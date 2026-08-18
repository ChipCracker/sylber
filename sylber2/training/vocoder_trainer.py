"""LightningModule for Sylber 2.0 acoustic-encoder + vocoder training
(Sec. 3.4, Appendix A.1.5). Manual optimization with hinge GAN losses,
feature matching, log-mel reconstruction, and optional WavLM perceptual loss.

One instance = one cycle of the cosine learning-rate schedule (Table 7);
run the cycles sequentially, warm-starting from the previous checkpoint.
"""
import torch
from lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR

from ..models.sylber2_model import Sylber2
from ..models.synthesis_model import SynthesisModel
from ..models.acoustic_encoder import AcousticEncoder
from ..models.vocoder import SylberVocoder
from ..models.discriminators import (MultiPeriodDiscriminator, MultiResolutionDiscriminator,
                                     discriminator_hinge_loss, generator_hinge_loss,
                                     feature_matching_loss, MelSpecLoss)
from ..utils.lr_schedule import COSLRLAMBDA


class VocoderTrainer(LightningModule):

    def __init__(self,
                 content_ckpt=None,
                 content_model_configs=None,
                 acoustic_configs=None,
                 vocoder_configs=None,
                 use_perceptual_loss=False,
                 freeze_acoustic_encoder=False,
                 mean_pool_acoustics_prob=0.2,
                 shuffle_acoustics_prob=0.0,
                 mel_coef=45.0, fm_coef=1.0, gan_coef=1.0, perceptual_coef=1.0,
                 lr=5e-5, warmup_steps=1000, total_steps=2000000,
                 betas=(0.8, 0.9), weight_decay=0.01, min_factor=0.01,
                 **kwargs):
        super().__init__()
        self.automatic_optimization = False

        content = Sylber2(stage=4, load_pretrained=False,
                          **(content_model_configs or {}))
        if content_ckpt:
            state = torch.load(content_ckpt, map_location="cpu", weights_only=False)
            sd = state.get("state_dict", state)
            sd = {k[len("net."):]: v for k, v in sd.items() if k.startswith("net.")} or sd
            sd = {k: v for k, v in sd.items() if k.startswith("student.")}
            missing, unexpected = content.load_state_dict(sd, strict=False)
            print(f"[content ckpt] missing={len(missing)} unexpected={len(unexpected)}")

        self.model = SynthesisModel(
            content_encoder=content,
            acoustic_encoder=AcousticEncoder(**(acoustic_configs or {})),
            vocoder=SylberVocoder(**(vocoder_configs or {})),
            mean_pool_acoustics_prob=mean_pool_acoustics_prob,
            shuffle_acoustics_prob=shuffle_acoustics_prob,
            freeze_acoustic_encoder=freeze_acoustic_encoder)

        self.mpd = MultiPeriodDiscriminator()
        self.mrd = MultiResolutionDiscriminator()
        self.mel_loss = MelSpecLoss()
        self.use_perceptual_loss = use_perceptual_loss
        if use_perceptual_loss:
            from ..models.perceptual import WavLMPerceptualLoss
            self.perceptual = WavLMPerceptualLoss()
        else:
            self.perceptual = None
        self.coefs = dict(mel=mel_coef, fm=fm_coef, gan=gan_coef,
                          perceptual=perceptual_coef)
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.betas = tuple(betas)
        self.weight_decay = weight_decay
        self.min_factor = min_factor

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        sch_g, sch_d = self.lr_schedulers()

        fake, _ = self.model(batch["wav16"], batch["wav24"])
        real = batch["wav24"][:, :fake.shape[-1]]

        # ---- discriminator step
        real_mpd, fake_mpd = self.mpd(real), self.mpd(fake.detach())
        real_mrd, fake_mrd = self.mrd(real), self.mrd(fake.detach())
        d_loss = discriminator_hinge_loss(real_mpd, fake_mpd) + \
                 discriminator_hinge_loss(real_mrd, fake_mrd)
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        self.clip_gradients(opt_d, gradient_clip_val=10.0, gradient_clip_algorithm="norm")
        opt_d.step()
        sch_d.step()

        # ---- generator step
        fake_mpd_g = self.mpd(fake)
        fake_mrd_g = self.mrd(fake)
        real_mpd_g = self.mpd(real)
        real_mrd_g = self.mrd(real)
        g_adv = generator_hinge_loss(fake_mpd_g) + generator_hinge_loss(fake_mrd_g)
        g_fm = feature_matching_loss(real_mpd_g, fake_mpd_g) + \
               feature_matching_loss(real_mrd_g, fake_mrd_g)
        g_mel = self.mel_loss(fake, real)
        g_loss = self.coefs["gan"] * g_adv + self.coefs["fm"] * g_fm + \
                 self.coefs["mel"] * g_mel
        if self.perceptual is not None:
            g_perc = self.perceptual(fake, real)
            g_loss = g_loss + self.coefs["perceptual"] * g_perc
            self.log("train_perceptual", g_perc, prog_bar=False)
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        self.clip_gradients(opt_g, gradient_clip_val=10.0, gradient_clip_algorithm="norm")
        opt_g.step()
        sch_g.step()

        self.log_dict({"train_d_loss": d_loss, "train_g_adv": g_adv,
                       "train_g_fm": g_fm, "train_g_mel": g_mel}, prog_bar=False)
        self.log("train_g_loss", g_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        fake, _ = self.model(batch["wav16"], batch["wav24"])
        real = batch["wav24"][:, :fake.shape[-1]]
        self.log("val_mel_loss", self.mel_loss(fake, real), sync_dist=True)

    def generator_parameters(self):
        params = [p for p in self.model.vocoder.parameters()]
        params += [p for p in self.model.content_model.student.content_proj.parameters()]
        if not self.model.freeze_acoustic_encoder:
            params += [p for p in self.model.acoustic_encoder.parameters()]
        return [p for p in params if p.requires_grad]

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(self.generator_parameters(), lr=self.lr,
                                  betas=self.betas, weight_decay=self.weight_decay)
        opt_d = torch.optim.AdamW(list(self.mpd.parameters()) + list(self.mrd.parameters()),
                                  lr=self.lr, betas=self.betas,
                                  weight_decay=self.weight_decay)
        lam = COSLRLAMBDA(self.warmup_steps, self.total_steps, self.min_factor, 0)
        sch_g = LambdaLR(opt_g, lam)
        sch_d = LambdaLR(opt_d, lam)
        return [opt_g, opt_d], [sch_g, sch_d]
