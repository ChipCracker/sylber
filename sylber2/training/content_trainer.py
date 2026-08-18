"""LightningModule for Sylber 2.0 content-encoder training (stages 1-4).

Optimizer hyperparameters follow Table 6 of the paper (AdamW; stage-specific
lr, warmup, betas, weight decay, iterations). After warmup the learning rate
is held constant (as in the original Sylber configs, min_factor=1).
"""
import torch
from lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR

from ..models.sylber2_model import Sylber2
from ..utils.lr_schedule import COSLRLAMBDA


class ContentTrainer(LightningModule):

    def __init__(self, loss_coefs=None, lr=1e-4, warmup_steps=2000,
                 total_steps=100000, betas=(0.9, 0.999), weight_decay=1e-3,
                 min_factor=1.0, **model_configs):
        super().__init__()
        self.loss_coefs = loss_coefs or {"distillation_loss": 1.0}
        self.net = Sylber2(**model_configs).to(torch.float)
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.betas = tuple(betas)
        self.weight_decay = weight_decay
        self.min_factor = min_factor

    def forward(self, **kwargs):
        return self.net(**kwargs)

    def training_step(self, batch, batch_idx):
        if self.net.stage == 1:
            self.net.ema_step()
        outputs = self.net(**batch)
        loss = 0.0
        for name, coef in self.loss_coefs.items():
            if name in outputs:
                loss = loss + coef * outputs[name]
                self.log(f"train_{name}", outputs[name], sync_dist=True)
        for name in ("num_segments", "boundary_f1_proxy"):
            if name in outputs:
                self.log(f"train_{name}", outputs[name], sync_dist=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.net(**batch)
        loss = 0.0
        for name, coef in self.loss_coefs.items():
            if name in outputs:
                loss = loss + coef * outputs[name]
                self.log(f"val_{name}", outputs[name], sync_dist=True)
        self.log("val_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        params = [p for p in self.net.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.lr, betas=self.betas,
                                weight_decay=self.weight_decay, eps=1e-6)
        lr_lambda = COSLRLAMBDA(self.warmup_steps, self.total_steps,
                                self.min_factor, 0)
        sch = LambdaLR(opt, lr_lambda)
        return [opt], [{"scheduler": sch, "interval": "step"}]

    def on_save_checkpoint(self, checkpoint):
        # persist the teacher/EMA so resuming within a stage is exact
        if self.net.stage == 1 and self.net.ema is not None:
            checkpoint["ema_state"] = self.net.ema.model.state_dict()
        if self.net.teacher_backbone is not None:
            checkpoint["teacher_backbone"] = self.net.teacher_backbone.state_dict()
        if self.net.teacher_boundary is not None:
            checkpoint["teacher_boundary"] = self.net.teacher_boundary.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if "ema_state" in checkpoint:
            self.net.setup_teacher()
            self.net.ema.model.load_state_dict(checkpoint["ema_state"])
        if "teacher_backbone" in checkpoint:
            self.net.setup_teacher()
            self.net.teacher_backbone.load_state_dict(checkpoint["teacher_backbone"])
        if "teacher_boundary" in checkpoint and self.net.stage == 4:
            self.net.teacher_boundary.load_state_dict(checkpoint["teacher_boundary"])
