"""GAN discriminators and losses for the Sylber 2.0 vocoder training.

Same setup as Vocos (Siuzdak, 2024): multi-period discriminator (HiFi-GAN)
and multi-resolution (spectrogram) discriminator, hinge loss, feature
matching loss, and a log-mel reconstruction loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.nn.utils import weight_norm


class PeriodDiscriminator(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        chs = [32, 128, 512, 1024]
        convs, in_ch = [], 1
        for ch in chs:
            convs.append(weight_norm(nn.Conv2d(in_ch, ch, (5, 1), (3, 1), padding=(2, 0))))
            in_ch = ch
        convs.append(weight_norm(nn.Conv2d(in_ch, 1024, (5, 1), 1, padding=(2, 0))))
        self.convs = nn.ModuleList(convs)
        self.post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):  # (B, T)
        b, t = x.shape
        pad = (self.period - t % self.period) % self.period
        x = F.pad(x, (0, pad), mode="reflect").view(b, 1, -1, self.period)
        fmaps = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmaps.append(x)
        x = self.post(x)
        fmaps.append(x)
        return x.flatten(1, -1), fmaps


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.discs = nn.ModuleList([PeriodDiscriminator(p) for p in periods])

    def forward(self, x):
        return [d(x) for d in self.discs]


class ResolutionDiscriminator(nn.Module):
    def __init__(self, n_fft, hop_length, win_length):
        super().__init__()
        self.n_fft, self.hop, self.win = n_fft, hop_length, win_length
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        ch = 32
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, ch, (3, 9), padding=(1, 4))),
            weight_norm(nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(ch, ch, (3, 3), padding=(1, 1))),
        ])
        self.post = weight_norm(nn.Conv2d(ch, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):  # (B, T)
        spec = torch.stft(x, self.n_fft, self.hop, self.win,
                          window=self.window, center=True, return_complex=True)
        x = spec.abs().unsqueeze(1)  # (B, 1, F, T)
        fmaps = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmaps.append(x)
        x = self.post(x)
        fmaps.append(x)
        return x.flatten(1, -1), fmaps


class MultiResolutionDiscriminator(nn.Module):
    def __init__(self, resolutions=((1024, 256, 1024), (2048, 512, 2048), (512, 128, 512))):
        super().__init__()
        self.discs = nn.ModuleList([ResolutionDiscriminator(*r) for r in resolutions])

    def forward(self, x):
        return [d(x) for d in self.discs]


# ----------------------------------------------------------------- loss helpers
def discriminator_hinge_loss(real_outs, fake_outs):
    loss = 0.0
    for (dr, _), (df, _) in zip(real_outs, fake_outs):
        loss = loss + torch.mean(F.relu(1 - dr)) + torch.mean(F.relu(1 + df))
    return loss / len(real_outs)


def generator_hinge_loss(fake_outs):
    loss = 0.0
    for df, _ in fake_outs:
        loss = loss + torch.mean(F.relu(1 - df))
    return loss / len(fake_outs)


def feature_matching_loss(real_outs, fake_outs):
    loss, n = 0.0, 0
    for (_, fr), (_, ff) in zip(real_outs, fake_outs):
        for r, f in zip(fr, ff):
            loss = loss + F.l1_loss(f, r.detach())
            n += 1
    return loss / max(n, 1)


class MelSpecLoss(nn.Module):
    """Log-mel L1 reconstruction loss (as in Vocos), for 24 kHz audio."""

    def __init__(self, sample_rate=24000, n_fft=1024, hop_length=256, n_mels=100):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, center=True, power=1)

    def forward(self, fake, real):
        m_f = torch.log(self.melspec(fake).clamp(min=1e-5))
        m_r = torch.log(self.melspec(real).clamp(min=1e-5))
        return F.l1_loss(m_f, m_r)
