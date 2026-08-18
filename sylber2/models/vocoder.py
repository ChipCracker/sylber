"""Sylber 2.0 vocoder for syllable-to-speech synthesis (Sec. 3.4).

Vocos-style backbone (Siuzdak, 2024): 12 ConvNeXt blocks predicting phase and
magnitude, generating 24 kHz audio through inverse STFT. Content and acoustic
embeddings are duplicated to 50 Hz frames according to segment durations, and
a within-segment positional encoding (wSegPE) with an 11-entry learnable
template (linearly interpolated over relative position in [0, 1]) is
concatenated to each frame.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, intermediate_dim, layer_scale_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))

    def forward(self, x):  # (B, L, D)
        residual = x
        x = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        return residual + self.gamma * x


class ISTFTHead(nn.Module):
    """Vocos iSTFT head: predicts log-magnitude and phase, overlap-adds with
    'same' padding so the output length is exactly L * hop."""

    def __init__(self, dim, n_fft=1920, hop_length=480):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop_length
        self.out = nn.Linear(dim, n_fft + 2)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def forward(self, x):  # (B, L, D) -> (B, L*hop)
        B, L, _ = x.shape
        x = self.out(x).transpose(1, 2)                    # (B, n_fft+2, L)
        mag, phase = x.chunk(2, dim=1)                     # (B, n_fft//2+1, L)
        mag = torch.exp(mag.clamp(max=1e2))
        spec = mag * (torch.cos(phase) + 1j * torch.sin(phase))
        audio = self._istft(spec)
        return audio

    def _istft(self, spec):
        B, F_, L = spec.shape
        ifft = torch.fft.irfft(spec.transpose(1, 2), n=self.n_fft, dim=-1)  # (B, L, n_fft)
        ifft = ifft * self.window[None, None, :]
        # overlap-add
        out_len = (L - 1) * self.hop + self.n_fft
        frames = ifft.permute(0, 2, 1)                                       # (B, n_fft, L)
        audio = F.fold(frames, output_size=(1, out_len), kernel_size=(1, self.n_fft),
                       stride=(1, self.hop)).squeeze(1).squeeze(1)
        wsq = (self.window ** 2)[None, :, None].expand(1, self.n_fft, L)
        win_sq = F.fold(wsq, output_size=(1, out_len), kernel_size=(1, self.n_fft),
                        stride=(1, self.hop)).squeeze(1).squeeze(1).squeeze(0)
        audio = audio / win_sq.clamp(min=1e-8)
        # 'same' padding: crop centered so output length == L * hop
        pad = (self.n_fft - self.hop) // 2
        return audio[:, pad:pad + L * self.hop]


class WSegPE(nn.Module):
    """Within-segment positional encoding: an 11-entry learnable template
    indexed by relative position in [0, 1] with linear interpolation."""

    def __init__(self, dim=64, num_entries=11):
        super().__init__()
        self.template = nn.Parameter(torch.randn(num_entries, dim) * 0.02)
        self.num_entries = num_entries

    def forward(self, positions):  # positions: (B, L) in [0, 1]
        scaled = positions.clamp(0, 1) * (self.num_entries - 1)
        lo = scaled.floor().long().clamp(max=self.num_entries - 1)
        hi = (lo + 1).clamp(max=self.num_entries - 1)
        frac = (scaled - lo.float()).unsqueeze(-1)
        return (1 - frac) * self.template[lo] + frac * self.template[hi]


def make_frame_inputs(content_embs, acoustic_embs, segments_batch, total_frames):
    """Duplicate per-segment embeddings back to 50 Hz frames and compute
    relative within-segment positions.

    Returns (content (B,L,Dc), acoustic (B,L,Da), positions (B,L)).
    """
    B = len(content_embs)
    device = content_embs[0].device
    Dc = content_embs[0].shape[-1]
    Da = acoustic_embs[0].shape[-1]
    content = torch.zeros(B, total_frames, Dc, device=device)
    acoustic = torch.zeros(B, total_frames, Da, device=device)
    positions = torch.zeros(B, total_frames, device=device)
    for b in range(B):
        segs = segments_batch[b]
        for i, (s, e) in enumerate(segs):
            s, e = int(s), int(min(e, total_frames))
            if e <= s:
                continue
            content[b, s:e] = content_embs[b][i]
            acoustic[b, s:e] = acoustic_embs[b][i]
            n = e - s
            if n == 1:
                positions[b, s] = 0.0
            else:
                positions[b, s:e] = torch.linspace(0, 1, n, device=device)
    return content, acoustic, positions


class SylberVocoder(nn.Module):
    """Vocos backbone consuming (content + acoustic + wSegPE) frame inputs."""

    def __init__(self,
                 content_dim=64,
                 acoustic_dim=64,
                 pe_dim=64,
                 dim=1024,
                 intermediate_dim=4096,
                 num_layers=12,
                 n_fft=1920,
                 hop_length=480,
                 sample_rate=24000):
        super().__init__()
        self.wsegpe = WSegPE(pe_dim)
        in_dim = content_dim + acoustic_dim + pe_dim
        self.embed = nn.Linear(in_dim, dim)
        self.norm_in = nn.LayerNorm(dim, eps=1e-6)
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(dim, intermediate_dim) for _ in range(num_layers)])
        self.norm_out = nn.LayerNorm(dim, eps=1e-6)
        self.head = ISTFTHead(dim, n_fft, hop_length)
        self.sample_rate = sample_rate
        self.hop_length = hop_length

    def forward(self, content, acoustic, positions):
        """content/acoustic: (B, L, 64); positions: (B, L) in [0,1] -> (B, L*hop)."""
        pe = self.wsegpe(positions)
        x = self.embed(torch.cat([content, acoustic, pe], dim=-1))
        x = self.norm_in(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm_out(x)
        return self.head(x)
