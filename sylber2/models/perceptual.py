"""Perceptual loss (Parker et al., 2025) computed with frozen WavLM-Large,
using layer 0 (CNN output) and transformer layers 3, 6, 9, and 12 (Sec. 3.4).
Audio is resampled from 24 kHz to WavLM's 16 kHz inside the loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import WavLMModel


class WavLMPerceptualLoss(nn.Module):

    def __init__(self, wavlm_upstream="microsoft/wavlm-large",
                 layers=(0, 3, 6, 9, 12), input_sr=24000, wavlm_sr=16000):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(wavlm_upstream)
        self.wavlm.requires_grad_(False).eval()
        self.layers = layers
        self.resample = torchaudio.transforms.Resample(input_sr, wavlm_sr)

    def train(self, mode=True):
        super().train(mode)
        self.wavlm.eval()
        return self

    def forward(self, fake, real):
        fake16 = self._norm(self.resample(fake))
        with torch.no_grad():
            real16 = self._norm(self.resample(real))
            h_real = self.wavlm(real16, output_hidden_states=True).hidden_states
        h_fake = self.wavlm(fake16, output_hidden_states=True).hidden_states
        loss = 0.0
        for l in self.layers:
            loss = loss + F.l1_loss(h_fake[l], h_real[l].detach())
        return loss / len(self.layers)

    @staticmethod
    def _norm(x):
        return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-9)
