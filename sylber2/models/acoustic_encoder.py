"""Sylber 2.0 syllable-guided acoustic encoder (Sec. 3.3).

CNN initialized from WavLM-Large, except the 2nd conv layer whose stride is
widened from 2 to 3 so the total stride becomes 480 (= 50 Hz at 24 kHz input).
Followed by 6 transformer layers. Frame outputs are averaged within segments
from the boundary detector and projected with residual FC layers to 64-d.
"""
import torch
import torch.nn as nn
from transformers import WavLMModel, WavLMConfig, HubertConfig
from transformers.models.hubert.modeling_hubert import HubertEncoder

from .content_encoder import EmbeddingProjector

WAVLM_REPO = "microsoft/wavlm-large"

# WavLM-Large CNN architecture (used when training from scratch/offline)
WAVLM_LARGE_CNN = dict(
    conv_dim=(512, 512, 512, 512, 512, 512, 512),
    conv_stride=(5, 2, 2, 2, 2, 2, 2),
    conv_kernel=(10, 3, 3, 3, 3, 2, 2),
    conv_bias=True,
    feat_extract_norm="layer",
)


class AcousticEncoder(nn.Module):

    def __init__(self,
                 wavlm_upstream=WAVLM_REPO,
                 load_pretrained=True,
                 hidden_size=768,
                 num_layers=6,
                 num_heads=12,
                 intermediate_size=3072,
                 embed_dim=64,
                 proj_blocks=2,
                 **kwargs):
        super().__init__()
        if load_pretrained:
            wavlm = WavLMModel.from_pretrained(wavlm_upstream)
            self.feature_extractor = wavlm.feature_extractor
            cnn_dim = wavlm.config.conv_dim[-1]
            del wavlm
        else:
            cfg = WavLMConfig(**WAVLM_LARGE_CNN)
            self.feature_extractor = WavLMModel(cfg).feature_extractor
            cnn_dim = cfg.conv_dim[-1]
        # widen the stride of the 2nd conv layer 2 -> 3 (total stride 320 -> 480,
        # i.e. 50 Hz on 24 kHz input); pretrained kernel weights are kept.
        self.feature_extractor.conv_layers[1].conv.stride = (3,)
        self.feature_extractor.requires_grad_(True)

        self.proj = nn.Sequential(nn.LayerNorm(cnn_dim), nn.Linear(cnn_dim, hidden_size))
        enc_cfg = HubertConfig(hidden_size=hidden_size,
                               num_hidden_layers=num_layers,
                               num_attention_heads=num_heads,
                               intermediate_size=intermediate_size,
                               hidden_dropout=0.0, attention_dropout=0.0,
                               layerdrop=0.0)
        self.encoder = HubertEncoder(enc_cfg)
        self.acoustic_proj = EmbeddingProjector(hidden_size, embed_dim, proj_blocks)
        self.hidden_size = hidden_size

    def forward_frames(self, wav24k):
        """wav24k: (B, T) at 24 kHz -> frame features (B, L, hidden) at 50 Hz."""
        feats = self.feature_extractor(wav24k)          # (B, C, L)
        feats = feats.transpose(1, 2)                   # (B, L, C)
        x = self.proj(feats)
        x = self.encoder(x).last_hidden_state
        return x

    def forward(self, wav24k, segments_batch):
        """Returns list (len B) of (N_b, 64) acoustic embeddings, one per segment."""
        frames = self.forward_frames(wav24k)
        embs = []
        for b, segments in enumerate(segments_batch):
            if len(segments) == 0:
                embs.append(frames[b, :0, :self.acoustic_proj.out.out_features])
                continue
            L = frames.shape[1]
            seg_fts = torch.stack([
                frames[b, min(s, L - 1):max(min(e, L), min(s, L - 1) + 1)].mean(0)
                for s, e in segments])
            embs.append(self.acoustic_proj(seg_fts))
        return embs
