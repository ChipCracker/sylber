"""Sylber 2.0 content encoder (Sec. 3.2).

- 9 Transformer layers initialized from mHuBERT-147 (last 3 randomly reinitialized)
- an additional student-only FC layer on top (not in the teacher)
- a boundary detector: 3 Transformer layers (same architecture) + binary logit
- residual FC layers projecting segment-averaged features to 64-d content
  embeddings (updated during synthesis-model training)
"""
import torch
import torch.nn as nn
from torch.nn import init
from transformers import HubertModel, HubertConfig
from transformers.models.hubert.modeling_hubert import HubertEncoderLayer

MHUBERT_REPO = "utter-project/mHuBERT-147"


def _xavier_reinit(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            init.ones_(m.weight)
            init.zeros_(m.bias)


class ResidualFC(nn.Module):
    """Pre-LN residual fully-connected block: x + W2 GELU(W1 LN(x))."""

    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class EmbeddingProjector(nn.Module):
    """Residual FC layers reducing the dimension to `out_dim` (=64)."""

    def __init__(self, in_dim, out_dim=64, num_blocks=2):
        super().__init__()
        self.blocks = nn.Sequential(*[ResidualFC(in_dim) for _ in range(num_blocks)])
        self.out = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.out(self.blocks(x))


class PredictorHead(nn.Module):
    """BYOL-style predictor: Linear -> BatchNorm -> GELU -> Linear.

    The paper describes "an additional fully-connected layer that is not in
    the teacher"; with a single linear layer (and with DINO-style centering
    added) our stage-1 self-distillation still collapsed to a single
    direction. BatchNorm in the predictor is the empirically decisive
    anti-collapse ingredient in BYOL-type recipes, so we use the standard
    BYOL predictor here (documented deviation, IMPLEMENTATION_NOTES #18)."""

    def __init__(self, dim, hidden_dim=2048):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):  # (B, L, D)
        B, L, D = x.shape
        h = self.fc1(x).reshape(B * L, -1)
        h = self.act(self.bn(h)).reshape(B, L, -1)
        return self.fc2(h)


class BoundaryDetector(nn.Module):
    """3 Transformer layers (same architecture as the main model) + binary logit."""

    def __init__(self, config, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([HubertEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.logit = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states, attention_mask=None):
        x = hidden_states
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)[0]
        return self.logit(self.norm(x)).squeeze(-1)  # (B, L)


class ContentEncoder(nn.Module):

    def __init__(self,
                 speech_upstream=MHUBERT_REPO,
                 num_hidden_layers=9,
                 num_reinit_layers=3,
                 target_layer=8,
                 boundary_layers=3,
                 embed_dim=64,
                 proj_blocks=2,
                 load_pretrained=True,
                 with_student_head=True,
                 with_boundary_detector=True,
                 **kwargs):
        super().__init__()
        if load_pretrained:
            self.backbone = HubertModel.from_pretrained(
                speech_upstream, num_hidden_layers=num_hidden_layers)
            # randomly reinitialize the last `num_reinit_layers` transformer layers
            for layer in self.backbone.encoder.layers[num_hidden_layers - num_reinit_layers:]:
                _xavier_reinit(layer)
        else:
            self.backbone = HubertModel(HubertConfig.from_pretrained(
                speech_upstream, num_hidden_layers=num_hidden_layers))
        cfg = self.backbone.config
        self.enc_dim = cfg.hidden_size
        self.target_layer = target_layer

        # student-only FC layer (not in the teacher)
        self.student_head = PredictorHead(self.enc_dim) if with_student_head else None
        self.boundary_detector = BoundaryDetector(cfg, boundary_layers) if with_boundary_detector else None
        # residual FC projection to the 64-d content embedding (used in synthesis training)
        self.content_proj = EmbeddingProjector(self.enc_dim, embed_dim, proj_blocks)

    def forward_frames(self, input_values, attention_mask=None, output_hidden_states=False):
        """Frame features from the last (9th) layer; optionally all hidden states."""
        out = self.backbone(input_values, attention_mask=attention_mask,
                            output_hidden_states=output_hidden_states)
        return out

    def student_frames(self, input_values, attention_mask=None):
        """Student prediction: last layer + student head. Returns (frames, head_out)."""
        h = self.backbone(input_values, attention_mask=attention_mask).last_hidden_state
        pred = self.student_head(h) if self.student_head is not None else h
        return h, pred

    @torch.no_grad()
    def teacher_frames(self, input_values, attention_mask=None):
        """Teacher target features from `target_layer` (8th), L2-normalized."""
        out = self.backbone(input_values, attention_mask=attention_mask,
                            output_hidden_states=True)
        h = out.hidden_states[self.target_layer]
        return h

    def boundary_logits(self, frames, attention_mask=None):
        assert self.boundary_detector is not None
        return self.boundary_detector(frames, attention_mask=attention_mask)
