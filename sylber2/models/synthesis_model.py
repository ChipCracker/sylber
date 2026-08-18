"""Sylber 2.0 synthesis model: frozen content encoder + boundary detector,
trainable acoustic encoder and Vocos vocoder (Sec. 3.4).

Disentanglement strategies during training (Table 7):
- the content encoder and boundary detector are frozen (only the final
  residual FC layers of the content projection are updated),
- random voice/audio perturbations are applied to the acoustic-encoder input
  and the reconstruction target, while the content input stays original,
- acoustic embeddings are randomly mean-pooled or shuffled across time.
"""
import numpy as np
import torch
import torch.nn as nn

from .sylber2_model import Sylber2
from .acoustic_encoder import AcousticEncoder
from .vocoder import SylberVocoder, make_frame_inputs


class SynthesisModel(nn.Module):

    def __init__(self,
                 content_encoder: Sylber2,
                 acoustic_encoder: AcousticEncoder = None,
                 vocoder: SylberVocoder = None,
                 mean_pool_acoustics_prob=0.2,
                 shuffle_acoustics_prob=0.0,
                 freeze_acoustic_encoder=False,
                 inference_prominence=0.1,
                 **kwargs):
        super().__init__()
        self.content_model = content_encoder
        # freeze everything in the content encoder except the residual FC projection
        self.content_model.requires_grad_(False).eval()
        self.content_model.student.content_proj.requires_grad_(True)

        self.acoustic_encoder = acoustic_encoder or AcousticEncoder()
        if freeze_acoustic_encoder:
            self.acoustic_encoder.requires_grad_(False)
        self.freeze_acoustic_encoder = freeze_acoustic_encoder
        self.vocoder = vocoder or SylberVocoder()
        self.mean_pool_acoustics_prob = mean_pool_acoustics_prob
        self.shuffle_acoustics_prob = shuffle_acoustics_prob
        self.inference_prominence = inference_prominence

    def train(self, mode=True):
        super().train(mode)
        self.content_model.eval()  # frozen backbone stays in eval
        if self.freeze_acoustic_encoder:
            self.acoustic_encoder.eval()
        return self

    @torch.no_grad()
    def encode_content(self, wav16k):
        """Frozen content forward: frame features + segments via boundary detector."""
        results = self.content_model.segment(
            wav16k, use_boundary_detector=True,
            inference_prominence=self.inference_prominence)
        segments_batch = [r['segments'] for r in results]
        seg_feats = [r['segment_features'] for r in results]
        return segments_batch, seg_feats

    def forward(self, wav16k, wav24k_acoustic, num_frames=None, augment=True):
        """
        wav16k: (B, T16) original speech for the (frozen) content encoder.
        wav24k_acoustic: (B, T24) possibly voice/audio-perturbed input for the
            acoustic encoder (perturbation happens in the data pipeline).
        Returns generated (B, T24') audio at 24 kHz.
        """
        segments_batch, seg_feats = self.encode_content(wav16k)
        # content projection (trainable residual FCs)
        content_embs = [self.content_model.student.content_proj(f) if len(f) else f
                        for f in seg_feats]
        acoustic_embs = self.acoustic_encoder(wav24k_acoustic, segments_batch)

        if augment and self.training:
            acoustic_embs = self._perturb_acoustics(acoustic_embs)

        if num_frames is None:
            num_frames = max(int(segs[-1][1]) if len(segs) else 1
                             for segs in segments_batch)
        content, acoustic, positions = make_frame_inputs(
            content_embs, acoustic_embs, segments_batch, num_frames)
        audio = self.vocoder(content, acoustic, positions)
        return audio, segments_batch

    def _perturb_acoustics(self, acoustic_embs):
        out = []
        for emb in acoustic_embs:
            if len(emb) == 0:
                out.append(emb)
                continue
            u = np.random.uniform()
            if u < self.mean_pool_acoustics_prob:
                emb = emb.mean(0, keepdim=True).expand(len(emb), -1)
            elif u < self.mean_pool_acoustics_prob + self.shuffle_acoustics_prob:
                emb = emb[torch.randperm(len(emb), device=emb.device)]
            out.append(emb)
        return out
