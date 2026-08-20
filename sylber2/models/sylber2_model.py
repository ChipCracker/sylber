"""Sylber 2.0 core self-distillation model (Sec. 3.2, Appendix A.1).

Four training stages for the content encoder:
  Stage 1: frame-wise self-distillation, EMA teacher (decay 0.999),
           both student and teacher inputs are (differently) augmented.
  Stage 2/3: self-segmentation distillation. Teacher = frozen copy of the
           student (weights at stage start). Targets are segment-averaged
           teacher features; segments come from greedy segmentation with
           refinement on the teacher's clean-input features. Augmentation
           is applied to the student input only.
  Stage 3: additionally trains the boundary detector with BCE against
           boundaries from the teacher segments.
  Stage 4: same as stage 3 but segments are produced by the (frozen)
           teacher's boundary detector via peak detection instead of the
           greedy algorithm.

Teacher target features are taken from the 8th transformer layer and
L2-normalized. The student predicts with 9 layers + an additional FC head.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .content_encoder import ContentEncoder
from .ema_module import EMAModule
from ..segmentation import (get_segments_greedy, detect_boundaries,
                            boundaries_to_segments, segments_to_boundary_targets)


class Sylber2(nn.Module):

    def __init__(self,
                 stage=1,
                 speech_upstream="utter-project/mHuBERT-147",
                 ema_decay=0.999,
                 merge_threshold_range=(0.5, 0.7),
                 refine_threshold=0.5,
                 peak_min_height=0.2,
                 peak_prominence=0.05,
                 peak_prob_override=0.8,
                 load_pretrained=True,
                 encoder_kwargs=None,
                 **kwargs):
        super().__init__()
        assert stage in (1, 2, 3, 4)
        self.stage = stage
        self.ema_decay = ema_decay
        self.merge_threshold_range = tuple(merge_threshold_range)
        self.refine_threshold = refine_threshold
        self.peak_kwargs = dict(min_height=peak_min_height,
                                prominence=peak_prominence,
                                prob_override=peak_prob_override)

        self.student = ContentEncoder(speech_upstream=speech_upstream,
                                      load_pretrained=load_pretrained and stage == 1,
                                      **(encoder_kwargs or {}))
        # teacher holders (built in setup_teacher, not part of state_dict)
        self.ema = None
        self.teacher_backbone = None
        self.teacher_boundary = None

    # ------------------------------------------------------------------ teacher
    def setup_teacher(self):
        """Initialize the teacher for the current stage. Stage 1 uses an EMA
        teacher; stages 2-4 use a frozen copy of the student at stage start."""
        if self.stage == 1:
            if self.ema is None:
                self.ema = EMAModule(self.student.backbone, ema_decay=self.ema_decay)
        else:
            if self.teacher_backbone is None:
                self.teacher_backbone = copy.deepcopy(self.student.backbone)
                self.teacher_backbone.requires_grad_(False).eval()
            if self.stage == 4 and self.teacher_boundary is None:
                self.teacher_boundary = copy.deepcopy(self.student.boundary_detector)
                self.teacher_boundary.requires_grad_(False).eval()

    def _sync_teacher_device(self):
        """EMAModule is not an nn.Module attribute, so Lightning's .to(device)
        does not move it (e.g. after on_load_checkpoint restores it on CPU)."""
        if self.ema is not None:
            dev = next(self.student.backbone.parameters()).device
            if next(self.ema.model.parameters()).device != dev:
                self.ema.model.to(dev)

    def ema_step(self):
        if self.stage == 1:
            if self.ema is None:
                self.setup_teacher()
            else:
                self._sync_teacher_device()
                self.ema.step(self.student.backbone)

    def _teacher_model(self):
        return self.ema.model if self.stage == 1 else self.teacher_backbone

    @torch.no_grad()
    def _teacher_features(self, wav):
        """Returns (layer-8 target features, layer-9 frames) of the teacher."""
        model = self._teacher_model()
        model.eval()
        out = model(wav, output_hidden_states=True)
        return out.hidden_states[self.student.target_layer], out.last_hidden_state

    # ------------------------------------------------------------------ forward
    def forward(self, student_input, teacher_input, **kwargs):
        self.setup_teacher()
        self._sync_teacher_device()
        device = student_input.device

        with torch.no_grad():
            trg_l8, trg_l9 = self._teacher_features(teacher_input)

        outputs = {}
        student_frames, student_pred = self.student.student_frames(student_input)
        B, L, D = student_pred.shape

        if self.stage == 1:
            target = F.normalize(trg_l8, dim=-1)
            outputs['distillation_loss'] = ((student_pred - target) ** 2).sum(-1).mean()
            return outputs

        # ---- stages 2-4: segment targets from the teacher
        segments_batch = self._teacher_segments(trg_l8, trg_l9)

        target = torch.zeros_like(trg_l8)
        for b, segments in enumerate(segments_batch):
            for s, e in segments:
                seg_mean = trg_l8[b, s:e].mean(0)
                target[b, s:e] = F.normalize(seg_mean, dim=-1)
        outputs['distillation_loss'] = ((student_pred - target) ** 2).sum(-1).mean()

        if self.stage >= 3:
            bnd_targets = torch.from_numpy(np.stack([
                segments_to_boundary_targets(segs, L) for segs in segments_batch
            ])).to(device)
            logits = self.student.boundary_logits(student_frames.detach())
            outputs['boundary_loss'] = F.binary_cross_entropy_with_logits(logits, bnd_targets)
            with torch.no_grad():
                pred_bnd = (torch.sigmoid(logits) > 0.5).float()
                outputs['boundary_f1_proxy'] = (
                    2 * (pred_bnd * bnd_targets).sum() /
                    (pred_bnd.sum() + bnd_targets.sum() + 1e-8)).detach()
        outputs['num_segments'] = torch.tensor(
            float(np.mean([len(s) for s in segments_batch])), device=device)
        return outputs

    @torch.no_grad()
    def _teacher_segments(self, trg_l8, trg_l9):
        """Segments per batch element, from greedy segmentation (stages 2/3) or
        from the teacher's boundary detector via peak detection (stage 4)."""
        if self.stage == 4:
            self.teacher_boundary.eval()
            probs = torch.sigmoid(self.teacher_boundary(trg_l9)).float().cpu().numpy()
            L = trg_l9.shape[1]
            return [boundaries_to_segments(detect_boundaries(p, **self.peak_kwargs), L)
                    for p in probs]
        lo, hi = self.merge_threshold_range
        merge_threshold = np.random.uniform(lo, hi) if lo < hi else lo
        feats = trg_l8.float().cpu().numpy()
        return [get_segments_greedy(f, merge_threshold, self.refine_threshold)
                for f in feats]

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def segment(self, input_values, use_boundary_detector=True, inference_prominence=0.1):
        """Inference-time segmentation. Returns list of dicts with segments,
        frame features, and 64-d content embeddings."""
        self.eval()
        frames, _ = self.student.student_frames(input_values)
        results = []
        if use_boundary_detector and self.student.boundary_detector is not None:
            probs = torch.sigmoid(self.student.boundary_logits(frames)).float().cpu().numpy()
            L = frames.shape[1]
            kw = dict(self.peak_kwargs)
            kw['prominence'] = inference_prominence
            segments_batch = [boundaries_to_segments(detect_boundaries(p, **kw), L)
                              for p in probs]
        else:
            feats = frames.float().cpu().numpy()
            lo, hi = self.merge_threshold_range
            segments_batch = [get_segments_greedy(f, (lo + hi) / 2, self.refine_threshold)
                              for f in feats]
        for b, segments in enumerate(segments_batch):
            if len(segments):
                seg_fts = torch.stack([frames[b, s:e].mean(0) for s, e in segments])
                content = self.student.content_proj(seg_fts)
            else:
                seg_fts = frames[b, :0]
                content = None
            results.append({'segments': segments,
                            'segment_features': seg_fts,
                            'content_embedding': content,
                            'frames': frames[b]})
        return results
