"""Unsupervised segmentation utilities for Sylber 2.0.

Implements the greedy segmentation algorithm from Sylber (Cho et al., 2025)
*without* the explicit silent-frame masking (removed in Sylber 2.0, Sec. 3.2.2),
plus the short-segment refinement described in Appendix A.1.1 of the
Sylber 2.0 paper, and peak-detection-based boundary decoding for the
boundary detector (Appendix A.1.2).
"""
import numpy as np
from scipy.signal import find_peaks

# 50 Hz frame rate -> 80 ms == 4 frames
FRAMES_PER_SECOND = 50
MIN_SEGMENT_FRAMES_80MS = 4


def cossim(x, y):
    return (x * y).sum(-1) / (((x ** 2).sum(-1) + 1e-8) ** .5) / (((y ** 2).sum(-1) + 1e-8) ** .5)


def greedy_segment(states, merge_threshold):
    """Greedy single-sweep segmentation (Cho et al., 2025) without silent masking.

    Groups adjacent frames whose cosine similarity to the running segment mean
    is above `merge_threshold`. After the sweep, mid-boundaries are locally
    re-optimized with a similarity sweep (as in the original Sylber code).

    Args:
        states: (L, D) numpy array of frame features.
        merge_threshold: cosine similarity threshold for merging frames.
    Returns:
        (N, 2) numpy array of [start, end) frame indices covering all frames.
    """
    L = len(states)
    if L == 0:
        return np.zeros((0, 2), dtype=np.int64)

    segments = []
    midboundaries = []
    curr = states[0]
    seg_cnt = 1
    s = 0
    for i in range(1, L):
        sim = cossim(curr, states[i])
        if sim >= merge_threshold:
            curr = (curr * seg_cnt + states[i]) / (seg_cnt + 1)
            seg_cnt += 1
        else:
            segments.append([s, i])
            midboundaries.append([i, len(segments) - 1])
            curr = states[i]
            seg_cnt = 1
            s = i
    segments.append([s, L])

    # local boundary refinement by similarity sweep (from original Sylber)
    merged = []
    for bd, segi in midboundaries:
        if segi >= len(segments) - 1:
            continue
        prev_center = states[segments[segi][0]:segments[segi][1]].mean(0)
        next_center = states[segments[segi + 1][0]:segments[segi + 1][1]].mean(0)
        if cossim(prev_center, next_center) >= merge_threshold:
            segments[segi + 1] = [segments[segi][0], segments[segi + 1][1]]
            merged.append(segi)
            continue
        lo = max(segments[segi][0], bd - max(1, (segments[segi][1] - segments[segi][0]) // 2))
        hi = min(segments[segi + 1][1], bd + max(1, (segments[segi + 1][1] - segments[segi + 1][0]) // 2))
        sim_prev = cossim(states[lo:hi], prev_center[None, :])
        sim_next = cossim(states[lo:hi], next_center[None, :])
        sim_sweep = [(sim_prev[:i].sum() + sim_next[i:].sum()) for i in range(0, hi - lo)]
        opt_b = np.arange(lo, hi)[np.argmax(sim_sweep)]
        segments[segi] = [segments[segi][0], opt_b]
        segments[segi + 1] = [opt_b, segments[segi + 1][1]]
    segments = [seg for segi, seg in enumerate(segments) if segi not in merged]
    segments = [seg for seg in segments if seg[1] > seg[0]]
    return np.array(segments, dtype=np.int64)


def refine_short_segments(states, segments, refine_threshold,
                          min_frames=MIN_SEGMENT_FRAMES_80MS):
    """Merge segments shorter than `min_frames` (80 ms at 50 Hz) into an
    adjacent segment when their cosine similarity exceeds `refine_threshold`
    (Sylber 2.0, Appendix A.1.1). The more similar neighbor absorbs the
    short segment; the process repeats until stable.
    """
    if len(segments) <= 1:
        return np.asarray(segments, dtype=np.int64)
    segs = [list(s) for s in segments]
    changed = True
    while changed and len(segs) > 1:
        changed = False
        means = [states[s:e].mean(0) for s, e in segs]
        for i in range(len(segs)):
            s, e = segs[i]
            if e - s >= min_frames:
                continue
            cand = []  # (sim, neighbor_index)
            if i > 0:
                cand.append((cossim(means[i], means[i - 1]), i - 1))
            if i < len(segs) - 1:
                cand.append((cossim(means[i], means[i + 1]), i + 1))
            if not cand:
                continue
            sim, j = max(cand, key=lambda t: t[0])
            if sim > refine_threshold:
                lo = min(segs[i][0], segs[j][0])
                hi = max(segs[i][1], segs[j][1])
                keep = min(i, j)
                segs[keep] = [lo, hi]
                del segs[max(i, j)]
                changed = True
                break
    return np.array(segs, dtype=np.int64)


def get_segments_greedy(states, merge_threshold, refine_threshold=None):
    """Full Sylber 2.0 unsupervised segmentation: greedy sweep + refinement."""
    segments = greedy_segment(states, merge_threshold)
    if refine_threshold is not None and len(segments) > 1:
        segments = refine_short_segments(states, segments, refine_threshold)
    return segments


def boundaries_to_segments(boundaries, length):
    """Convert a sorted list of boundary frame indices into [start, end) segments
    covering [0, length)."""
    bounds = [b for b in boundaries if 0 < b < length]
    edges = [0] + bounds + [length]
    segs = [[s, e] for s, e in zip(edges[:-1], edges[1:]) if e > s]
    return np.array(segs, dtype=np.int64) if segs else np.zeros((0, 2), dtype=np.int64)


def segments_to_boundary_targets(segments, length):
    """Binary per-frame boundary targets (1 at each segment start except frame 0)."""
    target = np.zeros(length, dtype=np.float32)
    for s, _ in segments:
        if 0 < s < length:
            target[s] = 1.0
    return target


def detect_boundaries(probs, min_height=0.2, prominence=0.05, prob_override=0.8):
    """Peak detection on boundary probabilities (Sylber 2.0, Appendix A.1.2).

    Boundaries are peaks with height >= `min_height` and either
    prominence > `prominence`, or probability > `prob_override`.
    At inference, `prominence=0.1` yields near-identical results.
    """
    probs = np.asarray(probs)
    peaks, props = find_peaks(probs, height=min_height, prominence=1e-6)
    keep = (props["prominences"] > prominence) | (probs[peaks] > prob_override)
    boundaries = peaks[keep]
    # frames above the override threshold that are plateaus (not strict peaks)
    override = np.where(probs > prob_override)[0]
    if len(override):
        merged = sorted(set(boundaries.tolist()) | set(_plateau_centers(probs, override)))
        boundaries = np.array(merged, dtype=np.int64)
    return boundaries


def _plateau_centers(probs, idxs):
    """Collapse consecutive above-threshold runs to their argmax frame."""
    centers = []
    run = [idxs[0]]
    for i in idxs[1:]:
        if i == run[-1] + 1:
            run.append(i)
        else:
            centers.append(run[int(np.argmax(probs[run]))])
            run = [i]
    centers.append(run[int(np.argmax(probs[run]))])
    return centers


def segment_average(features, segments):
    """Average frame `features` (L, D) within each [s, e) segment -> (N, D)."""
    if len(segments) == 0:
        return features[:0]
    import torch
    if isinstance(features, torch.Tensor):
        return torch.stack([features[s:e].mean(0) for s, e in segments])
    return np.stack([features[s:e].mean(0) for s, e in segments])
