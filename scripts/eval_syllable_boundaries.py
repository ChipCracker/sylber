#!/usr/bin/env python3
"""Table-1 reference test: syllable boundary detection vs. text-based
references (MFA phone alignments + maximal-onset syllabification),
Pr/Re/F1/R-value with 50 ms tolerance, plus short-segment filtering.

Usage:
  python scripts/eval_syllable_boundaries.py CKPT ALIGN_DIR AUDIO_DIR \
      [--limit N] [--device cuda]
ALIGN_DIR: directory with LibriSpeech TextGrids (words+phones tiers)
AUDIO_DIR: matching LibriSpeech split (flac files)
"""
import argparse
import glob
import re
from pathlib import Path

import numpy as np

VOWEL_RE = re.compile(r"^[A-Z]+[0-9]$")
LEGAL_ONSETS = {
    (), ("P",), ("B",), ("T",), ("D",), ("K",), ("G",), ("F",), ("V",),
    ("TH",), ("DH",), ("S",), ("Z",), ("SH",), ("ZH",), ("HH",), ("CH",),
    ("JH",), ("M",), ("N",), ("L",), ("R",), ("W",), ("Y",),
    ("P", "L"), ("P", "R"), ("P", "Y"), ("B", "L"), ("B", "R"), ("B", "Y"),
    ("T", "R"), ("T", "W"), ("T", "Y"), ("D", "R"), ("D", "W"), ("D", "Y"),
    ("K", "L"), ("K", "R"), ("K", "W"), ("K", "Y"), ("G", "L"), ("G", "R"),
    ("G", "W"), ("F", "L"), ("F", "R"), ("F", "Y"), ("TH", "R"), ("TH", "W"),
    ("S", "L"), ("S", "M"), ("S", "N"), ("S", "P"), ("S", "T"), ("S", "K"),
    ("S", "W"), ("S", "F"), ("SH", "R"), ("V", "Y"), ("M", "Y"), ("N", "Y"),
    ("HH", "Y"), ("L", "Y"),
    ("S", "P", "L"), ("S", "P", "R"), ("S", "P", "Y"), ("S", "T", "R"),
    ("S", "T", "Y"), ("S", "K", "L"), ("S", "K", "R"), ("S", "K", "W"),
    ("S", "K", "Y"),
}


def parse_textgrid_phones(path):
    """Minimal TextGrid parser -> list of (phone, start, end) from the
    'phones' tier."""
    text = open(path, errors="ignore").read()
    m = re.search(r'name = "phones".*?intervals: size = \d+(.*?)(?:item \[|\Z)',
                  text, re.S)
    if not m:
        return []
    out = []
    for iv in re.finditer(r'xmin = ([\d.]+)\s*\n\s*xmax = ([\d.]+)\s*\n\s*text = "([^"]*)"',
                          m.group(1)):
        s, e, p = float(iv.group(1)), float(iv.group(2)), iv.group(3).strip()
        out.append((p, s, e))
    return out


def syllable_boundaries(phones):
    """Reference syllable start times via maximal-onset syllabification.
    `phones`: (label, start, end); silences ('', 'sil', 'sp', 'spn') split words."""
    bounds = []
    word = []
    for p, s, e in phones + [("", 0, 0)]:
        if p in ("", "sil", "sp", "spn", "SIL"):
            if word:
                bounds += _syllabify_word(word)
                word = []
            continue
        word.append((p, s, e))
    return sorted(bounds)


def _syllabify_word(word):
    labels = [p for p, _, _ in word]
    nuclei = [i for i, l in enumerate(labels) if VOWEL_RE.match(l)]
    if len(nuclei) <= 1:
        return [word[0][1]]  # monosyllabic: syllable starts at word start
    starts = [0]
    for prev_n, n in zip(nuclei[:-1], nuclei[1:]):
        cluster = [l[:2] if l[:2] in ("TH", "DH", "SH", "ZH", "CH", "JH", "HH")
                   else re.sub(r"[0-9]", "", l) for l in labels[prev_n + 1:n]]
        cluster = [re.sub(r"[0-9]", "", l) for l in labels[prev_n + 1:n]]
        # maximal onset: give the next syllable the longest legal onset
        k = 0
        for k in range(len(cluster), -1, -1):
            if tuple(cluster[len(cluster) - k:]) in LEGAL_ONSETS:
                break
        starts.append(n - k)
    return [word[i][1] for i in starts]


def merge_short(segments, min_dur):
    """Iteratively merge segments shorter than min_dur into the shorter
    neighbor (paper: 'filtering out short segments')."""
    segs = [list(s) for s in segments]
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, (s, e) in enumerate(segs):
            if e - s >= min_dur:
                continue
            if i == 0:
                j = 1
            elif i == len(segs) - 1:
                j = i - 1
            else:
                j = i - 1 if (segs[i-1][1]-segs[i-1][0]) <= (segs[i+1][1]-segs[i+1][0]) else i + 1
            lo, hi = min(segs[i][0], segs[j][0]), max(segs[i][1], segs[j][1])
            segs[min(i, j)] = [lo, hi]
            del segs[max(i, j)]
            changed = True
            break
    return segs


def prf_r(pred, ref, tol=0.05):
    ref = sorted(ref)
    used = [False] * len(ref)
    tp = 0
    for p in sorted(pred):
        best, bi = tol + 1, -1
        for i, r in enumerate(ref):
            if used[i]:
                continue
            d = abs(p - r)
            if d < best:
                best, bi = d, i
        if bi >= 0 and best <= tol:
            used[bi] = True
            tp += 1
    pr = tp / max(len(pred), 1)
    re_ = tp / max(len(ref), 1)
    f1 = 2 * pr * re_ / max(pr + re_, 1e-9)
    os_ = re_ / max(pr, 1e-9) - 1
    r1 = np.sqrt((1 - re_) ** 2 + os_ ** 2)
    r2 = (-os_ + re_ - 1) / np.sqrt(2)
    rv = 1 - (abs(r1) + abs(r2)) / 2
    return pr, re_, f1, rv, tp, len(pred), len(ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("align_dir")
    ap.add_argument("audio_dir")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from sylber2.inference import Segmenter2
    seg = Segmenter2(args.ckpt, device=args.device)

    tgs = sorted(glob.glob(f"{args.align_dir}/**/*.TextGrid", recursive=True))
    if args.limit:
        tgs = tgs[:args.limit]
    filters = [0.0, 0.06, 0.08, 0.10, 0.12]
    agg = {f: np.zeros(3) for f in filters}  # tp, n_pred, n_ref
    n_utts = 0
    for tg in tgs:
        utt = Path(tg).stem
        spk, chap = utt.split("-")[0], utt.split("-")[1]
        wav = Path(args.audio_dir) / spk / chap / f"{utt}.flac"
        if not wav.exists():
            continue
        phones = parse_textgrid_phones(tg)
        ref = syllable_boundaries(phones)
        ref = [b for b in ref if b > 0.01]  # inner boundaries only
        if len(ref) < 2:
            continue
        out = seg(wav_file=str(wav))
        segments = out["segments"]
        if len(segments) < 2:
            continue
        for f in filters:
            segs = merge_short(segments, f) if f > 0 else segments
            pred = [s for s, _ in segs[1:]]  # inner boundaries
            _, _, _, _, tp, npred, nref = prf_r(pred, ref)
            agg[f] += [tp, npred, nref]
        n_utts += 1
        if n_utts % 200 == 0:
            print(f"  {n_utts} utts...", flush=True)

    print(f"\n=== Syllable boundary detection, LibriSpeech ({n_utts} utts, 50 ms tol) ===")
    print(f"{'filter':>8} {'Pr':>6} {'Re':>6} {'F1':>6} {'R':>6}")
    for f in filters:
        tp, npred, nref = agg[f]
        pr, re_ = tp / npred, tp / nref
        f1 = 2 * pr * re_ / (pr + re_)
        os_ = re_ / pr - 1
        rv = 1 - (abs(np.sqrt((1 - re_) ** 2 + os_ ** 2)) + abs((-os_ + re_ - 1) / np.sqrt(2))) / 2
        label = "raw" if f == 0 else f">={int(f*1000)}ms"
        print(f"{label:>8} {pr*100:6.1f} {re_*100:6.1f} {f1*100:6.1f} {rv*100:6.1f}")


if __name__ == "__main__":
    main()
