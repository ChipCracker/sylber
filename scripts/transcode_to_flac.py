#!/usr/bin/env python3
"""Transcode mp3 entries of content manifests to 16 kHz mono FLAC.

libsndfile's mp3 decoder leaks memory and can segfault under heavy dataloader
use; FLAC also seeks cleanly (no full-file decode per crop). Worker processes
are recycled (maxtasksperchild) so decoder leaks cannot accumulate.

Rewrites the manifests in place (mp3 path -> flac path, duration from the
written file) and deletes each source mp3 after successful verification
(use --keep-source to keep them).
"""
import argparse
import os
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import soundfile as sf
import librosa

DATA_ROOT = Path(os.environ.get("SYLBER2_DATA", "data"))
TARGET_SR = 16000


def transcode(args):
    src, keep = args
    src = Path(src)
    dst = src.with_suffix(".flac")
    try:
        if not dst.exists():
            y, sr = sf.read(src, dtype="float32", always_2d=True)
            y = y.mean(-1)
            if sr != TARGET_SR:
                y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
            sf.write(dst, y, TARGET_SR)
        dur = sf.info(dst).duration
        if dur < 0.5:
            dst.unlink(missing_ok=True)
            return (str(src), None, 0.0)
        if not keep:
            src.unlink(missing_ok=True)
        return (str(src), str(dst), dur)
    except Exception:
        dst.unlink(missing_ok=True)
        return (str(src), None, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", default=str(DATA_ROOT / "manifests" / "content"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--keep-source", action="store_true")
    args = ap.parse_args()

    for tsv in sorted(Path(args.manifest_dir).glob("*.tsv")):
        rows = [l.rstrip("\n").split("\t") for l in open(tsv) if l.strip()]
        mp3s = [(p, args.keep_source) for p, _ in rows if p.lower().endswith(".mp3")]
        if not mp3s:
            continue
        print(f"[{tsv.name}] {len(mp3s)} mp3 files", flush=True)
        mapping = {}
        with Pool(args.workers, maxtasksperchild=200) as pool:
            for i, (src, dst, dur) in enumerate(pool.imap_unordered(transcode, mp3s, chunksize=8)):
                mapping[src] = (dst, dur)
                if (i + 1) % 5000 == 0:
                    print(f"  {i+1}/{len(mp3s)}", flush=True)
        ok, dropped = 0, 0
        with open(tsv, "w") as f:
            for p, d in rows:
                if p in mapping:
                    dst, dur = mapping[p]
                    if dst is None:
                        dropped += 1
                        continue
                    f.write(f"{dst}\t{dur:.2f}\n")
                    ok += 1
                else:
                    f.write(f"{p}\t{d}\n")
        print(f"DONE {tsv.name}: {ok} transcoded, {dropped} dropped", flush=True)


if __name__ == "__main__":
    main()
