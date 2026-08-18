#!/usr/bin/env python3
"""Build Sylber 2.0 content manifests from a local MLS installation
(e.g. /nfs/data/mls on kiz0) without downloading anything.

Durations are taken from MLS's segments.txt (id, url, start, end), so no
audio files need to be opened. English and French are excluded (Sec. 3.5:
covered by Emilia).

Usage:
    python scripts/mls_manifest_from_local.py /nfs/data/mls \
        [--languages german spanish ...] [--split train]
"""
import argparse
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("SYLBER2_DATA", "data"))
DEFAULT_LANGS = ["dutch", "german", "italian", "polish", "portuguese", "spanish"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mls_root")
    ap.add_argument("--languages", nargs="*", default=DEFAULT_LANGS)
    ap.add_argument("--split", default="train")
    ap.add_argument("--min-seconds", type=float, default=1.0)
    args = ap.parse_args()

    out_dir = DATA_ROOT / "manifests" / "content"
    out_dir.mkdir(parents=True, exist_ok=True)
    for lang in args.languages:
        assert lang not in ("english", "french"), \
            "en/fr are excluded (covered by Emilia; Sec. 3.5)"
        base = Path(args.mls_root) / f"mls_{lang}" / args.split
        seg_file = base / "segments.txt"
        if not seg_file.exists():
            print(f"skip {lang}: {seg_file} missing")
            continue
        manifest = out_dir / f"mls_{lang}.tsv"
        n, total = 0, 0.0
        with open(seg_file) as f, open(manifest, "w") as mf:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                seg_id, _, start, end = parts[0], parts[1], parts[2], parts[3]
                dur = float(end) - float(start)
                if dur < args.min_seconds:
                    continue
                spk, book, _ = seg_id.split("_")
                path = base / "audio" / spk / book / f"{seg_id}.flac"
                mf.write(f"{path}\t{dur:.2f}\n")
                n += 1
                total += dur
        print(f"mls_{lang}: {n} files, {total/3600:.1f} h -> {manifest}")


if __name__ == "__main__":
    main()
