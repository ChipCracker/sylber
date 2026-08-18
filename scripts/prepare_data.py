#!/usr/bin/env python3
"""Data preparation for Sylber 2.0 training.

Downloads corpora (via HuggingFace streaming) into $SYLBER2_DATA/audio/...,
writes flac files, and creates manifests:

  content (16 kHz training, Sec. 3.5):
    manifests/content/emilia_<lang>.tsv   (weight 1 each)
    manifests/content/mls_<lang>.tsv      (weight 1 each; en/fr excluded)
    manifests/content/fleurs.tsv          (one source, weight 2)
  resynthesis (acoustic encoder + vocoder):
    manifests/resynth/fleurs_r.tsv        (weight 7)
    manifests/resynth/expresso.tsv
    manifests/resynth/globe.tsv
    manifests/resynth/gtsinger.tsv

Subcommands: fleurs, mls, emilia, fleurs_r, expresso, globe, gtsinger,
noise, rir, speech_clips. Use --max-hours to cap per-source size (subsets for
smoke tests); omit for full downloads.

Emilia and GTSinger are gated on HuggingFace: accept their terms on the hub
and set HF_TOKEN before running.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

DATA_ROOT = Path(os.environ.get("SYLBER2_DATA", "data"))

MLS_LANGS = ["dutch", "german", "italian", "polish", "portuguese", "spanish"]
EMILIA_LANGS = ["DE", "EN", "FR", "JA", "KO", "ZH"]


def _manifest_path(kind, name):
    p = DATA_ROOT / "manifests" / kind / f"{name}.tsv"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _write_stream(iterator, out_dir, manifest, max_hours=None, min_seconds=1.0,
                  audio_key="audio"):
    """Consume a HF streaming dataset, write flac files + manifest lines."""
    out_dir.mkdir(parents=True, exist_ok=True)
    total_sec, n = 0.0, 0
    with open(manifest, "w") as mf:
        for i, ex in enumerate(iterator):
            audio = ex[audio_key]
            arr, sr = np.asarray(audio["array"]), audio["sampling_rate"]
            dur = len(arr) / sr
            if dur < min_seconds:
                continue
            sub = out_dir / f"{n // 10000:04d}"
            sub.mkdir(exist_ok=True)
            path = sub / f"{n:08d}.flac"
            sf.write(path, arr.astype(np.float32), sr)
            mf.write(f"{path.resolve()}\t{dur:.2f}\n")
            total_sec += dur
            n += 1
            if n % 500 == 0:
                print(f"  {n} files, {total_sec/3600:.2f} h", flush=True)
            if max_hours and total_sec >= max_hours * 3600:
                break
    print(f"DONE {manifest.name}: {n} files, {total_sec/3600:.2f} h")


def _fleurs_langs(repo):
    from huggingface_hub import HfApi
    api = HfApi()
    return sorted(e.path.split("/")[-1]
                  for e in api.list_repo_tree(repo, "data", repo_type="dataset")
                  if e.path.count("/") == 1)


def _fleurs_tar(repo, out_name, kind, args):
    """Download FLEURS(-R) language tarballs directly (the datasets-lib loading
    script is no longer supported) and build one combined manifest."""
    import tarfile
    from huggingface_hub import hf_hub_download
    langs = args.languages or _fleurs_langs(repo)
    split = getattr(args, "split", "train")
    out_dir = DATA_ROOT / "audio" / out_name
    manifest = _manifest_path(kind, out_name)
    per_lang = (args.max_hours / len(langs)) if args.max_hours else None
    with open(manifest, "w") as mf:
        for lang in langs:
            print(f"[{out_name}] {lang} ({split})", flush=True)
            try:
                tar_path = hf_hub_download(repo, f"data/{lang}/audio/{split}.tar.gz",
                                           repo_type="dataset")
            except Exception as e:
                print(f"  skip {lang}: {e}")
                continue
            ld = out_dir / lang
            ld.mkdir(parents=True, exist_ok=True)
            sec, n = 0.0, 0
            with tarfile.open(tar_path, "r:gz") as tf:
                for m in tf:
                    if not m.isfile() or not m.name.endswith(".wav"):
                        continue
                    dst = ld / Path(m.name).name
                    if not dst.exists():
                        with tf.extractfile(m) as f, open(dst, "wb") as o:
                            o.write(f.read())
                    try:
                        dur = sf.info(dst).duration
                    except Exception:
                        dst.unlink(missing_ok=True)
                        continue
                    if dur < 1.0:
                        continue
                    mf.write(f"{dst.resolve()}\t{dur:.2f}\n")
                    sec += dur; n += 1
                    if per_lang and sec >= per_lang * 3600:
                        break
            print(f"  {lang}: {n} files, {sec/3600:.2f} h", flush=True)
            if not getattr(args, "keep_archives", False):
                # free the cached tarball (blob + symlink) after extraction
                real = os.path.realpath(tar_path)
                for p in (tar_path, real):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
    print(f"DONE {manifest}")


def cmd_fleurs(args):
    _fleurs_tar("google/fleurs", "fleurs", "content", args)


def cmd_mls(args):
    from datasets import load_dataset
    langs = args.languages or MLS_LANGS
    for lang in langs:
        assert lang not in ("english", "french"), \
            "en/fr are excluded (covered by Emilia; Sec. 3.5)"
        print(f"[mls] {lang}")
        ds = load_dataset("facebook/multilingual_librispeech", lang,
                          split="train", streaming=True)
        _write_stream(ds, DATA_ROOT / "audio" / "mls" / lang,
                      _manifest_path("content", f"mls_{lang}"),
                      max_hours=args.max_hours)


def cmd_emilia(args):
    from datasets import load_dataset
    langs = args.languages or EMILIA_LANGS
    for lang in langs:
        print(f"[emilia] {lang} (gated: requires accepted terms + HF_TOKEN)")
        ds = load_dataset("amphion/Emilia-Dataset", data_dir=f"Emilia/{lang}",
                          split="train", streaming=True)
        def gen():
            for ex in ds:
                key = next((k for k in ("mp3", "wav", "audio") if k in ex), None)
                if key is None:
                    continue
                yield {"audio": ex[key]}
        _write_stream(gen(), DATA_ROOT / "audio" / "emilia" / lang,
                      _manifest_path("content", f"emilia_{lang.lower()}"),
                      max_hours=args.max_hours)


def cmd_fleurs_r(args):
    _fleurs_tar("google/fleurs-r", "fleurs_r", "resynth", args)


def cmd_expresso(args):
    from datasets import load_dataset
    ds = load_dataset("ylacombe/expresso", split="train", streaming=True)
    _write_stream(ds, DATA_ROOT / "audio" / "expresso",
                  _manifest_path("resynth", "expresso"), max_hours=args.max_hours)


def cmd_globe(args):
    from datasets import load_dataset
    ds = load_dataset("MushanW/GLOBE_V2", split="train", streaming=True)
    _write_stream(ds, DATA_ROOT / "audio" / "globe",
                  _manifest_path("resynth", "globe"), max_hours=args.max_hours)


def cmd_gtsinger(args):
    """GTSinger (gated). Downloads the repo's wav files via snapshot_download."""
    from huggingface_hub import snapshot_download
    local = snapshot_download("GTSinger/GTSinger", repo_type="dataset",
                              allow_patterns=["*.wav", "*.flac"],
                              local_dir=DATA_ROOT / "audio" / "gtsinger")
    make_manifest_from_dir(Path(local), _manifest_path("resynth", "gtsinger"),
                           max_hours=args.max_hours)


def cmd_noise(args):
    """DNS-Challenge fullband noise via the official download script."""
    out = DATA_ROOT / "noise"
    out.mkdir(parents=True, exist_ok=True)
    url = ("https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/"
           "download-dns-challenge-5-noise-ir.sh")
    script = out / "download-dns5-noise-ir.sh"
    subprocess.run(["curl", "-sL", "-o", str(script), url], check=True)
    print(f"Official DNS script saved to {script}.")
    print("It downloads noise_fullband archives; run (large download!):")
    print(f"  cd {out} && bash {script.name} && "
          "find . -name '*.tar.bz2' -exec tar xjf {} \;")
    if args.run:
        subprocess.run(["bash", str(script.name)], cwd=out, check=True)
        subprocess.run("find . -name '*.tar.bz2' -exec tar xjf {} \;",
                       shell=True, cwd=out, check=True)


def cmd_rir(args):
    """RIRs. Paper uses the GTU-RIR corpus (github.com/mehmetpekmezci/gtu-rir);
    its bulk download is external (Google Drive). Default here: OpenSLR-28
    real+simulated RIRs as a drop-in replacement (--gtu prints instructions)."""
    out = DATA_ROOT / "rir"
    out.mkdir(parents=True, exist_ok=True)
    if args.gtu:
        print("GTU-RIR: clone https://github.com/mehmetpekmezci/gtu-rir and follow "
              f"its data links; place wavs under {out}/")
        return
    zip_path = out / "rirs_noises.zip"
    if not zip_path.exists():
        subprocess.run(["curl", "-L", "-o", str(zip_path),
                        "https://www.openslr.org/resources/28/rirs_noises.zip"],
                       check=True)
    subprocess.run(["unzip", "-oq", str(zip_path), "-d", str(out)], check=True)
    n = len(list(out.rglob("*.wav")))
    print(f"RIRs ready: {n} wav files under {out}")


def cmd_speech_clips(args):
    """Interfering speech-clip pool for augmentation: sample files from the
    content manifests into data/speech_clips (symlinks)."""
    import random
    out = DATA_ROOT / "speech_clips"
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for tsv in (DATA_ROOT / "manifests" / "content").glob("*.tsv"):
        with open(tsv) as f:
            files += [l.split("\t")[0] for l in f.read().splitlines() if l]
    random.shuffle(files)
    n = 0
    for p in files[:args.num_clips]:
        dst = out / f"{n:06d}{Path(p).suffix}"
        if not dst.exists():
            try:
                dst.symlink_to(p)
            except OSError:
                continue
        n += 1
    print(f"speech_clips: {n} links in {out}")


def make_manifest_from_dir(root, manifest, exts=(".wav", ".flac", ".ogg"),
                           max_hours=None):
    total, n = 0.0, 0
    with open(manifest, "w") as mf:
        for f in sorted(Path(root).rglob("*")):
            if f.suffix.lower() not in exts:
                continue
            try:
                dur = sf.info(f).duration
            except Exception:
                continue
            if dur < 1.0:
                continue
            mf.write(f"{f.resolve()}\t{dur:.2f}\n")
            total += dur; n += 1
            if max_hours and total >= max_hours * 3600:
                break
    print(f"DONE {manifest}: {n} files, {total/3600:.2f} h")


def cmd_manifest(args):
    make_manifest_from_dir(args.audio_dir, _manifest_path(args.kind, args.name),
                           max_hours=args.max_hours)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fleurs", "mls", "emilia", "fleurs_r", "expresso", "globe",
                 "gtsinger"):
        p = sub.add_parser(name)
        p.add_argument("--languages", nargs="*", default=None)
        p.add_argument("--max-hours", type=float, default=None,
                       help="cap total (fleurs/fleurs_r: total across languages; "
                            "others: per language)")
        p.add_argument("--split", default="train",
                       help="fleurs/fleurs_r only: train|dev|test")
        p.add_argument("--keep-archives", action="store_true",
                       help="fleurs/fleurs_r only: keep cached tarballs")
    p = sub.add_parser("noise"); p.add_argument("--run", action="store_true")
    p = sub.add_parser("rir"); p.add_argument("--gtu", action="store_true")
    p = sub.add_parser("speech_clips"); p.add_argument("--num-clips", type=int, default=20000)
    p = sub.add_parser("manifest")
    p.add_argument("audio_dir"); p.add_argument("kind", choices=["content", "resynth"])
    p.add_argument("name"); p.add_argument("--max-hours", type=float, default=None)
    args = ap.parse_args()
    globals()[f"cmd_{args.cmd}"](args)


if __name__ == "__main__":
    main()
