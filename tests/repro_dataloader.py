"""Isolate the dataloader hang: iterate a few real batches under different
worker/augmentation settings. Run on a CPU node with SYLBER2_DATA set."""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
from sylber2.data.datasets import ContentDataset
from sylber2.data.datamodules import resolve_sources

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--formant-prob", type=float, default=0.3)
ap.add_argument("--context", default=None, choices=[None, "fork", "spawn", "forkserver"])
ap.add_argument("--batches", type=int, default=8)
ap.add_argument("--batch-size", type=int, default=24)
ap.add_argument("--seed", type=int, default=None)
args = ap.parse_args()
if args.seed is not None:
    import lightning as pl
    pl.seed_everything(args.seed)

DATA = os.environ["SYLBER2_DATA"]
sources = resolve_sources(f"{DATA}/manifests/content", {"fleurs": 2.0, "default": 1.0})
ds = ContentDataset(sources, dummy_len=2000, both_augmented=True, augment_configs=dict(
    noise_dir=f"{DATA}/noise", rir_dir=f"{DATA}/rir", speech_dir=f"{DATA}/speech_clips",
    formant_prob=args.formant_prob))
print(f"dataset ready ({len(sources)} sources); workers={args.workers} "
      f"formant={args.formant_prob} ctx={args.context}", flush=True)
dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers,
                collate_fn=ContentDataset.collate, timeout=120 if args.workers else 0,
                multiprocessing_context=args.context, persistent_workers=args.workers > 0)
t0 = time.time()
for i, batch in enumerate(dl):
    print(f"batch {i}: {batch['student_input'].shape} t={time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    if i + 1 >= args.batches:
        break
print("REPRO OK", flush=True)
