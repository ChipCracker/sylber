"""Lightning data modules wrapping the Sylber 2.0 datasets."""
from pathlib import Path
from torch.utils.data import DataLoader
from lightning import LightningDataModule

from .datasets import ContentDataset, ResynthesisDataset


def resolve_sources(spec, weight_map=None):
    """`spec` is either an explicit [[weight, manifest_path], ...] list or a
    directory containing one TSV per source. For a directory, weights come
    from `weight_map` by file stem (e.g. {"fleurs": 2.0, "default": 1.0})."""
    if isinstance(spec, (str, Path)):
        weight_map = dict(weight_map or {})
        default = weight_map.pop("default", 1.0)
        files = sorted(Path(spec).glob("*.tsv"))
        assert files, f"no manifests found in {spec}"
        return [[float(weight_map.get(f.stem, default)), str(f)] for f in files]
    return [[float(w), str(p)] for w, p in spec]


class ContentDataModule(LightningDataModule):
    def __init__(self, train_sources, val_sources=None, batch_size=72,
                 val_batch_size=None, num_workers=8, dummy_len=100000,
                 crop_seconds=5.0, both_augmented=True, augment_configs=None,
                 **kwargs):
        super().__init__()
        weight_map = kwargs.get("source_weights")
        self.train_sources = resolve_sources(train_sources, weight_map)
        self.val_sources = resolve_sources(val_sources, weight_map) if val_sources \
            else self.train_sources
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers
        self.dummy_len = dummy_len
        self.crop_seconds = crop_seconds
        self.both_augmented = both_augmented
        self.augment_configs = augment_configs or {}

    def _make(self, sources, dummy_len, augment):
        cfg = dict(self.augment_configs)
        if not augment:
            cfg.update(formant_prob=0, noise_prob=0, speech_clip_prob=0,
                       rir_prob=0, white_noise_prob=0)
        return ContentDataset(sources, crop_seconds=self.crop_seconds,
                              dummy_len=dummy_len,
                              both_augmented=self.both_augmented and augment,
                              augment_configs=cfg)

    def train_dataloader(self):
        ds = self._make(self.train_sources, self.dummy_len, augment=True)
        return DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers,
                          collate_fn=ContentDataset.collate, drop_last=True,
                          pin_memory=True, persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        ds = self._make(self.val_sources, 500, augment=False)
        return DataLoader(ds, batch_size=self.val_batch_size, num_workers=2,
                          collate_fn=ContentDataset.collate, drop_last=True)


class ResynthesisDataModule(LightningDataModule):
    def __init__(self, train_sources, val_sources=None, batch_size=12,
                 num_workers=8, dummy_len=100000, crop_seconds=3.0,
                 perturb_voice_prob=0.2, perturb_audio_prob=0.2,
                 noise_dir=None, rir_dir=None, **kwargs):
        super().__init__()
        weight_map = kwargs.get("source_weights")
        self.train_sources = resolve_sources(train_sources, weight_map)
        self.val_sources = resolve_sources(val_sources, weight_map) if val_sources \
            else self.train_sources
        self.kw = dict(crop_seconds=crop_seconds, noise_dir=noise_dir, rir_dir=rir_dir)
        self.perturb = dict(perturb_voice_prob=perturb_voice_prob,
                            perturb_audio_prob=perturb_audio_prob)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dummy_len = dummy_len

    def train_dataloader(self):
        ds = ResynthesisDataset(self.train_sources, dummy_len=self.dummy_len,
                                **self.perturb, **self.kw)
        return DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers,
                          collate_fn=ResynthesisDataset.collate, drop_last=True,
                          pin_memory=True, persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        ds = ResynthesisDataset(self.val_sources, dummy_len=200,
                                perturb_voice_prob=0, perturb_audio_prob=0, **self.kw)
        return DataLoader(ds, batch_size=self.batch_size, num_workers=2,
                          collate_fn=ResynthesisDataset.collate, drop_last=True)
