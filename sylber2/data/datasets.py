"""Datasets for Sylber 2.0 training.

Content-encoder training (Sec. 3.5, Appendix A.1.3): a collection of sources
(each language of Emilia and MLS is a source with weight 1; FLEURS is one
source with weight 2). Audio is randomly cropped to 5 seconds at 16 kHz.

Synthesis training: FLEURS-R (weight 7), EXPRESSO, Globe, GTSinger; a random
3-second window is cropped, provided at 16 kHz (content input, original) and
24 kHz (acoustic input & reconstruction target, possibly voice/audio
perturbed).

Manifest format: one TSV per source; each line is an absolute audio path
(optionally `path\tduration_sec`). Sources are given as [weight, path] pairs.
"""
import numpy as np
import soundfile as sf
import librosa
import torch
from pathlib import Path
from torch.utils.data import Dataset
from lightning import LightningDataModule

from ..augment import ContentAugment, random_formant_perturb, mix_at_snr, FileSampler, _rms
from scipy.signal import fftconvolve


def _load_manifest(path):
    lines = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                lines.append(line.split("\t")[0])
    return lines


def _load_audio(path, target_sr, offset_sec=None, duration_sec=None):
    """Load (a crop of) an audio file, mono, resampled to target_sr.

    mp3 files are decoded fully and cropped in memory: libsndfile/mpg123 frame
    seeking is unreliable and floods stderr with layer3 errors."""
    info = sf.info(path)
    sr = info.samplerate
    if offset_sec is None:
        y, _ = sf.read(path, dtype="float32", always_2d=True)
    elif str(path).lower().endswith(".mp3"):
        y, _ = sf.read(path, dtype="float32", always_2d=True)
        start = int(offset_sec * sr)
        frames = int(duration_sec * sr) if duration_sec else len(y)
        y = y[start:start + frames]
    else:
        start = int(offset_sec * sr)
        frames = int(duration_sec * sr) if duration_sec else -1
        y, _ = sf.read(path, start=start, frames=frames, dtype="float32",
                       always_2d=True, fill_value=0.0)
    y = y.mean(-1)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y.astype(np.float32), info.duration


class WeightedSources:
    def __init__(self, sources):
        """sources: list of [weight, manifest_path]."""
        self.manifests = [_load_manifest(p) for _, p in sources]
        ws = np.array([w for w, _ in sources], dtype=np.float64)
        keep = [i for i, m in enumerate(self.manifests) if len(m)]
        assert keep, "all manifests are empty"
        self.manifests = [self.manifests[i] for i in keep]
        ws = ws[keep]
        self.weights = ws / ws.sum()

    def sample(self):
        si = np.random.choice(len(self.manifests), p=self.weights)
        files = self.manifests[si]
        return files[np.random.randint(len(files))]


class ContentDataset(Dataset):
    """5-second random crops at 16 kHz with Sylber 2.0 augmentation.

    Returns z-normalized `student_input` (augmented) and `teacher_input`
    (independently augmented in stage 1, clean otherwise).
    """

    def __init__(self, sources, crop_seconds=5.0, sr=16000, dummy_len=100000,
                 both_augmented=True, augment_configs=None):
        super().__init__()
        self.sources = WeightedSources(sources)
        self.crop = int(crop_seconds * sr)
        self.crop_seconds = crop_seconds
        self.sr = sr
        self.dummy_len = dummy_len
        self.both_augmented = both_augmented
        self.augment = ContentAugment(sr=sr, **(augment_configs or {}))

    def __len__(self):
        return self.dummy_len

    def _sample_crop(self):
        for _ in range(5):
            path = self.sources.sample()
            try:
                dur = sf.info(path).duration
                if dur > 600:  # skip pathological files (full-decode cost)
                    continue
                off = np.random.uniform(0, max(dur - self.crop_seconds, 0))
                y, _ = _load_audio(path, self.sr, off, self.crop_seconds)
            except Exception:
                continue
            if _rms(y) < 1e-5:
                continue
            if len(y) < self.crop:
                y = np.pad(y, (0, self.crop - len(y)))
            return y[:self.crop]
        return np.zeros(self.crop, dtype=np.float32)

    @staticmethod
    def _znorm(y):
        return (y - y.mean()) / (y.std() + 1e-9)

    def __getitem__(self, i):
        y = self._sample_crop()
        student = self.augment(y)
        teacher = self.augment(y) if self.both_augmented else y
        return {"student_input": torch.from_numpy(self._znorm(student)),
                "teacher_input": torch.from_numpy(self._znorm(teacher))}

    @staticmethod
    def collate(batch):
        return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


class ResynthesisDataset(Dataset):
    """3-second windows for acoustic-encoder + vocoder training.

    Returns:
      wav16: original window at 16 kHz (content input)
      wav24: possibly perturbed window at 24 kHz (acoustic input AND target)
    """

    def __init__(self, sources, crop_seconds=3.0, dummy_len=100000,
                 perturb_voice_prob=0.2, perturb_audio_prob=0.2,
                 noise_dir=None, rir_dir=None,
                 noise_snr=(5, 25), sr_content=16000, sr_audio=24000):
        super().__init__()
        self.sources = WeightedSources(sources)
        self.crop_seconds = crop_seconds
        self.sr_content = sr_content
        self.sr_audio = sr_audio
        self.dummy_len = dummy_len
        self.perturb_voice_prob = perturb_voice_prob
        self.perturb_audio_prob = perturb_audio_prob
        self.noise_snr = noise_snr
        self.noise_sampler = FileSampler(noise_dir, sr_audio) if noise_dir else None
        self.rir_sampler = FileSampler(rir_dir, sr_audio) if rir_dir else None

    def __len__(self):
        return self.dummy_len

    def __getitem__(self, i):
        n16 = int(self.crop_seconds * self.sr_content)
        n24 = int(self.crop_seconds * self.sr_audio)
        for _ in range(5):
            path = self.sources.sample()
            try:
                dur = sf.info(path).duration
                off = np.random.uniform(0, max(dur - self.crop_seconds, 0))
                y24, _ = _load_audio(path, self.sr_audio, off, self.crop_seconds)
                y16 = librosa.resample(y24, orig_sr=self.sr_audio, target_sr=self.sr_content)
            except Exception:
                continue
            if _rms(y24) < 1e-5:
                continue
            break
        else:
            y24 = np.zeros(n24, dtype=np.float32)
            y16 = np.zeros(n16, dtype=np.float32)
        y24 = np.pad(y24, (0, max(0, n24 - len(y24))))[:n24]
        y16 = np.pad(y16, (0, max(0, n16 - len(y16))))[:n16].astype(np.float32)

        # voice / audio perturbation of the acoustic input AND target;
        # the 16 kHz content input stays original.
        if np.random.uniform() < self.perturb_voice_prob:
            y24 = random_formant_perturb(y24, self.sr_audio)
        if np.random.uniform() < self.perturb_audio_prob:
            y24 = self._audio_perturb(y24)

        peak = np.abs(y24).max()
        if peak > 1.0:
            y24 = y24 / peak
        return {"wav16": torch.from_numpy(self._znorm(y16)),
                "wav24": torch.from_numpy(y24.astype(np.float32))}

    def _audio_perturb(self, y):
        did = False
        if self.rir_sampler and len(self.rir_sampler) and np.random.uniform() < 0.5:
            rir = self.rir_sampler.sample_rir()
            if rir is not None:
                wet = fftconvolve(y, rir)[:len(y)]
                if _rms(wet) > 1e-6:
                    y = (wet * (_rms(y) / _rms(wet))).astype(np.float32)
                    did = True
        if self.noise_sampler and len(self.noise_sampler) and not did:
            n = self.noise_sampler.sample(len(y))
            if n is not None and _rms(n) > 1e-6:
                y = mix_at_snr(y, n, np.random.uniform(*self.noise_snr))
        return y

    @staticmethod
    def _znorm(y):
        return (y - y.mean()) / (y.std() + 1e-9)

    @staticmethod
    def collate(batch):
        return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
