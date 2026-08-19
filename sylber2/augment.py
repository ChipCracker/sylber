"""Data augmentation for Sylber 2.0 (Sec. 3.2.1).

- Speaker identity perturbation by modifying formant levels (p=0.3), in the
  style of Qian et al. 2022 / Komatsu & Shinozaki 2024 (NANSY-like random
  formant/pitch perturbation via Praat's "Change gender").
- Environmental noise (p=0.2) or randomly cropped speech clips (p=0.05); if
  both fire, one is chosen with equal probability.
- Room impulse responses sampled from the GTU-RIR corpus (probability not
  specified in the paper; default 0.25 here, configurable).
- Random white noise (p=0.3).

All operations preserve the signal length (duration factor 1.0) so that frame
alignment is kept.
"""
import signal
import threading
import numpy as np
from pathlib import Path
from scipy.signal import fftconvolve
import soundfile as sf
import librosa


class _PraatTimeout(Exception):
    pass


class _alarm_guard:
    """Hard SIGALRM timeout around native calls that may deadlock (Praat is
    not fork-safe and can hang after thousands of calls). Only active in a
    process main thread; otherwise a no-op."""

    def __init__(self, seconds):
        self.seconds = seconds
        self.active = threading.current_thread() is threading.main_thread()

    def _raise(self, *a):
        raise _PraatTimeout()

    def __enter__(self):
        if self.active:
            self._old = signal.signal(signal.SIGALRM, self._raise)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, *exc):
        if self.active:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self._old)
        return False

try:
    import parselmouth
    from parselmouth.praat import call as praat_call
    HAS_PARSELMOUTH = True
except ImportError:  # pragma: no cover
    HAS_PARSELMOUTH = False


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def mix_at_snr(clean, noise, snr_db):
    """Mix `noise` into `clean` at the given SNR (lengths must match)."""
    g = _rms(clean) / (_rms(noise) * (10 ** (snr_db / 20)) + 1e-12)
    return clean + g * noise


def random_formant_perturb(wav, sr,
                           formant_range=(1.0, 1.4),
                           pitch_shift_range=(1.0, 2.0),
                           pitch_range_range=(1.0, 1.5),
                           perturb_pitch=True):
    """NANSY-style speaker perturbation via Praat 'Change gender'.

    Ratios are sampled uniformly and inverted with p=0.5 (i.e. r^{+/-1}).
    Falls back to the input on failure (e.g. unvoiced audio).
    """
    if not HAS_PARSELMOUTH:
        return wav
    fs = np.random.uniform(*formant_range) ** np.random.choice([-1, 1])
    if perturb_pitch:
        ps = np.random.uniform(*pitch_shift_range) ** np.random.choice([-1, 1])
        pr = np.random.uniform(*pitch_range_range) ** np.random.choice([-1, 1])
    else:
        ps, pr = 1.0, 1.0
    try:
        with _alarm_guard(10):
            snd = parselmouth.Sound(wav.astype(np.float64), sampling_frequency=sr)
            pitch = snd.to_pitch()
            f0_vals = pitch.selected_array['frequency']
            f0_vals = f0_vals[f0_vals > 0]
            median_f0 = float(np.median(f0_vals)) if len(f0_vals) else 0.0
            new_median = median_f0 * ps if median_f0 > 0 else 0.0
            out = praat_call(snd, "Change gender", 75, 600, fs, new_median, pr, 1.0)
            y = out.values[0].astype(np.float32)
    except Exception:
        return wav
    if len(y) < len(wav):
        y = np.pad(y, (0, len(wav) - len(y)))
    return y[:len(wav)]


class FileSampler:
    """Uniformly samples audio files from a directory tree, loads at `sr`."""

    def __init__(self, root, sr, exts=(".wav", ".flac", ".ogg")):
        self.files = [f for ext in exts for f in Path(root).rglob(f"*{ext}")] if root else []
        self.sr = sr

    def __len__(self):
        return len(self.files)

    def sample(self, num_samples):
        if not self.files:
            return None
        for _ in range(3):
            f = self.files[np.random.randint(len(self.files))]
            try:
                y, sr = sf.read(f, dtype="float32", always_2d=True)
            except Exception:
                continue
            y = y.mean(-1)
            if sr != self.sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.sr)
            if len(y) == 0:
                continue
            if len(y) >= num_samples:
                p = np.random.randint(len(y) - num_samples + 1)
                return y[p:p + num_samples]
            out = np.zeros(num_samples, dtype=np.float32)
            p = np.random.randint(num_samples - len(y) + 1)
            out[p:p + len(y)] = y
            return out
        return None


class ContentAugment:
    """Full augmentation pipeline for content-encoder training."""

    def __init__(self, sr=16000,
                 noise_dir=None, rir_dir=None, speech_dir=None,
                 formant_prob=0.3,
                 noise_prob=0.2,
                 speech_clip_prob=0.05,
                 rir_prob=0.25,
                 white_noise_prob=0.3,
                 noise_snr=(5, 25),
                 speech_snr=(5, 20),
                 white_snr=(10, 40),
                 perturb_pitch=True):
        self.sr = sr
        self.formant_prob = formant_prob
        self.noise_prob = noise_prob
        self.speech_clip_prob = speech_clip_prob
        self.rir_prob = rir_prob
        self.white_noise_prob = white_noise_prob
        self.noise_snr = noise_snr
        self.speech_snr = speech_snr
        self.white_snr = white_snr
        self.perturb_pitch = perturb_pitch
        self.noise_sampler = FileSampler(noise_dir, sr) if noise_dir else None
        self.rir_sampler = FileSampler(rir_dir, sr) if rir_dir else None
        self.speech_sampler = FileSampler(speech_dir, sr) if speech_dir else None

    def __call__(self, wav):
        wav = wav.astype(np.float32)
        if np.random.uniform() < self.formant_prob:
            wav = random_formant_perturb(wav, self.sr, perturb_pitch=self.perturb_pitch)

        # environmental noise (p=0.2) XOR cropped speech clip (p=0.05)
        add_noise = np.random.uniform() < self.noise_prob and self.noise_sampler and len(self.noise_sampler)
        add_speech = np.random.uniform() < self.speech_clip_prob and self.speech_sampler and len(self.speech_sampler)
        if add_noise and add_speech:
            if np.random.uniform() < 0.5:
                add_speech = False
            else:
                add_noise = False
        if add_noise:
            n = self.noise_sampler.sample(len(wav))
            if n is not None and _rms(n) > 1e-6:
                wav = mix_at_snr(wav, n, np.random.uniform(*self.noise_snr))
        elif add_speech:
            n = self.speech_sampler.sample(len(wav))
            if n is not None and _rms(n) > 1e-6:
                wav = mix_at_snr(wav, n, np.random.uniform(*self.speech_snr))

        if self.rir_sampler and len(self.rir_sampler) and np.random.uniform() < self.rir_prob:
            rir = self.rir_sampler.sample_rir()
            if rir is not None:
                wet = fftconvolve(wav, rir)[:len(wav)]
                if _rms(wet) > 1e-6:
                    wav = (wet * (_rms(wav) / _rms(wet))).astype(np.float32)

        if np.random.uniform() < self.white_noise_prob:
            wn = np.random.randn(len(wav)).astype(np.float32)
            wav = mix_at_snr(wav, wn, np.random.uniform(*self.white_snr))
        return wav.astype(np.float32)


def _sample_rir(self, max_seconds=1.0):
    """Load a random RIR, trimmed to its main support and normalized."""
    if not self.files:
        return None
    for _ in range(3):
        f = self.files[np.random.randint(len(self.files))]
        try:
            y, sr = sf.read(f, dtype="float32", always_2d=True)
        except Exception:
            continue
        y = y.mean(-1)
        if sr != self.sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sr)
        if len(y) == 0:
            continue
        # align direct path to start, trim tail
        onset = int(np.argmax(np.abs(y)))
        y = y[onset:onset + int(max_seconds * self.sr)]
        norm = np.sqrt((y ** 2).sum() + 1e-12)
        return (y / norm).astype(np.float32)
    return None


FileSampler.sample_rir = _sample_rir
