"""Inference API for Sylber 2.0.

Segmenter2 mirrors the Sylber 1.0 `Segmenter` interface: it takes audio and
returns syllable segments plus embeddings; with a synthesis checkpoint it can
also resynthesize 24 kHz audio from the syllabic embeddings.
"""
import numpy as np
import soundfile as sf
import torch
import torchaudio

from .models.sylber2_model import Sylber2
from .models.synthesis_model import SynthesisModel
from .models.acoustic_encoder import AcousticEncoder
from .models.vocoder import SylberVocoder, make_frame_inputs


def _strip_prefix(sd, prefix):
    out = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    return out or sd


class Segmenter2:

    def __init__(self, content_ckpt, synthesis_ckpt=None, device="cuda",
                 model_configs=None, acoustic_configs=None, vocoder_configs=None,
                 inference_prominence=0.1):
        if "cuda" in device and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.model = Sylber2(stage=4, load_pretrained=False, **(model_configs or {}))
        state = torch.load(content_ckpt, map_location="cpu", weights_only=False)
        sd = _strip_prefix(state.get("state_dict", state), "net.")
        self.model.load_state_dict(
            {k: v for k, v in sd.items() if k.startswith("student.")}, strict=False)
        self.model.eval().to(device)
        self.inference_prominence = inference_prominence

        self.synthesis = None
        if synthesis_ckpt is not None:
            ac = dict(acoustic_configs or {}); ac["load_pretrained"] = False
            syn = SynthesisModel(content_encoder=self.model,
                                 acoustic_encoder=AcousticEncoder(**ac),
                                 vocoder=SylberVocoder(**(vocoder_configs or {})))
            state = torch.load(synthesis_ckpt, map_location="cpu", weights_only=False)
            sd = state.get("state_dict", state)
            sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
            syn.load_state_dict(sd, strict=False)
            self.synthesis = syn.eval().to(device)

    def _load(self, wav_file=None, wav=None, sr=None, target_sr=16000):
        if wav_file is not None:
            arr, file_sr = sf.read(wav_file, dtype="float32", always_2d=True)
            y = torch.from_numpy(arr.mean(-1))
            sr = file_sr
        else:
            y = torch.as_tensor(wav, dtype=torch.float32)
            if y.ndim == 2:
                y = y.mean(0)
        if sr != target_sr:
            y = torchaudio.functional.resample(y, sr, target_sr)
        return y

    @torch.no_grad()
    def __call__(self, wav_file=None, wav=None, sr=16000, in_second=True):
        y16 = self._load(wav_file, wav, sr, 16000)
        y16 = (y16 - y16.mean()) / (y16.std() + 1e-9)
        res = self.model.segment(y16[None].to(self.device),
                                 use_boundary_detector=True,
                                 inference_prominence=self.inference_prominence)[0]
        segments = res["segments"]
        out = {
            "segments": segments * (1.0 / 50) if in_second else segments,
            "durations": (segments[:, 1] - segments[:, 0]) if len(segments) else np.array([]),
            "segment_features": res["segment_features"].cpu().numpy() if len(segments) else np.array([]),
            "content_embedding": (res["content_embedding"].cpu().numpy()
                                  if res["content_embedding"] is not None else None),
            "hidden_states": res["frames"].cpu().numpy(),
        }
        return out

    @torch.no_grad()
    def resynthesize(self, wav_file=None, wav=None, sr=24000):
        """Encode to ~5 Hz syllabic embeddings and decode back to 24 kHz audio."""
        assert self.synthesis is not None, "synthesis_ckpt required"
        y24 = self._load(wav_file, wav, sr, 24000)
        y16 = torchaudio.functional.resample(y24, 24000, 16000)
        y16 = (y16 - y16.mean()) / (y16.std() + 1e-9)
        audio, segments = self.synthesis(y16[None].to(self.device),
                                         y24[None].to(self.device), augment=False)
        return audio[0].cpu().numpy(), segments[0]
