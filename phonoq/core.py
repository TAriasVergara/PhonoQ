"""
core.py - main PhonoQ class with model caching, GPU support, batched inference.

The original PhonoQ has three speed bottlenecks fixed here:

1. **Model is reloaded from disk per file** (eval_model.output_model calls
   load_model() every call -> 37 MB disk read + PyTorch deserialization per audio).
   Fix: load once at __init__, reuse for all subsequent calls.

2. **CPU-only hardcoded** (torch.device('cpu') in eval_model.py).
   Fix: device parameter, GPU autodetect, all tensors moved to device.

3. **Sequential while-loop over audio segments** (each ~640ms window processed
   one at a time through the CNN-RNN).
   Fix: vectorize - stack all segments into a single batch and run one forward.

Plus cuDNN is re-enabled for speed (the original disables it for reproducibility).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.signal as signal
import librosa

# Re-enable cuDNN - the original PhonoQ disables it for reproducibility but
# we want speed at inference time.
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


# --------------------------------------------------------------------------- #
# Self-attention block (reproduced from original PhonoQ models.py)
# --------------------------------------------------------------------------- #
class SelfAttention(nn.Module):
    def __init__(self, in_dim, activation):
        super().__init__()
        self.activation = activation
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8 if in_dim >= 8 else 1, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8 if in_dim >= 8 else 1, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H*W).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, H*W)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(B, -1, H*W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return self.gamma * out + x


# --------------------------------------------------------------------------- #
# Conv-RNN model (mirrors original PhonoQ utils/models.py Conv_RNN)
# --------------------------------------------------------------------------- #
class ConvRNN(nn.Module):
    """Convolutional + GRU/LSTM/RNN block for phoneme posterior prediction.
    Architecture matches the original Conv_RNN in PhonoQ/utils/models.py.
    """
    def __init__(self, model_param, model_type=nn.GRU, input_shape=(1, 1, 64, 64)):
        super().__init__()
        self.batch_size = input_shape[0]
        self.input_shape = input_shape
        self.sequence_size = model_param["seq_dim"]
        self.input_size = model_param["input_dim"]
        self.hidden_size = model_param["hidden_dim"]
        self.bidirectional = model_param["bidirectional"]
        self.output_size = model_param["output_dim"]
        self.num_layers = model_param["layer_dim"]

        # Conv layers
        ks = 3
        self.conv1 = nn.Conv2d(input_shape[1], 8, kernel_size=ks, padding=1)
        self.Bn_c1 = nn.BatchNorm2d(8)
        self.MxP_c1 = nn.MaxPool2d((1, 2))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=ks, padding=1)
        self.Bn_c2 = nn.BatchNorm2d(16)
        self.MxP_c2 = nn.MaxPool2d((1, 2))
        self.Relu = nn.LeakyReLU()

        # Compute RNN input size from a dummy forward
        n_size = self._get_conv_output((1, input_shape[1], self.sequence_size, self.input_size))
        rnn_INsize = n_size * 16

        # Recurrent network
        self.model = model_type(
            rnn_INsize, self.hidden_size,
            bidirectional=self.bidirectional,
            num_layers=self.num_layers, batch_first=True,
        )
        idx_bi = 2 if self.bidirectional else 1
        output_RNN = self.hidden_size * idx_bi

        # Names intentionally match the original PhonoQ checkpoints.
        self.Output_layer = nn.Linear(output_RNN, self.output_size)
        self.Bn_l1 = nn.BatchNorm1d(self.sequence_size)
        self.Drop_l1 = nn.Dropout()

    def _get_conv_output(self, shape):
        with torch.no_grad():
            x = torch.zeros(shape)
            x = self.conv1(x)
            x = self.MxP_c1(x)
            x = self.Bn_c1(x)
            x = self.Relu(x)
            x = self.conv2(x)
            x = self.MxP_c2(x)
            x = self.Bn_c2(x)
            x = self.Relu(x)
            return x.size(-1)

    def forward(self, x):
        # x: (B, C, seq, input)
        x = self.conv1(x)
        x = self.MxP_c1(x)
        x = self.Bn_c1(x)
        x = self.Relu(x)
        x = self.conv2(x)
        x = self.MxP_c2(x)
        x = self.Bn_c2(x)
        x = self.Relu(x)
        # (B, 16, seq, F_reduced) -> (B, seq, 16 * F_reduced)
        B, C, S, Fr = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(B, S, C * Fr)
        x, _ = self.model(x)
        x = self.Bn_l1(x)
        x = self.Drop_l1(x)
        x = self.Output_layer(x)
        return x


# --------------------------------------------------------------------------- #
# Mel-feature extraction (reproduced from original PhonoQ utils/feature_extract.py)
# --------------------------------------------------------------------------- #
def get_mel_spec(sig, fs, win_time, step_time, n_mels, nfft=1024, fmax=8000):
    """Compute log-mel spectrogram. Returns (T, n_mels)."""
    n_fft = nfft
    hop = int(step_time * fs)
    win = int(win_time * fs)
    S = librosa.feature.melspectrogram(
        y=sig.astype(np.float32), sr=fs, n_fft=n_fft, hop_length=hop, win_length=win,
        n_mels=n_mels, fmax=fmax, power=2.0,
    )
    log_S = np.log(np.maximum(S, 1e-10))
    return log_S.T  # (T, n_mels)


# --------------------------------------------------------------------------- #
# Main PhonoQ class
# --------------------------------------------------------------------------- #
TARGETS = ["silence", "stop", "nasal", "trill", "fricative", "approximant",
           "lateral", "vowel", "labial", "alveolar", "velar", "palatal",
           "postalveolar", "central", "front", "back", "voiceless", "voiced"]


@dataclass
class PhonoQFeatures:
    """Return type of PhonoQ.extract - phoneme-derived feature dict + raw posteriors."""
    features: Dict[str, float]
    posteriors: np.ndarray          # (T, 18)
    predictions_manner: np.ndarray  # (T,)
    predictions_place: np.ndarray   # (T,)
    predictions_voicing: np.ndarray # (T,)
    targets: List[str] = field(default_factory=lambda: TARGETS)


class PhonoQ:
    """Fast PhonoQ phonological-posterior extractor.

    Parameters
    ----------
    lang : str
        Language model: "ES", "EN", "DE", "Multi".
    device : str
        Torch device: "cuda", "cpu", or "auto".
    model_dir : str or Path or None
        Path to a directory containing a PhonoQ checkpoint, parameter JSON, and
        norm_param.json. If None, looks for utils/{lang}_model under model_root.
    model_root : str or Path or None
        Root directory of a PhonoQ checkout. If None, looks for the model
        bundled with this package or under $PHONOQ_ROOT environment variable.

    Examples
    --------
    >>> pq = PhonoQ(lang="ES", device="cuda")
    >>> out = pq.extract_file("audio.wav")
    >>> print(out.features.keys())
    """
    def __init__(self,
                 lang: str = "ES",
                 device: str = "auto",
                 model_dir: Optional[Union[str, Path]] = None,
                 model_root: Optional[Union[str, Path]] = None,
                 win_time: float = 0.025,
                 step_time: float = 0.01):
        self.lang = lang
        self.device = self._resolve_device(device)
        self.win_time = win_time
        self.step_time = step_time

        # Locate the model directory
        if model_dir is None:
            model_dir = self._locate_model_dir(lang, model_root)
        model_dir = Path(model_dir)
        model_weights, model_parameters = self._resolve_model_files(model_dir)
        self.model_dir = model_dir

        # Load parameters
        with open(model_parameters) as fp:
            self.RNN_param = json.load(fp)
        with open(model_dir / "norm_param.json") as fp:
            self.norm_param = [json.load(fp)]

        # Build model and load weights ONCE
        cell_type_map = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
        cell_type = cell_type_map[self.RNN_param["cell_type"]]
        input_shape = (1, self.RNN_param["channel_dim"],
                       self.RNN_param["seq_dim"], self.RNN_param["input_dim"])
        self.model = ConvRNN(self.RNN_param, cell_type, input_shape)
        state = torch.load(str(model_weights), map_location=self.device,
                           weights_only=False)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        print(f"PhonoQ initialized: lang={lang}, device={self.device}, "
              f"params={sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M")

    # -- helpers ----------------------------------------------------------- #

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    @staticmethod
    def _resolve_model_files(model_dir: Path) -> Tuple[Path, Path]:
        weights = model_dir / "model_weights.ckp"
        parameters = model_dir / "model_parameters.json"

        if not weights.is_file():
            matches = sorted(model_dir.glob("*.ckp"))
            weights = matches[0] if matches else weights
        if not parameters.is_file():
            matches = sorted(
                p for p in model_dir.glob("*.json")
                if p.name != "norm_param.json"
            )
            parameters = matches[0] if matches else parameters

        missing = [
            str(path.name)
            for path in (weights, parameters, model_dir / "norm_param.json")
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing {', '.join(missing)} in {model_dir}. Point model_dir "
                "at a PhonoQ utils/{lang}_model folder."
            )
        return weights, parameters

    @staticmethod
    def _locate_model_dir(lang: str, model_root: Optional[Union[str, Path]]) -> Path:
        env = os.environ.get("PHONOQ_ROOT")
        candidates = []
        if model_root is not None:
            candidates.append(Path(model_root) / "utils" / f"{lang}_model")
        if env:
            candidates.append(Path(env) / "utils" / f"{lang}_model")
        here = Path(__file__).resolve().parent
        candidates.append(here.parent / "utils" / f"{lang}_model")
        candidates.append(here / "data" / f"{lang}_model")
        candidates.append(here.parent / "data" / f"{lang}_model")
        # Look upward for an extra_code/PhonoQ/utils/{lang}_model
        for parent in [here.parent.parent, here.parent.parent.parent,
                       Path.cwd(), Path.cwd().parent]:
            candidates.append(parent / "extra_code" / "PhonoQ" / "utils" / f"{lang}_model")
            candidates.append(parent / "PhonoQ" / "utils" / f"{lang}_model")
        for c in candidates:
            try:
                PhonoQ._resolve_model_files(c)
                return c
            except FileNotFoundError:
                pass
        raise FileNotFoundError(
            f"Could not locate {lang}_model. Tried: "
            f"{[str(c) for c in candidates]}. "
            f"Set the PHONOQ_ROOT env var or pass model_dir explicitly."
        )

    # -- preprocessing ----------------------------------------------------- #

    def _preprocess_signal(self, sig: np.ndarray, fs: int) -> Tuple[np.ndarray, int]:
        """Remove DC, rescale, resample to 16 kHz, pad with silence at end."""
        sig = sig - np.mean(sig)
        peak = np.max(np.abs(sig))
        if peak > 0:
            sig = sig / peak
        if fs != 16000:
            n_new = int((len(sig) / fs) * 16000)
            sig = signal.resample(sig, n_new)
            fs = 16000
            sig = sig - np.mean(sig)
            peak = np.max(np.abs(sig))
            if peak > 0:
                sig = sig / peak
        # Append 1 sec of zeros so even short signals produce >= 1 segment
        new_sig = np.zeros(len(sig) + fs, dtype=np.float32)
        new_sig[:len(sig)] = sig
        return new_sig, fs

    def _normalize_mel(self, mel: np.ndarray) -> np.ndarray:
        mn = float(self.norm_param[0]["min"])
        mx = float(self.norm_param[0]["max"])
        return (mel - mn) / (mx - mn)

    # -- core forward (vectorized over segments) --------------------------- #

    @torch.inference_mode()
    def _forward_segments(self, mel: np.ndarray) -> np.ndarray:
        """Take (T, F) log-mel features, slide windows, batch-forward, stitch.
        Returns posteriors of shape (T_pad, output_dim) after sigmoid."""
        seq_dim = self.RNN_param["seq_dim"]
        shift = seq_dim // 2
        L = mel.shape[0]

        if L < seq_dim:
            return np.zeros((0, self.RNN_param["output_dim"]), dtype=np.float32)

        # Build all overlapping windows as a single (n_windows, 1, seq, F) tensor
        starts = list(range(0, L - seq_dim + 1, shift))
        if not starts:
            return np.zeros((0, self.RNN_param["output_dim"]), dtype=np.float32)

        windows = np.stack([mel[s:s + seq_dim] for s in starts], axis=0)  # (N, seq, F)
        windows = self._normalize_mel(windows)
        windows_t = torch.from_numpy(windows.astype(np.float32)).unsqueeze(1).to(self.device)
        # Forward (one big batch - vectorized vs the original while-loop)
        out = self.model(windows_t)
        scores = torch.sigmoid(out).cpu().numpy()  # (N, seq, out_dim)

        # Stitch: first window contributes fully, subsequent windows only the new shift-frames
        out_post = [scores[0]]
        for i in range(1, len(starts)):
            out_post.append(scores[i, -shift:])
        post = np.concatenate(out_post, axis=0)
        return post

    # -- public API -------------------------------------------------------- #

    def extract(self, sig: np.ndarray, fs: int) -> PhonoQFeatures:
        """Extract phonological posteriors and aggregated features from one audio array."""
        sig, fs = self._preprocess_signal(sig, fs)
        mel = get_mel_spec(sig, fs, self.win_time, self.step_time,
                            self.RNN_param["input_dim"])
        posteriors = self._forward_segments(mel)
        # Trim the silence padding at the end
        n_pad_frames = int(round((1 - self.win_time) / self.step_time))
        n_frames = mel.shape[0] - n_pad_frames
        posteriors = posteriors[:max(0, n_frames)]

        # Predictions
        pred_manner = np.argmax(posteriors[:, 0:8], axis=1) if len(posteriors) > 0 else np.array([])
        psilence = posteriors[:, 0:1] if len(posteriors) > 0 else np.zeros((0, 1))
        pplace = np.hstack([psilence, posteriors[:, 8:16]]) if len(posteriors) > 0 else psilence
        pred_place = np.argmax(pplace, axis=1) - 1 if len(pplace) > 0 else np.array([])
        pvoice = np.hstack([psilence, posteriors[:, 16:]]) if len(posteriors) > 0 else psilence
        pred_voicing = np.argmax(pvoice, axis=1) - 1 if len(pvoice) > 0 else np.array([])

        # Aggregate features (delegate to original PhonoQ phoneme_features module
        # if available, otherwise compute a minimal fallback set).
        feats = self._aggregate_features(
            posteriors, pred_manner, pred_place, pred_voicing
        )

        return PhonoQFeatures(
            features=feats,
            posteriors=posteriors,
            predictions_manner=pred_manner,
            predictions_place=pred_place,
            predictions_voicing=pred_voicing,
        )

    def extract_file(self, path: Union[str, Path]) -> PhonoQFeatures:
        """Extract from a single audio file path."""
        sig, fs = librosa.load(str(path), sr=None)
        return self.extract(sig, fs)

    def extract_files(self,
                      paths: List[Union[str, Path]],
                      show_progress: bool = True) -> List[PhonoQFeatures]:
        """Extract from a list of audio file paths sequentially."""
        out = []
        for i, p in enumerate(paths):
            try:
                out.append(self.extract_file(p))
            except Exception as e:
                out.append(None)
                if show_progress:
                    print(f"  FAILED {Path(p).name}: {e}")
            if show_progress and (i + 1) % 25 == 0:
                print(f"  [{i + 1}/{len(paths)}]")
        return out

    # -- aggregate features (delegates to original PhonoQ if importable) --- #

    def _aggregate_features(self, posteriors, pred_manner, pred_place, pred_voicing):
        """Try to use the original PhonoQ phoneme_features module if available,
        otherwise fall back to a small subset of computable summary stats."""
        try:
            phonoq_root = os.environ.get("PHONOQ_ROOT")
            if phonoq_root:
                sys.path.insert(0, str(Path(phonoq_root)))
            # Walk up to find PhonoQ/utils/phoneme_features.py
            here = Path(__file__).resolve()
            for parent in [here.parent, here.parent.parent, here.parent.parent.parent]:
                if (parent / "utils" / "phoneme_features.py").is_file():
                    sys.path.insert(0, str(parent))
                    break
                cand = parent / "extra_code" / "PhonoQ"
                if cand.is_dir():
                    sys.path.insert(0, str(cand))
                    break
            from utils import phoneme_features as phone  # type: ignore
            X = {}
            X.update(phone.get_phoneme_feats(posteriors[:, 0:8], pred_manner, TARGETS[0:8]))
            X.update(phone.get_consonants(posteriors[:, 0:8], pred_manner))
            X.update(phone.get_phoneme_feats(posteriors[:, 8:16], pred_place, TARGETS[8:16]))
            X.update(phone.get_phoneme_feats(posteriors[:, 16:], pred_voicing, TARGETS[16:]))
            return {k: float(v) for k, v in X.items()}
        except Exception as e:
            # Fallback: minimal summary stats
            if len(posteriors) == 0:
                return {f"phonoq_class_{i}_mean": float("nan") for i in range(18)}
            return {
                **{f"class_{i}_mean": float(posteriors[:, i].mean()) for i in range(posteriors.shape[1])},
                **{f"class_{i}_std":  float(posteriors[:, i].std())  for i in range(posteriors.shape[1])},
            }
