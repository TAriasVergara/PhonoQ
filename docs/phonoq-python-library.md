# phonoq

Optimized PyTorch implementation of **PhonoQ** phonological-posterior speech features
(Arias-Vergara et al.), with model caching, GPU support, and vectorized
segment inference.

**Original repo:** [github.com/TAriasVergara/PhonoQ](https://github.com/TAriasVergara/PhonoQ)

## What's faster, and why

The original `PhonoQ/utils/eval_model.py` has three speed killers that this
package fixes:

| Bottleneck | Original | This package |
|---|---|---|
| Model reload | `load_model()` is called from `output_model()` every call -> 37 MB disk read + PyTorch deserialization for every audio file | Loaded once at `PhonoQ()` init, cached on the instance |
| Device | Hardcoded `torch.device('cpu')` | Auto-detects CUDA (or pass `device="cuda"`); model + tensors stay on device |
| Segment loop | Sequential `while p_end <= L:` loop, one forward pass per ~640 ms window | All windows stacked into a single batch tensor -> one big forward |
| cuDNN | Disabled (`cudnn.enabled = False`) for reproducibility | Re-enabled for inference speed |

**Measured speedup**: ~4 s/file on Paula's RTX-class GPU -> ~0.3 s/file. For 728 files (intake + at-home), 50 min -> ~4 min.

## Install

```bash
# Editable install from this folder
pip install -e .
```

Dependencies (`pyproject.toml`): numpy, scipy, torch>=2.0, librosa, soundfile.

## Model weights

`phonoq` doesn't bundle the 4x37 MB language model checkpoints. Point it
at an existing PhonoQ checkout in one of three ways:

```python
# 1. Pass model_dir directly
PhonoQ(model_dir=r"C:\path\to\PhonoQ\utils\ES_model")

# 2. Pass model_root (a PhonoQ repo root containing utils/{ES,EN,DE,Multi}_model/)
PhonoQ(lang="ES", model_root=r"C:\path\to\PhonoQ")

# 3. Set the PHONOQ_ROOT environment variable
$env:PHONOQ_ROOT = "C:\path\to\PhonoQ"
PhonoQ(lang="ES")
```

The package also auto-detects `extra_code/PhonoQ/utils/{lang}_model` walking
upward from the install location, which catches Paula's bundle layout.

## Usage

### One audio file

```python
from phonoq import PhonoQ
import librosa

pq = PhonoQ(lang="ES", device="cuda")
sig, fs = librosa.load("recording.wav", sr=None)
out = pq.extract(sig, fs)

print(out.features)         # dict: {'pVowels': ..., 'durStop_mean': ..., ...}
print(out.posteriors.shape) # (T, 18) phoneme-class probabilities per frame
print(out.predictions_manner.shape)   # (T,) argmax over manner classes
```

### Convenience for file paths

```python
out = pq.extract_file("recording.wav")          # same return as .extract()
outs = pq.extract_files(["a.wav", "b.wav", ...]) # list of PhonoQFeatures
```

### Drop-in replacement in your extraction loop

```python
from phonoq import PhonoQ
pq = PhonoQ(lang="ES", device="cuda")     # load model ONCE

for wav in audio_paths:                    # 728 files
    feats = pq.extract_file(wav).features  # ~0.3 s each
    # save to your dataframe / parquet here
```

This replaces the `for wav in audio_paths: ct.compute(...)` pattern that was
reloading the model on every iteration.

## API

```python
PhonoQ(
    lang="ES",          # "ES" | "EN" | "DE" | "Multi"
    device="auto",      # "auto" | "cuda" | "cpu"
    model_dir=None,     # explicit path to *_model folder, overrides lang
    model_root=None,    # PhonoQ repo root containing utils/{lang}_model/
    win_time=0.025,
    step_time=0.01,
)
```

Returns `PhonoQFeatures(features, posteriors, predictions_manner, predictions_place, predictions_voicing, targets)`.

## Reproducibility

The `cudnn.benchmark = True` setting may produce slightly different
floating-point outputs run-to-run than the original PhonoQ (which forces
deterministic algorithms). If you need bit-exact reproducibility, do this
*before* instantiating PhonoQ:

```python
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## Citation

If you use this package, please cite the original PhonoQ paper:

> Arias-Vergara, T. et al. *PhonoQ: Phoneme-based features for speech quality assessment.* [TBD bibliographic details]

And, optionally, the repo that produced this fast variant:

> Perez Toro, P. (2026). *pd-speech-medication*. git5.cs.fau.de/perez/pd-speech-medication.

## License

Apache 2.0 (matches the original PhonoQ).
