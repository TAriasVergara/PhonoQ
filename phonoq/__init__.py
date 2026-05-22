"""
phonoq - optimized, batched PhonoQ phonological-posterior speech features.

Original: TAriasVergara/PhonoQ (https://github.com/TAriasVergara/PhonoQ)
This package: refactored for speed (model caching, GPU support, batched inference).

Usage:
    from phonoq import PhonoQ
    pq = PhonoQ(lang="ES", device="cuda")
    feats, posteriors = pq.extract_file("audio.wav")
    feats_all = pq.extract_files(["a.wav", "b.wav", ...])

The 'features' return is a dict with keys like 'pVowels', 'durStop_mean', etc.
The 'posteriors' is a (T, 18) numpy array of phoneme-class probabilities per
10-ms frame: manner classes 0-7, place 8-15, voicing 16-17.
"""
from .core import PhonoQ, PhonoQFeatures

__all__ = ["PhonoQ", "PhonoQFeatures"]
__version__ = "0.1.0"
