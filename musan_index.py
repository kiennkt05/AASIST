"""
Build the MUSAN clip index, filtering out clips whose raw average power
is too low relative to the rest of their category. Low-power clips force
very large `scale` values during SNR mixing (scale = sqrt(sig_power /
(noise_power * snr_linear))), which causes destructive clipping and
corrupts the achieved SNR -- confirmed empirically to be the actual
root cause, NOT crest factor (peak/RMS), which was tested and rejected
as a filtering criterion.
"""

import json
from pathlib import Path
import numpy as np
import soundfile as sf


def _clip_power(path: str) -> float:
    wav, _ = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return float(np.mean(wav.astype(np.float64) ** 2) + 1e-12)


def build_musan_index(
    musan_root: str,
    out_json: str = "musan_index.json",
    min_power_percentile: float = 10.0,
):
    """Build MUSAN index with per-category minimum-power filtering.

    Clips below the `min_power_percentile`-th percentile of average power
    within their own category are excluded. Default (p10) keeps ~90% of
    clips per category while eliminating the vast majority of pathological
    scale-cap triggers seen during SNR mixing (empirically verified via
    diagnose_musan_full.py: p10 dropped noise's cap-trigger rate from
    5.0% -> 0.4%, and reduced worst-case deviation from tens of dB to ~1.5dB).
    """
    musan_root = Path(musan_root)
    raw_entries = {"noise": [], "music": [], "speech": []}

    for category in raw_entries.keys():
        for wav_path in (musan_root / category).rglob("*.wav"):
            info = sf.info(str(wav_path))
            power = _clip_power(str(wav_path))
            raw_entries[category].append({
                "path": str(wav_path),
                "frames": info.frames,
                "samplerate": info.samplerate,
                "power": power,
            })

    index = {}
    for category, entries in raw_entries.items():
        powers = np.array([e["power"] for e in entries])
        threshold = np.percentile(powers, min_power_percentile)

        kept = [e for e in entries if e["power"] >= threshold]
        n_excluded = len(entries) - len(kept)

        # strip the diagnostic 'power' field before saving -- not needed at
        # runtime by MusanNoiseAugmentor, keeps the index file lean
        for e in kept:
            del e["power"]

        index[category] = kept
        print(f"{category}: kept {len(kept)}/{len(entries)}  "
              f"(excluded {n_excluded} below p{min_power_percentile} "
              f"power threshold={threshold:.6f})")

    with open(out_json, "w") as f:
        json.dump(index, f)
    return index