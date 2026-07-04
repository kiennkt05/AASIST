"""
Build the MUSAN clip index. Does NOT exclude files based on whole-file
average power -- that was tested and found ineffective, since runtime
mixing uses a random 4-second CROP of each file, not the whole file, and
a file's average power doesn't predict a given crop's power (a file can
be loud on average but contain long near-silent stretches a crop can land
in). Instead, per-category power thresholds are computed and stored in
the index's `_meta` field, for use as a CROP-level rejection-sampling
threshold at runtime inside MusanNoiseAugmentor._load_random_clip.
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
    """Build MUSAN index. Keeps ALL clips (no file-level exclusion) but
    computes a per-category min-power threshold (from whole-file averages,
    as a reasonable proxy for setting the threshold) and stores it in
    index['_meta']['min_power_thresholds']. MusanNoiseAugmentor uses this
    threshold to reject-and-retry individual CROPS at runtime that fall
    below it, which is the level at which power actually matters for
    the SNR-mixing scale calculation.
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
    thresholds = {}
    for category, entries in raw_entries.items():
        powers = np.array([e["power"] for e in entries])
        threshold = float(np.percentile(powers, min_power_percentile))
        thresholds[category] = threshold

        for e in entries:
            del e["power"]

        index[category] = entries
        print(f"{category}: {len(entries)} files indexed "
              f"(min_power_threshold=p{min_power_percentile}={threshold:.6f}, "
              f"applied at CROP level, not file-exclusion level)")

    index["_meta"] = {"min_power_thresholds": thresholds}

    with open(out_json, "w") as f:
        json.dump(index, f)
    return index