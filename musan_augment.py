import json
import random
import numpy as np
import soundfile as sf
import os


class MusanNoiseAugmentor:
    def __init__(
        self,
        index_json: str,
        sample_rate: int = 16000,
        snr_ranges: dict = None,
        category_weights: dict = None,
        speech_num_clips_range: tuple = (3, 7),
        seed: int = None,
    ):
        with open(index_json, "r") as f:
            self.index = json.load(f)

        self.sr = sample_rate
        self.snr_ranges = snr_ranges or {
            "noise": (0, 15),
            "music": (5, 15),
            "speech": (13, 20),
        }
        self.category_weights = category_weights or {
            "noise": 0.5, "music": 0.25, "speech": 0.25
        }
        self.speech_num_clips_range = speech_num_clips_range
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        for cat in self.category_weights:
            if cat not in self.index or len(self.index[cat]) == 0:
                raise ValueError(f"MUSAN category '{cat}' has no files indexed. Please check your musan_index.json")

        assert set(self.category_weights.keys()) <= set(self.snr_ranges.keys()), \
            "musan_category_weights keys must be a subset of musan_snr_ranges keys"

    def _load_random_clip(self, category: str, min_len: int) -> np.ndarray:
        """Load a random clip from `category`, tiling/cropping to at least min_len."""
        entry = self.rng.choice(self.index[category])
        wav, _ = sf.read(entry["path"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)  # ensure mono

        # tile if shorter than needed
        if len(wav) < min_len:
            n_tiles = int(np.ceil(min_len / len(wav)))
            wav = np.tile(wav, n_tiles)

        # random crop to exact min_len
        if len(wav) > min_len:
            start = self.rng.randint(0, len(wav) - min_len)
            wav = wav[start:start + min_len]

        return wav

    def _build_noise(self, category: str, length: int) -> np.ndarray:
        if category == "speech":
            n_clips = self.rng.randint(*self.speech_num_clips_range)
            babble = np.zeros(length, dtype=np.float32)
            for _ in range(n_clips):
                babble += self._load_random_clip("speech", length)
            return babble
        else:
            return self._load_random_clip(category, length)

    @staticmethod
    def _mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
        sig_power = np.mean(signal ** 2) + 1e-10
        noise_power = np.mean(noise ** 2) + 1e-10
        snr_linear = 10 ** (snr_db / 10)
        scale = np.sqrt(sig_power / (noise_power * snr_linear))
        mixed = signal + scale * noise

        # peak-normalize defensively to avoid clipping if input wasn't [-1, 1]
        peak = np.max(np.abs(mixed))
        if peak > 1.0:
            mixed = mixed / peak
        return mixed.astype(np.float32)

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """Apply MUSAN augmentation to a padded/cut waveform (1D np.float32)."""
        categories = list(self.category_weights.keys())
        weights = list(self.category_weights.values())
        category = self.rng.choices(categories, weights=weights, k=1)[0]

        low, high = self.snr_ranges[category]
        snr_db = self.rng.uniform(low, high)

        noise = self._build_noise(category, length=len(signal))
        return self._mix_at_snr(signal, noise, snr_db)
