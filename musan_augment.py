import json
import random
import numpy as np
import soundfile as sf


class MusanNoiseAugmentor:
    def __init__(
        self,
        index_json: str,
        sample_rate: int = 16000,
        snr_ranges: dict = None,
        category_weights: dict = None,
        speech_num_clips_range: tuple = (3, 7),
        seed: int = None,
        max_scale: float = 6.0,
        max_retries: int = 8,
        correlation_warn_threshold: float = 0.6,
        hf_cutoff_hz: float = 4000.0,
    ):
        with open(index_json, "r") as f:
            self.index = json.load(f)

        # pull per-category crop-level power thresholds out of the index,
        # written there by build_musan_index(); default to {} (no
        # rejection sampling) if the index predates this feature
        self.min_power_thresholds = self.index.pop("_meta", {}).get(
            "min_power_thresholds", {}
        )

        self.sr = sample_rate
        self.snr_ranges = snr_ranges or {
            "noise": (3, 15),
            "music": (5, 15),
            "speech": (13, 20),
        }
        self.category_weights = category_weights or {
            "noise": 0.5, "music": 0.25, "speech": 0.25
        }
        self.speech_num_clips_range = speech_num_clips_range
        self.max_scale = max_scale
        self.max_retries = max_retries
        self.correlation_warn_threshold = correlation_warn_threshold
        self.hf_cutoff_hz = hf_cutoff_hz
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        # visibility counters
        self._n_capped = 0
        self._n_total = 0
        self._n_retry_exhausted = 0
        self._n_low_correlation = 0
        self._n_correlation_checked = 0

        for cat in self.category_weights:
            if cat not in self.index or len(self.index[cat]) == 0:
                raise ValueError(
                    f"MUSAN category '{cat}' has no files indexed. "
                    f"Please check your musan_index.json"
                )

        assert set(self.category_weights.keys()) <= set(self.snr_ranges.keys()), \
            "musan_category_weights keys must be a subset of musan_snr_ranges keys"

    def _load_random_clip(self, category: str, min_len: int,
                           min_power: float = None) -> np.ndarray:
        """Load a random clip from `category`, tiling/cropping to at least
        min_len. If min_power is set, retries up to self.max_retries times
        to find a CROP (not just a file) whose own power meets the
        threshold -- since a file's whole-file average power doesn't
        guarantee any particular random crop from it is loud enough."""
        last_wav = None
        for attempt in range(self.max_retries):
            entry = self.rng.choice(self.index[category])
            wav, _ = sf.read(entry["path"], dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)

            if len(wav) < min_len:
                n_tiles = int(np.ceil(min_len / len(wav)))
                wav = np.tile(wav, n_tiles)

            if len(wav) > min_len:
                start = self.rng.randint(0, len(wav) - min_len)
                wav = wav[start:start + min_len]

            last_wav = wav
            if min_power is None:
                return wav

            crop_power = np.mean(wav.astype(np.float64) ** 2) + 1e-12
            if crop_power >= min_power:
                return wav
            # else: retry with a new random clip/crop draw

        self._n_retry_exhausted += 1
        return last_wav

    def _build_noise(self, category: str, length: int) -> np.ndarray:
        min_power = self.min_power_thresholds.get(category)
        if category == "speech":
            n_clips = self.rng.randint(*self.speech_num_clips_range)
            babble = np.zeros(length, dtype=np.float32)
            for _ in range(n_clips):
                babble += self._load_random_clip("speech", length, min_power=min_power)
            return babble
        else:
            return self._load_random_clip(category, length, min_power=min_power)

    def _mix_at_snr(self, signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
        sig_power = np.mean(signal ** 2) + 1e-10
        noise_power = np.mean(noise ** 2) + 1e-10
        snr_linear = 10 ** (snr_db / 10)
        scale = np.sqrt(sig_power / (noise_power * snr_linear))

        self._n_total += 1
        if scale > self.max_scale:
            # this noise clip is too "peaky" relative to the target SNR to
            # mix cleanly at the exact requested level — cap the boost
            # rather than let it force destructive clipping. Logged via
            # self._n_capped so the trigger rate can be monitored.
            scale = self.max_scale
            self._n_capped += 1

        mixed = signal + scale * noise

        # clip only the samples that actually exceed range, rather than
        # globally rescaling the whole waveform (which would also rescale
        # the signal component and corrupt the achieved SNR)
        mixed = np.clip(mixed, -1.0, 1.0)
        return mixed.astype(np.float32)

    def _hf_correlation(self, clean: np.ndarray, mixed: np.ndarray) -> float:
        """Cosine similarity between the high-frequency (>hf_cutoff_hz)
        magnitude spectra of `clean` and `mixed`. Measures whether the
        augmented signal's HF structure still resembles the original's
        (spoof artifacts plausibly intact) vs. having been overwritten by
        unrelated noise content. 1.0 = fully preserved, 0.0 = unrelated."""
        freqs = np.fft.rfftfreq(len(clean), d=1 / self.sr)
        mask = freqs >= self.hf_cutoff_hz
        c = np.abs(np.fft.rfft(clean))[mask]
        m = np.abs(np.fft.rfft(mixed))[mask]
        c = c / (np.linalg.norm(c) + 1e-12)
        m = m / (np.linalg.norm(m) + 1e-12)
        return float(np.dot(c, m))

    def apply(self, signal: np.ndarray, track_correlation: bool = False) -> np.ndarray:
        """Apply MUSAN augmentation to a padded/cut waveform (1D np.float32).

        If track_correlation=True, additionally computes the HF structural
        correlation between the clean and augmented signal and updates
        self.low_correlation_rate accordingly. This adds an FFT's worth of
        overhead, so it's meant to be sampled on a subset of calls (e.g.
        ~2% of training steps) rather than every call -- see
        Dataset.__getitem__ for the sampling pattern.
        """
        categories = list(self.category_weights.keys())
        weights = list(self.category_weights.values())
        category = self.rng.choices(categories, weights=weights, k=1)[0]

        low, high = self.snr_ranges[category]
        snr_db = self.rng.uniform(low, high)

        noise = self._build_noise(category, length=len(signal))
        mixed = self._mix_at_snr(signal, noise, snr_db)

        if track_correlation:
            corr = self._hf_correlation(signal, mixed)
            self._n_correlation_checked += 1
            if corr < self.correlation_warn_threshold:
                self._n_low_correlation += 1

        return mixed

    @property
    def cap_trigger_rate(self) -> float:
        """Fraction of _mix_at_snr calls where the scale cap fired
        (noise clip too quiet/peaky relative to target SNR to mix
        cleanly without forcing destructive clipping)."""
        return self._n_capped / self._n_total if self._n_total > 0 else 0.0

    @property
    def low_correlation_rate(self) -> float:
        """Fraction of tracked apply() calls where high-frequency
        structural correlation between clean and augmented signal fell
        below self.correlation_warn_threshold -- a proxy for how often
        augmentation may be overwriting rather than merely perturbing
        signal-relevant high-frequency structure. Only meaningful if
        apply() has been called with track_correlation=True at least
        occasionally; returns 0.0 if never tracked."""
        if self._n_correlation_checked == 0:
            return 0.0
        return self._n_low_correlation / self._n_correlation_checked