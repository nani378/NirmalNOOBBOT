"""
feedback_learning.py — Persistent adaptive learning through user feedback.

When FEEDBACK_ENABLED = True in config.py the user can press:
    H → store / correct current prediction to Happy
    S → store / correct current prediction to Sad
    A → store / correct current prediction to Angry
    N → acknowledge No Emotion (resets stability votes; no sample stored)

Each key-press extracts the raw facial feature vector produced by
EmotionDetector (6 geometric values from MediaPipe Face Mesh), appends it
to the matching .npy dataset on disk, and recomputes the emotion centroid
in-place — no GPU, no heavy retraining.

On start-up the saved datasets are loaded, centroids are computed, and
injected directly into EmotionDetector._emotion_centroids so future
predictions are immediately biased toward the user's personal face geometry.

Dataset layout
──────────────
    data/
        happy_samples.npy   → float32 array, shape (N, 6)
        sad_samples.npy     → float32 array, shape (N, 6)
        angry_samples.npy   → float32 array, shape (N, 6)

The 6 feature columns (in fixed order):
    corner_lift, brow_slope, brow_gap, eye_open, lip_h, crinkle

Safety guards
─────────────
• Requires a valid detected face    (raw_features is not None)
• Requires confidence ≥ min_confidence
• Requires stable emotion flag        (emotion held across STABLE_FRAMES frames)
• Skips identical frames              (duplicate-key-press protection)
"""

import os

import numpy as np

# ── Canonical feature-key order (must match EmotionDetector._extract_raw) ─────
EMOTIONS: list[str] = ["happy", "sad", "angry"]

_FEATURE_KEYS: list[str] = [
    "corner_lift", "brow_slope",  "brow_gap",
    "eye_open",    "lip_h",       "crinkle",
    "brow_raise",  "mouth_width",             # added with v2 geometry
]


def _to_vec(raw: dict) -> np.ndarray:
    """Convert a feature-key dict → fixed-order float32 (6,) vector."""
    return np.array([raw[k] for k in _FEATURE_KEYS], dtype=np.float32)


def _to_centroid(arr: np.ndarray) -> dict:
    """
    Compute mean / std centroid from a (N, 6) sample array.
    Returns a dict compatible with EmotionDetector._emotion_centroids format:
        {"mean": {key: float, ...}, "std": {key: float, ...}}
    std is clipped to ≥ 1e-4 to avoid zero-division in Gaussian similarity.
    """
    mean_vec = arr.mean(axis=0)
    std_vec  = np.maximum(arr.std(axis=0), 1e-4)
    return {
        "mean": {k: float(mean_vec[i]) for i, k in enumerate(_FEATURE_KEYS)},
        "std":  {k: float(std_vec[i])  for i, k in enumerate(_FEATURE_KEYS)},
    }


class FeedbackLearner:
    """
    Persistent feedback-based learning module.

    Adaptive learning mechanism
    ───────────────────────────
    Emotion detection normally uses population-level geometric thresholds.
    After feedback samples accumulate, a *personal centroid* is computed as
    the mean feature vector of all confirmed samples for that emotion.

    At inference time (inside EmotionDetector._centroid_similarity) a
    Gaussian similarity score is computed between the current raw geometry
    and the centroid.  This score multiplies the raw landmark score by up
    to ×1.5, pulling the detector toward the user's idiosyncratic expression
    patterns without any retraining.

    The update is purely numpy (O(n) where n = total samples) — safe for
    Raspberry Pi real-time processing.
    """

    def __init__(self, data_dir: str = "data") -> None:
        self._data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # Persistent sample arrays — shape (N, 6) per emotion
        self._arrays: dict[str, np.ndarray] = {}
        self._centroids: dict[str, dict | None] = {e: None for e in EMOTIONS}

        # Duplicate-frame guard: last successfully stored feature vector
        self._last_vec: np.ndarray | None = None

        self._load_all()

    # ── I/O helpers ───────────────────────────────────────────────────────────

    def _path(self, emotion: str) -> str:
        return os.path.join(self._data_dir, f"{emotion}_samples.npy")

    def _load_all(self) -> None:
        _empty = lambda: np.empty((0, len(_FEATURE_KEYS)), dtype=np.float32)
        for emotion in EMOTIONS:
            p = self._path(emotion)
            if os.path.exists(p):
                try:
                    arr = np.load(p)
                    if arr.ndim == 2 and arr.shape[1] == len(_FEATURE_KEYS):
                        self._arrays[emotion] = arr.astype(np.float32)
                        print(f"[FEEDBACK] Loaded {len(arr):3d} {emotion} samples")
                    else:
                        print(f"[FEEDBACK] Unexpected shape in {p}, resetting.")
                        self._arrays[emotion] = _empty()
                except Exception as exc:
                    print(f"[FEEDBACK] Could not load {emotion} samples — {exc}")
                    self._arrays[emotion] = _empty()
            else:
                self._arrays[emotion] = _empty()
            self._recompute_centroid(emotion)

    def _save(self, emotion: str) -> None:
        try:
            np.save(self._path(emotion), self._arrays[emotion])
        except Exception as exc:
            print(f"[FEEDBACK] Save failed for {emotion}: {exc}")

    # ── Centroid maths ────────────────────────────────────────────────────────

    def _recompute_centroid(self, emotion: str) -> None:
        arr = self._arrays.get(emotion)
        if arr is None or len(arr) == 0:
            self._centroids[emotion] = None
            return
        self._centroids[emotion] = _to_centroid(arr)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_centroids(self) -> dict[str, dict | None]:
        """
        Return centroid dicts compatible with EmotionDetector._emotion_centroids.
        Inject the result into the detector at startup and after each update.
        """
        return dict(self._centroids)

    def get_sample_counts(self) -> dict[str, int]:
        """Total persistently stored samples per emotion (for overlay display)."""
        return {e: len(self._arrays.get(e, [])) for e in EMOTIONS}

    def store_sample(
        self,
        emotion: str,
        raw_features: dict | None,
        confidence: float,
        is_stable: bool,
        min_confidence: float,
    ) -> tuple[bool, str]:
        """
        Validate and persistently store one feedback sample.

        Parameters
        ----------
        emotion        : Target label  ("happy" | "sad" | "angry").
        raw_features   : Geometry dict from EmotionDetector._last_raw (or None).
        confidence     : Fused confidence for the current frame  (0–1).
        is_stable      : True when detector has a confirmed stable emotion.
        min_confidence : Minimum fused score required to accept the sample.

        Returns
        -------
        (stored: bool, message: str)
            stored  — True if the sample was accepted and written to disk.
            message — Human-readable result suitable for on-screen display.
        """
        if emotion not in EMOTIONS:
            return False, ""

        # ── Safety: face must be detected ─────────────────────────────────────
        if raw_features is None:
            return False, "No face detected — sample not stored"

        # ── Safety: minimum confidence ────────────────────────────────────────
        if confidence < min_confidence:
            return False, f"Confidence too low ({confidence:.0%}) — sample not stored"

        # ── Safety: emotion must be stable ────────────────────────────────────
        if not is_stable:
            return False, "Emotion not stable yet — sample not stored"

        vec = _to_vec(raw_features)

        # ── Safety: reject duplicate frames ───────────────────────────────────
        if self._last_vec is not None and np.allclose(vec, self._last_vec, atol=1e-6):
            return False, "Duplicate frame — sample not stored"

        # ── Append → persist → update centroid ────────────────────────────────
        existing = self._arrays.get(
            emotion, np.empty((0, len(_FEATURE_KEYS)), dtype=np.float32)
        )
        self._arrays[emotion] = np.vstack([existing, vec[np.newaxis, :]])
        self._last_vec = vec.copy()
        self._save(emotion)
        self._recompute_centroid(emotion)

        total = len(self._arrays[emotion])
        msg   = f"Stored sample for {emotion.upper()} | Total samples: {total}"
        print(f"[FEEDBACK] {msg}")
        return True, msg
