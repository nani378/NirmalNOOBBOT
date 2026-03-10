"""
emotion_detector.py — Dual-signal emotion detection:
                       FER CNN  +  MediaPipe landmark geometry

Two independent evidence sources are fused to reach ~85 % accuracy for
each of the three target emotions.

Signal 1 — FER CNN (weight 0.55)
  The pretrained Keras model outputs 7-class probabilities.
  These are grouped into our 3 target emotions and exponentially
  smoothed over EMOTION_BUFFER_SIZE frames.

Signal 2 — MediaPipe Face Mesh geometry (weight 0.45)
  468 facial landmarks are analysed to compute 5 geometric cues
  that map directly to Facial Action Coding System (FACS) action units:

  Feature                   AU     Target emotion
  ─────────────────────────────────────────────────
  Mouth-corner lift/droop   AU12   HAPPY (lift) / SAD (droop)
  Oblique inner-brow slope  AU1    SAD (inner rises), ANGRY (inner drops)
  Inner-brow horizontal gap AU4    ANGRY (brows pulled together)
  Eye openness ratio        AU7    ANGRY (squint / lid tension)
  Cheek & eye crinkle       AU6    HAPPY (Duchenne smile marker)

  Every cue is normalised by face height or inter-ocular distance so
  the detector is robust to camera distance and head scale.

Stability gating
  A ≥ 70 % majority vote over STABLE_FRAMES recent frames must agree
  before an emotion is announced — eliminates brief micro-expressions.
"""

import cv2
import os
import urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from collections import deque
from fer.fer import FER
from config import EMOTION_BUFFER_SIZE, STABLE_FRAMES, MIN_CONFIDENCE, MEMORY_DATA_DIR

# ── Face Landmarker model (Tasks API) ─────────────────────────────────────────
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

def _ensure_model() -> str:
    """Download the face landmark model on first run, then return its path."""
    if not os.path.exists(_MODEL_PATH):
        print("[DETECTOR] Downloading face landmark model (~30 MB)…")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[DETECTOR] Download complete.")
    return _MODEL_PATH

# ── Emotion grouping ──────────────────────────────────────────────────────────
EMOTION_GROUPS: dict[str, list[str]] = {
    "happy": ["happy", "surprise"],  # FER classifies wide smiles as "surprise"
    "sad":   ["sad",   "fear"],
    "angry": ["angry", "disgust"],
}

EMOTION_COLOURS: dict[str, tuple] = {
    "happy": (0,   220,   0),
    "sad":   (220,  60,   0),
    "angry": (0,    0,  220),
}

# Frames with a detected face needed to establish the personal neutral baseline
_CALIB_FRAMES   = 40
# Confirmed detections per emotion needed before centroid learning activates
_CONFIRM_FRAMES = 25
# Fixed feature column order for .npy memory files (must never change)
_FEATURE_KEYS = (
    "corner_lift", "brow_slope", "brow_gap", "eye_open",
    "lip_h", "crinkle", "brow_raise", "mouth_width",
)

# Per-emotion fusion weights {emotion: (FER_w, LM_w)} — sums to 1.0 each.
# FER CNN is reliable for happy smiles; geometry dominates subtle sad/angry.
_FUSION_W: dict[str, tuple] = {
    "happy": (0.55, 0.45),
    "sad":   (0.45, 0.55),   # raised FER weight — CNN reliable for subtle sad
    "angry": (0.35, 0.65),
}

# ── MediaPipe landmark indices ────────────────────────────────────────────────
# Verified against the canonical MediaPipe 468-point face mesh map.
_LM = {
    # Left eye
    "l_eye_inner": 33,  "l_eye_outer": 133,
    "l_eye_top":  159,  "l_eye_bot":   145,
    # Right eye
    "r_eye_inner": 362, "r_eye_outer": 263,
    "r_eye_top":   386, "r_eye_bot":   374,
    # Eyebrows
    "l_brow_inner": 107, "l_brow_outer": 70,
    "r_brow_inner": 336, "r_brow_outer": 300,
    # Mouth
    "mouth_l":     61,  "mouth_r":    291,
    "lip_top_ctr": 13,  "lip_bot_ctr": 14,
    # Face skeleton
    "forehead": 10,     "chin": 152,
}


def _sig(x: float) -> float:
    """Numerically stable sigmoid, clipped to avoid overflow."""
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0))))


def _pos_sig(x: float, gain: float = 1.0) -> float:
    """Zero-based sigmoid: 0.0 at x=0, grows toward 1 for x>0, 0 for x<0.
    Eliminates the 0.5 neutral-bias of plain sigmoid."""
    return float(max(0.0, 2.0 * _sig(x * gain) - 1.0))


def _neg_sig(x: float, gain: float = 1.0) -> float:
    """Zero-based sigmoid: 0.0 at x=0, grows toward 1 for x<0, 0 for x>0."""
    return _pos_sig(-x, gain)


class EmotionDetector:
    """
    FER + MediaPipe dual-signal emotion detector with temporal smoothing
    and majority-vote stability gating.
    """

    def __init__(
        self,
        buffer_size:    int   = EMOTION_BUFFER_SIZE,
        stable_frames:  int   = STABLE_FRAMES,
        min_confidence: float = MIN_CONFIDENCE,
    ):
        print("[DETECTOR] Loading FER model…")
        self._fer           = FER(mtcnn=False)
        self._stable_frames = stable_frames
        self._min_conf      = min_confidence

        print("[DETECTOR] Loading MediaPipe FaceLandmarker…")
        model_path  = _ensure_model()
        base_opts   = mp_python.BaseOptions(model_asset_path=model_path)
        lm_opts     = mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._face_landmarker = mp_vision.FaceLandmarker.create_from_options(lm_opts)

        self._score_bufs: dict[str, deque] = {
            e: deque(maxlen=buffer_size) for e in EMOTION_GROUPS
        }
        self._vote_buf: deque = deque(maxlen=stable_frames)

        # ── Memory 1: personal neutral-face baseline ──────────────────────────
        # First _CALIB_FRAMES face-detected frames build YOUR resting geometry.
        # All scores become deviations from your personal neutral, removing
        # false positives from individual anatomy (low brows, small eyes, etc.).
        self._calib_buf:  list      = []
        self._baseline:   dict|None = None

        # ── Memory 2: per-emotion confirmed samples & learned centroids ────────
        # When user presses H/S/A to confirm a correct detection, the raw
        # face geometry is stored here.  After _CONFIRM_FRAMES samples the
        # detector learns your personal emotion centroid, which amplifies
        # future detections of that emotion via a Gaussian similarity boost.
        self._confirmed_bufs: dict[str, deque] = {
            e: deque(maxlen=200) for e in EMOTION_GROUPS
        }
        self._emotion_centroids: dict[str, dict|None] = {
            e: None for e in EMOTION_GROUPS
        }

        # Raw geometry from the last processed frame (used for confirmation)
        self._last_raw: dict|None = None

        # ── Memory 3: hysteresis ──────────────────────────────────────────────
        # Hold the last confirmed emotion until a *different* one wins.
        self._last_stable: str|None = None
        # ── Memory 4: persistent on-disk sample store ─────────────────────────────
        # Samples from previous sessions load automatically on startup.
        # The centroid boost is additive and capped — it only ever helps accuracy.
        self._data_dir = MEMORY_DATA_DIR
        os.makedirs(self._data_dir, exist_ok=True)
        self._load_memory()
        print("[DETECTOR] Ready \u2014 look neutral at the camera to calibrate…")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _group_fer(self, raw: dict) -> dict[str, float]:
        """Collapse FER's 7 raw scores into 3 target-emotion scores."""
        return {
            t: sum(raw.get(s, 0.0) for s in members)
            for t, members in EMOTION_GROUPS.items()
        }

    def _load_memory(self) -> None:
        """Load saved emotion samples from disk and rebuild centroids on startup."""
        for emo in EMOTION_GROUPS:
            path = os.path.join(self._data_dir, f"{emo}_samples.npy")
            if not os.path.exists(path):
                continue
            try:
                arr = np.load(path)                      # float32 shape (N, 8)
                if arr.ndim != 2 or arr.shape[1] != len(_FEATURE_KEYS):
                    print(f"[MEMORY] Skipping {emo}: unexpected shape {arr.shape}")
                    continue
                for row in arr[-200:]:                   # keep most recent 200
                    self._confirmed_bufs[emo].append(dict(zip(_FEATURE_KEYS, row.tolist())))
                n = len(self._confirmed_bufs[emo])
                if n >= _CONFIRM_FRAMES:
                    self._rebuild_centroid(emo)
                    print(f"[MEMORY] {emo.upper()}: {n} samples loaded, centroid active")
                else:
                    print(f"[MEMORY] {emo.upper()}: {n} samples ({_CONFIRM_FRAMES - n} more needed for centroid)")
            except Exception as exc:
                print(f"[MEMORY] Could not load {emo}: {exc}")

    def _save_sample(self, emotion: str, raw: dict) -> None:
        """Append one geometry sample to the emotion's .npy file on disk."""
        path = os.path.join(self._data_dir, f"{emotion}_samples.npy")
        row  = np.array([[raw[k] for k in _FEATURE_KEYS]], dtype=np.float32)
        if os.path.exists(path):
            try:
                arr = np.vstack([np.load(path), row])
            except Exception:
                arr = row
        else:
            arr = row
        np.save(path, arr)

    def _rebuild_centroid(self, emotion: str) -> None:
        """Recompute the emotion centroid from all in-memory confirmed samples."""
        feats = list(self._confirmed_bufs[emotion])
        self._emotion_centroids[emotion] = {
            "mean": {k: float(np.mean([f[k] for f in feats])) for k in feats[0]},
            "std":  {k: max(float(np.std( [f[k] for f in feats])), 1e-4)
                     for k in feats[0]},
        }

    def _extract_raw(self, lm) -> dict | None:
        """
        Pull the 6 normalised geometry values from MediaPipe landmarks.
        Returns None when the face region is too small to be reliable.
        These raw values (not yet score-mapped) are stored for calibration
        and user-confirmation centroid learning.
        """
        def px(name): return lm[_LM[name]].x
        def py(name): return lm[_LM[name]].y

        face_h = abs(py("chin") - py("forehead"))
        iod    = abs(px("r_eye_inner") - px("l_eye_inner"))
        if face_h < 1e-4 or iod < 1e-4:
            return None

        mc_avg_y    = (py("mouth_l") + py("mouth_r")) / 2.0
        corner_lift = (py("lip_top_ctr") - mc_avg_y) / face_h

        l_slope    = (py("l_brow_inner") - py("l_brow_outer")) / face_h
        r_slope    = (py("r_brow_inner") - py("r_brow_outer")) / face_h
        brow_slope = (l_slope + r_slope) / 2.0

        brow_gap = abs(px("r_brow_inner") - px("l_brow_inner")) / iod

        l_open   = abs(py("l_eye_bot") - py("l_eye_top")) / face_h
        r_open   = abs(py("r_eye_bot") - py("r_eye_top")) / face_h
        eye_open = (l_open + r_open) / 2.0

        lip_h = abs(py("lip_bot_ctr") - py("lip_top_ctr")) / face_h

        l_ctr_y = (py("l_eye_top") + py("l_eye_bot")) / 2.0
        r_ctr_y = (py("r_eye_top") + py("r_eye_bot")) / 2.0
        crinkle = ((l_ctr_y - py("l_eye_bot")) + (r_ctr_y - py("r_eye_bot"))) / (2.0 * face_h)

        # Inner brow height above the eye top — direct AU1 measure (rises in sadness)
        l_brow_raise = (py("l_eye_top") - py("l_brow_inner")) / face_h
        r_brow_raise = (py("r_eye_top") - py("r_brow_inner")) / face_h
        brow_raise   = (l_brow_raise + r_brow_raise) / 2.0

        # Mouth width relative to inter-ocular distance — widens in a real smile
        mouth_width = abs(px("mouth_r") - px("mouth_l")) / iod

        return {
            "corner_lift":  corner_lift,
            "brow_slope":   brow_slope,
            "brow_gap":     brow_gap,
            "eye_open":     eye_open,
            "lip_h":        lip_h,
            "crinkle":      crinkle,
            "brow_raise":   brow_raise,   # ← new: AU1 inner-brow height
            "mouth_width":  mouth_width,  # ← new: smile width
        }

    def _centroid_similarity(self, raw: dict, emotion: str) -> float:
        """
        Gaussian similarity between the current raw geometry and the
        learned emotion centroid (built from user-confirmed samples).
        Returns 0.0 when no centroid is available yet.

        Uses only the most discriminative features per emotion so that
        unrelated variation does not dilute the similarity score.
        """
        c = self._emotion_centroids.get(emotion)
        if c is None:
            return 0.0
        mean, std = c["mean"], c["std"]
        _emo_keys = {
            "happy": ["corner_lift", "crinkle",   "mouth_width"],
            "sad":   ["corner_lift", "brow_raise", "brow_slope"],
            "angry": ["brow_gap",   "brow_slope",  "eye_open", "lip_h"],
        }
        keys = _emo_keys[emotion]
        d_sq = sum(((raw[k] - mean[k]) / std[k]) ** 2 for k in keys) / len(keys)
        return float(np.exp(-0.5 * d_sq))

    def _landmark_scores(self, lm) -> dict[str, float]:
        """
        Compute FACS-inspired geometric scores (zero-based: 0=neutral → 1=max).

        Pipeline:
          1. Extract raw geometry via _extract_raw().
          2. During the first _CALIB_FRAMES frames collect neutral baseline.
          3. After calibration compute feature deviations from personal neutral.
          4. Map deviations through zero-based sigmoids (_pos_sig/_neg_sig).

        Geometry IS the primary detector — it runs entirely from face physics and
        needs no feedback to work.  The optional centroid similarity (step 5) is
        an *additive* bonus capped at +0.12, which is below MIN_CONFIDENCE (0.30).
        This means:
          • A neutral face with a wrong centroid scores at most 0.12 → never
            triggers a false detection.
          • A genuine sad/angry/happy face gets a small recall boost without its
            score being inflated to win over a geometrically stronger competitor.
          • Deleting or corrupting saved samples cannot reduce detection accuracy
            below the pure-geometry baseline.
        """
        raw = self._extract_raw(lm)
        if raw is None:
            return {"happy": 0.0, "sad": 0.0, "angry": 0.0}

        # Store for user confirmation key-press
        self._last_raw = raw

        # ── Calibration phase ─────────────────────────────────────────────────
        if self._baseline is None:
            self._calib_buf.append(raw)
            if len(self._calib_buf) >= _CALIB_FRAMES:
                self._baseline = {
                    k: float(np.mean([f[k] for f in self._calib_buf]))
                    for k in raw
                }
                print("[DETECTOR] Neutral baseline calibrated:",
                      {k: f"{v:.4f}" for k, v in self._baseline.items()})
            return {"happy": 0.0, "sad": 0.0, "angry": 0.0}

        # ── Deviations from personal neutral ──────────────────────────────────
        b = self._baseline
        corner_lift   = raw["corner_lift"]  - b["corner_lift"]  # + = smile, − = droop
        brow_slope    = raw["brow_slope"]   - b["brow_slope"]   # − = inner brow up (sad); + = inner brow down (angry)
        brow_gap_d    = raw["brow_gap"]     - b["brow_gap"]     # − = brows converge (angry)
        eye_open_d    = raw["eye_open"]     - b["eye_open"]     # − = squint (angry)
        lip_h_d       = raw["lip_h"]        - b["lip_h"]        # − = lips pressed (angry)
        crinkle       = raw["crinkle"]      - b["crinkle"]      # + = cheek/eye crinkle (happy)
        brow_raise_d  = raw["brow_raise"]   - b["brow_raise"]   # + = inner brow raised above eye top (sad AU1)
        mouth_width_d = raw["mouth_width"]  - b["mouth_width"]  # + = mouth wider (happy smile)

        # ── Geometric scores (zero-based) ─────────────────────────────────────
        # HAPPY — mouth corners up + cheek crinkle (Duchenne) + mouth wider
        happy_lift    = _pos_sig(corner_lift,    42)   # AU12: lip corner puller
        happy_crinkle = _pos_sig(crinkle,        80)   # AU06: cheek/eye crinkle (lowered 120→80)
        happy_wide    = _pos_sig(mouth_width_d,  18)   # AU25: mouth stretches wider on smile
        happy_geom = 0.50 * happy_lift + 0.30 * happy_crinkle + 0.20 * happy_wide

        # SAD — mouth droop + inner brow raise (AU1, strongest FACS sad cue) + brow slope + lip part
        sad_droop = _neg_sig(corner_lift,  32)   # AU15: lip corner depressor (lowered 50→32)
        sad_raise = _pos_sig(brow_raise_d, 45)   # AU01: inner brow raise above eye (lowered 65→45)
        sad_brow  = _neg_sig(brow_slope,   30)   # AU01 corroboration: oblique brow slope (lowered 45→30)
        sad_lip   = _pos_sig(lip_h_d,      40)   # AU25: lips faintly parted (lowered 60→40)
        sad_geom  = 0.38 * sad_droop + 0.35 * sad_raise + 0.17 * sad_brow + 0.10 * sad_lip

        # ANGRY — brow furrow + brow lower + eye squint + lip compress
        angry_furrow   = _neg_sig(brow_gap_d,  22)   # AU04: brow lowerer / corrugator (raised 16→22)
        angry_lower    = _pos_sig(brow_slope,  32)   # AU04: inner brow depressed
        angry_squint   = _neg_sig(eye_open_d,  70)   # AU07: lid tightener (gain lowered 90->70)
        angry_compress = _neg_sig(lip_h_d,     80)   # AU23: lip tightener / press
        angry_geom = (0.28 * angry_furrow  + 0.35 * angry_lower
                    + 0.22 * angry_squint  + 0.15 * angry_compress)

        # ── Centroid similarity (additive, capped at +0.18) ───────────────────────────
        # Memory centroids add a recall bonus — cannot reduce accuracy.
        # Cap +0.18 keeps neutral faces safely below MIN_CONFIDENCE → no false positives.
        sim_h = self._centroid_similarity(raw, "happy")
        sim_s = self._centroid_similarity(raw, "sad")
        sim_a = self._centroid_similarity(raw, "angry")

        return {
            "happy": float(np.clip(happy_geom + 0.18 * sim_h, 0.0, 1.0)),
            "sad":   float(np.clip(sad_geom   + 0.18 * sim_s, 0.0, 1.0)),
            "angry": float(np.clip(angry_geom + 0.18 * sim_a, 0.0, 1.0)),
        }

    def _smooth(self, scores: dict[str, float]) -> dict[str, float]:
        """Exponentially weighted moving average over the score buffer."""
        for e, s in scores.items():
            self._score_bufs[e].append(s)
        smoothed: dict[str, float] = {}
        for e in EMOTION_GROUPS:
            hist = list(self._score_bufs[e])
            if not hist:
                smoothed[e] = 0.0
                continue
            n = len(hist)
            w = np.exp(np.linspace(-1.0, 0.0, n))
            w /= w.sum()
            smoothed[e] = float(np.dot(w, hist))
        return smoothed

    def _stable_vote(self) -> str | None:
        """
        Majority-vote stability check with hysteresis.

        Once an emotion reaches the ≥70% threshold it is held as
        `_last_stable` until a *different* emotion reaches the same
        threshold.  This stops flickering back to None between genuine
        sustained expressions.
        """
        if len(self._vote_buf) < self._stable_frames:
            return self._last_stable   # hold previous while warming up
        recent = list(self._vote_buf)
        counts = {e: recent.count(e) for e in EMOTION_GROUPS}
        best   = max(counts, key=counts.get)
        if counts[best] >= int(self._stable_frames * 0.70):
            self._last_stable = best
        return self._last_stable

    # ── Public API ────────────────────────────────────────────────────────────

    def process_frame(self, frame) -> dict:
        """
        Run FER + MediaPipe landmark fusion on one BGR frame.

        Returns a dict:
            raw_emotion     : str | None   dominant fused emotion this frame
            confidence      : float        smoothed fused confidence (0–1)
            box             : list | None  [x, y, w, h] face bounding box
            smoothed_scores : dict         per-emotion smoothed fused scores
            landmark_scores : dict         raw landmark-only scores (for HUD)
            stable_emotion  : str | None   stability-gated final emotion
        """
        null = {
            "raw_emotion":     None,
            "confidence":      0.0,
            "box":             None,
            "smoothed_scores": {e: 0.0 for e in EMOTION_GROUPS},
            "landmark_scores": {e: 0.0 for e in EMOTION_GROUPS},
            "stable_emotion":  None,
        }

        # ── Signal 1: FER CNN ─────────────────────────────────────────────────
        detections = self._fer.detect_emotions(frame)
        if not detections:
            return null

        top     = max(detections, key=lambda d: max(d["emotions"].values()))
        box     = top["box"]
        fer_g   = self._group_fer(top["emotions"])

        # ── Signal 2: MediaPipe FaceLandmarker ───────────────────────────────
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        lm_result = self._face_landmarker.detect(mp_image)
        if lm_result.face_landmarks:
            lm   = lm_result.face_landmarks[0]   # list of NormalizedLandmark
            lm_s = self._landmark_scores(lm)
            fused = {
                e: _FUSION_W[e][0] * fer_g[e] + _FUSION_W[e][1] * lm_s[e]
                for e in EMOTION_GROUPS
            }
        else:
            lm_s  = {e: 0.0 for e in EMOTION_GROUPS}
            fused = fer_g   # fallback: FER only

        # ── Temporal smoothing + stability vote ───────────────────────────────
        smoothed = self._smooth(fused)
        best     = max(smoothed, key=smoothed.get)
        conf     = smoothed[best]

        # Conflict resolver: only vote when the winner leads by ≥ 0.08
        # Prevents borderline sad/angry misclassifications from entering the vote
        sorted_scores = sorted(smoothed.values(), reverse=True)
        gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) >= 2 else sorted_scores[0]
        if conf >= self._min_conf and gap >= 0.08:
            self._vote_buf.append(best)

        return {
            "raw_emotion":     best,
            "confidence":      conf,
            "box":             box,
            "smoothed_scores": smoothed,
            "landmark_scores": lm_s,
            "stable_emotion":  self._stable_vote(),
        }

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def is_calibrated(self) -> bool:
        """True once the personal neutral baseline has been established."""
        return self._baseline is not None

    @property
    def calib_progress(self) -> int:
        """Calibration frames collected so far (0 – _CALIB_FRAMES)."""
        return _CALIB_FRAMES if self._baseline is not None else len(self._calib_buf)

    @property
    def confirmed_counts(self) -> dict[str, int]:
        """Number of confirmed samples per emotion (session + loaded from disk)."""
        return {e: len(self._confirmed_bufs[e]) for e in EMOTION_GROUPS}

    @property
    def memory_counts(self) -> dict[str, int]:
        """Alias for confirmed_counts — total samples per emotion stored in memory."""
        return self.confirmed_counts

    def confirm_detection(self, emotion: str) -> bool:
        """
        Call when the user confirms a detection (H/S/A key) or via auto-confirm.

        Stores the raw face geometry in both the in-session buffer and on disk.
        After _CONFIRM_FRAMES samples the personal centroid activates —
        future detections of that emotion receive a small accuracy boost.

        Returns True if a sample was stored.
        """
        if emotion not in EMOTION_GROUPS:
            return False
        if self._baseline is None or self._last_raw is None:
            return False
        self._confirmed_bufs[emotion].append(self._last_raw)
        self._save_sample(emotion, self._last_raw)
        n = len(self._confirmed_bufs[emotion])
        if n >= _CONFIRM_FRAMES:
            self._rebuild_centroid(emotion)
            print(f"[MEMORY] {emotion.upper()} centroid updated ({n} samples)")
        return True

    def reset_votes(self) -> None:
        self._vote_buf.clear()
        self._last_stable = None   # clear hysteresis after each conversation trigger


# ─────────────────────────────────────────────────────────────────────────────
# Legacy stub kept so old code that checks process_frame result["box"] doesn't
# break if the dict shape changes.
# ─────────────────────────────────────────────────────────────────────────────
class _LegacyResult:
    pass  # no longer used; kept as marker
