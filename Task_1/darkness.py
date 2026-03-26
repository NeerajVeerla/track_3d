# checks/darkness.py
# ─────────────────────────────────────────────
# Darkness detection using HSV Value channel.
#
# Why HSV over grayscale?
#   Construction sites have colorful equipment (yellow machinery,
#   orange vests). Grayscale mixes color into brightness — HSV V
#   channel gives pure brightness, decoupled from color.
#
# Three conditions checked — frame discarded if any TWO are met.
# Using multiple conditions prevents single-metric false positives.
# ─────────────────────────────────────────────

import numpy as np
from config import CONFIG
from utils.regions import get_analysis_region_hsv


def compute_darkness_score(frame: np.ndarray) -> dict:
    """
    Compute brightness metrics on the equatorial V channel.

    Returns:
        dict with:
          - v_mean          : mean brightness
          - v_median        : median brightness (p50)
          - dark_pixel_ratio: fraction of pixels below dark threshold
          - shadow_ratio    : fraction of pixels below shadow threshold
    """
    hsv       = get_analysis_region_hsv(frame)
    v_channel = hsv[:, :, 2].astype(np.float32)

    dark_px_thresh   = CONFIG["dark_pixel_thresh"]   # e.g. 30
    shadow_px_thresh = CONFIG["shadow_thresh"]        # e.g. 15

    dark_pixel_ratio = float(np.mean(v_channel < dark_px_thresh))
    shadow_ratio     = float(np.mean(v_channel < shadow_px_thresh))
    v_median         = float(np.median(v_channel))
    v_mean           = float(np.mean(v_channel))

    return {
        "v_mean"          : round(v_mean, 2),
        "v_median"        : round(v_median, 2),
        "dark_pixel_ratio": round(dark_pixel_ratio, 3),
        "shadow_ratio"    : round(shadow_ratio, 3),
    }


def is_dark(score_data: dict) -> tuple[bool, str]:
    """
    Decision logic — frame is dark if at least 2 conditions are met.
    Using 2-of-3 prevents false positives from any single metric.

    Conditions:
      1. dark_pixel_ratio > 0.60  → majority of scene is dark
      2. v_median < 35            → median pixel is very dark
      3. shadow_ratio > 0.80      → extreme darkness / lens issue

    Returns:
        (is_dark: bool, reason: str)
    """
    conditions = []

    if score_data["dark_pixel_ratio"] > CONFIG["dark_pixel_ratio"]:
        conditions.append(
            f"dark_pixel_ratio={score_data['dark_pixel_ratio']:.2f}"
        )

    if score_data["v_median"] < CONFIG["median_brightness"]:
        conditions.append(
            f"v_median={score_data['v_median']:.1f}"
        )

    if score_data["shadow_ratio"] >= CONFIG["shadow_ratio"]:
        conditions.append(
            f"shadow_ratio={score_data['shadow_ratio']:.2f}"
        )

    triggered = len(conditions) >= CONFIG["dark_min_conditions"]

    if triggered:
        return True, "too_dark (" + ", ".join(conditions) + ")"

    return False, ""


def check_autoexposure_lag(
    scores: list[dict],
    idx: int,
    window: int = 3
) -> bool:
    """
    Detect auto-exposure lag frames:
      - Current frame is dark
      - But neighboring frames are bright
      → Camera was adjusting exposure, not a genuinely dark scene

    Args:
        scores : list of all darkness score dicts (one per frame)
        idx    : index of current frame
        window : how many frames to look ahead/behind

    Returns:
        True if this looks like auto-exposure lag
    """
    # Can't check if no neighbors
    if idx < window or idx >= len(scores) - window:
        return False

    # Check if current frame is dark
    current_dark, _ = is_dark(scores[idx])
    if not current_dark:
        return False

    # Check if surrounding frames are bright
    neighbors = (
        scores[max(0, idx - window) : idx] +
        scores[idx + 1 : min(len(scores), idx + window + 1)]
    )
    bright_neighbors = sum(
        1 for s in neighbors
        if s["v_median"] > CONFIG["median_brightness"] * 2
    )

    # If most neighbors are bright → auto-exposure lag
    return bright_neighbors >= (window)
