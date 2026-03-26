# checks/blur.py
# ─────────────────────────────────────────────
# Blur detection using Laplacian variance on patches.
#
# Why patches instead of global score?
#   Construction sites have low-texture areas (concrete walls,
#   empty floors) that score low on Laplacian even when sharp.
#   Patches let us sample different scene regions — as long as
#   enough patches are sharp, the frame is considered sharp.
#
# Three-layer decision:
#   1. Absolute  — hard minimum score
#   2. Relative  — sudden drop vs rolling mean (motion blur event)
#   3. Patch vote— majority of patches must be sharp
# ─────────────────────────────────────────────

import cv2
import numpy as np
from config import CONFIG
from utils.regions import get_analysis_region_gray


# ── Rolling Mean ─────────────────────────────

class RollingMean:
    """
    Tracks rolling mean of blur scores over recent sharp frames.
    Only updated with sharp frames — blurry frames don't drag it down.
    """
    def __init__(self, window: int):
        self.window  = window
        self.history : list[float] = []

    def update(self, value: float):
        self.history.append(value)
        if len(self.history) > self.window:
            self.history.pop(0)

    def get(self) -> float:
        return float(np.mean(self.history)) if self.history else 0.0


# ── Core Functions ────────────────────────────

def _laplacian_variance(gray_patch: np.ndarray) -> float:
    """Blur score for a single grayscale patch. Higher = sharper."""
    lap = cv2.Laplacian(gray_patch, cv2.CV_64F)
    return float(lap.var())


def compute_blur_score(frame: np.ndarray) -> dict:
    """
    Compute blur scores across N horizontal patches of the equatorial band.

    Args:
        frame: BGR image

    Returns:
        dict with:
          - mean_score   : mean across all patches
          - median_score : median across all patches
          - patch_scores : individual score per patch
          - min_patch    : lowest patch score
          - max_patch    : highest patch score
    """
    gray         = get_analysis_region_gray(frame)
    h, w         = gray.shape
    n            = CONFIG["num_patches"]
    patch_width  = w // n
    patch_scores = []

    for i in range(n):
        x1 = i * patch_width
        x2 = x1 + patch_width if i < n - 1 else w
        patch = gray[:, x1:x2]
        patch_scores.append(_laplacian_variance(patch))

    return {
        "mean_score"   : round(float(np.mean(patch_scores)),   2),
        "median_score" : round(float(np.median(patch_scores)), 2),
        "patch_scores" : [round(s, 2) for s in patch_scores],
        "min_patch"    : round(float(min(patch_scores)), 2),
        "max_patch"    : round(float(max(patch_scores)), 2),
    }


def is_blurry(score_data: dict, rolling_mean: float) -> tuple[bool, str]:
    """
    Three-layer blur decision.

    Layer 1 — Absolute threshold:
        mean score below hard minimum → always blurry

    Layer 2 — Relative drop:
        score dropped significantly vs recent frames
        → sudden blur event (motion blur, camera shake)

    Layer 3 — Patch vote:
        too many individual patches are blurry
        → scene-wide blur even if mean looks okay

    Args:
        score_data   : output of compute_blur_score()
        rolling_mean : current rolling mean of recent sharp frames

    Returns:
        (is_blurry: bool, reason: str)
    """
    mean      = score_data["mean_score"]
    threshold = CONFIG["abs_blur_threshold"]
    min_sharp = CONFIG["min_sharp_patches"]

    # Layer 1 — absolute
    if mean < threshold:
        return True, f"absolute_blur (score={mean:.1f} < {threshold})"

    

    # Layer 2 — patch vote
    sharp_patches = sum(
        1 for s in score_data["patch_scores"]
        if s >= threshold
    )
    if sharp_patches < min_sharp:
        return True, (
            f"patch_blur ({sharp_patches}/{CONFIG['num_patches']} "
            f"patches sharp, need {min_sharp})"
        )

    # Layer 3 — relative drop vs rolling mean
    # if rolling_mean > 0:
    #     ratio = mean / rolling_mean
    #     if ratio < CONFIG["relative_drop"]:
    #         return True, (
    #             f"relative_blur (score={mean:.1f} is "
    #             f"{ratio:.0%} of rolling_mean={rolling_mean:.1f})"
    #         )
    return False, ""
