# utils/regions.py
# ─────────────────────────────────────────────
# Region masking for equirectangular images.
# 
# Extracts the analysis region by masking:
#   - Top 20%: pole distortion artifacts
#   - Bottom 20%: camera operator + floor
#
# All checks (blur, darkness, duplicates) operate
# on these consistently masked regions.
# ─────────────────────────────────────────────

import cv2
import numpy as np
from config import CONFIG


def get_analysis_bounds(frame_shape: tuple[int, ...]) -> tuple[int, int]:
    """
    Compute vertical bounds of the analysis region.

    Args:
        frame_shape: shape tuple from image array (H, W, C) or (H, W)

    Returns:
        (y1, y2): start and end row (end-exclusive) of the analysis region
    """
    h = frame_shape[0]
    top_crop = int(h * CONFIG["top_crop"])
    bottom_crop = int(h * CONFIG["bottom_crop"])
    y1 = top_crop
    y2 = h - bottom_crop
    return y1, y2


def get_analysis_region(frame: np.ndarray) -> np.ndarray:
    """
    Extract equatorial analysis region from equirectangular frame.
    Removes pole distortion (top) and camera/floor (bottom).
    
    Args:
        frame: BGR image (H, W, 3)
    
    Returns:
        BGR image of analysis region
    """
    y1, y2 = get_analysis_bounds(frame.shape)
    region = frame[y1:y2, :, :]
    
    return region


def get_analysis_region_gray(frame: np.ndarray) -> np.ndarray:
    """
    Extract analysis region and convert to grayscale.
    Used for blur detection and optical flow.
    
    Args:
        frame: BGR image (H, W, 3)
    
    Returns:
        Grayscale image of analysis region
    """
    region = get_analysis_region(frame)
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    return gray


def get_analysis_region_hsv(frame: np.ndarray) -> np.ndarray:
    """
    Extract analysis region and convert to HSV.
    Used for darkness detection (HSV V channel = brightness).
    
    Args:
        frame: BGR image (H, W, 3)
    
    Returns:
        HSV image of analysis region (H, W, 3)
    """
    region = get_analysis_region(frame)
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    return hsv


def draw_analysis_overlay(frame: np.ndarray) -> np.ndarray:
    """
    Draw overlay showing ignored/used bands on the input frame.

    - Top ignored band: red tint
    - Analysis region : green tint
    - Bottom ignored band: red tint

    Args:
        frame: BGR image

    Returns:
        BGR image with visualization overlay and boundary lines
    """
    y1, y2 = get_analysis_bounds(frame.shape)
    vis = frame.copy()

    # Tint masks
    red = np.array([0, 0, 255], dtype=np.uint8)
    green = np.array([0, 255, 0], dtype=np.uint8)

    top = vis[:y1, :, :]
    mid = vis[y1:y2, :, :]
    bot = vis[y2:, :, :]

    if top.size > 0:
        top[:] = cv2.addWeighted(top, 0.65, np.full_like(top, red), 0.35, 0)
    if mid.size > 0:
        mid[:] = cv2.addWeighted(mid, 0.75, np.full_like(mid, green), 0.25, 0)
    if bot.size > 0:
        bot[:] = cv2.addWeighted(bot, 0.65, np.full_like(bot, red), 0.35, 0)

    # Boundaries
    cv2.line(vis, (0, y1), (vis.shape[1] - 1, y1), (255, 255, 255), 2)
    cv2.line(vis, (0, y2), (vis.shape[1] - 1, y2), (255, 255, 255), 2)

    # Labels
    cv2.putText(vis, "IGNORED (TOP)", (20, max(25, y1 // 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(vis, "ANALYSIS REGION", (20, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(vis, "IGNORED (BOTTOM)", (20, min(vis.shape[0] - 20, y2 + 35)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return vis
