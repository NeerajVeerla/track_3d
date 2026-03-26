# config.py
# ─────────────────────────────────────────────
# All thresholds and settings in one place.
# Tune these without touching any logic files.
# ─────────────────────────────────────────────

CONFIG = {

    # ── Region Masking ───────────────────────
    # Applied consistently across ALL checks
    "top_crop"           : 0.20,   # ignore top 20%    → pole distortion
    "bottom_crop"        : 0.20,   # ignore bottom 20% → person + floor

    # ── Darkness ─────────────────────────────
    "dark_pixel_ratio"   : 0.60,   # >60% pixels below V=30  → globally dark
    "dark_pixel_thresh"  : 30,     # V channel threshold for "dark pixel"
    "median_brightness"  : 35,     # median V < 35            → dark frame
    "shadow_ratio"       : 0.80,   # >80% pixels below V=15   → extreme dark
    "shadow_thresh"      : 15,     # V channel threshold for "shadow pixel"
    "dark_min_conditions": 2,      # how many conditions must trigger → discard

    # ── Blur ─────────────────────────────────
    "abs_blur_threshold" : 40.0,   # hard min Laplacian variance → always blurry
    "relative_drop"      : 0.50,   # score < 40% of rolling mean → sudden blur
    "rolling_window"     : 15,     # frames in rolling mean window
    "num_patches"        : 10,      # horizontal patches to analyse
    "min_sharp_patches"  : 4,      # min patches that must be sharp

    # ── Duplicate ────────────────────────────
    "flow_threshold"     : 3.0,    # mean optical flow magnitude (px) → candidate dupe
    "cosine_threshold"   : 0.90,   # embedding cosine similarity → confirmed dupe
    "flow_downsample"    : 0.25,   # downsample factor before computing flow (speed)

    # ── Output ───────────────────────────────
    "save_discarded"     : True,   # save discarded frames (False = just log them)
    "report_name"        : "report.csv",
}
