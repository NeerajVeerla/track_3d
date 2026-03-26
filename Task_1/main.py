# main.py
# ═════════════════════════════════════════════════════════════
# Task 1: Video Quality Assessment — Main Pipeline Orchestrator
# ═════════════════════════════════════════════════════════════
#
# Three-pass video quality filtering pipeline:
#   Pass 1: Score all frames (blur, darkness, embeddings)
#   Pass 2: Filter dark and blurry frames
#   Pass 3: Detect and remove near-duplicates
#
# ═════════════════════════════════════════════════════════════

import sys
import logging
from pathlib import Path
import cv2
import numpy as np

# Local imports
from config import CONFIG
from file_utils import load_frames, setup_output_dirs, save_frame, write_report
from blur import compute_blur_score, is_blurry, RollingMean
from darkness import compute_darkness_score, is_dark
from duplicate import EmbeddingModel, find_duplicate_clusters


# ── Logging Setup ────────────────────────────────

def setup_logging(output_dir: str) -> logging.Logger:
    """Configure logging to both console and file."""
    log_file = Path(output_dir) / "pipeline.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


# ── Pass 1: Score All Frames ────────────────────

def score_all_frames(
    frame_paths: list[Path],
    logger: logging.Logger
) -> tuple[list[dict], list[float], np.ndarray]:
    """
    Pass 1: Extract all quality scores for all frames.
    
    Returns:
        - scores: list of score dicts (per-frame metrics)
        - blur_scores: list of blur mean scores (for dedup clustering)
        - embeddings: (N, 768) DINOv2 embeddings for duplicate detection
    """
    logger.info(f"Pass 1: Scoring {len(frame_paths)} frames...")
    
    scores = []
    blur_scores = []
    model = EmbeddingModel()
    logger.info(f"Embedding model loaded on device: {model.device}")
    
    # Pre-load all frames for batched processing
    logger.info("Loading frames into memory...")
    frames = []
    for i, frame_path in enumerate(frame_paths):
        try:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                logger.warning(f"Failed to load frame {i}: {frame_path.name}")
                continue
            frames.append(frame)
        except Exception as e:
            logger.error(f"Error loading frame {frame_path.name}: {e}")
            continue
    
    if len(frames) == 0:
        raise RuntimeError("No frames could be loaded!")
    
    logger.info(f"Loaded {len(frames)} frames successfully")
    
    # Batch process embeddings
    logger.info("Extracting DINOv2 embeddings (batched)...")
    embeddings = model.get_embeddings_batch(frames, batch_size=8)
    logger.info(f"Extracted {embeddings.shape[0]} embeddings")
    
    # Per-frame scoring
    logger.info("Computing blur and darkness scores...")
    for i, frame in enumerate(frames):
        try:
            # Blur score
            blur_data = compute_blur_score(frame)
            blur_mean = blur_data["mean_score"]
            blur_scores.append(blur_mean)
            
            # Darkness score
            darkness_data = compute_darkness_score(frame)
            
            # Combine into one score record
            score_record = {
                "frame_idx": i,
                "frame_name": frame_paths[i].name,
                **blur_data,
                **darkness_data,
            }
            scores.append(score_record)
            
            if (i + 1) % 10 == 0 or i == len(frames) - 1:
                logger.info(f"  Scored {i + 1}/{len(frames)} frames")
        
        except Exception as e:
            logger.error(f"Error scoring frame {i} ({frame_paths[i].name}): {e}")
            continue
    
    logger.info(f"Pass 1 complete: {len(scores)} frames scored")
    return scores, blur_scores, embeddings


# ── Pass 2: Filter Dark + Blurry ─────────────────

def filter_dark_and_blurry(
    scores: list[dict],
    frame_paths: list[Path],
    logger: logging.Logger
) -> list[dict]:
    """
    Pass 2: Mark frames as DISCARD if too dark or blurry.
    
    Returns:
        - results: list of decision dicts {frame, status, reason, ...}
    """
    logger.info("Pass 2: Filtering dark and blurry frames...")
    
    results = [
        {
            "frame": frame_paths[i].name,
            "status": "KEEP",
            "reason": "",
            "blur_mean": scores[i]["mean_score"],
            "brightness_median": scores[i]["v_median"],
            "dark_pixel_ratio": scores[i]["dark_pixel_ratio"],
        }
        for i in range(len(scores))
    ]
    
    rolling_mean_blur = RollingMean(window=CONFIG["rolling_window"])
    
    for i, score in enumerate(scores):
        reasons = []
        
        # Check darkness
        is_dark_result, dark_reason = is_dark(score)
        if is_dark_result:
            reasons.append(dark_reason)
        
        # Check blur (with rolling mean)
        is_blurry_result, blur_reason = is_blurry(score, rolling_mean_blur.get())
        if is_blurry_result:
            reasons.append(blur_reason)
        else:
            # Update rolling mean only if frame is NOT blurry
            rolling_mean_blur.update(score["mean_score"])
        
        # Decision
        if reasons:
            results[i]["status"] = "DISCARD"
            results[i]["reason"] = " | ".join(reasons)
        
        if (i + 1) % 20 == 0 or i == len(scores) - 1:
            kept = sum(1 for r in results[:i+1] if r["status"] == "KEEP")
            logger.info(f"  Processed {i + 1}/{len(scores)} | Kept: {kept}")
    
    discarded = sum(1 for r in results if r["status"] == "DISCARD")
    logger.info(f"Pass 2 complete: {discarded} frames marked for discard")
    return results


# ── Pass 3: Detect Duplicates ────────────────────

def detect_duplicates(
    results: list[dict],
    frame_paths: list[Path],
    blur_scores: list[float],
    embeddings: np.ndarray,
    logger: logging.Logger
) -> list[dict]:
    """
    Pass 3: Among surviving frames, detect near-duplicates.
    Uses optical flow + embedding similarity two-stage detection.
    """
    logger.info("Pass 3: Detecting near-duplicates...")
    
    # Filter to only KEEP frames
    surviving_indices = [i for i, r in enumerate(results) if r["status"] == "KEEP"]
    surviving_paths = [frame_paths[i] for i in surviving_indices]
    surviving_blur_scores = [blur_scores[i] for i in surviving_indices]
    surviving_embeddings = embeddings[surviving_indices, :]
    
    logger.info(f"  {len(surviving_indices)} frames survived Pass 2")
    
    if len(surviving_paths) < 2:
        logger.info("  Fewer than 2 surviving frames — no duplicates possible")
        return results
    
    # Load surviving frames for optical flow
    logger.info("Loading surviving frames for duplicate detection...")
    surviving_frames = []
    for idx in surviving_indices:
        try:
            frame = cv2.imread(str(frame_paths[idx]))
            if frame is not None:
                surviving_frames.append(frame)
        except Exception as e:
            logger.warning(f"Failed to load frame {frame_paths[idx].name}: {e}")
    
    if len(surviving_frames) < len(surviving_paths):
        logger.warning(f"Only {len(surviving_frames)}/{len(surviving_paths)} frames loaded")
    
    # Run duplicate detection
    try:
        duplicate_results = find_duplicate_clusters(
            surviving_paths,
            surviving_embeddings,
            surviving_blur_scores,
            load_frame_fn=lambda p: cv2.imread(str(p))
        )
        
        # Map results back to original indices
        for orig_idx, dup_result in zip(surviving_indices, duplicate_results):
            if dup_result["status"] == "DISCARD":
                results[orig_idx]["status"] = "DISCARD"
                results[orig_idx]["reason"] = dup_result["reason"]
        
        discarded_in_pass3 = sum(
            1 for r in duplicate_results if r["status"] == "DISCARD"
        )
        logger.info(f"Pass 3 complete: {discarded_in_pass3} duplicates detected")
    
    except Exception as e:
        logger.error(f"Error during duplicate detection: {e}")
        logger.warning("Skipping Pass 3 — continuing with current results")
    
    return results


# ── Main Pipeline ───────────────────────────────

def main(input_dir: str, output_dir: str = "output"):
    """
    Main pipeline orchestrator.
    
    Args:
        input_dir: Path to directory containing input frames
        output_dir: Path where output will be saved
    """
    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(str(output_path))
    logger.info("=" * 60)
    logger.info("VIDEO QUALITY ASSESSMENT PIPELINE - START")
    logger.info("=" * 60)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Config: top_crop={CONFIG['top_crop']}, bottom_crop={CONFIG['bottom_crop']}")
    
    try:
        # Load frames
        logger.info("\n[LOADING] Reading frame list...")
        frame_paths = load_frames(input_dir)
        logger.info(f"Found {len(frame_paths)} frames")
        
        # Setup output directories
        dirs = setup_output_dirs(output_dir)
        logger.info(f"Output structure created: {dirs['base']}")
        
        # Pass 1: Score all frames
        logger.info("\n[PASS 1] Scoring all frames...")
        scores, blur_scores, embeddings = score_all_frames(frame_paths, logger)
        
        # Pass 2: Filter dark + blurry
        logger.info("\n[PASS 2] Filtering dark and blurry...")
        results = filter_dark_and_blurry(scores, frame_paths, logger)
        
        # Pass 3: Detect duplicates
        logger.info("\n[PASS 3] Detecting near-duplicates...")
        results = detect_duplicates(
            results, frame_paths, blur_scores, embeddings, logger
        )
        
        # Save results
        logger.info("\n[SAVING] Writing results...")
        for i, result in enumerate(results):
            try:
                save_frame(frame_paths[i], dirs, result["status"])
            except Exception as e:
                logger.error(f"Error saving frame {i} ({frame_paths[i].name}): {e}")
        
        # Report
        logger.info("\n[REPORTING] Generating report...")
        report_path = write_report(results, output_dir)
        logger.info(f"Report saved: {report_path}")
        
        # Summary
        total = len(results)
        kept = sum(1 for r in results if r["status"] == "KEEP")
        discarded = total - kept
        
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total frames: {total}")
        logger.info(f"Kept: {kept} ({100*kept/total:.1f}%)")
        logger.info(f"Discarded: {discarded} ({100*discarded/total:.1f}%)")
        logger.info("=" * 60)
        
        print("\n✅ Pipeline completed successfully!")
        print(f"   Output: {output_dir}")
        
    except Exception as e:
        logger.error(f"PIPELINE FAILED: {e}", exc_info=True)
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="360° Video Quality Assessment Pipeline"
    )
    parser.add_argument(
        "input",
        help="Input directory containing equirectangular frames"
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Output directory (default: output)"
    )
    
    args = parser.parse_args()
    main(args.input, args.output)
