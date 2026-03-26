# utils/io.py
# ─────────────────────────────────────────────
# Frame loading, saving, and CSV report writing.
# ─────────────────────────────────────────────

import cv2
import csv
import shutil
from pathlib import Path
from collections import Counter
from config import CONFIG


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_frames(input_dir: str) -> list[Path]:
    """
    Collect all image paths from input directory, sorted by name.
    Sorting ensures temporal order is preserved.
    """
    input_path = Path(input_dir)
    frames = sorted([
        f for f in input_path.iterdir()
        if f.suffix.lower() in EXTENSIONS
    ])
    if not frames:
        raise FileNotFoundError(f"No image files found in {input_dir}")
    return frames


def setup_output_dirs(output_dir: str) -> dict[str, Path]:
    """
    Create output folder structure.
    Returns dict of named paths.
    """
    base         = Path(output_dir)
    kept_path    = base / "kept"
    discard_path = base / "discarded"

    kept_path.mkdir(parents=True, exist_ok=True)
    discard_path.mkdir(parents=True, exist_ok=True)

    return {
        "base"     : base,
        "kept"     : kept_path,
        "discarded": discard_path,
    }


def save_frame(frame_path: Path, dirs: dict, status: str):
    """
    Copy frame to kept/ or discarded/ folder based on status.
    Uses shutil.copy2 to preserve metadata.
    """
    if status == "KEEP":
        dest = dirs["kept"] / frame_path.name
    else:
        dest = dirs["discarded"] / frame_path.name

    if CONFIG["save_discarded"] or status == "KEEP":
        shutil.copy2(str(frame_path), str(dest))


def write_report(report: list[dict], output_dir: str):
    """
    Write full audit CSV — one row per frame with all scores and decision.
    """
    if not report:
        return

    report_path = Path(output_dir) / CONFIG["report_name"]
    fieldnames = list(report[0].keys())

    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report)

        # Append summary section at end of CSV
        writer.writerow({k: "" for k in fieldnames})
        for row in _build_summary_rows(report, fieldnames):
            writer.writerow(row)

    return report_path


def _build_summary_rows(report: list[dict], fieldnames: list[str]) -> list[dict]:
    """
    Build summary rows appended at end of report.csv.
    """
    total = len(report)
    kept = sum(1 for r in report if r.get("status") == "KEEP")
    discarded = total - kept

    bucket_counts = Counter()
    combo_counts = Counter()

    for row in report:
        reason = (row.get("reason") or "").strip()
        if not reason:
            continue

        tags = set()
        parts = [p.strip() for p in reason.split("|")]
        for part in parts:
            if part.startswith("absolute_blur"):
                tags.add("absolute_blur")
            elif part.startswith("patch_blur"):
                tags.add("patch_blur")
            elif part.startswith("too_dark"):
                tags.add("too_dark")
            elif "duplicate_of_" in part or part.startswith("duplicate_of_"):
                tags.add("duplicate")
            else:
                tags.add("other")

        for tag in tags:
            bucket_counts[tag] += 1
        combo_counts[" + ".join(sorted(tags))] += 1

    def row(frame: str, status: str = "", reason: str = "") -> dict:
        base = {k: "" for k in fieldnames}
        if "frame" in base:
            base["frame"] = frame
        if "status" in base:
            base["status"] = status
        if "reason" in base:
            base["reason"] = reason
        return base

    summary_rows = [
        row("__SUMMARY__"),
        row("total_frames", str(total)),
        row("kept_frames", str(kept)),
        row("discarded_frames", str(discarded)),
        row("discard_rate", f"{(100.0 * discarded / total):.1f}%" if total else "0.0%"),
        row(""),
        row("__REASON_BUCKETS__"),
        row("absolute_blur", str(bucket_counts.get("absolute_blur", 0))),
        row("patch_blur", str(bucket_counts.get("patch_blur", 0))),
        row("too_dark", str(bucket_counts.get("too_dark", 0))),
        row("duplicate", str(bucket_counts.get("duplicate", 0))),
        row("other", str(bucket_counts.get("other", 0))),
        row(""),
        row("__REASON_COMBINATIONS__"),
    ]

    for combo_name, count in sorted(combo_counts.items(), key=lambda x: (-x[1], x[0])):
        summary_rows.append(row(combo_name, str(count)))

    return summary_rows


def print_summary(report: list[dict]):
    """
    Print a clean summary to terminal after processing.
    """
    total    = len(report)
    kept     = sum(1 for r in report if r["status"] == "KEEP")
    dark     = sum(1 for r in report if r["reason"] and "dark" in r["reason"])
    blurry   = sum(1 for r in report if r["reason"] and "blur" in r["reason"])
    dupes    = sum(1 for r in report if r["reason"] and "duplicate" in r["reason"])
    discarded= total - kept

    print("\n" + "=" * 52)
    print(f"  ✅  PIPELINE COMPLETE")
    print("=" * 52)
    print(f"  Total frames     : {total}")
    print(f"  Kept (clean)     : {kept}  ({100*kept/total:.1f}%)")
    print(f"  Discarded        : {discarded}  ({100*discarded/total:.1f}%)")
    print(f"    ↳ Too dark     : {dark}")
    print(f"    ↳ Blurry       : {blurry}")
    print(f"    ↳ Duplicates   : {dupes}")
    print("=" * 52)
