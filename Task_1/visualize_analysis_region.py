import argparse
from pathlib import Path

import cv2

from file_utils import load_frames
from utils.regions import get_analysis_region, draw_analysis_overlay


def _pick_indices(total: int, count: int) -> list[int]:
    if total <= 0:
        return []
    if total <= count:
        return list(range(total))
    step = max(1, total // count)
    indices = list(range(0, total, step))[:count]
    if (total - 1) not in indices:
        indices[-1] = total - 1
    return sorted(set(indices))


def main(input_dir: str, output_dir: str, samples: int) -> None:
    input_path = Path(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    frame_paths = load_frames(str(input_path))
    chosen = _pick_indices(len(frame_paths), samples)

    print(f"Found {len(frame_paths)} frames")
    print(f"Visualizing {len(chosen)} sample frames")

    for idx in chosen:
        frame_path = frame_paths[idx]
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"Skipping unreadable frame: {frame_path.name}")
            continue

        overlay = draw_analysis_overlay(frame)
        crop = get_analysis_region(frame)

        # Resize crop to full frame width for easy visual comparison
        crop_resized = cv2.resize(crop, (frame.shape[1], frame.shape[0]))

        combined = cv2.hconcat([overlay, crop_resized])

        out_name = f"sample_{idx+1:04d}_{frame_path.stem}.jpg"
        cv2.imwrite(str(out / out_name), combined)

    print(f"Saved previews to: {out}")
    print("Left: overlay on original | Right: extracted analysis region")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize analysis crop region for equirectangular frames")
    parser.add_argument("input", help="Input directory containing frames")
    parser.add_argument("--output", default="region_viz", help="Output directory for preview images")
    parser.add_argument("--samples", type=int, default=12, help="Number of sample frames to visualize")

    args = parser.parse_args()
    main(args.input, args.output, args.samples)
