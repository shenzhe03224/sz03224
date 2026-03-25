from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from utils import ensure_dir, print_header, print_paths, require_existing_dir


SUPPORTED_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg"}


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment wheel masks with traditional image processing and export a report."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs") / "prepared_images_v2" / "images_selected",
        help="Directory that contains prepared wheel images.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=Path("data") / "masks",
        help="Directory used to save binary masks.",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=Path("outputs") / "segmentation" / "mask_preview",
        help="Directory used to save mask overlay previews.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("outputs") / "segmentation" / "mask_report.csv",
        help="CSV report path for segmentation results.",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.03,
        help="Minimum candidate area ratio relative to image area.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate images for cases without a valid mask.",
    )
    return parser.parse_args()


def collect_image_paths(input_dir: Path) -> list[Path]:
    root = require_existing_dir(input_dir, "Segmentation input directory")
    image_paths = [
        path for path in root.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(image_paths, key=lambda path: path.name.lower())


def apply_morphology(binary: np.ndarray) -> np.ndarray:
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)
    return cleaned


def score_contour(
    contour: np.ndarray,
    image_shape: tuple[int, int],
    min_area_ratio: float,
    variant_name: str,
) -> dict[str, object] | None:
    image_height, image_width = image_shape
    image_area = float(image_height * image_width)
    area = float(cv2.contourArea(contour))
    if area <= 0:
        return None

    area_ratio = area / image_area
    if area_ratio < float(min_area_ratio):
        return None

    x, y, w, h = cv2.boundingRect(contour)
    perimeter = float(cv2.arcLength(contour, True))
    circularity = 0.0 if perimeter <= 0 else (4.0 * math.pi * area) / (perimeter * perimeter)

    moments = cv2.moments(contour)
    if moments["m00"] > 0:
        center_x = float(moments["m10"] / moments["m00"])
        center_y = float(moments["m01"] / moments["m00"])
    else:
        center_x = float(x + w / 2.0)
        center_y = float(y + h / 2.0)

    image_center_x = image_width / 2.0
    image_center_y = image_height / 2.0
    max_distance = math.hypot(image_center_x, image_center_y)
    center_distance = math.hypot(center_x - image_center_x, center_y - image_center_y)
    center_score = 1.0 - min(center_distance / max_distance, 1.0)

    aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0.0
    score = (area_ratio * 2.0) + (max(0.0, circularity) * 1.5) + center_score + (aspect_ratio * 0.5)

    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    mask_area = int(cv2.countNonZero(mask))

    return {
        "mask": mask,
        "mask_area": mask_area,
        "image_area": int(image_area),
        "area_ratio": float(mask_area / image_area),
        "bbox_x": int(x),
        "bbox_y": int(y),
        "bbox_w": int(w),
        "bbox_h": int(h),
        "score": float(score),
        "variant": variant_name,
    }


def find_best_mask(gray: np.ndarray, min_area_ratio: float) -> tuple[dict[str, object] | None, dict[str, np.ndarray]]:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(clahe, (5, 5), 0)

    _, otsu_binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu_binary_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    processed_variants = {
        "otsu": apply_morphology(otsu_binary),
        "otsu_inv": apply_morphology(otsu_binary_inv),
    }
    debug_images = {
        "gray": gray,
        "clahe": clahe,
        "blurred": blurred,
        "otsu": processed_variants["otsu"],
        "otsu_inv": processed_variants["otsu_inv"],
    }

    best_candidate: dict[str, object] | None = None
    for variant_name, binary in processed_variants.items():
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            candidate = score_contour(contour, gray.shape[:2], min_area_ratio, variant_name)
            if candidate is None:
                continue
            if best_candidate is None or float(candidate["score"]) > float(best_candidate["score"]):
                best_candidate = candidate

    return best_candidate, debug_images


def build_preview(image: np.ndarray, candidate: dict[str, object]) -> np.ndarray:
    preview = image.copy()
    mask = candidate["mask"]

    color_mask = np.zeros_like(preview)
    color_mask[mask > 0] = (0, 255, 0)
    preview = cv2.addWeighted(preview, 0.7, color_mask, 0.3, 0.0)

    x = int(candidate["bbox_x"])
    y = int(candidate["bbox_y"])
    w = int(candidate["bbox_w"])
    h = int(candidate["bbox_h"])
    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return preview


def save_debug_images(debug_dir: Path, image_stem: str, debug_images: dict[str, np.ndarray]) -> None:
    ensure_dir(debug_dir)
    for stage_name, debug_image in debug_images.items():
        debug_path = debug_dir / f"{image_stem}_{stage_name}.png"
        cv2.imwrite(str(debug_path), debug_image)


def build_empty_row(image_path: Path, readable: bool) -> dict[str, object]:
    return {
        "filename": image_path.name,
        "readable": readable,
        "mask_found": False,
        "mask_area": 0,
        "image_area": 0,
        "area_ratio": 0.0,
        "bbox_x": None,
        "bbox_y": None,
        "bbox_w": None,
        "bbox_h": None,
    }


def run(
    input_dir: Path,
    mask_dir: Path,
    preview_dir: Path,
    report_path: Path,
    min_area_ratio: float,
    debug: bool,
) -> Path:
    image_paths = collect_image_paths(input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {input_dir}")

    mask_dir = ensure_dir(mask_dir)
    preview_dir = ensure_dir(preview_dir)
    ensure_dir(report_path.parent)
    debug_dir = Path("outputs") / "segmentation" / "debug"

    print_header("Segment Wheel")
    print_paths(
        [
            ("Input directory", input_dir),
            ("Mask directory", mask_dir),
            ("Preview directory", preview_dir),
            ("Report path", report_path),
        ]
    )
    print(f"Min area ratio: {float(min_area_ratio)}")
    print(f"Debug mode: {debug}")

    rows: list[dict[str, object]] = []
    readable_count = 0
    mask_found_count = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            rows.append(build_empty_row(image_path, readable=False))
            continue

        readable_count += 1
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        candidate, debug_images = find_best_mask(gray, float(min_area_ratio))

        if candidate is None:
            rows.append(build_empty_row(image_path, readable=True))
            if debug:
                save_debug_images(debug_dir, image_path.stem, debug_images)
            continue

        mask_found_count += 1
        mask_path = mask_dir / f"{image_path.stem}.png"
        preview_path = preview_dir / f"{image_path.stem}.png"
        preview = build_preview(image, candidate)

        cv2.imwrite(str(mask_path), candidate["mask"])
        cv2.imwrite(str(preview_path), preview)

        rows.append(
            {
                "filename": image_path.name,
                "readable": True,
                "mask_found": True,
                "mask_area": int(candidate["mask_area"]),
                "image_area": int(candidate["image_area"]),
                "area_ratio": round(float(candidate["area_ratio"]), 6),
                "bbox_x": int(candidate["bbox_x"]),
                "bbox_y": int(candidate["bbox_y"]),
                "bbox_w": int(candidate["bbox_w"]),
                "bbox_h": int(candidate["bbox_h"]),
            }
        )

    report = pd.DataFrame(rows)
    report.to_csv(report_path, index=False, encoding="utf-8-sig")

    failure_count = len(image_paths) - mask_found_count
    print(f"Report saved to: {report_path}")
    print(f"Total images: {len(image_paths)}")
    print(f"Readable images: {readable_count}")
    print(f"Generated masks: {mask_found_count}")
    print(f"Failed images: {failure_count}")
    print(f"Output directory: {mask_dir}")

    return report_path


def main() -> int:
    args = parse_args()
    try:
        run(
            input_dir=args.input_dir,
            mask_dir=args.mask_dir,
            preview_dir=args.preview_dir,
            report_path=args.report_path,
            min_area_ratio=args.min_area_ratio,
            debug=args.debug,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
