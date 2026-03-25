from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
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
        description="Prepare wheel images, compute basic quality metrics, and export a screening report."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data") / "images_2d",
        help="Directory that contains the original 2D images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "prepared_images",
        help="Directory for the image report and filtered image folders.",
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=100.0,
        help="Reject images when Laplacian variance is below this threshold.",
    )
    parser.add_argument(
        "--brightness-threshold",
        type=float,
        default=40.0,
        help="Mark images as dim when mean grayscale brightness is below this threshold.",
    )
    parser.add_argument(
        "--copy-selected",
        type=parse_bool,
        default=False,
        help="Whether to copy passed images to images_selected and rejected images to images_rejected.",
    )
    return parser.parse_args()


def collect_image_paths(input_dir: Path) -> list[Path]:
    root = require_existing_dir(input_dir, "Input image directory")
    image_paths = [
        path for path in root.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(image_paths, key=lambda path: path.name.lower())


def analyze_image(image_path: Path, blur_threshold: float, brightness_threshold: float) -> dict[str, object]:
    blur_threshold = float(blur_threshold)
    brightness_threshold = float(brightness_threshold)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    readable = image is not None

    record: dict[str, object] = {
        "filename": image_path.name,
        "width": None,
        "height": None,
        "brightness_mean": None,
        "sharpness_laplacian_var": None,
        "readable": readable,
        "rejected": False,
        "reject_reason": "",
        "is_dim": False,
        "source_path": str(image_path),
    }

    if not readable:
        record["rejected"] = True
        record["reject_reason"] = "unreadable"
        return record

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness_mean = float(gray.mean())
    sharpness_laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    record["width"] = width
    record["height"] = height
    record["brightness_mean"] = round(brightness_mean, 4)
    record["sharpness_laplacian_var"] = round(sharpness_laplacian_var, 4)
    record["is_dim"] = float(brightness_mean) < float(brightness_threshold)

    if float(sharpness_laplacian_var) < float(blur_threshold):
        record["rejected"] = True
        record["reject_reason"] = "blur"

    return record


def copy_image(image_path: Path, target_dir: Path) -> Path:
    ensure_dir(target_dir)
    target_path = target_dir / image_path.name
    shutil.copy2(image_path, target_path)
    return target_path


def run(
    input_dir: Path,
    output_dir: Path,
    blur_threshold: float,
    brightness_threshold: float,
    copy_selected: bool,
) -> Path:
    blur_threshold = float(blur_threshold)
    brightness_threshold = float(brightness_threshold)
    image_paths = collect_image_paths(input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {input_dir}")

    output_dir = ensure_dir(output_dir)
    selected_dir = ensure_dir(output_dir / "images_selected")
    rejected_dir = ensure_dir(output_dir / "images_rejected")
    report_path = output_dir / "image_report.csv"

    print_header("Prepare Images")
    print_paths(
        [
            ("Input directory", input_dir),
            ("Output directory", output_dir),
            ("Selected directory", selected_dir),
            ("Rejected directory", rejected_dir),
        ]
    )
    print(f"Blur threshold: {blur_threshold}")
    print(f"Brightness threshold: {brightness_threshold}")
    print(f"Copy selected images: {copy_selected}")

    rows: list[dict[str, object]] = []
    readable_count = 0
    passed_count = 0
    rejected_count = 0

    for image_path in image_paths:
        record = analyze_image(image_path, blur_threshold, brightness_threshold)
        rows.append(record)

        if record["readable"]:
            readable_count += 1

        if record["rejected"]:
            rejected_count += 1
            if copy_selected:
                copy_image(image_path, rejected_dir)
        else:
            passed_count += 1
            if copy_selected:
                copy_image(image_path, selected_dir)

    report = pd.DataFrame(rows)
    try:
        report.to_csv(report_path, index=False, encoding="utf-8-sig")
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot write report to {report_path}. "
            "The file may be open in another program, so an older report can remain unchanged."
        ) from exc

    print(f"Report saved to: {report_path}")
    print(f"Total images: {len(image_paths)}")
    print(f"Readable images: {readable_count}")
    print(f"Passed filter: {passed_count}")
    print(f"Rejected images: {rejected_count}")
    print(f"Output directory: {output_dir}")

    return report_path


def main() -> int:
    args = parse_args()
    try:
        run(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            blur_threshold=args.blur_threshold,
            brightness_threshold=args.brightness_threshold,
            copy_selected=args.copy_selected,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
