from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from utils import ensure_dir, list_xyz_files, print_header, print_paths, require_existing_dir, require_existing_file


matplotlib.use("Agg")
import matplotlib.pyplot as plt


CROP_RATIO = 0.65


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a basic point-cloud validation workflow. This is not an industrial-grade accuracy evaluation."
    )
    parser.add_argument(
        "--pred-xyz",
        type=Path,
        default=Path("outputs") / "reconstruction" / "predicted.xyz",
        help="Predicted point cloud in XYZ format.",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("data") / "gt_xyz",
        help="Directory that contains ground-truth XYZ files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "evaluation",
        help="Directory for evaluation summaries.",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default="xyz_summary.csv",
        help="CSV file name written into the output directory.",
    )
    return parser.parse_args()


def require_open3d():
    try:
        import open3d as o3d
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "open3d is required for the basic ICP validation workflow. Please install open3d in this Python environment."
        ) from exc
    return o3d


def load_xyz_points(xyz_path: Path) -> np.ndarray:
    points_frame = pd.read_csv(
        xyz_path,
        sep=r"\s+",
        header=None,
        usecols=[0, 1, 2],
        engine="python",
    )
    points = points_frame.to_numpy(dtype=float)
    if points.size == 0:
        raise ValueError(f"XYZ file is empty: {xyz_path}")
    return points


def points_to_point_cloud(points: np.ndarray):
    o3d = require_open3d()
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud


def load_pred_point_cloud(point_path: Path):
    o3d = require_open3d()
    suffix = point_path.suffix.lower()
    if suffix == ".ply":
        point_cloud = o3d.io.read_point_cloud(str(point_path))
        points = np.asarray(point_cloud.points)
        if points.size == 0:
            raise ValueError(f"Predicted point cloud is empty or unreadable: {point_path}")
        return point_cloud, points, "ply"
    if suffix == ".xyz":
        points = load_xyz_points(point_path)
        return points_to_point_cloud(points), points, "xyz"
    raise ValueError(f"Unsupported predicted point cloud format: {point_path.suffix}")


def load_gt_point_cloud(point_path: Path):
    points = load_xyz_points(point_path)
    return points_to_point_cloud(points), points


def summarise_points_file(point_path: Path, points: np.ndarray, role: str, selected_for_icp: bool) -> dict[str, object]:
    xyz_min = points.min(axis=0)
    xyz_max = points.max(axis=0)
    extents = xyz_max - xyz_min
    centroid = points.mean(axis=0)
    return {
        "role": role,
        "selected_for_icp": selected_for_icp,
        "file_name": point_path.name,
        "file_path": str(point_path),
        "point_count": int(points.shape[0]),
        "bbox_size_x": float(extents[0]),
        "bbox_size_y": float(extents[1]),
        "bbox_size_z": float(extents[2]),
        "bbox_diagonal": float(np.linalg.norm(extents)),
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "centroid_z": float(centroid[2]),
        "x_min": float(xyz_min[0]),
        "y_min": float(xyz_min[1]),
        "z_min": float(xyz_min[2]),
        "x_max": float(xyz_max[0]),
        "y_max": float(xyz_max[1]),
        "z_max": float(xyz_max[2]),
    }


def choose_icp_reference(gt_items: list[dict[str, object]]) -> dict[str, object]:
    if not gt_items:
        raise FileNotFoundError("No usable ground-truth XYZ files were found for ICP.")
    return max(gt_items, key=lambda item: int(item["point_count"]))


def compute_cloud_stats(points: np.ndarray) -> dict[str, np.ndarray | float]:
    xyz_min = points.min(axis=0)
    xyz_max = points.max(axis=0)
    extents = xyz_max - xyz_min
    centroid = points.mean(axis=0)
    diagonal = float(np.linalg.norm(extents))
    return {
        "min": xyz_min,
        "max": xyz_max,
        "extents": extents,
        "centroid": centroid,
        "diagonal": diagonal,
    }


def crop_points_center_region(points: np.ndarray, crop_ratio: float = CROP_RATIO) -> np.ndarray:
    # This is a basic geometry crop intended to reduce support/background interference.
    # It is not a precise target segmentation for the wheel region.
    stats = compute_cloud_stats(points)
    centroid = np.asarray(stats["centroid"], dtype=float)
    extents = np.asarray(stats["extents"], dtype=float)
    half_range = (extents * float(crop_ratio)) / 2.0

    lower_bound = centroid - half_range
    upper_bound = centroid + half_range
    keep_mask = np.all((points >= lower_bound) & (points <= upper_bound), axis=1)
    cropped_points = points[keep_mask]
    if cropped_points.size == 0:
        raise ValueError("Center crop produced an empty point cloud. Please check the input point cloud geometry.")
    return cropped_points


def estimate_voxel_size(gt_diag: float) -> float:
    return max(float(gt_diag) / 120.0, 1e-3)


def build_prealigned_pred_cloud(pred_points: np.ndarray, gt_points: np.ndarray):
    o3d = require_open3d()
    pred_stats = compute_cloud_stats(pred_points)
    gt_stats = compute_cloud_stats(gt_points)

    pred_diag = float(pred_stats["diagonal"])
    gt_diag = float(gt_stats["diagonal"])
    if pred_diag <= 0 or gt_diag <= 0:
        raise ValueError("Predicted or ground-truth point cloud has zero bbox diagonal, cannot perform scale alignment.")

    scale_factor = gt_diag / pred_diag
    pred_centroid = np.asarray(pred_stats["centroid"], dtype=float)
    gt_centroid = np.asarray(gt_stats["centroid"], dtype=float)

    scaled_points = ((pred_points - pred_centroid) * scale_factor) + pred_centroid
    centroid_translation = gt_centroid - pred_centroid
    aligned_points = scaled_points + centroid_translation

    aligned_cloud = o3d.geometry.PointCloud()
    aligned_cloud.points = o3d.utility.Vector3dVector(aligned_points)
    return aligned_cloud, scale_factor, centroid_translation, pred_stats, gt_stats


def run_icp(pred_cloud, gt_cloud, gt_diag: float):
    o3d = require_open3d()
    voxel_size = estimate_voxel_size(gt_diag)
    pred_down = pred_cloud.voxel_down_sample(voxel_size)
    gt_down = gt_cloud.voxel_down_sample(voxel_size)
    if len(pred_down.points) == 0 or len(gt_down.points) == 0:
        raise ValueError("Downsampled point cloud is empty. Try using denser inputs or adjusting the data scale.")

    distance_threshold = max(float(gt_diag) * 0.03, voxel_size * 2.0)
    registration = o3d.pipelines.registration.registration_icp(
        pred_down,
        gt_down,
        distance_threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    pred_registered = o3d.geometry.PointCloud(pred_down)
    pred_registered.transform(registration.transformation.copy())
    distances = np.asarray(pred_registered.compute_point_cloud_distance(gt_down), dtype=float)
    return pred_down, gt_down, pred_registered, registration, distances, distance_threshold


def save_icp_result(result_path: Path, selected_gt_path: Path, voxel_size: float, distance_threshold: float, registration) -> Path:
    lines = [
        "Basic ICP validation result",
        "This is a basic verification version and does not represent industrial-grade precision evaluation.",
        "If the GT point cloud is only a single-view local capture, the result is for reference only.",
        f"Selected GT file: {selected_gt_path}",
        f"Voxel size: {voxel_size:.6f}",
        f"ICP distance threshold: {distance_threshold:.6f}",
        f"Fitness: {registration.fitness:.6f}",
        f"Inlier RMSE: {registration.inlier_rmse:.6f}",
        "Transformation matrix:",
        np.array2string(registration.transformation, precision=6, suppress_small=False),
        "",
    ]
    result_path.write_text("\n".join(lines), encoding="utf-8")
    return result_path


def save_icp_result_extended(
    result_path: Path,
    selected_gt_path: Path,
    crop_ratio: float,
    cropped_pred_point_count: int,
    cropped_gt_point_count: int,
    pred_diag: float,
    gt_diag: float,
    scale_factor: float,
    centroid_translation: np.ndarray,
    voxel_size: float,
    distance_threshold: float,
    registration,
) -> Path:
    lines = [
        "Basic ICP validation result",
        "This is a basic verification version and does not represent industrial-grade precision evaluation.",
        "If the GT point cloud is only a single-view local capture, the result is for reference only.",
        "Current cropping is a basic geometric crop intended to reduce support/background interference.",
        "It does not represent precise wheel-only segmentation.",
        f"Selected GT file: {selected_gt_path}",
        f"Crop ratio: {crop_ratio:.6f}",
        f"Cropped pred point count: {cropped_pred_point_count}",
        f"Cropped GT point count: {cropped_gt_point_count}",
        f"Pred diag: {pred_diag:.6f}",
        f"GT diag: {gt_diag:.6f}",
        f"Scale factor: {scale_factor:.6f}",
        f"Centroid translation: [{centroid_translation[0]:.6f}, {centroid_translation[1]:.6f}, {centroid_translation[2]:.6f}]",
        f"Voxel size: {voxel_size:.6f}",
        f"ICP distance threshold: {distance_threshold:.6f}",
        f"Fitness: {registration.fitness:.6f}",
        f"Inlier RMSE: {registration.inlier_rmse:.6f}",
        "Transformation matrix:",
        np.array2string(registration.transformation, precision=6, suppress_small=False),
        "",
    ]
    result_path.write_text("\n".join(lines), encoding="utf-8")
    return result_path


def save_distance_histogram(hist_path: Path, distances: np.ndarray) -> Path:
    if distances.size == 0:
        raise ValueError("No nearest-neighbor distances were produced after ICP.")

    plt.figure(figsize=(8, 5))
    plt.hist(distances, bins=50, color="#4C78A8", edgecolor="black")
    plt.title("Pred-to-GT Distance Histogram")
    plt.xlabel("Distance")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(hist_path, dpi=150)
    plt.close()
    return hist_path


def run(pred_xyz: Path, gt_dir: Path, output_dir: Path, summary_name: str) -> Path:
    o3d = require_open3d()

    require_existing_file(pred_xyz, "Predicted XYZ file")
    require_existing_dir(gt_dir, "Ground-truth XYZ directory")
    output_dir = ensure_dir(output_dir)

    gt_files = list_xyz_files(gt_dir)
    if not gt_files:
        raise FileNotFoundError(f"No XYZ files found in: {gt_dir}")

    pred_cloud, pred_points, pred_file_type = load_pred_point_cloud(pred_xyz)

    gt_items: list[dict[str, object]] = []
    for gt_file in gt_files:
        gt_cloud, gt_points = load_gt_point_cloud(gt_file)
        gt_items.append(
            {
                "path": gt_file,
                "cloud": gt_cloud,
                "points": gt_points,
                "point_count": int(gt_points.shape[0]),
            }
        )

    selected_gt = choose_icp_reference(gt_items)
    cropped_pred_points = crop_points_center_region(pred_points, CROP_RATIO)
    cropped_gt_points = crop_points_center_region(selected_gt["points"], CROP_RATIO)
    cropped_pred_cloud = points_to_point_cloud(cropped_pred_points)
    cropped_gt_cloud = points_to_point_cloud(cropped_gt_points)

    pred_aligned_pre_icp, scale_factor, centroid_translation, pred_stats, gt_stats = build_prealigned_pred_cloud(
        cropped_pred_points,
        cropped_gt_points,
    )
    voxel_size = estimate_voxel_size(float(gt_stats["diagonal"]))
    pred_down, gt_down, pred_registered, registration, distances, distance_threshold = run_icp(
        pred_aligned_pre_icp,
        cropped_gt_cloud,
        float(gt_stats["diagonal"]),
    )

    print_header("Evaluate XYZ")
    print_paths(
        [
            ("Predicted XYZ", pred_xyz),
            ("Ground-truth directory", gt_dir),
            ("Output directory", output_dir),
        ]
    )
    print("Note: this is a basic validation version, not an industrial-grade precision evaluation.")
    print("Note: if the GT is a single-view local point cloud, the ICP result is only for reference.")
    print("Note: current cropping is only a basic geometric crop to reduce support/background interference.")
    print(f"Predicted file type: {pred_file_type}")
    print(f"Predicted point count: {int(pred_points.shape[0])}")
    print(f"Ground-truth files discovered: {len(gt_files)}")
    print(f"Selected GT for ICP: {selected_gt['path'].name}")
    print(f"Crop ratio: {CROP_RATIO:.2f}")
    print(f"Cropped pred point count: {cropped_pred_points.shape[0]}")
    print(f"Cropped GT point count: {cropped_gt_points.shape[0]}")
    print(f"Pred bbox diagonal: {float(pred_stats['diagonal']):.6f}")
    print(f"GT bbox diagonal: {float(gt_stats['diagonal']):.6f}")
    print(f"Scale factor: {scale_factor:.6f}")
    print(
        "Centroid translation: "
        f"[{centroid_translation[0]:.6f}, {centroid_translation[1]:.6f}, {centroid_translation[2]:.6f}]"
    )

    summary_rows = [summarise_points_file(pred_xyz, pred_points, "pred", False)]
    for gt_item in gt_items:
        summary_rows.append(
            summarise_points_file(
                gt_item["path"],
                gt_item["points"],
                "gt",
                gt_item["path"] == selected_gt["path"],
            )
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / summary_name
    icp_result_path = output_dir / "icp_result.txt"
    pred_cropped_path = output_dir / "pred_cropped_pre_icp.ply"
    gt_cropped_path = output_dir / "gt_cropped_pre_icp.ply"
    pred_aligned_pre_icp_path = output_dir / "pred_scaled_aligned_pre_icp.ply"
    pred_down_path = output_dir / "pred_downsampled.ply"
    gt_down_path = output_dir / "gt_downsampled.ply"
    pred_registered_path = output_dir / "pred_registered.ply"
    distance_hist_path = output_dir / "distance_hist.png"

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    save_icp_result_extended(
        icp_result_path,
        selected_gt["path"],
        CROP_RATIO,
        int(cropped_pred_points.shape[0]),
        int(cropped_gt_points.shape[0]),
        float(pred_stats["diagonal"]),
        float(gt_stats["diagonal"]),
        scale_factor,
        centroid_translation,
        voxel_size,
        distance_threshold,
        registration,
    )
    o3d.io.write_point_cloud(str(pred_cropped_path), cropped_pred_cloud)
    o3d.io.write_point_cloud(str(gt_cropped_path), cropped_gt_cloud)
    o3d.io.write_point_cloud(str(pred_aligned_pre_icp_path), pred_aligned_pre_icp)
    o3d.io.write_point_cloud(str(pred_down_path), pred_down)
    o3d.io.write_point_cloud(str(gt_down_path), gt_down)
    o3d.io.write_point_cloud(str(pred_registered_path), pred_registered)
    save_distance_histogram(distance_hist_path, distances)

    print(f"ICP fitness: {registration.fitness:.6f}")
    print(f"ICP inlier RMSE: {registration.inlier_rmse:.6f}")
    print(f"Summary saved to: {summary_path}")
    print(f"ICP result saved to: {icp_result_path}")

    return summary_path


def main() -> int:
    args = parse_args()
    try:
        run(args.pred_xyz, args.gt_dir, args.output_dir, args.summary_name)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
