from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

from utils import ensure_dir, print_header, print_paths, require_existing_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean COLMAP fused point clouds with downsampling, denoising, and DBSCAN clustering."
    )
    parser.add_argument(
        "--input-ply",
        type=Path,
        default=Path("outputs") / "colmap_workspace" / "dense" / "fused.ply",
        help="Input fused point cloud in PLY format.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "cleaned_pointcloud",
        help="Directory for cleaned point cloud outputs.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.005,
        help="Voxel size for downsampling.",
    )
    parser.add_argument(
        "--nb-neighbors",
        type=int,
        default=20,
        help="Number of neighbors used by statistical outlier removal.",
    )
    parser.add_argument(
        "--std-ratio",
        type=float,
        default=2.0,
        help="Standard deviation ratio used by statistical outlier removal.",
    )
    parser.add_argument(
        "--use-radius",
        action="store_true",
        help="Enable radius outlier removal after statistical filtering.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.015,
        help="Radius used by radius outlier removal.",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=16,
        help="Minimum neighbors used by radius outlier removal.",
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=0.03,
        help="DBSCAN epsilon parameter.",
    )
    parser.add_argument(
        "--dbscan-min-points",
        type=int,
        default=30,
        help="DBSCAN minimum points parameter.",
    )
    return parser.parse_args()


def count_points(point_cloud: o3d.geometry.PointCloud) -> int:
    return int(len(point_cloud.points))


def require_non_empty(point_cloud: o3d.geometry.PointCloud, stage_name: str) -> None:
    if count_points(point_cloud) == 0:
        raise ValueError(f"Point cloud is empty after {stage_name}.")


def keep_indices(point_cloud: o3d.geometry.PointCloud, indices: list[int]) -> o3d.geometry.PointCloud:
    if not indices:
        raise ValueError("No points were kept after filtering.")
    return point_cloud.select_by_index(indices)


def extract_main_cluster(
    point_cloud: o3d.geometry.PointCloud,
    eps: float,
    min_points: int,
) -> tuple[o3d.geometry.PointCloud, int, int]:
    labels = np.asarray(point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False), dtype=int)
    if labels.size == 0:
        raise ValueError("DBSCAN did not return any labels.")

    valid_labels = labels[labels >= 0]
    if valid_labels.size == 0:
        raise ValueError("DBSCAN found no valid clusters. Try adjusting dbscan-eps or dbscan-min-points.")

    unique_labels, counts = np.unique(valid_labels, return_counts=True)
    cluster_count = int(unique_labels.size)
    main_label = int(unique_labels[np.argmax(counts)])
    main_cluster_indices = np.where(labels == main_label)[0].tolist()
    main_cluster = keep_indices(point_cloud, main_cluster_indices)
    return main_cluster, cluster_count, count_points(main_cluster)


def write_report(
    report_path: Path,
    original_count: int,
    downsampled_count: int,
    denoised_count: int,
    cluster_count: int,
    main_cluster_count: int,
) -> Path:
    main_cluster_ratio = 0.0 if denoised_count == 0 else main_cluster_count / denoised_count
    lines = [
        "Clean Fused Point Cloud Report",
        f"Original point count: {original_count}",
        f"Downsampled point count: {downsampled_count}",
        f"Denoised point count: {denoised_count}",
        f"Cluster count: {cluster_count}",
        f"Largest main cluster point count: {main_cluster_count}",
        f"Main cluster ratio: {main_cluster_ratio:.6f}",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run(
    input_ply: Path,
    output_dir: Path,
    voxel_size: float,
    nb_neighbors: int,
    std_ratio: float,
    use_radius: bool,
    radius: float,
    min_points: int,
    dbscan_eps: float,
    dbscan_min_points: int,
) -> Path:
    input_ply = require_existing_file(input_ply, "Input PLY file")
    output_dir = ensure_dir(output_dir)

    downsampled_path = output_dir / "fused_downsampled.ply"
    denoised_path = output_dir / "fused_denoised.ply"
    main_cluster_path = output_dir / "fused_main_cluster.ply"
    report_path = output_dir / "clean_report.txt"

    print_header("Clean Fused Point Cloud")
    print_paths(
        [
            ("Input PLY", input_ply),
            ("Output directory", output_dir),
            ("Downsampled output", downsampled_path),
            ("Denoised output", denoised_path),
            ("Main cluster output", main_cluster_path),
            ("Report output", report_path),
        ]
    )
    print(f"Voxel size: {voxel_size}")
    print(f"Statistical nb_neighbors: {nb_neighbors}")
    print(f"Statistical std_ratio: {std_ratio}")
    print(f"Use radius outlier removal: {use_radius}")
    print(f"Radius outlier radius: {radius}")
    print(f"Radius outlier min_points: {min_points}")
    print(f"DBSCAN eps: {dbscan_eps}")
    print(f"DBSCAN min_points: {dbscan_min_points}")

    point_cloud = o3d.io.read_point_cloud(str(input_ply))
    require_non_empty(point_cloud, "loading")
    original_count = count_points(point_cloud)
    print(f"Original point count: {original_count}")

    downsampled_cloud = point_cloud.voxel_down_sample(voxel_size)
    require_non_empty(downsampled_cloud, "voxel downsampling")
    downsampled_count = count_points(downsampled_cloud)
    print(f"Downsampled point count: {downsampled_count}")

    statistical_cloud, statistical_indices = downsampled_cloud.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    denoised_cloud = statistical_cloud
    require_non_empty(denoised_cloud, "statistical outlier removal")
    print(f"After statistical outlier removal: {count_points(denoised_cloud)}")

    if use_radius:
        radius_cloud, radius_indices = denoised_cloud.remove_radius_outlier(nb_points=min_points, radius=radius)
        denoised_cloud = radius_cloud
        require_non_empty(denoised_cloud, "radius outlier removal")
        print(f"After radius outlier removal: {count_points(denoised_cloud)}")

    denoised_count = count_points(denoised_cloud)
    main_cluster_cloud, cluster_count, main_cluster_count = extract_main_cluster(
        denoised_cloud,
        eps=dbscan_eps,
        min_points=dbscan_min_points,
    )
    print(f"Cluster count: {cluster_count}")
    print(f"Main cluster point count: {main_cluster_count}")
    print(f"Main cluster ratio: {main_cluster_count / denoised_count:.6f}")

    o3d.io.write_point_cloud(str(downsampled_path), downsampled_cloud)
    o3d.io.write_point_cloud(str(denoised_path), denoised_cloud)
    o3d.io.write_point_cloud(str(main_cluster_path), main_cluster_cloud)
    write_report(
        report_path,
        original_count=original_count,
        downsampled_count=downsampled_count,
        denoised_count=denoised_count,
        cluster_count=cluster_count,
        main_cluster_count=main_cluster_count,
    )

    print(f"Saved downsampled point cloud: {downsampled_path}")
    print(f"Saved denoised point cloud: {denoised_path}")
    print(f"Saved main cluster point cloud: {main_cluster_path}")
    print(f"Saved clean report: {report_path}")
    return report_path


def main() -> int:
    args = parse_args()
    try:
        run(
            input_ply=args.input_ply,
            output_dir=args.output_dir,
            voxel_size=args.voxel_size,
            nb_neighbors=args.nb_neighbors,
            std_ratio=args.std_ratio,
            use_radius=args.use_radius,
            radius=args.radius,
            min_points=args.min_points,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_points=args.dbscan_min_points,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
