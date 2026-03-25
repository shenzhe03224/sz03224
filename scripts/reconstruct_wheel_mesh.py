from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

from utils import ensure_dir, print_header, print_paths, require_existing_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct a smoother wheel surface mesh from the cleaned main point-cloud cluster."
    )
    parser.add_argument(
        "--input-ply",
        type=Path,
        default=Path("outputs") / "cleaned_pointcloud" / "fused_main_cluster.ply",
        help="Input cleaned point cloud in PLY format.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "wheel_mesh",
        help="Directory for mesh reconstruction outputs.",
    )
    parser.add_argument(
        "--normal-radius",
        type=float,
        default=0.03,
        help="Radius used for point-cloud normal estimation.",
    )
    parser.add_argument(
        "--max-nn",
        type=int,
        default=30,
        help="Maximum neighbors used for point-cloud normal estimation.",
    )
    parser.add_argument(
        "--poisson-depth",
        type=int,
        default=8,
        help="Depth parameter for Poisson surface reconstruction.",
    )
    parser.add_argument(
        "--density-quantile",
        type=float,
        default=0.02,
        help="Lower density quantile removed from the Poisson mesh.",
    )
    parser.add_argument(
        "--smooth-iterations",
        type=int,
        default=5,
        help="Number of simple smoothing iterations applied to the mesh.",
    )
    return parser.parse_args()


def count_points(point_cloud: o3d.geometry.PointCloud) -> int:
    return int(len(point_cloud.points))


def count_vertices(mesh: o3d.geometry.TriangleMesh) -> int:
    return int(len(mesh.vertices))


def count_triangles(mesh: o3d.geometry.TriangleMesh) -> int:
    return int(len(mesh.triangles))


def require_non_empty_point_cloud(point_cloud: o3d.geometry.PointCloud, stage_name: str) -> None:
    if count_points(point_cloud) == 0:
        raise ValueError(f"Point cloud is empty after {stage_name}.")


def require_non_empty_mesh(mesh: o3d.geometry.TriangleMesh, stage_name: str) -> None:
    if count_vertices(mesh) == 0 or count_triangles(mesh) == 0:
        raise ValueError(f"Mesh reconstruction failed at {stage_name}. No valid vertices or triangles were produced.")


def filter_low_density_vertices(
    mesh: o3d.geometry.TriangleMesh,
    densities: np.ndarray,
    density_quantile: float,
) -> o3d.geometry.TriangleMesh:
    filtered_mesh = o3d.geometry.TriangleMesh(mesh)
    density_threshold = float(np.quantile(densities, density_quantile))
    remove_mask = densities < density_threshold
    filtered_mesh.remove_vertices_by_mask(remove_mask)
    filtered_mesh.remove_unreferenced_vertices()
    filtered_mesh.remove_degenerate_triangles()
    filtered_mesh.remove_duplicated_triangles()
    filtered_mesh.remove_duplicated_vertices()
    filtered_mesh.remove_non_manifold_edges()
    require_non_empty_mesh(filtered_mesh, "low-density filtering")
    return filtered_mesh


def write_report(
    report_path: Path,
    input_point_count: int,
    poisson_depth: int,
    initial_vertex_count: int,
    initial_triangle_count: int,
    filtered_vertex_count: int,
    filtered_triangle_count: int,
    smooth_vertex_count: int,
    smooth_triangle_count: int,
) -> Path:
    lines = [
        "Wheel Mesh Reconstruction Report",
        "This mesh is intended to look more continuous and presentation-friendly.",
        "It does not imply that geometric accuracy is necessarily improved.",
        f"Input point count: {input_point_count}",
        f"Poisson depth: {poisson_depth}",
        f"Initial mesh vertex count: {initial_vertex_count}",
        f"Initial mesh triangle count: {initial_triangle_count}",
        f"Filtered mesh vertex count: {filtered_vertex_count}",
        f"Filtered mesh triangle count: {filtered_triangle_count}",
        f"Smoothed mesh vertex count: {smooth_vertex_count}",
        f"Smoothed mesh triangle count: {smooth_triangle_count}",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run(
    input_ply: Path,
    output_dir: Path,
    normal_radius: float,
    max_nn: int,
    poisson_depth: int,
    density_quantile: float,
    smooth_iterations: int,
) -> Path:
    input_ply = require_existing_file(input_ply, "Input PLY file")
    output_dir = ensure_dir(output_dir)

    poisson_mesh_path = output_dir / "wheel_mesh_poisson.ply"
    smooth_mesh_path = output_dir / "wheel_mesh_smooth.ply"
    obj_mesh_path = output_dir / "wheel_mesh.obj"
    report_path = output_dir / "mesh_report.txt"

    print_header("Reconstruct Wheel Mesh")
    print_paths(
        [
            ("Input PLY", input_ply),
            ("Output directory", output_dir),
            ("Poisson mesh", poisson_mesh_path),
            ("Smooth mesh", smooth_mesh_path),
            ("OBJ mesh", obj_mesh_path),
            ("Report", report_path),
        ]
    )
    print(f"Normal radius: {normal_radius}")
    print(f"Max NN: {max_nn}")
    print(f"Poisson depth: {poisson_depth}")
    print(f"Density quantile: {density_quantile}")
    print(f"Smooth iterations: {smooth_iterations}")

    point_cloud = o3d.io.read_point_cloud(str(input_ply))
    require_non_empty_point_cloud(point_cloud, "loading")
    input_point_count = count_points(point_cloud)
    if input_point_count < 10:
        raise ValueError("Input point cloud has too few points for stable mesh reconstruction.")
    print(f"Input point count: {input_point_count}")

    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=max_nn)
    )
    orientation_neighbors = max(3, min(max_nn, input_point_count - 1))
    point_cloud.orient_normals_consistent_tangent_plane(orientation_neighbors)
    print("Normals estimated and oriented.")

    poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud,
        depth=poisson_depth,
    )
    require_non_empty_mesh(poisson_mesh, "Poisson reconstruction")
    densities = np.asarray(densities, dtype=float)
    initial_vertex_count = count_vertices(poisson_mesh)
    initial_triangle_count = count_triangles(poisson_mesh)
    print(f"Initial mesh vertices: {initial_vertex_count}")
    print(f"Initial mesh triangles: {initial_triangle_count}")

    filtered_mesh = filter_low_density_vertices(poisson_mesh, densities, density_quantile)
    filtered_vertex_count = count_vertices(filtered_mesh)
    filtered_triangle_count = count_triangles(filtered_mesh)
    print(f"After density filtering vertices: {filtered_vertex_count}")
    print(f"After density filtering triangles: {filtered_triangle_count}")

    smooth_mesh = filtered_mesh.filter_smooth_simple(number_of_iterations=smooth_iterations)
    smooth_mesh.remove_unreferenced_vertices()
    smooth_mesh.remove_degenerate_triangles()
    smooth_mesh.compute_vertex_normals()
    require_non_empty_mesh(smooth_mesh, "mesh smoothing")
    smooth_vertex_count = count_vertices(smooth_mesh)
    smooth_triangle_count = count_triangles(smooth_mesh)
    print(f"After smoothing vertices: {smooth_vertex_count}")
    print(f"After smoothing triangles: {smooth_triangle_count}")

    poisson_mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(poisson_mesh_path), poisson_mesh, write_vertex_normals=True)
    o3d.io.write_triangle_mesh(str(smooth_mesh_path), smooth_mesh, write_vertex_normals=True)
    o3d.io.write_triangle_mesh(str(obj_mesh_path), smooth_mesh, write_vertex_normals=True)
    write_report(
        report_path,
        input_point_count=input_point_count,
        poisson_depth=poisson_depth,
        initial_vertex_count=initial_vertex_count,
        initial_triangle_count=initial_triangle_count,
        filtered_vertex_count=filtered_vertex_count,
        filtered_triangle_count=filtered_triangle_count,
        smooth_vertex_count=smooth_vertex_count,
        smooth_triangle_count=smooth_triangle_count,
    )

    print(f"Saved Poisson mesh: {poisson_mesh_path}")
    print(f"Saved smooth mesh: {smooth_mesh_path}")
    print(f"Saved OBJ mesh: {obj_mesh_path}")
    print(f"Saved mesh report: {report_path}")
    return report_path


def main() -> int:
    args = parse_args()
    try:
        run(
            input_ply=args.input_ply,
            output_dir=args.output_dir,
            normal_radius=args.normal_radius,
            max_nn=args.max_nn,
            poisson_depth=args.poisson_depth,
            density_quantile=args.density_quantile,
            smooth_iterations=args.smooth_iterations,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
