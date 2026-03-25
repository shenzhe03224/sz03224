from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from utils import check_command_available, list_images, print_header, print_paths, require_existing_dir


COMMAND_FILE_NAME = "run_colmap_commands.txt"


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare COLMAP reconstruction commands and optionally run them."
    )
    parser.add_argument(
        "--colmap-path",
        type=str,
        default="colmap",
        help="COLMAP executable name or absolute path.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("outputs") / "prepared_images_v2" / "images_selected",
        help="Directory that contains input images for COLMAP.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=Path("data") / "masks",
        help="Directory that contains PNG masks with the same stem as the input images.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("outputs") / "colmap_workspace",
        help="Workspace directory for COLMAP outputs.",
    )
    parser.add_argument(
        "--database-path",
        type=Path,
        default=Path("outputs") / "colmap_workspace" / "database.db",
        help="COLMAP database path.",
    )
    parser.add_argument(
        "--camera-model",
        type=str,
        default="SIMPLE_RADIAL",
        help="COLMAP camera model passed to feature_extractor.",
    )
    parser.add_argument(
        "--matcher",
        type=str,
        choices=("exhaustive", "sequential"),
        default="exhaustive",
        help="Matcher type used before mapping.",
    )
    parser.add_argument(
        "--use-masks",
        type=parse_bool,
        default=False,
        help="Whether to use masks when matching PNG masks are available.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute COLMAP commands after generating the command file.",
    )
    return parser.parse_args()


def resolve_command_name(matcher: str) -> str:
    if matcher == "sequential":
        return "sequential_matcher"
    return "exhaustive_matcher"


def has_matching_masks(image_paths: list[Path], mask_dir: Path) -> bool:
    if not mask_dir.exists() or not mask_dir.is_dir():
        return False
    for image_path in image_paths:
        if (mask_dir / f"{image_path.stem}.png").exists():
            return True
    return False


def build_colmap_commands(
    colmap_path: str,
    image_dir: Path,
    database_path: Path,
    sparse_dir: Path,
    camera_model: str,
    matcher: str,
    use_masks: bool,
    mask_dir: Path,
) -> list[list[str]]:
    feature_command = [
        colmap_path,
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_dir),
        "--ImageReader.camera_model",
        camera_model,
    ]
    if use_masks:
        feature_command.extend(["--ImageReader.mask_path", str(mask_dir)])

    matcher_command = [
        colmap_path,
        resolve_command_name(matcher),
        "--database_path",
        str(database_path),
    ]
    mapper_command = [
        colmap_path,
        "mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_dir),
        "--output_path",
        str(sparse_dir),
    ]
    return [feature_command, matcher_command, mapper_command]


def format_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def write_command_file(command_path: Path, commands: list[list[str]]) -> Path:
    command_path.parent.mkdir(parents=True, exist_ok=True)
    command_text = "\n".join(format_command(command) for command in commands) + "\n"
    command_path.write_text(command_text, encoding="utf-8")
    return command_path


def execute_command(step_name: str, command: list[str]) -> None:
    print(f"{step_name} command:")
    print(format_command(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{step_name} failed with return code {completed.returncode}")


def run(
    colmap_path: str,
    image_dir: Path,
    mask_dir: Path,
    workspace: Path,
    database_path: Path,
    camera_model: str,
    matcher: str,
    use_masks: bool,
    execute: bool,
) -> Path:
    require_existing_dir(image_dir, "COLMAP image directory")
    workspace.mkdir(parents=True, exist_ok=True)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    sparse_dir = workspace / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(image_dir)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {image_dir}")

    colmap_available = check_command_available(colmap_path) if Path(colmap_path).name == colmap_path else Path(colmap_path).exists()
    if execute and not colmap_available:
        raise FileNotFoundError(f"COLMAP executable not found: {colmap_path}")

    masks_enabled = bool(use_masks) and has_matching_masks(image_paths, mask_dir)
    command_path = workspace / COMMAND_FILE_NAME
    commands = build_colmap_commands(
        colmap_path=colmap_path,
        image_dir=image_dir,
        database_path=database_path,
        sparse_dir=sparse_dir,
        camera_model=camera_model,
        matcher=matcher,
        use_masks=masks_enabled,
        mask_dir=mask_dir,
    )
    write_command_file(command_path, commands)

    print_header("Run COLMAP")
    print_paths(
        [
            ("COLMAP path", Path(colmap_path) if Path(colmap_path).name != colmap_path else Path(colmap_path)),
            ("Image directory", image_dir),
            ("Mask directory", mask_dir),
            ("Workspace directory", workspace),
            ("Database path", database_path),
            ("Sparse output directory", sparse_dir),
            ("Command file", command_path),
        ]
    )
    print(f"Discovered images: {len(image_paths)}")
    print(f"COLMAP executable available: {colmap_available}")
    print(f"Camera model: {camera_model}")
    print(f"Matcher: {matcher}")
    print(f"Use masks requested: {use_masks}")
    print(f"Use masks enabled: {masks_enabled}")
    print(f"Run commands: {execute}")
    print("Planned commands:")
    for command in commands:
        print(format_command(command))
    print(f"Command list saved to: {command_path}")

    if execute:
        execute_command("feature_extractor", commands[0])
        execute_command(resolve_command_name(matcher), commands[1])
        execute_command("mapper", commands[2])

    return command_path


def main() -> int:
    args = parse_args()
    try:
        run(
            colmap_path=args.colmap_path,
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            workspace=args.workspace,
            database_path=args.database_path,
            camera_model=args.camera_model,
            matcher=args.matcher,
            use_masks=args.use_masks,
            execute=args.run,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
