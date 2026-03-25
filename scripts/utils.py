from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def to_path(path_str: str | Path) -> Path:
    """Convert user input to a normalized Path object."""
    return Path(path_str).expanduser()


def ensure_dir(path: str | Path) -> Path:
    """Create a directory when it does not exist and return its Path."""
    directory = to_path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def require_existing_dir(path: str | Path, name: str) -> Path:
    """Validate that a directory exists."""
    directory = to_path(path)
    if not directory.exists():
        raise FileNotFoundError(f"{name} does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"{name} is not a directory: {directory}")
    return directory


def require_existing_file(path: str | Path, name: str) -> Path:
    """Validate that a file exists."""
    file_path = to_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{name} does not exist: {file_path}")
    if not file_path.is_file():
        raise FileNotFoundError(f"{name} is not a file: {file_path}")
    return file_path


def list_images(directory: str | Path) -> list[Path]:
    """Collect supported image files from a directory."""
    root = require_existing_dir(directory, "Image directory")
    return sorted(
        path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def list_xyz_files(directory: str | Path) -> list[Path]:
    """Collect xyz files from a directory."""
    root = require_existing_dir(directory, "XYZ directory")
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() == ".xyz")


def print_header(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{title}\n{line}")


def print_paths(items: Iterable[tuple[str, Path]]) -> None:
    """Print a compact list of path values."""
    for label, path in items:
        print(f"{label}: {path}")


def check_command_available(command: str) -> bool:
    """Return True when an executable can be found on PATH."""
    return shutil.which(command) is not None
