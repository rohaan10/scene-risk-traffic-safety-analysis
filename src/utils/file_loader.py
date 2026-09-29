"""
File Loader Utilities
─────────────────────
Directory walker and image helpers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

# Supported image extensions (case-insensitive check)
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(directory: str | Path) -> List[Path]:
    """
    Recursively find all image files under *directory*, sorted by name.

    Parameters
    ----------
    directory : str | Path
        Root folder to scan.

    Returns
    -------
    list[Path]
        Sorted list of absolute ``Path`` objects for each image found.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    images: list[Path] = []
    for root, _dirs, files in os.walk(directory):
        for fname in files:
            if Path(fname).suffix.lower() in _IMAGE_EXTENSIONS:
                images.append(Path(root) / fname)

    images.sort(key=lambda p: p.name)
    return images


def image_id_from_path(path: str | Path) -> str:
    """
    Derive a canonical ``image_id`` from a file path (stem without extension).

    Example
    -------
    >>> image_id_from_path("/data/raw_images/scene_042.jpg")
    'scene_042'
    """
    return Path(path).stem
