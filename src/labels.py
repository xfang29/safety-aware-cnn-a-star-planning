"""Supervision-label generation for learned path planning."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from src.planners import GridPoint, validate_path
from src.scene_generator import DrivingScene


@dataclass
class PathLabelMaps:
    """Binary and soft supervision maps generated from an expert path."""

    path_mask: np.ndarray
    distance_to_path: np.ndarray
    soft_path_label: np.ndarray


def build_soft_path_label(
    scene: DrivingScene,
    path: list[GridPoint],
    sigma: float = 2.0,
) -> PathLabelMaps:
    """
    Convert a valid expert path into a Gaussian soft path label.

    The expert path itself has label value 1.0. Label values decrease
    smoothly as the distance from the path increases.

    Cells outside the drivable free space are assigned zero.

    Args:
        scene:
            Synthetic driving scene associated with the expert path.
        path:
            Expert path represented as a list of (row, column) cells.
        sigma:
            Width of the Gaussian path corridor, measured in grid cells.

    Returns:
        PathLabelMaps containing:
        - binary path mask,
        - Euclidean distance to the expert path,
        - Gaussian soft path label.
    """
    if sigma <= 0:
        raise ValueError(
            "sigma must be positive."
        )

    path_is_valid = validate_path(
        free_space=scene.free_space,
        path=path,
        start=scene.start,
        goal=scene.goal,
    )

    if not path_is_valid:
        raise ValueError(
            "The expert path must be valid before generating labels."
        )

    height, width = scene.road_mask.shape

    path_mask = np.zeros(
        (height, width),
        dtype=bool,
    )

    path_array = np.asarray(
        path,
        dtype=np.int64,
    )

    path_rows = path_array[:, 0]
    path_cols = path_array[:, 1]

    path_mask[
        path_rows,
        path_cols,
    ] = True

    # The path cells are zero-valued in ~path_mask, so the distance
    # transform returns the distance to the nearest path cell.
    distance_to_path = distance_transform_edt(
        ~path_mask
    ).astype(np.float32)

    soft_path_label = np.exp(
        -(
            distance_to_path**2
        )
        / (
            2.0 * sigma**2
        )
    )

    # Obstacles and cells outside the road are not valid targets.
    soft_path_label *= scene.free_space

    soft_path_label = soft_path_label.astype(
        np.float32
    )

    return PathLabelMaps(
        path_mask=path_mask,
        distance_to_path=distance_to_path,
        soft_path_label=soft_path_label,
    )