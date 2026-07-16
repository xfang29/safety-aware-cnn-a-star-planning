"""Evaluation metrics for grid-based planned paths."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from src.planners import (
    GridPoint,
    calculate_path_length,
    validate_path,
)
from src.scene_generator import DrivingScene


@dataclass
class PathMetrics:
    """Quantitative properties of one planned path."""

    success: bool
    path_length: float

    minimum_obstacle_clearance: float
    mean_obstacle_clearance: float

    minimum_boundary_clearance: float

    mean_lane_deviation: float
    maximum_lane_deviation: float

    mean_turn_angle: float
    total_turn_angle: float


def calculate_turn_metrics(
    path: list[GridPoint],
) -> tuple[float, float]:
    """
    Calculate mean and total absolute heading changes in radians.

    A smaller value indicates a smoother path.
    """
    if len(path) < 3:
        return 0.0, 0.0

    path_array = np.asarray(
        path,
        dtype=np.float64,
    )

    steps = np.diff(
        path_array,
        axis=0,
    )

    headings = np.arctan2(
        steps[:, 0],
        steps[:, 1],
    )

    heading_changes = np.diff(headings)

    # Wrap angular differences into [-pi, pi].
    heading_changes = np.arctan2(
        np.sin(heading_changes),
        np.cos(heading_changes),
    )

    absolute_changes = np.abs(
        heading_changes
    )

    return (
        float(np.mean(absolute_changes)),
        float(np.sum(absolute_changes)),
    )


def evaluate_path(
    scene: DrivingScene,
    path: list[GridPoint],
) -> PathMetrics:
    """
    Evaluate the safety and geometric quality of a planned path.

    All distance-related values are measured in grid cells.
    Turn-angle values are measured in radians.
    """
    path_is_valid = validate_path(
        free_space=scene.free_space,
        path=path,
        start=scene.start,
        goal=scene.goal,
    )

    if not path_is_valid:
        return PathMetrics(
            success=False,
            path_length=np.inf,
            minimum_obstacle_clearance=np.nan,
            mean_obstacle_clearance=np.nan,
            minimum_boundary_clearance=np.nan,
            mean_lane_deviation=np.nan,
            maximum_lane_deviation=np.nan,
            mean_turn_angle=np.nan,
            total_turn_angle=np.nan,
        )

    path_array = np.asarray(
        path,
        dtype=np.int64,
    )

    path_rows = path_array[:, 0]
    path_cols = path_array[:, 1]

    obstacle_distance = distance_transform_edt(
        ~scene.obstacle_mask
    )

    boundary_distance = distance_transform_edt(
        scene.road_mask
    )

    obstacle_clearances = obstacle_distance[
        path_rows,
        path_cols,
    ]

    boundary_clearances = boundary_distance[
        path_rows,
        path_cols,
    ]

    centerline_columns = scene.centerline_x[
        path_rows
    ]

    lane_deviations = np.abs(
        path_cols - centerline_columns
    )

    mean_turn_angle, total_turn_angle = (
        calculate_turn_metrics(path)
    )

    return PathMetrics(
        success=True,
        path_length=calculate_path_length(path),
        minimum_obstacle_clearance=float(
            np.min(obstacle_clearances)
        ),
        mean_obstacle_clearance=float(
            np.mean(obstacle_clearances)
        ),
        minimum_boundary_clearance=float(
            np.min(boundary_clearances)
        ),
        mean_lane_deviation=float(
            np.mean(lane_deviations)
        ),
        maximum_lane_deviation=float(
            np.max(lane_deviations)
        ),
        mean_turn_angle=mean_turn_angle,
        total_turn_angle=total_turn_angle,
    )