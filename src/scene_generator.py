"""Synthetic driving-scene generation utilities."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DrivingScene:
    """A synthetic bird's-eye-view driving scene."""

    road_mask: np.ndarray
    obstacle_mask: np.ndarray
    lane_center_map: np.ndarray
    start_map: np.ndarray
    goal_map: np.ndarray

    start: tuple[int, int]
    goal: tuple[int, int]

    centerline_x: np.ndarray

    @property
    def free_space(self) -> np.ndarray:
        """Return drivable cells that are not occupied by obstacles."""
        return self.road_mask & (~self.obstacle_mask)

    @property
    def model_input(self) -> np.ndarray:
        """
        Stack the five input channels.

        Returns:
            Array with shape (5, height, width).
        """
        return np.stack(
            [
                self.road_mask.astype(np.float32),
                self.obstacle_mask.astype(np.float32),
                self.lane_center_map.astype(np.float32),
                self.start_map.astype(np.float32),
                self.goal_map.astype(np.float32),
            ],
            axis=0,
        )


def make_gaussian_point_map(
    height: int,
    width: int,
    point: tuple[int, int],
    sigma: float = 1.5,
) -> np.ndarray:
    """Create a Gaussian heatmap centered at a grid point."""
    row, col = point

    yy, xx = np.meshgrid(
        np.arange(height),
        np.arange(width),
        indexing="ij",
    )

    squared_distance = (
        (yy - row) ** 2
        + (xx - col) ** 2
    )

    heatmap = np.exp(
        -squared_distance / (2.0 * sigma**2)
    )

    return heatmap.astype(np.float32)


def generate_driving_scene(
    seed: Optional[int] = None,
    height: int = 64,
    width: int = 64,
    road_half_width: int = 10,
    min_obstacles: int = 2,
    max_obstacles: int = 5,
    max_generation_attempts: int = 100,
) -> DrivingScene:
    """
    Generate one valid synthetic driving-like grid scene.

    A valid scene contains a collision-free route between its start
    and goal. The path-existence function is imported locally to avoid
    a circular module dependency.
    """
    from src.planners import path_exists

    if height < 20 or width < 20:
        raise ValueError(
            "height and width must both be at least 20."
        )

    if road_half_width < 3:
        raise ValueError(
            "road_half_width must be at least 3."
        )

    if min_obstacles < 0:
        raise ValueError(
            "min_obstacles cannot be negative."
        )

    if max_obstacles < min_obstacles:
        raise ValueError(
            "max_obstacles must be greater than or equal "
            "to min_obstacles."
        )

    rng = np.random.default_rng(seed)

    _, xx = np.meshgrid(
        np.arange(height),
        np.arange(width),
        indexing="ij",
    )

    for _ in range(max_generation_attempts):
        normalized_row = (
            np.arange(height)
            / max(height - 1, 1)
        )

        curve_amplitude = rng.uniform(1.0, 5.0)
        curve_phase = rng.uniform(
            0.0,
            2.0 * np.pi,
        )

        centerline_x = (
            width / 2
            + curve_amplitude
            * np.sin(
                np.pi * normalized_row
                + curve_phase
            )
        )

        centerline_grid = centerline_x[:, None]

        distance_to_centerline = np.abs(
            xx - centerline_grid
        )

        road_mask = (
            distance_to_centerline
            <= road_half_width
        )

        start_row = height - 3
        goal_row = 2

        start_col = int(
            np.clip(
                round(centerline_x[start_row]),
                0,
                width - 1,
            )
        )

        goal_col = int(
            np.clip(
                round(centerline_x[goal_row]),
                0,
                width - 1,
            )
        )

        start = (start_row, start_col)
        goal = (goal_row, goal_col)

        obstacle_mask = np.zeros(
            (height, width),
            dtype=bool,
        )

        number_of_obstacles = int(
            rng.integers(
                min_obstacles,
                max_obstacles + 1,
            )
        )

        for _ in range(number_of_obstacles):
            obstacle_height = int(
                rng.integers(3, 8)
            )

            obstacle_width = int(
                rng.integers(3, 7)
            )

            obstacle_center_row = int(
                rng.integers(
                    8,
                    height - 8,
                )
            )

            local_centerline = centerline_x[
                obstacle_center_row
            ]

            lateral_offset = rng.uniform(
                -road_half_width + 3,
                road_half_width - 3,
            )

            obstacle_center_col = int(
                round(
                    local_centerline
                    + lateral_offset
                )
            )

            row_min = max(
                0,
                obstacle_center_row
                - obstacle_height // 2,
            )

            row_max = min(
                height,
                row_min + obstacle_height,
            )

            col_min = max(
                0,
                obstacle_center_col
                - obstacle_width // 2,
            )

            col_max = min(
                width,
                col_min + obstacle_width,
            )

            proposed_obstacle = np.zeros(
                (height, width),
                dtype=bool,
            )

            proposed_obstacle[
                row_min:row_max,
                col_min:col_max,
            ] = True

            proposed_obstacle &= road_mask

            proposed_obstacle[
                max(0, start_row - 5):
                min(height, start_row + 6),
                max(0, start_col - 5):
                min(width, start_col + 6),
            ] = False

            proposed_obstacle[
                max(0, goal_row - 5):
                min(height, goal_row + 6),
                max(0, goal_col - 5):
                min(width, goal_col + 6),
            ] = False

            obstacle_mask |= proposed_obstacle

        free_space = road_mask & (~obstacle_mask)

        if not path_exists(
            free_space=free_space,
            start=start,
            goal=goal,
        ):
            continue

        lane_sigma = 2.0

        lane_center_map = np.exp(
            -(distance_to_centerline**2)
            / (2.0 * lane_sigma**2)
        )

        lane_center_map *= road_mask
        lane_center_map = lane_center_map.astype(
            np.float32
        )

        start_map = make_gaussian_point_map(
            height=height,
            width=width,
            point=start,
            sigma=1.5,
        )

        goal_map = make_gaussian_point_map(
            height=height,
            width=width,
            point=goal,
            sigma=1.5,
        )

        return DrivingScene(
            road_mask=road_mask,
            obstacle_mask=obstacle_mask,
            lane_center_map=lane_center_map,
            start_map=start_map,
            goal_map=goal_map,
            start=start,
            goal=goal,
            centerline_x=centerline_x,
        )

    raise RuntimeError(
        "Unable to generate a valid scene after "
        f"{max_generation_attempts} attempts."
    )