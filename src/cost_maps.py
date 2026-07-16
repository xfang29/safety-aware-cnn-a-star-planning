"""Rule-based safety cost maps for grid-based path planning."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from src.scene_generator import DrivingScene


@dataclass
class SafetyCostMaps:
    """Components of the rule-based safety-aware traversal cost."""

    obstacle_distance: np.ndarray
    boundary_distance: np.ndarray

    obstacle_cost: np.ndarray
    boundary_cost: np.ndarray
    lane_cost: np.ndarray

    traversal_cost: np.ndarray


def build_safety_cost_maps(
    scene: DrivingScene,
    obstacle_weight: float = 8.0,
    boundary_weight: float = 4.0,
    lane_weight: float = 2.0,
    obstacle_sigma: float = 3.0,
    boundary_sigma: float = 2.5,
) -> SafetyCostMaps:
    """
    Construct safety-aware traversal costs for an expert planner.

    The total free-space traversal cost is

        1
        + obstacle_weight * obstacle_cost
        + boundary_weight * boundary_cost
        + lane_weight * lane_cost

    Obstacle cells and cells outside the road are assigned infinite
    cost and therefore remain hard constraints.

    Args:
        scene:
            Synthetic driving scene.
        obstacle_weight:
            Strength of the obstacle-proximity penalty.
        boundary_weight:
            Strength of the road-boundary penalty.
        lane_weight:
            Strength of the lane-deviation penalty.
        obstacle_sigma:
            Spatial decay scale of the obstacle penalty.
        boundary_sigma:
            Spatial decay scale of the boundary penalty.

    Returns:
        SafetyCostMaps containing all individual cost components and
        the combined traversal-cost map.
    """
    if obstacle_weight < 0:
        raise ValueError(
            "obstacle_weight must be nonnegative."
        )

    if boundary_weight < 0:
        raise ValueError(
            "boundary_weight must be nonnegative."
        )

    if lane_weight < 0:
        raise ValueError(
            "lane_weight must be nonnegative."
        )

    if obstacle_sigma <= 0:
        raise ValueError(
            "obstacle_sigma must be positive."
        )

    if boundary_sigma <= 0:
        raise ValueError(
            "boundary_sigma must be positive."
        )

    # For each non-obstacle cell, calculate the Euclidean distance
    # to the nearest obstacle cell.
    obstacle_distance = distance_transform_edt(
        ~scene.obstacle_mask
    ).astype(np.float64)

    # For each road cell, calculate the Euclidean distance to the
    # nearest non-road cell, which represents the road boundary.
    boundary_distance = distance_transform_edt(
        scene.road_mask
    ).astype(np.float64)

    # High near obstacles and exponentially smaller farther away.
    obstacle_cost = np.exp(
        -obstacle_distance / obstacle_sigma
    )

    # High near the road boundary and smaller toward the road interior.
    boundary_cost = np.exp(
        -boundary_distance / boundary_sigma
    )

    # The lane-center map is near 1 at the centerline and near 0 away
    # from it, so this conversion produces a deviation penalty.
    lane_cost = (
        1.0
        - scene.lane_center_map.astype(np.float64)
    )

    traversal_cost = (
        1.0
        + obstacle_weight * obstacle_cost
        + boundary_weight * boundary_cost
        + lane_weight * lane_cost
    )

    traversal_cost = traversal_cost.astype(
        np.float64
    )

    # Obstacles and non-road cells remain hard constraints.
    traversal_cost[~scene.free_space] = np.inf

    return SafetyCostMaps(
        obstacle_distance=obstacle_distance,
        boundary_distance=boundary_distance,
        obstacle_cost=obstacle_cost,
        boundary_cost=boundary_cost,
        lane_cost=lane_cost,
        traversal_cost=traversal_cost,
    )