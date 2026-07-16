"""Visualization utilities for scenes, cost maps, and planned paths."""

import matplotlib.pyplot as plt
import numpy as np

from src.cost_maps import SafetyCostMaps
from src.labels import PathLabelMaps
from src.planners import (
    AStarResult,
    CostAwareAStarResult,
    GridPoint,
)
from src.scene_generator import DrivingScene
def plot_safety_cost_maps(
    scene: DrivingScene,
    cost_maps: SafetyCostMaps,
) -> None:
    """
    Visualize the individual rule-based safety-cost components.

    Cells outside the free space are hidden so that infinite costs do
    not distort the displayed color scale.
    """
    displayed_maps = [
        cost_maps.obstacle_distance,
        cost_maps.boundary_distance,
        cost_maps.obstacle_cost,
        cost_maps.boundary_cost,
        cost_maps.lane_cost,
        cost_maps.traversal_cost,
    ]

    titles = [
        "Distance to Obstacles",
        "Distance to Road Boundary",
        "Obstacle Proximity Cost",
        "Boundary Proximity Cost",
        "Lane-Deviation Cost",
        "Total Traversal Cost",
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14, 9),
    )

    axes = axes.ravel()

    for axis, map_data, title in zip(
        axes,
        displayed_maps,
        titles,
    ):
        visible_map = np.where(
            scene.free_space,
            map_data,
            np.nan,
        )

        image = axis.imshow(
            visible_map,
            origin="upper",
        )

        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect("equal")

        fig.colorbar(
            image,
            ax=axis,
            fraction=0.046,
            pad=0.04,
        )

    plt.tight_layout()
    plt.show()


def _plot_scene_background(
    axis: plt.Axes,
    scene: DrivingScene,
) -> None:
    """
    Draw the common road, obstacle, centerline, start, and goal layers.
    """
    axis.imshow(
        scene.road_mask,
        cmap="gray",
        origin="upper",
    )

    obstacle_overlay = np.ma.masked_where(
        ~scene.obstacle_mask,
        scene.obstacle_mask,
    )

    axis.imshow(
        obstacle_overlay,
        cmap="Reds",
        vmin=0,
        vmax=1,
        alpha=1.0,
        origin="upper",
    )

    axis.plot(
        scene.centerline_x,
        np.arange(scene.road_mask.shape[0]),
        linestyle="--",
        linewidth=1.2,
        label="Lane center",
    )

    axis.scatter(
        scene.start[1],
        scene.start[0],
        marker="o",
        s=90,
        edgecolors="black",
        label="Start",
        zorder=5,
    )

    axis.scatter(
        scene.goal[1],
        scene.goal[0],
        marker="*",
        s=160,
        edgecolors="black",
        label="Goal",
        zorder=5,
    )

    axis.set_xlim(
        0,
        scene.road_mask.shape[1] - 1,
    )

    axis.set_ylim(
        scene.road_mask.shape[0] - 1,
        0,
    )

    axis.set_aspect("equal")
    axis.set_xlabel("Grid column")
    axis.set_ylabel("Grid row")


def plot_path_comparison(
    scene: DrivingScene,
    standard_result: AStarResult,
    expert_result: CostAwareAStarResult,
) -> None:
    """
    Compare standard geometric A* and safety-aware expert A*.
    """
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 6),
    )

    # Standard A*
    _plot_scene_background(
        axes[0],
        scene,
    )

    if standard_result.path:
        standard_path = np.asarray(
            standard_result.path
        )

        axes[0].plot(
            standard_path[:, 1],
            standard_path[:, 0],
            linewidth=2.5,
            label="Standard A* path",
        )

    axes[0].set_title(
        "Standard A*\n"
        f"Length: {standard_result.path_length:.2f}, "
        f"Expanded: {standard_result.expanded_nodes}"
    )

    axes[0].legend(
        loc="upper right"
    )

    # Safety-aware Expert A*
    _plot_scene_background(
        axes[1],
        scene,
    )

    if expert_result.path:
        expert_path = np.asarray(
            expert_result.path
        )

        axes[1].plot(
            expert_path[:, 1],
            expert_path[:, 0],
            linewidth=2.5,
            label="Expert A* path",
        )

    axes[1].set_title(
        "Safety-Aware Expert A*\n"
        f"Length: {expert_result.geometric_length:.2f}, "
        f"Expanded: {expert_result.expanded_nodes}"
    )

    axes[1].legend(
        loc="upper right"
    )

    plt.tight_layout()
    plt.show()

def plot_path_label_maps(
    scene: DrivingScene,
    label_maps: PathLabelMaps,
    expert_path: list[GridPoint],
) -> None:
    """
    Visualize the binary expert path, distance transform,
    Gaussian soft label, and expert path overlay.
    """
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 9),
    )

    axes[0, 0].imshow(
        label_maps.path_mask,
        origin="upper",
    )
    axes[0, 0].set_title(
        "Binary Expert Path Mask"
    )

    visible_distance = np.where(
        scene.free_space,
        label_maps.distance_to_path,
        np.nan,
    )

    distance_image = axes[0, 1].imshow(
        visible_distance,
        origin="upper",
    )
    axes[0, 1].set_title(
        "Distance to Expert Path"
    )

    fig.colorbar(
        distance_image,
        ax=axes[0, 1],
        fraction=0.046,
        pad=0.04,
    )

    soft_label_image = axes[1, 0].imshow(
        label_maps.soft_path_label,
        origin="upper",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1, 0].set_title(
        "Gaussian Soft Path Label"
    )

    fig.colorbar(
        soft_label_image,
        ax=axes[1, 0],
        fraction=0.046,
        pad=0.04,
    )

    _plot_scene_background(
        axes[1, 1],
        scene,
    )

    if expert_path:
        path_array = np.asarray(
            expert_path,
            dtype=np.int64,
        )

        axes[1, 1].plot(
            path_array[:, 1],
            path_array[:, 0],
            linewidth=2.5,
            label="Expert path",
        )

    axes[1, 1].set_title(
        "Expert Path on Driving Scene"
    )
    axes[1, 1].legend(
        loc="upper right"
    )

    for axis in axes.ravel():
        axis.set_aspect("equal")

    plt.tight_layout()
    plt.show()