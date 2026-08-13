"""Evaluation utilities for CNN-guided path planning."""

from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn

from src.learned_cost import (
    build_learned_traversal_cost,
    predict_preference_map,
)
from src.metrics import evaluate_path
from src.cost_maps import build_safety_cost_maps
from src.planners import (
    astar_search,
    cost_aware_astar,
)
from src.scene_generator import generate_driving_scene


def evaluate_learned_weight_sweep(
    model: nn.Module,
    device: torch.device,
    seeds: Iterable[int],
    learned_weights: Iterable[float],
) -> dict[float, list[dict]]:
    """
    Evaluate CNN-guided A* over several learned-cost weights.

    The CNN prediction is computed only once per scene and reused
    for all candidate weights.
    """
    seed_values = [
        int(seed)
        for seed in seeds
    ]

    weight_values = [
        float(weight)
        for weight in learned_weights
    ]

    results = {
        weight: []
        for weight in weight_values
    }

    total_scenes = len(seed_values)

    for scene_index, seed in enumerate(
        seed_values,
        start=1,
    ):
        scene = generate_driving_scene(
            seed=seed
        )

        preference_map = predict_preference_map(
            model=model,
            scene=scene,
            device=device,
        )

        for weight in weight_values:
            learned_cost_maps = (
                build_learned_traversal_cost(
                    scene=scene,
                    preference_map=preference_map,
                    learned_weight=weight,
                )
            )

            planner_result = cost_aware_astar(
                free_space=scene.free_space,
                traversal_cost=(
                    learned_cost_maps.traversal_cost
                ),
                start=scene.start,
                goal=scene.goal,
            )

            metrics = evaluate_path(
                scene=scene,
                path=planner_result.path,
            )

            results[weight].append(
                {
                    "success": metrics.success,
                    "path_length":
                        metrics.path_length,
                    "minimum_obstacle_clearance":
                        metrics.minimum_obstacle_clearance,
                    "minimum_boundary_clearance":
                        metrics.minimum_boundary_clearance,
                    "mean_lane_deviation":
                        metrics.mean_lane_deviation,
                    "mean_turn_angle":
                        metrics.mean_turn_angle,
                    "expanded_nodes":
                        planner_result.expanded_nodes,
                    "planning_time_ms":
                        planner_result.planning_time * 1000,
                }
            )

        if (
            scene_index % 20 == 0
            or scene_index == total_scenes
        ):
            print(
                f"Evaluated "
                f"{scene_index}/{total_scenes} scenes."
            )

    return results


def summarize_planner_records(
    records: list[dict],
) -> dict[str, float]:
    """
    Calculate average metrics over successful planning runs.

    Metrics that are not present in the supplied records are skipped.
    """
    successful_records = [
        record
        for record in records
        if record["success"]
    ]

    if not successful_records:
        return {
            "success_rate": 0.0,
        }

    summary = {
        "success_rate":
            len(successful_records)
            / len(records)
    }

    metric_names = [
        "path_length",
        "minimum_obstacle_clearance",
        "minimum_boundary_clearance",
        "mean_lane_deviation",
        "mean_turn_angle",
        "expanded_nodes",
        "planning_time_ms",
        "distance_to_expert",
    ]

    for metric_name in metric_names:

        # Older evaluation records may not contain every metric.
        if not all(
            metric_name in record
            for record in successful_records
        ):
            continue

        values = np.asarray(
            [
                record[metric_name]
                for record in successful_records
            ],
            dtype=np.float64,
        )

        summary[metric_name] = float(
            np.mean(values)
        )

    return summary



def evaluate_validation_baselines(
    seeds: Iterable[int],
) -> tuple[list[dict], list[dict]]:
    """
    Evaluate standard A* and safety-aware expert A*
    on the same validation scenes.
    """
    standard_records = []
    expert_records = []

    seed_values = [
        int(seed)
        for seed in seeds
    ]

    total_scenes = len(seed_values)

    for scene_index, seed in enumerate(
        seed_values,
        start=1,
    ):
        scene = generate_driving_scene(
            seed=seed
        )

        # Standard A*
        standard_result = astar_search(
            free_space=scene.free_space,
            start=scene.start,
            goal=scene.goal,
        )

        standard_metrics = evaluate_path(
            scene=scene,
            path=standard_result.path,
        )

        standard_records.append(
            {
                "success": standard_metrics.success,
                "path_length":
                    standard_metrics.path_length,
                "minimum_obstacle_clearance":
                    standard_metrics.minimum_obstacle_clearance,
                "minimum_boundary_clearance":
                    standard_metrics.minimum_boundary_clearance,
                "mean_lane_deviation":
                    standard_metrics.mean_lane_deviation,
                "mean_turn_angle":
                    standard_metrics.mean_turn_angle,
                "expanded_nodes":
                    standard_result.expanded_nodes,
                "planning_time_ms":
                    standard_result.planning_time * 1000,
            }
        )

        # Expert A*
        safety_cost_maps = build_safety_cost_maps(
            scene
        )

        expert_result = cost_aware_astar(
            free_space=scene.free_space,
            traversal_cost=(
                safety_cost_maps.traversal_cost
            ),
            start=scene.start,
            goal=scene.goal,
        )

        expert_metrics = evaluate_path(
            scene=scene,
            path=expert_result.path,
        )

        expert_records.append(
            {
                "success": expert_metrics.success,
                "path_length":
                    expert_metrics.path_length,
                "minimum_obstacle_clearance":
                    expert_metrics.minimum_obstacle_clearance,
                "minimum_boundary_clearance":
                    expert_metrics.minimum_boundary_clearance,
                "mean_lane_deviation":
                    expert_metrics.mean_lane_deviation,
                "mean_turn_angle":
                    expert_metrics.mean_turn_angle,
                "expanded_nodes":
                    expert_result.expanded_nodes,
                "planning_time_ms":
                    expert_result.planning_time * 1000,
            }
        )

        if (
            scene_index % 20 == 0
            or scene_index == total_scenes
        ):
            print(
                f"Evaluated "
                f"{scene_index}/{total_scenes} scenes."
            )

    return (
        standard_records,
        expert_records,
    )

def evaluate_final_test(
    model: nn.Module,
    device: torch.device,
    seeds: Iterable[int],
    learned_weight: float = 2.0,
) -> dict[str, list[dict]]:
    """
    Evaluate Standard A*, Expert A*, and CNN-guided A*
    on a held-out set of scenes.
    """
    from src.metrics import symmetric_path_distance

    results = {
        "standard": [],
        "expert": [],
        "cnn": [],
    }

    seed_values = [
        int(seed)
        for seed in seeds
    ]

    total_scenes = len(seed_values)

    for scene_index, seed in enumerate(
        seed_values,
        start=1,
    ):
        scene = generate_driving_scene(
            seed=seed
        )

        # -------------------------------------------------
        # Standard A*
        # -------------------------------------------------
        standard_result = astar_search(
            free_space=scene.free_space,
            start=scene.start,
            goal=scene.goal,
        )

        standard_metrics = evaluate_path(
            scene=scene,
            path=standard_result.path,
        )

        # -------------------------------------------------
        # Safety-aware Expert A*
        # -------------------------------------------------
        expert_cost_maps = build_safety_cost_maps(
            scene
        )

        expert_result = cost_aware_astar(
            free_space=scene.free_space,
            traversal_cost=(
                expert_cost_maps.traversal_cost
            ),
            start=scene.start,
            goal=scene.goal,
        )

        expert_metrics = evaluate_path(
            scene=scene,
            path=expert_result.path,
        )

        # -------------------------------------------------
        # CNN-guided A*
        # -------------------------------------------------
        preference_map = predict_preference_map(
            model=model,
            scene=scene,
            device=device,
        )

        learned_cost_maps = (
            build_learned_traversal_cost(
                scene=scene,
                preference_map=preference_map,
                learned_weight=learned_weight,
            )
        )

        cnn_result = cost_aware_astar(
            free_space=scene.free_space,
            traversal_cost=(
                learned_cost_maps.traversal_cost
            ),
            start=scene.start,
            goal=scene.goal,
        )

        cnn_metrics = evaluate_path(
            scene=scene,
            path=cnn_result.path,
        )

        # -------------------------------------------------
        # Distance to expert path
        # -------------------------------------------------
        standard_expert_distance = (
            symmetric_path_distance(
                standard_result.path,
                expert_result.path,
            )
        )

        cnn_expert_distance = (
            symmetric_path_distance(
                cnn_result.path,
                expert_result.path,
            )
        )

        results["standard"].append(
            {
                "success": standard_metrics.success,
                "path_length":
                    standard_metrics.path_length,
                "minimum_obstacle_clearance":
                    standard_metrics.minimum_obstacle_clearance,
                "minimum_boundary_clearance":
                    standard_metrics.minimum_boundary_clearance,
                "mean_lane_deviation":
                    standard_metrics.mean_lane_deviation,
                "mean_turn_angle":
                    standard_metrics.mean_turn_angle,
                "expanded_nodes":
                    standard_result.expanded_nodes,
                "planning_time_ms":
                    standard_result.planning_time * 1000,
                "distance_to_expert":
                    standard_expert_distance,
            }
        )

        results["expert"].append(
            {
                "success": expert_metrics.success,
                "path_length":
                    expert_metrics.path_length,
                "minimum_obstacle_clearance":
                    expert_metrics.minimum_obstacle_clearance,
                "minimum_boundary_clearance":
                    expert_metrics.minimum_boundary_clearance,
                "mean_lane_deviation":
                    expert_metrics.mean_lane_deviation,
                "mean_turn_angle":
                    expert_metrics.mean_turn_angle,
                "expanded_nodes":
                    expert_result.expanded_nodes,
                "planning_time_ms":
                    expert_result.planning_time * 1000,
                "distance_to_expert": 0.0,
            }
        )

        results["cnn"].append(
            {
                "success": cnn_metrics.success,
                "path_length":
                    cnn_metrics.path_length,
                "minimum_obstacle_clearance":
                    cnn_metrics.minimum_obstacle_clearance,
                "minimum_boundary_clearance":
                    cnn_metrics.minimum_boundary_clearance,
                "mean_lane_deviation":
                    cnn_metrics.mean_lane_deviation,
                "mean_turn_angle":
                    cnn_metrics.mean_turn_angle,
                "expanded_nodes":
                    cnn_result.expanded_nodes,
                "planning_time_ms":
                    cnn_result.planning_time * 1000,
                "distance_to_expert":
                    cnn_expert_distance,
            }
        )

        if (
            scene_index % 20 == 0
            or scene_index == total_scenes
        ):
            print(
                f"Final test: "
                f"{scene_index}/{total_scenes} scenes."
            )

    return results