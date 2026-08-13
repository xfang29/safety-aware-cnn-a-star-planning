"""Learned traversal-cost construction from U-Net predictions."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from src.scene_generator import DrivingScene


@dataclass
class LearnedCostMaps:
    """CNN preference prediction and resulting traversal-cost map."""

    preference_map: np.ndarray
    learned_penalty: np.ndarray
    traversal_cost: np.ndarray


@torch.no_grad()
def predict_preference_map(
    model: nn.Module,
    scene: DrivingScene,
    device: torch.device,
) -> np.ndarray:
    """
    Predict a path-preference map for one driving scene.

    Returns:
        Array with shape (H, W) and values in [0, 1].
    """
    model.eval()

    model_input = torch.from_numpy(
        scene.model_input
    ).unsqueeze(0)

    model_input = model_input.to(
        device=device,
        dtype=torch.float32,
    )

    logits = model(
        model_input
    )

    prediction = torch.sigmoid(
        logits
    )

    preference_map = (
        prediction[0, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    return preference_map


def build_learned_traversal_cost(
    scene: DrivingScene,
    preference_map: np.ndarray,
    learned_weight: float = 8.0,
) -> LearnedCostMaps:
    """
    Convert a CNN preference map into an A* traversal-cost map.

    High predicted preference produces a low penalty.
    Low predicted preference produces a high penalty.

    Obstacles and cells outside the road remain hard constraints.
    """
    if preference_map.shape != scene.road_mask.shape:
        raise ValueError(
            "preference_map must have the same spatial shape as the scene."
        )

    if learned_weight < 0:
        raise ValueError(
            "learned_weight must be nonnegative."
        )

    if not np.all(
        np.isfinite(preference_map)
    ):
        raise ValueError(
            "preference_map must contain only finite values."
        )

    preference_map = np.clip(
        preference_map,
        0.0,
        1.0,
    )

    # High preference -> low cost.
    learned_penalty = (
        1.0 - preference_map
    )

    traversal_cost = (
        1.0
        + learned_weight
        * learned_penalty
    )

    traversal_cost = traversal_cost.astype(
        np.float64
    )

    # Preserve hard safety constraints.
    traversal_cost[
        ~scene.free_space
    ] = np.inf

    return LearnedCostMaps(
        preference_map=preference_map,
        learned_penalty=learned_penalty,
        traversal_cost=traversal_cost,
    )