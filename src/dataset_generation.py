"""Dataset generation utilities for learned path planning."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.cost_maps import build_safety_cost_maps
from src.labels import build_soft_path_label
from src.planners import cost_aware_astar, validate_path
from src.scene_generator import generate_driving_scene


@dataclass
class DatasetSplit:
    """Arrays stored for one generated dataset split."""

    inputs: np.ndarray
    soft_labels: np.ndarray
    path_masks: np.ndarray
    starts: np.ndarray
    goals: np.ndarray
    seeds: np.ndarray
    expert_lengths: np.ndarray

    @property
    def size(self) -> int:
        """Return the number of samples."""
        return int(self.inputs.shape[0])


def generate_dataset_split(
    seeds: Iterable[int],
    label_sigma: float = 2.0,
    scene_kwargs: dict | None = None,
    cost_kwargs: dict | None = None,
    progress_every: int = 100,
) -> DatasetSplit:
    """
    Generate one dataset split from a sequence of random seeds.
    """
    seed_values = [int(seed) for seed in seeds]

    if not seed_values:
        raise ValueError("At least one seed is required.")

    if label_sigma <= 0:
        raise ValueError("label_sigma must be positive.")

    scene_kwargs = {} if scene_kwargs is None else dict(scene_kwargs)
    cost_kwargs = {} if cost_kwargs is None else dict(cost_kwargs)

    input_samples = []
    soft_label_samples = []
    path_mask_samples = []
    starts = []
    goals = []
    expert_lengths = []

    total_samples = len(seed_values)

    for sample_index, seed in enumerate(seed_values, start=1):
        scene = generate_driving_scene(
            seed=seed,
            **scene_kwargs,
        )

        safety_cost_maps = build_safety_cost_maps(
            scene,
            **cost_kwargs,
        )

        expert_result = cost_aware_astar(
            free_space=scene.free_space,
            traversal_cost=safety_cost_maps.traversal_cost,
            start=scene.start,
            goal=scene.goal,
        )

        path_is_valid = validate_path(
            free_space=scene.free_space,
            path=expert_result.path,
            start=scene.start,
            goal=scene.goal,
        )

        if not expert_result.success or not path_is_valid:
            raise RuntimeError(
                f"Expert planning failed for seed {seed}."
            )

        label_maps = build_soft_path_label(
            scene=scene,
            path=expert_result.path,
            sigma=label_sigma,
        )

        input_samples.append(scene.model_input)

        # Add a channel dimension: (1, height, width)
        soft_label_samples.append(
            label_maps.soft_path_label[None, ...]
        )

        path_mask_samples.append(
            label_maps.path_mask[None, ...].astype(np.uint8)
        )

        starts.append(scene.start)
        goals.append(scene.goal)
        expert_lengths.append(expert_result.geometric_length)

        should_report = (
            progress_every > 0
            and (
                sample_index % progress_every == 0
                or sample_index == total_samples
            )
        )

        if should_report:
            print(
                f"Generated {sample_index}/{total_samples} samples."
            )

    return DatasetSplit(
        inputs=np.stack(
            input_samples,
            axis=0,
        ).astype(np.float32),

        soft_labels=np.stack(
            soft_label_samples,
            axis=0,
        ).astype(np.float32),

        path_masks=np.stack(
            path_mask_samples,
            axis=0,
        ).astype(np.uint8),

        starts=np.asarray(
            starts,
            dtype=np.int16,
        ),

        goals=np.asarray(
            goals,
            dtype=np.int16,
        ),

        seeds=np.asarray(
            seed_values,
            dtype=np.int64,
        ),

        expert_lengths=np.asarray(
            expert_lengths,
            dtype=np.float32,
        ),
    )


def save_dataset_split(
    dataset_split: DatasetSplit,
    output_path: str | Path,
) -> Path:
    """Save a dataset split as a compressed NumPy archive."""
    output_path = Path(output_path)

    if output_path.suffix.lower() != ".npz":
        output_path = output_path.with_suffix(".npz")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        inputs=dataset_split.inputs,
        soft_labels=dataset_split.soft_labels,
        path_masks=dataset_split.path_masks,
        starts=dataset_split.starts,
        goals=dataset_split.goals,
        seeds=dataset_split.seeds,
        expert_lengths=dataset_split.expert_lengths,
    )

    return output_path


def load_dataset_split(
    input_path: str | Path,
) -> DatasetSplit:
    """Load a previously saved dataset split."""
    input_path = Path(input_path)

    with np.load(
        input_path,
        allow_pickle=False,
    ) as archive:
        return DatasetSplit(
            inputs=archive["inputs"],
            soft_labels=archive["soft_labels"],
            path_masks=archive["path_masks"],
            starts=archive["starts"],
            goals=archive["goals"],
            seeds=archive["seeds"],
            expert_lengths=archive["expert_lengths"],
        )