"""Training utilities for learned path-preference prediction."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train the model for one epoch and return the mean loss."""
    model.train()

    total_loss = 0.0
    total_samples = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(inputs)

        loss = criterion(
            logits,
            targets,
        )

        loss.backward()

        optimizer.step()

        batch_size = inputs.shape[0]

        total_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate the model and return the mean validation loss."""
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        logits = model(inputs)

        loss = criterion(
            logits,
            targets,
        )

        batch_size = inputs.shape[0]

        total_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

    return total_loss / total_samples


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    number_of_epochs: int,
    checkpoint_path: str | Path,
) -> dict[str, list[float]]:
    """
    Train a model and save the checkpoint with the best validation loss.

    Returns:
        Dictionary containing training and validation loss histories.
    """
    checkpoint_path = Path(
        checkpoint_path
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = {
        "train_loss": [],
        "validation_loss": [],
    }

    best_validation_loss = float(
        "inf"
    )

    for epoch in range(
        1,
        number_of_epochs + 1,
    ):
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        validation_loss = validate_one_epoch(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(
            train_loss
        )

        history[
            "validation_loss"
        ].append(
            validation_loss
        )

        improved = (
            validation_loss
            < best_validation_loss
        )

        if improved:
            best_validation_loss = (
                validation_loss
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "validation_loss":
                        validation_loss,
                },
                checkpoint_path,
            )

        marker = " *" if improved else ""

        print(
            f"Epoch {epoch:02d}/"
            f"{number_of_epochs:02d} | "
            f"Train loss: {train_loss:.6f} | "
            f"Validation loss: "
            f"{validation_loss:.6f}"
            f"{marker}"
        )

    return history