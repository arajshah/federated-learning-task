import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.common.config import load_config
from src.common.data import load_tabular_data
from src.common.metrics import binary_classification_metrics
from src.common.model import TabularMLP
from src.common.utils import set_seed


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / total_examples


def evaluate(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float | None]]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = loss_fn(logits, labels)
            probabilities = torch.sigmoid(logits)

            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            all_labels.append(labels.cpu().numpy().ravel())
            all_probabilities.append(probabilities.cpu().numpy().ravel())

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probabilities)
    metrics = binary_classification_metrics(y_true, y_prob)
    return total_loss / total_examples, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one client locally.")
    parser.add_argument("--config", required=True, help="Path to a client YAML config.")
    return parser.parse_args()


def resolve_project_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    project_root = config_path.resolve().parent.parent
    return project_root / path


def get_hidden_dims(hidden_dim: int | list[int]) -> list[int]:
    if isinstance(hidden_dim, int):
        return [hidden_dim]
    return hidden_dim


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)

    client_id = config["client_id"]
    seed = int(config.get("seed", 42))
    set_seed(seed)

    data_path = resolve_project_path(config_path, config["data_path"])
    output_dir = resolve_project_path(config_path, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader, metadata = load_tabular_data(
        data_path,
        id_column=config.get("id_col", "ID"),
        target_column=config.get("target_col", "Target"),
        test_size=float(config.get("test_size", 0.2)),
        seed=seed,
        batch_size=int(config.get("batch_size", 32)),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabularMLP(
        input_dim=metadata.num_features,
        hidden_dims=get_hidden_dims(config.get("hidden_dim", 64)),
        dropout=float(config.get("dropout", 0.0)),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("learning_rate", 0.001)),
    )
    loss_fn = nn.BCEWithLogitsLoss()

    epochs = int(config.get("epochs", 5))
    print(
        f"Training {client_id} on {data_path} "
        f"({metadata.train_size} train / {metadata.test_size} test)"
    )
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        _, metrics = evaluate(model, test_loader, loss_fn, device)
        roc_auc = metrics["roc_auc"]
        roc_auc_text = "undefined" if roc_auc is None else f"{roc_auc:.4f}"
        print(
            f"Epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.4f} "
            f"test_accuracy={metrics['accuracy']:.4f} "
            f"test_roc_auc={roc_auc_text}"
        )

    checkpoint_path = output_dir / f"{client_id}_model.pt"
    torch.save(
        {
            "client_id": client_id,
            "model_state_dict": model.state_dict(),
            "config": config,
            "feature_columns": metadata.feature_columns,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
