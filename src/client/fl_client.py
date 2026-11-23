"""Flower client for local radiomics training on tabular data."""

import argparse
from pathlib import Path
from typing import Any

import flwr as fl
import numpy as np
import torch
from torch import nn

from src.client.train import evaluate, get_hidden_dims, resolve_project_path, train_one_epoch
from src.common.config import load_config
from src.common.data import TabularDataMetadata, load_tabular_data
from src.common.model import TabularMLP
from src.common.utils import set_seed


class RadiomicsClient(fl.client.NumPyClient):
    def __init__(
        self,
        *,
        client_id: str,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        train_size: int,
        test_size: int,
        local_epochs: int,
        learning_rate: float,
        device: torch.device,
    ) -> None:
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_size = train_size
        self.test_size = test_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.device = device
        self.loss_fn = nn.BCEWithLogitsLoss()

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        return get_model_parameters(self.model)

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        set_model_parameters(self.model, parameters)

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[list[np.ndarray], int, dict[str, float]]:
        self.set_parameters(parameters)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        train_loss = 0.0
        for _ in range(self.local_epochs):
            train_loss = train_one_epoch(
                self.model,
                self.train_loader,
                optimizer,
                self.loss_fn,
                self.device,
            )

        print(
            f"{self.client_id} fit complete: "
            f"local_epochs={self.local_epochs} train_loss={train_loss:.4f}"
        )
        return self.get_parameters(config), self.train_size, {"train_loss": train_loss}

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[float, int, dict[str, float | bool]]:
        self.set_parameters(parameters)
        loss, metrics = evaluate(self.model, self.test_loader, self.loss_fn, self.device)

        roc_auc = metrics["roc_auc"]
        roc_auc_text = "undefined" if roc_auc is None else f"{roc_auc:.4f}"
        print(
            f"{self.client_id} evaluate: "
            f"loss={loss:.4f} accuracy={metrics['accuracy']:.4f} "
            f"roc_auc={roc_auc_text}"
        )

        flower_metrics: dict[str, float | bool] = {
            "loss": loss,
            "accuracy": float(metrics["accuracy"]),
            "roc_auc_available": roc_auc is not None,
        }
        if roc_auc is not None:
            flower_metrics["roc_auc"] = float(roc_auc)
        return loss, self.test_size, flower_metrics


def get_model_parameters(model: nn.Module) -> list[np.ndarray]:
    return [parameter.cpu().numpy() for parameter in model.state_dict().values()]


def set_model_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    state_dict = model.state_dict()
    if len(parameters) != len(state_dict):
        raise ValueError(
            f"Expected {len(state_dict)} parameter arrays, got {len(parameters)}"
        )

    updated_state = {
        name: torch.tensor(parameter, dtype=tensor.dtype)
        for (name, tensor), parameter in zip(state_dict.items(), parameters)
    }
    model.load_state_dict(updated_state, strict=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start a Flower radiomics client.")
    parser.add_argument("--config", required=True, help="Path to a client YAML config.")
    parser.add_argument(
        "--server_address",
        help="Flower server address. Overrides server_address in the config.",
    )
    return parser.parse_args()


def require_config_value(config: dict[str, Any], key: str) -> Any:
    if key not in config:
        raise ValueError(f"Missing required client config key: {key}")
    return config[key]


def build_client(config_path: Path, server_address_override: str | None) -> tuple[str, RadiomicsClient]:
    config = load_config(config_path)
    client_id = str(require_config_value(config, "client_id"))
    seed = int(config.get("seed", 42))
    set_seed(seed)

    data_path = resolve_project_path(config_path, str(require_config_value(config, "data_path")))
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file for {client_id} not found: {data_path}. "
            "Generate synthetic data with `python scripts/generate_synthetic_data.py` "
            "or update data_path in the client config."
        )

    train_loader, test_loader, metadata = load_client_data(config, data_path, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabularMLP(
        input_dim=metadata.num_features,
        hidden_dims=get_hidden_dims(config.get("hidden_dim", 64)),
        dropout=float(config.get("dropout", 0.0)),
    ).to(device)

    server_address = server_address_override or str(
        require_config_value(config, "server_address")
    )
    client = RadiomicsClient(
        client_id=client_id,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        train_size=metadata.train_size,
        test_size=metadata.test_size,
        local_epochs=int(config.get("local_epochs", 1)),
        learning_rate=float(config.get("learning_rate", 0.001)),
        device=device,
    )
    print(
        f"Starting {client_id}: server={server_address} data={data_path} "
        f"({metadata.train_size} train / {metadata.test_size} test)"
    )
    return server_address, client


def load_client_data(
    config: dict[str, Any],
    data_path: Path,
    seed: int,
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    TabularDataMetadata,
]:
    return load_tabular_data(
        data_path,
        id_column=str(config.get("id_col", "ID")),
        target_column=str(config.get("target_col", "Target")),
        test_size=float(config.get("test_size", 0.2)),
        seed=seed,
        batch_size=int(config.get("batch_size", 32)),
    )


def main() -> None:
    args = parse_args()
    server_address, client = build_client(Path(args.config), args.server_address)
    fl.client.start_client(server_address=server_address, client=client.to_client())


if __name__ == "__main__":
    main()
