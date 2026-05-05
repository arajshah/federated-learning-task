import argparse
import csv
from pathlib import Path
from typing import Any

import flwr as fl
import numpy as np
from flwr.common import NDArrays, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from src.common.config import load_config


class MetricsLogger:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.round_number = 0
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=["round", "loss", "accuracy", "roc_auc"],
            )
            writer.writeheader()

    def aggregate(self, metrics: list[tuple[int, dict[str, Scalar]]]) -> dict[str, Scalar]:
        self.round_number += 1
        aggregated = weighted_average(metrics)
        with self.output_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=["round", "loss", "accuracy", "roc_auc"],
            )
            writer.writerow(
                {
                    "round": self.round_number,
                    "loss": aggregated.get("loss", ""),
                    "accuracy": aggregated.get("accuracy", ""),
                    "roc_auc": aggregated.get("roc_auc", ""),
                }
            )
        print(f"Round {self.round_number} aggregated metrics: {aggregated}")
        return aggregated


class CheckpointingFedAvg(FedAvg):
    def __init__(self, *, final_parameters_path: Path, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.final_parameters_path = final_parameters_path
        self.latest_parameters: NDArrays | None = None

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, fl.common.FitRes]],
        failures: list[tuple[ClientProxy, fl.common.FitRes] | BaseException],
    ) -> tuple[fl.common.Parameters | None, dict[str, Scalar]]:
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if parameters is not None:
            self.latest_parameters = fl.common.parameters_to_ndarrays(parameters)
        return parameters, metrics

    def save_final_parameters(self) -> None:
        if self.latest_parameters is None:
            print("No global parameters were available to save.")
            return

        self.final_parameters_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            f"arr_{index}": array
            for index, array in enumerate(self.latest_parameters)
        }
        np.savez(self.final_parameters_path, **payload)
        print(f"Saved final global parameters: {self.final_parameters_path}")


def weighted_average(metrics: list[tuple[int, dict[str, Scalar]]]) -> dict[str, Scalar]:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    aggregated: dict[str, Scalar] = {}
    for metric_name in ("loss", "accuracy", "roc_auc"):
        weighted_sum = 0.0
        contributing_examples = 0
        for num_examples, client_metrics in metrics:
            value = client_metrics.get(metric_name)
            if isinstance(value, bool) or value is None:
                continue
            weighted_sum += float(value) * num_examples
            contributing_examples += num_examples
        if contributing_examples > 0:
            aggregated[metric_name] = weighted_sum / contributing_examples
    return aggregated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the local Flower FedAvg server.")
    parser.add_argument("--config", required=True, help="Path to server YAML config.")
    return parser.parse_args()


def require_config_value(config: dict[str, Any], key: str) -> Any:
    if key not in config:
        raise ValueError(f"Missing required server config key: {key}")
    return config[key]


def resolve_project_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    project_root = config_path.resolve().parent.parent
    return project_root / path


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)

    host = str(require_config_value(config, "host"))
    port = int(require_config_value(config, "port"))
    num_rounds = int(require_config_value(config, "num_rounds"))
    output_dir = resolve_project_path(config_path, str(require_config_value(config, "output_dir")))
    metrics_path = output_dir / "server_metrics.csv"
    final_parameters_path = output_dir / "final_global_parameters.npz"

    metrics_logger = MetricsLogger(metrics_path)
    strategy = CheckpointingFedAvg(
        final_parameters_path=final_parameters_path,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=int(config.get("min_fit_clients", 2)),
        min_evaluate_clients=int(config.get("min_evaluate_clients", 2)),
        min_available_clients=int(config.get("min_available_clients", 2)),
        evaluate_metrics_aggregation_fn=metrics_logger.aggregate,
    )

    server_address = f"{host}:{port}"
    print(
        f"Starting Flower FedAvg server on {server_address} "
        f"for {num_rounds} rounds. Waiting for clients..."
    )
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    strategy.save_final_parameters()


if __name__ == "__main__":
    main()
