import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_client_dataframe(
    *,
    client_id: str,
    num_rows: int,
    num_features: int,
    mean_shift: float,
    scale: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    features = rng.normal(loc=mean_shift, scale=scale, size=(num_rows, num_features))
    weights = rng.normal(size=num_features)
    logits = features @ weights / np.sqrt(num_features)
    logits += rng.normal(scale=0.5, size=num_rows)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    targets = rng.binomial(1, probabilities)

    feature_columns = {
        f"feature_{index:02d}": features[:, index]
        for index in range(num_features)
    }
    dataframe = pd.DataFrame(feature_columns)
    dataframe.insert(0, "Target", targets)
    dataframe.insert(0, "ID", [f"{client_id}_{index:04d}" for index in range(num_rows)])
    return dataframe


def write_client_csv(
    *,
    output_path: Path,
    client_id: str,
    num_rows: int,
    num_features: int,
    mean_shift: float,
    scale: float,
    seed: int,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        print(f"Skipping existing file: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = make_client_dataframe(
        client_id=client_id,
        num_rows=num_rows,
        num_features=num_features,
        mean_shift=mean_shift,
        scale=scale,
        seed=seed,
    )
    dataframe.to_csv(output_path, index=False)
    print(f"Wrote {len(dataframe)} rows to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic heterogeneous client CSV files."
    )
    parser.add_argument("--rows", type=int, default=120)
    parser.add_argument("--features", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows < 100:
        raise ValueError("--rows must be at least 100")
    if args.features < 1:
        raise ValueError("--features must be positive")

    write_client_csv(
        output_path=Path("data/client1/radiomics.csv"),
        client_id="client1",
        num_rows=args.rows,
        num_features=args.features,
        mean_shift=0.0,
        scale=1.0,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    write_client_csv(
        output_path=Path("data/client2/radiomics.csv"),
        client_id="client2",
        num_rows=args.rows,
        num_features=args.features,
        mean_shift=0.6,
        scale=1.3,
        seed=args.seed + 1,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
