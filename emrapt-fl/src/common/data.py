from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class TabularDataMetadata:
    feature_columns: list[str]
    num_features: int
    train_size: int
    test_size: int
    target_column: str
    id_column: str


def load_tabular_data(
    csv_path: str | Path,
    *,
    id_column: str = "ID",
    target_column: str = "Target",
    test_size: float = 0.2,
    seed: int = 42,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, TabularDataMetadata]:
    """Load a tabular radiomics CSV into train and test DataLoaders."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    dataframe = pd.read_csv(path)
    _validate_columns(dataframe, id_column=id_column, target_column=target_column)

    feature_frame = dataframe.drop(columns=[id_column, target_column])
    feature_columns = list(feature_frame.columns)
    labels = dataframe[target_column].astype("float32")

    if feature_frame.empty:
        raise ValueError("CSV must include at least one feature column")

    features = feature_frame.astype("float32")
    stratify = labels if labels.nunique() > 1 else None

    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)

    train_loader = _build_loader(
        train_features_scaled,
        train_labels.to_numpy(),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = _build_loader(
        test_features_scaled,
        test_labels.to_numpy(),
        batch_size=batch_size,
        shuffle=False,
    )

    metadata = TabularDataMetadata(
        feature_columns=feature_columns,
        num_features=len(feature_columns),
        train_size=len(train_labels),
        test_size=len(test_labels),
        target_column=target_column,
        id_column=id_column,
    )
    return train_loader, test_loader, metadata


def _validate_columns(dataframe: pd.DataFrame, *, id_column: str, target_column: str) -> None:
    missing_columns = [
        column for column in (id_column, target_column) if column not in dataframe.columns
    ]
    if missing_columns:
        joined_columns = ", ".join(missing_columns)
        raise ValueError(f"CSV is missing required column(s): {joined_columns}")


def _build_loader(
    features: Any,
    labels: Any,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    feature_tensor = torch.tensor(features, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    dataset = TensorDataset(feature_tensor, label_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
