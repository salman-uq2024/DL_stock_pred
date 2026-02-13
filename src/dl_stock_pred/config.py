from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class SplitConfig:
    """Chronological split configuration by target year."""

    train_end_year: int = 2023
    val_year: int = 2024
    test_year: int = 2025


@dataclass
class TrainingConfig:
    """Model training controls."""

    window_size: int = 20
    batch_size: int = 64
    max_epochs: int = 60
    patience: int = 10
    weight_decay: float = 0.0
    seed: int = 42
    device: str = "auto"


@dataclass
class SearchSpace:
    """Hyperparameter grid for recurrent models."""

    hidden_sizes: Sequence[int] = (32, 64, 128)
    num_layers: Sequence[int] = (1, 2)
    learning_rates: Sequence[float] = (1e-3, 5e-4)
    dropout: Sequence[float] = (0.0,)


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    data_files: dict[str, Path] = field(
        default_factory=lambda: {
            "sp500": Path("data/raw/sp500_5y.csv"),
            "nasdaq100": Path("data/raw/nasdaq100_5y.csv"),
            "dowjones": Path("data/raw/dowjones_5y.csv"),
        }
    )
    feature_candidates: Sequence[str] = (
        "open",
        "high",
        "low",
        "close",
        "range_hl",
        "return_1d",
        "change_pct_lag1",
        "volume",
    )
    target_column: str = "close"
    model_types: Sequence[str] = ("gru", "lstm", "rnn")
    output_dir: Path = Path("outputs/latest")
    max_trials_per_model: int | None = None
    save_plots: bool = True

    split: SplitConfig = field(default_factory=SplitConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    search: SearchSpace = field(default_factory=SearchSpace)
