from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from .config import ExperimentConfig
from .pipeline import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate RNN/LSTM/GRU models for index close forecasting."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write all results. Defaults to outputs/run_<timestamp>",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max training epochs.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Override early stopping patience.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Override sequence window size.",
    )
    parser.add_argument(
        "--max-trials-per-model",
        type=int,
        default=None,
        help="Limit number of hyperparameter combinations per model for quick runs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string for torch (e.g., cpu, cuda, auto).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable saving prediction plots.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path(f"outputs/run_{timestamp}")

    config = ExperimentConfig(output_dir=output_dir)

    if args.max_trials_per_model is not None:
        config.max_trials_per_model = args.max_trials_per_model

    if args.max_epochs is not None:
        config.train.max_epochs = args.max_epochs

    if args.patience is not None:
        config.train.patience = args.patience

    if args.batch_size is not None:
        config.train.batch_size = args.batch_size

    if args.window_size is not None:
        config.train.window_size = args.window_size

    if args.device is not None:
        config.train.device = args.device

    if args.no_plots:
        config.save_plots = False

    return config


def main() -> None:
    args = parse_args()
    config = build_config(args)
    summary = run_experiment(config)
    print("\nTop models by validation RMSE:")
    print(summary[["symbol", "model_type", "val_rmse", "test_rmse"]].head(12).to_string(index=False))
    print(f"\nSaved outputs to: {config.output_dir}")


if __name__ == "__main__":
    main()
