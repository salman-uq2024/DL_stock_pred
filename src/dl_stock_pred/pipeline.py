from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data import load_index_dataframe, prepare_supervised_data
from .models import RecurrentRegressor
from .train import (
    EvaluationOutput,
    evaluate_model,
    make_loader,
    resolve_device,
    set_seed,
    train_with_early_stopping,
)


def _serialize_config(config: ExperimentConfig) -> dict[str, Any]:
    raw = asdict(config)
    data_files = {
        key: str(path)
        for key, path in config.data_files.items()
    }
    raw["data_files"] = data_files
    raw["output_dir"] = str(config.output_dir)
    return raw


def _iter_hyperparams(config: ExperimentConfig) -> list[dict[str, Any]]:
    combos = [
        {
            "hidden_size": hidden,
            "num_layers": layers,
            "learning_rate": lr,
            "dropout": dropout,
        }
        for hidden, layers, lr, dropout in product(
            config.search.hidden_sizes,
            config.search.num_layers,
            config.search.learning_rates,
            config.search.dropout,
        )
    ]

    if config.max_trials_per_model is not None:
        return combos[: config.max_trials_per_model]
    return combos


def _save_predictions(
    output_path: Path,
    dates: pd.Series,
    eval_output: EvaluationOutput,
) -> None:
    df = pd.DataFrame(
        {
            "date": dates.astype("datetime64[ns]"),
            "actual": eval_output.y_true,
            "predicted": eval_output.y_pred,
            "residual": eval_output.y_pred - eval_output.y_true,
        }
    )
    df.to_csv(output_path, index=False)


def _save_plot(
    output_path: Path,
    title: str,
    dates: pd.Series,
    eval_output: EvaluationOutput,
) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(dates.values, eval_output.y_true, label="actual", linewidth=2)
    plt.plot(dates.values, eval_output.y_pred, label="predicted", linewidth=1.5)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def _naive_baseline_rmse(
    test_X: np.ndarray,
    test_y: np.ndarray,
    target_scaler,
    close_feature_index: int,
) -> float:
    # Last available close in the sequence predicts next-step close.
    if test_X.shape[2] == 0:
        return float("nan")

    y_hat_scaled = test_X[:, -1, close_feature_index]

    y_hat = target_scaler.inverse_transform(y_hat_scaled.reshape(-1, 1)).reshape(-1)
    y_true = target_scaler.inverse_transform(test_y.reshape(-1, 1)).reshape(-1)
    return float(np.sqrt(np.mean((y_hat - y_true) ** 2)))


def run_experiment(config: ExperimentConfig) -> pd.DataFrame:
    set_seed(config.train.seed)
    device = resolve_device(config.train.device)

    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "config": _serialize_config(config),
    }
    (output_root / "run_config.json").write_text(json.dumps(metadata, indent=2))

    summary_rows: list[dict[str, Any]] = []

    for symbol, csv_path in config.data_files.items():
        index_dir = output_root / symbol
        index_dir.mkdir(parents=True, exist_ok=True)

        df = load_index_dataframe(Path(csv_path), symbol=symbol)
        prepared = prepare_supervised_data(
            df=df,
            feature_candidates=list(config.feature_candidates),
            target_column=config.target_column,
            split_cfg=config.split,
            train_cfg=config.train,
        )

        train_loader = make_loader(
            prepared.train_X,
            prepared.train_y,
            batch_size=config.train.batch_size,
            shuffle=True,
        )
        val_loader = make_loader(
            prepared.val_X,
            prepared.val_y,
            batch_size=config.train.batch_size,
            shuffle=False,
        )
        test_loader = make_loader(
            prepared.test_X,
            prepared.test_y,
            batch_size=config.train.batch_size,
            shuffle=False,
        )

        hyperparams = _iter_hyperparams(config)
        baseline_rmse = _naive_baseline_rmse(
            test_X=prepared.test_X,
            test_y=prepared.test_y,
            target_scaler=prepared.target_scaler,
            close_feature_index=prepared.feature_columns.index("close"),
        )

        per_model_best: list[dict[str, Any]] = []

        for model_type in config.model_types:
            best_trial: dict[str, Any] | None = None

            for trial_id, hp in enumerate(hyperparams, start=1):
                model = RecurrentRegressor(
                    model_type=model_type,
                    input_size=len(prepared.feature_columns),
                    hidden_size=int(hp["hidden_size"]),
                    num_layers=int(hp["num_layers"]),
                    dropout=float(hp["dropout"]),
                )

                train_out = train_with_early_stopping(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    target_scaler=prepared.target_scaler,
                    learning_rate=float(hp["learning_rate"]),
                    max_epochs=config.train.max_epochs,
                    patience=config.train.patience,
                    weight_decay=config.train.weight_decay,
                )

                val_eval = evaluate_model(
                    model=model,
                    loader=val_loader,
                    device=device,
                    target_scaler=prepared.target_scaler,
                )
                test_eval = evaluate_model(
                    model=model,
                    loader=test_loader,
                    device=device,
                    target_scaler=prepared.target_scaler,
                )

                candidate = {
                    "symbol": symbol,
                    "model_type": model_type,
                    "trial_id": trial_id,
                    "params": hp,
                    "history": train_out.history,
                    "val_eval": val_eval,
                    "test_eval": test_eval,
                    "val_rmse": val_eval.metrics.rmse,
                    "test_rmse": test_eval.metrics.rmse,
                    "test_mae": test_eval.metrics.mae,
                    "test_mape": test_eval.metrics.mape,
                    "epochs_trained": int(train_out.history["epoch"].max()),
                }

                if best_trial is None or candidate["val_rmse"] < best_trial["val_rmse"]:
                    best_trial = candidate

            assert best_trial is not None

            model_tag = f"{symbol}_{model_type}"
            best_trial["history"].to_csv(index_dir / f"history_{model_tag}.csv", index=False)
            _save_predictions(
                output_path=index_dir / f"predictions_{model_tag}.csv",
                dates=prepared.test_dates,
                eval_output=best_trial["test_eval"],
            )

            if config.save_plots:
                _save_plot(
                    output_path=index_dir / f"plot_{model_tag}.png",
                    title=f"{symbol.upper()} - {model_type.upper()} (test year {config.split.test_year})",
                    dates=prepared.test_dates,
                    eval_output=best_trial["test_eval"],
                )

            record = {
                "symbol": symbol,
                "model_type": model_type,
                "window_size": config.train.window_size,
                "batch_size": config.train.batch_size,
                "hidden_size": int(best_trial["params"]["hidden_size"]),
                "num_layers": int(best_trial["params"]["num_layers"]),
                "learning_rate": float(best_trial["params"]["learning_rate"]),
                "dropout": float(best_trial["params"]["dropout"]),
                "epochs_trained": best_trial["epochs_trained"],
                "val_rmse": best_trial["val_rmse"],
                "test_rmse": best_trial["test_rmse"],
                "test_mae": best_trial["test_mae"],
                "test_mape_pct": best_trial["test_mape"],
                "naive_baseline_test_rmse": baseline_rmse,
                "feature_columns": ",".join(prepared.feature_columns),
                "n_train": prepared.train_X.shape[0],
                "n_val": prepared.val_X.shape[0],
                "n_test": prepared.test_X.shape[0],
            }

            summary_rows.append(record)
            per_model_best.append(record)

        champion = min(per_model_best, key=lambda x: x["val_rmse"])
        (index_dir / "champion.json").write_text(json.dumps(champion, indent=2))

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["symbol", "val_rmse", "test_rmse"]
    )
    summary_df.to_csv(output_root / "summary.csv", index=False)

    # A compact human-readable report
    report_lines = [
        "# Experiment Summary",
        "",
    ]
    for symbol, symbol_df in summary_df.groupby("symbol"):
        champ = symbol_df.nsmallest(1, "val_rmse").iloc[0]
        report_lines.extend(
            [
                f"## {symbol}",
                f"- Champion model: `{champ['model_type']}`",
                f"- Val RMSE: `{champ['val_rmse']:.4f}`",
                f"- Test RMSE: `{champ['test_rmse']:.4f}`",
                f"- Test MAE: `{champ['test_mae']:.4f}`",
                f"- Test MAPE: `{champ['test_mape_pct']:.2f}%`",
                f"- Naive baseline test RMSE: `{champ['naive_baseline_test_rmse']:.4f}`",
                "",
            ]
        )

    (output_root / "SUMMARY.md").write_text("\n".join(report_lines))

    return summary_df
