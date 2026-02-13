from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class Metrics:
    rmse: float
    mae: float
    mape: float


@dataclass
class EvaluationOutput:
    y_true: np.ndarray
    y_pred: np.ndarray
    metrics: Metrics


@dataclass
class TrainOutput:
    history: pd.DataFrame
    best_state_dict: dict
    best_val_rmse: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _inverse_target(values_scaled: np.ndarray, target_scaler: RobustScaler) -> np.ndarray:
    return target_scaler.inverse_transform(values_scaled.reshape(-1, 1)).reshape(-1)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    error = y_pred - y_true
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))

    eps = 1e-8
    denom = np.maximum(np.abs(y_true), eps)
    mape = float(np.mean(np.abs(error) / denom) * 100.0)

    return Metrics(rmse=rmse, mae=mae, mape=mape)


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_scaler: RobustScaler,
) -> EvaluationOutput:
    model.eval()
    preds_scaled: list[np.ndarray] = []
    true_scaled: list[np.ndarray] = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            preds_scaled.append(preds)
            true_scaled.append(y_batch.numpy())

    y_pred_scaled = np.concatenate(preds_scaled)
    y_true_scaled = np.concatenate(true_scaled)

    y_pred = _inverse_target(y_pred_scaled, target_scaler)
    y_true = _inverse_target(y_true_scaled, target_scaler)

    metrics = compute_metrics(y_true=y_true, y_pred=y_pred)
    return EvaluationOutput(y_true=y_true, y_pred=y_pred, metrics=metrics)


def train_with_early_stopping(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    target_scaler: RobustScaler,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    weight_decay: float = 0.0,
) -> TrainOutput:
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    model.to(device)

    best_val_rmse = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    records: list[dict] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses: list[float] = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        val_eval = evaluate_model(
            model=model,
            loader=val_loader,
            device=device,
            target_scaler=target_scaler,
        )

        records.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_rmse": val_eval.metrics.rmse,
                "val_mae": val_eval.metrics.mae,
                "val_mape": val_eval.metrics.mape,
            }
        )

        if val_eval.metrics.rmse < best_val_rmse:
            best_val_rmse = val_eval.metrics.rmse
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)

    return TrainOutput(
        history=pd.DataFrame.from_records(records),
        best_state_dict=best_state,
        best_val_rmse=best_val_rmse,
    )
