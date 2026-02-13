from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from .config import SplitConfig, TrainingConfig


@dataclass
class PreparedData:
    feature_columns: list[str]
    train_X: np.ndarray
    train_y: np.ndarray
    val_X: np.ndarray
    val_y: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray
    train_dates: pd.Series
    val_dates: pd.Series
    test_dates: pd.Series
    target_scaler: RobustScaler
    feature_scaler: RobustScaler


def _normalize_col(col: str) -> str:
    return (
        col.strip()
        .lower()
        .replace(" ", "")
        .replace("/", "")
        .replace(".", "")
        .replace("%", "pct")
    )


def _pick_column(raw_columns: list[str], candidates: list[str]) -> str | None:
    normalized_lookup = {_normalize_col(c): c for c in raw_columns}
    for candidate in candidates:
        key = _normalize_col(candidate)
        if key in normalized_lookup:
            return normalized_lookup[key]
    return None


def _to_float(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()
    if text == "" or text == "-":
        return np.nan

    text = text.replace(",", "").replace("$", "")
    try:
        return float(text)
    except ValueError:
        return np.nan


def _clean_volume(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip().upper().replace(",", "")
    if text in {"", "-"}:
        return np.nan

    multipliers = {"K": 1e3, "M": 1e6, "B": 1e9}
    suffix = text[-1]
    if suffix in multipliers:
        try:
            return float(text[:-1]) * multipliers[suffix]
        except ValueError:
            return np.nan

    return _to_float(text)


def _clean_pct(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip().replace(",", "")
    if text in {"", "-"}:
        return np.nan

    if text.endswith("%"):
        text = text[:-1]
        try:
            return float(text) / 100.0
        except ValueError:
            return np.nan

    try:
        return float(text)
    except ValueError:
        return np.nan


def load_index_dataframe(path: Path, symbol: str) -> pd.DataFrame:
    """Load and normalize raw index CSV into a canonical schema."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

    col_date = _pick_column(df.columns.tolist(), ["Date"])
    col_close = _pick_column(df.columns.tolist(), ["Close/Last", "CloseLast", "Close", "Price"])
    col_open = _pick_column(df.columns.tolist(), ["Open"])
    col_high = _pick_column(df.columns.tolist(), ["High"])
    col_low = _pick_column(df.columns.tolist(), ["Low"])
    col_volume = _pick_column(df.columns.tolist(), ["Vol.", "Vol", "Volume"])
    col_change = _pick_column(df.columns.tolist(), ["Change %", "ChangePct", "Change"])

    required = {
        "Date": col_date,
        "Close": col_close,
        "Open": col_open,
        "High": col_high,
        "Low": col_low,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"{symbol}: missing required columns: {missing}")

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[col_date], errors="coerce")
    out["close"] = df[col_close].map(_to_float)
    out["open"] = df[col_open].map(_to_float)
    out["high"] = df[col_high].map(_to_float)
    out["low"] = df[col_low].map(_to_float)

    if col_volume is not None:
        out["volume"] = df[col_volume].map(_clean_volume)
    else:
        out["volume"] = np.nan

    if col_change is not None:
        out["change_pct"] = df[col_change].map(_clean_pct)
    else:
        out["change_pct"] = np.nan

    out["symbol"] = symbol
    out = out.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

    # Feature engineering
    out["range_hl"] = out["high"] - out["low"]
    out["return_1d"] = out["close"].pct_change()

    # If source has no change %, fall back to observed daily return.
    out["change_pct"] = out["change_pct"].fillna(out["return_1d"])
    out["change_pct_lag1"] = out["change_pct"].shift(1)

    out = out.dropna(subset=["date", "close", "open", "high", "low"]).reset_index(drop=True)

    return out


def _finalize_sequence_array(values: list[np.ndarray], shape_tail: tuple[int, ...]) -> np.ndarray:
    if not values:
        return np.empty((0, *shape_tail), dtype=np.float32)
    return np.stack(values).astype(np.float32)


def prepare_supervised_data(
    df: pd.DataFrame,
    feature_candidates: list[str],
    target_column: str,
    split_cfg: SplitConfig,
    train_cfg: TrainingConfig,
) -> PreparedData:
    """Prepare scaled train/val/test sequence data with chronological splits."""

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not present in dataframe")

    available = [c for c in feature_candidates if c in df.columns]
    if not available:
        raise ValueError("No requested feature columns are available.")

    # Train period mask is based on target date.
    years = df["date"].dt.year
    train_rows_mask = years <= split_cfg.train_end_year
    if not train_rows_mask.any():
        raise ValueError("No rows in training years; check split configuration.")

    train_features_raw = df.loc[train_rows_mask, available]
    medians = train_features_raw.median(numeric_only=True)

    # Drop features that cannot be imputed from training data.
    valid_feature_cols = [c for c in available if pd.notna(medians.get(c, np.nan))]
    if not valid_feature_cols:
        raise ValueError("No valid feature columns remain after cleaning/imputation.")
    if "close" not in valid_feature_cols:
        raise ValueError("Required feature 'close' is missing after preprocessing.")

    fill_values = medians[valid_feature_cols]
    all_features = df[valid_feature_cols].copy().fillna(fill_values)

    feature_scaler = RobustScaler()
    feature_scaler.fit(all_features.loc[train_rows_mask].values)
    features_scaled = feature_scaler.transform(all_features.values)

    target_scaler = RobustScaler()
    target_scaler.fit(df.loc[train_rows_mask, [target_column]].values)
    target_scaled = target_scaler.transform(df[[target_column]].values).reshape(-1)

    window = train_cfg.window_size
    if len(df) <= window:
        raise ValueError(
            f"Not enough rows ({len(df)}) for window size {window}."
        )

    train_X_list: list[np.ndarray] = []
    train_y_list: list[np.ndarray] = []
    train_dates: list[pd.Timestamp] = []

    val_X_list: list[np.ndarray] = []
    val_y_list: list[np.ndarray] = []
    val_dates: list[pd.Timestamp] = []

    test_X_list: list[np.ndarray] = []
    test_y_list: list[np.ndarray] = []
    test_dates: list[pd.Timestamp] = []

    for idx in range(window, len(df)):
        sample_X = features_scaled[idx - window : idx, :]
        sample_y = np.array([target_scaled[idx]], dtype=np.float32)
        sample_date = df.iloc[idx]["date"]
        sample_year = int(sample_date.year)

        if sample_year <= split_cfg.train_end_year:
            train_X_list.append(sample_X)
            train_y_list.append(sample_y)
            train_dates.append(sample_date)
        elif sample_year == split_cfg.val_year:
            val_X_list.append(sample_X)
            val_y_list.append(sample_y)
            val_dates.append(sample_date)
        elif sample_year == split_cfg.test_year:
            test_X_list.append(sample_X)
            test_y_list.append(sample_y)
            test_dates.append(sample_date)

    n_features = len(valid_feature_cols)

    train_X = _finalize_sequence_array(train_X_list, (window, n_features))
    val_X = _finalize_sequence_array(val_X_list, (window, n_features))
    test_X = _finalize_sequence_array(test_X_list, (window, n_features))

    train_y = _finalize_sequence_array(train_y_list, (1,)).reshape(-1)
    val_y = _finalize_sequence_array(val_y_list, (1,)).reshape(-1)
    test_y = _finalize_sequence_array(test_y_list, (1,)).reshape(-1)

    if train_X.shape[0] == 0 or val_X.shape[0] == 0 or test_X.shape[0] == 0:
        raise ValueError(
            "One or more splits are empty after sequence generation. "
            "Check data coverage and split years."
        )

    return PreparedData(
        feature_columns=valid_feature_cols,
        train_X=train_X,
        train_y=train_y,
        val_X=val_X,
        val_y=val_y,
        test_X=test_X,
        test_y=test_y,
        train_dates=pd.Series(train_dates, name="date"),
        val_dates=pd.Series(val_dates, name="date"),
        test_dates=pd.Series(test_dates, name="date"),
        target_scaler=target_scaler,
        feature_scaler=feature_scaler,
    )
