from pathlib import Path

from dl_stock_pred.config import SplitConfig, TrainingConfig
from dl_stock_pred.data import load_index_dataframe, prepare_supervised_data


DATA_FILES = {
    "sp500": Path("data/raw/sp500_5y.csv"),
    "nasdaq100": Path("data/raw/nasdaq100_5y.csv"),
    "dowjones": Path("data/raw/dowjones_5y.csv"),
}


def test_load_index_dataframe_schema() -> None:
    required = {"date", "open", "high", "low", "close", "range_hl", "return_1d", "change_pct_lag1"}

    for symbol, path in DATA_FILES.items():
        df = load_index_dataframe(path=path, symbol=symbol)
        assert len(df) > 1000
        assert required.issubset(set(df.columns))
        assert df["date"].is_monotonic_increasing
        assert df["close"].notna().all()


def test_prepare_supervised_data_has_non_empty_splits() -> None:
    df = load_index_dataframe(path=DATA_FILES["dowjones"], symbol="dowjones")

    prepared = prepare_supervised_data(
        df=df,
        feature_candidates=["open", "high", "low", "close", "range_hl", "return_1d", "change_pct_lag1", "volume"],
        target_column="close",
        split_cfg=SplitConfig(train_end_year=2023, val_year=2024, test_year=2025),
        train_cfg=TrainingConfig(window_size=20),
    )

    assert prepared.train_X.shape[0] > 0
    assert prepared.val_X.shape[0] > 0
    assert prepared.test_X.shape[0] > 0

    assert (prepared.train_dates.dt.year <= 2023).all()
    assert (prepared.val_dates.dt.year == 2024).all()
    assert (prepared.test_dates.dt.year == 2025).all()

    assert "close" in prepared.feature_columns
