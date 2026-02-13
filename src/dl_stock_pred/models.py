from __future__ import annotations

import torch
import torch.nn as nn


class RecurrentRegressor(nn.Module):
    """Simple recurrent next-step regressor for time-series forecasting."""

    def __init__(
        self,
        model_type: str,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        model_type = model_type.lower()
        if model_type == "rnn":
            recurrent_cls = nn.RNN
        elif model_type == "lstm":
            recurrent_cls = nn.LSTM
        elif model_type == "gru":
            recurrent_cls = nn.GRU
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        effective_dropout = dropout if num_layers > 1 else 0.0

        self.recurrent = recurrent_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.recurrent(x)
        out = out[:, -1, :]
        out = self.head(out).squeeze(-1)
        return out
