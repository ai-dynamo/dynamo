# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional model adapter test with real, tiny random t0 weights; no downloads."""

import pytest

try:
    import torch
    from t0 import T0Forecaster

    from dynamo.planner.examples.external_plugin.t0_beta.model import T0Forecast
except ImportError as error:
    pytest.skip(
        f"Optional t0 example dependencies unavailable: {error}",
        allow_module_level=True,
    )

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def test_local_model_adapter_preserves_signal_order_and_quantile() -> None:
    torch.manual_seed(42)
    model = T0Forecaster(
        embed_dim=32,
        num_layers=1,
        num_heads=2,
        mlp_hidden_dim=64,
        patch_size=4,
        group_every_n=1,
        dropout=0.0,
        quantile_levels=(0.1, 0.5, 0.9),
        scaler_eps_mode="std_clamp",
    ).eval()
    history = [(10.0, 100.0, 20.0), (20.0, 120.0, 30.0)]
    with torch.inference_mode():
        expected = model.predict(
            torch.tensor([[10.0, 20.0], [100.0, 120.0], [20.0, 30.0]]),
            horizon=1,
            quantile_levels=[0.9],
        ).quantiles[:, 0, 0]
    actual = T0Forecast(model)(history, 0.9)
    assert actual == pytest.approx(expected.tolist())
    assert all(parameter.grad is None for parameter in model.parameters())
