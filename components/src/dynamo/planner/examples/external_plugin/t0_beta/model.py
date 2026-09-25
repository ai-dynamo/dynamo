# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional local t0 runtime; imported only by the example server."""

from collections.abc import Sequence

import torch
from jaxtyping import Float
from t0 import T0Forecaster

from dynamo.planner.examples.external_plugin.t0_beta.predictor import Traffic


class T0Forecast:
    """Batch three independent traffic series and forecast their next interval."""

    def __init__(self, model: T0Forecaster) -> None:
        self._model = model.eval()

    @torch.inference_mode()
    def __call__(self, history: Sequence[Traffic], quantile: float) -> Traffic:
        context: Float[torch.Tensor, "signal time"] = torch.tensor(  # noqa: F722
            history, dtype=torch.float32
        ).T.contiguous()
        result = self._model.predict(context, horizon=1, quantile_levels=[quantile])
        if result.quantiles.shape != (3, 1, 1):
            raise ValueError("Expected t0 quantiles with shape (3, 1, 1)")
        values: Float[torch.Tensor, "3"] = result.quantiles[:, 0, 0]  # noqa: UP037
        return float(values[0]), float(values[1]), float(values[2])
