# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The configs of ``dynamo.common.ais_perf_model`` next to the AISimulate config class."""

import importlib

import pytest

from dynamo.common import ais_perf_model
from tests.utils.ais_perf_model import IDENTITY, normalized

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def test_router_perf_config_equals_forward_pass_config_dict() -> None:
    untyped = {**IDENTITY, "tp": 2, "backend_version": "1.0"}
    entry = {**untyped, "worker_type": "prefill"}
    # The entry with the defaults of the AISimulate config class.
    expected = normalized(entry)

    assert ais_perf_model.router_perf_config(entry) == expected
    assert ais_perf_model.router_perf_config(untyped, "prefill") == expected
    # Only a direct call can give a config without worker_type.
    with pytest.raises(ValueError, match="^ais_perf_model requires worker_type$"):
        ais_perf_model.normalized_config(untyped)


def _decode_pass(requests: int, kv_tokens: int) -> list[dict]:
    """One decode forward pass on one rank, 6 ms plus 30 ns for each KV token."""
    return [
        {
            "version": 1,
            "wall_time": 0.006 + 3e-8 * kv_tokens,
            "scheduled_requests": {
                "num_decode_requests": requests,
                "sum_decode_kv_tokens": kv_tokens,
            },
        }
    ]


_REGRESSION = {
    **IDENTITY,
    "worker_type": "decode",
    "estimation_mode": "fpm_regression",
}


def test_tuned_state_is_restored_by_a_state_snapshot_value(tmp_path) -> None:
    passes = [
        _decode_pass(requests, requests * context)
        for requests, context in ((1, 300), (8, 900), (16, 2000), (32, 5000), (64, 700))
    ]

    saved = ais_perf_model.save_tuned_state_snapshot(
        _REGRESSION, passes, tmp_path / "state"
    )
    restored = ais_perf_model.engine_perf_config(
        {"from": {"state_snapshot": {"path": str(tmp_path / "state")}}}, "decode"
    )

    assert restored["estimator_config"]["state_snapshot"] == (
        saved["estimator_config"]["state_snapshot"]
    )
    # The restored model answers without more observations.
    sdk = importlib.import_module("aisim_core.sdk")
    model = sdk.RustForwardPassPerfModel.best_available(restored)
    estimate = model.estimate_forward_pass_time_ms(
        {
            "scheduled_requests": {
                "num_decode_requests": 10,
                "sum_decode_kv_tokens": 100_000,
            }
        }
    )
    assert estimate == pytest.approx(9.0, rel=1e-3)


def test_tuning_that_leaves_the_model_unready_saves_nothing(tmp_path) -> None:
    # One point is not enough for a fit.
    with pytest.raises(ValueError) as error:
        ais_perf_model.save_tuned_state_snapshot(
            _REGRESSION, [_decode_pass(8, 8000)], tmp_path / "state"
        )

    assert str(error.value).startswith(
        "ais_perf_model: the model is not ready after tuning: pure_decode "
        "(1 observations, "
    )
    assert not (tmp_path / "state").exists()
