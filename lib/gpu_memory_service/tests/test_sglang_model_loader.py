# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sglang")

from gpu_memory_service.integrations.sglang.model_loader import (  # noqa: E402
    GMSModelLoader,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
]


def test_import_model_rejects_custom_post_load_without_executing_it():
    called = False

    class Model(torch.nn.Module):
        def post_load_weights(self):
            nonlocal called
            called = True

    with pytest.raises(RuntimeError, match="custom post_load_weights"):
        GMSModelLoader._validate_import_model(Model())

    assert called is False


def test_import_model_accepts_model_without_custom_post_load():
    GMSModelLoader._validate_import_model(torch.nn.Linear(2, 2, device="meta"))


def test_import_runtime_state_restores_only_missing_flashinfer_state():
    prepared = []
    quant_method = SimpleNamespace(
        _prepare_flashinfer_trtllm_activation_params=lambda module: prepared.append(
            module
        )
    )
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2, device="meta"),
        torch.nn.Linear(2, 2, device="meta"),
    )
    model[0].quant_method = quant_method
    model[1].quant_method = quant_method
    model[1]._flashinfer_trtllm_gemm1_alpha = object()

    GMSModelLoader._restore_import_runtime_state(model)

    assert prepared == [model[0]]
