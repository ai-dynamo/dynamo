# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from unittest.mock import Mock, patch

import pytest

from dynamo.experimental.workflow.vllm import EncoderStage
from dynamo.experimental.workflow.vllm import encoder_worker as worker_module
from dynamo.experimental.workflow import StageContext

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


class _Stage:
    contract = EncoderStage.contract

    def __init__(self) -> None:
        self.close_calls = 0

    async def run(
        self, inputs: Mapping[str, Any], context: StageContext
    ) -> Mapping[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        self.close_calls += 1


class _Carrier:
    def __init__(self) -> None:
        self.close_calls = 0

    def can_export(self, value: Any) -> bool:
        return False

    async def export_tensor(self, tensor: Any, transfer_id: str) -> Any:
        raise NotImplementedError

    async def export_tensor_fanout(
        self, tensor: Any, transfer_ids: tuple[str, ...]
    ) -> Any:
        raise NotImplementedError

    async def import_tensor(self, reference: Any) -> Any:
        raise NotImplementedError

    async def close(self) -> None:
        self.close_calls += 1


class _Endpoint:
    def __init__(self, error: Exception | None = None) -> None:
        self.handler: Any = None
        self._error = error

    async def serve_endpoint(self, handler: Any) -> None:
        self.handler = handler
        if self._error is not None:
            raise self._error


class _Runtime:
    def __init__(self, error: Exception | None = None) -> None:
        self.endpoint_ids: list[str] = []
        self.endpoint_value = _Endpoint(error)

    def endpoint(self, endpoint_id: str) -> _Endpoint:
        self.endpoint_ids.append(endpoint_id)
        return self.endpoint_value


@pytest.mark.parametrize("serve_error", [None, RuntimeError("serve failed")])
async def test_encoder_worker_serves_and_cleans_up(serve_error):
    runtime = _Runtime(serve_error)
    stage = _Stage()
    carrier = _Carrier()
    backend_class = Mock(return_value=object())

    with (
        patch.object(
            worker_module,
            "resolve_vision_encoder_backend_class",
            return_value=backend_class,
        ) as resolve,
        patch.object(
            worker_module.EncoderStage,
            "from_backend",
            return_value=stage,
        ) as from_backend,
        patch.object(worker_module, "NixlTensorCarrier", return_value=carrier),
    ):
        if serve_error is None:
            await worker_module.encoder_worker.__wrapped__(
                runtime,
                "workflows.encoder.generate",
                "org/model",
                "author.encoder.Backend",
                "vision",
            )
        else:
            with pytest.raises(RuntimeError, match="serve failed"):
                await worker_module.encoder_worker.__wrapped__(
                    runtime,
                    "workflows.encoder.generate",
                    "org/model",
                    "author.encoder.Backend",
                    "vision",
                )

    resolve.assert_called_once_with("author.encoder.Backend")
    from_backend.assert_called_once_with(
        backend_class.return_value,
        model="org/model",
        name="workflow-vision",
    )
    assert runtime.endpoint_ids == ["workflows.encoder.generate"]
    assert runtime.endpoint_value.handler is not None
    assert carrier.close_calls == 1
    assert stage.close_calls == 1
