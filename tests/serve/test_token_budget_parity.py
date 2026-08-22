# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Differential request-budget tests against each backend's native server."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest
import requests
import yaml

from tests.serve.common import WORKSPACE_DIR, managed_serve_deployment
from tests.utils.constants import CROSS_BACKEND_SMOKE_MODEL, DynamoPortRange
from tests.utils.engine_process import EngineConfig
from tests.utils.managed_process import ManagedProcess
from tests.utils.payloads import check_models_api
from tests.utils.port_utils import allocate_port, deallocate_port

MODEL = CROSS_BACKEND_SMOKE_MODEL
CONTEXT_LIMIT = 128


@dataclass(frozen=True)
class BackendParitySpec:
    framework: str
    mode: str
    reject_prompt_overflow: bool
    reject_total_overflow: bool
    prompt_reject_after_headers: bool = False

    @property
    def name(self) -> str:
        return f"{self.framework}-{self.mode}"


@dataclass(frozen=True)
class RequestCase:
    name: str
    prompt: str
    max_tokens: int
    stream: bool


@dataclass(frozen=True)
class RequestOutcome:
    status_code: int
    body: dict | None
    stream_finished: bool
    usage: TokenUsage | None
    stream_error: dict | None


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


PARITY_SPECS = (
    pytest.param(
        BackendParitySpec(
            "vllm",
            "default",
            reject_prompt_overflow=True,
            reject_total_overflow=True,
        ),
        id="vllm-default",
        marks=[
            pytest.mark.vllm,
            pytest.mark.profiled_vram_gib(8.0),
            pytest.mark.requested_vllm_kv_cache_bytes(559_693_824),
        ],
    ),
    pytest.param(
        BackendParitySpec(
            "sglang",
            "default",
            reject_prompt_overflow=True,
            reject_total_overflow=True,
        ),
        id="sglang-default",
        marks=[
            pytest.mark.sglang,
            pytest.mark.profiled_vram_gib(8.0),
            pytest.mark.requested_sglang_kv_tokens(2048),
        ],
    ),
    pytest.param(
        BackendParitySpec(
            "sglang",
            "auto-truncate",
            reject_prompt_overflow=False,
            reject_total_overflow=False,
        ),
        id="sglang-auto-truncate",
        marks=[
            pytest.mark.sglang,
            pytest.mark.profiled_vram_gib(8.0),
            pytest.mark.requested_sglang_kv_tokens(2048),
        ],
    ),
    pytest.param(
        BackendParitySpec(
            "trtllm",
            "default",
            reject_prompt_overflow=True,
            reject_total_overflow=False,
            prompt_reject_after_headers=True,
        ),
        id="trtllm-default",
        marks=[
            pytest.mark.trtllm,
            pytest.mark.profiled_vram_gib(8.5),
            pytest.mark.requested_trtllm_kv_tokens(512),
        ],
    ),
)

REQUEST_CASES = (
    RequestCase("within-budget", "Hello", 1, False),
    # The prompt consumes at least one token, so requesting CONTEXT_LIMIT output
    # tokens necessarily exceeds the combined budget.
    RequestCase("output-overflow", "Hello", CONTEXT_LIMIT, True),
    # For Qwen3, each repeated " hello" contributes a token. The 2x margin
    # keeps this unambiguously above the configured prompt budget.
    RequestCase("prompt-overflow", " hello" * (CONTEXT_LIMIT * 2), 1, True),
)


@pytest.fixture
def native_server_port():
    port = allocate_port(DynamoPortRange.FRONTEND.value)
    try:
        yield port
    finally:
        deallocate_port(port)


def _write_trtllm_options(tmp_path: Path) -> Path:
    path = tmp_path / "trtllm-token-budget.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "tensor_parallel_size": 1,
                "max_batch_size": 2,
                "max_num_tokens": CONTEXT_LIMIT,
                "max_seq_len": CONTEXT_LIMIT,
                "trust_remote_code": True,
                "backend": "pytorch",
                "enable_chunked_prefill": True,
                "kv_cache_config": {"free_gpu_memory_fraction": 0.2},
                "cuda_graph_config": None,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _native_command(spec: BackendParitySpec, port: int, tmp_path: Path) -> list[str]:
    if spec.framework == "vllm":
        return [
            "vllm",
            "serve",
            MODEL,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--max-model-len",
            str(CONTEXT_LIMIT),
            "--max-num-seqs",
            "2",
            "--gpu-memory-utilization",
            "0.2",
            "--enforce-eager",
        ]

    if spec.framework == "sglang":
        command = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            MODEL,
            "--served-model-name",
            MODEL,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--context-length",
            str(CONTEXT_LIMIT),
            "--mem-fraction-static",
            "0.2",
            "--max-running-requests",
            "2",
            "--skip-server-warmup",
            "--disable-cuda-graph",
            "--disable-piecewise-cuda-graph",
            "--trust-remote-code",
        ]
        if spec.mode == "auto-truncate":
            command.append("--allow-auto-truncate")
        return command

    if spec.framework == "trtllm":
        return [
            "trtllm-serve",
            MODEL,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--config",
            str(_write_trtllm_options(tmp_path)),
        ]

    raise AssertionError(f"Unsupported framework: {spec.framework}")


def _dynamo_config(spec: BackendParitySpec, tmp_path: Path) -> EngineConfig:
    backend_dir = os.path.join(WORKSPACE_DIR, "examples", "backends", spec.framework)
    script_args: list[str] = []
    env: dict[str, str] = {}
    delayed_start = 0

    if spec.framework == "vllm":
        env = {
            "MAX_MODEL_LEN": str(CONTEXT_LIMIT),
            "MAX_CONCURRENT_SEQS": "2",
        }
    elif spec.framework == "sglang":
        script_args = [
            "--context-length",
            str(CONTEXT_LIMIT),
            "--skip-server-warmup",
        ]
        if spec.mode == "auto-truncate":
            script_args.append("--allow-auto-truncate")
    elif spec.framework == "trtllm":
        delayed_start = 5
        env = {
            "MODEL_PATH": MODEL,
            "SERVED_MODEL_NAME": MODEL,
            "AGG_ENGINE_ARGS": str(_write_trtllm_options(tmp_path)),
        }
    else:
        raise AssertionError(f"Unsupported framework: {spec.framework}")

    return EngineConfig(
        name=f"dynamo-{spec.name}",
        directory=backend_dir,
        script_name="agg.sh",
        script_args=script_args,
        marks=[],
        request_payloads=[],
        model=MODEL,
        timeout=600,
        delayed_start=delayed_start,
        env=env,
    )


def _native_process(
    spec: BackendParitySpec, port: int, tmp_path: Path, request
) -> ManagedProcess:
    return ManagedProcess(
        command=_native_command(spec, port, tmp_path),
        env=os.environ.copy(),
        health_check_urls=[(f"http://127.0.0.1:{port}/v1/models", check_models_api)],
        timeout=600,
        working_dir=WORKSPACE_DIR,
        display_output=True,
        terminate_all_matching_process_names=False,
        log_dir=f"{request.node.name}-native",
        display_name=f"{spec.name}-native",
    )


def _send_case(port: int, case: RequestCase) -> RequestOutcome:
    payload = {
        "model": MODEL,
        "prompt": case.prompt,
        "max_tokens": case.max_tokens,
        "temperature": 0.0,
        "stream": case.stream,
    }
    if case.stream:
        payload["stream_options"] = {"include_usage": True}
    if case.name == "output-overflow":
        # Force generation to the effective length limit so a successful
        # response proves the backend clamped max_tokens rather than stopping
        # early at EOS.
        payload["ignore_eos"] = True

    response = requests.post(
        f"http://127.0.0.1:{port}/v1/completions",
        json=payload,
        timeout=180,
        stream=case.stream,
    )

    try:
        if not case.stream or response.status_code != 200:
            try:
                body = response.json()
            except requests.exceptions.JSONDecodeError:
                body = None
            return RequestOutcome(
                response.status_code,
                body,
                False,
                _parse_usage(body),
                None,
            )

        stream_finished = False
        saw_data = False
        usage = None
        stream_error = None
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            saw_data = True
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                stream_finished = True
                continue
            event = json.loads(data)
            if event.get("error"):
                stream_error = event["error"]
                continue
            usage = _parse_usage(event) or usage

        assert saw_data, "Successful streaming response did not contain SSE data"
        return RequestOutcome(
            response.status_code,
            None,
            stream_finished,
            usage,
            stream_error,
        )
    finally:
        response.close()


def _parse_usage(body: dict | None) -> TokenUsage | None:
    if not body or not body.get("usage"):
        return None
    usage = body["usage"]
    return TokenUsage(
        prompt_tokens=int(usage["prompt_tokens"]),
        completion_tokens=int(usage["completion_tokens"]),
        total_tokens=int(usage["total_tokens"]),
    )


def _is_rejected(spec: BackendParitySpec, case: RequestCase) -> bool:
    if case.name == "output-overflow":
        return spec.reject_total_overflow
    if case.name == "prompt-overflow":
        return spec.reject_prompt_overflow
    return False


def _expected_status(
    spec: BackendParitySpec,
    case: RequestCase,
    *,
    dynamo: bool,
) -> int:
    if case.name == "within-budget":
        return 200
    if (
        not dynamo
        and case.name == "prompt-overflow"
        and spec.prompt_reject_after_headers
    ):
        return 200
    return 400 if _is_rejected(spec, case) else 200


def _assert_outcome_semantics(
    spec: BackendParitySpec,
    case: RequestCase,
    outcome: RequestOutcome,
    *,
    dynamo: bool,
) -> None:
    if _is_rejected(spec, case):
        if outcome.status_code == 400:
            assert outcome.stream_error is None
            assert outcome.body is not None, "400 response was not JSON"
            error = (
                outcome.body.get("error")
                or outcome.body.get("message")
                or outcome.body.get("detail")
            )
        else:
            # TRT-LLM currently discovers prompt overflow after committing
            # streaming headers. Dynamo deliberately moves that same rejection
            # to request time using the engine-published Reject policy.
            assert not dynamo
            assert spec.prompt_reject_after_headers
            assert outcome.stream_finished
            error = outcome.stream_error
        assert error, f"Rejected response lacked an error message: {outcome}"
        return

    assert outcome.status_code == 200
    assert outcome.stream_error is None, (
        f"Accepted response contained an inline stream error: "
        f"{outcome.stream_error}"
    )

    if case.stream:
        assert (
            outcome.stream_finished
        ), "Successful stream did not terminate with [DONE]"
    else:
        assert outcome.body is not None
        assert outcome.body.get(
            "choices"
        ), f"Completion response lacked choices: {outcome.body}"

    assert outcome.usage is not None, "Successful response did not report token usage"
    usage = outcome.usage
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
    assert usage.total_tokens <= CONTEXT_LIMIT

    if case.name == "output-overflow":
        assert 0 < usage.completion_tokens < case.max_tokens, (
            "Output-overflow request was accepted but max_tokens was not "
            f"observably clamped: {usage}"
        )
    elif case.name == "prompt-overflow":
        assert usage.prompt_tokens < CONTEXT_LIMIT, (
            "Prompt-overflow request was accepted but the prompt was not "
            f"observably truncated: {usage}"
        )


# This guards the externally visible request contract. It is deliberately
# pre-merge even though it starts two servers: backend upgrades must not merge
# with stale Dynamo token-budget metadata.
@pytest.mark.pre_merge
@pytest.mark.gpu_1
@pytest.mark.e2e
@pytest.mark.core
@pytest.mark.token_budget_parity
@pytest.mark.model(MODEL)
@pytest.mark.timeout(900)
@pytest.mark.parametrize("spec", PARITY_SPECS)
def test_native_and_dynamo_token_budget_parity(
    spec,
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    native_server_port,
    predownload_models,
    tmp_path,
):
    """Pin native overflow semantics and require Dynamo to match them."""
    dynamo_config = _dynamo_config(spec, tmp_path)

    with _native_process(spec, native_server_port, tmp_path, request):
        with managed_serve_deployment(
            dynamo_config, request, ports=dynamo_dynamic_ports
        ):
            native = {
                case.name: _send_case(native_server_port, case)
                for case in REQUEST_CASES
            }
            dynamo = {
                case.name: _send_case(dynamo_dynamic_ports.frontend_port, case)
                for case in REQUEST_CASES
            }

    for case in REQUEST_CASES:
        expected_native = _expected_status(spec, case, dynamo=False)
        expected_dynamo = _expected_status(spec, case, dynamo=True)
        native_outcome = native[case.name]
        dynamo_outcome = dynamo[case.name]

        assert native_outcome.status_code == expected_native, (
            f"{spec.name} native behavior changed for {case.name}: "
            f"expected HTTP {expected_native}, got {native_outcome.status_code}"
        )
        assert dynamo_outcome.status_code == expected_dynamo, (
            f"Dynamo behavior changed for {spec.name} {case.name}: "
            f"expected HTTP {expected_dynamo}, got {dynamo_outcome.status_code}"
        )
        _assert_outcome_semantics(spec, case, native_outcome, dynamo=False)
        _assert_outcome_semantics(spec, case, dynamo_outcome, dynamo=True)
        if not _is_rejected(spec, case):
            assert dynamo_outcome.usage == native_outcome.usage, (
                f"Dynamo token accounting diverged from native {spec.name} for "
                f"{case.name}: native={native_outcome.usage}, "
                f"Dynamo={dynamo_outcome.usage}"
            )
