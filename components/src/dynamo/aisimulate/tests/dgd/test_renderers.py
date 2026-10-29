# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import functools
import json
import subprocess
import sys
from dataclasses import dataclass
from itertools import pairwise
from types import SimpleNamespace

import pytest
import yaml
from aisimulate.generator.request.schema import ModelFacts
from aisimulate.sweeper.config import Workload

from dynamo.aisimulate.output.dgd.renderers import (
    CandidateMaterializationError,
    DGDGenerationOptions,
)
from dynamo.aisimulate.output.dgd.renderers import base as base_module
from dynamo.aisimulate.output.dgd.renderers import render_dgd
from dynamo.aisimulate.output.dgd.renderers.aic import renderer as aic_renderer
from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig
from dynamo.profiler.utils.config_modifiers import vllm

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]


@pytest.mark.timeout(60)
def test_renderer_contract_import_does_not_load_runtime_modifiers() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import dynamo.aisimulate.output.dgd.renderers.base; "
                "assert 'dynamo.profiler.utils.dgd_materialization' not in sys.modules; "
                "assert 'dynamo.planner' not in sys.modules"
            ),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr


def _candidate(**overrides):
    config = {
        "backend": "vllm",
        "backend_version": "0.20.1",
    }
    config.update(overrides)
    return SimpleNamespace(config=config)


def _options(**overrides) -> DGDGenerationOptions:
    values = {
        "runtime_image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.0",
        "namespace": "demo",
        "num_gpus_per_node": 8,
    }
    values.update(overrides)
    return DGDGenerationOptions(**values)


def _stub_dgd_finalization(monkeypatch) -> None:
    monkeypatch.setattr(base_module, "_materialize_dgd", lambda dgd, **_kwargs: dgd)


@pytest.mark.parametrize(
    "make_workload", [dict, lambda **fields: SimpleNamespace(**fields)]
)
def test_aic_renderer_gives_trace_workloads_placeholder_lengths(
    monkeypatch, make_workload
) -> None:
    captured = []

    def fake_from_sweeper_candidate(candidate, *, workload, **_kwargs):
        captured.append(workload)
        return "request"

    monkeypatch.setattr(
        aic_renderer,
        "_load_generator_api",
        lambda: (
            fake_from_sweeper_candidate,
            lambda _request: {"k8s_deploy.yaml": "kind: DynamoGraphDeployment\n"},
        ),
    )
    monkeypatch.setattr(
        aic_renderer, "patch_dgd_manifest", lambda rendered, *_args, **_kwargs: rendered
    )
    fields = {
        "trace_path": "/traces/window.jsonl",
        "trace_paths": ["/traces/window.jsonl"],
        "trace_format": "dynamo",
        "isl": None,
        "osl": None,
    }
    workload = make_workload(**fields)

    render_dgd(_candidate(), workload, _options(), dgd_name="sweeper-dgd")

    assert captured == [{**fields, "isl": 1, "osl": 1}]
    assert _workload_lengths(workload) == (None, None)


def _workload_lengths(workload):
    if isinstance(workload, dict):
        return workload["isl"], workload["osl"]
    return workload.isl, workload.osl


@dataclass(frozen=True)
class _SweeperCandidate:
    config: dict
    used_gpus: int


def _disagg_candidate(backend: str, backend_version: str) -> _SweeperCandidate:
    roles = {
        f"{role}_{field}": value
        for role in ("prefill", "decode")
        for field, value in {
            "tp": 1,
            "pp": 1,
            "attention_dp": 1,
            "moe_tp": 1,
            "moe_ep": 1,
            "strategy": "tp",
            "replicas": 1,
            "max_num_batched_tokens": 4096,
            "max_num_seqs": 64,
            "block_size": 32,
            "gpu_memory_utilization": 0.9,
            "enable_prefix_caching": role == "prefill",
        }.items()
    }
    return _SweeperCandidate(
        config={
            "deployment_mode": "disagg",
            "backend": backend,
            "backend_version": backend_version,
            "model_name": "Qwen/Qwen3-32B",
            "hardware_sku": "h200_sxm",
            **roles,
        },
        used_gpus=2,
    )


def _transfer_buffer_args(rendered: str) -> dict[str, dict[str, str]]:
    """Map each disaggregated role to its ``cache_transceiver_config`` flags."""
    flags = {}
    for component in yaml.safe_load(rendered)["spec"]["components"]:
        if component.get("type") not in {"prefill", "decode"}:
            continue
        args = next(
            container
            for container in component["podTemplate"]["spec"]["containers"]
            if container.get("name") == "main"
        )["args"]
        flags[component["type"]] = {
            name: value
            for name, value in pairwise(args)
            if name.startswith("--trtllm.cache_transceiver_config.")
        }
    return flags


@pytest.mark.parametrize(
    ("backend", "backend_version", "image"),
    [
        ("trtllm", "1.3.0rc23", "tensorrtllm-runtime"),
        ("vllm", "0.20.1", "vllm-runtime"),
        ("sglang", "0.5.9", "sglang-runtime"),
    ],
)
def test_installed_aic_renders_trace_workloads_like_synthetic_ones(
    monkeypatch, tmp_path, backend, backend_version, image
) -> None:
    # Keep the renders offline: model facts come from the request below.
    monkeypatch.setattr(vllm, "get_model_context_length", lambda _model: None)
    monkeypatch.setattr(vllm, "get_mamba_cache_align_block_size", lambda _model: None)
    from_candidate, generate = aic_renderer._load_generator_api()
    monkeypatch.setattr(
        aic_renderer,
        "_load_generator_api",
        lambda: (
            functools.partial(
                from_candidate,
                model_facts=ModelFacts(is_moe=False, architecture="Qwen3ForCausalLM"),
            ),
            generate,
        ),
    )
    _stub_dgd_finalization(monkeypatch)
    trace = tmp_path / "window.jsonl"
    trace.write_text("{}\n")
    candidate = _disagg_candidate(backend, backend_version)
    options = _options(runtime_image=f"nvcr.io/nvidia/ai-dynamo/{image}:1.5.0")

    def render(workload) -> str:
        return render_dgd(candidate, workload, options, dgd_name="sweeper-dgd")

    traced = render(
        Workload(trace_path=str(trace), trace_paths=[str(trace)], trace_format="dynamo")
    )
    synthetic = render({"isl": 4000, "osl": 1000})

    # Apart from TensorRT-LLM's transfer buffer size, nothing may depend on
    # the placeholder lengths a trace workload gives AIC.
    assert yaml.safe_load(traced) == yaml.safe_load(
        aic_renderer._drop_transfer_buffer_size(synthetic)
    )
    if backend == "trtllm":
        # The trace keeps the transceiver backend; TensorRT-LLM sizes the buffer.
        assert _transfer_buffer_args(traced) == {
            role: {"--trtllm.cache_transceiver_config.backend": "DEFAULT"}
            for role in ("prefill", "decode")
        }
        assert all(
            "--trtllm.cache_transceiver_config.max_tokens_in_buffer" in flags
            for flags in _transfer_buffer_args(synthetic).values()
        )


def test_dropping_transfer_buffer_size_keeps_other_arguments() -> None:
    buffer_arg = "--trtllm.cache_transceiver_config.max_tokens_in_buffer"

    def component(component_type, args):
        return {
            "type": component_type,
            "podTemplate": {"spec": {"containers": [{"name": "main", "args": args}]}},
        }

    rendered = yaml.safe_dump(
        {
            "kind": "DynamoGraphDeployment",
            "spec": {
                "components": [
                    {"name": "Frontend", "type": "frontend"},
                    component(
                        "prefill",
                        [
                            "--trtllm.cache_transceiver_config.backend",
                            "DEFAULT",
                            buffer_arg,
                            "6528",
                            "--max-num-tokens",
                            "4096",
                        ],
                    ),
                    component(
                        "decode", [f"{buffer_arg}=6528", "--max-batch-size", "64"]
                    ),
                ]
            },
        },
        sort_keys=False,
    )
    unrelated = yaml.safe_dump(
        {"spec": {"components": [component("worker", ["--block-size", "64"])]}}
    )

    dgd = yaml.safe_load(aic_renderer._drop_transfer_buffer_size(rendered))

    args = {
        component["type"]: component["podTemplate"]["spec"]["containers"][0]["args"]
        for component in dgd["spec"]["components"]
        if "podTemplate" in component
    }
    assert args == {
        "prefill": [
            "--trtllm.cache_transceiver_config.backend",
            "DEFAULT",
            "--max-num-tokens",
            "4096",
        ],
        "decode": ["--max-batch-size", "64"],
    }
    assert aic_renderer._drop_transfer_buffer_size(unrelated) == unrelated


def test_direct_renderer_preserves_evaluation_context(monkeypatch) -> None:
    monkeypatch.setattr(vllm, "get_model_context_length", lambda _model: None)
    monkeypatch.setattr(vllm, "get_mamba_cache_align_block_size", lambda _model: None)
    candidate = _disagg_candidate("vllm", "0.20.1")
    candidate.config["concurrency"] = 64
    rendered = render_dgd(
        candidate,
        SimpleNamespace(isl=4000, osl=1000),
        _options(),
        dgd_name="sweeper-dgd",
        renderer="direct",
    )

    dgd = yaml.safe_load(rendered)
    metadata = dgd["metadata"]
    evaluation_context = metadata.pop("annotations")[
        "nvidia.com/sweeper-evaluation-context"
    ]
    assert json.loads(evaluation_context) == {
        "concurrency": 64,
        "hardware_sku": "h200_sxm",
    }
    assert metadata == {
        "name": "sweeper-dgd",
        "namespace": "demo",
    }
    assert "runtimeVersionOverride" not in dgd["spec"]["components"][0]


@pytest.mark.parametrize(
    ("candidate", "message"),
    [
        (_candidate(backend="unknown"), "candidate backend must be one of"),
        (_candidate(backend_version=""), "candidate backend_version must be"),
    ],
)
def test_materialize_requires_renderer_candidate_fields(candidate, message) -> None:
    with pytest.raises(CandidateMaterializationError, match=message):
        render_dgd(
            candidate,
            SimpleNamespace(isl=4000, osl=1000),
            _options(),
            dgd_name="sweeper-dgd",
        )


def test_patch_manifest_preserves_non_dgd_documents(monkeypatch) -> None:
    _stub_dgd_finalization(monkeypatch)
    rendered = """
apiVersion: v1
kind: ConfigMap
metadata:
  name: engine-config
data:
  engine.yaml: preserved
---
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: generated
spec:
  components: []
"""

    patched = base_module.patch_dgd_manifest(
        rendered,
        _candidate(),
        _options(),
        dgd_name="sweeper-dgd",
    )

    documents = list(yaml.safe_load_all(patched))
    assert [document["kind"] for document in documents] == [
        "ConfigMap",
        "DynamoGraphDeployment",
    ]
    assert documents[0] == {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": "engine-config"},
        "data": {"engine.yaml": "preserved"},
    }
    assert documents[1]["metadata"]["name"] == "sweeper-dgd"


def test_patch_manifest_applies_shared_legacy_finalization(monkeypatch) -> None:
    calls = []

    def fake_materialize(dgd, *, purpose, runtime_backend, model_name_or_path):
        calls.append((purpose, runtime_backend, model_name_or_path))
        return dgd

    monkeypatch.setattr(base_module, "_materialize_dgd", fake_materialize)

    patched = base_module.patch_dgd_manifest(
        """
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: generated
spec:
  components:
  - name: TRTLLMWorker
    type: worker
    podTemplate:
      spec:
        containers:
        - name: main
          args: []
""",
        _candidate(backend="trtllm", model_name="Qwen/Qwen3-32B"),
        _options(),
        dgd_name="sweeper-dgd",
    )

    dgd = yaml.safe_load(patched)
    args = dgd["spec"]["components"][0]["podTemplate"]["spec"]["containers"][0]["args"]
    index = args.index("--trtllm.enable_chunked_prefill")
    assert args[index + 1] == "true"
    assert len(calls) == 1
    assert calls[0][0].value == "final output"
    assert calls[0][1:] == ("trtllm", "Qwen/Qwen3-32B")


def test_patch_manifest_router_configuration_round_trips_through_frontend(
    monkeypatch,
) -> None:
    _stub_dgd_finalization(monkeypatch)
    candidate = _candidate(
        adapters={
            "dynamo.router": {
                "policy": "kv_router",
                "prefill_load_model": {"type": "none"},
                "overlap_score_credit": 0.5,
                "prefill_load_scale": 0.25,
                "temperature": 0.2,
            }
        }
    )

    patched = base_module.patch_dgd_manifest(
        """
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: generated
spec:
  components:
  - name: Frontend
    type: frontend
    podTemplate:
      spec:
        containers:
        - name: main
          args: []
""",
        candidate,
        _options(),
        dgd_name="sweeper-dgd",
    )

    frontend = yaml.safe_load(patched)["spec"]["components"][0]
    env = {
        item["name"]: item["value"]
        for item in frontend["podTemplate"]["spec"]["containers"][0]["env"]
    }
    assert env == {
        "DYN_ROUTER_MODE": "kv",
        "DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT": "0.5",
        "DYN_ROUTER_PREFILL_LOAD_SCALE": "0.25",
        "DYN_ROUTER_TEMPERATURE": "0.2",
        "DYN_ROUTER_PREFILL_LOAD_MODEL": "none",
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    parser = argparse.ArgumentParser()
    with pytest.warns(FutureWarning, match="router-policy-config"):
        FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(parser.parse_args([]))
    config.validate()

    assert config.router_mode == "kv"
    assert config.overlap_score_credit == 0.5
    assert config.prefill_load_scale == 0.25
    assert config.router_temperature == 0.2
    assert config.router_prefill_load_model == "none"


@pytest.mark.parametrize(
    ("router_config", "expected_temperature"),
    [
        ({"mode": "kv_router", "router_temperature": 0.7}, "0.7"),
        (
            {
                "mode": "kv_router",
                "router_temperature": 0.7,
                "temperature": 0.2,
            },
            "0.2",
        ),
    ],
)
def test_patch_manifest_preserves_native_router_temperature(
    monkeypatch, router_config, expected_temperature
) -> None:
    _stub_dgd_finalization(monkeypatch)

    patched = base_module.patch_dgd_manifest(
        """
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: generated
spec:
  components:
  - name: Frontend
    type: frontend
    podTemplate:
      spec:
        containers:
        - name: main
          args: []
""",
        _candidate(adapters={"dynamo.router": router_config}),
        _options(),
        dgd_name="sweeper-dgd",
    )

    frontend = yaml.safe_load(patched)["spec"]["components"][0]
    env = {
        item["name"]: item["value"]
        for item in frontend["podTemplate"]["spec"]["containers"][0]["env"]
    }
    assert env["DYN_ROUTER_TEMPERATURE"] == expected_temperature


def test_patch_manifest_rejects_router_runtime_model_without_configuration(
    monkeypatch,
) -> None:
    _stub_dgd_finalization(monkeypatch)
    candidate = _candidate(
        adapters={
            "dynamo.router": {
                "policy": "kv_router",
                "prefill_load_model": {"type": "ais"},
            }
        }
    )

    with pytest.raises(CandidateMaterializationError, match="runtime model"):
        base_module.patch_dgd_manifest(
            """
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: generated
spec:
  components:
  - name: Frontend
    type: frontend
    podTemplate:
      spec:
        containers:
        - name: main
          args: []
""",
            candidate,
            _options(),
            dgd_name="sweeper-dgd",
        )


def test_patch_manifest_preserves_candidate_context_length(monkeypatch) -> None:
    _stub_dgd_finalization(monkeypatch)

    patched = base_module.patch_dgd_manifest(
        """
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: generated
spec:
  components:
  - name: VllmWorker
    type: worker
    podTemplate:
      spec:
        containers:
        - name: main
          args: [--max-model-len, "40960"]
""",
        _candidate(context_length=8192),
        _options(),
        dgd_name="sweeper-dgd",
    )

    worker = yaml.safe_load(patched)["spec"]["components"][0]
    args = worker["podTemplate"]["spec"]["containers"][0]["args"]
    assert args.count("--max-model-len") == 1
    assert args[args.index("--max-model-len") + 1] == "8192"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("runtime_image", "", "runtime_image must not be empty"),
        ("num_gpus_per_node", 0, "num_gpus_per_node must be positive"),
        (
            "runtime_version_override",
            "1.5",
            "runtime_version_override must be a canonical",
        ),
    ],
)
def test_generation_options_validate_inputs(
    field: str, value: object, message: str
) -> None:
    values = {
        "runtime_image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.0",
        "num_gpus_per_node": 8,
    }
    values[field] = value
    with pytest.raises(ValueError, match=message):
        DGDGenerationOptions(**values)


def test_generation_options_require_semver_image_without_override() -> None:
    with pytest.raises(ValueError, match="runtime_image must have a canonical"):
        DGDGenerationOptions(
            runtime_image="nvcr.io/nvidia/ai-dynamo/vllm-runtime:latest",
            num_gpus_per_node=8,
        )
