# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import yaml

from dynamo.profiler.sweeper.renderers import (
    CandidateMaterializationError,
    DGDGenerationOptions,
)
from dynamo.profiler.sweeper.renderers import base as base_module
from dynamo.profiler.sweeper.renderers import render_dgd
from dynamo.profiler.sweeper.renderers.aic import renderer as aic_renderer
from dynamo.profiler.sweeper.renderers.direct import renderer as direct_renderer

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]


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


def test_materialize_uses_official_candidate_bridge(monkeypatch) -> None:
    captured = {}

    def fake_from_sweeper_candidate(
        candidate, *, workload, deployment_target, generator_overrides
    ):
        captured.update(
            candidate=candidate,
            workload=workload,
            deployment_target=deployment_target,
            generator_overrides=generator_overrides,
        )
        return "request"

    def fake_generate_from_request(request):
        captured["request"] = request
        return {
            "k8s_deploy.yaml": """
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: generated-name
spec:
  components:
  - name: Frontend
  - name: VllmDecodeWorker
"""
        }

    monkeypatch.setattr(
        aic_renderer,
        "_load_generator_api",
        lambda: (fake_from_sweeper_candidate, fake_generate_from_request),
    )
    candidate = _candidate()
    workload = SimpleNamespace(isl=4000, osl=1000)

    rendered = render_dgd(
        candidate,
        workload,
        _options(),
        dgd_name="sweeper-dgd",
    )

    assert captured == {
        "candidate": candidate,
        "workload": workload,
        "deployment_target": "dynamo-python",
        "generator_overrides": {
            "generator_dynamo_version": "1.5.0",
            "K8sConfig": {
                "k8s_image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.0",
                "k8s_namespace": "demo",
                "name_prefix": "sweeper-dgd",
            },
            "NodeConfig": {"num_gpus_per_node": 8},
        },
        "request": "request",
    }
    dgd = yaml.safe_load(rendered)
    assert dgd["metadata"] == {
        "name": "sweeper-dgd",
        "namespace": "demo",
    }
    assert all(
        "runtimeVersionOverride" not in component
        for component in dgd["spec"]["components"]
    )


def test_materialize_direct_uses_config_modifiers(monkeypatch) -> None:
    captured = {}

    class FakeMaterializationError(Exception):
        pass

    class FakeResult:
        def __init__(self) -> None:
            self.dgd = {
                "apiVersion": "nvidia.com/v1beta1",
                "kind": "DynamoGraphDeployment",
                "metadata": {"name": "direct"},
                "spec": {"components": [{"name": "Worker"}]},
            }

    def fake_materialize(config, *, image, num_gpus_per_node):
        captured.update(
            config=config,
            image=image,
            num_gpus_per_node=num_gpus_per_node,
        )
        return FakeResult()

    monkeypatch.setattr(
        direct_renderer,
        "_load_materializer",
        lambda: (FakeMaterializationError, fake_materialize),
    )

    candidate = _candidate(deployment_mode="agg")
    rendered = render_dgd(
        candidate,
        SimpleNamespace(isl=4000, osl=1000),
        _options(),
        dgd_name="sweeper-dgd",
        renderer="direct",
    )

    assert captured == {
        "config": candidate.config,
        "image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.0",
        "num_gpus_per_node": 8,
    }
    dgd = yaml.safe_load(rendered)
    assert dgd["metadata"] == {
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


def test_patch_manifest_preserves_non_dgd_documents() -> None:
    rendered = """
apiVersion: v1
kind: ConfigMap
metadata:
  name: engine-config
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
        _options(),
        dgd_name="sweeper-dgd",
    )

    documents = list(yaml.safe_load_all(patched))
    assert [document["kind"] for document in documents] == [
        "ConfigMap",
        "DynamoGraphDeployment",
    ]
    assert documents[1]["metadata"]["name"] == "sweeper-dgd"


def test_runtime_version_override_is_only_written_when_explicit() -> None:
    rendered = """
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: generated
spec:
  components:
  - name: Worker
"""

    patched = base_module.patch_dgd_manifest(
        rendered,
        _options(runtime_version_override="1.4.2"),
        dgd_name="sweeper-dgd",
    )

    component = yaml.safe_load(patched)["spec"]["components"][0]
    assert component["runtimeVersionOverride"] == "1.4.2"


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
