# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import yaml

from dynamo.profiler.sweeper import dgd as dgd_module
from dynamo.profiler.sweeper.dgd import (
    CandidateMaterializationError,
    DGDMaterializationOptions,
    materialize_candidate_dgd,
)

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


def _options(**overrides) -> DGDMaterializationOptions:
    values = {
        "backend": "vllm",
        "backend_version": "0.20.1",
        "backend_image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.0",
        "dynamo_version": "1.5.0",
        "namespace": "demo",
        "num_gpus_per_node": 8,
    }
    values.update(overrides)
    return DGDMaterializationOptions(**values)


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
        dgd_module,
        "_load_generator_api",
        lambda: (fake_from_sweeper_candidate, fake_generate_from_request),
    )
    candidate = _candidate()
    workload = SimpleNamespace(isl=4000, osl=1000)

    rendered = materialize_candidate_dgd(
        candidate,
        workload,
        _options(),
        candidate_index=4,
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
                "name_prefix": "sweeper-candidate",
            },
            "NodeConfig": {"num_gpus_per_node": 8},
        },
        "request": "request",
    }
    dgd = yaml.safe_load(rendered)
    assert dgd["metadata"] == {
        "name": "sweeper-candidate-004",
        "namespace": "demo",
    }
    assert {
        component["runtimeVersionOverride"] for component in dgd["spec"]["components"]
    } == {"1.5.0"}


@pytest.mark.parametrize(
    ("candidate", "message"),
    [
        (_candidate(backend="sglang"), "does not match target backend"),
        (
            _candidate(backend_version="0.19.0"),
            "does not match target backend version",
        ),
    ],
)
def test_materialize_requires_candidate_to_match_runtime_target(
    candidate, message
) -> None:
    with pytest.raises(CandidateMaterializationError, match=message):
        materialize_candidate_dgd(
            candidate,
            SimpleNamespace(isl=4000, osl=1000),
            _options(),
            candidate_index=0,
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

    patched = dgd_module._patch_dgd_manifest(
        rendered,
        _options(),
        candidate_index=1,
    )

    documents = list(yaml.safe_load_all(patched))
    assert [document["kind"] for document in documents] == [
        "ConfigMap",
        "DynamoGraphDeployment",
    ]
    assert documents[1]["metadata"]["name"] == "sweeper-candidate-001"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("backend", "unknown", "backend must be one of"),
        ("backend_version", "", "backend_version must not be empty"),
        ("backend_image", "", "backend_image must not be empty"),
        ("dynamo_version", "", "dynamo_version must not be empty"),
        ("num_gpus_per_node", 0, "num_gpus_per_node must be positive"),
    ],
)
def test_materialization_options_validate_required_inputs(
    field: str, value: object, message: str
) -> None:
    values = {
        "backend": "vllm",
        "backend_version": "0.20.1",
        "backend_image": "image",
        "dynamo_version": "1.5.0",
        "num_gpus_per_node": 8,
    }
    values[field] = value
    with pytest.raises(ValueError, match=message):
        DGDMaterializationOptions(**values)
