# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import render_mocker_campaign
import resolve_campaign_image_attestation
import yaml

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.planner,
]

PLANNER_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = PLANNER_ROOT.parent
SENTINEL_NAMESPACE = "mocker-contract-test"
SENTINEL_IMAGE = "registry.example/dynamo:immutable-mocker-build"
SENTINEL_RUNTIME_VERSION = "1.5.0"
SENTINEL_TENANT = "mocker-contract-tenant"
SENTINEL_WARM_REPLICAS = "1"
SENTINEL_MODEL_CACHE_CLAIM = "mocker-contract-model-cache"
SENTINEL_IMAGE_PULL_SECRET = "mocker-contract-registry"
MODEL = "Qwen/Qwen3-0.6B"
FIXED_CONFIG_PATH = "/etc/dynamo/mocker/engine-args.json"
PROFILE_PATH = (
    "/workspace/components/src/dynamo/planner/tests/data/"
    "profiling_results/H200_TP1P_TP1D"
)


def _render(path: Path, replacements: dict[str, str]) -> str:
    rendered = path.read_text(encoding="utf-8")
    for name, value in replacements.items():
        rendered = rendered.replace(f"${{{name}}}", value)
    assert "${" not in rendered
    return rendered


def _option(args: list[str], name: str) -> str:
    position = args.index(name)
    return args[position + 1]


def _resource_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for nested in value.values():
            keys.update(_resource_keys(nested))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for nested in value:
            keys.update(_resource_keys(nested))
        return keys
    return set()


def _mocker_documents(
    *,
    exact_head: bool = True,
    model_cache_claim: str | None = None,
    image_pull_secret: str | None = None,
) -> tuple[dict, dict]:
    if exact_head:
        max_num_seqs = "16"
        timing_flag = "--extra-engine-args"
        timing_value = FIXED_CONFIG_PATH
    else:
        max_num_seqs = "10"
        timing_flag = "--planner-profile-data"
        timing_value = PROFILE_PATH

    template = (PLANNER_ROOT / "mocker-campaign.yaml").read_text(encoding="utf-8")
    # CI parses checked-in YAML before rendering. Keep the template valid with
    # its literal placeholder scalars.
    assert len([document for document in yaml.safe_load_all(template) if document]) == 2
    rendered = render_mocker_campaign.render_manifest(
        template,
        {
            "NAMESPACE": SENTINEL_NAMESPACE,
            "DYNAMO_IMAGE": SENTINEL_IMAGE,
            "DYNAMO_RUNTIME_VERSION": SENTINEL_RUNTIME_VERSION,
            "MOCKER_MAX_NUM_SEQS": max_num_seqs,
            "MOCKER_TIMING_FLAG": timing_flag,
            "MOCKER_TIMING_VALUE": timing_value,
        },
        model_cache_claim=model_cache_claim,
        image_pull_secret=image_pull_secret,
    )
    documents = [document for document in yaml.safe_load_all(rendered) if document]
    assert len(documents) == 2
    assert documents[0]["kind"] == "ConfigMap"
    assert documents[1]["kind"] == "DynamoGraphDeployment"
    return documents[0], documents[1]


def test_mocker_template_preserves_the_live_batch_contract_on_cpu() -> None:
    configmap, dgd = _mocker_documents()

    assert configmap["metadata"] == {
        "name": "qwen3-0-6b-batch-mocker-config",
        "namespace": SENTINEL_NAMESPACE,
    }

    assert dgd["metadata"] == {
        "name": "qwen3-0-6b-batch",
        "namespace": SENTINEL_NAMESPACE,
    }
    assert "annotations" not in dgd["metadata"]
    assert "minAvailable" not in _resource_keys(dgd)
    assert "nvidia.com/gpu" not in _resource_keys(dgd)

    components = {
        component["name"]: component for component in dgd["spec"]["components"]
    }
    assert set(components) == {"Frontend", "worker"}

    frontend = components["Frontend"]
    assert frontend["type"] == "frontend"
    assert frontend["replicas"] == 1
    assert frontend["runtimeVersionOverride"] == SENTINEL_RUNTIME_VERSION
    frontend_container = frontend["podTemplate"]["spec"]["containers"][0]
    assert frontend_container["image"] == SENTINEL_IMAGE
    assert frontend_container["envFrom"] == [{"secretRef": {"name": "hf-token-secret"}}]
    assert frontend_container["command"] == ["python3", "-m", "dynamo.frontend"]
    assert frontend_container["args"] == [
        "--router-mode",
        "round-robin",
        "--http-port",
        "8000",
    ]

    worker = components["worker"]
    assert worker["type"] == "worker"
    assert worker["replicas"] == 1
    assert worker["runtimeVersionOverride"] == SENTINEL_RUNTIME_VERSION
    assert worker["scalingAdapter"] == {}
    worker_container = worker["podTemplate"]["spec"]["containers"][0]
    assert worker_container["image"] == SENTINEL_IMAGE
    assert worker_container["envFrom"] == [{"secretRef": {"name": "hf-token-secret"}}]
    assert worker_container["command"] == ["python3", "-m", "dynamo.mocker"]
    assert _option(worker_container["args"], "--model-path") == MODEL
    assert _option(worker_container["args"], "--model-name") == MODEL
    assert _option(worker_container["args"], "--num-workers") == "1"
    assert _option(worker_container["args"], "--disaggregation-mode") == "agg"
    assert frontend["podTemplate"]["spec"]["imagePullSecrets"] == []
    assert worker["podTemplate"]["spec"]["imagePullSecrets"] == []
    assert frontend["podTemplate"]["spec"]["volumes"] == [
        {
            "name": "model-cache",
            "emptyDir": {},
        }
    ]
    assert worker["podTemplate"]["spec"]["volumes"] == [
        {
            "name": "model-cache",
            "emptyDir": {},
        },
        {
            "name": "mocker-config",
            "configMap": {"name": "qwen3-0-6b-batch-mocker-config"},
        },
    ]


def test_mocker_template_accepts_optional_cache_and_pull_secret_overrides() -> None:
    _, dgd = _mocker_documents(
        model_cache_claim=SENTINEL_MODEL_CACHE_CLAIM,
        image_pull_secret=SENTINEL_IMAGE_PULL_SECRET,
    )
    components = {
        component["name"]: component for component in dgd["spec"]["components"]
    }

    for name in ("Frontend", "worker"):
        pod_spec = components[name]["podTemplate"]["spec"]
        assert pod_spec["imagePullSecrets"] == [{"name": SENTINEL_IMAGE_PULL_SECRET}]
        model_cache = next(
            volume for volume in pod_spec["volumes"] if volume["name"] == "model-cache"
        )
        assert model_cache["persistentVolumeClaim"] == {
            "claimName": SENTINEL_MODEL_CACHE_CLAIM
        }


def test_exact_head_mocker_uses_deterministic_fixed_timing() -> None:
    configmap, dgd = _mocker_documents()
    worker = next(
        component
        for component in dgd["spec"]["components"]
        if component["name"] == "worker"
    )
    args = worker["podTemplate"]["spec"]["containers"][0]["args"]

    assert _option(args, "--num-gpu-blocks-override") == "512"
    assert _option(args, "--block-size") == "64"
    assert _option(args, "--max-model-len") == "4096"
    assert _option(args, "--max-num-seqs") == "16"
    assert _option(args, "--max-num-batched-tokens") == "8192"
    assert "--no-enable-prefix-caching" in args
    assert _option(args, "--speedup-ratio") == "1.0"
    assert _option(args, "--extra-engine-args") == FIXED_CONFIG_PATH

    engine_args = json.loads(configmap["data"]["engine-args.json"])
    assert engine_args["engine_type"] == "vllm"
    assert engine_args["worker_type"] == "aggregated"
    assert engine_args["max_num_seqs"] == 16
    assert engine_args["timing_model"] == {
        "type": "fixed",
        "prefill_ms": 75.0,
        "decode_ms": 12.0,
    }


def test_release_pilot_profile_creates_the_configured_contention_envelope() -> None:
    _, dgd = _mocker_documents(exact_head=False)
    worker = next(
        component
        for component in dgd["spec"]["components"]
        if component["name"] == "worker"
    )
    args = worker["podTemplate"]["spec"]["containers"][0]["args"]

    assert _option(args, "--max-num-seqs") == "10"
    assert _option(args, "--planner-profile-data") == PROFILE_PATH

    planner_documents = list(
        yaml.safe_load_all(
            _render(
                PLANNER_ROOT / "planner-poc.yaml",
                {
                    "NAMESPACE": SENTINEL_NAMESPACE,
                    "PLANNER_IMAGE": SENTINEL_IMAGE,
                    "BATCH_TENANT": SENTINEL_TENANT,
                    "BATCH_WARM_REPLICAS": SENTINEL_WARM_REPLICAS,
                },
            )
        )
    )
    planner_configmap = next(
        document
        for document in planner_documents
        if document
        and document["kind"] == "ConfigMap"
        and document["metadata"]["name"] == "qwen3-0-6b-batch-planner-config"
    )
    policy = yaml.safe_load(planner_configmap["data"]["planner.yaml"])[
        "batch_scheduling"
    ]["pool"]

    # Approximate steady-state occupancy using Little's Law. The campaign uses
    # 64 +/- 8 online output tokens, 128 batch output tokens, 5/8/5 online
    # RPS, and eight stock Async workers. Derive the Planner caps from the
    # deployed 5.5-RPS safety policy instead of the superseded 10-RPS pilot
    # assumption. The bundled profile is approximately 10 ms prefill and
    # 7 ms/token in this region. This is a calibration check, not a GPU claim.
    online_seconds = (10.0 + 72 * 7.0) / 1000.0
    batch_seconds = (10.0 + 128 * 7.0) / 1000.0
    high_online_occupancy = 8 * online_seconds
    low_online_occupancy = 5 * online_seconds
    stock_batch_occupancy = 8.0
    safe_rps = policy["safe_rps_per_ready_replica"]
    max_batch_rps = policy["max_batch_admission_rps"]
    high_planner_cap = min(max_batch_rps, max(0.0, safe_rps - 8.0))
    low_planner_cap = min(max_batch_rps, max(0.0, safe_rps - 5.0))

    max_num_seqs = int(_option(args, "--max-num-seqs"))
    assert safe_rps == 5.5
    assert high_planner_cap == 0.0
    assert low_planner_cap == 0.5
    assert high_online_occupancy < max_num_seqs
    assert high_online_occupancy + stock_batch_occupancy > max_num_seqs
    assert high_online_occupancy + high_planner_cap * batch_seconds < max_num_seqs
    assert low_online_occupancy + low_planner_cap * batch_seconds < max_num_seqs


def test_mocker_model_matches_async_readiness_and_planner_online_labels() -> None:
    _, dgd = _mocker_documents()
    worker = next(
        component
        for component in dgd["spec"]["components"]
        if component["name"] == "worker"
    )
    worker_args = worker["podTemplate"]["spec"]["containers"][0]["args"]
    mocker_model = _option(worker_args, "--model-name")

    async_values = yaml.safe_load(
        _render(
            EXAMPLE_ROOT / "llm-d-async-planner-values.yaml",
            {"NAMESPACE": SENTINEL_NAMESPACE},
        )
    )
    readiness_query = async_values["ap"]["redis"]["queuesConfig"][0]["gate_params"][
        "query"
    ]

    planner_documents = list(
        yaml.safe_load_all(
            _render(
                PLANNER_ROOT / "planner-poc.yaml",
                {
                    "NAMESPACE": SENTINEL_NAMESPACE,
                    "PLANNER_IMAGE": SENTINEL_IMAGE,
                    "BATCH_TENANT": SENTINEL_TENANT,
                    "BATCH_WARM_REPLICAS": SENTINEL_WARM_REPLICAS,
                },
            )
        )
    )
    planner_configmap = next(
        document
        for document in planner_documents
        if document
        and document["kind"] == "ConfigMap"
        and document["metadata"]["name"] == "qwen3-0-6b-batch-planner-config"
    )
    planner_config = yaml.safe_load(planner_configmap["data"]["planner.yaml"])

    assert mocker_model == MODEL
    assert f'model="{mocker_model}"' in readiness_query
    assert f'namespace="{SENTINEL_NAMESPACE}"' in readiness_query
    assert planner_config["model_name"] == mocker_model
    assert planner_config["batch_scheduling"]["gateway"]["tenant"] == (SENTINEL_TENANT)
    # The Mocker explicitly simulates the vLLM engine protocol. Planner still
    # detects `dynamo.mocker` from the command and applies its logical-GPU
    # fallback; keeping the live vLLM backend preserves the campaign contract.
    assert planner_config["backend"] == "vllm"
    assert planner_config["decode_engine_num_gpu"] == 1
    assert planner_config["min_endpoint"] == 0
    assert planner_config["batch_scheduling"]["pool"]["min_replicas"] == 1
    assert (
        planner_config["batch_scheduling"]["pool"]["safe_rps_per_ready_replica"] == 5.5
    )
    assert planner_config["batch_scheduling"]["metrics"]["online_match_labels"] == {
        "endpoint": "chat_completions",
        "model": mocker_model,
        "request_type": "stream",
    }


def test_experiment_prometheus_is_namespace_local_and_scrapes_only_frontend() -> None:
    rendered = _render(
        PLANNER_ROOT / "experiment-prometheus.yaml",
        {"NAMESPACE": SENTINEL_NAMESPACE},
    )
    documents = [document for document in yaml.safe_load_all(rendered) if document]
    assert [document["kind"] for document in documents] == [
        "ConfigMap",
        "Deployment",
        "Service",
    ]
    assert all(
        document["metadata"]["namespace"] == SENTINEL_NAMESPACE
        for document in documents
    )

    config = yaml.safe_load(documents[0]["data"]["prometheus.yml"])
    assert config["global"]["scrape_interval"] == "1s"
    assert config["scrape_configs"] == [
        {
            "job_name": "dynamo-batch-frontend",
            "metrics_path": "/metrics",
            "static_configs": [
                {
                    "labels": {"namespace": SENTINEL_NAMESPACE},
                    "targets": ["qwen3-0-6b-batch-frontend:8000"],
                }
            ],
        }
    ]

    container = documents[1]["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == "quay.io/prometheus/prometheus:v3.5.0"
    assert "--storage.tsdb.retention.time=3h" in container["args"]
    assert documents[2]["spec"]["ports"] == [
        {"name": "http", "port": 9090, "targetPort": "http"}
    ]

    base_values = yaml.safe_load(
        (EXAMPLE_ROOT / "llm-d-async-values.yaml").read_text(encoding="utf-8")
    )
    assert base_values["ap"]["prometheusURL"] == (
        "http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090"
    )

    campaign_values = yaml.safe_load(
        (PLANNER_ROOT / "llm-d-async-mocker-campaign-values.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert campaign_values == {
        "ap": {
            "prometheusURL": "http://planner-impact-prometheus:9090",
            "podMonitor": {"enabled": False},
        }
    }


def test_campaign_driver_is_portable_and_runs_the_frozen_five_round_order() -> None:
    script = (
        PLANNER_ROOT
        / "experiment"
        / "workloads"
        / "run_local_mocker_impact_campaign.zsh"
    ).read_text(encoding="utf-8")

    assert "/Users/" not in script
    assert "/private/tmp/" not in script
    assert "run-cell.zsh" not in script
    assert "llm-d-async-stock-campaign-values.yaml" in script
    assert "llm-d-async-planner-values.yaml" in script
    assert "planner-poc.yaml" in script
    assert "batch_warm_replicas=1" in script
    assert "${BATCH_WARM_REPLICAS}" in script
    assert "DYNAMO_IMAGE_ID_ATTESTATION" in script
    assert "--image-id-attestation" in script
    assert "--minimum-repetitions 5" in script
    assert "--atomic" in script
    assert "--rollback-on-failure" not in script
    assert re.findall(r"^run_cell (r\d-\S+) (\S+)$", script, re.MULTILINE) == [
        ("r1-online", "online-only"),
        ("r1-stock", "stock"),
        ("r1-native", "planner-native"),
        ("r2-stock", "stock"),
        ("r2-native", "planner-native"),
        ("r2-online", "online-only"),
        ("r3-native", "planner-native"),
        ("r3-online", "online-only"),
        ("r3-stock", "stock"),
        ("r4-online", "online-only"),
        ("r4-stock", "stock"),
        ("r4-native", "planner-native"),
        ("r5-stock", "stock"),
        ("r5-native", "planner-native"),
        ("r5-online", "online-only"),
    ]


def test_live_async_readiness_check_matches_json_escaped_promql() -> None:
    readme = (EXAMPLE_ROOT / "planner-scheduling" / "README.md").read_text(
        encoding="utf-8"
    )
    queue_config = json.dumps(
        [
            {
                "gate_type": "prometheus-query",
                "gate_params": {
                    "query": (
                        "min(dynamo_frontend_model_ready{"
                        f'namespace="{SENTINEL_NAMESPACE}"'
                        "})"
                    )
                },
            }
        ],
        separators=(",", ":"),
    )

    assert f'namespace=\\"{SENTINEL_NAMESPACE}\\"' in queue_config
    assert r'grep -Fq -- "namespace=\\\"${NAMESPACE}\\\""' in readme


def _write_attestation_run(
    root: Path,
    *,
    label: str,
    arm: str,
    inventory: list[dict],
) -> tuple[str, str, Path]:
    run_directory = root / label
    run_directory.mkdir()
    (run_directory / "metadata.json").write_text(
        json.dumps({"arm": arm}), encoding="utf-8"
    )
    if arm == "online-only":
        evidence_root = run_directory / "online-evidence"
    else:
        evidence_root = run_directory / "batch-harness" / "results" / "raw" / "child"
    for phase in ("start", "end"):
        phase_directory = evidence_root / "kubernetes" / phase
        phase_directory.mkdir(parents=True)
        (phase_directory / "images.json").write_text(
            json.dumps(inventory), encoding="utf-8"
        )
    return label, arm, run_directory


def _write_attestation_index(root: Path, rows: list[tuple[str, str, Path]]) -> Path:
    index = root / "runs.tsv"
    index.write_text(
        "".join(
            f"{label}\t{arm}\t{run_directory}\n" for label, arm, run_directory in rows
        ),
        encoding="utf-8",
    )
    return index


def test_campaign_attestation_is_added_only_for_the_exact_sanitized_image(
    tmp_path: Path,
) -> None:
    inventory = [{"image": SENTINEL_IMAGE, "image_id": "<redacted>"}]
    rows = [
        _write_attestation_run(
            tmp_path, label="online", arm="online-only", inventory=inventory
        ),
        _write_attestation_run(
            tmp_path, label="stock", arm="stock", inventory=inventory
        ),
    ]
    index = _write_attestation_index(tmp_path, rows)

    with pytest.raises(
        resolve_campaign_image_attestation.AttestationError,
        match="DYNAMO_IMAGE_ID_ATTESTATION",
    ):
        resolve_campaign_image_attestation.resolve_attestation(
            index, SENTINEL_IMAGE, None
        )

    resolved = "docker-pullable://registry.example/dynamo@sha256:abc123"
    assert (
        resolve_campaign_image_attestation.resolve_attestation(
            index, SENTINEL_IMAGE, resolved
        )
        == f"{SENTINEL_IMAGE}={resolved}"
    )
    with pytest.raises(
        resolve_campaign_image_attestation.AttestationError,
        match="resolved imageID only",
    ):
        resolve_campaign_image_attestation.resolve_attestation(
            index, SENTINEL_IMAGE, f"{SENTINEL_IMAGE}={resolved}"
        )


def test_campaign_attestation_is_not_emitted_for_resolved_inventories(
    tmp_path: Path,
) -> None:
    inventory = [
        {
            "image": SENTINEL_IMAGE,
            "image_id": "docker-pullable://registry.example/dynamo@sha256:abc123",
        }
    ]
    index = _write_attestation_index(
        tmp_path,
        [
            _write_attestation_run(
                tmp_path, label="online", arm="online-only", inventory=inventory
            )
        ],
    )

    assert (
        resolve_campaign_image_attestation.resolve_attestation(
            index, SENTINEL_IMAGE, "unused-resolved-id"
        )
        is None
    )


def test_campaign_attestation_rejects_an_additional_redacted_image_ref(
    tmp_path: Path,
) -> None:
    other_image = "registry.example/llm-d-async:immutable-build"
    inventory = [
        {"image": SENTINEL_IMAGE, "image_id": "<redacted>"},
        {"image": other_image, "image_id": "<redacted>"},
    ]
    index = _write_attestation_index(
        tmp_path,
        [
            _write_attestation_run(
                tmp_path, label="online", arm="online-only", inventory=inventory
            )
        ],
    )

    with pytest.raises(
        resolve_campaign_image_attestation.AttestationError,
        match=re.escape(other_image),
    ):
        resolve_campaign_image_attestation.resolve_attestation(
            index, SENTINEL_IMAGE, "resolved-id"
        )
