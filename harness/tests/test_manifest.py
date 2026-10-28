# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`dynamo_test.manifest`."""

import pytest

yaml = pytest.importorskip("yaml")

from dynamo_test.manifest import (  # noqa: E402
    ManifestError,
    NoGraphDeployment,
    Plan,
    Schema,
    role_of,
)
from dynamo_test.roles import PortName, Role  # noqa: E402

V1BETA1 = """
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: demo
spec:
  components:
    - name: Frontend
      replicas: 1
      podTemplate:
        spec:
          containers:
            - name: main
              image: nvcr.io/dynamo:latest
              command: ["/bin/bash", "-lc"]
              args: ["exec python3 -m dynamo.frontend --http-port 8000"]
              ports:
                - name: http
                  containerPort: 8000
                - name: metrics
                  containerPort: 9090
    - name: VllmDecodeWorker
      replicas: 2
      podTemplate:
        spec:
          containers:
            - name: main
              image: nvcr.io/dynamo:latest
              command: ["/bin/bash", "-lc"]
              args:
                - "ulimit -n 65536 && exec python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B --max-model-len 2048"
              resources:
                limits:
                  nvidia.com/gpu: 2
"""

V1ALPHA1 = """
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: legacy
spec:
  services:
    Frontend:
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: nvcr.io/dynamo:latest
          command: ["python3", "-m", "dynamo.frontend"]
          args: ["--http-port", "8000"]
    SglangWorker:
      replicas: 1
      resources:
        limits:
          gpu: 4
      extraPodSpec:
        mainContainer:
          image: nvcr.io/dynamo:latest
          command: ["python3", "-m", "dynamo.sglang"]
          args: ["--model-path", "Qwen/Qwen3-0.6B", "--tp", "4"]
"""

WITH_CONFIGMAP = (
    """
apiVersion: v1
kind: ConfigMap
metadata:
  name: engine-config
data:
  prefill.yaml: "model_path: Qwen/Qwen3-0.6B"
---
"""
    + V1BETA1
)


# ------------------------------------------------------------ role inference


@pytest.mark.parametrize(
    "name, role",
    [
        ("Frontend", Role.FRONTEND),
        ("frontend", Role.FRONTEND),
        ("VllmPrefillWorker", Role.PREFILL),  # "prefill" must beat "worker"
        ("prefill", Role.PREFILL),
        ("VllmDecodeWorker", Role.DECODE),
        ("decode", Role.DECODE),
        ("EncodeWorker", Role.ENCODE),
        ("Planner", Role.PLANNER),
        ("GlobalPlanner", Role.PLANNER),
        ("LocalRouter", Role.ROUTER),
        ("Epp", Role.ROUTER),
        ("agg", Role.WORKER),
        ("VllmWorker", Role.WORKER),
        # Both casings are in the corpus; matching one and not the other is how
        # a log lookup silently returns nothing for a healthy service.
        ("TrtllmWorker", Role.WORKER),
        ("TRTLLMWorker", Role.WORKER),
    ],
)
def test_role_inference_covers_the_names_in_use(name, role):
    assert role_of(name).require() is role


def test_an_unrecognised_name_is_absent_not_guessed():
    fact = role_of("Zookeeper")
    assert fact.is_absent
    assert "frontend" in fact.detail


# -------------------------------------------------------------------- loading


def test_reads_v1beta1_components():
    plan = Plan.from_yaml(V1BETA1)
    assert plan.schema is Schema.V1BETA1
    assert plan.name == "demo"
    assert [c.name for c in plan] == ["Frontend", "VllmDecodeWorker"]


def test_reads_v1alpha1_services():
    """75 documents in the corpus use this schema; it is not legacy."""
    plan = Plan.from_yaml(V1ALPHA1)
    assert plan.schema is Schema.V1ALPHA1
    assert plan["SglangWorker"].role is Role.WORKER


def test_a_configmap_alongside_the_deployment_is_ignored():
    """`safe_load` alone reads the first document, which is the wrong one."""
    plan = Plan.from_yaml(WITH_CONFIGMAP)
    assert plan.name == "demo"


def test_a_file_with_no_deployment_says_what_it_does_contain():
    with pytest.raises(NoGraphDeployment, match="ConfigMap"):
        Plan.from_yaml("kind: ConfigMap\nmetadata:\n  name: x\n")


def test_invalid_yaml_is_reported_as_such():
    with pytest.raises(ManifestError, match="not valid YAML"):
        Plan.from_yaml("${TEMPLATE_VAR}\n  broken: [")


# ------------------------------------------------------- several deployments


MULTI = V1BETA1 + "\n---\n" + V1BETA1.replace("name: demo", "name: demo-two")


def test_several_deployments_in_one_file_are_rejected_with_a_way_forward():
    with pytest.raises(ManifestError) as exc:
        Plan.from_yaml(MULTI)
    assert "all_from_yaml" in str(exc.value)
    assert "demo-two" in str(exc.value)


def test_all_from_yaml_gives_one_plan_each():
    """Four `global_planner` examples describe several deployments in one file."""
    plans = Plan.all_from_yaml(MULTI)
    assert [p.name for p in plans] == ["demo", "demo-two"]


def test_select_picks_one_by_name():
    assert Plan.from_yaml(MULTI, select="demo-two").name == "demo-two"


def test_select_of_an_absent_name_lists_what_is_there():
    with pytest.raises(ManifestError, match="demo-two"):
        Plan.from_yaml(MULTI, select="nope")


# ------------------------------------------------------------------ reading


def test_component_resolves_its_engine_and_model():
    worker = Plan.from_yaml(V1BETA1)["VllmDecodeWorker"]
    assert worker.backend.require() == "vllm"
    assert worker.read("model").require() == "Qwen/Qwen3-0.6B"
    assert worker.read("context_length").require() == "2048"


def test_the_frontend_has_no_engine_and_says_so_as_unknown():
    frontend = Plan.from_yaml(V1BETA1)["Frontend"]
    assert frontend.backend.is_absent
    assert frontend.read("model").is_unknown  # no dialect, not "no model"


def test_gpus_read_from_either_schema():
    assert Plan.from_yaml(V1BETA1)["VllmDecodeWorker"].gpus.require() == 2
    assert Plan.from_yaml(V1ALPHA1)["SglangWorker"].gpus.require() == 4


def test_lookup_by_role_or_by_name():
    plan = Plan.from_yaml(V1BETA1)
    assert plan[Role.DECODE].name == "VllmDecodeWorker"
    assert plan["VllmDecodeWorker"].role is Role.DECODE


def test_an_absent_component_lists_the_ones_present():
    with pytest.raises(KeyError, match="VllmDecodeWorker"):
        Plan.from_yaml(V1BETA1)["PrefillWorker"]


# -------------------------------------------------------------- role table


def test_the_role_table_is_derived_from_the_plan():
    table = Plan.from_yaml(V1BETA1).roles()
    assert table.require(Role.DECODE).service == "VllmDecodeWorker"
    assert table.require(Role.DECODE).log_key == "vllmdecodeworker"
    assert table.require(Role.FRONTEND).port(PortName.METRICS) == 9090


def test_two_components_sharing_a_role_is_an_error():
    """A role must name one service, or a selector cannot say which it means."""
    doubled = V1BETA1.replace("name: Frontend", "name: VllmDecode", 1)
    with pytest.raises(ManifestError, match="share a role"):
        Plan.from_yaml(doubled).roles()


# ------------------------------------------------------------------ editing


def test_setting_a_semantic_writes_the_engine_s_own_flag():
    plan = Plan.from_yaml(V1ALPHA1)
    plan.set("SglangWorker", context_length=4096)
    # SGLang spells it --context-length, not --max-model-len.
    assert "--context-length" in plan["SglangWorker"].argv.as_container_args()
    assert plan["SglangWorker"].read("context_length").require() == "4096"


def test_editing_preserves_the_shell_command_around_the_flag():
    plan = Plan.from_yaml(V1BETA1)
    plan.set(Role.DECODE, context_length=4096)
    command = plan["VllmDecodeWorker"].argv.as_shell_string()
    assert command.startswith("ulimit -n 65536 && exec python3")
    assert "'&&'" not in command
    assert command.count("--max-model-len") == 1


def test_editing_survives_a_yaml_round_trip():
    plan = Plan.from_yaml(V1BETA1)
    plan.set(Role.DECODE, context_length=4096, max_batch_size=32)
    reloaded = Plan.from_yaml(plan.to_yaml())
    assert reloaded[Role.DECODE].read("context_length").require() == "4096"
    assert reloaded[Role.DECODE].read("max_batch_size").require() == "32"
    assert reloaded[Role.DECODE].argv.as_shell_string().startswith("ulimit -n 65536 &&")


def test_scaling_a_role():
    plan = Plan.from_yaml(V1BETA1).scale(Role.DECODE, 4)
    assert plan[Role.DECODE].replicas == 4
    assert Plan.from_yaml(plan.to_yaml())[Role.DECODE].replicas == 4


def test_setting_on_a_component_with_no_engine_is_refused():
    with pytest.raises(ManifestError, match="Frontend"):
        Plan.from_yaml(V1BETA1).set("Frontend", context_length=4096)


# ------------------------------------------------------------------- record


def test_the_plan_serialises_for_the_run_record():
    import json

    record = Plan.from_yaml(V1BETA1).to_record()
    assert record["schema"] == "v1beta1"
    decode = next(c for c in record["components"] if c["role"] == "decode")
    assert decode["backend"] == "vllm"
    assert decode["model"] == "Qwen/Qwen3-0.6B"
    assert decode["gpus"] == 2
    assert json.loads(json.dumps(record))
