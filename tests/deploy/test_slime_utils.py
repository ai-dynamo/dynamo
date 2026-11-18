# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Slime tutorial test's helpers (tests/deploy/slime_utils.py)
and for the wiring of the tutorial's manifests in tests/deploy/slime/."""

import ast
import math
import shlex
from urllib.parse import urlsplit

import pytest
import yaml

from tests.deploy.slime_utils import (
    ROLE_LABEL,
    TUTORIAL_CLAIM,
    SlimeTrainingPlan,
    SlimeTutorialManifests,
    SlimeTutorialOptions,
    command_flags,
    count_weight_reloads,
    missing_paths,
    parse_ls_listing,
    routed_requests,
    summarize_driver_log,
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

TUTORIAL_SLIME_IMAGE = "slimerl/slime:nightly-dev-20261027a-cu130"


def _load(**options: str) -> SlimeTutorialManifests:
    return SlimeTutorialManifests.load(SlimeTutorialOptions(**options))


def _pod_specs(manifests: SlimeTutorialManifests) -> dict[str, dict]:
    """Every Pod spec in the tutorial, by owner."""
    head = manifests.rayjob["spec"]["rayClusterSpec"]["headGroupSpec"]
    return {
        "setup": manifests.setup_job["spec"]["template"]["spec"],
        "ray-head": head["template"]["spec"],
        **{
            component["name"]: component["podTemplate"]["spec"]
            for component in manifests.dgd["spec"]["components"]
        },
    }


def _claim_names(manifests: SlimeTutorialManifests) -> set[str]:
    return {
        volume["persistentVolumeClaim"]["claimName"]
        for spec in _pod_specs(manifests).values()
        for volume in spec.get("volumes", [])
        if "persistentVolumeClaim" in volume
    }


def test_tutorial_manifests_change_only_the_storage_class_by_default() -> None:
    manifests = _load(storage_class="rwx-class")

    assert manifests.pvc is not None
    assert manifests.pvc["metadata"]["name"] == TUTORIAL_CLAIM
    assert manifests.pvc["spec"]["storageClassName"] == "rwx-class"
    assert _claim_names(manifests) == {TUTORIAL_CLAIM}
    specs = _pod_specs(manifests)
    assert not any("tolerations" in spec for spec in specs.values())
    for owner in ("setup", "ray-head"):
        assert [c["image"] for c in specs[owner]["containers"]] == [
            TUTORIAL_SLIME_IMAGE
        ]


def test_existing_claim_replaces_the_tutorial_claim() -> None:
    manifests = _load(existing_claim="lustre-data")

    assert manifests.pvc is None
    assert _claim_names(manifests) == {"lustre-data"}
    assert manifests.creation_order()[0] == [manifests.setup_job]


def test_slime_image_and_gpu_toleration_reach_only_their_pods() -> None:
    manifests = _load(
        storage_class="rwx-class",
        slime_image="registry.example/slime:dev",
        gpu_toleration="nvidia.com/gpu",
    )

    specs = _pod_specs(manifests)
    for owner in ("setup", "ray-head"):
        assert [c["image"] for c in specs[owner]["containers"]] == [
            "registry.example/slime:dev"
        ]
    toleration = {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
    for owner in ("setup", "ray-head", "worker-0", "worker-1"):
        assert specs[owner]["tolerations"] == [toleration], owner
    assert "tolerations" not in specs["frontend"]
    frontend = specs["frontend"]["containers"][0]
    assert frontend["image"] == "nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.6.1"


def test_training_plan_reads_the_tutorial_counts() -> None:
    plan = SlimeTrainingPlan.from_rayjob(_load(storage_class="rwx").rayjob)

    assert plan.engine_addresses == (
        "slime-sglang-engine-0:30000",
        "slime-sglang-engine-1:30000",
    )
    assert plan.rollout_url == "http://slime-sglang-frontend:8000"
    # The tutorial's "Verify the Run" numbers: 640 routed requests, 10 steps,
    # and 11 weight loads per engine.
    assert (plan.rollout_requests, plan.train_steps, plan.weight_syncs) == (
        640,
        10,
        11,
    )
    assert [str(path) for path in plan.staged_paths] == [
        "/data/models/Qwen3-0.6B",
        "/data/models/Qwen3-0.6B_torch_dist",
        "/data/datasets/gsm8k",
    ]


def test_trainer_addresses_lead_to_the_dgd() -> None:
    """Each engine address names an engine Service whose selector matches one
    single-replica DGD worker on that port, and the rollout URL names the
    frontend Service that the operator creates for the DGD."""
    manifests = _load(storage_class="rwx")
    plan = SlimeTrainingPlan.from_rayjob(manifests.rayjob)
    services = {
        service["metadata"]["name"]: service for service in manifests.engine_services
    }
    workers = {
        component["podTemplate"]["metadata"]["labels"][ROLE_LABEL]: component
        for component in manifests.dgd["spec"]["components"]
        if component["type"] == "worker"
    }

    assert len(plan.engine_addresses) == len(workers)
    for address in plan.engine_addresses:
        host, port = address.rsplit(":", 1)
        service = services[host]
        assert int(port) in {p["port"] for p in service["spec"]["ports"]}
        worker = workers[service["spec"]["selector"][ROLE_LABEL]]
        assert worker["replicas"] == 1
        args = worker["podTemplate"]["spec"]["containers"][0]["args"]
        assert command_flags(args)["--port"] == [port]
        assert "--incremental-streaming-output" in args
    assert sorted(manifests.engine_roles()) == sorted(workers)

    trainer_flags = command_flags(shlex.split(manifests.rayjob["spec"]["entrypoint"]))
    assert "--sglang-incremental-streaming-output" in trainer_flags
    frontend = urlsplit(plan.rollout_url)
    dgd_name = manifests.dgd["metadata"]["name"]
    assert (frontend.hostname, frontend.port) == (f"{dgd_name}-frontend", 8000)
    assert manifests.served_model_name() == "Qwen/Qwen3-0.6B"


def test_custom_functions_define_the_hooks_that_the_rayjob_loads() -> None:
    """The RayJob's custom-function flags name functions in the ConfigMap, and
    the ConfigMap is mounted on the Ray actors' PYTHONPATH."""
    manifests = _load(storage_class="rwx")
    rayjob = manifests.rayjob["spec"]
    flags = command_flags(shlex.split(rayjob["entrypoint"]))
    sources = manifests.functions["data"]

    for flag in (
        "--custom-generate-function-path",
        "--custom-update-weight-post-write-path",
    ):
        (path,) = flags[flag]
        module, function = path.rsplit(".", 1)
        tree = ast.parse(sources[f"{module}.py"])
        defined = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert function in defined, flag

    head = rayjob["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]
    (volume,) = [
        volume
        for volume in head["volumes"]
        if volume.get("configMap", {}).get("name")
        == manifests.functions["metadata"]["name"]
    ]
    (mount,) = [
        mount
        for container in head["containers"]
        for mount in container["volumeMounts"]
        if mount["name"] == volume["name"]
    ]
    env = yaml.safe_load(rayjob["runtimeEnvYAML"])["env_vars"]
    assert mount["mountPath"] in env["PYTHONPATH"].split(":")


def _frontend_line(worker_id: int) -> str:
    """A routing line as the 1.6.1 frontend logs it, color codes included."""
    dim, italic, reset = "\x1b[2m", "\x1b[3m", "\x1b[0m"
    return (
        f"{dim}2026-11-11T02:16:37.907153Z{reset} \x1b[32m INFO{reset} "
        f"{dim}dynamo_runtime::pipeline::network::egress::push_router{reset}"
        f"{dim}:{reset} Selected worker {italic}router_mode{reset}{dim}={reset}"
        f'"round-robin" {italic}worker_id{reset}{dim}={reset}{worker_id} '
        f"{dim}{italic}method{reset}{dim}={reset}POST"
    )


def test_routed_requests_reads_the_colored_frontend_log() -> None:
    log = "\n".join(
        [
            _frontend_line(3134858258051881),
            "INFO dynamo_llm::http::service: listening on 0.0.0.0:8000",
            _frontend_line(5886276872228916),
            _frontend_line(3134858258051881),
        ]
    )

    routed = routed_requests(log)

    assert routed.total == 3
    assert routed.by_worker == {"3134858258051881": 2, "5886276872228916": 1}
    assert routed.router_modes == {"round-robin"}


def test_count_weight_reloads_counts_only_successful_reloads() -> None:
    log = (
        '[2026-11-11 02:16:36] INFO:     10.244.7.45:59390 - "POST '
        '/update_weights_from_disk HTTP/1.1" 200 OK\n'
        '[2026-11-11 02:16:40] INFO:     127.0.0.1:41234 - "POST /generate '
        'HTTP/1.1" 200 OK\n'
        '[2026-11-11 02:16:49] INFO:     10.244.7.45:49724 - "POST '
        '/update_weights_from_disk HTTP/1.1" 500 Internal Server Error\n'
        '[2026-11-11 02:16:54] INFO:     10.244.7.45:49768 - "POST '
        '/update_weights_from_disk HTTP/1.1" 200 OK\n'
    )

    assert count_weight_reloads(log) == 2


ACTOR = "\x1b[36m(MegatronTrainRayActor pid=2396)\x1b[0m"


def _step_line(step: int, diff: str | None) -> str:
    metrics = "{'train/loss': 1.8e-09, "
    if diff is not None:
        metrics += f"'train/train_rollout_logprob_abs_diff': {diff}, "
    metrics += "'train/grad_norm': 2.36}"
    return f"{ACTOR} [2026-11-11 02:16:45] model.py:976 - step {step}: {metrics}"


def test_summarize_driver_log_counts_each_rollout_and_step_once() -> None:
    rollout = (
        ACTOR + " [2026-11-11 02:16:39] train_metric_utils.py:135 - rollout {}: "
        "{{'rollout/response_lengths': 145.2, 'rollout/raw_reward': 0.73}}"
    )
    log = "\n".join(
        [
            "[2026-11-11 02:15:58] external.py:147 - Detected external SGLang "
            "engines: [{'url': 'http://slime-sglang-engine-0:30000', "
            "'worker_type': 'regular'}, {'url': "
            "'http://slime-sglang-engine-1:30000', 'worker_type': 'regular'}]",
            rollout.format(0),
            _step_line(0, "0.009679259732365608"),
            rollout.format(1),
            _step_line(1, "0.0098"),
            # The submitter can replay lines after it reconnects to the job.
            _step_line(1, "0.0098"),
            _step_line(2, None),
            "Job 'slime-grpo-kn4mx' succeeded",
        ]
    )

    summary = summarize_driver_log(log)

    assert summary.engine_urls == (
        "http://slime-sglang-engine-0:30000",
        "http://slime-sglang-engine-1:30000",
    )
    assert summary.rollouts == {0, 1}
    assert sorted(summary.logprob_abs_diff_by_step) == [0, 1, 2]
    assert summary.logprob_abs_diff_by_step[1] == pytest.approx(0.0098)
    assert math.isnan(summary.logprob_abs_diff_by_step[2])


def test_staged_data_comes_from_the_setup_jobs_listing() -> None:
    log = (
        "Fetching 12 files: 100% 12/12\n"
        "/data/models/Qwen3-0.6B\n"
        "/data/datasets:\n"
        "gsm8k\n"
        "\n"
        "/data/models:\n"
        "Qwen3-0.6B\n"
        "Qwen3-0.6B_torch_dist\n"
    )
    plan = SlimeTrainingPlan.from_rayjob(_load(storage_class="rwx").rayjob)

    listing = parse_ls_listing(log)

    assert listing == {
        "/data/datasets": {"gsm8k"},
        "/data/models": {"Qwen3-0.6B", "Qwen3-0.6B_torch_dist"},
    }
    assert missing_paths(listing, plan.staged_paths) == []
    assert missing_paths({"/data/models": {"Qwen3-0.6B"}}, plan.staged_paths) == [
        "/data/models/Qwen3-0.6B_torch_dist",
        "/data/datasets/gsm8k",
    ]
