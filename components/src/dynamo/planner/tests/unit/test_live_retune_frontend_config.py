# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the retune imports.

"""Frontend startup parsing and the spec premises of live retune.

Covers ``find_frontend``, ``parse_frontend`` (the operator's argv
defaulting, env merging and ``$(VAR)`` expansion, the frontend's own parser
built against an explicit env, a golden comparison with the real frontend
parser, disaggregated maps) and
``evaluate_premises`` (P4, P6-P10) on table-driven DGD fixtures, including
P4 on workers that advertise their own router configuration (compared with
the real worker parser) and the P7 check of ``retune.dgdr.model``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import types
import warnings
from typing import Any

import pytest

pytest.importorskip(
    "aisim.sweeper",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from dynamo.common.configuration.groups.kv_router_args import _KV_ROUTER_FIELDS
from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig
from dynamo.llm import KvRouterConfig
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.retune.live import frontend_config as fc
from dynamo.planner.retune.live.contracts import FrontendStartup, PremiseReport

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

DGD = "llm"
NAMESPACE = "team-a"
CLAIM = "request-traces"
MODEL = "Qwen/Qwen3-32B"
TRACE_PATH = "/trace/segments/$(DYN_PARENT_DGD_K8S_NAME)/$(DYN_POD_NAME)/req"
KV_ARGS = ("-m", "dynamo.frontend", "--router-mode", "kv")
DEFAULT_AGG_MAP = {
    "decode_active_request_weight": 0.0,
    "disk_cache_hit_weight": 0.25,
    "host_cache_hit_weight": 0.75,
    "overlap_score_credit": 1.0,
    "overlap_score_credit_decay": 0.0,
    "prefill_load_scale": 1.0,
    "router_temperature": 0.0,
    "shared_cache_multiplier": 0.0,
}

PRESET = {
    "tensor": 2,
    "pipeline": 1,
    "attention_data": 1,
    "moe_tensor": 1,
    "moe_expert": 1,
}
ENGINE_WORKER = {
    "parallelism": {"preset": [PRESET]},
    "scheduler": {"max_batched_tokens": 8192, "max_sequences": 256},
    "kv_cache": {"block_size": 64, "capacity": {"memory_fraction": 0.9}},
}
ENGINES = {
    "agg": {
        "mode": "aggregated",
        "backend": "vllm",
        "backend_version": "0.11.0",
        "context_length": 32768,
        "workers": {"aggregated": ENGINE_WORKER, "prefill": None, "decode": None},
    },
    "disagg": {
        "mode": "disaggregated",
        "backend": "vllm",
        "backend_version": "0.11.0",
        "context_length": 32768,
        "workers": {"prefill": ENGINE_WORKER, "decode": ENGINE_WORKER},
    },
}


# ------------------------------------------------------------- fixtures


def trace_env(**overrides: Any) -> list[dict[str, Any]]:
    """The P10 trace recipe env; an override of None drops the entry."""

    entries: dict[str, dict[str, Any]] = {
        "DYN_POD_NAME": {
            "name": "DYN_POD_NAME",
            "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}},
        },
        "DYN_REQUEST_TRACE": {"name": "DYN_REQUEST_TRACE", "value": "1"},
        "DYN_REQUEST_TRACE_RECORDS": {
            "name": "DYN_REQUEST_TRACE_RECORDS",
            "value": "request_end",
        },
        "DYN_REQUEST_TRACE_SINKS": {"name": "DYN_REQUEST_TRACE_SINKS", "value": "file"},
        "DYN_REQUEST_TRACE_FILE_FORMAT": {
            "name": "DYN_REQUEST_TRACE_FILE_FORMAT",
            "value": "jsonl_gz",
        },
        "DYN_REQUEST_TRACE_FILE_PATH": {
            "name": "DYN_REQUEST_TRACE_FILE_PATH",
            "value": TRACE_PATH,
        },
        "DYN_REQUEST_TRACE_FILE_ROLL_BYTES": {
            "name": "DYN_REQUEST_TRACE_FILE_ROLL_BYTES",
            "value": "67108864",
        },
        "DYN_REQUEST_TRACE_CAPACITY": {
            "name": "DYN_REQUEST_TRACE_CAPACITY",
            "value": "8192",
        },
        # The admin listener that serves /router_policy (P10).
        "DYN_ENABLE_FRONTEND_ADMIN_API": {
            "name": "DYN_ENABLE_FRONTEND_ADMIN_API",
            "value": "1",
        },
        "DYN_FRONTEND_ADMIN_HOST": {
            "name": "DYN_FRONTEND_ADMIN_HOST",
            "value": "0.0.0.0",
        },
    }
    for name, entry in overrides.items():
        if entry is None:
            entries.pop(name, None)
        elif isinstance(entry, dict):
            entries[name] = {"name": name, **entry}
        else:
            entries[name] = {"name": name, "value": entry}
    return list(entries.values())


def frontend(
    *,
    args: tuple[str, ...] | list[str] | None = KV_ARGS,
    command: tuple[str, ...] | list[str] | None = ("python3",),
    env: list[dict[str, Any]] | None = None,
    replicas: int | None = 1,
    mounts: list[dict[str, Any]] | None = None,
    volumes: list[dict[str, Any]] | None = None,
    name: str = "Frontend",
    **extra: Any,
) -> dict[str, Any]:
    main: dict[str, Any] = {
        "name": "main",
        "image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0",
        "env": trace_env() if env is None else env,
        "volumeMounts": (
            [{"name": "traces", "mountPath": "/trace"}] if mounts is None else mounts
        ),
    }
    if command is not None:
        main["command"] = list(command)
    if args is not None:
        main["args"] = list(args)
    component: dict[str, Any] = {
        "name": name,
        "type": "frontend",
        "podTemplate": {
            "spec": {
                "containers": [main],
                "volumes": (
                    [{"name": "traces", "persistentVolumeClaim": {"claimName": CLAIM}}]
                    if volumes is None
                    else volumes
                ),
            }
        },
        **extra,
    }
    if replicas is not None:
        component["replicas"] = replicas
    return component


def worker(
    name: str = "agg", *, type_: str = "worker", model: str = MODEL, **extra: Any
) -> dict[str, Any]:
    return {
        "name": name,
        "type": type_,
        "replicas": 2,
        "podTemplate": {
            "spec": {
                "containers": [
                    {
                        "name": "main",
                        "command": ["python3", "-m", "dynamo.vllm"],
                        "args": [f"--model={model}", "--tensor-parallel-size=2"],
                    }
                ]
            }
        },
        **extra,
    }


def planner(replicas: int = 1) -> dict[str, Any]:
    return {
        "name": "Planner",
        "type": "planner",
        "replicas": replicas,
        "podTemplate": {"spec": {"containers": [{"name": "main"}]}},
    }


def dgd(*components: dict[str, Any], **spec: Any) -> dict[str, Any]:
    if not components:
        components = (frontend(), worker(), planner())
    return {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": DGD, "namespace": NAMESPACE, "uid": "u-1"},
        "spec": {"components": list(components), **spec},
    }


def planner_config(
    mode: str = "agg", router: Any = "default", **dgdr: Any
) -> PlannerConfig:
    """A validated live retune config (P1-P3, P5 and P13 hold)."""

    return PlannerConfig(
        namespace="test-ns",
        environment="kubernetes",
        mode=mode,
        enable_load_scaling=False,
        enable_throughput_scaling=False,
        retune={
            "interval_seconds": 21600.0,
            "timeout_seconds": 20400.0,
            "router": router,
            "executor": "dgdr",
            "dgdr": {
                "claim_name": CLAIM,
                "model": MODEL,
                "sku": "h200_sxm",
                "engine": copy.deepcopy(ENGINES[mode]),
                **dgdr,
            },
        },
    )


def parse(deployment: dict[str, Any], mode: str = "agg") -> FrontendStartup:
    startup = fc.parse_frontend(deployment, planner_mode=mode)
    assert isinstance(startup, FrontendStartup), startup
    return startup


def parse_error(deployment: dict[str, Any], mode: str = "agg") -> str:
    result = fc.parse_frontend(deployment, planner_mode=mode)
    assert isinstance(result, str), result
    return result


def evaluate(
    deployment: dict[str, Any], *, config: PlannerConfig | None = None
) -> PremiseReport:
    return fc.evaluate_premises(deployment, planner_config=config or planner_config())


def reasons_matching(report: PremiseReport, prefix: str) -> list[str]:
    return [reason for reason in report.reasons if reason.startswith(prefix)]


def with_frontend(**kwargs: Any) -> dict[str, Any]:
    return dgd(frontend(**kwargs), worker(), planner())


# --------------------------------------------------------- find_frontend


def test_find_frontend_v1beta1():
    name, spec = fc.find_frontend(dgd())
    assert name == "Frontend"
    assert spec["type"] == "frontend"


@pytest.mark.parametrize(
    ("components", "match"),
    [
        ((worker(), planner()), "no frontend component"),
        (
            (frontend(), frontend(name="Frontend2"), worker()),
            "2 frontend components",
        ),
        ((worker(frontendSidecar="sidecar-frontend"),), "frontendSidecar of agg"),
        (({"name": "Epp", "type": "epp"}, worker()), "EPP component"),
    ],
)
def test_find_frontend_rejects(components, match):
    with pytest.raises(fc.FrontendParseError, match=match):
        fc.find_frontend(dgd(*components))
    assert parse_error(dgd(*components)).startswith("P7:")


# ----------------------------------------------------- argv and parsing


def test_parse_recipe_frontend():
    startup = parse(dgd())
    assert startup.router_mode == "kv_router"
    assert startup.router_config == {}
    assert startup.startup_maps == {"aggregated": DEFAULT_AGG_MAP}
    assert startup.frontend_component == "Frontend"
    assert startup.replicas == 1
    assert startup.replica_sync is False
    assert json.loads(startup.fingerprint_source) == {
        "router_mode": "kv_router",
        "router_config": {},
        "startup_maps": {"aggregated": DEFAULT_AGG_MAP},
    }
    assert startup.fingerprint_source == fc.canonical_json(
        json.loads(startup.fingerprint_source)
    )


@pytest.mark.parametrize(
    ("command", "args", "mode"),
    [
        # Operator default: python3 -m dynamo.frontend (round-robin).
        (None, None, "round_robin"),
        # Authored args keep the default python3 command.
        (None, KV_ARGS, "kv_router"),
        (("python3",), KV_ARGS, "kv_router"),
        (("python3", "-m", "dynamo.frontend"), ("--router-mode", "kv"), "kv_router"),
        (
            ("/usr/bin/python3", "-u", "-X", "dev", "-m", "dynamo.frontend"),
            ("--router-mode=kv",),
            "kv_router",
        ),
        (("python", "-mdynamo.frontend", "--router-mode", "kv"), None, "kv_router"),
        (("python3",), ("-m", "dynamo.frontend", "--router-mode", "random"), "random"),
        (
            ("python3",),
            ("-m", "dynamo.frontend", "--router-mode", "device-aware-weighted"),
            "device_aware_weighted",
        ),
    ],
)
def test_effective_argv(command, args, mode):
    startup = parse(with_frontend(command=command, args=args))
    assert startup.router_mode == mode


def test_authored_command_drops_default_args():
    # graph.go: an authored frontend command clears the default args, so the
    # command alone must name the module.
    error = parse_error(with_frontend(command=("python3",), args=()))
    assert "does not run python -m dynamo.frontend" in error


@pytest.mark.parametrize(
    "command,args",
    [
        (("/bin/sh", "-c"), ("python3 -m dynamo.frontend --router-mode kv",)),
        (("bash", "-lc", "exec python3 -m dynamo.frontend --router-mode kv"), None),
        (("python3", "-c", "import runpy"), None),
    ],
)
def test_shell_wrapped_command_is_unparseable(command, args):
    error = parse_error(with_frontend(command=command, args=args))
    assert error.startswith("P4:")
    assert "shell-wrapped" in error


@pytest.mark.parametrize(
    "command,args,match",
    [
        (("dynamo-frontend",), ("--router-mode", "kv"), "not a python interpreter"),
        (("python3",), ("-m", "dynamo.vllm"), "does not run python -m dynamo.frontend"),
        (("python3",), ("serve.py",), "does not run python -m dynamo.frontend"),
    ],
)
def test_non_frontend_command_is_unparseable(command, args, match):
    assert match in parse_error(with_frontend(command=command, args=args))


def test_frontend_roles_are_unparseable():
    error = parse_error(dgd(frontend(roles=[{"name": "leader"}]), worker()))
    assert "uses roles" in error


def test_env_sets_router_mode_and_knobs():
    env = trace_env(
        DYN_ROUTER_MODE="kv",
        DYN_ROUTER_TEMPERATURE="0.5",
        DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT="0.25",
    )
    startup = parse(with_frontend(args=("-m", "dynamo.frontend"), env=env))
    assert startup.router_mode == "kv_router"
    assert startup.router_config == {
        "router_temperature": 0.5,
        "overlap_score_credit": 0.25,
    }
    assert startup.startup_maps["aggregated"]["router_temperature"] == 0.5
    assert startup.startup_maps["aggregated"]["overlap_score_credit"] == 0.25


def test_cli_wins_over_env_and_container_env_over_graph_env():
    env = trace_env(DYN_ROUTER_TEMPERATURE="0.5", DYN_ROUTER_PREFILL_LOAD_SCALE="2")
    deployment = dgd(
        frontend(args=(*KV_ARGS, "--router-temperature", "0.2"), env=env),
        worker(),
        env=[
            {"name": "DYN_ROUTER_PREFILL_LOAD_SCALE", "value": "3"},
            {"name": "DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT", "value": "0.5"},
        ],
    )
    assert parse(deployment).router_config == {
        "router_temperature": 0.2,
        "prefill_load_scale": 2.0,
        "overlap_score_credit": 0.5,
    }


def test_planner_process_env_is_ignored(monkeypatch):
    monkeypatch.setenv("DYN_ROUTER_MODE", "kv")
    monkeypatch.setenv("DYN_ROUTER_TEMPERATURE", "0.9")
    monkeypatch.setenv("DYN_ROUTER_REPLICA_SYNC", "1")
    before = dict(os.environ)
    startup = parse(with_frontend(args=("-m", "dynamo.frontend")))
    assert startup.router_mode == "round_robin"
    assert startup.router_config == {}
    assert startup.replica_sync is False
    assert dict(os.environ) == before


def test_abbreviated_flag_matches_frontend_parser():
    # The frontend parses with argparse abbreviations enabled.
    startup = parse(with_frontend(args=(*KV_ARGS, "--router-temp", "0.4")))
    assert startup.router_config == {"router_temperature": 0.4}


def test_unknown_frontend_args_are_ignored():
    args = (
        *KV_ARGS,
        "--http-port",
        "8000",
        "--kv-cache-block-size",
        "64",
        "--trust-remote-code",
        "--some-vllm-processor-flag",
        "x",
    )
    assert parse(with_frontend(args=args)).router_config == {}


def test_legacy_and_removed_flags_parse_quietly(caplog):
    args = (
        *KV_ARGS,
        "--no-kv-events",
        "--admission-control",
        "none",
        "--enforce-disagg",
    )
    startup = parse(with_frontend(args=args))
    assert startup.router_config == {"use_kv_events": False}
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_replica_sync_and_disagg_maps():
    startup = parse(
        with_frontend(
            args=(*KV_ARGS, "--router-replica-sync", "--router-temperature", "0.3")
        ),
        mode="disagg",
    )
    assert startup.replica_sync is True
    assert startup.router_config == {
        "router_replica_sync": True,
        "router_temperature": 0.3,
    }
    assert set(startup.startup_maps) == {"prefill", "decode"}
    prefill, decode = startup.startup_maps["prefill"], startup.startup_maps["decode"]
    assert prefill["overlap_score_credit"] == 1.0
    assert prefill["host_cache_hit_weight"] == 0.75
    # Plain decode routers report zero cache credits.
    assert decode["overlap_score_credit"] == 0.0
    assert decode["host_cache_hit_weight"] == 0.0
    assert decode["disk_cache_hit_weight"] == 0.0
    assert decode["router_temperature"] == 0.3


def test_zero_credit_links_cache_weights():
    startup = parse(
        with_frontend(args=(*KV_ARGS, "--router-kv-overlap-score-credit", "0"))
    )
    agg = startup.startup_maps["aggregated"]
    assert agg["overlap_score_credit"] == 0.0
    assert agg["host_cache_hit_weight"] == 0.0
    assert agg["disk_cache_hit_weight"] == 0.0


def test_load_aware_implies_kv_mode():
    startup = parse(with_frontend(args=("-m", "dynamo.frontend", "--load-aware")))
    assert startup.router_mode == "kv_router"
    assert startup.router_config["use_kv_events"] is False
    assert startup.router_config["overlap_score_credit"] == 0.0


def test_fingerprint_ignores_replicas_but_not_router_config():
    base = parse(dgd(frontend(replicas=1), worker()))
    scaled = parse(dgd(frontend(replicas=3), worker()))
    tuned = parse(
        dgd(frontend(args=(*KV_ARGS, "--router-temperature", "0.1")), worker())
    )
    assert scaled.replicas == 3
    assert base.fingerprint_source == scaled.fingerprint_source
    assert base.fingerprint_source != tuned.fingerprint_source


def test_omitted_replicas_default_to_one():
    assert parse(dgd(frontend(replicas=None), worker())).replicas == 1


# -------------------------------------------- env references and sources


def test_arg_expands_literal_env():
    env = trace_env(TEMP="0.3")
    args = (*KV_ARGS, "--router-temperature", "$(TEMP)")
    assert parse(with_frontend(args=args, env=env)).router_config == {
        "router_temperature": 0.3
    }


def test_escaped_reference_stays_literal():
    # $$(TEMP) is the literal "$(TEMP)", which is not a float.
    env = trace_env(TEMP="0.3")
    args = (*KV_ARGS, "--router-temperature", "$$(TEMP)")
    error = parse_error(with_frontend(args=args, env=env))
    assert "cannot parse the frontend args" in error


def test_arg_referencing_value_from_is_unparseable():
    env = trace_env(TEMP={"valueFrom": {"configMapKeyRef": {"name": "c", "key": "t"}}})
    args = (*KV_ARGS, "--router-temperature", "$(TEMP)")
    error = parse_error(with_frontend(args=args, env=env))
    assert "$(TEMP)" in error and "cannot read" in error


def test_opaque_string_router_arg_is_unparseable():
    env = trace_env(MODE={"valueFrom": {"secretKeyRef": {"name": "s", "key": "k"}}})
    args = ("-m", "dynamo.frontend", "--router-tracking-key-id", "id-$(MODE)")
    error = parse_error(with_frontend(args=args, env=env))
    assert "router_tracking_key_id" in error


def test_arg_that_is_an_opaque_reference_is_unparseable():
    env = trace_env(EXTRA={"valueFrom": {"secretKeyRef": {"name": "s", "key": "k"}}})
    error = parse_error(with_frontend(args=(*KV_ARGS, "$(EXTRA)"), env=env))
    assert "$(EXTRA)" in error


def test_opaque_reference_in_a_non_router_arg_is_fine():
    args = (*KV_ARGS, "--metrics-prefix", "fe-$(POD_NAME)")
    assert parse(with_frontend(args=args)).router_config == {}


OPAQUE_PORT = {"valueFrom": {"configMapKeyRef": {"name": "c", "key": "port"}}}


@pytest.mark.parametrize(
    "args",
    [
        # Typed non-router flags keep the unreadable value unconverted.
        (*KV_ARGS, "--http-port", "$(HTTP_PORT)"),
        (*KV_ARGS, "--http-port=$(HTTP_PORT)"),
        (*KV_ARGS, "--grpc-metrics-port", "$(POD_UID)"),
        # A non-router choices flag.
        (*KV_ARGS, "--dyn-chat-processor", "$(HTTP_PORT)"),
        # The value of a flag the frontend hands to its chat processor.
        (
            *KV_ARGS,
            "--dyn-chat-processor",
            "vllm",
            "--tool-call-parser",
            "$(HTTP_PORT)",
        ),
        (*KV_ARGS, "--dyn-chat-processor", "vllm", "--tool-call-parser=$(HTTP_PORT)"),
        # Attached to an option that takes no value, it is the option's value.
        (
            *KV_ARGS,
            "--dyn-chat-processor",
            "vllm",
            "--enable-auto-tool-choice",
            "--some-vllm-opt=$(HTTP_PORT)",
        ),
    ],
)
def test_opaque_reference_in_a_typed_non_router_arg_is_fine(args):
    startup = parse(with_frontend(args=args, env=trace_env(HTTP_PORT=OPAQUE_PORT)))
    assert startup.router_mode == "kv_router"
    assert startup.router_config == {}


@pytest.mark.parametrize(
    ("args", "match"),
    [
        # Router flags whose value is unreadable.
        (
            ("-m", "dynamo.frontend", "--router-mode", "$(HTTP_PORT)"),
            "router arg references $(HTTP_PORT)",
        ),
        (
            (*KV_ARGS, "--router-temp", "$(HTTP_PORT)"),
            "router arg references $(HTTP_PORT)",
        ),
        (
            (*KV_ARGS, "--router-ttl-secs=$(HTTP_PORT)"),
            "router arg references $(HTTP_PORT)",
        ),
        # Left over where a flag could stand.
        (
            (*KV_ARGS, "--router-kv-events", "$(HTTP_PORT)"),
            "could set a router flag",
        ),
        ((*KV_ARGS, "--$(HTTP_PORT)"), "could set a router flag"),
        ((*KV_ARGS, "--$(HTTP_PORT)=1"), "could set a router flag"),
        (
            (*KV_ARGS, "--tool-call-parser", "--$(HTTP_PORT)", "$(HTTP_PORT)"),
            "could set a router flag",
        ),
        # The chat processor's flag takes no value, so the frontend would
        # parse what follows it as its own flag.
        (
            (
                *KV_ARGS,
                "--dyn-chat-processor",
                "vllm",
                "--enable-auto-tool-choice",
                "$(HTTP_PORT)",
            ),
            "could set a router flag",
        ),
        (
            (
                *KV_ARGS,
                "--dyn-chat-processor",
                "vllm",
                "--tool-call-parser",
                "hermes",
                "--enable-auto-tool-choice",
                "--trust-remote-code",
                "$(HTTP_PORT)",
            ),
            "could set a router flag",
        ),
        # Inert env is still unreadable as a router value or an option name.
        ((*KV_ARGS, "--router-temp", "$(POD_UID)"), "router arg references $(POD_UID)"),
        ((*KV_ARGS, "--$(POD_NAME)"), "could set a router flag"),
    ],
)
def test_opaque_reference_that_could_set_a_router_flag_is_unparseable(args, match):
    error = parse_error(with_frontend(args=args, env=trace_env(HTTP_PORT=OPAQUE_PORT)))
    assert error.startswith("P4:")
    assert match in error


@pytest.mark.parametrize(
    "args",
    [
        (*KV_ARGS, "$(POD_NAME)"),
        (*KV_ARGS, "--router-kv-events", "$(POD_UID)"),
        (*KV_ARGS, "$(NODE)"),
    ],
)
def test_inert_reference_where_a_flag_could_stand_is_fine(args):
    # A pod name, uid or node name never starts with "-".
    env = trace_env(NODE={"valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}})
    assert parse(with_frontend(args=args, env=env)).router_config == {}


@pytest.mark.parametrize(
    ("args", "prefix", "match"),
    [
        ((*KV_ARGS, "$(ROUTER_ARG)"), None, "could set a router flag"),
        (
            (*KV_ARGS, "--router-temperature", "$(T)"),
            None,
            "router arg references $(T)",
        ),
        ((*KV_ARGS, "$(APP_ROUTER_ARG)"), "APP_", "could set a router flag"),
        # A prefix that cannot produce the name, and command substitution.
        ((*KV_ARGS, "$(ROUTER_ARG)"), "APP_", None),
        ((*KV_ARGS, "--metrics-prefix", "$(seq 1 3)"), None, None),
    ],
)
def test_env_from_may_define_an_undefined_reference(args, prefix, match):
    deployment = with_frontend(args=args)
    source: dict[str, Any] = {"configMapRef": {"name": "router-options"}}
    if prefix is not None:
        source["prefix"] = prefix
    deployment["spec"]["components"][0]["podTemplate"]["spec"]["containers"][0][
        "envFrom"
    ] = [source]
    result = fc.parse_frontend(deployment, planner_mode="agg")
    if match is None:
        assert isinstance(result, FrontendStartup), result
    else:
        assert isinstance(result, str) and match in result, result


def test_router_env_from_value_from_is_unparseable():
    env = trace_env(
        DYN_ROUTER_TEMPERATURE={
            "valueFrom": {"configMapKeyRef": {"name": "c", "key": "t"}}
        }
    )
    error = parse_error(with_frontend(env=env))
    assert "DYN_ROUTER_TEMPERATURE" in error and "valueFrom" in error


def test_router_env_referencing_value_from_is_unparseable():
    env = trace_env(
        A_TEMP={"valueFrom": {"configMapKeyRef": {"name": "c", "key": "t"}}},
        DYN_ROUTER_TEMPERATURE="$(A_TEMP)",
    )
    assert "DYN_ROUTER_TEMPERATURE" in parse_error(with_frontend(env=env))


def test_router_env_references_earlier_literal():
    env = trace_env(A_TEMP="0.7", DYN_ROUTER_TEMPERATURE="$(A_TEMP)")
    assert parse(with_frontend(env=env)).router_config == {"router_temperature": 0.7}


def test_non_router_value_from_is_fine():
    env = trace_env(
        HF_TOKEN={"valueFrom": {"secretKeyRef": {"name": "hf", "key": "t"}}}
    )
    assert parse(with_frontend(env=env)).router_mode == "kv_router"


@pytest.mark.parametrize(
    ("source", "warned"),
    [
        ({"secretRef": {"name": "hf-token-secret"}}, True),
        ({"configMapRef": {"name": "router"}, "prefix": "DYN_"}, True),
        ({"configMapRef": {"name": "router"}, "prefix": "D"}, True),
        ({"secretRef": {"name": "hf-token-secret"}, "prefix": "HF_"}, False),
    ],
)
def test_env_from_is_parsed_and_warned_when_it_may_set_router_env(source, warned):
    deployment = with_frontend()
    deployment["spec"]["components"][0]["podTemplate"]["spec"]["containers"][0][
        "envFrom"
    ] = [source]
    assert isinstance(
        fc.parse_frontend(deployment, planner_mode="agg"), FrontendStartup
    )
    report = evaluate(deployment)
    envfrom_warnings = [w for w in report.warnings if "envFrom" in w]
    assert bool(envfrom_warnings) == warned
    assert not reasons_matching(report, "P4")


def test_router_policy_config_is_unparseable():
    error = parse_error(
        with_frontend(args=(*KV_ARGS, "--router-policy-config", "/etc/policy.yaml"))
    )
    assert "--router-policy-config" in error


@pytest.mark.parametrize(
    "args,env,match",
    [
        ((*KV_ARGS, "--router-kv-overlap-score-credit", "-1"), {}, "invalid"),
        ((*KV_ARGS, "--router-temperature", "hot"), {}, "cannot parse"),
        (
            ("-m", "dynamo.frontend", "--router-mode", "power-of-two"),
            {},
            "cannot parse",
        ),
        ((*KV_ARGS,), {"DYN_ROUTER_TEMPERATURE": "hot"}, "invalid frontend env"),
        (
            (*KV_ARGS, "--router-approximate-cache-policy", "lru"),
            {},
            "invalid",
        ),
    ],
)
def test_invalid_router_config_is_unparseable(args, env, match):
    error = parse_error(with_frontend(args=args, env=trace_env(**env)))
    assert error.startswith("P4:")
    assert match in error


# ------------------------------------------------------ parser fidelity


GOLDEN_CASES = [
    ((*KV_ARGS,), {}),
    ((*KV_ARGS, "--router-temperature", "0.5", "--router-replica-sync"), {}),
    (
        ("-m", "dynamo.frontend"),
        {"DYN_ROUTER_MODE": "kv", "DYN_ROUTER_USE_KV_EVENTS": "0"},
    ),
    ((*KV_ARGS, "--no-kv-events", "--router-ttl-secs", "30"), {}),
    (
        (*KV_ARGS, "--router-kv-overlap-score-credit", "0"),
        {"DYN_ROUTER_TEMPERATURE": "0.1"},
    ),
    (("-m", "dynamo.frontend", "--load-aware"), {}),
    ((*KV_ARGS,), {"DYN_OVERLAP_SCORE_WEIGHT": "2"}),
    (
        (*KV_ARGS,),
        {"DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT": "0", "DYN_OVERLAP_SCORE_WEIGHT": "2"},
    ),
    ((*KV_ARGS, "--router-temp", "0.25", "--router-event-threads", "8"), {}),
    (
        (*KV_ARGS, "--router-tracking-hash", "keyed-xxh3-v1"),
        {"DYN_ROUTER_TRACKING_KEY_FILE": "/k", "DYN_ROUTER_TRACKING_KEY_ID": "e1"},
    ),
]


@pytest.mark.parametrize(("args", "env"), GOLDEN_CASES)
def test_matches_real_frontend_parser(monkeypatch, args, env):
    """The frontend's own parser, run with its env in os.environ, agrees."""

    for name in list(os.environ):
        if name.startswith("DYN_"):
            monkeypatch.delenv(name)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parser = argparse.ArgumentParser()
        FrontendArgGroup().add_arguments(parser)
        namespace, _ = parser.parse_known_args(list(args[2:]))
    config = FrontendConfig.from_cli_args(namespace)
    if config.load_aware:
        config.router_mode = "kv"
    config.apply_load_aware_preset()
    config.apply_conditional_disagg_config()
    expected = json.loads(
        KvRouterConfig(**{f: getattr(config, f) for f in _KV_ROUTER_FIELDS}).to_json()
    )

    for name in env:
        monkeypatch.delenv(name)
    startup = parse(with_frontend(args=args, env=trace_env(**env)))
    actual = json.loads(
        KvRouterConfig.from_json(json.dumps(startup.router_config)).to_json()
    )
    assert actual == expected
    assert startup.router_mode == fc._router_mode(config.router_mode)


def test_router_fields_are_all_covered():
    """Every frontend router field is in the JSON or checked on the flags."""

    serialized = set(
        json.loads(
            KvRouterConfig(
                router_tracking_hash="keyed-xxh3-v1",
                router_tracking_key_file="/k",
                router_tracking_key_id="e",
                decode_active_request_weight=1.0,
                enable_session_prefix_index=True,
                shared_cache_multiplier=0.5,
                conditional_disagg_enabled=True,
                conditional_disagg_policy="isl_or_load",
                conditional_disagg_eff_isl_threshold=1,
                conditional_disagg_eff_isl_ratio_threshold=0.5,
                conditional_disagg_prefill_busy_threshold=0.5,
                conditional_disagg_decode_busy_threshold=0.5,
            ).to_json()
        )
    )
    covered = serialized | set(fc._SERDE_SKIPPED_DEFAULTS) | {"overlap_score_weight"}
    assert set(_KV_ROUTER_FIELDS) <= covered


# --------------------------------------------------------- P4 premises


def test_recipe_dgd_satisfies_every_premise():
    report = evaluate(dgd())
    assert report.p4_ok and report.spec_ok, report.reasons
    assert report.reasons == ()
    assert report.startup is not None
    assert not [w for w in report.warnings if w.startswith("P6")]


def test_parse_error_fails_p4_only():
    deployment = with_frontend(
        command=("/bin/sh", "-c"), args=("python3 -m dynamo.frontend",)
    )
    report = evaluate(deployment)
    assert not report.p4_ok and not report.spec_ok
    assert len(report.reasons) == 1 and "shell-wrapped" in report.reasons[0]
    assert report.startup is None


@pytest.mark.parametrize(
    ("args", "env", "match"),
    [
        (("-m", "dynamo.frontend"), {}, "routes with 'round_robin'"),
        (
            ("-m", "dynamo.frontend", "--router-mode", "random"),
            {},
            "routes with 'random'",
        ),
        ((*KV_ARGS, "--no-router-kv-events"), {}, "use_kv_events=False"),
        (
            (*KV_ARGS, "--router-host-cache-hit-weight", "0.5"),
            {},
            "host_cache_hit_weight=0.5",
        ),
        (
            (*KV_ARGS, "--router-queue-threshold", "0.5"),
            {},
            "router_queue_threshold=0.5",
        ),
        (
            (*KV_ARGS, "--router-kv-overlap-score-credit-decay", "1"),
            {},
            "overlap_score_credit_decay",
        ),
        (
            (*KV_ARGS, "--router-decode-active-request-weight", "1"),
            {},
            "decode_active_request_weight",
        ),
        (
            (*KV_ARGS, "--router-prefill-load-model", "ais"),
            {},
            "router_prefill_load_model is 'aic'",
        ),
        (
            (*KV_ARGS, "--router-conditional-disagg"),
            {},
            "conditional disaggregation",
        ),
        (
            (
                *KV_ARGS,
                "--no-router-kv-events",
                "--router-approximate-cache-policy",
                "lru",
            ),
            {},
            "router_approximate_cache_policy='lru'",
        ),
        ((*KV_ARGS, "--router-prefill-policy", "default"), {}, "router_prefill_policy"),
        ((*KV_ARGS, "--router-decode-policy", "default"), {}, "router_decode_policy"),
        ((*KV_ARGS,), {"DYN_OVERLAP_SCORE_WEIGHT": "2"}, "overlap_score_weight"),
        (
            ("-m", "dynamo.frontend"),
            {"DYN_ROUTER_LOAD_AWARE": "1"},
            "load-aware preset",
        ),
        (
            (*KV_ARGS,),
            {"DYN_ROUTER_WORKER_SELECTION_POLICY": "mine"},
            "worker-selection",
        ),
        (
            (*KV_ARGS,),
            {"DYN_ROUTER_POSITIONAL_SEARCH_MODE": "x"},
            "DYN_ROUTER_POSITIONAL_SEARCH_MODE",
        ),
        (
            (*KV_ARGS,),
            {"DYN_ROUTER_SESSION_AFFINITY_TTL_SECS": "60"},
            "session_affinity_ttl_secs=60",
        ),
        (
            (*KV_ARGS, "--active-decode-blocks-threshold", "0.9"),
            {},
            "active_decode_blocks_threshold",
        ),
        (
            (*KV_ARGS, "--active-prefill-tokens-threshold", "100"),
            {},
            "active_prefill_tokens_threshold",
        ),
    ],
)
def test_p4_rejects(args, env, match):
    report = evaluate(with_frontend(args=args, env=trace_env(**env)))
    assert not report.p4_ok and not report.spec_ok
    assert any(
        match in reason for reason in reasons_matching(report, "P4:")
    ), report.reasons


@pytest.mark.parametrize(
    ("args", "env"),
    [
        # Searched knobs may start anywhere.
        (
            (
                *KV_ARGS,
                "--router-temperature",
                "0.5",
                "--router-prefill-load-scale",
                "2",
            ),
            {},
        ),
        ((*KV_ARGS, "--router-kv-overlap-score-credit", "0.5"), {}),
        # Tolerated fields.
        ((*KV_ARGS, "--router-event-threads", "8", "--router-replica-sync"), {}),
        ((*KV_ARGS, "--enable-session-prefix-index", "--serve-indexer"), {}),
        (
            (*KV_ARGS, "--router-tracking-hash", "keyed-xxh3-v1"),
            {"DYN_ROUTER_TRACKING_KEY_FILE": "/k", "DYN_ROUTER_TRACKING_KEY_ID": "e"},
        ),
        # Frontend-only and transport settings.
        (
            (*KV_ARGS,),
            {
                "DYN_ROUTER_MIN_INITIAL_WORKERS": "2",
                "DYN_ROUTER_ZMQ_ENDPOINTS_PER_SUB": "4",
            },
        ),
        ((*KV_ARGS, "--router-session-affinity-mode", "soft"), {}),
    ],
)
def test_p4_accepts(args, env):
    report = evaluate(with_frontend(args=args, env=trace_env(**env)))
    assert report.p4_ok, report.reasons


def test_p4_unsearched_knob_when_router_domain_is_pinned():
    # A scalar still searches (pins) the knob: it stays in the domain.
    config = planner_config(
        router={
            "overlap_score_credit": 1.0,
            "prefill_load_scale": 1.0,
            "temperature": 0.0,
        }
    )
    report = evaluate(
        with_frontend(args=(*KV_ARGS, "--router-temperature", "0.5")), config=config
    )
    assert report.p4_ok, report.reasons


def test_p4_zero_startup_credit_rejects_positive_candidates():
    zero = (*KV_ARGS, "--router-kv-overlap-score-credit", "0")
    # The default domain collapses the omitted credit to [0.0].
    assert evaluate(with_frontend(args=zero)).p4_ok
    config = planner_config(
        router={"overlap_score_credit": {"choices": [0.0, 0.5]}, "temperature": 0.0}
    )
    report = evaluate(with_frontend(args=zero), config=config)
    assert not report.p4_ok
    assert "overlap_score_credit > 0" in report.reasons[0]


def test_p4_disagg_decode_credit_zero_is_not_a_zero_credit_startup():
    # Plain decode routers always report credit 0; only prefill counts.
    config = planner_config(mode="disagg")
    report = evaluate(
        dgd(
            frontend(),
            worker("prefill", type_="prefill"),
            worker("decode", type_="decode"),
            planner(),
        ),
        config=config,
    )
    assert report.p4_ok, report.reasons
    assert report.startup is not None
    assert set(report.startup.startup_maps) == {"prefill", "decode"}


# ------------------------------------------- P4: worker router advertisement


SECRET = {"valueFrom": {"secretKeyRef": {"name": "s", "key": "k"}}}
# A Secret that may hold any key.
ENV_FROM = [{"secretRef": {"name": "worker-launch-options"}}]
# A worker launch in a shell script.
WORKER = f"python3 -m dynamo.vllm --model {MODEL}"


def router_worker(
    *args: str,
    env: list[dict[str, Any]] | None = None,
    env_from: list[dict[str, Any]] | None = None,
    name: str = "agg",
    type_: str = "worker",
) -> dict[str, Any]:
    component = worker(name, type_=type_)
    main = component["podTemplate"]["spec"]["containers"][0]
    main["args"] = [*main["args"], *args]
    if env is not None:
        main["env"] = env
    if env_from is not None:
        main["envFrom"] = env_from
    return component


def shell_worker(
    script: str,
    *,
    command: tuple[str, ...] | None = ("/bin/sh", "-c"),
    env: list[dict[str, Any]] | None = None,
    env_from: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """A worker running ``script``; ``command=None`` omits the command."""

    component = worker()
    main = component["podTemplate"]["spec"]["containers"][0]
    main["args"] = [script]
    if command is None:
        del main["command"]
    else:
        main["command"] = list(command)
    if env is not None:
        main["env"] = env
    if env_from is not None:
        main["envFrom"] = env_from
    return component


def field_ref_env(name: str, path: str) -> dict[str, Any]:
    return {"name": name, "valueFrom": {"fieldRef": {"fieldPath": path}}}


def router_arg_env() -> list[dict[str, Any]]:
    """ROUTER_ARG from a ConfigMap the planner cannot read."""

    return [
        {
            "name": "ROUTER_ARG",
            "valueFrom": {
                "configMapKeyRef": {
                    "name": "worker-launch-options",
                    "key": "router-arg",
                }
            },
        }
    ]


def minimax_prefill_worker(*router_args: str) -> dict[str, Any]:
    """The PrefillWorker of recipes/minimax-m3/vllm/disagg-gb200-agentic."""

    return {
        "name": "PrefillWorker",
        "type": "prefill",
        "replicas": 3,
        "podTemplate": {
            "spec": {
                "containers": [
                    {
                        "name": "main",
                        "command": ["python3"],
                        "args": [
                            "-m",
                            "dynamo.vllm",
                            *router_args,
                            "--model",
                            "$(MODEL_NAME)",
                            "--served-model-name",
                            "$(SERVED_MODEL_NAME)",
                            "--tensor-parallel-size",
                            "4",
                            "--speculative-config",
                            "$(SPECULATIVE_CONFIG)",
                            "--disaggregation-mode",
                            "prefill",
                            "--kv-transfer-config",
                            "$(KV_TRANSFER_CONFIG)",
                            "--kv-events-config",
                            '{"enable_kv_cache_events":true,"publisher":"zmq"}',
                        ],
                        "env": [
                            {
                                "name": "KV_TRANSFER_CONFIG",
                                "value": '{"kv_connector":"NixlConnector",'
                                '"kv_role":"kv_both"}',
                            },
                            {
                                "name": "SPECULATIVE_CONFIG",
                                "valueFrom": {
                                    "configMapKeyRef": {
                                        "name": "minimax-m3-config",
                                        "key": "speculative-config",
                                    }
                                },
                            },
                            {"name": "MODEL_NAME", "value": "nvidia/MiniMax-M3-NVFP4"},
                            {
                                "name": "SERVED_MODEL_NAME",
                                "value": "nvidia/MiniMax-M3-NVFP4",
                            },
                        ],
                    }
                ]
            }
        },
    }


@pytest.mark.parametrize(
    ("args", "env", "match"),
    [
        (("--router-mode", "kv"), None, "advertises its own router configuration"),
        (("--router-mode=round-robin",), None, "router mode 'round-robin'"),
        (("--router-m", "kv", "--no-router-kv-events"), None, "router mode 'kv'"),
        ((), [{"name": "DYN_ROUTER_MODE", "value": "kv"}], "router mode 'kv'"),
        ((), [{"name": "DYN_ROUTER_MODE", "value": ""}], "router mode ''"),
        (
            (),
            [{"name": "DYN_ROUTER_MODE", **SECRET}],
            "DYN_ROUTER_MODE through valueFrom",
        ),
        (
            (),
            [
                {"name": "A_MODE", **SECRET},
                {"name": "DYN_ROUTER_MODE", "value": "$(A_MODE)"},
            ],
            "DYN_ROUTER_MODE through valueFrom",
        ),
        (
            ("--router-mode", "$(A_MODE)"),
            [{"name": "A_MODE", **SECRET}],
            "router flag that references $(A_MODE)",
        ),
        (
            ("--router-mode", "$(A_MODE)"),
            [{"name": "A_MODE", "value": "kv"}],
            "router mode 'kv'",
        ),
        (("--router-mode", "fastest"), None, "cannot parse"),
        (("--router-mode", "power-of-two"), None, "cannot parse"),
        (
            (),
            [{"name": "DYN_ROUTER_TEMPERATURE", "value": "hot"}],
            "invalid router env",
        ),
        # An unreadable argument where an option could stand, e.g. a
        # ConfigMap value "--router-mode=kv".
        (
            ("--no-router-kv-events", "$(ROUTER_ARG)"),
            router_arg_env(),
            "has an argument that references $(ROUTER_ARG)",
        ),
        (
            ("$(ROUTER_ARG)",),
            router_arg_env(),
            "has an argument that references $(ROUTER_ARG)",
        ),
        (
            ("--$(A_MODE)=kv",),
            [{"name": "A_MODE", **SECRET}],
            "has an option whose name references $(A_MODE)",
        ),
        (
            ("--router-$(A_MODE)", "kv"),
            [{"name": "A_MODE", **SECRET}],
            "has an option whose name references $(A_MODE)",
        ),
        # Inert env could still spell an option's name.
        (
            ("--x-$(MY_POD)=1",),
            [field_ref_env("MY_POD", "metadata.name")],
            "has an option whose name references $(MY_POD)",
        ),
        (
            ("--speculative-config", "$(A_SECRET)", "$(A_SECRET)"),
            [{"name": "A_SECRET", **SECRET}],
            "has an argument that references $(A_SECRET)",
        ),
        # Only the first value of --served-model-name (nargs "+") is exempt:
        # a later one starting with "-" is parsed as a flag.
        (
            ("--served-model-name", "x", "$(ROUTER_ARG)"),
            router_arg_env(),
            "has an argument that references $(ROUTER_ARG)",
        ),
        # After an engine flag that takes no value or optional values, the
        # worker would read "--router-mode=kv" as a flag.
        (
            ("--enforce-eager", "$(ROUTER_ARG)"),
            router_arg_env(),
            "has an argument that references $(ROUTER_ARG)",
        ),
        (
            ("--connector", "$(ROUTER_ARG)"),
            router_arg_env(),
            "has an argument that references $(ROUTER_ARG)",
        ),
        (
            ("$(NOTE)",),
            [field_ref_env("NOTE", "metadata.annotations['router']")],
            "has an argument that references $(NOTE)",
        ),
    ],
)
def test_p4_rejects_a_worker_that_advertises_a_router_config(args, env, match):
    report = evaluate(dgd(frontend(), router_worker(*args, env=env), planner()))
    assert not report.p4_ok and not report.spec_ok
    assert report.startup is not None  # the frontend itself parses
    rejected = reasons_matching(report, "P4: worker component agg ")
    assert any(match in reason for reason in rejected), report.reasons
    assert all("on the frontend only" in reason for reason in rejected)


@pytest.mark.parametrize(
    ("args", "env"),
    [
        # Router flags and env without a mode advertise nothing.
        (("--no-router-kv-events", "--router-temperature", "0.5"), None),
        ((), [{"name": "DYN_ROUTER_TEMPERATURE", "value": "0.5"}]),
        ((), [{"name": "DYN_ROUTER_MODE_HINT", "value": "kv"}]),
        # Unreadable env as the value of a non-mode router flag, of an
        # option known to take one, or attached to its option.
        (("--router-temperature", "$(A_SECRET)"), [{"name": "A_SECRET", **SECRET}]),
        (("--speculative-config", "$(A_SECRET)"), [{"name": "A_SECRET", **SECRET}]),
        (("--speculative-config", "$(ROUTER_ARG)"), router_arg_env()),
        (("--model", "$(ROUTER_ARG)"), router_arg_env()),
        (("--model-path", "$(ROUTER_ARG)"), router_arg_env()),
        (("--served-model-name", "$(ROUTER_ARG)"), router_arg_env()),
        (("--some-opt=$(ROUTER_ARG)",), router_arg_env()),
        # Inert env never stands for an option.
        (("$(POD_NAME)",), None),
        (("--enforce-eager", "$(MY_POD)"), [field_ref_env("MY_POD", "metadata.name")]),
    ],
)
def test_p4_accepts_workers_that_advertise_nothing(args, env):
    report = evaluate(dgd(frontend(), router_worker(*args, env=env), planner()))
    assert report.p4_ok and report.spec_ok, report.reasons


@pytest.mark.parametrize(
    ("args", "env_from", "match"),
    [
        # A Secret or ConfigMap key may define what env leaves undefined.
        (
            ("$(ROUTER_ARG)",),
            ENV_FROM,
            "references $(ROUTER_ARG), which the planner cannot read (the "
            "container's env leaves it to an envFrom source)",
        ),
        (
            ("$(APP_ROUTER_ARG)",),
            [{**ENV_FROM[0], "prefix": "APP_"}],
            "references $(APP_ROUTER_ARG)",
        ),
        # A prefix that cannot produce the name, the worker's own env, and
        # the value of an option known to take one.
        (("$(ROUTER_ARG)",), [{**ENV_FROM[0], "prefix": "APP_"}], None),
        (("--model", "$(MODEL_ID)"), ENV_FROM, None),
        (("--model", "$(MODEL_NAME)"), ENV_FROM, None),
        # The operator sets these on the container, whose env wins over
        # envFrom, and none of them holds an option.
        (("--draft-endpoint", "$(DYN_NAMESPACE)/backend/generate"), ENV_FROM, None),
        (("--x", "$(DYN_COMPONENT)", "$(DYN_SYSTEM_PORT)"), ENV_FROM, None),
        # The operator sets this one only for some discovery configurations.
        (("$(NATS_SERVER)",), ENV_FROM, "references $(NATS_SERVER)"),
    ],
)
def test_p4_worker_reference_an_env_from_source_may_define(args, env_from, match):
    component = router_worker(
        *args, env=[{"name": "MODEL_ID", "value": MODEL}], env_from=env_from
    )
    report = evaluate(dgd(frontend(), component, planner()))
    rejected = reasons_matching(report, "P4: worker component agg ")
    if match is None:
        assert report.p4_ok, report.reasons
    else:
        assert any(match in reason for reason in rejected), report.reasons


@pytest.mark.parametrize(
    ("command", "script", "match"),
    [
        (("/bin/sh", "-c"), "$(ROUTER_ARG)", "references $(ROUTER_ARG)"),
        (None, "$(ROUTER_ARG)", "references $(ROUTER_ARG)"),
        # Image env too, which a shell splits unless quoted; a script's own
        # assignment overrides what such a source may define.
        (
            ("/bin/sh", "-c"),
            f"export LD_LIBRARY_PATH=/opt/lib:$LD_LIBRARY_PATH; exec {WORKER}",
            "references $LD_LIBRARY_PATH where it could become an option, and "
            "the planner cannot read it (the container's env leaves it to an "
            "envFrom source)",
        ),
        (
            ("/bin/sh", "-c"),
            f'export LD_LIBRARY_PATH="/opt/lib:$LD_LIBRARY_PATH"; exec {WORKER}',
            None,
        ),
        (("/bin/sh", "-c"), f"ARGS=; exec {WORKER} $ARGS", None),
        # A redirection target is a file, never an argument.
        (("/bin/sh", "-c"), f'exec {WORKER} > "$HOME/worker.log" 2>&1', None),
        # Operator env the container's env defines.
        (None, f"echo $DYN_NAMESPACE $DYN_COMPONENT; exec {WORKER}", None),
        (None, f"exec {WORKER} --endpoint dyn://$DYN_NAMESPACE.backend.generate", None),
        (None, f"echo $DYN_SYSTEM_PORT; exec {WORKER}", None),
    ],
)
def test_p4_worker_script_reference_an_env_from_source_may_define(
    command, script, match
):
    component = shell_worker(
        script,
        command=command,
        env=[{"name": "MODEL_ID", "value": MODEL}],
        env_from=ENV_FROM,
    )
    report = evaluate(dgd(frontend(), component, planner()))
    rejected = reasons_matching(report, "P4: worker component agg ")
    if match is None:
        assert report.p4_ok, report.reasons
    else:
        assert any(match in reason for reason in rejected), report.reasons


@pytest.mark.parametrize(
    "env_from", [ENV_FROM, [{**ENV_FROM[0], "prefix": "APP_"}]], ids=["bare", "APP_"]
)
def test_p4_names_the_env_a_dangerous_script_variable_holds(env_from):
    component = shell_worker(
        f"X=$ROUTER_ARG; exec {WORKER} $X", env=router_arg_env(), env_from=env_from
    )
    report = evaluate(dgd(frontend(), component, planner()))
    (reason,) = reasons_matching(report, "P4: worker component agg ")
    assert "references $X (assigned from $ROUTER_ARG) where" in reason
    assert "envFrom source" not in reason


@pytest.mark.parametrize(
    ("env", "match"),
    [
        (None, None),
        # User env of an operator name overrides the operator's.
        (
            [
                {
                    "name": "DYN_NAMESPACE",
                    "valueFrom": {"configMapKeyRef": {"name": "c", "key": "ns"}},
                }
            ],
            "references $(DYN_NAMESPACE)",
        ),
    ],
)
def test_p4_user_env_overrides_operator_env(env, match):
    component = router_worker(
        "--draft-endpoint", "$(DYN_NAMESPACE)/backend/generate", env=env
    )
    report = evaluate(dgd(frontend(), component, planner()))
    rejected = reasons_matching(report, "P4: worker component agg ")
    if match is None:
        assert report.p4_ok, report.reasons
    else:
        assert any(match in reason for reason in rejected), report.reasons


def test_p4_untyped_component_gets_no_worker_operator_env():
    # ComponentDefaultsFactory gives an untyped component the common base
    # container only, so an envFrom source may define DYN_SYSTEM_PORT.
    component = router_worker("$(DYN_SYSTEM_PORT)", env_from=ENV_FROM, type_="")
    report = evaluate(dgd(frontend(), component, planner()))
    assert any(
        "references $(DYN_SYSTEM_PORT)" in reason
        for reason in reasons_matching(report, "P4: worker component agg ")
    ), report.reasons


def test_p4_checks_an_untyped_worker():
    report = evaluate(
        dgd(
            frontend(),
            router_worker("--router-mode", "kv", name="w", type_=""),
            planner(),
        )
    )
    assert any(
        "router mode 'kv'" in reason
        for reason in reasons_matching(report, "P4: worker component w ")
    ), report.reasons


def test_p4_rejects_graph_env_router_mode_reaching_the_workers():
    # recipes/solar-open2-250b: a graph-level DYN_ROUTER_MODE reaches the
    # frontend and every worker.
    deployment = dgd(
        frontend(args=("-m", "dynamo.frontend")),
        worker("prefill", type_="prefill"),
        worker("decode", type_="decode"),
        planner(),
        env=[{"name": "DYN_ROUTER_MODE", "value": "kv"}],
    )
    report = evaluate(deployment, config=planner_config("disagg"))
    assert report.startup is not None and report.startup.router_mode == "kv_router"
    assert not report.p4_ok
    assert [r.split(" advertises")[0] for r in reasons_matching(report, "P4:")] == [
        "P4: worker component prefill",
        "P4: worker component decode",
    ]


def test_p4_checks_the_real_recipe_prefill_worker():
    config = planner_config("disagg", model="nvidia/MiniMax-M3-NVFP4")
    advertising = dgd(
        frontend(),
        minimax_prefill_worker(
            "--router-mode", "kv", "--no-router-track-output-blocks"
        ),
        worker("DecodeWorker", type_="decode", model="nvidia/MiniMax-M3-NVFP4"),
        planner(),
    )
    report = evaluate(advertising, config=config)
    assert not report.p4_ok
    assert reasons_matching(
        report,
        "P4: worker component PrefillWorker advertises its own router "
        "configuration (router mode 'kv'",
    )

    # Without the router flags the worker inherits the frontend's routing;
    # its unreadable --speculative-config value does not matter, and its
    # $(MODEL_NAME) / $(SERVED_MODEL_NAME) expand.
    inheriting = dgd(
        frontend(),
        minimax_prefill_worker(),
        worker("DecodeWorker", type_="decode", model="nvidia/MiniMax-M3-NVFP4"),
        planner(),
    )
    report = evaluate(inheriting, config=config)
    assert report.p4_ok and report.spec_ok, report.reasons
    assert not [w for w in report.warnings if w.startswith("P7:")], report.warnings
    assert fc.served_model_name(inheriting) == "nvidia/MiniMax-M3-NVFP4"


@pytest.mark.parametrize(
    "command",
    # The operator's /bin/sh -c applies to an omitted or empty command; a
    # launcher may start the shell.
    [
        ("/bin/sh", "-c"),
        None,
        (),
        ("/usr/bin/env", "bash", "-c"),
        ("tini", "--", "/bin/sh", "-c"),
        ("dumb-init", "--", "bash", "-c"),
        ("/bin/bash", "-o", "pipefail", "-c"),
        ("bash", "-euo", "pipefail", "-c"),
    ],
    ids=[
        "explicit",
        "omitted",
        "empty",
        "env-bash",
        "tini-sh",
        "dumb-init-bash",
        "bash-o-pipefail",
        "bash-euo-pipefail",
    ],
)
@pytest.mark.parametrize(
    ("script", "match"),
    [
        (f"python3 -m dynamo.vllm --model {MODEL} --router-mode kv", "router mode"),
        (
            f"python3 -m dynamo.vllm --model {MODEL} --router-mode kv "
            "--no-router-kv-events",
            "router mode 'kv'",
        ),
        (
            f"export DYN_ROUTER_MODE=kv && exec python3 -m dynamo.vllm --model {MODEL}",
            "sets DYN_ROUTER_MODE in its shell command",
        ),
        (
            f"DYN_ROUTER_MODE=kv python3 -m dynamo.vllm --model {MODEL}",
            "sets DYN_ROUTER_MODE in its shell command",
        ),
        ('python3 -m dynamo.vllm --router-mode "$MODE"', "cannot parse"),
        (f"python3 -m dynamo.vllm --model {MODEL} --no-router-kv-events", None),
    ],
)
def test_p4_shell_wrapped_worker(command, script, match):
    report = evaluate(dgd(frontend(), shell_worker(script, command=command), planner()))
    rejected = reasons_matching(report, "P4: worker component agg ")
    if match is None:
        assert report.p4_ok, report.reasons
    else:
        assert not report.p4_ok
        assert any(match in reason for reason in rejected), report.reasons


@pytest.mark.parametrize(
    "command",
    # A shell option's operand (pipefail) is no script.
    [
        ("/bin/sh", "-c"),
        ("bash", "-lc"),
        None,
        ("/bin/bash", "-o", "pipefail", "-c"),
        ("bash", "-euo", "pipefail", "-c"),
    ],
    ids=["sh", "bash-lc", "omitted", "bash-o-pipefail", "bash-euo-pipefail"],
)
@pytest.mark.parametrize(
    ("script", "match"),
    [
        ("$(WORKER_CMD)", "has a shell script that references $(WORKER_CMD)"),
        (f"{WORKER} $(ROUTER_ARG)", "has a shell script that references $(ROUTER_ARG)"),
        # Kubernetes substitutes $(VAR) before the shell parses quotes, so
        # even an option's value could carry --router-mode.
        (
            "python3 -m dynamo.vllm --speculative-config $(CFG)",
            "has a shell script that references $(CFG)",
        ),
        (
            f"{WORKER} --router-temperature $(CFG)",
            "has a shell script that references $(CFG)",
        ),
        ("python3 -m dynamo.vllm --model $(MODEL_NAME)", None),
        # The shell splits an unquoted variable into words.
        (f"{WORKER} $ROUTER_ARG", "has a shell script that references $ROUTER_ARG"),
        (f"{WORKER} ${{ROUTER_ARG}}", "has a shell script that references $ROUTER_ARG"),
        (
            "python3 -m dynamo.vllm --speculative-config $SPECULATIVE_CONFIG",
            "has a shell script that references $SPECULATIVE_CONFIG",
        ),
        (
            f"X=$ROUTER_ARG; exec {WORKER} $X",
            "has a shell script that references $X (assigned from $ROUTER_ARG)",
        ),
        # An assignment may not run, so it clears neither env nor an
        # earlier dangerous value.
        (
            'if [ -z "$ROUTER_ARG" ]; then ROUTER_ARG=x; fi; '
            f"exec {WORKER} $ROUTER_ARG",
            "has a shell script that references $ROUTER_ARG",
        ),
        (
            f'[ -n "$ROUTER_ARG" ] || ROUTER_ARG=""; exec {WORKER} $ROUTER_ARG',
            "has a shell script that references $ROUTER_ARG",
        ),
        (
            f'X=$ROUTER_ARG; [ -n "$X" ] || X=; exec {WORKER} $X',
            "has a shell script that references $X",
        ),
        (
            f'[ -n "$OVERRIDE" ] && EXTRA="$OVERRIDE"; exec {WORKER} $EXTRA',
            "router mode 'kv'",
        ),
        # A quoted one stays one word: an option's value, unless it stands
        # alone as an argument of the worker.
        (f'{WORKER} "$ROUTER_ARG"', "has a shell script that references $ROUTER_ARG"),
        (
            f'exec env A=1 {WORKER} "${{ROUTER_ARG:-}}"',
            "has a shell script that references $ROUTER_ARG",
        ),
        (
            f'exec python3 -um dynamo.vllm --model {MODEL} "$ROUTER_ARG"',
            "has a shell script that references $ROUTER_ARG",
        ),
        # eval and a shell's -c parse their arguments again, also behind a
        # launcher and within a larger word.
        ('eval "$ROUTER_ARG"', "has a shell script that references $ROUTER_ARG"),
        (
            f'eval "{WORKER} $ROUTER_ARG"',
            "has a shell script that references $ROUTER_ARG",
        ),
        ('exec sh -c "$ROUTER_ARG"', "has a shell script that references $ROUTER_ARG"),
        (
            f'exec sh -c "{WORKER} $ROUTER_ARG"',
            "has a shell script that references $ROUTER_ARG",
        ),
        (
            f"exec bash -c 'exec {WORKER} $@' bash \"$ROUTER_ARG\"",
            "has a shell script that references $ROUTER_ARG",
        ),
        # source, env, export and a shell turn one quoted word into code or
        # env, also by path and behind exec, nohup, command or builtin.
        (
            f'env "$ROUTER_ARG" {WORKER}',
            "has a shell script that references $ROUTER_ARG",
        ),
        (
            f'exec env "$ROUTER_ARG" {WORKER}',
            "has a shell script that references $ROUTER_ARG",
        ),
        (
            f'exec env A=1 "$ROUTER_ARG" {WORKER}',
            "has a shell script that references $ROUTER_ARG",
        ),
        (
            f'/usr/bin/env "$ROUTER_ARG" {WORKER}',
            "has a shell script that references $ROUTER_ARG",
        ),
        (
            f'nohup env "$ROUTER_ARG" {WORKER}',
            "has a shell script that references $ROUTER_ARG",
        ),
        (
            f'command export "$ROUTER_ARG"; exec {WORKER}',
            "has a shell script that references $ROUTER_ARG",
        ),
        (
            'command eval "$ROUTER_ARG"',
            "has a shell script that references $ROUTER_ARG",
        ),
        (f'cp -r /cfg/. "$ROUTER_ARG"; exec {WORKER}', None),
        # A redirection target names a file, unless a here-string feeds
        # its text to the command.
        (f'exec {WORKER} > "$CFG" 2>&1', None),
        (f"exec {WORKER} 2>$CFG", None),
        (
            f'exec {WORKER} > "$CFG" "$ROUTER_ARG"',
            "has a shell script that references $ROUTER_ARG",
        ),
        ("exec bash <<< $ROUTER_ARG", "has a shell script that references $ROUTER_ARG"),
        # The value of an option known to take one.
        ('exec python3 -m dynamo.vllm --model "$ROUTER_ARG"', None),
        (f'{WORKER} --speculative-config="${{SPECULATIVE_CONFIG}}"', None),
        (f'{WORKER} --speculative-config "$SPECULATIVE_CONFIG"', None),
        (f'{WORKER} "--some-opt=$ROUTER_ARG"', None),
        (f'[ -n "$ROUTER_ARG" ] && echo "$ROUTER_ARG"; exec {WORKER}', None),
        (f'export LD_LIBRARY_PATH="/opt/lib:$ROUTER_ARG"; exec {WORKER}', None),
        # Literal env reached through the shell is read, and so are the
        # default and alternate words the shell may use.
        (f"{WORKER} $EXTRA", "router mode 'kv'"),
        (f'{WORKER} --router-mode "${{MODE:-random}}"', "router mode 'kv'"),
        (f"{WORKER} ${{MODE:+--router-mode=kv}}", "router mode 'kv'"),
        (
            f"{WORKER} ${{UNSET:-$ROUTER_ARG}}",
            "has a shell script that references $ROUTER_ARG",
        ),
        # Inert env never stands for an option.
        (f'{WORKER} $POD_NAME "$MY_POD" $(MY_POD) $(POD_UID)', None),
        (f'[ "$POD_INDEX" = "0" ] && exec {WORKER}', None),
        (f"{WORKER} $NOTE", "has a shell script that references $NOTE"),
        # Neither evaluated nor rejected: command substitution and script
        # variables built from it.
        (f"{WORKER} $(cat /etc/router-args)", None),
        (f"ARGS=`cat /etc/router-args`; {WORKER} $ARGS", None),
    ],
)
def test_p4_shell_wrapped_worker_with_unreadable_env(command, script, match):
    env = [
        {"name": name, "valueFrom": {"configMapKeyRef": {"name": "c", "key": "k"}}}
        for name in ("CFG", "ROUTER_ARG", "SPECULATIVE_CONFIG", "WORKER_CMD")
    ]
    env += [
        {"name": "MODEL_NAME", "value": MODEL},
        {"name": "EXTRA", "value": "--router-mode kv"},
        {"name": "MODE", "value": "kv"},
        field_ref_env("MY_POD", "metadata.name"),
        field_ref_env("POD_INDEX", "metadata.labels['grove.io/podclique-pod-index']"),
        field_ref_env("NOTE", "metadata.annotations['router']"),
    ]
    report = evaluate(
        dgd(frontend(), shell_worker(script, command=command, env=env), planner())
    )
    rejected = reasons_matching(report, "P4: worker component agg ")
    if match is None:
        assert report.p4_ok, report.reasons
    else:
        assert not report.p4_ok
        assert any(match in reason for reason in rejected), report.reasons
        assert all("on the frontend only" in reason for reason in rejected)


@pytest.mark.parametrize(
    ("command", "args", "match"),
    [
        # A launcher drops its "--", so the worker still sees later flags.
        (
            ("tini", "--", "python3", "-m", "dynamo.vllm"),
            ("--router-mode", "kv"),
            "router mode 'kv'",
        ),
        (
            ("dumb-init", "--"),
            ("python3", "-m", "dynamo.vllm", "--router-mode", "kv"),
            "router mode 'kv'",
        ),
        (
            ("/bin/sh", "-c"),
            ("exec tini -- python3 -m dynamo.vllm --router-mode kv",),
            "router mode 'kv'",
        ),
        (("tini", "--", "python3", "-m", "dynamo.vllm"), ("--model", MODEL), None),
    ],
)
def test_p4_worker_behind_a_launcher(command, args, match):
    component = worker()
    component["podTemplate"]["spec"]["containers"][0].update(
        command=list(command), args=list(args)
    )
    report = evaluate(dgd(frontend(), component, planner()))
    rejected = reasons_matching(report, "P4: worker component agg ")
    if match is None:
        assert report.p4_ok, report.reasons
    else:
        assert any(match in reason for reason in rejected), report.reasons


MODEL_FROM_CONFIG_MAP = [
    {"name": "MODEL", "valueFrom": {"configMapKeyRef": {"name": "c", "key": "model"}}}
]
RUNTIME_SIDECAR = {"name": "runtime", "restartPolicy": "Always"}


@pytest.mark.parametrize(
    ("command", "args"),
    [
        (None, ["--model-path", "$(MODEL)", "--tp", "2"]),
        (["python3", "-m", "tensorrt_llm.commands.serve"], ["$(MODEL)"]),
        (None, ["--model-path=$(MODEL)", "--router-mode", "kv"]),
    ],
)
@pytest.mark.parametrize("leader_only", [False, True], ids=["component", "leader"])
def test_p4_skips_native_sidecar_mains(command, args, leader_only):
    # The main runs the engine with its image entrypoint, and the runtime
    # sidecar registers no router configuration.
    component = worker()
    spec = component["podTemplate"]["spec"]
    main = spec["containers"][0]
    main.update(args=args, env=MODEL_FROM_CONFIG_MAP)
    if command is None:
        del main["command"]
    else:
        main["command"] = command
    if leader_only:
        component["multinode"] = {"nodeCount": 2}
        component["roles"] = [
            {
                "name": "leader",
                "podTemplate": {"spec": {"initContainers": [RUNTIME_SIDECAR]}},
            }
        ]
    else:
        spec["initContainers"] = [RUNTIME_SIDECAR]
    report = evaluate(dgd(frontend(), component, planner()))
    assert report.p4_ok, report.reasons


def test_p4_reads_an_omitted_command_without_a_native_sidecar_as_a_script():
    component = worker()
    main = component["podTemplate"]["spec"]["containers"][0]
    del main["command"]
    main.update(args=["--model-path=$(MODEL)", "--tp", "2"], env=MODEL_FROM_CONFIG_MAP)
    report = evaluate(dgd(frontend(), component, planner()))
    (reason,) = reasons_matching(report, "P4: worker component agg ")
    # Quoting or attaching never helps a $(VAR): Kubernetes substitutes it
    # before the shell parses the script.
    assert "has a shell script that references $(MODEL), which Kubernetes" in reason
    assert '(--option="$MODEL")' in reason
    assert "quote the reference" not in reason


def role_worker(
    leader: dict[str, Any], worker_: dict[str, Any], **extra: Any
) -> dict[str, Any]:
    """A multinode worker whose launches are only in its role pod templates."""

    roles = []
    for name, main in (("leader", leader), ("worker", worker_)):
        container = {"name": "main", "command": ["python3", "-m", "dynamo.vllm"]}
        container.update(main)
        roles.append(
            {"name": name, "podTemplate": {"spec": {"containers": [container]}}}
        )
    return {
        "name": "agg",
        "type": "worker",
        "replicas": 1,
        "multinode": {"nodeCount": 2},
        "roles": roles,
        **extra,
    }


ROLE_ARGS = ["--model", MODEL, "--tensor-parallel-size", "16"]


@pytest.mark.parametrize(
    ("leader", "worker_", "match"),
    [
        (
            {"args": [*ROLE_ARGS, "--router-mode", "kv"]},
            {"args": ROLE_ARGS},
            "agg (role leader) advertises its own router configuration",
        ),
        (
            {"args": ROLE_ARGS},
            {"args": ROLE_ARGS, "env": [{"name": "DYN_ROUTER_MODE", "value": "kv"}]},
            "agg (role worker) advertises its own router configuration",
        ),
        (
            {"args": [*ROLE_ARGS, "$(ROUTER_ARG)"], "env": router_arg_env()},
            {"args": ROLE_ARGS},
            "agg (role leader) has an argument that references $(ROUTER_ARG)",
        ),
        ({"args": ROLE_ARGS}, {"args": ROLE_ARGS}, None),
        # The operator sets the rank, so envFrom cannot.
        (
            {"args": ROLE_ARGS, "envFrom": ENV_FROM},
            {
                "args": [*ROLE_ARGS, "--node-rank", "$(DYNAMO_RANK)"],
                "envFrom": ENV_FROM,
            },
            None,
        ),
    ],
)
def test_p4_checks_each_role_pod_template(leader, worker_, match):
    report = evaluate(dgd(frontend(), role_worker(leader, worker_), planner()))
    if match is None:
        assert report.p4_ok, report.reasons
        assert fc.served_model_names(dgd(role_worker(leader, worker_)))[0] == {
            "agg": MODEL
        }
    else:
        assert any(
            match in reason
            for reason in reasons_matching(report, "P4: worker component ")
        ), report.reasons


def test_worker_check_ignores_the_planner_env(monkeypatch):
    monkeypatch.setenv("DYN_ROUTER_MODE", "kv")
    fc._worker_router_cached.cache_clear()
    try:
        report = evaluate(dgd())
    finally:
        fc._worker_router_cached.cache_clear()
    assert report.p4_ok, report.reasons
    assert os.environ["DYN_ROUTER_MODE"] == "kv"


WORKER_GOLDEN_CASES = [
    ((), {}),
    (("--router-mode", "kv"), {}),
    (("--router-m", "random"), {}),
    (("--router-mode=device-aware-weighted",), {}),
    ((), {"DYN_ROUTER_MODE": "kv"}),
    (("--router-mode", "least-loaded"), {"DYN_ROUTER_MODE": "kv"}),
    (("--no-router-kv-events", "--router-temperature", "0.5"), {}),
    ((), {"DYN_ROUTER_TEMPERATURE": "0.5"}),
    # What an unreadable "$(ROUTER_ARG)" after --no-router-kv-events can be.
    (("--no-router-kv-events", "--router-mode=kv"), {}),
]


@pytest.mark.parametrize(("args", "env"), WORKER_GOLDEN_CASES)
def test_worker_router_mode_matches_the_real_worker_parser(monkeypatch, args, env):
    """Backends parse their left-over argv with parse_worker_router_config."""

    from dynamo.common.configuration.groups.router_args import (
        parse_worker_router_config,
    )

    for name in list(os.environ):
        if name.startswith("DYN_"):
            monkeypatch.delenv(name)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config, _ = parse_worker_router_config([f"--model={MODEL}", *args])
    for name in env:
        monkeypatch.delenv(name)

    worker_env = [{"name": name, "value": value} for name, value in env.items()]
    report = evaluate(dgd(frontend(), router_worker(*args, env=worker_env), planner()))
    rejected = reasons_matching(report, "P4: worker component agg ")
    if config.router_mode is None:
        assert rejected == [], rejected
    else:
        assert any(
            f"router mode {config.router_mode!r}" in reason for reason in rejected
        ), rejected


def test_worker_env_from_is_warned_once():
    deployment = dgd(frontend(), worker("a"), worker("b"), planner())
    for component in deployment["spec"]["components"]:
        component["podTemplate"]["spec"]["containers"][0]["envFrom"] = [
            {"secretRef": {"name": "hf-token-secret"}}
        ]
    report = evaluate(deployment)
    assert report.p4_ok and report.spec_ok, report.reasons
    env_from = sorted(w for w in report.warnings if "envFrom" in w)
    assert env_from == [
        "P10: the frontend uses envFrom, which the planner cannot read; it "
        "assumes envFrom sets no DYN_FRONTEND_ADMIN_PORT and calls port 8002. "
        "Set DYN_FRONTEND_ADMIN_PORT in the container's env to be sure",
        "P4: the frontend container uses envFrom secretRef 'hf-token-secret'; "
        "the planner cannot read it and assumes it sets no router env (P14 "
        "still checks the routers' policy parameters)",
        "P4: worker components a, b use envFrom; the planner cannot read it and "
        "assumes it sets no router env (a DYN_ROUTER_MODE there would make the "
        "worker advertise its own router configuration)",
    ]


# ---------------------------------------------------------- P6-P10


def test_p6_warns_on_unknown_model():
    report = evaluate(dgd(), config=planner_config(model="acme/private-model"))
    assert report.spec_ok
    assert any(w.startswith("P6:") for w in report.warnings)


@pytest.fixture
def uncached_model_warning():
    fc._model_warning.cache_clear()
    yield
    fc._model_warning.cache_clear()


@pytest.mark.usefixtures("uncached_model_warning")
def test_p6_without_aisimulate_is_not_checked(monkeypatch):
    monkeypatch.setitem(sys.modules, "aisim_core.sdk.common", None)
    assert fc._model_warning("acme/private-model") is None


@pytest.mark.usefixtures("uncached_model_warning")
def test_p6_broken_aisimulate_warns(monkeypatch, caplog):
    def broken(name: str) -> Any:
        raise RuntimeError("broken install")

    module = types.ModuleType("aisim_core.sdk.common")
    module.__getattr__ = broken  # type: ignore[method-assign]
    monkeypatch.setitem(sys.modules, "aisim_core.sdk.common", module)
    warning = fc._model_warning(MODEL)
    assert warning is not None and warning.startswith("P6: cannot check")
    assert "RuntimeError: broken install" in warning
    assert any(r.levelname == "ERROR" for r in caplog.records)


def test_p7_served_models():
    two = dgd(frontend(), worker("a"), worker("b", model="meta/llama"), planner())
    report = evaluate(two)
    assert reasons_matching(report, "P7:") == [
        f"P7: the DGD serves 2 models ({MODEL}, meta/llama); live retune needs one"
    ]
    assert fc.served_model_name(two) is None
    assert fc.served_model_name(dgd()) == MODEL


def test_p7_served_model_prefers_current_status_and_warns_when_unknown():
    unknown = worker("b")
    unknown["podTemplate"]["spec"]["containers"][0]["args"] = ["--model", "$(MODEL)"]
    deployment = dgd(frontend(), worker("a"), unknown, planner())
    report = evaluate(deployment)
    assert report.spec_ok
    assert any("served model of b" in w for w in report.warnings)

    deployment["metadata"]["generation"] = 3
    deployment["status"] = {
        "observedGeneration": 3,
        "components": {"b": {"servedModelName": "other/model"}},
    }
    assert fc.served_model_names(deployment) == ({"a": MODEL, "b": "other/model"}, ())
    deployment["status"]["observedGeneration"] = 2
    assert fc.served_model_names(deployment)[1] == ("b",)


def test_p7_shell_wrapped_worker_model():
    shell = worker("a")
    shell["podTemplate"]["spec"]["containers"][0].update(
        command=["/bin/sh", "-c"],
        args=[
            f"python3 -m dynamo.sglang --model-path {MODEL} --served-model-name qwen"
        ],
    )
    assert fc.served_model_names(dgd(frontend(), shell)) == ({"a": "qwen"}, ())


def p7_model_warnings(report: PremiseReport) -> list[str]:
    return [w for w in report.warnings if w.startswith("P7: retune.dgdr.model")]


def test_p7_warns_when_dgdr_model_is_not_what_the_workers_serve():
    deployment = dgd(frontend(), worker(model="meta-llama/Llama-3.1-8B"), planner())
    report = evaluate(deployment)
    assert report.p4_ok and report.spec_ok, report.reasons
    assert p7_model_warnings(report) == [
        f"P7: retune.dgdr.model {MODEL!r} is not among the workers' "
        "--model/--model-path/--served-model-name values "
        "(meta-llama/Llama-3.1-8B); every DGDR Run simulates retune.dgdr.model, "
        "so make sure it is the model the DGD serves"
    ]
    assert not p7_model_warnings(evaluate(dgd()))


@pytest.mark.parametrize(
    "args",
    [
        # An alias: the HF id is still among the values.
        ["--model", "/models/qwen", "--served-model-name", MODEL],
        ["--model-path", MODEL, "--served-model-name", "qwen"],
        ["--model", "/models/qwen", f"--served-model-name=qwen,{MODEL}"],
        # Expanded against the worker's env.
        ["--model", "$(MODEL_ID)"],
    ],
)
def test_p7_dgdr_model_matches_any_worker_model_value(args):
    component = worker()
    component["podTemplate"]["spec"]["containers"][0].update(
        args=args, env=[{"name": "MODEL_ID", "value": MODEL}]
    )
    report = evaluate(dgd(frontend(), component, planner()))
    assert not p7_model_warnings(report), report.warnings


def test_p7_dgdr_model_counts_the_current_served_model_status():
    component = worker()
    component["podTemplate"]["spec"]["containers"][0]["args"] = ["--model", "$(X)"]
    deployment = dgd(frontend(), component, planner())
    deployment["metadata"]["generation"] = 1
    deployment["status"] = {
        "observedGeneration": 1,
        "components": {"agg": {"servedModelName": "other/model"}},
    }
    assert p7_model_warnings(evaluate(deployment))
    deployment["status"]["components"]["agg"]["servedModelName"] = MODEL
    assert not p7_model_warnings(evaluate(deployment))
    # Nothing known: only the undetermined-model warning.
    del deployment["status"]
    report = evaluate(deployment)
    assert not p7_model_warnings(report)
    assert any("served model of agg" in w for w in report.warnings)


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ({"name": "Epp", "type": "epp"}, "EPP component"),
        (
            worker("side", frontendSidecar="sidecar-frontend"),
            "side run a frontendSidecar",
        ),
    ],
)
def test_p7_gaie_layouts_with_a_frontend(extra, match):
    report = evaluate(dgd(frontend(), worker(), planner(), extra))
    assert report.p4_ok
    assert any(match in r for r in reasons_matching(report, "P7:")), report.reasons


def test_p7_no_frontend():
    report = evaluate(dgd(worker(frontendSidecar="sidecar-frontend"), planner()))
    assert not report.p4_ok
    assert all(r.startswith("P7:") for r in report.reasons)
    assert len(report.reasons) == 1


@pytest.mark.parametrize(
    ("args", "env"),
    [
        ((*KV_ARGS, "--tls-cert-path", "/c.pem", "--tls-key-path", "/k.pem"), {}),
        ((*KV_ARGS,), {"DYN_TLS_CERT_PATH": "/c.pem", "DYN_TLS_KEY_PATH": "/k.pem"}),
    ],
)
def test_p7_tls(args, env):
    report = evaluate(with_frontend(args=args, env=trace_env(**env)))
    assert report.p4_ok
    assert reasons_matching(report, "P7:") == [
        "P7: the frontend serves TLS (--tls-cert-path / DYN_TLS_CERT_PATH)"
    ]


@pytest.mark.parametrize(
    ("replicas", "args", "ok"),
    [
        (1, KV_ARGS, True),
        (0, KV_ARGS, True),
        (2, KV_ARGS, False),
        (2, (*KV_ARGS, "--router-replica-sync"), True),
    ],
)
def test_p8_frontend_replicas(replicas, args, ok):
    report = evaluate(dgd(frontend(replicas=replicas, args=args), worker(), planner()))
    assert bool(reasons_matching(report, "P8:")) is not ok
    assert report.p4_ok


def test_p9_scaling_adapter_and_planner_replicas():
    adapter = dgd(frontend(), worker(scalingAdapter={}), planner())
    assert reasons_matching(evaluate(adapter), "P9:") == [
        "P9: worker component agg has a scalingAdapter; its replicas could change during a window"
    ]
    two_planners = dgd(frontend(), worker(), planner(replicas=2))
    assert "2 replicas" in reasons_matching(evaluate(two_planners), "P9:")[0]
    no_planner = evaluate(dgd(frontend(), worker()))
    assert no_planner.spec_ok
    assert any(w.startswith("P9:") for w in no_planner.warnings)


P10_CASES = [
    ({"DYN_REQUEST_TRACE": "0"}, None, "DYN_REQUEST_TRACE=1"),
    ({"DYN_REQUEST_TRACE": "true"}, None, None),
    ({"DYN_REQUEST_TRACE_SINKS": "stderr"}, None, "DYN_REQUEST_TRACE_SINKS=file"),
    (
        {"DYN_REQUEST_TRACE_SINKS": "file,jsonl_gz"},
        None,
        "DYN_REQUEST_TRACE_SINKS=file",
    ),
    ({"DYN_REQUEST_TRACE_SINKS": "file,stderr"}, None, None),
    ({"DYN_REQUEST_TRACE_FILE_FORMAT": None}, None, None),
    ({"DYN_REQUEST_TRACE_FILE_FORMAT": "jsonl"}, None, "DYN_REQUEST_TRACE_FILE_FORMAT"),
    (
        {"DYN_REQUEST_TRACE_RECORDS": None},
        None,
        "DYN_REQUEST_TRACE_RECORDS=request_end",
    ),
    (
        {"DYN_REQUEST_TRACE_RECORDS": "request_end,tool"},
        None,
        "DYN_REQUEST_TRACE_RECORDS",
    ),
    ({"DYN_REQUEST_TRACE_RECORDS": " Request_End "}, None, None),
    (
        {"DYN_REQUEST_TRACE_FILE_PATH": None},
        None,
        "must set DYN_REQUEST_TRACE_FILE_PATH",
    ),
    (
        {"DYN_REQUEST_TRACE_FILE_PATH": f"/trace/segments/{DGD}/$(DYN_POD_NAME)/req"},
        None,
        None,
    ),
    # The reconciler prunes only the top level of the pod's directory.
    (
        {
            "DYN_REQUEST_TRACE_FILE_PATH": "/trace/segments/$(DYN_PARENT_DGD_K8S_NAME)/$(DYN_POD_NAME)/a/req"
        },
        None,
        "no further '/'",
    ),
    (
        {
            "DYN_REQUEST_TRACE_FILE_PATH": "/trace/segments/$(DYN_PARENT_DGD_K8S_NAME)/$(DYN_POD_NAME)/req-$(DYN_POD_NAME)"
        },
        None,
        None,
    ),
    (
        {
            "DYN_REQUEST_TRACE_FILE_PATH": "/trace/segments/$(DYN_PARENT_DGD_K8S_NAME)/$(DYN_POD_NAME)/"
        },
        None,
        "file prefix",
    ),
    (
        {
            "DYN_REQUEST_TRACE_FILE_PATH": "/trace/segments/$(DYN_PARENT_DGD_K8S_NAME)/$(POD_NAME)/req"
        },
        None,
        "file prefix",
    ),
    (
        {
            "DYN_REQUEST_TRACE_FILE_PATH": "/trace/other/$(DYN_PARENT_DGD_K8S_NAME)/$(DYN_POD_NAME)/req"
        },
        None,
        "file prefix",
    ),
    (
        {
            "DYN_REQUEST_TRACE_FILE_PATH": "/data/segments/$(DYN_PARENT_DGD_K8S_NAME)/$(DYN_POD_NAME)/req"
        },
        None,
        "file prefix",
    ),
    (
        {
            "DYN_REQUEST_TRACE_FILE_PATH": {
                "valueFrom": {"configMapKeyRef": {"name": "c", "key": "p"}}
            }
        },
        None,
        "must be a literal value",
    ),
    (
        {"DYN_POD_NAME": {"valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}}}},
        None,
        "DYN_POD_NAME from fieldRef",
    ),
    ({"DYN_PARENT_DGD_K8S_NAME": "other"}, None, "overrides DYN_PARENT_DGD_K8S_NAME"),
    (
        {},
        {"volumes": [{"name": "traces", "persistentVolumeClaim": {"claimName": "x"}}]},
        "no volume for claim",
    ),
    ({}, {"mounts": []}, "does not mount claim"),
    (
        {},
        {"mounts": [{"name": "traces", "mountPath": "/trace", "readOnly": True}]},
        "read-only or through a subPath",
    ),
    ({}, {"mounts": [{"name": "traces", "mountPath": "/trace/"}]}, None),
    (
        {},
        {
            "mounts": [
                {"name": "traces", "mountPath": "/ro", "readOnly": True},
                {"name": "traces", "mountPath": "/trace"},
            ]
        },
        None,
    ),
    # The admin listener that serves /router_policy.
    (
        {"DYN_ENABLE_FRONTEND_ADMIN_API": None},
        None,
        "must set DYN_ENABLE_FRONTEND_ADMIN_API=1",
    ),
    (
        {"DYN_FRONTEND_ADMIN_HOST": None},
        None,
        "its admin listener binds 127.0.0.1 by default",
    ),
    ({"DYN_FRONTEND_ADMIN_PORT": "9100"}, None, None),
    # The planner reads the pod spec, where $(VAR) is not expanded yet.
    (
        {"DYN_FRONTEND_ADMIN_PORT": "$(DYN_POD_NAME)"},
        None,
        "DYN_FRONTEND_ADMIN_PORT must be a literal value",
    ),
]


@pytest.mark.parametrize(("env", "frontend_kwargs", "match"), P10_CASES)
def test_p10_trace_recipe(env, frontend_kwargs, match):
    deployment = dgd(
        frontend(env=trace_env(**env), **(frontend_kwargs or {})), worker(), planner()
    )
    report = evaluate(deployment)
    assert report.p4_ok, report.reasons
    p10 = reasons_matching(report, "P10:")
    if match is None:
        assert p10 == [], p10
        assert report.spec_ok
    else:
        assert any(match in r for r in p10), p10
        assert not report.spec_ok


def test_p10_warns_when_env_from_could_set_the_admin_port():
    def with_env_from(env: list[dict[str, Any]], source: dict[str, Any]) -> dict:
        component = frontend(env=env)
        component["podTemplate"]["spec"]["containers"][0]["envFrom"] = [source]
        return dgd(component, worker(), planner())

    def port_warnings(report: Any) -> list[str]:
        return [w for w in report.warnings if "DYN_FRONTEND_ADMIN_PORT" in w]

    source = {"configMapRef": {"name": "frontend-env"}}
    report = evaluate(with_env_from(trace_env(), source))
    assert report.spec_ok, report.reasons
    assert port_warnings(report), report.warnings
    # An explicit env entry wins over envFrom.
    report = evaluate(with_env_from(trace_env(DYN_FRONTEND_ADMIN_PORT="8002"), source))
    assert report.spec_ok and not port_warnings(report)
    # A prefixed source cannot define the name.
    report = evaluate(with_env_from(trace_env(), {**source, "prefix": "APP_"}))
    assert report.spec_ok and not port_warnings(report)


def test_p10_custom_segments_dir_and_graph_env():
    path = "/trace/rt/segs/$(DYN_PARENT_DGD_K8S_NAME)/$(DYN_POD_NAME)/req"
    deployment = dgd(
        frontend(env=trace_env(DYN_REQUEST_TRACE_FILE_PATH=None)),
        worker(),
        planner(),
        env=[{"name": "DYN_REQUEST_TRACE_FILE_PATH", "value": path}],
    )
    report = evaluate(deployment, config=planner_config(segments_dir="rt/segs"))
    assert report.spec_ok, report.reasons


def test_p10_extra_sinks_warn():
    report = evaluate(
        with_frontend(env=trace_env(DYN_REQUEST_TRACE_SINKS="file,stderr"))
    )
    assert report.spec_ok
    assert any("also enables stderr" in w for w in report.warnings)


def test_p10_is_checked_when_parsing_fails():
    deployment = with_frontend(
        command=("/bin/sh", "-c"),
        args=("python3 -m dynamo.frontend",),
        env=trace_env(DYN_REQUEST_TRACE=None),
    )
    report = evaluate(deployment)
    assert reasons_matching(report, "P4:")
    assert reasons_matching(report, "P10:")


def test_evaluate_requires_dgdr():
    config = PlannerConfig(namespace="test-ns", environment="kubernetes", mode="agg")
    with pytest.raises(ValueError, match="retune.dgdr"):
        fc.evaluate_premises(dgd(), planner_config=config)


def test_malformed_dgd_never_raises():
    odd_frontend = {
        "name": "f",
        "type": "frontend",
        "replicas": "many",
        "podTemplate": "not-a-template",
    }
    for deployment in (
        {},
        {"spec": None},
        {"spec": {"components": [None, 3]}},
        {"spec": {"components": [odd_frontend]}},
        {"spec": {"components": [frontend(volumes=["x"], mounts=[None])]}},
    ):
        report = evaluate(deployment)
        assert not report.spec_ok


def test_unexpected_parser_failure_fails_closed(monkeypatch, caplog):
    def broken(env):
        raise RuntimeError("frontend arg groups changed")

    monkeypatch.setattr(fc, "_frontend_parser", broken)
    fc._parse_cached.cache_clear()
    try:
        error = parse_error(dgd())
    finally:
        fc._parse_cached.cache_clear()
    assert error.startswith("P4: the planner failed to parse")
    assert "RuntimeError: frontend arg groups changed" in error
    assert any(r.levelname == "ERROR" for r in caplog.records)


# ------------------------------------------------------------- helpers


def test_worker_types_and_canonical_json():
    assert fc.worker_types("disagg") == ("prefill", "decode")
    assert fc.worker_types("agg") == ("aggregated",)
    assert fc.canonical_json({"b": 1, "a": [1.5]}) == '{"a":[1.5],"b":1}'


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("plain", "plain"),
        ("$(A)/x", "1/x"),
        ("$$(A)", "$(A)"),
        ("$(MISSING)", "$(MISSING)"),
        ("$(A", "$(A"),
        ("$", "$"),
        ("a$$b", "a$b"),
        ("x-$(SECRET)", "x-{opaque}$(SECRET)\x00"),
        ("$(POD)", "{inert}$(POD)\x00"),
        # Undefined names an envFrom source with prefix APP_ may define.
        ("$(APP_ARG)", "{opaque}$(APP_ARG)\x00"),
        ("$(APP_a.b-c)", "{opaque}$(APP_a.b-c)\x00"),
        # The container's env wins over envFrom.
        ("$(APP_DEFINED)", "2"),
        ("$(ARG)", "$(ARG)"),
        ("$(APP_ seq 1 60)", "$(APP_ seq 1 60)"),
    ],
)
def test_kubernetes_expansion(value, expected):
    expanded = fc._expand(
        value,
        {"A": "1", "APP_DEFINED": "2"},
        {"SECRET", "POD"},
        inert={"POD"},
        env_from_prefixes=("APP_",),
    )
    assert expanded == expected.format(opaque=fc._OPAQUE, inert=fc._INERT)


def test_parse_is_cached_per_spec():
    fc._parse_cached.cache_clear()
    parse(dgd())
    parse(dgd())
    info = fc._parse_cached.cache_info()
    assert info.hits >= 1 and info.misses == 1
    parse(with_frontend(args=(*KV_ARGS, "--router-temperature", "0.5")))
    assert fc._parse_cached.cache_info().misses == 2
