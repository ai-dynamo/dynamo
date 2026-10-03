# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the live deployment recipe (run with the worktree's .venv python)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

DEPLOY = Path(__file__).resolve().parents[1]
ENGINE_JSON = (
    Path(os.environ.get("LR_ROOT") or Path.home() / "learned-routing")
    / "config"
    / "engine.json"
)


def load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"lr_deploy_{name}", DEPLOY / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


plan = load("plan")
weights = load("weights")
health = load("health_check")
facts = load("live_facts")

CAMPAIGN_ENGINE = {
    "model": "Qwen/Qwen3-32B",
    "tp": 2,
    "backend": "vllm",
    "backend_version": "0.24.0",
    "hardware": "h100_sxm",
    "max_model_len": 131072,
    "kv_capacity": {"block_size": 16, "num_gpu_blocks": 18863},
    "mock_engine_args": {
        "ais_perf_config": {
            "backend": "vllm",
            "backend_version": "0.24.0",
            "model": "Qwen/Qwen3-32B",
            "system": "h100_sxm",
            "tp": 2,
            "worker_type": "aggregated",
        },
        "block_size": 16,
        "decode_speedup_ratio": 1.0,
        "dp_size": 1,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True,
        "engine_type": "vllm",
        "max_model_len": 131072,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 1024,
        "num_gpu_blocks": 18863,
        "speedup_ratio": 1.0,
        "worker_type": "aggregated",
    },
}


def flag_value(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def test_engine_plan_mirrors_every_fidelity_knob():
    engine = plan.engine_plan(json.loads(json.dumps(CAMPAIGN_ENGINE)))
    args = engine["vllm_args"]
    assert flag_value(args, "--block-size") == "16"
    assert flag_value(args, "--max-model-len") == "131072"
    assert flag_value(args, "--max-num-seqs") == "1024"
    assert flag_value(args, "--max-num-batched-tokens") == "8192"
    assert flag_value(args, "--tensor-parallel-size") == "2"
    assert "--enable-chunked-prefill" in args and "--enable-prefix-caching" in args
    # vLLM's null block: one extra total block gives replay's usable capacity.
    assert flag_value(args, "--num-gpu-blocks-override") == "18864"
    assert engine["rope_scaling"] == {
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 32768,
    }


@pytest.mark.skipif(not ENGINE_JSON.exists(), reason="campaign engine.json not present")
def test_engine_plan_accepts_the_frozen_campaign_engine():
    engine = plan.engine_plan(json.loads(ENGINE_JSON.read_text()))
    assert engine["total_kv_blocks"] == engine["usable_kv_blocks"] + 1


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda e: e["mock_engine_args"].update(preemption_mode="swap"), "unmapped"),
        (
            lambda e: e["mock_engine_args"].update(speedup_ratio=2.0),
            "cannot be mirrored",
        ),
        (
            lambda e: e["mock_engine_args"].update(enable_prefix_caching=False),
            "prefix caching",
        ),
        (lambda e: e["kv_capacity"].update(num_gpu_blocks=1), "kv_capacity"),
        (lambda e: e.update(hardware="b200"), "H100"),
    ],
)
def test_engine_plan_refuses_unmirrorable_engines(mutate, message):
    engine = json.loads(json.dumps(CAMPAIGN_ENGINE))
    mutate(engine)
    with pytest.raises(plan.PlanError, match=message):
        plan.engine_plan(engine)


def test_policy_plan_yaml_and_frontend_kwargs_match_replay(tmp_path):
    from learned_routing.policy import spec_from_dict

    specs = [
        {
            "name": "knobs",
            "router_mode": "kv_router",
            "type": "dynamo-default-cost-fn",
            "parameters": {},
            "router_config": {
                "router_temperature": 0.25,
                "overlap_score_credit": 1.5,
                "router_track_output_blocks": True,
                "router_queue_threshold": 0.8,
            },
        },
        {
            "name": "lc",
            "router_mode": "kv_router",
            "type": "learned-choice",
            "parameters": {"feature_set": "v1", "theta": [-1.0] + [0.0] * 7},
        },
        {"name": "rr", "router_mode": "round_robin"},
    ]
    spec_file = tmp_path / "specs.json"
    spec_file.write_text(json.dumps(specs))
    plans = {
        p["name"]: p
        for p in plan.policy_plans([str(spec_file)], 3, tmp_path / "plan", 16)
    }
    for raw in specs:
        spec = spec_from_dict(raw)
        expected = spec.replay_yaml_text(
            spec.effective_seed(4)
        )  # replicate 3 -> seed 4
        got = plans[raw["name"]]
        if expected is None:
            assert got["policy_yaml"] is None and got["frontend_flags"][:2] == [
                "--router-mode",
                "round-robin",
            ]
            continue
        written = (
            tmp_path / "plan" / "policies" / got["slug"] / "policy.yaml"
        ).read_text()
        assert written == expected
        assert (
            got["policy_yaml_sha256"] == hashlib.sha256(expected.encode()).hexdigest()
        )
    kwargs = plans["knobs"]["kv_router_kwargs"]
    assert (
        kwargs["router_temperature"] == 0.25 and kwargs["overlap_score_credit"] == 1.5
    )
    assert (
        kwargs["router_track_output_blocks"] is True
        and kwargs["router_queue_threshold"] == 0.8
    )
    assert plans["lc"]["kv_router_kwargs"]["router_prefill_load_model"] == "none"


def test_frontend_parity_rejects_a_knob_without_a_flag(tmp_path):
    from learned_routing.policy import PolicySpec

    spec = PolicySpec(
        name="x",
        router_mode="kv_router",
        type="dynamo-default-cost-fn",
        router_config={"not_a_knob": 1},
        seeded=True,
    )
    yaml_path = tmp_path / "p.yaml"
    yaml_path.write_text(spec.replay_yaml_text(1))
    with pytest.raises(plan.PlanError, match="no frontend flag"):
        plan.frontend_parity(spec, yaml_path, 16)


def manifest_for(root: Path, names: dict[str, bool]) -> dict:
    files = []
    for name, lfs in names.items():
        data = (root / name).read_bytes()
        entry = {"path": name, "size": len(data)}
        if lfs:
            entry["sha256"] = hashlib.sha256(data).hexdigest()
        else:
            entry["git_blob_sha1"] = subprocess.run(
                ["git", "hash-object", str(root / name)],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        files.append(entry)
    return {"schema": "learned-routing.model-manifest.v1", "files": files}


def test_weights_copy_verify_and_patch(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text(
        json.dumps({"rope_scaling": None, "rope_theta": 1000000})
    )
    (src / "model-00001.safetensors").write_bytes(bytes(range(256)) * 1000)
    manifest = manifest_for(
        src, {"config.json": False, "model-00001.safetensors": True}
    )
    assert weights.verify(src, manifest, 2, sizes_only=False)["ok"]

    dst = tmp_path / "dst"
    weights.copy(src, dst, manifest, 2)
    rope = {
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 32768,
    }
    result = weights.patch_config(dst, manifest, rope)
    assert json.loads((dst / "config.json").read_text())["rope_scaling"] == rope
    assert (dst / "config.json.orig").read_bytes() == (src / "config.json").read_bytes()
    assert result["patched_sha256"] != result["original_sha256"]
    # Patching again is idempotent: it re-verifies the preserved original.
    weights.patch_config(dst, manifest, rope)
    assert weights.verify(dst, manifest, 2, False, frozenset({"config.json"}))["ok"]

    corrupt = bytearray((dst / "model-00001.safetensors").read_bytes())
    corrupt[1234] ^= 1
    (dst / "model-00001.safetensors").write_bytes(bytes(corrupt))
    report = weights.verify(dst, manifest, 2, False, frozenset({"config.json"}))
    assert not report["ok"] and report["problems"] == ["hash model-00001.safetensors"]


def test_weights_patch_refuses_an_unpinned_config(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    manifest = {
        "schema": "learned-routing.model-manifest.v1",
        "files": [{"path": "config.json", "size": 2, "git_blob_sha1": "0" * 40}],
    }
    with pytest.raises(SystemExit, match="pinned"):
        weights.patch_config(tmp_path, manifest, {"rope_type": "yarn"})


def test_pinned_manifest_lists_the_full_checkpoint():
    manifest = json.loads((DEPLOY / "qwen3-32b-9216db57.manifest.json").read_text())
    names = {f["path"] for f in manifest["files"]}
    shards = {f"model-{i:05d}-of-00017.safetensors" for i in range(1, 18)}
    assert (
        shards <= names
        and {"config.json", "tokenizer.json", "tokenizer_config.json"} <= names
    )
    assert all("sha256" in f for f in manifest["files"] if f["path"] in shards)


def run_lock(lock_dir: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(DEPLOY / "hold_lock.sh"), *args],
        capture_output=True,
        text=True,
        env={"LR_HOLD_LOCK_DIR": str(lock_dir), "PATH": "/usr/bin:/bin"},
    )


def test_hold_lock_is_exclusive_and_never_deletes(tmp_path):
    assert run_lock(tmp_path, "acquire", "test", "start-a", "2").returncode == 0
    assert run_lock(tmp_path, "acquire", "other", "start-b", "1").returncode != 0
    assert run_lock(tmp_path, "release", "start-b").returncode != 0
    assert run_lock(tmp_path, "acquire", "x", "y", "4").returncode != 0  # > 3 h
    assert run_lock(tmp_path, "release", "start-a").returncode == 0
    names = sorted(p.name for p in tmp_path.iterdir())
    assert len(names) == 1 and names[0].startswith("released-start-a-")
    assert run_lock(tmp_path, "status").stdout.strip() == "free"


def test_hold_lock_takes_an_expired_lock_only_when_asked(tmp_path):
    (tmp_path / "ACTIVE").write_text(
        "owner\told\nstart\told-start\ncreated_utc\t2026-01-01T00:00:00Z\n"
        "hard_expiry_utc\t2026-01-01T01:00:00Z\n"
    )
    assert run_lock(tmp_path, "acquire", "me", "new-start", "1").returncode != 0
    assert (
        run_lock(
            tmp_path, "acquire", "me", "new-start", "1", "--take-expired"
        ).returncode
        == 0
    )
    assert any(p.name.startswith("expired-old-start-") for p in tmp_path.iterdir())


def test_live_facts_records_and_merges_jobs(tmp_path):
    path = tmp_path / "live.json"
    path.write_text(json.dumps({"other_stage": {"keep": True}}))
    facts.update(
        path,
        "123",
        {"cancel": "ssh gpu-cluster scancel 123", "state": "SUBMITTED"},
        True,
    )
    with pytest.raises(SystemExit, match="already recorded"):
        facts.update(path, "123", {}, True)
    facts.update(path, "123", {"state": "COMPLETED"}, False)
    data = json.loads(path.read_text())
    assert data["other_stage"] == {"keep": True}
    (job,) = data["jobs"]
    assert (
        job["state"] == "COMPLETED" and job["cancel"] == "ssh gpu-cluster scancel 123"
    )
    with pytest.raises(SystemExit, match="not recorded"):
        facts.update(path, "999", {}, False)


def test_metric_parsing_matches_counters_with_labels_and_total_suffix():
    text = (
        "# HELP x\n"
        'vllm:prefix_cache_hits_total{engine="0",model_name="m"} 4096.0\n'
        'dynamo_kvrouter_kv_cache_events_applied{event_type="stored",status="ok"} 12\n'
        'dynamo_kvrouter_kv_cache_events_applied{event_type="removed",status="ok"} 3\n'
        'dynamo_component_inflight_requests{dynamo_endpoint="generate",worker_id="2a"} 0\n'
    )
    samples = []
    for line in text.splitlines():
        match = health.SAMPLE.match(line)
        if match and not line.startswith("#"):
            name, labels, value = match.groups()
            samples.append(
                (name, dict(health.LABEL.findall(labels or "")), float(value))
            )
    assert health.metric_sum(samples, "vllm:prefix_cache_hits") == 4096.0
    assert (
        health.metric_sum_suffix(
            samples, "kv_cache_events_applied", event_type="stored"
        )
        == 12
    )
    assert health.metric_sum(samples, "vllm:num_requests_running") is None
    assert int(samples[-1][1]["worker_id"], 16) == 42
