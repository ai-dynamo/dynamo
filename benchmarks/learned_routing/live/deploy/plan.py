# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Freeze a live deployment plan from the campaign's engine config and harness policy specs.

The live lane must run the same engine and the same router as offline replay. This tool turns the
two sources of truth into the exact launch inputs and checks the mapping:

* ``CR/config/engine.json`` -> vLLM worker flags. Every ``mock_engine_args`` key is either mirrored
  by a flag or explicitly sim-only; an unknown key or a value the live engine cannot reproduce fails.
* Harness policy specs (``learned_routing.policy``) -> per-policy frontend inputs: the replay YAML
  for replicate ``k`` (byte-identical to what ``lr-eval`` hands replay) and the CLI flags that carry
  the spec's ``router_config`` knobs.

The frontend's own argument parser then parses those flags, and the resulting ``KvRouterConfig``
keyword arguments must equal the ones replay passes (signature defaults + ``router_config`` +
``router_policy_config``). Run with the worktree's ``.venv`` python (it imports ``dynamo`` and
``learned_routing``)::

    .venv/bin/python benchmarks/learned_routing/live/deploy/plan.py \\
        --engine CR/config/engine.json \\
        --spec default --spec round_robin --spec specs.json --replicate 0 --out PLAN_DIR
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any

PLAN_SCHEMA = "learned-routing.live-plan.v1"
NATIVE_CONTEXT = (
    40960  # Qwen3-32B max_position_embeddings (config.json at the pinned revision)
)
YARN_ORIGINAL_CONTEXT = 32768  # Qwen3 model card: YaRN base for the 131072 context
REFERENCE_TREES = ("lib", "components", "Cargo.lock")

# Every MockEngineArgs key the campaign may set. Each maps to a live vLLM flag or is sim-only with a
# required value; anything else fails the plan instead of silently diverging.
SIM_ONLY = {
    "speedup_ratio": 1.0,
    "decode_speedup_ratio": 1.0,
}

# Fixed live-only choices. They are not in engine.json because replay has no equivalent knob; each
# is the value AIS and the mocker assume (bf16 weights and KV, vLLM defaults otherwise).
FIXED_VLLM_ARGS = [
    "--dtype",
    "bfloat16",
    "--kv-cache-dtype",
    "auto",
    "--gpu-memory-utilization",
    "0.90",
    "--generation-config",
    "vllm",
]


class PlanError(ValueError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _bool_flag(name: str, value: Any) -> list[str]:
    if not isinstance(value, bool):
        raise PlanError(f"mock_engine_args.{name} must be a bool, got {value!r}")
    flag = name.replace("_", "-")
    return [f"--{flag}" if value else f"--no-{flag}"]


def engine_plan(engine: dict) -> dict:
    """vLLM worker flags and the fidelity map for one ``engine.json``."""
    args = dict(engine.get("mock_engine_args") or {})
    if not args:
        raise PlanError("engine.json has no mock_engine_args")
    ais = args.pop("ais_perf_config", None) or {}
    fidelity: list[dict] = []
    vllm_args: list[str] = []

    def mirror(key: str, flags: list[str], note: str = "") -> None:
        fidelity.append(
            {
                "engine_json": f"mock_engine_args.{key}",
                "live": " ".join(flags),
                "note": note,
            }
        )
        vllm_args.extend(flags)

    def require(key: str, value: Any, expected: Any, note: str) -> None:
        if value != expected:
            raise PlanError(
                f"mock_engine_args.{key}={value!r} cannot be mirrored live (needs {expected!r})"
            )
        fidelity.append(
            {"engine_json": f"mock_engine_args.{key}", "live": "n/a", "note": note}
        )

    engine_type = args.pop("engine_type", None)
    require("engine_type", engine_type, "vllm", "live workers run dynamo.vllm")
    worker_type = args.pop("worker_type", None)
    require("worker_type", worker_type, "aggregated", "no disaggregation flags")
    for key, expected in SIM_ONLY.items():
        if key in args:
            require(key, float(args.pop(key)), expected, "simulator time scale only")

    model = engine.get("model") or ais.get("model")
    tp = int(engine.get("tp") or ais.get("tp"))
    if ais and (ais.get("model") != model or int(ais.get("tp")) != tp):
        raise PlanError("engine.json model/tp disagree with ais_perf_config")
    backend_version = engine.get("backend_version") or ais.get("backend_version")
    hardware = engine.get("hardware") or ais.get("system")
    if hardware != "h100_sxm":
        raise PlanError(f"hardware {hardware!r}: the recipe targets 8 x H100 SXM nodes")
    vllm_args += ["--served-model-name", model, "--tensor-parallel-size", str(tp)]
    fidelity.append(
        {
            "engine_json": "model, tp, ais_perf_config",
            "live": f"--served-model-name {model} --tensor-parallel-size {tp}",
            "note": f"vLLM {backend_version} on {hardware}; checked in the container and by nvidia-smi",
        }
    )

    dp = int(args.pop("dp_size", 1))
    mirror("dp_size", ["--data-parallel-size", str(dp)])
    block_size = int(args.pop("block_size"))
    mirror("block_size", ["--block-size", str(block_size)], "router block size too")
    max_model_len = int(args.pop("max_model_len"))
    if engine.get("max_model_len") not in (None, max_model_len):
        raise PlanError("engine.json max_model_len disagrees with mock_engine_args")
    rope_scaling = None
    note = "native context"
    if max_model_len > NATIVE_CONTEXT:
        factor = max_model_len / YARN_ORIGINAL_CONTEXT
        rope_scaling = {
            "rope_type": "yarn",
            "factor": factor,
            "original_max_position_embeddings": YARN_ORIGINAL_CONTEXT,
        }
        note = f"YaRN factor {factor} written into the node-local config.json"
    mirror("max_model_len", ["--max-model-len", str(max_model_len)], note)
    mirror("max_num_seqs", ["--max-num-seqs", str(int(args.pop("max_num_seqs")))])
    mirror(
        "max_num_batched_tokens",
        ["--max-num-batched-tokens", str(int(args.pop("max_num_batched_tokens")))],
    )
    mirror(
        "enable_chunked_prefill",
        _bool_flag("enable_chunked_prefill", args.pop("enable_chunked_prefill")),
    )
    prefix = args.pop("enable_prefix_caching")
    if prefix is not True:
        raise PlanError("the live recipe needs prefix caching (KV events depend on it)")
    mirror("enable_prefix_caching", _bool_flag("enable_prefix_caching", prefix))
    usable_blocks = int(args.pop("num_gpu_blocks"))
    total_blocks = usable_blocks + 1
    mirror(
        "num_gpu_blocks",
        ["--num-gpu-blocks-override", str(total_blocks)],
        f"vLLM reserves one null block, so {total_blocks} total = {usable_blocks} usable "
        "(MockEngineArgs.num_gpu_blocks convention); the router sees total_kv_blocks="
        f"{total_blocks} vs replay {usable_blocks} (feature 4 denominator, rel {1 / usable_blocks:.1e})",
    )
    if args:
        raise PlanError(f"unmapped mock_engine_args keys: {sorted(args)}")
    capacity = engine.get("kv_capacity") or {}
    if capacity and (
        int(capacity.get("num_gpu_blocks", usable_blocks)) != usable_blocks
        or int(capacity.get("block_size", block_size)) != block_size
    ):
        raise PlanError("engine.json kv_capacity disagrees with mock_engine_args")
    for flag, value in zip(FIXED_VLLM_ARGS[::2], FIXED_VLLM_ARGS[1::2]):
        fidelity.append(
            {
                "engine_json": "(live only)",
                "live": f"{flag} {value}",
                "note": "fixed by the recipe",
            }
        )
    vllm_args += FIXED_VLLM_ARGS
    return {
        "model": model,
        "tp": tp,
        "dp": dp,
        "backend": engine.get("backend", "vllm"),
        "backend_version": backend_version,
        "hardware": hardware,
        "block_size": block_size,
        "max_model_len": max_model_len,
        "usable_kv_blocks": usable_blocks,
        "total_kv_blocks": total_blocks,
        "rope_scaling": rope_scaling,
        "vllm_args": vllm_args,
        "fidelity": fidelity,
    }


@contextmanager
def _without_dyn_env():
    """The frontend reads DYN_* defaults while building its parser; plan against a clean env."""
    saved = {k: os.environ.pop(k) for k in list(os.environ) if k.startswith("DYN_")}
    try:
        yield
    finally:
        os.environ.update(saved)


def replay_kwarg_defaults() -> dict[str, Any]:
    """Defaults of the ``KvRouterConfig`` constructor that replay calls."""
    from dynamo.llm import KvRouterConfig

    signature = KvRouterConfig.__text_signature__ or ""
    defaults: dict[str, Any] = {}
    for part in signature.strip("()").split(","):
        name, sep, raw = part.partition("=")
        if sep:
            defaults[name.strip()] = ast.literal_eval(raw.strip())
    if "router_policy_config" not in defaults:
        raise PlanError(f"cannot parse KvRouterConfig signature: {signature[:120]}")
    return defaults


def _frontend_parser():
    from dynamo.frontend.frontend_args import FrontendArgGroup

    parser = argparse.ArgumentParser(add_help=False)
    FrontendArgGroup().add_arguments(parser)
    return parser


def _knob_flags(parser, knobs: dict) -> list[str]:
    by_dest: dict[str, argparse.Action] = {}
    for action in parser._actions:
        if action.option_strings and action.dest not in by_dest:
            by_dest[action.dest] = action
    flags: list[str] = []
    for name in sorted(knobs):
        value = knobs[name]
        action = by_dest.get(name)
        if action is None:
            raise PlanError(f"router_config knob {name!r} has no frontend flag")
        long_flags = [s for s in action.option_strings if s.startswith("--")]
        positive = next(s for s in long_flags if not s.startswith("--no-"))
        if isinstance(value, bool):
            flags.append(positive if value else "--no-" + positive[2:])
        elif value is None:
            flags += [positive, "None"]
        else:
            flags += [positive, repr(value) if isinstance(value, float) else str(value)]
    return flags


def frontend_parity(spec, yaml_path: Path | None, block_size: int) -> dict:
    """Parse the live frontend flags with the frontend's own parser and compare with replay."""
    from dynamo.frontend.frontend_args import FrontendConfig

    with _without_dyn_env(), warnings.catch_warnings():
        # Knob flags such as --router-temperature are deprecated aliases of the same
        # KvRouterConfig fields; the parity check below proves they land where replay puts them.
        warnings.simplefilter("ignore", FutureWarning)
        parser = _frontend_parser()
        if spec.router_mode == "round_robin":
            flags = ["--router-mode", "round-robin"]
        else:
            flags = ["--router-mode", "kv", "--router-policy-config", str(yaml_path)]
            flags += _knob_flags(parser, spec.router_config)
        argv = flags + ["--kv-cache-block-size", str(block_size)]
        config = FrontendConfig.from_cli_args(parser.parse_args(argv))
        config.validate()
    router = config.router_kwargs()
    checks = {
        "router_mode": config.router_mode,
        "session_affinity_ttl_secs": router.get("session_affinity_ttl_secs"),
        "active_decode_blocks_threshold": router.get("active_decode_blocks_threshold"),
        "active_prefill_tokens_threshold": router.get(
            "active_prefill_tokens_threshold"
        ),
        "active_prefill_tokens_threshold_frac": router.get(
            "active_prefill_tokens_threshold_frac"
        ),
        "migration_limit": config.migration_limit,
    }
    expected_mode = "round-robin" if spec.router_mode == "round_robin" else "kv"
    bad = [
        k
        for k, v in checks.items()
        if (k == "router_mode" and v != expected_mode)
        or (k == "migration_limit" and v != 0)
        or (k not in ("router_mode", "migration_limit") and v is not None)
    ]
    if bad:
        raise PlanError(
            f"frontend config differs from replay semantics: {bad} in {checks}"
        )
    result = {"frontend_flags": flags, "router_checks": checks}
    if spec.router_mode == "round_robin":
        return result
    live = config.kv_router_kwargs()
    replay = replay_kwarg_defaults()
    replay.update(spec.router_config)
    replay["router_policy_config"] = str(yaml_path)
    keys = sorted(set(live) | set(replay))
    diffs = {
        k: (live.get(k), replay.get(k)) for k in keys if live.get(k) != replay.get(k)
    }
    if diffs:
        raise PlanError(f"live KvRouterConfig kwargs differ from replay: {diffs}")
    result["kv_router_kwargs"] = {k: live[k] for k in keys}
    return result


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "policy"


def policy_plans(
    spec_sources: list[str], replicate: int, out: Path, block_size: int
) -> list[dict]:
    from learned_routing.policy import load_specs
    from learned_routing.replicates import policy_seed

    plans = []
    seen: set[str] = set()
    for spec in load_specs(spec_sources):
        slug = _slug(spec.name)
        if slug in seen:
            slug = f"{slug}-{spec.sha[:8]}"
        seen.add(slug)
        pdir = out / "policies" / slug
        pdir.mkdir(parents=True, exist_ok=True)
        seed = spec.effective_seed(policy_seed(replicate))
        text = spec.replay_yaml_text(seed)
        yaml_path = None
        if text is not None:
            yaml_path = pdir / "policy.yaml"
            yaml_path.write_text(text)
        parity = frontend_parity(spec, yaml_path, block_size)
        plan = {
            "name": spec.name,
            "slug": slug,
            "policy_sha": spec.sha,
            "policy_type": spec.policy_type,
            "router_mode": spec.router_mode,
            "spec": spec.to_dict(),
            "replicate": replicate,
            "policy_seed": seed,
            "policy_yaml": None if yaml_path is None else "policy.yaml",
            "policy_yaml_sha256": None if text is None else sha256_bytes(text.encode()),
            **parity,
        }
        # The frontend reads the YAML from the node; the flag is rewritten there.
        plan["frontend_flags"] = [
            "{policy_yaml}" if yaml_path is not None and f == str(yaml_path) else f
            for f in plan["frontend_flags"]
        ]
        (pdir / "policy_plan.json").write_text(
            json.dumps(plan, indent=1, sort_keys=True) + "\n"
        )
        plans.append(plan)
    return plans


def source_identity(worktree: Path, reference: str | None) -> dict:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(worktree), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    head = git("rev-parse", "HEAD")
    trees = {p: git("rev-parse", f"HEAD:{p}") for p in REFERENCE_TREES}
    dirty = git("status", "--porcelain", "--", *REFERENCE_TREES)
    identity = {"commit": head, "trees": trees, "serving_paths_dirty": bool(dirty)}
    if reference:
        ref_trees = {p: git("rev-parse", f"{reference}:{p}") for p in REFERENCE_TREES}
        identity["reference_commit"] = git("rev-parse", reference)
        identity["matches_reference"] = ref_trees == trees
    return identity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument(
        "--spec",
        action="append",
        required=True,
        help="builtin name or spec file (repeatable)",
    )
    parser.add_argument(
        "--replicate", type=int, default=0, help="CRN replicate k (seed k + 1)"
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--reference-commit",
        default="78f13ba65f5135e64b6c3a76673adc0a10afa518",
        help="a commit whose lib/, components/ and Cargo.lock trees equal the tuning bindings "
        "build's (build_id 6955b0ee); the deployed trees must equal it",
    )
    parser.add_argument("--allow-tree-mismatch", action="store_true")
    args = parser.parse_args(argv)

    out: Path = args.out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"error: {out} exists and is not empty; plans are write-once")
    out.mkdir(parents=True, exist_ok=True)
    engine_bytes = args.engine.read_bytes()
    engine = engine_plan(json.loads(engine_bytes))
    engine["engine_json"] = str(args.engine)
    engine["engine_json_sha256"] = sha256_bytes(engine_bytes)
    worktree = Path(__file__).resolve().parents[4]
    source = source_identity(worktree, args.reference_commit)
    if source["serving_paths_dirty"]:
        raise SystemExit(
            "error: lib/, components/ or Cargo.lock has uncommitted changes"
        )
    if not source.get("matches_reference", True) and not args.allow_tree_mismatch:
        raise SystemExit(
            f"error: serving trees at {source['commit'][:12]} differ from the tuning build "
            f"reference {args.reference_commit[:12]}; pass --allow-tree-mismatch to override"
        )
    (out / "engine_plan.json").write_text(
        json.dumps(engine, indent=1, sort_keys=True) + "\n"
    )
    policies = policy_plans(args.spec, args.replicate, out, engine["block_size"])
    shutil.copy2(args.engine, out / "engine.json")
    index = {
        "schema": PLAN_SCHEMA,
        "source": source,
        "engine_plan_sha256": sha256_bytes((out / "engine_plan.json").read_bytes()),
        "replicate": args.replicate,
        "policies": [
            {
                "slug": p["slug"],
                "name": p["name"],
                "policy_sha": p["policy_sha"],
                "policy_yaml_sha256": p["policy_yaml_sha256"],
            }
            for p in policies
        ],
    }
    index["plan_id"] = sha256_bytes(canonical_json(index).encode())[:16]
    (out / "PLAN.json").write_text(json.dumps(index, indent=1, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"plan_id": index["plan_id"], "out": str(out), "policies": len(policies)}
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
