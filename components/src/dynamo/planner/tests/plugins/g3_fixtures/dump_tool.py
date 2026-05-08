# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""G3 behavior parity fixture dump / verify tool.

Usage::

    # Generate all fixtures
    python -m dynamo.planner.tests.plugins.g3_fixtures.dump_tool

    # Single scenario
    python -m dynamo.planner.tests.plugins.g3_fixtures.dump_tool \\
        --scenario baseline_disagg_throughput_only_sla

    # Verify-only mode (no write; re-run + compare to existing fixtures)
    python -m dynamo.planner.tests.plugins.g3_fixtures.dump_tool --verify

    # Custom output dir
    python -m dynamo.planner.tests.plugins.g3_fixtures.dump_tool \\
        --output-dir /tmp/g3_fixtures

Output: one JSONL per scenario in ``golden/<scenario_name>.jsonl``.
First line is a metadata header; subsequent lines are tick records:

    {"_meta": {"fixture_format_version": 1, "git_commit": "...",
               "python_version": "...", "scenario": "...",
               "config": {...}, "caps": {...}}}
    {"tick_index": 0, "now_s": 5.0, "tick_input": {...},
     "scheduled_tick": {...}, "planner_effects": {...}}
    {"tick_index": 1, ...}
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from dynamo.planner.core.state_machine import PlannerStateMachine
from dynamo.planner.core.types import ScheduledTick, TickInput

from .scenarios import ALL_SCENARIOS, Scenario, find_scenario
from .serializers import (
    encode_planner_effects,
    encode_scheduled_tick,
    encode_tick_input,
    encode_worker_capabilities,
)

logger = logging.getLogger(__name__)

FIXTURE_FORMAT_VERSION = 1
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "golden"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tick_for(tick_input: TickInput) -> ScheduledTick:
    """Build a ScheduledTick matching the data present in a TickInput.

    Mirrors ``tests/unit/test_state_machine.py::_tick_for``—keep in sync.
    """
    has_fpm = tick_input.fpm_observations is not None
    has_traffic = tick_input.traffic is not None
    return ScheduledTick(
        at_s=tick_input.now_s,
        run_load_scaling=has_fpm,
        run_throughput_scaling=has_traffic,
        need_worker_states=True,
        need_worker_fpm=has_fpm,
        need_traffic_metrics=has_traffic,
        traffic_metrics_duration_s=tick_input.traffic.duration_s
        if has_traffic
        else 0.0,
    )


def _git_commit() -> str:
    """Best-effort: return current git HEAD commit hash. Empty string on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _python_version() -> str:
    return sys.version.split()[0]


def _config_to_dict(config) -> dict:
    """Pydantic config -> dict for fixture metadata."""
    if hasattr(config, "model_dump"):
        return config.model_dump()
    return asdict(config)


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------


def _run_scenario(scenario: Scenario) -> list[dict]:
    """Run a scenario through PSM and capture (input, scheduled, effects) tuples.

    Returns:
        List of dict records, first record being metadata header.
    """
    config = scenario.make_config()
    caps = scenario.caps_factory()
    core = PlannerStateMachine(config, caps)

    # Bootstrap (load_benchmark_fpms / warm_load_predictors).
    if scenario.bootstrap_fn is not None:
        scenario.bootstrap_fn(core)

    # initial_tick - record but don't pass to on_tick (that's done by tick loop)
    initial_tick = core.initial_tick(scenario.initial_tick_at_s)

    records: list[dict] = []

    # Metadata header (first JSONL line)
    records.append({
        "_meta": {
            "fixture_format_version": FIXTURE_FORMAT_VERSION,
            "git_commit": _git_commit(),
            "python_version": _python_version(),
            "scenario": scenario.name,
            "description": scenario.description,
            "config": _config_to_dict(config),
            "caps": encode_worker_capabilities(caps),
            "initial_tick_at_s": scenario.initial_tick_at_s,
            "initial_tick": encode_scheduled_tick(initial_tick),
            "num_ticks": len(scenario.ticks),
        }
    })

    # Tick loop
    for tick_idx, tick_input in enumerate(scenario.ticks):
        scheduled_tick = _tick_for(tick_input)
        effects = core.on_tick(scheduled_tick, tick_input)
        records.append({
            "tick_index": tick_idx,
            "now_s": tick_input.now_s,
            "tick_input": encode_tick_input(tick_input),
            "scheduled_tick": encode_scheduled_tick(scheduled_tick),
            "planner_effects": encode_planner_effects(effects),
        })

    return records


def _write_fixture(scenario: Scenario, records: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{scenario.name}.jsonl"
    with output_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True, ensure_ascii=False))
            f.write("\n")
    return output_path


def _read_fixture(path: Path) -> list[dict]:
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _compare_records(expected: list[dict], actual: list[dict], scenario_name: str) -> list[str]:
    """Compare two record lists; return list of diff descriptions (empty = match)."""
    diffs: list[str] = []
    if len(expected) != len(actual):
        diffs.append(
            f"[{scenario_name}] record count: expected {len(expected)}, got {len(actual)}"
        )
        return diffs

    # Skip metadata header (records[0]); compare tick records only.
    for i, (exp_rec, act_rec) in enumerate(zip(expected[1:], actual[1:])):
        if exp_rec != act_rec:
            diffs.append(
                f"[{scenario_name}] tick {i} diff:\n"
                f"  expected: {json.dumps(exp_rec, sort_keys=True)[:300]}...\n"
                f"  actual:   {json.dumps(act_rec, sort_keys=True)[:300]}..."
            )
    return diffs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="G3 behavior parity fixture dump/verify tool"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Run a single scenario by name (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify mode: re-run scenarios + compare to existing fixtures (no write)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available scenarios and exit",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()

    if args.list:
        print(f"Available scenarios ({len(ALL_SCENARIOS)}):")
        for s in ALL_SCENARIOS:
            print(f"  - {s.name}")
            print(f"      {s.description.splitlines()[0]}")
        return 0

    # Resolve scenarios to run
    if args.scenario:
        scenario = find_scenario(args.scenario)
        if scenario is None:
            print(f"ERROR: scenario not found: {args.scenario}", file=sys.stderr)
            return 2
        scenarios = [scenario]
    else:
        scenarios = ALL_SCENARIOS

    if args.verify:
        # Verify mode: re-run + compare
        all_diffs: list[str] = []
        for scenario in scenarios:
            fixture_path = args.output_dir / f"{scenario.name}.jsonl"
            if not fixture_path.exists():
                all_diffs.append(f"[{scenario.name}] missing fixture: {fixture_path}")
                continue
            expected = _read_fixture(fixture_path)
            actual = _run_scenario(scenario)
            diffs = _compare_records(expected, actual, scenario.name)
            if diffs:
                all_diffs.extend(diffs)
            else:
                logger.info(f"  ✓ {scenario.name}")
        if all_diffs:
            print("\nVERIFY FAILED:")
            for d in all_diffs:
                print(d)
            return 1
        print(f"\nVERIFY OK: all {len(scenarios)} scenarios match fixtures.")
        return 0

    # Generate mode: run + write
    for scenario in scenarios:
        records = _run_scenario(scenario)
        fixture_path = _write_fixture(scenario, records, args.output_dir)
        logger.info(
            f"  wrote {scenario.name} -> {fixture_path} ({len(records) - 1} ticks)"
        )

    print(f"\nGenerated {len(scenarios)} fixtures in {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
