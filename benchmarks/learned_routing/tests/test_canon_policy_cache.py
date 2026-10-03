# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import shutil

import pytest
import yaml
from learned_routing import cache, canon, policy
from learned_routing.cells import Cell, CellError, resolve_replicate
from learned_routing.paths import Layout


# -- canonical YAML -----------------------------------------------------------------------
def test_canonical_yaml_is_key_order_independent_and_round_trips():
    a = {
        "b": [1, 2.5, 1e-05],
        "a": {"z": True, "y": None, "x": "s p"},
        "l": [{"k": 1, "j": [0.1]}],
    }
    b = {
        "l": [{"j": [0.1], "k": 1}],
        "a": {"x": "s p", "y": None, "z": True},
        "b": [1, 2.5, 1e-05],
    }
    text = canon.canonical_yaml(a)
    assert text == canon.canonical_yaml(b)
    assert yaml.safe_load(text) == a  # YAML 1.1 reader agrees, including 1e-05


@pytest.mark.parametrize("value", [1e-05, 1e20, -3.5e-300, 0.1, 1.0, 123456789.123])
def test_floats_are_repr_exact_for_yaml_11_and_12(value):
    text = canon.float_text(value)
    assert float(text) == value
    assert yaml.safe_load(f"x: {text}")["x"] == value


def test_non_finite_floats_are_rejected():
    with pytest.raises(ValueError):
        canon.canonical_json({"x": float("nan")})
    with pytest.raises(ValueError):
        canon.canonical_yaml({"x": [float("inf")]})


# -- policy specs -------------------------------------------------------------------------
def spec(**raw):
    return policy.spec_from_dict(raw, validate_knobs=False)


def test_seed_is_injected_per_replicate_and_not_part_of_identity():
    s = spec(type="dynamo-default-cost-fn", parameters={"overlap_score_credit": 2.0})
    assert "seed" not in s.canonical_text()
    y1, y2 = yaml.safe_load(s.replay_yaml_text(1)), yaml.safe_load(
        s.replay_yaml_text(2)
    )
    assert y1["worker_selection"]["instances"][0]["parameters"] == {
        "overlap_score_credit": 2.0,
        "seed": 1,
    }
    assert y2["worker_selection"]["instances"][0]["parameters"]["seed"] == 2
    assert set(y1) == {
        "worker_selection"
    }  # the replay parser rejects unknown top-level keys
    with pytest.raises(policy.PolicySpecError):
        spec(type="dynamo-default-cost-fn", parameters={"seed": 3})


def test_unseeded_types_get_no_seed_and_seeded_override_works():
    lmetric = spec(type="lmetric")
    assert lmetric.effective_seed(5) is None
    assert "seed" not in lmetric.replay_yaml_text(5)
    forced = spec(type="lmetric", seeded=True)
    assert forced.effective_seed(5) == 5
    assert "seed: 5" in forced.replay_yaml_text(5)
    assert (
        forced.sha == lmetric.sha
    )  # seeding is replicate plumbing, not policy identity


def test_identity_covers_router_config_and_ignores_names():
    base = spec(type="dynamo-default-cost-fn", name="a")
    assert base.sha == spec(type="dynamo-default-cost-fn", name="b").sha
    assert (
        base.sha
        != spec(
            type="dynamo-default-cost-fn", router_config={"router_temperature": 0.5}
        ).sha
    )
    assert (
        base.sha
        != spec(
            type="dynamo-default-cost-fn", parameters={"prefill_load_scale": 2.0}
        ).sha
    )


def test_contract_worker_selection_form_equals_type_form():
    contract = spec(
        worker_selection={
            "aggregated": "candidate",
            "instances": [{"name": "candidate", "type": "lmetric", "parameters": {}}],
        }
    )
    assert contract.sha == spec(type="lmetric").sha
    with pytest.raises(policy.PolicySpecError, match="aggregated"):
        spec(worker_selection={"instances": [{"name": "c", "type": "lmetric"}]})


def test_round_robin_and_untyped_kv_router_rules():
    assert spec(router_mode="round_robin").replay_yaml_text(1) is None
    with pytest.raises(policy.PolicySpecError):
        spec(router_mode="round_robin", router_config={"router_temperature": 1.0})
    with pytest.raises(policy.PolicySpecError, match="unseeded"):
        spec(router_mode="kv_router")


def test_router_config_knobs_are_validated_against_kv_router_config():
    pytest.importorskip("dynamo.llm")
    with pytest.raises(policy.PolicySpecError, match="unknown KvRouterConfig knobs"):
        policy.spec_from_dict({"type": "lmetric", "router_config": {"no_such_knob": 1}})
    ok = policy.spec_from_dict(
        {"type": "lmetric", "router_config": {"router_temperature": 0.5}}
    )
    assert ok.router_config == {"router_temperature": 0.5}


# -- cell identity ------------------------------------------------------------------------
def make_cell(tmp_path, trace_name="t.jsonl", **overrides):
    trace = tmp_path / "traces" / trace_name
    trace.parent.mkdir(parents=True, exist_ok=True)
    if not trace.exists():
        trace.write_text(
            "".join(
                json.dumps(
                    {
                        "timestamp": 3000 * (i // 4),
                        "input_length": 100 + i,
                        "output_length": 4,
                        "hash_ids": [i],
                    }
                )
                + "\n"
                for i in range(12)
            )
        )
    engine = tmp_path / "config" / "engine.json"
    engine.parent.mkdir(parents=True, exist_ok=True)
    if not engine.exists():
        engine.write_text(
            json.dumps(
                {
                    "mock_engine_args": {"block_size": 16},
                    "model": "m",
                    "ais_perf_config": {},
                }
            )
        )
    raw = {
        "cell_id": "c1",
        "split": "train",
        "trace_files": [f"traces/{trace_name}"],
        "trace_format": "mooncake",
        "trace_block_size": 512,
        "load": {"mode": "open_speedup", "value": 1.0},
        "num_workers": 4,
        "sla": {"itl_ms": 30.0, "e2e_slowdown": 5.0},
        "engine_ref": "config/engine.json",
    }
    raw.update(overrides)
    return Cell(raw=raw, layout=Layout(tmp_path))


def test_cell_sha_ignores_location_and_split_but_tracks_content(tmp_path):
    cell = make_cell(tmp_path)
    moved_dir = tmp_path / "elsewhere"
    moved_dir.mkdir()
    shutil.copy(tmp_path / "traces" / "t.jsonl", moved_dir / "t.jsonl")
    moved = make_cell(
        tmp_path,
        split="test",
        holdout_axis="window",
        trace_files=[str(moved_dir / "t.jsonl")],
    )
    assert cell.content_sha() == moved.content_sha()
    assert (
        cell.content_sha()
        != make_cell(tmp_path, sla={"itl_ms": 31.0, "e2e_slowdown": 5.0}).content_sha()
    )
    assert (
        cell.content_sha()
        != make_cell(
            tmp_path, load={"mode": "open_speedup", "value": 2.0}
        ).content_sha()
    )
    other = make_cell(tmp_path, trace_name="u.jsonl")
    (tmp_path / "traces" / "u.jsonl").write_text(
        (tmp_path / "traces" / "t.jsonl").read_text() + "\n"
    )
    assert cell.content_sha() != other.content_sha()


def test_cell_sha_ignores_generator_metadata(tmp_path):
    cell = make_cell(tmp_path)
    annotated = make_cell(
        tmp_path,
        expected_cost_s=9.5,
        measure_trace={"warmup_ms": 240000.0, "window_ms": 480000},
        trace_rows=12,
        cache_pressure_ref={"kv_tokens_per_worker": 301808},
        transform_tag="base",
    )
    assert cell.content_sha() == annotated.content_sha()
    assert (
        cell.content_sha()
        != make_cell(
            tmp_path, measure={"basis": "arrival", "warmup_ms": 10.0}
        ).content_sha()
    )


def test_declared_trace_sha_mismatch_is_an_error(tmp_path):
    cell = make_cell(tmp_path, trace_sha256=["0" * 64])
    with pytest.raises(CellError, match="does not match"):
        cell.content_sha()


def test_replicates_are_shared_across_cells_with_the_same_trace(tmp_path):
    a = make_cell(tmp_path)
    b = make_cell(tmp_path, cell_id="c2", num_workers=8)
    for k in range(3):
        ra, rb = resolve_replicate(a, k), resolve_replicate(b, k)
        assert ra.trace_sha256 == rb.trace_sha256 and ra.policy_seed == k + 1
    assert len({resolve_replicate(a, k).trace_sha256 for k in range(3)}) == 3


# -- cache --------------------------------------------------------------------------------
KEY_FIELDS = dict(
    policy_sha="p",
    cell_id="c",
    cell_sha="s",
    repeat=0,
    protocol="crn-order-v1",
    harness_version="h",
    build_id="b",
)


def test_cache_key_depends_on_every_component():
    base = cache.cache_key(**KEY_FIELDS)
    assert base == cache.cache_key(**dict(KEY_FIELDS))
    for field, value in [
        ("policy_sha", "p2"),
        ("cell_id", "c2"),
        ("cell_sha", "s2"),
        ("repeat", 1),
        ("protocol", "other"),
        ("harness_version", "h2"),
        ("build_id", "b2"),
    ]:
        assert cache.cache_key(**{**KEY_FIELDS, field: value}) != base, field


def test_result_cache_round_trip_refuses_errors_and_wrong_keys(tmp_path):
    store = cache.ResultCache(tmp_path)
    key = cache.cache_key(**KEY_FIELDS)
    record = {"cache_key": key, "goodput_rps_window": 1.5, "error": None}
    store.put(key, record)
    assert store.get(key) == record
    assert list(store.keys()) == [key]
    assert not list(tmp_path.rglob("*.tmp"))
    with pytest.raises(ValueError):
        store.put(key, {**record, "error": "boom"})
    with pytest.raises(ValueError):
        store.put(key, {**record, "cache_key": "other"})


def test_build_id_tracks_extension_content(tmp_path):
    so = tmp_path / "_core.abi3.so"
    so.write_bytes(b"one")
    first = cache.bindings_build_id(tmp_path / "cache", so)
    assert cache.bindings_build_id(tmp_path / "cache", so) == first
    so.write_bytes(b"two!")  # rebuilt: different size and content
    second = cache.bindings_build_id(tmp_path / "cache", so)
    assert second["build_id"] != first["build_id"]
    assert second["core_so_sha256"] != first["core_so_sha256"]
