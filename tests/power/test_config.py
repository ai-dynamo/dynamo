# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the power_control block, FoM specs, GPU-group layouts and shared rules."""

import pytest

from dynamo.power import config as power_config
from dynamo.power.config import (
    FomSpec,
    MetricsUrl,
    NodeAgentSettings,
    PowerControlConfig,
    PowerControlConfigError,
    WorkerPlacement,
    build_pool_layouts,
    check_stale_window,
    check_visible_gpus,
    epd_environment,
    nupd_budget_binds,
    parse_fom,
    parse_gpu_indices,
    parse_metrics_urls,
    parse_nodelist,
    stale_after_s_for,
    urls_for_scope,
    validate_gpu_identity,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

FOUR = (0, 1, 2, 3)
FRONTEND = "http://frontend:8000/metrics"


def _block(**overrides):
    block = {
        "enabled": True,
        "mode": "nupd",
        "pools": "shared",
        "budget_w_per_gpu": 1500,
        "fom": "tokens_per_gpu",
    }
    block.update(overrides)
    return block


def _tep6_placements():
    """AgentX DSV4: 5 TEP6 prefill workers on 8 four-GPU nodes + one DEP16 decode."""
    prefill = [
        ("P0", "A", FOUR),
        ("P0", "B", (0, 1)),
        ("P1", "B", (2, 3)),
        ("P1", "C", FOUR),
        ("P2", "D", FOUR),
        ("P2", "E", (0, 1)),
        ("P3", "E", (2, 3)),
        ("P3", "F", FOUR),
        ("P4", "G", FOUR),
        ("P4", "H", (0, 1)),
    ]
    decode = [("D0", node, FOUR) for node in ("I", "J", "K", "L")]
    return [
        WorkerPlacement(role="prefill", node=node, gpu_indices=gpus, worker=worker)
        for worker, node, gpus in prefill
    ] + [
        WorkerPlacement(role="decode", node=node, gpu_indices=gpus, worker=worker)
        for worker, node, gpus in decode
    ]


def _settings(**overrides) -> NodeAgentSettings:
    values = {
        "pool_name": "shared",
        "rank": 0,
        "nodelist": ("hecate0001", "hecate0002"),
        "log_dir": "/logs/power_control/shared",
        "gpu_indices": FOUR,
        "mode": "nupd",
        "budget_w_per_gpu": 1500,
        "fom": parse_fom("tokens_per_gpu"),
        "metrics_urls": parse_metrics_urls([FRONTEND]),
    }
    values.update(overrides)
    return NodeAgentSettings(**values)


# ------------------------------------------------------------------ FoM specs


@pytest.mark.parametrize("name", power_config.FOM_PRESETS)
def test_parse_fom_preset(name):
    spec = parse_fom(name)
    assert spec == FomSpec(source="preset", name=name)
    assert spec.to_cli() == name


def test_parse_fom_prom_keeps_colons_in_metric_names_and_parses_labels():
    text = "prom:vllm:generation_tokens_total:model=foo,engine=bar:counter:sum"
    spec = parse_fom(text)
    assert spec.source == "prom"
    assert spec.name == "vllm:generation_tokens_total"
    assert spec.labels == (("model", "foo"), ("engine", "bar"))
    assert spec.metric_kind == "counter"
    assert spec.aggregation == "sum"
    assert spec.scope == "all"
    assert spec.to_cli() == text


def test_parse_fom_prom_without_labels():
    spec = parse_fom("prom:dynamo_frontend_inflight_requests:gauge:avg")
    assert spec.name == "dynamo_frontend_inflight_requests"
    assert spec.labels == ()
    assert spec.metric_kind == "gauge"
    assert spec.aggregation == "avg"


def test_parse_fom_prom_scope_field_round_trips():
    text = "prom:vllm:generation_tokens_total:model=foo:scope=pool_workers:counter:sum"
    spec = parse_fom(text)
    assert spec.name == "vllm:generation_tokens_total"
    assert spec.labels == (("model", "foo"),)
    assert spec.scope == "pool_workers"
    assert spec.to_cli() == text
    bare = parse_fom("prom:dynamo_frontend_requests_total:scope=frontend:counter:sum")
    assert (bare.scope, bare.labels) == ("frontend", ())
    assert parse_fom(bare.to_cli()) == bare
    mapping = parse_fom(
        {
            "metric": "x_total",
            "kind": "counter",
            "labels": {"model_name": "m"},
            "scope": "pool_workers",
        }
    )
    assert parse_fom(mapping.to_cli()) == mapping
    assert "on pool_workers" in mapping.describe()


def test_parse_fom_python():
    spec = parse_fom("python:tests.power.fake_provider:FakeProvider")
    assert spec.source == "python"
    assert spec.name == "tests.power.fake_provider:FakeProvider"
    assert spec.to_cli() == "python:tests.power.fake_provider:FakeProvider"


@pytest.mark.parametrize(
    "text",
    [
        "bogus",
        "prom:x",
        "prom:metric:histogram:sum",
        "prom:metric:counter:max",
        "prom:bad metric:counter:sum",
        "prom::counter:sum",
        "prom:m:scope=everywhere:counter:sum",
        "prom:m:scope=frontend,model=a:counter:sum",
        "prom:m:scope=frontend:model=a:counter:sum",
        "python:NoColon",
        "python:1bad:Class",
        "python:pkg.mod:not-an-identifier",
    ],
)
def test_parse_fom_rejects_bad_specs(text):
    with pytest.raises(PowerControlConfigError):
        parse_fom(text)


def test_parse_fom_rejects_a_label_named_scope():
    with pytest.raises(PowerControlConfigError, match="reserved"):
        parse_fom({"metric": "x", "kind": "gauge", "labels": {"scope": "a"}})


def test_parse_fom_mapping_forms():
    assert parse_fom({"preset": "output_tokens"}) == FomSpec("preset", "output_tokens")
    prom = parse_fom(
        {
            "metric": "trtllm_request_success_total",
            "kind": "counter",
            "agg": "sum",
            "labels": {"model_name": "m"},
            "scope": "pool_workers",
        }
    )
    assert prom.name == "trtllm_request_success_total"
    assert prom.labels == (("model_name", "m"),)
    assert prom.scope == "pool_workers"
    assert parse_fom({"provider": "pkg.mod:Cls"}).source == "python"
    with pytest.raises(PowerControlConfigError):
        parse_fom({"metric": "x"})
    with pytest.raises(PowerControlConfigError):
        parse_fom({"preset": "output_tokens", "kind": "counter"})
    with pytest.raises(PowerControlConfigError):
        parse_fom(42)


# --------------------------------------------------------- power_control block


def test_from_dict_minimal_shared_block():
    cfg = PowerControlConfig.from_dict(_block())
    assert cfg.enabled is True
    assert cfg.mode == "nupd"
    assert cfg.use_ubpc is False
    assert cfg.pools == "shared"
    assert cfg.budget_w_per_gpu == 1500
    assert cfg.fom == FomSpec("preset", "tokens_per_gpu")
    assert cfg.interval_s == 30
    assert cfg.stale_after_s == 90
    assert cfg.perf_threshold == pytest.approx(0.03)
    assert dict(cfg.ubpc_buffer) == {"prefill": 2.0, "decode": 1.1}
    assert cfg.min_tgp_w is None
    assert cfg.warmup_policy == "adaptive"
    assert cfg.applier == "sudo-smi"
    assert cfg.critical is True


def test_from_dict_full_block_from_the_contract():
    cfg = PowerControlConfig.from_dict(
        {
            "enabled": True,
            "mode": "both",
            "pools": "per-role",
            "budget_w_per_gpu": 1500,
            "fom": "tokens_per_gpu",
            "interval_s": 30,
            "perf_threshold": 0.03,
            "ubpc_buffer": {"prefill": 2.0, "decode": 1.1},
            "min_tgp_w": None,
            "warmup_policy": "adaptive",
            "applier": "sudo-smi",
            "epd_source": "/lustre/x/nvidia-epd/src/epd",
            "dynamo_power_source": "/lustre/x/dynamo/components/src/dynamo/power",
            "critical": True,
        }
    )
    assert cfg.pools == "per-role"
    assert cfg.use_ubpc is True
    assert cfg.epd_source == "/lustre/x/nvidia-epd/src/epd"
    assert cfg.dynamo_power_source.endswith("dynamo/power")


def test_from_dict_requires_fom():
    block = _block()
    del block["fom"]
    with pytest.raises(PowerControlConfigError, match="fom is required"):
        PowerControlConfig.from_dict(block)


def test_from_dict_requires_budget_for_shorthand_pools():
    block = _block()
    del block["budget_w_per_gpu"]
    with pytest.raises(PowerControlConfigError, match="budget_w_per_gpu is required"):
        PowerControlConfig.from_dict(block)


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(PowerControlConfigError, match="unknown key"):
        PowerControlConfig.from_dict(_block(budget_w=1500))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("mode", "turbo"),
        ("applier", "bmc"),
        ("warmup_policy", "slow"),
        ("perf_threshold", 1.5),
        ("perf_threshold", 0),
        ("interval_s", 0),
        ("budget_w_per_gpu", -5),
        ("budget_w_per_gpu", True),
        ("ubpc_buffer", 0),
        ("ubpc_buffer", {}),
        ("critical", "yes"),
        ("pools", "everything"),
        ("pools", []),
        ("min_tgp_w", -1),
        ("min_tgp_w", 0),
        ("min_tgp_w", 1600),  # above the 1500 W budget
    ],
)
def test_from_dict_rejects_bad_values(key, value):
    with pytest.raises(PowerControlConfigError):
        PowerControlConfig.from_dict(_block(**{key: value}))


def test_from_dict_floor_is_kept_and_checked_against_the_budget():
    cfg = PowerControlConfig.from_dict(_block(min_tgp_w=900))
    assert cfg.min_tgp_w == 900
    (pool,) = power_config.resolve_pools(cfg, _tep6_placements()[:1])
    assert pool.min_tgp_w == 900
    with pytest.raises(PowerControlConfigError, match="above budget_w_per_gpu"):
        PowerControlConfig.from_dict(
            _block(
                min_tgp_w=1400,
                pools=[{"name": "d", "roles": ["decode"], "budget_w_per_gpu": 1300}],
            )
        )
    with pytest.raises(PowerControlConfigError, match="pool d: min_tgp_w"):
        PowerControlConfig.from_dict(
            _block(
                pools=[
                    {
                        "name": "d",
                        "roles": ["decode"],
                        "budget_w_per_gpu": 1300,
                        "min_tgp_w": 1400,
                    }
                ]
            )
        )


def test_from_dict_explicit_pools_with_their_own_fom_and_budget():
    block = {
        "enabled": True,
        "mode": "ubpc",
        "pools": [
            {
                "name": "prefill",
                "roles": ["prefill"],
                "budget_w_per_gpu": 1800,
                "fom": "ctx_tokens",
            },
            {
                "name": "decode",
                "roles": ["decode"],
                "budget_w_per_gpu": 1300,
                "fom": "gen_tokens",
                "ubpc_buffer": 1.05,
                "min_tgp_w": 900,
            },
        ],
    }
    cfg = PowerControlConfig.from_dict(block)
    prefill, decode = cfg.pools
    assert prefill.budget_w_per_gpu == 1800
    assert prefill.fom.name == "ctx_tokens"
    assert prefill.ubpc_buffer == 2.0  # from the default prefill buffer
    assert prefill.min_tgp_w is None
    assert decode.ubpc_buffer == 1.05
    assert decode.min_tgp_w == 900


def test_from_dict_explicit_pool_inherits_top_level_defaults():
    cfg = PowerControlConfig.from_dict(
        _block(pools=[{"name": "everything", "nodes": ["n1", "n2"]}])
    )
    (pool,) = cfg.pools
    assert pool.nodes == ("n1", "n2")
    assert pool.budget_w_per_gpu == 1500
    assert pool.fom.name == "tokens_per_gpu"
    assert pool.ubpc_buffer == pytest.approx(power_config.EPD_DEFAULT_UBPC_BUFFER)


@pytest.mark.parametrize(
    "pool",
    [
        {"name": "x"},
        {"name": "x", "roles": ["prefill"], "nodes": ["n1"]},
        {"roles": ["prefill"]},
        {"name": "x", "roles": ["prefill"], "colour": "red"},
    ],
)
def test_from_dict_explicit_pool_validation(pool):
    with pytest.raises(PowerControlConfigError):
        PowerControlConfig.from_dict(_block(pools=[pool]))


def test_from_dict_explicit_pool_without_any_fom_is_rejected():
    block = _block(pools=[{"name": "x", "roles": ["prefill"]}])
    del block["fom"]
    with pytest.raises(PowerControlConfigError, match="fom is required"):
        PowerControlConfig.from_dict(block)


def test_from_dict_duplicate_pool_names_rejected():
    pools = [{"name": "x", "roles": ["prefill"]}, {"name": "x", "roles": ["decode"]}]
    with pytest.raises(PowerControlConfigError, match="duplicate pool names"):
        PowerControlConfig.from_dict(_block(pools=pools))


def test_buffer_for_roles():
    cfg = PowerControlConfig.from_dict(_block())
    assert cfg.buffer_for(("prefill", "decode")) == 2.0
    assert cfg.buffer_for(("decode",)) == 1.1
    assert cfg.buffer_for(("encode",)) == pytest.approx(
        power_config.EPD_DEFAULT_UBPC_BUFFER
    )
    scalar = PowerControlConfig.from_dict(_block(ubpc_buffer=1.3))
    assert scalar.buffer_for(("prefill", "decode")) == pytest.approx(1.3)


def test_ubpc_and_both_configure_epd_identically():
    ubpc = PowerControlConfig.from_dict(_block(mode="ubpc"))
    both = PowerControlConfig.from_dict(_block(mode="both"))
    assert ubpc.use_ubpc and both.use_ubpc
    assert epd_environment(
        _settings(mode="ubpc"), this_node="hecate0001", max_tgp_w=2300.0
    ) == epd_environment(
        _settings(mode="both"), this_node="hecate0001", max_tgp_w=2300.0
    )


# ---------------------------------------------------------------- shared rules


def test_stale_window_follows_the_controller_interval():
    assert stale_after_s_for(30) == 90
    assert stale_after_s_for(5) == 90
    assert stale_after_s_for(60) == 180
    assert PowerControlConfig.from_dict(_block(interval_s=90)).stale_after_s == 270
    for interval_s in (1, 5, 30, 90, 600):
        check_stale_window(stale_after_s_for(interval_s), interval_s)
    check_stale_window(60, 30)
    with pytest.raises(PowerControlConfigError, match="flap"):
        check_stale_window(59, 30)


def test_validate_gpu_identity():
    assert validate_gpu_identity(4, {0: 0, 1: 1, 2: 2, 3: 3}, (0, 1, 2, 3)) == []
    assert validate_gpu_identity(8, {2: 2, 3: 3}, (2, 3)) == []
    with pytest.raises(PowerControlConfigError, match="does not exist"):
        validate_gpu_identity(2, {0: 0, 1: 1}, (0, 1, 2))


def test_validate_gpu_identity_reports_minor_numbers_without_refusing():
    # VR200 nodes enumerate NVML device 0 as /dev/nvidia2: caps go by NVML index, so this is a note.
    notes = validate_gpu_identity(4, {0: 2, 1: 0, 2: 3, 3: 1}, (0, 1, 2, 3))
    assert len(notes) == 4
    assert "NVML device 0 is /dev/nvidia2" in notes[0]
    assert validate_gpu_identity(4, {}, (0,)) == [
        "GPU index 0: NVML reported no minor number"
    ]


def test_nupd_budget_binds_only_below_the_max_tgp():
    assert nupd_budget_binds(1500, 2300.0)
    assert not nupd_budget_binds(2300, 2300.0)
    assert not nupd_budget_binds(2400, 2300.0)


# ------------------------------------------------------------------- layouts


def test_shared_layout_merges_tep6_workers_sharing_a_node():
    cfg = PowerControlConfig.from_dict(_block())
    (layout,) = build_pool_layouts(cfg, _tep6_placements())
    assert layout.name == "shared"
    assert layout.nodes == tuple("ABCDEFGHIJKL")
    assert layout.controller_node == "A"
    assert layout.rank_of("B") == 1
    assert layout.group_for("B").gpu_indices == FOUR  # P0's tail + P1's head
    assert layout.group_for("H").gpu_indices == (0, 1)  # two idle GPUs excluded
    assert layout.gpu_count == 30 + 16
    assert layout.spec.ubpc_buffer == 2.0  # largest buffer across the roles


def test_per_role_layout_builds_one_pool_per_role():
    cfg = PowerControlConfig.from_dict(_block(pools="per-role", mode="both"))
    prefill, decode = build_pool_layouts(cfg, _tep6_placements())
    assert prefill.name == "prefill"
    assert prefill.gpu_count == 30
    assert prefill.spec.ubpc_buffer == 2.0
    assert decode.name == "decode"
    assert decode.nodes == ("I", "J", "K", "L")
    assert decode.controller_node == "I"
    assert decode.gpu_count == 16
    assert decode.spec.ubpc_buffer == 1.1


def test_node_split_between_two_pools_is_rejected():
    pools = [{"name": "x", "nodes": ["A", "B"]}, {"name": "y", "nodes": ["B", "C"]}]
    cfg = PowerControlConfig.from_dict(_block(pools=pools))
    placements = [
        WorkerPlacement("prefill", "A", FOUR),
        WorkerPlacement("prefill", "B", (0, 1)),
        WorkerPlacement("prefill", "B", (2, 3)),
        WorkerPlacement("prefill", "C", FOUR),
    ]
    with pytest.raises(PowerControlConfigError, match="pools must not overlap"):
        build_pool_layouts(cfg, placements)


def test_colocated_roles_on_one_node_rejected_with_per_role_pools():
    cfg = PowerControlConfig.from_dict(_block(pools="per-role"))
    placements = [
        WorkerPlacement("prefill", "N", (0, 1)),
        WorkerPlacement("decode", "N", (2, 3)),
    ]
    with pytest.raises(PowerControlConfigError, match="split between pools"):
        build_pool_layouts(cfg, placements)


def test_gpu_claimed_twice_within_a_pool_is_rejected():
    cfg = PowerControlConfig.from_dict(_block())
    placements = [
        WorkerPlacement("prefill", "B", (0, 1), worker="P0"),
        WorkerPlacement("prefill", "B", (1, 2), worker="P1"),
    ]
    with pytest.raises(PowerControlConfigError, match="claimed twice"):
        build_pool_layouts(cfg, placements)


def test_worker_outside_every_pool_is_rejected():
    cfg = PowerControlConfig.from_dict(
        _block(pools=[{"name": "d", "roles": ["decode"]}])
    )
    with pytest.raises(PowerControlConfigError, match="belongs to no pool"):
        build_pool_layouts(cfg, _tep6_placements())


def test_pool_matching_no_worker_is_rejected():
    pools = [
        {"name": "p", "roles": ["prefill"]},
        {"name": "d", "roles": ["decode"]},
        {"name": "e", "roles": ["encode"]},
    ]
    cfg = PowerControlConfig.from_dict(_block(pools=pools))
    with pytest.raises(PowerControlConfigError, match="matches no worker"):
        build_pool_layouts(cfg, _tep6_placements())


def test_empty_placements_rejected():
    cfg = PowerControlConfig.from_dict(_block())
    with pytest.raises(PowerControlConfigError, match="no worker placements"):
        build_pool_layouts(cfg, [])


def test_negative_gpu_index_in_placement_rejected():
    cfg = PowerControlConfig.from_dict(_block())
    with pytest.raises(PowerControlConfigError, match="negative GPU index"):
        build_pool_layouts(cfg, [WorkerPlacement("prefill", "A", (-1, 0))])


# --------------------------------------------------------- GPU visibility check


def test_check_visible_gpus_accepts_restricted_container():
    check_visible_gpus(2, (2, 3))


def test_check_visible_gpus_accepts_the_whole_node():
    # Minor numbers are not consulted: a node whose /dev/nvidiaN order differs from NVML order passes.
    check_visible_gpus(4, (0, 1, 2, 3))


def test_check_visible_gpus_accepts_prefix_of_all_gpus():
    check_visible_gpus(4, (0, 1))


def test_check_visible_gpus_rejects_wrong_first_devices():
    with pytest.raises(PowerControlConfigError, match="NVIDIA_VISIBLE_DEVICES=2,3"):
        check_visible_gpus(4, (2, 3))


def test_check_visible_gpus_rejects_too_few_devices():
    with pytest.raises(PowerControlConfigError, match="sees 1 GPUs"):
        check_visible_gpus(1, (0, 1))


# ------------------------------------------------------------- metrics URLs


def test_parse_metrics_urls_keeps_tags_and_flattens():
    urls = parse_metrics_urls(
        [
            "frontend=http://f:8000/metrics,http://w1:9090/metrics",
            "worker=http://w2:9090/metrics?x=1",
        ]
    )
    assert urls == (
        MetricsUrl("http://f:8000/metrics", "frontend"),
        MetricsUrl("http://w1:9090/metrics"),
        MetricsUrl("http://w2:9090/metrics?x=1", "worker"),
    )
    assert [entry.to_cli() for entry in urls] == [
        "frontend=http://f:8000/metrics",
        "http://w1:9090/metrics",
        "worker=http://w2:9090/metrics?x=1",
    ]
    assert urls_for_scope(urls, "all") == (
        "http://f:8000/metrics",
        "http://w1:9090/metrics",
        "http://w2:9090/metrics?x=1",
    )
    assert urls_for_scope(urls, "frontend") == ("http://f:8000/metrics",)
    assert urls_for_scope(urls, "pool_workers") == ("http://w2:9090/metrics?x=1",)
    with pytest.raises(PowerControlConfigError):
        parse_metrics_urls(["frontend:8000/metrics"])
    with pytest.raises(PowerControlConfigError):
        parse_metrics_urls(["worker=w2:9090/metrics"])
    with pytest.raises(PowerControlConfigError):
        urls_for_scope(urls, "everywhere")


# ---------------------------------------------------- node_agent settings/env


def test_settings_defaults_and_pool_gpu_count():
    settings = _settings()
    assert settings.is_controller
    assert settings.use_ubpc is False
    assert settings.interval_s == power_config.DEFAULT_INTERVAL_S
    assert settings.ubpc_buffer == power_config.DEFAULT_NODE_UBPC_BUFFER
    assert settings.pool_gpu_count == 8
    assert _settings(gpus_per_node=4, gpu_indices=(0, 1)).pool_gpu_count == 8
    assert _settings(gpu_indices=(0, 1)).pool_gpu_count == 4
    # The launcher's exact total wins over the nodes x GPUs estimate (TEP6 tails).
    assert _settings(gpus_per_node=4, pool_gpu_count_override=6).pool_gpu_count == 6
    assert not hasattr(settings, "min_tgp_w")


@pytest.mark.parametrize(
    "overrides",
    [
        {"rank": 2},
        {"rank": -1},
        {"metrics_urls": ()},
        {"metrics_urls": (FRONTEND,)},  # plain strings: use parse_metrics_urls
        {"interval_s": 0},
        {"perf_threshold": 1.0},
        {"ubpc_buffer": 0.0},
        {"budget_w_per_gpu": 0},
        {"mode": "turbo"},
        {"warmup_policy": "slow"},
        {"gpu_indices": (1, 1)},
        {"gpus_per_node": 0},
        {"pool_gpu_count_override": 0},
    ],
)
def test_settings_validation(overrides):
    with pytest.raises(PowerControlConfigError):
        _settings(**overrides)


def test_non_controller_rank_needs_no_metrics_url():
    settings = _settings(rank=1, metrics_urls=())
    assert not settings.is_controller


def test_controller_scope_needs_matching_tagged_urls():
    worker_only = parse_metrics_urls(["worker=http://w1:9090/metrics"])
    frontend_only = parse_metrics_urls(["frontend=http://f:8000/metrics"])
    untagged = parse_metrics_urls(["http://w1:9090/metrics"])
    pool_workers = parse_fom(
        "prom:vllm:generation_tokens_total:scope=pool_workers:counter:sum"
    )
    frontend = parse_fom(
        {
            "metric": "dynamo_frontend_requests_total",
            "kind": "counter",
            "scope": "frontend",
        }
    )
    assert _settings(
        fom=pool_workers, metrics_urls=worker_only
    ).worker_metrics_urls == ("http://w1:9090/metrics",)
    assert _settings(
        fom=frontend, metrics_urls=frontend_only
    ).frontend_metrics_urls == ("http://f:8000/metrics",)
    with pytest.raises(PowerControlConfigError, match="worker=<url>"):
        _settings(fom=pool_workers, metrics_urls=frontend_only)
    with pytest.raises(PowerControlConfigError, match="frontend=<url>"):
        _settings(fom=frontend, metrics_urls=worker_only)
    with pytest.raises(PowerControlConfigError, match="worker=<url>"):
        _settings(fom=pool_workers, metrics_urls=untagged)
    assert not _settings(rank=1, fom=pool_workers, metrics_urls=()).is_controller
    assert _settings(metrics_urls=untagged).all_metrics_urls == (
        "http://w1:9090/metrics",
    )


def test_epd_environment_uses_epd_variable_names():
    env = epd_environment(_settings(), this_node="hecate0002", max_tgp_w=2300.0)
    assert env["LOG_DIR"] == "/logs/power_control/shared"
    assert env["EPD_LOG_DIR"] == env["LOG_DIR"]
    assert env["EPD_NODELIST"] == "hecate0001,hecate0002"
    assert env["EPD_NUM_NODES"] == "2"
    assert env["SLURMD_NODENAME"] == "hecate0002"
    assert env["GPUS_PER_NODE"] == "4"
    assert env["MAX_TGP"] == "2300"
    assert env["AVERAGE_GPU_POWER_LIMIT"] == "1500"
    assert env["MIN_TGP"] == "0"
    assert env["NUPD_SKIP_PERFORMANCE_FEEDBACK"] == "0"
    assert env["NUPD_SKIP_HISTORY"] == "1"
    assert env["NUPD_SKIP_UBPC"] == "1"
    assert env["EPD_PERF_THRESHOLD"] == "0.03"
    assert env["UBPC_BUFFER"] == "1.1"
    assert env["EPD_WARMUP_POLICY"] == "adaptive"
    assert env["EPD_ACTUATOR"] == "file"
    assert env["EPD_RUN_TRACE"] == "1"
    assert env["DYNAMO_POWER_METRICS_URLS"] == FRONTEND
    assert env["DYNAMO_POWER_FRONTEND_METRICS_URLS"] == ""
    assert env["DYNAMO_POWER_WORKER_METRICS_URLS"] == ""


def test_epd_environment_mode_both_keeps_min_tgp_pinned_to_zero():
    both = epd_environment(
        _settings(mode="both", ubpc_buffer=2.0),
        this_node="hecate0001",
        max_tgp_w=1400.0,
    )
    assert both["NUPD_SKIP_UBPC"] == "0"
    assert both["MIN_TGP"] == "0"  # EPD's cap law has no floor; the applier has
    assert both["UBPC_BUFFER"] == "2"
    assert both["MAX_TGP"] == "1400"


def test_epd_environment_splits_tagged_metrics_urls():
    settings = _settings(
        metrics_urls=parse_metrics_urls(
            [
                "frontend=http://f:8000/metrics",
                "worker=http://w1:9090/metrics,worker=http://w2:9090/metrics",
            ]
        )
    )
    env = epd_environment(settings, this_node="hecate0001", max_tgp_w=2300.0)
    assert (
        env["DYNAMO_POWER_METRICS_URLS"]
        == "http://f:8000/metrics,http://w1:9090/metrics,http://w2:9090/metrics"
    )
    assert env["DYNAMO_POWER_FRONTEND_METRICS_URLS"] == "http://f:8000/metrics"
    assert (
        env["DYNAMO_POWER_WORKER_METRICS_URLS"]
        == "http://w1:9090/metrics,http://w2:9090/metrics"
    )


# ------------------------------------------------------------------- parsers


def test_parse_gpu_indices_sorts_and_validates():
    assert parse_gpu_indices("3,1,2") == (1, 2, 3)
    assert parse_gpu_indices(" 0 ") == (0,)
    for bad in ("", "1,1", "a", "-1"):
        with pytest.raises(PowerControlConfigError):
            parse_gpu_indices(bad)


def test_parse_nodelist():
    assert parse_nodelist("hecate0001, hecate0002") == ("hecate0001", "hecate0002")
    with pytest.raises(PowerControlConfigError):
        parse_nodelist("a,a")
    with pytest.raises(PowerControlConfigError):
        parse_nodelist(" ")
