# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-rank NIXL exporter ports for co-located SGLang schedulers.

These tests replay the arguments SGLang hands each scheduler process rather
than calling the derivation with hand-picked ranks, because what has to hold is
that every scheduler a real launch starts gets a port of its own inside the
range the pod reserves.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

import dynamo.sglang._compat as sglang_compat
from dynamo.common.utils.nixl_telemetry import NIXL_TELEMETRY_PROMETHEUS_PORT_ENV
from dynamo.sglang.nixl_telemetry import (
    _assign_nixl_prometheus_port,
    install_per_rank_nixl_prometheus_ports,
    run_scheduler_process_with_nixl_port,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

BASE_PORT = 19090


def _server_args(**overrides) -> SimpleNamespace:
    """A ServerArgs stand-in carrying only the fields the derivation reads."""
    fields = {
        "nnodes": 1,
        "node_rank": 0,
        "tp_size": 1,
        "pp_size": 1,
        "dp_size": 1,
        "base_gpu_id": 0,
        "gpu_id_step": 1,
        "enable_dp_attention": False,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _run_scheduler_process(
    server_args,
    port_args,
    gpu_id,
    tp_rank,
    attn_cp_rank,
    moe_dp_rank,
    moe_ep_rank,
    pp_rank,
    dp_rank,
    pipe_writer,
):
    """Stands in for ``sglang.srt.managers.scheduler.run_scheduler_process``.

    Mirrors that function's parameter list because the wrapper binds the call by
    signature: it reads the rank arguments by name out of a purely positional
    call, so the position of every other parameter matters too.
    """


def _scheduler_calls(server_args) -> list[tuple[int, int, int, int | None]]:
    """The ``(gpu_id, tp_rank, pp_rank, dp_rank)`` tuples one node launches.

    Mirrors ``DataParallelController.launch_dp_schedulers`` and
    ``launch_tensor_parallel_group``, which the non-data-parallel path in
    ``Engine._launch_subprocesses`` reproduces with ``dp_rank`` left unset.
    Under ``--enable-dp-attention`` there is one launch group and the scheduler
    is handed a ``dp_rank`` recomputed from its ``tp_rank``.
    """
    pp_size_per_node = max(server_args.pp_size // server_args.nnodes, 1)
    nnodes_per_pp_rank = max(server_args.nnodes // server_args.pp_size, 1)
    tp_size_per_node = server_args.tp_size // nnodes_per_pp_rank
    pp_ranks = range(
        pp_size_per_node * (server_args.node_rank // nnodes_per_pp_rank),
        pp_size_per_node * (server_args.node_rank // nnodes_per_pp_rank + 1),
    )
    tp_ranks = range(
        tp_size_per_node * (server_args.node_rank % nnodes_per_pp_rank),
        tp_size_per_node * (server_args.node_rank % nnodes_per_pp_rank + 1),
    )

    attn_tp_size = max(server_args.tp_size // server_args.dp_size, 1)
    dp_groups: list[tuple[int, int | None]] = [(0, None)]
    if server_args.dp_size > 1 and not server_args.enable_dp_attention:
        dp_groups = [
            (
                dp_rank
                * server_args.tp_size
                * server_args.pp_size
                * server_args.gpu_id_step,
                dp_rank,
            )
            for dp_rank in range(server_args.dp_size)
        ]

    calls = []
    for group_gpu_offset, dp_rank in dp_groups:
        for pp_rank in pp_ranks:
            for tp_rank in tp_ranks:
                gpu_id = (
                    server_args.base_gpu_id
                    + group_gpu_offset
                    + (pp_rank % pp_size_per_node) * tp_size_per_node
                    + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
                )
                launched_dp_rank = dp_rank
                if server_args.enable_dp_attention:
                    launched_dp_rank = tp_rank // attn_tp_size
                calls.append((gpu_id, tp_rank, pp_rank, launched_dp_rank))
    return calls


def _port_for_scheduler(
    server_args, gpu_id, tp_rank, pp_rank, dp_rank, base_port=BASE_PORT
) -> int:
    """The exporter port the wrapper installs in one scheduler's process.

    SGLang calls the scheduler entry point entirely positionally, so this passes
    positionally too rather than by keyword.
    """
    # Every scheduler is its own process and reads the base the operator
    # injected. Sharing one interpreter across a launch would instead let each
    # call read the port the previous call installed.
    os.environ[NIXL_TELEMETRY_PROMETHEUS_PORT_ENV] = str(base_port)

    port_args, attn_cp_rank, moe_dp_rank, moe_ep_rank = SimpleNamespace(), 0, 0, 0
    _assign_nixl_prometheus_port(
        _run_scheduler_process,
        (
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            attn_cp_rank,
            moe_dp_rank,
            moe_ep_rank,
            pp_rank,
            dp_rank,
            None,
        ),
        {},
    )
    return int(os.environ[NIXL_TELEMETRY_PROMETHEUS_PORT_ENV])


def _ports_for_launch(server_args, base_port=BASE_PORT) -> list[int]:
    return [
        _port_for_scheduler(server_args, *call, base_port=base_port)
        for call in _scheduler_calls(server_args)
    ]


class _LazyProxyModule:
    """Stands in for ``sglang``, whose ``Engine`` is a lazy import proxy.

    A proxy makes the read look harmless -- it succeeds -- and then absorbs the
    write that follows instead of passing it to the class. Raising on any
    attribute turns that silent no-op back into a test failure.
    """

    def __getattr__(self, name: str):
        raise AssertionError(f"install must not reach sglang.{name}")


@pytest.fixture
def telemetry_env(monkeypatch):
    monkeypatch.setenv("NIXL_TELEMETRY_ENABLE", "y")
    monkeypatch.setenv("NIXL_TELEMETRY_EXPORTER", "prometheus")
    monkeypatch.setenv(NIXL_TELEMETRY_PROMETHEUS_PORT_ENV, str(BASE_PORT))
    monkeypatch.setenv("DYN_SYSTEM_PORT", "9090")
    monkeypatch.setenv("DYN_FORWARDPASS_METRIC_PORT", "20380")
    # The stubs below carry effective values already, so pin resolution to the
    # identity rather than let the installed SGLang project a stand-in object.
    # The resolved-configuration case supplies its own projection.
    monkeypatch.setattr(sglang_compat, "sglang_resolved_view", None)
    return monkeypatch


class TestPerRankPortAssignment:
    @pytest.mark.parametrize(
        "name,server_args,expected_ranks",
        [
            ("tensor parallel", _server_args(tp_size=8), 8),
            ("data parallel", _server_args(tp_size=1, dp_size=8), 8),
            (
                "attention data parallel",
                _server_args(tp_size=8, dp_size=8, enable_dp_attention=True),
                8,
            ),
            ("offset devices", _server_args(tp_size=4, base_gpu_id=4), 4),
            (
                "stepped pipeline with data parallelism",
                _server_args(tp_size=1, pp_size=2, dp_size=2, gpu_id_step=2),
                4,
            ),
            (
                "stepped pipeline with tensor parallelism",
                _server_args(tp_size=4, pp_size=2, gpu_id_step=2),
                8,
            ),
            (
                "pipeline split across nodes",
                _server_args(tp_size=4, pp_size=2, nnodes=2),
                4,
            ),
            ("tensor group split across nodes", _server_args(tp_size=8, nnodes=2), 4),
        ],
    )
    def test_every_scheduler_gets_its_own_port_in_the_reserved_range(
        self, telemetry_env, name, server_args, expected_ranks
    ):
        """A pod reserves one consecutive port per node-local rank and no more.

        Two schedulers on one port leave the second unable to bind, and a rank
        past the reserved range has no declared container port to be scraped on,
        so the ports a node hands out have to be exactly ``base .. base + n-1``.
        """
        ports = _ports_for_launch(server_args)
        assert len(ports) == expected_ranks
        assert set(ports) == set(range(BASE_PORT, BASE_PORT + expected_ranks))

    def test_each_node_restarts_at_the_reserved_base(self, telemetry_env):
        """The range is reserved per pod, so node 1 uses the same ports as node 0."""
        first, second = (
            _ports_for_launch(_server_args(tp_size=8, nnodes=2, node_rank=node_rank))
            for node_rank in (0, 1)
        )
        assert first == second == [19090, 19091, 19092, 19093]

    def test_a_narrow_launch_fits_where_a_full_range_would_not(self, telemetry_env):
        """The pod reserves one port per rank it launches, not eight regardless.

        A four-rank launch is measured against the four ports it is given, so a
        base leaving exactly that much room is usable rather than refused for
        ranks this launch never places.
        """
        base = 65535 - 3
        assert _ports_for_launch(_server_args(tp_size=4), base_port=base) == [
            65532,
            65533,
            65534,
            65535,
        ]

    def test_a_launch_with_no_room_for_its_own_ranks_is_rejected(self, telemetry_env):
        """Rank 0 alone fits; the ranks after it are what have nowhere to bind."""
        with pytest.raises(ValueError, match="exceeds the maximum port"):
            _ports_for_launch(_server_args(tp_size=8), base_port=65535 - 3)

    def test_disabled_telemetry_leaves_the_environment_alone(self, telemetry_env):
        telemetry_env.setenv("NIXL_TELEMETRY_ENABLE", "n")
        assert _ports_for_launch(_server_args(tp_size=8)) == [BASE_PORT] * 8


class TestResolvedLaunchConfiguration:
    def test_ranks_are_numbered_from_the_resolved_configuration(self, telemetry_env):
        """``--tp-size 8 --dwdp-size 8`` turns attention DP on during resolution.

        The scheduler is handed the raw arguments, where ``dp_size`` still reads
        1 and ``enable_dp_attention`` still reads False, while the launch places
        one group of eight schedulers and derives each ``dp_rank`` from its
        ``tp_rank``. Numbering from the raw values folds ``dp_rank`` in a second
        time, so the scheduler at ``tp_rank=1, dp_rank=1`` claims node-local
        rank 9 and is refused a port instead of taking 19091.
        """
        raw = _server_args(tp_size=8)
        resolved = _server_args(tp_size=8, dp_size=8, enable_dp_attention=True)
        telemetry_env.setattr(
            sglang_compat, "sglang_resolved_view", lambda server_args: resolved
        )

        ports = [_port_for_scheduler(raw, *call) for call in _scheduler_calls(resolved)]
        assert ports == list(range(BASE_PORT, BASE_PORT + 8))


class TestInstall:
    def test_install_is_a_no_op_when_telemetry_is_disabled(self, telemetry_env):
        """No SGLang import, so a non-telemetry deployment cannot regress on it."""
        telemetry_env.setenv("NIXL_TELEMETRY_ENABLE", "n")
        install_per_rank_nixl_prometheus_ports()

    def test_install_points_sglang_at_the_wrapper(self, telemetry_env):
        """The override has to land on the class, not on the ``sglang`` proxy."""
        engine = type("Engine", (), {"run_scheduler_process_func": None})
        telemetry_env.setitem(sys.modules, "sglang", _LazyProxyModule())
        telemetry_env.setitem(
            sys.modules,
            "sglang.srt.entrypoints.engine",
            SimpleNamespace(Engine=engine),
        )
        install_per_rank_nixl_prometheus_ports()
        assert engine.run_scheduler_process_func is run_scheduler_process_with_nixl_port

    def test_missing_override_point_is_rejected(self, telemetry_env):
        """Serving on would leave every rank one port and all but one rank dead."""
        telemetry_env.setitem(
            sys.modules,
            "sglang.srt.entrypoints.engine",
            SimpleNamespace(Engine=type("Engine", (), {})),
        )
        with pytest.raises(RuntimeError, match="run_scheduler_process_func"):
            install_per_rank_nixl_prometheus_ports()
