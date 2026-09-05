# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-rank NIXL exporter ports for co-located SGLang schedulers.

These tests replay the arguments SGLang hands each scheduler process rather
than calling the derivation with hand-picked ranks, because the bug being fixed
is about which SGLang argument identifies a rank -- not about the arithmetic.
``tp_rank`` restarts at 0 in every data-parallel group, so a derivation keyed on
it looks correct for plain tensor parallelism and hands every attention-DP rank
the same port, which is the deployment that reported this.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

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

GPUS_PER_NODE = 8


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
    signature: it reads ``server_args`` and ``gpu_id`` by name out of a purely
    positional call, so the position of every other parameter matters too.
    """


def _scheduler_gpu_id(
    server_args, tp_rank: int, dp_group_offset: int = 0, pp_rank: int = 0
) -> int:
    """The gpu_id SGLang computes for one scheduler on a single node.

    Mirrors ``Engine._launch_scheduler_processes`` and the data-parallel
    controller's ``launch_tensor_parallel_group``. A pipeline stage shifts the
    device by a whole tensor-parallel group and that shift is deliberately not
    multiplied by ``gpu_id_step``, so a stepped pipeline's devices are dense
    while a stepped tensor-parallel group's are not.
    """
    tp_size_per_node = min(server_args.tp_size, GPUS_PER_NODE)
    return (
        server_args.base_gpu_id
        + dp_group_offset
        + pp_rank * tp_size_per_node
        + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
    )


def _port_for_scheduler(
    monkeypatch, server_args, gpu_id, tp_rank, dp_rank, pp_rank: int = 0
) -> int:
    """The exporter port the wrapper installs in one scheduler's process.

    SGLang calls the scheduler entry point entirely positionally, so this passes
    positionally too rather than by keyword.
    """
    monkeypatch.setenv(NIXL_TELEMETRY_PROMETHEUS_PORT_ENV, "19090")
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
    monkeypatch.setenv("DYN_SYSTEM_PORT", "9090")
    monkeypatch.setenv("DYN_FORWARDPASS_METRIC_PORT", "20380")
    monkeypatch.delenv("SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS", raising=False)
    return monkeypatch


class TestPerRankPortAssignment:
    def test_tensor_parallel_ranks_get_distinct_ports(self, telemetry_env):
        server_args = SimpleNamespace(tp_size=8, base_gpu_id=0, gpu_id_step=1)
        ports = {
            _port_for_scheduler(
                telemetry_env,
                server_args,
                _scheduler_gpu_id(server_args, tp_rank),
                tp_rank,
                dp_rank=None,
            )
            for tp_rank in range(8)
        }
        assert len(ports) == 8

    def test_attention_dp_ranks_get_distinct_ports(self, telemetry_env):
        """Every scheduler here has ``tp_rank == 0``; only gpu_id separates them."""
        server_args = SimpleNamespace(tp_size=1, base_gpu_id=0, gpu_id_step=1)
        ports = {
            _port_for_scheduler(
                telemetry_env,
                server_args,
                _scheduler_gpu_id(server_args, tp_rank=0, dp_group_offset=dp_rank),
                tp_rank=0,
                dp_rank=dp_rank,
            )
            for dp_rank in range(8)
        }
        assert len(ports) == 8

    def test_offset_devices_still_start_at_the_reserved_base(self, telemetry_env):
        """``base_gpu_id`` shifts devices, not the pod's reserved port range."""
        server_args = SimpleNamespace(tp_size=4, base_gpu_id=4, gpu_id_step=1)
        ports = [
            _port_for_scheduler(
                telemetry_env,
                server_args,
                _scheduler_gpu_id(server_args, tp_rank),
                tp_rank,
                dp_rank=None,
            )
            for tp_rank in range(4)
        ]
        assert ports == [19090, 19091, 19092, 19093]

    def test_stepped_pipeline_stages_get_distinct_ports(self, telemetry_env):
        """A pipeline shift is not scaled by ``gpu_id_step``, so it is already dense.

        These two schedulers hold devices 0 and 1, and dividing either by the
        step would put both on the base port and back on one bind.
        """
        server_args = SimpleNamespace(
            tp_size=1, pp_size=2, base_gpu_id=0, gpu_id_step=2
        )
        ports = {
            _port_for_scheduler(
                telemetry_env,
                server_args,
                _scheduler_gpu_id(server_args, tp_rank=0, pp_rank=pp_rank),
                tp_rank=0,
                dp_rank=None,
                pp_rank=pp_rank,
            )
            for pp_rank in range(2)
        }
        assert ports == {19090, 19091}

    def test_hidden_device_index_is_rejected(self, telemetry_env):
        """With devices reindexed per process every gpu_id is 0; refuse to collide."""
        telemetry_env.setenv("SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS", "1")
        server_args = SimpleNamespace(tp_size=8, base_gpu_id=0, gpu_id_step=1)
        with pytest.raises(ValueError, match="SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS"):
            _port_for_scheduler(
                telemetry_env, server_args, gpu_id=0, tp_rank=0, dp_rank=None
            )

    def test_disabled_telemetry_leaves_the_environment_alone(self, telemetry_env):
        telemetry_env.setenv("NIXL_TELEMETRY_ENABLE", "n")
        server_args = SimpleNamespace(tp_size=8, base_gpu_id=0, gpu_id_step=1)
        assert (
            _port_for_scheduler(
                telemetry_env, server_args, gpu_id=3, tp_rank=3, dp_rank=None
            )
            == 19090
        )


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
