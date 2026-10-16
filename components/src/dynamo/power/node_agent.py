# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""node_agent: one process per GPU group, hosting EPD's pull driver.

``python -m dynamo.power.node_agent`` is started once per node per pool by
the launcher (srt-slurm today). Every rank runs EPD's per-node power monitor,
which writes this node's NVML energy snapshot each cycle; rank 0 of the pool
also runs EPD's NUPD/UBPC controller with the pool's figure-of-merit
provider and publishes ``power_decision.json`` through EPD's ``file``
actuator. The per-node applier (:mod:`dynamo.power.applier`) makes that
decision real.

This is the only module in the package that imports ``epd``; EPD is
installed separately (its license is still being settled) and is bind-mounted
into the node_agent container by the launcher. The command line itself is
parsed in :mod:`dynamo.power.node_agent_cli`, which imports neither EPD nor
NVML and is unit-tested.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
from collections.abc import Sequence

import pynvml
from epd.config import JobContext, RuntimeConfig
from epd.drivers import pull

from dynamo.power.config import (
    NodeAgentSettings,
    PowerControlConfigError,
    check_visible_gpus,
    epd_environment,
    nupd_budget_binds,
)
from dynamo.power.decision import decision_file_path
from dynamo.power.fom import (
    ProviderContext,
    build_provider,
    describe_provider,
    probe_provider,
)
from dynamo.power.node_agent_cli import (
    build_parser,
    resolve_node_name,
    settings_from_args,
)

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_EPD_FAILED = 1
EXIT_BAD_CONFIG = 2


def inspect_gpus(settings: NodeAgentSettings) -> float:
    """Check NVML visibility against the GPU group and return the max TGP in W.

    EPD senses NVML devices ``0 .. n-1``, so the container must see exactly
    this group's GPUs, or every GPU with the group being the node's first
    ``n`` indices (:func:`check_visible_gpus`). The devices' bus ids and minor
    numbers are logged for the record. The max TGP comes from ``--max-tgp-w``
    or the largest power-management upper limit among the group's GPUs.
    """
    pynvml.nvmlInit()
    try:
        count = pynvml.nvmlDeviceGetCount()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(index) for index in range(count)]
        for index, handle in enumerate(handles):
            logger.info(
                "visible NVML device %d: bus id %s, /dev/nvidia%d",
                index,
                pynvml.nvmlDeviceGetPciInfo(handle).busId,
                pynvml.nvmlDeviceGetMinorNumber(handle),
            )
        check_visible_gpus(count, settings.gpu_indices)
        if settings.max_tgp_w is not None:
            return float(settings.max_tgp_w)
        upper_limits_w = [
            pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0
            for handle in handles[: len(settings.gpu_indices)]
        ]
    finally:
        pynvml.nvmlShutdown()
    max_tgp_w = max(upper_limits_w)
    logger.info("max TGP auto-detected via NVML: %.0f W per GPU", max_tgp_w)
    return max_tgp_w


def build_runtime_config(
    settings: NodeAgentSettings, max_tgp_w: float
) -> RuntimeConfig:
    """EPD's behaviour knobs, built explicitly from the flags.

    ``min_tgp`` stays at EPD's default of ``0``: with ``use_history=False``
    EPD never reads it (its cap law has no floor), so the floor the recipe's
    ``min_tgp_w`` asks for is enforced by the applier instead.
    """
    return RuntimeConfig(
        max_tgp=max_tgp_w,
        gpus_per_node=len(settings.gpu_indices),
        average_gpu_power_limit=float(settings.budget_w_per_gpu),
        min_tgp=0.0,
        use_performance_feedback=True,
        use_history=False,
        use_ubpc=settings.use_ubpc,
        perf_threshold=settings.perf_threshold,
        actuator="file",
        epd_update_interval=None,
        run_trace=True,
        smi_log=False,
        ubpc_buffer=settings.ubpc_buffer,
        warmup_policy=settings.warmup_policy,
    )


def build_job_context(settings: NodeAgentSettings, node: str) -> JobContext:
    """EPD's rank identity: one process per node, rank 0 decides, local rank 0 senses."""
    return JobContext(
        job_id=os.environ.get("SLURM_JOB_ID", settings.pool_name),
        job_name=settings.pool_name,
        job_nodelist=",".join(settings.nodelist),
        num_nodes=len(settings.nodelist),
        global_rank=settings.rank,
        local_rank=0,
        node_name=node,
    )


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    try:
        settings = settings_from_args(build_parser().parse_args(argv))
        node = resolve_node_name(settings)
        max_tgp_w = inspect_gpus(settings)
    except (PowerControlConfigError, pynvml.NVMLError) as exc:
        logger.error("node_agent cannot start: %s", exc)
        return EXIT_BAD_CONFIG
    budget_binds = nupd_budget_binds(settings.budget_w_per_gpu, max_tgp_w)
    if settings.mode == "nupd" and not budget_binds:
        logger.error(
            "mode nupd needs a binding budget: %d W per GPU is not below the max TGP "
            "of %.0f W (use mode ubpc/both or lower the budget)",
            settings.budget_w_per_gpu,
            max_tgp_w,
        )
        return EXIT_BAD_CONFIG
    if settings.mode == "ubpc" and budget_binds:
        logger.warning(
            "mode ubpc with a binding budget (%d W per GPU < max TGP %.0f W): NUPD "
            "redistributes whenever the budget binds, so this run behaves exactly "
            "like mode both",
            settings.budget_w_per_gpu,
            max_tgp_w,
        )

    os.environ.update(epd_environment(settings, this_node=node, max_tgp_w=max_tgp_w))
    os.makedirs(settings.log_dir, exist_ok=True)
    config = build_runtime_config(settings, max_tgp_w)
    context = build_job_context(settings, node)

    provider = None
    if settings.is_controller:
        try:
            provider = build_provider(
                settings.fom,
                ProviderContext(
                    urls=settings.all_metrics_urls,
                    pool_gpu_count=settings.pool_gpu_count,
                    pool_nodes=settings.nodelist,
                    log_dir=settings.log_dir,
                    frontend_urls=settings.frontend_metrics_urls,
                    worker_urls=settings.worker_metrics_urls,
                ),
            )
            problems = probe_provider(provider)
        except (PowerControlConfigError, ImportError) as exc:
            logger.error("FoM %s cannot be used: %s", settings.fom.describe(), exc)
            return EXIT_BAD_CONFIG
        if problems:
            for problem in problems:
                logger.error("FoM probe: %s", problem)
            return EXIT_BAD_CONFIG
        logger.info("FoM: %s", describe_provider(provider))

    logger.info(
        "pool %s rank %d/%d on %s, GPUs %s: mode %s, budget %d W per GPU, max TGP "
        "%.0f W, interval %d s, perf threshold %.3f, UBPC buffer %.2f, warm-up %s, "
        "log dir %s",
        settings.pool_name,
        settings.rank,
        len(settings.nodelist),
        node,
        list(settings.gpu_indices),
        settings.mode,
        settings.budget_w_per_gpu,
        max_tgp_w,
        settings.interval_s,
        settings.perf_threshold,
        settings.ubpc_buffer,
        settings.warmup_policy,
        settings.log_dir,
    )
    if settings.is_controller:
        logger.info("decision file: %s", decision_file_path(settings.log_dir))

    stop = threading.Event()

    def _request_stop(signum, _frame) -> None:
        logger.info("signal %d received; stopping EPD", signum)
        stop.set()

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    started = pull.start(
        log_dir=settings.log_dir,
        perf_signal_provider=provider,
        actuator="file",
        interval_s=float(settings.interval_s),
        config=config,
        context=context,
    )
    if not started:
        logger.error("EPD did not start (see the epd log lines above)")
        return EXIT_EPD_FAILED
    logger.info("EPD running; one decision cycle every %d s", settings.interval_s)
    while not stop.wait(1.0):
        pass
    stopped = pull.stop()
    logger.info("EPD stopped (%s)", "clean" if stopped else "teardown reported errors")
    return EXIT_OK if stopped else EXIT_EPD_FAILED


if __name__ == "__main__":
    sys.exit(main())
