# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-node applier: makes EPD's decisions real on the GPUs of one GPU group.

``python -m dynamo.power.applier`` runs on the bare host (SLURM) or as the
privileged sidecar (Kubernetes), polls ``power_decision.json`` and applies
this node's entry to each GPU index of its group. The state machine, the
``sudo-smi`` and ``dcgmi`` actuators and the verification verdicts live in
:mod:`dynamo.power.applier_core` (no NVML import, unit-tested); this module
adds the pynvml-backed pieces and ``main``:

* one NVML session for the applier's lifetime (:class:`NvmlSession`);
* the read-only verification reader (:class:`NvmlLimitReader`:
  ``nvmlDeviceGetPowerManagementLimit`` / ``nvmlDeviceGetEnforcedPowerLimit``
  / ``nvmlDeviceGetPowerManagementDefaultLimit`` / limit constraints);
* the ``nvml`` actuator (``nvmlDeviceSetPowerManagementLimit``, root only);
* the start-up identity check: every ``--gpu-indices`` entry must be an
  existing host NVML index (the key ``nvidia-smi -i``, DCGM and
  ``NVIDIA_VISIBLE_DEVICES`` share); a ``/dev/nvidiaN`` minor number that
  differs from the index is logged, not refused.

Host requirements: Python >= 3.9 and ``nvidia-ml-py`` in the interpreter
that runs this module; the ``ai-dynamo`` wheel itself is not needed.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
from collections.abc import Mapping, Sequence

import pynvml

from dynamo.power.applier_core import (
    COMMAND_TIMEOUT_S,
    MATCH_TOLERANCE_W,
    STATUS_CLAMPED,
    STATUS_NOT_LANDED,
    STATUS_OK,
    STATUS_UNVERIFIED,
    TRACE_COLUMNS,
    Applier,
    CommandResult,
    DcgmiActuator,
    GpuLimits,
    GpuState,
    LimitReader,
    PowerActuator,
    SudoSmiActuator,
    TraceWriter,
    build_parser,
    classify,
    run_command,
    setpoint_missed,
)
from dynamo.power.config import (
    PowerControlConfigError,
    parse_gpu_indices,
    validate_gpu_identity,
)

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_BAD_CONFIG = 2


class NvmlSession:
    """One NVML session for the applier's lifetime, with cached device handles."""

    def __init__(self) -> None:
        pynvml.nvmlInit()
        self._handles: dict[int, object] = {}

    def device_count(self) -> int:
        return int(pynvml.nvmlDeviceGetCount())

    def handle(self, gpu: int):
        if gpu not in self._handles:
            self._handles[gpu] = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        return self._handles[gpu]

    def minor_numbers(self, gpu_indices: Sequence[int]) -> Mapping[int, int]:
        """``{index: /dev/nvidiaN minor}`` for the indices that exist."""
        count = self.device_count()
        return {
            gpu: int(pynvml.nvmlDeviceGetMinorNumber(self.handle(gpu)))
            for gpu in gpu_indices
            if gpu < count
        }

    def close(self) -> None:
        self._handles.clear()
        pynvml.nvmlShutdown()


class NvmlActuator:
    """``nvmlDeviceSetPowerManagementLimit`` per GPU; the process must be root."""

    name = "nvml"

    def __init__(self, session: NvmlSession) -> None:
        self._session = session

    def apply(self, limits_w: Mapping[int, int]) -> None:
        for gpu in sorted(limits_w):
            try:
                pynvml.nvmlDeviceSetPowerManagementLimit(
                    self._session.handle(gpu), int(limits_w[gpu]) * 1000
                )
            except pynvml.NVMLError as exc:
                logger.error(
                    "GPU %d: NVML set %d W failed: %s", gpu, limits_w[gpu], exc
                )

    def close(self) -> None:
        return None


class NvmlLimitReader:
    """NVML-backed :class:`LimitReader`; a failed read yields ``None``, never a guess."""

    def __init__(self, session: NvmlSession) -> None:
        self._session = session

    def read(self, gpu: int) -> GpuLimits:
        setpoint = None
        enforced = None
        try:
            handle = self._session.handle(gpu)
            setpoint = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            enforced = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
        except pynvml.NVMLError as exc:
            logger.warning("GPU %d: NVML limit read failed: %s", gpu, exc)
        return GpuLimits(setpoint_w=setpoint, enforced_w=enforced)

    def default_limit_w(self, gpu: int) -> float | None:
        try:
            handle = self._session.handle(gpu)
            return pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle) / 1000.0
        except pynvml.NVMLError as exc:
            logger.warning("GPU %d: NVML default-limit read failed: %s", gpu, exc)
            return None

    def constraints_w(self, gpu: int) -> tuple[float, float] | None:
        try:
            handle = self._session.handle(gpu)
            low, high = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
        except pynvml.NVMLError as exc:
            logger.warning("GPU %d: NVML limit-constraints read failed: %s", gpu, exc)
            return None
        return (low / 1000.0, high / 1000.0)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    try:
        gpu_indices = parse_gpu_indices(args.gpu_indices)
    except PowerControlConfigError as exc:
        logger.error("applier cannot start: %s", exc)
        return EXIT_BAD_CONFIG
    if args.budget_w_per_gpu <= 0:
        logger.error("--budget-w-per-gpu must be > 0 W")
        return EXIT_BAD_CONFIG
    if args.min_w_per_gpu is not None and not (
        0 < args.min_w_per_gpu <= args.budget_w_per_gpu
    ):
        logger.error(
            "--min-w-per-gpu must be > 0 W and <= --budget-w-per-gpu (%d W); got %d W",
            args.budget_w_per_gpu,
            args.min_w_per_gpu,
        )
        return EXIT_BAD_CONFIG
    if args.mode == "nvml" and os.geteuid() != 0:
        logger.error("--mode nvml sets caps through NVML and needs root")
        return EXIT_BAD_CONFIG

    try:
        session = NvmlSession()
    except pynvml.NVMLError as exc:
        logger.error("applier cannot start: NVML is unavailable on this host: %s", exc)
        return EXIT_BAD_CONFIG
    try:
        notes = validate_gpu_identity(
            session.device_count(), session.minor_numbers(gpu_indices), gpu_indices
        )
    except (pynvml.NVMLError, PowerControlConfigError) as exc:
        logger.error("applier cannot start: GPU identity check failed: %s", exc)
        session.close()
        return EXIT_BAD_CONFIG
    for note in notes:
        logger.info("GPU identity: %s", note)

    limits = NvmlLimitReader(session)
    actuator: PowerActuator
    if args.mode == "sudo-smi":
        actuator = SudoSmiActuator()
    elif args.mode == "dcgmi":
        dcgmi = DcgmiActuator()
        dcgmi.cleanup_stale_groups()
        actuator = dcgmi
    else:
        actuator = NvmlActuator(session)
    applier = Applier(
        decision_file=args.decision_file,
        node=args.node,
        gpu_indices=gpu_indices,
        budget_w_per_gpu=args.budget_w_per_gpu,
        actuator=actuator,
        limits=limits,
        trace=TraceWriter(args.trace_csv),
        stale_after_s=args.stale_after_s,
        restore_default_on_exit=args.restore_default_on_exit,
        min_w_per_gpu=args.min_w_per_gpu,
    )
    stop = threading.Event()

    def _request_stop(signum, _frame) -> None:
        logger.info("signal %d received; stopping", signum)
        stop.set()

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    logger.info(
        "applier for node %s GPUs %s: mode %s, decision file %s, poll every %g s, "
        "stale after %g s, fail-safe budget %d W per GPU, floor %s, restore default "
        "on exit: %s",
        args.node,
        list(gpu_indices),
        args.mode,
        args.decision_file,
        args.poll_s,
        args.stale_after_s,
        args.budget_w_per_gpu,
        "none" if args.min_w_per_gpu is None else f"{args.min_w_per_gpu} W per GPU",
        args.restore_default_on_exit,
    )
    try:
        applier.run(stop, args.poll_s)
    finally:
        applier.shutdown()
        session.close()
    return EXIT_OK


__all__ = [
    "COMMAND_TIMEOUT_S",
    "EXIT_BAD_CONFIG",
    "EXIT_OK",
    "MATCH_TOLERANCE_W",
    "STATUS_CLAMPED",
    "STATUS_NOT_LANDED",
    "STATUS_OK",
    "STATUS_UNVERIFIED",
    "TRACE_COLUMNS",
    "Applier",
    "CommandResult",
    "DcgmiActuator",
    "GpuLimits",
    "GpuState",
    "LimitReader",
    "NvmlActuator",
    "NvmlLimitReader",
    "NvmlSession",
    "PowerActuator",
    "SudoSmiActuator",
    "TraceWriter",
    "build_parser",
    "classify",
    "main",
    "run_command",
    "setpoint_missed",
]


if __name__ == "__main__":
    sys.exit(main())
