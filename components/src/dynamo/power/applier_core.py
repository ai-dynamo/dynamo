# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The applier's state machine, actuators and verification, without NVML.

:mod:`dynamo.power.applier` adds the pynvml-backed pieces (the NVML session,
limit reader and actuator) and ``main``; everything here runs on injected
collaborators so it is unit-testable on a machine without ``nvidia-ml-py``.

Backends behind the same :class:`PowerActuator` interface:

* ``sudo-smi`` -- ``sudo -n nvidia-smi -pl <W> -i <N>`` per GPU, in that
  argument order (the form hecate's sudoers rule allows; ``-n`` makes a
  missing rule fail at once instead of hanging on a password prompt);
* ``dcgmi`` -- one DCGM group of the GPUs that share a cap plus
  ``dcgmi config -g <id> --set -P <W>`` (AWS CMH, no privilege needed);
* ``nvml`` -- ``nvmlDeviceSetPowerManagementLimit`` (root only; in the
  other module).

Every apply is verified through a read-only :class:`LimitReader` (setpoint
and enforced limit) and recorded in a trace CSV with the columns ``ts, gpu,
requested, setpoint, enforced, status`` (watts; ``ts`` is unix seconds) where
``status`` is ``ok``, ``not_landed``, ``clamped`` or ``unverified``.

Fail-safe rules (design section 5): when the decision file is older than
``--stale-after-s`` or disappears after it existed, every GPU of the group is
set to the pool's uniform budget (``--budget-w-per-gpu``), never to the
hardware default -- a provisioned budget is a promise to the facility. The
hardware default is restored only on exit with ``--restore-default-on-exit``
(DCGM caps persist as hostengine target config, so the end-of-job reset is
mandatory there). An optional floor (``--min-w-per-gpu``, the recipe's
``min_tgp_w``) raises any requested cap below it before writing; EPD's own
cap law has no floor.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from dynamo.power.config import APPLIERS, DEFAULT_POLL_S, DEFAULT_STALE_AFTER_S
from dynamo.power.decision import (
    DecisionFormatError,
    PowerDecisionFile,
    node_watts,
    read_decision_file,
)

logger = logging.getLogger(__name__)

STATUS_OK = "ok"
STATUS_NOT_LANDED = "not_landed"
STATUS_CLAMPED = "clamped"
STATUS_UNVERIFIED = "unverified"
TRACE_COLUMNS = ("ts", "gpu", "requested", "setpoint", "enforced", "status")
MATCH_TOLERANCE_W = 5.0
COMMAND_TIMEOUT_S = 30.0
DCGM_GROUP_PREFIX = "dynamo-power"


# ------------------------------------------------------------------ commands


@dataclass(frozen=True)
class CommandResult:
    """Outcome of one external command."""

    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0


CommandRunner = Callable[[Sequence[str]], CommandResult]


def run_command(
    argv: Sequence[str], timeout_s: float = COMMAND_TIMEOUT_S
) -> CommandResult:
    """Run ``argv`` without a shell; a timeout is reported as a failed result."""
    try:
        completed = subprocess.run(
            list(argv), capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except subprocess.TimeoutExpired:
        return CommandResult(
            returncode=-1, stderr=f"timed out after {timeout_s:g} s: {' '.join(argv)}"
        )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def pid_alive(pid: int) -> bool:
    """Whether a process with ``pid`` exists (signal 0; EPERM counts as alive)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ----------------------------------------------------------------- actuators


class PowerActuator(Protocol):
    """Sets per-GPU power caps; verification is the applier's job, not the actuator's."""

    name: str

    def apply(self, limits_w: Mapping[int, int]) -> None:
        """Set each GPU index in ``limits_w`` to its watts (best effort)."""

    def close(self) -> None:
        """Release backend resources (DCGM groups, NVML session)."""


class SudoSmiActuator:
    """``sudo -n nvidia-smi -pl <W> -i <N>`` per GPU, ascending GPU order."""

    name = "sudo-smi"

    def __init__(self, runner: CommandRunner = run_command) -> None:
        self._runner = runner

    def apply(self, limits_w: Mapping[int, int]) -> None:
        for gpu in sorted(limits_w):
            argv = [
                "sudo",
                "-n",
                "nvidia-smi",
                "-pl",
                str(limits_w[gpu]),
                "-i",
                str(gpu),
            ]
            result = self._runner(argv)
            if not result.ok:
                logger.error(
                    "GPU %d: %s failed (exit %d): %s",
                    gpu,
                    " ".join(argv),
                    result.returncode,
                    (result.stderr or result.stdout).strip(),
                )

    def close(self) -> None:
        return None


_GROUP_LIST_ID_RE = re.compile(r"Group ID\s*\|\s*(\d+)")
_GROUP_LIST_NAME_RE = re.compile(r"Group Name\s*\|\s*(\S+)")


def parse_dcgmi_group_listing(text: str) -> list[tuple[str, str]]:
    """``(group_id, name)`` pairs from ``dcgmi group -l`` output.

    The listing prints one block per group with a ``Group ID | <n>`` line
    followed by a ``Group Name | <name>`` line; anything else is ignored, so
    an unexpected format yields no pairs rather than wrong ones.
    """
    pairs = []
    current_id: str | None = None
    for line in text.splitlines():
        id_match = _GROUP_LIST_ID_RE.search(line)
        if id_match:
            current_id = id_match.group(1)
            continue
        name_match = _GROUP_LIST_NAME_RE.search(line)
        if name_match and current_id is not None:
            pairs.append((current_id, name_match.group(1)))
            current_id = None
    return pairs


class DcgmiActuator:
    """DCGM groups of the GPUs sharing a cap plus ``dcgmi config --set -P``.

    Groups are named ``<prefix>-<pid>-<gpu>-<gpu>...``, created lazily per
    distinct GPU set (one per node in the common case, since EPD caps every
    GPU of a node alike) and deleted on :meth:`close`. An applier that dies
    without :meth:`close` leaks its groups in the hostengine;
    :meth:`cleanup_stale_groups` removes those of dead appliers at start-up.
    """

    name = "dcgmi"
    _GROUP_ID_RE = re.compile(r"group ID of (\d+)")

    def __init__(
        self,
        runner: CommandRunner = run_command,
        group_prefix: str = DCGM_GROUP_PREFIX,
    ) -> None:
        self._runner = runner
        self._group_prefix = group_prefix
        self._groups: dict[tuple[int, ...], str] = {}

    def apply(self, limits_w: Mapping[int, int]) -> None:
        by_watts: dict[int, list[int]] = {}
        for gpu in sorted(limits_w):
            by_watts.setdefault(limits_w[gpu], []).append(gpu)
        for watts, gpus in sorted(by_watts.items()):
            group_id = self._group_id(tuple(gpus))
            if group_id is None:
                continue
            argv = ["dcgmi", "config", "-g", group_id, "--set", "-P", str(watts)]
            result = self._runner(argv)
            if not result.ok:
                logger.error(
                    "GPUs %s: %s failed (exit %d): %s",
                    gpus,
                    " ".join(argv),
                    result.returncode,
                    (result.stderr or result.stdout).strip(),
                )

    def cleanup_stale_groups(
        self, alive: Callable[[int], bool] = pid_alive
    ) -> list[str]:
        """Delete ``<prefix>-<pid>-...`` groups whose applier ``pid`` is gone.

        Returns the ids deleted. A group whose pid is still alive (another
        applier on this node) or whose name does not carry a pid is left alone.
        """
        result = self._runner(["dcgmi", "group", "-l"])
        if not result.ok:
            logger.warning(
                "listing DCGM groups failed (exit %d): %s",
                result.returncode,
                (result.stderr or result.stdout).strip(),
            )
            return []
        deleted = []
        marker = f"{self._group_prefix}-"
        for group_id, name in parse_dcgmi_group_listing(result.stdout):
            if not name.startswith(marker):
                continue
            pid_text = name[len(marker) :].split("-", 1)[0]
            if not pid_text.isdigit():
                continue
            pid = int(pid_text)
            if pid == os.getpid() or alive(pid):
                logger.info(
                    "leaving DCGM group %s (%s): applier pid %d is alive",
                    group_id,
                    name,
                    pid,
                )
                continue
            delete = self._runner(["dcgmi", "group", "-d", group_id])
            if delete.ok:
                logger.info("deleted stale DCGM group %s (%s)", group_id, name)
                deleted.append(group_id)
            else:
                logger.warning(
                    "deleting stale DCGM group %s (%s) failed (exit %d)",
                    group_id,
                    name,
                    delete.returncode,
                )
        return deleted

    def _group_id(self, gpus: tuple[int, ...]) -> str | None:
        if gpus in self._groups:
            return self._groups[gpus]
        name = f"{self._group_prefix}-{os.getpid()}-{'-'.join(str(g) for g in gpus)}"
        argv = ["dcgmi", "group", "-c", name, "-a", ",".join(str(g) for g in gpus)]
        result = self._runner(argv)
        match = self._GROUP_ID_RE.search(result.stdout) if result.ok else None
        if match is None:
            logger.error(
                "creating DCGM group for GPUs %s failed (exit %d): %s",
                list(gpus),
                result.returncode,
                (result.stderr or result.stdout).strip(),
            )
            return None
        self._groups[gpus] = match.group(1)
        logger.info(
            "created DCGM group %s (id %s) for GPUs %s",
            name,
            match.group(1),
            list(gpus),
        )
        return match.group(1)

    def close(self) -> None:
        for gpus, group_id in self._groups.items():
            result = self._runner(["dcgmi", "group", "-d", group_id])
            if not result.ok:
                logger.warning(
                    "deleting DCGM group %s for GPUs %s failed (exit %d)",
                    group_id,
                    list(gpus),
                    result.returncode,
                )
        self._groups.clear()


# -------------------------------------------------------------- verification


@dataclass(frozen=True)
class GpuLimits:
    """Read-back of one GPU: the programmed setpoint and the enforced limit, in watts."""

    setpoint_w: float | None
    enforced_w: float | None


class LimitReader(Protocol):
    """Read-only view of a GPU's power limits (NVML in production, a fake in tests)."""

    def read(self, gpu: int) -> GpuLimits:
        """Current setpoint and enforced limit."""

    def default_limit_w(self, gpu: int) -> float | None:
        """The hardware default limit, restored at job end."""

    def constraints_w(self, gpu: int) -> tuple[float, float] | None:
        """``(min, max)`` settable limits, or ``None`` when unknown."""


def classify(
    requested_w: int,
    target_w: int,
    limits: GpuLimits,
    tolerance_w: float = MATCH_TOLERANCE_W,
) -> str:
    """EPD's enforcement verdicts for one GPU after an apply.

    ``target_w`` is what was actually written (``requested_w`` raised to the
    floor and clamped into the GPU's settable range). ``ok``: the enforced
    limit matches the request; ``clamped``: it matches an adjusted target, or
    the setpoint landed but the hardware enforces less; ``not_landed``: the
    setpoint is not what was written, or it is but the hardware enforces more
    (EPD's "diverged" case); ``unverified``: a read-back value is missing.
    """
    if limits.setpoint_w is None or limits.enforced_w is None:
        return STATUS_UNVERIFIED
    if abs(limits.enforced_w - target_w) <= tolerance_w:
        return STATUS_OK if target_w == requested_w else STATUS_CLAMPED
    if setpoint_missed(target_w, limits, tolerance_w):
        return STATUS_NOT_LANDED
    if limits.enforced_w < limits.setpoint_w - tolerance_w:
        return STATUS_CLAMPED
    return STATUS_NOT_LANDED


def setpoint_missed(
    target_w: int, limits: GpuLimits, tolerance_w: float = MATCH_TOLERANCE_W
) -> bool:
    """Whether the programmed setpoint differs from what was written.

    Only this case is worth retrying: rewriting the same cap cannot change an
    enforced limit the hardware chose, nor make an unreadable GPU readable.
    """
    return (
        limits.setpoint_w is not None
        and abs(limits.setpoint_w - target_w) > tolerance_w
    )


class TraceWriter:
    """Appends verification rows to the trace CSV (header on first write)."""

    def __init__(self, path: str | None) -> None:
        self.path = path

    def write(
        self,
        ts: float,
        gpu: int,
        requested_w: int,
        setpoint_w: float | None,
        enforced_w: float | None,
        status: str,
    ) -> None:
        if self.path is None:
            return
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        new_file = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        with open(self.path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if new_file:
                writer.writerow(TRACE_COLUMNS)
            writer.writerow(
                [
                    f"{ts:.3f}",
                    gpu,
                    requested_w,
                    "" if setpoint_w is None else f"{setpoint_w:.0f}",
                    "" if enforced_w is None else f"{enforced_w:.0f}",
                    status,
                ]
            )


# ------------------------------------------------------------------- applier


@dataclass
class GpuState:
    """What the applier last did to one GPU.

    ``retry`` is set when the setpoint itself missed, the one verdict a
    rewrite can fix; a ``not_landed`` whose setpoint landed (the hardware
    enforces more) is reported but not rewritten every poll.
    """

    applied_w: int | None = None
    status: str | None = None
    retry: bool = False


class Applier:
    """The decision-file follower for one node's GPU group.

    Pure state machine over injected collaborators (actuator, limit reader,
    clocks, trace) so it can be exercised without GPUs: :meth:`poll_once`
    performs one poll, :meth:`run` loops until a stop event, :meth:`shutdown`
    restores defaults when asked and closes the actuator.
    """

    def __init__(
        self,
        *,
        decision_file: str,
        node: str,
        gpu_indices: Sequence[int],
        budget_w_per_gpu: int,
        actuator: PowerActuator,
        limits: LimitReader,
        trace: TraceWriter,
        stale_after_s: float = DEFAULT_STALE_AFTER_S,
        restore_default_on_exit: bool = False,
        min_w_per_gpu: int | None = None,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if budget_w_per_gpu <= 0:
            raise ValueError("budget_w_per_gpu must be > 0 W")
        if stale_after_s <= 0:
            raise ValueError("stale_after_s must be > 0 s")
        if min_w_per_gpu is not None:
            if min_w_per_gpu <= 0:
                raise ValueError("min_w_per_gpu must be > 0 W")
            if min_w_per_gpu > budget_w_per_gpu:
                raise ValueError(
                    f"min_w_per_gpu {min_w_per_gpu} W is above the uniform budget "
                    f"{budget_w_per_gpu} W; the fail-safe cap would sit below the floor"
                )
        self._decision_file = decision_file
        self._node = node
        self._gpu_indices = tuple(sorted(int(index) for index in gpu_indices))
        self._budget_w = int(budget_w_per_gpu)
        self._min_w = None if min_w_per_gpu is None else int(min_w_per_gpu)
        self._actuator = actuator
        self._limits = limits
        self._trace = trace
        self._stale_after_s = float(stale_after_s)
        self._restore_default_on_exit = restore_default_on_exit
        self._clock = clock
        self._monotonic = monotonic
        self._states: dict[int, GpuState] = {
            gpu: GpuState() for gpu in self._gpu_indices
        }
        self._seen_file = False
        self._last_sequence: int | None = None
        self._sequence_changes = 0
        self._last_change_at = 0.0
        self._fail_safe_active = False
        self._missing_entry_sequence: int | None = None

    @property
    def states(self) -> Mapping[int, GpuState]:
        return self._states

    @property
    def fail_safe_active(self) -> bool:
        return self._fail_safe_active

    def run(self, stop: threading.Event, poll_s: float) -> None:
        """Poll until ``stop`` is set; the final poll happens before returning."""
        while True:
            self.poll_once()
            if stop.wait(poll_s):
                return

    def poll_once(self) -> None:
        decision = self._load_decision()
        if decision is None:
            if self._seen_file:
                self._enter_fail_safe("decision file missing or unreadable")
            return
        self._seen_file = True
        self._note_sequence(decision)
        age_s = self._clock() - decision.timestamp
        if self._is_stale(age_s):
            self._enter_fail_safe(
                f"decision #{decision.sequence} is {age_s:.0f} s old "
                f"(> {self._stale_after_s:g} s)"
            )
            return
        watts = node_watts(decision, self._node)
        if watts is None:
            if self._missing_entry_sequence != decision.sequence:
                self._missing_entry_sequence = decision.sequence
                logger.warning(
                    "decision #%d has no entry for node %s (nodes: %s); nothing applied",
                    decision.sequence,
                    self._node,
                    list(decision.nodes()),
                )
            return
        if self._fail_safe_active:
            logger.info(
                "decision #%d is fresh again; leaving fail-safe", decision.sequence
            )
            self._fail_safe_active = False
        self._apply(
            {gpu: int(watts) for gpu in self._gpu_indices},
            reason=f"decision #{decision.sequence}",
        )

    def shutdown(self) -> None:
        """Restore hardware defaults when requested, then close the actuator."""
        if self._restore_default_on_exit:
            defaults = {}
            for gpu in self._gpu_indices:
                default_w = self._limits.default_limit_w(gpu)
                if default_w is None:
                    logger.error(
                        "GPU %d: default limit unknown; leaving its cap in place", gpu
                    )
                    continue
                defaults[gpu] = int(round(default_w))
                self._states[gpu] = GpuState()
            if defaults:
                # The hardware default is restored as-is: the floor protects
                # the pool's caps, not the state the job hands back.
                self._apply(
                    defaults, reason="restore hardware default on exit", floor=False
                )
        self._actuator.close()

    # ---- internals ----------------------------------------------------

    def _load_decision(self) -> PowerDecisionFile | None:
        try:
            return read_decision_file(self._decision_file)
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, DecisionFormatError) as exc:
            logger.warning("decision file %s unreadable: %s", self._decision_file, exc)
            return None

    def _note_sequence(self, decision: PowerDecisionFile) -> None:
        if decision.sequence != self._last_sequence:
            if self._last_sequence is not None:
                self._sequence_changes += 1
            self._last_sequence = decision.sequence
            self._last_change_at = self._monotonic()

    def _is_stale(self, age_s: float) -> bool:
        # The controller's timestamp is the primary signal. A sequence that keeps
        # advancing proves the controller is alive even when clocks disagree.
        observed_live = (
            self._sequence_changes >= 1
            and (self._monotonic() - self._last_change_at) <= self._stale_after_s
        )
        return age_s > self._stale_after_s and not observed_live

    def _enter_fail_safe(self, reason: str) -> None:
        if not self._fail_safe_active:
            logger.warning(
                "fail-safe: %s; holding the uniform budget of %d W per GPU on GPUs %s",
                reason,
                self._budget_w,
                list(self._gpu_indices),
            )
        self._fail_safe_active = True
        self._apply(
            {gpu: self._budget_w for gpu in self._gpu_indices},
            reason=f"fail-safe ({reason})",
        )

    def _apply(
        self, targets: Mapping[int, int], reason: str, *, floor: bool = True
    ) -> None:
        pending = {
            gpu: watts
            for gpu, watts in targets.items()
            if self._states[gpu].applied_w != watts or self._states[gpu].retry
        }
        if not pending:
            return
        adjusted = {
            gpu: self._adjust(gpu, watts, floor=floor) for gpu, watts in pending.items()
        }
        logger.info(
            "applying %s: %s",
            reason,
            ", ".join(
                f"GPU {gpu}: {watts} W" for gpu, watts in sorted(adjusted.items())
            ),
        )
        self._actuator.apply(adjusted)
        ts = self._clock()
        for gpu in sorted(pending):
            limits = self._limits.read(gpu)
            status = classify(pending[gpu], adjusted[gpu], limits)
            retry = status == STATUS_NOT_LANDED and setpoint_missed(
                adjusted[gpu], limits
            )
            self._states[gpu] = GpuState(
                applied_w=pending[gpu], status=status, retry=retry
            )
            self._trace.write(
                ts, gpu, pending[gpu], limits.setpoint_w, limits.enforced_w, status
            )
            if status != STATUS_OK:
                logger.warning(
                    "GPU %d: requested %d W (wrote %d W), setpoint %s, enforced %s -> %s%s",
                    gpu,
                    pending[gpu],
                    adjusted[gpu],
                    _fmt_w(limits.setpoint_w),
                    _fmt_w(limits.enforced_w),
                    status,
                    " (will retry)" if retry else "",
                )

    def _adjust(self, gpu: int, watts: int, *, floor: bool) -> int:
        """The watts actually written: raised to the floor, then clamped to NVML's range."""
        target = watts
        if floor and self._min_w is not None and target < self._min_w:
            logger.warning(
                "GPU %d: %d W is below the pool floor of %d W; writing %d W",
                gpu,
                watts,
                self._min_w,
                self._min_w,
            )
            target = self._min_w
        constraints = self._limits.constraints_w(gpu)
        if constraints is None:
            return target
        low, high = constraints
        bounded = int(round(min(max(float(target), low), high)))
        if bounded != target:
            logger.warning(
                "GPU %d: %d W is outside the settable range [%.0f W, %.0f W]; writing %d W",
                gpu,
                target,
                low,
                high,
                bounded,
            )
        return bounded


def _fmt_w(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f} W"


# ----------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dynamo.power.applier",
        description="Apply EPD power decisions to this node's GPUs and verify them.",
    )
    parser.add_argument(
        "--decision-file", required=True, help="EPD's power_decision.json"
    )
    parser.add_argument("--node", required=True, help="this node's key in the decision")
    parser.add_argument(
        "--gpu-indices",
        required=True,
        help="comma-separated GPU indices of the group (host NVML index == "
        "/dev/nvidiaN minor number; checked at start)",
    )
    parser.add_argument("--mode", required=True, choices=APPLIERS)
    parser.add_argument(
        "--budget-w-per-gpu",
        required=True,
        type=int,
        help="uniform fail-safe cap in W per GPU",
    )
    parser.add_argument(
        "--stale-after-s",
        type=float,
        default=DEFAULT_STALE_AFTER_S,
        help="fall back to the budget when the decision is older than this many s "
        "(at least 2 x the controller's interval_s; the launcher renders "
        "3 x interval_s)",
    )
    parser.add_argument(
        "--poll-s", type=float, default=DEFAULT_POLL_S, help="poll period in s"
    )
    parser.add_argument(
        "--min-w-per-gpu",
        type=int,
        default=None,
        help="floor in W per GPU: a requested cap below it is raised to it and "
        "reported as clamped (the recipe's min_tgp_w)",
    )
    parser.add_argument(
        "--restore-default-on-exit",
        action="store_true",
        help="restore each GPU's hardware default limit on exit",
    )
    parser.add_argument("--trace-csv", default=None, help="verification trace CSV path")
    return parser


__all__ = [
    "COMMAND_TIMEOUT_S",
    "DCGM_GROUP_PREFIX",
    "MATCH_TOLERANCE_W",
    "STATUS_CLAMPED",
    "STATUS_NOT_LANDED",
    "STATUS_OK",
    "STATUS_UNVERIFIED",
    "TRACE_COLUMNS",
    "Applier",
    "CommandResult",
    "CommandRunner",
    "DcgmiActuator",
    "GpuLimits",
    "GpuState",
    "LimitReader",
    "PowerActuator",
    "SudoSmiActuator",
    "TraceWriter",
    "build_parser",
    "classify",
    "parse_dcgmi_group_listing",
    "pid_alive",
    "run_command",
    "setpoint_missed",
]
