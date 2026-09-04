# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Roles as local subprocesses.

The substrate most of the existing suite already uses, so it is the one worth
implementing first. It is also the one that makes the façade testable without a
cluster: a test can start a role, query it, restart it, and judge the evidence,
entirely on a laptop.

Two behaviours here are deliberate and both come from watching real suites:

**Stop means stop this role.** A ``stop`` that quietly tears down the whole
deployment makes every fault-tolerance test vacuous — the thing it was supposed
to isolate is gone along with everything else. Selection resolves to one role's
processes and nothing else.

**A dead process is a fact, not an exception.** ``logs`` and ``restart_count``
answer for a role that is not running rather than raising, because a check that
wants to know *whether* something died should not have to catch an error to find
out. What they must never do is answer "nothing" for a role that was never
started; that is ``UNKNOWN``.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..facts import Fact
from ..roles import Role, Sel

__all__ = ["LocalRole", "LocalProvider"]


@dataclass
class LocalRole:
    """How to run one role locally."""

    role: Role
    argv: Sequence[str]
    port: int | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    cwd: str | None = None
    ready_path: str = "/v1/models"

    # runtime state
    process: subprocess.Popen | None = field(default=None, repr=False)
    starts: int = 0
    log_path: Path | None = field(default=None, repr=False)


class LocalProvider:
    """Run roles as subprocesses on this machine."""

    def __init__(self, roles: Mapping[Role | str, LocalRole], log_dir: str | Path):
        self._roles: dict[Role, LocalRole] = {}
        for key, spec in roles.items():
            role = key if isinstance(key, Role) else Role(str(key).lower())
            self._roles[role] = spec
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------- lookup

    def _role(self, sel: Sel) -> LocalRole:
        try:
            return self._roles[sel.role]
        except KeyError:
            raise KeyError(
                f"no local role {sel.role}; this provider runs: "
                f"{', '.join(sorted(str(r) for r in self._roles)) or '<none>'}"
            ) from None

    def _source(self, sel: Sel) -> str:
        return f"local:{sel.describe()}"

    # ------------------------------------------------------------- lifecycle

    def start(self, sel: Sel) -> Mapping[str, Any]:
        spec = self._role(sel)
        if spec.process is not None and spec.process.poll() is None:
            return {"already_running": True, "pid": spec.process.pid}

        spec.starts += 1
        spec.log_path = self.log_dir / f"{sel.role}.{spec.starts}.log"
        env = {**os.environ, **spec.env}
        # Line-buffered, so a check reading logs mid-run sees whole lines rather
        # than whatever happened to be flushed.
        env.setdefault("PYTHONUNBUFFERED", "1")
        handle = spec.log_path.open("w")
        spec.process = subprocess.Popen(
            list(spec.argv),
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=spec.cwd,
            env=env,
            start_new_session=True,  # so stop() cannot signal the test runner
        )
        return {
            "pid": spec.process.pid,
            "starts": spec.starts,
            "log": str(spec.log_path),
        }

    def stop(
        self, sel: Sel, *, graceful: bool = True, grace: float = 5.0
    ) -> Mapping[str, Any]:
        spec = self._role(sel)
        if spec.process is None or spec.process.poll() is not None:
            return {"already_stopped": True}

        pid = spec.process.pid
        if graceful:
            spec.process.terminate()
            try:
                spec.process.wait(timeout=grace)
                return {
                    "pid": pid,
                    "signal": "SIGTERM",
                    "returncode": spec.process.returncode,
                }
            except subprocess.TimeoutExpired:
                pass
        spec.process.kill()
        spec.process.wait(timeout=grace)
        return {"pid": pid, "signal": "SIGKILL", "returncode": spec.process.returncode}

    def restart(self, sel: Sel, **settings: Any) -> Mapping[str, Any]:
        """Stop, apply any flag changes, start.

        Settings go through :class:`~dynamo_test.argv.ArgV`, so a flag that is
        already set is **replaced** rather than appended. Appending is a measured
        defect elsewhere: the engine takes one occurrence and the test believes
        the other.
        """
        spec = self._role(sel)
        before = self.restart_count(sel).or_else(0)
        stopped = self.stop(sel)
        if settings:
            from ..argv import ArgV

            argv = ArgV.argv(spec.argv[1:], command=spec.argv[:1], source=str(sel.role))
            for flag, value in settings.items():
                flag = flag if flag.startswith("-") else f"--{flag.replace('_', '-')}"
                argv = argv.set(flag, value)
            spec.argv = list(spec.argv[:1]) + argv.as_container_args()
        started = self.start(sel)
        return {
            "before": before,
            "stopped": dict(stopped),
            **started,
            "restart_count": spec.starts - 1,
        }

    # ---------------------------------------------------------------- reads

    def address(self, sel: Sel) -> Fact[str]:
        spec = self._role(sel)
        if spec.port is None:
            return Fact.absent(self._source(sel), f"{sel.role} declares no port")
        return Fact.known(f"http://127.0.0.1:{spec.port}", self._source(sel))

    def replicas(self, sel: Sel) -> Fact[int]:
        spec = self._role(sel)
        running = spec.process is not None and spec.process.poll() is None
        return Fact.known(1 if running else 0, self._source(sel))

    def restart_count(self, sel: Sel) -> Fact[int]:
        """How many times this role has been (re)started, minus the first.

        ``UNKNOWN`` before the first start: "never started" and "started once and
        never restarted" are different, and only one of them means a restart
        assertion can be evaluated.
        """
        spec = self._role(sel)
        if spec.starts == 0:
            return Fact.unknown(self._source(sel), "role has never been started")
        return Fact.known(spec.starts - 1, self._source(sel))

    def logs(self, sel: Sel) -> Fact[str]:
        spec = self._role(sel)
        if spec.log_path is None:
            return Fact.unknown(
                self._source(sel), "role has never been started, so there is no log"
            )
        if not spec.log_path.exists():
            return Fact.unknown(
                self._source(sel), f"log file {spec.log_path} does not exist"
            )
        return Fact.known(
            spec.log_path.read_text(), f"{self._source(sel)}:{spec.log_path}"
        )

    def all_logs(self) -> dict[str, str]:
        """Every role's current log, for the COLLECT phase."""
        out = {}
        for role, spec in self._roles.items():
            if spec.log_path and spec.log_path.exists():
                out[str(role)] = spec.log_path.read_text()
        return out

    def request(
        self,
        sel: Sel,
        path: str,
        *,
        method: str = "GET",
        body: Any = None,
        timeout: float = 30.0,
    ) -> Fact[Any]:
        """An HTTP request to a role.

        Every failure mode is a ``Fact``, not an exception: a check asking "did
        this respond?" should read the answer, not catch it. The detail carries
        the status or the errno so a failure names itself.
        """
        base = self.address(sel)
        if not base.is_known:
            return Fact.unknown(base.source, base.detail)
        url = base.require().rstrip("/") + path
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data else {},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode()
                try:
                    return Fact.known(
                        json.loads(raw), f"{url}", f"HTTP {response.status}"
                    )
                except json.JSONDecodeError:
                    return Fact.known(
                        raw, f"{url}", f"HTTP {response.status}, not JSON"
                    )
        except urllib.error.HTTPError as exc:
            return Fact.absent(url, f"HTTP {exc.code}: {exc.reason}")
        except urllib.error.URLError as exc:
            return Fact.unknown(url, f"unreachable: {exc.reason}")
        except TimeoutError:
            return Fact.unknown(url, f"timed out after {timeout}s")

    # ------------------------------------------------------------ teardown

    def shutdown(self) -> None:
        """Stop every role. Safe to call twice."""
        for role in list(self._roles):
            try:
                self.stop(Sel(role=role))
            except Exception:  # noqa: BLE001 - teardown must not mask a real failure
                spec = self._roles[role]
                if spec.process and spec.process.poll() is None:
                    try:
                        os.killpg(os.getpgid(spec.process.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass

    def collect_into(self, sut: Any, recorder: Any) -> None:
        """A COLLECT-phase collector: write every role's log into the bundle.

        Declares before writing, so a role whose log never arrives shows up in
        the seal rather than as a check that quietly found nothing.
        """
        for role, spec in sorted(self._roles.items(), key=lambda kv: str(kv[0])):
            artifact = f"roles/{role}/current.log"
            recorder.declare(artifact, "local-provider", required=False)
            if spec.log_path and spec.log_path.exists():
                recorder.write_text(artifact, spec.log_path.read_text())
