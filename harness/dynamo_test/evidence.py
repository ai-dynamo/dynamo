# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collecting evidence, sealing it, and judging it afterwards.

A run has two questions to answer and they are not the same one:

*Did the system under test behave?* — the **verdict**.
*Did we actually measure it?* — the **validity** of the run.

Conflating them is how a suite reports green after collecting nothing. A check
that globs for a log file, finds none, and concludes "no errors" has answered the
first question using the failure of the second.

So collection and judgement are separated by a seal:

**COLLECT** — a :class:`Recorder` *declares* what it intends to produce, then
produces it. Declaring first is what makes a missing artifact detectable: without
a promise there is nothing for absence to contradict.

**CHECK** — an :class:`Evidence` view over a sealed bundle. It is read-only,
synchronous, and needs no cluster, so the same judgement runs in CI and hours
later on a laptop from the bundle directory alone. Checks resolve artifacts
**through the producer index, never by globbing** — an artifact nobody declared
is `UNKNOWN`, and a glob that matches nothing is indistinguishable from a
measurement of nothing.

The seal compares promises to delivery. An artifact that was declared and never
arrived, or that arrived with fewer rows than promised, makes the run
`INVALID` — which is a different outcome from `FAILED`, and must be, because
re-running is the right response to one and filing a bug is the right response
to the other.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .facts import Fact

__all__ = [
    "Verdict",
    "Outcome",
    "Producer",
    "Promise",
    "Seal",
    "Recorder",
    "Evidence",
    "BundleError",
    "Sealed",
    "NotDeclared",
    "SCHEMA_VERSION",
]

SCHEMA_VERSION = 1


class BundleError(Exception):
    """The evidence bundle was used in a way that would produce a wrong answer."""


class Sealed(BundleError):
    """A write was attempted after the bundle was sealed."""


class NotDeclared(BundleError):
    """An artifact was written that nothing declared."""


class Verdict(str, Enum):
    """What a run concluded, in precedence order.

    ``INVALID`` outranks ``FAILED`` deliberately. If the measurement did not
    happen, the system under test has not been judged, and reporting a failure
    would attribute a harness problem to the product.
    """

    INVALID = "invalid"
    FAILED = "failed"
    PASSED = "passed"
    OBSERVED = "observed"
    SKIPPED = "skipped"

    @property
    def rank(self) -> int:
        return _RANK[self]

    @property
    def exit_code(self) -> int:
        return _EXIT[self]


_RANK = {
    Verdict.INVALID: 0,
    Verdict.FAILED: 1,
    Verdict.PASSED: 2,
    Verdict.OBSERVED: 3,
    Verdict.SKIPPED: 4,
}

# Distinct codes so a CI lane can retry an invalid run and escalate a failed one.
_EXIT = {
    Verdict.INVALID: 3,
    Verdict.FAILED: 1,
    Verdict.PASSED: 0,
    Verdict.OBSERVED: 0,
    Verdict.SKIPPED: 0,
}


@dataclass(frozen=True)
class Outcome:
    """One judgement: what was asked, what was found, and what it counts as."""

    check: str
    verdict: Verdict
    detail: str = ""
    evidence: tuple[str, ...] = ()

    def to_record(self) -> dict:
        return {
            "check": self.check,
            "verdict": self.verdict.value,
            "detail": self.detail,
            "evidence": list(self.evidence),
        }


def combine(outcomes: Iterable[Outcome]) -> Verdict:
    """The run's verdict: the most severe outcome present.

    An empty set of outcomes is ``INVALID``, not ``PASSED``. A run that judged
    nothing has not established that anything works, and reporting success for it
    is the largest false green available.
    """
    outcomes = list(outcomes)
    if not outcomes:
        return Verdict.INVALID
    return min((o.verdict for o in outcomes), key=lambda v: v.rank)


@dataclass(frozen=True)
class Producer:
    """Who made an artifact, and with what."""

    artifact: str
    producer: str
    tool: str = ""
    tool_version: str = ""

    def to_record(self) -> dict:
        return {
            "producer": self.producer,
            "tool": self.tool,
            "tool_version": self.tool_version,
        }


@dataclass(frozen=True)
class Promise:
    """What a producer said it would deliver, before it delivered it.

    ``min_rows`` is the part that matters. "The file exists" is a weak promise —
    an empty ``requests.jsonl`` satisfies it while proving nothing. A load
    generator that promises at least one record and delivers none has not
    measured the system, and the run is invalid rather than passing.
    """

    artifact: str
    producer: str
    min_rows: int | None = None
    required: bool = True

    def to_record(self) -> dict:
        return {
            "producer": self.producer,
            "min_rows": self.min_rows,
            "required": self.required,
        }


@dataclass(frozen=True)
class Seal:
    """Declared promises against measured delivery."""

    kept: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    short: tuple[tuple[str, int, int], ...] = ()  # (artifact, promised, actual)
    undeclared: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.missing and not self.short

    def verdict(self) -> Verdict:
        return Verdict.PASSED if self.is_valid else Verdict.INVALID

    def why(self) -> str:
        parts = []
        if self.missing:
            parts.append(f"declared but never written: {', '.join(self.missing)}")
        if self.short:
            parts.append(
                "fewer rows than promised: "
                + ", ".join(f"{a} promised {p}, got {g}" for a, p, g in self.short)
            )
        if self.undeclared:
            parts.append(f"written without a promise: {', '.join(self.undeclared)}")
        return "; ".join(parts) or "every promise kept"

    def to_record(self) -> dict:
        return {
            "valid": self.is_valid,
            "kept": list(self.kept),
            "missing": list(self.missing),
            "short": [list(s) for s in self.short],
            "undeclared": list(self.undeclared),
            "why": self.why(),
        }


class Recorder:
    """The COLLECT phase: declare, then write, then seal.

    Runs before teardown, unconditionally — including when the test has already
    failed, because a failed run is exactly when its evidence matters most.
    """

    def __init__(self, root: str | Path, run_id: str) -> None:
        self.root = Path(root) / run_id
        self.run_id = run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self._promises: dict[str, Promise] = {}
        self._producers: dict[str, Producer] = {}
        self._rows: dict[str, int] = {}
        self._sealed = False
        self._meta: dict[str, Any] = {}

    # ------------------------------------------------------------- declare

    def declare(
        self,
        artifact: str,
        producer: str,
        *,
        min_rows: int | None = None,
        required: bool = True,
        tool: str = "",
        tool_version: str = "",
    ) -> Promise:
        """Promise an artifact before producing it.

        Declaring is not bookkeeping. It is what turns "the file is not there"
        from an unanswerable question into a contradiction of a specific claim.
        """
        self._refuse_if_sealed(f"declare {artifact!r}")
        promise = Promise(artifact, producer, min_rows=min_rows, required=required)
        self._promises[artifact] = promise
        self._producers[artifact] = Producer(artifact, producer, tool, tool_version)
        return promise

    # --------------------------------------------------------------- write

    def write_text(self, artifact: str, text: str) -> Path:
        path = self._path_for(artifact)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        self._rows[artifact] = text.count("\n") + (0 if text.endswith("\n") else 1)
        if not text:
            self._rows[artifact] = 0
        return path

    def write_json(self, artifact: str, payload: Any) -> Path:
        path = self._path_for(artifact)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        self._rows[artifact] = len(payload) if isinstance(payload, (list, dict)) else 1
        return path

    def write_rows(self, artifact: str, rows: Sequence[Mapping[str, Any]]) -> Path:
        """Write JSONL. The row count is what a promise is checked against."""
        path = self._path_for(artifact)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        self._rows[artifact] = len(rows)
        return path

    def append_row(self, artifact: str, row: Mapping[str, Any]) -> None:
        path = self._path_for(artifact)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        self._rows[artifact] = self._rows.get(artifact, 0) + 1

    def note(self, key: str, value: Any) -> None:
        """Record run-level metadata: plan digest, site, roles, grants."""
        self._refuse_if_sealed(f"note {key!r}")
        self._meta[key] = value

    # ---------------------------------------------------------------- seal

    def seal(self) -> Seal:
        """Close the bundle and compare promises to delivery."""
        if self._sealed:
            raise Sealed(f"{self.run_id} is already sealed")

        kept, missing, short = [], [], []
        for artifact, promise in sorted(self._promises.items()):
            path = self._path_for(artifact)
            if not path.exists():
                if promise.required:
                    missing.append(artifact)
                continue
            rows = self._rows.get(artifact, 0)
            if promise.min_rows is not None and rows < promise.min_rows:
                short.append((artifact, promise.min_rows, rows))
                continue
            kept.append(artifact)

        undeclared = sorted(set(self._written()) - set(self._promises))
        seal = Seal(
            kept=tuple(kept),
            missing=tuple(missing),
            short=tuple(short),
            undeclared=tuple(undeclared),
        )

        (self.root / "producers.json").write_text(
            json.dumps(
                {a: p.to_record() for a, p in sorted(self._producers.items())},
                indent=2,
                sort_keys=True,
            )
        )
        (self.root / "declared.json").write_text(
            json.dumps(
                {a: p.to_record() for a, p in sorted(self._promises.items())},
                indent=2,
                sort_keys=True,
            )
        )
        (self.root / "run.json").write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": self.run_id,
                    "seal": seal.to_record(),
                    "rows": dict(sorted(self._rows.items())),
                    **self._meta,
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
        self._sealed = True
        return seal

    # -------------------------------------------------------------- helpers

    def _path_for(self, artifact: str) -> Path:
        self._refuse_if_sealed(f"write {artifact!r}")
        if artifact not in self._promises:
            raise NotDeclared(
                f"{artifact!r} was written but never declared. Call declare() first "
                "— an artifact with no promise cannot be checked for absence, which "
                "is the whole point of the seal."
            )
        if os.path.isabs(artifact) or ".." in Path(artifact).parts:
            raise BundleError(f"artifact name must be a relative path: {artifact!r}")
        return self.root / artifact

    def _written(self) -> list[str]:
        out = []
        for path in self.root.rglob("*"):
            if path.is_file() and path.name not in {
                "run.json",
                "producers.json",
                "declared.json",
                "findings.json",
            }:
                out.append(str(path.relative_to(self.root)))
        return out

    def _refuse_if_sealed(self, action: str) -> None:
        if self._sealed:
            raise Sealed(
                f"cannot {action}: {self.run_id} is sealed. Collection happens "
                "before the seal; judgement happens after it, and mixing them "
                "makes a run unreplayable."
            )

    def evidence(self) -> "Evidence":
        """The read view. Legal only once sealed."""
        if not self._sealed:
            raise BundleError(
                f"{self.run_id} is not sealed; call seal() at the end of COLLECT. "
                "Judging a bundle still being written gives a different answer on "
                "a replay."
            )
        return Evidence(self.root)


class Evidence:
    """The CHECK phase: a read-only view over a sealed bundle.

    Synchronous and free of I/O beyond the bundle directory, so the same
    judgement runs in CI and later on a laptop with only the directory. Nothing
    here can reach the system under test — by the time checks run, it may not
    exist.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        run = self.root / "run.json"
        if not run.exists():
            raise BundleError(
                f"{self.root} has no run.json; it is not a sealed evidence bundle"
            )
        self.run = json.loads(run.read_text())
        self.producers: dict[str, dict] = json.loads(
            (self.root / "producers.json").read_text()
        )
        self.declared: dict[str, dict] = json.loads(
            (self.root / "declared.json").read_text()
        )

    @classmethod
    def open(cls, root: str | Path) -> "Evidence":
        return cls(root)

    @property
    def run_id(self) -> str:
        return self.run.get("run_id", "<unknown>")

    @property
    def seal(self) -> Seal:
        s = self.run.get("seal", {})
        return Seal(
            kept=tuple(s.get("kept", [])),
            missing=tuple(s.get("missing", [])),
            short=tuple(tuple(x) for x in s.get("short", [])),
            undeclared=tuple(s.get("undeclared", [])),
        )

    # ---------------------------------------------------------------- reads

    def _resolve(self, artifact: str) -> Fact[Path]:
        """Locate an artifact through the producer index, never by globbing.

        A glob that matches nothing and an artifact that was never produced look
        identical to the caller. Going through the index means "not collected"
        is a distinct, reportable answer.
        """
        if artifact not in self.producers:
            return Fact.unknown(
                f"{self.run_id}/{artifact}",
                f"nothing declared this artifact; the run produced: "
                f"{', '.join(sorted(self.producers)) or '<nothing>'}",
            )
        path = self.root / artifact
        if not path.exists():
            return Fact.unknown(
                f"{self.run_id}/{artifact}",
                f"declared by {self.producers[artifact]['producer']} but never written",
            )
        return Fact.known(path, f"{self.run_id}/{artifact}")

    def text(self, artifact: str) -> Fact[str]:
        path = self._resolve(artifact)
        if not path.is_known:
            return Fact.unknown(path.source, path.detail)
        return Fact.known(path.require().read_text(), path.source)

    def json(self, artifact: str) -> Fact[Any]:
        path = self._resolve(artifact)
        if not path.is_known:
            return Fact.unknown(path.source, path.detail)
        try:
            return Fact.known(json.loads(path.require().read_text()), path.source)
        except json.JSONDecodeError as exc:
            return Fact.unknown(path.source, f"not valid JSON: {exc}")

    def rows(self, artifact: str) -> Fact[tuple[dict, ...]]:
        """JSONL rows. A malformed line is UNKNOWN for the whole artifact.

        Skipping bad lines would silently change the denominator of every rate
        computed from them.
        """
        path = self._resolve(artifact)
        if not path.is_known:
            return Fact.unknown(path.source, path.detail)
        out = []
        for n, line in enumerate(path.require().read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                return Fact.unknown(path.source, f"line {n} is not valid JSON: {exc}")
        return Fact.known(tuple(out), path.source, f"{len(out)} row(s)")

    def artifacts(self) -> tuple[str, ...]:
        return tuple(sorted(self.producers))

    def produced_by(self, artifact: str) -> Fact[str]:
        if artifact not in self.producers:
            return Fact.absent(f"{self.run_id}/{artifact}", "not in the producer index")
        return Fact.known(
            self.producers[artifact]["producer"], f"{self.run_id}/{artifact}"
        )

    # ------------------------------------------------------------- judging

    def require_valid_run(self) -> Outcome:
        """The default admissibility check: did collection keep its promises?"""
        seal = self.seal
        return Outcome(
            check="require_valid_run",
            verdict=seal.verdict(),
            detail=seal.why(),
            evidence=seal.missing + tuple(a for a, _, _ in seal.short),
        )

    def judge(self, checks: Sequence[Any]) -> tuple[Verdict, tuple[Outcome, ...]]:
        """Run every check and combine, with validity evaluated first.

        An invalid run short-circuits: there is no point asking whether the
        system behaved when the measurement did not happen, and reporting a
        product failure from missing evidence is how a harness bug becomes
        someone else's bug report.
        """
        validity = self.require_valid_run()
        if validity.verdict is Verdict.INVALID:
            return Verdict.INVALID, (validity,)

        outcomes = [validity]
        for check in checks:
            name = getattr(check, "__name__", repr(check))
            try:
                result = check(self)
            except (
                Exception
            ) as exc:  # noqa: BLE001 - a broken check is not a SUT failure
                outcomes.append(
                    Outcome(
                        check=name,
                        verdict=Verdict.INVALID,
                        detail=f"the check itself raised {type(exc).__name__}: {exc}",
                    )
                )
                continue
            if isinstance(result, Outcome):
                outcomes.append(result)
            else:
                outcomes.append(
                    Outcome(
                        check=name,
                        verdict=Verdict.INVALID,
                        detail=f"returned {type(result).__name__}, expected an Outcome",
                    )
                )
        return combine(outcomes), tuple(outcomes)

    def write_findings(self, outcomes: Sequence[Outcome]) -> Path:
        """Record the judgement so a run can be re-scored and diffed."""
        path = self.root / "findings.json"
        path.write_text(
            json.dumps(
                {
                    "verdict": combine(outcomes).value,
                    "outcomes": [o.to_record() for o in outcomes],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return path

    def __iter__(self) -> Iterator[str]:
        return iter(self.artifacts())

    def __repr__(self) -> str:
        return f"Evidence({self.run_id!r}, {len(self.producers)} artifacts)"
