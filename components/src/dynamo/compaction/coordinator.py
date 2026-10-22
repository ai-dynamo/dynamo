# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed evidence selection and summary verification, without persistence."""

import asyncio
import math
from dataclasses import asdict, dataclass
from typing import Callable, Protocol

from .protocol import Invalid, Request, Result, Unit, canonical, require


class Models(Protocol):
    async def select(
        self, units: tuple[Unit, ...], context: tuple[Unit, ...], session_id: str
    ) -> dict:
        ...

    async def compact(
        self, units: tuple[Unit, ...], context: tuple[Unit, ...], session_id: str
    ) -> dict:
        ...

    async def verify(
        self,
        units: tuple[Unit, ...],
        claims: tuple[dict, ...],
        context: tuple[Unit, ...],
        session_id: str,
    ) -> dict:
        ...


@dataclass(frozen=True)
class Counter:
    name: str
    count: Callable[[str], int]
    qualified: bool

    def measure(self, text: str) -> int:
        value = self.count(text)
        require(type(value) is int and value >= 0, "invalid_counter")
        return value


def projection(summary: str | None, units: tuple[Unit, ...]) -> str:
    return canonical({"summary": summary, "retained": [asdict(unit) for unit in units]})


def selected_sources(
    response: dict, eligible: tuple[Unit, ...], minimum: float
) -> tuple[Unit, ...]:
    require(
        type(response) is dict and set(response) == {"selections"},
        "invalid_selection_schema",
    )
    rows = response["selections"]
    require(
        type(rows) is list and len(rows) <= len(eligible), "invalid_selection_count"
    )
    known = {unit.id for unit in eligible}
    seen, selected = set(), set()
    for row in rows:
        require(
            type(row) is dict and set(row) == {"id", "action", "score"},
            "invalid_selection_schema",
        )
        require(
            type(row["id"]) is str and row["id"] in known and row["id"] not in seen,
            "invalid_selection_id",
        )
        seen.add(row["id"])
        require(
            type(row["action"]) is str
            and row["action"] in {"keep", "summarize", "unknown"},
            "invalid_selection_action",
        )
        score = row["score"]
        if score is None:
            continue
        require(
            type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1,
            "invalid_selection_score",
        )
        if row["action"] == "summarize" and score >= minimum:
            selected.add(row["id"])
    return tuple(unit for unit in eligible if unit.id in selected)


def validate_claims(
    response: dict, evidence: tuple[Unit, ...], max_bytes: int
) -> tuple[dict, ...]:
    require(
        type(response) is dict and set(response) == {"claims"}, "invalid_summary_schema"
    )
    require(len(canonical(response).encode()) <= max_bytes, "summary_bytes_exceeded")
    rows = response["claims"]
    require(type(rows) is list and 0 < len(rows) <= 256, "invalid_claim_count")
    known = {unit.id for unit in evidence}
    covered = set()
    for row in rows:
        require(
            type(row) is dict and set(row) == {"text", "source_ids"},
            "invalid_claim_schema",
        )
        require(
            type(row["text"]) is str and bool(row["text"].strip()), "invalid_claim_text"
        )
        ids = row["source_ids"]
        require(
            type(ids) is list and bool(ids) and all(type(id_) is str for id_ in ids),
            "invalid_citations",
        )
        require(len(set(ids)) == len(ids) and set(ids) <= known, "invalid_citations")
        covered.update(ids)
    require(covered == known, "uncovered_evidence")
    return tuple(
        {"text": row["text"], "source_ids": tuple(row["source_ids"])} for row in rows
    )


def verified(response: dict, count: int) -> bool:
    require(
        type(response) is dict and set(response) == {"supported", "unsupported_claims"},
        "invalid_verifier_schema",
    )
    require(type(response["supported"]) is bool, "invalid_verifier_verdict")
    indices = response["unsupported_claims"]
    require(
        type(indices) is list
        and all(type(i) is int and 0 <= i < count for i in indices),
        "invalid_verifier_indices",
    )
    require(len(set(indices)) == len(indices), "invalid_verifier_indices")
    return response["supported"] and not indices


@dataclass(frozen=True)
class Pipeline:
    models: Models
    counter: Counter
    mode: str = "fixture"
    minimum_selection_score: float = 0.9

    async def run(self, request: Request) -> Result:
        try:
            return await asyncio.wait_for(
                self._run(request), request.budget.deadline_seconds
            )
        except asyncio.TimeoutError:
            return Result(
                "rejected",
                request.snapshot_sha256,
                None,
                tuple(unit.id for unit in request.units),
                "deadline_exceeded",
                self.mode,
                None,
            )

    async def _run(self, request: Request) -> Result:
        calls = 0
        try:
            require(self.mode in {"fixture", "model"}, "invalid_mode")
            require(
                type(self.minimum_selection_score) in (int, float)
                and math.isfinite(self.minimum_selection_score)
                and 0 < self.minimum_selection_score <= 1,
                "invalid_selection_policy",
            )
            require(
                self.mode == "fixture" or self.counter.qualified is True,
                "unqualified_counter",
            )
            require(request.budget.max_model_calls >= 3, "call_budget_exceeded")
            protected = tuple(unit for unit in request.units if unit.must_retain)
            require(
                self.counter.measure(projection(None, protected))
                <= request.budget.max_output_tokens,
                "protected_budget_exceeded",
            )
            eligible = tuple(unit for unit in request.units if not unit.must_retain)
            require(bool(eligible), "no_evidence")
            if eligible:
                calls += 1
                context = request.units + request.context_units
                selected = selected_sources(
                    await self.models.select(eligible, context, request.session_id),
                    eligible,
                    self.minimum_selection_score,
                )
                require(bool(selected), "no_selected_evidence")
                calls += 1
                claims = validate_claims(
                    await self.models.compact(selected, context, request.session_id),
                    selected,
                    request.budget.max_summary_bytes,
                )
                roles = {unit.id: unit.role for unit in selected}
                summary = canonical(
                    {
                        "kind": "untrusted_compaction_evidence",
                        "claims": [
                            {
                                **claim,
                                "source_roles": [
                                    roles[id_] for id_ in claim["source_ids"]
                                ],
                            }
                            for claim in claims
                        ],
                    }
                )
                require(
                    len(summary.encode()) <= request.budget.max_summary_bytes,
                    "summary_bytes_exceeded",
                )
                selected_ids = {unit.id for unit in selected}
                retained = tuple(
                    unit for unit in request.units if unit.id not in selected_ids
                )
                require(
                    all(unit in retained for unit in protected),
                    "protected_content_changed",
                )
                count = self.counter.measure(projection(summary, retained))
                require(
                    count <= request.budget.max_output_tokens, "output_budget_exceeded"
                )
                calls += 1
                require(
                    verified(
                        await self.models.verify(
                            selected, claims, context, request.session_id
                        ),
                        len(claims),
                    ),
                    "semantic_rejection",
                )
            result = Result(
                "accepted",
                request.snapshot_sha256,
                summary,
                tuple(unit.id for unit in retained),
                "validated_candidate",
                self.mode,
                calls,
                count,
                self.counter.name,
                self.counter.qualified,
            )
            require(
                len(canonical(result.wire()).encode())
                <= request.budget.max_output_bytes,
                "output_bytes_exceeded",
            )
            return result
        except (Invalid, TimeoutError) as error:
            reason = (
                "deadline_exceeded" if isinstance(error, TimeoutError) else str(error)
            )
            return Result(
                "rejected",
                request.snapshot_sha256,
                None,
                tuple(unit.id for unit in request.units),
                reason,
                self.mode,
                calls,
            )
