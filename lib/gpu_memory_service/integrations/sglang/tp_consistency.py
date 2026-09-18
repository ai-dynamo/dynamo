# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small, fail-closed agreement protocol for rank-local persistent KV state.

Each rank-local directory describes its shard of a logical TP page. Every
rank must agree on the content/page layout before native SGLang state uses it;
lease generations remain local fencing tokens and may differ between ranks.
These collectives use SGLang's existing CPU cache group, not the GPU serving
communicator.
"""

from __future__ import annotations


class GmsTPConsistencyError(RuntimeError):
    """A TP cohort cannot safely continue using shared KV metadata."""


class TPConsistency:
    def __init__(self, group=None, world_size: int = 1):
        self.group = group
        self.world_size = int(world_size)

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    def _gather(self, value):
        if not self.enabled:
            return [value]
        import torch.distributed as dist

        values = [None] * self.world_size
        try:
            dist.all_gather_object(values, value, group=self.group)
        except Exception as exc:
            # A broken collective is not a local cache miss. The caller must
            # stop the cohort instead of continuing with an unagreed prefix.
            raise GmsTPConsistencyError(
                "SGLang GMS TP agreement channel failed"
            ) from exc
        return values

    def agree(self, stage: str, value) -> None:
        values = self._gather((stage, value))
        if any(candidate != values[0] for candidate in values[1:]):
            raise GmsTPConsistencyError(f"SGLang GMS TP disagreement during {stage}")

    def attempt(self, stage: str, operation):
        """Vote on a reversible reservation attempt without losing its result."""
        result = None
        ok = True
        try:
            result = operation()
        except Exception:  # noqa: BLE001
            ok = False
        votes = self._gather((stage, ok))
        if any(vote[0] != stage for vote in votes):
            raise GmsTPConsistencyError(f"SGLang GMS TP stage mismatch: {stage}")
        return all(vote[1] for vote in votes), result

    def leader_true(self, stage: str, value: bool) -> bool:
        """Broadcast the leader's conservative candidate predicate.

        A stale false on the leader only delays a cache hit and falls back to
        recompute. A stale true is verified by the existing common-prefix
        lookup before mutation. Using a broadcast avoids an unnecessary
        reduction on every native prefix miss.
        """
        if not self.enabled:
            return bool(value)
        import torch
        import torch.distributed as dist

        vote = torch.tensor(
            [bool(value) if self._rank() == 0 else False], dtype=torch.uint8
        )
        try:
            dist.broadcast(vote, src=0, group=self.group)
        except Exception as exc:
            raise GmsTPConsistencyError(
                f"SGLang GMS TP agreement channel failed during {stage}"
            ) from exc
        return bool(vote.item())

    def _rank(self) -> int:
        if not self.enabled:
            return 0
        import torch.distributed as dist

        return dist.get_rank(group=self.group)

    def leader_call(self, stage: str, operation):
        """Execute one destructive directory operation and share its outcome."""
        result = None
        error = None
        try:
            if self._rank() == 0:
                result = operation()
        except Exception as exc:  # noqa: BLE001
            error = exc
        votes = self._gather((stage, error is None, result))
        if any(vote[:2] != (stage, True) for vote in votes):
            raise GmsTPConsistencyError(
                f"SGLang GMS TP leader operation failed during {stage}"
            ) from error
        return votes[0][2]

    def intersection(self, stage: str, candidates: list[int]) -> list[int]:
        votes = self._gather((stage, candidates))
        if any(vote[0] != stage for vote in votes):
            raise GmsTPConsistencyError(f"SGLang GMS TP stage mismatch: {stage}")
        common = set(votes[0][1])
        for _stage, values in votes[1:]:
            common.intersection_update(values)
        return sorted(common)

    def _gather_digest(
        self, stage: str, ok: bool, agreement: bytes
    ) -> list[tuple[bool, bytes]]:
        from hashlib import sha256

        import torch
        import torch.distributed as dist

        identity = sha256(stage.encode("utf-8") + b"\0" + bytes(agreement)).digest()
        vote = torch.tensor([ok, *identity], dtype=torch.uint8)
        votes = [torch.empty_like(vote) for _ in range(self.world_size)]
        try:
            dist.all_gather(votes, vote, group=self.group)
        except Exception as exc:
            raise GmsTPConsistencyError(
                f"SGLang GMS TP agreement channel failed during {stage}"
            ) from exc
        return [
            (bool(candidate[0].item()), bytes(candidate[1:].tolist()))
            for candidate in votes
        ]

    def transact_digest(self, stage: str, agreement: bytes, operation):
        """Run one local transaction and agree a fixed-size identity once.

        Publication is on SGLang's completion hot path. A 33-byte tensor vote
        avoids pickling a page-layout object on every rank while preserving the
        same fail-closed success and identity checks. Callers must compensate
        local side effects if this raises.
        """
        if not self.enabled:
            return operation()
        error = None
        result = None
        try:
            result = operation()
        except Exception as exc:  # noqa: BLE001
            error = exc
        votes = self._gather_digest(stage, error is None, agreement)
        if any(not ok for ok, _identity in votes):
            raise GmsTPConsistencyError(
                f"SGLang GMS TP transaction failed during {stage}"
            ) from error
        if any(identity != votes[0][1] for _ok, identity in votes[1:]):
            raise GmsTPConsistencyError(
                f"SGLang GMS TP disagreement during {stage}"
            )
        return result

    def run(self, stage: str, operation):
        """No rank proceeds after a peer's local operation failed."""
        if not self.enabled:
            return operation()
        error = None
        result = None
        try:
            result = operation()
        # The operation is deliberately opaque: every rank must reach the vote
        # even when engine or directory code raises an arbitrary exception.
        except Exception as exc:  # noqa: BLE001
            error = exc
        votes = self._gather((stage, error is None))
        if any(vote != (stage, True) for vote in votes):
            raise GmsTPConsistencyError(
                f"SGLang GMS TP operation failed during {stage}"
            ) from error
        return result

    def common_prefix(self, stage: str, candidates: list) -> list:
        """An independently stale read view may shorten, never split, a TP hit."""
        votes = self._gather((stage, candidates))
        if any(vote[0] != stage for vote in votes):
            raise GmsTPConsistencyError(f"SGLang GMS TP stage mismatch: {stage}")
        result = []
        for entries in zip(*(vote[1] for vote in votes)):
            if any(entry != entries[0] for entry in entries[1:]):
                break
            result.append(entries[0])
        return result

    def run_common_prefix(self, stage: str, operation):
        """Run a local lookup and agree its usable prefix in one collective."""
        value = None
        candidates = []
        error = None
        try:
            value, candidates = operation()
        except Exception as exc:  # noqa: BLE001
            error = exc
        votes = self._gather((stage, error is None, candidates))
        if any(vote[:2] != (stage, True) for vote in votes):
            raise GmsTPConsistencyError(
                f"SGLang GMS TP operation failed during {stage}"
            ) from error
        common = []
        for entries in zip(*(vote[2] for vote in votes)):
            if any(entry != entries[0] for entry in entries[1:]):
                break
            common.append(entries[0])
        return value, common
