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
        import pickle
        import struct

        import torch
        import torch.distributed as dist

        try:
            payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:  # noqa: BLE001
            # Vote on encoding failure too; never strand peers in a collective.
            payload = b""
        try:
            # Fixed frames remove all_gather_object's separate size exchange.
            # Keep the full payload/stage checks and EVERY transaction vote.
            frame_size = 4096
            frame = bytearray(frame_size)
            struct.pack_into("<I", frame, 0, len(payload))
            if len(payload) <= frame_size - 4:
                frame[4 : 4 + len(payload)] = payload
            vote = torch.frombuffer(frame, dtype=torch.uint8)
            outputs = [torch.empty_like(vote) for _ in range(self.world_size)]
            dist.all_gather(outputs, vote, group=self.group)
            frames = [output.numpy().tobytes() for output in outputs]
            sizes = [struct.unpack_from("<I", item)[0] for item in frames]
            if any(size == 0 for size in sizes):
                raise ValueError("could not encode TP agreement frame")
            if any(size > frame_size - 4 for size in sizes):
                # Every rank takes this fallback when ANY payload is oversized.
                values = [None] * self.world_size
                dist.all_gather_object(values, value, group=self.group)
                return values
            return [
                pickle.loads(item[4 : 4 + size]) for item, size in zip(frames, sizes)
            ]
        except Exception as exc:
            # A broken collective is not a local cache miss. The caller must
            # stop the cohort instead of continuing with an unagreed prefix.
            raise GmsTPConsistencyError(
                "SGLang GMS TP agreement channel failed"
            ) from exc

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

    def all_true(self, stage: str, value: bool) -> bool:
        """Return true only when every rank reached the same readiness point."""
        if not self.enabled:
            return bool(value)
        votes = self._gather((stage, bool(value)))
        if any(vote[0] != stage for vote in votes):
            raise GmsTPConsistencyError(f"SGLang GMS TP stage mismatch: {stage}")
        return all(bool(vote[1]) for vote in votes)

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

    def run_agreed(self, stage: str, operation):
        """Vote local success and logical identity in one transaction.

        The operation returns (local result, shared identity). Callers must
        compensate reversible local side effects if this raises. Generation
        tokens may remain rank-local; only the supplied identity is compared.
        """
        result = identity = None
        error = None
        try:
            result, identity = operation()
        except Exception as exc:  # noqa: BLE001
            error = exc
        votes = self._gather((stage, error is None, identity))
        if any(vote[:2] != (stage, True) for vote in votes):
            raise GmsTPConsistencyError(
                f"SGLang GMS TP operation failed during {stage}"
            ) from error
        if any(vote[2] != votes[0][2] for vote in votes[1:]):
            raise GmsTPConsistencyError(f"SGLang GMS TP disagreement during {stage}")
        return result

    def run_intersection(self, stage: str, operation):
        """Read local candidates and vote success plus membership together.

        No mutation is allowed in operation. A peer's read failure must still
        reach the same collective before any rank uses the candidate set.
        """
        local = []
        error = None
        try:
            local = list(operation())
        except Exception as exc:  # noqa: BLE001
            error = exc
        votes = self._gather((stage, error is None, local))
        if any(vote[:2] != (stage, True) for vote in votes):
            raise GmsTPConsistencyError(
                f"SGLang GMS TP operation failed during {stage}"
            ) from error
        common = set(votes[0][2])
        for _stage, _ok, candidates in votes[1:]:
            common.intersection_update(candidates)
        return local, sorted(common)

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
            raise GmsTPConsistencyError(f"SGLang GMS TP disagreement during {stage}")
        return result

    def run_agreed_digest(self, stage: str, operation):
        """Run a local transaction and agree its compact identity once.

        ``operation`` returns ``(local_result, agreement_bytes)``. This is the
        fixed-size counterpart of :meth:`run_agreed` for large identities such
        as a pressure-eviction batch. Sending the full Python object can cross
        the fixed-frame limit and fall back to ``all_gather_object`` on the
        scheduler thread; the digest retains fail-closed agreement without
        serializing the whole batch across ranks.
        """
        if not self.enabled:
            result, _agreement = operation()
            return result
        error = None
        result = None
        agreement = b""
        try:
            result, agreement = operation()
            agreement = bytes(agreement)
        except Exception as exc:  # noqa: BLE001
            error = exc
        votes = self._gather_digest(stage, error is None, agreement)
        if any(not ok for ok, _identity in votes):
            raise GmsTPConsistencyError(
                f"SGLang GMS TP operation failed during {stage}"
            ) from error
        if any(identity != votes[0][1] for _ok, identity in votes[1:]):
            raise GmsTPConsistencyError(f"SGLang GMS TP disagreement during {stage}")
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
