# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Three-valued facts.

Every observation the harness makes about a system under test is one of three
things, and conflating the last two is how test suites go falsely green:

``KNOWN``
    The source was readable and the thing is there. ``value`` is set.
``ABSENT``
    The source was readable and the thing is genuinely not there.
``UNKNOWN``
    The source could not be read, or could not represent the form the thing is
    written in. **Nothing** may be concluded.

The distinction is not academic. Two measured examples from this repository:

* ``_declared_tool_call_parser`` scanned argv tokens only. For recipes whose
  worker is launched through a shell the whole command is a single token, so the
  scan found nothing and the skip message said the recipe "configures no
  tool-call parser" — a false statement, in a green run, for 34 of 101 recipes.
* ``k8s_utils`` swallowed every exception and returned ``{}``, so a
  crash-looping pod reported "no restarts".

Both are ``UNKNOWN`` reported as ``ABSENT``. :class:`Fact` makes that confusion
expressible only on purpose: you cannot get at the value without naming which
case you are in, and :meth:`Fact.__bool__` raises so that ``if not fact:`` — the
shape both bugs took — does not compile into a silent wrong answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, NoReturn, TypeVar

T = TypeVar("T")

__all__ = ["Status", "Fact", "FactNotKnown"]


class Status(str, Enum):
    KNOWN = "known"
    ABSENT = "absent"
    UNKNOWN = "unknown"


class FactNotKnown(Exception):
    """Raised by :meth:`Fact.require` when a fact is not ``KNOWN``."""


@dataclass(frozen=True)
class Fact(Generic[T]):
    """An observation, its provenance, and whether it is trustworthy.

    ``source`` names *where the answer came from* precisely enough to go and
    look: ``"DGD spec.components[VllmDecodeWorker].args[0]"``, ``"nvidia-smi"``,
    ``"node label nvidia.com/gpu.present"``. It is not a description of the
    check; it is an address.
    """

    status: Status
    value: T | None
    source: str
    detail: str = ""

    @classmethod
    def known(cls, value: T, source: str, detail: str = "") -> "Fact[T]":
        return cls(Status.KNOWN, value, source, detail)

    @classmethod
    def absent(cls, source: str, why: str) -> "Fact[T]":
        """The source was read and the thing is genuinely not there.

        ``why`` must justify the *positive* claim of absence — what was read,
        and why that reading is complete. "not found" is not a justification.
        """
        return cls(Status.ABSENT, None, source, why)

    @classmethod
    def unknown(cls, source: str, why: str) -> "Fact[T]":
        """The source could not be read, or cannot represent this form."""
        return cls(Status.UNKNOWN, None, source, why)

    @property
    def is_known(self) -> bool:
        return self.status is Status.KNOWN

    @property
    def is_absent(self) -> bool:
        return self.status is Status.ABSENT

    @property
    def is_unknown(self) -> bool:
        return self.status is Status.UNKNOWN

    def require(self) -> T:
        """Return the value, or raise with the provenance attached.

        For callers that genuinely cannot proceed. The message carries
        ``source`` and ``detail`` so the failure names the thing that could not
        be read rather than the symptom three frames later.
        """
        if self.status is Status.KNOWN:
            return self.value  # type: ignore[return-value]
        raise FactNotKnown(
            f"{self.status.value} at {self.source}"
            + (f": {self.detail}" if self.detail else "")
        )

    def or_else(self, default: T) -> T:
        """Return the value if ``KNOWN``, else ``default``.

        Collapses ``ABSENT`` and ``UNKNOWN`` together, which is exactly the
        conflation this module exists to prevent. Legitimate only where the
        default is correct for *both* — a display string, a metrics label. If
        the caller would behave differently for a thing that is missing versus a
        thing that could not be read, branch on :attr:`status` instead.
        """
        return self.value if self.status is Status.KNOWN else default  # type: ignore[return-value]

    def __bool__(self) -> NoReturn:
        raise TypeError(
            "Fact has no truth value: `if fact:` cannot distinguish ABSENT from "
            "UNKNOWN, which is the bug this type exists to prevent. Branch on "
            "fact.status, or use fact.is_known / fact.require() / fact.or_else()."
        )
