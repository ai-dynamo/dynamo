# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Match the components of a v1beta1 DGD to a planner role.

Shared by the planner, which reads a live DGD, and the profiler, which edits a
DGD template. This module only reports the candidates. What to do with
duplicates, a generic worker, or no match is each caller's policy.
"""

from dataclasses import dataclass
from typing import Iterable, Optional

from dynamo.planner.config.defaults import SubComponentType

V1BETA1_GENERIC_WORKER_COMPONENT_TYPE = "worker"


@dataclass(frozen=True)
class RoleCandidates:
    """Names of the components that can serve a role, in ``spec.components`` order."""

    typed: tuple[str, ...]
    generic_workers: tuple[str, ...]

    @property
    def sole_generic_worker(self) -> Optional[str]:
        """The generic worker when the spec has exactly one, else ``None``."""
        if len(self.generic_workers) == 1:
            return self.generic_workers[0]
        return None


def match_role_components(
    components: Iterable[tuple[str, str]], role: SubComponentType
) -> RoleCandidates:
    """Return the components typed as ``role`` and the generic ``type: worker`` ones.

    Args:
        components: ``(name, type)`` of every component, in spec order.
        role: The role to resolve.
    """
    named_types = list(components)
    return RoleCandidates(
        typed=tuple(
            name for name, component_type in named_types if component_type == role.value
        ),
        generic_workers=tuple(
            name
            for name, component_type in named_types
            if component_type == V1BETA1_GENERIC_WORKER_COMPONENT_TYPE
        ),
    )
