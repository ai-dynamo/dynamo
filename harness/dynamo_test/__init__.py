# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test harness for Dynamo.

Tier 0 is values: stdlib only, no pytest, no HTTP, no Kubernetes, and — by
design — no ``dynamo.*``. Welding the harness to the runtime would make it
impossible to point one suite at an older release.
"""

from . import catalog as _catalog  # noqa: F401  (registers the standard verbs)
from .argv import ArgForm, ArgV, is_shell_command_flag
from .bringup import BringUp, bring_up
from .dialect import DIALECTS, Dialect, EngineDialect, detect, for_backend
from .evidence import Evidence, Outcome, Producer, Promise, Recorder, Seal, Verdict
from .facts import Fact, FactNotKnown, Status
from .roles import (
    Policy,
    PortName,
    Process,
    Role,
    RoleBinding,
    RoleTable,
    Sel,
    UnknownRole,
    at,
)
from .site import (
    BUILTIN_SITES,
    LOCAL,
    Capability,
    Ownership,
    Site,
    Topology,
    UnknownSite,
)
from .stack import DYNAMO_DEFAULTS, Defaults, Render, Stack
from .sut import Handle, NotGranted, PhaseError, Provider, Sut
from .verbs import (
    REGISTRY,
    Contribution,
    Grant,
    Phase,
    Receiver,
    VerbCall,
    VerbRegistry,
    VerbSpec,
    verb,
)

__all__ = [
    # values
    "ArgForm",
    "ArgV",
    "Fact",
    "FactNotKnown",
    "Status",
    "is_shell_command_flag",
    # engine dialects
    "DIALECTS",
    "Dialect",
    "EngineDialect",
    "detect",
    "for_backend",
    # evidence
    "Evidence",
    "Outcome",
    "Producer",
    "Promise",
    "Recorder",
    "Seal",
    "Verdict",
    # the system under test
    "Handle",
    "NotGranted",
    "Phase",
    "PhaseError",
    "Provider",
    "Sut",
    # roles and selection
    "Policy",
    "PortName",
    "Process",
    "Role",
    "RoleBinding",
    "RoleTable",
    "Sel",
    "UnknownRole",
    # sites and topology
    "BUILTIN_SITES",
    "LOCAL",
    "Capability",
    "Ownership",
    "Site",
    "Topology",
    "UnknownSite",
    "at",
    # verbs
    "REGISTRY",
    "Grant",
    "Phase",
    "Receiver",
    "VerbCall",
    "VerbRegistry",
    "Contribution",
    "VerbSpec",
    "verb",
    # stacks: a plan plus the defaults the operator would have supplied
    "DYNAMO_DEFAULTS",
    "Defaults",
    "Render",
    "Stack",
    # bring-up
    "BringUp",
    "bring_up",
]
