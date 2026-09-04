# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test harness for Dynamo.

Tier 0 is values: stdlib only, no pytest, no HTTP, no Kubernetes, and — by
design — no ``dynamo.*``. Welding the harness to the runtime would make it
impossible to point one suite at an older release.
"""

from .argv import ArgForm, ArgV, is_shell_command_flag
from .dialect import DIALECTS, Dialect, EngineDialect, detect, for_backend
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
from .verbs import (
    REGISTRY,
    Grant,
    Phase,
    Receiver,
    VerbCall,
    VerbRegistry,
    VerbSpec,
    Verdict,
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
    # roles and selection
    "Policy",
    "PortName",
    "Process",
    "Role",
    "RoleBinding",
    "RoleTable",
    "Sel",
    "UnknownRole",
    "at",
    # verbs
    "REGISTRY",
    "Grant",
    "Phase",
    "Receiver",
    "VerbCall",
    "VerbRegistry",
    "VerbSpec",
    "Verdict",
    "verb",
]
