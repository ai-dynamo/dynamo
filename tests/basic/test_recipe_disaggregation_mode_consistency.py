# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repo-consistency test for ``--disaggregation-mode`` in recipe deployments.

A disaggregated deployment splits work across a ``prefill`` and a ``decode``
worker, and the prefill worker declares ``needs: [["decode"]]``. A worker that
never receives ``--disaggregation-mode`` registers as ``worker_type:
aggregated``, so the frontend's readiness check finds no worker of the required
type and never publishes the model.

That failure is silent in the worst way: every pod reaches ``1/1 Running`` and
``/health`` reports healthy with all instances up, while ``/v1/models`` returns
an empty list and ``/v1/chat/completions`` answers 503. The real cause surfaces
only in ``/v1/models/<name>/ready`` as "no namespace has all required worker
types live". Nothing fails loudly enough to catch in review.

The invariant asserted here is **symmetry**, not presence. Some recipe families
drive the mode by another mechanism and declare it on neither role; those are
left alone deliberately. What is never correct is one role declaring the mode
while its peer does not, because that is precisely the shape that produces a
deployment which looks healthy and serves nothing.

Also asserts the declared value matches the service's own ``subComponentType``,
so a copy-paste that leaves ``prefill`` on the decode worker is caught too.
"""

from pathlib import Path
from typing import Dict, List, Optional

import pytest
import yaml

# Reads files only, so it is cheap enough to gate every merge.
pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.post_merge,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]

REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPES_DIR = REPO_ROOT / "recipes"
FLAG = "--disaggregation-mode"
ROLES = ("prefill", "decode")


def _worker_args(service: Dict) -> List[str]:
    """Args for a service, whether nested under extraPodSpec or given directly."""
    container = (service.get("extraPodSpec") or {}).get("mainContainer") or {}
    args = container.get("args") or service.get("args") or []
    return [str(a) for a in args] if isinstance(args, list) else []


def _declared_mode(args: List[str]) -> Optional[str]:
    """The value passed to --disaggregation-mode, or None when absent."""
    if FLAG not in args:
        return None
    i = args.index(FLAG)
    return args[i + 1] if i + 1 < len(args) else ""


def _deployment_docs():
    """Yield (path, doc) for every parsable YAML doc under recipes/."""
    for path in sorted(RECIPES_DIR.rglob("*.y*ml")):
        try:
            docs = list(yaml.safe_load_all(path.read_text()))
        except yaml.YAMLError:
            continue  # non-YAML or templated; other tests cover parse validity
        for doc in docs:
            if isinstance(doc, dict) and doc.get("spec", {}).get("services"):
                yield path, doc


def _role_modes(doc: Dict) -> Dict[str, Dict[str, Optional[str]]]:
    """Map role -> {service_name: declared mode or None} for prefill/decode."""
    out: Dict[str, Dict[str, Optional[str]]] = {r: {} for r in ROLES}
    for name, service in (doc["spec"]["services"] or {}).items():
        if not isinstance(service, dict):
            continue
        role = str(service.get("subComponentType") or "").lower()
        if role in ROLES:
            out[role][name] = _declared_mode(_worker_args(service))
    return out


def test_disaggregation_mode_is_declared_symmetrically():
    """One role declaring the mode while its peer does not is always a defect."""
    offenders = []
    for path, doc in _deployment_docs():
        modes = _role_modes(doc)
        if not (modes["prefill"] and modes["decode"]):
            continue  # not a prefill/decode pair; nothing to compare
        declared = {
            role: [n for n, m in svcs.items() if m is not None]
            for role, svcs in modes.items()
        }
        missing = {
            role: [n for n, m in svcs.items() if m is None]
            for role, svcs in modes.items()
        }
        if any(declared.values()) and any(missing.values()):
            rel = path.relative_to(REPO_ROOT)
            offenders.append(f"{rel}: declared by {declared} but missing on {missing}")
    assert not offenders, (
        "Disaggregated recipes must declare --disaggregation-mode on both the "
        "prefill and decode workers, or on neither. A one-sided declaration "
        "leaves the undeclared worker registering as 'aggregated', so the "
        "model never becomes ready while every pod still reports healthy:\n  "
        + "\n  ".join(offenders)
    )


def test_declared_disaggregation_mode_matches_role():
    """A decode worker must not claim 'prefill', and vice versa."""
    offenders = []
    for path, doc in _deployment_docs():
        for role, services in _role_modes(doc).items():
            for name, mode in services.items():
                if mode is not None and mode != role:
                    rel = path.relative_to(REPO_ROOT)
                    offenders.append(
                        f"{rel}: service '{name}' has subComponentType "
                        f"'{role}' but passes {FLAG} '{mode}'"
                    )
    assert not offenders, (
        "--disaggregation-mode must match the service's subComponentType:\n  "
        + "\n  ".join(offenders)
    )
