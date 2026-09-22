# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime binding resolution (tracking issue #13545, item 3): binds one
Candidate's target runtime image, GPU topology, and compatible renderer.
The private aiconfigurator GPU-topology lookup is isolated behind an
injectable dependency so it can be replaced without touching any caller.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

# TRT-LLM TEP and DEP strategies require the AIC renderer: the direct
# renderer raises NotImplementedError for them.
_DIRECT_RENDERER_UNSUPPORTED_STRATEGIES: frozenset[tuple[str, str]] = frozenset(
    {("trtllm", "tep"), ("trtllm", "dep")}
)

_RUNTIME_IMAGE_REGISTRY = "nvcr.io/nvidia/ai-dynamo"

# Matches renderers/base.py's _RUNTIME_VERSION_PATTERN exactly -- this
# module's canonical-version check must never disagree with the real
# downstream validation in DGDGenerationOptions.__post_init__.
_RUNTIME_VERSION_PATTERN = re.compile(
    r"^(0|[1-9][0-9]{0,3})\.(0|[1-9][0-9]{0,3})\.(0|[1-9][0-9]{0,3})$"
)


class RuntimeBindingError(ValueError):
    """The candidate cannot be bound to a runtime -- e.g. no matching
    performance data. Per #13200's error taxonomy this is an ordinary,
    per-candidate outcome (skip this candidate, keep searching), not a
    terminal failure of the whole run -- mirrors how MaterializationError
    is already handled one layer up in materializer.py.
    """


class SupportChecker(Protocol):
    def __call__(
        self,
        *,
        model: str,
        system: str,
        backend: str,
        version: str,
        architecture: str | None,
    ) -> Any: ...


class NumGPUsPerNodeLookup(Protocol):
    def __call__(self, system: str) -> int: ...


@dataclass(frozen=True)
class RuntimeBinding:
    runtime_image: str
    num_gpus_per_node: int
    renderer: str  # "direct" | "aic"
    runtime_version_override: str | None = None


def _default_support_checker() -> SupportChecker:
    from aiconfigurator.sdk import common

    return common.check_support


def _candidate_strategies(
    candidate_config: Mapping[str, Any],
) -> tuple[str | None, ...]:
    """Agg candidates carry a single `strategy`; disagg candidates carry
    per-role `prefill_strategy`/`decode_strategy` instead, and `strategy`
    itself is None for disagg. Returns every strategy that actually
    applies to this candidate so renderer selection checks all of them,
    not only a top-level field that's absent for half of all candidates.
    """
    if candidate_config.get("deployment_mode") == "disagg":
        return (
            candidate_config.get("prefill_strategy"),
            candidate_config.get("decode_strategy"),
        )
    return (candidate_config.get("strategy"),)


def resolve_runtime_binding(
    candidate_config: Mapping[str, Any],
    *,
    model: str,
    dynamo_version: str,
    architecture: str | None = None,
    check_support: SupportChecker | None = None,
    lookup_num_gpus_per_node: NumGPUsPerNodeLookup,
) -> RuntimeBinding:
    """Resolve runtime image, num_gpus_per_node, and renderer for one
    Candidate. Raises RuntimeBindingError if no matching performance data
    exists for this candidate's deployment_mode.

    lookup_num_gpus_per_node has no default: the only known real
    implementation, aiconfigurator.sdk.task_v2._lookup_num_gpus_per_node,
    does not exist in the latest released aiconfigurator (0.11.0) --
    confirmed by direct import attempt, not assumed -- and no on-disk
    hardware-catalog fallback ships in that release either. A silent
    default here would raise an opaque ImportError from deep inside
    aiconfigurator on every normal call instead of failing clearly at the
    call site. Callers must supply this explicitly until the real function
    ships in a release; requiring it now makes that an explicit, visible
    choice at every call site rather than a hidden landmine.

    dynamo_version is required for the same reason runtime_version_override
    exists on DGDGenerationOptions: a candidate's backend_version (e.g.
    "1.3.0rc10") is frequently not canonical MAJOR.MINOR.PATCH, and nothing
    about a Candidate can tell us Dynamo's own release version -- that is
    an environment-wide constant the caller must supply, not something
    derivable from search data.

    check_support defaults to the real aiconfigurator.sdk.common.check_support
    (imported lazily so this module stays importable without aiconfigurator
    installed) since that one IS confirmed present and working in the
    pinned release -- unlike the GPU lookup, it does not need to be
    required.
    """
    check_support = check_support or _default_support_checker()

    backend = candidate_config["backend"]
    backend_version = candidate_config["backend_version"]
    hardware_sku = candidate_config["hardware_sku"]
    deployment_mode = candidate_config["deployment_mode"]  # "agg" | "disagg"

    result = check_support(
        model=model,
        system=hardware_sku,
        backend=backend,
        version=backend_version,
        architecture=architecture,
    )
    supported = (
        result.disagg_supported if deployment_mode == "disagg" else result.agg_supported
    )
    if not supported:
        raise RuntimeBindingError(
            f"no matching performance data for model={model!r} system={hardware_sku!r} "
            f"backend={backend!r} version={backend_version!r} mode={deployment_mode!r}"
        )

    num_gpus_per_node = lookup_num_gpus_per_node(hardware_sku)

    strategies = _candidate_strategies(candidate_config)
    renderer = (
        "aic"
        if any(
            (backend, strategy) in _DIRECT_RENDERER_UNSUPPORTED_STRATEGIES
            for strategy in strategies
        )
        else "direct"
    )

    is_canonical = bool(_RUNTIME_VERSION_PATTERN.fullmatch(backend_version))

    return RuntimeBinding(
        runtime_image=f"{_RUNTIME_IMAGE_REGISTRY}/{backend}-runtime:{backend_version}",
        num_gpus_per_node=num_gpus_per_node,
        renderer=renderer,
        runtime_version_override=None if is_canonical else dynamo_version,
    )
