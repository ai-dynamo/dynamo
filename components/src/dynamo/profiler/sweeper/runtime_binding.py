# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime binding resolution (tracking issue #13545, item 3): "Bind the
target runtime/backend version, compatible renderer, and matching
performance data" for one Candidate.

Confirmed against the real aiconfigurator source
(python/aisimulate/src/aiconfigurator/cli/main.py, _run_support_mode):

    result = common.check_support(
        model=model, system=system, backend=backend,
        version=version, architecture=architecture,
    )
    result.agg_supported / result.disagg_supported / result.exact_match

and against aiconfigurator.sdk.task_v2._lookup_num_gpus_per_node, which
resolves GPUs-per-node for a hardware SKU -- the proper API surface for
this, not a hand-rolled read of the systems/*.yaml catalog files.

RISK, stated plainly: _lookup_num_gpus_per_node is underscore-prefixed --
not confirmed as a stable, externally-callable interface. A message asking
aiconfigurator's maintainers to confirm this (or point at a public
equivalent) has been sent but not yet answered as of this module's
authorship. common.check_support is not underscore-prefixed and reads as
the intended public entry point, but has not been independently confirmed
stable either. Both are wrapped in a single injectable dependency so a
maintainer-provided public replacement can swap in without touching any
caller of resolve_runtime_binding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

# Confirmed empirically this session: TrtllmConfigModifier.set_config_tep_size/
# set_config_dep_size raise NotImplementedError for the direct renderer.
# Candidates using these strategies must use the aic renderer instead.
_DIRECT_RENDERER_UNSUPPORTED_STRATEGIES: frozenset[tuple[str, str]] = frozenset(
    {("trtllm", "tep"), ("trtllm", "dep")}
)

_RUNTIME_IMAGE_REGISTRY = "nvcr.io/nvidia/ai-dynamo"  # matches every real run this session


class RuntimeBindingError(ValueError):
    """The candidate cannot be bound to a runtime -- e.g. no matching
    performance data. Per #13200's error taxonomy this is an ordinary,
    per-candidate outcome (skip this candidate, keep searching), not a
    terminal failure of the whole run -- mirrors how MaterializationError
    is already handled one layer up in materializer.py.
    """


class SupportChecker(Protocol):
    """Matches common.check_support's real, confirmed signature and return
    shape (.agg_supported / .disagg_supported / .exact_match)."""

    def __call__(
        self, *, model: str, system: str, backend: str, version: str, architecture: str | None
    ) -> Any: ...


class NumGPUsPerNodeLookup(Protocol):
    def __call__(self, system: str) -> int: ...


@dataclass(frozen=True)
class RuntimeBinding:
    runtime_image: str
    num_gpus_per_node: int
    renderer: str  # "direct" | "aic"


def _default_support_checker() -> SupportChecker:
    from aiconfigurator.sdk import common  # noqa: PLC0415 -- lazy, same reason as stack_provider.py

    return common.check_support


def _default_num_gpus_per_node_lookup() -> NumGPUsPerNodeLookup:
    from aiconfigurator.sdk.task_v2 import _lookup_num_gpus_per_node  # noqa: PLC0415

    return _lookup_num_gpus_per_node


def resolve_runtime_binding(
    candidate_config: Mapping[str, Any],
    *,
    model: str,
    architecture: str | None = None,
    check_support: SupportChecker | None = None,
    lookup_num_gpus_per_node: NumGPUsPerNodeLookup | None = None,
) -> RuntimeBinding:
    """Resolve runtime image, num_gpus_per_node, and renderer for one
    Candidate. Raises RuntimeBindingError if no matching performance data
    exists for this candidate's deployment_mode.

    check_support/lookup_num_gpus_per_node default to the real
    aiconfigurator functions (imported lazily, so this module stays
    importable without aiconfigurator installed) but are injectable for
    testing without the real package -- same pattern as create_stack()'s
    _load_runner_factory in stack_provider.py.
    """
    check_support = check_support or _default_support_checker()
    lookup_num_gpus_per_node = lookup_num_gpus_per_node or _default_num_gpus_per_node_lookup()

    backend = candidate_config["backend"]
    backend_version = candidate_config["backend_version"]
    hardware_sku = candidate_config["hardware_sku"]
    deployment_mode = candidate_config["deployment_mode"]  # "agg" | "disagg"
    strategy = candidate_config.get("strategy")

    result = check_support(
        model=model,
        system=hardware_sku,
        backend=backend,
        version=backend_version,
        architecture=architecture,
    )
    # deployment_mode-specific, deliberately not a blanket "is it supported at
    # all" check -- agg_supported and disagg_supported are independent in the
    # real result shape, and checking the wrong one (or both) would silently
    # accept a candidate that can't actually be rendered.
    supported = result.disagg_supported if deployment_mode == "disagg" else result.agg_supported
    if not supported:
        raise RuntimeBindingError(
            f"no matching performance data for model={model!r} system={hardware_sku!r} "
            f"backend={backend!r} version={backend_version!r} mode={deployment_mode!r}"
        )

    num_gpus_per_node = lookup_num_gpus_per_node(hardware_sku)

    renderer = (
        "aic"
        if (backend, strategy) in _DIRECT_RENDERER_UNSUPPORTED_STRATEGIES
        else "direct"
    )

    return RuntimeBinding(
        runtime_image=f"{_RUNTIME_IMAGE_REGISTRY}/{backend}-runtime:{backend_version}",
        num_gpus_per_node=num_gpus_per_node,
        renderer=renderer,
    )
