# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize one AI-Simulate-Sweeper ``Candidate`` into a real DGD.

This is the Python-side counterpart to DGDR v1beta2's DGDC
(``DynamoGraphDeploymentCandidate``): ``DynamoGraphDeploymentCandidate.Spec``
is typed as the real ``v1beta1.DynamoGraphDeploymentSpec`` (confirmed against
deploy/operator/api/v1beta2/dynamographdeploymentcandidate_types.go), and
``.Status.Experimental`` is a schema-free ``RawExtension`` -- the defined home
for evaluation-context values that never belonged in a DGD spec to begin
with. This module's return shape follows that split exactly: ``dgd`` maps to
DGDC.spec, ``experimental`` maps to DGDC.status.experimental.

Deliberately takes a plain dict (the shape of AI-Simulate-Sweeper's
``Candidate.config``) rather than importing ``aisimulate.sweeper.Candidate``
directly. Two reasons, both load-bearing, not just convenience:

1. AI-Simulate-Sweeper is published as a separate, independently-versioned
   package (aisimulate==0.1.0.dev1 per pyproject.toml / requirements
   .aisimulate.txt) and explicitly does not depend on Dynamo. A hard import
   the other direction would not break that boundary technically (Dynamo
   already depends on aisimulate), but pinning this module's contract to one
   exact aisimulate release's class shape would recreate the coupling the
   decoupling work (#12441) was for. A documented dict shape is a looser,
   more stable contract across aisimulate releases.
2. It keeps this module's own tests independent of having a real aisimulate
   wheel installed.

Runs inside the Sweeper Job's Python process, alongside (not inside)
aisimulate -- see the DGDR v1beta2 architecture discussion: a Go publisher
cannot reuse this Python templating logic, so the materialized DGD has to be
produced here, before the payload ever reaches the publisher over the
observer socket.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dynamo.planner.config.defaults import SubComponentType
from dynamo.profiler.utils.config import update_image
from dynamo.profiler.utils.config_modifiers import CONFIG_MODIFIERS

# Candidate.config fields that describe the deployment shape and have a
# direct, confirmed CONFIG_MODIFIERS destination. Everything else on
# Candidate.config is evaluation context (see EVALUATION_CONTEXT_FIELDS)
# and is preserved in `experimental`, never silently dropped -- the DEP's
# own materialization rule: "An unmapped value is an error; silently using
# an engine default changes the evaluated Candidate."
_STRATEGY_SETTERS = {
    "tp": "set_config_tp_size",
    "tep": "set_config_tep_size",
    "dep": "set_config_dep_size",
}

# Fields on Candidate.config that describe *what the candidate was evaluated
# against*, not the deployment itself. Confirmed via a real Pareto search
# fixture that kv_load_ratio/concurrency can vary across candidates with an
# otherwise byte-identical deployment shape -- these must never be folded
# into the DGD spec, and must not be silently dropped either.
EVALUATION_CONTEXT_FIELDS = frozenset(
    {
        "concurrency",
        "kv_load_ratio",
        "hardware_sku",
        "gpu_budget",
        "min_gpu_budget",
        "context_length",
        "startup_time",
        "aic_nextn",
    }
)


class MaterializationError(Exception):
    """One Candidate could not be materialized into a DGD.

    Per the DEP: "A materialization failure is reported as a failed
    candidate outcome and does not create a DGDC." Callers should catch this
    specifically (not a bare Exception) and translate it into that failure
    outcome, rather than letting the Job crash uninformatively.
    """


@dataclass
class MaterializationResult:
    """The DGDC.spec / DGDC.status.experimental split, ready to publish."""

    dgd: dict[str, Any]
    experimental: dict[str, Any] = field(default_factory=dict)


def materialize_dgd_from_candidate(
    candidate_config: dict[str, Any],
    *,
    image: str,
    component_type: SubComponentType = SubComponentType.DECODE,
) -> MaterializationResult:
    """Materialize one Candidate.config dict into a real DGD.

    ``image`` must already be resolved by the caller (backend + backend
    version -> container image). That lookup is a separate, still-open
    concern -- confirmed absent from build_dgd_config and everywhere else in
    this package -- and deliberately not this function's job.

    Raises MaterializationError for a known, explicit gap (e.g. an
    unsupported strategy/backend combination) rather than letting the
    underlying NotImplementedError/KeyError surface directly, so callers get
    one exception type to catch regardless of which CONFIG_MODIFIERS call
    failed and why.
    """
    backend = candidate_config.get("backend")
    mode = candidate_config.get("deployment_mode")
    strategy = candidate_config.get("strategy")

    modifier = CONFIG_MODIFIERS.get(backend)
    if modifier is None:
        raise MaterializationError(
            f"no CONFIG_MODIFIERS entry for backend {backend!r}"
        )
    if mode not in ("agg", "disagg"):
        raise MaterializationError(f"unsupported deployment_mode {mode!r}")

    setter_name = _STRATEGY_SETTERS.get(strategy)
    if setter_name is None:
        raise MaterializationError(f"unknown parallelism strategy {strategy!r}")
    setter = getattr(modifier, setter_name, None)
    if setter is None:
        raise MaterializationError(
            f"{backend} modifier has no {setter_name} implementation"
        )

    try:
        config = modifier.load_default_config(mode=mode)
        config = update_image(config, image)

        if strategy == "tp":
            config = setter(config, candidate_config["tp"], component_type)
        else:
            # tep/dep setters additionally require num_gpus_per_node; the
            # DEP's MVP scope is single-node agg, so this is pinned rather
            # than threaded through from the candidate for now -- a real
            # multi-node materializer needs this to come from deployment
            # context, not the candidate itself.
            tp_or_ep_size = candidate_config["moe_tp" if strategy == "tep" else "moe_ep"]
            config = setter(
                config,
                tp_or_ep_size,
                num_gpus_per_node=8,
                component_type=component_type,
            )

        config = modifier.set_config_kv_cache(
            config,
            block_size=candidate_config["agg_block_size"],
            memory_fraction=candidate_config["agg_gpu_memory_utilization"],
            prefix_caching=candidate_config["agg_enable_prefix_caching"],
            component_type=component_type,
        )
    except NotImplementedError as exc:
        # e.g. TRT-LLM's set_config_tep_size: confirmed real and reachable --
        # a real Sweeper-produced Candidate (strategy="tep", backend="trtllm")
        # hits exactly this path. This is the DEP's "materialization failure"
        # case, not a bug in this function.
        raise MaterializationError(
            f"{backend}/{strategy} materialization not supported: {exc}"
        ) from exc
    except KeyError as exc:
        raise MaterializationError(
            f"candidate_config missing required field: {exc}"
        ) from exc

    experimental = {
        key: candidate_config[key]
        for key in EVALUATION_CONTEXT_FIELDS
        if key in candidate_config
    }
    return MaterializationResult(dgd=config, experimental=experimental)
