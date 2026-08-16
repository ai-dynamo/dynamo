# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Remote model placement with a selectable application response stage."""

from __future__ import annotations

import os
from typing import Literal, cast

from dynamo.workflow import (
    DeploymentSpec,
    ExecutionPlan,
    GenerateEndpointBinding,
    InlineBinding,
    RemoteBinding,
    compile_workflow,
)
from examples.custom_backend.user_ensemble.workflow import define_workflow

CLASSIFIER_ENDPOINT = "user-ensemble.classifier.generate"
GENERATOR_ENDPOINT = "user-ensemble.generator.generate"
RESPONSE_ENDPOINT = "user-ensemble.response.generate"
RESPONSE_PLACEMENT_ENV = "DYN_USER_ENSEMBLE_RESPONSE_PLACEMENT"
ResponsePlacement = Literal["inline", "remote"]


def configured_response_placement() -> ResponsePlacement:
    placement = os.environ.get(RESPONSE_PLACEMENT_ENV, "inline")
    if placement not in {"inline", "remote"}:
        raise ValueError(
            f"{RESPONSE_PLACEMENT_ENV} must be 'inline' or 'remote', got "
            f"{placement!r}"
        )
    return cast(ResponsePlacement, placement)


def compile_remote_workflow(
    *, response_placement: ResponsePlacement = "inline"
) -> ExecutionPlan:
    if response_placement not in {"inline", "remote"}:
        raise ValueError(
            "response_placement must be 'inline' or 'remote', got "
            f"{response_placement!r}"
        )
    response_binding = (
        InlineBinding("response")
        if response_placement == "inline"
        else RemoteBinding(RESPONSE_ENDPOINT)
    )
    return compile_workflow(
        define_workflow(),
        DeploymentSpec(
            {
                "classifier": RemoteBinding(CLASSIFIER_ENDPOINT),
                "generator": GenerateEndpointBinding(
                    GENERATOR_ENDPOINT,
                    tensor_carrier=None,
                ),
                "response": response_binding,
            }
        ),
    )
