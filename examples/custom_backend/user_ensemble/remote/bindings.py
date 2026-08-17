# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Remote placement for two workflow workers and one stock vLLM worker."""

import os

from dynamo.workflow import (
    DeploymentSpec,
    ExecutionPlan,
    GenerateEndpointBinding,
    InlineBinding,
    RemoteBinding,
    compile_workflow,
)
from examples.custom_backend.user_ensemble.workflow import define_workflow

WORKFLOW_NAMESPACE = os.environ.get("DYN_USER_ENSEMBLE_NAMESPACE", "user-ensemble")
ENCODER_ENDPOINT = f"{WORKFLOW_NAMESPACE}.encoder.generate"
CLASSIFIER_ENDPOINT = f"{WORKFLOW_NAMESPACE}.classifier.generate"
GENERATOR_ENDPOINT = f"{WORKFLOW_NAMESPACE}.generator.generate"


def compile_remote_workflow() -> ExecutionPlan:
    return compile_workflow(
        define_workflow(),
        DeploymentSpec(
            {
                "encoder": RemoteBinding(
                    ENCODER_ENDPOINT,
                    tensor_carrier="nixl",
                ),
                "classifier": RemoteBinding(
                    CLASSIFIER_ENDPOINT,
                    tensor_carrier="nixl",
                ),
                "generator": GenerateEndpointBinding(
                    GENERATOR_ENDPOINT,
                    tensor_carrier="nixl",
                ),
                "response": InlineBinding("response"),
            }
        ),
    )
