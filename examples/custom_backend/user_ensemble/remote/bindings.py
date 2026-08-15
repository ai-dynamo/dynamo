# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Remote model placement with an inline application response stage."""

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


def compile_remote_workflow() -> ExecutionPlan:
    return compile_workflow(
        define_workflow(),
        DeploymentSpec(
            {
                "classifier": RemoteBinding(CLASSIFIER_ENDPOINT),
                "generator": GenerateEndpointBinding(
                    GENERATOR_ENDPOINT,
                    tensor_carrier=None,
                ),
                "response": InlineBinding("response"),
            }
        ),
    )
