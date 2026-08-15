# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Remote placement for two workflow workers and one stock vLLM worker."""

from dynamo.workflow import (
    DeploymentSpec,
    ExecutionPlan,
    GenerateEndpointBinding,
    InlineBinding,
    RemoteBinding,
    compile_workflow,
)
from examples.custom_backend.user_ensemble.workflow import define_workflow

ENCODER_ENDPOINT = "user-ensemble.encoder.generate"
CLASSIFIER_ENDPOINT = "user-ensemble.classifier.generate"
GENERATOR_ENDPOINT = "user-ensemble.generator.generate"


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
                "generator": GenerateEndpointBinding(GENERATOR_ENDPOINT),
                "response": InlineBinding("response"),
            }
        ),
    )
