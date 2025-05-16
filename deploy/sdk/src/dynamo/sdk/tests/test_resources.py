# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from dynamo.sdk.cli.utils import configure_target_environment
from dynamo.sdk.core.runner import TargetEnum

pytestmark = pytest.mark.pre_merge


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    configure_target_environment(TargetEnum.BENTO)
    yield
    configure_target_environment(TargetEnum.DYNAMO)


def test_gpu_resources(setup_and_teardown):
    """Test resource configurations"""
    from _bentoml_sdk import Service as BentoService

    from dynamo.sdk import service

    @service(
        resources={"cpu": "2", "gpu": "1", "memory": "4Gi"},
        dynamo={"namespace": "test"},
    )
    class MyService:
        def __init__(self) -> None:
            pass

    svc: BentoService = MyService.get_bentoml_service()  # type: ignore
    assert svc.config["resources"]["cpu"] == "2"
    assert svc.config["resources"]["gpu"] == "1"
    assert svc.config["resources"]["memory"] == "4Gi"
