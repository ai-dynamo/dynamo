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
from tests.utils import find_free_port
from tests.e2e.dynamo_client import DynamoRunProcess

# pytest fixture for DynamoRunProcess
@pytest.fixture()
def dynamo_run(backend, model, input_type, timeout):
    """
    Create and start a DynamoRunProcess for testing.
    """
    port = find_free_port()
    with DynamoRunProcess(
        model=model, backend=backend, port=port, input_type=input_type, timeout=timeout
    ) as process:
        yield process 
