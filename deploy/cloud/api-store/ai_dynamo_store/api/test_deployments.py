#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from .deployments import get_deployment_status, get_urls


def test_get_deployment_status():
    # Test case 1: Ready condition present with message
    resource = {
        "status": {"conditions": [{"type": "Ready", "message": "Deployment is ready"}]}
    }
    assert get_deployment_status(resource) == "Deployment is ready"

    # Test case 2: Ready condition not present
    resource = {
        "status": {
            "conditions": [{"type": "Available", "message": "Some other condition"}]
        }
    }
    assert get_deployment_status(resource) == "unknown"

    # Test case 3: Empty conditions list
    resource = {"status": {"conditions": []}}
    assert get_deployment_status(resource) == "unknown"

    # Test case 4: No status field
    resource = {}
    assert get_deployment_status(resource) == "unknown"

    # Test case 5: No conditions field in status
    resource = {"status": {}}
    assert get_deployment_status(resource) == "unknown"

    # Test case 6: Ready condition present without message
    resource = {"status": {"conditions": [{"type": "Ready"}]}}
    assert get_deployment_status(resource) == "unknown"


def test_get_urls():
    resource = {
        "status": {
            "conditions": [
                {"type": "EndpointExposed", "message": "https://example.com"}
            ]
        }
    }
    assert get_urls(resource) == ["https://example.com"]
