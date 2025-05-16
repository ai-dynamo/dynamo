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


from ..api.utils import build_latest_revision_from_cr, get_deployment_status, get_urls


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


def test_build_latest_revision_from_cr_minimal():
    cr = {
        "metadata": {
            "uid": "u1",
            "name": "n1",
            "creationTimestamp": "2024-01-01T00:00:00Z",
        },
        "spec": {
            "dynamoGraph": "repo:ver",
            "services": {"svc": {}},
            "envs": [{"name": "A", "value": "B"}],
        },
    }
    rev = build_latest_revision_from_cr(cr)
    assert rev["uid"] == "u1"
    assert rev["name"] == "n1"
    assert rev["targets"][0]["bento"]["repository"]["name"] == "repo"
    assert rev["targets"][0]["bento"]["name"] == "ver"
    assert rev["targets"][0]["config"]["services"] == {"svc": {}}
    assert rev["targets"][0]["config"]["envs"] == [{"name": "A", "value": "B"}]


def test_build_latest_revision_from_cr_missing_fields():
    cr = {"spec": {}}
    rev = build_latest_revision_from_cr(cr)
    assert rev["uid"] == "dummy-uid"
    assert rev["name"] == "dummy-revision"
    assert rev["targets"][0]["bento"]["repository"]["name"] == "unknown"
    assert rev["targets"][0]["bento"]["name"] == "unknown"
    assert rev["targets"][0]["config"]["services"] == {}
    assert rev["targets"][0]["config"]["envs"] == []


def test_build_latest_revision_from_cr_bento_colonless():
    cr = {"spec": {"dynamoGraph": "justrepo"}}
    rev = build_latest_revision_from_cr(cr)
    assert rev["targets"][0]["bento"]["repository"]["name"] == "unknown"
    assert rev["targets"][0]["bento"]["name"] == "unknown"
