# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Checks for the per-DGD values renderer; no cluster or model is loaded."""

import copy
import importlib.util
from pathlib import Path

import pytest

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

PROFILE_DIR = Path(__file__).parents[2] / "examples" / "deployments" / "batch-gateway"
SPEC = importlib.util.spec_from_file_location(
    "batch_profile_render_values", PROFILE_DIR / "render_values.py"
)
RENDERER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RENDERER)


@pytest.fixture
def profile():
    return {
        "targetDGD": {"name": "parse", "namespace": "offline"},
        "model": "nvidia/parse",
        "asyncConcurrency": 512,
        "storage": {
            "secretName": "parse-storage",
            "s3": {
                "region": "us-east-1",
                "bucket": "parse-files",
                "accessKeyId": "example-access-id",
            },
        },
    }


def test_values_share_the_dgd_contract(profile):
    defaults = RENDERER.load_values("chart/values.yaml")
    assert (
        defaults["batch-gateway"]["processor"]["config"]["asyncDispatch"]["models"]
        == {}
    )
    values = RENDERER.render_values(profile)
    assert list(values) == ["dynamo-batch-values.yaml"]
    parent = values["dynamo-batch-values.yaml"]
    sync = parent["batch-gateway"]
    overlay = sync["processor"]["config"]
    ap = parent["async"]["ap"]
    mapping = overlay["asyncDispatch"]["models"]["nvidia/parse"]
    assert set(overlay["asyncDispatch"]["models"]) == {"nvidia/parse"}
    queue = ap["transportConfig"]["queues"][0]

    assert mapping["requestQueueName"] == queue["queue_name"]
    assert mapping["resultQueueName"] == ap["transportConfig"]["result_queue_name"]
    assert "result_queue_name" not in queue
    assert queue["igw_base_url"] == "http://parse-frontend.offline.svc:8000"
    assert queue["gate_params"]["url"] == f'{queue["igw_base_url"]}/metrics'
    assert queue["gate_params"]["labels"] == {"model": "nvidia/parse"}
    assert queue["gate_params"]["value_type"] == "budget"
    assert queue["gate_params"]["fallback"] == "0"
    assert overlay["concurrency"]["global"] >= ap["concurrency"] == 512
    assert overlay["concurrency"]["perEndpoint"] >= 512
    assert sync["global"]["secretName"] == ap["transportConfig"]["urlSecret"]["name"]
    assert sync["global"]["dbClient"]["type"] == "postgresql"
    assert sync["global"]["fileClient"]["type"] == "s3"
    assert sync["apiserver"]["image"]["tag"] == "v0.6.0"
    assert ap["image"]["tag"] == "v0.10.0"
    assert sync["apiserver"]["fullnameOverride"] == "parse-batch-api"
    assert parent["async"]["fullnameOverride"] == "parse-batch-async"
    assert overlay["dispatchMode"] == "async"
    assert overlay["modelGateways"] is None


def test_dgds_do_not_share_queue_or_file_names(profile):
    other = copy.deepcopy(profile)
    other["targetDGD"]["namespace"] = "other"
    first = RENDERER.render_values(profile)["dynamo-batch-values.yaml"]
    second = RENDERER.render_values(other)["dynamo-batch-values.yaml"]
    for key in ("requestQueueName", "resultQueueName"):
        first_mapping = first["batch-gateway"]["processor"]["config"]["asyncDispatch"][
            "models"
        ]["nvidia/parse"]
        second_mapping = second["batch-gateway"]["processor"]["config"][
            "asyncDispatch"
        ]["models"]["nvidia/parse"]
        assert first_mapping[key] != second_mapping[key]
    first_files = first["batch-gateway"]["global"]["fileClient"]["s3"]
    second_files = second["batch-gateway"]["global"]["fileClient"]["s3"]
    assert first_files["prefix"] != second_files["prefix"]


def test_storage_is_referenced_not_provisioned(profile):
    parent = RENDERER.render_values(profile)["dynamo-batch-values.yaml"]
    assert "developmentStorage" not in parent
    global_config = parent["batch-gateway"]["global"]
    assert global_config["secretName"] == "parse-storage"
    assert global_config["dbClient"]["type"] == "postgresql"
    assert global_config["fileClient"]["type"] == "s3"
    assert "fs" not in global_config["fileClient"]
    assert global_config["fileClient"]["s3"]["autoCreateBucket"] is False
    assert (
        parent["async"]["ap"]["transportConfig"]["urlSecret"]["name"] == "parse-storage"
    )


def test_shipped_profile_requires_user_supplied_storage():
    profile = RENDERER.load_values("profile-external-storage.yaml")
    assert profile["storage"]["mode"] == "external"
    with pytest.raises(ValueError, match="configured string"):
        RENDERER.render_values(profile)


@pytest.mark.parametrize("length", [51, 52])
def test_dgd_name_fits_generated_services(profile, length):
    profile["targetDGD"]["name"] = "x" * length
    if length == 52:
        with pytest.raises(ValueError, match="integration resource names"):
            RENDERER.render_values(profile)
    else:
        parent = RENDERER.render_values(profile)["dynamo-batch-values.yaml"]
        assert len(parent["async"]["fullnameOverride"]) == 63


@pytest.mark.parametrize(
    "error",
    [
        "placeholder",
        "misspelled",
        "concurrency",
        "service-name",
        "endpoint",
        "storage-mode",
        "storage-mix",
        "access-mode",
        "secret-name",
    ],
)
def test_invalid_profile_fails_before_rendering(profile, error):
    if error == "placeholder":
        profile["storage"]["s3"]["bucket"] = "REPLACE_ME"
    elif error == "misspelled":
        profile["asyncConcurency"] = 512
    elif error == "concurrency":
        profile["asyncConcurrency"] = 0
    elif error == "service-name":
        profile["targetDGD"]["name"] = "x" * 63
    elif error == "endpoint":
        profile["storage"]["s3"]["endpoint"] = None
    elif error == "storage-mode":
        profile["storage"]["mode"] = "redis"
    elif error == "storage-mix":
        profile["storage"]["mode"] = "development"
    elif error == "access-mode":
        profile["storage"] = {"mode": "development", "filesAccessMode": "BadMode"}
    else:
        profile["storage"]["secretName"] = "Bad_Secret"
    with pytest.raises(ValueError):
        RENDERER.render_values(profile)
