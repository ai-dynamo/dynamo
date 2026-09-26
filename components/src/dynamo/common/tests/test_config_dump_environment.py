# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for sensitive-value redaction in the config dump."""

import pytest

from dynamo.common.config_dump.environment import get_environment_vars

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.parallel,
]


@pytest.mark.parametrize(
    "name",
    ["DYN_KVBM_OBJECT_ACCESS_KEY"],
)
def test_credentials_are_redacted(monkeypatch, name: str) -> None:
    """An object-storage access key is a credential and must be redacted."""
    monkeypatch.setenv(name, "s3cr3t-value")

    assert get_environment_vars()[name] == "<REDACTED>"


@pytest.mark.parametrize(
    "name",
    ["DYN_SGL_DISAGG_CONFIG_KEY"],
)
def test_non_credential_key_names_stay_readable(monkeypatch, name: str) -> None:
    """Matching on KEY alone would blank out config keys and key file paths,
    which is the diagnostic value a dump exists for."""
    monkeypatch.setenv(name, "plain-value")

    assert get_environment_vars()[name] == "plain-value"


def test_include_sensitive_still_returns_the_value(monkeypatch) -> None:
    monkeypatch.setenv("DYN_KVBM_OBJECT_ACCESS_KEY", "s3cr3t-value")

    values = get_environment_vars(include_sensitive=True)

    assert values["DYN_KVBM_OBJECT_ACCESS_KEY"] == "s3cr3t-value"
