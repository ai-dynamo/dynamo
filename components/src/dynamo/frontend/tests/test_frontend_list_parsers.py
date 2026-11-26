# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`python -m dynamo.frontend --list-parsers` against the installed parser registry."""

import json
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def _frontend(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "dynamo.frontend", *args],
        capture_output=True,
        text=True,
        timeout=25,
    )


@pytest.mark.timeout(30)
def test_list_parsers_prints_the_registry_and_exits():
    result = _frontend("--list-parsers")

    assert result.returncode == 0, result.stderr
    registry = json.loads(result.stdout)
    parsers = {entry["family"]: entry for entry in registry["parsers"]}
    assert parsers["mistral_4"]["legacy_configuration_selectors"] == ["mistral"]
    assert parsers["mistral_4"]["required_tokens"][0]["token"] == "[ARGS]"
    assert parsers["gpt_oss"]["aliases"] == ["harmony"]
    assert parsers["gpt_oss"]["token_id_input"] is True
    assert {entry["source"] for entry in registry["parsers"]} == {"builtin"}


@pytest.mark.timeout(30)
def test_list_parsers_rejects_a_mistyped_flag():
    result = _frontend("--list-parsers", "--list-parsr")

    assert result.returncode != 0
    assert "unrecognized arguments: --list-parsr" in result.stderr
