# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys

import pytest

from dynamo._core import (
    get_parser_capabilities,
    get_reasoning_parser_names,
    get_tool_parser_names,
)
from dynamo.frontend import parser_capabilities as frontend

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def test_capability_cli_reports_linked_registry_before_runtime_setup(
    monkeypatch, capsys
):
    monkeypatch.setattr(sys, "argv", ["dynamo.frontend", "--list-parser-capabilities"])
    frontend.main()
    catalog = json.loads(capsys.readouterr().out)
    assert catalog == get_parser_capabilities()
    families = {entry["family"]: entry for entry in catalog["unified"]}
    assert "mistral" in families["mistral_4"]["aliases"]
    assert families["mistral_4"]["required_tokens"][0]["token"] == "[ARGS]"
    assert families["mistral_4"]["guidance"]["mistral_lark"] is True
    assert "qwen3_coder" in families["nemotron_v3"]["compatible_tool_selectors"]
    assert families["beam"]["guidance"]["whole_turn"] is True
    assert families["qwen3"]["auto_selection"] == "requires_tool_selector"
    assert families["gpt_oss"]["token_id_input"] is True
    for descriptor in families.values():
        for selector in [descriptor["family"], *descriptor["aliases"]]:
            assert selector in get_tool_parser_names()
            assert selector in get_reasoning_parser_names()
    legacy = catalog["legacy"]
    assert (
        sorted(entry["selector"] for entry in legacy["tools"])
        == legacy["tool_selectors"]
    )
    assert (
        sorted(entry["selector"] for entry in legacy["reasoning"])
        == legacy["reasoning_selectors"]
    )


def test_capability_cli_rejects_unknown_arguments(monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", ["dynamo.frontend", "--list-parser-capabilities", "--misspelled"]
    )
    with pytest.raises(SystemExit) as result:
        frontend.main()
    assert result.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def subprocess_environment():
    return {
        name: os.environ[name]
        for name in ("PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "HOME")
        if name in os.environ
    }


def test_module_cli_does_not_import_serving_runtime():
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy,sys; sys.argv=['dynamo.frontend','--list-parser-capabilities']; runpy.run_module('dynamo.frontend',run_name='__main__'); assert 'dynamo.frontend.main' not in sys.modules; assert 'dynamo.runtime' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=True,
        env=subprocess_environment(),
    )
    assert json.loads(process.stdout) == get_parser_capabilities()


def test_independently_built_plugin_is_discoverable():
    library = os.environ.get("DYNAMO_REGISTRY_FIXTURE_PLUGIN")
    if not library:
        pytest.skip(
            "set DYNAMO_REGISTRY_FIXTURE_PLUGIN to the independently built fixture"
        )

    command = [sys.executable, "-m", "dynamo.frontend", "--list-parser-capabilities"]
    environment = subprocess_environment()
    linked = subprocess.run(
        command, capture_output=True, text=True, check=True, env=environment
    )
    assert all(
        row["family"] != "dis3562_lines" for row in json.loads(linked.stdout)["unified"]
    )
    loaded = subprocess.run(
        [*command, "--tool-parser-plugin", library],
        capture_output=True,
        text=True,
        check=True,
        env=environment,
    )
    catalog = json.loads(loaded.stdout)
    entry = next(row for row in catalog["unified"] if row["family"] == "dis3562_lines")
    assert entry["aliases"] == ["dis3562_alias"]
    assert entry["guidance"]["structural_tag"]
    assert entry["plugin"]["inspect_prompt"]
    assert entry["plugin"]["identity"].startswith("v1:")

    conflict = subprocess.run(
        [*command, "--tool-parser-plugin", library, "--tool-parser-plugin", library],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert conflict.returncode == 2
    assert "conflicting parser selector" in conflict.stderr
