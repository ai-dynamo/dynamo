# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dynamo._core import (
    get_tool_parser_names,
    supports_structural_tag_parser,
    validate_structural_tag_parser,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.timeout(30),
]


def _probe(case, parser_name, reasoning_parser=None):
    if case == "available":
        assert parser_name in get_tool_parser_names()
        assert supports_structural_tag_parser(parser_name, reasoning_parser)
        assert validate_structural_tag_parser(parser_name, reasoning_parser) is None
    elif case == "missing":
        assert parser_name in get_tool_parser_names()
        assert supports_structural_tag_parser(parser_name) is False
        with pytest.raises(ValueError, match="No structural-tag grammar"):
            validate_structural_tag_parser(parser_name)
    elif case == "unknown":
        assert parser_name not in get_tool_parser_names()
        assert supports_structural_tag_parser(parser_name) is False
        with pytest.raises(ValueError, match="No structural-tag grammar"):
            validate_structural_tag_parser(parser_name)
    elif case == "routing_error":
        with pytest.raises(ValueError, match="DYN_PARSER_VERSION"):
            supports_structural_tag_parser(parser_name, reasoning_parser)
        with pytest.raises(ValueError, match="DYN_PARSER_VERSION"):
            validate_structural_tag_parser(parser_name, reasoning_parser)
    else:
        raise ValueError(f"Unknown capability probe: {case}")


def _run_probe(case, parser_name, reasoning_parser=None, *, version=None):
    # Parser generation is cached at process startup, so each configuration needs
    # its own process rather than environment mutation after the first lookup.
    allowed = {
        "PATH",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "HOME",
        "USER",
        "LANG",
        "LC_ALL",
    }
    environment = {key: value for key, value in os.environ.items() if key in allowed}
    if version is not None:
        environment["DYN_PARSER_VERSION"] = version
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            json.dumps(
                {
                    "case": case,
                    "parser_name": parser_name,
                    "reasoning_parser": reasoning_parser,
                }
            ),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "parser_name", ["deepseek_v4", "deepseek-v4", "deepseekv4", "deepseek_v41"]
)
def test_deepseek_grammar_is_available(parser_name):
    _run_probe("available", parser_name)


def test_selectable_parser_without_grammar_is_rejected_by_preflight():
    _run_probe("missing", "gemma4")


def test_unknown_family_is_not_grammar_capable():
    _run_probe("unknown", "no_such_parser")


@pytest.mark.parametrize("version", ["1", "2", "invalid"])
def test_capability_query_respects_parser_generation(version):
    _run_probe(
        "available" if version == "2" else "routing_error",
        "deepseek_v41",
        "deepseek_v41",
        version=version,
    )


def test_incompatible_v2_parser_pair_is_rejected():
    _run_probe("routing_error", "deepseek_v41", "qwen3", version="2")


if __name__ == "__main__":
    _probe(**json.loads(sys.argv[1]))
