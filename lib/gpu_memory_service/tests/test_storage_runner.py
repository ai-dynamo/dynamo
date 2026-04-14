# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from gpu_memory_service.cli.storage_runner import _build_parser


def test_cli_parser_builds_save_and_load_commands() -> None:
    parser = _build_parser()

    save_args = parser.parse_args(["save", "--output-dir", "/tmp/out"])
    load_args = parser.parse_args(["load", "--input-dir", "/tmp/in"])

    assert save_args.subcommand == "save"
    assert save_args.device == 0
    assert load_args.subcommand == "load"
    assert load_args.workers == 4
