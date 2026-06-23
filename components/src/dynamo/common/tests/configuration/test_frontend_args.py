# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import pytest

from dynamo.frontend.frontend_args import (
    _U32_MAX,
    FrontendArgGroup,
    FrontendConfig,
    _preprocess_for_encode_config,
    validate_model_name,
    validate_model_path,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def _frontend_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    return parser


def _frontend_config(cli_args: list[str]) -> FrontendConfig:
    args = _frontend_parser().parse_args(cli_args)
    return FrontendConfig.from_cli_args(args)


@pytest.mark.parametrize("value", ["", "   ", None])
def test_validate_model_name_rejects_empty_values(value) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="non-empty string"):
        validate_model_name(value)


def test_validate_model_name_trims_whitespace() -> None:
    assert validate_model_name("  llama3  ") == "llama3"


def test_validate_model_path_requires_existing_directory(tmp_path) -> None:
    assert validate_model_path(str(tmp_path)) == str(tmp_path)
    with pytest.raises(argparse.ArgumentTypeError, match="valid directory"):
        validate_model_path(str(tmp_path / "missing"))


def test_frontend_arg_group_version_flag_prints_frontend_version(capsys) -> None:
    parser = _frontend_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--version"])

    assert "Dynamo Frontend" in capsys.readouterr().out


def test_frontend_arg_group_registers_new_frontend_flags() -> None:
    help_text = _frontend_parser().format_help()
    for flag in (
        "--namespace-prefix",
        "--migration-max-seq-len",
        "--enable-streaming-tool-dispatch",
        "--enable-streaming-reasoning-dispatch",
        "--exclude-tools-when-tool-choice-none",
        "--dyn-chat-processor",
        "--dyn-debug-perf",
        "--dyn-preprocess-workers",
        "--tokenizer",
        "--trust-remote-code",
    ):
        assert flag in help_text


def test_preprocess_for_encode_config_returns_internal_dict() -> None:
    config = _frontend_config([])
    encoded = _preprocess_for_encode_config(config)

    assert encoded is config.__dict__
    assert encoded["http_port"] == 8000


def test_frontend_validate_load_aware_forces_kv_router_mode() -> None:
    config = _frontend_config(["--load-aware"])
    config.validate()

    assert config.router_mode == "kv"


def test_frontend_validate_requires_tls_key_and_cert_together() -> None:
    config = _frontend_config(["--tls-cert-path", "/tmp/cert.pem"])
    with pytest.raises(
        ValueError, match="--tls-cert-path and --tls-key-path must be provided together"
    ):
        config.validate()


@pytest.mark.parametrize("migration_limit", [-1, _U32_MAX + 1])
def test_frontend_validate_rejects_migration_limit_out_of_u32_bounds(
    migration_limit: int,
) -> None:
    config = _frontend_config(["--migration-limit", str(migration_limit)])
    with pytest.raises(ValueError, match="--migration-limit must be between 0 and"):
        config.validate()


@pytest.mark.parametrize("max_seq_len", [0, _U32_MAX + 1])
def test_frontend_validate_rejects_migration_max_seq_len_out_of_u32_bounds(
    max_seq_len: int,
) -> None:
    config = _frontend_config(["--migration-max-seq-len", str(max_seq_len)])
    with pytest.raises(
        ValueError, match="--migration-max-seq-len must be between 1 and"
    ):
        config.validate()


def test_frontend_validate_rejects_negative_min_initial_workers() -> None:
    config = _frontend_config(["--router-min-initial-workers", "-1"])
    with pytest.raises(ValueError, match="--router-min-initial-workers must be >= 0"):
        config.validate()


def test_frontend_validate_rejects_invalid_tokenizer_backend() -> None:
    config = _frontend_config([])
    config.tokenizer_backend = "invalid"
    with pytest.raises(ValueError, match="--tokenizer: invalid value"):
        config.validate()


def test_frontend_validate_router_prefill_load_model_aic_requires_kv_mode() -> None:
    config = _frontend_config(["--router-prefill-load-model", "aic"])
    with pytest.raises(
        ValueError, match="--router-prefill-load-model=aic requires --router-mode=kv"
    ):
        config.validate()


def test_frontend_validate_router_prefill_load_model_aic_requires_dynamo_processor() -> (
    None
):
    config = _frontend_config(
        [
            "--router-mode",
            "kv",
            "--router-prefill-load-model",
            "aic",
            "--dyn-chat-processor",
            "vllm",
        ]
    )
    with pytest.raises(ValueError, match="--dyn-chat-processor=dynamo"):
        config.validate()


def test_frontend_validate_router_prefill_load_model_aic_requires_aic_fields() -> None:
    config = _frontend_config(
        [
            "--router-mode",
            "kv",
            "--router-prefill-load-model",
            "aic",
        ]
    )
    with pytest.raises(ValueError, match="--aic-backend"):
        config.validate()


def test_frontend_validate_router_prefill_load_model_aic_requires_prefill_tracking() -> (
    None
):
    config = _frontend_config(
        [
            "--router-mode",
            "kv",
            "--router-prefill-load-model",
            "aic",
            "--aic-backend",
            "vllm",
            "--aic-system",
            "h200_sxm",
            "--aic-model-path",
            "repo/model",
        ]
    )
    with pytest.raises(
        ValueError,
        match="--router-prefill-load-model=aic requires --router-track-prefill-tokens",
    ):
        config.validate()


def test_frontend_validate_serve_indexer_requires_kv_mode() -> None:
    config = _frontend_config(["--serve-indexer"])
    with pytest.raises(ValueError, match="--serve-indexer requires --router-mode=kv"):
        config.validate()


def test_frontend_validate_serve_indexer_and_remote_indexer_are_mutually_exclusive() -> (
    None
):
    config = _frontend_config(
        [
            "--serve-indexer",
            "--router-mode",
            "kv",
            "--use-remote-indexer",
        ]
    )
    with pytest.raises(
        ValueError,
        match="--serve-indexer and --use-remote-indexer are mutually exclusive",
    ):
        config.validate()
