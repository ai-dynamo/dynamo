# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from unittest.mock import patch
import os 
from vllm.engine.arg_utils import AsyncEngineArgs
from dynamo.vllm.args import Config, overwrite_args, create_kv_events_config, set_side_channel_host_and_port, KVEventsConfig

def test_set_side_channel_host_and_port():
    with patch("dynamo.vllm.args.get_host_ip", return_value="127.0.0.1"):
        set_side_channel_host_and_port(6379)
        assert os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] == "127.0.0.1"
        assert os.environ["VLLM_NIXL_SIDE_CHANNEL_PORT"] == "6379"


def test_overwrite_args():
    config = Config()

    # Set engine args different than default
    config.engine_args = AsyncEngineArgs(
        task="classify",
        skip_tokenizer_init=True,
        enable_log_requests=True,
        disable_log_stats=True,
    )

    assert config.engine_args.task == "classify"
    assert config.engine_args.skip_tokenizer_init == True
    assert config.engine_args.enable_log_requests == True
    assert config.engine_args.disable_log_stats == True

    overwrite_args(config)
    assert config.engine_args.task == "generate"
    assert config.engine_args.skip_tokenizer_init == False
    assert config.engine_args.enable_log_requests == False
    assert config.engine_args.disable_log_stats == False

def test_create_kv_events_config_prefix_caching_disabled():
    config = Config()
    config.engine_args = AsyncEngineArgs(enable_prefix_caching=False)
    result = create_kv_events_config(config)
    assert result is None

def test_create_kv_events_config_user_provided(monkeypatch):
    config = Config()
    config.engine_args = AsyncEngineArgs(enable_prefix_caching=True)
    # Simulate user-provided config
    setattr(config.engine_args, "kv_events_config", object())
    result = create_kv_events_config(config)
    assert result is None

def test_create_kv_events_config_missing_kv_port():
    config = Config()
    config.engine_args = AsyncEngineArgs(enable_prefix_caching=True)
    # No user-provided config, kv_port is None
    config.kv_port = None
    # Remove kv_events_config if present
    if hasattr(config.engine_args, "kv_events_config"):
        delattr(config.engine_args, "kv_events_config")
    with pytest.raises(ValueError) as excinfo:
        create_kv_events_config(config)
    assert "config.kv_port is not set" in str(excinfo.value)

def test_create_kv_events_config_default(monkeypatch):
    config = Config()
    config.engine_args = AsyncEngineArgs(enable_prefix_caching=True)
    config.kv_port = 12345
    config.engine_args.data_parallel_rank = 2
    # Remove kv_events_config if present
    if hasattr(config.engine_args, "kv_events_config"):
        delattr(config.engine_args, "kv_events_config")
    result = create_kv_events_config(config)
    assert isinstance(result, KVEventsConfig)
    assert result.enable_kv_cache_events is True
    assert result.publisher == "zmq"
    # Should subtract dp_rank from kv_port
    assert result.endpoint == "tcp://*:12343"

def test_create_kv_events_config_default_dp_rank_none():
    config = Config()
    config.engine_args = AsyncEngineArgs(enable_prefix_caching=True)
    config.kv_port = 23456
    config.engine_args.data_parallel_rank = None
    # Remove kv_events_config if present
    if hasattr(config.engine_args, "kv_events_config"):
        delattr(config.engine_args, "kv_events_config")
    result = create_kv_events_config(config)
    assert isinstance(result, KVEventsConfig)
    assert result.endpoint == "tcp://*:23456"
