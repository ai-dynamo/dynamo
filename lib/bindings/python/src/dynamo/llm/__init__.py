# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa

import logging

from dynamo._core import AicPerfConfig as AicPerfConfig
from dynamo._core import EngineType
from dynamo._core import EntrypointArgs as EntrypointArgs
from dynamo._core import FpmEventRelay as FpmEventRelay
from dynamo._core import FpmEventSubscriber as FpmEventSubscriber
from dynamo._core import HttpAsyncEngine as HttpAsyncEngine
from dynamo._core import HttpService as HttpService
from dynamo._core import KserveGrpcService as KserveGrpcService
from dynamo._core import KvEventPublisher as KvEventPublisher
from dynamo._core import KvRouter as KvRouter
from dynamo._core import KvRouterConfig as KvRouterConfig
from dynamo._core import LoRADownloader as LoRADownloader
from dynamo._core import MediaDecoder as MediaDecoder
from dynamo._core import MediaFetcher as MediaFetcher
from dynamo._core import MockEngineArgs as MockEngineArgs
from dynamo._core import ModelCardInstanceId as ModelCardInstanceId
from dynamo._core import ModelInput as ModelInput
from dynamo._core import ModelRuntimeConfig as ModelRuntimeConfig
from dynamo._core import ModelType as ModelType
from dynamo._core import OverlapScores as OverlapScores
from dynamo._core import PlannerReplayBridge as PlannerReplayBridge
from dynamo._core import PythonAsyncEngine as PythonAsyncEngine
from dynamo._core import RadixTree as RadixTree
from dynamo._core import ReasoningConfig as ReasoningConfig
from dynamo._core import RouterConfig as RouterConfig
from dynamo._core import RouterMode as RouterMode
from dynamo._core import SglangArgs as SglangArgs
from dynamo._core import WorkerMetricsPublisher as WorkerMetricsPublisher
from dynamo._core import compute_block_hash_for_seq as compute_block_hash_for_seq
from dynamo._core import fetch_model as fetch_model
from dynamo._core import lora_name_to_id as lora_name_to_id
from dynamo._core import make_engine
from dynamo._core import register_model as register_model
from dynamo._core import run_input
from dynamo._core import run_kv_indexer as run_kv_indexer
from dynamo._core import run_mocker_trace_replay as _run_mocker_trace_replay
from dynamo._core import unregister_model as unregister_model

from .exceptions import HttpError

# Backward-compatible aliases
fetch_llm = fetch_model
register_llm = register_model
unregister_llm = unregister_model


def run_mocker_trace_replay(
    trace_file,
    extra_engine_args=None,
    router_config=None,
    num_workers=1,
    replay_concurrency=None,
    router_mode="round_robin",
    arrival_speedup_ratio=1.0,
    trace_block_size=512,
    trace_format="mooncake",
    trace_shared_prefix_ratio=0.0,
    trace_num_prefix_groups=0,
):
    return _run_mocker_trace_replay(
        trace_file,
        extra_engine_args=extra_engine_args,
        router_config=router_config,
        num_workers=num_workers,
        replay_concurrency=replay_concurrency,
        replay_mode="offline",
        router_mode=router_mode,
        arrival_speedup_ratio=arrival_speedup_ratio,
        trace_block_size=trace_block_size,
        trace_format=trace_format,
        trace_shared_prefix_ratio=trace_shared_prefix_ratio,
        trace_num_prefix_groups=trace_num_prefix_groups,
    )
