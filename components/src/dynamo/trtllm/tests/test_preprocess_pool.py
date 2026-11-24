# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the multimodal preprocessing process pool.

Pool processes here run standard-library callables in place of TensorRT-LLM
input processors, so no TensorRT-LLM install, model, or GPU is needed.
"""

import functools
import logging
import multiprocessing.spawn
import operator
import pickle
import threading
from concurrent.futures.process import BrokenProcessPool
from types import SimpleNamespace
from unittest import mock

import pytest

from dynamo.trtllm.preprocess_pool import (
    InputProcessorSpec,
    PreprocessPool,
    record_input_processor_calls,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.multimodal,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

# Pool processes unpickle the builder by reference, so it uses only builtins.
_BUILD = functools.partial(operator.itemgetter(0), [operator.add])


@pytest.fixture
def local():
    return mock.Mock(return_value="in-process")


@pytest.fixture
def pool(local):
    pool = PreprocessPool(2, _BUILD, local)
    yield pool
    pool.close()


def _module():
    return SimpleNamespace(
        create_input_processor_with_hash=mock.Mock(return_value="in-process")
    )


def _llm(encoder_cache_max_bytes=0):
    return SimpleNamespace(
        input_processor=object(),
        args=SimpleNamespace(
            multimodal_config=SimpleNamespace(
                encoder_cache_max_bytes=encoder_cache_max_bytes
            )
        ),
    )


def _calls(llm):
    """The create_input_processor call that built ``llm``'s input processor."""
    return [(("model-dir", {"vocab": 3}), {}, llm.input_processor)]


@pytest.mark.timeout(60)
def test_every_process_is_ready_before_requests_are_processed(pool, local):
    assert len(pool.wait_until_ready()) == 2

    assert pool([1], [2]) == [1, 2]
    local.assert_not_called()


@pytest.mark.timeout(60)
def test_processing_error_is_raised_without_local_retry(pool, local):
    with pytest.raises(TypeError, match="unsupported operand"):
        pool(1, "a")
    local.assert_not_called()


@pytest.mark.timeout(60)
def test_unpicklable_request_is_processed_locally(pool, local):
    request = threading.Lock()

    assert pool(request, None) == "in-process"
    local.assert_called_once_with(request, None)


@pytest.mark.timeout(60)
def test_pool_processes_do_not_run_the_main_module(monkeypatch, tmp_path, local):
    get_preparation_data = multiprocessing.spawn.get_preparation_data

    def with_unimportable_main(name):
        data = get_preparation_data(name)
        data["init_main_from_name"] = "dynamo_missing_worker_main"
        data["init_main_from_path"] = str(tmp_path / "missing_worker_main.py")
        return data

    monkeypatch.setattr(
        multiprocessing.spawn, "get_preparation_data", with_unimportable_main
    )
    pool = PreprocessPool(1, _BUILD, local)
    try:
        pool.wait_until_ready()
        assert pool([1], [2]) == [1, 2]
    finally:
        pool.close()
    assert multiprocessing.spawn.get_preparation_data is with_unimportable_main


@pytest.mark.timeout(60)
def test_startup_failure_is_raised(local):
    pool = PreprocessPool(1, functools.partial(int, "not a number"), local)
    try:
        with pytest.raises(BrokenProcessPool):
            pool.wait_until_ready()
    finally:
        pool.close()


def test_routes_only_the_llm_input_processor(pool):
    module = _module()
    original = module.create_input_processor_with_hash
    processor, other = object(), object()
    pool.route(module, processor, encoder_cache_enabled=True)
    create = module.create_input_processor_with_hash

    assert create(processor, encoder_cache_enabled=True) is pool
    assert create(other, encoder_cache_enabled=True) == "in-process"
    assert create(processor) == "in-process"
    assert create(processor, "hasher", encoder_cache_enabled=True) == "in-process"
    assert create(processor, hash_lib="hasher") == "in-process"
    assert original.call_args_list == [
        mock.call(other, encoder_cache_enabled=True),
        mock.call(processor),
        mock.call(processor, "hasher", encoder_cache_enabled=True),
        mock.call(processor, hash_lib="hasher"),
    ]


def test_warns_once_when_the_llm_input_processor_is_hashed_differently(pool, caplog):
    module = _module()
    processor = object()
    pool.route(module, processor, encoder_cache_enabled=True)
    create = module.create_input_processor_with_hash

    with caplog.at_level(logging.DEBUG, logger="dynamo.trtllm.preprocess_pool"):
        create(processor)
        create(processor, hash_lib="hasher")
        create(object(), encoder_cache_enabled=True)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "processing multimodal requests in the worker process" in (
        warnings[0].getMessage()
    )


def test_start_routes_once_the_processes_are_ready():
    module, llm = _module(), _llm()
    original = module.create_input_processor_with_hash
    with mock.patch.object(PreprocessPool, "wait_until_ready") as wait_until_ready:
        pool = PreprocessPool.start(module, llm, _calls(llm), 2)
    try:
        wait_until_ready.assert_called_once_with()
        original.assert_called_once_with(
            llm.input_processor, encoder_cache_enabled=False
        )
        routed = module.create_input_processor_with_hash(
            llm.input_processor, encoder_cache_enabled=False
        )
        assert routed is pool
    finally:
        pool.close()
    assert module.create_input_processor_with_hash is original


def test_start_does_not_route_when_the_processes_fail():
    module, llm = _module(), _llm()
    original = module.create_input_processor_with_hash
    with mock.patch.object(
        PreprocessPool, "wait_until_ready", side_effect=TimeoutError("slow")
    ):
        with pytest.raises(TimeoutError):
            PreprocessPool.start(module, llm, _calls(llm), 2)

    assert module.create_input_processor_with_hash is original


def test_records_input_processor_calls_until_the_block_ends():
    module = SimpleNamespace(create_input_processor=mock.Mock(return_value="built"))
    original = module.create_input_processor
    with record_input_processor_calls(module) as calls:
        assert module.create_input_processor("model-dir", tokenizer=None) == "built"
    module.create_input_processor("other-dir")

    assert calls == [(("model-dir",), {"tokenizer": None}, "built")]
    assert module.create_input_processor is original


@pytest.mark.parametrize("encoder_cache_max_bytes", [0, 1024])
def test_spec_takes_the_call_that_built_the_llm_input_processor(
    encoder_cache_max_bytes,
):
    llm = _llm(encoder_cache_max_bytes)
    args = ("model-dir", {"vocab": 3}, "HF")
    # Arguments this module does not know about are passed on as well.
    kwargs = {"trust_remote_code": False, "vision_gpu_normalize": True}
    calls = [(("draft-dir",), {}, object()), (args, kwargs, llm.input_processor)]
    spec = InputProcessorSpec.from_llm(llm, calls)

    assert pickle.loads(spec.arguments) == (args, kwargs)
    assert spec.encoder_cache_enabled is (encoder_cache_max_bytes > 0)


def test_spec_needs_the_call_that_built_the_llm_input_processor():
    llm = _llm()
    with pytest.raises(RuntimeError, match="did not build its input processor"):
        InputProcessorSpec.from_llm(llm, [(("model-dir",), {}, object())])
