# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the multimodal preprocessing process pool.

Pool processes here run standard-library callables in place of TensorRT-LLM
input processors, so no TensorRT-LLM install, model, or GPU is needed.
"""

import functools
import multiprocessing.spawn
import operator
import pickle
import threading
from concurrent.futures.process import BrokenProcessPool
from types import SimpleNamespace
from unittest import mock

import pytest

from dynamo.trtllm.preprocess_pool import InputProcessorSpec, PreprocessPool

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


def _llm(tmp_path, video_pruning_rate=None, encoder_cache_max_bytes=0):
    return SimpleNamespace(
        _hf_model_dir=tmp_path,
        input_processor=object(),
        tokenizer={"vocab": 3},
        args=SimpleNamespace(
            enable_tokenization_cache=True,
            multimodal_config=SimpleNamespace(
                video_pruning_rate=video_pruning_rate,
                encoder_cache_max_bytes=encoder_cache_max_bytes,
            ),
            checkpoint_format="HF",
            trust_remote_code=False,
        ),
    )


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


def test_start_routes_once_the_processes_are_ready(tmp_path):
    module, llm = _module(), _llm(tmp_path)
    original = module.create_input_processor_with_hash
    with mock.patch.object(PreprocessPool, "wait_until_ready") as wait_until_ready:
        pool = PreprocessPool.start(module, llm, 2)
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


def test_start_does_not_route_when_the_processes_fail(tmp_path):
    module, llm = _module(), _llm(tmp_path)
    original = module.create_input_processor_with_hash
    with mock.patch.object(
        PreprocessPool, "wait_until_ready", side_effect=TimeoutError("slow")
    ):
        with pytest.raises(TimeoutError):
            PreprocessPool.start(module, llm, 2)

    assert module.create_input_processor_with_hash is original


@pytest.mark.parametrize(
    ("video_pruning_rate", "encoder_cache_max_bytes"), [(None, 0), (0.5, 1024)]
)
def test_spec_mirrors_how_the_llm_built_its_input_processor(
    tmp_path, video_pruning_rate, encoder_cache_max_bytes
):
    llm = _llm(tmp_path, video_pruning_rate, encoder_cache_max_bytes)
    spec = InputProcessorSpec.from_llm(llm)

    assert spec.model_dir == str(tmp_path)
    assert pickle.loads(spec.tokenizer) == {"vocab": 3}
    assert spec.checkpoint_format == "HF"
    assert spec.trust_remote_code is False
    expected_kwargs = {"enable_tokenization_cache": True}
    if video_pruning_rate is not None:
        expected_kwargs["video_pruning_rate"] = video_pruning_rate
    assert spec.kwargs == expected_kwargs
    assert spec.encoder_cache_enabled is (encoder_cache_max_bytes > 0)
