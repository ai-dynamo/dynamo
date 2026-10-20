# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contract checks at the optional DeepInfer dependency boundary."""

import asyncio
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from dynamo.deepinfer.args import parse_args
from dynamo.deepinfer.llm_engine import DeepInferLLMEngine, create_engine
from dynamo.deepinfer.output import RequestOutput

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.fixture
def checkpoint(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text("{}")
    return str(tmp_path)


@pytest.fixture
def native_api(monkeypatch):
    """Simulate native pointer reuse, batched output and deferred completion."""
    arena = {}
    calls = []

    class OutputAPI:
        pass

    for field in RequestOutput.__dataclass_fields__:

        def read(ptr, field=field):
            calls.append((f"get_{field}", threading.get_ident()))
            return arena[int(ptr)][field]

        setattr(OutputAPI, f"get_{field}", staticmethod(read))

    class Engine:
        def __init__(self, args):
            self.args = args
            self.pending = {}
            self.multi_modal_processor = object()
            self.needs_step = False
            self.closed = False

        @classmethod
        def from_engine_args(cls, args):
            calls.append(("start", threading.get_ident()))
            return cls(args)

        def add_request(self, **kwargs):
            calls.append(("add_request", threading.get_ident()))
            assert kwargs["prompt_token_ids"].dtype == np.int32
            assert kwargs["multi_modal_data"] == {"image_bytes": [], "image_sizes": []}
            self.pending[kwargs["request_id"]] = len(kwargs["prompt_token_ids"])
            self.needs_step = True

        def has_unfinished_requests(self):
            calls.append(("has_unfinished_requests", threading.get_ident()))
            return self.needs_step

        def outputs(self, finished):
            arena.clear()
            for rid, prompt_len in self.pending.items():
                for index in range(1 if finished else 2):
                    ptr = len(arena) + 100
                    arena[ptr] = {
                        "request_id": rid,
                        "prompt_len": prompt_len,
                        "token_id": -1 if finished else index + 7,
                        "num_output_tokens": 2 if finished else index + 1,
                        "finished": finished,
                        "finish_reason": (
                            (
                                "insufficient_system_resource"
                                if rid == "failed"
                                else "stop"
                            )
                            if finished
                            else ""
                        ),
                    }
            if finished:
                self.pending.clear()
            return np.asarray(list(arena), dtype=np.uint64)

        def step(self):
            calls.append(("step", threading.get_ident()))
            self.needs_step = False
            return self.outputs(False)

        def flush_inflight_reqs(self):
            calls.append(("flush", threading.get_ident()))
            return self.outputs(True)

        def abort_request(self, rid):
            calls.append(("abort", threading.get_ident()))
            self.pending.pop(rid, None)

        def close(self):
            calls.append(("close", threading.get_ident()))
            self.closed = True

    dependencies = {
        "torch": {},
        "deepinfer._C.engine": {"RequestOutput": OutputAPI},
        "deepinfer.engine.arg_utils": {"EngineArgs": SimpleNamespace},
        "deepinfer.engine.llm_engine": {"LLMEngine": Engine},
    }
    for name, members in dependencies.items():
        module = ModuleType(name)
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, module)
    # Import the real adapter against the dependency boundary above; restore its
    # module cache along with the external dependency modules after each test.
    monkeypatch.delitem(sys.modules, "dynamo.deepinfer.native_engine", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    yield SimpleNamespace(arena=arena, calls=calls)
    sys.modules.pop("dynamo.deepinfer.native_engine", None)


def test_native_is_default_and_copies_pooled_outputs(checkpoint, native_api):
    config = parse_args(
        [
            "--model",
            checkpoint,
            "--engine-args",
            '{"context_parallel_size":2,"expert_parallel_size":2}',
        ]
    )
    assert config.mock is False and config.served_model_name == "deepinfer"
    engine = create_engine(config)
    assert engine.engine.args.data_parallel_size == 1
    assert engine.engine.args.context_parallel_size == 2
    assert engine.engine.args.model == checkpoint
    engine.add_request(
        user_id="user",
        request_id="one",
        sampling_params={"max_tokens": 3},
        prompt_token_ids=np.array([1, 2], dtype=np.int32),
    )
    first = engine.step()
    terminal = engine.step()
    assert [out.token_id for out in first] == [7, 8]
    assert terminal == [RequestOutput("one", 2, -1, 2, True, "stop")]
    assert native_api.arena[100]["token_id"] == -1
    engine.abort_request("one")
    engine.close()
    assert engine.engine.closed


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_native_worker_flush_failure_isolation_and_owner_cleanup(
    checkpoint, native_api
):
    engine, _ = await DeepInferLLMEngine.from_args(["--model", checkpoint])
    await engine.start(1)
    native = engine.engine.engine
    try:

        async def collect(rid):
            return [
                chunk
                async for chunk in engine.generate(
                    {"token_ids": [1, 2]}, SimpleNamespace(id=lambda: rid)
                )
            ]

        good, failed = await asyncio.gather(
            collect("good"), collect("failed"), return_exceptions=True
        )
        assert [chunk["token_ids"] for chunk in good] == [[7], [8], []]
        assert good[-1]["finish_reason"] == "stop"
        assert good[-1]["completion_usage"] == {
            "prompt_tokens": 2,
            "completion_tokens": 2,
            "total_tokens": 4,
        }
        assert isinstance(
            failed, RuntimeError
        ) and "insufficient_system_resource" in str(failed)
        assert engine.failure is None and not engine.streams
    finally:
        await engine.cleanup()
    assert native.closed
    assert {name for name, _ in native_api.calls} >= {
        "start",
        "add_request",
        "step",
        "flush",
        "abort",
        "close",
    }
    assert len({tid for _, tid in native_api.calls}) == 1
    assert all(tid != threading.get_ident() for _, tid in native_api.calls)


@pytest.mark.parametrize(
    "options",
    [
        [],
        ["--engine-args", "[]"],
        ["--engine-args", '{"data_parallel_size":2}'],
        ["--engine-args", '{"engine_rank":1}'],
        ["--engine-args", '{"engram_parallel_size":8}'],
        ["--engine-args", '{"model":"other"}'],
        ["--mock", "--engine-args", '{"max_num_seqs":4}'],
    ],
)
def test_invalid_native_configuration_rejected(checkpoint, options):
    if not options:
        # A tokenizer-only directory is valid only with --mock.
        Path(checkpoint, "config.json").unlink()
    with pytest.raises(SystemExit):
        parse_args(["--model", checkpoint, *options])
