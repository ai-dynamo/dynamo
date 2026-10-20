# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-owner DeepInfer driver. JSON lines on stdin/stdout are private IPC.

Only this process touches the engine. Diagnostics go to stderr so DeepInfer
startup messages cannot corrupt the Rust server's IPC stream.
"""

import argparse
import contextlib
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from tokenizers import Tokenizer


class NativeDeepInferEngine:
    """Real API adapter; GPU execution is deliberately outside the CPU mock check."""

    def __init__(self, model, context_length, options):
        from deepinfer.engine.arg_utils import EngineArgs
        from deepinfer.engine.llm_engine import LLMEngine

        arguments = {
            "model": model,
            "max_model_len": context_length,
            "max_num_batched_tokens": context_length,
            "chunked_prefill": True,
            "data_parallel_size": 1,
            "offline_inference": False,
        } | options
        if arguments["data_parallel_size"] != 1:
            raise ValueError(
                "This example owns one DP rank; a multi-rank launcher is required for DP > 1"
            )
        self.inner = LLMEngine.from_engine_args(EngineArgs(**arguments))

    def add_request(self, *, prompt_token_ids, sampling_params, **kwargs):
        import numpy as np
        from deepinfer.sampling_params import SamplingParams

        return self.inner.add_request(
            prompt_token_ids=np.asarray(prompt_token_ids, dtype=np.int32),
            sampling_params=SamplingParams(**vars(sampling_params)),
            **kwargs,
        )

    def step(self):
        from deepinfer._C.engine import RequestOutput

        # Never let a pooled native pointer cross IPC or survive the next step.
        return [RequestOutput.clone(int(ptr)) for ptr in self.inner.step()]

    def flush(self):
        from deepinfer._C.engine import RequestOutput

        return [
            RequestOutput.clone(int(ptr)) for ptr in self.inner.flush_inflight_reqs()
        ]

    def abort_request(self, request_id):
        self.inner.abort_request(request_id)

    def close(self):
        self.inner.close()

    def get_num_unfinished_requests(self):
        return self.inner.get_num_unfinished_requests()


def serve(args):
    # Redirect fd 1 too: native C++/CUDA libraries can bypass Python's stdout.
    output = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    def emit(data):
        output.write(json.dumps(data, ensure_ascii=False) + "\n")
        output.flush()

    with contextlib.redirect_stdout(sys.stderr):
        tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
        if args.engine == "mock":
            from mock_engine import MockDeepInferEngine

            engine = MockDeepInferEngine(tokenizer)
        else:
            options = (
                json.loads(args.engine_args.read_text()) if args.engine_args else {}
            )
            engine = NativeDeepInferEngine(
                str(args.model_dir), args.context_length, options
            )
    emit(
        {
            "type": "ready",
            "vocab_size": tokenizer.get_vocab_size(),
            "eos_token_id": tokenizer.token_to_id("<｜end▁of▁sentence｜>"),
        }
    )
    commands = queue.Queue(maxsize=128)

    def read_commands():
        try:
            for line in sys.stdin:
                commands.put(json.loads(line))
        finally:
            commands.put({"type": "shutdown"})

    threading.Thread(target=read_commands, daemon=True).start()
    active = {}
    previous_load = 0
    try:
        while True:
            command = (
                commands.get()
                if not active and not engine.get_num_unfinished_requests()
                else None
            )
            pending = [command] if command else []
            while not commands.empty():
                pending.append(commands.get_nowait())
            with contextlib.redirect_stdout(sys.stderr):
                for command in pending:
                    kind = command["type"]
                    request_id = command.get("request_id")
                    if kind == "shutdown":
                        return
                    if kind == "abort":
                        if request_id in active:
                            engine.abort_request(request_id)
                            del active[request_id]
                        continue
                    if kind != "generate":
                        raise ValueError(f"Unknown engine command: {kind}")
                    params = SimpleNamespace(**command["sampling"])
                    try:
                        accepted = engine.add_request(
                            user_id="openengine",
                            request_id=request_id,
                            prompt_token_ids=command["token_ids"],
                            sampling_params=params,
                        )
                        if accepted is None:
                            raise ValueError("DeepInfer rejected token IDs")
                        active[request_id] = set(command["stop_token_ids"])
                    except (TypeError, ValueError) as exc:
                        emit(
                            {
                                "type": "output",
                                "request_id": request_id,
                                "error": str(exc),
                            }
                        )
                outputs = (
                    engine.step()
                    if active or engine.get_num_unfinished_requests()
                    else []
                )
                if (
                    active
                    and hasattr(engine, "flush")
                    and not engine.inner.has_unfinished_requests()
                ):
                    outputs.extend(engine.flush())
                for item in outputs:
                    if item.request_id not in active:
                        continue
                    stopped = item.token_id in active[item.request_id]
                    finished = item.finished or stopped
                    emit(
                        {
                            "type": "output",
                            "request_id": item.request_id,
                            "token_id": item.token_id,
                            "finished": finished,
                            "finish_reason": "stop" if stopped else item.finish_reason,
                            "logprob": item.token_logprobs,
                            "stop_token_id": item.token_id if stopped else None,
                        }
                    )
                    if finished:
                        if stopped and not item.finished:
                            engine.abort_request(item.request_id)
                        del active[item.request_id]
                load = engine.get_num_unfinished_requests()
                if load != previous_load:
                    emit({"type": "load", "running_requests": load})
                    previous_load = load
            if active and args.tick_ms:
                time.sleep(args.tick_ms / 1000)
    finally:
        with contextlib.redirect_stdout(sys.stderr):
            engine.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("mock", "deepinfer"), default="mock")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--engine-args", type=Path)
    parser.add_argument("--tick-ms", type=int, default=2)
    serve(parser.parse_args())
