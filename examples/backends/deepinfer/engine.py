# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python engine owner called by the Rust server through PyO3."""

import logging
import queue
import threading
from concurrent.futures import Future
from types import SimpleNamespace

from tokenizers import Tokenizer

logger = logging.getLogger(__name__)


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

        # Clone pooled native outputs before another step or inflight flush.
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


class RuntimeHandle:
    """Only the owner thread touches the engine; Rust calls submit/abort."""

    def __init__(self, args, engine_factory=None):
        self.args = args
        self._factory = engine_factory
        self._commands = queue.Queue(maxsize=args.max_requests)
        self._requests = {}
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._status_callback = None
        self.load = 0
        self.error = None
        self._ready = Future()
        self._thread = threading.Thread(
            target=self._run, name="deepinfer-engine", daemon=True
        )
        self._thread.start()
        try:
            self.vocab_size, self.eos_token_id = self._ready.result(timeout=120)
        except BaseException:
            self.close()
            raise

    def set_status_callback(self, callback):
        self._status_callback = callback
        callback(self.load, self.error)

    def submit(self, request_id, token_ids, stop_ids, sampling, callback):
        with self._lock:
            if self.error or self._stopping.is_set():
                raise RuntimeError(self.error or "engine is stopping")
            if len(self._requests) >= self.args.max_requests:
                raise OverflowError("engine request capacity reached")
            cancelled = threading.Event()
            self._requests[request_id] = cancelled
            try:
                self._commands.put_nowait(
                    (request_id, token_ids, stop_ids, sampling, callback, cancelled)
                )
            except queue.Full:
                del self._requests[request_id]
                raise OverflowError("engine command queue full") from None

    def abort(self, request_id):
        with self._lock:
            cancelled = self._requests.get(request_id)
        if cancelled is not None:
            cancelled.set()

    def close(self):
        self._stopping.set()
        try:
            self._commands.put_nowait(None)
        except queue.Full:
            pass  # The owner also checks _stopping between steps.
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise RuntimeError("DeepInfer engine thread did not stop within 10 seconds")
        self._status_callback = None

    def _release(self, request_id):
        with self._lock:
            self._requests.pop(request_id, None)

    def _report(self, load):
        changed = self.load != load
        self.load = load
        if self._status_callback and (changed or self.error):
            self._status_callback(load, self.error)

    def _run(self):
        engine = None
        active = {}
        try:
            tokenizer = Tokenizer.from_file(str(self.args.model_dir / "tokenizer.json"))
            if self._factory is not None:
                engine = self._factory(tokenizer)
            elif self.args.engine == "mock":
                from mock_engine import MockDeepInferEngine

                engine = MockDeepInferEngine(tokenizer)
            else:
                engine = NativeDeepInferEngine(
                    str(self.args.model_dir),
                    self.args.context_length,
                    self.args.engine_options,
                )
            self._ready.set_result(
                (
                    tokenizer.get_vocab_size(),
                    tokenizer.token_to_id("<｜end▁of▁sentence｜>"),
                )
            )
            while not self._stopping.is_set():
                command = (
                    self._commands.get()
                    if not active and not engine.get_num_unfinished_requests()
                    else None
                )
                commands = [command] if command else []
                while True:
                    try:
                        queued = self._commands.get_nowait()
                    except queue.Empty:
                        break
                    if queued is not None:
                        commands.append(queued)
                if self._stopping.is_set():
                    break
                for (
                    request_id,
                    token_ids,
                    stop_ids,
                    sampling,
                    callback,
                    cancelled,
                ) in commands:
                    if cancelled.is_set():
                        self._release(request_id)
                        continue
                    try:
                        accepted = engine.add_request(
                            user_id="openengine",
                            request_id=request_id,
                            prompt_token_ids=token_ids,
                            sampling_params=SimpleNamespace(**sampling),
                        )
                        if accepted is None:
                            raise ValueError("DeepInfer rejected token IDs")
                        active[request_id] = (set(stop_ids), callback, cancelled)
                    except (TypeError, ValueError) as exc:
                        callback(error=str(exc))
                        self._release(request_id)
                for request_id, (_, _, cancelled) in list(active.items()):
                    if cancelled.is_set():
                        engine.abort_request(request_id)
                        del active[request_id]
                        self._release(request_id)
                # Publish admission before a potentially long prefill step.
                self._report(engine.get_num_unfinished_requests())
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
                    request = active.get(item.request_id)
                    if request is None:
                        continue
                    stop_ids, callback, _ = request
                    stopped = item.token_id in stop_ids
                    finished = item.finished or stopped
                    consumed = callback(
                        token_id=item.token_id,
                        finished=finished,
                        finish_reason="stop" if stopped else item.finish_reason,
                        logprob=item.token_logprobs,
                        stop_token_id=item.token_id if stopped else None,
                    )
                    if finished or not consumed:
                        if not item.finished:
                            engine.abort_request(item.request_id)
                        del active[item.request_id]
                        self._release(item.request_id)
                self._report(engine.get_num_unfinished_requests())
                if active and self.args.tick_ms:
                    self._stopping.wait(self.args.tick_ms / 1000)
        except Exception as exc:
            # Engine failures must terminate Rust streams instead of leaving them hanging.
            self.error = f"DeepInfer engine failed: {exc}"
            logger.exception("DeepInfer engine failed")
            if not self._ready.done():
                self._ready.set_exception(exc)
        finally:
            try:
                if engine is not None:
                    engine.close()
            except Exception as exc:
                self.error = self.error or f"DeepInfer close failed: {exc}"
                logger.exception("DeepInfer engine cleanup failed")
            with self._lock:
                self._requests.clear()
            self._report(0)
