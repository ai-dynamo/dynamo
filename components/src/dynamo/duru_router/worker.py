# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observe accepted work without changing native vLLM queue order."""

import os
import time

from vllm.v1.core.sched.async_scheduler import AsyncScheduler

from dynamo.duru_router.events import CompactFeed


class DuruScheduler(AsyncScheduler):
    """Preserve native scheduling while recording only work accepted by the engine."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.num_spec_tokens or self.num_sampled_tokens_per_step != 1:
            raise ValueError("Duru requires non-speculative autoregressive generation")
        self._pending_work = {}
        self.compact_feed = CompactFeed(os.environ["DURU_WORK_DIR"])

    def identity(self, request):
        extra = request.sampling_params.extra_args or {}
        client_id = extra.get("duru_request_id") or request.request_id
        root = extra.get("duru_root_id") or client_id
        return dict(
            client_request_id=client_id,
            root=root,
            agent=extra.get("duru_agent_id") or root,
        )

    def add_request(self, request, *args, **kwargs):
        if request.sampling_params is None or request.resumable:
            raise ValueError("Duru requires non-resumable generation requests")
        return super().add_request(request, *args, **kwargs)

    def _update_after_schedule(self, scheduler_output):
        if scheduler_output.scheduled_spec_decode_tokens:
            raise ValueError("Speculative work is outside Duru's accounting model")
        with self.compact_feed.lock:
            started = time.monotonic()
            pending = {}
            for request_id, count in scheduler_output.num_scheduled_tokens.items():
                request = self.requests[request_id]
                computed = request.num_computed_tokens
                pending[request_id] = (
                    request,
                    dict(
                        **self.identity(request),
                        scheduled_at=started,
                        p=count,
                        d=computed,
                        k=computed + count,
                        phase=(
                            "prefill"
                            if computed < request.num_prompt_tokens or count > 1
                            else "decode"
                        ),
                    ),
                )
            self._pending_work[id(scheduler_output)] = pending
            return super()._update_after_schedule(scheduler_output)

    def update_from_output(self, scheduler_output, runner_output):
        if scheduler_output.scheduled_spec_decode_tokens:
            raise ValueError("Speculative work is outside Duru's accounting model")
        connector = runner_output.kv_connector_output
        if connector is not None and (
            connector.invalid_block_ids or connector.failed_recving
        ):
            raise ValueError(
                "Invalid KV blocks or failed KV transfers are outside Duru's accounting model"
            )
        with self.compact_feed.lock:
            pending = self._pending_work.pop(id(scheduler_output))
            acceptance = {}
            for request_id, (request, _) in pending.items():
                accepted = (
                    self.requests.get(request_id) is request
                    and not request.is_finished()
                    and not (
                        request.num_stale_output_tokens > 0
                        and request.drop_stale_output
                    )
                )
                acceptance[request_id] = (accepted, request.num_output_tokens)
            result = super().update_from_output(scheduler_output, runner_output)
            ended = time.monotonic()
            steps = []
            for request_id, (request, step) in pending.items():
                accepted, previous_tokens = acceptance[request_id]
                steps.append(
                    dict(
                        step,
                        accepted=accepted,
                        end=ended,
                        generated_tokens=(
                            max(0, request.num_output_tokens - previous_tokens)
                            if accepted
                            else 0
                        ),
                    )
                )
            self.compact_feed.capture("accepted_output", dict(steps=steps))
            finished = [
                self.identity(request)
                for rid, (request, _) in pending.items()
                if acceptance[rid][0] and request.is_finished()
            ]
            if finished:
                self.compact_feed.emit("worker_request_finished", requests=finished)
            return result

    def finish_requests(self, *args, **kwargs):
        # Abort acknowledgement follows native removal. Later stale output cannot
        # be accepted, so accounting may release a terminal root after this event.
        with self.compact_feed.lock:
            finished = super().finish_requests(*args, **kwargs)
            if finished:
                self.compact_feed.emit(
                    "worker_request_finished",
                    requests=[self.identity(r) for r in finished],
                )
            return finished

    def shutdown(self):
        try:
            return super().shutdown()
        finally:
            self.compact_feed.close()
