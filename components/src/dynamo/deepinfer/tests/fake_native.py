# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A scripted stand-in for DeepInfer's native ``LLMEngine`` (no GPU needed)."""

import threading
import time
from types import SimpleNamespace


class Slot:
    """Mimics a native output buffer that the next step overwrites."""

    def __init__(self):
        self.request_id = ""
        self.token_id = 0
        self.finished = False
        self.finish_reason = ""
        self.prompt_len = 0
        self.token_logprobs = None
        self.top_logprobs = []
        self.top_token_ids = []


class FakeEngine:
    """Generates ``max_tokens`` tokens per request, one per step.

    ``max_tokens`` is read from ``sampling_params.max_tokens`` or, when the
    caller passes a bare int, from the int itself.
    """

    def __init__(self):
        self.slots = {}
        self.pending = {}
        self.added = []
        self.aborted = []
        self.closed = threading.Event()
        self.step_threads = set()
        self.fail_on_step = None
        self.reject = set()
        self.steps = 0
        self.step_delay = 0.001  # keep long requests in flight while tests act
        self.finish_reason = "length"
        self.final_token = None
        self.logprob = -0.5
        self.top_k = 0

    def add_request(self, *, user_id, request_id, sampling_params, prompt_token_ids):
        if request_id in self.reject:
            raise ValueError("prompt too long")
        self.added.append((request_id, sampling_params, list(prompt_token_ids)))
        max_tokens = getattr(sampling_params, "max_tokens", sampling_params)
        self.pending[request_id] = [0, max_tokens, len(prompt_token_ids)]

    def abort_request(self, request_id):
        self.aborted.append(request_id)
        self.pending.pop(request_id, None)

    def has_unfinished_requests(self):
        return bool(self.pending)

    def step(self):
        self.steps += 1
        time.sleep(self.step_delay)
        self.step_threads.add(threading.current_thread().name)
        if self.fail_on_step is not None and self.steps >= self.fail_on_step:
            raise RuntimeError("cuda error")
        pointers = []
        for request_id in list(self.pending):
            produced, max_tokens, prompt_len = self.pending[request_id]
            slot = self.slots.setdefault(request_id, Slot())
            slot.request_id = request_id
            slot.token_id = 100 + produced
            slot.prompt_len = prompt_len
            slot.token_logprobs = self.logprob
            slot.top_token_ids = [slot.token_id + k for k in range(self.top_k)]
            slot.top_logprobs = [-0.1 * (k + 1) for k in range(self.top_k)]
            slot.finished = produced + 1 >= max_tokens
            slot.finish_reason = self.finish_reason if slot.finished else ""
            if slot.finished and self.final_token is not None:
                slot.token_id = self.final_token
            self.pending[request_id][0] += 1
            if slot.finished:
                del self.pending[request_id]
            pointers.append(request_id)
        return pointers

    def flush_inflight_reqs(self):
        return []

    def clone(self, request_id):
        """Like RequestOutput.clone: an owned copy that survives the next step."""
        return SimpleNamespace(**vars(self.slots[request_id]))

    def close(self):
        self.closed.set()
