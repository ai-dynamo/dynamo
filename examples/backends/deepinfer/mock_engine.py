# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The only synthetic part of the local example: DeepInfer execution/output.

The tokenizer, gRPC transport, sidecar, frontend renderer and parsers are real.
The method boundary mirrors DeepInfer's add_request/step/abort_request driver.
"""

from dataclasses import dataclass


@dataclass
class RequestOutput:
    request_id: str
    token_id: int
    finished: bool
    finish_reason: str
    token_logprobs: float = 0.0


class MockDeepInferEngine:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.requests = {}

    def add_request(self, *, request_id, prompt_token_ids, sampling_params, **_):
        prompt = self.tokenizer.decode(prompt_token_ids, skip_special_tokens=False)
        user = prompt.rsplit("<｜User｜>", 1)[-1].split("<｜Assistant｜>", 1)[0]
        thinking = prompt.rsplit("<｜Assistant｜>", 1)[-1].startswith("<think>")
        if "mock:tool" in user:
            body = (
                '<｜DSML｜ calls><｜DSML｜ invoke name="get_weather">'
                '<｜DSML｜ parameter name="city" string="true">Tokyo</｜DSML｜ parameter>'
                "</｜DSML｜ invoke></｜DSML｜ calls>"
            )
        elif "mock:unicode" in user:
            body = "東京 café 𐍈"
        elif "mock:slow" in user:
            body = "slow " * 4096
        elif "mock:reasoning" in user:
            body = "42"
        else:
            body = "Hello from DeepInfer mock."
        if thinking:
            body = "I will check the request carefully.</think>" + body
        ids = self.tokenizer.encode(body, add_special_tokens=False).ids
        maximum = sampling_params.max_tokens
        if sampling_params.ignore_eos:
            ids = (ids * (maximum // len(ids) + 1))[:maximum]
            reason = "length"
        else:
            ids.append(self.tokenizer.token_to_id("<｜end▁of▁sentence｜>"))
            reason = "stop" if len(ids) <= maximum else "length"
        self.requests[request_id] = [
            iter(ids[:maximum]),
            min(len(ids), maximum),
            reason,
        ]
        return True

    def step(self):
        outputs = []
        for request_id, state in list(self.requests.items()):
            token_id = next(state[0])
            state[1] -= 1
            finished = state[1] == 0
            outputs.append(
                RequestOutput(
                    request_id, token_id, finished, state[2] if finished else ""
                )
            )
            if finished:
                del self.requests[request_id]
        return outputs

    def abort_request(self, request_id):
        self.requests.pop(request_id, None)

    def close(self):
        self.requests.clear()

    def get_num_unfinished_requests(self):
        return len(self.requests)
