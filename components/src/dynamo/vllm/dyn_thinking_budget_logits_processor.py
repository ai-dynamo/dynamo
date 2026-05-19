# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo-owned thinking-token budget logits processor.

Equivalent to vLLM's builtin ``ThinkingTokenBudgetLogitsProcessor`` /
``ThinkingBudgetStateHolder`` (see ``vllm/v1/sample/thinking_budget_state.py``),
but installed via the user-provided logits-processor mechanism so the worker
does **not** need ``--reasoning-parser`` and ``vllm_config.reasoning_config``
stays ``None``.

Why this matters: when ``reasoning_config`` is non-None,
``gpu_model_runner.py`` flips ``logitsprocs_need_output_token_ids=True``, which
changes the sampling pipeline's batch handling even when no request carries a
budget. That change empirically destabilizes ``tool_choice=required`` on Qwen3
family models — the model drifts from XML to a JSON-array shape (which the
qwen3_coder parser cannot recover when truncated), producing ~20% leaks.

This processor enforces the budget without touching ``reasoning_config``:
- token IDs for the reasoning start / end markers are derived from the
  model tokenizer at init time
- per-request budget comes from ``SamplingParams.extra_args[BUDGET_KEY]``,
  not ``SamplingParams.thinking_token_budget`` (which would trip vLLM's
  ``input_processor`` validation in the absence of ``reasoning_config``)

The runtime algorithm mirrors the wrapped path of
``ThinkingBudgetStateHolder._update_think_state`` and ``_apply_forcing_to_logits``,
restricted to the non-speculative-decoding case which is all Dynamo currently
needs.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch

from vllm.v1.sample.logits_processor.builtin import process_dict_updates
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
)

if TYPE_CHECKING:
    from vllm import SamplingParams
    from vllm.config import VllmConfig


# Per-request state stored in the dict.
ReqState = dict[str, Any]

# Key under which the budget is propagated from the request through to this
# logits processor. Mirrors the BUDGET_KEY contract documented in
# ``handlers.build_sampling_params``.
BUDGET_KEY = "dyn_thinking_token_budget"

# Environment variable carrying the dyn-reasoning-parser name from worker
# launch into the engine subprocess. Required because LogitsProcessor
# instances are constructed inside the engine process with only
# ``vllm_config`` available.
PARSER_ENV = "DYN_REASONING_PARSER_FOR_BUDGET"

# Reasoning markers per parser family. Derived from the upstream
# ``ReasoningParserManager`` registrations.
_PARSER_MARKERS: dict[str, tuple[str, str]] = {
    "qwen3": ("<think>", "</think>"),
    "deepseek_r1": ("<think>", "</think>"),
}


class DynThinkingBudgetLogitsProcessor(LogitsProcessor):
    """Thinking-token budget enforcement independent of vLLM reasoning_config."""

    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ) -> None:
        _ = is_pin_memory
        self.device = device

        parser_name = os.environ.get(PARSER_ENV, "").strip()
        markers = _PARSER_MARKERS.get(parser_name)
        if markers is None:
            # Unknown / unset parser: install as a no-op rather than crashing
            # so the engine still boots when the env var is missing.
            self.think_start_token_ids: list[int] = []
            self.think_end_token_ids: list[int] = []
        else:
            from vllm.transformers_utils.tokenizer import get_tokenizer

            tokenizer = get_tokenizer(
                vllm_config.model_config.tokenizer,
                tokenizer_mode=vllm_config.model_config.tokenizer_mode,
                trust_remote_code=vllm_config.model_config.trust_remote_code,
                revision=vllm_config.model_config.tokenizer_revision,
            )
            self.think_start_token_ids = tokenizer.encode(
                markers[0], add_special_tokens=False
            )
            self.think_end_token_ids = tokenizer.encode(
                markers[1], add_special_tokens=False
            )

        self.is_enabled = bool(
            self.think_start_token_ids and self.think_end_token_ids
        )

        # index -> per-request state
        self._state: dict[int, ReqState] = {}

    @classmethod
    def validate_params(cls, sampling_params: "SamplingParams") -> None:
        budget = (sampling_params.extra_args or {}).get(BUDGET_KEY)
        if budget is None:
            return
        if not isinstance(budget, int) or budget <= 0:
            raise ValueError(
                f"SamplingParams.extra_args[{BUDGET_KEY!r}] must be a positive int; "
                f"got {budget!r}"
            )

    def is_argmax_invariant(self) -> bool:
        # Forcing the </think> token series can flip greedy outcomes.
        return False

    # -------------------------------------------------------------------
    # State management
    # -------------------------------------------------------------------

    def _add_request(
        self,
        params: "SamplingParams",
        prompt_tok_ids: list[int] | None,
        output_tok_ids: list[int],
    ) -> ReqState | None:
        if not params.extra_args:
            return None
        budget = params.extra_args.get(BUDGET_KEY)
        if budget is None:
            return None

        # Detect whether the prompt already entered ``<think>...`` without
        # ``</think>``. Qwen3-style chat templates inject ``<think>\n`` into
        # the prompt when ``enable_thinking=True``, so the model never emits
        # the start sequence itself — we must seed the state with the
        # already-consumed thinking tokens from the prompt.
        prompt_think_count = 0
        in_prompt_think = False
        if prompt_tok_ids:
            last_start = self._find_last_subseq(
                prompt_tok_ids, self.think_start_token_ids
            )
            last_end = self._find_last_subseq(
                prompt_tok_ids, self.think_end_token_ids
            )
            in_prompt_think = last_start > last_end
            if in_prompt_think:
                prompt_think_count = len(prompt_tok_ids) - (
                    last_start + len(self.think_start_token_ids)
                )

        # ``think_token_start`` is expressed in *output* coordinates: a
        # negative value means the start tag fell inside the prompt, with
        # magnitude equal to thinking tokens already accumulated before
        # generation. ``in_think`` is the explicit flag (start_offset alone
        # cannot encode "not in think" without colliding with a -1 from a
        # prompt that consumed exactly one thinking token).
        return {
            "budget": int(budget),
            "output_tok_ids": output_tok_ids,
            "in_think": in_prompt_think,
            "think_token_start": -prompt_think_count if in_prompt_think else 0,
            "in_end": False,
            "end_count": 0,
        }

    @staticmethod
    def _find_last_subseq(haystack: list[int], needle: list[int]) -> int:
        if not needle or len(haystack) < len(needle):
            return -1
        for i in range(len(haystack) - len(needle), -1, -1):
            if haystack[i : i + len(needle)] == needle:
                return i
        return -1

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if not self.is_enabled:
            return
        process_dict_updates(self._state, batch_update, self._add_request)

    # -------------------------------------------------------------------
    # Per-step enforcement
    # -------------------------------------------------------------------

    def _refresh_think_position(self, state: ReqState) -> None:
        """Update ``in_think`` based on the latest emitted output tokens.

        Detects:
          - A new ``<think>`` sequence emitted by the model — flips
            ``in_think`` on and records the position after the start tokens.
          - A ``</think>`` sequence — exits ``in_think`` (model closed the
            reasoning block on its own).
        """
        output = state["output_tok_ids"]
        n = len(output)

        if state["in_think"] and self._ends_with(
            output, self.think_end_token_ids
        ):
            state["in_think"] = False
            state["in_end"] = False
            state["end_count"] = 0
            return

        if (not state["in_think"]) and self._ends_with(
            output, self.think_start_token_ids
        ):
            state["in_think"] = True
            state["think_token_start"] = n

    @staticmethod
    def _ends_with(seq: list[int], suffix: list[int]) -> bool:
        if not suffix or len(seq) < len(suffix):
            return False
        return seq[-len(suffix) :] == suffix

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.is_enabled or not self._state:
            return logits

        end_len = len(self.think_end_token_ids)
        vocab = logits.shape[-1]
        for index, state in list(self._state.items()):
            if index >= logits.shape[0]:
                continue

            self._refresh_think_position(state)

            if not state["in_end"] and state["in_think"]:
                # think_token_start is in *output* coordinates: negative means
                # the prompt already consumed -think_token_start thinking
                # tokens before generation started.
                think_count = (
                    len(state["output_tok_ids"]) - state["think_token_start"]
                )
                if think_count >= state["budget"]:
                    state["in_end"] = True
                    state["end_count"] = 0

            if state["in_end"]:
                if state["end_count"] < end_len:
                    forced = self.think_end_token_ids[state["end_count"]]
                    if 0 <= forced < vocab:
                        # Mask everything except the forced token. Use -inf
                        # so softmax assigns probability 1.0 to ``forced``.
                        row = logits[index]
                        row.fill_(float("-inf"))
                        row[forced] = 1.0e9
                    state["end_count"] += 1
                else:
                    # All </think> tokens emitted; resume normal generation.
                    state["in_end"] = False
                    state["in_think"] = False
                    state["end_count"] = 0

        return logits
