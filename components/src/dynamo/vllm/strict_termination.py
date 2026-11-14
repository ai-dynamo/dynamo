# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request-local terminal evidence for strict RL token generation."""

from dataclasses import dataclass, replace
from typing import Any

from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest

from dynamo.llm.exceptions import InvalidArgument

try:
    from vllm.v1.engine.detokenizer import IncrementalDetokenizer
except ModuleNotFoundError as exc:
    # Older optional vLLM backends must still import non-strict handlers.
    if exc.name != "vllm.v1.engine.detokenizer":
        raise
    IncrementalDetokenizer = None


@dataclass(frozen=True)
class _Delta:
    """Immutable chunks keep string-stop evidence without copying earlier output."""

    token_ids: tuple[int, ...]
    text: str
    previous: "_Delta | None"


def _deltas(history: _Delta | None):
    while history is not None:
        yield history
        history = history.previous


@dataclass(frozen=True)
class StrictRLTermination:
    sampling_params: SamplingParams
    model_max_len: int
    prompt_token_ids: tuple[int, ...]
    request_id: str
    tokenizer: Any
    token_count: int = 0
    last_token: int | None = None
    history: _Delta | None = None
    finished: bool = False

    @classmethod
    def from_engine(cls, engine, sampling_params, prompt_token_ids, request_id):
        """Mirror the engine's request-local EOS configuration, never model guesses."""
        processor = engine.input_processor
        effective = sampling_params.clone()
        effective.update_from_generation_config(
            processor.generation_config_fields, processor.renderer.get_eos_token_id()
        )
        limit = engine.model_config.max_model_len
        if type(limit) is not int or limit < 1:
            raise RuntimeError("strict RL requires the actual engine context limit")
        if not prompt_token_ids or len(prompt_token_ids) >= limit:
            raise InvalidArgument(
                "strict RL requires a nonempty prompt below the context limit"
            )
        if effective.max_tokens is None:
            effective.max_tokens = limit - len(prompt_token_ids)
        if type(effective.max_tokens) is not int or effective.max_tokens < 1:
            raise InvalidArgument("strict RL requires a positive effective max_tokens")
        if effective.stop and (
            not effective.detokenize
            or processor.tokenizer is None
            or IncrementalDetokenizer is None
        ):
            raise InvalidArgument("strict RL string stops require the engine tokenizer")
        return cls(
            effective, limit, tuple(prompt_token_ids), request_id, processor.tokenizer
        )

    def observe(self, output):
        """Accept one DELTA output only when its count and terminal proof agree."""
        if self.finished:
            raise ValueError("strict RL received output after termination")
        if any(type(token) is not int or token < 0 for token in output.token_ids):
            raise ValueError("strict RL received invalid token IDs")
        if not isinstance(output.text, str):
            raise ValueError("strict RL received invalid output text")
        state = replace(
            self,
            token_count=self.token_count + len(output.token_ids),
            last_token=output.token_ids[-1] if output.token_ids else self.last_token,
            history=(
                _Delta(tuple(output.token_ids), output.text, self.history)
                if self.sampling_params.stop
                else None
            ),
        )
        count = state.token_count
        params = self.sampling_params
        if (
            count > params.max_tokens
            or len(self.prompt_token_ids) + count > self.model_max_len
        ):
            raise ValueError("strict RL output exceeds the request or context limit")
        finish, reason = output.finish_reason, output.stop_reason
        if finish is None:
            if reason is not None:
                raise ValueError("strict RL received a stop reason without termination")
            return state
        if finish == "length":
            if reason is not None or not (
                count == params.max_tokens
                or len(self.prompt_token_ids) + count == self.model_max_len
            ):
                raise ValueError(
                    "strict RL length termination lacks a token/context boundary"
                )
        elif finish == "stop":
            state._validate_stop(reason)
        else:
            raise ValueError("strict RL generation did not finish normally")
        return replace(state, finished=True)

    def _validate_stop(self, reason):
        params = self.sampling_params
        if self.token_count <= params.min_tokens:
            raise ValueError(
                "strict RL stop termination lacks generated token evidence"
            )
        if isinstance(reason, str):
            if not reason or reason not in params.stop:
                raise ValueError("strict RL stop string was not configured")
            self._validate_string_stop(reason)
            return
        if reason is None:
            accepted = set(params.stop_token_ids)
            if not params.ignore_eos and params.eos_token_id is not None:
                accepted.add(params.eos_token_id)
            if self.last_token in accepted:
                return
        elif type(reason) is int:
            if reason in params.stop_token_ids and self.last_token == reason:
                return
        raise ValueError(
            "strict RL stop termination lacks an accepted EOS or stop token"
        )

    def _validate_string_stop(self, reason):
        chunks = tuple(_deltas(self.history))
        token_ids = [token for chunk in reversed(chunks) for token in chunk.token_ids]
        text = "".join(chunk.text for chunk in reversed(chunks))
        request = EngineCoreRequest(
            request_id=self.request_id,
            prompt_token_ids=list(self.prompt_token_ids),
            mm_features=None,
            sampling_params=self.sampling_params,
            pooling_params=None,
            arrival_time=0.0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        )
        replay = IncrementalDetokenizer.from_new_request(self.tokenizer, request)
        # A string stop is detected by the frontend detokenizer, not EngineCore's
        # token-stop flag. Replaying with that flag would hide the trigger token.
        matched = replay.update(token_ids, stop_terminated=False)
        if (
            matched != reason
            or replay.output_token_ids != token_ids
            or replay.get_next_output_text(finished=True, delta=False) != text
        ):
            raise ValueError(
                "strict RL stop string does not match engine token/text evidence"
            )
