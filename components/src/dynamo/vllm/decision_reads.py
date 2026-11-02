# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional source-qualified native reads; this module registers no capabilities."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .decision_native_errors import (
    NativeExecutionError,
    NativeRuntimeError,
    NativeValidationError,
)
from .decision_native_lifecycle import NativeAdapter, NativeChildBatch
from .decision_native_registry import DIFFUSION_PROFILE, LIQUID_PROFILE, NativeProfile
from .decision_native_wire import (
    BudgetTokenizer,
    NativeRequest,
    NativeUsage,
    aggregate_usage,
    measured_cache_tokens,
    parse_native_request,
)
from .decision_runtime_policy import qualified_runtime

# Match the frontend reducer's tolerance for float32 vocabulary logprobs.
_VOCABULARY_MASS_TOLERANCE = 1e-6


def _runtime(engine: Any, selected: NativeProfile, tasks: tuple[str, ...]) -> None:
    profile = selected.name
    model, config = engine.model_config, engine.vllm_config
    if (
        not qualified_runtime(
            config,
            runner_type="generate",
            diffusion=profile == DIFFUSION_PROFILE,
            min_model_len=2,
        )
        or model.architecture != selected.architecture
        or model.logprobs_mode != "raw_logprobs"
        or selected.task not in tasks
        or (profile == LIQUID_PROFILE and config.diffusion_config is not None)
        or (
            profile == DIFFUSION_PROFILE
            and (config.diffusion_config is None or model.is_hybrid is not False)
        )
    ):
        raise NativeRuntimeError(
            "effective runtime is incompatible with the native read profile"
        )
    return None


@dataclass(frozen=True)
class PreparedRead:
    prompt: dict[str, Any]
    sampling_params: Any
    rendered_ids: tuple[int, ...]
    label_groups: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class PreparedReadRequest:
    questions: tuple[Any, ...]
    cache_salt: str
    reads: tuple[PreparedRead, ...]
    prefix: PreparedRead | None

    @property
    def admission_count(self) -> int:
        return len(self.reads) + int(self.prefix is not None)


@dataclass(frozen=True)
class PreparedReadText:
    request: NativeRequest
    questions: tuple[Any, ...]
    budget: BudgetTokenizer
    contents: tuple[str, ...]
    label_groups: tuple[tuple[tuple[int, ...], ...], ...]


class ReadExecutionAdapter(NativeAdapter):
    """Use native whole-request strategy semantics on one existing generation engine."""

    kind = "read"
    validate_runtime = staticmethod(_runtime)

    def _prepare_text(self, wire: Any) -> PreparedReadText:
        if not isinstance(wire, dict) or set(wire) != {"decision_read"}:
            raise NativeExecutionError("invalid internal native read envelope")
        request = parse_native_request(
            wire["decision_read"],
            self.spec.name,
            self.native,
            self.strategy.limits(),
        )
        questions = tuple(
            self.native.questions.Question(
                id=q.id,
                type=q.type,
                instructions=q.instructions,
                options=q.options,
                labels=q.type.label_set[: len(q.options)],
            )
            for q in request.questions
        )
        budget = BudgetTokenizer(
            self.preflight_tokenizer,
            request.max_tokenization_work,
            request.max_encoding_bytes,
        )
        # Only this private strategy copy uses the serialized preflight tokenizer.
        labels = copy.copy(self.strategy)
        labels._tokenizer = lambda: budget
        state = self.strategy.render_state(request.state)
        contents, groups = [], []
        for question in questions:
            content = self.strategy._content(state, question)
            budget.consume_bytes(content)
            contents.append(content)
            groups.append(tuple(tuple(g) for g in labels._label_groups(question)))
        return PreparedReadText(
            request, questions, budget, tuple(contents), tuple(groups)
        )

    async def _prepare(self, wire: Any) -> PreparedReadRequest:
        """Render and validate all native prompts, label groups, and canvas capacity."""
        try:
            text = await self.run_preflight(lambda: self._prepare_text(wire))
            request = text.request
            strategy = self.strategy
            read_request = strategy._read_request(None, request.cache_salt)
            prepared = []
            for content, groups in zip(text.contents, text.label_groups):
                rendered, prompt_ids = await self._render(read_request, content)
                text.budget.consume_tokens(len(prompt_ids))
                if prompt_ids[len(prompt_ids) - len(strategy.tail) :] != strategy.tail:
                    raise NativeExecutionError(
                        "native read prompt changed the trained label suffix"
                    )
                params = strategy._sampling_params(
                    [token for group in groups for token in group], prompt_ids
                )
                prompt = strategy._read_input(rendered, prompt_ids)
                self._validate_plan(prompt, params, request.cache_salt)
                prepared.append(
                    PreparedRead(
                        copy.deepcopy(prompt),
                        params,
                        tuple(prompt_ids),
                        groups,
                    )
                )
            prefix = self._prefix(prepared, request.cache_salt)
            count = sum(len(read.prompt["prompt_token_ids"]) for read in prepared)
            if prefix is not None:
                count += len(prefix.prompt["prompt_token_ids"])
            if count > request.max_input_tokens:
                raise NativeValidationError(
                    "native read total input token budget exceeded"
                )
            return PreparedReadRequest(
                text.questions,
                request.cache_salt,
                tuple(prepared),
                prefix,
            )
        except (
            self.native.questions.StructuredDecisionError,
            self.native.validation_error,
            UnicodeEncodeError,
        ) as cause:
            raise NativeValidationError(
                "invalid or unsupported native read request"
            ) from cause

    async def _render(self, read_request, content):
        try:
            return await self.strategy._render(
                read_request, [{"role": "user", "content": content}]
            )
        except (
            NativeValidationError,
            self.native.questions.StructuredDecisionError,
            self.native.validation_error,
            UnicodeEncodeError,
        ):
            raise
        except (AttributeError, KeyError, TypeError, ValueError) as cause:
            raise NativeExecutionError(
                "native renderer returned invalid prompt input"
            ) from cause

    def _validate_plan(self, prompt: Any, params: Any, salt: str) -> None:
        if (
            not isinstance(prompt, dict)
            or prompt.get("type") != "token"
            or set(prompt)
            - {
                "prompt_token_ids",
                "cache_salt",
                "type",
                "prompt",
                "prompt_token_offsets",
                "arrival_time",
            }
        ):
            raise NativeExecutionError("native read requires a text-only engine input")
        ids = prompt.get("prompt_token_ids")
        if (
            not isinstance(ids, list)
            or not ids
            or any(type(token) is not int or token < 0 for token in ids)
        ):
            raise NativeExecutionError("native read rendered invalid prompt tokens")
        if prompt.get("cache_salt") != salt:
            raise NativeExecutionError(
                "native renderer did not preserve isolation salt"
            )
        if len(ids) + params.max_tokens > self.engine.model_config.max_model_len:
            raise NativeValidationError(
                "native read exceeds effective context capacity"
            )
        if self.spec.name == DIFFUSION_PROFILE:
            extra = params.extra_args
            if (
                params.max_tokens != 2
                or extra["diffusion_canvas_length"] != 16
                or extra["diffusion_max_steps"] != 1
                or extra["diffusion_read_only"] is not True
                or len(extra["diffusion_seed_canvas"]) != 16
            ):
                raise NativeExecutionError(
                    "native diffusion canvas differs from qualified read profile"
                )

    def _prefix(self, prepared, salt) -> PreparedRead | None:
        if (
            len(prepared) < 2
            or not self.engine.model_config.is_hybrid
            or not self.engine.vllm_config.cache_config.enable_prefix_caching
        ):
            return None
        prompts = [read.rendered_ids for read in prepared]
        shared = self.native.strategies.common_prefix_len(prompts)
        if shared < self.native.strategies.MIN_WARM_PREFIX_TOKENS:
            return None
        prompt = self.native.strategies.tokens_input(
            list(prompts[0][:shared]), cache_salt=salt
        )
        params = self.native.strategies.SamplingParams(max_tokens=1, temperature=0.0)
        return PreparedRead(prompt, params, tuple(prompt["prompt_token_ids"]))

    def _children(
        self,
        prepared: PreparedReadRequest,
        context: Any,
        *,
        request_id: str,
        cancelled: Callable[[], bool],
        trace_headers: Mapping[str, str] | None = None,
    ):
        if prepared.prefix is not None:
            yield NativeChildBatch(
                1,
                lambda _: self._read(
                    prepared.prefix,
                    context,
                    f"{request_id}-prefix",
                    cancelled,
                    trace_headers,
                ),
            )

        async def child(index):
            return await self._read(
                prepared.reads[index],
                context,
                f"{request_id}-{index}",
                cancelled,
                trace_headers,
            )

        yield NativeChildBatch(len(prepared.reads), child)

    async def _read(self, read, context, request_id, cancelled, trace_headers):
        def validate(output):
            return output, self._validate_output(read, output)

        return await self.read_child(
            lambda: self.engine.generate(
                read.prompt,
                read.sampling_params,
                request_id,
                lora_request=None,
                trace_headers=trace_headers,
                priority=0,
            ),
            request_id,
            context,
            cancelled,
            validate,
        )

    def _validate_output(self, read, output) -> NativeUsage:
        if not isinstance(output.prompt_token_ids, list) or tuple(
            output.prompt_token_ids
        ) != tuple(read.prompt["prompt_token_ids"]):
            raise NativeExecutionError("native read changed the prepared prompt tokens")
        if len(output.outputs) != 1:
            raise NativeExecutionError("native read requires one complete output")
        completion = output.outputs[0]
        if (
            completion.index != 0
            or completion.finish_reason not in ("stop", "length")
            or not isinstance(completion.token_ids, (list, tuple))
            or len(completion.token_ids) > read.sampling_params.max_tokens
            or any(
                type(token) is not int or token < 0 for token in completion.token_ids
            )
        ):
            raise NativeExecutionError(
                "native read returned invalid completion evidence"
            )
        cached = measured_cache_tokens(
            getattr(output, "num_cached_tokens", None), len(output.prompt_token_ids)
        )
        label_ids = read.sampling_params.logprob_token_ids
        if label_ids:
            if not completion.logprobs or any(
                token not in completion.logprobs[0] for token in label_ids
            ):
                raise NativeExecutionError(
                    "native read omitted selected label logprobs"
                )
            if any(
                type(completion.logprobs[0][token].logprob) not in (int, float)
                or math.isnan(completion.logprobs[0][token].logprob)
                or completion.logprobs[0][token].logprob > 0
                for token in label_ids
            ):
                raise NativeExecutionError(
                    "native read returned invalid label logprobs"
                )
        return NativeUsage(
            input_tokens=len(output.prompt_token_ids),
            output_tokens=len(completion.token_ids),
            cached_tokens=cached,
        )

    def _reduce(self, prepared, children):
        """Use the pinned stateless reducer, never replay engine execution."""
        reduced = []
        for read, output in zip(prepared.reads, children):
            logprobs = [
                output.outputs[0].logprobs[0][token].logprob
                for token in read.sampling_params.logprob_token_ids
            ]
            reduced.append(
                self.native.strategies.NextTokenStrategy._question_read(
                    output, [list(group) for group in read.label_groups], logprobs
                )
            )
        return reduced

    def _respond(self, prepared, results) -> dict[str, Any]:
        children = [output for output, _ in results[int(prepared.prefix is not None) :]]
        native_reads = self._reduce(prepared, children)
        if len(native_reads) != len(prepared.questions):
            raise NativeExecutionError("native reducer returned incomplete outcomes")
        outcomes = []
        for ordinal, (question, result) in enumerate(
            zip(prepared.questions, native_reads)
        ):
            probabilities = result.probs
            if (
                len(probabilities) != len(question.options)
                or any(
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or not 0 <= value <= 1
                    for value in probabilities
                )
                or abs(sum(probabilities) - 1) > 1e-6
            ):
                raise NativeExecutionError(
                    "native reducer returned invalid probabilities"
                )
            mass = result.label_mass
            if (
                type(mass) not in (int, float)
                or not math.isfinite(mass)
                or not 0 <= mass <= 1 + _VOCABULARY_MASS_TOLERANCE
            ):
                raise NativeExecutionError(
                    "native reducer returned invalid vocabulary label mass"
                )
            outcomes.append(
                {
                    "ordinal": ordinal,
                    "probabilities": probabilities,
                    "label_mass": min(mass, 1.0),
                }
            )
        return {
            "protocol_version": 1,
            "profile": self.spec.name,
            "outcomes": outcomes,
            "usage": aggregate_usage([usage for _, usage in results]),
        }
