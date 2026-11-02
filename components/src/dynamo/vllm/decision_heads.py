# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional native D3/Clef execution and fail-closed readiness advertisement."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Any

from .decision_native_errors import (
    NativeExecutionError,
    NativeRuntimeError,
    NativeValidationError,
)
from .decision_native_lifecycle import (
    NATIVE_HEALTH_CACHE_SALT,
    NativeAdapter,
    NativeChildBatch,
    run_with_context_cancellation,
    translate_native_errors,
)
from .decision_native_registry import NativeProfile
from .decision_native_wire import (
    BudgetTokenizer,
    NativeUsage,
    aggregate_usage,
    measured_cache_tokens,
    parse_native_request,
)
from .decision_runtime_policy import qualified_runtime


@dataclass(frozen=True)
class PreparedHeadRead:
    prompt_ids: tuple[int, ...]
    pooling_params: Any
    ordinals: tuple[int, ...]
    orders: tuple[tuple[int, ...], ...]
    counts: tuple[int, ...]


@dataclass(frozen=True)
class PreparedHeadRequest:
    cache_salt: str
    reads: tuple[PreparedHeadRead, ...]

    @property
    def admission_count(self) -> int:
        return len(self.reads)


def _runtime_profile(
    engine: Any, selected: NativeProfile, tasks: tuple[str, ...]
) -> None:
    config = engine.vllm_config
    if (
        not qualified_runtime(
            config, runner_type="pooling", diffusion=False, min_model_len=1
        )
        or engine.model_config.architecture != selected.architecture
        or selected.task not in tasks
        or config.cache_config.enable_prefix_caching is not False
    ):
        raise NativeRuntimeError(
            "effective runtime is incompatible with the head profile"
        )
    return None


def _validate_order(order: Any, count: int) -> tuple[int, ...]:
    if any(type(index) is not int for index in order) or sorted(order) != list(
        range(count)
    ):
        raise NativeExecutionError(
            "native head preparation returned an invalid option order"
        )
    return tuple(order)


@dataclass
class D3Decoder:
    engine: Any
    native: Any
    strategy: Any

    def __post_init__(self):
        strategy = self.strategy
        if (
            not isinstance(strategy.codes, list)
            or not strategy.codes
            or len(strategy.codes) > 255
            or any(not isinstance(code, str) or not code for code in strategy.codes)
            or len(set(strategy.codes)) != len(strategy.codes)
            or type(strategy.temperature) not in (int, float)
            or not math.isfinite(strategy.temperature)
            or strategy.temperature <= 0
        ):
            raise NativeRuntimeError("D3 requires valid answer codes and temperature")

    def prepare(self, request, strategy, state) -> list[PreparedHeadRead]:
        reads = []
        for ordinal, question in enumerate(request.questions):
            prompt = strategy._tokens(strategy._prompt(state, question, []))
            if len(prompt) > self.engine.model_config.max_model_len:
                raise NativeValidationError(
                    "D3 prompt exceeds the effective context limit"
                )
            _, order = strategy._options(question)
            count = len(question.options)
            reads.append(
                PreparedHeadRead(
                    tuple(prompt),
                    self.native.pooling_params(task="classify", use_activation=False),
                    (ordinal,),
                    (_validate_order(order, count),),
                    (count,),
                )
            )
        return reads

    def decode(self, read, values):
        if len(values) != len(self.strategy.codes):
            raise NativeExecutionError("head engine returned an invalid score vector")
        probabilities = self.native.questions.label_softmax(
            [value / self.strategy.temperature for value in values[: read.counts[0]]]
        )
        _validate_distribution(probabilities)
        return [
            {
                "ordinal": read.ordinals[0],
                "probabilities": [probabilities[index] for index in read.orders[0]],
            }
        ]


@dataclass
class JointDecoder:
    engine: Any
    native: Any
    strategy: Any

    def prepare(self, request, strategy, state) -> list[PreparedHeadRead]:
        prompt, schema, orders = strategy.encode(list(request.questions), state)
        if len(prompt) > self.engine.model_config.max_model_len:
            raise NativeValidationError(
                "joint prompt exceeds the effective context limit"
            )
        counts = tuple(len(q.options) for q in request.questions)
        if schema["option_counts"] != list(counts) or len(orders) != len(counts):
            raise NativeExecutionError("native joint schema cardinality is invalid")
        return [
            PreparedHeadRead(
                tuple(prompt),
                self.native.pooling_params(
                    task="token_classify", extra_kwargs={"joint_schema": schema}
                ),
                tuple(range(len(counts))),
                tuple(
                    _validate_order(order, count)
                    for order, count in zip(orders, counts)
                ),
                counts,
            )
        ]

    def decode(self, read, values):
        if len(values) != sum(read.counts):
            raise NativeExecutionError("head engine returned an invalid score vector")
        outcomes, offset = [], 0
        for ordinal, order, size in zip(read.ordinals, read.orders, read.counts):
            segment = values[offset : offset + size]
            _validate_distribution(segment)
            # Native float32 probabilities need float64 unit mass at the core boundary.
            total = math.fsum(segment)
            probabilities = [0.0] * size
            for index, probability in zip(order, segment):
                probabilities[index] = probability / total
            outcomes.append({"ordinal": ordinal, "probabilities": probabilities})
            offset += size
        return outcomes


def _validate_distribution(values):
    if any(not 0 <= value <= 1 for value in values) or abs(sum(values) - 1) > 1e-6:
        raise NativeExecutionError(
            "head engine returned an invalid candidate distribution"
        )


class HeadExecutionAdapter(NativeAdapter):
    """Prepare native head reads and execute them on one existing pooling engine."""

    kind = "head"
    validate_runtime = staticmethod(_runtime_profile)

    def _initialize_profile(self):
        decoder = D3Decoder if self.spec.task == "classify" else JointDecoder
        self.decoder = decoder(self.engine, self.native, self.strategy)

    def _check_canary(self, prepared):
        if len(prepared.reads) != 1:
            raise NativeExecutionError("native health requires one pooling read")

    async def _prepare(self, wire: Any) -> PreparedHeadRequest:
        return await self.run_preflight(lambda: self._prepare_text(wire))

    def _prepare_text(self, wire: Any) -> PreparedHeadRequest:
        """Validate and tokenize the complete request before any engine admission."""
        if not isinstance(wire, dict) or set(wire) != {"decision_head"}:
            raise NativeExecutionError("invalid internal native head envelope")
        request = parse_native_request(
            wire["decision_head"], self.spec.name, self.native, self.strategy.limits()
        )
        strategy = copy.copy(self.strategy)
        strategy.tokenizer = BudgetTokenizer(
            self.preflight_tokenizer,
            request.max_tokenization_work,
            request.max_encoding_bytes,
        )
        state = strategy.render_state(request.state)
        try:
            reads = self.decoder.prepare(request, strategy, state)
        except self.native.questions.StructuredDecisionError as cause:
            raise NativeValidationError(
                "native head cannot prepare this request"
            ) from cause
        except UnicodeEncodeError as cause:
            raise NativeValidationError("head text must contain valid UTF-8") from cause
        if sum(len(read.prompt_ids) for read in reads) > request.max_input_tokens:
            raise NativeValidationError("head input token budget exceeded")
        return PreparedHeadRequest(request.cache_salt, tuple(reads))

    def _children(
        self,
        prepared: PreparedHeadRequest,
        context: Any,
        *,
        request_id: str,
        cancelled: Callable[[], bool],
        trace_headers: Mapping[str, str] | None = None,
    ) -> tuple[NativeChildBatch, ...]:
        async def child(index):
            return await self._read(
                prepared.reads[index],
                prepared.cache_salt,
                context,
                f"{request_id}-head-{index}",
                cancelled,
                trace_headers,
                lambda final: self._decode(prepared.reads[index], final),
            )

        return (NativeChildBatch(len(prepared.reads), child),)

    def _respond(self, prepared, results) -> dict[str, Any]:
        outcomes = [outcome for result in results for outcome in result[0]]
        return {
            "protocol_version": 1,
            "profile": self.spec.name,
            "outcomes": sorted(outcomes, key=lambda outcome: outcome["ordinal"]),
            "usage": aggregate_usage([result[1] for result in results]),
        }

    async def _read(
        self,
        read,
        cache_salt,
        context,
        request_id,
        cancelled,
        trace_headers,
        validate,
        *,
        admission=None,
    ):
        return await self.read_child(
            lambda: self.engine.encode(
                prompt=self.native.tokens_input(
                    list(read.prompt_ids), cache_salt=cache_salt
                ),
                pooling_params=read.pooling_params,
                request_id=request_id,
                trace_headers=trace_headers,
                priority=0,
            ),
            request_id,
            context,
            cancelled,
            validate,
            admission=admission,
        )

    async def verify_liveness(self, read, context):
        """Check engine acceptance without interpreting decision scores or usage."""
        read = replace(read, pooling_params=copy.deepcopy(read.pooling_params))

        def cancelled():
            return context.is_stopped() or context.is_killed()

        operation = self._read(
            read,
            NATIVE_HEALTH_CACHE_SALT,
            context,
            f"{context.id()}-head-0",
            cancelled,
            context.trace_headers(),
            lambda final: None,
            admission=nullcontext(),
        )
        await translate_native_errors(
            run_with_context_cancellation(operation, context, cancelled),
            self.native,
            self.on_engine_dead,
        )

    def _decode(self, read, final):
        if (
            not isinstance(final.prompt_token_ids, list)
            or any(type(token) is not int for token in final.prompt_token_ids)
            or tuple(final.prompt_token_ids) != read.prompt_ids
        ):
            raise NativeExecutionError("head engine changed the prepared prompt tokens")
        count = len(final.prompt_token_ids)
        cached = measured_cache_tokens(getattr(final, "num_cached_tokens", None), count)
        data = final.outputs.data
        if data.ndim != 1:
            raise NativeExecutionError("head engine returned a non-vector output")
        values = data.tolist()
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise NativeExecutionError("head engine returned an invalid score vector")
        return self.decoder.decode(read, values), NativeUsage(
            input_tokens=count, output_tokens=0, cached_tokens=cached
        )
