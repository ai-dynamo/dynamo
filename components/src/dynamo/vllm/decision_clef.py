# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text-only Clef Flash joint-head execution on the existing pooling worker."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
from dataclasses import dataclass
from importlib import import_module
from types import SimpleNamespace
from typing import Any

from dynamo.common.constants import DisaggregationMode
from dynamo.health_check import HEALTH_CHECK_KEY, HealthCheckPayload
from dynamo.llm import ModelInput, WorkerType
from dynamo.llm.exceptions import HttpError, InvalidArgument, WorkerOverloaded
from dynamo.runtime import Context

from .decision_cache_provenance import CACHE_SOURCE_BLOBS, verify_installed_sources
from .decision_execution import _released_runtime

logger = logging.getLogger(__name__)
CLEF_FLASH_PROFILE = "vllm_clef_flash_joint_decision_v1"
DECISION_METADATA = "dynamo.internal.decision"
_SOURCE_BLOBS = (
    (
        "vllm/entrypoints/openai/decisions/strategies.py",
        "2159657dcbff622079875452129060f3887de124",
    ),
    (
        "vllm/entrypoints/openai/decisions/question_types.py",
        "7ff319edc0c274ef355b2180ff26dbf17f7fe8af",
    ),
    (
        "vllm/model_executor/models/clef_joint_schema.py",
        "678c354387ce6ca9d0c399057e9a4f6ad744806f",
    ),
    ("vllm/pooling_params.py", "7d9a022c5ae05126561c4131d7c564c1cb8d4806"),
    ("vllm/renderers/online_renderer.py", "941d173bd1d5cb2ad2c6e3f07f8d052c54770739"),
    *CACHE_SOURCE_BLOBS,
)


class ClefHealthCheckPayload(HealthCheckPayload):
    def __init__(self, model: str):
        self.default_payload = {"native_pooling_health": True, "model": model}
        super().__init__()

    def to_dict(self):
        return {**super().to_dict(), HEALTH_CHECK_KEY: True}


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return _json(value)
    raise InvalidArgument("decision text must be a string, object or array")


def _integer(value, name, maximum):
    if type(value) is not int or not 1 <= value <= maximum:
        raise InvalidArgument(f"invalid native decision {name}")
    return value


class _BudgetTokenizer:
    def __init__(self, tokenizer, tokens, byte_budget):
        self.tokenizer, self.tokens, self.bytes = tokenizer, tokens, byte_budget

    def encode(self, text, **kwargs):
        self.bytes -= len(text.encode("utf-8"))
        if self.bytes < 0:
            raise InvalidArgument("native encoding byte budget exceeded")
        output = self.tokenizer.encode(text, **kwargs)
        self.tokens -= len(output)
        if self.tokens < 0:
            raise InvalidArgument("native tokenization work budget exceeded")
        return output


@dataclass(frozen=True)
class _Prepared:
    prompt_ids: tuple[int, ...]
    schema: dict
    orders: tuple[tuple[int, ...], ...]
    cache_salt: str
    questions: tuple[Any, ...]


class ClefDecisionExecutor:
    def __init__(self, handler, native, strategy, *, max_concurrency):
        self.handler, self.native, self.strategy = handler, native, strategy
        self._admission = asyncio.Semaphore(max_concurrency)
        self._preflight = asyncio.Semaphore(max_concurrency)

    def _question(self, raw, ordinal):
        if (
            not isinstance(raw, dict)
            or type(raw.get("ordinal")) is not int
            or raw["ordinal"] != ordinal
        ):
            raise InvalidArgument("native questions must be ordered by ordinal")
        kinds = {"Choice": "choice", "Predicate": "noul", "Score": "score"}
        kind = raw.get("kind")
        if kind not in kinds:
            raise InvalidArgument("unsupported native decision question kind")
        candidates = raw.get("candidates")
        if not isinstance(candidates, list) or not 2 <= len(candidates) <= 26:
            raise InvalidArgument("native decisions require 2 to 26 candidates")
        options, seen = [], set()
        typed = any(
            isinstance(item, dict) and type(item.get("value")) is bool
            for item in candidates
        )
        for index, item in enumerate(candidates):
            if not isinstance(item, dict):
                raise InvalidArgument("invalid native candidate")
            value = item.get("value")
            if type(value) not in (str, bool) or (type(value), value) in seen:
                raise InvalidArgument(
                    "candidate values must be distinct strings or booleans"
                )
            seen.add((type(value), value))
            description = (
                _text(item["description"])
                if item.get("description") is not None
                else None
            )
            name = _json(value) if typed else value
            if kind == "Score":
                if value != str(index) or len(candidates) > 10:
                    raise InvalidArgument("score candidates must be zero-based levels")
                label = item.get("label")
                if label is not None and not isinstance(label, str):
                    raise InvalidArgument("score labels must be strings")
                name = label if label is not None else description or ""
                if label is not None and description:
                    name = f"{label} - {description}"
                description = None
            options.append(self.native.questions.Option(name, description))
        if kind == "Predicate" and [item["value"] for item in candidates] != [
            True,
            False,
        ]:
            raise InvalidArgument("predicate candidates must be true then false")
        return self.native.questions.Question(
            id=f"question-{ordinal}",
            type=self.native.questions.get_question_type(kinds[kind]),
            instructions=_text(raw.get("instructions")),
            options=tuple(options),
            labels=(),
        )

    def prepare(self, wire):
        if not isinstance(wire, dict) or set(wire) != {"decision_head"}:
            raise HttpError(502, "invalid native decision envelope")
        raw = wire["decision_head"]
        required = {
            "protocol_version",
            "profile",
            "model",
            "input",
            "questions",
            "temperature",
            "cache_salt",
            "max_input_tokens",
            "max_tokenization_work",
            "max_encoding_bytes",
        }
        if not isinstance(raw, dict) or set(raw) != required:
            raise HttpError(502, "invalid native decision fields")
        if (
            type(raw["protocol_version"]) is not int
            or raw["protocol_version"] != 1
            or raw["profile"] != CLEF_FLASH_PROFILE
        ):
            raise HttpError(502, "unsupported native decision protocol")
        if type(raw["temperature"]) not in (int, float) or raw["temperature"] != 1:
            raise InvalidArgument("Clef decisions require temperature 1")
        if (
            not isinstance(raw["model"], str)
            or not raw["model"]
            or not isinstance(raw["cache_salt"], str)
            or not raw["cache_salt"]
        ):
            raise HttpError(502, "native model and cache salt must be nonempty strings")
        input_limit = _integer(raw["max_input_tokens"], "input limit", 1_048_576)
        work_limit = _integer(
            raw["max_tokenization_work"], "tokenization limit", 1_048_576
        )
        byte_limit = _integer(raw["max_encoding_bytes"], "byte limit", 16 * 1024 * 1024)
        try:
            if len(_json(wire).encode("utf-8")) > byte_limit:
                raise InvalidArgument("native decision wire exceeds byte budget")
            rows = raw["questions"]
            if not isinstance(rows, list) or not 1 <= len(rows) <= 64:
                raise InvalidArgument("native decisions require 1 to 64 questions")
            questions = tuple(
                self._question(row, index) for index, row in enumerate(rows)
            )
            strategy = copy.copy(self.strategy)
            strategy.tokenizer = _BudgetTokenizer(
                strategy.tokenizer, work_limit, byte_limit
            )
            prompt, schema, orders = strategy.encode(
                list(questions), strategy.render_state(raw["input"])
            )
        except (UnicodeError, TypeError, ValueError, RecursionError) as cause:
            raise InvalidArgument(
                "native decision text or schema is invalid"
            ) from cause
        if len(prompt) > min(
            input_limit, self.handler.engine_client.model_config.max_model_len
        ):
            raise InvalidArgument("native decision input-token budget exceeded")
        if schema["option_counts"] != [len(q.options) for q in questions] or len(
            orders
        ) != len(questions):
            raise HttpError(502, "native schema cardinality is invalid")
        for order, question in zip(orders, questions):
            if any(type(index) is not int for index in order) or sorted(order) != list(
                range(len(question.options))
            ):
                raise HttpError(502, "native schema option order is invalid")
        return _Prepared(
            tuple(prompt),
            schema,
            tuple(tuple(order) for order in orders),
            raw["cache_salt"],
            questions,
        )

    async def _prepare(self, wire):
        await self._preflight.acquire()
        task = asyncio.create_task(asyncio.to_thread(self.prepare, wire))

        # A cancelled coroutine cannot stop tokenization already running in a thread.
        def settled(result):
            self._preflight.release()
            if not result.cancelled():
                result.exception()

        task.add_done_callback(settled)
        return await asyncio.shield(task)

    def decode(self, prepared, output):
        if (
            output.error is not None
            or output.finished is not True
            or output.prompt_token_ids != list(prepared.prompt_ids)
        ):
            raise HttpError(502, "native engine returned invalid terminal metadata")
        cached = output.num_cached_tokens
        if type(cached) is not int or not 0 <= cached <= len(prepared.prompt_ids):
            raise HttpError(502, "native engine returned invalid cache accounting")
        data = output.outputs.data
        values = data.tolist()
        if (
            data.ndim != 1
            or len(values) != sum(prepared.schema["option_counts"])
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
        ):
            raise HttpError(502, "native engine returned invalid candidate scores")
        outcomes, offset = [], 0
        for ordinal, order in enumerate(prepared.orders):
            segment = values[offset : offset + len(order)]
            total = math.fsum(segment)
            if any(not 0 <= value <= 1 for value in segment) or abs(total - 1) > 1e-6:
                raise HttpError(
                    502, "native engine returned an invalid probability distribution"
                )
            probabilities = [0.0] * len(order)
            for index, probability in zip(order, segment):
                probabilities[index] = probability / total
            outcomes.append({"ordinal": ordinal, "probabilities": probabilities})
            offset += len(order)
        return {
            "protocol_version": 1,
            "profile": CLEF_FLASH_PROFILE,
            "outcomes": outcomes,
            "finish_reason": "stop",
            "usage": {
                "input_tokens": len(prepared.prompt_ids),
                "output_tokens": 0,
                "cached_tokens": cached,
            },
        }

    async def _execute(self, wire, context):
        async with self._admission:
            prepared = await self._prepare(wire)
            if context.is_stopped() or context.is_killed():
                raise asyncio.CancelledError
            request_id = f"{context.id()}-head-0"
            if len(request_id) > 256:
                raise InvalidArgument("native request ID is too long")
            completed = False
            iterator = None
            try:
                async with self.handler._abort_monitor(context, request_id):
                    result = None
                    iterator = self.handler.engine_client.encode(
                        prompt=self.native.tokens_input(
                            list(prepared.prompt_ids), cache_salt=prepared.cache_salt
                        ),
                        pooling_params=self.native.pooling_params(
                            task="token_classify",
                            extra_kwargs={"joint_schema": prepared.schema},
                        ),
                        request_id=request_id,
                        trace_headers=context.trace_headers(),
                        priority=0,
                    )
                    async for output in iterator:
                        if result is not None:
                            raise HttpError(
                                502, "native pooling returned multiple outputs"
                            )
                        result = self.decode(prepared, output)
                    if result is None:
                        raise HttpError(502, "native pooling produced no result")
                    completed = True
                    return result
            finally:
                await self._settle(iterator, request_id, completed)

    async def _settle(self, iterator, request_id, completed):
        operations = []
        if iterator is not None:
            operations.append(iterator.aclose())
        if not completed:
            operations.append(self.handler.engine_client.abort(request_id))
        if not operations:
            return
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*operations, return_exceptions=True), timeout=5
            )
            errors = [result for result in results if isinstance(result, BaseException)]
        except TimeoutError as cause:
            errors = [cause]
        if errors:
            if (
                any(
                    isinstance(cause, self.native.engine_dead_error) for cause in errors
                )
                and self.handler.shutdown_event is not None
            ):
                self.handler.shutdown_event.set()
            if completed:
                raise HttpError(
                    503, "native decision cleanup did not settle"
                ) from errors[0]
            logger.warning(
                "Native decision cleanup failed; retaining the request error"
            )

    async def execute(self, wire, context):
        if context.is_stopped() or context.is_killed():
            raise asyncio.CancelledError
        task = asyncio.create_task(self._execute(wire, context))
        signal = asyncio.ensure_future(context.async_killed_or_stopped())
        try:
            done, _ = await asyncio.wait(
                (task, signal), return_when=asyncio.FIRST_COMPLETED
            )
            if signal in done:
                await signal
                raise asyncio.CancelledError
            return await task
        except self.native.engine_dead_error:
            if self.handler.shutdown_event is not None:
                self.handler.shutdown_event.set()
            raise
        except self.native.overload_errors as cause:
            raise WorkerOverloaded("native decision capacity is exhausted") from cause
        except self.native.retryable_errors as cause:
            raise HttpError(
                503, "native decision engine is temporarily unavailable"
            ) from cause
        finally:
            for owned in (task, signal):
                if not owned.done():
                    owned.cancel()
            await asyncio.gather(task, signal, return_exceptions=True)

    async def health(self, context):
        wire = {
            "decision_head": {
                "protocol_version": 1,
                "profile": CLEF_FLASH_PROFILE,
                "model": "native-health",
                "input": "A short health check.",
                "temperature": 1,
                "cache_salt": "dynamo-clef-health-v1",
                "max_input_tokens": 4096,
                "max_tokenization_work": 16384,
                "max_encoding_bytes": 65536,
                "questions": [
                    {
                        "ordinal": 0,
                        "kind": "Choice",
                        "instructions": "Is the state short?",
                        "candidates": [{"value": "yes"}, {"value": "no"}],
                    }
                ],
            }
        }
        await self.execute(wire, context)
        return {"status": "ok"}


async def initialize_clef(handler, runtime, config):
    """Qualify a recognized Clef worker before publishing its decision capability."""
    if runtime.model_config.architecture != "ClefFlashForJointSchema":
        return None
    model = runtime.model_config
    scheduler = runtime.scheduler_config
    if not (
        _released_runtime()
        and config.disaggregation_mode == DisaggregationMode.AGGREGATED
        and config.enable_multimodal is False
        and config.enable_rl is False
        and config.route_to_encoder is False
        and config.custom_encoder_class is None
        and config.realtime is False
        and model.runner_type == "pooling"
        and model.is_encoder_decoder is False
        and model.is_diffusion is False
        and model.convert_type == "none"
        and model.model_impl in ("auto", "vllm")
        and not model.model_class_overrides
        and type(runtime.parallel_config.data_parallel_size) is int
        and runtime.parallel_config.data_parallel_size == 1
        and type(scheduler.max_num_seqs) is int
        and 1 <= scheduler.max_num_seqs <= 4
        and scheduler.async_scheduling is False
        and scheduler.enable_chunked_prefill is False
        and runtime.cache_config.enable_prefix_caching is False
        and scheduler.scheduler_cls in (None, "vllm.v1.core.sched.scheduler.Scheduler")
        and all(
            extension is None
            for extension in (
                runtime.lora_config,
                runtime.speculative_config,
                runtime.kv_transfer_config,
                runtime.ec_transfer_config,
            )
        )
        and verify_installed_sources(_SOURCE_BLOBS)
    ):
        raise RuntimeError(
            "Clef decisions require the qualified text-only vLLM 0.33 pooling configuration"
        )
    strategies = import_module("vllm.entrypoints.openai.decisions.strategies")
    renderer_type = import_module("vllm.renderers.online_renderer").OnlineRenderer
    exceptions = import_module("vllm.exceptions")
    engine_errors = import_module("vllm.v1.engine.exceptions")
    renderer = renderer_type(
        model_config=model,
        renderer=handler.engine_client.renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    context = strategies.ReadContext(
        engine_client=handler.engine_client,
        online_renderer=renderer,
        chat_template=None,
        chat_template_content_format="auto",
        default_chat_template_kwargs={},
    )
    native = SimpleNamespace(
        questions=import_module("vllm.entrypoints.openai.decisions.question_types"),
        pooling_params=import_module("vllm.pooling_params").PoolingParams,
        tokens_input=import_module("vllm.inputs").tokens_input,
        overload_errors=(
            exceptions.QueueOverflowError,
            exceptions.MaxQueuedTokensError,
        ),
        retryable_errors=(
            exceptions.RetryableRequestError,
            engine_errors.EngineGenerateError,
        ),
        engine_dead_error=engine_errors.EngineDeadError,
    )
    executor = ClefDecisionExecutor(
        handler,
        native,
        strategies.ClefFlashStrategy(context),
        max_concurrency=min(scheduler.max_num_seqs, 4),
    )
    supported = await asyncio.wait_for(
        handler.engine_client.get_supported_tasks(), timeout=30
    )
    if "token_classify" not in supported:
        raise RuntimeError("Clef engine does not support native joint-head pooling")
    await asyncio.wait_for(executor.health(Context()), timeout=30)
    return executor


def publish_clef_capability(
    runtime_config, enabled, model_input, model_type, worker_type, needs
):
    if (
        enabled is True
        and model_input == ModelInput.Text
        and model_type.supports_classify()
        and model_type.supports_pooling()
        and worker_type == WorkerType.Aggregated
        and not any(needs or ())
    ):
        runtime_config.set_engine_specific(CLEF_FLASH_PROFILE, "true")
        runtime_config.set_engine_specific("vllm_native_decision_max_candidates", "26")
