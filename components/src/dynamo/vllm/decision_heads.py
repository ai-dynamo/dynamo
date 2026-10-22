# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional native D3/Clef execution and fail-closed readiness advertisement."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from dynamo.common.constants import DisaggregationMode

from .decision_lifecycle import run_with_context_cancellation
from .decision_runtime import is_native_decision_model, translate_native_overload

PINNED_VLLM_COMMIT = "9da5f2d3103df460dfca53df4ab4884def5c935f"
D3_PROFILE = "vllm_d3_decision_v1"
CLEF_FLASH_PROFILE = "vllm_clef_flash_joint_decision_v1"
CLEF_OMNI_PROFILE = "vllm_clef_omni_joint_decision_v1"
_PROFILES = (
    (D3_PROFILE, "D3ForSequenceClassification", "classify", "D3Strategy"),
    (
        CLEF_FLASH_PROFILE,
        "ClefFlashForJointSchema",
        "token_classify",
        "ClefFlashStrategy",
    ),
    (
        CLEF_OMNI_PROFILE,
        "ClefOmniForJointSchema",
        "token_classify",
        "JointSchemaStrategy",
    ),
)
_PINNED_SOURCES = (
    (
        "vllm.entrypoints.openai.decisions.strategies",
        "0b56a7a2b778dd64eb752bb7db8e4b072822e45d6c83af534adc79cfc04c741e",
    ),
    (
        "vllm.entrypoints.openai.decisions.question_types",
        "1008066e550fd64ce0a0b973c0f563c38c07e01850898cf82e924bc8e7098491",
    ),
    (
        "vllm.outputs",
        "d6fc34a8766a8fa5364713e6c2f995d9f6c393e50de391370c78c21529b98893",
    ),
    (
        "vllm.v1.engine.async_llm",
        "1ae2a81f39dc56f7136ec9aa19aedc4dd2d3f39601228f00a43d2d18687009a9",
    ),
    (
        "vllm.model_executor.models.d3",
        "c632494024c2ce5b4b46ceae209f08ad38bbcc23d358105a6bc730e740f31cc4",
    ),
    (
        "vllm.model_executor.models.clef_joint_schema",
        "592c4624ef8cbc7f42e93d457e95c0e91834032e3b84da9192f249e688656824",
    ),
    (
        "vllm.exceptions",
        "b7289dc5d796a40f7a40cd74060035475b54c8e65b57c467d644509b50689c71",
    ),
)


class HeadRuntimeError(RuntimeError):
    """The optional installed runtime cannot implement this head profile."""


class HeadValidationError(ValueError):
    """The internal request is invalid or unsupported before GPU dispatch."""


class HeadExecutionError(RuntimeError):
    """Native execution did not produce complete, trustworthy head evidence."""


def select_decision_head_profile(model_config: Any, config: Any) -> str | None:
    """Select a recognized pooling head only on an eligible worker path.

    Native API, source, task, cache, and parallelism validation happens in
    ``HeadExecutionAdapter.create`` before any profile can be advertised.
    """
    if model_config is None:
        return None
    profile = next(
        (
            name
            for name, architecture, _, _ in _PROFILES
            if model_config.architecture == architecture
        ),
        None,
    )
    if profile is None:
        return None
    if model_config.runner_type != "pooling" or model_config.is_encoder_decoder:
        raise HeadRuntimeError("native head requires a decoder-only pooling model")
    if (
        config.disaggregation_mode != DisaggregationMode.AGGREGATED
        or config.enable_rl
        or config.route_to_encoder
        or config.custom_encoder_class is not None
        or config.enable_multimodal
        or config.realtime
    ):
        raise HeadRuntimeError("worker configuration is incompatible with native heads")
    return profile


def publish_decision_head_capability(
    runtime_config: Any, ready_profile: str | None
) -> bool:
    """Publish only the profile returned by a successfully initialized handler."""
    if ready_profile not in (profile for profile, _, _, _ in _PROFILES):
        return False
    runtime_config.set_engine_specific(ready_profile, json.dumps(True))
    return True


@dataclass(frozen=True)
class _NativeApi:
    strategies: Any
    questions: Any
    pooling_params: Any
    tokens_input: Any
    online_renderer: Any
    overload_errors: tuple[type[Exception], ...]
    http_error: type[Exception]


def _load_native(source_commit: str) -> _NativeApi:
    if source_commit != PINNED_VLLM_COMMIT:
        raise HeadRuntimeError("head execution requires the pinned vLLM source")
    try:
        modules = tuple(import_module(name) for name, _ in _PINNED_SOURCES)
        for module, (_, expected) in zip(modules, _PINNED_SOURCES):
            if (
                hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                != expected
            ):
                raise HeadRuntimeError(
                    "installed head source differs from the pinned runtime"
                )
        return _NativeApi(
            modules[0],
            modules[1],
            import_module("vllm.pooling_params").PoolingParams,
            import_module("vllm.inputs").tokens_input,
            import_module("vllm.renderers.online_renderer").OnlineRenderer,
            (modules[6].QueueOverflowError, modules[6].MaxQueuedTokensError),
            import_module("dynamo.llm.exceptions").HttpError,
        )
    except (ImportError, OSError, AttributeError, TypeError) as cause:
        raise HeadRuntimeError("pinned optional head runtime is unavailable") from cause


def _positive_integer(value: Any, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise HeadValidationError(f"invalid {name}")
    return value


def _json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as cause:
        raise HeadValidationError("head input must contain valid JSON") from cause


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return _json(value)
    raise HeadValidationError("question text must be a string, object, or array")


def _object(value: Any, keys: set[str], required: set[str]) -> dict:
    if (
        not isinstance(value, dict)
        or not required <= value.keys()
        or value.keys() - keys
    ):
        raise HeadValidationError("invalid internal head request fields")
    return value


@dataclass(frozen=True)
class _Request:
    profile: str
    state: Any
    questions: tuple[Any, ...]
    cache_salt: str
    max_input_tokens: int
    max_tokenization_work: int
    max_encoding_bytes: int


@dataclass(frozen=True)
class PreparedHeadRead:
    prompt_ids: tuple[int, ...]
    pooling_params: Any
    ordinals: tuple[int, ...]
    orders: tuple[tuple[int, ...], ...]
    counts: tuple[int, ...]


@dataclass(frozen=True)
class PreparedHeadRequest:
    profile: str
    cache_salt: str
    reads: tuple[PreparedHeadRead, ...]


class _BudgetTokenizer:
    def __init__(self, tokenizer: Any, tokens: int, encoding_bytes: int):
        self.tokenizer = tokenizer
        self.tokens = tokens
        self.encoding_bytes = encoding_bytes

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        self.encoding_bytes -= len(text.encode("utf-8"))
        if self.encoding_bytes < 0:
            raise HeadValidationError("head encoding byte budget exceeded")
        encoded = self.tokenizer.encode(text, **kwargs)
        self.tokens -= len(encoded)
        if self.tokens < 0:
            raise HeadValidationError("head tokenization work budget exceeded")
        return encoded

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        return self.tokenizer.apply_chat_template(*args, **kwargs)


def _candidate_options(raw: dict, native: Any) -> tuple[Any, ...]:
    candidates = raw["candidates"]
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 255:
        raise HeadValidationError("invalid head candidate count")
    typed = any(
        isinstance(c, dict) and type(c.get("value")) is bool for c in candidates
    )
    options, seen = [], set()
    for index, candidate in enumerate(candidates):
        item = _object(candidate, {"value", "description", "label"}, {"value"})
        value = item["value"]
        if type(value) not in (str, bool) or (type(value), value) in seen:
            raise HeadValidationError(
                "head candidates must have unique string/bool values"
            )
        seen.add((type(value), value))
        label = item.get("label")
        if label is not None and not isinstance(label, str):
            raise HeadValidationError("invalid head score label")
        description = (
            _text(item.get("description"))
            if item.get("description") is not None
            else None
        )
        name = _json(value) if typed else value
        if raw["kind"] == "Score":
            if value != str(index):
                raise HeadValidationError("score values must be canonical ordinals")
            name = label if label is not None else description or ""
            if label is not None and description:
                name = f"{label} - {description}"
            description = None
        options.append(native.questions.Option(name, description))
    if raw["kind"] == "Predicate" and [c["value"] for c in candidates] != [True, False]:
        raise HeadValidationError("predicate candidates must be true then false")
    if raw["kind"] == "Score" and not 2 <= len(options) <= 10:
        raise HeadValidationError("head scores require two to ten levels")
    return tuple(options)


def _question(raw: Any, ordinal: int, native: Any, max_options: int) -> Any:
    item = _object(
        raw,
        {"ordinal", "id", "kind", "instructions", "candidates"},
        {"ordinal", "kind", "candidates"},
    )
    if type(item["ordinal"]) is not int or item["ordinal"] != ordinal:
        raise HeadValidationError(
            "head question ordinals must be contiguous and ordered"
        )
    if item.get("id") is not None and not isinstance(item["id"], str):
        raise HeadValidationError("invalid head question ID")
    kinds = {"Predicate": "noul", "Choice": "choice", "Score": "score"}
    if not isinstance(item["kind"], str) or item["kind"] not in kinds:
        raise HeadValidationError("unsupported head question kind")
    options = _candidate_options(item, native)
    if len(options) > max_options:
        raise HeadValidationError("question exceeds the native head option limit")
    return native.questions.Question(
        id=f"question-{ordinal}",
        type=native.questions.get_question_type(kinds[item["kind"]]),
        instructions=_text(item.get("instructions")),
        options=options,
        labels=(),
    )


def _parse_request(wire: Any, profile: str, native: Any, limits: Any) -> _Request:
    envelope = _object(wire, {"decision_head"}, {"decision_head"})
    fields = {
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
    raw = _object(envelope["decision_head"], fields, fields)
    if type(raw["protocol_version"]) is not int or raw["protocol_version"] != 1:
        raise HeadValidationError("unsupported head protocol version")
    if raw["profile"] != profile:
        raise HeadValidationError("head profile differs from the selected worker")
    if not isinstance(raw["model"], str) or not raw["model"]:
        raise HeadValidationError("head model must be a nonempty string")
    if type(raw["temperature"]) not in (int, float) or raw["temperature"] != 1:
        raise HeadValidationError("native head profiles require temperature one")
    if not isinstance(raw["cache_salt"], str) or not 1 <= len(raw["cache_salt"]) <= 256:
        raise HeadValidationError("head cache salt must be bounded and nonempty")
    budgets = tuple(
        _positive_integer(raw[name], name, maximum)
        for name, maximum in (
            ("max_input_tokens", 1_048_576),
            ("max_tokenization_work", 1_048_576),
            ("max_encoding_bytes", 16 * 1024 * 1024),
        )
    )
    try:
        if len(_json(raw).encode("utf-8")) > budgets[2]:
            raise HeadValidationError("head encoding byte budget exceeded")
    except UnicodeEncodeError as cause:
        raise HeadValidationError("head input must contain valid UTF-8") from cause
    questions = raw["questions"]
    if not isinstance(questions, list) or not 1 <= len(questions) <= min(
        limits.max_questions, 64
    ):
        raise HeadValidationError("invalid head question count")
    parsed = tuple(
        _question(q, i, native, limits.max_options) for i, q in enumerate(questions)
    )
    return _Request(profile, raw["input"], parsed, raw["cache_salt"], *budgets)


def _runtime_profile(engine: Any, profile: str, tasks: tuple[str, ...]) -> str:
    selected = next((entry for entry in _PROFILES if entry[0] == profile), None)
    if selected is None:
        raise HeadRuntimeError("unsupported head execution profile")
    _, architecture, task, strategy_name = selected
    config = engine.vllm_config
    if (
        engine.model_config.architecture != architecture
        or not is_native_decision_model(engine.model_config, diffusion=False)
        or engine.model_config.runner_type != "pooling"
        or task not in tasks
        or config.parallel_config.data_parallel_size != 1
        or type(engine.model_config.max_model_len) is not int
        or engine.model_config.max_model_len < 1
        or any(
            value is not None
            for value in (
                config.lora_config,
                config.speculative_config,
                config.kv_transfer_config,
                config.ec_transfer_config,
            )
        )
        or config.cache_config.enable_prefix_caching is not False
        or (
            profile == D3_PROFILE
            and config.scheduler_config.enable_chunked_prefill is not False
        )
    ):
        raise HeadRuntimeError(
            "effective runtime is incompatible with the head profile"
        )
    return strategy_name


def _validate_order(order: Any, count: int) -> tuple[int, ...]:
    if any(type(index) is not int for index in order) or sorted(order) != list(
        range(count)
    ):
        raise HeadExecutionError(
            "native head preparation returned an invalid option order"
        )
    return tuple(order)


def _softmax(values: list[float], temperature: float) -> list[float]:
    maximum = max(values)
    weights = [math.exp((value - maximum) / temperature) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


class HeadExecutionAdapter:
    """Prepare native head reads and execute them on one existing pooling engine."""

    def __init__(self, engine, profile, native, strategy, abort_monitor, concurrency):
        self.engine = engine
        self.profile = profile
        self.native = native
        self.strategy = strategy
        self.abort_monitor = abort_monitor
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)

    @classmethod
    async def create(
        cls,
        engine: Any,
        profile: str,
        *,
        source_commit: str,
        abort_monitor: Callable[[Any, str], AbstractAsyncContextManager[Any]],
        max_concurrency: int = 4,
    ) -> HeadExecutionAdapter:
        """Fail closed unless the runtime implements the exact pinned head APIs."""
        if source_commit != PINNED_VLLM_COMMIT:
            raise HeadRuntimeError("head execution requires the pinned vLLM source")
        native = _load_native(source_commit)
        strategy_name = _runtime_profile(
            engine, profile, tuple(await engine.get_supported_tasks())
        )
        if type(max_concurrency) is not int or not 1 <= max_concurrency <= 64:
            raise HeadRuntimeError("head concurrency must be between one and 64")
        concurrency = min(
            max_concurrency, engine.vllm_config.scheduler_config.max_num_seqs
        )
        if type(concurrency) is not int or concurrency < 1:
            raise HeadRuntimeError("head scheduler has no usable capacity")
        renderer = native.online_renderer(
            model_config=engine.model_config,
            renderer=engine.renderer,
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
        )
        context = native.strategies.ReadContext(
            engine_client=engine,
            online_renderer=renderer,
            chat_template=None,
            chat_template_content_format="auto",
            default_chat_template_kwargs={},
        )
        strategy = getattr(native.strategies, strategy_name)(context)
        if profile == D3_PROFILE and (
            not isinstance(strategy.codes, list)
            or not strategy.codes
            or len(strategy.codes) > 255
            or any(not isinstance(code, str) or not code for code in strategy.codes)
            or len(set(strategy.codes)) != len(strategy.codes)
            or type(strategy.temperature) not in (int, float)
            or not math.isfinite(strategy.temperature)
            or strategy.temperature <= 0
        ):
            raise HeadRuntimeError(
                "D3 requires valid answer codes and checkpoint temperature"
            )
        return cls(engine, profile, native, strategy, abort_monitor, concurrency)

    def prepare(self, wire: Any) -> PreparedHeadRequest:
        """Validate and tokenize the complete request before any engine admission."""
        request = _parse_request(
            wire, self.profile, self.native, self.strategy.limits()
        )
        strategy = copy.copy(self.strategy)
        strategy.tokenizer = _BudgetTokenizer(
            self.strategy.tokenizer,
            request.max_tokenization_work,
            request.max_encoding_bytes,
        )
        state = strategy.render_state(request.state)
        try:
            reads = (
                self._prepare_d3(request, strategy, state)
                if self.profile == D3_PROFILE
                else self._prepare_joint(request, strategy, state)
            )
        except self.native.questions.StructuredDecisionError as cause:
            raise HeadValidationError(
                "native head cannot prepare this request"
            ) from cause
        except UnicodeEncodeError as cause:
            raise HeadValidationError("head text must contain valid UTF-8") from cause
        if sum(len(read.prompt_ids) for read in reads) > request.max_input_tokens:
            raise HeadValidationError("head input token budget exceeded")
        return PreparedHeadRequest(self.profile, request.cache_salt, tuple(reads))

    def _prepare_d3(self, request, strategy, state) -> list[PreparedHeadRead]:
        reads = []
        for ordinal, question in enumerate(request.questions):
            prompt = strategy._tokens(strategy._prompt(state, question, []))
            if len(prompt) > self.engine.model_config.max_model_len:
                raise HeadValidationError(
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

    def _prepare_joint(self, request, strategy, state) -> list[PreparedHeadRead]:
        prompt, schema, orders = strategy.encode(list(request.questions), state)
        if len(prompt) > self.engine.model_config.max_model_len:
            raise HeadValidationError(
                "joint prompt exceeds the effective context limit"
            )
        counts = tuple(len(q.options) for q in request.questions)
        if schema["option_counts"] != list(counts) or len(orders) != len(counts):
            raise HeadExecutionError("native joint schema cardinality is invalid")
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

    async def execute(
        self,
        wire: Any,
        context: Any,
        *,
        request_id: str,
        cancelled: Callable[[], bool],
        trace_headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Execute a whole request, including cancellation while awaiting capacity."""
        return await translate_native_overload(
            run_with_context_cancellation(
                self._execute(
                    wire,
                    context,
                    request_id=request_id,
                    cancelled=cancelled,
                    trace_headers=trace_headers,
                ),
                context,
                cancelled,
            ),
            self.native,
        )

    async def _execute(
        self,
        wire: Any,
        context: Any,
        *,
        request_id: str,
        cancelled: Callable[[], bool],
        trace_headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Return ordinal-tagged head distributions and actual per-read usage."""
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 256:
            raise HeadValidationError("invalid head engine request ID")
        if cancelled():
            raise asyncio.CancelledError
        prepared = self.prepare(wire)
        if cancelled():
            raise asyncio.CancelledError
        self.engine.check_admission(len(prepared.reads))
        results: list[Any] = [None] * len(prepared.reads)
        indices = iter(range(len(prepared.reads)))

        async def worker():
            for index in indices:
                async with self.semaphore:
                    if cancelled():
                        raise asyncio.CancelledError
                    results[index] = await self._read(
                        prepared.reads[index],
                        prepared.cache_salt,
                        context,
                        f"{request_id}-head-{index}",
                        trace_headers,
                    )

        tasks = [
            asyncio.create_task(worker())
            for _ in range(min(self.concurrency, len(prepared.reads)))
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        if cancelled():
            raise asyncio.CancelledError
        outcomes = [outcome for result in results for outcome in result[0]]
        cached = [result[2] for result in results]
        return {
            "protocol_version": 1,
            "profile": self.profile,
            "outcomes": sorted(outcomes, key=lambda outcome: outcome["ordinal"]),
            "usage": {
                "input_tokens": sum(result[1] for result in results),
                "output_tokens": 0,
                "cached_tokens": sum(cached)
                if all(value is not None for value in cached)
                else None,
            },
        }

    async def _read(self, read, cache_salt, context, request_id, trace_headers):
        generator = self.engine.encode(
            prompt=self.native.tokens_input(
                list(read.prompt_ids), cache_salt=cache_salt
            ),
            pooling_params=read.pooling_params,
            request_id=request_id,
            trace_headers=trace_headers,
            priority=0,
        )
        succeeded = False
        try:
            async with self.abort_monitor(context, request_id):
                final = await self._terminal(generator, request_id)
                result = self._decode(read, final)
            succeeded = True
            return result
        finally:
            try:
                await generator.aclose()
            finally:
                if not succeeded:
                    await self.engine.abort(request_id)

    @staticmethod
    async def _terminal(generator: AsyncIterator, request_id: str) -> Any:
        terminal = None
        async for output in generator:
            if terminal is not None:
                raise HeadExecutionError("head engine returned data after its terminal")
            if output.request_id != request_id or output.error is not None:
                raise HeadExecutionError(
                    "head engine failed or returned a different request"
                )
            if output.finished is True:
                terminal = output
            elif output.finished is not False:
                raise HeadExecutionError(
                    "head engine returned an invalid terminal marker"
                )
        if terminal is None:
            raise HeadExecutionError("head engine did not return a finished result")
        return terminal

    def _decode(self, read, final):
        if (
            not isinstance(final.prompt_token_ids, list)
            or any(type(token) is not int for token in final.prompt_token_ids)
            or tuple(final.prompt_token_ids) != read.prompt_ids
        ):
            raise HeadExecutionError("head engine changed the prepared prompt tokens")
        count = len(final.prompt_token_ids)
        cached = final.num_cached_tokens
        if cached is not None and (type(cached) is not int or not 0 <= cached <= count):
            raise HeadExecutionError(
                "head engine returned invalid cached-token accounting"
            )
        data = final.outputs.data
        if data.ndim != 1:
            raise HeadExecutionError("head engine returned a non-vector output")
        values = data.tolist()
        width = (
            len(self.strategy.codes) if self.profile == D3_PROFILE else sum(read.counts)
        )
        if len(values) != width or any(
            type(v) not in (int, float) or not math.isfinite(v) for v in values
        ):
            raise HeadExecutionError("head engine returned an invalid score vector")
        if self.profile == D3_PROFILE:
            values = _softmax(values[: read.counts[0]], self.strategy.temperature)
        outcomes, offset = [], 0
        for ordinal, order, size in zip(read.ordinals, read.orders, read.counts):
            segment = values[offset : offset + size]
            total = sum(segment)
            if (
                any(not 0 <= probability <= 1 for probability in segment)
                or abs(total - 1) > 1e-6
            ):
                raise HeadExecutionError(
                    "head engine returned an invalid candidate distribution"
                )
            probabilities = (
                [segment[index] / total for index in order]
                if self.profile == D3_PROFILE
                else self._joint_order(segment, order, total)
            )
            outcomes.append({"ordinal": ordinal, "probabilities": probabilities})
            offset += size
        return outcomes, count, cached

    @staticmethod
    def _joint_order(segment, order, total):
        probabilities = [0.0] * len(order)
        for index, probability in zip(order, segment):
            probabilities[index] = probability / total
        return probabilities
