# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-qualified native Liquid and DiffusionGemma whole-request reads.

Preparation and numerical reduction belong to pinned native strategies. Actual
engine children have Dynamo-owned cancellation, concurrency, and usage accounting.
No routes or model capabilities are registered by this module.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import math
from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .decision_heads import PINNED_VLLM_COMMIT, HeadValidationError, _parse_request
from .decision_lifecycle import run_with_context_cancellation
from .decision_runtime import is_native_decision_model, translate_native_overload

LIQUID_PROFILE = "vllm_liquid_decision_v1"
DIFFUSION_PROFILE = "vllm_diffusion_canvas_decision_v1"
_PROFILES = {
    LIQUID_PROFILE: ("Lfm2VlForConditionalGeneration", "LiquidStrategy"),
    DIFFUSION_PROFILE: (
        "DiffusionGemmaForBlockDiffusion",
        "DiffusionGemmaCanvasStrategy",
    ),
}
# Git blob identities from the named source commit; compare the complete source
# including Git's blob header, never an installed package's claimed version.
_PINNED_SOURCE_BLOBS = (
    ("vllm/exceptions.py", "85863c2fbf1db06e3d3c3119ac0f2ad7526faf15"),
    ("vllm/inputs/engine.py", "aa8c9054335e75242dc2d89b80fc364e590f0de5"),
    ("vllm/config/diffusion.py", "6f59c40a8366910b41630ad62bfe824def738364"),
    (
        "vllm/entrypoints/generate/label_reads.py",
        "c677ebc7903efc0a931598e0282ed7a94a35fe11",
    ),
    (
        "vllm/entrypoints/openai/decisions/question_types.py",
        "0672fa1709970bf623739534e24f6bc9f3039396",
    ),
    (
        "vllm/entrypoints/openai/decisions/strategies.py",
        "cff24cf81b3e3b9053adc294b73c2c93e1bc5bd2",
    ),
    (
        "vllm/model_executor/models/diffusion_gemma.py",
        "541ae4e2f1bad325fcde3a727f38a58dad2f26ef",
    ),
    (
        "vllm/model_executor/models/diffusion_gemma_sampler.py",
        "9b1523379197e2d460e935812af734bb372dd2a6",
    ),
    (
        "vllm/model_executor/models/lfm2_vl.py",
        "dd58707b855ff7b28327c072b82b4623c323aa11",
    ),
    ("vllm/outputs.py", "2149cf7df5e5447270d08e0f1d0561899647de04"),
    ("vllm/renderers/inputs/preprocess.py", "9daf38fdbd3bd54a9cb32175329af9ca5bc07099"),
    ("vllm/renderers/online_renderer.py", "65771211d0a5eca6ee6bd4c8e20cb0096b4c6ef0"),
    ("vllm/sampling_params.py", "a0ff636155660d81999899ee0a41b93f097968d1"),
    ("vllm/utils/async_utils.py", "b2c0d57dc921bd71eda85b5c7472dfc03c6cdd3e"),
    ("vllm/utils/diffusion.py", "f3018c18e7bdeaee0850c96773f0b6638950d3d2"),
    (
        "vllm/v1/core/sched/diffusion_scheduler.py",
        "9803307037670ea0be87d286111e732ed4bedf98",
    ),
    ("vllm/v1/engine/async_llm.py", "a5b72fa3fb64f74f9d02aeccc83b10be66d38ae1"),
    ("vllm/v1/worker/gpu_model_runner.py", "d1271dc7f61d1a1550011e5100c9fbdaa76cd378"),
)


class ReadRuntimeError(RuntimeError):
    """The optional effective runtime cannot implement this native profile."""


class ReadValidationError(ValueError):
    """The entire canonical request failed validation before engine admission."""


class ReadExecutionError(RuntimeError):
    """Native execution did not produce complete, trustworthy evidence."""


def _load_native(source_commit: str) -> Any:
    if source_commit != PINNED_VLLM_COMMIT:
        raise ReadRuntimeError("native reads require the pinned vLLM source")
    try:
        root = Path(import_module("vllm").__file__).resolve().parent.parent
        for relative, expected in _PINNED_SOURCE_BLOBS:
            data = (root / relative).read_bytes()
            digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            if digest != expected:
                raise ReadRuntimeError(
                    "installed native read source differs from pinned runtime"
                )
        return SimpleNamespace(
            strategies=import_module("vllm.entrypoints.openai.decisions.strategies"),
            questions=import_module("vllm.entrypoints.openai.decisions.question_types"),
            validation_error=import_module("vllm.exceptions").VLLMValidationError,
            overload_errors=(
                import_module("vllm.exceptions").QueueOverflowError,
                import_module("vllm.exceptions").MaxQueuedTokensError,
            ),
            http_error=import_module("dynamo.llm.exceptions").HttpError,
            online_renderer=import_module(
                "vllm.renderers.online_renderer"
            ).OnlineRenderer,
        )
    except (ImportError, OSError, AttributeError, TypeError) as cause:
        raise ReadRuntimeError(
            "pinned optional native read runtime is unavailable"
        ) from cause


def _runtime(engine: Any, profile: str, tasks: tuple[str, ...]) -> str:
    if profile not in _PROFILES:
        raise ReadRuntimeError("unsupported native read profile")
    architecture, strategy = _PROFILES[profile]
    model, config = engine.model_config, engine.vllm_config
    if (
        model.architecture != architecture
        or not is_native_decision_model(model, diffusion=profile == DIFFUSION_PROFILE)
        or model.runner_type != "generate"
        or model.is_encoder_decoder is not False
        or model.logprobs_mode != "raw_logprobs"
        or "generate" not in tasks
        or type(model.max_model_len) is not int
        or model.max_model_len < 2
        or config.parallel_config.data_parallel_size != 1
        or any(
            value is not None
            for value in (
                config.lora_config,
                config.speculative_config,
                config.kv_transfer_config,
                config.ec_transfer_config,
            )
        )
        or (profile == LIQUID_PROFILE and config.diffusion_config is not None)
        or (profile == DIFFUSION_PROFILE and config.diffusion_config is None)
    ):
        raise ReadRuntimeError(
            "effective runtime is incompatible with the native read profile"
        )
    return strategy


class _BudgetTokenizer:
    def __init__(self, tokenizer: Any, token_budget: int, byte_budget: int):
        self.tokenizer = tokenizer
        self.tokens, self.bytes = token_budget, byte_budget

    def consume_bytes(self, text: str) -> None:
        self.bytes -= len(text.encode("utf-8"))
        if self.bytes < 0:
            raise ReadValidationError("native read encoding byte budget exceeded")

    def consume_tokens(self, count: int) -> None:
        self.tokens -= count
        if self.tokens < 0:
            raise ReadValidationError("native read tokenization work budget exceeded")

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        self.consume_bytes(text)
        result = self.tokenizer.encode(text, **kwargs)
        self.consume_tokens(len(result))
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tokenizer, name)


class _Renderer:
    def __init__(self, renderer: Any, tokenizer: Any):
        self.renderer, self.tokenizer = renderer, tokenizer

    def get_tokenizer(self) -> Any:
        return self.tokenizer

    def __getattr__(self, name: str) -> Any:
        return getattr(self.renderer, name)


@dataclass(frozen=True)
class PreparedRead:
    prompt: dict[str, Any]
    sampling_params: Any
    rendered_prompt: dict[str, Any]
    rendered_ids: tuple[int, ...]
    label_groups: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class PreparedReadRequest:
    profile: str
    state: str
    questions: tuple[Any, ...]
    cache_salt: str
    reads: tuple[PreparedRead, ...]
    prefix: PreparedRead | None


class _ReplayEngine:
    def __init__(self, engine: Any, results: dict[str, Any]):
        self.model_config = engine.model_config
        self.vllm_config = engine.vllm_config
        self.results = results
        self.seen: set[str] = set()

    async def generate(self, prompt, sampling_params, request_id, **kwargs):
        if request_id in self.seen or request_id not in self.results:
            raise ReadExecutionError(
                "native reducer changed its prepared child requests"
            )
        self.seen.add(request_id)
        result = self.results[request_id]
        if tuple(prompt["prompt_token_ids"]) != tuple(result.prompt_token_ids):
            raise ReadExecutionError("native reducer changed its prepared prompt")
        yield result


class ReadExecutionAdapter:
    """Use native whole-request strategy semantics on one existing generation engine."""

    def __init__(self, engine, profile, native, strategy, abort_monitor, concurrency):
        self.engine, self.profile, self.native = engine, profile, native
        self.strategy, self.abort_monitor = strategy, abort_monitor
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
    ) -> ReadExecutionAdapter:
        if source_commit != PINNED_VLLM_COMMIT:
            raise ReadRuntimeError("native reads require the pinned vLLM source")
        native = _load_native(source_commit)
        strategy_name = _runtime(
            engine, profile, tuple(await engine.get_supported_tasks())
        )
        capacity = engine.vllm_config.scheduler_config.max_num_seqs
        if (
            type(max_concurrency) is not int
            or not 1 <= max_concurrency <= 64
            or type(capacity) is not int
            or capacity < 1
        ):
            raise ReadRuntimeError("native read concurrency is invalid")
        try:
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
        except (ValueError, AttributeError, TypeError) as cause:
            raise ReadRuntimeError(
                "native strategy cannot initialize on the effective runtime"
            ) from cause
        return cls(
            engine,
            profile,
            native,
            strategy,
            abort_monitor,
            min(max_concurrency, capacity),
        )

    async def prepare(self, wire: Any) -> PreparedReadRequest:
        """Render and validate all native prompts, label groups, and canvas capacity."""
        if not isinstance(wire, dict) or set(wire) != {"decision_read"}:
            raise ReadValidationError("invalid internal native read envelope")
        try:
            request = _parse_request(
                {"decision_head": wire["decision_read"]},
                self.profile,
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
            budget = _BudgetTokenizer(
                self.engine.renderer.get_tokenizer(),
                request.max_tokenization_work,
                request.max_encoding_bytes,
            )
            strategy = copy.copy(self.strategy)
            strategy.context = copy.copy(self.strategy.context)
            strategy.context.online_renderer = copy.copy(
                self.strategy.context.online_renderer
            )
            strategy.context.online_renderer.renderer = _Renderer(
                self.engine.renderer, budget
            )
            state = strategy.render_state(request.state)
            budget.consume_bytes(state)
            read_request = strategy._read_request(None, request.cache_salt)
            prepared = []
            for question in questions:
                content = strategy._content(state, question)
                budget.consume_bytes(content)
                rendered, prompt_ids = await strategy._render(
                    read_request, [{"role": "user", "content": content}]
                )
                budget.consume_tokens(len(prompt_ids))
                if prompt_ids[len(prompt_ids) - len(strategy.tail) :] != strategy.tail:
                    raise ReadValidationError(
                        "native read prompt changed the trained label suffix"
                    )
                groups = strategy._label_groups(question)
                params = strategy._sampling_params(
                    [token for group in groups for token in group], prompt_ids
                )
                prompt = strategy._read_input(rendered, prompt_ids)
                self._validate_plan(prompt, params, request.cache_salt)
                prepared.append(
                    PreparedRead(
                        copy.deepcopy(prompt),
                        params,
                        copy.deepcopy(rendered),
                        tuple(prompt_ids),
                        tuple(tuple(group) for group in groups),
                    )
                )
            prefix = self._prefix(strategy, prepared, request.cache_salt)
            count = sum(len(read.prompt["prompt_token_ids"]) for read in prepared)
            if prefix is not None:
                count += len(prefix.prompt["prompt_token_ids"])
            if count > request.max_input_tokens:
                raise ReadValidationError(
                    "native read total input token budget exceeded"
                )
            return PreparedReadRequest(
                self.profile,
                state,
                questions,
                request.cache_salt,
                tuple(prepared),
                prefix,
            )
        except (
            HeadValidationError,
            self.native.questions.StructuredDecisionError,
            self.native.validation_error,
            UnicodeEncodeError,
        ) as cause:
            raise ReadValidationError(
                "invalid or unsupported native read request"
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
            raise ReadValidationError("native read requires a text-only engine input")
        ids = prompt["prompt_token_ids"]
        if (
            not isinstance(ids, list)
            or not ids
            or any(type(token) is not int or token < 0 for token in ids)
        ):
            raise ReadValidationError("native read rendered invalid prompt tokens")
        if prompt.get("cache_salt") != salt:
            raise ReadValidationError("native renderer did not preserve isolation salt")
        if len(ids) + params.max_tokens > self.engine.model_config.max_model_len:
            raise ReadValidationError("native read exceeds effective context capacity")
        if self.profile == DIFFUSION_PROFILE:
            extra = params.extra_args
            if (
                params.max_tokens != 2
                or extra["diffusion_canvas_length"] != 16
                or extra["diffusion_max_steps"] != 1
                or extra["diffusion_read_only"] is not True
                or len(extra["diffusion_seed_canvas"]) != 16
            ):
                raise ReadRuntimeError(
                    "native diffusion canvas differs from qualified read profile"
                )

    def _prefix(self, strategy, prepared, salt) -> PreparedRead | None:
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
        return PreparedRead(prompt, params, prompt, tuple(prompt["prompt_token_ids"]))

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
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 240:
            raise ReadValidationError("invalid native read request ID")
        if cancelled():
            raise asyncio.CancelledError
        prepared = await self.prepare(wire)
        if cancelled():
            raise asyncio.CancelledError
        self.engine.check_admission(
            len(prepared.reads) + int(prepared.prefix is not None)
        )
        results: dict[str, Any] = {}
        if prepared.prefix is not None:
            prefix_id = f"{request_id}-prefix"
            results[prefix_id] = await self._read(
                prepared.prefix, context, prefix_id, cancelled, trace_headers
            )
        indices = iter(range(len(prepared.reads)))

        async def worker():
            for index in indices:
                child_id = f"{request_id}-{index}"
                results[child_id] = await self._read(
                    prepared.reads[index], context, child_id, cancelled, trace_headers
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
        native_reads = await self._reduce(prepared, results, request_id, trace_headers)
        if cancelled():
            raise asyncio.CancelledError
        return self._response(prepared, native_reads, results)

    async def _read(self, read, context, request_id, cancelled, trace_headers):
        async with self.semaphore:
            if cancelled():
                raise asyncio.CancelledError
            generator = self.engine.generate(
                read.prompt,
                read.sampling_params,
                request_id,
                lora_request=None,
                trace_headers=trace_headers,
                priority=0,
            )
            succeeded = False
            try:
                terminal = None
                async with self.abort_monitor(context, request_id):
                    async for output in generator:
                        if cancelled():
                            raise asyncio.CancelledError
                        if terminal is not None or output.request_id != request_id:
                            raise ReadExecutionError(
                                "native read returned a changed or duplicate terminal"
                            )
                        if output.finished is True:
                            terminal = output
                        elif output.finished is not False:
                            raise ReadExecutionError(
                                "native read returned an invalid terminal marker"
                            )
                    if terminal is None:
                        raise ReadExecutionError(
                            "native read did not return a finished output"
                        )
                    self._validate_output(read, terminal)
                succeeded = True
                return terminal
            finally:
                try:
                    if not succeeded:
                        await self.engine.abort(request_id)
                finally:
                    await generator.aclose()

    def _validate_output(self, read, output) -> None:
        if not isinstance(output.prompt_token_ids, list) or tuple(
            output.prompt_token_ids
        ) != tuple(read.prompt["prompt_token_ids"]):
            raise ReadExecutionError("native read changed the prepared prompt tokens")
        if len(output.outputs) != 1:
            raise ReadExecutionError("native read requires one complete output")
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
            raise ReadExecutionError("native read returned invalid completion evidence")
        cached = output.num_cached_tokens
        if cached is not None and (
            type(cached) is not int or not 0 <= cached <= len(output.prompt_token_ids)
        ):
            raise ReadExecutionError("native read returned invalid cache accounting")
        label_ids = read.sampling_params.logprob_token_ids
        if label_ids:
            if not completion.logprobs or any(
                token not in completion.logprobs[0] for token in label_ids
            ):
                raise ReadExecutionError("native read omitted selected label logprobs")
            if any(
                type(completion.logprobs[0][token].logprob) not in (int, float)
                or math.isnan(completion.logprobs[0][token].logprob)
                or completion.logprobs[0][token].logprob > 0
                for token in label_ids
            ):
                raise ReadExecutionError("native read returned invalid label logprobs")

    async def _reduce(self, prepared, results, request_id, trace_headers):
        strategy = copy.copy(self.strategy)
        strategy.context = copy.copy(self.strategy.context)
        replay = _ReplayEngine(self.engine, results)
        strategy.context.engine_client = replay
        renders = iter(prepared.reads)

        async def render(read_request, messages):
            plan = next(renders)
            return copy.deepcopy(plan.rendered_prompt), list(plan.rendered_ids)

        strategy._render = render
        groups = {
            question.id: read.label_groups
            for question, read in zip(prepared.questions, prepared.reads)
        }
        strategy._label_groups = lambda question: [
            list(group) for group in groups[question.id]
        ]
        reduced = await strategy.read(
            list(prepared.questions),
            None,
            prepared.state,
            request_id=request_id,
            chat_template_kwargs=None,
            lora_request=None,
            priority=0,
            cache_salt=prepared.cache_salt,
            trace_headers=trace_headers,
        )
        if replay.seen != set(results):
            raise ReadExecutionError(
                "native reducer did not consume every prepared child"
            )
        return reduced

    def _response(self, prepared, native_reads, results) -> dict[str, Any]:
        if len(native_reads) != len(prepared.questions):
            raise ReadExecutionError("native reducer returned incomplete outcomes")
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
                raise ReadExecutionError(
                    "native reducer returned invalid probabilities"
                )
            mass = result.label_mass
            if (
                type(mass) not in (int, float)
                or not math.isfinite(mass)
                or not 0 <= mass <= 1
            ):
                raise ReadExecutionError(
                    "native reducer returned invalid vocabulary label mass"
                )
            outcomes.append(
                {
                    "ordinal": ordinal,
                    "probabilities": probabilities,
                    "label_mass": mass,
                }
            )
        cached = [result.num_cached_tokens for result in results.values()]
        return {
            "protocol_version": 1,
            "profile": self.profile,
            "outcomes": outcomes,
            "usage": {
                "input_tokens": sum(
                    len(result.prompt_token_ids) for result in results.values()
                ),
                "output_tokens": sum(
                    len(result.outputs[0].token_ids) for result in results.values()
                ),
                "cached_tokens": sum(cached)
                if all(value is not None for value in cached)
                else None,
            },
        }
