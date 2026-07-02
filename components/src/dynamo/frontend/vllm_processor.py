#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

#
# Use vllm for input and output processing
#

import asyncio
import json
import logging
import os
import time
from argparse import Namespace
from collections.abc import AsyncGenerator, Sequence
from typing import Any

from msgspec.structs import replace as msgspec_replace
from vllm.config import CacheConfig, LoadConfig, ModelConfig, VllmConfig
from vllm.entrypoints.chat_utils import get_tool_call_id_type, load_chat_template
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.tasks import GENERATION_TASKS
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser, ToolParserManager
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.output_processor import OutputProcessor, OutputProcessorOutput
from vllm.v1.engine.parallel_sampling import ParentRequest

from dynamo._internal import ModelDeploymentCard
from dynamo.common.multimodal.mm_kwargs_transfer import (
    MmKwargsNixlSender,
    MmKwargsSender,
    MmKwargsShmSender,
)
from dynamo.common.multimodal.routing_utils import build_mm_routing_info_from_features
from dynamo.common.utils import nvtx_utils as _nvtx
from dynamo.frontend.frontend_args import FrontendConfig
from dynamo.llm import ModelCardInstanceId, PythonAsyncEngine, RoutedEngine
from dynamo.vllm.capacity import (
    normalize_vllm_reasoning_parser,
    normalize_vllm_tool_parser,
    reasoning_parsers_compatible,
)

from .prepost import StreamingPostProcessor, preprocess_chat_request
from .utils import (
    extract_mm_urls,
    handle_engine_error,
    make_internal_error,
    random_uuid,
    resolve_chat_template,
)

logger = logging.getLogger(__name__)

REASONING_AWARE_GUIDED_DECODING_RUNTIME_KEY = "reasoning_aware_guided_decoding"


_FINISH_REASON_MAP: dict[str, FinishReason] = {
    "eos": FinishReason.STOP,
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "error": FinishReason.ERROR,
    "cancelled": FinishReason.ABORT,
    "content_filter": FinishReason.STOP,
}

_DYNAMO_NON_FORCE_REASONING = frozenset({"basic", "nemotron_deci"})


class _ExplicitThinkReasoningParser(BaseThinkingReasoningParser):
    """A <think> parser that starts in content mode when no marker is present."""

    dynamo_generated_reasoning_start = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initial_in_reasoning = False

    def adjust_initial_state_from_prompt(self, prompt_token_ids: list[int]) -> None:
        self._initial_in_reasoning = not self.is_reasoning_end(prompt_token_ids)

    def adjust_request(self, request: Any) -> Any:
        request.skip_special_tokens = False
        return request

    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        for token_id in reversed(input_ids):
            if token_id == self.start_token_id:
                return False
            if token_id == self.end_token_id:
                return True
        return True

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        for index in range(len(input_ids) - 1, -1, -1):
            if input_ids[index] == self.start_token_id:
                return []
            if input_ids[index] == self.end_token_id:
                return input_ids[index + 1 :]
        return input_ids

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        del previous_text, current_text, current_token_ids
        if self.end_token_id in previous_token_ids:
            return DeltaMessage(content=delta_text or None)

        if self._initial_in_reasoning or self.start_token_id in previous_token_ids:
            if self.end_token_id in delta_token_ids:
                reasoning, _, content = delta_text.partition(self.end_token)
                return DeltaMessage(
                    reasoning=reasoning or None, content=content or None
                )
            return DeltaMessage(reasoning=delta_text or None)

        if self.start_token_id in delta_token_ids:
            before_start, _, after_start = delta_text.partition(self.start_token)
            if self.end_token_id in delta_token_ids:
                reasoning, _, content = after_start.partition(self.end_token)
                return DeltaMessage(
                    reasoning=reasoning or None,
                    content=(before_start + content) or None,
                )
            return DeltaMessage(
                reasoning=after_start or None, content=before_start or None
            )

        return DeltaMessage(content=delta_text or None)

    def extract_reasoning(self, model_output: str, request: Any):
        del request
        if self.start_token not in model_output and not self._initial_in_reasoning:
            return None, model_output or None
        if self.start_token in model_output:
            before_start, _, after_start = model_output.partition(self.start_token)
        else:
            before_start, after_start = "", model_output
        if self.end_token not in after_start:
            return after_start or None, before_start or None
        reasoning, _, content = after_start.partition(self.end_token)
        return reasoning or None, (before_start + content) or None


class _ImplicitStartThinkReasoningParser(BaseThinkingReasoningParser):
    """A force parser for models whose generated output omits ``<think>``.

    MiniMax's append-think chat template places the model in reasoning mode,
    but the generated completion starts with reasoning text and emits only the
    closing marker.  vLLM's native adapter prepends ``<think>`` to content;
    Dynamo instead needs to split that implicit reasoning prefix into the
    OpenAI ``reasoning_content`` field without leaking either marker.
    """

    def adjust_request(self, request: Any) -> Any:
        request.skip_special_tokens = False
        return request

    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        del previous_text, current_text, current_token_ids
        if self.end_token_id in previous_token_ids:
            return DeltaMessage(content=delta_text or None)

        # The opening marker is normally prompt-injected rather than emitted.
        # Accept it defensively and never surface it in either response field.
        reasoning_delta = delta_text.replace(self.start_token, "", 1)
        if self.end_token_id in delta_token_ids or self.end_token in reasoning_delta:
            reasoning, separator, content = reasoning_delta.partition(self.end_token)
            if not separator:
                # A special-token-aware decoder can expose the token ID while
                # omitting its text.  In that case the whole visible delta is
                # still pre-boundary reasoning.
                return DeltaMessage(reasoning=reasoning_delta or None)
            return DeltaMessage(
                reasoning=reasoning or None,
                content=content or None,
            )
        return DeltaMessage(reasoning=reasoning_delta or None)


def map_finish_reason(raw_reason: str | None) -> FinishReason | None:
    if raw_reason is None:
        return None
    if raw_reason.startswith("error"):
        return FinishReason.ERROR
    if raw_reason.startswith("abort"):
        return FinishReason.ABORT
    if raw_reason.startswith("content_filter"):
        logger.info("Router finish_reason indicates content filtering: %s", raw_reason)
        raw_reason = "content_filter"
    mapped = _FINISH_REASON_MAP.get(raw_reason)
    if mapped is None:
        logger.warning("Unknown finish_reason from router: %s", raw_reason)
    return mapped


def _runtime_config_context_length(mdc: ModelDeploymentCard) -> int | None:
    runtime_config = mdc.runtime_config()
    if not isinstance(runtime_config, dict):
        return None

    context_length = runtime_config.get("context_length")
    if type(context_length) is not int or context_length <= 0:
        return None
    return context_length


def _validated_reasoning_aware_guided_decoding(
    mdc: ModelDeploymentCard,
    effective_reasoning_parser: str | None,
) -> bool:
    """Trust the worker capability only for the frontend's effective parser.

    A frontend parser override is local to this process; the worker that
    published the capability could only validate its own Dynamo parser.  Fail
    closed for old/malformed cards and reject an incompatible override rather
    than pairing two different reasoning wire formats.
    """
    runtime_config = mdc.runtime_config()
    if not isinstance(runtime_config, dict):
        return False
    runtime_data = runtime_config.get("runtime_data")
    if (
        not isinstance(runtime_data, dict)
        or runtime_data.get(REASONING_AWARE_GUIDED_DECODING_RUNTIME_KEY) is not True
    ):
        return False

    worker_reasoning_parser = runtime_config.get("reasoning_parser")
    if not isinstance(worker_reasoning_parser, str) or not worker_reasoning_parser:
        return False
    native_frontend_parser = normalize_vllm_reasoning_parser(effective_reasoning_parser)
    if not native_frontend_parser:
        return False
    if not reasoning_parsers_compatible(
        native_frontend_parser, worker_reasoning_parser
    ):
        raise ValueError(
            "Frontend reasoning parser override is incompatible with the "
            "worker's reasoning-aware guided-decoding parser: "
            f"frontend={effective_reasoning_parser!r}, "
            f"worker={worker_reasoning_parser!r}"
        )
    return True


def _is_named_tool_choice(tool_choice: Any) -> bool:
    if isinstance(tool_choice, dict):
        function = tool_choice.get("function")
        return (
            tool_choice.get("type") == "function"
            and isinstance(function, dict)
            and bool(function.get("name"))
        )
    function = getattr(tool_choice, "function", None)
    return getattr(tool_choice, "type", None) == "function" and bool(
        getattr(function, "name", None)
    )


def _response_reasoning_parser_class(
    request_for_sampling: Any,
    reasoning_parser_class: type[ReasoningParser] | None,
    reasoning_ended: bool | None,
    reasoning_aware_guided_decoding: bool,
) -> type[ReasoningParser] | None:
    """Disable only force-reasoning parsing for bare guided JSON.

    Without backend support the required/named JSON grammar starts at token
    zero. A parser whose prompt state is still in reasoning would otherwise
    classify that bare JSON as reasoning. Structural-tag/custom grammars carry
    model-native reasoning syntax and must keep the parser active.
    """
    if reasoning_parser_class is None or reasoning_aware_guided_decoding:
        return reasoning_parser_class

    tool_choice = getattr(request_for_sampling, "tool_choice", None)
    guided_tool_choice = tool_choice == "required" or _is_named_tool_choice(tool_choice)
    structured_outputs = getattr(request_for_sampling, "structured_outputs", None)
    structural_tag = getattr(structured_outputs, "structural_tag", None)
    grammar_from_tool_parser = getattr(
        request_for_sampling, "_grammar_from_tool_parser", False
    )
    if (
        guided_tool_choice
        and reasoning_ended is False
        and structural_tag is None
        and not grammar_from_tool_parser
    ):
        return None
    return reasoning_parser_class


def _request_reasoning_policy(
    parser_name: str | None,
    request: dict[str, Any],
    request_for_sampling: Any,
    reasoning_parser_class: type[ReasoningParser] | None,
    reasoning_ended: bool | None,
    reasoning_aware_guided_decoding: bool,
    normalized_template_kwargs: dict[str, Any] | None = None,
) -> tuple[bool | None, type[ReasoningParser] | None]:
    normalized_parser_name = (
        parser_name.strip().lower() if isinstance(parser_name, str) else None
    )
    template_kwargs: dict[str, Any] = {}
    for key in ("chat_template_kwargs", "chat_template_args"):
        value = request.get(key)
        if isinstance(value, dict):
            template_kwargs.update(value)
    root_thinking = request.get("thinking")
    if isinstance(root_thinking, bool):
        template_kwargs.setdefault("thinking", root_thinking)
        template_kwargs.setdefault("enable_thinking", root_thinking)
    elif isinstance(root_thinking, dict):
        thinking_type = root_thinking.get("type")
        if thinking_type in {"enabled", "disabled"}:
            enabled = thinking_type == "enabled"
            template_kwargs.setdefault("thinking", enabled)
            template_kwargs.setdefault("enable_thinking", enabled)
    if normalized_template_kwargs:
        template_kwargs.update(normalized_template_kwargs)
    messages = request.get("messages") or []
    last_message = messages[-1] if messages else None
    last_role = (
        last_message.get("role")
        if isinstance(last_message, dict)
        else getattr(last_message, "role", None)
    )
    if normalized_parser_name == "kimi_k25" and last_role == "tool":
        # Kimi K2.5 post-tool turns generate the visible answer directly.
        # Native vLLM's kimi_k2 parser otherwise defaults back into reasoning.
        return True, None

    if normalized_parser_name == "mistral":
        reasoning_effort = request.get("reasoning_effort")
        if reasoning_effort is None:
            reasoning_effort = template_kwargs.get("reasoning_effort")
        if reasoning_effort is None or reasoning_effort == "none":
            return True, None

    if normalized_parser_name == "deepseek_r1":
        disabled = (
            template_kwargs.get("thinking") is False
            or template_kwargs.get("enable_thinking") is False
        )
        disabled = disabled or template_kwargs.get("thinking_mode") == "chat"
        if disabled:
            return True, None

    if normalized_parser_name in {"deepseek_v4", "deepseek-v4", "deepseekv4"}:
        disabled = (
            template_kwargs.get("thinking") is False
            or template_kwargs.get("enable_thinking") is False
        )
        disabled = disabled or template_kwargs.get("thinking_mode") == "chat"
        if disabled:
            return True, None

    if normalized_parser_name in {"minimax_m3", "minimax-m3"} and (
        template_kwargs.get("thinking_mode") == "disabled"
        or template_kwargs.get("enable_thinking") is False
        or template_kwargs.get("thinking") is False
    ):
        return True, None

    return reasoning_ended, _response_reasoning_parser_class(
        request_for_sampling,
        reasoning_parser_class,
        reasoning_ended,
        reasoning_aware_guided_decoding,
    )


def _resolve_reasoning_parser_names(
    worker_parser_name: str | None,
    frontend_override: str | None,
) -> tuple[str | None, str | None, str | None]:
    """Resolve policy/effective/native names without erasing Dyn semantics."""
    effective_name = frontend_override or worker_parser_name
    policy_name = worker_parser_name or effective_name
    native_name = normalize_vllm_reasoning_parser(effective_name)
    return policy_name, effective_name, native_name


def _reasoning_parser_class(
    policy_name: str | None,
    native_name: str | None,
) -> type[ReasoningParser] | None:
    if isinstance(policy_name, str) and policy_name.strip().lower() == "kimi":
        raise ValueError(
            "Dynamo reasoning parser 'kimi' uses the original Kimi "
            "◁think▷ wire format, which this vLLM version does not support"
        )
    if not native_name:
        return None
    if (
        isinstance(policy_name, str)
        and policy_name.strip().lower() in _DYNAMO_NON_FORCE_REASONING
    ):
        return _ExplicitThinkReasoningParser
    if isinstance(policy_name, str) and policy_name.strip().lower() in {
        "minimax_append_think",
        "minimax-append-think",
        "minimax_m2_append_think",
    }:
        return _ImplicitStartThinkReasoningParser
    return ReasoningParserManager.get_reasoning_parser(native_name)


def _build_reasoning_parser_metadata(
    reasoning_parser_class: type[ReasoningParser] | None,
    tokenizer: TokenizerLike,
    chat_template_kwargs: dict[str, Any],
    request_for_sampling: Any,
    prompt_token_ids: list[int],
    reasoning_aware_guided_decoding: bool = False,
) -> tuple[bool | None, dict[str, Any] | None]:
    if reasoning_parser_class is None and not reasoning_aware_guided_decoding:
        return None, None

    parser_kwargs = {"chat_template_kwargs": chat_template_kwargs}
    if getattr(request_for_sampling, "_grammar_from_tool_parser", False):
        return True, parser_kwargs
    if reasoning_parser_class is None:
        # The backend-native parser can derive its initial state from these
        # kwargs even when this frontend is not separating reasoning output.
        return None, parser_kwargs

    reasoning_parser = reasoning_parser_class(
        tokenizer,
        chat_template_kwargs=chat_template_kwargs,
    )
    return reasoning_parser.is_reasoning_end(prompt_token_ids), parser_kwargs


def _inject_routing_metadata(
    dynamo_preproc: dict[str, Any],
    target: dict[str, Any],
    mm_routing_info: dict[str, Any] | None = None,
) -> None:
    extra_updates: dict[str, Any] = {}
    for key in ("reasoning_ended", "reasoning_parser_kwargs"):
        if key in dynamo_preproc:
            extra_updates[key] = dynamo_preproc[key]
    if dynamo_preproc.get("mm_processor_kwargs") is not None:
        extra_updates["mm_processor_kwargs"] = dynamo_preproc["mm_processor_kwargs"]

    if extra_updates:
        extra_args = target.get("extra_args")
        if not isinstance(extra_args, dict):
            extra_args = {}
            target["extra_args"] = extra_args
        extra_args.update(extra_updates)

    if mm_routing_info is not None:
        target["mm_routing_info"] = mm_routing_info


class VllmProcessor:
    def __init__(
        self,
        tokenizer: TokenizerLike,
        input_processor: InputProcessor,
        output_processor: OutputProcessor,
        tool_parser_class: type[ToolParser] | None,
        reasoning_parser_class: type[ReasoningParser] | None,
        reasoning_parser_name: str | None,
        routed_engine: RoutedEngine,
        block_size: int = 16,
        enable_auto_tool_choice: bool = False,
        reasoning_aware_guided_decoding: bool = False,
        tool_call_id_type: str = "random",
        model_config: ModelConfig | None = None,
    ):
        self.tokenizer = tokenizer
        self.input_processor = input_processor
        self.routed_engine = routed_engine
        self.output_processor = output_processor
        self.tool_parser_class = tool_parser_class
        self.reasoning_parser_class = reasoning_parser_class
        self.reasoning_parser_name = reasoning_parser_name
        self.exclude_tools_when_tool_choice_none = True
        self.block_size = block_size
        self.enable_auto_tool_choice = enable_auto_tool_choice
        self.reasoning_aware_guided_decoding = reasoning_aware_guided_decoding
        self.tool_call_id_type = tool_call_id_type
        self.model_config = model_config
        # Sender for mm_kwargs transfer — instantiated lazily on first MM request.
        # MmKwargsShmSender for same-node transfers (default), MmKwargsNixlSender
        # for cross-node RDMA. Controlled by DYNAMO_MM_TRANSFER env var.
        self._sender: MmKwargsSender | None = None
        # Set DYNAMO_DISABLE_NIXL_MM=1 to disable mm_kwargs transfer entirely.
        # Set DYNAMO_MM_TRANSFER to choose transfer mode:
        #   shm (default): shared memory. Same-node only (~2ms). If the
        #     backend can't read the segment (cross-node), it falls back to
        #     normal processing (backend runs HF processor).
        #   nixl: NIXL RDMA. Works cross-node via IB.
        self.nixl_mm_enabled = os.environ.get("DYNAMO_DISABLE_NIXL_MM", "") != "1"
        transfer_mode = os.environ.get("DYNAMO_MM_TRANSFER", "shm").lower()
        self.use_shm_transfer = transfer_mode == "shm"
        logger.info("[mm-routing] Transfer mode: %s", transfer_mode)

    def _get_eos_token_ids(self) -> list[int]:
        """Return EOS token ids using tokenizer metadata.

        vLLM 0.17.0 removed EngineCoreRequest.eos_token_id, so Dynamo can no
        longer read EOS ids from the preprocessed request object.
        """
        eos_token_ids = getattr(self.tokenizer, "eos_token_ids", None)
        if eos_token_ids is not None and not isinstance(eos_token_ids, int):
            return list(eos_token_ids)

        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            return []
        return [eos_token_id]

    async def _prepare_mm_routing(
        self,
        vllm_preproc: EngineCoreRequest,
        dynamo_preproc: dict[str, Any],
    ) -> tuple[dict | None, list, bool]:
        """Extract MM routing info and prepare mm_kwargs transfer.

        Returns:
            (mm_routing_info, cleanup_items, transferred)
            cleanup_items: passed to self._sender.cleanup() after streaming.
            transferred: True when all features with data were sent successfully.
        """
        mm_routing_info = None
        cleanup_items: list = []
        nixl_transferred = False

        rng_routing = _nvtx.start_range("mm_frontend:build_routing_info", color="cyan")
        if vllm_preproc.mm_features:
            mm_routing_info = build_mm_routing_info_from_features(
                vllm_preproc.mm_features,
                prompt_token_ids=list(vllm_preproc.prompt_token_ids),
            )
            # Forward mm_hashes to backend for hash consistency — the backend
            # will use these directly instead of recomputing.
            mm_hashes_list = [f.mm_hash for f in vllm_preproc.mm_features]
            mm_placeholders_list = [
                (f.mm_position.offset, f.mm_position.length)
                for f in vllm_preproc.mm_features
            ]
            # Transport mm_hashes and mm_placeholders to backend via extra_args.
            if "extra_args" not in dynamo_preproc:
                dynamo_preproc["extra_args"] = {}
            dynamo_preproc["extra_args"]["mm_hashes"] = mm_hashes_list
            dynamo_preproc["extra_args"]["mm_placeholders"] = mm_placeholders_list
            # Forward the expanded prompt_token_ids (with image placeholders)
            # so the backend can use them in the pre-rendered MultiModalInput.
            dynamo_preproc["extra_args"]["expanded_token_ids"] = list(
                vllm_preproc.prompt_token_ids
            )

            n_blocks = len(mm_routing_info["block_mm_infos"]) if mm_routing_info else 0
            n_mm_blocks = sum(
                1 for b in (mm_routing_info or {}).get("block_mm_infos", []) if b
            )
            logger.debug(
                "[mm-routing] Built mm_routing_info: %d mm_features, "
                "%d hashes, %d total blocks, %d blocks with MM content, "
                "block_size=%d",
                len(vllm_preproc.mm_features),
                len(mm_hashes_list),
                n_blocks,
                n_mm_blocks,
                self.block_size,
            )
            if logger.isEnabledFor(logging.DEBUG):
                for i, f in enumerate(vllm_preproc.mm_features):
                    logger.debug(
                        "[mm-routing]   feature[%d]: modality=%s, hash=%s..., "
                        "offset=%d, length=%d",
                        i,
                        f.modality,
                        f.mm_hash[:16] if f.mm_hash else "None",
                        f.mm_position.offset,
                        f.mm_position.length,
                    )

            # Transfer pre-processed mm_kwargs to the backend so it can skip
            # the HF processor.  Strategy:
            #   - shm (default): shared memory, same-node only (~2ms).
            #     Cross-node backends fail gracefully and fall back to
            #     normal processing.
            #   - nixl: NIXL RDMA (works cross-node via IB).
            if not self.nixl_mm_enabled:
                logger.debug(
                    "[mm-routing] mm_kwargs transfer disabled via DYNAMO_DISABLE_NIXL_MM"
                )
            else:
                try:
                    if self._sender is None:
                        self._sender = (
                            MmKwargsShmSender()
                            if self.use_shm_transfer
                            else MmKwargsNixlSender()
                        )
                    # NVTX annotation is owned by MmKwargsSender.prepare via
                    # the subclass's _nvtx_label/_nvtx_color class attrs.
                    extra_update, cleanup_items = await self._sender.prepare(
                        vllm_preproc.mm_features, modality="image"
                    )
                    if extra_update is not None:
                        dynamo_preproc["extra_args"].update(extra_update)
                        nixl_transferred = True
                    else:
                        logger.debug(
                            "[mm-routing] sender returned None — no tensors to transfer"
                        )
                except Exception:
                    logger.warning(
                        "[mm-routing] sender failed, backend will run HF processor",
                        exc_info=True,
                    )
                    cleanup_items = []

        else:
            logger.debug("[mm-routing] No mm_features — text-only request")
        _nvtx.end_range(rng_routing)

        return mm_routing_info, cleanup_items, nixl_transferred

    # Ideally we would map NVCreateChatCompletionRequest into Python so it can be type checked, but
    # it has a lot of fields.
    # request: dynamo.NVCreateChatCompletionRequest
    async def generator(
        self, request: dict[str, Any], context: Any | None = None
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Run a single request through the engine. Does pre and post processing on this machine, delegates
        model inference to a backend using the router.
        """
        with _nvtx.annotate("mm_frontend:generator", color="blue"):
            async for item in self._generator_inner(request, context=context):
                yield item

    async def _generator_inner(
        self, request: dict[str, Any], context: Any | None = None
    ) -> AsyncGenerator[dict[str, Any], None]:
        request_id = random_uuid()

        # vLLM's Pydantic model requires image_url.detail to be 'auto'/'low'/'high'.
        # The Rust HTTP layer accepts None/missing, so normalize before validation.
        messages = request.get("messages") or []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "image_url":
                    img_url = part.get("image_url")
                    if isinstance(img_url, dict) and img_url.get("detail") is None:
                        img_url["detail"] = "auto"

        # Images are fetched by vLLM's renderer via DynamoMediaConnector,
        # which wraps our ImageLoader (LRU cache + in-flight dedup).
        # No data URI encoding needed.
        with _nvtx.annotate("mm_frontend:preprocess_chat", color="yellow"):
            pre = await preprocess_chat_request(
                request,
                tokenizer=self.tokenizer,
                renderer=self.input_processor.renderer,
                tool_parser_class=self.tool_parser_class,
                reasoning_parser_class=self.reasoning_parser_class,
                reasoning_policy_name=self.reasoning_parser_name,
                model_config=self.model_config,
                exclude_tools_when_tool_choice_none=self.exclude_tools_when_tool_choice_none,
                enable_auto_tool_choice=self.enable_auto_tool_choice,
            )

        request_for_sampling = pre.request_for_sampling
        tool_parser = pre.tool_parser
        chat_template_kwargs = pre.chat_template_kwargs
        engine_prompt = pre.engine_prompt
        tokens = pre.prompt_token_ids

        if request_for_sampling.max_completion_tokens is not None:
            max_tokens = request_for_sampling.max_completion_tokens
        elif request_for_sampling.max_tokens is not None:
            max_tokens = request_for_sampling.max_tokens
        else:
            # This should mean model max - prompt len.
            max_tokens = None

        sampling_params = SamplingParams(
            output_kind=RequestOutputKind.DELTA,
            max_tokens=max_tokens,
        )
        # generation_config.json
        # Skip eos_token_id: vLLM 0.17.0 made SamplingParams.eos_token_id a
        # read-only property; eos tokens are handled via eos_token_ids below.
        for k, v in self.input_processor.generation_config_fields.items():
            if k == "eos_token_id":
                continue
            if hasattr(sampling_params, k):
                setattr(sampling_params, k, v)

        # User request: copy fields supported by both request schema and
        # SamplingParams, excluding fields handled separately below.
        sampling_fields = (
            set(getattr(SamplingParams, "__annotations__", ()))
            & set(type(request_for_sampling).model_fields)
        ) - {"max_tokens", "logprobs", "output_kind"}
        for k in sorted(sampling_fields):
            v = getattr(request_for_sampling, k, None)
            if v is not None:
                setattr(sampling_params, k, v)
        # nvext.max_thinking_tokens is enforced on the worker, not here. The
        # frontend's InputProcessor is built without reasoning_config (it only
        # tokenizes), so setting sampling_params.thinking_token_budget would
        # cause process_inputs._validate_params to reject the request. Pluck
        # the value out of nvext and pass it directly into dynamo_preproc
        # below.
        nvext_max_thinking_tokens = (request.get("nvext") or {}).get(
            "max_thinking_tokens"
        )
        logprobs = request_for_sampling.logprobs
        top_logprobs = request_for_sampling.top_logprobs
        if logprobs is True:
            sampling_params.logprobs = top_logprobs or 1
        elif isinstance(logprobs, int) and not isinstance(logprobs, bool):
            sampling_params.logprobs = logprobs
        elif top_logprobs not in (None, 0):
            sampling_params.logprobs = top_logprobs
        if sampling_params.logprobs is not None and sampling_params.logprobs > 0:
            logger.warning(
                "Logprobs requested but not supported in distributed inference mode"
            )

        # The renderer's process_for_engine() always returns a fully processed
        # EngineInput (TokenInputs or MultiModalInputs) with a "type" key.
        # Pass it directly to process_inputs() — no need to rebuild a
        # TokensPrompt, and this avoids the deprecation warning.
        prompt_inputs = engine_prompt
        if request_for_sampling.cache_salt is not None:
            prompt_inputs["cache_salt"] = request_for_sampling.cache_salt
        if request_for_sampling.mm_processor_kwargs is not None:
            prompt_inputs[
                "mm_processor_kwargs"
            ] = request_for_sampling.mm_processor_kwargs

        with _nvtx.annotate("mm_frontend:process_inputs", color="orange"):
            vllm_preproc: EngineCoreRequest = self.input_processor.process_inputs(
                request_id,
                prompt_inputs,
                sampling_params,
                GENERATION_TASKS,  # vLLM 0.17.0: required supported_tasks arg
            )

        InputProcessor.assign_request_id(vllm_preproc)

        # vLLM 0.17.0 removed EngineCoreRequest.eos_token_id. Dynamo now uses
        # tokenizer metadata for EOS ids when constructing the router payload.

        reasoning_ended, reasoning_parser_kwargs = _build_reasoning_parser_metadata(
            self.reasoning_parser_class,
            self.tokenizer,
            chat_template_kwargs,
            request_for_sampling,
            tokens,
            self.reasoning_aware_guided_decoding,
        )
        reasoning_ended, response_reasoning_parser_class = _request_reasoning_policy(
            self.reasoning_parser_name,
            request,
            request_for_sampling,
            self.reasoning_parser_class,
            reasoning_ended,
            self.reasoning_aware_guided_decoding,
            chat_template_kwargs,
        )

        # Convert to a Python object that has fields that match our PreprocessedRequest
        sp = vllm_preproc.sampling_params
        dynamo_preproc = {
            "model": request["model"],
            "token_ids": tokens,
            "stop_conditions": {
                "max_tokens": sp.max_tokens,
                "stop": sp.stop,
                "stop_token_ids": sp.stop_token_ids,
                "min_tokens": sp.min_tokens,
                "ignore_eos": sp.ignore_eos,
                "max_thinking_tokens": nvext_max_thinking_tokens,
            },
            "sampling_options": {
                "n": sp.n,
                "presence_penalty": sp.presence_penalty,
                "frequency_penalty": sp.frequency_penalty,
                "repetition_penalty": sp.repetition_penalty,
                "temperature": sp.temperature,
                "top_p": sp.top_p,
                "top_k": sp.top_k,
                "min_p": sp.min_p,
                "seed": sp.seed,
            },
            "output_options": {
                "logprobs": sp.logprobs,
                "prompt_logprobs": sp.prompt_logprobs,
                "skip_special_tokens": sp.skip_special_tokens,
            },
            "eos_token_ids": self._get_eos_token_ids(),
            "annotations": [],
            "routing": request.get("routing"),
        }
        if reasoning_ended is not None:
            dynamo_preproc["reasoning_ended"] = reasoning_ended
        if reasoning_parser_kwargs is not None:
            dynamo_preproc["reasoning_parser_kwargs"] = reasoning_parser_kwargs

        # Extract MM routing metadata and prepare transfer.
        cleanup_items: list = []
        try:
            (
                mm_routing_info,
                cleanup_items,
                nixl_transferred,
            ) = await self._prepare_mm_routing(vllm_preproc, dynamo_preproc)

            # Forward multimodal URLs so the backend handler can load the media.
            # Only skip when ALL features were transferred — a partial transfer
            # (some features had data=None due to processor cache) still needs
            # URLs for the backend to process the missing features.
            n_features = (
                len(vllm_preproc.mm_features) if vllm_preproc.mm_features else 0
            )
            n_with_data = sum(
                1 for f in (vllm_preproc.mm_features or []) if f.data is not None
            )
            all_transferred = nixl_transferred and n_with_data == n_features
            if not all_transferred:
                mm_data = extract_mm_urls(request.get("messages") or [])
                if mm_data:
                    dynamo_preproc["multi_modal_data"] = mm_data

            # Forward mm_processor_kwargs (e.g. use_audio_in_video) to the backend.
            if request_for_sampling.mm_processor_kwargs is not None:
                dynamo_preproc[
                    "mm_processor_kwargs"
                ] = request_for_sampling.mm_processor_kwargs

            def new_post_processor() -> StreamingPostProcessor:
                return StreamingPostProcessor(
                    tokenizer=self.tokenizer,
                    request_for_sampling=request_for_sampling,
                    sampling_params=sampling_params,
                    prompt_token_ids=tokens,
                    tool_parser=tool_parser,
                    reasoning_parser_class=response_reasoning_parser_class,
                    chat_template_kwargs=chat_template_kwargs,
                    stream_response=bool(request.get("stream", False)),
                    tool_call_id_type=self.tool_call_id_type,
                    model_config=self.model_config,
                )

            # StreamingPostProcessor keeps delta/tool/reasoning parser state, so
            # parallel choices must not share one instance. Keep one state machine
            # per choice index while the backend interleaves n>1 token chunks.
            post_processors = {
                output_idx: new_post_processor() for output_idx in range(sp.n)
            }

            async for item in self._generate_and_stream(
                request_id,
                request,
                dynamo_preproc,
                tokens,
                vllm_preproc,
                post_processors,
                mm_routing_info=mm_routing_info,
                context=context,
            ):
                yield item
        finally:
            if cleanup_items and self._sender is not None:
                await self._sender.cleanup(cleanup_items)

    async def _generate_and_stream(
        self,
        request_id: str,
        request: dict[str, Any],
        dynamo_preproc: dict[str, Any],
        tokens: list[int],
        vllm_preproc: EngineCoreRequest,
        post_processors: dict[int, StreamingPostProcessor],
        mm_routing_info: dict[str, Any] | None = None,
        context: Any | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        sp = vllm_preproc.sampling_params
        output_request_ids: dict[int, str]
        registered_request_ids: list[str]

        if sp.n == 1:
            self.output_processor.add_request(vllm_preproc, None)
            output_request_ids = {0: vllm_preproc.request_id}
            registered_request_ids = [vllm_preproc.request_id]
        else:
            # vLLM's normal engine path fans out SamplingParams.n>1 into
            # ParentRequest children before registering with OutputProcessor.
            # Dynamo bypasses that path here: the backend generates indexed
            # token chunks and this frontend feeds those chunks directly into
            # vLLM's OutputProcessor. Recreate the same parent/child request
            # state so each choice has its own request id, sampling params,
            # detokenizer/logprob state, and OpenAI choice index.
            #
            # See vLLM's implementation:
            # https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/v1/engine/async_llm.py
            # https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/v1/engine/output_processor.py
            # https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/v1/engine/parallel_sampling.py
            parent_preproc = vllm_preproc
            if parent_preproc.external_req_id is None:
                parent_preproc = msgspec_replace(
                    parent_preproc, external_req_id=parent_preproc.request_id
                )
            parent_req = ParentRequest(parent_preproc)
            output_request_ids = {}
            registered_request_ids = []
            for output_idx in range(sp.n):
                child_request_id, child_sampling_params = parent_req.get_child_info(
                    output_idx
                )
                child_preproc = msgspec_replace(
                    parent_preproc,
                    request_id=child_request_id,
                    sampling_params=child_sampling_params,
                )
                self.output_processor.add_request(
                    child_preproc,
                    None,
                    parent_req=parent_req,
                    request_index=output_idx,
                )
                output_request_ids[output_idx] = child_request_id
                registered_request_ids.append(child_request_id)

        # llm_metrics totals; Rust postprocessor is bypassed on this path.
        input_tokens = len(tokens)
        cumulative_output_tokens = 0

        try:
            _inject_routing_metadata(dynamo_preproc, dynamo_preproc, mm_routing_info)
            with _nvtx.annotate("mm_frontend:routed_engine_generate", color="red"):
                dynamo_stream = await self.routed_engine.generate(
                    dynamo_preproc, context=context
                )

            rng_stream = _nvtx.start_range(
                "mm_frontend:stream_response", color="purple"
            )
            async for dynamo_response in dynamo_stream:
                if dynamo_response.is_error():
                    comments = dynamo_response.comments() or []
                    message = "; ".join(comments) or "unknown routed_engine error"
                    logger.error(
                        "routed_engine error for request %s: %s",
                        request_id,
                        message,
                    )
                    yield make_internal_error(request_id, message)
                    break
                engine_response = dynamo_response.data()

                if engine_response is None:
                    if dynamo_response.is_error():
                        yield handle_engine_error(engine_response, request_id, logger)
                        break
                    # No data or error fields, means we may have a comment or other kind of event.
                    # I'm not sure what those are used for, so TODO. Skip for now.
                    continue

                if "token_ids" not in engine_response:
                    yield handle_engine_error(engine_response, request_id, logger)
                    break

                # Count before any choice gate — tool/reasoning parsers may
                # consume tokens without emitting a visible delta.
                chunk_tokens = len(engine_response.get("token_ids") or [])
                cumulative_output_tokens += chunk_tokens

                output_idx = engine_response.get("index", 0) or 0
                output_request_id = output_request_ids.get(output_idx)
                if output_request_id is None:
                    yield {
                        "error": {
                            "message": (
                                f"Invalid engine choice index {output_idx} "
                                f"for request {request_id}"
                            ),
                            "type": "internal_error",
                        }
                    }
                    break

                raw_finish_reason = engine_response.get("finish_reason")
                finish_reason = map_finish_reason(raw_finish_reason)
                stop_reason = engine_response.get("stop_reason")

                output_kwargs: dict[str, Any] = {
                    "request_id": output_request_id,
                    "new_token_ids": engine_response["token_ids"],
                    "finish_reason": finish_reason,
                    "stop_reason": stop_reason,
                }
                output_fields = getattr(EngineCoreOutput, "__struct_fields__", ())
                if "is_segment_finished" in output_fields:
                    output_kwargs["is_segment_finished"] = engine_response.get(
                        "is_segment_finished", False
                    )
                if "new_prompt_len_snapshot" in output_fields:
                    output_kwargs["new_prompt_len_snapshot"] = engine_response.get(
                        "new_prompt_len_snapshot"
                    )
                vllm_response = EngineCoreOutput(**output_kwargs)

                vllm_out: OutputProcessorOutput = self.output_processor.process_outputs(
                    [vllm_response]
                )

                if vllm_out.reqs_to_abort:
                    pass

                choices = []
                postprocess_error = False
                if vllm_out.request_outputs:
                    for output in vllm_out.request_outputs[0].outputs:
                        post = post_processors.get(output.index)
                        if post is None:
                            yield {
                                "error": {
                                    "message": (
                                        f"Invalid postprocessor choice index {output.index} "
                                        f"for request {request_id}"
                                    ),
                                    "type": "internal_error",
                                }
                            }
                            postprocess_error = True
                            break
                        choice = post.process_output(output)
                        if choice:
                            choices.append(choice)

                if postprocess_error:
                    continue

                # One envelope per iteration carries both data and metrics so
                # client cancellation can't drop the annotation between yields.
                envelope: dict[str, Any] = {"_dynamo_annotated": True}
                if choices:
                    dynamo_out = {
                        "id": request_id,
                        "choices": choices,
                        "created": int(time.time()),
                        "model": request["model"],
                        "object": "chat.completion.chunk",
                    }
                    if usage := engine_response.get("completion_usage"):
                        dynamo_out["usage"] = usage
                    envelope["data"] = dynamo_out

                metrics = {
                    "input_tokens": input_tokens,
                    "output_tokens": cumulative_output_tokens,
                    "chunk_tokens": chunk_tokens,
                }
                envelope["event"] = "llm_metrics"
                envelope["comment"] = [json.dumps(metrics)]

                yield envelope
            _nvtx.end_range(rng_stream)
        except Exception as e:
            logger.exception("Error generating response for request %s", request_id)
            yield make_internal_error(request_id, str(e))
        finally:
            for output_request_id in registered_request_ids:
                if output_request_id in self.output_processor.request_states:
                    self.output_processor.abort_requests(
                        [output_request_id], internal=True
                    )


class EngineFactory:
    def __init__(
        self,
        config: FrontendConfig,
        flags: Namespace,
    ):
        self.config = config
        self.flags = flags
        self.stream_interval = 20
        raw_stream_interval = os.getenv("DYN_VLLM_STREAM_INTERVAL")
        if raw_stream_interval:
            try:
                self.stream_interval = max(1, int(raw_stream_interval))
            except ValueError:
                logger.warning(
                    "Invalid DYN_VLLM_STREAM_INTERVAL=%r, using default=%d",
                    raw_stream_interval,
                    self.stream_interval,
                )

    async def chat_engine_factory(
        self,
        instance_id: ModelCardInstanceId,
        mdc: ModelDeploymentCard,
        routed_engine: RoutedEngine,
    ) -> PythonAsyncEngine:
        """
        Called by Rust when a model is discovered.
        """
        model_type = mdc.model_type()
        if not model_type.supports_chat():
            raise RuntimeError(
                f"model type {model_type} is not supported by this factory"
            )
        if self.config.preprocess_workers != 0:
            raise RuntimeError(
                "preprocess_workers is not supported for vllm processor. "
                "Use the sglang processor for worker-pool preprocessing."
            )
        loop = asyncio.get_running_loop()

        local_dir = mdc.local_dir()
        if not os.path.isdir(local_dir):
            raise RuntimeError(
                f"MDC local_dir {local_dir!r} not populated for model {mdc.name()!r}; "
                f"download_config must run before the engine factory."
            )

        tokenizer_mode = getattr(self.flags, "tokenizer_mode", None) or "auto"
        config_format = getattr(self.flags, "config_format", None) or "auto"
        load_format = getattr(self.flags, "load_format", None) or "dummy"
        trust_remote_code = self.config.trust_remote_code
        enable_auto_tool_choice = getattr(self.flags, "enable_auto_tool_choice", False)

        model_config_kwargs = {
            "model": local_dir,
            "tokenizer_mode": tokenizer_mode,
            "config_format": config_format,
            "trust_remote_code": trust_remote_code,
        }
        context_length = _runtime_config_context_length(mdc)
        if context_length:
            os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
            model_config_kwargs["max_model_len"] = context_length
            logger.info(
                "vLLM frontend ModelConfig max_model_len=%d "
                "from runtime_config.context_length",
                context_length,
            )
        model_config = ModelConfig(**model_config_kwargs)
        # Use processor_only cache so tensor data persists across requests.
        # The default "lru" sender cache drops tensor data on cache hits
        # (designed for disagg where P1 holds tensors), but we need the
        # data to pickle and send via NIXL on repeated requests.
        if model_config.multimodal_config is not None:
            nixl_enabled = os.environ.get("DYNAMO_DISABLE_NIXL_MM", "") != "1"
            if nixl_enabled:
                model_config.multimodal_config.mm_processor_cache_type = (
                    "processor_only"
                )
        vllm_config = VllmConfig(
            model_config=model_config,
            load_config=LoadConfig(load_format=load_format),
            cache_config=CacheConfig(),
            # scheduler_config=SchedulerConfig(),
        )

        # Register dynamo's ImageLoader as vLLM's media connector so the
        # renderer uses our LRU cache + in-flight dedup for image fetching.
        # This eliminates data URI encoding overhead entirely.
        if os.environ.get("VLLM_MEDIA_CONNECTOR") != "dynamo":
            os.environ["VLLM_MEDIA_CONNECTOR"] = "dynamo"
        import dynamo.common.multimodal.media_connector  # noqa: F401

        input_processor = InputProcessor(vllm_config)
        tokenizer = input_processor.get_tokenizer()

        # vLLM's renderer skips its AutoProcessor fallback when tools are present,
        # so tool calls crash unless tokenizer.chat_template is set; load from disk.
        if tokenizer.chat_template is None:
            tokenizer.chat_template = resolve_chat_template(local_dir)

        # --chat-template overrides; load_chat_template accepts either a file path
        # or an inline Jinja template string.
        chat_template_flag = getattr(self.flags, "chat_template", None)
        if chat_template_flag:
            tokenizer.chat_template = load_chat_template(chat_template_flag)

        # Resolve stream_interval: env var override > backend config > default (20)
        stream_interval = self.stream_interval
        if not os.getenv("DYN_VLLM_STREAM_INTERVAL"):
            backend_interval = (
                mdc.runtime_config().get("runtime_data", {}).get("stream_interval")
            )
            if backend_interval is not None:
                try:
                    stream_interval = max(1, int(backend_interval))
                except (TypeError, ValueError):
                    logger.warning(
                        "Invalid stream_interval=%r from backend runtime_config, "
                        "using default=%d",
                        backend_interval,
                        stream_interval,
                    )

        output_processor = OutputProcessor(
            tokenizer,
            log_stats=False,
            stream_interval=stream_interval,
        )
        logger.info("vLLM OutputProcessor stream_interval=%d", stream_interval)

        runtime_config = mdc.runtime_config()
        if not isinstance(runtime_config, dict):
            runtime_config = {}

        tool_parser_name = self.flags.tool_call_parser
        if not tool_parser_name:
            runtime_tool_parser = runtime_config.get("tool_call_parser")
            tool_parser_name = (
                runtime_tool_parser.strip()
                if isinstance(runtime_tool_parser, str) and runtime_tool_parser.strip()
                else None
            )
        if tool_parser_name:
            configured_tool_parser = tool_parser_name
            tool_parser_name = normalize_vllm_tool_parser(configured_tool_parser)
            try:
                tool_parser_class = ToolParserManager.get_tool_parser(tool_parser_name)
            except KeyError as exc:
                raise ValueError(
                    f"Dynamo tool parser {configured_tool_parser!r} maps to vLLM "
                    f"parser {tool_parser_name!r}, which this vLLM version does "
                    "not support"
                ) from exc
        else:
            tool_parser_class = None

        worker_reasoning_parser = runtime_config.get("reasoning_parser")
        if not isinstance(worker_reasoning_parser, str):
            worker_reasoning_parser = None
        (
            reasoning_policy_name,
            effective_reasoning_parser,
            native_reasoning_parser,
        ) = _resolve_reasoning_parser_names(
            worker_reasoning_parser,
            self.flags.reasoning_parser,
        )
        reasoning_aware_guided_decoding = _validated_reasoning_aware_guided_decoding(
            mdc, effective_reasoning_parser
        )
        reasoning_parser_class = _reasoning_parser_class(
            reasoning_policy_name, native_reasoning_parser
        )

        block_size = self.config.kv_cache_block_size or 16

        gen = VllmProcessor(
            tokenizer,
            input_processor,
            output_processor,
            tool_parser_class,
            reasoning_parser_class,
            reasoning_policy_name,
            routed_engine,
            block_size=block_size,
            enable_auto_tool_choice=enable_auto_tool_choice,
            reasoning_aware_guided_decoding=reasoning_aware_guided_decoding,
            tool_call_id_type=get_tool_call_id_type(model_config),
            model_config=model_config,
        )
        gen.exclude_tools_when_tool_choice_none = (
            self.config.exclude_tools_when_tool_choice_none
        )

        return PythonAsyncEngine(gen.generator, loop)
