# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM worker handling for the ``/classify`` and ``/pooling`` APIs.

This module is intentionally separate from the generation and embeddings
handlers. Pooling-family models use vLLM's pooling runner and do not need the
KV-cache, disaggregation, or multimodal generation setup used by those paths.
"""

import asyncio
import base64
import importlib
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, AsyncIterator, Final, Optional, cast

import torch
from vllm import PoolingParams
from vllm.config import ModelConfig
from vllm.inputs import TextPrompt, TokensPrompt

from dynamo._core import Context
from dynamo.llm import ModelType

from .args import Config
from .handlers import EmbeddingWorkerHandler

logger = logging.getLogger(__name__)

_POOLING_BATCH_CONCURRENCY_FALLBACK: Final = 256
_POOLING_EMBED_DTYPES: Final = frozenset(
    ("float32", "float16", "bfloat16", "fp8_e4m3", "fp8_e5m2")
)
_POOLING_ENDIANNESS: Final = frozenset(("native", "big", "little"))


class ClassifyWorkerHandler(EmbeddingWorkerHandler):
    """Handle ``/classify`` and ``/pooling`` on a vLLM pooling engine.

    The handler reuses the embeddings worker's engine lifecycle and abort
    monitoring, but supplies the request-specific pooling task and response
    shape. The Rust pooling wire request always contains ``encoding_format``;
    the classify wire request does not, which provides an unambiguous internal
    dispatch key.
    """

    def __init__(
        self,
        runtime: Any,
        engine: Any,
        config: Config,
        model_config: ModelConfig | None = None,
        shutdown_event: Optional[asyncio.Event] = None,
        generate_endpoint=None,
    ) -> None:
        super().__init__(
            runtime=runtime,
            engine=engine,
            config=config,
            shutdown_event=shutdown_event,
            generate_endpoint=generate_endpoint,
            model_config=model_config,
        )
        self._default_pooling_task: str | None = None
        logger.info("Classify worker handler initialized")

    def _lora_model_type(self) -> ModelType:
        """One worker registers both pooling-family types, so adapter cards
        must carry the same pair as the base card."""
        return ModelType.Classify | ModelType.Pooling

    def _id2label(self) -> dict[int, str]:
        """Return the model's Hugging Face label map with integer keys."""
        hf_config = getattr(self.model_config, "hf_config", None)
        raw = getattr(hf_config, "id2label", None) or {}
        normalized: dict[int, str] = {}
        for key, value in raw.items():
            try:
                normalized[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return normalized

    async def _resolve_pooling_task(self, requested_task: str | None) -> str:
        if requested_task is not None:
            return requested_task
        if self._default_pooling_task is not None:
            return self._default_pooling_task
        if self.model_config is None:
            raise RuntimeError("Model config is required to resolve the pooling task")

        supported_tasks = await self.engine_client.get_supported_tasks()
        pooling_task = self.model_config.get_pooling_task(supported_tasks)
        if pooling_task is None:
            raise ValueError(
                f"No default pooling task is available; supported tasks: {supported_tasks}"
            )
        self._default_pooling_task = pooling_task
        return pooling_task

    async def _run_pooling_batch(
        self,
        prompts: list[Any],
        encode_one: Callable[[int, Any], Awaitable[Any]],
    ) -> list[Any]:
        """Run a request batch without exceeding vLLM scheduler capacity."""
        scheduler_config = getattr(
            getattr(self.engine_client, "vllm_config", None),
            "scheduler_config",
            None,
        )
        max_num_seqs = getattr(scheduler_config, "max_num_seqs", None)
        if (
            not isinstance(max_num_seqs, int)
            or isinstance(max_num_seqs, bool)
            or max_num_seqs < 1
        ):
            max_num_seqs = _POOLING_BATCH_CONCURRENCY_FALLBACK

        outputs: list[Any | None] = [None] * len(prompts)
        indices = iter(range(len(prompts)))

        async def _worker() -> None:
            for idx in indices:
                outputs[idx] = await encode_one(idx, prompts[idx])

        tasks = [
            asyncio.create_task(_worker())
            for _ in range(min(len(prompts), max_num_seqs))
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            pending = [task for task in tasks if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        if any(output is None for output in outputs):
            raise RuntimeError("vLLM engine.encode did not return every batch item")
        return cast(list[Any], outputs)

    async def _encode_batch(
        self,
        request: dict[str, Any],
        context: Context,
        prompts: list[Any],
        pooling_params: PoolingParams,
    ) -> list[Any]:
        """Encode all inputs with common request controls and preprocessing."""
        tokenization_kwargs = _build_tokenization_kwargs(request)
        tokenize_params = _build_pooling_tokenize_params(
            self.engine_client, tokenization_kwargs
        )
        tokenizer = getattr(self.engine_client.renderer, "tokenizer", None)
        engine_request_id = context.id()
        priority = request.get("priority", 0)
        # Resolved once per request: every prompt in the batch names the same
        # model. Raises for an unknown non-base name rather than quietly
        # pooling with the base weights.
        lora_request = self._resolve_lora_request(
            request.get("model") or self.config.served_model_name or ""
        )
        self._track_lora_request_activation(lora_request)

        async def _encode_one(idx: int, prompt: Any) -> Any:
            request_id = f"{engine_request_id}-{idx}"
            encode_arg = _prepare_pooling_prompt(
                prompt,
                request,
                tokenize_params,
                tokenizer,
            )
            final_output = None
            async with self._abort_monitor(context, request_id):
                encode_kwargs: dict[str, Any] = {
                    "prompt": encode_arg,
                    "pooling_params": pooling_params,
                    "request_id": request_id,
                }
                # Omitting this silently pools with the base model for a
                # request that named an adapter.
                if lora_request is not None:
                    encode_kwargs["lora_request"] = lora_request
                if priority != 0:
                    encode_kwargs["priority"] = priority
                # Token-ID prompts have already been passed through vLLM's
                # TokenizeParams. Text prompts are tokenized inside AsyncLLM.
                if tokenization_kwargs is not None and isinstance(prompt, str):
                    encode_kwargs["tokenization_kwargs"] = tokenization_kwargs

                async for output in self.engine_client.encode(**encode_kwargs):
                    final_output = output

            if final_output is None:
                raise RuntimeError(
                    f"vLLM engine.encode produced no output for input index {idx}"
                )
            return final_output

        return await self._run_pooling_batch(prompts, _encode_one)

    async def generate(
        self, request: dict, context: Context
    ) -> AsyncIterator[dict[str, Any]]:
        """Handle one internal classify or pooling wire request."""
        if "encoding_format" in request:
            async for response in self._generate_pooling(request, context):
                yield response
            return

        model_name = request.get("model") or self.config.served_model_name or ""
        input_field = request.get("input")
        if input_field is None:
            raise ValueError("Classify request missing required 'input' field")
        prompts = _classify_pooling_input(input_field)

        pooling_kwargs: dict[str, Any] = {"task": "classify"}
        use_activation = request.get("use_activation")
        if use_activation is not None:
            pooling_kwargs["use_activation"] = use_activation
        pooling_params = PoolingParams(**pooling_kwargs)
        outputs = await self._encode_batch(request, context, prompts, pooling_params)

        id2label = self._id2label()
        data: list[dict[str, Any]] = []
        prompt_tokens = 0
        for idx, final_output in enumerate(outputs):
            probs = _classification_output_to_list(final_output.outputs.data)
            predicted_index = (
                max(range(len(probs)), key=probs.__getitem__) if probs else None
            )
            data.append(
                {
                    "index": idx,
                    "label": (
                        id2label.get(predicted_index)
                        if predicted_index is not None
                        else None
                    ),
                    "probs": probs,
                    "num_classes": len(probs),
                }
            )
            token_ids = getattr(final_output, "prompt_token_ids", None) or []
            prompt_tokens += len(token_ids)

        response_request_id = request.get("request_id") or context.id()
        yield {
            "id": f"classify-{response_request_id}",
            "object": "list",
            "created": int(time.time()),
            "model": model_name,
            "data": data,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens,
                "completion_tokens": 0,
            },
        }

    async def _generate_pooling(
        self, request: dict[str, Any], context: Context
    ) -> AsyncIterator[dict[str, Any]]:
        """Handle one internal pooling wire request."""
        model_name = request.get("model") or self.config.served_model_name or ""
        input_field = request.get("input")
        if input_field is None:
            raise ValueError("Pooling request missing required 'input' field")
        prompts = _classify_pooling_input(input_field)

        if request.get("dimensions") is not None:
            raise ValueError("dimensions is currently not supported")

        encoding_format = request.get("encoding_format", "float")
        if encoding_format not in ("float", "base64", "bytes", "bytes_only"):
            raise ValueError(
                f"Invalid 'encoding_format' value {encoding_format!r}; "
                "expected 'float', 'base64', 'bytes', or 'bytes_only'"
            )

        embed_dtype = request.get("embed_dtype", "float32")
        if not isinstance(embed_dtype, str) or embed_dtype not in _POOLING_EMBED_DTYPES:
            raise ValueError(f"Invalid 'embed_dtype' value {embed_dtype!r}")

        endianness = request.get("endianness", "native")
        if not isinstance(endianness, str) or endianness not in _POOLING_ENDIANNESS:
            raise ValueError(f"Invalid 'endianness' value {endianness!r}")

        pooling_task = await self._resolve_pooling_task(request.get("task"))
        pooling_kwargs: dict[str, Any] = {"task": pooling_task}
        use_activation = request.get("use_activation")
        if use_activation is not None:
            pooling_kwargs["use_activation"] = use_activation
        pooling_params = PoolingParams(**pooling_kwargs)
        outputs = await self._encode_batch(request, context, prompts, pooling_params)

        data: list[dict[str, Any]] = []
        prompt_tokens = 0
        for idx, final_output in enumerate(outputs):
            item: dict[str, Any] = {
                "index": idx,
                "object": "pooling",
            }
            if encoding_format == "float":
                item["data"] = _pooling_output_to_nested(final_output.outputs.data)
            else:
                encoded, shape = _encode_pooling_output(
                    final_output.outputs.data,
                    embed_dtype,
                    endianness,
                )
                item["data"] = encoded
                if encoding_format == "bytes":
                    item["shape"] = shape
            data.append(item)
            token_ids = getattr(final_output, "prompt_token_ids", None) or []
            prompt_tokens += len(token_ids)

        response_request_id = request.get("request_id") or context.id()
        yield {
            "id": f"pool-{response_request_id}",
            "object": "list",
            "created": int(time.time()),
            "model": model_name,
            "data": data,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens,
                "completion_tokens": 0,
            },
        }


def _build_pooling_tokenize_params(
    engine_client: Any, tokenization_kwargs: dict[str, Any] | None
) -> Any:
    """Resolve the TokenizeParams applied to pre-tokenized inputs.

    With no caller overrides this returns the renderer's defaults rather than
    ``None``: upstream vLLM runs every pooling/classify input through
    ``apply_post_tokenization``, which is also where default truncation to
    ``max_model_len`` happens. Returning ``None`` here would skip that for
    token-ID inputs, so an over-long pre-tokenized prompt would reach the engine
    untruncated instead of being truncated the way native vLLM truncates it.
    """
    defaults = engine_client.renderer.default_cmpl_tok_params
    if tokenization_kwargs is None:
        return defaults
    return defaults.with_kwargs(**tokenization_kwargs)


def _prepare_pooling_prompt(
    prompt: Any,
    request: dict[str, Any],
    tokenize_params: Any,
    tokenizer: Any,
) -> Any:
    """Build a vLLM prompt and preprocess token-ID inputs with its renderer."""
    mm_processor_kwargs = request.get("mm_processor_kwargs")
    cache_salt = request.get("cache_salt")
    if isinstance(prompt, str):
        if mm_processor_kwargs is None and cache_salt is None:
            return prompt
        encode_arg: Any = TextPrompt(prompt=prompt)
    else:
        encode_arg = TokensPrompt(prompt_token_ids=prompt)

    if mm_processor_kwargs is not None:
        encode_arg["mm_processor_kwargs"] = mm_processor_kwargs
    if cache_salt is not None:
        encode_arg["cache_salt"] = cache_salt

    # AsyncLLM's compatibility preprocessor does not apply tokenization kwargs
    # to TokensPrompt. Run the same vLLM TokenizeParams post-tokenization path
    # explicitly, including the tokenizer's configured default truncation side.
    if tokenize_params is not None and not isinstance(prompt, str):
        encode_arg = tokenize_params.apply_post_tokenization(tokenizer, encode_arg)
    return encode_arg


def _build_tokenization_kwargs(request: dict[str, Any]) -> dict[str, Any] | None:
    """Validate and collect vLLM completion tokenization controls."""
    kwargs: dict[str, Any] = {}

    truncate_prompt_tokens = request.get("truncate_prompt_tokens")
    if truncate_prompt_tokens is not None:
        if not isinstance(truncate_prompt_tokens, int) or isinstance(
            truncate_prompt_tokens, bool
        ):
            raise TypeError(
                "Invalid 'truncate_prompt_tokens' type "
                f"{type(truncate_prompt_tokens).__name__}; expected int"
            )
        if truncate_prompt_tokens < -1:
            raise ValueError(
                f"truncate_prompt_tokens must be >= -1, got {truncate_prompt_tokens}"
            )
        kwargs["truncate_prompt_tokens"] = truncate_prompt_tokens

    truncation_side = request.get("truncation_side")
    if truncation_side is not None:
        if truncation_side not in ("left", "right"):
            raise ValueError(
                f"Invalid 'truncation_side' value {truncation_side!r}; "
                "expected 'left' or 'right'"
            )
        kwargs["truncation_side"] = truncation_side

    add_special_tokens = request.get("add_special_tokens")
    if add_special_tokens is not None:
        if not isinstance(add_special_tokens, bool):
            raise TypeError(
                "Invalid 'add_special_tokens' type "
                f"{type(add_special_tokens).__name__}; expected bool"
            )
        kwargs["add_special_tokens"] = add_special_tokens

    return kwargs or None


def _is_token_id(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _classify_pooling_input(input_field: Any) -> list[Any]:
    """Normalize the four vLLM completion input forms without coercion."""
    if isinstance(input_field, str):
        return [input_field]
    if not isinstance(input_field, list):
        raise TypeError(
            f"Invalid 'input' type {type(input_field).__name__}; "
            "expected str, list[str], list[int], or list[list[int]]"
        )
    if not input_field:
        raise ValueError("Request 'input' must be non-empty")

    first = input_field[0]
    if isinstance(first, str):
        texts: list[str] = []
        for item in input_field:
            if not isinstance(item, str):
                raise TypeError(
                    "'input' list mixes str and non-str entries; pass either "
                    "all strings or all token-ID arrays"
                )
            texts.append(item)
        return texts

    if _is_token_id(first):
        token_ids: list[int] = []
        for item in input_field:
            if not _is_token_id(item):
                raise TypeError(
                    "'input' list mixes int and non-int entries; for tokenized "
                    "input pass all integers or list[list[int]]"
                )
            token_ids.append(item)
        return [token_ids]

    if isinstance(first, list):
        prompts: list[list[int]] = []
        for idx, item in enumerate(input_field):
            if not isinstance(item, list) or not all(
                _is_token_id(token_id) for token_id in item
            ):
                raise TypeError(
                    f"'input' list element at index {idx} must be a list of "
                    "ints (token IDs); mixed batches are not supported"
                )
            prompts.append(item)
        return prompts

    raise TypeError(
        f"Unsupported 'input' element type {type(first).__name__}; "
        "expected str, int, or list[int]"
    )


def _classification_output_to_list(data: Any) -> list[float]:
    """Convert sequence-level classification output to one probability vector.

    ``/classify`` reports a single vector of shape ``[num_classes]``. vLLM's
    pooling pipeline may wrap it in a singleton batch dim (``[1, num_classes]``),
    which is unwrapped here. Any other multi-row shape means the model or pooler
    produced token-level output: flattening it would report ``num_classes`` as
    ``rows * classes`` and pick an ``argmax`` outside the label range, yielding a
    200 with a null label instead of a visible failure. Native vLLM rejects
    classification output with ``ndim != 1``; so does this.
    """
    if isinstance(data, torch.Tensor):
        tensor = data.detach().cpu()
        if tensor.ndim == 2 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim != 1:
            raise ValueError(
                "Classification output must be one probability vector of shape "
                f"[num_classes]; got tensor of shape {tuple(tensor.shape)}. "
                "This usually means the model or pooler emits token-level output."
            )
        return tensor.tolist()
    if isinstance(data, (list, tuple)):
        if data and isinstance(data[0], (list, tuple)):
            if len(data) != 1:
                raise ValueError(
                    "Classification output must be one probability vector of "
                    f"shape [num_classes]; got {len(data)} rows. This usually "
                    "means the model or pooler emits token-level output."
                )
            return [float(value) for value in data[0]]
        return [float(value) for value in data]
    raise TypeError(
        f"Unsupported PoolingOutput.data type {type(data).__name__}; "
        "expected torch.Tensor or list"
    )


def _pooling_output_to_nested(data: Any) -> Any:
    """Convert pooling output to JSON values while preserving tensor shape."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().tolist()
    if isinstance(data, (list, tuple)):
        return [
            _pooling_output_to_nested(value)
            if isinstance(value, (list, tuple))
            else float(value)
            for value in data
        ]
    raise TypeError(
        f"Unsupported PoolingOutput.data type {type(data).__name__}; "
        "expected torch.Tensor or list"
    )


def _encode_pooling_output(
    data: Any, embed_dtype: str, endianness: str
) -> tuple[str, list[int]]:
    """Encode a pooling tensor with vLLM's binary response implementation."""
    if isinstance(data, torch.Tensor):
        tensor = data.detach().cpu()
    elif isinstance(data, (list, tuple)):
        tensor = torch.tensor(data)
    else:
        raise TypeError(
            f"Unsupported PoolingOutput.data type {type(data).__name__}; "
            "expected torch.Tensor or list"
        )

    serial_utils = importlib.import_module("vllm.utils.serial_utils")
    binary = serial_utils.tensor2binary(tensor, embed_dtype, endianness)
    return base64.b64encode(binary).decode("ascii"), list(tensor.shape)
