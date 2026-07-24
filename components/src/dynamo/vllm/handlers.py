# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import importlib
import inspect
import logging
import math
import os
import struct
import tempfile
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Mapping
from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Final,
    Generic,
    Iterator,
    NoReturn,
    Optional,
    TypeVar,
    cast,
)

import torch
from vllm import PoolingParams
from vllm.config import ModelConfig, VllmConfig
from vllm.inputs import EmbedsPrompt, TextPrompt, TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.renderers.embed_utils import safe_load_prompt_embeds
from vllm.sampling_params import (
    RequestOutputKind,
    SamplingParams,
    StructuredOutputsParams,
)
from vllm.v1.engine.exceptions import EngineDeadError

from dynamo._core import Context
from dynamo.common.backend import logprobs as _shared_logprobs
from dynamo.common.lora.manager import LoRAInfo, get_lora_manager
from dynamo.common.memory.multimodal_embedding_cache_manager import (
    MultimodalEmbeddingCacheManager,
)
from dynamo.common.multimodal.embedding_transfer import (
    LocalEmbeddingReceiver,
    NixlReadEmbeddingReceiver,
    NixlWriteEmbeddingReceiver,
)
from dynamo.common.rl import (
    RLAdminValidationError,
    RLRouteRegistry,
    env_bool,
    require_lora_load_request,
    require_lora_unload_request,
)
from dynamo.common.utils import nvtx_utils as _nvtx
from dynamo.common.utils.engine_response import normalize_finish_reason
from dynamo.common.utils.input_params import InputParamManager
from dynamo.common.utils.structural_tag import serialize_structural_tag
from dynamo.common.utils.time_section import time_and_log_code_section
from dynamo.llm import (
    KvEventPublisher,
    ModelInput,
    ModelRuntimeConfig,
    ModelType,
    WorkerType,
    lora_name_to_id,
    register_model,
    unregister_model,
)
from dynamo.llm.exceptions import EngineShutdown
from dynamo.runtime import Client
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.vllm.kv_connector_protocols import (
    KvConnectorProtocol,
    make_kv_connector_protocol,
)

from .args import Config
from .cache_info import get_configured_kv_event_block_size
from .constants import DisaggregationMode, EmbeddingTransferMode
from .engine_monitor import VllmEngineMonitor
from .multimodal_utils.async_vision_encoder import AsyncVisionEncoder
from .multimodal_utils.custom_encoder_adapter import (
    CustomEncoderAdapter,
    create_custom_encoder_adapter,
)
from .multimodal_utils.prefill_worker_utils import MultiModalEmbeddingLoader
from .multimodal_utils.request_processor import (
    IMAGE_URL_KEY,
    URL_VARIANT_KEY,
    MissingMultimodalHandoffError,
    VllmMultimodalRequestProcessor,
)
from .multimodal_utils.vision_encoder_backend import VisionEncoderBackend

configure_dynamo_logging()
logger = logging.getLogger(__name__)

_GENERATE_REASONING_SUPPORT_CACHE_ATTR = "_dynamo_generate_reasoning_support"
_DELTA_REQUEST_OUTPUT_KIND = RequestOutputKind.DELTA
_RL_INIT_WEIGHTS_TIMEOUT_ENV = "DYN_RL_INIT_WEIGHTS_TIMEOUT_S"
_RL_INIT_WEIGHTS_TIMEOUT_DEFAULT_S = 30.0
_LORA_LOCK_STRIPES = 64
_DISTRIBUTED_WEIGHT_UPDATE_RESERVED_KEYS: Final = frozenset(
    {
        "allow_unpaused",
        "engine_rpc",
        "reset_prefix_cache",
        "weight_version",
    }
)


def build_prompt_tokens_details(
    num_cached_tokens: int | None,
) -> dict[str, int] | None:
    """Preserve the distinction between unavailable and zero cached tokens."""
    if num_cached_tokens is None:
        return None
    return {"cached_tokens": num_cached_tokens}


def _rl_init_weights_timeout_s() -> float:
    return float(
        os.environ.get(
            _RL_INIT_WEIGHTS_TIMEOUT_ENV,
            str(_RL_INIT_WEIGHTS_TIMEOUT_DEFAULT_S),
        )
    )


class _DeferredAbort:
    """Defers engine_client.abort(request_id) until the first engine output.

    In disaggregated decode mode, calling engine_client.abort() while a NIXL
    KV transfer is still in flight on the decode worker can crash EngineCore.
    This guard delays the real abort call until the first generation result
    has been yielded (which, for a decode worker, means the KV transfer has
    completed and the engine has produced at least one token).

    When abort() is called before the first token, a background asyncio.Task
    is spawned to wait for the first-token signal and then perform the real
    abort.
    """

    def __init__(
        self,
        engine_client: Any,
        request_id: str,
        on_engine_dead: Optional[Any] = None,
    ):
        self._engine_client = engine_client
        self._request_id = request_id
        # Escalation hook invoked if the (possibly deferred/background) engine
        # abort hits EngineDeadError, so engine death shuts the runtime down even
        # when the abort runs outside the request's own error handling (the
        # disconnect-monitor or deferred-after-first-token path).
        self._on_engine_dead = on_engine_dead
        self._first_token_received = False
        self._first_token_event = asyncio.Event()
        # Strong reference to the deferred-abort background task so it is not
        # garbage collected mid-execution (asyncio.create_task only holds a
        # weak reference via the event loop).
        self._abort_task: Optional[asyncio.Task] = None
        # Exception the real engine abort raised (if it has run), so the admin
        # abort_request route can report failure instead of a false "ok".
        self._abort_exc: Optional[BaseException] = None

    def signal_first_token(self) -> None:
        """Called when the first engine output for the request is received."""
        if not self._first_token_received:
            self._first_token_received = True
            self._first_token_event.set()

    async def abort(self) -> None:
        """Abort the request. Creates a Task to hold a strong reference so the
        engine abort cannot be dropped if this coroutine is concurrently
        cancelled."""
        if self._abort_task is None:
            if self._first_token_received:
                logger.debug(
                    f"Deferred abort: first token already received, "
                    f"aborting request {self._request_id} now"
                )
                self._abort_task = asyncio.create_task(self._run_abort())
            else:
                logger.debug(
                    f"Deferred abort: first token not received for request "
                    f"{self._request_id}, spawning background task"
                )
                self._abort_task = asyncio.create_task(self._wait_and_abort())
        # Only block on completion when the abort runs immediately (post first
        # token). A pre-first-token deferred abort fires in the background when
        # the first token arrives; awaiting it here would hang the caller (admin
        # route or disconnect monitor) until — or unless — generation produces
        # output. _abort_task keeps the background task alive; close() reaps it.
        if not self._first_token_received:
            return
        try:
            # shield() so that if the caller (e.g. a cancelled system-route
            # request or a disconnected client) is cancelled while awaiting, the
            # cancellation is NOT propagated into the abort task — it really does
            # continue in the background. A bare `await self._abort_task` would
            # cancel the task too, silently dropping the abort.
            await asyncio.shield(self._abort_task)
        except asyncio.CancelledError:
            logger.debug(
                f"Deferred abort: shielded from cancellation for request "
                f"{self._request_id}, abort continues in background"
            )

    async def _run_abort(self) -> None:
        """Execute engine.abort() and emit the canonical completion log."""
        try:
            await self._engine_client.abort(self._request_id)
            logger.debug(f"Aborted Request ID: {self._request_id}")
        except Exception as e:
            # Record so abort_request can report the failure rather than a false
            # success. Also escalate engine death here, since a deferred or
            # disconnect-monitor abort runs in the background with no caller
            # awaiting the result to handle EngineDeadError.
            self._abort_exc = e
            logger.warning(
                f"Deferred abort: engine abort raised for request "
                f"{self._request_id}: {e}"
            )
            if isinstance(e, EngineDeadError) and self._on_engine_dead is not None:
                self._on_engine_dead(e)

    async def _wait_and_abort(self) -> None:
        """Background task: wait for first token, then abort."""
        try:
            await self._first_token_event.wait()
        except Exception:
            pass
        await self._run_abort()

    async def close(self) -> None:
        """Clean up the deferred-abort waiter when generation exits.

        Handles case 1b: if abort() was requested before the first token
        arrived AND the generation loop exits without ever producing output,
        the background _wait_and_abort task would otherwise remain parked on
        first_token_event.wait() forever. Cancel it so it does not leak.

        Safety invariant: this method must NOT call engine_client.abort() in
        the pre-first-token window. Issuing abort before the engine has
        produced output is exactly what this guard exists to avoid (it can
        crash EngineCore while a NIXL KV transfer is still in flight on the
        decode worker).
        """
        if self._abort_task is None:
            return

        if not self._first_token_received:
            # Case 1b: cancel the local waiter without firing the real abort.
            self._abort_task.cancel()

        try:
            # shield so that if cleanup is awaiting a real post-first-token abort
            # and the caller is cancelled, the abort still completes (the
            # pre-first-token path was cancelled just above and resolves here).
            await asyncio.shield(self._abort_task)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(
                f"Deferred abort: cleanup observed error for request "
                f"{self._request_id}: {e}"
            )


@asynccontextmanager
async def _deferred_abort_guard(
    engine_client: Any,
    request_id: str,
    is_decode_only: bool,
    registry: Optional[dict[str, "_DeferredAbort"]] = None,
    on_engine_dead: Optional[Any] = None,
) -> AsyncIterator[Optional[_DeferredAbort]]:
    """Own the _DeferredAbort lifecycle for a single request.

    Yields a _DeferredAbort in disaggregated-decode mode, otherwise yields
    None. On exit, awaits guard.close() so the background waiter cannot leak
    when generation finishes without producing output (case 1b). close() is
    specifically designed not to call engine_client.abort() in the unsafe
    pre-first-token window.

    When `registry` is provided, the guard registers itself under `request_id`
    for the request's lifetime so out-of-band callers (the admin abort_request
    route) can route their abort through this same deferred path instead of
    calling engine_client.abort() directly in the unsafe window.
    """
    guard = (
        _DeferredAbort(engine_client, request_id, on_engine_dead)
        if is_decode_only
        else None
    )
    if guard is not None and registry is not None:
        registry[request_id] = guard
    try:
        yield guard
    finally:
        if guard is not None:
            # Keep the guard registered until close() finishes: close() may
            # await a deferred abort, and an out-of-band admin abort_request
            # during that window must still find the guard and route through
            # the deferred path instead of taking the unsafe direct abort.
            try:
                await guard.close()
            finally:
                if registry is not None:
                    registry.pop(request_id, None)


class VllmEnginePauseController:
    def __init__(self, engine_client: Any):
        self._engine_client = engine_client
        self._is_paused = False
        self._generation_paused = False

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    @property
    def needs_resume_recovery(self) -> bool:
        return self._generation_paused

    async def pause(self, *args: object) -> bool:
        if self._is_paused or self._generation_paused:
            return False

        level = args[0] if args else None
        await self._engine_client.pause_generation()
        self._generation_paused = True
        try:
            if level is None:
                await self._engine_client.sleep()
            else:
                await self._engine_client.sleep(level)
        except Exception:
            try:
                await self._engine_client.resume_generation()
                self._generation_paused = False
            except Exception:
                logger.exception(
                    "Failed to resume generation after native vLLM sleep failure"
                )
            raise
        self._is_paused = True
        return True

    async def resume(self, tags: list[str] | None = None) -> bool:
        if not self._is_paused and not self._generation_paused:
            return False

        if self._is_paused:
            if tags is None:
                await self._engine_client.wake_up()
            else:
                await self._engine_client.wake_up(tags)
        if self._generation_paused:
            await self._engine_client.resume_generation()
            self._generation_paused = False
        return True

    def mark_resumed(self) -> None:
        self._is_paused = False
        self._generation_paused = False


# Helpers for nvext response fields requested through `nvext.extra_fields`.


# Logprobs can be -inf (log of probability 0) for masked/disallowed tokens (e.g.
# via bad_words_token_ids / allowed_token_ids) or full-vocab prompt logprobs.
# JSON has no inf/nan, so pythonize -> serde_json rewrites them to `null`, which
# then fails typed deserialization on the Rust side and SILENTLY DROPS the whole
# logprobs payload. Clamp non-finite logprobs to a large finite-negative
# sentinel so the value survives transport while still meaning "effectively
# impossible".
_MIN_FINITE_LOGPROB = -1e30


def _finite_logprob(value: Any) -> float:
    lp = float(value)
    return lp if math.isfinite(lp) else _MIN_FINITE_LOGPROB


def _serialize_prompt_logprobs(
    raw_prompt_logprobs: list,
) -> list:
    """Convert vLLM's ``RequestOutput.prompt_logprobs`` into the dict shape
    expected by Dynamo's Rust ``PromptLogprobEntry`` (serde deserialization).

    vLLM shape: ``list[dict[int, Logprob] | None]`` where ``Logprob`` has
    ``.logprob``, ``.rank``, ``.decoded_token`` attributes.

    Output shape: ``list[dict[str, {"logprob": float, ...}] | None]``.
    Workers carry it under ``engine_data.prompt_logprobs`` so the Rust
    postprocessor can surface it on ``NvExtResponse.prompt_logprobs`` without
    changing the public ``LLMEngineOutput`` struct shape.

    NOTE: token_id keys are emitted as **strings** (not ints) so that
    pythonize → serde → JSON survives the worker→frontend transport. JSON
    object keys are required to be strings; ``HashMap<u32, _>`` on the Rust
    side deserializes string keys via ``u32::from_str``. Emitting int keys
    here causes the chunk to be silently dropped on pythonize → JSON, which
    surfaces as ``"Stream ended before generation completed"`` on the
    frontend (worker emits cleanly, frontend never sees ``complete_final``).
    """
    result: list = []
    for entry in raw_prompt_logprobs:
        if entry is None:
            result.append(None)
        else:
            converted: Dict[str, Dict[str, Any]] = {}
            for token_id, logprob_obj in entry.items():
                try:
                    key = str(int(token_id))
                except (TypeError, ValueError):
                    # vLLM only emits int token-id keys; skip a non-int key
                    # rather than aborting the whole prompt_logprobs payload.
                    continue
                lp_dict: Dict[str, Any] = {
                    "logprob": _finite_logprob(logprob_obj.logprob),
                }
                rank = getattr(logprob_obj, "rank", None)
                if rank is not None:
                    lp_dict["rank"] = int(rank)
                decoded = getattr(logprob_obj, "decoded_token", None)
                if decoded is not None:
                    lp_dict["decoded_token"] = decoded
                converted[key] = lp_dict
            result.append(converted)
    return result


def _attach_prompt_logprobs_engine_data(
    tok: Dict[str, Any], prompt_logprobs: list
) -> None:
    engine_data = tok.setdefault("engine_data", {})
    if isinstance(engine_data, dict):
        engine_data["prompt_logprobs"] = prompt_logprobs


def _attach_routed_experts_engine_data(
    tok: Dict[str, Any], routed_experts: Dict[str, Any]
) -> None:
    # routed_experts rides the engine's opaque engine_data passthrough (surfaced
    # by the Rust postprocessor as nvext.routed_experts); disaggregated_params
    # stays KV-transfer only. Merge via setdefault so we don't clobber any
    # prompt_logprobs already attached to engine_data on the final chunk.
    engine_data = tok.setdefault("engine_data", {})
    if isinstance(engine_data, dict):
        engine_data["routed_experts"] = routed_experts


def _iter_nvext_sources(request: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yield each nvext dict on the request, in priority order:

      1. ``request["nvext"]``               — raw OpenAI shape (SGLang / tests).
      2. ``request["extra_args"]["nvext"]`` — what the Rust preprocessor stashes
         when building PreprocessedRequest from an OpenAI request
         (PreprocessedRequest itself has no nvext field).

    Centralizing this lookup keeps ``_nvext_extra_field_requested``,
    ``_is_token_in_request`` and ``_apply_nvext_cache_salt`` consistent so an
    nvext field is never honored by one helper but silently missed by another.
    """
    extra_args = request.get("extra_args")
    for source in (
        request.get("nvext"),
        extra_args.get("nvext") if isinstance(extra_args, dict) else None,
    ):
        if isinstance(source, dict):
            yield source


def _nvext_extra_field_requested(request: Dict[str, Any], field: str) -> bool:
    """Return True iff the request opted into the given nvext extra field."""
    return any(
        isinstance(source.get("extra_fields"), list) and field in source["extra_fields"]
        for source in _iter_nvext_sources(request)
    )


# Must match DYNAMO_CACHE_SALT_PREFIX in lib/kv-router/src/zmq_wire/extra_keys.rs.
_DYNAMO_CACHE_SALT_PREFIX = "dynamo-cache-salt:"


def _apply_nvext_cache_salt(request: Dict[str, Any], prompt: Any) -> None:
    """Pass an internally tagged cache salt to vLLM.

    vLLM publishes cache salts as otherwise-untyped strings in ``extra_keys``
    alongside LoRA and multimodal metadata. The tag lets Dynamo recover the
    namespace without guessing from the user-controlled value. It is removed
    again by the Rust KV-event decoder, so Dynamo's public namespace is
    unchanged.
    """
    if not isinstance(prompt, dict):
        return
    for source in _iter_nvext_sources(request):
        cache_salt = source.get("cache_salt")
        if cache_salt:
            prompt["cache_salt"] = f"{_DYNAMO_CACHE_SALT_PREFIX}{cache_salt}"
            return


def _prompt_token_ids_for_engine_data(
    request: Dict[str, Any],
    prompt: Any,
) -> list[int]:
    prompt_token_ids = (
        prompt.get("prompt_token_ids")
        if isinstance(prompt, dict)
        else request.get("token_ids")
    )
    return list(prompt_token_ids or [])


def _flatten_logprobs(
    log_probs: Any,
) -> Optional[list[float]]:
    """Coerce a per-token logprob sequence into a flat list[float].

    The backend's per-chunk `log_probs` field is one float per emitted
    token (the logprob the engine sampled). Some upstream paths wrap it
    in dicts or nested lists; this helper accepts:
      - list[float]               -> returned as-is
      - list[list[float]]         -> flattened
      - list[dict{logprob: ...}]  -> .logprob extracted
      - None                      -> None

    Any element that isn't coercible to float is dropped silently rather
    than poisoning the entire payload. The trainer treats missing
    per-token logprobs as off-policy correction = 0 for that token, so a
    partial list is preferable to a hard failure.
    """
    if log_probs is None:
        return None
    if not isinstance(log_probs, list):
        return None
    out: list[float] = []
    # deque + popleft/extendleft avoids the O(n^2) of list.pop(0)/front-splice
    # on long RL/TITO logprob payloads. reversed() keeps original token order.
    pending: deque = deque(log_probs)
    while pending:
        item = pending.popleft()
        if isinstance(item, bool):
            # bool is an int subclass; a True/False here is not a real logprob.
            continue
        if isinstance(item, (int, float)):
            out.append(_finite_logprob(item))
        elif isinstance(item, list):
            pending.extendleft(reversed(item))
        elif isinstance(item, dict) and "logprob" in item:
            try:
                out.append(_finite_logprob(item["logprob"]))
            except (TypeError, ValueError):
                continue
    return out or None


def _is_token_in_request(request: Dict[str, Any]) -> bool:
    return any(
        source.get("token_data") or source.get("token_in")
        for source in _iter_nvext_sources(request)
    )


def _accumulate_engine_data(
    tok: Dict[str, Any],
    request_prompt_token_ids: Optional[list[int]],
    accumulated_token_ids: dict[int, list[int]],
    accumulated_log_probs: dict[int, list[float]],
) -> None:
    output_index = int(tok.get("index") or 0)
    token_accumulator = accumulated_token_ids.setdefault(output_index, [])
    logprob_accumulator = accumulated_log_probs.setdefault(output_index, [])

    new_token_ids = tok.get("token_ids")
    if isinstance(new_token_ids, list):
        for t in new_token_ids:
            try:
                token_accumulator.append(int(t))
            except (TypeError, ValueError):
                # Defensive: a malformed token id should degrade (drop the
                # token) rather than abort the whole generation stream.
                continue

    flat_lp = _flatten_logprobs(tok.get("log_probs"))
    if flat_lp:
        logprob_accumulator.extend(flat_lp)

    finish_reason = tok.get("finish_reason")
    if finish_reason is None:
        return

    # An engine error is delivered as a finish_reason like "error: ..."; do not
    # report it to the RL client as a clean finish (which would look like a
    # successful empty completion).
    finished = not (
        isinstance(finish_reason, str) and finish_reason.startswith("error")
    )

    engine_data: Dict[str, Any] = dict(tok.get("engine_data") or {})
    engine_data.update(
        {
            "completion_token_ids": list(token_accumulator),
            "finished": finished,
        }
    )
    if logprob_accumulator:
        # completion_logprobs must stay positionally aligned 1:1 with
        # completion_token_ids — the trainer indexes them together. A dropped or
        # None logprob on any chunk shifts every later logprob onto the wrong
        # token, so emit them only when the counts match; otherwise omit (a
        # silently misaligned list corrupts the off-policy correction).
        if len(logprob_accumulator) == len(token_accumulator):
            engine_data["completion_logprobs"] = list(logprob_accumulator)
        else:
            logger.warning(
                "Dropping completion_logprobs for output index %d: logprob "
                "count %d != token count %d (misaligned)",
                output_index,
                len(logprob_accumulator),
                len(token_accumulator),
            )
    if request_prompt_token_ids:
        engine_data["prompt_token_ids"] = list(request_prompt_token_ids)
    tok["engine_data"] = engine_data


def _serialize_routed_experts(
    routed_experts: Any, start: int = 0
) -> Optional[Dict[str, Any]]:
    if routed_experts is None:
        return None

    shape = getattr(routed_experts, "shape", None)
    tobytes = getattr(routed_experts, "tobytes", None)
    if shape is None or not callable(tobytes):
        logger.warning(
            "Unable to serialize routed_experts of type %s",
            type(routed_experts).__name__,
        )
        return None

    return {
        # base64, matching vLLM-native encoding.
        "data": base64.b64encode(tobytes()).decode("ascii"),
        "shape": [int(dim) for dim in shape],
        # Row offset of the first returned routing entry within the full
        # sequence (= SamplingParams.routed_experts_prompt_start; vLLM trims the
        # leading prompt rows). Lets the RL consumer align the completion.
        "start": int(start),
        # Encode dtype so the consumer decodes the raw bytes with the right
        # element type instead of assuming a fixed width.
        "dtype": str(getattr(routed_experts, "dtype", "")),
    }


def build_sampling_params(
    request: Dict[str, Any],
    default_sampling_params: Dict[str, Any],
    model_max_len: int | None = None,
    enable_rl: bool = False,
) -> SamplingParams:
    """
    Build SamplingParams from a PreprocessedRequest (internal protocol format).

    Args:
        request: The PreprocessedRequest dict with 'sampling_options', 'stop_conditions',
                 and 'output_options'
        default_sampling_params: Default sampling parameters from the model's
            ``generation_config.json`` (vLLM ``ModelConfig.get_diff_sampling_param``).
            Used for non-RL/chat clients that want the model's recommended
            sampling defaults applied transparently.

    Returns:
        SamplingParams configured from the request

    RL token-in requests use vLLM's default sampling values instead of model
    `generation_config.json` sampling overrides. Ordinary token-data requests
    keep generation_config defaults for Gateway/backward-compatible traffic.
    Stop-token defaults from the model config are still applied later.
    """
    if enable_rl and _is_token_in_request(request):
        # Use vLLM defaults without model generation_config overlays.
        sampling_params = SamplingParams()
    else:
        sampling_params = SamplingParams(**default_sampling_params)

    # Handle guided_decoding - convert to StructuredOutputsParams
    sampling_options = dict(request.get("sampling_options") or {})
    extra_args = request.get("extra_args") or {}
    if isinstance(extra_args, dict):
        passthrough_sampling_options = extra_args.get("sampling_options")
        if isinstance(passthrough_sampling_options, dict):
            sampling_options.update(passthrough_sampling_options)
    guided_decoding = sampling_options.get("guided_decoding")
    if guided_decoding is not None and isinstance(guided_decoding, dict):
        sampling_params.structured_outputs = StructuredOutputsParams(
            json=guided_decoding.get("json"),
            regex=guided_decoding.get("regex"),
            choice=guided_decoding.get("choice"),
            grammar=guided_decoding.get("grammar"),
            whitespace_pattern=guided_decoding.get("whitespace_pattern"),
            structural_tag=serialize_structural_tag(
                guided_decoding.get("structural_tag")
            ),
        )

    # Apply remaining sampling_options
    for key, value in sampling_options.items():
        # Skip guided_decoding - already handled above
        if key == "guided_decoding":
            continue
        if key == "bad_words_token_ids" and value is not None:
            # vLLM has no public setter for token-id bad words; we write the
            # private field directly. Guard so a vLLM upgrade that renames it
            # fails loudly here instead of silently dropping the constraint.
            if not hasattr(sampling_params, "_bad_words_token_ids"):
                raise AttributeError(
                    "vLLM SamplingParams._bad_words_token_ids missing; TITO "
                    "bad_words_token_ids passthrough needs updating for this "
                    "vLLM version"
                )
            sampling_params._bad_words_token_ids = value
            continue
        if value is not None and hasattr(sampling_params, key):
            setattr(sampling_params, key, value)

    # routed_experts_prompt_start (RL capture offset) must be a non-negative
    # int; reject bad client values so the worker emits a sane `start` instead
    # of a bogus offset the consumer cannot align (vLLM clamps the upper bound).
    reps = getattr(sampling_params, "routed_experts_prompt_start", None)
    if reps is not None and (
        isinstance(reps, bool) or not isinstance(reps, int) or reps < 0
    ):
        logger.warning(
            "Ignoring invalid routed_experts_prompt_start=%r (want non-negative int)",
            reps,
        )
        sampling_params.routed_experts_prompt_start = 0

    # Apply stop_conditions
    for key, value in request.get("stop_conditions", {}).items():
        if value is not None and hasattr(sampling_params, key):
            # Do not add stop key to sampling params - dynamo handles stop conditions directly
            if key == "stop":
                continue
            setattr(sampling_params, key, value)
        if (
            key == "stop_token_ids_hidden"
            and value is not None
            and hasattr(sampling_params, "stop_token_ids")
        ):
            existing = sampling_params.stop_token_ids or []
            sampling_params.stop_token_ids = list(set(existing).union(value))
        # Dynamo's StopConditions uses `max_thinking_tokens`; vLLM 0.20+ exposes
        # the same concept as `thinking_token_budget` on SamplingParams and
        # enforces it via the builtin thinking-budget logits processor.
        if (
            key == "max_thinking_tokens"
            and value is not None
            and hasattr(sampling_params, "thinking_token_budget")
        ):
            sampling_params.thinking_token_budget = value

    # Apply output_options (logprobs, prompt_logprobs, etc.)
    output_options = request.get("output_options", {}) or {}
    logprobs, prompt_logprobs = _shared_logprobs.parse_logprob_options(output_options)
    if logprobs is not None:
        sampling_params.logprobs = logprobs
    if prompt_logprobs is not None:
        sampling_params.prompt_logprobs = prompt_logprobs

    # skip_special_tokens is intentionally NOT forwarded to vLLM here: this path
    # forces detokenize=False (below), so vLLM never detokenizes and ignores it.
    # Dynamo detokenizes in the Rust backend, which reads skip_special_tokens
    # directly from the request output_options (lib/llm/src/backend.rs).

    # If max_tokens wasn't provided (None or missing), compute a dynamic default
    provided_max_tokens = request.get("stop_conditions", {}).get("max_tokens", None)
    token_ids = request.get("token_ids", [])
    input_length = len(token_ids)
    if model_max_len is not None and provided_max_tokens is None:
        # Ensure at least 1 token generation by default when possible
        dynamic_default = max(1, model_max_len - input_length)
        configured_default = default_sampling_params.get("max_tokens", dynamic_default)
        sampling_params.max_tokens = min(configured_default, dynamic_default)

    # Dynamo's internal token path consumes disjoint token deltas. This mirrors
    # the SGLang integration and lets vLLM's stream_interval gate reduce backend
    # bridge pressure before chunks cross into Dynamo.
    sampling_params.detokenize = False
    sampling_params.output_kind = _DELTA_REQUEST_OUTPUT_KIND

    return sampling_params


def build_sampling_params_openai(
    request: Dict[str, Any],
    default_sampling_params: Dict[str, Any],
) -> SamplingParams:
    """
    Build SamplingParams from an OpenAI-compatible request format.

    Args:
        request: The OpenAI-style request dict with parameters like temperature, max_tokens, etc.
        default_sampling_params: Default sampling parameters to initialize with

    Returns:
        SamplingParams configured from the request
    """
    sampling_params = SamplingParams(**default_sampling_params)
    sampling_params.detokenize = True

    # Map common OpenAI parameters to SamplingParams
    openai_mapping = {
        "n": "n",
        "temperature": "temperature",
        "top_p": "top_p",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
        "seed": "seed",
        "top_k": "top_k",
        "repetition_penalty": "repetition_penalty",
        "min_p": "min_p",
        "length_penalty": "length_penalty",
        "use_beam_search": "use_beam_search",
    }

    for req_key, param_key in openai_mapping.items():
        if req_key in request and request[req_key] is not None:
            if hasattr(sampling_params, param_key):
                setattr(sampling_params, param_key, request[req_key])

    # Handle max_tokens
    if "max_tokens" in request and request["max_tokens"] is not None:
        sampling_params.max_tokens = request["max_tokens"]

    # Handle stop sequences
    if "stop" in request and request["stop"] is not None:
        sampling_params.stop = request["stop"]

    # Handle ignore_eos (custom extension)
    if "ignore_eos" in request and request["ignore_eos"] is not None:
        sampling_params.ignore_eos = request["ignore_eos"]

    # Handle min_tokens (custom extension)
    if "min_tokens" in request and request["min_tokens"] is not None:
        sampling_params.min_tokens = request["min_tokens"]

    nvext_max_thinking_tokens = (request.get("nvext") or {}).get("max_thinking_tokens")
    if nvext_max_thinking_tokens is not None and hasattr(
        sampling_params, "thinking_token_budget"
    ):
        sampling_params.thinking_token_budget = nvext_max_thinking_tokens

    return sampling_params


def _engine_generate_reasoning_kwargs(
    engine_client: Any,
    reasoning_ended: bool | None,
    reasoning_parser_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    if reasoning_ended is None and reasoning_parser_kwargs is None:
        return {}

    support = _engine_generate_reasoning_support(engine_client)
    if support is None:
        return {}
    accepts_reasoning_ended, accepts_reasoning_parser_kwargs = support

    kwargs: dict[str, Any] = {}
    if accepts_reasoning_ended:
        kwargs["reasoning_ended"] = reasoning_ended
    if accepts_reasoning_parser_kwargs:
        kwargs["reasoning_parser_kwargs"] = reasoning_parser_kwargs

    if not kwargs:
        logger.debug(
            "vLLM generate does not accept reasoning parser kwargs; "
            "running without request-local reasoning parser metadata"
        )
    return kwargs


def _engine_generate_reasoning_support(
    engine_client: Any,
) -> tuple[bool, bool] | None:
    try:
        cached = vars(engine_client).get(_GENERATE_REASONING_SUPPORT_CACHE_ATTR)
    except TypeError:
        cached = None
    if cached is not None:
        return cached

    try:
        parameters = inspect.signature(engine_client.generate).parameters
    except (TypeError, ValueError):
        logger.debug(
            "Unable to inspect vLLM generate signature; dropping reasoning parser kwargs"
        )
        return None

    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()
    )
    support = (
        accepts_kwargs or "reasoning_ended" in parameters,
        accepts_kwargs or "reasoning_parser_kwargs" in parameters,
    )
    try:
        setattr(engine_client, _GENERATE_REASONING_SUPPORT_CACHE_ATTR, support)
    except Exception:
        pass
    return support


def _request_reasoning_metadata(
    request: Mapping[str, Any],
) -> tuple[bool | None, dict[str, Any] | None]:
    reasoning_ended = request.get("reasoning_ended")
    reasoning_parser_kwargs = request.get("reasoning_parser_kwargs")

    extra_args = request.get("extra_args")
    if isinstance(extra_args, dict):
        if reasoning_ended is None:
            reasoning_ended = extra_args.get("reasoning_ended")
        if reasoning_parser_kwargs is None:
            reasoning_parser_kwargs = extra_args.get("reasoning_parser_kwargs")

    return reasoning_ended, reasoning_parser_kwargs


def get_dp_range_for_worker(vllm_config: VllmConfig) -> tuple[int, int]:
    """
    Get the global DP rank range that this worker is responsible for based on vLLM config.
    Note that the 'vllm_config' is normalized so the load balancing flags are set properly.
    The return value is in the format of (start_dp_rank, managed_dp_size)."""
    if vllm_config.parallel_config.data_parallel_external_lb:
        # external load balancing, each worker is responsible for exactly 1 rank
        return (vllm_config.parallel_config.data_parallel_rank, 1)
    elif vllm_config.parallel_config.data_parallel_hybrid_lb:
        # hybrid load balancing, each worker is responsible for a subset of local ranks
        return (
            vllm_config.parallel_config.data_parallel_rank,
            vllm_config.parallel_config.data_parallel_size_local,
        )
    else:
        # internal load balancing, the worker is responsible for all DP ranks
        logger.warning(
            "vLLM selects internal DP load balancing. If you are launching multiple workers for DP deployment,"
            " hybrid or external load balancing is recommended."
        )
        return (
            vllm_config.parallel_config.data_parallel_rank,
            vllm_config.parallel_config.data_parallel_size,
        )


RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")


class BaseWorkerHandler(ABC, Generic[RequestT, ResponseT]):
    """
    Request handler for the generate and clear_kv_blocks endpoints.
    """

    _benchmark_results: Optional[dict] = None
    _scale_ep_in_progress: bool = False

    def __init__(
        self,
        runtime,
        config: Config,
        engine,
        default_sampling_params,
        model_max_len: int | None = None,
        model_config: ModelConfig | None = None,
        enable_multimodal: bool = False,
        generate_endpoint=None,
        use_vllm_tokenizer: bool = False,
        shutdown_event: asyncio.Event | None = None,
        enable_frontend_decoding: bool = False,
        encode_worker_client: Optional[Client] = None,
    ):
        self.runtime = runtime
        self.engine_client = engine
        self.default_sampling_params = default_sampling_params
        self.kv_publishers: list[KvEventPublisher] | None = None
        self.fpm_relays: list | None = None
        self.generate_endpoint = generate_endpoint
        self.config = config
        self.engine_monitor = VllmEngineMonitor(runtime, engine, shutdown_event)
        self.temp_dirs: list[tempfile.TemporaryDirectory] = []
        self.model_max_len = model_max_len
        self.model_config = model_config
        # LoRA tracking: name -> LoRAInfo(id, path)
        self.loaded_loras: dict[str, LoRAInfo] = {}
        # Adapters known to have been handed to vLLM. Prefill registration is
        # metadata-only, but vLLM activates a prefill adapter lazily when an
        # inference request supplies its LoRARequest.
        self._engine_loaded_loras: set[str] = set()
        # Requests and load/unload operations for the same adapter share a lock,
        # so an unload cannot race with vLLM's lazy adapter activation. A fixed
        # number of shared locks bounds memory as adapter names come and go.
        self._lora_load_locks = [asyncio.Lock() for _ in range(_LORA_LOCK_STRIPES)]
        self._paused: bool = False
        self._weight_version: str = "initial"

        embedding_loader = self.init_embedding_loader(config, encode_worker_client)

        # Aggregated partial encoder. The attribute is set here so cleanup() is
        # always safe; the encoder itself is loaded last in __init__ (it starts
        # the actor thread) — see _load_custom_encoder below for why.
        self._custom_encoder: Optional[AsyncVisionEncoder] = None
        self._custom_encoder_adapter: Optional[CustomEncoderAdapter] = None

        self.use_vllm_tokenizer = use_vllm_tokenizer

        self.dp_range = get_dp_range_for_worker(self.engine_client.vllm_config)
        self._pause_controller = VllmEnginePauseController(self.engine_client)
        self._pause_lock = asyncio.Lock()
        # Maps request_id -> _DeferredAbort for in-flight decode-only requests so
        # admin abort_request can route through the deferred-abort path instead
        # of calling engine_client.abort() during the unsafe pre-first-token
        # NIXL-KV-transfer window.
        self._deferred_aborts: dict[str, _DeferredAbort] = {}

        self._multimodal_request_processor = VllmMultimodalRequestProcessor(
            model=config.model,
            engine_client=engine,
            enable_multimodal=enable_multimodal,
            enable_frontend_decoding=enable_frontend_decoding,
            embedding_loader=embedding_loader,
            trust_remote_code=config.engine_args.trust_remote_code,
        )

        # Serialise concurrent scale_elastic_ep calls.  vLLM's elastic-EP
        # bootstrap creates a fresh TCPStore per scale operation and stores it
        # in engine_client._coord_store.  When two callers race through
        # _setup_elastic_ep_reconfig_bootstrap concurrently the first caller's
        # store gets garbage-collected before the new Ray actor has had a chance
        # to connect, causing a 300 s TCPStore timeout on the worker node.
        # One handler is created per worker process (worker_factory.py), so all
        # concurrent HTTP callers share this lock and only one scale operation
        # can mutate _coord_store at a time.
        self._scale_ep_lock = asyncio.Lock()
        self._scale_ep_in_progress = False

        # Initialize InputParamManager for text-in-text-out mode
        tokenizer = None
        if use_vllm_tokenizer and hasattr(engine, "tokenizer"):
            tokenizer = engine.tokenizer
        self.input_param_manager = InputParamManager(tokenizer)

        # Store shutdown event for graceful shutdown monitoring
        self.shutdown_event = shutdown_event
        # Request-plane RL method map served by rl_dispatch on
        # dyn://<namespace>.<component>.rl when --enable-rl / DYN_ENABLE_RL is set.
        self.rl_route_registry = RLRouteRegistry(self.runtime, logger_=logger)

        # Load the custom encoder last. If a later init step raised, executor
        # GC would eventually reap the idle actor thread — but only once the
        # exception's traceback stops pinning `self` (unbounded while the
        # failure is handled upstream), and GC never runs backend.close() or
        # frees the encoder's GPU memory in the meantime. Ordering the encoder
        # after all other fallible setup removes that window by construction.
        self._load_custom_encoder(config)

    def _load_custom_encoder(self, config: Config) -> None:
        """Import, instantiate, and load the --custom-encoder-class encoder."""
        custom_encoder_class = config.custom_encoder_class
        if not custom_encoder_class:
            return
        module_path, _, class_name = custom_encoder_class.rpartition(".")
        backend_cls = getattr(importlib.import_module(module_path), class_name)
        if not (
            isinstance(backend_cls, type)
            and issubclass(backend_cls, VisionEncoderBackend)
        ):
            raise TypeError(
                f"--custom-encoder-class {custom_encoder_class!r} must resolve to a "
                f"VisionEncoderBackend subclass, got {backend_cls!r}."
            )
        # The author writes the VisionEncoderBackend; Dynamo wraps it in the
        # AsyncVisionEncoder glue, which owns the preprocess pool and
        # ThreadedMicroBatcher actor thread. load() runs backend.build() there
        # (the backend picks its own device) and cleans that thread up on failure.
        backend = backend_cls()
        adapter = create_custom_encoder_adapter(
            backend,
            self.model_config,
            config.engine_args,
            self.engine_client.vllm_config,
        )
        encoder = AsyncVisionEncoder(backend)
        encoder.load(config.model)
        # Assign only after a successful load so a failed load (which already shut
        # its own thread down) leaves _custom_encoder None.
        self._custom_encoder = encoder
        self._custom_encoder_adapter = adapter
        logger.info(
            "Loaded CustomEncoder %s from %s with %s",
            custom_encoder_class,
            config.model,
            type(adapter).__name__,
        )

    def _shutdown_worker(self) -> NoReturn:
        logger.warning("Initiating Dynamo Runtime shutdown.")
        self.runtime.shutdown()
        os._exit(1)

    def _shutdown_on_engine_dead(self, e: EngineDeadError) -> NoReturn:
        logger.error(f"vLLM EngineDeadError: {e}")
        self._shutdown_worker()

    def init_embedding_loader(
        self, config: Config, encode_worker_client: Optional[Client] = None
    ) -> Optional[MultiModalEmbeddingLoader]:
        """Initialize the embedding loader with the given encode worker client."""
        # Without encode worker, the embedding will be generated internally by vLLM.
        if encode_worker_client is None:
            return None
        logger.warning(
            "Separate multimodal encode-worker routing only applies to image_url "
            "inputs. video_url inputs are not sent to the encode worker and will "
            "be processed on the prefill/PD worker instead."
        )
        # Embedding loader consist of two main components:
        # 1) An remote encode worker client and matching embedding receiver,
        #    which can request remote encode and handle the transfer of embeddings
        #    from the encode worker to this prefill worker.
        # 2) A local embedding cache manager, which can store previously fetched embeddings
        #    and used to determine whether remote encode is necessary for a given mm data.
        self.encode_worker_client = encode_worker_client
        if config.embedding_transfer_mode == EmbeddingTransferMode.LOCAL:
            self.embedding_receiver = LocalEmbeddingReceiver()  # type: ignore
        elif config.embedding_transfer_mode == EmbeddingTransferMode.NIXL_WRITE:
            self.embedding_receiver = NixlWriteEmbeddingReceiver()  # type: ignore
        elif config.embedding_transfer_mode == EmbeddingTransferMode.NIXL_READ:
            # [gluo FIXME] can't use pre-registered tensor as NIXL requires descriptors
            # to be at matching size, need to overwrite nixl connect library
            self.embedding_receiver = NixlReadEmbeddingReceiver(max_items=0)  # type: ignore
        else:
            raise ValueError(
                f"Invalid embedding transfer mode: {config.embedding_transfer_mode}"
            )
        # [gluo FIXME/NOTE] This embedding cache manager is purely used for caching embedding
        # results from encode worker, but 'config.multimodal_embedding_cache_capacity_gb' is
        # also used to configure the DynamoMultimodalEmbeddingCacheConnector within the vLLM.
        # This results in duplication of memory and ideally we should have single cache manager
        # which can be used by vLLM internal and here. Then we can explore asynchrous embedding
        # transfer as we can process and block until the embedding is actually used within vLLM.
        self.embedding_cache_manager: MultimodalEmbeddingCacheManager | None = None
        if config.multimodal_embedding_cache_capacity_gb > 0:
            capacity_bytes = int(
                config.multimodal_embedding_cache_capacity_gb * 1024**3
            )
            self.embedding_cache_manager = MultimodalEmbeddingCacheManager(
                capacity_bytes
            )
        return MultiModalEmbeddingLoader(
            encode_worker_client=self.encode_worker_client,  # type: ignore
            receiver=self.embedding_receiver,
            embedding_cache_manager=self.embedding_cache_manager,
        )

    async def sleep(self, body: dict) -> dict:
        """Sleep the engine to release GPU memory and unregister from discovery.

        Args:
            body: Dict with optional 'level' key (1=weights only, 2=weights+buffers, 3=everything)

        Order of operations:
        1. Unregister from discovery - stop accepting new requests
        2. Abort and drain in-flight requests
        3. Sleep engine - safe once generation has stopped
        """
        if body is None:
            body = {}
        elif not isinstance(body, dict):
            return {
                "status": "error",
                "message": "request body must be a JSON object",
            }
        level = body.get("level", 1)
        async with self._pause_lock:
            if self._pause_controller.is_paused:
                return {
                    "status": "ok",
                    "message": "Engine already sleeping",
                }
            if self._pause_controller.needs_resume_recovery:
                return {
                    "status": "error",
                    "message": "wake_up required before retrying sleep",
                }

            unregistered = False
            try:
                # Step 1: Unregister endpoint instance before memory transitions.
                if self.generate_endpoint is not None:
                    await self.generate_endpoint.unregister_endpoint_instance()
                    unregistered = True
                    logger.info(
                        "[Sleep] Unregistered endpoint from discovery - worker removed from routing pool"
                    )

                # Step 2: Abort in-flight requests and wait for them to drain so
                # generation is fully paused before unmapping memory.
                if not await self._pause_controller.pause(level):
                    return {
                        "status": "ok",
                        "message": "Engine already sleeping",
                    }

                return {
                    "status": "ok",
                    "message": f"Engine slept (level={level})",
                }
            except Exception as e:
                logger.error(f"Failed to sleep engine: {e}")
                # If pause rolled back cleanly the engine is serving-safe again,
                # but discovery still shows us unregistered and wake_up will
                # early-return. Re-register so the worker rejoins the routing pool.
                if (
                    unregistered
                    and not self._pause_controller.is_paused
                    and not self._pause_controller.needs_resume_recovery
                    and self.generate_endpoint is not None
                ):
                    try:
                        await self.generate_endpoint.register_endpoint_instance()
                        logger.info(
                            "[Sleep] Re-registered endpoint after failed sleep rollback"
                        )
                    except Exception as reg_err:
                        logger.error(
                            f"Failed to re-register endpoint after sleep failure: {reg_err}"
                        )
                return {"status": "error", "message": str(e)}

    async def scale_elastic_ep(self, body: dict) -> dict:
        """Scale the elastic expert-parallelism data-parallel size live.

        Args:
            body: Dict with required 'new_data_parallel_size' key (int).
                Example::

                    {"new_data_parallel_size": 4}

        The vLLM Ray DP backend will spin up / tear down DP workers on the GPUs
        already reserved by the pod, then hot-swap the expert routing table.
        No pod restart is needed.
        """
        if body is None:
            body = {}
        elif not isinstance(body, dict):
            return {
                "status": "error",
                "message": "request body must be a JSON object",
            }
        new_dp_size = body.get("new_data_parallel_size")
        if new_dp_size is None:
            return {
                "status": "error",
                "message": "Missing required field: new_data_parallel_size",
            }
        try:
            new_dp_size = int(new_dp_size)
        except (TypeError, ValueError):
            return {
                "status": "error",
                "message": f"new_data_parallel_size must be an integer, got: {new_dp_size!r}",
            }
        if new_dp_size < 2:
            return {
                "status": "error",
                "message": (
                    "new_data_parallel_size must be >= 2 when elastic EP/ePLB is enabled"
                ),
            }

        logger.info(f"[ElasticEP] Scaling to new_data_parallel_size={new_dp_size}")

        # Early-reject if another scale is already in progress rather than
        # queuing behind it: a queued caller would garbage-collect the first
        # caller's TCPStore before its Ray actor connects, causing a 300 s
        # timeout on the worker node.
        async with self._scale_ep_lock:
            if self._scale_ep_in_progress:
                msg = (
                    "A scale_elastic_ep operation is already in progress; "
                    f"rejecting concurrent request for new_data_parallel_size={new_dp_size}"
                )
                logger.warning("[ElasticEP] %s", msg)
                return {"status": "error", "message": msg}
            self._scale_ep_in_progress = True

        try:
            # TODO(upstream-vllm): remove this patch once vLLM fixes
            # add_dp_placement_groups in vllm/v1/engine/utils.py to use ray.nodes()
            # instead of ray.util.state.list_nodes().
            #
            # Patch ray.util.state.list_nodes to use the GCS API instead of the
            # dashboard HTTP API (127.0.0.1:8265/api/v0/nodes). The dynamo image
            # installs ray core only (not ray[default]), so the dashboard HTTP server
            # starts in --minimal mode with the HTTP server disabled. vLLM's
            # add_dp_placement_groups calls list_nodes() which requires that HTTP
            # endpoint, causing scale_elastic_ep to fail with "Failed to connect to
            # API server".
            #
            # ray.nodes() uses the GCS gRPC channel directly (no dashboard process
            # needed) and returns the same information. Imported lazily so ray is not
            # required at module load time (absent in non-elastic-EP deployments).
            #
            # Format mapping:
            #   list_nodes() -> objects with .node_ip and .node_id
            #   ray.nodes()  -> dicts with "NodeManagerAddress" and "NodeID"
            import ray
            import ray.util.state as _ray_util_state

            class _NodeInfo:
                __slots__ = ("node_id", "node_ip")

                def __init__(self, d: dict) -> None:
                    self.node_ip: str = d["NodeManagerAddress"]
                    self.node_id: str = d["NodeID"]

            original_list_nodes = _ray_util_state.list_nodes
            try:
                _ray_util_state.list_nodes = lambda **kw: [
                    _NodeInfo(n) for n in ray.nodes() if n.get("Alive", False)
                ]
                await self.engine_client.scale_elastic_ep(new_dp_size)
            finally:
                _ray_util_state.list_nodes = original_list_nodes

            logger.info(f"[ElasticEP] Scaling to dp={new_dp_size} complete")
            return {
                "status": "ok",
                "message": f"Scaled to data_parallel_size={new_dp_size}",
                "new_data_parallel_size": new_dp_size,
            }
        except Exception as e:
            logger.error(f"[ElasticEP] Scaling failed: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            async with self._scale_ep_lock:
                self._scale_ep_in_progress = False

    async def wake_up(self, body: dict) -> dict:
        """Wake the engine to restore GPU memory and re-register to discovery.

        Args:
            body: Optional dict with "tags" to request a partial wake.

        Order of operations:
        1. Wake engine - restore GPU memory
        2. Re-register endpoint instance - allow frontend to route requests here again
        """
        if body is None:
            body = {}
        elif not isinstance(body, dict):
            return {
                "status": "error",
                "message": "request body must be a JSON object",
            }
        tags = body.get("tags")
        async with self._pause_lock:
            needs_recovery = self._pause_controller.needs_resume_recovery
            if not self._pause_controller.is_paused and not needs_recovery:
                return {"status": "ok", "message": "Engine already awake"}

            try:
                # Step 1: Wake engine first - must be ready before accepting requests
                await self._pause_controller.resume(tags)
                if self.generate_endpoint is not None:
                    await self.generate_endpoint.register_endpoint_instance()
                    logger.info(
                        "[Wake] Re-registered endpoint to discovery - worker added back to routing pool"
                    )
                self._pause_controller.mark_resumed()

                return {
                    "status": "ok",
                    "message": "Engine woke",
                }
            except Exception as e:
                logger.error(f"Failed to wake up engine: {e}")
                return {"status": "error", "message": str(e)}

    async def start_profile(self, body: dict) -> dict:
        """Start profiling on the engine.

        Args:
            body: Dict with profiling parameters. Supported keys:
                - profile_prefix (str|None): Optional prefix for profile output files.
        """
        profile_prefix = body.get("profile_prefix")
        try:
            await self.engine_client.start_profile(profile_prefix=profile_prefix)
            return {"status": "ok", "message": "Profiling started"}
        except Exception as e:
            logger.error(f"Failed to start profiling: {e}")
            return {"status": "error", "message": str(e)}

    async def stop_profile(self, body: dict) -> dict:
        """Stop profiling on the engine.

        Args:
            body: Unused, but required for handler signature.
        """
        try:
            await self.engine_client.stop_profile()
            return {"status": "ok", "message": "Profiling stopped"}
        except Exception as e:
            logger.error(f"Failed to stop profiling: {e}")
            return {"status": "error", "message": str(e)}

    async def rl_dispatch(self, request=None):
        """Request-plane dispatcher for the worker's ``rl`` endpoint."""
        async for response in self.rl_route_registry.dispatch_stream(request):
            yield response

    async def liveness_probe(self, body: dict) -> dict:
        """Engine event-loop liveness probe."""
        if body is None:
            body = {}
        elif not isinstance(body, dict):
            return {
                "status": "error",
                "message": "request body must be a JSON object",
            }
        try:
            if hasattr(self.engine_client, "check_health"):
                await self.engine_client.check_health()
            else:
                await self.engine_client.collective_rpc("liveness_probe", kwargs={})
            return {"status": "ok", "alive": True}
        except EngineDeadError as e:
            self._shutdown_on_engine_dead(e)
        except Exception as e:
            logger.warning(f"[RL] liveness_probe failed: {e}")
            return {"status": "error", "alive": False, "message": str(e)}

    async def pause_generation(self, body: dict) -> dict:
        """Pause generation before a weight update."""
        if body is None:
            body = {}
        elif not isinstance(body, dict):
            return {
                "status": "error",
                "message": "request body must be a JSON object",
            }
        mode = body.get("mode", "keep")
        clear_cache = bool(body.get("clear_cache", False))
        if mode not in ("keep", "wait", "abort"):
            return {
                "status": "error",
                "message": f"Invalid mode '{mode}'; expected keep|wait|abort",
            }
        async with self._pause_lock:
            try:
                try:
                    await self.engine_client.pause_generation(
                        mode=mode, clear_cache=clear_cache
                    )
                except TypeError:
                    await self.engine_client.pause_generation()
                    if clear_cache:
                        await self.engine_client.reset_prefix_cache()
                self._paused = True
                logger.info(
                    f"[RL] Engine paused (mode={mode}, clear_cache={clear_cache})"
                )
                return {
                    "status": "ok",
                    "message": "Engine paused",
                    "mode": mode,
                    "clear_cache": clear_cache,
                }
            except EngineDeadError as e:
                self._shutdown_on_engine_dead(e)
            except Exception as e:
                logger.error(f"[RL] Failed to pause: {e}")
                return {"status": "error", "message": str(e)}

    async def resume_generation(self, body: dict) -> dict:
        """Resume generation after a weight update."""
        if body is None:
            body = {}
        elif not isinstance(body, dict):
            return {
                "status": "error",
                "message": "request body must be a JSON object",
            }
        # Serialize pause / resume / weight-update so a concurrent resume cannot
        # re-enable generation while an update's collective_rpc is still
        # mutating weights (dynamo-ops).
        async with self._pause_lock:
            try:
                await self.engine_client.resume_generation()
                self._paused = False
                logger.info("[RL] Engine resumed")
                return {"status": "ok", "message": "Engine resumed"}
            except EngineDeadError as e:
                self._shutdown_on_engine_dead(e)
            except Exception as e:
                logger.error(f"[RL] Failed to resume: {e}")
                return {"status": "error", "message": str(e)}

    async def flush_cache(self, body: dict) -> dict:
        """Invalidate prefix / KV cache."""
        if body is None:
            body = {}
        elif not isinstance(body, dict):
            return {
                "status": "error",
                "message": "request body must be a JSON object",
            }
        # Serialize under _pause_lock so a flush cannot race with a locked
        # weight-update / pause / resume mutating engine cache state.
        async with self._pause_lock:
            try:
                await self.engine_client.reset_prefix_cache()
                logger.debug("[RL] Prefix cache flushed")
                return {"status": "ok", "message": "Cache flushed"}
            except EngineDeadError as e:
                self._shutdown_on_engine_dead(e)
            except Exception as e:
                logger.error(f"[RL] Failed to flush cache: {e}")
                return {"status": "error", "message": str(e)}

    async def abort_request(self, body: dict) -> dict:
        """Abort a single in-flight request by request_id."""
        if body is None:
            body = {}
        elif not isinstance(body, dict):
            return {
                "status": "error",
                "message": "request body must be a JSON object",
            }
        request_id = body.get("request_id")
        if not request_id:
            return {"status": "error", "message": "Missing 'request_id' in body"}
        try:
            guard = self._deferred_aborts.get(request_id)
            if guard is not None:
                # Route through the per-request deferred-abort guard so that in
                # disaggregated decode mode the real engine abort is deferred
                # until the first token, never firing during an in-flight NIXL
                # KV transfer (which can crash EngineCore).
                await guard.abort()
                # If the abort already ran (post-first-token) and failed, report
                # it instead of a false "ok"; escalate engine death like the
                # direct path does. (Pre-first-token aborts are queued and have
                # no result yet, so they correctly report accepted/ok.)
                abort_exc = guard._abort_exc
                if abort_exc is not None:
                    if isinstance(abort_exc, EngineDeadError):
                        self._shutdown_on_engine_dead(abort_exc)
                    return {
                        "status": "error",
                        "request_id": request_id,
                        "message": f"abort failed: {abort_exc}",
                    }
            else:
                await self.engine_client.abort(request_id)
            logger.debug(f"[RL] Aborted request {request_id}")
            return {"status": "ok", "request_id": request_id}
        except EngineDeadError as e:
            self._shutdown_on_engine_dead(e)
        except Exception as e:
            logger.error(f"[RL] Failed to abort request {request_id}: {e}")
            return {"status": "error", "message": str(e)}

    async def get_weight_version(self, body: dict) -> dict:
        """Return the current weight version tag."""
        if body is None:
            body = {}
        elif not isinstance(body, dict):
            return {
                "status": "error",
                "message": "request body must be a JSON object",
            }
        return {"status": "ok", "version": getattr(self, "_weight_version", "initial")}

    async def update_weights_from_disk(self, body: dict) -> dict:
        """Load weights from a shared filesystem checkpoint."""
        if body is None:
            body = {}
        elif not isinstance(body, dict):
            return {
                "status": "error",
                "message": "request body must be a JSON object",
            }
        # Hold _pause_lock across the paused-state check and the weight RPC so a
        # concurrent resume cannot re-enable generation mid-update (dynamo-ops).
        async with self._pause_lock:
            if not getattr(self, "_paused", False):
                return {
                    "status": "error",
                    "message": (
                        "Worker must be paused via pause_generation() before "
                        "updating weights. Call pause_generation() first, then "
                        "update, then resume_generation()."
                    ),
                }
            path = body.get("model_path")
            if not path:
                return {"status": "error", "message": "Missing 'model_path' in body"}
            version = body.get("weight_version", "unknown")
            rpc = body.get("engine_rpc", "reload_weights")
            kwargs = (
                {"weights_path": path}
                if rpc == "reload_weights"
                else {"weight_path": path}
            )
            try:
                await self.engine_client.collective_rpc(rpc, kwargs=kwargs)
                # Weights changed: any prefix/KV cache computed under the old
                # weights is now stale and must not be reused. Invalidate it
                # while still holding _pause_lock (generation is paused).
                await self.engine_client.reset_prefix_cache()
                self._weight_version = version
                logger.info(
                    f"[RL] Weights loaded from {path} (version={version}, rpc={rpc})"
                )
                return {"status": "ok", "version": version}
            except EngineDeadError as e:
                self._shutdown_on_engine_dead(e)
            except Exception as e:
                logger.error(f"[RL] update_weights_from_disk failed: {e}")
                return {"status": "error", "message": str(e)}

    async def update_weights_from_distributed(self, body: dict) -> dict:
        """Receive weights via a distributed transport such as NCCL."""
        if body is None:
            body = {}
        elif not isinstance(body, dict):
            return {
                "status": "error",
                "message": "request body must be a JSON object",
            }
        allow_unpaused = body.get("allow_unpaused", False)
        reset_prefix_cache = body.get("reset_prefix_cache", True)
        if not isinstance(allow_unpaused, bool):
            return {
                "status": "error",
                "message": "'allow_unpaused' must be a boolean",
            }
        if not isinstance(reset_prefix_cache, bool):
            return {
                "status": "error",
                "message": "'reset_prefix_cache' must be a boolean",
            }
        if allow_unpaused and reset_prefix_cache:
            return {
                "status": "error",
                "message": (
                    "Unpaused weight updates cannot reset the prefix cache. "
                    "Set 'reset_prefix_cache' to false or pause generation first."
                ),
            }
        async with self._pause_lock:
            if not self._paused and not allow_unpaused:
                return {
                    "status": "error",
                    "message": (
                        "Worker must be paused via pause_generation() before "
                        "updating weights. Call pause_generation() first, then "
                        "update, then resume_generation()."
                    ),
                }
            version = body.get("weight_version", "unknown")
            rpc = body.get("engine_rpc", "update_weights_from_path")
            rpc_kwargs = {
                k: v
                for k, v in body.items()
                if k not in _DISTRIBUTED_WEIGHT_UPDATE_RESERVED_KEYS
            }
            try:
                await self.engine_client.collective_rpc(rpc, kwargs=rpc_kwargs)
                if reset_prefix_cache:
                    # Weights changed: stale prefix/KV cache must be invalidated
                    # before resume so it is not reused under the new weights.
                    await self.engine_client.reset_prefix_cache()
                self._weight_version = version
                logger.info(
                    f"[RL] Weights received via distributed "
                    f"(version={version}, rpc={rpc})"
                )
                return {"status": "ok", "version": version}
            except EngineDeadError as e:
                self._shutdown_on_engine_dead(e)
            except Exception as e:
                logger.error(f"[RL] update_weights_from_distributed failed: {e}")
                return {"status": "error", "message": str(e)}

    async def update_weights_from_tensor(self, body: dict) -> dict:
        """Not implemented: in-process tensor transfer is not yet supported."""
        if body is None:
            body = {}
        elif not isinstance(body, dict):
            return {
                "status": "error",
                "message": "request body must be a JSON object",
            }
        return {
            "status": "error",
            "message": "update_weights_from_tensor is not implemented",
        }

    async def init_weights_update_group(self, body: dict) -> dict:
        """Initialize the distributed weight-update communication group."""
        if body is None:
            body = {}
        elif not isinstance(body, dict):
            return {
                "status": "error",
                "message": "request body must be a JSON object",
            }
        rpc = body.get("engine_rpc", "init_broadcaster")
        kwargs = {k: v for k, v in body.items() if k != "engine_rpc"}
        async with self._pause_lock:
            try:
                timeout_s = _rl_init_weights_timeout_s()
                rpc_task = asyncio.create_task(
                    self.engine_client.collective_rpc(rpc, kwargs=kwargs)
                )
                try:
                    done, _ = await asyncio.wait({rpc_task}, timeout=timeout_s)
                except asyncio.CancelledError:
                    rpc_task.cancel()
                    await asyncio.gather(rpc_task, return_exceptions=True)
                    raise
                if rpc_task not in done:
                    rpc_task.cancel()
                    await asyncio.gather(rpc_task, return_exceptions=True)
                    logger.error(
                        f"[RL] init_weights_update_group timed out after "
                        f"{timeout_s:.1f} seconds (rpc={rpc}); terminating the "
                        "worker because EngineCore may still be blocked"
                    )
                    self._shutdown_worker()

                await rpc_task
                logger.info(f"[RL] Weight update group initialized (rpc={rpc})")
                return {"status": "ok", "message": "Weight update group initialized"}
            except EngineDeadError as e:
                self._shutdown_on_engine_dead(e)
            except Exception as e:
                logger.error(f"[RL] init_weights_update_group failed: {e}")
                return {"status": "error", "message": str(e)}

    async def destroy_weights_update_group(self, body: dict) -> dict:
        """Tear down the distributed weight-update communication group."""
        if body is None:
            body = {}
        elif not isinstance(body, dict):
            return {
                "status": "error",
                "message": "request body must be a JSON object",
            }
        rpc = body.get("engine_rpc", "destroy_broadcaster")
        kwargs = {k: v for k, v in body.items() if k != "engine_rpc"}
        async with self._pause_lock:
            try:
                await self.engine_client.collective_rpc(rpc, kwargs=kwargs)
                logger.info(f"[RL] Weight update group destroyed (rpc={rpc})")
                return {"status": "ok", "message": "Weight update group destroyed"}
            except EngineDeadError as e:
                self._shutdown_on_engine_dead(e)
            except Exception as e:
                logger.error(f"[RL] destroy_weights_update_group failed: {e}")
                return {"status": "error", "message": str(e)}

    @abstractmethod
    def generate(self, request: RequestT, context: Context) -> AsyncIterator[ResponseT]:
        raise NotImplementedError

    async def _monitor_abort(self, context, request_id, is_prefill, abort_guard=None):
        """
        Background task that monitors for context cancellation and shutdown.
        Aborts the request if either occurs. Raises EngineShutdown if shutdown was triggered.

        If abort_guard is provided, the abort call is routed through it so that
        it can be deferred until the first engine output (used in disagg decode
        mode to avoid aborting during an active NIXL KV transfer).
        """
        try:
            # Build list of futures/tasks to wait for
            wait_for = [context.async_killed_or_stopped()]
            shutdown_task = None

            if self.shutdown_event:
                # Create task for shutdown monitoring and add to wait list
                shutdown_task = asyncio.create_task(self.shutdown_event.wait())
                wait_for.append(shutdown_task)

            # Wait for whichever happens first
            done, pending = await asyncio.wait(
                wait_for,
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel the pending task/future
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Log intent before the abort — synchronous, no await, so it is
            # guaranteed to appear even if this task is concurrently cancelled
            # by _abort_monitor's cleanup.
            logger.debug(
                f"Aborting {'Prefill ' if is_prefill else ''}Request ID: {request_id}"
            )

            # Abort the request (via guard if provided, otherwise direct).
            if abort_guard is not None:
                # Guard owns completion logging: "Aborted Request ID:" is
                # emitted from _run_abort() once engine.abort() returns,
                # even if this task is concurrently cancelled.
                await abort_guard.abort()
            else:
                # No guard: shield the abort so a concurrent CancelledError
                # from _abort_monitor cleanup cannot interrupt it mid-flight.
                try:
                    await asyncio.shield(self.engine_client.abort(request_id))
                except asyncio.CancelledError:
                    logger.debug(
                        f"Abort shielded from cancellation for request "
                        f"{request_id}, continuing in background"
                    )
                logger.debug(
                    f"Aborted {'Prefill ' if is_prefill else ''}Request ID: {request_id}"
                )

            # Check which event triggered and raise EngineShutdown if shutdown
            if shutdown_task and shutdown_task in done:
                raise EngineShutdown("Engine was shut down during generation.")

        except asyncio.CancelledError:
            # Task was cancelled, normal cleanup if not aborted
            pass
        except EngineShutdown:
            raise
        except Exception as e:
            logger.error(f"Error in abort monitor for request {request_id}: {e}")

    @asynccontextmanager
    async def _abort_monitor(
        self, context, request_id, is_prefill=False, abort_guard=None
    ):
        """
        Context manager that creates and automatically cleans up an abort monitoring task.
        If shutdown event was triggered, raises EngineShutdown on exit.

        If abort_guard is provided, the abort call is routed through it so the
        abort can be deferred until the first engine output.
        """
        task = asyncio.create_task(
            self._monitor_abort(context, request_id, is_prefill, abort_guard)
        )
        try:
            yield task
        finally:
            # Clean up the abort monitoring task
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            else:
                # If the task completed, check if it raised EngineShutdown
                task.result()

    async def clear_kv_blocks(self, request=None):
        try:
            reset_successful = await self.engine_client.reset_prefix_cache(
                reset_connector=True
            )
            if reset_successful is False:
                yield {"status": "error", "message": "KV cache reset failed"}
                return
            yield {"status": "success", "message": "KV cache cleared"}
        except Exception as e:
            yield {"status": "error", "message": str(e)}

    async def get_perf_metrics(self, request=None):
        """Return self-benchmark FPM results, or an error dict if none."""
        result = getattr(self, "_benchmark_results", None)
        if result is None:
            yield {"status": "error", "message": "no benchmark data"}
        else:
            yield result

    def add_temp_dir(self, temp_dir: tempfile.TemporaryDirectory) -> None:
        """Add a temporary directory to be cleaned up later."""
        if temp_dir is not None:
            self.temp_dirs.append(temp_dir)

    def _to_local_dp_rank(self, dp_rank: int | None) -> int | None:
        """Convert global DP rank to local DP rank based on engine config."""
        if dp_rank is None:
            return None
        if dp_rank < self.dp_range[0] or dp_rank >= self.dp_range[0] + self.dp_range[1]:
            logger.warning(
                f"Received DP rank {dp_rank} is out of range [{self.dp_range[0]} - {self.dp_range[0] + self.dp_range[1]}), fallback to vLLM internal DP selection"
            )
            return None
        local_dp_rank = (dp_rank - self.dp_range[0]) % self.dp_range[1]
        logger.debug(
            f"Converted global DP rank {dp_rank} to local DP rank {local_dp_rank}"
        )
        return local_dp_rank

    def _resolve_lora_request(self, model_name: str | None) -> LoRARequest | None:
        """Return a LoRARequest if model_name is a loaded adapter, else None."""
        if model_name and (lora := self.loaded_loras.get(model_name)):
            return LoRARequest(
                lora_name=model_name,
                lora_int_id=lora.id,
                lora_path=lora.path,
            )
        return None

    def _track_lora_request_activation(self, lora_request: LoRARequest | None) -> None:
        """Record adapters handed to vLLM for request-time lazy activation."""
        if lora_request is not None:
            self._engine_loaded_loras.add(lora_request.lora_name)

    @staticmethod
    def _is_lora_not_loaded_error(error: Exception) -> bool:
        """Return whether vLLM reports an idempotent remove of a missing LoRA."""
        message = str(error).lower()
        return "not loaded" in message or "not found" in message

    def _get_lora_lock(self, lora_name: str) -> asyncio.Lock:
        """Return a bounded lock stripe for a LoRA name."""
        return self._lora_load_locks[hash(lora_name) % _LORA_LOCK_STRIPES]

    async def _generate_with_lora_admission_lock(
        self,
        lora_request: LoRARequest | None,
        create_generator: Callable[[LoRARequest | None], AsyncIterator[Any]],
    ) -> AsyncIterator[Any]:
        """Yield results after atomically admitting a lazy LoRA request.

        vLLM admits an ``AsyncLLM.generate`` request on its first iteration.
        Holding the adapter lifecycle lock through that iteration prevents an
        unload from deleting bookkeeping before lazy activation completes.
        """
        if lora_request is None or self._preload_lora_into_engine():
            self._track_lora_request_activation(lora_request)
            async for result in create_generator(lora_request):
                yield result
            return

        lock = self._get_lora_lock(lora_request.lora_name)
        async with lock:
            # The adapter may have been unloaded or reloaded at a different path
            # while this request waited. Look it up again while holding the lock.
            admitted_lora_request = self._resolve_lora_request(lora_request.lora_name)
            if admitted_lora_request is None:
                logger.warning(
                    "LoRA adapter %s was unloaded before vLLM admission; "
                    "rejecting the request",
                    lora_request.lora_name,
                )
                raise ValueError(
                    f"unknown model or LoRA adapter: '{lora_request.lora_name}'"
                )
            generator = create_generator(admitted_lora_request)
            self._track_lora_request_activation(admitted_lora_request)
            try:
                first_result = await anext(generator)
            except StopAsyncIteration:
                return

        yield first_result
        async for result in generator:
            yield result

    def _preload_lora_into_engine(self) -> bool:
        """Whether lifecycle registration should eagerly activate the adapter.

        Prefill keeps the downloaded adapter metadata and supplies its path in
        the inference-time ``LoRARequest``. Decode and aggregated workers must
        be immediately ready to generate and therefore continue to preload.
        """
        return self.config.disaggregation_mode != DisaggregationMode.PREFILL

    async def load_lora(self, request=None):
        """
        Load a LoRA adapter dynamically into the vLLM's AsyncLLM engine.

        Request format:
        {
            "lora_name": str,
            "source": {
                "uri": str  # e.g., "s3://bucket/path" or "file:///path"
            }
        }

        Concurrent calls for the same LoRA are serialized. Re-loading an already
        loaded LoRA is idempotent by default. Set
        ``DYN_LORA_HOTSWAP_ENABLED=true`` to replace an already loaded LoRA with
        a new URI.
        """
        try:
            try:
                lora_name, lora_uri = require_lora_load_request(request)
            except RLAdminValidationError as e:
                yield {"status": "error", "message": str(e)}
                return

            # Debug: Log the incoming request
            logger.debug(f"load_lora request keys: {list(request.keys())}")
            logger.debug(f"load_lora request: {request}")

            # Use LoRAManager to download from URI
            lora_manager = get_lora_manager()
            if lora_manager is None:
                yield {
                    "status": "error",
                    "message": "LoRAManager not initialized. Set DYN_LORA_ENABLED=true to enable URI-based LoRA loading.",
                }
                return

            # Serialize load/unload operations per lora_name.
            lock = self._get_lora_lock(lora_name)
            async with lock:
                try:
                    old_info = self.loaded_loras.get(lora_name)
                    hot_swap_enabled = env_bool("DYN_LORA_HOTSWAP_ENABLED")
                    is_hot_swap = old_info is not None and hot_swap_enabled
                    old_engine_loaded = lora_name in self._engine_loaded_loras

                    if old_info is not None and not hot_swap_enabled:
                        logger.info(
                            f"LoRA adapter already loaded: {lora_name} "
                            f"with ID {old_info.id}"
                        )
                        yield {
                            "status": "success",
                            "message": f"LoRA adapter '{lora_name}' already loaded",
                            "lora_name": lora_name,
                            "lora_id": old_info.id,
                            "hot_swap": False,
                        }
                        return

                    logger.info(
                        f"Downloading LoRA adapter: {lora_name} from {lora_uri}"
                    )
                    download_result = await lora_manager.download_lora(lora_uri)

                    if download_result["status"] != "success":
                        yield {
                            "status": "error",
                            "message": f"Failed to download LoRA: {download_result.get('message', 'Unknown error')}",
                        }
                        return

                    lora_path = download_result["local_path"]
                    logger.debug(f"LoRA downloaded to: {lora_path}")

                    # Generate deterministic ID from lora_name before using it
                    lora_id = lora_name_to_id(lora_name)

                    if is_hot_swap and old_info is not None and old_engine_loaded:
                        try:
                            await self.engine_client.remove_lora(old_info.id)
                            self._engine_loaded_loras.discard(lora_name)
                        except Exception as e:
                            logger.error(
                                f"Failed to remove existing LoRA '{lora_name}' "
                                f"before hot-swap: {e}"
                            )
                            yield {
                                "status": "error",
                                "message": (
                                    f"Failed to remove existing LoRA '{lora_name}' "
                                    f"before hot-swap: {e}"
                                ),
                                "lora_name": lora_name,
                            }
                            return

                    # Initial prefill registration is metadata-only. A hot
                    # swap must still replace any lazily activated old adapter
                    # atomically before the prefix cache is reset.
                    preload_into_engine = (
                        self._preload_lora_into_engine() or is_hot_swap
                    )
                    if preload_into_engine:
                        try:
                            await self.engine_client.add_lora(
                                LoRARequest(
                                    lora_name=lora_name,
                                    lora_int_id=lora_id,
                                    lora_path=lora_path,
                                )
                            )
                            self._engine_loaded_loras.add(lora_name)
                        except Exception as e:
                            if (
                                is_hot_swap
                                and old_info is not None
                                and old_engine_loaded
                            ):
                                try:
                                    await self.engine_client.add_lora(
                                        LoRARequest(
                                            lora_name=lora_name,
                                            lora_int_id=old_info.id,
                                            lora_path=old_info.path,
                                        )
                                    )
                                    self._engine_loaded_loras.add(lora_name)
                                except Exception as rollback_error:
                                    self.loaded_loras.pop(lora_name, None)
                                    logger.exception(
                                        f"Rollback failed for LoRA {lora_name}: "
                                        f"{rollback_error}"
                                    )
                            yield {
                                "status": "error",
                                "message": f"Failed to add LoRA '{lora_name}': {e}",
                                "lora_name": lora_name,
                            }
                            return

                    # Track the LoRA
                    self.loaded_loras[lora_name] = LoRAInfo(id=lora_id, path=lora_path)
                    logger.info(
                        f"Successfully {'hot-swapped' if is_hot_swap else 'loaded'} "
                        f"LoRA adapter: {lora_name} with ID {lora_id}"
                    )

                    if is_hot_swap:
                        try:
                            await self.engine_client.reset_prefix_cache()
                        except Exception as e:
                            # The new adapter is already active in the engine, but
                            # the prefix cache still holds entries computed under
                            # the old adapter and could be reused incorrectly.
                            # Roll the ENGINE back to old_info (remove new, re-add
                            # old) so engine state and our tracking stay consistent
                            # — a metadata-only rollback would leave the new adapter
                            # live while we report/route the old one (codex).
                            rolled_back = "tracking only"
                            if old_info is not None:
                                try:
                                    if preload_into_engine:
                                        await self.engine_client.remove_lora(lora_id)
                                        self._engine_loaded_loras.discard(lora_name)
                                    if old_engine_loaded:
                                        await self.engine_client.add_lora(
                                            LoRARequest(
                                                lora_name=lora_name,
                                                lora_int_id=old_info.id,
                                                lora_path=old_info.path,
                                            )
                                        )
                                        self._engine_loaded_loras.add(lora_name)
                                    self.loaded_loras[lora_name] = old_info
                                    rolled_back = (
                                        "engine+tracking"
                                        if old_engine_loaded
                                        else "tracking only"
                                    )
                                except Exception as rollback_error:
                                    # Engine is in an indeterminate adapter state;
                                    # drop tracking so we never claim a clean swap.
                                    self.loaded_loras.pop(lora_name, None)
                                    logger.exception(
                                        f"LoRA '{lora_name}' hot-swap engine "
                                        f"rollback failed: {rollback_error}"
                                    )
                            else:
                                self.loaded_loras.pop(lora_name, None)
                            logger.error(
                                f"LoRA '{lora_name}' hot-swap rolled back "
                                f"({rolled_back}): prefix cache reset failed: {e}"
                            )
                            yield {
                                "status": "error",
                                "message": (
                                    f"LoRA '{lora_name}' hot-swap aborted; prefix "
                                    f"cache reset failed: {e}"
                                ),
                                "lora_name": lora_name,
                                "lora_id": lora_id,
                            }
                            return

                    # Publish LoRA as a ModelDeploymentCard with format:
                    # v1/mdc/{namespace}/{component}/{endpoint}/{instance_id}/{lora_slug}
                    # This allows the frontend to discover it and route correctly to the worker instance
                    if not is_hot_swap and self.generate_endpoint is not None:
                        logger.debug(
                            f"Publishing LoRA '{lora_name}' ModelDeploymentCard to {self.generate_endpoint}"
                        )
                        try:
                            logger.debug(
                                f"Publishing LoRA '{lora_name}' ModelDeploymentCard"
                            )

                            # Mark this as a LoRA in user_data
                            user_data = {
                                "lora_adapter": True,
                                "lora_id": lora_id,
                            }

                            runtime_config = ModelRuntimeConfig()
                            runtime_config.context_length = self.model_max_len
                            runtime_config.tool_call_parser = (
                                self.config.dyn_tool_call_parser
                            )
                            runtime_config.reasoning_parser = (
                                self.config.dyn_reasoning_parser
                            )

                            # Match the base-model registration topology (see
                            # worker_factory.py _create_decode_worker /
                            # _create_prefill_worker) so the router activates for the
                            # LoRA model name the same way it does for the base model.
                            # A prefill worker carries its role via worker_type=Prefill;
                            # we register the legacy ModelType.Prefill marker bit (not a
                            # surface) so an old frontend still detects it during the
                            # cross-version rollout. Decode and aggregated workers expose the
                            # LoRA on the same chat/completions surface.
                            # --route-to-encoder adds Encode to the AND-set of peers.
                            if (
                                self.config.disaggregation_mode
                                == DisaggregationMode.PREFILL
                            ):
                                lora_model_type = ModelType.Prefill
                                lora_worker_type = WorkerType.Prefill
                                lora_needs_set: list[WorkerType] = [WorkerType.Decode]
                            elif (
                                self.config.disaggregation_mode
                                == DisaggregationMode.DECODE
                            ):
                                lora_model_type = ModelType.Chat | ModelType.Completions
                                lora_worker_type = WorkerType.Decode
                                lora_needs_set = [WorkerType.Prefill]
                            else:  # AGGREGATED
                                lora_model_type = ModelType.Chat | ModelType.Completions
                                lora_worker_type = WorkerType.Aggregated
                                lora_needs_set = []
                            if self.config.route_to_encoder:
                                lora_needs_set.append(WorkerType.Encode)
                            lora_needs: list[list[WorkerType]] = (
                                [lora_needs_set] if lora_needs_set else []
                            )

                            # Publish with format: v1/mdc/dynamo/backend/generate/{instance_id}/{lora_slug}
                            # Use the engine's main-attention KV block size — the
                            # same value the base-model registration publishes —
                            # not the CLI arg. On hybrid-attention models vLLM
                            # inflates the attention block size at engine init
                            # (e.g. 16 -> 1056), so a card that carries the CLI
                            # value makes the frontend's KV router book blocks in
                            # units far smaller than the worker's real blocks and
                            # falsely mark the worker overloaded.
                            await register_model(
                                model_input=ModelInput.Tokens,
                                model_type=lora_model_type,
                                endpoint=self.generate_endpoint,
                                model_path=self.config.model,
                                kv_cache_block_size=get_configured_kv_event_block_size(
                                    self.engine_client.vllm_config
                                ),
                                runtime_config=runtime_config,
                                user_data=user_data,
                                lora_name=lora_name,
                                base_model_path=self.config.model,
                                worker_type=lora_worker_type,
                                needs=lora_needs,
                                # Publish the worker's per-worker LoRA slot budget so the frontend
                                # allocator sizes placement against real capacity instead of the
                                # hard-coded default.
                                max_gpu_lora_count=getattr(
                                    self.config.engine_args, "max_loras", None
                                ),
                            )
                            logger.info(
                                f"Successfully published LoRA '{lora_name}' ModelDeploymentCard"
                            )
                        except Exception as e:
                            logger.exception(
                                f"Failed to publish LoRA {lora_name} ModelDeploymentCard: {e}"
                            )

                            # Roll back engine state when this worker preloaded;
                            # prefill only needs to discard the cached metadata.
                            try:
                                if preload_into_engine:
                                    logger.debug(
                                        f"Rolling back: removing LoRA '{lora_name}' from engine"
                                    )
                                    await self.engine_client.remove_lora(lora_id)
                                    self._engine_loaded_loras.discard(lora_name)
                                self.loaded_loras.pop(lora_name, None)
                                logger.debug(
                                    f"Successfully rolled back LoRA '{lora_name}'"
                                )
                            except Exception as rollback_error:
                                logger.exception(
                                    f"Failed to rollback LoRA {lora_name}: {rollback_error}"
                                )

                            # Return error status since registration failed
                            yield {
                                "status": "error",
                                "message": f"Failed to register LoRA '{lora_name}' in discovery registry: {str(e)}",
                                "lora_name": lora_name,
                            }
                            return
                    elif not is_hot_swap:
                        logger.debug(
                            f"Cannot publish LoRA '{lora_name}': generate_endpoint={self.generate_endpoint}, config={self.config}"
                        )

                    yield {
                        "status": "success",
                        "message": (
                            f"LoRA adapter '{lora_name}' "
                            f"{'hot-swapped' if is_hot_swap else 'loaded'} successfully"
                        ),
                        "lora_name": lora_name,
                        "lora_id": lora_id,
                        "hot_swap": is_hot_swap,
                    }
                finally:
                    # Stripes are intentionally retained. Evicting a lock here
                    # can separate a waiting request from a later lifecycle op.
                    pass
        except Exception as e:
            logger.exception(f"Failed to load LoRA adapter: {e}")
            yield {"status": "error", "message": str(e)}

    async def unload_lora(self, request=None):
        """
        Unload a LoRA adapter dynamically from the vLLM's AsyncLLM engine.
        Expected request format:
        {
            "lora_name": str,
        }
        """
        try:
            try:
                lora_name = require_lora_unload_request(request)
            except RLAdminValidationError as e:
                yield {"status": "error", "message": str(e)}
                return

            # Serialize load/unload operations per lora_name.
            lock = self._get_lora_lock(lora_name)
            async with lock:
                try:
                    # Check if the LoRA exists *after* waiting for any in-progress load.
                    lora = self.loaded_loras.get(lora_name)
                    if lora is None:
                        yield {
                            "status": "error",
                            "message": f"LoRA adapter '{lora_name}' not found. Available LoRAs: {list(self.loaded_loras.keys())}",
                        }
                        return

                    logger.debug(f"Unloading LoRA adapter: {lora_name}")
                    lora_id = lora.id

                    # Stop advertising the adapter before mutating engine or
                    # tracking state. Otherwise requests can still route here
                    # after _resolve_lora_request has forgotten the adapter and
                    # silently execute against the base model.
                    if self.generate_endpoint is not None:
                        logger.debug(
                            f"Unregistering LoRA '{lora_name}' ModelDeploymentCard"
                        )
                        try:
                            await unregister_model(
                                endpoint=self.generate_endpoint,
                                lora_name=lora_name,
                            )
                            logger.info(
                                f"Successfully unregistered LoRA '{lora_name}' ModelDeploymentCard"
                            )
                        except Exception as e:
                            logger.exception(
                                f"Failed to unregister LoRA {lora_name} ModelDeploymentCard: {e}"
                            )
                            yield {
                                "status": "error",
                                "message": f"Failed to unregister LoRA '{lora_name}' from discovery registry: {str(e)}",
                                "lora_name": lora_name,
                            }
                            return
                    else:
                        logger.debug(
                            f"Cannot unregister LoRA '{lora_name}': generate_endpoint={self.generate_endpoint}"
                        )

                    # Prefill lifecycle registration is metadata-only, but
                    # vLLM may have activated the adapter lazily for an
                    # inference request. Remove only adapters known to have
                    # reached vLLM.
                    if lora_name in self._engine_loaded_loras:
                        try:
                            await self.engine_client.remove_lora(lora_id)
                        except Exception as e:
                            if not self._is_lora_not_loaded_error(e):
                                raise
                        self._engine_loaded_loras.discard(lora_name)
                    del self.loaded_loras[lora_name]

                    logger.info(
                        f"Successfully unloaded LoRA adapter: {lora_name} with ID {lora_id}"
                    )
                    yield {
                        "status": "success",
                        "message": f"LoRA adapter '{lora_name}' unloaded successfully",
                        "lora_name": lora_name,
                        "lora_id": lora_id,
                    }
                finally:
                    # Stripes are intentionally retained. Evicting a lock here
                    # can separate a waiting request from a later lifecycle op.
                    pass
        except Exception as e:
            logger.exception(f"Failed to unload LoRA adapter: {e}")
            yield {"status": "error", "message": str(e)}

    async def list_loras(self, request=None):
        """
        List all loaded LoRA adapters.
        Returns a dictionary of lora_name -> lora_id mappings.
        """
        try:
            loras = {name: lora.id for name, lora in self.loaded_loras.items()}
            yield {
                "status": "success",
                "loras": loras,
                "count": len(loras),
            }
        except Exception as e:
            logger.error(f"Failed to list LoRA adapters: {e}")
            yield {"status": "error", "message": str(e)}

    def cleanup(self):
        """Clean up resources including temporary directories."""
        if self._custom_encoder is not None:
            # Run backend.close() on the actor thread, then stop it — executor
            # GC would only end the thread, never call close().
            self._custom_encoder.shutdown()
            self._custom_encoder = None
            self._custom_encoder_adapter = None
        for temp_dir in self.temp_dirs:
            try:
                temp_dir.cleanup()
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {e}")

    def _decode_prompt_embeds(self, prompt_embeds_base64: str):
        """
        Decode base64-encoded prompt embeddings in PyTorch format.

        Use vllm's safe loader to prevent out-of-bounds writes from maliciously crafted tensors.

        Format: PyTorch tensor serialized with torch.save() and base64-encoded.

        Args:
            prompt_embeds_base64: Base64-encoded PyTorch tensor

        Returns:
            torch.Tensor: Decoded prompt embeddings with dim == 2

        Raises:
            ValueError: If decoding fails or format is invalid
        """
        if not isinstance(prompt_embeds_base64, str):
            raise ValueError(
                f"Prompt embeds must be base64 encoded string. Got {type(prompt_embeds_base64)}."
            )

        if self.model_config is None:
            raise ValueError("ModelConfig is unavailable for prompt_embeds validation.")

        try:
            return safe_load_prompt_embeds(
                self.model_config, prompt_embeds_base64.encode()
            )
        except Exception as e:
            logger.error(f"Failed to decode prompt_embeds: {e}")
            raise ValueError(f"Failed to decode prompt_embeds as PyTorch tensor: {e}")

    def _create_prompt_from_embeddings(
        self, prompt_embeds_base64: str
    ) -> tuple[EmbedsPrompt, torch.Tensor]:
        """
        Decode prompt embeddings and create EmbedsPrompt for vLLM.

        Args:
            prompt_embeds_base64: Base64-encoded PyTorch tensor

        Returns:
            Tuple of (EmbedsPrompt, tensor) where:
            - EmbedsPrompt: The vLLM prompt input
            - tensor: The decoded tensor (for logging shape/dtype)

        Raises:
            ValueError: If decoding fails or tensor is invalid
        """
        embeddings_tensor = self._decode_prompt_embeds(prompt_embeds_base64)
        if embeddings_tensor.dim() != 2:
            raise ValueError(
                f"prompt embeds should have dim 2 after vllm processing, but found dim {embeddings_tensor.dim()}"
            )

        # EmbedsInputs TypedDict has: {type: 'embeds', prompt_embeds: Tensor, cache_salt?: str}
        prompt = EmbedsPrompt(prompt_embeds=embeddings_tensor)

        return prompt, embeddings_tensor

    def _build_prompt_from_request(
        self,
        request: Dict[str, Any],
        request_id: str,
        multi_modal_data: Dict[str, Any] | None,
        log_prefix: str = "",
        mm_processor_kwargs: Dict[str, Any] | None = None,
    ) -> tuple[TokensPrompt | EmbedsPrompt | None, Dict[str, Any] | None]:
        """
        Build a prompt from request, handling both prompt_embeds and token_ids.

        Args:
            request: The request dict containing either prompt_embeds or token_ids
            request_id: Request ID for logging
            multi_modal_data: Optional multimodal data to attach to TokensPrompt
            log_prefix: Prefix for log messages (e.g., "Prefill " for prefill requests)
            mm_processor_kwargs: Optional multimodal processor kwargs (e.g.
                use_audio_in_video) forwarded to the vLLM engine.

        Returns:
            Tuple of (prompt, error_dict) where:
            - On success: (prompt, None)
            - On failure: (None, error_dict to yield)
        """
        if "prompt_embeds" in request and request["prompt_embeds"]:
            if not self.config.engine_args.enable_prompt_embeds:
                msg = (
                    "Set `--enable-prompt-embeds` to allow `prompt_embeds` in request."
                )
                logger.error(
                    f"Rejected prompt_embeds for {log_prefix.lower().strip() or 'request'} "
                    f"{request_id}: {msg}"
                )
                return (
                    None,
                    {
                        "finish_reason": f"error: Invalid prompt_embeds: {msg}",
                        "token_ids": [],
                    },
                )
            try:
                prompt, tensor = self._create_prompt_from_embeddings(
                    request["prompt_embeds"]
                )
                logger.info(
                    f"{log_prefix}Using prompt embeddings: shape={tensor.shape}, "
                    f"dtype={tensor.dtype}, sequence_length={tensor.shape[0]}, "
                    f"request_id={request_id}"
                )
                return prompt, None
            except Exception as e:
                logger.error(
                    f"Failed to process prompt_embeds for {log_prefix.lower().strip() or 'request'} "
                    f"{request_id}: {e}"
                )
                return (
                    None,
                    {
                        "finish_reason": f"error: Invalid prompt_embeds: {e}",
                        "token_ids": [],
                    },
                )
        # Text-only PD + encoder-worker path.
        # Normal path: use token IDs.
        # Prefer frontend-forwarded mm_hashes for hash consistency with the
        # routing layer. Fall back to computing from loaded image data when
        # not in EPD mode — in EPD mode multi_modal_data carries pre-computed
        # embeddings from the encode worker, not raw images, and raw-image
        # identity lives upstream at the Router / URL-keyed encoder cache.
        prompt = self._multimodal_request_processor.build_tokens_prompt(
            request,
            multi_modal_data,
            mm_processor_kwargs,
        )
        return prompt, None

    @staticmethod
    def _build_completion_usage(
        request_output: RequestOutput,
        completion_token_counts: dict[int, int] | None = None,
    ) -> Dict[str, Any]:
        """
        Build completion usage statistics.

        Args:
            request_output: vLLM RequestOutput object
            completion_token_counts: Optional cumulative generated-token counts by
                                     output index. DELTA-mode streams need this
                                     because the final vLLM chunk is not cumulative.

        Returns:
            Dict with prompt_tokens, completion_tokens, total_tokens, prompt_tokens_details
        """
        prompt_tokens = (
            len(request_output.prompt_token_ids)
            if request_output.prompt_token_ids
            else None
        )

        if completion_token_counts is not None:
            completion_tokens = sum(completion_token_counts.values())
        else:
            completion_tokens = sum(
                len(output.token_ids) for output in request_output.outputs
            )

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": (
                prompt_tokens + completion_tokens if prompt_tokens is not None else None
            ),
            "prompt_tokens_details": build_prompt_tokens_details(
                getattr(request_output, "num_cached_tokens", None)
            ),
        }

    @staticmethod
    def _extract_logprobs(
        output, num_output_tokens_so_far: int, tokenizer=None
    ) -> tuple[list[float] | None, list[list[dict]] | None]:
        # Legacy vLLM handler always emits when vLLM returned a dict.
        return _shared_logprobs.extract_from_completion_output(
            output,
            num_output_tokens_so_far,
            tokenizer=tokenizer,
            fallback_to_first_on_missing=True,
            include_bytes=True,
        )

    @staticmethod
    def _log_with_lora_context(
        message: str,
        request_id: str,
        lora_request=None,
        level: str = "debug",
        **kwargs,
    ) -> None:
        """
        Log a message with optional LoRA context.

        Args:
            message: Base message to log (can include {lora_info} placeholder)
            request_id: Request ID for correlation
            lora_request: Optional LoRA request object
            level: Log level ("debug" or "info")
            **kwargs: Additional format arguments for the message
        """
        if lora_request:
            lora_info = f" with LoRA {lora_request.lora_name}"
        else:
            lora_info = ""

        formatted_message = message.format(
            request_id=request_id,
            lora_info=lora_info,
            **kwargs,
        )

        if level == "info":
            logger.info(formatted_message)
        else:
            logger.debug(formatted_message)

    async def generate_tokens(
        self,
        prompt,
        sampling_params,
        request_id,
        data_parallel_rank=None,
        lora_request=None,
        trace_headers=None,
        priority=0,
        reasoning_ended=None,
        reasoning_parser_kwargs=None,
    ):
        try:
            # Log LoRA usage for this generation (debug level to avoid log spam)
            self._log_with_lora_context(
                "Starting token generation for request {request_id}{lora_info}",
                request_id,
                lora_request,
            )
            gen = self._generate_with_lora_admission_lock(
                lora_request,
                lambda admitted_lora_request: self.engine_client.generate(
                    prompt,
                    sampling_params,
                    request_id,
                    lora_request=admitted_lora_request,
                    data_parallel_rank=data_parallel_rank,
                    trace_headers=trace_headers,
                    priority=priority,
                    **_engine_generate_reasoning_kwargs(
                        self.engine_client,
                        reasoning_ended,
                        reasoning_parser_kwargs,
                    ),
                ),
            )

            total_output_tokens_by_index: dict[int, int] = {}
            raw_routed_experts_by_output: dict[int, Any] = {}
            # vLLM surfaces prompt_logprobs once (at end-of-prefill) and clears
            # them on subsequent chunks, so the generation-finish chunk often
            # carries None. Capture the first non-None payload and attach it to
            # the final chunk instead of reading res.prompt_logprobs there.
            prompt_logprobs_payload: Optional[list] = None
            async for res in gen:
                # res is vllm's RequestOutput
                if (
                    prompt_logprobs_payload is None
                    and getattr(res, "prompt_logprobs", None) is not None
                ):
                    prompt_logprobs_payload = _serialize_prompt_logprobs(
                        res.prompt_logprobs
                    )

                if not res.outputs:
                    self._log_with_lora_context(
                        "Request {request_id}{lora_info} returned no outputs",
                        request_id,
                        lora_request,
                    )
                    # Use string format "error: message" for consistency with vLLM's string-based finish_reason
                    # Rust will parse this into FinishReason::Error(message)
                    yield {
                        "finish_reason": "error: No outputs from vLLM engine",
                        "index": 0,
                        "token_ids": [],
                    }
                    break

                prepared_outputs = []
                for output in res.outputs:
                    output_idx = getattr(output, "index", 0) or 0
                    token_ids = list(output.token_ids or [])
                    total_output_tokens_by_index[
                        output_idx
                    ] = total_output_tokens_by_index.get(output_idx, 0) + len(token_ids)
                    finish_reason = getattr(output, "finish_reason", None)
                    stop_reason = getattr(output, "stop_reason", None)
                    if not token_ids and not finish_reason and not stop_reason:
                        continue
                    prepared_outputs.append(
                        (output, output_idx, token_ids, finish_reason, stop_reason)
                    )

                for (
                    output,
                    output_idx,
                    token_ids,
                    finish_reason,
                    stop_reason,
                ) in prepared_outputs:
                    out = {
                        "index": output_idx,
                        "token_ids": token_ids,
                    }
                    # Capture the raw routed_experts cheaply here; serialize it
                    # only once on the final chunk (base64-encoding a tensor on
                    # every streamed chunk would be wasted work, since only the
                    # final value is emitted).
                    raw_routed_experts = getattr(output, "routed_experts", None)
                    if raw_routed_experts is not None:
                        raw_routed_experts_by_output[output_idx] = raw_routed_experts

                    # vLLM DELTA outputs already align token_ids/logprobs to this chunk.
                    tokenizer = getattr(self.engine_client, "tokenizer", None)
                    log_probs, top_logprobs = self._extract_logprobs(
                        output, 0, tokenizer=tokenizer
                    )
                    if log_probs is not None:
                        out["log_probs"] = log_probs
                    if top_logprobs is not None:
                        out["top_logprobs"] = top_logprobs

                    if finish_reason:
                        out["finish_reason"] = normalize_finish_reason(finish_reason)
                        out[
                            "completion_usage"
                        ] = BaseWorkerHandler._build_completion_usage(
                            request_output=res,
                            completion_token_counts=total_output_tokens_by_index,
                        )
                        if prompt_logprobs_payload is not None:
                            _attach_prompt_logprobs_engine_data(
                                out, prompt_logprobs_payload
                            )
                        # Emit the EFFECTIVE trim offset: clamp the requested
                        # routed_experts_prompt_start to the prompt length. vLLM
                        # clamps the returned routing rows the same way, so an
                        # out-of-range request (e.g. start=999 on a 100-token
                        # prompt) would otherwise publish a `start` the consumer
                        # cannot align to the (clamped) tensor.
                        raw_start = int(
                            getattr(sampling_params, "routed_experts_prompt_start", 0)
                            or 0
                        )
                        prompt_len = len(getattr(res, "prompt_token_ids", None) or [])
                        effective_start = min(raw_start, prompt_len)
                        routed_experts = _serialize_routed_experts(
                            raw_routed_experts_by_output.get(output_idx),
                            start=effective_start,
                        )
                        if routed_experts is not None:
                            _attach_routed_experts_engine_data(out, routed_experts)
                        # Log completion with LoRA info (debug level to avoid log spam)
                        self._log_with_lora_context(
                            "Completed token generation for request {request_id}{lora_info}: "
                            "{output_tokens} output tokens, finish_reason={finish_reason}",
                            request_id,
                            lora_request,
                            output_tokens=total_output_tokens_by_index.get(
                                output_idx, 0
                            ),
                            finish_reason=finish_reason,
                        )
                    if stop_reason:
                        out["stop_reason"] = stop_reason
                    yield out

        except EngineDeadError as e:
            logger.error(f"vLLM EngineDeadError: {e}")
            logger.warning("Initiating Dynamo Runtime shutdown.")
            self.runtime.shutdown()
            os._exit(1)


class DecodeWorkerHandler(BaseWorkerHandler):
    def __init__(
        self,
        runtime,
        config: Config,
        engine,
        default_sampling_params,
        model_max_len: int | None = None,
        model_config: ModelConfig | None = None,
        enable_multimodal: bool = False,
        generate_endpoint=None,
        use_vllm_tokenizer: bool = False,
        shutdown_event: asyncio.Event | None = None,
        enable_frontend_decoding: bool = False,
        encode_worker_client: Client | None = None,
    ):
        super().__init__(
            runtime,
            config,
            engine,
            default_sampling_params,
            model_max_len=model_max_len,
            model_config=model_config,
            enable_multimodal=enable_multimodal,
            generate_endpoint=generate_endpoint,
            use_vllm_tokenizer=use_vllm_tokenizer,
            shutdown_event=shutdown_event,
            enable_frontend_decoding=enable_frontend_decoding,
            encode_worker_client=encode_worker_client,
        )

    async def generate(self, request, context):
        # Use context ID for request tracking and correlation
        request_id = context.id()
        logger.debug(f"Decode Request ID: {request_id}")
        self._multimodal_request_processor.validate_multimodal_request(request)
        first_token = True
        with time_and_log_code_section(
            f"[DECODE] request: {request_id} generate"
        ) as decode_timer:
            if self.use_vllm_tokenizer:
                # Text-in-text-out mode: use InputParamManager and OpenAI-compatible format
                generator = self._generate_text_mode(request, context, request_id)
            else:
                # Token-in-token-out mode: internal protocol format
                generator = self._generate_token_mode(request, context, request_id)

            async for chunk in generator:
                if first_token:
                    decode_timer.stop_interval()
                    first_token = False
                yield chunk

    async def _assemble_custom_encoder_prompt(
        self,
        request: Dict[str, Any],
        request_id: str,
    ) -> tuple[EmbedsPrompt | TokensPrompt | None, Dict[str, Any] | None]:
        """Run the in-process CustomEncoder and prepare its engine prompt.

        The CustomEncoder consumes image URLs directly and emits artifacts. Returns
        ``(prepared_prompt, error)``:
        - images present: ``(prepared_prompt, None)``,
        - no image content: ``(None, None)`` — text-only request, nothing
          to assemble or extract (non-image modalities are rejected above),
        - failure: ``(None, error_dict)`` for the caller to yield.
        """
        # Internal invariant: callers guard on `self._custom_encoder is not None`
        # before reaching here. Use an explicit raise (not assert, which is
        # stripped under `python -O`) so a future mis-wire fails loudly.
        if self._custom_encoder is None:
            raise RuntimeError(
                "_assemble_custom_encoder_prompt called without a CustomEncoder"
            )
        if self._custom_encoder_adapter is None:
            raise RuntimeError(
                "_assemble_custom_encoder_prompt called without an adapter"
            )
        mm_map = request.get("multi_modal_data") or {}
        # CustomEncoder handles images only. Reject any non-image modality
        # (video/audio/...) explicitly instead of silently dropping it.
        unsupported = sorted(k for k in mm_map if k != IMAGE_URL_KEY and mm_map.get(k))
        if unsupported:
            msg = (
                "CustomEncoder supports image inputs only; got "
                f"unsupported multimodal data: {unsupported}"
            )
            logger.error("Request %s: %s", request_id, msg)
            return None, {"finish_reason": f"error: {msg}", "token_ids": []}

        image_items = mm_map.get(IMAGE_URL_KEY) or []
        image_urls = [
            item[URL_VARIANT_KEY]
            for item in image_items
            if isinstance(item, dict) and URL_VARIANT_KEY in item
        ]
        if len(image_urls) != len(image_items):
            # At least one image item was malformed — not a dict with a 'Url'
            # key (e.g. a pre-'Decoded' variant the CustomEncoder can't take).
            # Reject the whole request instead of silently dropping images.
            msg = (
                "CustomEncoder received image multimodal data but only "
                f"{len(image_urls)} of {len(image_items)} item(s) had a usable "
                "'Url'; each item must be a dict with a 'Url' key"
            )
            logger.error("Request %s: %s", request_id, msg)
            return None, {"finish_reason": f"error: {msg}", "token_ids": []}

        if not image_urls:
            # No image items at all — and non-image modalities were already
            # rejected above — so there is nothing to assemble → text-only.
            return None, None

        token_ids: list[int] = request.get("token_ids") or []
        # Both encode() and adapter preparation run user/model-specific code, so
        # keep them inside one guard. A failure becomes a structured request error
        # instead of escaping the coroutine and tearing down the stream.
        try:
            # AsyncVisionEncoder preprocesses off-thread; its ThreadedMicroBatcher
            # coalesces concurrent calls onto one dedicated actor thread.
            encodings = await self._custom_encoder.encode(image_urls)
            prepared = self._custom_encoder_adapter.prepare_prompt(
                token_ids,
                encodings,
            )
        except Exception as exc:
            msg = f"CustomEncoder failed: {exc}"
            logger.exception("Request %s: %s", request_id, msg)
            return None, {"finish_reason": f"error: {msg}", "token_ids": []}

        logger.debug(
            "Request %s: CustomEncoder prepared prompt for %d image(s)",
            request_id,
            len(encodings),
        )
        return prepared, None

    async def _generate_token_mode(self, request, context, request_id):
        """Generate tokens using internal protocol format (token-in-token-out)."""
        # Firstly extract disaggregated params from prefill result if available
        prefill_result = request.get("prefill_result")
        if prefill_result and isinstance(prefill_result, dict):
            # `disaggregated_params` may be explicitly None (prefill error path,
            # or _build_disaggregated_params returning None when empty), so use
            # `or {}` rather than a .get default — the default only applies when
            # the key is absent, not when it is present-but-None.
            disaggregated_params = prefill_result.get("disaggregated_params") or {}
            kv_params = disaggregated_params.get("kv_transfer_params")
        else:
            kv_params = None

        mode = cast(DisaggregationMode, self.config.disaggregation_mode)
        is_decode_only = mode == DisaggregationMode.DECODE
        has_mm_data = request.get("multi_modal_data") is not None
        custom_prompt: EmbedsPrompt | TokensPrompt | None = None

        if (
            mode == DisaggregationMode.AGGREGATED
            and self._custom_encoder is not None
            and has_mm_data
        ):
            # A configured CustomEncoder owns the aggregated image path. Bypass
            # raw-media loading and let its decoder-selected adapter prepare the
            # final engine prompt.
            custom_prompt, assemble_error = await self._assemble_custom_encoder_prompt(
                request,
                request_id,
            )
            if assemble_error is not None:
                yield assemble_error
                return
            multi_modal_data = None
            mm_processor_kwargs = None
            pre_rendered = None
        else:
            try:
                prepared_input = await self._multimodal_request_processor.prepare_input(
                    request,
                    request_id,
                    context,
                    mode,
                )
            except MissingMultimodalHandoffError as exc:
                logger.error("Request %s: %s", request_id, exc)
                yield {
                    "finish_reason": f"error: {exc}",
                    "index": 0,
                    "token_ids": [],
                }
                return

            request = prepared_input.request
            multi_modal_data = prepared_input.multi_modal_data
            mm_processor_kwargs = prepared_input.mm_processor_kwargs
            pre_rendered = prepared_input.pre_rendered_prompt

        # Build prompt from request. `prompt` is either a pre-rendered
        # MultiModalInput dict (fast path) or a TokensPrompt/EmbedsPrompt from
        # `_build_prompt_from_request`. Declare as Any so mypy accepts both
        # branches without spelling out the full union.
        prompt: Any
        with _nvtx.annotate("mm_backend:build_prompt", color="yellow"):
            if custom_prompt is not None:
                prompt = custom_prompt
                error = None
            elif pre_rendered is not None:
                # pre_rendered is a MultiModalInput dict with "type": "multimodal".
                # The engine's InputProcessor.process_inputs() will see the "type"
                # key and skip the HF processor entirely.
                prompt = pre_rendered
                error = None
                logger.debug(
                    "[mm-routing] Request %s: using pre-rendered MultiModalInput",
                    request_id,
                )
            else:
                prompt, error = self._build_prompt_from_request(
                    request,
                    request_id,
                    multi_modal_data,
                    mm_processor_kwargs=mm_processor_kwargs,
                )
        if error is not None:
            yield error
            return

        _apply_nvext_cache_salt(request, prompt)

        # Build sampling params from request
        sampling_params = build_sampling_params(
            request,
            self.default_sampling_params,
            self.model_max_len,
            enable_rl=self.config.enable_rl,
        )

        if kv_params is not None:
            if sampling_params.extra_args is None:
                sampling_params.extra_args = {}
            sampling_params.extra_args["kv_transfer_params"] = kv_params
            logger.debug(
                f"Using disaggregated params from prefill for request {request_id}"
            )
        prefill_prompt_tokens_details = (
            prefill_result.get("prompt_tokens_details") if prefill_result else None
        )

        # Extract LoRA request if present
        model_name = request.get("model")
        lora_request = self._resolve_lora_request(model_name)
        if lora_request:
            logger.info(
                f"Decode request {request_id} will use LoRA adapter: {model_name} (ID: {lora_request.lora_int_id})"
            )
        else:
            logger.debug(
                f"Decode request {request_id} has no LoRA specified (model: {model_name})"
            )
        routing = request.get("routing") or {}
        dp_rank = self._to_local_dp_rank(routing.get("dp_rank"))
        priority = -int(routing.get("priority", 0))

        trace_headers = context.trace_headers()
        reasoning_ended, reasoning_parser_kwargs = _request_reasoning_metadata(request)

        # In disagg decode mode, defer engine_client.abort() until the first
        # token so we don't abort while a NIXL KV transfer is still in flight
        # on the decode worker (which can crash EngineCore). The guard's
        # cleanup runs after _abort_monitor tears down its monitor task, so
        # any deferred-abort waiter spawned by the monitor is in a stable
        # state when close() is awaited.
        async with _deferred_abort_guard(
            self.engine_client,
            request_id,
            is_decode_only,
            self._deferred_aborts,
            self._shutdown_on_engine_dead,
        ) as abort_guard:
            async with self._abort_monitor(
                context, request_id, abort_guard=abort_guard
            ):
                # nvext.engine_data opt-in: if the client requested
                # `nvext.extra_fields=["engine_data"]`, we accumulate
                # per-chunk token_ids and logprobs and attach them to the
                # FINAL chunk (the one carrying `finish_reason`). The Rust
                # frontend's response builder (delta.rs) gates emission via
                # `NvExtResponseFieldSelection.engine_data` so this payload
                # only reaches clients that asked for it.
                want_engine_data = _nvext_extra_field_requested(request, "engine_data")
                # Prompt token IDs the engine actually saw. Either the
                # pre-tokenized `nvext.token_data` (TITO) or whatever the
                # preprocessor produced from messages (MITO). We echo them
                # back in engine_data so the client doesn't have to re-derive
                # them from a request it might no longer hold.
                request_prompt_token_ids = (
                    _prompt_token_ids_for_engine_data(request, prompt)
                    if want_engine_data
                    else None
                )
                accumulated_token_ids: dict[int, list[int]] = {}
                accumulated_log_probs: dict[int, list[float]] = {}
                try:
                    async for tok in self.generate_tokens(
                        prompt,
                        sampling_params,
                        request_id,
                        data_parallel_rank=dp_rank,
                        lora_request=lora_request,
                        trace_headers=trace_headers,
                        priority=priority,
                        reasoning_ended=reasoning_ended,
                        reasoning_parser_kwargs=reasoning_parser_kwargs,
                    ):
                        if abort_guard is not None:
                            abort_guard.signal_first_token()
                        if prefill_result is not None and "completion_usage" in tok:
                            tok["completion_usage"][
                                "prompt_tokens_details"
                            ] = prefill_prompt_tokens_details

                        if want_engine_data:
                            _accumulate_engine_data(
                                tok,
                                request_prompt_token_ids,
                                accumulated_token_ids,
                                accumulated_log_probs,
                            )
                        yield tok
                except EngineDeadError as e:
                    logger.error(f"vLLM EngineDeadError: {e}")
                    logger.warning("Initiating Dynamo Runtime shutdown.")
                    self.runtime.shutdown()
                    os._exit(1)

    async def _generate_text_mode(self, request, context, request_id):
        """Generate text using OpenAI-compatible format (text-in-text-out)."""
        # Get text input using InputParamManager
        input_data = self.input_param_manager.get_input_param(
            request, use_tokenizer=True
        )

        # Build prompt for vLLM
        if isinstance(input_data, list):
            prompt = TokensPrompt(prompt_token_ids=input_data)
        else:
            prompt = TextPrompt(prompt=input_data)

        _apply_nvext_cache_salt(request, prompt)

        # Build sampling params from OpenAI-style request
        sampling_params = build_sampling_params_openai(
            request, self.default_sampling_params
        )

        routing = request.get("routing") or {}
        dp_rank = self._to_local_dp_rank(routing.get("dp_rank"))
        priority = -int(routing.get("priority", 0))
        openai_request_id = request.get("id") or request.get("request_id", request_id)
        previous_text_per_choice: dict[int, str] = {}

        trace_headers = context.trace_headers()

        # Mirror _generate_token_mode: in disagg decode mode route aborts through
        # the per-request deferred guard so engine_client.abort() never fires in
        # the unsafe pre-first-token window, and the admin abort_request route can
        # reach this request via self._deferred_aborts.
        is_decode_only = self.config.disaggregation_mode == DisaggregationMode.DECODE
        async with _deferred_abort_guard(
            self.engine_client,
            request_id,
            is_decode_only,
            self._deferred_aborts,
            self._shutdown_on_engine_dead,
        ) as abort_guard, self._abort_monitor(
            context, request_id, abort_guard=abort_guard
        ):
            try:
                gen = self.engine_client.generate(
                    prompt,
                    sampling_params,
                    request_id,
                    data_parallel_rank=dp_rank,
                    trace_headers=trace_headers,
                    priority=priority,
                )

                async for res in gen:
                    if not res.outputs:
                        yield {
                            "id": openai_request_id,
                            "created": int(time.time()),
                            "object": "chat.completion.chunk",
                            "model": "unknown",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": ""},
                                    "finish_reason": "error",
                                }
                            ],
                        }
                        break

                    for output in res.outputs:
                        if abort_guard is not None:
                            abort_guard.signal_first_token()
                        output_idx = getattr(output, "index", 0) or 0
                        previous_text = previous_text_per_choice.get(output_idx, "")
                        # Calculate the delta text (new text since last chunk)
                        delta_text = output.text[len(previous_text) :]

                        choice_data = {
                            "index": output_idx,
                            "delta": {
                                "role": "assistant",
                                "content": delta_text,
                            },
                            "finish_reason": normalize_finish_reason(
                                output.finish_reason
                            ),
                        }

                        chunk = {
                            "id": openai_request_id,
                            "created": int(time.time()),
                            "object": "chat.completion.chunk",
                            "model": "unknown",
                            "choices": [choice_data],
                        }

                        if output.finish_reason:
                            chunk["usage"] = BaseWorkerHandler._build_completion_usage(
                                request_output=res,
                            )

                        yield chunk
                        previous_text_per_choice[output_idx] = output.text

            except EngineDeadError as e:
                logger.error(f"vLLM EngineDeadError: {e}")
                logger.warning("Initiating Dynamo Runtime shutdown.")
                self.runtime.shutdown()
                os._exit(1)


class PrefillWorkerHandler(BaseWorkerHandler):
    def __init__(
        self,
        runtime,
        config: Config,
        engine,
        default_sampling_params,
        model_max_len: int | None = None,
        model_config: ModelConfig | None = None,
        enable_multimodal: bool = False,
        generate_endpoint=None,
        use_vllm_tokenizer: bool = False,
        shutdown_event: asyncio.Event | None = None,
        enable_frontend_decoding: bool = False,
        encode_worker_client: Client | None = None,
    ):
        super().__init__(
            runtime,
            config,
            engine,
            default_sampling_params,
            model_max_len=model_max_len,
            model_config=model_config,
            enable_multimodal=enable_multimodal,
            generate_endpoint=generate_endpoint,
            use_vllm_tokenizer=use_vllm_tokenizer,
            shutdown_event=shutdown_event,
            enable_frontend_decoding=enable_frontend_decoding,
            encode_worker_client=encode_worker_client,
        )

        self._multimodal_request_processor.initialize_prefill_handoff()

    async def generate(self, request, context):
        # Use context ID for request tracking and correlation with decode phase
        request_id = context.id()
        logger.debug("Prefill Request ID: %s", request_id)
        try:
            self._multimodal_request_processor.validate_multimodal_request(request)
        except ValueError as exc:
            logger.error("Request %s: %s", request_id, exc)
            yield {
                "status": "error",
                "message": str(exc),
                "disaggregated_params": None,
            }
            return

        # Token-in-token-out mode: internal protocol format
        with time_and_log_code_section(f"[PREFILL] request: {request_id} generate"):
            async for chunk in self._generate_token_mode(request, context, request_id):
                yield chunk

    async def _generate_token_mode(self, request, context, request_id):
        """Generate prefill using internal protocol format (token-in-token-out)."""
        prepared_input = await self._multimodal_request_processor.prepare_input(
            request,
            request_id,
            context,
            DisaggregationMode.PREFILL,
        )
        request = prepared_input.request
        multi_modal_data = prepared_input.multi_modal_data
        mm_processor_kwargs = prepared_input.mm_processor_kwargs

        # Build prompt from request (handles both prompt_embeds and token_ids)
        prompt, error = self._build_prompt_from_request(
            request,
            request_id,
            multi_modal_data,
            log_prefix="Prefill ",
            mm_processor_kwargs=mm_processor_kwargs,
        )
        if error is not None:
            # Prefill errors need disaggregated_params field
            error["disaggregated_params"] = None
            yield error
            return

        _apply_nvext_cache_salt(request, prompt)

        # Build sampling params from request using shared utility
        sampling_params = build_sampling_params(
            request,
            self.default_sampling_params,
            self.model_max_len,
            enable_rl=self.config.enable_rl,
        )

        # One protocol instance per request; carries per-request state
        # (e.g. Mooncake's transfer_id) into the response loop below.
        kv_protocol: KvConnectorProtocol = make_kv_connector_protocol(
            self.engine_client.vllm_config
        )
        if sampling_params.extra_args is None:
            sampling_params.extra_args = {}
        sampling_params.extra_args[
            "kv_transfer_params"
        ] = kv_protocol.prefill_request_kv_transfer_params()
        # Override for prefill: only generate 1 token
        sampling_params.max_tokens = 1
        sampling_params.min_tokens = 1

        # Extract LoRA request if present
        model_name = request.get("model")
        lora_request = self._resolve_lora_request(model_name)
        if lora_request:
            logger.info(
                f"Prefill request {request_id} will use LoRA adapter: {model_name} "
                f"(ID: {lora_request.lora_int_id}), path: {lora_request.lora_path}"
            )
        else:
            logger.debug(
                f"Prefill request {request_id} has no LoRA specified (model: {model_name})"
            )

        routing = request.get("routing") or {}
        dp_rank = self._to_local_dp_rank(routing.get("dp_rank"))
        priority = -int(routing.get("priority", 0))

        trace_headers = context.trace_headers()
        reasoning_ended, reasoning_parser_kwargs = _request_reasoning_metadata(request)

        async with self._abort_monitor(context, request_id, is_prefill=True):
            try:
                gen = self._generate_with_lora_admission_lock(
                    lora_request,
                    lambda admitted_lora_request: self.engine_client.generate(
                        prompt,
                        sampling_params,
                        request_id,
                        data_parallel_rank=dp_rank,
                        lora_request=admitted_lora_request,
                        trace_headers=trace_headers,
                        priority=priority,
                        **_engine_generate_reasoning_kwargs(
                            self.engine_client,
                            reasoning_ended,
                            reasoning_parser_kwargs,
                        ),
                    ),
                )
            except EngineDeadError as e:
                logger.error(f"vLLM EngineDeadError: {e}")
                logger.warning("Initiating Dynamo Runtime shutdown.")
                self.runtime.shutdown()
                os._exit(1)

            async for res in gen:
                logger.debug(f"kv transfer params: {res.kv_transfer_params}")

                token_ids = res.outputs[0].token_ids if res.outputs else []

                # For prefill worker, only one res will be generated,
                # so we can always build embedding params here without conditionals
                embedding_params = (
                    self._multimodal_request_processor.build_prefill_handoff(
                        multi_modal_data=multi_modal_data,
                        prompt_token_ids=list(res.prompt_token_ids or []),
                        mm_processor_kwargs=mm_processor_kwargs,
                    )
                )

                output: Dict[str, Any] = {
                    "token_ids": list(token_ids),
                    "disaggregated_params": self._build_disaggregated_params(
                        kv_protocol.decode_request_kv_transfer_params(res),
                        embedding_params,
                    ),
                    "completion_usage": BaseWorkerHandler._build_completion_usage(
                        request_output=res,
                    ),
                }

                # Log prefill completion with LoRA info
                self._log_with_lora_context(
                    "Prefill completed for request {request_id}{lora_info}: "
                    "generated {token_count} token(s), has_kv_params={has_kv_params}",
                    request_id,
                    lora_request,
                    level="info" if lora_request else "debug",
                    token_count=len(token_ids),
                    has_kv_params=res.kv_transfer_params is not None,
                )

                yield output

    def _build_disaggregated_params(
        self, kv_transfer_params, embedding_params=None, expanded_prompt_token_ids=None
    ):
        disaggregated_params = {}
        if kv_transfer_params is not None:
            disaggregated_params["kv_transfer_params"] = kv_transfer_params
        if embedding_params is not None:
            disaggregated_params["embedding_params"] = embedding_params
        if expanded_prompt_token_ids is not None:
            disaggregated_params[
                "expanded_prompt_token_ids"
            ] = expanded_prompt_token_ids

        return disaggregated_params if disaggregated_params else None


class EmbeddingWorkerHandler:
    """Standalone handler for OpenAI /v1/embeddings requests on vLLM.

    Does NOT inherit BaseWorkerHandler. The base class does generation-only
    init (media loaders, KV-block lookup via get_dp_range_for_worker, embedding
    cache manager) that would either fail or be meaningless on a pooling
    engine. Embedding inference is a single forward pass with no KV cache, no
    multimodal data, and no streamed decode.
    """

    def __init__(
        self,
        runtime,
        engine: Any,
        config: Config,
        shutdown_event: Optional[asyncio.Event] = None,
    ) -> None:
        self.runtime = runtime
        self.engine_client = engine
        self.config = config
        self.shutdown_event = shutdown_event
        # Dead-engine detection: VllmEngineMonitor polls AsyncLLM and triggers
        # shutdown_event + process exit on EngineDeadError. Without this, a
        # crashed pooling engine leaves the endpoint registered and serves
        # failures.
        self.engine_monitor = VllmEngineMonitor(runtime, engine, shutdown_event)
        logger.info("Embedding worker handler initialized")

    def cleanup(self) -> None:
        """Release resources owned by this handler.

        AsyncLLM lifecycle is owned by the worker factory / runtime; the
        engine monitor cancels its background tasks via ``__del__``.
        """
        return None

    async def _monitor_abort(self, context: Context, request_id: str) -> None:
        """Background task: abort the encode if context is cancelled or
        shutdown_event fires. Raises EngineShutdown on shutdown so the
        ``_abort_monitor`` context manager can propagate it.

        Mirrors ``BaseWorkerHandler._monitor_abort`` but trimmed for the
        embedding path (no ``is_prefill``, no ``abort_guard``).
        """
        shutdown_task: Optional[asyncio.Task] = None
        try:
            # `list[Any]` mirrors BaseWorkerHandler._monitor_abort: the
            # iterable mixes the Future from async_killed_or_stopped() with
            # the Task from shutdown_event.wait().
            wait_for: list[Any] = [context.async_killed_or_stopped()]
            if self.shutdown_event is not None:
                shutdown_task = asyncio.create_task(self.shutdown_event.wait())
                wait_for.append(shutdown_task)

            done, pending = await asyncio.wait(
                wait_for, return_when=asyncio.FIRST_COMPLETED
            )

            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            logger.debug(f"Aborting embedding request ID: {request_id}")
            try:
                await asyncio.shield(self.engine_client.abort(request_id))
            except asyncio.CancelledError:
                logger.debug(
                    f"Abort shielded from cancellation for embedding request "
                    f"{request_id}, continuing in background"
                )

            if shutdown_task is not None and shutdown_task in done:
                raise EngineShutdown("Engine was shut down during embedding.")
        except asyncio.CancelledError:
            pass
        except EngineShutdown:
            raise
        except Exception as e:
            # Unexpected failure in the monitor task — log and propagate so
            # `_abort_monitor.__aexit__` surfaces it via ``task.result()``
            # rather than silently leaving the encode unmanaged.
            logger.error(
                f"Error in embedding abort monitor for request {request_id}: {e}"
            )
            raise
        finally:
            # On the success path the wrapping ``_abort_monitor`` cancels
            # this coroutine while it's blocked in ``asyncio.wait``, which
            # short-circuits past the pending-task cleanup loop above and
            # leaves ``shutdown_task`` (the ``shutdown_event.wait()`` task)
            # pending forever — one leaked task per embedding request.
            # Cancel it here on every exit path.
            if shutdown_task is not None and not shutdown_task.done():
                shutdown_task.cancel()
                try:
                    await shutdown_task
                except asyncio.CancelledError:
                    pass

    @asynccontextmanager
    async def _abort_monitor(self, context: Context, request_id: str):
        """Create + tear down an abort monitor task around one encode call.

        On exit, re-raises EngineShutdown if the monitor caught a shutdown.
        """
        task = asyncio.create_task(self._monitor_abort(context, request_id))
        try:
            yield task
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            else:
                # Re-raise EngineShutdown if the monitor task raised it.
                task.result()

    async def generate(
        self, request: dict, context: Context
    ) -> AsyncIterator[Dict[str, Any]]:
        """Handle one OpenAI /v1/embeddings request.

        The Rust frontend forwards the request dict directly. Expected keys:
        ``model: str``, ``input: str | list[str] | list[int] | list[list[int]]``.
        Optional ``dimensions`` (Matryoshka dimensionality reduction):
        forwarded to vLLM's pooler, which truncates to N dims and
        re-normalizes; vLLM requires the model to declare Matryoshka support.
        Optional ``encoding_format`` (``"float"`` -- default -- or
        ``"base64"``); when ``"base64"`` is requested, each per-input vector is
        serialized as a base64-encoded string of little-endian ``f32`` bytes
        per the OpenAI spec, so the byte count matches the (possibly reduced)
        dimensionality. Optional ``truncate_prompt_tokens`` is forwarded to
        vLLM's tokenizer path for raw-text inputs.
        """
        model_name = request.get("model") or self.config.served_model_name or ""
        input_field = request.get("input")
        if input_field is None:
            raise ValueError("Embedding request missing required 'input' field")

        # Per OpenAI spec, `input` can be:
        #   - str           : single text prompt
        #   - list[str]     : batch of text prompts
        #   - list[int]     : single pre-tokenized prompt (token IDs)
        #   - list[list[int]]: batch of pre-tokenized prompts
        # Token-id forms must be passed to vLLM as TokensPrompt so the engine
        # skips its own tokenizer; the previous str()-coercion path turned
        # `[1, 2, 3]` into three text prompts ("1", "2", "3") instead of one.
        prompts: list[Any] = _classify_embedding_input(input_field)

        dimensions = request.get("dimensions")
        if dimensions is not None and (
            not isinstance(dimensions, int) or isinstance(dimensions, bool)
        ):
            raise TypeError(
                f"Invalid 'dimensions' type {type(dimensions).__name__}; expected int"
            )
        if dimensions is not None and dimensions < 1:
            raise ValueError(f"dimensions must be >= 1, got {dimensions}")

        encoding_format = request.get("encoding_format", "float")
        if encoding_format not in ("float", "base64"):
            raise ValueError(
                f"Invalid 'encoding_format' value {encoding_format!r}; "
                "expected 'float' or 'base64'"
            )

        truncate_prompt_tokens = request.get("truncate_prompt_tokens")
        tokenization_kwargs: dict[str, Any] | None = None
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
                    "truncate_prompt_tokens must be >= -1, "
                    f"got {truncate_prompt_tokens}"
                )
            tokenization_kwargs = {
                "truncate_prompt_tokens": truncate_prompt_tokens,
            }

        # Request the pooled sentence embedding. With no task, vLLM's
        # encode() resolves to per-token output (the full ``n_tokens x
        # hidden`` hidden-state matrix), so the OpenAI ``/v1/embeddings``
        # response ends up with the wrong shape (dim scales with input
        # length) instead of one vector per input. ``task="embed"`` selects
        # the pooled embedding and runs the model's configured pooler
        # (normalization included for models like Qwen3-Embedding), matching
        # vLLM's own embedding server. ``use_activation`` is intentionally
        # left at the pooler default so per-model behaviour isn't overridden.
        #
        # ``dimensions`` (OpenAI Matryoshka truncation) is forwarded to vLLM
        # rather than applied here: vLLM's pooler truncates to ``dimensions``
        # and then re-normalizes (the correct MRL behaviour) and validates
        # that the model actually supports Matryoshka -- raising rather than
        # silently returning a degraded, un-normalized vector for models that
        # don't. This matches bare ``vllm serve``. Models whose HF config
        # doesn't declare Matryoshka support (e.g. Qwen3-Embedding) must be
        # launched with ``--hf-overrides '{"is_matryoshka": true}'`` for
        # ``dimensions`` requests to be accepted.
        pooling_kwargs: dict[str, Any] = {"task": "embed"}
        if dimensions is not None:
            pooling_kwargs["dimensions"] = dimensions
        pooling_params = PoolingParams(**pooling_kwargs)
        # Use the per-request context id (same as the chat/completion paths
        # in this file) so concurrent embeddings never collide inside
        # ``AsyncLLM``. ``context.trace_id`` is a distributed-trace id
        # shared by every request in a trace and ``id(context)`` can be
        # reused across short-lived ``Context`` objects, so neither is
        # unique enough to scope a vLLM ``request_id``.
        base_request_id = context.id()

        async def _encode_one(idx: int, prompt: Any):
            request_id = f"{base_request_id}-{idx}"
            encode_arg: Any = (
                prompt
                if isinstance(prompt, str)
                else TokensPrompt(prompt_token_ids=prompt)
            )
            final_output = None
            async with self._abort_monitor(context, request_id):
                encode_kwargs: dict[str, Any] = {
                    "prompt": encode_arg,
                    "pooling_params": pooling_params,
                    "request_id": request_id,
                }
                if tokenization_kwargs is not None and isinstance(encode_arg, str):
                    encode_kwargs["tokenization_kwargs"] = tokenization_kwargs

                async for out in self.engine_client.encode(**encode_kwargs):
                    final_output = out
            if final_output is None:
                raise RuntimeError(
                    f"vLLM engine.encode produced no output for input index {idx}"
                )
            return final_output

        # Submit every prompt to the engine in the same event-loop tick so
        # vLLM's continuous-batching scheduler can coalesce them into a
        # single forward pass instead of N sequential ones. ``asyncio.gather``
        # returns results in input order, so ``outputs[k]`` matches ``prompts[k]``
        # regardless of engine completion order.
        #
        # Use explicit tasks + a ``finally`` cancellation pass so that if one
        # ``_encode_one`` raises, we cancel siblings still in flight instead
        # of leaving them running -- otherwise vLLM keeps consuming engine
        # capacity for output that this handler will discard.
        tasks = [asyncio.create_task(_encode_one(i, p)) for i, p in enumerate(prompts)]
        try:
            outputs = await asyncio.gather(*tasks)
        finally:
            pending = [t for t in tasks if not t.done()]
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        embedding_objects: list[Dict[str, Any]] = []
        prompt_tokens = 0
        for idx, final_output in enumerate(outputs):
            # vLLM has already applied any ``dimensions`` Matryoshka reduction
            # (truncate + re-normalize) inside the pooler, so this is the
            # final per-input vector -- no post-hoc truncation here.
            embedding = _pooling_output_to_list(final_output.outputs.data)

            # vLLM rejects an unsupported ``dimensions`` for models that
            # declare a ``matryoshka_dimensions`` list, but a model enabled
            # via ``--hf-overrides '{"is_matryoshka": true}'`` (no explicit
            # list) is only validated for ``dimensions >= 1`` -- the pooler
            # then silently clamps an oversized request to the model's native
            # size (``embeddings[..., :dimensions]``). Surface the same clear
            # error the old post-hoc path raised instead of returning a
            # shorter-than-requested vector.
            if dimensions is not None and len(embedding) < dimensions:
                raise ValueError(
                    f"dimensions={dimensions} exceeds model embedding "
                    f"dimension {len(embedding)}"
                )

            # Always emit base64 over the worker->frontend wire format. The
            # Rust frontend decodes back to float when the client's
            # ``encoding_format`` is float (or unset). 15x1024-float responses
            # serialized as JSON arrays cost ~110 ms in Python json.dumps +
            # Rust serde parse; base64 bytes are ~3x smaller and ~10x faster
            # to (de)serialize. Client-visible wire format is preserved
            # because Rust converts at the HTTP boundary.
            embedding_objects.append(
                {
                    "object": "embedding",
                    "embedding": _encode_floats_to_base64(embedding),
                    "index": idx,
                }
            )
            token_ids = getattr(final_output, "prompt_token_ids", None) or []
            prompt_tokens += len(token_ids)

        yield {
            "object": "list",
            "data": embedding_objects,
            "model": model_name,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens,
            },
        }


def _is_token_id(x: Any) -> bool:
    """True iff ``x`` is an int that could be a vLLM token id.

    Filters out ``bool`` (subclass of int) so ``[True, False]`` is not
    accepted as a tokenized prompt.
    """
    return isinstance(x, int) and not isinstance(x, bool)


def _classify_embedding_input(input_field: Any) -> list[Any]:
    """Map an OpenAI ``input`` payload to a list of vLLM-ready prompts.

    Returns a list whose elements are either:
      - ``str``        — passed straight to ``engine.encode`` as text, or
      - ``list[int]``  — wrapped in ``TokensPrompt`` by the caller.

    Rejects mixed lists (e.g. ``["foo", 42]`` or ``[[1, 2], "bar"]``) with
    a clear ``TypeError`` rather than silently coercing.
    """
    if isinstance(input_field, str):
        return [input_field]
    if not isinstance(input_field, list):
        raise TypeError(
            f"Invalid 'input' type {type(input_field).__name__}; "
            "expected str, list[str], list[int], or list[list[int]]"
        )
    if not input_field:
        raise ValueError("Embedding request 'input' must be non-empty")

    first = input_field[0]
    if isinstance(first, str):
        texts: list[str] = []
        for item in input_field:
            if not isinstance(item, str):
                raise TypeError(
                    "'input' list mixes str and non-str entries; pass either "
                    "all strings or all token-id arrays"
                )
            texts.append(item)
        return texts
    if _is_token_id(first):
        token_ids: list[int] = []
        for item in input_field:
            if not _is_token_id(item):
                raise TypeError(
                    "'input' list mixes int and non-int entries; for tokenized "
                    "input pass all integers (single prompt) or list[list[int]]"
                )
            token_ids.append(item)
        # Single tokenized prompt.
        return [token_ids]
    if isinstance(first, list):
        prompts: list[list[int]] = []
        for i, item in enumerate(input_field):
            if not isinstance(item, list):
                raise TypeError(
                    f"'input' list element at index {i} must be a list of "
                    "ints (token IDs); mixed batches are not supported"
                )
            inner: list[int] = []
            for x in item:
                if not _is_token_id(x):
                    raise TypeError(
                        f"'input' list element at index {i} must be a list of "
                        "ints (token IDs); mixed batches are not supported"
                    )
                inner.append(x)
            prompts.append(inner)
        return prompts
    raise TypeError(
        f"Unsupported 'input' element type {type(first).__name__}; "
        "expected str, int, or list[int]"
    )


def _pooling_output_to_list(data: Any) -> list[float]:
    """Convert a vLLM PoolingOutput.data tensor (or list) to a flat list[float].

    vLLM's pooling pipeline can return a tensor with a singleton batch dim
    (shape ``(1, hidden_dim)``) instead of a 1D vector (shape ``(hidden_dim,)``).
    The OpenAI ``/v1/embeddings`` response expects ``data[].embedding`` to be a
    flat array of floats, so we flatten unconditionally.
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().flatten().tolist()
    if isinstance(data, (list, tuple)):
        # Already a list — flatten one level if it's a list-of-lists.
        if data and isinstance(data[0], (list, tuple)):
            return [float(x) for row in data for x in row]
        return [float(x) for x in data]
    raise TypeError(
        f"Unsupported PoolingOutput.data type {type(data).__name__}; "
        "expected torch.Tensor or list"
    )


def _encode_floats_to_base64(floats: list[float]) -> str:
    """Encode an embedding vector as a base64 string per the OpenAI
    ``encoding_format=base64`` spec: raw little-endian ``float32`` bytes
    are concatenated and base64-encoded with the standard alphabet.

    Mirrors the Rust ``encode_floats_to_base64`` helper in
    ``lib/llm/src/preprocessor.rs`` so the two backend code paths
    produce identical bytes for the same input.
    """
    packed = struct.pack(f"<{len(floats)}f", *floats)
    return base64.b64encode(packed).decode("ascii")
