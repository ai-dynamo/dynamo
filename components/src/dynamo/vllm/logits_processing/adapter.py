# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM realizer for the shared logits-processor spec layer.

Unlike TRT-LLM, vLLM cannot accept a live Python callable per request. Its
custom logits processors are *engine-loaded*: a class is registered at engine
init (`AsyncEngineArgs.logits_processors`) and vLLM instantiates it once, then
calls it for every batch. Per-request activation rides on
`SamplingParams.extra_args` (vLLM's `vllm_xargs`).

So this module is two halves:

* :class:`DynamoVllmLogitsProcessor` — the engine-loaded adapter. vLLM hands
  it each request's `SamplingParams`; it reads the serialized spec entries we
  stashed under ``extra_args["dynamo_logits"]``, realizes a *fresh* per-request
  processor, and returns it (or ``None`` for requests with no activation).
* :func:`activate_logits_processors` — called from ``generate()`` to stash the
  serialized entries onto a request's `SamplingParams`.

The shared generation-stage gating and per-request freshness policy live in
``dynamo.common.backend.engine``; this module only translates the
backend-neutral entries into vLLM's mechanism.

API assumptions (vLLM v1, ~0.11+; not introspectable in CI without a GPU):

* ``vllm.v1.sample.logits_processor`` exposes ``AdapterLogitsProcessor`` whose
  subclasses implement ``is_argmax_invariant()`` and
  ``new_req_logits_processor(params) -> RequestLogitsProcessor | None``. The
  base class handles batching, persistence, and the request callable's arity.
* The returned ``RequestLogitsProcessor`` is called as
  ``(past_output_token_ids, logits) -> logits`` with a 1-D ``logits`` vector
  for one request — matching Dynamo's ``BaseLogitsProcessor(input_ids,
  logits)`` in-place contract.
* ``engine_args.logits_processors`` accepts ``"module.path:ClassName"`` import
  strings.

If a vLLM upgrade moves these, route the import through a compat shim rather
than scattering try/except across call sites.
"""

import logging
from typing import Any, Optional, Sequence

from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)

from dynamo.common.backend.engine import (
    ForcedTokenSequenceSpec,
    LogitsProcessorEntry,
    deserialize_logits_processor_entries,
    serialize_logits_processor_entries,
)
from dynamo.logits_processing import BaseLogitsProcessor
from dynamo.logits_processing.examples import ForcedSequenceLogitsProcessor

logger = logging.getLogger(__name__)

#: Import path vLLM uses to load the engine-level adapter. Appended to
#: ``AsyncEngineArgs.logits_processors`` at startup by
#: :func:`register_dynamo_logits_processor`.
DYNAMO_VLLM_LOGITS_PROCESSOR_PATH = (
    "dynamo.vllm.logits_processing.adapter:DynamoVllmLogitsProcessor"
)

#: Key under ``SamplingParams.extra_args`` carrying the serialized entries.
_DYNAMO_LOGITS_KEY = "dynamo_logits"


def _realize_entry(entry: LogitsProcessorEntry) -> BaseLogitsProcessor:
    """Build one fresh, stateful processor from a serializable spec entry.

    vLLM only ever sees serializable entries (``PythonProcessorSpec`` is
    rejected at serialize time), so ``ForcedTokenSequenceSpec`` is the only
    case here. Anything else is a forward-incompatibility and raises.
    """
    if isinstance(entry, ForcedTokenSequenceSpec):
        return ForcedSequenceLogitsProcessor(entry.token_ids, entry.eos_token_id)
    raise TypeError(
        f"vLLM logits-processor adapter cannot realize entry of type "
        f"{type(entry).__name__}"
    )


class DynamoVllmLogitsProcessor(AdapterLogitsProcessor):
    """Engine-loaded vLLM adapter for Dynamo logits processors.

    Registered once via ``engine_args.logits_processors`` (only when the smoke
    hook is enabled on a generation worker). vLLM calls
    :meth:`new_req_logits_processor` once per request; we realize fresh
    per-request processor state there so concurrent requests never share a
    counter.
    """

    def is_argmax_invariant(self) -> bool:
        # Forced-sequence masking deliberately changes which token is the
        # argmax, so the processor is NOT argmax-invariant. Returning False
        # makes vLLM apply it even on greedy (temperature=0) requests.
        return False

    def new_req_logits_processor(self, params: Any) -> Optional[RequestLogitsProcessor]:
        extra_args = getattr(params, "extra_args", None) or {}
        payload = extra_args.get(_DYNAMO_LOGITS_KEY)
        if not payload:
            return None
        entries = deserialize_logits_processor_entries(payload)
        processors = [_realize_entry(entry) for entry in entries]
        if not processors:
            return None

        def _request_logits_processor(
            output_token_ids: Sequence[int], logits: Any
        ) -> Any:
            # Dynamo processors mutate the 1-D logits vector in place; vLLM's
            # request-callable contract wants the (mutated) tensor returned.
            for processor in processors:
                processor(output_token_ids, logits)
            return logits

        return _request_logits_processor


def activate_logits_processors(
    sampling_params: Any,
    entries: Sequence[LogitsProcessorEntry],
) -> None:
    """Stash serialized spec entries onto a request's ``SamplingParams``.

    On empty entries, clear any pre-existing payload rather than no-op: vLLM
    exposes ``extra_args`` to clients via ``vllm_xargs``, so a caller could
    otherwise leave a stale ``dynamo_logits`` that the engine-loaded
    :class:`DynamoVllmLogitsProcessor` would activate even when the shared
    gating produced no entries for this request.
    """
    extra_args = getattr(sampling_params, "extra_args", None)
    if not entries:
        if extra_args is not None:
            extra_args.pop(_DYNAMO_LOGITS_KEY, None)
        return
    if extra_args is None:
        sampling_params.extra_args = extra_args = {}
    extra_args[_DYNAMO_LOGITS_KEY] = serialize_logits_processor_entries(entries)


def register_dynamo_logits_processor(engine_args: Any) -> None:
    """Append the engine-loaded adapter to ``engine_args.logits_processors``.

    Idempotent. Call at startup only when the smoke hook is enabled on a
    generation worker; production paths leave ``logits_processors`` untouched.
    """
    existing = list(getattr(engine_args, "logits_processors", None) or [])
    if DYNAMO_VLLM_LOGITS_PROCESSOR_PATH not in existing:
        existing.append(DYNAMO_VLLM_LOGITS_PROCESSOR_PATH)
    engine_args.logits_processors = existing
