# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang realizer for the shared logits-processor spec layer.

SGLang's custom-processor surface is different from both TRT-LLM (live
per-request callables) and vLLM (engine-loaded class + per-request
``extra_args``). SGLang:

* gates the whole feature behind the server flag
  ``--enable-custom-logit-processor``;
* takes a *serialized* processor class as a top-level ``async_generate``
  argument (``custom_logit_processor``), reconstructed and cached on the
  scheduler;
* hands that processor a *batch* logits tensor plus a ``custom_param_list``
  (one ``sampling_params["custom_params"]`` dict per request in the batch).

Two halves, mirroring the other backends:

* :class:`DynamoSglangLogitProcessor` — the serialized, scheduler-side
  processor. It maps each batch row to its request's serialized entries and
  applies a per-request Dynamo processor to that row.
* :func:`activate_logits_processors` — called from ``generate()`` to stash
  the serialized entries into ``sampling_params["custom_params"]`` and return
  the ``custom_logit_processor`` kwarg for ``async_generate``.

Per-request state — the central assumption
-------------------------------------------
SGLang's callback is ``__call__(logits, custom_param_list)``. It does NOT
pass the per-request generated-token position, and ``custom_param_list`` is
static across decode steps. So this adapter only supports
``ForcedTokenSequenceSpec``, whose realized
``ForcedSequenceLogitsProcessor`` advances a purely internal counter and
ignores ``input_ids`` — the narrow, spec-backed case the design scoped for
SGLang. Arbitrary ``BaseLogitsProcessor`` instances that need the real token
history are deliberately out of scope until SGLang exposes that state at this
callback.

To make the internal counter advance correctly, per-request processor state
is kept in a dict keyed by a request UID we inject into ``custom_params``.
This relies on SGLang caching ONE :class:`DynamoSglangLogitProcessor`
instance (keyed by its serialized string) and reusing it across decode steps
and requests — true in the SGLang versions we target. The per-UID map is not
pruned (no completion signal reaches this callback); acceptable for the
env-gated smoke hook, not a production processor surface.

API assumptions (SGLang; pre-1.0, moves between releases — see the
component CLAUDE.md ``_compat.py`` policy if an upgrade breaks these):

* ``sglang.srt.sampling.custom_logit_processor.CustomLogitProcessor`` is an
  ABC with ``__call__(self, logits, custom_param_list)`` and a ``to_str()``
  serializer.
* ``logits`` is a 2-D ``[batch, vocab]`` tensor; row ``i`` corresponds to
  ``custom_param_list[i]``.
"""

import logging
from typing import Any, Optional, Sequence

from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

from dynamo.common.backend.engine import (
    ForcedTokenSequenceSpec,
    LogitsProcessorEntry,
    deserialize_logits_processor_entries,
    serialize_logits_processor_entries,
)
from dynamo.logits_processing import BaseLogitsProcessor
from dynamo.logits_processing.examples import ForcedSequenceLogitsProcessor

logger = logging.getLogger(__name__)

#: Keys under ``sampling_params["custom_params"]``.
_ENTRIES_KEY = "dynamo_logits"
_UID_KEY = "dynamo_uid"


def _realize_entry(entry: LogitsProcessorEntry) -> BaseLogitsProcessor:
    """Build one fresh, stateful processor from a serializable spec entry.

    Only ``ForcedTokenSequenceSpec`` is supported here (see the module
    docstring on SGLang's per-request-state constraint).
    """
    if isinstance(entry, ForcedTokenSequenceSpec):
        return ForcedSequenceLogitsProcessor(entry.token_ids, entry.eos_token_id)
    raise TypeError(
        f"SGLang logits-processor adapter cannot realize entry of type "
        f"{type(entry).__name__}"
    )


class DynamoSglangLogitProcessor(CustomLogitProcessor):
    """Scheduler-side SGLang adapter for Dynamo logits processors.

    A single instance is reconstructed from :meth:`to_str` and cached by
    SGLang, so :attr:`_per_request` persists across decode steps — that is
    what lets the forced-sequence counter advance. See the module docstring.
    """

    def __init__(self) -> None:
        super().__init__()
        # request UID -> realized per-request processors (fresh per request).
        self._per_request: dict[Any, list[BaseLogitsProcessor]] = {}

    def __call__(
        self, logits: Any, custom_param_list: Optional[Sequence[dict]] = None
    ) -> Any:
        if not custom_param_list:
            return logits
        for row_idx, params in enumerate(custom_param_list):
            params = params or {}
            payload = params.get(_ENTRIES_KEY)
            if not payload:
                continue
            uid = params.get(_UID_KEY)
            processors = self._per_request.get(uid)
            if processors is None:
                processors = [
                    _realize_entry(entry)
                    for entry in deserialize_logits_processor_entries(payload)
                ]
                self._per_request[uid] = processors
            # Row view: in-place mutation by the Dynamo processor writes back
            # into the batch tensor. input_ids is unused by forced-sequence.
            row = logits[row_idx]
            for processor in processors:
                processor((), row)
        return logits


# `to_str()` only depends on the class, not instance state, so every request
# shares one serialized string -> SGLang caches a single instance. Compute it
# once and reuse.
_SERIALIZED_PROCESSOR: Optional[str] = None


def _serialized_processor() -> str:
    global _SERIALIZED_PROCESSOR
    if _SERIALIZED_PROCESSOR is None:
        _SERIALIZED_PROCESSOR = DynamoSglangLogitProcessor().to_str()
    return _SERIALIZED_PROCESSOR


def activate_logits_processors(
    sampling_params: dict,
    entries: Sequence[LogitsProcessorEntry],
    *,
    request_uid: Any,
) -> dict[str, Any]:
    """Wire entries into SGLang's per-request custom-processor mechanism.

    Mutates ``sampling_params`` to carry the serialized entries under
    ``custom_params`` and returns the ``async_generate`` kwargs (the
    serialized ``custom_logit_processor``). Returns ``{}`` on empty entries
    (hook-off / non-generation worker), so the caller adds nothing.

    ``request_uid`` MUST be a non-empty, request-unique id (pass
    ``context.id()``, not ``context.trace_id``). The processor keys its
    per-request state on it; a ``None``/empty or reused uid would collide
    that state across requests and corrupt the forced sequence, so a falsy
    uid is rejected loudly here rather than silently mis-tracked.

    ``n`` is forced to 1: SGLang expands ``n > 1`` into multiple batch rows
    sharing this request's ``custom_params`` (and uid), which would all hit
    the same stateful processor and corrupt each other. A forced sequence has
    no meaningful ``n > 1`` anyway, so the env-gated hook pins it to 1.
    """
    if not entries:
        return {}
    if not request_uid:
        raise ValueError(
            "request_uid must be a non-empty, request-unique id "
            "(e.g. context.id()); got "
            f"{request_uid!r}"
        )
    requested_n = sampling_params.get("n")
    if requested_n not in (None, 1):
        logger.warning(
            "forcing n=1 for the logits-processor hook (requested n=%s); "
            "parallel completions share per-request processor state",
            requested_n,
        )
    sampling_params["n"] = 1
    # Merge into any existing custom_params rather than replacing, so a
    # caller-supplied entry isn't silently dropped (mirrors the vLLM adapter's
    # extra_args handling).
    custom_params = sampling_params.setdefault("custom_params", {})
    custom_params[_ENTRIES_KEY] = serialize_logits_processor_entries(entries)
    custom_params[_UID_KEY] = request_uid
    return {"custom_logit_processor": _serialized_processor()}
