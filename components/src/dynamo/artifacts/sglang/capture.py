# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hidden-state artifacts from SGLang's capture-only aux hidden states.

A worker launched with ``--enable-return-aux-hidden-states``,
``--speculative-algorithm EAGLE3`` (which only names the capture convention; no
draft loads) and ``--return-aux-hidden-state-layer-ids`` returns the
concatenated aux layers in ``meta_info["hidden_states"]`` for requests that set
``return_hidden_states``. The session here reads those rows; admission, the
format, chunking and delivery are the vLLM session's.

SGLang specifics the handler and this session work around:

- The rows arrive as Python float lists, and every streamed output re-sends the
  whole list. The handler asks SGLang to stream only the first token and the
  finish (``stream_interval`` above ``max_new_tokens``).
- Prefill skips prefix-cached positions and keeps only the last chunk of a
  chunked prefill, so the rows are a suffix of the sequence. The handler caps
  the prefix match at ``token_start`` (``logprob_start_len``); a suffix that
  still starts after ``token_start`` fails the capture.
- A retracted request re-prefills and appends a second block of rows, so a
  retracted request fails the capture.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from dynamo.artifacts.vllm.capture import VllmArtifactSession
from dynamo.artifacts.vllm.hidden_states import HiddenStateProfile
from dynamo.common.metadata_upload import _backend_metadata_upload_settings

# Python list slot plus float object, plus the float32 conversion copy.
LIST_BYTES_PER_VALUE = 36
_DTYPE_NAMES = {
    "torch.bfloat16": "bf16",
    "torch.float16": "fp16",
    "torch.float32": "fp32",
}
_REQUIRED_FLAGS = (
    "hidden-state capture requires an SGLang worker launched with "
    "--enable-return-aux-hidden-states, --speculative-algorithm EAGLE3 and "
    "--return-aux-hidden-state-layer-ids"
)


def hidden_state_profile_from_sglang(
    server_args: Any, model_config: Any
) -> HiddenStateProfile | None:
    """What an SGLang worker captures, or None when capture is off.

    ``layer_ids`` follow the artifact convention, shared with vLLM: id ``i`` is
    the input of decoder layer ``i``. SGLang's explicit layer ids name layer
    outputs (it captures ``id + 1``), so each is reported plus one.
    """
    if not getattr(server_args, "enable_return_aux_hidden_states", False):
        return None
    method = (
        getattr(server_args, "aux_hidden_state_capture", None)
        or getattr(server_args, "speculative_algorithm", None)
        or ""
    )
    if str(method).lower() != "eagle3":
        raise ValueError(
            "Dynamo hidden-state artifacts support SGLang's EAGLE3 capture "
            f"convention, got {method!r}"
        )
    layer_ids = getattr(server_args, "return_aux_hidden_state_layer_ids", None)
    if not layer_ids:
        # The EAGLE3 default list is defined per model file, so Dynamo can't
        # publish it without loading the model.
        raise ValueError(
            "hidden-state artifacts need --return-aux-hidden-state-layer-ids so "
            "Dynamo knows which layers SGLang captures"
        )
    dtype = _DTYPE_NAMES.get(str(getattr(model_config, "dtype", None)))
    hidden_size = getattr(model_config, "hidden_size", None)
    if dtype is None or type(hidden_size) is not int or hidden_size <= 0:
        raise ValueError("unsupported SGLang model dtype or hidden size for capture")
    return HiddenStateProfile(
        layer_ids=tuple(sorted(int(layer) + 1 for layer in layer_ids)),
        hidden_size=hidden_size,
        dtype=dtype,
        storage_root="",
    )


def sglang_hidden_rows(raw: Any, *, width: int, dtype: str) -> tuple[int, bytes]:
    """Flatten SGLang's hidden-state list into packed rows of ``dtype``.

    Items are 2-D blocks (a prefill) or single rows (a decode step). The values
    must be exactly representable in ``dtype``, so a dtype mismatch fails
    instead of rounding.
    """
    if not isinstance(raw, list):
        raise ValueError("SGLang returned no hidden states")
    rows: list[Any] = []
    for item in raw:
        if not isinstance(item, list) or not item:
            raise ValueError("unexpected SGLang hidden-state layout")
        if isinstance(item[0], list):
            rows.extend(item)
        else:
            rows.append(item)
    if not rows:
        return 0, b""
    try:
        values = np.asarray(rows, dtype=np.float32)
    except (ValueError, TypeError) as exc:
        raise ValueError("unexpected SGLang hidden-state layout") from exc
    if values.shape != (len(rows), width):
        raise ValueError("hidden-state geometry differs from profile")
    if dtype == "bf16":
        bits = values.view(np.uint32)
        if np.any(bits & 0xFFFF):
            raise ValueError("hidden states are not bf16 values")
        return len(rows), (bits >> 16).astype(np.uint16).tobytes()
    if dtype == "fp16":
        half = values.astype(np.float16)
        if not np.array_equal(half.astype(np.float32), values, equal_nan=True):
            raise ValueError("hidden states are not fp16 values")
        return len(rows), half.tobytes()
    if dtype == "fp32":
        return len(rows), values.tobytes()
    raise ValueError("unsupported hidden-state dtype")


class SglangArtifactSession(VllmArtifactSession):
    """Generation-artifact capture for SGLang; ``hidden_states`` only for now."""

    hidden_rows: Any = None
    retractions: Any = None

    @classmethod
    def from_request(  # type: ignore[override]
        cls,
        request,
        *,
        prompt_token_count=0,
        max_tokens=0,
        choice_count=1,
        budget=None,
        runtime=None,
        hidden_state_profile=None,
    ):
        settings = _backend_metadata_upload_settings(request)
        if settings is None or "kind" not in settings:
            return None
        contents = settings.get("contents")
        if isinstance(contents, list):
            if any(name != "hidden_states" for name in contents):
                raise ValueError(
                    "SGLang artifacts currently support hidden_states only"
                )
            if "hidden_states" in contents and hidden_state_profile is None:
                raise ValueError(_REQUIRED_FLAGS)
        session = super().from_request(
            request,
            prompt_token_count=prompt_token_count,
            max_tokens=max_tokens,
            choice_count=choice_count,
            budget=budget,
            runtime=runtime,
            hidden_state_profile=hidden_state_profile,
        )
        if session.hidden_profile is not None:
            # SGLang may return rows from position 0, as Python floats.
            profile = cast(HiddenStateProfile, session.hidden_profile)
            width = len(profile.layer_ids) * profile.hidden_size
            extra = (prompt_token_count + max_tokens) * width * LIST_BYTES_PER_VALUE
            try:
                session.budget.reserve(extra)
            except BaseException:
                session.close()
                raise
            session.reserved_bytes += extra
        return session

    def observe_tokens(self, token_ids, prompt_token_ids):
        """Record a streamed output's new tokens."""
        self.observe(SimpleNamespace(index=0, token_ids=token_ids), prompt_token_ids)

    def observe_finish(self, meta_info):
        """Take the rows and retraction count from the final output."""
        if self._closed or self.hidden_profile is None:
            return
        self.hidden_rows = meta_info.get("hidden_states")
        self.retractions = meta_info.get("num_retractions")

    def _load_hidden_states(self):
        raw, self.hidden_rows = self.hidden_rows, None
        if raw is None:
            raise ValueError("requested hidden states unavailable")
        if self.retractions != 0:
            # A re-prefill appends a second block of rows to the same list.
            raise ValueError("SGLang retracted the request; its rows may repeat")
        if self.error or self.prompt is None:
            raise ValueError("invalid artifact capture")
        profile = cast(HiddenStateProfile, self.hidden_profile)
        count, data = sglang_hidden_rows(
            raw,
            width=len(profile.layer_ids) * profile.hidden_size,
            dtype=profile.dtype,
        )
        del raw
        extent = len(self.prompt) + len(self.tokens) - bool(self.tokens)
        start = self.hidden_token_start
        if start > len(self.prompt):
            raise ValueError("hidden-state token_start exceeds observed prompt")
        first = extent - count
        if first < 0:
            raise ValueError("SGLang returned more hidden-state rows than positions")
        if first > start:
            raise ValueError(
                "SGLang returned no hidden states before position "
                f"{first}: served from the prefix cache or an earlier prefill chunk"
            )
        skip = (start - first) * profile.row_bytes
        self._bind_hidden_rows(start=start, rows=extent - start, data=data[skip:])

    def _discard(self):
        self.hidden_rows = None
        super()._discard()
