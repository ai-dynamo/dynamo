# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Admission and owned capture for the vLLM generation-artifact producer."""

from __future__ import annotations

import asyncio
import base64
import math
import os
import sqlite3
import threading
from copy import deepcopy
from functools import cache
from typing import Any, Protocol, cast

import numpy as np

from dynamo.artifacts.vllm.format import (
    GenerationArtifactChoice,
    GenerationArtifactView,
    MoeRoutes,
    SelectedLogprobs,
    Tensor,
    TopKLogprobs,
    compact_routes,
    encode_generation_artifact,
)
from dynamo.common.metadata_upload import _backend_metadata_upload_settings

_CONTENTS = frozenset({"moe_routes", "selected_logprobs", "topk_logprobs"})


class CaptureCapacityError(ValueError):
    """Process capture memory is temporarily unavailable."""


class _DeliveryRuntime(Protocol):
    def validate(self, destination: dict[str, Any], mode: str) -> None:
        ...

    async def submit(
        self,
        data: bytes,
        contents: list[str],
        destination: dict[str, Any],
        *,
        mode: str,
    ) -> dict[str, Any]:
        ...


def _validated_router_profile(profile):
    if not isinstance(profile, (tuple, list)):
        raise ValueError("router profile must be a list of router entries")
    # The format allows 4096 tensors; reserve one sequence and up to three logprob tensors.
    if len(profile) > 4092:
        raise ValueError(
            "router profile supports at most 4092 routers under the 4096-tensor format limit"
        )
    result: list[tuple[int, int, int]] = []
    ids = set()
    for site in profile:
        if not isinstance(site, (tuple, list)) or len(site) != 3:
            raise ValueError("invalid router profile entry")
        router_id, expert_count, width = site
        if (
            any(type(value) is not int for value in site)
            or not 0 <= router_id < 1 << 32
            or not 1 <= expert_count <= 65536
            or not 1 <= width <= expert_count
            or router_id in ids
            or (result and width != result[0][2])
        ):
            raise ValueError("unsupported router geometry")
        ids.add(router_id)
        result.append(cast(tuple[int, int, int], tuple(site)))
    return tuple(result)


def _validated_logprob_row(entry, token, width):
    if not isinstance(entry, dict) or not 1 <= len(entry) <= width + 1:
        raise ValueError("logprob row exceeds admitted width")
    if token not in entry:
        raise ValueError("selected token logprob unavailable")
    result = {}
    for candidate, value in entry.items():
        if type(candidate) is not int or not 0 <= candidate < 1 << 63:
            raise ValueError("invalid logprob candidate token")
        score = value.logprob
        if not isinstance(score, (int, float, np.integer, np.floating)) or isinstance(
            score, bool
        ):
            raise ValueError("invalid logprob value")
        try:
            score = float(score)
        except OverflowError as exc:
            raise ValueError("logprob value cannot be represented") from exc
        if (
            math.isnan(score)
            or score > np.finfo(np.float32).max
            or score < -np.finfo(np.float32).max
            and score != -math.inf
        ):
            raise ValueError("logprob value cannot be represented")
        result[candidate] = score
    return result


class CaptureBudget:
    """One atomic admission budget shared by all event loops in the process."""

    def __init__(self, max_bytes: int):
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("capture budget must be positive")
        self.max_bytes = max_bytes
        self._used = 0
        self._lock = threading.Lock()

    @property
    def used(self):
        with self._lock:
            return self._used

    def reserve(self, size):
        with self._lock:
            if self._used + size > self.max_bytes:
                raise CaptureCapacityError("artifact capture capacity exhausted")
            self._used += size

    def release(self, size):
        with self._lock:
            self._used -= size


@cache
def process_capture_budget():
    return CaptureBudget(
        int(os.environ.get("DYNAMO_ARTIFACT_CAPTURE_BYTES", str(2 << 30)))
    )


class VllmArtifactSession:
    """Single-choice capture; all queued route buffers are already compact."""

    settings: dict[str, Any]
    contents: frozenset[str]
    profile: tuple[tuple[int, int, int], ...]
    token_start: int
    max_prompt: int
    max_completion: int
    runtime: _DeliveryRuntime | None
    logprob_width: int
    prompt: tuple[int, ...] | None
    tokens: list[int]
    selected: list[float]
    candidates: list[list[int]]
    candidate_scores: list[list[float]]
    routes: MoeRoutes | None
    error: bool
    _closed: bool
    budget: CaptureBudget
    reserved_bytes: int

    @classmethod
    def from_request(
        cls,
        request,
        *,
        prompt_token_count=0,
        max_tokens=0,
        choice_count=1,
        route_capture_enabled=False,
        router_profile=(),
        token_start=0,
        logprob_width=1,
        budget=None,
        runtime=None,
    ):
        settings = _backend_metadata_upload_settings(request)
        if settings is None or "kind" not in settings:
            return None
        if settings.get("kind") != "generation_artifact":
            raise ValueError("unsupported metadata_upload.kind")
        if (request.get("stop_conditions") or {}).get("stop"):
            raise ValueError("artifact capture does not support frontend string stops")
        if settings.keys() - {"kind", "contents", "mode", "destination"}:
            raise ValueError("invalid artifact request fields")
        contents = settings.get("contents")
        if (
            not isinstance(contents, list)
            or any(type(x) is not str for x in contents)
            or len(contents) != len(set(contents))
            or set(contents) - _CONTENTS
        ):
            raise ValueError("invalid artifact contents")
        mode = settings.get("mode")
        if mode not in {"inline_base64", "object_store", "relay_store"}:
            raise ValueError("invalid artifact delivery mode")
        if mode == "inline_base64" and "destination" in settings:
            raise ValueError("inline artifact cannot contain destination")
        if mode != "inline_base64" and not isinstance(
            settings.get("destination"), dict
        ):
            raise ValueError("artifact destination is required")
        if type(choice_count) is not int or choice_count != 1:
            raise ValueError("artifact capture currently requires one choice")
        if type(logprob_width) is not int or not 1 <= logprob_width <= 1024:
            raise ValueError("unsupported artifact logprob width")
        if (
            type(prompt_token_count) is not int
            or prompt_token_count < 0
            or type(max_tokens) is not int
            or max_tokens < 0
            or prompt_token_count + max_tokens > 1 << 20
        ):
            raise ValueError("invalid artifact token limits")
        if type(token_start) is not int or not 0 <= token_start <= prompt_token_count:
            raise ValueError("invalid artifact token_start")
        profile = _validated_router_profile(router_profile)
        if "moe_routes" in contents and (
            not route_capture_enabled or not router_profile
        ):
            raise ValueError(
                "routed-expert capture and declared router profile are required"
            )
        owned_settings = deepcopy(settings)
        if mode != "inline_base64":
            if runtime is None:
                raise ValueError("artifact delivery authority must be configured")
            runtime.validate(owned_settings["destination"], mode)
        session = cls()
        session.settings = owned_settings
        session.contents = frozenset(contents)
        session.profile = profile
        session.token_start = token_start
        session.max_prompt = prompt_token_count
        session.max_completion = max_tokens
        session.runtime = runtime
        session.logprob_width = logprob_width
        session.prompt = None
        session.tokens = []
        session.selected = []
        session.candidates = []
        session.candidate_scores = []
        session.routes = None
        session.error = False
        session._closed = False
        session.budget = budget or process_capture_budget()
        extent = prompt_token_count + max_tokens
        route_elements = (
            extent * sum(site[2] for site in session.profile)
            if "moe_routes" in contents
            else 0
        )
        # Wide device capture, host scratch, compact source and encoding workspace.
        logprob_bytes = max_tokens * (
            4 * ("selected_logprobs" in contents)
            + 12 * logprob_width * ("topk_logprobs" in contents)
        )
        source_bound = (
            extent * 8
            + route_elements * 2
            + logprob_bytes
            + 8192
            + len(session.profile) * 512
        )
        session.reserved_bytes = route_elements * 16 + source_bound * 4
        if source_bound > 64 << 20:
            raise ValueError("artifact exceeds supported worst-case size")
        if mode == "inline_base64" and source_bound * 4 // 3 + 4 > 1 << 20:
            raise ValueError("artifact exceeds inline worst-case size")
        session.budget.reserve(session.reserved_bytes)
        return session

    async def observe_async(self, output, prompt_token_ids):
        native = getattr(output, "routed_experts", None)
        if native is None or "moe_routes" not in self.contents:
            self.observe(output, prompt_token_ids)
            return
        task = asyncio.create_task(
            asyncio.to_thread(self.observe, output, prompt_token_ids)
        )
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
            if not task.cancelled():
                task.exception()
            raise

    def observe(self, output, prompt_token_ids):
        if self.error or self._closed:
            return
        try:
            if getattr(output, "index", 0) != 0:
                raise ValueError("unexpected artifact choice")
            prompt = tuple(prompt_token_ids) if prompt_token_ids is not None else None
            if prompt is None or len(prompt) > self.max_prompt:
                raise ValueError("exact prompt IDs unavailable or exceed admission")
            if self.token_start > len(prompt):
                raise ValueError("token_start exceeds observed prompt")
            if self.prompt is not None and self.prompt != prompt:
                raise ValueError("prompt identity changed")
            self.prompt = prompt
            tokens = list(output.token_ids or [])
            if len(self.tokens) + len(tokens) > self.max_completion:
                raise ValueError("completion exceeds admission")
            self.tokens.extend(tokens)
            if self.contents & {"selected_logprobs", "topk_logprobs"}:
                entries = getattr(output, "logprobs", None)
                if entries is None and not tokens:
                    entries = []
                if entries is None or len(entries) != len(tokens):
                    raise ValueError("requested logprobs unavailable")
                for token, entry in zip(tokens, entries):
                    row = _validated_logprob_row(entry, token, self.logprob_width)
                    if "selected_logprobs" in self.contents:
                        self.selected.append(row[token])
                    if "topk_logprobs" in self.contents:
                        ranked = sorted(
                            row.items(),
                            key=lambda pair: (-pair[1], pair[0]),
                        )[: self.logprob_width]
                        if len(ranked) != self.logprob_width:
                            raise ValueError("requested candidate width unavailable")
                        self.candidates.append([candidate for candidate, _ in ranked])
                        self.candidate_scores.append([value for _, value in ranked])
            native = getattr(output, "routed_experts", None)
            if native is not None and "moe_routes" in self.contents:
                if (
                    not isinstance(native, np.ndarray)
                    or native.ndim != 3
                    or native.shape[1:] != (len(self.profile), self.profile[0][2])
                ):
                    raise ValueError("native route geometry differs from profile")
                extent = len(prompt) + len(self.tokens) - bool(self.tokens)
                if native.shape[0] > extent - self.token_start:
                    raise ValueError("native route extent exceeds observed sequence")
                self.routes = compact_routes(
                    native,
                    tuple(x[0] for x in self.profile),
                    tuple(x[1] for x in self.profile),
                    token_start=self.token_start,
                )
        except (ValueError, TypeError, AttributeError):
            self.error = True
            self._discard()

    def _view(self):
        if self.error or self.prompt is None:
            raise ValueError("invalid artifact capture")
        components: list[MoeRoutes | SelectedLogprobs | TopKLogprobs] = []
        if "moe_routes" in self.contents:
            if self.routes is None:
                raise ValueError("requested routes unavailable")
            components.append(self.routes)
        if "selected_logprobs" in self.contents:
            components.append(
                SelectedLogprobs(
                    len(self.prompt),
                    Tensor.from_array(np.asarray(self.selected, dtype=np.float32)),
                )
            )
        if "topk_logprobs" in self.contents:
            candidates = np.asarray(self.candidates, dtype=np.int64).reshape(
                -1, self.logprob_width
            )
            scores = np.asarray(self.candidate_scores, dtype=np.float32).reshape(
                -1, self.logprob_width
            )
            if candidates.ndim != 2 or scores.shape != candidates.shape:
                raise ValueError("top-k logprobs do not have constant width")
            components.append(
                TopKLogprobs(
                    len(self.prompt),
                    Tensor.from_array(candidates),
                    Tensor.from_array(scores),
                )
            )
        tokens = Tensor.from_array(
            np.asarray((*self.prompt, *self.tokens), dtype=np.int64)
        )
        return GenerationArtifactView(
            (GenerationArtifactChoice(0, len(self.prompt), tokens, tuple(components)),)
        )

    async def finalize(self, *, release_capture=True):
        encoding = None
        try:
            encoding = asyncio.create_task(
                asyncio.to_thread(encode_generation_artifact, self._view())
            )
            encoded = await asyncio.shield(encoding)
            if self.settings["mode"] == "inline_base64":
                if encoded.byte_count > (1 << 20) * 3 // 4:
                    raise ValueError("encoded artifact exceeds inline size")
                return {
                    "kind": "generation_artifact",
                    "mode": "inline_base64",
                    "format": "generation_artifact_v1",
                    "contents": sorted(self.contents),
                    "byte_count": encoded.byte_count,
                    "data_base64": base64.b64encode(encoded.data).decode("ascii"),
                }
            try:
                return await cast(_DeliveryRuntime, self.runtime).submit(
                    encoded.data,
                    sorted(self.contents),
                    self.settings["destination"],
                    mode=self.settings["mode"],
                )
            except (ValueError, TypeError, RuntimeError, OSError, sqlite3.Error):
                return {
                    "kind": "generation_artifact",
                    "mode": self.settings["mode"],
                    "format": "generation_artifact_v1",
                    "contents": sorted(self.contents),
                    "state": "failed",
                    "error_code": "delivery_rejected",
                }
        except asyncio.CancelledError:
            if encoding is not None:
                while not encoding.done():
                    try:
                        await asyncio.shield(encoding)
                    except asyncio.CancelledError:
                        continue
                encoding.exception()
            raise
        except (ValueError, TypeError, RuntimeError, OSError):
            return {
                "kind": "generation_artifact",
                "mode": self.settings["mode"],
                "format": "generation_artifact_v1",
                "contents": sorted(self.contents),
                "state": "failed",
                "error_code": "capture_invalid",
            }
        finally:
            if release_capture:
                self.close()

    def close(self):
        if self.reserved_bytes:
            self.budget.release(self.reserved_bytes)
            self.reserved_bytes = 0
        self._closed = True
        self._discard()

    def _discard(self):
        self.routes = None
        self.prompt = None
        self.tokens = []
        self.selected = []
        self.candidates = []
        self.candidate_scores = []
