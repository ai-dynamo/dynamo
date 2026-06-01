# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async client for the prompt-enhanced text-to-video pipeline.

Chains a small LLM rewriter (OpenAI chat-completions API) and a diffusion
text-to-video backend (OpenAI /v1/videos API) behind a single ``generate``
call. Supports a per-request ``enhancer`` override that skips the LLM hop
entirely, which is the cheapest way to honor empirical findings that
larger T2V backbones do not benefit from short-prompt rewriting.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

import aiohttp


class EnhancerMode(str, Enum):
    AUTO = "auto"
    OFF = "off"


class T2VEnhancedClientError(RuntimeError):
    """Raised when either the LLM or T2V endpoint returns a non-2xx status."""


_DEFAULT_SYSTEM_PROMPT = (
    "You are a cinematographer. Given a short user idea, rewrite it as ONE "
    "rich, vivid English sentence (no more than 40 words) suitable as a "
    "text-to-video prompt. Include camera framing, lighting, mood, and "
    "motion. Output the rewritten prompt only - no preamble, no quotes, no "
    "thinking tags."
)


@dataclass
class T2VTimings:
    """Per-request timing breakdown in milliseconds (monotonic clock)."""

    submit_ms: float = 0.0
    enhance_start_ms: float = 0.0
    enhance_end_ms: float = 0.0
    t2v_start_ms: float = 0.0
    t2v_end_ms: float = 0.0
    response_ms: float = 0.0

    @property
    def enhance_ms(self) -> float:
        if not self.enhance_end_ms:
            return 0.0
        return self.enhance_end_ms - self.enhance_start_ms

    @property
    def t2v_ms(self) -> float:
        return self.t2v_end_ms - self.t2v_start_ms

    @property
    def e2e_ms(self) -> float:
        return self.response_ms - self.submit_ms


@dataclass
class T2VResult:
    """Returned by ``T2VEnhancedClient.generate``."""

    url: str
    rewritten_prompt: Optional[str]
    timings: T2VTimings
    raw_t2v_response: Mapping[str, Any] = field(default_factory=dict)


class T2VEnhancedClient:
    """Two-endpoint async client: LLM enhancer -> T2V backend.

    The two endpoints are two independent Dynamo deployments. This client
    chains them request-by-request and provides:
      * a single ``.generate(prompt, ...)`` entrypoint;
      * a per-request ``enhancer`` knob ("auto" or "off") so callers can
        skip the LLM hop when the diffusion backbone is large enough that
        prompt rewriting hurts quality;
      * a chained timing record so the caller can correlate LLM and T2V
        latencies under a single request id.

    Parameters
    ----------
    llm_url : str
        Base URL of the Dynamo HTTP frontend serving the enhancer model.
    t2v_url : str
        Base URL of the Dynamo HTTP frontend serving the diffusion model.
    llm_model : str
        Model name registered against ``llm_url``, e.g. ``Qwen/Qwen3-0.6B``.
    t2v_model : str
        Model name registered against ``t2v_url``.
    default_enhancer : EnhancerMode | str
        Mode used when ``generate`` is called without an explicit
        ``enhancer`` argument. ``"auto"`` runs the LLM enhancer; ``"off"``
        skips it and forwards the user prompt verbatim.
    system_prompt : Optional[str]
        Override the built-in cinematographer system prompt.
    enhancer_max_tokens : int
        Cap on rewritten-prompt length.
    enhancer_temperature : float
        Sampling temperature for the enhancer. Defaults to 0 for
        determinism.
    timeout_s : float
        Per-request timeout applied to both legs.
    """

    def __init__(
        self,
        *,
        llm_url: str,
        t2v_url: str,
        llm_model: str,
        t2v_model: str,
        default_enhancer: EnhancerMode | str = EnhancerMode.AUTO,
        system_prompt: Optional[str] = None,
        enhancer_max_tokens: int = 80,
        enhancer_temperature: float = 0.0,
        timeout_s: float = 120.0,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        self._llm_url = llm_url.rstrip("/")
        self._t2v_url = t2v_url.rstrip("/")
        self._llm_model = llm_model
        self._t2v_model = t2v_model
        self._default_enhancer = EnhancerMode(default_enhancer)
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self._enhancer_max_tokens = int(enhancer_max_tokens)
        self._enhancer_temperature = float(enhancer_temperature)
        self._timeout = aiohttp.ClientTimeout(total=timeout_s)
        self._owns_session = session is None
        self._session = session

    async def __aenter__(self) -> "T2VEnhancedClient":
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
            self._owns_session = True
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def close(self) -> None:
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    def _resolve_enhancer(self, override: Optional[EnhancerMode | str]) -> EnhancerMode:
        if override is None:
            return self._default_enhancer
        return EnhancerMode(override)

    def _session_or_raise(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise T2VEnhancedClientError(
                "T2VEnhancedClient is not entered: use 'async with client:' "
                "or pass an external aiohttp.ClientSession to the constructor."
            )
        return self._session

    async def _enhance(self, prompt: str) -> str:
        session = self._session_or_raise()
        payload = {
            "model": self._llm_model,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self._enhancer_max_tokens,
            "temperature": self._enhancer_temperature,
        }
        async with session.post(
            f"{self._llm_url}/v1/chat/completions", json=payload
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise T2VEnhancedClientError(
                    f"LLM enhancer returned HTTP {resp.status}: {body!r}"
                )
            data = await resp.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError) as exc:
            raise T2VEnhancedClientError(
                f"Unexpected enhancer response shape: {data!r}"
            ) from exc

    async def _t2v(self, prompt: str, t2v_params: Mapping[str, Any]) -> Mapping[str, Any]:
        session = self._session_or_raise()
        payload = {"model": self._t2v_model, "prompt": prompt, **t2v_params}
        async with session.post(f"{self._t2v_url}/v1/videos", json=payload) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise T2VEnhancedClientError(
                    f"T2V backend returned HTTP {resp.status}: {body!r}"
                )
            return await resp.json()

    async def generate(
        self,
        prompt: str,
        *,
        enhancer: Optional[EnhancerMode | str] = None,
        **t2v_params: Any,
    ) -> T2VResult:
        """Run the chained enhancer + T2V pipeline.

        When ``enhancer`` resolves to ``EnhancerMode.OFF`` the LLM hop is
        skipped entirely and ``rewritten_prompt`` in the result is
        ``None``. Otherwise the LLM enhancer is invoked first and its
        output replaces the ``prompt`` field passed to the T2V backend.
        Any additional ``t2v_params`` are forwarded verbatim to the T2V
        endpoint (for example ``size``, ``num_inference_steps``,
        ``num_frames``, ``nvext``).
        """
        loop = asyncio.get_event_loop()
        now = loop.time

        timings = T2VTimings(submit_ms=now() * 1000.0)
        rewritten: Optional[str] = None
        mode = self._resolve_enhancer(enhancer)

        if mode is EnhancerMode.OFF:
            text_prompt = prompt
        else:
            timings.enhance_start_ms = now() * 1000.0
            rewritten = await self._enhance(prompt)
            timings.enhance_end_ms = now() * 1000.0
            text_prompt = rewritten

        timings.t2v_start_ms = now() * 1000.0
        raw = await self._t2v(text_prompt, t2v_params)
        timings.t2v_end_ms = now() * 1000.0
        timings.response_ms = timings.t2v_end_ms

        try:
            url = raw["data"][0]["url"]
        except (KeyError, IndexError, TypeError) as exc:
            raise T2VEnhancedClientError(
                f"Unexpected T2V response shape: {raw!r}"
            ) from exc

        return T2VResult(
            url=url,
            rewritten_prompt=rewritten,
            timings=timings,
            raw_t2v_response=raw,
        )
