# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async clients for prompt-enhanced visual generation.

Chains a small LLM rewriter (OpenAI chat-completions API) and a diffusion
backend behind a single ``generate`` call. The concrete clients target
OpenAI-compatible text-to-video (``/v1/videos``) and text-to-image
(``/v1/images/generations``) endpoints. Supports a per-request
``enhancer`` override that skips the LLM hop entirely, which is the cheapest
way to honor empirical findings that larger diffusion backbones do not always
benefit from short-prompt rewriting.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, Mapping, Optional, TypeVar

import aiohttp


class EnhancerMode(str, Enum):
    AUTO = "auto"
    OFF = "off"


class OutputModality(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class PromptEnhancedClientError(RuntimeError):
    """Raised when either the LLM or diffusion endpoint returns an error."""


T2VEnhancedClientError = PromptEnhancedClientError
T2IEnhancedClientError = PromptEnhancedClientError


_DEFAULT_SYSTEM_PROMPTS = {
    OutputModality.VIDEO: (
        "You are a cinematographer. Given a short user idea, rewrite it as ONE "
        "rich, vivid English sentence (no more than 40 words) suitable as a "
        "text-to-video prompt. Include camera framing, lighting, mood, and "
        "motion. Output the rewritten prompt only - no preamble, no quotes, no "
        "thinking tags."
    ),
    OutputModality.IMAGE: (
        "You are a visual prompt engineer. Given a short user idea, rewrite it "
        "as ONE rich, vivid English sentence (no more than 40 words) suitable "
        "as a text-to-image prompt. Include subject, setting, style, lighting, "
        "composition, and mood. Output the rewritten prompt only - no preamble, "
        "no quotes, no thinking tags."
    ),
}


@dataclass
class PromptEnhanceTimings:
    """Per-request timing breakdown in milliseconds (monotonic clock)."""

    submit_ms: float = 0.0
    enhance_start_ms: float = 0.0
    enhance_end_ms: float = 0.0
    generation_start_ms: float = 0.0
    generation_end_ms: float = 0.0
    response_ms: float = 0.0

    @property
    def enhance_ms(self) -> float:
        if not self.enhance_end_ms:
            return 0.0
        return self.enhance_end_ms - self.enhance_start_ms

    @property
    def generation_ms(self) -> float:
        return self.generation_end_ms - self.generation_start_ms

    @property
    def t2v_start_ms(self) -> float:
        return self.generation_start_ms

    @t2v_start_ms.setter
    def t2v_start_ms(self, value: float) -> None:
        self.generation_start_ms = value

    @property
    def t2v_end_ms(self) -> float:
        return self.generation_end_ms

    @t2v_end_ms.setter
    def t2v_end_ms(self, value: float) -> None:
        self.generation_end_ms = value

    @property
    def t2v_ms(self) -> float:
        return self.generation_ms

    @property
    def t2i_start_ms(self) -> float:
        return self.generation_start_ms

    @t2i_start_ms.setter
    def t2i_start_ms(self, value: float) -> None:
        self.generation_start_ms = value

    @property
    def t2i_end_ms(self) -> float:
        return self.generation_end_ms

    @t2i_end_ms.setter
    def t2i_end_ms(self, value: float) -> None:
        self.generation_end_ms = value

    @property
    def t2i_ms(self) -> float:
        return self.generation_ms

    @property
    def e2e_ms(self) -> float:
        return self.response_ms - self.submit_ms


T2VTimings = PromptEnhanceTimings
T2ITimings = PromptEnhanceTimings


@dataclass
class PromptEnhancedResult:
    """Returned by ``PromptEnhancedClient.generate``."""

    url: Optional[str]
    b64_json: Optional[str]
    rewritten_prompt: Optional[str]
    timings: PromptEnhanceTimings
    raw_response: Mapping[str, Any] = field(default_factory=dict)


class T2VResult(PromptEnhancedResult):
    """Returned by ``T2VEnhancedClient.generate``."""

    @property
    def raw_t2v_response(self) -> Mapping[str, Any]:
        return self.raw_response


class T2IResult(PromptEnhancedResult):
    """Returned by ``T2IEnhancedClient.generate``."""

    @property
    def raw_t2i_response(self) -> Mapping[str, Any]:
        return self.raw_response


_ResultT = TypeVar("_ResultT", bound=PromptEnhancedResult)


class PromptEnhancedClient(Generic[_ResultT]):
    """Two-endpoint async client: LLM enhancer -> diffusion backend.

    The two endpoints are two independent Dynamo deployments. This client
    chains them request-by-request and provides:
      * a single ``.generate(prompt, ...)`` entrypoint;
      * a per-request ``enhancer`` knob ("auto" or "off") so callers can
        skip the LLM hop when the diffusion backbone is large enough that
        prompt rewriting hurts quality;
      * a chained timing record so the caller can correlate LLM and generation
        latencies under a single request id.

    Parameters
    ----------
    llm_url : str
        Base URL of the Dynamo HTTP frontend serving the enhancer model.
    backend_url : str
        Base URL of the Dynamo HTTP frontend serving the diffusion model.
    llm_model : str
        Model name registered against ``llm_url``, e.g. ``Qwen/Qwen3-0.6B``.
    backend_model : str
        Model name registered against ``backend_url``.
    backend_endpoint : str
        OpenAI-compatible generation endpoint path.
    default_enhancer : EnhancerMode | str
        Mode used when ``generate`` is called without an explicit
        ``enhancer`` argument. ``"auto"`` runs the LLM enhancer; ``"off"``
        skips it and forwards the user prompt verbatim.
    system_prompt : Optional[str]
        Override the built-in modality-specific system prompt.
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
        backend_url: str,
        llm_model: str,
        backend_model: str,
        backend_endpoint: str,
        backend_name: str,
        modality: OutputModality | str,
        result_type: type[_ResultT],
        default_enhancer: EnhancerMode | str = EnhancerMode.AUTO,
        system_prompt: Optional[str] = None,
        enhancer_max_tokens: int = 80,
        enhancer_temperature: float = 0.0,
        timeout_s: float = 120.0,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        self._llm_url = llm_url.rstrip("/")
        self._backend_url = backend_url.rstrip("/")
        self._llm_model = llm_model
        self._backend_model = backend_model
        self._backend_endpoint = backend_endpoint
        self._backend_name = backend_name
        self._modality = OutputModality(modality)
        self._result_type = result_type
        self._default_enhancer = EnhancerMode(default_enhancer)
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPTS[self._modality]
        self._enhancer_max_tokens = int(enhancer_max_tokens)
        self._enhancer_temperature = float(enhancer_temperature)
        self._timeout = aiohttp.ClientTimeout(total=timeout_s)
        self._owns_session = session is None
        self._session = session

    async def __aenter__(self) -> "PromptEnhancedClient[_ResultT]":
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
            raise PromptEnhancedClientError(
                f"{self.__class__.__name__} is not entered: use 'async with client:' "
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
            f"{self._llm_url}/v1/chat/completions",
            json=payload,
            timeout=self._timeout,
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise PromptEnhancedClientError(
                    f"LLM enhancer returned HTTP {resp.status}: {body!r}"
                )
            try:
                data = await resp.json()
            except (aiohttp.ContentTypeError, ValueError) as exc:
                raise PromptEnhancedClientError(
                    f"LLM enhancer returned a non-JSON response: {exc}"
                ) from exc
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError) as exc:
            raise PromptEnhancedClientError(
                f"Unexpected enhancer response shape: {data!r}"
            ) from exc

    async def _generate_backend(
        self, prompt: str, generation_params: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        session = self._session_or_raise()
        payload = {"model": self._backend_model, "prompt": prompt, **generation_params}
        async with session.post(
            f"{self._backend_url}{self._backend_endpoint}",
            json=payload,
            timeout=self._timeout,
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise PromptEnhancedClientError(
                    f"{self._backend_name} returned HTTP {resp.status}: {body!r}"
                )
            try:
                return await resp.json()
            except (aiohttp.ContentTypeError, ValueError) as exc:
                raise PromptEnhancedClientError(
                    f"{self._backend_name} returned a non-JSON response: {exc}"
                ) from exc

    def _extract_generation_media(
        self, raw: Mapping[str, Any]
    ) -> tuple[Optional[str], Optional[str]]:
        try:
            item = raw["data"][0]  # type: ignore[index]
            url = item.get("url")
            b64_json = item.get("b64_json")
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise PromptEnhancedClientError(
                f"Unexpected {self._backend_name} response shape: {raw!r}"
            ) from exc

        if url is None and b64_json is None:
            raise PromptEnhancedClientError(
                f"Unexpected {self._backend_name} response shape: {raw!r}"
            )
        return url, b64_json

    async def generate(
        self,
        prompt: str,
        *,
        enhancer: Optional[EnhancerMode | str] = None,
        **generation_params: Any,
    ) -> _ResultT:
        """Run the chained enhancer + diffusion generation pipeline.

        When ``enhancer`` resolves to ``EnhancerMode.OFF`` the LLM hop is
        skipped entirely and ``rewritten_prompt`` in the result is
        ``None``. Otherwise the LLM enhancer is invoked first and its
        output replaces the ``prompt`` field passed to the generation backend.
        Any additional ``generation_params`` are forwarded verbatim to the
        backend endpoint (for example ``size``, ``response_format``,
        ``input_reference``, ``num_frames``, ``nvext``).
        """
        loop = asyncio.get_running_loop()
        now = loop.time

        timings = PromptEnhanceTimings(submit_ms=now() * 1000.0)
        rewritten: Optional[str] = None
        mode = self._resolve_enhancer(enhancer)

        if mode is EnhancerMode.OFF:
            text_prompt = prompt
        else:
            timings.enhance_start_ms = now() * 1000.0
            rewritten = await self._enhance(prompt)
            timings.enhance_end_ms = now() * 1000.0
            text_prompt = rewritten

        timings.generation_start_ms = now() * 1000.0
        raw = await self._generate_backend(text_prompt, generation_params)
        timings.generation_end_ms = now() * 1000.0
        timings.response_ms = timings.generation_end_ms

        url, b64_json = self._extract_generation_media(raw)

        return self._result_type(
            url=url,
            b64_json=b64_json,
            rewritten_prompt=rewritten,
            timings=timings,
            raw_response=raw,
        )


class T2VEnhancedClient(PromptEnhancedClient[T2VResult]):
    """Two-endpoint async client: LLM enhancer -> T2V backend."""

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
        super().__init__(
            llm_url=llm_url,
            backend_url=t2v_url,
            llm_model=llm_model,
            backend_model=t2v_model,
            backend_endpoint="/v1/videos",
            backend_name="T2V backend",
            modality=OutputModality.VIDEO,
            result_type=T2VResult,
            default_enhancer=default_enhancer,
            system_prompt=system_prompt,
            enhancer_max_tokens=enhancer_max_tokens,
            enhancer_temperature=enhancer_temperature,
            timeout_s=timeout_s,
            session=session,
        )

    async def __aenter__(self) -> "T2VEnhancedClient":
        await super().__aenter__()
        return self


class T2IEnhancedClient(PromptEnhancedClient[T2IResult]):
    """Two-endpoint async client: LLM enhancer -> T2I backend."""

    def __init__(
        self,
        *,
        llm_url: str,
        t2i_url: str,
        llm_model: str,
        t2i_model: str,
        default_enhancer: EnhancerMode | str = EnhancerMode.AUTO,
        system_prompt: Optional[str] = None,
        enhancer_max_tokens: int = 80,
        enhancer_temperature: float = 0.0,
        timeout_s: float = 120.0,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        super().__init__(
            llm_url=llm_url,
            backend_url=t2i_url,
            llm_model=llm_model,
            backend_model=t2i_model,
            backend_endpoint="/v1/images/generations",
            backend_name="T2I backend",
            modality=OutputModality.IMAGE,
            result_type=T2IResult,
            default_enhancer=default_enhancer,
            system_prompt=system_prompt,
            enhancer_max_tokens=enhancer_max_tokens,
            enhancer_temperature=enhancer_temperature,
            timeout_s=timeout_s,
            session=session,
        )

    async def __aenter__(self) -> "T2IEnhancedClient":
        await super().__aenter__()
        return self
