# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construct vLLM serving adapters and callbacks for Realtime handlers."""

from __future__ import annotations

import asyncio
import contextvars
import logging
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine
from contextlib import aclosing
from importlib.util import find_spec
from typing import Any

import numpy as np

#: Adapt float32 audio chunks and generated-token feedback to vLLM StreamingInput.
StreamingInputFactory = Callable[
    [AsyncGenerator[np.ndarray, None], "asyncio.Queue[list[int]]"],
    AsyncGenerator[Any, None],
]
#: Await with chat messages, an optional output-token limit, chat ``tools`` /
#: ``tool_choice`` options, and the turn's held prefill request, if any, to obtain
#: a stream of OpenAI chat-completion SSE frames. The caller owns closing that
#: stream and the held request. The factory only borrows the held request.
ChatCompletionFactory = Callable[
    [list[dict[str, Any]], int | None, dict[str, Any], "HeldTextRequest | None"],
    Awaitable[AsyncGenerator[str, None]],
]
#: Prefill cumulative user-text updates alongside committed chat history, and
#: return the engine request holding that prefix once updates end. Cancellation
#: must complete engine cleanup before returning. Takes the same tool options as
#: ``ChatCompletionFactory`` so both render one prefix.
TextPrefillFactory = Callable[
    [list[dict[str, Any]], AsyncGenerator[str, None], dict[str, Any]],
    Coroutine[Any, Any, "HeldTextRequest | None"],
]

logger = logging.getLogger(__name__)

_HELD_REQUEST: contextvars.ContextVar[HeldTextRequest | None] = contextvars.ContextVar(
    "realtime_held_text_request", default=None
)


def vllm_parser_names(
    tool_parser: str | None, reasoning_parser: str | None
) -> tuple[str | None, str | None]:
    """Keep only the parser names that vLLM registers.

    Realtime text parses with vLLM's chat serving, but the names come from the
    Dynamo parser flags, and some Dynamo parsers have no vLLM counterpart. vLLM
    rejects unknown names, so an unknown name disables that parsing instead of
    failing the worker.
    """
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager

    def known(name: str | None, kind: str, lookup: Callable[[str], Any]) -> bool:
        if not name:
            return False
        try:
            lookup(name)
        except KeyError:
            logger.warning(
                "%s parser %r is not a vLLM parser, so realtime text sessions "
                "run without it",
                kind,
                name,
            )
            return False
        return True

    return (
        tool_parser
        if known(tool_parser, "Tool call", ToolParserManager.get_tool_parser)
        else None,
        reasoning_parser
        if known(
            reasoning_parser, "Reasoning", ReasoningParserManager.get_reasoning_parser
        )
        else None,
    )


class HeldTextRequest:
    """One StreamingInput engine request that prefills a turn and then generates its answer.

    Prefill chunks generate one throwaway token, which vLLM drops when the next
    chunk arrives, so the engine keeps the prompt's KV cache and recurrent state
    without relying on prefix caching. The answer must be the last chunk: a
    resumable request whose input stream stays open can keep decoding past EOS.
    """

    def __init__(
        self,
        engine_client: Any,
        make_chunk: Callable[[list[int], Any], Any],
        prefill_params: Any,
    ) -> None:
        self.fed: list[int] = []
        self.prefill_params = prefill_params
        self._engine = engine_client
        self._make_chunk = make_chunk
        self._pending: list[int] = []
        self._next_answer: tuple[Any, asyncio.Queue[Any]] | None = None
        self._answer: asyncio.Queue[Any] | None = None
        self._wake = asyncio.Event()
        self._chunk_done = asyncio.Event()
        self._request_id = f"rt-prefill-{uuid.uuid4().hex}"
        self._task: asyncio.Task[None] | None = None
        self._closed = False

    def extends(self, token_ids: list[int]) -> bool:
        """Whether ``token_ids`` continue the fed prompt and can still be appended."""
        return not self._closed and token_ids[: len(self.fed)] == self.fed

    def feed(self, token_ids: list[int]) -> None:
        """Queue the tokens of ``token_ids`` beyond the fed prompt, which it must extend."""
        delta = token_ids[len(self.fed) :]
        if delta:
            self._queue(delta)

    def answer(
        self, token_ids: list[int], sampling_params: Any
    ) -> AsyncGenerator[Any, None]:
        """Append the rest of the final prompt as the last chunk and stream its outputs."""
        queue: asyncio.Queue[Any] = asyncio.Queue()
        self._next_answer = (sampling_params, queue)
        self._queue(token_ids[len(self.fed) :])
        self._closed = True
        return self._answer_outputs(queue)

    async def close(self) -> None:
        """Cancel the request and abort it on the engine."""
        self._closed = True
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await self._engine.abort(self._request_id)

    def _queue(self, token_ids: list[int]) -> None:
        self.fed += token_ids
        self._pending += token_ids
        if self._task is None:
            self._task = asyncio.create_task(self._run())
        self._wake.set()

    async def _answer_outputs(
        self, queue: asyncio.Queue[Any]
    ) -> AsyncGenerator[Any, None]:
        while (output := await queue.get()) is not None:
            if isinstance(output, Exception):
                raise output
            yield output

    def _claim_answer(self) -> asyncio.Queue[Any] | None:
        if self._answer is None and self._next_answer is not None:
            _, self._answer = self._next_answer
            self._next_answer = None
        return self._answer

    async def _inputs(self) -> AsyncGenerator[Any, None]:
        while True:
            await self._wake.wait()
            self._wake.clear()
            token_ids, self._pending = self._pending, []
            answer, self._next_answer = self._next_answer, None
            if not token_ids and answer is None:
                continue
            self._chunk_done.clear()
            if answer is not None:
                params, self._answer = answer
                yield self._make_chunk(token_ids, params)
                return
            yield self._make_chunk(token_ids, self.prefill_params)
            await self._chunk_done.wait()

    async def _run(self) -> None:
        try:
            async for output in self._engine.generate(
                prompt=self._inputs(),
                sampling_params=self.prefill_params,
                request_id=self._request_id,
            ):
                if self._answer is None:
                    self._chunk_done.set()
                    continue
                self._answer.put_nowait(output)
                if output.finished:
                    break
        except Exception as exc:  # noqa: BLE001 - surfaced as an answer failure
            if (answer := self._claim_answer()) is None:
                logger.warning("realtime text prefill stopped: %s", exc)
            else:
                answer.put_nowait(exc)
        finally:
            self._closed = True
            self._chunk_done.set()
            if (answer := self._claim_answer()) is not None:
                answer.put_nowait(None)


class _HeldRequestEngine:
    """Engine client whose ``generate`` continues the turn's held prefill request.

    vLLM's chat serving still validates the request, renders the prompt, and
    builds the sampling parameters. Only the engine call reuses the held
    request, when the final prompt extends it and the answer needs no
    per-request engine options or seed.
    """

    def __init__(self, engine_client: Any) -> None:
        self._engine = engine_client

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)

    def generate(
        self, prompt: Any, sampling_params: Any, request_id: str, **kwargs: Any
    ) -> AsyncGenerator[Any, None]:
        held = _HELD_REQUEST.get()
        token_ids = prompt.get("prompt_token_ids") if isinstance(prompt, dict) else None
        if (
            held is not None
            and token_ids is not None
            and len(token_ids) > len(held.fed)
            and held.extends(list(token_ids))
            and sampling_params.structured_outputs is None
            and sampling_params.seed is None
            and sampling_params.skip_special_tokens
            == held.prefill_params.skip_special_tokens
            and sampling_params.spaces_between_special_tokens
            == held.prefill_params.spaces_between_special_tokens
            and kwargs.get("lora_request") is None
        ):
            return held.answer(list(token_ids), sampling_params)
        return self._engine.generate(prompt, sampling_params, request_id, **kwargs)


def _build_models(*, engine_client: Any, model_name: str, model_path: str) -> Any:
    from vllm.entrypoints.openai.models.protocol import BaseModelPath
    from vllm.entrypoints.openai.models.serving import OpenAIServingModels

    return OpenAIServingModels(
        engine_client=engine_client,
        base_model_paths=[BaseModelPath(name=model_name, model_path=model_path)],
        lora_modules=None,
    )


def build_realtime_serving(
    *,
    engine_client: Any,
    model_name: str,
    model_path: str,
) -> Any:
    """Build vLLM's OpenAI realtime serving adapter for one model."""
    from vllm.entrypoints.speech_to_text.realtime.serving import OpenAIServingRealtime

    models = _build_models(
        engine_client=engine_client,
        model_name=model_name,
        model_path=model_path,
    )
    return OpenAIServingRealtime(
        engine_client=engine_client,
        models=models,
        request_logger=None,
    )


def build_realtime_text_factories(
    *,
    engine_client: Any,
    model_name: str,
    model_path: str,
    chat_template_path: str | None,
    tool_parser: str | None = None,
    reasoning_parser: str | None = None,
    default_chat_template_kwargs: dict[str, Any] | None = None,
) -> tuple[ChatCompletionFactory, TextPrefillFactory]:
    """Bind final-generation and prefix-warming callbacks to one vLLM engine.

    ``model_name`` is the served API name; ``model_path`` identifies its weights.
    ``chat_template_path`` optionally overrides the model's default chat template.
    Both callbacks use vLLM's chat renderer to apply that template and tokenize.
    ``tool_parser`` enables tool calls in the SSE stream. ``reasoning_parser``
    moves reasoning out of ``delta.content``. ``default_chat_template_kwargs``
    applies to every render, since Realtime sessions carry no template options.

    Awaiting the first callback with chat messages and an optional output-token
    limit returns an SSE stream that the caller must close. The second prefills
    cumulative pending-user text with committed chat history on one vLLM
    StreamingInput request and returns that request when its updates end. Final
    generation renders the exact committed conversation and appends it to the
    held request when it extends what the request holds, so the answer starts
    from the prefilled state instead of a prefix-cache lookup. Hybrid models
    cache recurrent state only at large block boundaries, so a separate request
    would recompute most of the prompt.
    """
    from vllm.engine.protocol import StreamingInput
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.entrypoints.serve.engine.protocol import ErrorResponse
    from vllm.inputs import tokens_input
    from vllm.renderers.online_renderer import OnlineRenderer
    from vllm.sampling_params import RequestOutputKind, SamplingParams

    # vLLM 0.32 moved the chat helpers to vllm.renderers.
    if find_spec("vllm.renderers.chat_utils") is not None:
        from vllm.renderers.chat_utils import load_chat_template
    else:
        from vllm.entrypoints.chat_utils import load_chat_template

    class RealtimeServingChat(OpenAIServingChat):
        async def chat_completion_stream_generator(
            self,
            request: ChatCompletionRequest,
            result_generator: Any,
            *args: Any,
            **kwargs: Any,
        ) -> AsyncGenerator[str, None]:
            # Closing the SSE generator alone does not close its engine iterator.
            async with aclosing(result_generator), aclosing(
                super().chat_completion_stream_generator(
                    request, result_generator, *args, **kwargs
                )
            ) as stream:
                async for frame in stream:
                    yield frame

    chat_template = load_chat_template(chat_template_path)
    online_renderer = OnlineRenderer(
        model_config=engine_client.model_config,
        renderer=engine_client.renderer,
        request_logger=None,
        chat_template=chat_template,
        chat_template_content_format="auto",
        enable_auto_tools=bool(tool_parser),
        tool_parser=tool_parser,
        reasoning_parser=reasoning_parser,
        default_chat_template_kwargs=default_chat_template_kwargs,
    )
    serving = RealtimeServingChat(
        engine_client=_HeldRequestEngine(engine_client),
        models=_build_models(
            engine_client=engine_client,
            model_name=model_name,
            model_path=model_path,
        ),
        response_role="assistant",
        online_renderer=online_renderer,
        request_logger=None,
        chat_template=chat_template,
        chat_template_content_format="auto",
        reasoning_parser=reasoning_parser or "",
        enable_auto_tools=bool(tool_parser),
        tool_parser=tool_parser,
        default_chat_template_kwargs=default_chat_template_kwargs,
    )

    async def create_chat_completion(
        messages: list[dict[str, Any]],
        max_output_tokens: int | None,
        tool_options: dict[str, Any],
        prefill: HeldTextRequest | None,
    ) -> AsyncGenerator[str, None]:
        request = ChatCompletionRequest(
            messages=messages,
            model=model_name,
            max_completion_tokens=max_output_tokens,
            stream=True,
            stream_options={"include_usage": True},
            **tool_options,
        )
        token = _HELD_REQUEST.set(prefill)
        try:
            response = await serving.create_chat_completion(request)
        finally:
            _HELD_REQUEST.reset(token)
        if isinstance(response, ErrorResponse):
            raise ValueError(response.error.message)
        return response

    async def render_tokens(
        messages: list[dict[str, Any]], tool_options: dict[str, Any]
    ) -> tuple[list[int], ChatCompletionRequest]:
        request = ChatCompletionRequest(
            messages=messages,
            model=model_name,
            add_generation_prompt=False,
            continue_final_message=True,
            **tool_options,
        )
        rendered = await serving.render_chat_request(request)
        if isinstance(rendered, ErrorResponse):
            raise ValueError(rendered.error.message)
        _, engine_inputs = rendered
        if len(engine_inputs) != 1:
            raise ValueError("Realtime text prefill requires one rendered prompt")
        token_ids = engine_inputs[0].get("prompt_token_ids")
        if token_ids is None:
            raise ValueError("Realtime text prefill requires tokenized chat input")
        return list(token_ids), request

    def make_chunk(token_ids: list[int], params: SamplingParams) -> StreamingInput:
        return StreamingInput(prompt=tokens_input(token_ids), sampling_params=params)

    async def prefill_text(
        messages: list[dict[str, Any]],
        updates: AsyncGenerator[str, None],
        tool_options: dict[str, Any],
    ) -> HeldTextRequest | None:
        held: HeldTextRequest | None = None
        try:
            async for text in updates:
                text = text[: text.rfind(" ") + 1].rstrip()
                if not text:
                    continue
                token_ids, request = await render_tokens(
                    [*messages, {"role": "user", "content": text}],
                    tool_options,
                )
                if held is not None and not held.extends(token_ids):
                    await held.close()
                    held = None
                if held is None:
                    held = HeldTextRequest(
                        engine_client,
                        make_chunk,
                        SamplingParams.from_optional(
                            temperature=0.0,
                            max_tokens=1,
                            ignore_eos=True,
                            output_kind=RequestOutputKind.DELTA,
                            skip_special_tokens=request.skip_special_tokens,
                            spaces_between_special_tokens=(
                                request.spaces_between_special_tokens
                            ),
                        ),
                    )
                held.feed(token_ids)
        except asyncio.CancelledError:
            if held is not None:
                await asyncio.shield(held.close())
            raise
        except Exception as exc:  # noqa: BLE001 - prefill is an optional optimization
            logger.warning("realtime text prefill disabled for this turn: %s", exc)
            if held is not None:
                await held.close()
            return None
        return held

    return create_chat_completion, prefill_text
