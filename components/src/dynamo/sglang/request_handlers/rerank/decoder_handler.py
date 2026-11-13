# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text Qwen3 reranking with Dynamo-owned scoring request lifetimes."""

import asyncio
import math
from typing import Any
from uuid import uuid4

from sglang.srt.entrypoints.openai.serving_rerank import (
    _is_qwen3_reranker_template,
    _render_jinja_chat_template,
)
from sglang.srt.managers.io_struct import GenerateReqInput

from dynamo._core import Context
from dynamo.llm.exceptions import EngineShutdown
from dynamo.sglang.protocol import RerankRequest
from dynamo.sglang.request_handlers.llm.decode_handler import _raise_if_sglang_error
from dynamo.sglang.request_handlers.rerank.rerank_handler import RerankWorkerHandler


class DecoderRerankWorkerHandler(RerankWorkerHandler):
    """Score text query/document pairs with Qwen3's yes/no next-token logits."""

    @staticmethod
    def validate_engine(engine: Any) -> None:
        manager = engine.tokenizer_manager
        architectures = manager.model_config.hf_config.architectures or []
        template = manager.tokenizer.chat_template
        if "Qwen3ForCausalLM" not in architectures:
            raise ValueError(
                "Decoder reranking supports text Qwen3-Reranker models only"
            )
        if not manager.model_config.is_generation:
            raise ValueError("Decoder reranking requires a generation engine")
        if (
            not isinstance(template, str)
            or not _is_qwen3_reranker_template(template)
            or "<|vision_start|>" in template
            or "<|image_pad|>" in template
        ):
            raise ValueError(
                "Decoder reranking requires a Qwen3 reranker chat template"
            )
        query, document = f"query-{uuid4().hex}", f"document-{uuid4().hex}"
        rendered = _render_jinja_chat_template(
            template, query=query, document=document, instruct=None
        )
        if query not in rendered or document not in rendered:
            raise ValueError(
                "Reranker template must render both query and document; use SGLang's "
                "examples/chat_template/qwen3_reranker.jinja"
            )
        # Do not use upstream's fallback IDs: a tokenizer mismatch must fail at startup.
        label_ids = []
        for label in ("yes", "no"):
            encoded = manager.tokenizer.encode(label, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(f"Reranker label {label!r} must encode to one token")
            label_ids.append(encoded[0])
        if label_ids[0] == label_ids[1]:
            raise ValueError("Reranker yes/no labels must have distinct token IDs")

    async def _score_document(
        self, query: str, document: str, labels: list[int], context: Context
    ) -> dict[str, Any]:
        manager = self.engine.tokenizer_manager
        tokenizer = manager.tokenizer
        prompt = _render_jinja_chat_template(
            tokenizer.chat_template, query=query, document=document, instruct=None
        )
        request_id = f"rerank-{uuid4().hex}"
        request = GenerateReqInput(
            text=prompt,
            rid=request_id,
            sampling_params={"max_new_tokens": 0},
            return_logprob=True,
            token_ids_logprob=labels,
            logprob_start_len=-1,
            stream=False,
            external_trace_header=context.trace_headers()
            if self.enable_trace
            else None,
        )
        responses = manager.generate_request(request, None)
        inference = asyncio.create_task(anext(responses))
        cancelled = asyncio.ensure_future(context.async_killed_or_stopped())
        shutdown = (
            asyncio.create_task(self.shutdown_event.wait())
            if self.shutdown_event is not None
            else None
        )
        signals = [cancelled] if shutdown is None else [cancelled, shutdown]
        completed = False
        try:
            done, _ = await asyncio.wait(
                [inference, *signals], return_when=asyncio.FIRST_COMPLETED
            )
            if shutdown is not None and shutdown in done:
                raise EngineShutdown("Engine shut down during reranking")
            if cancelled in done:
                raise asyncio.CancelledError
            result = await inference
            completed = True
        finally:
            try:
                if not completed:
                    # Abort while the native request is still registered. Cancelling
                    # the waiter first removes that registration on older SGLang.
                    await self._abort_sglang_request(
                        manager, request_id, manager.rid_to_state, context.id()
                    )
            finally:
                for task in [inference, *signals]:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(inference, *signals, return_exceptions=True)
                await responses.aclose()

        meta = result["meta_info"]
        reason = meta.get("finish_reason")
        if isinstance(reason, dict) and reason.get("type") == "abort":
            _raise_if_sglang_error(reason)
            raise asyncio.CancelledError
        rows = meta.get("output_token_ids_logprobs")
        if not rows or not rows[0]:
            raise ValueError("SGLang returned no rerank label logprobs")
        logprobs = {entry[1]: entry[0] for entry in rows[0]}
        if any(label not in logprobs for label in labels):
            raise ValueError("SGLang omitted a rerank label logprob")
        yes, no = (float(logprobs[label]) for label in labels)
        if (
            any(math.isnan(value) or value == math.inf for value in (yes, no))
            or max(yes, no) == -math.inf
        ):
            raise ValueError("SGLang returned invalid rerank label logprobs")
        # Equivalent to p(yes)/(p(yes)+p(no)), without probability underflow.
        maximum = max(yes, no)
        weight = math.exp(yes - maximum)
        score = weight / (weight + math.exp(no - maximum))
        return {"embedding": [score], "meta_info": meta}

    async def _run_pairs(
        self, rerank_request: RerankRequest, context: Context
    ) -> list[dict[str, Any]]:
        tokenizer = self.engine.tokenizer_manager.tokenizer
        labels = [
            tokenizer.encode(label, add_special_tokens=False)[0]
            for label in ("yes", "no")
        ]
        tasks = [
            asyncio.create_task(
                self._score_document(rerank_request.query, doc, labels, context)
            )
            for doc in rerank_request.documents
        ]
        try:
            return await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
