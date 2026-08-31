# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import heapq
from collections.abc import AsyncGenerator
from typing import Any

import sglang as sgl
from sglang.srt.managers.io_struct import EmbeddingReqInput

from dynamo._core import Context
from dynamo.sglang.protocol import RerankRequest


def _looks_like_decoder_reranker(tokenizer_manager: Any) -> bool:
    """Detect decoder-only Qwen3 rerankers that need SGLang's generation path."""
    tokenizer = getattr(tokenizer_manager, "tokenizer", None)
    chat_template = getattr(tokenizer, "chat_template", "")
    template = chat_template.lower() if isinstance(chat_template, str) else ""
    model_config = getattr(tokenizer_manager, "model_config", None)
    model_path = str(getattr(model_config, "model_path", "")).lower()
    has_yes_no_template = (
        "answer can only be" in template and "yes" in template and "no" in template
    )
    return (
        has_yes_no_template
        or "qwen3-reranker" in model_path
        or "qwen3-vl-reranker" in model_path
    )


class RerankWorkerHandler:
    """Runs stable text-only cross-encoder reranking through SGLang pooling."""

    def __init__(self, engine: sgl.Engine, *, enable_trace: bool) -> None:
        self.engine = engine
        self.enable_trace = enable_trace

    @staticmethod
    def _validate(request: RerankRequest) -> None:
        if not request.query.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        if not request.documents:
            raise ValueError("Documents cannot be empty")
        if any(not document.strip() for document in request.documents):
            raise ValueError("Each document cannot be empty or whitespace only")
        if request.top_n is not None and request.top_n < 1:
            raise ValueError("Parameter 'top_n' must be larger than 0")

    @staticmethod
    def _score(item: Any, index: int) -> float:
        if not isinstance(item, dict) or "embedding" not in item:
            raise ValueError(f"Missing embedding score for rerank at index {index}")
        score = item["embedding"]
        if isinstance(score, list):
            if not score or not isinstance(score[0], (int, float)):
                raise ValueError(
                    f"Invalid embedding score for rerank at index {index}: {score!r}"
                )
            score = score[0]
        if not isinstance(score, (int, float)):
            raise ValueError(
                f"Invalid embedding score for rerank at index {index}: {score!r}"
            )
        return float(score)

    async def generate(
        self, request: dict[str, Any], context: Context
    ) -> AsyncGenerator[list[dict[str, Any]], None]:
        rerank_request = RerankRequest(**request)
        self._validate(rerank_request)

        tokenizer_manager = self.engine.tokenizer_manager
        if _looks_like_decoder_reranker(tokenizer_manager):
            raise ValueError(
                "Dynamo's SGLang /v1/rerank integration currently supports "
                "cross-encoder rerankers only; decoder-only Qwen3 and Qwen3-VL "
                "rerankers are not supported"
            )

        pairs = [
            [rerank_request.query, document] for document in rerank_request.documents
        ]
        trace_id = context.trace_id
        request_ids = (
            [f"{trace_id}-{index}" for index in range(len(pairs))]
            if trace_id is not None
            else None
        )
        internal_request = EmbeddingReqInput(
            text=pairs,
            is_cross_encoder_request=True,
            external_trace_header=(
                context.trace_headers() if self.enable_trace else None
            ),
            rid=request_ids,
        )
        responses = tokenizer_manager.generate_request(internal_request, None)
        result = await anext(responses)
        if not isinstance(result, list):
            result = [result]
        if len(result) != len(rerank_request.documents):
            raise ValueError(
                "SGLang returned an unexpected number of rerank scores: "
                f"expected {len(rerank_request.documents)}, got {len(result)}"
            )

        ranked = []
        for index, item in enumerate(result):
            response = {"score": self._score(item, index), "index": index}
            if rerank_request.return_documents:
                response["document"] = rerank_request.documents[index]
            meta_info = item.get("meta_info") if isinstance(item, dict) else None
            if meta_info is not None:
                response["meta_info"] = meta_info
            ranked.append(response)

        if rerank_request.top_n is not None:
            ranked = heapq.nlargest(
                rerank_request.top_n, ranked, key=lambda item: item["score"]
            )
        else:
            ranked.sort(key=lambda item: item["score"], reverse=True)
        yield ranked
