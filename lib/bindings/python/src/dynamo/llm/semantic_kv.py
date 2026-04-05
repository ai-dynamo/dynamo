# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Semantic KV cache lookup interface for Dynamo.

When RadixTree exact-prefix matching misses (e.g., same document but different
instruction), a semantic provider finds cached prompts with similar content.
The router then queries the RadixTree with the donor's tokens to locate the
worker holding the reusable KV blocks.

This module defines the provider-agnostic interface. Implementations may use
embeddings, learned routers, or any other similarity signal.

Example usage with the standalone router::

    from dynamo.llm import SemanticKvCacheProvider, SemanticMatch

    class MyProvider:
        async def find_semantic_match(self, token_ids, prompt_text=None):
            # Your similarity search logic here
            return SemanticMatch(donor_token_ids=[...], similarity=0.95, donor_id="req-123")

        async def register_donor(self, donor_id, token_ids, prompt_text=None):
            # Store embedding for future lookups
            pass

        def on_eviction(self, donor_id):
            pass
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

__all__ = ["SemanticMatch", "SemanticKvCacheProvider"]


@dataclass(frozen=True)
class SemanticMatch:
    """Result from a semantic KV cache lookup.

    Returned by :meth:`SemanticKvCacheProvider.find_semantic_match` when a
    similar cached prompt is found.

    Attributes:
        donor_token_ids: Token IDs of the matched donor prompt. The router
            uses these to query the RadixTree for overlap scores.
        similarity: Similarity score between query and donor (0.0–1.0).
            Interpretation depends on the provider (cosine, Jaccard, etc.).
        donor_id: Opaque identifier for the donor request. Used for
            eviction callbacks and debugging.
        metadata: Optional provider-specific metadata (e.g., embedding model,
            match latency, donor age). Not used by the router.
    """

    donor_token_ids: Sequence[int]
    similarity: float
    donor_id: str
    metadata: Optional[Mapping[str, Any]] = field(default=None)

    def __post_init__(self) -> None:
        if not (0.0 <= self.similarity <= 1.0):
            raise ValueError(f"similarity must be in [0.0, 1.0], got {self.similarity}")


@runtime_checkable
class SemanticKvCacheProvider(Protocol):
    """Provider for semantic KV cache lookup.

    When the RadixTree exact-prefix match returns low overlap for a request,
    the router calls :meth:`find_semantic_match` to search for cached prompts
    with semantically similar content. If a match is found, the router queries
    the RadixTree with the donor's ``token_ids`` to get overlap scores and
    route the request to the worker holding those KV blocks.

    After each request completes prefill, the router calls
    :meth:`register_donor` so the provider can index the request for future
    lookups.

    Implementations are external to Dynamo — this interface is intentionally
    provider-agnostic. No assumptions about embedding models, vector stores,
    or similarity metrics.

    Thread safety: all methods may be called concurrently from multiple
    asyncio tasks. Implementations must handle their own synchronization.
    """

    async def find_semantic_match(
        self,
        token_ids: Sequence[int],
        prompt_text: Optional[str] = None,
    ) -> Optional[SemanticMatch]:
        """Find a semantically similar cached prompt.

        Args:
            token_ids: Token IDs of the current request.
            prompt_text: Optional decoded prompt text. Passing text avoids
                the provider needing to decode tokens, but is not required.

        Returns:
            A :class:`SemanticMatch` if a similar donor is found, or ``None``
            if no match exceeds the provider's similarity threshold.

        Note:
            Should complete quickly (< 10ms) to avoid adding latency to the
            request critical path. Providers with expensive lookups should
            use background indexing and return cached results.
        """
        ...

    async def register_donor(
        self,
        donor_id: str,
        token_ids: Sequence[int],
        prompt_text: Optional[str] = None,
    ) -> None:
        """Register a completed request as a potential donor.

        Called after a request completes prefill. The provider stores whatever
        representation it needs (embedding, token hashes, etc.) for future
        :meth:`find_semantic_match` lookups.

        Args:
            donor_id: Unique identifier for this request (e.g., request UUID).
            token_ids: The request's token IDs.
            prompt_text: Optional decoded prompt text.
        """
        ...

    def on_eviction(self, donor_id: str) -> None:
        """Notify the provider that a donor's KV blocks were evicted.

        Called when the RadixTree evicts blocks for a previously registered
        donor. The provider should remove or invalidate the entry so it is
        no longer returned by :meth:`find_semantic_match`.

        This method is synchronous because eviction notifications are
        fire-and-forget — the router does not wait for the provider to
        finish cleanup.

        Args:
            donor_id: Identifier previously passed to :meth:`register_donor`.
        """
        ...
