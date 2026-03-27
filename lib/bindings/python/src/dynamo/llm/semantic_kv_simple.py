# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Simple reference implementation of SemanticKvCacheProvider.

Uses Jaccard similarity over sentence-segmented word trigrams (shingles)
for fast approximate matching. No GPU, no embedding model — just stdlib.
Note that this implementation should never be used in a production setting...
Jaccard similarity on shingles is a very rough heuristic and does not capture semantic meaning well.
It is also O(N) over donors, which can be slow for large counts.

So... just here as an example... Demonstrates the interface contract and serves as a
starting point for testing and prototyping. For production, I would use an
embedding-based provider with a vector index.

Concurrency: uses ``threading.Lock`` for safe access from multiple asyncio
tasks or threads. The O(N) donor scan is bounded by a time budget to avoid
blocking the event loop.

Usage::

    from dynamo.llm.semantic_kv_simple import SimpleSemanticProvider

    provider = SimpleSemanticProvider(min_similarity=0.6)
    await provider.register_donor("req-1", [1, 2, 3], "The quick brown fox...")
    match = await provider.find_semantic_match([1, 2, 4], "The quick brown dog...")
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import FrozenSet, Optional, Sequence, Set

from .semantic_kv import SemanticMatch

logger = logging.getLogger(__name__)

# Cap shingle count per donor to prevent memory blowup from very long prompts.
_MAX_SHINGLES_PER_DONOR = 2_000

# Maximum characters of prompt text to process (longer text is truncated).
_MAX_TEXT_CHARS = 50_000

# Time budget for the O(N) donor scan (seconds).
_SEARCH_BUDGET_SECS = 0.005


@dataclass
class _DonorEntry:
    """Internal donor record."""

    donor_id: str
    token_ids: list[int]
    shingles: FrozenSet[str]


class SimpleSemanticProvider:
    """Reference SemanticKvCacheProvider using Jaccard similarity.

    Stores donors in an OrderedDict with sentence-segmented word trigrams.
    Matching is O(N) scan over donors with a time budget. Suitable for
    small donor counts (< 10K).

    Args:
        min_similarity: Minimum Jaccard similarity (0.0–1.0) for a match.
        max_donors: Maximum number of donors to keep (LRU eviction).

    Raises:
        ValueError: If min_similarity is not in [0.0, 1.0] or max_donors < 1.
    """

    def __init__(
        self,
        min_similarity: float = 0.6,
        max_donors: int = 10_000,
    ) -> None:
        if not (0.0 <= min_similarity <= 1.0):
            raise ValueError(
                f"min_similarity must be in [0.0, 1.0], got {min_similarity}"
            )
        if max_donors < 1:
            raise ValueError(f"max_donors must be >= 1, got {max_donors}")

        self._min_similarity = min_similarity
        self._max_donors = max_donors
        self._donors: OrderedDict[str, _DonorEntry] = OrderedDict()
        self._lock = threading.Lock()

    async def find_semantic_match(
        self,
        token_ids: Sequence[int],
        prompt_text: Optional[str] = None,
    ) -> Optional[SemanticMatch]:
        """Find the most similar donor by Jaccard similarity over shingles."""
        if not prompt_text:
            return None

        query_shingles = self._text_to_shingles(prompt_text)
        if not query_shingles:
            return None

        best_entry: Optional[_DonorEntry] = None
        best_sim = 0.0
        deadline = time.monotonic() + _SEARCH_BUDGET_SECS

        with self._lock:
            if not self._donors:
                return None

            for entry in self._donors.values():
                sim = self._jaccard(query_shingles, entry.shingles)
                if sim > best_sim and sim >= self._min_similarity:
                    best_entry = entry
                    best_sim = sim
                if time.monotonic() > deadline:
                    logger.debug(
                        "Semantic search hit time budget; returning best so far"
                    )
                    break

        if best_entry is None:
            return None

        return SemanticMatch(
            donor_token_ids=list(best_entry.token_ids),
            similarity=best_sim,
            donor_id=best_entry.donor_id,
            metadata={"method": "jaccard_shingles"},
        )

    async def register_donor(
        self,
        donor_id: str,
        token_ids: Sequence[int],
        prompt_text: Optional[str] = None,
    ) -> None:
        """Register a donor with sentence-segmented word trigrams."""
        if not prompt_text:
            return

        shingles = self._text_to_shingles(prompt_text)
        if not shingles:
            return

        with self._lock:
            # Remove existing entry if re-registering (OrderedDict handles order)
            if donor_id in self._donors:
                del self._donors[donor_id]

            # Evict oldest if at capacity
            while len(self._donors) >= self._max_donors:
                self._donors.popitem(last=False)

            self._donors[donor_id] = _DonorEntry(
                donor_id=donor_id,
                token_ids=list(token_ids),
                shingles=shingles,
            )

    def on_eviction(self, donor_id: str) -> None:
        """Remove an evicted donor."""
        with self._lock:
            self._donors.pop(donor_id, None)

    @property
    def donor_count(self) -> int:
        """Number of registered donors."""
        return len(self._donors)

    @staticmethod
    def _text_to_shingles(text: str) -> FrozenSet[str]:
        """Convert text to a set of normalized 3-word shingles.

        Splits on sentence boundaries, normalizes whitespace and case,
        then creates 3-word shingles for robust partial matching.
        Caps output to ``_MAX_SHINGLES_PER_DONOR`` to bound memory.
        """
        # Truncate to prevent excessive processing
        text = text[:_MAX_TEXT_CHARS]

        # Normalize: collapse whitespace, lowercase
        normalized = " ".join(text.lower().split())

        # Split into sentences (simple heuristic)
        for sep in (".", "!", "?", "\n"):
            normalized = normalized.replace(sep, "|")

        sentences = [s.strip() for s in normalized.split("|") if len(s.strip()) > 10]

        # Create 3-word shingles from each sentence
        shingles: Set[str] = set()
        for sentence in sentences:
            words = sentence.split()
            for i in range(len(words) - 2):
                shingles.add(" ".join(words[i : i + 3]))
                if len(shingles) >= _MAX_SHINGLES_PER_DONOR:
                    return frozenset(shingles)

        return frozenset(shingles)

    @staticmethod
    def _jaccard(a: FrozenSet[str], b: FrozenSet[str]) -> float:
        """Jaccard similarity between two sets."""
        if not a or not b:
            return 0.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union > 0 else 0.0
