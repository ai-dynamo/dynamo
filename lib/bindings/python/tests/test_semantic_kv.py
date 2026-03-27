# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the semantic KV cache provider interface and reference implementation.

Run: pytest lib/bindings/python/tests/test_semantic_kv.py -v
"""

import os
import sys
import types

import pytest

# ── Bootstrap: import semantic_kv modules without compiled _core ──────
# dynamo.llm.__init__ unconditionally imports from dynamo._core (compiled
# Rust bindings). We create a minimal stub package so our pure-Python
# modules can be imported for testing without _core.
_src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# Stub out dynamo.llm as a minimal package (skip __init__.py)
_llm_pkg = types.ModuleType("dynamo.llm")
_llm_pkg.__path__ = [os.path.join(_src_dir, "dynamo", "llm")]
_llm_pkg.__package__ = "dynamo.llm"
sys.modules.setdefault("dynamo", types.ModuleType("dynamo"))
sys.modules["dynamo.llm"] = _llm_pkg

# Now normal imports work
from dynamo.llm.semantic_kv import SemanticKvCacheProvider, SemanticMatch  # noqa: E402
from dynamo.llm.semantic_kv_simple import SimpleSemanticProvider  # noqa: E402

pytestmark = pytest.mark.pre_merge


# ── Protocol compliance ─────────────────────────────────────────────


class TestProtocolCompliance:
    """Verify the Protocol contract is correctly defined."""

    def test_simple_provider_implements_protocol(self):
        """SimpleSemanticProvider must satisfy the runtime-checkable Protocol."""
        provider = SimpleSemanticProvider()
        assert isinstance(provider, SemanticKvCacheProvider)

    def test_minimal_implementation_satisfies_protocol(self):
        """A bare-bones class with the right signatures should satisfy the Protocol."""

        class MinimalProvider:
            async def find_semantic_match(self, token_ids, prompt_text=None):
                return None

            async def register_donor(self, donor_id, token_ids, prompt_text=None):
                pass

            def on_eviction(self, donor_id):
                pass

        assert isinstance(MinimalProvider(), SemanticKvCacheProvider)

    def test_incomplete_implementation_fails_protocol(self):
        """A class missing methods should NOT satisfy the Protocol."""

        class IncompleteProvider:
            async def find_semantic_match(self, token_ids, prompt_text=None):
                return None

            # Missing register_donor and on_eviction

        assert not isinstance(IncompleteProvider(), SemanticKvCacheProvider)


# ── SemanticMatch dataclass ──────────────────────────────────────────


class TestSemanticMatch:
    """Verify SemanticMatch is immutable and well-formed."""

    def test_creation(self):
        match = SemanticMatch(
            donor_token_ids=[1, 2, 3],
            similarity=0.95,
            donor_id="req-123",
        )
        assert match.donor_token_ids == [1, 2, 3]
        assert match.similarity == 0.95
        assert match.donor_id == "req-123"
        assert match.metadata is None

    def test_with_metadata(self):
        match = SemanticMatch(
            donor_token_ids=[1],
            similarity=0.8,
            donor_id="req-456",
            metadata={"model": "minilm", "latency_ms": 2.3},
        )
        assert match.metadata["model"] == "minilm"

    def test_frozen(self):
        match = SemanticMatch(donor_token_ids=[1], similarity=0.9, donor_id="x")
        with pytest.raises(AttributeError):
            match.similarity = 0.5  # type: ignore[misc]


# ── SimpleSemanticProvider unit tests ────────────────────────────────


class TestSimpleSemanticProvider:
    """Test the reference implementation."""

    @pytest.fixture
    def provider(self):
        return SimpleSemanticProvider(min_similarity=0.5)

    @pytest.mark.asyncio
    async def test_empty_store_returns_none(self, provider):
        """No donors registered → always None."""
        result = await provider.find_semantic_match(
            [1, 2, 3], "The quick brown fox jumps over the lazy dog."
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_no_prompt_text_returns_none(self, provider):
        """Without prompt text, matching is impossible."""
        await provider.register_donor("d1", [1, 2, 3], "Some text here for testing.")
        result = await provider.find_semantic_match([1, 2, 3])
        assert result is None

    @pytest.mark.asyncio
    async def test_register_and_match_similar(self, provider):
        """Register a donor, query with similar text → match."""
        document = (
            "Quantum computing uses qubits to perform calculations. "
            "Unlike classical bits, qubits can exist in superposition states. "
            "This enables quantum computers to solve certain problems exponentially faster."
        )
        await provider.register_donor("donor-1", [10, 20, 30], document)

        # Same document, different question appended
        query = (
            "Quantum computing uses qubits to perform calculations. "
            "Unlike classical bits, qubits can exist in superposition states. "
            "This enables quantum computers to solve certain problems exponentially faster. "
            "What are the main advantages of quantum computing?"
        )
        result = await provider.find_semantic_match([10, 20, 99], query)

        assert result is not None
        assert result.donor_id == "donor-1"
        assert result.donor_token_ids == [10, 20, 30]
        assert result.similarity > 0.5

    @pytest.mark.asyncio
    async def test_unrelated_text_no_match(self, provider):
        """Completely different text → no match."""
        await provider.register_donor(
            "donor-1",
            [1, 2, 3],
            "Quantum computing uses qubits to perform calculations. "
            "This enables solving problems exponentially faster than classical computers.",
        )

        result = await provider.find_semantic_match(
            [99, 98, 97],
            "The recipe calls for two cups of flour and one tablespoon of baking powder. "
            "Mix the dry ingredients together before adding the wet ingredients slowly.",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_eviction_removes_donor(self, provider):
        """Evicted donor should no longer be findable."""
        text = (
            "Neural networks consist of layers of interconnected nodes. "
            "Each node applies a nonlinear activation function to its weighted inputs."
        )
        await provider.register_donor("donor-1", [1, 2, 3], text)

        # Verify it's findable
        result = await provider.find_semantic_match([1, 2, 4], text)
        assert result is not None

        # Evict and verify it's gone
        provider.on_eviction("donor-1")
        result = await provider.find_semantic_match([1, 2, 4], text)
        assert result is None

    @pytest.mark.asyncio
    async def test_multiple_donors_best_match(self, provider):
        """With multiple donors, the most similar one should be returned."""
        await provider.register_donor(
            "cooking",
            [1, 2, 3],
            "The recipe requires flour, sugar, and butter to make a delicious cake. "
            "Mix the dry ingredients first, then add wet ingredients gradually. "
            "Bake at 350 degrees for 30 minutes until golden brown on top. "
            "Let the cake cool completely before applying the frosting layer.",
        )
        await provider.register_donor(
            "physics",
            [4, 5, 6],
            "Quantum entanglement allows particles to be correlated across vast distances. "
            "Measuring one particle instantly determines the state of its entangled partner. "
            "This phenomenon was famously described by Einstein as spooky action at a distance. "
            "Modern experiments have confirmed entanglement over thousands of kilometers.",
        )

        result = await provider.find_semantic_match(
            [10, 11, 12],
            "Quantum entanglement allows particles to be correlated across vast distances. "
            "This phenomenon was famously described by Einstein as spooky action at a distance. "
            "Modern experiments have confirmed entanglement over thousands of kilometers. "
            "What are the practical applications of quantum entanglement in computing?",
        )

        assert result is not None
        # Should match physics donor, not cooking
        assert result.donor_id == "physics"

    @pytest.mark.asyncio
    async def test_min_similarity_threshold(self):
        """Matches below the threshold should be filtered out."""
        provider = SimpleSemanticProvider(min_similarity=0.99)
        await provider.register_donor(
            "d1",
            [1, 2, 3],
            "Some moderately long text about artificial intelligence and machine learning.",
        )

        result = await provider.find_semantic_match(
            [4, 5, 6],
            "Some different moderately long text about deep learning and neural networks.",
        )
        # With 0.99 threshold, partial overlap should be rejected
        assert result is None

    @pytest.mark.asyncio
    async def test_max_donors_eviction(self):
        """Exceeding max_donors should evict the oldest entry."""
        provider = SimpleSemanticProvider(min_similarity=0.3, max_donors=3)

        for i in range(5):
            await provider.register_donor(
                f"donor-{i}",
                [i],
                f"Document number {i} has unique content about topic {i} specifically. "
                f"It discusses topic {i} in great detail with many examples and references.",
            )

        assert provider.donor_count == 3
        # Oldest donors (0, 1) should have been evicted
        assert "donor-0" not in provider._donors
        assert "donor-1" not in provider._donors
        # Newest donors should remain
        assert "donor-4" in provider._donors

    @pytest.mark.asyncio
    async def test_donor_count_property(self, provider):
        """donor_count should track registered donors."""
        assert provider.donor_count == 0
        await provider.register_donor("d1", [1], "Some text for testing the count.")
        assert provider.donor_count == 1
        await provider.register_donor("d2", [2], "More text for testing the count.")
        assert provider.donor_count == 2
        provider.on_eviction("d1")
        assert provider.donor_count == 1

    @pytest.mark.asyncio
    async def test_register_without_text_is_noop(self, provider):
        """Registering without prompt_text should be silently ignored."""
        await provider.register_donor("d1", [1, 2, 3])
        assert provider.donor_count == 0

    @pytest.mark.asyncio
    async def test_evict_nonexistent_is_safe(self, provider):
        """Evicting a non-existent donor should not raise."""
        provider.on_eviction("nonexistent")  # Should not raise


# ── Integration test: SemanticMatch → RadixTree handoff ──────────────


class TestRadixTreeIntegration:
    """Test that semantic matches can be used with RadixTree overlap scoring.

    This verifies the core handoff: semantic provider returns donor_token_ids,
    which are then used to query the RadixTree for overlap scores.
    """

    @pytest.mark.asyncio
    async def test_semantic_to_radixtree_handoff(self):
        """Full flow: register donor → semantic match → RadixTree lookup."""
        try:
            from dynamo.llm import RadixTree, compute_block_hash_for_seq
        except ImportError:
            pytest.skip("dynamo._core not available (requires compiled bindings)")

        provider = SimpleSemanticProvider(min_similarity=0.3)

        # Simulate a donor request with specific token IDs
        donor_tokens = list(range(64))  # 64 tokens = 2 blocks of 32
        donor_text = (
            "Quantum computing leverages quantum mechanical phenomena. "
            "Superposition allows qubits to represent multiple states simultaneously. "
            "Entanglement enables correlated measurements across distances."
        )

        # 1. Register the donor
        await provider.register_donor("donor-1", donor_tokens, donor_text)

        # 2. Store the donor's KV blocks in a RadixTree
        tree = RadixTree()
        block_hashes = compute_block_hash_for_seq(donor_tokens, 32)

        import json

        # Store block 0
        event_0 = json.dumps(
            {
                "event_id": 1,
                "data": {
                    "stored": {
                        "parent_hash": None,
                        "blocks": [
                            {
                                "block_hash": block_hashes[0],
                                "tokens_hash": block_hashes[0],
                            }
                        ],
                    }
                },
            }
        ).encode()
        tree.apply_event(0, event_0)  # worker_id=0

        # Store block 1
        event_1 = json.dumps(
            {
                "event_id": 2,
                "data": {
                    "stored": {
                        "parent_hash": block_hashes[0],
                        "blocks": [
                            {
                                "block_hash": block_hashes[1],
                                "tokens_hash": block_hashes[1],
                            }
                        ],
                    }
                },
            }
        ).encode()
        tree.apply_event(0, event_1)

        # 3. Simulate a new request with the same document but different question
        query_text = (
            "Quantum computing leverages quantum mechanical phenomena. "
            "Superposition allows qubits to represent multiple states simultaneously. "
            "Entanglement enables correlated measurements across distances. "
            "How does quantum error correction work?"
        )

        match = await provider.find_semantic_match([50, 51, 52], query_text)
        assert match is not None
        assert match.donor_id == "donor-1"

        # 4. Use the donor's token IDs to query the RadixTree
        donor_block_hashes = compute_block_hash_for_seq(match.donor_token_ids, 32)
        overlap = tree.find_matches(donor_block_hashes)
        scores = overlap.scores

        # Worker 0 should have a non-zero score (it has the donor's KV blocks)
        assert len(scores) > 0, "RadixTree should find overlap for donor tokens"
        assert any(
            score > 0 for score in scores.values()
        ), "At least one worker should have overlapping KV blocks"
