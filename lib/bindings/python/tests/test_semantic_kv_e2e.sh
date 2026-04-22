#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end smoke test for the semantic KV cache interface.
# Requires compiled dynamo bindings (run in CI or Linux environment).
#
# Usage: bash lib/bindings/python/tests/test_semantic_kv_e2e.sh
#
# Validates:
# 1. Protocol imports work through the real dynamo.llm package
# 2. RadixTree integration test passes (semantic → RadixTree handoff)
# 3. SimpleSemanticProvider works with real RadixTree overlap scoring

set -euo pipefail

echo "=== Semantic KV Cache E2E Smoke Test ==="

# 1. Verify imports through real dynamo package
python3 -c "
from dynamo.llm import SemanticKvCacheProvider, SemanticMatch
from dynamo.llm.semantic_kv_simple import SimpleSemanticProvider

provider = SimpleSemanticProvider()
assert isinstance(provider, SemanticKvCacheProvider), 'Protocol check failed'
print('OK: Protocol imports and compliance check passed')
"

# 2. Run full test suite including RadixTree integration
python3 -m pytest lib/bindings/python/tests/test_semantic_kv.py -v \
    --noconftest --override-ini="addopts=" --override-ini="filterwarnings="

# 3. Standalone integration: register donor → semantic match → RadixTree lookup
python3 -c "
import asyncio
import json
from dynamo.llm import RadixTree, compute_block_hash_for_seq
from dynamo.llm.semantic_kv_simple import SimpleSemanticProvider
from dynamo.llm.semantic_kv import SemanticMatch

async def main():
    provider = SimpleSemanticProvider(min_similarity=0.3)

    # Register a donor
    donor_tokens = list(range(128))
    donor_text = (
        'Quantum computing leverages quantum mechanical phenomena like superposition. '
        'Qubits can represent multiple states simultaneously unlike classical bits. '
        'Entanglement enables correlated measurements across vast distances instantly. '
        'Error correction is essential for practical quantum computation at scale.'
    )
    await provider.register_donor('donor-1', donor_tokens, donor_text)

    # Store donor KV blocks in RadixTree
    tree = RadixTree()
    block_hashes = compute_block_hash_for_seq(donor_tokens, 32)
    parent = None
    for i, bh in enumerate(block_hashes):
        event = json.dumps({
            'event_id': i + 1,
            'data': {'stored': {'parent_hash': parent, 'blocks': [{'block_hash': bh, 'tokens_hash': bh}]}},
        }).encode()
        tree.apply_event(0, event)
        parent = bh

    # Query with similar text (same document, different question)
    query_text = (
        'Quantum computing leverages quantum mechanical phenomena like superposition. '
        'Qubits can represent multiple states simultaneously unlike classical bits. '
        'Entanglement enables correlated measurements across vast distances instantly. '
        'How does quantum error correction improve computation reliability?'
    )
    match = await provider.find_semantic_match([200, 201, 202], query_text)
    assert match is not None, 'Semantic match should find the donor'
    assert match.donor_id == 'donor-1'
    print(f'Semantic match: donor={match.donor_id}, sim={match.similarity:.3f}')

    # Use donor tokens to query RadixTree
    donor_hashes = compute_block_hash_for_seq(match.donor_token_ids, 32)
    overlap = tree.find_matches(donor_hashes)
    scores = overlap.scores
    assert len(scores) > 0, 'RadixTree should find overlap'
    print(f'RadixTree overlap: {dict(scores)}')
    print('OK: Full semantic → RadixTree handoff works end-to-end')

asyncio.run(main())
"

echo ""
echo "=== ALL E2E TESTS PASSED ==="
