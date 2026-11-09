# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional tokenizer-only matrix; all artifacts must be explicitly staged."""
import json
import os

import pytest

from dynamo.frontend import toktier_session as adapter
from dynamo.frontend.delta_tokenizer import DeltaConfig

pytestmark = [pytest.mark.post_merge, pytest.mark.integration, pytest.mark.gpu_0]

_MATRIX = json.loads(os.environ.get("DYNAMO_DELTA_TOKENIZER_MATRIX", "[]"))


@pytest.mark.parametrize("case", _MATRIX, ids=lambda case: case["name"])
def test_staged_family_matrix(case):
    pytest.importorskip("toktier")
    transformers = pytest.importorskip("transformers")
    reference = transformers.AutoTokenizer.from_pretrained(
        case["directory"], local_files_only=True
    )
    manager = adapter.create_delta_tokenizer(
        reference, case["directory"], case["name"], DeltaConfig(verify_rate=1)
    )
    if not case["admitted"]:
        assert manager is None
        return
    assert manager is not None
    text = "A multi-turn conversation with code and punctuation.\n" * 2048
    with manager:
        for turn in range(20):
            text += (
                " Continue the conversation."
                if turn < 6
                else ["a", "\u0301", "🙂漢字", "  \n", "tool result: {}"][turn % 5]
            )
            actual = manager.tokenize(
                "session", text, scope="tenant", template_identity="v1"
            )
            assert actual == reference.encode(text, add_special_tokens=False)
            manager.wait_for_idle()
        assert manager.stats().get("incremental", 0) > 0
        assert manager.stats()["mismatches"] == 0
