# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in tokenizer-only fixtures; collection never downloads model artifacts."""
import os

import pytest


@pytest.fixture
def delta_staged_tokenizer():
    pytest.importorskip("toktier")
    transformers = pytest.importorskip("transformers")
    directory = os.environ.get("DYNAMO_DELTA_TOKENIZER_MODEL_DIR")
    if not directory:
        pytest.skip("requires explicitly staged tokenizer artifacts")
    return (
        transformers.AutoTokenizer.from_pretrained(directory, local_files_only=True),
        directory,
    )
