# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest


@pytest.fixture
def draft_checkpoint(tmp_path):
    """Reuse the local target geometry for a one-layer draft, without weights."""
    target = Path(__file__).parent / "e2e/configs/unified_cli/fixtures/tiny-model"
    config = json.loads((target / "config.json").read_text())
    config.update(
        num_hidden_layers=1,
        block_size=8,
        target_layer_ids=[0, 1, 2],
        dflash_config={"target_layer_ids": [0, 1, 2]},
        markov_rank=8,
    )
    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "config.json").write_text(json.dumps(config))
    return str(draft), config
