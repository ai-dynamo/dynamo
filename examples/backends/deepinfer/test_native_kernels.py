# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optional real-kernel qualification of the image's DeepInfer compatibility patch."""

from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu_1, pytest.mark.post_merge]


@pytest.mark.timeout(120)
@pytest.mark.parametrize("width", [128, 16384, 32768])
def test_candidate_topk_short_exact_and_long_prefills(width):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    index_attn = pytest.importorskip(
        "deepinfer.model_executor.models.v4_modules.index_attn"
    )
    state = SimpleNamespace(
        candidate_topk_cfg=SimpleNamespace(topk_block=2048, block_size=8)
    )
    output = torch.full((2, 2048), -99, dtype=torch.int32, device="cuda")
    runtime = SimpleNamespace(block_candidate_idxs=output)
    generator = torch.Generator(device="cuda").manual_seed(42)
    scores = torch.randn(
        2, width, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    lengths = torch.tensor([width, width // 2], dtype=torch.int32, device="cuda")
    index_attn.MQA._topk_candidate_block(state, scores, lengths, 0, runtime)
    torch.cuda.synchronize()
    assert ((output >= 0) | (output == -1)).all()
    for row, length in enumerate(lengths.tolist()):
        selected = output[row][output[row] >= 0]
        available = length // 8
        count = min(2048, available)
        assert selected.numel() == count
        assert selected.unique().numel() == count
        assert (selected < available).all()
        block_scores = scores[row, :length].reshape(-1, 8).max(1).values
        # The native candidate policy always retains the last valid block.
        block_scores[-1] = torch.inf
        cutoff = block_scores.topk(count).values.min()
        assert (block_scores[selected.long()] >= cutoff).all()
