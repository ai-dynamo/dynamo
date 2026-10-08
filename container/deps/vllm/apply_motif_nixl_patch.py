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
"""Apply guarded NIXL fixes while building the pinned Motif image.

The model mixes MLA and GQA. Keep its global TP/replication policy unchanged,
validate the first backend's actual layout, and retain Motif's NHD layout.
Only homogeneous TP2 is supported by this recipe.
"""

import hashlib
import importlib.util
from pathlib import Path

PATCHES = [
    (
        "distributed/kv_transfer/kv_connector/utils.py",
        "a005d542161b71d824338a0fbbf329291ee580c5113787d8e080bb8cf6909a9b",
        """            if not self.is_mla:
                assert len(kv_cache_shape) == 4, (
                    "Attention KV cache layout must be standardized as "
                    "[num_blocks, num_kv_heads, block_size, content_size], "
                    f"got shape {kv_cache_shape}."
                )""",
        """            # Motif mixes MLA and GQA: model-wide is_mla is False.
            # Layout belongs to this backend; global replication stays unchanged.
            expected_rank = 3 if attn_backend.is_mla() else 4
            if len(kv_cache_shape) != expected_rank:
                raise ValueError(
                    f"Unexpected {attn_backend.get_name()} KV cache rank: "
                    f"expected {expected_rank}, got shape {kv_cache_shape}."
                )""",
    ),
    (
        "distributed/kv_transfer/kv_connector/v1/nixl/connector.py",
        "95b0a138309ff6de80440a5cf604acda6187ab3125ccaa2d04ccf745afb809e4",
        """        use_mla = vllm_config.model_config.use_mla
        if use_mla:""",
        """        if vllm_config.model_config.hf_config.model_type == "Motif":
            # The pinned Motif cute-dsl / DiffKV kernels use NHD in aggregate.
            # Equal TP transfers complete blocks and needs no HND head slicing.
            if vllm_config.parallel_config.tensor_parallel_size != 2:
                raise ValueError(
                    "This Motif NIXL compatibility patch is validated only for TP2"
                )
            logger.info_once("Motif NIXL compatibility: retaining NHD cache layout")
            return "NHD"
        use_mla = vllm_config.model_config.use_mla
        if use_mla:""",
    ),
]


def apply(root: Path) -> None:
    pending = []
    for relative, expected_hash, old, new in PATCHES:
        path = root / relative
        source = path.read_text()
        if source.count(new) == 1:
            # Idempotence is checked against the original pinned file too.
            original = source.replace(new, old, 1)
            if hashlib.sha256(original.encode()).hexdigest() != expected_hash:
                raise RuntimeError(f"Unrecognized patched runtime source: {path}")
            continue
        if hashlib.sha256(source.encode()).hexdigest() != expected_hash:
            raise RuntimeError(
                f"Unrecognized runtime source: {path}; refusing to patch"
            )
        if source.count(old) != 1:
            raise RuntimeError(f"Runtime patch target is not unique: {path}")
        patched = source.replace(old, new, 1)
        compile(patched, str(path), "exec")
        pending.append((path, patched))
    for path, patched in pending:
        path.write_text(patched)
        print(
            f"Motif NIXL patch {path}: "
            f"sha256={hashlib.sha256(patched.encode()).hexdigest()}",
            flush=True,
        )


if __name__ == "__main__":
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise RuntimeError("The pinned Motif vLLM runtime is required")
    apply(Path(spec.origin).parent)
