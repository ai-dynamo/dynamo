# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The load-diff descriptor: who wrote it, and how it is stored in GMS.

The descriptor is JSON. It is stored zlib-compressed in GMS metadata chunks
``__gms__.load_diff/<i>`` that keep every RPC frame small, and the header
``__gms__.load_diff`` (schema, chunk count, size, digest) is written last, so
a reader never sees a partial descriptor.

A reader may only apply a descriptor written by the same build for the same
model and layout: ``identity`` and ``source_hashes`` fingerprint both, and
``check_match`` refuses any difference.

Anyone who can write GMS metadata can already change the weights, so the
descriptor's trust boundary is the GMS socket's permissions. It is JSON only:
reading it imports nothing and runs no code.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
import zlib
from pathlib import Path

import torch
from gpu_memory_service.client.torch.tensor import RESERVED_METADATA_PREFIX
from gpu_memory_service.integrations.common.utils import torch_device
from gpu_memory_service.integrations.vllm.load_diff.state import owners

# Bump on any change to the descriptor's layout or to how state.describe
# encodes values: writer and reader must agree on both.
SCHEMA = 2
KEY = RESERVED_METADATA_PREFIX + "load_diff"
_ENCODING = "zlib+json"
# The GMS protocol does not cap a frame; chunks keep each one small.
_CHUNK_BYTES = 256 * 1024
_MAX_DESCRIPTOR_BYTES = 256 << 20
_PACKAGES = ("torch", "vllm", "transformers", "flashinfer-python", "gpu-memory-service")
# Packages whose Python sources build and post-process the model. Models with
# remote code live under ``transformers_modules``.
_SOURCE_PACKAGES = ("vllm", "flashinfer", "gpu_memory_service", "transformers_modules")
# vLLM environment variables that choose kernels, weight layouts or numerics
# on the devices GMS supports (CUDA and XPU), as of vLLM 0.33. vLLM's own
# list, envs.compile_factors(), also hashes ports, timeouts and the DP rank,
# which legitimately differ between a writer and its readers, so it would
# refuse every attach.
_KERNEL_ENV = (
    "VLLM_BATCH_INVARIANT",
    "VLLM_USE_DEEP_GEMM",
    "VLLM_MOE_USE_DEEP_GEMM",
    "VLLM_USE_DEEP_GEMM_E8M0",
    "VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES",
    "VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER",
    "VLLM_USE_FLASHINFER_MOE_INT4",
    "VLLM_FLASHINFER_MOE_A2A_LOW_PRECISION_COMBINE",
    "VLLM_B12X_MOE_FP4_FORCE_A16",
    "VLLM_DEEPEPLL_NVFP4_DISPATCH",
    "VLLM_USE_FUSED_MOE_GROUPED_TOPK",
    "VLLM_ALLREDUCE_USE_FLASHINFER",
    "VLLM_FLASHINFER_ALLREDUCE_BACKEND",
    "VLLM_ALLREDUCE_USE_SYMM_MEM",
    "VLLM_ENABLE_GEMM_RS",
    "VLLM_DISABLED_KERNELS",
    "VLLM_MLA_DISABLE",
    "VLLM_KV_CACHE_LAYOUT",
    "VLLM_FLOAT32_MATMUL_PRECISION",
    "VLLM_MARLIN_USE_ATOMIC_ADD",
    "VLLM_MARLIN_INPUT_DTYPE",
    "VLLM_HUMMING_ONLINE_QUANT_CONFIG",
    "VLLM_HUMMING_INPUT_QUANT_CONFIG",
    "VLLM_USE_TRITON_AWQ",
    "VLLM_MXFP4_EMULATION_DEQUANT_AT_LOAD",
    "VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD",
    "VLLM_PP_LAYER_PARTITION",
    "VLLM_USE_OINK_OPS",
    "VLLM_REPLICATE_EMBED",
    "VLLM_STEP5_SHARD_SP_SHARED_EXPERT",
    "VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT",
    "VLLM_GDN_DECODE_KERNEL",
    "VLLM_XPU_FORCE_N_CONTIG_WEIGHT",
    "VLLM_XPU_INC_WNA16_BACKEND",
)


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def kernel_env() -> dict:
    """The kernel-selecting vLLM environment variables that are set.

    Raw strings, so this module does not import vLLM: a variable that is
    unset on one side and set to its default on the other is refused.
    """
    return {name: os.environ[name] for name in _KERNEL_ENV if name in os.environ}


def identity(vllm_config, model_config) -> dict:
    """What a reader must share with the writer for the diff to apply.

    Take it before the model is constructed: construction can change the HF
    config, and readers check theirs before constructing.
    """
    parallel = vllm_config.parallel_config
    config = json.dumps(model_config.hf_config.to_dict(), sort_keys=True, default=str)
    kernel = json.dumps(vars(vllm_config.kernel_config), sort_keys=True, default=str)
    attention = json.dumps(
        vars(vllm_config.attention_config), sort_keys=True, default=str
    )
    return dict(
        schema=SCHEMA,
        model=str(model_config.model),
        revision=model_config.revision,
        architectures=list(model_config.hf_config.architectures or []),
        config_sha256=hashlib.sha256(config.encode()).hexdigest(),
        versions={package: _version(package) for package in _PACKAGES},
        cuda=torch.version.cuda,
        device=torch_device().get_device_name(),
        dtype=str(model_config.dtype),
        quantization=model_config.quantization,
        kv_cache_dtype=vllm_config.cache_config.cache_dtype,
        max_model_len=model_config.max_model_len,
        max_num_batched_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
        tp=parallel.tensor_parallel_size,
        pp=parallel.pipeline_parallel_size,
        dp=parallel.data_parallel_size,
        enable_expert_parallel=parallel.enable_expert_parallel,
        expert_placement=parallel.expert_placement_strategy,
        enable_eplb=parallel.enable_eplb,
        # With the sizes above, these fix the TP, PP and EP ranks.
        rank=parallel.rank,
        dp_rank=parallel.data_parallel_rank,
        kernel_config=json.loads(kernel),
        attention_config=json.loads(attention),
        kernel_env=kernel_env(),
    )


def _defining_module(qualified: str) -> str:
    """The module of a ``module.qualname`` class name.

    A nested class has dots in its qualname too. Its module is found among
    the imported ones, so nothing is imported to tell the two apart.
    """
    name = qualified.rsplit(".", 1)[0]
    parent = name
    while parent not in sys.modules and "." in parent:
        parent = parent.rsplit(".", 1)[0]
    module = sys.modules.get(parent)
    if parent != name and module is not None:
        if name[len(parent) + 1 :].split(".")[0] in vars(module):
            return parent
    return name


def source_modules(model: torch.nn.Module, final: dict | None = None) -> set[str]:
    """Python modules that define the model's modules, quant methods and
    attention impls, and the classes of objects in ``final``."""
    names = {
        type(owner).__module__
        for _, module in model.named_modules()
        for _, owner in owners(module)
    }
    for row in (final or {}).values():
        names.update(_defining_module(d["o"]) for d in row.values() if "o" in d)
    return {name for name in names if name.split(".")[0] in _SOURCE_PACKAGES}


def source_hashes(names) -> dict[str, str]:
    """sha256 of each named module's source file. Names outside the source
    packages are not looked up, so they never match."""
    hashes = {}
    for name in sorted(names):
        if name.split(".")[0] not in _SOURCE_PACKAGES:
            continue
        try:
            # Imports parent packages only, all from the source packages.
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            continue
        if spec is not None and spec.origin and spec.origin.endswith(".py"):
            hashes[name] = hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
    return hashes


def check_match(writer: dict, reader: dict, what: str) -> None:
    """Raise, naming the differing keys with both values, unless equal."""
    if writer == reader:
        return
    keys = sorted(
        k for k in set(writer) | set(reader) if writer.get(k) != reader.get(k)
    )
    detail = {k: (writer.get(k), reader.get(k)) for k in keys[:10]}
    raise RuntimeError(f"Load-diff {what} mismatch (writer, reader): {detail}")


def put(manager, allocation_id: str, descriptor: dict) -> int:
    """Store ``descriptor`` on ``allocation_id``; returns the compressed size.

    Must run before the layout is committed, once per layout: the chunks of
    an earlier, larger descriptor are not removed. GMS drops an allocation's
    metadata with the allocation, so ``allocation_id`` must outlive the write.
    """
    body = zlib.compress(json.dumps(descriptor, sort_keys=True).encode(), 6)
    chunks = [body[i : i + _CHUNK_BYTES] for i in range(0, len(body), _CHUNK_BYTES)]
    for index, chunk in enumerate(chunks):
        manager.metadata_put(f"{KEY}/{index}", allocation_id, 0, chunk)
    header = dict(
        schema=SCHEMA,
        encoding=_ENCODING,
        chunks=len(chunks),
        bytes=len(body),
        sha256=hashlib.sha256(body).hexdigest(),
    )
    manager.metadata_put(KEY, allocation_id, 0, json.dumps(header).encode())
    return len(body)


def get(manager) -> dict:
    """Fetch and verify the descriptor; raises if it is absent or invalid."""
    got = manager.metadata_get(KEY)
    if got is None:
        raise RuntimeError(
            "GMS has no load-diff descriptor: the writer did not run with "
            "gms_reconstruction_mode=load_diff"
        )
    try:
        header = json.loads(got[2])
    except ValueError as exc:
        raise RuntimeError("Load-diff descriptor header is not JSON") from exc
    schema = header.get("schema") if isinstance(header, dict) else None
    if schema != SCHEMA:
        raise RuntimeError(
            f"Load-diff descriptor schema {schema!r}; this reader supports {SCHEMA}"
        )
    if header.get("encoding") != _ENCODING:
        raise RuntimeError(
            f"Unknown load-diff descriptor encoding {header.get('encoding')!r}"
        )
    count, size = header.get("chunks"), header.get("bytes")
    if type(count) is not int or type(size) is not int:
        raise RuntimeError("Load-diff descriptor header is incomplete")
    chunks = []
    for index in range(count):
        chunk = manager.metadata_get(f"{KEY}/{index}")
        if chunk is None:
            raise RuntimeError(f"Load-diff descriptor chunk {index} is missing")
        chunks.append(chunk[2])
    body = b"".join(chunks)
    if len(body) != size or hashlib.sha256(body).hexdigest() != header.get("sha256"):
        raise RuntimeError("Load-diff descriptor is corrupt (size or digest)")
    inflate = zlib.decompressobj()
    data = inflate.decompress(body, _MAX_DESCRIPTOR_BYTES)
    if inflate.unconsumed_tail or not inflate.eof:
        raise RuntimeError("Load-diff descriptor is truncated or too large")
    return json.loads(data)
