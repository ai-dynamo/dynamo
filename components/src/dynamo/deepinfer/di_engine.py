# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepInfer engine construction and the ``LLMEngine`` subclass with Dynamo's hooks.

This is the only module that reaches into DeepInfer. DeepInfer (or the CPU
fake in ``testing/fake_deepinfer.py``) is imported lazily by :func:`load_api`,
after the launcher has set the ``DEEPINFER_*`` environment, because DeepInfer
reads several of those variables at import time.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from typing import TYPE_CHECKING, Any, Optional

from dynamo.deepinfer.disagg import DecodeFlow, PrefillFlow
from dynamo.deepinfer.kv_cache.base import make_cache_manager
from dynamo.deepinfer.kv_transfer import base as kv_transfer
from dynamo.deepinfer.kv_transfer.base import KVLayout, KVRegion
from dynamo.deepinfer.protocol import LAYOUT_ENV, LOCKSTEP_ENV, EngineMeta

if TYPE_CHECKING:
    from dynamo.deepinfer.rank import RankSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeepInferAPI:
    LLMEngine: type
    EngineArgs: type
    EngineType: Any
    SamplingParams: type
    RequestOutput: Any
    real: bool
    # DeepInfer's settings as it resolved them from DEEPINFER_* (``deepinfer.environ.envs``).
    envs: Any


def load_api(backend: str) -> DeepInferAPI:
    """``"deepinfer"`` for the real engine, ``"fake"`` for the CPU fake."""
    if backend == "fake":
        fake = importlib.import_module("dynamo.deepinfer.testing.fake_deepinfer")
        return DeepInferAPI(
            fake.LLMEngine,
            fake.EngineArgs,
            fake.EngineType,
            fake.SamplingParams,
            fake.RequestOutput,
            real=False,
            envs=fake.envs,
        )
    if backend != "deepinfer":
        raise ValueError(f"unknown DeepInfer backend {backend!r}")
    return DeepInferAPI(
        importlib.import_module("deepinfer.engine.llm_engine").LLMEngine,
        importlib.import_module("deepinfer.engine.arg_utils").EngineArgs,
        importlib.import_module("deepinfer.engine.engine_type").EngineType,
        importlib.import_module("deepinfer.sampling_params").SamplingParams,
        importlib.import_module("deepinfer._C.engine").RequestOutput,
        real=True,
        envs=importlib.import_module("deepinfer.environ").envs,
    )


@functools.lru_cache(maxsize=None)
def make_engine_class(base: type) -> type:
    """Subclass DeepInfer's ``LLMEngine`` (or the fake) with Dynamo's hooks.

    The ``dynamo_*`` attributes are attached by :func:`build_engine` after the
    base constructor ran; hooks that fire during construction see ``None``.
    """

    class DynamoLLMEngine(base):  # type: ignore[valid-type, misc]
        dynamo_cache: Any = None
        dynamo_prefill: Any = None
        dynamo_decode: Any = None
        dynamo_transfer: Any = None

        def post_schedule(self, scheduler_outputs: Any) -> None:
            super().post_schedule(scheduler_outputs)
            if self.dynamo_cache is not None:
                self.dynamo_cache.on_post_schedule(scheduler_outputs, self)

        def finalize_seq(self, seq: Any) -> None:
            if self.dynamo_cache is None:
                super().finalize_seq(seq)
            else:
                self.dynamo_cache.on_finished(seq)

        def pe_do_save_kvcache(self, scheduled: list[Any], finished: list[Any]) -> None:
            super().pe_do_save_kvcache(scheduled, finished)
            if self.dynamo_prefill is not None:
                self.dynamo_prefill.on_kv_saved(scheduled, finished)

        def send_processed_prompts(self) -> None:
            super().send_processed_prompts()
            if self.dynamo_prefill is not None:
                self.dynamo_prefill.on_processed_prompts()

    DynamoLLMEngine.__qualname__ = f"DynamoLLMEngine[{base.__module__}]"
    return DynamoLLMEngine


ENGRAM_ALLOC_ATTEMPTS = 8
ENGRAM_ALLOC_RETRY_S = 15.0


def evict_page_cache(model_dir: str) -> tuple[int, int]:
    """Drop the checkpoint's clean pages from the page cache (``POSIX_FADV_DONTNEED``).

    Returns ``(files, bytes)`` advised. Needs no privileges; pages another
    process still maps or dirties stay cached.
    """
    files = nbytes = 0
    for root, _, names in os.walk(model_dir, followlinks=True):
        for name in names:
            if not name.endswith(".safetensors"):
                continue
            try:
                fd = os.open(os.path.join(root, name), os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                files += 1
                nbytes += os.fstat(fd).st_size
            except OSError:
                pass
            finally:
                os.close(fd)
    return files, nbytes


def evict_page_cache_before_engram(model_dir: str) -> None:
    """Evict the checkpoint's page cache right before DeepEP pins Engram storage.

    DeepEP allocates each rank's Engram table (91.6 GiB per rank at DEP2) as
    pinned ``cuMemCreate`` host memory on the GPU's NUMA node. That allocation
    fails with ``CUDA_ERROR_OUT_OF_MEMORY`` instead of reclaiming page cache, and
    loading a ~510 GB checkpoint just filled it: two DEP2 engines on one GB300
    node (4 x 91.6 GiB pinned) could not start. ``DYN_DEEPINFER_EVICT_CHECKPOINT_CACHE=0``
    turns this off.
    """
    if os.environ.get("DYN_DEEPINFER_EVICT_CHECKPOINT_CACHE", "1") in (
        "0",
        "false",
        "False",
    ):
        return
    try:
        engram = importlib.import_module(
            "deepinfer.model_executor.models.v4_modules.engram"
        )
    except ImportError:
        return
    runtime = engram.EngramRuntime
    allocate = runtime.allocate
    if getattr(allocate, "_dynamo_evicts_page_cache", False):
        return

    def allocate_after_eviction(*args: Any, **kwargs: Any) -> Any:
        # Other engines on the node may still be reading the checkpoint back into
        # the cache: on OOM, evict again and retry. cuMemCreate fails before
        # DeepEP builds any state, so a retry starts clean.
        for attempt in range(ENGRAM_ALLOC_ATTEMPTS):
            files, nbytes = evict_page_cache(model_dir)
            logger.info(
                "Evicted page cache of %d checkpoint files (%.1f GiB) before Engram allocation",
                files,
                nbytes / 2**30,
            )
            try:
                return allocate(*args, **kwargs)
            except RuntimeError as e:
                if (
                    "OUT_OF_MEMORY" not in str(e)
                    or attempt + 1 == ENGRAM_ALLOC_ATTEMPTS
                ):
                    raise
                logger.warning(
                    "Engram host allocation failed (attempt %d/%d): %s; retrying in %.0f s",
                    attempt + 1,
                    ENGRAM_ALLOC_ATTEMPTS,
                    e,
                    ENGRAM_ALLOC_RETRY_S,
                )
                time.sleep(ENGRAM_ALLOC_RETRY_S)
        raise AssertionError("unreachable")

    allocate_after_eviction._dynamo_evicts_page_cache = True  # type: ignore[attr-defined]
    runtime.allocate = staticmethod(allocate_after_eviction)


def build_engine(api: DeepInferAPI, spec: RankSpec) -> Any:
    """Build this rank's engine and attach its cache manager and P/D flow."""
    cls = make_engine_class(api.LLMEngine)
    engine_args = api.EngineArgs(
        **spec.engine_args, engine_type=api.EngineType[spec.engine_type]
    )
    if api.real:
        evict_page_cache_before_engram(str(spec.engine_args["model"]))
    engine = cls.from_engine_args(engine_args)
    engine.dynamo_cache = make_cache_manager(
        spec.cache_manager, engine, **spec.cache_options
    )
    transfer = kv_transfer.make_kv_transfer(
        spec.kv_transfer,
        rank=spec.rank,
        engine_id=spec.engine_id,
        **spec.transfer_options,
    )
    engine.dynamo_transfer = transfer
    layout = kv_layout(engine, api.envs)
    if transfer is not None:
        transfer.register(layout)
    if spec.engine_type == "PROMPT":
        engine.dynamo_prefill = PrefillFlow(
            engine,
            engine.dynamo_cache,
            transfer,
            dp_rank=spec.rank,
            engine_id=spec.engine_id,
            layout_hash=layout.layout_hash,
            lease_s=spec.kv_lease_s,
        )
    elif spec.engine_type == "COMPLETION":
        if transfer is None:
            raise ValueError("decode workers need a KV transfer (--kv-transfer)")
        engine.dynamo_decode = DecodeFlow(
            engine, transfer, layout_hash=layout.layout_hash
        )
    return engine


def engine_meta(engine: Any) -> EngineMeta:
    cc, sc = engine.cache_config, engine.scheduler_config
    bos = getattr(engine.tokenizer, "bos_token_id", None)
    return EngineMeta(
        block_size=int(cc.block_size),
        num_gpu_blocks=int(cc.num_gpu_blocks),
        max_num_seqs=int(sc.max_num_seqs),
        max_model_len=int(sc.max_model_len),
        max_num_batched_tokens=int(sc.max_num_batched_tokens),
        prefill_any_attempt_interval=int(sc.prefill_any_attempt_interval),
        bos_token_id=int(bos) if bos is not None else 0,
    )


def bind_thread(api: DeepInferAPI, engine: Any) -> None:
    """Make the calling thread drive the engine's GPU (CUDA's current device is
    per thread; rank 0 is built on the main thread and stepped on another)."""
    if api.real:
        torch = importlib.import_module("torch")
        torch.cuda.set_device(engine.driver_worker.device)


def has_local_work(engine: Any) -> bool:
    """DeepInfer's ``has_unfinished_requests`` without its all-reduce: this
    rank's scheduler holds a request, or a structured-output matcher is still
    being built for one (``llm_engine.py`` ``has_unfinished_requests``)."""
    return bool(
        engine.scheduler.has_unfinished_seqs() or engine.so_builder_has_pending()
    )


# (tag, values) -> elementwise MAX over every DP rank of the engine.
DPMax = Callable[[str, list[int]], list[int]]


def dp_max(api: DeepInferAPI, engine: Any, *, cpu: bool = False) -> DPMax:
    """An int64 MAX all-reduce over the engine's DP ranks.

    On the group DeepInfer's ``has_unfinished_requests`` uses: the gloo
    ``cpu_data_parallel_group`` with zero-overhead scheduling, else the NCCL
    DP group. ``cpu`` always takes the gloo group, which works whatever the
    ranks' zero-overhead settings are. Only the fake reads ``tag``: it fails
    ranks that reach different collectives.
    """
    if not api.real:
        return engine.all_reduce_max
    torch = importlib.import_module("torch")
    dist = importlib.import_module("torch.distributed")
    pg = importlib.import_module(
        "deepinfer.model_executor.parallel_utils.parallel_state"
    ).PG
    if cpu or api.envs.DEEPINFER_ZERO_OVERHEAD_SCHEDULING:
        group, device = pg.cpu_data_parallel_group(), "cpu"
    else:
        group, device = pg.data_parallel_group(), "cuda"
    if group.size() == 1:
        return lambda tag, values: list(values)
    op = dist.ReduceOp.MAX

    def all_reduce(tag: str, values: list[int]) -> list[int]:
        flags = torch.tensor(values, dtype=torch.int64, device=device)
        dist.all_reduce(flags, op=op, group=group)
        return flags.tolist()

    return all_reduce


def lockstep_settings(api: DeepInferAPI, engine_type: str) -> dict[str, Optional[str]]:
    """``LOCKSTEP_ENV`` as this rank's engine runs with it.

    DeepInfer sets ``DEEPINFER_ZERO_OVERHEAD_SCHEDULING`` from the engine type
    when it is unset (``ENGINE_DERIVED_ENV``); a rank that lacks it (the fake)
    reports that derived value, so it matches a rank whose DeepInfer set it.
    """
    env = {k: os.environ.get(k) for k in sorted(LOCKSTEP_ENV)}
    zos = "DEEPINFER_ZERO_OVERHEAD_SCHEDULING"
    if env[zos] is None:
        derived = api.EngineType[engine_type].default_zero_overhead_scheduling
        env[zos] = "1" if derived else "0"
    return env


def _region(name: str, buf: Any) -> KVRegion:
    if hasattr(buf, "data_ptr"):  # torch.Tensor
        return KVRegion(
            name=name,
            base_ptr=buf.data_ptr(),
            num_items=buf.shape[0],
            item_bytes=buf.stride(0) * buf.element_size(),
            device_id=buf.device.index if buf.device.type == "cuda" else -1,
            buffer=buf,
        )
    return KVRegion(  # numpy array (CPU fake)
        name=name,
        base_ptr=buf.ctypes.data,
        num_items=buf.shape[0],
        item_bytes=buf.strides[0],
        device_id=-1,
        buffer=buf,
    )


def kv_layout(engine: Any, envs: Any) -> KVLayout:
    """The two per-rank KV regions: block rows and per-request state slots.

    The hash covers ``LAYOUT_ENV`` as DeepInfer resolved it (``envs``), not
    the raw environment: a role that leaves a setting unset and one that
    spells out its default have the same layout.
    """
    cache_engine = engine.driver_worker.cache_engine
    regions = [_region("gpu_blocks", cache_engine.gpu_blocks)]
    if cache_engine.gpu_states is not None:
        regions.append(_region("gpu_states", cache_engine.gpu_states))
    try:
        version = metadata.version("deepinfer")
    except metadata.PackageNotFoundError:
        version = "unknown"
    key = {
        "model": os.path.basename(
            os.path.normpath(str(getattr(engine.model_config, "model", "")))
        ),
        "deepinfer": version,
        "block_size": int(engine.cache_config.block_size),
        "regions": [(r.name, r.item_bytes) for r in regions],
        "env": {k: getattr(envs, k) for k in LAYOUT_ENV},
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    return KVLayout(regions=regions, layout_hash=digest[:16])
