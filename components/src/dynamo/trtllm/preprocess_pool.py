# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run TensorRT-LLM multimodal input processing in spawned processes, each with
its own GIL, so that it does not compete with the worker's event loop.

Pool processes import this module before they set ``TRT_LLM_NO_LIB_INIT``, so it
imports only the standard library at module level.
"""

import concurrent.futures
import dataclasses
import functools
import logging
import multiprocessing.context
import multiprocessing.spawn
import multiprocessing.synchronize
import os
import pickle
import signal
from collections.abc import Callable
from types import ModuleType, SimpleNamespace
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Intra-op threads per process, so that the processes do not oversubscribe the CPUs.
TORCH_THREADS_PER_PROCESS = 4
STARTUP_TIMEOUT_S = 600.0

# A processor returned by create_input_processor_with_hash.
Processor = Callable[[Any, Any], Any]

# The processor and ready barrier, set in each pool process by _init_process.
_state = SimpleNamespace()


@dataclasses.dataclass
class InputProcessorSpec:
    """The arguments an LLM built its hashing input processor from."""

    model_dir: str
    # Pickled so that pool processes unpickle it after TRT_LLM_NO_LIB_INIT is set.
    tokenizer: bytes
    checkpoint_format: Optional[str]
    trust_remote_code: bool
    kwargs: dict[str, Any]
    encoder_cache_enabled: bool

    @classmethod
    def from_llm(cls, llm: Any) -> "InputProcessorSpec":
        """Mirror how TensorRT-LLM's LLM creates and hashes its input processor."""
        multimodal_config = llm.args.multimodal_config
        kwargs: dict[str, Any] = {
            "enable_tokenization_cache": llm.args.enable_tokenization_cache
        }
        if multimodal_config.video_pruning_rate is not None:
            kwargs["video_pruning_rate"] = multimodal_config.video_pruning_rate
        return cls(
            # The local checkpoint directory the LLM resolved its model to.
            model_dir=str(llm._hf_model_dir),
            tokenizer=pickle.dumps(llm.tokenizer),
            checkpoint_format=llm.args.checkpoint_format,
            trust_remote_code=llm.args.trust_remote_code,
            kwargs=kwargs,
            encoder_cache_enabled=multimodal_config.encoder_cache_max_bytes > 0,
        )


def build_input_processor(spec: InputProcessorSpec) -> Processor:
    """Build the LLM's hashing input processor in a pool process."""
    # Skip TensorRT-LLM's CUDA library initialization. It runs when tensorrt_llm
    # is first imported, so these imports must follow the assignment.
    os.environ["TRT_LLM_NO_LIB_INIT"] = "1"
    import torch
    from tensorrt_llm.inputs import (
        create_input_processor,
        create_input_processor_with_hash,
    )

    torch.set_num_threads(TORCH_THREADS_PER_PROCESS)
    # Return tensors through named shared-memory files. The default strategy
    # passes a file descriptor per tensor over a socket, and that handshake
    # stalls whenever the worker's event loop holds the GIL.
    torch.multiprocessing.set_sharing_strategy("file_system")
    processor = create_input_processor(
        spec.model_dir,
        pickle.loads(spec.tokenizer),
        spec.checkpoint_format,
        trust_remote_code=spec.trust_remote_code,
        **spec.kwargs,
    )
    return create_input_processor_with_hash(
        processor, encoder_cache_enabled=spec.encoder_cache_enabled
    )


class _SpawnProcess(multiprocessing.context.SpawnProcess):
    """A spawned process that does not re-run the parent's main module, which
    may import TensorRT-LLM or start serving."""

    @staticmethod
    def _Popen(process_obj):
        get_preparation_data = multiprocessing.spawn.get_preparation_data

        def without_main_module(name):
            data = get_preparation_data(name)
            data.pop("init_main_from_name", None)
            data.pop("init_main_from_path", None)
            return data

        # Popen reads the preparation data while it starts the child.
        multiprocessing.spawn.get_preparation_data = without_main_module
        try:
            return multiprocessing.context.SpawnProcess._Popen(process_obj)
        finally:
            multiprocessing.spawn.get_preparation_data = get_preparation_data


class _SpawnContext(multiprocessing.context.SpawnContext):
    Process = _SpawnProcess


def _init_process(
    build: Callable[[], Processor], ready_barrier: multiprocessing.synchronize.Barrier
) -> None:
    # The worker process handles interrupts and stops the pool.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _state.processor = build()
    _state.ready_barrier = ready_barrier


def _ready(timeout_s: float) -> int:
    # Every process waits here until all have built their processor, so each
    # process takes exactly one of these tasks.
    _state.ready_barrier.wait(timeout_s)
    return os.getpid()


def _process(inputs: Any, sampling_params: Any) -> tuple[Any, Optional[Exception]]:
    # Processing errors are returned rather than raised, so the caller can tell
    # them apart from failures to pickle the request or its result.
    try:
        return _state.processor(inputs, sampling_params), None
    except Exception as e:
        return None, e


class PreprocessPool:
    """Runs a hashing input processor in ``num_processes`` spawned processes.

    Processing errors are raised to the caller. A request or result that cannot
    be pickled is processed in this process by ``local`` instead.
    """

    def __init__(
        self, num_processes: int, build: Callable[[], Processor], local: Processor
    ) -> None:
        self._num_processes = num_processes
        self._local = local
        self._fallback_logged = False
        self._restore: Optional[Callable[[], None]] = None
        # Forking would copy the worker's CUDA, MPI and thread state.
        mp_context = _SpawnContext()
        self._executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=num_processes,
            mp_context=mp_context,
            initializer=_init_process,
            initargs=(build, mp_context.Barrier(num_processes)),
        )

    @classmethod
    def start(
        cls, llm_module: ModuleType, llm: Any, num_processes: int
    ) -> "PreprocessPool":
        """Start a pool and route ``llm``'s multimodal input processing to it.

        ``llm_module`` is the module whose ``create_input_processor_with_hash``
        TensorRT-LLM's LLM calls to process each multimodal request.
        """
        spec = InputProcessorSpec.from_llm(llm)
        local = llm_module.create_input_processor_with_hash(
            llm.input_processor, encoder_cache_enabled=spec.encoder_cache_enabled
        )
        build = functools.partial(build_input_processor, spec)
        pool = cls(num_processes, build, local)
        try:
            pool.wait_until_ready()
        except BaseException:
            pool.close()
            raise
        pool.route(llm_module, llm.input_processor, spec.encoder_cache_enabled)
        logger.info("Multimodal input processing runs in %d processes", num_processes)
        return pool

    def wait_until_ready(self, timeout_s: float = STARTUP_TIMEOUT_S) -> set[int]:
        """Start the processes, wait until each has built its processor and
        return their process ids."""
        futures = [
            self._executor.submit(_ready, timeout_s) for _ in range(self._num_processes)
        ]
        _, not_done = concurrent.futures.wait(futures, timeout=timeout_s)
        if not_done:
            raise TimeoutError(
                "Multimodal preprocessing processes did not start within "
                f"{timeout_s:.0f} s"
            )
        return {future.result() for future in futures}

    def route(
        self, llm_module: ModuleType, input_processor: Any, encoder_cache_enabled: bool
    ) -> None:
        """Send ``input_processor``'s hashing processor to this pool until close."""
        original = llm_module.create_input_processor_with_hash

        def create_input_processor_with_hash(processor, *args, **kwargs):
            # Other processors and arguments keep the in-process path.
            if (
                processor is not input_processor
                or args
                or kwargs.keys() - {"encoder_cache_enabled"}
                or kwargs.get("encoder_cache_enabled", False) != encoder_cache_enabled
            ):
                return original(processor, *args, **kwargs)
            return self

        setattr(
            llm_module,
            "create_input_processor_with_hash",
            create_input_processor_with_hash,
        )
        self._restore = functools.partial(
            setattr, llm_module, "create_input_processor_with_hash", original
        )

    def __call__(self, inputs: Any, sampling_params: Any) -> Any:
        future = self._executor.submit(_process, inputs, sampling_params)
        try:
            processed, error = future.result()
        except (pickle.PicklingError, TypeError, AttributeError) as e:
            # The request or its result could not be pickled.
            log = logger.debug if self._fallback_logged else logger.warning
            self._fallback_logged = True
            log(
                "A multimodal request could not be sent to a preprocessing process "
                "(%s); processing it in the worker process",
                e,
            )
            return self._local(inputs, sampling_params)
        if error is not None:
            raise error
        return processed

    def close(self) -> None:
        """Restore in-process input processing and stop the processes."""
        if self._restore is not None:
            self._restore()
            self._restore = None
        self._executor.shutdown(wait=False, cancel_futures=True)
