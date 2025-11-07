# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import asyncio
import contextlib
import errno
import multiprocessing
import os
import pty
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import AsyncGenerator, Mapping
from typing import Any, Optional, Union
import logging
import cloudpickle
import msgspec
import zmq
import zmq.asyncio

from vllm.config import LiteVllmConfig, ModelConfig, VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import PromptType
from vllm.inputs.preprocess import InputPreprocessor
from vllm.lora.request import LoRARequest
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Device, get_open_port
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.output_processor import RequestOutputCollector
from vllm.v1.executor.abstract import Executor
from vllm.v1.metrics.loggers import StatLoggerFactory

from .utils import (
    find_gpu_worker_pids,
    collect_process_tree_pids,
    verify_processes_exited, get_tty_info,
)
from .metadata import CheckpointMetadata
from .zmq_async_llm_server import (
    RPCMessageType, RPCResponse, PropertyRequest, PropertyResponse,
    run_async_llm_server,
)

logger = logging.getLogger(__name__)


def _create_engine_config_in_subprocess(queue: multiprocessing.Queue, engine_args_pickle: bytes) -> None:
    """Create engine config in a subprocess with isolated CUDA context.

    This function runs in a separate process to ensure complete CUDA context isolation.
    The subprocess will have its own CUDA context that won't interfere with the main
    process or any subsequent subprocesses.

    Args:
        queue: Multiprocessing queue to send results back to parent process
        engine_args_pickle: Pickled AsyncEngineArgs object
    """
    try:
        # Unpickle the engine args
        engine_args = cloudpickle.loads(engine_args_pickle)

        # Create the config (this may initialize CUDA)
        org_vllm_config = engine_args.create_engine_config()
        vllm_config = LiteVllmConfig.from_vllm_config(org_vllm_config)

        # Pickle and send back the result
        config_pickle = cloudpickle.dumps(vllm_config)
        queue.put(("success", config_pickle))
    except Exception as e:
        # Send back the error with traceback for debugging
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        queue.put(("error", error_msg))


def create_engine_config_isolated(engine_args: AsyncEngineArgs) -> VllmConfig:
    """Create engine config in a short-lived subprocess with separate CUDA context.

    This function spawns a new process to create the VllmConfig, ensuring that any
    CUDA initialization happens in a completely isolated context. This prevents
    CUDA context conflicts between the configuration phase and the actual engine
    execution.

    The subprocess uses the 'spawn' start method to ensure a clean state without
    inheriting any CUDA context from the parent process.

    Args:
        engine_args: AsyncEngineArgs object containing engine configuration

    Returns:
        VllmConfig: The created configuration object

    Raises:
        RuntimeError: If the subprocess fails to create the config
        TimeoutError: If the subprocess takes too long
    """
    # Use spawn to ensure a clean CUDA context
    ctx = multiprocessing.get_context('spawn')
    queue = ctx.Queue()

    # Pickle the engine args
    engine_args_pickle = cloudpickle.dumps(engine_args)

    # Start the subprocess
    process = ctx.Process(
        target=_create_engine_config_in_subprocess,
        args=(queue, engine_args_pickle),
        daemon=True
    )
    logger.info("Spawning subprocess for isolated config creation...")
    process.start()

    # Wait for the result
    try:
        status, result = queue.get(timeout=60)  # 60 second timeout
        process.join(timeout=5)  # Wait for process to cleanup

        if status == "error":
            raise RuntimeError(f"Failed to create engine config in subprocess:\n{result}")

        # Unpickle and return the config
        return cloudpickle.loads(result)
    except multiprocessing.TimeoutError:
        # Handle timeout specifically
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        raise TimeoutError("Subprocess timed out while creating engine config")
    except Exception as e:
        # Make sure to terminate the subprocess if something goes wrong
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        raise e


class CheckpointableAsyncLLM(AsyncLLM):
    """
    A wrapper around AsyncLLM that runs it in a subprocess and
    communicates via ZMQ. Supports CRIU checkpoint/restore functionality.

    This class is designed for checkpointing workflows:
    1. Start the AsyncLLM subprocess
    2. Checkpoint the process using CRIU
    3. Restore from checkpoint later

    Example usage:
        # Start and checkpoint
        engine_args = AsyncEngineArgs(model="...")
        llm = CheckpointableAsyncLLM.from_engine_args(engine_args)
        await llm.criu_checkpoint("/path/to/checkpoint/dir")

        # Restore from existing checkpoint
        llm = CheckpointableAsyncLLM.from_engine_args(engine_args,
                                                       auto_start=False)
        llm.checkpoint_dir = "/path/to/existing/checkpoint"
        await llm.criu_resume()

        # Now use normally
        async for output in llm.generate(...):
            print(output)
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        log_requests: bool = True,
        start_engine_loop: bool = True,
        stat_loggers: Optional[list[StatLoggerFactory]] = None,
        client_addresses: Optional[dict[str, str]] = None,
        client_count: int = 1,
        client_index: int = 0,
        auto_start: bool = True,
    ) -> None:
        if auto_start:
            # Use TCP socket instead of IPC for better CRIU compatibility
            # Only set these attributes if not restoring (auto_start=True)
            self.port = get_open_port()
            self.socket_url = f"tcp://127.0.0.1:{self.port}"

        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        # Processor and io_processor are managed by the subprocess AsyncLLM
        # Set to None as we proxy through ZMQ RPC
        self.processor = None  # type: ignore
        self.io_processor = None
        self.process: Optional[multiprocessing.Process] = None
        self.ctx = zmq.asyncio.Context()
        self.socket: Optional[zmq.asyncio.Socket] = None
        self.checkpoint_dir: Optional[str] = None
        self._is_running = False
        self._subprocess_started = False
        # For CRIU TTY forwarding
        self._pty_main_fd: Optional[int] = None
        self._pty_forwarder_thread: Optional[threading.Thread] = None

        # Serialize configuration for subprocess
        self.vllm_config_pickle = cloudpickle.dumps(vllm_config)
        self.executor_class_pickle = cloudpickle.dumps(executor_class)
        kwargs = {
            'log_stats': log_stats,
            'usage_context': usage_context,
            'log_requests': log_requests,
            'start_engine_loop': start_engine_loop,
            'stat_loggers': None,  # Set to None - stat_loggers contain unpicklable Component objects
            'client_addresses': client_addresses,
            'client_count': client_count,
            'client_index': client_index,
        }
        self.kwargs_pickle = cloudpickle.dumps(kwargs)

        # Only start the subprocess if auto_start is True
        # This allows creating the instance without starting a subprocess
        # when we plan to restore from CRIU
        if auto_start:
            self._start_subprocess()

    def _start_subprocess(self):
        """Start the AsyncLLM subprocess."""
        if self._subprocess_started:
            logger.warning("Subprocess already started")
            return

        ctx = multiprocessing.get_context('spawn')
        # Note: daemon=False is required because AsyncLLM may need to spawn
        # its own child processes (e.g., DPCoordinator for data parallel)
        # Create a private PTY for the child so it can become a session leader
        # with a controlling TTY. We pass the parent's PID and the worker fd
        # number so the child can open it via /proc and call TIOCSCTTY.
        pty_main_fd, pty_worker_fd = pty.openpty()
        os.set_inheritable(pty_worker_fd, True)
        self.process = ctx.Process(
            target=run_async_llm_server,
            args=(self.socket_url, self.vllm_config_pickle,
                  self.executor_class_pickle, self.kwargs_pickle,
                  os.getpid(), pty_worker_fd),
            daemon=False
        )
        self.process.start()
        self._is_running = True
        self._subprocess_started = True

        # Close the worker fd in parent process (child has it)
        os.close(pty_worker_fd)

        # Forward child's PTY main to our stdout so logs are visible
        # Do this BEFORE connecting so we can see any startup errors
        try:
            self._start_pty_forwarder(pty_main_fd)
        except Exception as e:
            logger.warning("Failed to start startup PTY forwarder: %s", e)

        # Connect to the subprocess
        self._connect()

    def _connect(self):
        """Connect to the AsyncLLM subprocess."""
        # Use a synchronous REQ socket for initial connection
        sync_ctx = zmq.Context()
        sync_socket = sync_ctx.socket(zmq.REQ)
        sync_socket.connect(self.socket_url)

        try:
            # Send initial hello and wait for acknowledgment
            sync_socket.send(b"HELLO")
            logger.debug("Sent HELLO to subprocess")
            hello_ack = sync_socket.recv()
            if hello_ack != b"HELLO_ACK":
                raise RuntimeError(f"Unexpected HELLO response: {hello_ack}")

            # Now check for readiness
            sync_socket.send(b"READY_CHECK")
            logger.debug("Sent READY_CHECK to subprocess")

            # Wait for ready signal
            # Must match or exceed the server's max_wait_time (900s in zmq_async_llm_server.py)
            timeout = 900  # 15 minutes for initial connection and engine initialization
            # With REQ socket, we just wait for the response
            if sync_socket.poll(timeout=timeout * 1000):  # Convert to milliseconds
                msg = sync_socket.recv()
                logger.debug("Received message: %s", msg)
                if msg == b"READY":
                    logger.info("Connected to AsyncLLM subprocess")
                    # Now create the async socket for normal operations
                    # Client side of REQ-REP pattern
                    self.socket = self.ctx.socket(zmq.REQ)
                    # REQ sockets automatically wait for connection establishment
                    self.socket.connect(self.socket_url)
                    return
                else:
                    raise RuntimeError(f"Unexpected READY response: {msg}")
            else:
                # Poll timed out - check if process is still alive
                if self.process and not self.process.is_alive():
                    exit_code = self.process.exitcode
                    raise RuntimeError(
                        f"AsyncLLM subprocess died during startup "
                        f"(exit code: {exit_code})")
                raise TimeoutError(
                    f"Timeout waiting for AsyncLLM subprocess to start "
                    f"after {timeout} seconds")
        finally:
            sync_socket.close()
            sync_ctx.term()

    async def start(self) -> None:
        """Explicitly start the AsyncLLM subprocess.

        This is used when auto_start=False was passed to __init__.
        """
        if self._subprocess_started:
            logger.info("Subprocess already started")
            return

        self._start_subprocess()
        logger.info("AsyncLLM subprocess started successfully")

    async def wait_until_ready(self, timeout: float = 300.0) -> None:
        """Wait until the AsyncLLM engine is fully initialized and ready.

        This method ensures the engine has completed all initialization steps:
        - Model loaded
        - torch.compile completed (if applicable)
        - CUDA graphs captured
        - KV cache allocated
        - Engine ready to handle requests

        Args:
            timeout: Maximum time to wait in seconds
                    (default: 300.0 = 5 minutes). Set higher for large models
                    or when torch.compile is enabled.

        Raises:
            TimeoutError: If engine doesn't become ready within timeout
            RuntimeError: If engine fails during initialization
        """
        if not self._subprocess_started:
            raise RuntimeError(
                "AsyncLLM subprocess not started. Call start() first.")

        start_time = time.time()
        last_error = None

        logger.info("Waiting for AsyncLLM engine to be fully initialized...")

        while time.time() - start_time < timeout:
            try:
                # Check if the engine is fully ready
                is_ready = await self._rpc_call("is_engine_ready")
                if is_ready:
                    logger.info(
                        "AsyncLLM engine is fully initialized and ready")
                    return
            except Exception as e:
                last_error = e
                # Check if process is still alive
                if self.process and not self.process.is_alive():
                    raise RuntimeError(
                        f"AsyncLLM subprocess died during initialization: "
                        f"{last_error}"
                    ) from e

            # Engine not ready yet, wait a bit before retrying
            await asyncio.sleep(0.5)

            # Log progress
            elapsed = time.time() - start_time
            if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                logger.info("Still waiting for engine initialization... "
                           "(%d seconds elapsed)", int(elapsed))

        raise TimeoutError(
            f"AsyncLLM engine did not become ready within {timeout} seconds. "
            f"Last error: {last_error}"
        )

    async def _rpc_call(self, method: str, *args, **kwargs) -> Any:
        """Make an RPC call to the AsyncLLM subprocess."""
        if not self._subprocess_started:
            raise RuntimeError(
                "AsyncLLM subprocess not started. "
                "Call start() first or use auto_start=True")
        if not self.socket:
            raise RuntimeError(
                "Not connected to AsyncLLM subprocess")

        request_id = str(uuid.uuid4())
        msg = RPCMessageType(
            request_id=request_id,
            method=method,
            args_pickle=cloudpickle.dumps(args),
            kwargs_pickle=cloudpickle.dumps(kwargs),
            is_generator=False
        )

        # Send request as single message
        await self.socket.send(b"RPC|" + msgspec.msgpack.encode(msg))

        # Wait for response
        raw_response = await self.socket.recv()
        # Parse response format: TYPE|PAYLOAD
        delimiter_idx = raw_response.find(b'|')
        if delimiter_idx == -1:
            raise RuntimeError("Invalid response format")
        response_type = raw_response[:delimiter_idx]
        if response_type != b"RPC_RESPONSE":
            raise RuntimeError(f"Unexpected response type: {response_type}")

        response = msgspec.msgpack.decode(raw_response[delimiter_idx+1:], type=RPCResponse)
        if response.error:
            raise RuntimeError(f"RPC error: {response.error}")

        return cloudpickle.loads(response.result_pickle)

    async def _rpc_generator(self, method: str, *args,
                             **kwargs) -> AsyncGenerator:
        """Make an RPC call that returns an async generator."""
        if not self._subprocess_started:
            raise RuntimeError(
                "AsyncLLM subprocess not started. "
                "Call start() first or use auto_start=True")
        if not self.socket:
            raise RuntimeError(
                "Not connected to AsyncLLM subprocess")

        request_id = str(uuid.uuid4())

        # Initial call to start the generator
        msg = RPCMessageType(
            request_id=request_id,
            method=method,
            args_pickle=cloudpickle.dumps(args),
            kwargs_pickle=cloudpickle.dumps(kwargs),
            is_generator=True
        )

        await self.socket.send(b"RPC|" + msgspec.msgpack.encode(msg))

        # Get initial response
        raw_response = await self.socket.recv()
        delimiter_idx = raw_response.find(b'|')
        if delimiter_idx == -1:
            raise RuntimeError("Invalid response format")
        response_type = raw_response[:delimiter_idx]
        if response_type != b"RPC_RESPONSE":
            raise RuntimeError(f"Unexpected response type: {response_type}")

        response = msgspec.msgpack.decode(raw_response[delimiter_idx+1:], type=RPCResponse)
        if response.error:
            raise RuntimeError(f"RPC error: {response.error}")

        # Now iterate through generator items
        while True:
            # Request next item
            next_msg = RPCMessageType(
                request_id=str(uuid.uuid4()),
                method="_generator_next",
                args_pickle=cloudpickle.dumps((request_id,))
            )

            await self.socket.send(b"RPC|" + msgspec.msgpack.encode(next_msg))

            raw_response = await self.socket.recv()
            delimiter_idx = raw_response.find(b'|')
            if delimiter_idx == -1:
                raise RuntimeError("Invalid response format")
            response_type = raw_response[:delimiter_idx]
            if response_type != b"RPC_RESPONSE":
                raise RuntimeError(f"Unexpected response type: {response_type}")

            response = msgspec.msgpack.decode(raw_response[delimiter_idx+1:], type=RPCResponse)
            if response.error:
                raise RuntimeError(f"RPC error: {response.error}")

            if response.generator_done:
                break

            yield cloudpickle.loads(response.result_pickle)

    async def _get_property(self, property_name: str) -> Any:
        """Get a property value from the AsyncLLM subprocess."""
        if not self._subprocess_started:
            raise RuntimeError(
                "AsyncLLM subprocess not started. "
                "Call start() first or use auto_start=True")
        if not self.socket:
            raise RuntimeError(
                "Not connected to AsyncLLM subprocess")

        request_id = str(uuid.uuid4())
        msg = PropertyRequest(request_id=request_id,
                              property_name=property_name)

        await self.socket.send(b"PROPERTY|" + msgspec.msgpack.encode(msg))

        raw_response = await self.socket.recv()
        delimiter_idx = raw_response.find(b'|')
        if delimiter_idx == -1:
            raise RuntimeError("Invalid response format")
        response_type = raw_response[:delimiter_idx]
        if response_type != b"PROPERTY_RESPONSE":
            raise RuntimeError(f"Unexpected response type: {response_type}")

        response = msgspec.msgpack.decode(raw_response[delimiter_idx+1:], type=PropertyResponse)
        if response.error:
            raise RuntimeError(f"Property error: {response.error}")

        return cloudpickle.loads(response.value_pickle)

    # Implement all AsyncLLM methods
    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return await self._rpc_call("get_supported_tasks")

    async def add_request(
        self,
        request_id: str,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        tokenization_kwargs: Optional[dict[str, Any]] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
        data_parallel_rank: Optional[int] = None,
    ) -> RequestOutputCollector:
        return await self._rpc_call(
            "add_request",
            request_id,
            prompt,
            params,
            arrival_time,
            lora_request,
            tokenization_kwargs,
            trace_headers,
            priority,
            data_parallel_rank
        )

    async def generate(
        self,
        prompt: PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
        data_parallel_rank: Optional[int] = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        async for output in self._rpc_generator(
            "generate",
            prompt,
            sampling_params,
            request_id,
            lora_request,
            trace_headers,
            priority,
            data_parallel_rank
        ):
            yield output

    async def abort(self, request_id: Union[str, list[str]]) -> None:
        await self._rpc_call("abort", request_id)

    async def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
        truncate_prompt_tokens: Optional[int] = None,
        tokenization_kwargs: Optional[dict[str, Any]] = None,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        async for output in self._rpc_generator(
            "encode",
            prompt,
            pooling_params,
            request_id,
            lora_request,
            trace_headers,
            priority,
            truncate_prompt_tokens,
            tokenization_kwargs
        ):
            yield output

    async def get_vllm_config(self) -> VllmConfig:
        return await self._rpc_call("get_vllm_config")

    async def get_model_config(self) -> ModelConfig:
        return await self._rpc_call("get_model_config")

    async def get_input_preprocessor(self) -> InputPreprocessor:
        return await self._rpc_call("get_input_preprocessor")

    async def get_tokenizer(self) -> AnyTokenizer:
        return await self._rpc_call("get_tokenizer")

    async def is_tracing_enabled(self) -> bool:
        return await self._rpc_call("is_tracing_enabled")

    async def do_log_stats(self) -> None:
        await self._rpc_call("do_log_stats")

    async def check_health(self) -> None:
        await self._rpc_call("check_health")

    async def start_profile(self) -> None:
        await self._rpc_call("start_profile")

    async def stop_profile(self) -> None:
        await self._rpc_call("stop_profile")

    async def reset_mm_cache(self) -> None:
        await self._rpc_call("reset_mm_cache")

    async def reset_prefix_cache(self,
                                 device: Optional[Device] = None) -> None:
        await self._rpc_call("reset_prefix_cache", device)

    async def sleep(self, level: int = 1) -> None:
        await self._rpc_call("sleep", level)

    async def wake_up(self, tags: Optional[list[str]] = None) -> None:
        await self._rpc_call("wake_up", tags)

    async def is_sleeping(self) -> bool:
        return await self._rpc_call("is_sleeping")

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        return await self._rpc_call("add_lora", lora_request)

    async def remove_lora(self, lora_id: int) -> bool:
        return await self._rpc_call("remove_lora", lora_id)

    async def list_loras(self) -> set[int]:
        return await self._rpc_call("list_loras")

    async def pin_lora(self, lora_id: int) -> bool:
        return await self._rpc_call("pin_lora", lora_id)

    async def collective_rpc(
        self,
        method: str,
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict] = None
    ):
        return await self._rpc_call(
            "collective_rpc", method, timeout, args, kwargs)

    async def wait_for_requests_to_drain(self, drain_timeout: int = 300):
        await self._rpc_call("wait_for_requests_to_drain", drain_timeout)

    async def scale_elastic_ep(self, new_data_parallel_size: int,
                               drain_timeout: int = 300):
        await self._rpc_call(
            "scale_elastic_ep", new_data_parallel_size, drain_timeout)

    # Properties
    @property
    def is_running(self) -> bool:
        return (self._is_running and
                (self.process is not None and self.process.is_alive()))

    @property
    def is_stopped(self) -> bool:
        return not self.is_running

    @property
    def errored(self) -> bool:
        # Synchronous property - cannot make async RPC call
        # Return False as default; actual error handling is via exceptions
        return False

    @property
    def dead_error(self) -> BaseException:
        # Synchronous property - cannot make async RPC call
        # Return a default exception; actual error handling is via exceptions
        return RuntimeError("Engine error - check logs for details")

    # CRIU-specific methods
    async def criu_checkpoint(
            self, checkpoint_dir: str,
            cuda_checkpoint_path: str = "cuda-checkpoint") -> None:
        """Checkpoint the AsyncLLM subprocess using CRIU.

        Args:
            checkpoint_dir: Directory to save checkpoint files
            cuda_checkpoint_path: Unused; kept for backward compatibility
        """
        if not self._subprocess_started:
            raise RuntimeError(
                "AsyncLLM subprocess not started. "
                "Call start() first")
        if not self.process or not self.process.is_alive():
            raise RuntimeError("AsyncLLM subprocess is not running")

        self.checkpoint_dir = checkpoint_dir
        try:
            os.makedirs(checkpoint_dir, exist_ok=False)
        except FileExistsError as err:
            raise RuntimeError(
                "Checkpoint directory "
                f"{checkpoint_dir} already exists. "
                "Please delete it before checkpointing."
            ) from err

        # Root of the subprocess tree
        root_pid = self.process.pid

        # Validate that only leaf processes have CUDA contexts
        logger.info("Getting CUDA process tree...")
        cuda_pids = find_gpu_worker_pids(root_pid)


        # Sleep the model (level 1) to free GPU memory before checkpointing
        logger.info("Putting model to sleep (level 1) before checkpoint...")
        await self.sleep(level=1)
        logger.info("Model sleep completed")

        

        # Create checkpoint metadata
        metadata = CheckpointMetadata()

        # Get TTY info from the subprocess
        rdev, dev = get_tty_info(root_pid)
        metadata.tty_rdev = rdev
        metadata.tty_dev = dev

        # Save process tree info
        metadata.tree_pid = root_pid
        metadata.zmq_port = self.port
        metadata.cuda_pids = cuda_pids

        # Take a snapshot of the process tree (for post-dump verification)
        pre_dump_tree = collect_process_tree_pids(root_pid)

        # Build CRIU dump command
        cmd = [
            "criu", "dump",
            "--shell-job",
            "--images-dir", checkpoint_dir,
            "-o", "criu-dump.log",
            "-v4",
            "--ext-unix-sk",
            "--tcp-established",
            "--external", "mnt[shm]:/dev/shm",
            "--link-remap",
            "--manage-cgroups=ignore",
            "--tree", str(root_pid),
            "--ghost-limit", "50M",
            "--timeout", "1800",
        ]

        if metadata.tty_external:
            cmd.extend(["--external", metadata.tty_external])

        logger.info("Running CRIU dump: %s", ' '.join(cmd))

        # Run CRIU dump
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"CRIU dump failed: {result.stderr}")

        logger.info(
            "Successfully checkpointed AsyncLLM to %s", checkpoint_dir)

        # Save all metadata to JSON file
        metadata.save(checkpoint_dir)

        # Reap the child process to avoid a zombie holding the PID.
        # This ensures /proc/<pid> disappears if the process is already dead.
        if self.process is not None:
            try:
                self.process.join(timeout=5)
            except Exception as err:
                raise RuntimeError("Failed to reap child process") from err

        # Verify that all processes in the pre-dump tree are gone.
        # This helps catch stray children that might linger due to plugins.
        verify_processes_exited(pre_dump_tree)

        # The process is now frozen, mark it as not running
        self._is_running = False
        self.process = None

        # Close the socket connection
        if self.socket:
            self.socket.close()
            self.socket = None

    async def criu_resume(
            self, cuda_checkpoint_path: str = "cuda-checkpoint") -> None:
        """Restore from CRIU checkpoint.

        This method:
        1. Restores the AsyncLLM subprocess from CRIU checkpoint
        2. Re-establishes the ZMQ connection
        3. Wakes up the model to restore GPU memory

        Args:
            cuda_checkpoint_path: Path to cuda-checkpoint utility (not used)
        """
        if not self.checkpoint_dir:
            raise RuntimeError(
                "No checkpoint directory set. Call criu_checkpoint first.")

        # Load checkpoint metadata
        metadata = CheckpointMetadata.load(self.checkpoint_dir)
        if metadata is None:
            raise RuntimeError(
                f"Could not load checkpoint metadata from {self.checkpoint_dir}")

        # Use the saved port from checkpoint
        assert metadata.zmq_port is not None
        self.port = metadata.zmq_port
        self.socket_url = f"tcp://127.0.0.1:{self.port}"
        logger.info("Using saved port %d from checkpoint", self.port)

        # Ensure the original tree PID from dump is fully gone to avoid
        # PID collisions when restoring into the same PID namespace.
        if metadata.tree_pid:
            start = time.time()
            # Wait up to a short grace period since dump should have killed it
            while time.time() - start < 5.0:
                if os.system(f"kill -0 {metadata.tree_pid} >/dev/null 2>&1") != 0:
                    break
                await asyncio.sleep(0.05)

            # Verify root PID is not taken now (PID could be reused by others)
            pid_path = f"/proc/{metadata.tree_pid}"
            if os.path.exists(pid_path):
                # Try to read the cmdline of the holder for diagnostics
                holder = ""
                try:
                    cmd_path = os.path.join(pid_path, "cmdline")
                    with open(cmd_path, "rb") as f:
                        raw = f.read().replace(b"\x00", b" ")
                        holder = raw.decode("utf-8", "ignore").strip()
                except Exception:
                    holder = ""
                msg = (
                    "CRIU restore pre-check failed: root PID "
                    f"{metadata.tree_pid} is in use in current PID namespace. "
                    "Restore will fail with EEXIST. Consider restoring in a "
                    "new PID namespace or wait until the PID is free."
                )
                if holder:
                    logger.error(
                        "%s Holder cmdline: %s",
                        msg,
                        holder,
                    )
                else:
                    logger.error("%s", msg)

        # Build CRIU restore command
        cmd = [
            "criu", "restore",
            "--shell-job",
            "--restore-detached",
            "--images-dir", self.checkpoint_dir,
            "-o", "criu-restore.log",
            "-v4",
            "--ext-unix-sk",
            "--tcp-established",
            "--external", "mnt[shm]:/dev/shm",
            "--link-remap",
            "--manage-cgroups=ignore",
            "--ghost-limit", "50M",
        ]

        # Provide a valid TTY to CRIU using a Python-created pty, so we can
        # mirror logs back to this terminal and support non-TTY parents.
        # We pass the pty worker fd to CRIU and map it to the saved TTY id.
        pass_fds = ()
        pty_main_fd = None
        pty_worker_fd = None
        if metadata.tty_external:
            try:
                pty_main_fd, pty_worker_fd = pty.openpty()
                os.set_inheritable(pty_worker_fd, True)
                # Map the saved tty id to the pty worker fd in CRIU
                cmd.extend([
                    "--inherit-fd", f"fd[{pty_worker_fd}]:{metadata.tty_external}",
                ])
                pass_fds = (pty_worker_fd,)
            except Exception as e:
                logger.warning("Failed to create PTY for CRIU restore: %s", e)

        logger.info("Running CRIU restore: %s", ' '.join(cmd))

        # Run CRIU restore
        if pass_fds:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    pass_fds=pass_fds)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"CRIU restore failed: {result.stderr}")

        logger.info("Successfully restored AsyncLLM from checkpoint")

        # Re-establish connection to the restored process
        self._is_running = True
        # Mark subprocess as started after restore
        self._subprocess_started = True

        # Create new socket and perform reconnection handshake
        # Re-establish client side of REQ-REP pattern
        self.socket = self.ctx.socket(zmq.REQ)
        # REQ sockets automatically wait for connection establishment
        self.socket.connect(self.socket_url)

        # Send reconnection signal and wait for acknowledgment
        logger.info("Sending reconnection signal to restored process...")

        # With REQ-REP pattern, the socket will block until connected
        # This should succeed on the first attempt
        await self.socket.send(b"RECONNECT")
        logger.info("Waiting for reconnection acknowledgment...")

        # REQ socket will wait for the response
        # Use a generous timeout for post-restore initialization
        if await self.socket.poll(timeout=500000):  # 5 second timeout
            ack = await self.socket.recv()
            if ack == b"RECONNECT_ACK":
                logger.info("Reconnection acknowledged by server")
            else:
                raise ValueError(f"Unexpected reconnection response: {ack}")
        else:
            raise TimeoutError("No response to reconnection signal after 30 seconds")

        # Wake up the model to restore GPU memory
        logger.info("Waking up model after restore...")
        await self.wake_up()
        logger.info("Model wake up completed")

        # Start forwarding the restored process output to our stdout
        if pty_main_fd is not None:
            try:
                self._start_pty_forwarder(pty_main_fd)
            except Exception as e:
                logger.warning("Failed to start PTY forwarder: %s", e)
        # Close the worker fd in this process; CRIU/restored proc holds it now
        if pty_worker_fd is not None:
            with contextlib.suppress(Exception):
                os.close(pty_worker_fd)

    def shutdown(self):
        """Shutdown the subprocess and clean up resources."""
        # Close PTY main if present
        if getattr(self, "_pty_main_fd", None) is not None:
            with contextlib.suppress(Exception):
                os.close(self._pty_main_fd)  # type: ignore[arg-type]
            self._pty_main_fd = None

        # Try to send shutdown signal if we have a socket connection
        # This works both for normal operation and after CRIU restore
        if self.socket and self._is_running:
            try:
                # Create a synchronous context for shutdown
                sync_ctx = zmq.Context()
                sync_socket = sync_ctx.socket(zmq.REQ)
                sync_socket.connect(self.socket_url)

                # Send shutdown signal and wait for acknowledgment
                sync_socket.send(b"SHUTDOWN")
                logger.info("Sent SHUTDOWN signal to AsyncLLM subprocess")
                # Wait for acknowledgment with timeout
                if sync_socket.poll(timeout=5000):  # 5 second timeout
                    ack = sync_socket.recv()
                    logger.debug("Received shutdown acknowledgment: %s", ack)
                else:
                    logger.warning("No shutdown acknowledgment received")

                # If we have a process reference (normal operation), wait for it
                if self.process and self.process.is_alive():
                    # Wait for the process to exit gracefully
                    # (give it more time)
                    shutdown_timeout = 30  # 30 seconds for graceful shutdown
                    self.process.join(timeout=shutdown_timeout)

                    if self.process.is_alive():
                        logger.warning(
                            "AsyncLLM subprocess did not exit gracefully "
                            "after %d seconds, terminating...",
                            shutdown_timeout)
                        self.process.terminate()
                        self.process.join(timeout=5)

                        if self.process.is_alive():
                            logger.error(
                                "AsyncLLM subprocess did not terminate, "
                                "killing...")
                            self.process.kill()
                            self.process.join()
                else:
                    # After CRIU restore, we don't have process reference
                    # Give processes time to shutdown gracefully
                    time.sleep(2)

                sync_socket.close()
                sync_ctx.term()
            except Exception as e:
                logger.error("Error during shutdown: %s", e)
                # Force kill if graceful shutdown fails and we have process ref
                if self.process and self.process.is_alive():
                    self.process.kill()

        # Close async socket
        if self.socket:
            self.socket.close()

        if self.ctx:
            self.ctx.term()

        self._is_running = False

    def __del__(self):
        self.shutdown()


    # ----- Internal helpers for CRIU PTY forwarding -----
    def _start_pty_forwarder(self, main_fd: int) -> None:
        """Forward data from PTY main fd to this process' stdout.

        This mirrors the restored process' stdout/stderr back
        into the controlling terminal running CheckpointableAsyncLLM.
        """
        self._pty_main_fd = main_fd
        logger.debug("Starting PTY forwarder for fd %d", main_fd)

        def _forward_loop(fd: int):
            try:
                os.set_blocking(fd, False)  # Non-blocking reads
                logger.debug("PTY forwarder started for fd %d", fd)
                while True:
                    try:
                        data = os.read(fd, 4096)
                        if not data:
                            break
                        # Write raw bytes to stdout to preserve ANSI
                        # sequences
                        sys.stdout.buffer.write(data)
                        sys.stdout.buffer.flush()
                    except OSError as e:
                        if e.errno == errno.EAGAIN:
                            time.sleep(0.01)
                            continue
                        raise
                    except InterruptedError:
                        continue
            except Exception as e:
                logger.debug("PTY forwarder stopped: %s", e)
            finally:
                with contextlib.suppress(Exception):
                    os.close(fd)
                if getattr(self, "_pty_main_fd", None) == fd:
                    self._pty_main_fd = None

        t = threading.Thread(target=_forward_loop,
                             args=(main_fd,),
                             name="criu-pty-forwarder",
                             daemon=True)
        t.start()
        self._pty_forwarder_thread = t

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[list[StatLoggerFactory]] = None,
        enable_log_requests: bool = False,
        disable_log_stats: bool = False,
        client_addresses: Optional[dict[str, str]] = None,
        client_count: int = 1,
        client_index: int = 0,
        auto_start: bool = True,
    ) -> "CheckpointableAsyncLLM":
        """Create CheckpointableAsyncLLM from VllmConfig."""
        # Validate required settings for checkpointing
        if not vllm_config.model_config.enable_sleep_mode:
            raise ValueError(
                "enable_sleep_mode must be True in vllm_config for checkpointing. "
                "Sleep mode is required to omit KV cache from the checkpoint image."
            )
        
        if hasattr(vllm_config.parallel_config, 'disable_custom_all_reduce'):
            if not vllm_config.parallel_config.disable_custom_all_reduce:
                raise ValueError(
                    "disable_custom_all_reduce must be True in vllm_config for checkpointing. "
                    "Custom all-reduce uses CUDA IPC which is not compatible with CRIU."
                )
        
        return cls(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            start_engine_loop=start_engine_loop,
            stat_loggers=stat_loggers,
            log_requests=enable_log_requests,
            log_stats=not disable_log_stats,
            usage_context=usage_context,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
            auto_start=auto_start,
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[list[StatLoggerFactory]] = None,
        auto_start: bool = True,
    ) -> "CheckpointableAsyncLLM":
        """Create CheckpointableAsyncLLM from EngineArgs.

        Args:
            engine_args: Engine configuration arguments
            start_engine_loop: Whether to start the engine loop
            usage_context: Usage context for the engine
            stat_loggers: Optional stat logger factories
            auto_start: Whether to automatically start the subprocess
            port: Optional port to use for ZMQ connection (for restore)

        Returns:
            CheckpointableAsyncLLM instance
        """
        # Force enable sleep mode and disable custom all reduce for checkpointing
        if not engine_args.enable_sleep_mode:
            logger.warning("enable_sleep_mode was False, forcing to True for checkpointing")
            engine_args.enable_sleep_mode = True
        
        if not engine_args.disable_custom_all_reduce:
            logger.warning("disable_custom_all_reduce was False, forcing to True for checkpointing")
            engine_args.disable_custom_all_reduce = True
        
        vllm_config = create_engine_config_isolated(engine_args)

        return cls(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            log_requests=engine_args.enable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            auto_start=auto_start,
        )
