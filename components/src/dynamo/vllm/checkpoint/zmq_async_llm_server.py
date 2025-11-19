# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ZMQ-based async LLM server for checkpoint/restore operations."""

import asyncio
import contextlib
import os
import sys
import time
import traceback
import logging
from collections.abc import AsyncGenerator
from typing import Any, Optional

import cloudpickle
import msgspec
import zmq
import zmq.asyncio

from vllm.config import VllmConfig
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.executor.abstract import Executor

logger = logging.getLogger(__name__)

# RPC message types
class RPCMessageType(msgspec.Struct):
    request_id: str
    method: str
    args_pickle: bytes = b''  # Cloudpickle serialized args
    kwargs_pickle: bytes = b''  # Cloudpickle serialized kwargs
    is_generator: bool = False

class RPCResponse(msgspec.Struct):
    request_id: str
    result_pickle: bytes = b''  # Cloudpickle serialized result
    error: Optional[str] = None
    is_generator_item: bool = False
    generator_done: bool = False

class PropertyRequest(msgspec.Struct):
    request_id: str
    property_name: str

class PropertyResponse(msgspec.Struct):
    request_id: str
    value_pickle: bytes = b''  # Cloudpickle serialized value
    error: Optional[str] = None


class ZMQAsyncLLMServer:
    """Server process that runs AsyncLLM and handles RPC calls via ZMQ."""

    def __init__(self, socket_url: str, vllm_config: VllmConfig,
                 executor_class: type[Executor], **kwargs):
        self.socket_url = socket_url
        self.vllm_config = vllm_config
        self.executor_class = executor_class
        self.kwargs = kwargs
        self.async_llm: Optional[AsyncLLM] = None
        self.running = True
        self.engine_ready = False

        # For handling async generators
        self.active_generators: dict[str, AsyncGenerator] = {}

    async def initialize(self):
        """Initialize the AsyncLLM instance and wait for engine to be ready."""
        logger.info("Initializing AsyncLLM...")
        self.async_llm = AsyncLLM(
            vllm_config=self.vllm_config,
            executor_class=self.executor_class,
            **self.kwargs
        )

        # The AsyncLLM constructor returns after starting the engine,
        # but the engine may still be initializing (loading model,
        # compiling, etc.)
        # Try to verify the engine is ready by calling a simple method
        max_wait_time = 900  # 15 minutes
        start_time = time.time()
        last_error = None

        while time.time() - start_time < max_wait_time:
            try:
                # Try to get model config - this will succeed when
                # engine is ready
                await self.async_llm.get_model_config()
                self.engine_ready = True
                logger.info("AsyncLLM engine is fully initialized and ready")
                return
            except Exception as e:
                last_error = e
                # Check if the engine is dead
                if (hasattr(self.async_llm, 'errored') and
                        self.async_llm.errored):
                    raise RuntimeError(
                        f"AsyncLLM engine failed during initialization: {e}"
                    ) from e
                # Engine not ready yet, wait a bit
                await asyncio.sleep(0.5)
                elapsed = time.time() - start_time
                if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                    logger.info("Waiting for AsyncLLM engine to initialize... "
                               "(%d seconds elapsed)", int(elapsed))

        raise RuntimeError(
            f"AsyncLLM engine failed to initialize within {max_wait_time} "
            f"seconds. Last error: {last_error}")

    async def is_engine_ready(self) -> bool:
        """Check if the AsyncLLM engine is fully initialized and ready."""
        return self.engine_ready and self.async_llm is not None

    async def handle_rpc_request(self, msg: RPCMessageType) -> Any:
        """Handle an RPC request and return the result."""
        if self.async_llm is None:
            raise RuntimeError("AsyncLLM not initialized")

        # Handle special server-side methods
        if msg.method == "is_engine_ready":
            return await self.is_engine_ready()

        method = getattr(self.async_llm, msg.method)

        # Deserialize args and kwargs
        args = cloudpickle.loads(msg.args_pickle) if msg.args_pickle else ()
        kwargs = (cloudpickle.loads(msg.kwargs_pickle)
                  if msg.kwargs_pickle else {})

        if msg.is_generator:
            # For generator methods, we store the generator and
            # return items one by one
            generator = method(*args, **kwargs)
            self.active_generators[msg.request_id] = generator
            return None  # Initial response for generator
        else:
            # Regular method call
            result = method(*args, **kwargs)
            # Handle both sync and async methods
            if asyncio.iscoroutine(result):
                result = await result
            return result

    async def handle_generator_next(self,
                                     request_id: str) -> tuple[Any, bool]:
        """Get the next item from a generator."""
        if request_id not in self.active_generators:
            raise RuntimeError(
                f"No active generator for request {request_id}")

        generator = self.active_generators[request_id]
        try:
            item = await generator.__anext__()
            return item, False  # not done
        except StopAsyncIteration:
            del self.active_generators[request_id]
            return None, True  # done

    async def handle_property_request(self, msg: PropertyRequest) -> Any:
        """Handle a property access request."""
        if self.async_llm is None:
            raise RuntimeError("AsyncLLM not initialized")

        return getattr(self.async_llm, msg.property_name)

    async def run(self):
        """Main server loop."""
        ctx = zmq.asyncio.Context()
        # Using REP socket for reliable request-reply pattern
        # REP ensures proper message pairing and connection establishment
        socket = ctx.socket(zmq.REP)
        socket.bind(self.socket_url)
        socket.setsockopt(zmq.LINGER, 0)  # Don't wait on socket close

        logger.info("AsyncLLM server listening on %s", self.socket_url)

        # Wait for initial HELLO from client
        logger.debug("Waiting for HELLO message...")
        hello_msg = await socket.recv()
        logger.debug("Received message: %s", hello_msg)
        if hello_msg != b"HELLO":
            logger.warning("Expected HELLO, got %s", hello_msg)
        else:
            # REP socket must always send a reply
            await socket.send(b"HELLO_ACK")

        # Initialize AsyncLLM
        logger.debug("Initializing AsyncLLM...")
        await self.initialize()
        logger.debug("AsyncLLM initialized")

        # With REQ-REP, client will need to explicitly ask for readiness
        # Wait for READY_CHECK from client
        ready_check = await socket.recv()
        if ready_check == b"READY_CHECK":
            logger.debug("Sending READY signal...")
            await socket.send(b"READY")
            logger.debug("READY signal sent")

        try:
            while self.running:
                # Use poll with timeout to handle CRIU restore properly
                # After CRIU restore, the socket's epoll registration might be stale
                # Using poll with timeout ensures we periodically check for messages
                # REP socket blocks on recv, no need for poll
                try:
                    # Receive single message with protocol: TYPE|PAYLOAD
                    raw_msg = await socket.recv()
                    # Handle simple text messages first
                    if raw_msg == b"RECONNECT":
                        msg_type = "RECONNECT"
                        frames = None
                    elif raw_msg == b"SHUTDOWN":
                        msg_type = "SHUTDOWN"
                        frames = None
                    else:
                        # Parse structured messages
                        delimiter_idx = raw_msg.find(b'|')
                        if delimiter_idx == -1:
                            logger.error("Invalid message format: %s", raw_msg)
                            await socket.send(b"ERROR: Invalid message format")
                            continue
                        msg_type = raw_msg[:delimiter_idx].decode()
                        frames = [raw_msg[:delimiter_idx], raw_msg[delimiter_idx+1:]]
                except asyncio.TimeoutError:
                    # This shouldn't happen with REP socket
                    continue

                if msg_type == "RPC":
                    msg = msgspec.msgpack.decode(frames[1], type=RPCMessageType)
                    try:
                        if msg.method == "_generator_next":
                            # Special case for getting next item from generator
                            # Deserialize the request_id from args_pickle
                            request_id = cloudpickle.loads(msg.args_pickle)[0]
                            result, done = await self.handle_generator_next(
                                request_id)
                            response = RPCResponse(
                                request_id=msg.request_id,
                                result_pickle=(cloudpickle.dumps(result)
                                              if not done else b''),
                                is_generator_item=True,
                                generator_done=done
                            )
                        else:
                            result = await self.handle_rpc_request(msg)
                            response = RPCResponse(
                                request_id=msg.request_id,
                                result_pickle=cloudpickle.dumps(result)
                            )
                    except Exception as e:
                        response = RPCResponse(
                            request_id=msg.request_id,
                            error=(f"{type(e).__name__}: {str(e)}\n"
                                   f"{traceback.format_exc()}")
                        )
                    # Send response as single message
                    await socket.send(b"RPC_RESPONSE|" + msgspec.msgpack.encode(response))

                elif msg_type == "PROPERTY":
                    msg = msgspec.msgpack.decode(
                        frames[1], type=PropertyRequest)
                    try:
                        value = await self.handle_property_request(msg)
                        response = PropertyResponse(
                            request_id=msg.request_id,
                            value_pickle=cloudpickle.dumps(value)
                        )
                    except Exception as e:
                        response = PropertyResponse(
                            request_id=msg.request_id,
                            error=f"{type(e).__name__}: {str(e)}"
                        )
                    # Send response as single message
                    await socket.send(b"PROPERTY_RESPONSE|" + msgspec.msgpack.encode(response))

                elif msg_type == "SHUTDOWN":
                    logger.info("Received shutdown signal")
                    # REP socket must send reply before breaking
                    await socket.send(b"SHUTDOWN_ACK")
                    break

                elif msg_type == "RECONNECT":
                    # Handle reconnection after CRIU restore
                    logger.info("Received reconnect signal after CRIU restore")
                    await socket.send(b"RECONNECT_ACK")
                    logger.info("Sent reconnect acknowledgment")

                    # After CRIU restore, log that we're back to normal operation
                    logger.info("Server successfully handling messages after CRIU restore")

        finally:
            logger.info("AsyncLLM server shutting down...")
            if self.async_llm:
                logger.info("Shutting down AsyncLLM instance...")
                self.async_llm.shutdown()
                logger.info("AsyncLLM instance shutdown complete")
            socket.close()
            ctx.term()
            logger.info("AsyncLLM server shutdown complete")


def run_async_llm_server(socket_url: str, vllm_config_pickle: bytes,
                        executor_class_pickle: bytes,
                        kwargs_pickle: bytes,
                        parent_pid_for_tty: Optional[int] = None,
                        parent_tty_worker_fd: Optional[int] = None):
    """Entry point for the subprocess running AsyncLLM."""
    # Set up a private PTY for logs without making it a controlling TTY.
    # We intentionally avoid TIOCSCTTY so no session in the checkpointed
    # subtree owns a controlling TTY (simplifies CRIU restore).
    if parent_pid_for_tty is not None and parent_tty_worker_fd is not None:
        try:
            # Become a new session leader with no controlling TTY
            with contextlib.suppress(Exception):
                os.setsid()

            # Open the PTY worker passed by the parent without acquiring
            # a controlling TTY.
            fd_path = f"/proc/{parent_pid_for_tty}/fd/{parent_tty_worker_fd}"
            tty_fd = os.open(fd_path, os.O_RDWR | os.O_NOCTTY)

            # Mirror logs to this PTY (stdout/stderr; stdin optional)
            with contextlib.suppress(Exception):
                os.dup2(tty_fd, 1)
                os.dup2(tty_fd, 2)
            with contextlib.suppress(Exception):
                os.dup2(tty_fd, 0)
        except Exception as e:
            # Fall back to default stdio if anything fails; logs still work.
            print(f"Failed to set PTY for logs: {e}", file=sys.stderr)
        finally:
            with contextlib.suppress(Exception):
                os.close(tty_fd)  # type: ignore[name-defined]

    # Ensure logger is initialized in subprocess AFTER stdio is set
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        print(f"AsyncLLM server starting on {socket_url}", file=sys.stderr)

        # Deserialize the configuration
        vllm_config = cloudpickle.loads(vllm_config_pickle)
        executor_class = cloudpickle.loads(executor_class_pickle)
        kwargs = cloudpickle.loads(kwargs_pickle)

        print(f"Model: {vllm_config.model_config.model}", file=sys.stderr)

        # Create and run the server
        server = ZMQAsyncLLMServer(
            socket_url,
            vllm_config,
            executor_class,
            **kwargs,
        )

        # Run the async event loop
        asyncio.run(server.run())
    except Exception as e:
        print(f"AsyncLLM server failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
