# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service allocation server component for Dynamo.

This component wraps the GMSRPCServer from gpu_memory_service to manage
GPU memory allocations with connection-based RW/RO locking.

Workers connect via the socket path, which should be passed to vLLM/SGLang via:
    --load-format gpu_memory_service
    --model-loader-extra-config '{"gpu_memory_service_socket_path": "/tmp/gpu_memory_service_{device}.sock"}'

Usage:
    python -m dynamo.gpu_memory_service --device 0
    python -m dynamo.gpu_memory_service --device 0 --socket-path /tmp/gpu_memory_service_{device}.sock
"""

import asyncio
import logging
import os
import signal
import threading
from typing import Optional

import uvloop
from gpu_memory_service.server import GMSRPCServer

from .args import parse_args

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class GMSRPCServerThread:
    """Wrapper to run GMSRPCServer in a background thread."""

    def __init__(self, socket_path: str, device: int):
        self.socket_path = socket_path
        self.device = device
        self._server: Optional[GMSRPCServer] = None
        self._thread: Optional[threading.Thread] = None
        self._started = threading.Event()
        self._error: Optional[Exception] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def start(self) -> None:
        """Start the allocation server in a background thread."""
        self._thread = threading.Thread(
            target=self._run_server,
            name=f"GMSRPCServer-GPU{self.device}",
            daemon=True,
        )
        self._thread.start()
        # Wait for server to be ready (socket file created)
        self._started.wait(timeout=10.0)
        if self._error is not None:
            raise self._error
        if not self._started.is_set():
            raise RuntimeError("GMSRPCServer failed to start within timeout")

    def _run_server(self) -> None:
        """Run the server (called in background thread).

        The GMSRPCServer is async-based, so we create a new event loop for this thread.
        """
        try:
            # Create a new event loop for this thread
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            self._server = GMSRPCServer(self.socket_path, device=self.device)

            # Start the server (creates the socket)
            self._loop.run_until_complete(self._server.start())
            logger.info(
                f"GMSRPCServer started on device {self.device} at {self.socket_path}"
            )
            self._started.set()

            # Run the main loop
            while self._server._running:
                self._loop.run_until_complete(asyncio.sleep(1))

        except Exception as e:
            logger.error(f"GMSRPCServer error: {e}")
            self._error = e
            self._started.set()  # Unblock waiter even on error
        finally:
            if self._loop is not None:
                self._loop.close()

    def stop(self) -> None:
        """Stop the allocation server."""
        if self._server is not None:
            logger.info(f"Stopping GMSRPCServer on device {self.device}")
            # Signal the server to stop - the loop in _run_server will exit
            self._server._running = False
            self._server._shutdown = True
            # Wake any blocked waiters from the server's event loop
            if self._loop is not None and self._loop.is_running():

                async def _notify():
                    async with self._server._condition:
                        self._server._condition.notify_all()

                asyncio.run_coroutine_threadsafe(_notify(), self._loop)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)


async def worker() -> None:
    """Main async worker function."""
    config = parse_args()

    # Configure logging level
    if config.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("dynamo.gpu_memory_service").setLevel(logging.DEBUG)

    logger.info(f"Starting GPU Memory Service Server for device {config.device}")
    logger.info(f"Socket path: {config.socket_path}")

    loop = asyncio.get_running_loop()

    # Clean up any existing socket file
    if config.socket_path and os.path.exists(config.socket_path):
        os.unlink(config.socket_path)
        logger.debug(f"Removed existing socket file: {config.socket_path}")

    # Start GMSRPCServer in a background thread
    server = GMSRPCServerThread(config.socket_path, config.device)
    server.start()

    # Set up shutdown event
    shutdown_event = asyncio.Event()

    def signal_handler():
        logger.info("Received shutdown signal")
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    logger.info("GPU Memory Service Server ready, waiting for connections...")
    logger.info(
        f"To connect vLLM workers, use: --load-format gpu_memory_service "
        f'--model-loader-extra-config \'{{"gpu_memory_service_socket_path": "{config.socket_path}"}}\''
    )

    # Wait for shutdown signal
    try:
        await shutdown_event.wait()
    finally:
        logger.info("Shutting down GPU Memory Service Server...")
        server.stop()
        logger.info("GPU Memory Service Server shutdown complete")


def main() -> None:
    """Entry point for GPU Memory Service server."""
    uvloop.install()
    asyncio.run(worker())


if __name__ == "__main__":
    main()
