# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service allocation server component for Dynamo.

This component wraps the AllocationServer from gpu_memory_service to manage
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
from gpu_memory_service.server import AllocationServer

from .args import parse_args

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class AllocationServerThread:
    """Wrapper to run AllocationServer in a background thread."""

    def __init__(self, socket_path: str, device: int):
        self.socket_path = socket_path
        self.device = device
        self._server: Optional[AllocationServer] = None
        self._thread: Optional[threading.Thread] = None
        self._started = threading.Event()
        self._error: Optional[Exception] = None

    def start(self) -> None:
        """Start the allocation server in a background thread."""
        self._thread = threading.Thread(
            target=self._run_server,
            name=f"AllocationServer-GPU{self.device}",
            daemon=True,
        )
        self._thread.start()
        # Wait for server to be ready
        self._started.wait(timeout=10.0)
        if self._error is not None:
            raise self._error
        if not self._started.is_set():
            raise RuntimeError("AllocationServer failed to start within timeout")

    def _run_server(self) -> None:
        """Run the server (called in background thread)."""
        try:
            self._server = AllocationServer(self.socket_path, device=self.device)
            logger.info(
                f"AllocationServer started on device {self.device} at {self.socket_path}"
            )
            self._started.set()
            self._server.serve_forever()
        except Exception as e:
            logger.error(f"AllocationServer error: {e}")
            self._error = e
            self._started.set()  # Unblock waiter even on error

    def stop(self) -> None:
        """Stop the allocation server."""
        if self._server is not None:
            logger.info(f"Stopping AllocationServer on device {self.device}")
            self._server.stop()
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

    # Start AllocationServer in a background thread
    server = AllocationServerThread(config.socket_path, config.device)
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
