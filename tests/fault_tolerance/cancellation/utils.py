# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re
import shutil
import socket
import threading
import time
from typing import Any, Callable, Dict

import pytest
import requests

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.engine_process import FRONTEND_PORT
from tests.utils.managed_process import ManagedProcess

logger = logging.getLogger(__name__)


class DynamoFrontendProcess(ManagedProcess):
    """Process manager for Dynamo frontend"""

    def __init__(self, request):
        command = ["python", "-m", "dynamo.frontend"]

        # Set debug logging environment
        env = os.environ.copy()
        env["DYN_LOG"] = "debug"

        log_dir = f"{request.node.name}_frontend"

        # Clean up any existing log directory from previous runs
        try:
            shutil.rmtree(log_dir)
            logger.info(f"Cleaned up existing log directory: {log_dir}")
        except FileNotFoundError:
            # Directory doesn't exist, which is fine
            pass

        super().__init__(
            command=command,
            env=env,
            display_output=True,
            terminate_existing=True,
            log_dir=log_dir,
        )


class CancellableRequest:
    """A wrapper for a single request that can be explicitly cancelled.

    Each instance supports only one post() call and should not be reused.
    """

    # Class-level tracking for thread-safe socket monitoring
    _socket_tracking_lock = threading.Lock()
    _socket_trackers: Dict[
        Any, Any
    ] = {}  # Maps thread ID to CancellableRequest instance
    _original_socket: Callable[..., Any] = socket.socket

    @classmethod
    def _global_tracked_socket(
        cls, family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0, fileno=None
    ):
        """Global socket tracker that routes to the appropriate CancellableRequest instance"""
        sock = cls._original_socket(family, type, proto, fileno)

        # Find which CancellableRequest should track this socket
        thread_id = threading.current_thread().ident
        with cls._socket_tracking_lock:
            tracker = cls._socket_trackers.get(thread_id)
            if tracker:
                tracker._active_sockets.append(sock)

        return sock

    def __init__(self):
        self.session = requests.Session()
        self.response = None
        self.exception = None
        self._cancelled = False
        self._request_thread = None
        self._lock = threading.Lock()
        self._active_sockets = []

    def post(self, *args, **kwargs):
        """Start a POST request in a separate thread. Can only be called once."""

        def make_request():
            thread_id = threading.current_thread().ident

            # Register this thread's tracker
            with self.__class__._socket_tracking_lock:
                self.__class__._socket_trackers[thread_id] = self
                # Install global monkey-patch if not already installed
                if socket.socket != self.__class__._global_tracked_socket:
                    socket.socket = self.__class__._global_tracked_socket  # type: ignore[assignment,misc]

            try:
                self.response = self.session.post(*args, **kwargs)
            except Exception as e:
                self.exception = e
            finally:
                # Unregister this thread's tracker
                with self.__class__._socket_tracking_lock:
                    self.__class__._socket_trackers.pop(thread_id, None)
                    # Only restore original socket if no other trackers are active
                    if (
                        not self.__class__._socket_trackers
                        and socket.socket == self.__class__._global_tracked_socket
                    ):
                        socket.socket = self.__class__._original_socket  # type: ignore[assignment,misc]

        with self._lock:
            if self._request_thread is not None:
                raise RuntimeError(
                    "This CancellableRequest instance has already been used. Create a new instance."
                )
            self._request_thread = threading.Thread(target=make_request)
        self._request_thread.start()

    def cancel(self):
        """Cancel the request by forcefully closing the underlying TCP socket"""
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True

        # Do the cleanup outside the lock to avoid holding it during I/O operations
        # Force close all tracked sockets (this is the actual TCP connection)
        for sock in self._active_sockets:
            # Set socket to non-blocking to avoid hanging
            try:
                sock.setblocking(0)
            except Exception as e:
                logger.warning(f"Failed to set socket to non-blocking: {e}")
            # Force shutdown both send and receive
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except Exception as e:
                logger.warning(f"Failed to shutdown socket: {e}")
            # Close the socket
            try:
                sock.close()
            except Exception as e:
                logger.warning(f"Failed to close socket: {e}")

        self._active_sockets.clear()

        # Also close at the requests level for cleanup
        if self.response:
            self.response.close()
        for adapter in self.session.adapters.values():
            adapter.close()
        self.session.close()

    def get_response(self):
        """Get the response or raise exception if there was one"""
        if self._cancelled:
            raise requests.exceptions.RequestException("Request was cancelled")
        if self.exception:
            raise self.exception
        return self.response


def send_completion_request(prompt: str, max_tokens: int) -> CancellableRequest:
    """Send a completion request to the frontend

    Args:
        prompt: The prompt for completion
        max_tokens: Maximum tokens to generate

    Returns:
        A CancellableRequest object that can be explicitly cancelled
    """
    payload = {
        "model": FAULT_TOLERANCE_MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
    }

    headers = {"Content-Type": "application/json"}

    logger.info(
        f"Sending completion request with prompt: '{prompt[:50]}...' and max_tokens: {max_tokens}"
    )

    # Return a cancellable request object
    cancellable_req = CancellableRequest()
    cancellable_req.post(
        f"http://localhost:{FRONTEND_PORT}/v1/completions",
        headers=headers,
        json=payload,
    )
    return cancellable_req


def send_chat_completion_request(
    prompt: str, max_tokens: int, stream: bool = False
) -> CancellableRequest:
    """Send a chat completion request to the frontend

    Args:
        prompt: The prompt for chat completion
        max_tokens: Maximum tokens to generate
        stream: Whether to stream the response

    Returns:
        A CancellableRequest object that can be explicitly cancelled
    """
    payload = {
        "model": FAULT_TOLERANCE_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": stream,
    }

    headers = {"Content-Type": "application/json"}

    logger.info(
        f"Sending chat completion request (stream={stream}) with prompt: '{prompt[:50]}...' and max_tokens: {max_tokens}"
    )

    # Return a cancellable request object
    cancellable_req = CancellableRequest()
    cancellable_req.post(
        f"http://localhost:{FRONTEND_PORT}/v1/chat/completions",
        headers=headers,
        json=payload,
        stream=stream,
    )
    return cancellable_req


def send_cancellable_request(
    request_type: str = "completion",
    use_long_prompt: bool = False,
) -> CancellableRequest:
    """Send a request that can be manually cancelled.

    Args:
        request_type: Type of request - "completion", "chat_completion", or "chat_completion_stream"
        use_long_prompt: Whether to use an extremely long prompt

    Returns:
        A CancellableRequest object that can be explicitly cancelled
    """
    prompt = "Tell me a very long and detailed story about the history of artificial intelligence, including all major milestones, researchers, and breakthroughs?"
    if use_long_prompt:
        prompt += " Make sure it is" + " long" * 8000 + "!"

    if request_type == "completion":
        return send_completion_request(prompt, 8192)
    elif request_type == "chat_completion":
        return send_chat_completion_request(prompt, 8192, stream=False)
    elif request_type == "chat_completion_stream":
        return send_chat_completion_request(prompt, 8192, stream=True)
    else:
        raise ValueError(f"Unknown request type: {request_type}")


def read_streaming_responses(
    cancellable_req: CancellableRequest,
    expected_count: int = 5,
) -> None:
    """Read a specific number of responses from a streaming request.

    Args:
        cancellable_req: The CancellableRequest object with an active stream
        expected_count: Number of responses to read before returning

    Raises:
        pytest.fail if stream ends before expected_count responses
    """
    response = cancellable_req.get_response()
    if not response or response.status_code != 200:
        pytest.fail(
            f"Failed to get streaming response: status_code={response.status_code if response else 'None'}"
        )

    response_count = 0
    for line in response.iter_lines():
        response_count += 1
        logger.info(
            f"Received streaming response {response_count}: {line.decode()[:100]}"
        )
        if response_count >= expected_count:
            logger.info(f"Successfully read {response_count} responses")
            return

    # If we get here, stream ended too early
    pytest.fail(
        f"Stream ended after only {response_count} lines - expected to read at least {expected_count}"
    )


def read_log_content(log_path: str | None) -> str:
    """Read log content from a file"""
    if log_path is None:
        pytest.fail("Log path is None - cannot read log content")

    try:
        with open(log_path, "r") as f:
            return f.read()
    except Exception as e:
        pytest.fail(f"Could not read log file {log_path}: {e}")


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI color codes from text"""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def poll_for_pattern(
    process: ManagedProcess,
    pattern: str,
    log_offset: int = 0,
    max_wait_ms: int = 500,
    poll_interval_ms: int = 5,
    match_type: str = "endswith",  # "contains" or "endswith"
) -> tuple[str, int]:
    """
    Poll process log for a specific pattern.

    Args:
        process: The process to monitor logs from
        pattern: The pattern to search for
        log_offset: Offset in the log to start reading from
        max_wait_ms: Maximum time to wait for the pattern in milliseconds
        poll_interval_ms: Interval between polls in milliseconds
        match_type: How to match the pattern - "contains" or "endswith"

    Returns:
        Tuple of (matched_content, new_log_offset) where matched_content is:
        - For "contains": everything after the pattern on the same line
        - For "endswith": empty string (since nothing follows)
    """
    max_iterations = max_wait_ms // poll_interval_ms
    iteration = 0
    current_offset = log_offset

    logger.info(
        f"Starting to poll for '{pattern}' pattern (max {max_iterations} iterations)..."
    )

    while iteration < max_iterations:
        # Read the process log
        log_content = read_log_content(process._log_path)
        new_content = log_content[current_offset:]

        # Look for the pattern
        for line in new_content.split("\n"):
            clean_line = strip_ansi_codes(line).strip()

            matched = False
            result = ""

            if match_type == "contains" and pattern in clean_line:
                # Find the pattern and return everything after it
                idx = clean_line.rfind(pattern)  # Use rfind to get last occurrence
                if idx != -1:
                    result = clean_line[idx + len(pattern) :].strip()
                    matched = True
            elif match_type == "endswith" and clean_line.endswith(pattern):
                # Pattern is at the end, nothing follows
                result = ""
                matched = True

            if matched:
                logger.info(f"Found pattern '{pattern}' at iteration {iteration}")
                if match_type == "contains":
                    logger.info(f"Content after pattern: '{result}'")
                # Update offset to current position
                current_offset = len(log_content)
                return result, current_offset

        # Update offset for next poll
        current_offset = len(log_content)

        # Wait before next poll
        time.sleep(poll_interval_ms / 1000.0)
        iteration += 1

    pytest.fail(
        f"Failed to find '{pattern}' pattern after {max_iterations} iterations ({max_wait_ms}ms)"
    )
