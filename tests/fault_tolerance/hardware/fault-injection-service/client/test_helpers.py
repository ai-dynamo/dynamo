#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
"""
Test helper utilities for fault injection tests.

This module provides:
- Color output formatting (Colors class)
- Frontend reachability checks
- Completion request helpers
- Response validation

Example usage:
    from fault_injection_client import FaultInjectionClient
    from test_helpers import (
        Colors,
        check_frontend_reachable,
        send_completion_request,
        validate_completion_response,
        get_config_from_env,
    )

    config = get_config_from_env()
    if not check_frontend_reachable(config['frontend_url']):
        print(f"{Colors.RED}Frontend not reachable{Colors.RESET}")
"""
import os
import sys
from typing import Dict, Optional

import requests


class Colors:
    """
    ANSI color codes for terminal output.

    Automatically disables colors when output is piped/redirected,
    unless FORCE_COLOR environment variable is set.

    Example:
        print(f"{Colors.GREEN}[OK]{Colors.RESET} Test passed")
        print(f"{Colors.RED}[FAIL]{Colors.RESET} Test failed")
    """

    RESET = "\033[0m"
    BOLD = "\033[1m"

    # Foreground colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"

    # Background colors
    BG_RED = "\033[101m"
    BG_GREEN = "\033[102m"
    BG_YELLOW = "\033[103m"

    @staticmethod
    def disable():
        """Disable all color codes (useful for non-TTY output)"""
        Colors.RESET = ""
        Colors.BOLD = ""
        Colors.RED = ""
        Colors.GREEN = ""
        Colors.YELLOW = ""
        Colors.BLUE = ""
        Colors.MAGENTA = ""
        Colors.CYAN = ""
        Colors.WHITE = ""
        Colors.GRAY = ""
        Colors.BG_RED = ""
        Colors.BG_GREEN = ""
        Colors.BG_YELLOW = ""

    @staticmethod
    def enable():
        """Re-enable color codes"""
        Colors.RESET = "\033[0m"
        Colors.BOLD = "\033[1m"
        Colors.RED = "\033[91m"
        Colors.GREEN = "\033[92m"
        Colors.YELLOW = "\033[93m"
        Colors.BLUE = "\033[94m"
        Colors.MAGENTA = "\033[95m"
        Colors.CYAN = "\033[96m"
        Colors.WHITE = "\033[97m"
        Colors.GRAY = "\033[90m"
        Colors.BG_RED = "\033[101m"
        Colors.BG_GREEN = "\033[102m"
        Colors.BG_YELLOW = "\033[103m"


# Auto-detect color support
# Check if output is a TTY (disable colors if piped/redirected)
# Allow forcing colors via environment variable (useful for kubectl logs)
FORCE_COLOR = os.getenv("FORCE_COLOR", "").lower() in ("1", "true", "yes")
if not FORCE_COLOR and not sys.stdout.isatty():
    Colors.disable()


def get_config_from_env() -> Dict[str, str]:
    """
    Get configuration from environment variables.

    Returns:
        Dictionary with api_url, frontend_url, and app_namespace

    Example:
        config = get_config_from_env()
        client = FaultInjectionClient(api_url=config['api_url'])
    """
    return {
        "api_url": os.getenv("API_URL", "http://localhost:8080"),
        "frontend_url": os.getenv("FRONTEND_URL", "http://localhost:8000"),
        "app_namespace": os.getenv("APP_NAMESPACE", "dynamo-oviya"),
    }


def check_frontend_reachable(
    frontend_url: Optional[str] = None, timeout: int = 5
) -> bool:
    """
    Check if frontend is reachable via health endpoint.

    Args:
        frontend_url: Frontend URL (defaults to FRONTEND_URL env var)
        timeout: Request timeout in seconds

    Returns:
        True if frontend is reachable, False otherwise

    Example:
        if check_frontend_reachable():
            print("Frontend is up!")
    """
    if frontend_url is None:
        frontend_url = os.getenv("FRONTEND_URL", "http://localhost:8000")

    try:
        response = requests.get(f"{frontend_url}/health", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def send_completion_request(
    prompt: str,
    max_tokens: int,
    frontend_url: Optional[str] = None,
    model: str = "Qwen/Qwen3-0.6B",
    temperature: float = 0.7,
    timeout: int = 30,
    verbose: bool = True,
) -> requests.Response:
    """
    Send a completion request to the frontend.

    Args:
        prompt: The prompt to send
        max_tokens: Maximum tokens to generate
        frontend_url: Frontend URL (defaults to FRONTEND_URL env var)
        model: Model name to use
        temperature: Sampling temperature
        timeout: Request timeout in seconds
        verbose: Print request details

    Returns:
        Response object

    Example:
        response = send_completion_request("Hello world", 10)
        validate_completion_response(response)
    """
    if frontend_url is None:
        frontend_url = os.getenv("FRONTEND_URL", "http://localhost:8000")

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    if verbose:
        print(
            f"{Colors.GRAY}Sending completion request: model='{model}', prompt='{prompt}', max_tokens={max_tokens}{Colors.RESET}"
        )

    return requests.post(
        f"{frontend_url}/v1/completions", json=payload, timeout=timeout
    )


def validate_completion_response(
    response: requests.Response,
    verbose: bool = True,
) -> str:
    """
    Validate that the response is a proper completion response.

    Args:
        response: Response object from send_completion_request
        verbose: Print validation details

    Returns:
        Completion text from the response

    Raises:
        AssertionError: If response is invalid

    Example:
        response = send_completion_request("Hello", 10)
        text = validate_completion_response(response)
        print(f"Got: {text}")
    """
    assert (
        response.status_code == 200
    ), f"Request failed with status {response.status_code}: {response.text}"

    data = response.json()
    assert "choices" in data, f"No 'choices' in response: {data}"
    assert len(data["choices"]) > 0, f"Empty choices: {data}"
    assert "text" in data["choices"][0], f"No 'text' in first choice: {data}"

    completion_text = data["choices"][0]["text"]

    if verbose:
        print(
            f"{Colors.GRAY}Received valid completion: {completion_text[:50]}...{Colors.RESET}"
        )

    return completion_text


def print_section_header(title: str, level: str = "major"):
    """
    Print a formatted section header.

    Args:
        title: Section title
        level: "major" (=== lines), "minor" (--- lines), or "subsection" (no lines)

    Example:
        print_section_header("Starting Test", "major")
        print_section_header("Setup Phase", "minor")
    """
    if level == "major":
        print(f"\n{Colors.CYAN}{'=' * 80}")
        print(f"{Colors.BOLD}{Colors.MAGENTA}{title}{Colors.RESET}")
        print(f"{Colors.CYAN}{'=' * 80}{Colors.RESET}")
    elif level == "minor":
        print(f"\n{Colors.BOLD}{Colors.BLUE}{title}{Colors.RESET}")
        print(f"{Colors.BLUE}{'-' * 80}{Colors.RESET}")
    else:  # subsection
        print(f"\n{Colors.BOLD}{title}{Colors.RESET}")


def print_status(status: str, message: str):
    """
    Print a status message with appropriate color.

    Args:
        status: Status type - "ok", "fail", "warn", "info", "note", "expected"
        message: Message to print

    Example:
        print_status("ok", "Test passed")
        print_status("fail", "Test failed")
        print_status("warn", "Retrying in 5s")
    """
    status_map = {
        "ok": (Colors.GREEN, "[OK]"),
        "fail": (Colors.RED, "[FAIL]"),
        "error": (Colors.RED, "[ERROR]"),
        "warn": (Colors.YELLOW, "[WARN]"),
        "warning": (Colors.YELLOW, "[WARNING]"),
        "info": (Colors.CYAN, "[INFO]"),
        "note": (Colors.CYAN, "[NOTE]"),
        "expected": (Colors.YELLOW, "[EXPECTED]"),
        "cleanup": (Colors.YELLOW, "[CLEANUP]"),
    }

    color, prefix = status_map.get(
        status.lower(), (Colors.WHITE, f"[{status.upper()}]")
    )
    print(f"{color}{prefix}{Colors.RESET} {message}")


def print_test_result(passed: bool, message: str = ""):
    """
    Print final test result.

    Args:
        passed: True if test passed, False otherwise
        message: Optional message to include

    Example:
        print_test_result(True)
        print_test_result(False, "Assertion failed")
    """
    if passed:
        print(f"\n{Colors.GREEN}{'=' * 80}")
        print(f"{Colors.BOLD}[PASS] TEST PASSED{Colors.RESET}")
        if message:
            print(f"{Colors.GREEN}{message}{Colors.RESET}")
        print(f"{Colors.GREEN}{'=' * 80}{Colors.RESET}")
    else:
        print(f"\n{Colors.RED}{'=' * 80}")
        print(f"{Colors.BOLD}[FAIL] TEST FAILED{Colors.RESET}")
        if message:
            print(f"{Colors.RED}{message}{Colors.RESET}")
        print(f"{Colors.RED}{'=' * 80}{Colors.RESET}")


# Export all public functions
__all__ = [
    "Colors",
    "get_config_from_env",
    "check_frontend_reachable",
    "send_completion_request",
    "validate_completion_response",
    "print_section_header",
    "print_status",
    "print_test_result",
]
