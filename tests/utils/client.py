# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import re
import time
from copy import deepcopy
from typing import Any, Dict

import requests

logger = logging.getLogger(__name__)


def _truncate_base64_url(url: str, max_length: int = 100) -> str:
    """Helper to truncate a single base64 data URL."""
    if (m := re.match(r"^(data:image/[^;]+;base64,)(.+)$", url)) and len(
        m.group(2)
    ) > max_length:
        data = m.group(2)
        return f"{m.group(1)}{data[:max_length]}...<{len(data)} chars, truncated>"
    return url


def _sanitize_payload_for_logging(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Truncate base64-encoded images in multimodal payloads for cleaner logging.
    Multimodal payloads can contain base64 images with multiple MB of data in
    the field "type": "image_url", "image_url": "data: ... <MB of data>"
    """
    sanitized = deepcopy(payload)

    # Handle chat completions with multimodal content
    if "messages" in sanitized:
        for message in sanitized["messages"]:
            content = message.get("content")
            # Content can be string or list of content parts (multimodal)
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        image_url = part.get("image_url", {})
                        if "url" in image_url:
                            image_url["url"] = _truncate_base64_url(image_url["url"])

    return sanitized


def send_request(
    url: str,
    payload: Dict[str, Any],
    timeout: float = 30.0,
    method: str = "POST",
    log_level: int = 20,
) -> requests.Response:
    """
    Send an HTTP request to the engine with detailed logging.

    Args:
        url: The endpoint URL
        payload: The request payload (for GET, sent as query params)
        timeout: Request timeout in seconds
        method: HTTP method ("POST" or "GET")

    Returns:
        The response object

    Raises:
        requests.RequestException: If the request fails
    """

    method_upper = method.upper()

    # Sanitize payload for logging (truncate base64 images)
    sanitized_payload = _sanitize_payload_for_logging(payload)
    payload_json = json.dumps(sanitized_payload, indent=2)

    curl_command = ""
    if method_upper == "GET":
        curl_command = f'curl "{url}"'
        if payload:
            # For GET requests, payload is sent as query parameters
            query_params = "&".join(f"{k}={v}" for k, v in payload.items())
            curl_command += f"?{query_params}"
    else:
        curl_command = f'curl -X {method_upper} "{url}"'
        if method_upper == "POST":
            curl_command += (
                ' \\\n  -H "Content-Type: application/json" \\\n  -d \''
                + payload_json
                + "'"
            )
    logger.log(log_level, "Sending request (curl equivalent):\n%s", curl_command)

    start_time = time.time()
    try:
        if method_upper == "GET":
            response = requests.get(url, params=payload, timeout=timeout)
        elif method_upper == "POST":
            response = requests.post(url, json=payload, timeout=timeout)
        else:
            # Fallback for other methods if needed
            response = requests.request(
                method_upper, url, json=payload, timeout=timeout
            )

        elapsed = time.time() - start_time

        # Log response details
        logger.log(
            log_level,
            "Received response: status=%d, elapsed=%.2fs",
            response.status_code,
            elapsed,
        )

        logger.debug("Response headers: %s", dict(response.headers))

        # Try to log response body (truncated if too long)
        try:
            if response.headers.get("content-type", "").startswith("application/json"):
                response_data = response.json()
                response_str = json.dumps(response_data, indent=2)
                if len(response_str) > 1000:
                    response_str = response_str[:1000] + "... (truncated)"
                logger.debug("Response body: %s", response_str)
            else:
                response_text = response.text
                if len(response_text) > 1000:
                    response_text = response_text[:1000] + "... (truncated)"
                logger.debug("Response body: %s", response_text)
        except Exception as e:
            logger.debug("Could not parse response body: %s", e)

        return response

    except requests.exceptions.Timeout:
        logger.error("Request timed out after %.2f seconds", timeout)
        raise
    except requests.exceptions.ConnectionError as e:
        logger.error("Connection error: %s", e)
        raise
    except requests.exceptions.RequestException as e:
        logger.error("Request failed: %s", e)
        raise


def wait_for_model_availability(
    url: str,
    endpoint: str,
    model: str,
    logger: logging.Logger,
    max_attempts: int = 15,
    attempt_timeouts: list[float] | None = None,
) -> bool:
    """
    Wait for model to be available by sending test requests.

    Polls the specified endpoint with test requests until the model responds
    successfully or max attempts are reached. Used to ensure a deployed model
    is ready before running tests.

    Args:
        url: Base URL for the service (e.g., "http://localhost:8000")
        endpoint: API endpoint path (e.g., "/v1/chat/completions")
        model: Model name to test
        logger: Logger instance for output
        max_attempts: Maximum number of attempts to check availability (default: 15)
        attempt_timeouts: List of timeout values for each attempt (default: decreasing from 60s)

    Returns:
        True if model is available and responding, False otherwise
    """
    if attempt_timeouts is None:
        # Default: Start with 60s timeout, then gradually decrease
        attempt_timeouts = [60, 60, 45, 30, 30, 20, 20, 15, 15, 15, 10, 10, 10, 10, 10]

    test_url = f"{url}{endpoint}"

    for attempt in range(max_attempts):
        try:
            test_payload = {
                "model": model,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1,
                "stream": False,
            }

            timeout_val = attempt_timeouts[min(attempt, len(attempt_timeouts) - 1)]
            logger.debug(
                f"Testing model availability at {test_url} (attempt {attempt+1}/{max_attempts}, timeout={timeout_val}s)"
            )
            response = requests.post(test_url, json=test_payload, timeout=timeout_val)

            if response.status_code == 200:
                logger.info(f"Model '{model}' is available and responding")
                # Give a bit more time for stabilization
                logger.info("Model ready, waiting 5s for stabilization...")
                time.sleep(5)
                return True
            elif response.status_code == 404:
                logger.warning(
                    f"Model '{model}' not found (404). Response: {response.text[:200]}"
                )
            elif response.status_code == 400:
                logger.warning(f"Bad request (400). Response: {response.text[:200]}")
            else:
                logger.warning(
                    f"Unexpected status code {response.status_code}: {response.text[:200]}"
                )

        except requests.Timeout as e:
            logger.warning(
                f"Model availability test timed out (attempt {attempt+1}): {e}"
            )
        except Exception as e:
            logger.warning(f"Model availability test failed (attempt {attempt+1}): {e}")

        if attempt < max_attempts - 1:
            wait_time = 10 if attempt < 5 else 5
            logger.info(f"Waiting {wait_time}s before retry...")
            time.sleep(wait_time)

    logger.warning("Could not confirm model availability after all attempts")
    return False
