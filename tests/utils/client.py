# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import time
from typing import Any, Dict

import requests

logger = logging.getLogger(__name__)


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
    payload_json = json.dumps(payload, indent=2)

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
