# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import time
from typing import Any, Callable, Dict

import requests

from tests.utils.managed_process import ManagedProcess

logger = logging.getLogger(__name__)


class EngineResponseError(Exception):
    """Custom exception for engine response errors"""

    pass


class EngineProcess(ManagedProcess):
    """Base class for LLM engine processes (vLLM, TRT-LLM, etc.)"""

    def _check_models_api(self, response):
        """Check if models API is working and returns models"""
        try:
            if response.status_code != 200:
                return False
            data = response.json()
            return data.get("data") and len(data["data"]) > 0
        except Exception:
            return False

    def get_metrics(self, port=8081):
        """Curl the metrics endpoint and return the response."""
        import requests

        metrics_url = f"http://localhost:{port}/metrics"
        logger.info(f"Curling metrics endpoint: {metrics_url}")

        try:
            response = requests.get(metrics_url, timeout=10)
            logger.info(
                f"Metrics endpoint responded with status: {response.status_code}"
            )
            return response
        except requests.RequestException as e:
            logger.error(f"Failed to curl metrics endpoint: {e}")
            raise

    def send_request(
        self, url: str, payload: Dict[str, Any], timeout: float = 30.0
    ) -> requests.Response:
        """
        Send a POST request to the engine with detailed logging.

        Args:
            url: The endpoint URL
            payload: The request payload
            timeout: Request timeout in seconds

        Returns:
            The response object

        Raises:
            requests.RequestException: If the request fails
        """

        # Log the request as a curl command for easy reproduction
        payload_json = json.dumps(payload, indent=2)
        curl_command = f'curl -X POST "{url}" \\\n  -H "Content-Type: application/json" \\\n  -d \'{payload_json}\''
        logger.info("Sending request (curl equivalent):\n%s", curl_command)

        start_time = time.time()
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            elapsed = time.time() - start_time

            # Log response details
            logger.info(
                "Received response: status=%d, elapsed=%.2fs",
                response.status_code,
                elapsed,
            )

            logger.debug("Response headers: %s", dict(response.headers))

            # Try to log response body (truncated if too long)
            try:
                if response.headers.get("content-type", "").startswith(
                    "application/json"
                ):
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

    def check_response(
        self,
        payload: Any,
        response: requests.Response,
        response_handler: Callable[[Any], str],
    ) -> None:
        """
        Check if the response is valid and contains expected content.

        Args:
            payload: The original payload (should have expected_response attribute)
            response: The response object
            response_handler: Function to extract content from response

        Raises:
            EngineResponseError: If the response is invalid or missing expected content
        """

        if response.status_code != 200:
            logger.error(
                "Response returned non-200 status code: %d", response.status_code
            )

            error_msg = f"Response returned non-200 status code: {response.status_code}"
            try:
                error_data = response.json()
                if "error" in error_data:
                    error_msg += f"\nError details: {error_data['error']}"
                logger.error(
                    "Response error details: %s", json.dumps(error_data, indent=2)
                )
            except Exception:
                logger.error("Response text: %s", response.text[:500])

            raise EngineResponseError(error_msg)

        # Extract content using the handler
        try:
            content = response_handler(response)
            logger.info(
                "Extracted content: \n%s",
                content[:200] + "..." if len(content) > 200 else content,
            )
        except Exception as e:
            raise EngineResponseError(f"Failed to extract content from response: {e}")

        if not content:
            raise EngineResponseError("Response contained empty content")

        if hasattr(payload, "expected_response") and payload.expected_response:
            missing_expected = []
            for expected in payload.expected_response:
                if expected not in content:
                    missing_expected.append(expected)

            if missing_expected:
                raise EngineResponseError(
                    f"Expected content not found in response. Missing: {missing_expected}"
                )
            else:
                logger.info(
                    f"SUCCESS: All expected content ({payload.expected_response}) found in response"
                )
