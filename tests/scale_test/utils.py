# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Helper utilities for scale testing.
"""

import logging
import time
from typing import List

import requests

# Configure module logger
logger = logging.getLogger(__name__)


def generate_namespace(index: int) -> str:
    """
    Generate a unique namespace for a deployment.

    Args:
        index: The deployment index (1-based)

    Returns:
        A unique namespace string like 'scale-test-1'
    """
    return f"scale-test-{index}"


def wait_for_url_ready(url: str, timeout: float = 30.0, interval: float = 0.5) -> bool:
    """
    Wait for a URL to become available.

    Args:
        url: The URL to check
        timeout: Maximum time to wait in seconds
        interval: Time between checks in seconds

    Returns:
        True if the URL is reachable, False if timeout exceeded
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(interval)
    return False


def wait_for_all_ready(
    urls: List[str], timeout: float = 120.0, interval: float = 1.0
) -> bool:
    """
    Wait for all URLs to become available.

    Args:
        urls: List of URLs to check
        timeout: Maximum time to wait in seconds
        interval: Time between check rounds in seconds

    Returns:
        True if all URLs are reachable, False if timeout exceeded
    """
    start_time = time.time()
    remaining_urls = set(urls)

    while time.time() - start_time < timeout and remaining_urls:
        newly_ready = set()
        for url in remaining_urls:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    newly_ready.add(url)
                    logger.info(f"URL ready: {url}")
            except requests.RequestException:
                pass

        remaining_urls -= newly_ready

        if remaining_urls:
            elapsed = time.time() - start_time
            logger.debug(
                f"Waiting for {len(remaining_urls)} URLs... ({elapsed:.1f}s elapsed)"
            )
            time.sleep(interval)

    if remaining_urls:
        logger.error(f"Timeout waiting for URLs: {remaining_urls}")
        return False

    return True


def wait_for_workers_registered(
    frontend_url: str,
    model: str,
    timeout: float = 120.0,
    interval: float = 1.0,
) -> bool:
    """
    Wait for workers to register with a frontend and be ready to serve requests.

    This performs a two-phase check:
    1. Poll GET /v1/models until at least one model is registered (workers connected)
    2. Send a test POST to /v1/completions to verify the request pipeline is functional

    This is more robust than just checking /health because the frontend health check
    can pass before workers have registered via NATS/etcd, causing 503 errors.

    Args:
        frontend_url: Base URL of the frontend (e.g., "http://localhost:8001")
        model: The model name to wait for
        timeout: Maximum time to wait in seconds
        interval: Time between check rounds in seconds

    Returns:
        True if workers are registered and can serve requests, False if timeout exceeded
    """
    start_time = time.time()
    models_url = f"{frontend_url}/v1/models"
    completions_url = f"{frontend_url}/v1/completions"

    # Phase 1: Wait for model to appear in /v1/models
    logger.info(f"Waiting for model '{model}' to register at {frontend_url}...")
    model_found = False
    while time.time() - start_time < timeout:
        try:
            response = requests.get(models_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                model_ids = [m.get("id") for m in models]
                if model in model_ids or len(models) > 0:
                    logger.info(f"Model(s) registered at {frontend_url}: {model_ids}")
                    model_found = True
                    break
        except requests.RequestException as e:
            logger.debug(f"Error checking models endpoint: {e}")

        elapsed = time.time() - start_time
        logger.debug(
            f"No models registered yet at {frontend_url} ({elapsed:.1f}s elapsed)"
        )
        time.sleep(interval)

    if not model_found:
        logger.error(f"Timeout waiting for model to register at {frontend_url}")
        return False

    # Phase 2: Send a test completion request to verify the pipeline is functional
    # This catches the case where /v1/models shows the model but routing isn't ready yet
    logger.info(f"Verifying completions pipeline at {frontend_url}...")
    test_payload = {"model": model, "prompt": "ping", "max_tokens": 1}

    while time.time() - start_time < timeout:
        try:
            response = requests.post(completions_url, json=test_payload, timeout=30)
            if response.status_code == 200:
                logger.info(f"Completions pipeline ready at {frontend_url}")
                return True
            elif response.status_code == 503:
                logger.debug(
                    f"Service unavailable at {frontend_url}, workers not ready yet"
                )
            elif response.status_code == 404 and "Model not found" in response.text:
                logger.debug(f"Model not routable yet at {frontend_url}")
            else:
                logger.debug(
                    f"Unexpected response from {frontend_url}: "
                    f"status={response.status_code}, body={response.text[:200]}"
                )
        except requests.RequestException as e:
            logger.debug(f"Error testing completions: {e}")

        elapsed = time.time() - start_time
        logger.debug(f"Pipeline not ready at {frontend_url} ({elapsed:.1f}s elapsed)")
        time.sleep(interval)

    logger.error(f"Timeout waiting for completions pipeline at {frontend_url}")
    return False


def wait_for_all_workers_registered(
    frontend_urls: List[str],
    model: str,
    timeout: float = 120.0,
    interval: float = 1.0,
) -> bool:
    """
    Wait for workers to register with all frontends.

    This is the multi-frontend version of wait_for_workers_registered.
    It ensures that each frontend has workers registered and can serve requests.

    Args:
        frontend_urls: List of frontend base URLs to check
        model: The model name to wait for
        timeout: Maximum time to wait in seconds
        interval: Time between check rounds in seconds

    Returns:
        True if all frontends have workers ready, False if timeout exceeded
    """
    start_time = time.time()
    remaining_urls = set(frontend_urls)

    logger.info(
        f"Waiting for workers to register with {len(frontend_urls)} frontends..."
    )

    while time.time() - start_time < timeout and remaining_urls:
        newly_ready = set()

        for url in remaining_urls:
            # Calculate remaining timeout for this URL
            elapsed = time.time() - start_time
            remaining_timeout = max(1.0, timeout - elapsed)

            # Use a shorter per-URL timeout to allow checking all URLs
            per_url_timeout = min(10.0, remaining_timeout)

            if wait_for_workers_registered(
                url, model, timeout=per_url_timeout, interval=interval
            ):
                newly_ready.add(url)

        remaining_urls -= newly_ready

        if remaining_urls:
            elapsed = time.time() - start_time
            logger.info(
                f"Waiting for {len(remaining_urls)}/{len(frontend_urls)} frontends... "
                f"({elapsed:.1f}s elapsed)"
            )

    if remaining_urls:
        logger.error(f"Timeout waiting for frontends: {remaining_urls}")
        return False

    logger.info(f"All {len(frontend_urls)} frontends have workers registered!")
    return True


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for the scale test tool.

    Args:
        verbose: If True, set log level to DEBUG; otherwise INFO
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
