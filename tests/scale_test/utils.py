# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from typing import List

import requests

logger = logging.getLogger(__name__)


def wait_for_url_ready(url: str, timeout: float = 30.0, interval: float = 0.5) -> bool:
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
    """Wait for workers to register with a frontend and serve requests."""
    start_time = time.time()
    models_url = f"{frontend_url}/v1/models"
    completions_url = f"{frontend_url}/v1/completions"

    # Wait for model to appear
    logger.info(f"Waiting for model '{model}' at {frontend_url}...")
    model_found = False
    while time.time() - start_time < timeout:
        try:
            response = requests.get(models_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                if model in [m.get("id") for m in models] or len(models) > 0:
                    logger.info(f"Model(s) registered at {frontend_url}")
                    model_found = True
                    break
        except requests.RequestException:
            pass
        time.sleep(interval)

    if not model_found:
        logger.error(f"Timeout waiting for model at {frontend_url}")
        return False

    # Verify completions work
    logger.info(f"Verifying completions at {frontend_url}...")
    test_payload = {"model": model, "prompt": "ping", "max_tokens": 1}

    while time.time() - start_time < timeout:
        try:
            response = requests.post(completions_url, json=test_payload, timeout=30)
            if response.status_code == 200:
                logger.info(f"Completions ready at {frontend_url}")
                return True
        except requests.RequestException:
            pass
        time.sleep(interval)

    logger.error(f"Timeout waiting for completions at {frontend_url}")
    return False


def wait_for_all_workers_registered(
    frontend_urls: List[str],
    model: str,
    timeout: float = 120.0,
    interval: float = 1.0,
) -> bool:
    start_time = time.time()
    remaining_urls = set(frontend_urls)

    logger.info(f"Waiting for workers at {len(frontend_urls)} frontends...")

    while time.time() - start_time < timeout and remaining_urls:
        newly_ready = set()

        for url in remaining_urls:
            elapsed = time.time() - start_time
            remaining_timeout = max(1.0, timeout - elapsed)
            per_url_timeout = min(10.0, remaining_timeout)

            if wait_for_workers_registered(
                url, model, timeout=per_url_timeout, interval=interval
            ):
                newly_ready.add(url)

        remaining_urls -= newly_ready

    if remaining_urls:
        logger.error(f"Timeout waiting for frontends: {remaining_urls}")
        return False

    logger.info(f"All {len(frontend_urls)} frontends have workers registered!")
    return True


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
