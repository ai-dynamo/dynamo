# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import random
import time
from datetime import datetime

import requests


def _get_random_prompt(length):
    word_list = [f"{i}" for i in range(10)]
    return " ".join(random.choices(word_list, k=length))


def _single_request(
    url,
    payload,
    logger,
    retry_attempts=1,
    input_token_length=100,
    output_token_length=100,
    timeout=30,
    retry_delay=1,
):
    prompt = _get_random_prompt(input_token_length)
    payload["messages"][0]["content"] = prompt
    payload["max_tokens"] = output_token_length
    response = None
    end_time = None
    start_time = time.time()
    results = []

    # Convert retries to total attempts (1 initial attempt + N retries)
    attempts_remaining = 1 + max(0, retry_attempts)

    while attempts_remaining:
        start_request_time = time.time()

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=timeout,
            )
            end_time = time.time()

            content = None

            try:
                content = response.json()
            except json.JSONDecodeError:
                pass

            results.append(
                {
                    "status": response.status_code,
                    "result": content,
                    "request_elapsed_time": end_time - start_request_time,
                }
            )

            # Success - exit immediately
            if response.status_code == 200:
                break

            # Failure - retry if we have attempts left
            attempts_remaining -= 1
            if attempts_remaining == 0:
                break
            time.sleep(retry_delay)

        except (requests.RequestException, requests.Timeout) as e:
            results.append(
                {
                    "status": str(e),
                    "result": None,
                    "request_elapsed_time": time.time() - start_request_time,
                }
            )

            # Exception - retry if we have attempts left
            attempts_remaining -= 1
            if attempts_remaining == 0:
                break
            logger.warning("Retrying due to Request failed: %s", e)
            time.sleep(retry_delay)

    return {
        "time": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "results": results,
        "total_time": time.time() - start_time,
    }


def client(
    deployment_graph,
    server_process,
    payload,
    log_dir,
    index,
    requests_per_client,
    input_token_length,
    output_token_length,
    max_retries,
    retry_delay=1,
):
    logger = logging.getLogger(f"CLIENT: {index}")

    try:
        log_path = os.path.join(log_dir, f"client_{index}.log.txt")
        with open(log_path, "w") as log:
            url = f"http://localhost:{server_process.port}/{deployment_graph.endpoints[0]}"

            for i in range(requests_per_client):
                result = _single_request(
                    url,
                    payload.payload_chat,
                    logger,
                    max_retries,
                    input_token_length=input_token_length,
                    output_token_length=output_token_length,
                    retry_delay=retry_delay,
                )
                logger.info(
                    f"Request: {i} Status: {result['results'][-1]['status']} Latency: {result['results'][-1]['request_elapsed_time']}"
                )

                log.write(json.dumps(result) + "\n")
                log.flush()
    except Exception as e:
        logger.error(str(e))
    logger.info("Exiting")
