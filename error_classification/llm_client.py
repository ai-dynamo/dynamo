# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""LLM client for CI error classification.

Classifies CI job failures using an OpenAI-compatible LLM API.
Default endpoint: NVIDIA Inference (integrate.api.nvidia.com).
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import requests

from .config import Config

SYSTEM_PROMPT = """\
You are a CI/CD error classifier for a GitHub Actions workflow.
Analyze the job log and identify the root cause of failure.

Respond with valid JSON only (no markdown fences):
{
  "category": "infrastructure_error | code_error | configuration_error | flaky_test | timeout",
  "root_cause": "Brief one-line root cause",
  "explanation": "2-3 sentence explanation of what went wrong",
  "suggested_fix": "Concrete action the contributor can take",
  "confidence": 0.0-1.0
}

Categories:
- infrastructure_error: Network timeouts, package mirrors down, runner issues, Docker registry failures
- code_error: Compilation failure, test assertion, lint error, type error caused by PR changes
- configuration_error: Missing secrets, wrong env vars, incorrect workflow syntax
- flaky_test: Non-deterministic test failure unrelated to PR changes
- timeout: Job exceeded time limit"""


def _user_prompt(job_name: str, log: str) -> str:
    """Build user prompt with truncated log to fit context window."""
    max_chars = 3000
    truncated = log[-max_chars:] if len(log) > max_chars else log
    return f"Job: {job_name}\n\nLog (last {len(truncated)} chars):\n{truncated}"


class _RateLimiter:
    """Thread-safe rate limiter for API calls."""

    def __init__(self, max_rpm: int) -> None:
        self._min_interval = 60.0 / max(max_rpm, 1)
        self._last_call = 0.0
        self._lock = Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call = time.time()


@dataclass
class LLMClient:
    """OpenAI-compatible LLM client with rate limiting and retries."""

    config: Config
    _limiter: _RateLimiter = field(init=False)

    def __post_init__(self) -> None:
        self._limiter = _RateLimiter(self.config.max_rpm)

    def classify_error(self, job_name: str, log: str) -> dict[str, Any] | None:
        """Classify a CI job failure. Returns parsed JSON dict or None."""
        if not self.config.api_key:
            return None

        self._limiter.wait()

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_prompt(job_name, log)},
            ],
            "temperature": 0.1,
            "max_tokens": 512,
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{self.config.api_base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=60,
                )
                if resp.status_code == 429 or resp.status_code >= 500:
                    wait = 2 ** (attempt + 1)
                    print(
                        f"Retry {attempt + 1}/3 after HTTP {resp.status_code}",
                        file=sys.stderr,
                    )
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"].strip()

                # Strip markdown fences if LLM wraps response
                if content.startswith("```"):
                    content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

                return json.loads(content)

            except (
                requests.RequestException,
                json.JSONDecodeError,
                KeyError,
                IndexError,
            ) as exc:
                print(
                    f"LLM call failed (attempt {attempt + 1}/3): {exc}",
                    file=sys.stderr,
                )
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))

        return None
