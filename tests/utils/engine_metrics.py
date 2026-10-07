# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine observations for cancellation and completed KV transfer checks."""

import json
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import requests

from tests.utils.prometheus import find_metric_samples


@dataclass
class EngineMetrics(ABC):
    url: str
    settle_timeout: float = 10

    def scrape(self) -> str:
        with requests.get(self.url, timeout=2) as response:
            response.raise_for_status()
            return response.text

    @staticmethod
    def samples(
        body: str, name: str, labels: dict[str, str] | None = None
    ) -> list[float]:
        values = find_metric_samples(body, name, labels or {})
        assert values, f"Missing engine metric {name}: {body}"
        assert all(
            math.isfinite(value) and value >= 0 for value in values
        ), f"Invalid engine metric {name}: {values}"
        return values

    @abstractmethod
    def scheduler_counts(self) -> tuple[float, float]:
        """Read running and queued requests from the same metrics scrape."""

    @abstractmethod
    def progress(self) -> float:
        """Read a monotonic measure of generation work."""

    @abstractmethod
    def completion_progress(self, max_tokens: int) -> float:
        """Validate capacity and return a lower bound on full-generation work.

        The test runs one non-speculative request with ignore_eos=True.
        Implementations must account for their counter's units and prefill.
        """

    @abstractmethod
    def assert_recovered(self, *, before: float, max_tokens: int) -> None:
        """Verify the backend's progress counter observed the recovery request."""

    @abstractmethod
    def transfer_progress(self) -> float:
        """Read a monotonic measure of completed KV transfers."""

    def wait_for_transfer(self, *, before: float) -> None:
        deadline = time.monotonic() + self.settle_timeout
        while True:
            progress = self.transfer_progress()
            assert progress >= before, "Completed KV transfer counter reset"
            if progress > before:
                return
            assert time.monotonic() < deadline, "No new completed KV transfer"
            time.sleep(0.05)

    def _wait_for_progress(
        self, *, before: float, minimum: float, maximum: float
    ) -> None:
        deadline = time.monotonic() + self.settle_timeout
        while True:
            delta = self.progress() - before
            assert 0 <= delta <= maximum, f"Unexpected recovery progress: {delta=}"
            if delta >= minimum:
                return
            assert (
                time.monotonic() < deadline
            ), f"Recovery progress did not reach {minimum}: {delta=}"
            time.sleep(0.05)

    def wait_for_scheduler(self, *, is_active: bool = False) -> None:
        deadline = time.monotonic() + self.settle_timeout
        while True:
            running, waiting = self.scheduler_counts()
            is_ready = running > 0 if is_active else running == waiting == 0
            if is_ready:
                return
            assert time.monotonic() < deadline, (
                f"Scheduler is_active={is_active}: running={running}, waiting={waiting}; "
                f"metrics={self.url}"
            )
            time.sleep(0.05)

    def assert_cancelled(self, *, before: float, completion_progress: float) -> None:
        self.wait_for_scheduler()
        # Scrapes are not atomic: read progress again after observing idle.
        delta = self.progress() - before
        assert delta >= 0, f"Engine progress counter reset: {delta}"
        assert delta < completion_progress, (
            f"Engine completed generation instead of cancelling: "
            f"{delta=}, {completion_progress=}"
        )


@dataclass
class VllmMetricsChecker(EngineMetrics):
    transfer_probe: Path | None = None

    def scheduler_counts(self) -> tuple[float, float]:
        body = self.scrape()
        return (
            sum(self.samples(body, "vllm:num_requests_running")),
            sum(self.samples(body, "vllm:num_requests_waiting")),
        )

    def progress(self) -> float:
        # With one non-speculative request, each iteration emits one token.
        return sum(self.samples(self.scrape(), "vllm:iteration_tokens_total_count"))

    def completion_progress(self, max_tokens: int) -> float:
        return max_tokens

    def assert_recovered(self, *, before: float, max_tokens: int) -> None:
        self._wait_for_progress(before=before, minimum=max_tokens, maximum=max_tokens)

    def transfer_progress(self) -> float:
        assert self.transfer_probe is not None, "Missing completed-transfer probe"
        return sum(
            json.loads(line)["bytes"]
            for line in self.transfer_probe.read_text().splitlines()
        )


@dataclass
class SGLangMetricsChecker(EngineMetrics):
    # Idle scheduler gauges may only be published every 30 seconds.
    settle_timeout: float = 40

    def scheduler_counts(self) -> tuple[float, float]:
        body = self.scrape()
        return (
            sum(self.samples(body, "sglang:num_running_reqs")),
            sum(self.samples(body, "sglang:num_queue_reqs")),
        )

    def progress(self) -> float:
        return sum(
            self.samples(
                self.scrape(), "sglang:realtime_tokens_total", {"mode": "decode"}
            )
        )

    def completion_progress(self, max_tokens: int) -> float:
        # Leave room for the short prompt and prevent capacity-driven truncation
        # from looking like successful cancellation.
        body = self.scrape()
        for name, minimum in (
            ("sglang:context_len", 2 * max_tokens),
            ("sglang:max_total_num_tokens", 4 * max_tokens),
        ):
            assert (
                min(self.samples(body, name)) >= minimum
            ), f"Cancellation request may be shortened: {name} must be >= {minimum}"
        # SGLang's prefill supplies the first token; this counter counts decode.
        return max_tokens - 1

    def assert_recovered(self, *, before: float, max_tokens: int) -> None:
        # Overlap scheduling can count an additional iteration after finishing.
        self._wait_for_progress(before=before, minimum=max_tokens - 1, maximum=math.inf)

    def transfer_progress(self) -> float:
        return sum(self.samples(self.scrape(), "sglang:kv_transfer_total_mb_sum"))
