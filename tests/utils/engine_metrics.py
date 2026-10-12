# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine observations for cancellation and completed KV transfer checks."""

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import requests

from tests.utils.prometheus import find_metric_samples


@dataclass
class EngineMetrics(ABC):
    port_env: str
    settle_timeout: float = 10
    url: str | None = field(default=None, init=False)

    def bind_environment(self, env: dict[str, str]) -> None:
        """Resolve the engine HTTP port from this deployment's launch settings."""
        self.url = f"http://127.0.0.1:{int(env[self.port_env])}/metrics"

    def scrape(self) -> str:
        if self.url is None:
            raise RuntimeError(
                f"Engine metrics port has not been bound: {self.port_env}"
            )
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
        return sum(
            self.samples(
                self.scrape(),
                "vllm:prompt_tokens_by_source_total",
                {"source": "external_kv_transfer"},
            )
        )


@dataclass
class SGLangMetricsChecker(EngineMetrics):
    minimum_context_length: int | None = field(default=None, kw_only=True)
    minimum_kv_capacity: int | None = field(default=None, kw_only=True)

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
        assert self.minimum_context_length is not None, "Missing context prerequisite"
        assert self.minimum_kv_capacity is not None, "Missing KV capacity prerequisite"
        assert (
            0 < max_tokens < self.minimum_context_length
        ), "Cancellation request needs context space for its prompt"
        assert (
            max_tokens < self.minimum_kv_capacity
        ), "Cancellation request exceeds its KV capacity prerequisite"
        body = self.scrape()
        for name, minimum in (
            ("sglang:context_len", self.minimum_context_length),
            ("sglang:max_total_num_tokens", self.minimum_kv_capacity),
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
