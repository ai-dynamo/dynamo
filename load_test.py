#!/usr/bin/env python3
"""
Simple, memory-bounded load test for the repro harness.

This is modeled after PR #5269's `pipeline_openai/load_test.py`, but with CLI flags:
  --concurrency, --durations, --payload-size, --base-url, --wait-period

Supports multiple "bumps" of traffic with idle periods between them:
  --durations 60,60,60  # Three 60-second load bursts
  --wait-period 30      # 30 seconds idle between each burst

Each duration runs until all in-flight requests complete before starting the wait period.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import time
import traceback
from collections import deque
from typing import Deque, Optional

import aiohttp


@functools.lru_cache(maxsize=1)
def _payload_text(payload_size: int) -> str:
    return "Hello from request. " * payload_size


def _powers_of_two_in_range(lo: int, hi: int) -> list[int]:
    """
    Return powers of 2 in [lo, hi] as an up-then-down cycle.

    Example: _powers_of_two_in_range(3, 20) -> [4, 8, 16, 8, 4]
    """
    if lo > hi or lo < 1:
        return []

    # Find powers of two within bounds.
    pows: list[int] = []
    p = 1
    while p <= hi:
        if p >= lo:
            pows.append(p)
        p *= 2

    if not pows:
        return []

    # Build up-then-down cycle (exclude endpoints on the way down to avoid duplicates).
    if len(pows) == 1:
        return pows
    return pows + list(reversed(pows[1:-1]))


class _DynamicConcurrencyLimiter:
    """A small async limiter with a dynamically adjustable limit."""

    def __init__(self, limit: int):
        self._limit = max(1, int(limit))
        self._in_flight = 0
        self._cond = asyncio.Condition()

    @property
    def limit(self) -> int:
        return self._limit

    def set_limit(self, new_limit: int) -> None:
        # Called from the event loop; keep it simple.
        self._limit = max(1, int(new_limit))

        # Wake waiters so they can re-check limit.
        async def _notify() -> None:
            async with self._cond:
                self._cond.notify_all()

        asyncio.create_task(_notify())

    async def acquire(self) -> None:
        async with self._cond:
            while self._in_flight >= self._limit:
                await self._cond.wait()
            self._in_flight += 1

    async def release(self) -> None:
        async with self._cond:
            self._in_flight -= 1
            self._cond.notify_all()


class LoadTest:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session: Optional[aiohttp.ClientSession] = None

        self.request_times: Deque[float] = deque(maxlen=10_000)
        self.errors: Deque[str] = deque(maxlen=1_000)
        self.total_requests = 0
        self.total_errors = 0

    async def start(self) -> None:
        connector = aiohttp.TCPConnector(
            limit=0,
            limit_per_host=2000,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True,
            force_close=False,
            ssl=False,
        )

        timeout = aiohttp.ClientTimeout(
            total=120,
            connect=10,
            sock_read=30,
        )

        self.session = aiohttp.ClientSession(
            connector=connector, timeout=timeout, headers={"Connection": "keep-alive"}
        )

    async def stop(self) -> None:
        if self.session:
            await self.session.close()
            self.session = None

    async def test_connection(self) -> bool:
        assert self.session is not None
        try:
            async with self.session.get(
                f"{self.base_url}/v1/models", timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    print(f"✓ Frontend is accessible at {self.base_url}")
                    return True
                print(f"✗ Frontend returned status {response.status}")
                return False
        except Exception as e:
            print(f"✗ Cannot connect to frontend at {self.base_url}: {e}")
            return False

    async def send_request(self, request_id: int, payload_size: int) -> None:
        assert self.session is not None
        start_time = time.time()

        payload = {
            "model": "mock_model",
            "messages": [
                {"role": "user", "content": _payload_text(payload_size)},
                {"role": "user", "content": f"request id {request_id}"},
            ],
            "max_tokens": 2,
        }
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )

        # Guardrail: NATS request plane has a 16MB payload limit (documented).
        # This check estimates the HTTP request size; the internal dyn RPC payload will be similar.
        if len(body) > 15 * 1024 * 1024:
            raise RuntimeError(
                f"Payload JSON is {len(body)/1024/1024:.2f}MB. "
                "This is likely to exceed NATS request-plane limits (~16MB). "
                "Reduce --payload-size or switch request plane to TCP for larger payloads."
            )

        try:
            async with self.session.post(
                f"{self.base_url}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status == 200:
                    # Consume streaming response without storing chunks.
                    async for line in response.content:
                        if line.strip():
                            pass

                    end_time = time.time()
                    self.request_times.append(end_time - start_time)
                    self.total_requests += 1

                    if request_id % 1000 == 0:
                        recent = list(self.request_times)[-100:] or [
                            end_time - start_time
                        ]
                        print(
                            f"Request {request_id} completed in {end_time - start_time:.3f}s "
                            f"(avg last {len(recent)}: {sum(recent)/len(recent):.3f}s)"
                        )
                else:
                    self.errors.append(
                        f"Request {request_id}: HTTP {response.status} {await response.text()}"
                    )
                    self.total_errors += 1
        except Exception as e:
            self.errors.append(f"Request {request_id}: {str(e)}")
            self.total_errors += 1
            if self.total_errors % 100 == 0:
                print(f"Total error count: {self.total_errors}")

    async def run_for_duration(
        self,
        *,
        duration_seconds: Optional[float],
        concurrency: int,
        payload_size: int,
        burst_num: int,
    ) -> int:
        """
        Run load test for a specified duration.
        Returns the number of requests sent in this burst.
        Waits for all in-flight requests to complete before returning.
        """
        if duration_seconds is None:
            banner = "RUN-FOREVER"
            duration_str = "∞"
        else:
            banner = f"BURST {burst_num}"
            duration_str = f"{duration_seconds}s"

        print(
            f"\n{'='*60}\n"
            f"{banner}: Starting {duration_str} load test "
            f"(concurrency={concurrency}, payload_size={payload_size})\n"
            f"{'='*60}"
        )

        sem = asyncio.Semaphore(concurrency)
        started = time.time()
        end_time = None if duration_seconds is None else started + duration_seconds
        request_id_offset = self.total_requests + self.total_errors
        burst_requests = 0
        active_tasks: set[asyncio.Task[None]] = set()
        last_report = started

        async def bounded(req_id: int) -> None:
            async with sem:
                await self.send_request(req_id, payload_size)

        def should_continue() -> bool:
            return True if end_time is None else (time.time() < end_time)

        try:
            # Keep spawning requests until duration expires (or forever).
            while should_continue():
                # Clean up completed tasks
                done_tasks = {t for t in active_tasks if t.done()}
                active_tasks -= done_tasks

                # Spawn new requests up to concurrency limit
                while len(active_tasks) < concurrency and should_continue():
                    req_id = request_id_offset + burst_requests
                    task = asyncio.create_task(bounded(req_id))
                    active_tasks.add(task)
                    burst_requests += 1

                now = time.time()
                if now - last_report >= 10.0:
                    elapsed = now - started
                    rate = burst_requests / elapsed if elapsed > 0 else 0.0
                    print(
                        f"{banner}: {burst_requests} requests sent, "
                        f"rate={rate:.1f} req/s, in-flight={len(active_tasks)}, "
                        f"ok={self.total_requests}, err={self.total_errors}"
                    )
                    last_report = now

                # Small sleep to avoid busy loop
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            # Ensure we don't leave a pile of tasks running if the load test is cancelled.
            for t in active_tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*active_tasks, return_exceptions=True)
            raise

        # Wait for all in-flight requests to complete (finite mode only).
        if end_time is not None and active_tasks:
            print(
                f"Burst {burst_num}: Duration ended, waiting for {len(active_tasks)} in-flight requests..."
            )
            await asyncio.gather(*active_tasks, return_exceptions=True)

        burst_time = time.time() - started
        if end_time is not None:
            print(
                f"Burst {burst_num} completed: {burst_requests} requests in {burst_time:.2f}s "
                f"(rate={burst_requests/burst_time:.1f} req/s)"
            )

        return burst_requests

    async def run_forever(self, *, concurrency: int, payload_size: int) -> None:
        """Run indefinitely, maintaining a fixed concurrency, until interrupted."""
        await self.run_for_duration(
            duration_seconds=None,
            concurrency=concurrency,
            payload_size=payload_size,
            burst_num=1,
        )

    async def run_forever_cycling_concurrency(
        self,
        *,
        payload_size: int,
        cycle_period_seconds: float,
        min_concurrency: int,
        max_concurrency: int,
    ) -> None:
        """
        Run indefinitely and vary the target concurrency over time to simulate bursty traffic.

        Concurrency cycles through powers of two within [min_concurrency, max_concurrency],
        going up and then back down (e.g. 2,4,...,512,256,...,4) and repeats.
        """
        # Compute powers of two within the given bounds.
        cycle = _powers_of_two_in_range(min_concurrency, max_concurrency)
        if not cycle:
            raise ValueError(
                f"No powers of two in range [{min_concurrency}, {max_concurrency}]"
            )

        limiter = _DynamicConcurrencyLimiter(cycle[0])
        started = time.time()
        request_id_offset = self.total_requests + self.total_errors
        requests_sent = 0
        active_tasks: set[asyncio.Task[None]] = set()

        print(
            f"\n{'='*60}\n"
            f"RUN-FOREVER (cycling concurrency)\n"
            f"  Range: {min_concurrency}..{max_concurrency}\n"
            f"  Cycle: {cycle}\n"
            f"  Switch every: {cycle_period_seconds/60:.1f} minutes\n"
            f"{'='*60}"
        )

        async def bounded(req_id: int) -> None:
            await limiter.acquire()
            try:
                await self.send_request(req_id, payload_size)
            finally:
                await limiter.release()

        async def cycle_task() -> None:
            idx = 0
            while True:
                current = cycle[idx % len(cycle)]
                limiter.set_limit(current)
                elapsed_min = (time.time() - started) / 60.0
                print(
                    f"Concurrency shift: target={current} "
                    f"(elapsed={elapsed_min:.1f} min, ok={self.total_requests}, err={self.total_errors})"
                )
                idx += 1
                await asyncio.sleep(cycle_period_seconds)

        cycler = asyncio.create_task(cycle_task())
        last_report = started
        try:
            while True:
                done_tasks = {t for t in active_tasks if t.done()}
                active_tasks -= done_tasks

                # Spawn new requests; limiter enforces current target.
                while len(active_tasks) < limiter.limit:
                    req_id = request_id_offset + requests_sent
                    task = asyncio.create_task(bounded(req_id))
                    active_tasks.add(task)
                    requests_sent += 1

                now = time.time()
                if now - last_report >= 10.0:
                    elapsed = now - started
                    rate = requests_sent / elapsed if elapsed > 0 else 0.0
                    print(
                        f"RUN-FOREVER: sent={requests_sent}, rate={rate:.1f} req/s, "
                        f"in-flight={len(active_tasks)}, target={limiter.limit}, "
                        f"ok={self.total_requests}, err={self.total_errors}"
                    )
                    last_report = now

                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            cycler.cancel()
            await asyncio.gather(cycler, return_exceptions=True)
            for t in active_tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*active_tasks, return_exceptions=True)
            raise
        finally:
            cycler.cancel()
            await asyncio.gather(cycler, return_exceptions=True)

    async def run_multi_burst(
        self,
        *,
        durations: list[float],
        wait_period: float,
        concurrencies: list[int],
        payload_size: int,
    ) -> None:
        """
        Run multiple bursts of load with idle periods between them.
        Each burst can have a different concurrency level.
        """
        burst_info = ", ".join(f"{d}s@c{c}" for d, c in zip(durations, concurrencies))
        print(
            f"\n{'#'*60}\n"
            f"MULTI-BURST LOAD TEST\n"
            f"  Bursts: {burst_info}\n"
            f"  Wait period: {wait_period}s between bursts\n"
            f"  Payload size: {payload_size}\n"
            f"{'#'*60}"
        )

        overall_start = time.time()
        total_requests_sent = 0

        for i, (duration, concurrency) in enumerate(zip(durations, concurrencies), 1):
            burst_requests = await self.run_for_duration(
                duration_seconds=duration,
                concurrency=concurrency,
                payload_size=payload_size,
                burst_num=i,
            )
            total_requests_sent += burst_requests

            # Wait period after each burst (except the last)
            if i < len(durations):
                print(f"\n>>> IDLE PERIOD: Waiting {wait_period}s before next burst...")
                await asyncio.sleep(wait_period)
                print(f">>> IDLE PERIOD: Complete, starting burst {i+1}\n")

        # Final summary
        overall_time = time.time() - overall_start
        print(
            f"\n{'#'*60}\n"
            f"MULTI-BURST LOAD TEST COMPLETE\n"
            f"{'#'*60}\n"
            f"Total requests sent: {total_requests_sent}\n"
            f"Successful requests: {self.total_requests}\n"
            f"Total errors: {self.total_errors}\n"
            f"Total time (including wait periods): {overall_time:.2f}s\n"
        )
        if self.request_times:
            print(
                f"Average response time: {sum(self.request_times) / len(self.request_times):.3f}s"
            )
        if self.errors:
            print(f"\nLast 10 errors (of {self.total_errors} total):")
            for err in list(self.errors)[-10:]:
                print(f"  {err}")

    async def run(
        self, *, request_count: int, concurrency: int, payload_size: int
    ) -> None:
        """Legacy method for backwards compatibility with --request-count."""
        print(
            f"Starting load test: {request_count} requests with concurrency={concurrency}, "
            f"payload_size={payload_size}"
        )
        sem = asyncio.Semaphore(concurrency)
        started = time.time()

        async def bounded(i: int) -> None:
            async with sem:
                await self.send_request(i, payload_size)

        batch_size = 500
        for batch_start in range(0, request_count, batch_size):
            batch_end = min(batch_start + batch_size, request_count)
            tasks = [
                asyncio.create_task(bounded(i)) for i in range(batch_start, batch_end)
            ]
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                # Help GC by removing task refs.
                for t in tasks:
                    if not t.done():
                        t.cancel()
                del tasks

            if batch_end % 5000 == 0:
                elapsed = time.time() - started
                rate = batch_end / elapsed if elapsed > 0 else 0.0
                print(
                    f"Progress: {batch_end}/{request_count} requests, rate={rate:.1f} req/s "
                    f"(ok={self.total_requests}, err={self.total_errors})"
                )

        total_time = time.time() - started
        print("\nLoad test completed!")
        print(f"Total requests sent: {request_count}")
        print(f"Successful requests: {self.total_requests}")
        print(f"Total errors: {self.total_errors}")
        print(f"Total time: {total_time:.2f}s")
        if self.request_times:
            print(
                f"Average response time: {sum(self.request_times) / len(self.request_times):.3f}s"
            )
        if self.errors:
            print(f"\nLast 10 errors (of {self.total_errors} total):")
            for err in list(self.errors)[-10:]:
                print(f"  {err}")


def parse_durations(value: str) -> list[float]:
    """Parse comma-separated durations (e.g., '60,60,60' -> [60.0, 60.0, 60.0])"""
    return [float(d.strip()) for d in value.split(",")]


def parse_concurrency(value: str) -> list[int]:
    """Parse comma-separated concurrency values (e.g., '64,128,256' -> [64, 128, 256])"""
    return [int(c.strip()) for c in value.split(",")]


async def _main() -> None:
    ap = argparse.ArgumentParser(
        description="Load test with support for multiple traffic bursts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single burst (legacy mode):
  python3 load_test.py --concurrency 128 --request-count 5000 --payload-size 1000

  # Run forever at fixed concurrency:
  python3 load_test.py --concurrency 128 --run-forever --payload-size 1000

  # Run forever cycling concurrency (bursty network simulation):
  python3 load_test.py --run-forever --cycle-concurrency --min-concurrency 2 --max-concurrency 1024 --payload-size 1000

  # Multi-burst mode (3 bursts of 60s each, same concurrency, 30s idle between):
  python3 load_test.py --concurrency 128 --durations 60,60,60 --wait-period 30 --payload-size 1000

  # Multi-burst with different concurrency per burst (ramp up pattern):
  python3 load_test.py --concurrency 64,128,256 --durations 60,60,60 --wait-period 30 --payload-size 1000

  # Two bumps with longer idle to observe RSS:
  python3 load_test.py --concurrency 128 --durations 120,120 --wait-period 60 --payload-size 1000
        """,
    )
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument(
        "--concurrency",
        type=parse_concurrency,
        default=None,
        help="Max concurrent requests. Single value or comma-separated per burst (e.g., '128' or '64,128,256'). "
        "Not required when using --cycle-concurrency.",
    )
    ap.add_argument(
        "--payload-size",
        type=int,
        required=True,
        help="Size multiplier for request payload",
    )

    # Two modes: --request-count (legacy) or --durations (new multi-burst)
    mode_group = ap.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--request-count",
        type=int,
        help="Total number of requests to send (legacy single-burst mode)",
    )
    mode_group.add_argument(
        "--durations",
        type=parse_durations,
        help="Comma-separated durations in seconds for each burst (e.g., '60,60,60')",
    )
    mode_group.add_argument(
        "--run-forever",
        action="store_true",
        help="Run indefinitely, maintaining a fixed concurrency (requires single --concurrency value)",
    )

    ap.add_argument(
        "--wait-period",
        type=float,
        default=30.0,
        help="Seconds to wait between bursts (default: 30). Only used with --durations.",
    )
    ap.add_argument(
        "--cycle-concurrency",
        action="store_true",
        help="When used with --run-forever, cycle through powers-of-two between --min-concurrency and --max-concurrency.",
    )
    ap.add_argument(
        "--min-concurrency",
        type=int,
        default=2,
        help="Lower bound for cycling concurrency (default: 2). Used with --cycle-concurrency.",
    )
    ap.add_argument(
        "--max-concurrency",
        type=int,
        default=1024,
        help="Upper bound for cycling concurrency (default: 1024). Used with --cycle-concurrency.",
    )
    ap.add_argument(
        "--cycle-period-seconds",
        type=float,
        default=600.0,
        help="Seconds between concurrency changes (default: 600 = 10 minutes). Used with --cycle-concurrency.",
    )

    args = ap.parse_args()

    # Validate --concurrency is provided when needed
    if args.cycle_concurrency:
        # --concurrency not required for cycling mode
        pass
    elif args.concurrency is None:
        ap.error("--concurrency is required (unless using --cycle-concurrency)")

    # Validate concurrency matches durations if using multi-burst mode
    if args.durations:
        if args.concurrency is None:
            ap.error("--concurrency is required with --durations")
        if len(args.concurrency) == 1:
            # Single concurrency value - expand to all bursts
            args.concurrency = args.concurrency * len(args.durations)
        elif len(args.concurrency) != len(args.durations):
            ap.error(
                f"--concurrency has {len(args.concurrency)} values but --durations has "
                f"{len(args.durations)} values. Provide either one concurrency for all bursts "
                f"or one per burst."
            )

    # Validate run-forever mode
    if args.run_forever and not args.cycle_concurrency:
        if args.concurrency is None or len(args.concurrency) != 1:
            ap.error(
                "--run-forever requires a single --concurrency value (e.g. --concurrency 128)"
            )
    if args.cycle_concurrency and not args.run_forever:
        ap.error("--cycle-concurrency can only be used with --run-forever")

    # Validate cycling bounds
    if args.cycle_concurrency:
        if args.min_concurrency < 1 or args.max_concurrency < 1:
            ap.error("--min-concurrency and --max-concurrency must be >= 1")
        if args.min_concurrency > args.max_concurrency:
            ap.error("--min-concurrency must be <= --max-concurrency")
        cycle = _powers_of_two_in_range(args.min_concurrency, args.max_concurrency)
        if not cycle:
            ap.error(
                f"No powers of two in range [{args.min_concurrency}, {args.max_concurrency}]"
            )

    tester = LoadTest(args.base_url)
    try:
        await tester.start()
        if not await tester.test_connection():
            print("Cannot connect to frontend. Make sure it's running.")
            return

        if args.durations:
            # New multi-burst mode
            await tester.run_multi_burst(
                durations=args.durations,
                wait_period=args.wait_period,
                concurrencies=args.concurrency,  # list of concurrencies
                payload_size=args.payload_size,
            )
        elif args.run_forever:
            if args.cycle_concurrency:
                await tester.run_forever_cycling_concurrency(
                    payload_size=args.payload_size,
                    cycle_period_seconds=args.cycle_period_seconds,
                    min_concurrency=args.min_concurrency,
                    max_concurrency=args.max_concurrency,
                )
            else:
                await tester.run_forever(
                    concurrency=args.concurrency[0],
                    payload_size=args.payload_size,
                )
        else:
            # Legacy single-burst mode (use first concurrency value)
            await tester.run(
                request_count=args.request_count,
                concurrency=args.concurrency[0],
                payload_size=args.payload_size,
            )
    except KeyboardInterrupt:
        print("\nLoad test interrupted")
    except Exception as e:
        print(f"Load test failed: {e}")
        traceback.print_exc()
    finally:
        await tester.stop()


if __name__ == "__main__":
    asyncio.run(_main())
