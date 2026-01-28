#!/usr/bin/env python3
"""
Stability test to verify memory stays bounded with MALLOC_ARENA_MAX=2.

Runs repeated load cycles and measures memory between runs to detect growth.
"""

import argparse
import asyncio
import subprocess
import time
import os
import signal
import sys
from datetime import datetime, timedelta

# Import the load test module
import aiohttp


async def run_load_cycle(session, url, payload_size, concurrency, requests):
    """Run a single load test cycle."""
    payload = "x" * payload_size
    completed = 0
    errors = 0
    semaphore = asyncio.Semaphore(concurrency)

    async def make_request():
        nonlocal completed, errors
        async with semaphore:
            try:
                data = {
                    "model": "mock_model",
                    "messages": [{"role": "user", "content": payload}],
                    "max_tokens": 10,
                    "stream": True,
                }
                async with session.post(url, json=data) as resp:
                    async for _ in resp.content:
                        pass
                    completed += 1
            except Exception as e:
                errors += 1

    tasks = [make_request() for _ in range(requests)]
    await asyncio.gather(*tasks)
    return completed, errors


def get_process_memory(pid):
    """Get RSS memory in MB for a process."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # KB to MB
    except:
        return None
    return None


def get_frontend_pid():
    """Find the frontend.py process PID - the one with highest memory."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "python frontend.py"],
            capture_output=True,
            text=True
        )
        pids = result.stdout.strip().split()
        if not pids:
            return None

        # Find the PID with highest RSS (the actual frontend, not a shell wrapper)
        max_rss = 0
        best_pid = None
        for pid_str in pids:
            try:
                pid = int(pid_str)
                rss = get_process_memory(pid)
                if rss and rss > max_rss:
                    max_rss = rss
                    best_pid = pid
            except:
                continue

        return best_pid if best_pid else int(pids[0])
    except:
        return None


async def main():
    parser = argparse.ArgumentParser(description="Memory stability test")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in minutes")
    parser.add_argument("--payload-size", type=int, default=200000, help="Payload size in chars")
    parser.add_argument("--concurrency", type=int, default=96, help="Concurrent requests")
    parser.add_argument("--requests", type=int, default=1000, help="Requests per cycle")
    parser.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    args = parser.parse_args()

    print("=" * 70)
    print("MEMORY STABILITY TEST")
    print("=" * 70)
    print(f"Duration:    {args.duration} minutes")
    print(f"Payload:     {args.payload_size:,} chars")
    print(f"Concurrency: {args.concurrency}")
    print(f"Requests:    {args.requests} per cycle")
    print("=" * 70)
    print()

    # Find frontend PID
    frontend_pid = get_frontend_pid()
    if not frontend_pid:
        print("ERROR: Could not find frontend.py process")
        sys.exit(1)
    print(f"Frontend PID: {frontend_pid}")

    # Get initial memory
    initial_memory = get_process_memory(frontend_pid)
    print(f"Initial RSS:  {initial_memory:.1f} MB")
    print()

    # Track memory over time
    memory_readings = [(0, initial_memory)]
    cycle_count = 0
    start_time = time.time()
    end_time = start_time + (args.duration * 60)

    print(f"{'Cycle':<8} {'Time':<10} {'RSS (MB)':<12} {'Delta':<12} {'Req/s':<10} {'Errors':<8}")
    print("-" * 70)

    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)
    timeout = aiohttp.ClientTimeout(total=60)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        while time.time() < end_time:
            cycle_count += 1
            cycle_start = time.time()

            # Run load cycle
            completed, errors = await run_load_cycle(
                session, args.url, args.payload_size, args.concurrency, args.requests
            )

            cycle_duration = time.time() - cycle_start
            elapsed_minutes = (time.time() - start_time) / 60

            # Measure memory after cycle
            current_memory = get_process_memory(frontend_pid)
            if current_memory is None:
                print(f"ERROR: Frontend process died!")
                break

            delta = current_memory - initial_memory
            req_per_sec = completed / cycle_duration if cycle_duration > 0 else 0

            memory_readings.append((elapsed_minutes, current_memory))

            print(f"{cycle_count:<8} {elapsed_minutes:>6.1f}m    {current_memory:>8.1f}    {delta:>+8.1f}     {req_per_sec:>6.1f}     {errors}")

            # Brief pause between cycles
            await asyncio.sleep(2)

    # Final summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    final_memory = memory_readings[-1][1]
    total_delta = final_memory - initial_memory
    min_memory = min(m for _, m in memory_readings)
    max_memory = max(m for _, m in memory_readings)

    print(f"Total cycles:     {cycle_count}")
    print(f"Initial RSS:      {initial_memory:.1f} MB")
    print(f"Final RSS:        {final_memory:.1f} MB")
    print(f"Total delta:      {total_delta:+.1f} MB")
    print(f"Min RSS:          {min_memory:.1f} MB")
    print(f"Max RSS:          {max_memory:.1f} MB")
    print(f"Range:            {max_memory - min_memory:.1f} MB")
    print()

    # Analyze trend
    if len(memory_readings) >= 3:
        first_third = memory_readings[:len(memory_readings)//3]
        last_third = memory_readings[-len(memory_readings)//3:]
        avg_first = sum(m for _, m in first_third) / len(first_third)
        avg_last = sum(m for _, m in last_third) / len(last_third)
        trend = avg_last - avg_first

        print(f"Avg (first 1/3):  {avg_first:.1f} MB")
        print(f"Avg (last 1/3):   {avg_last:.1f} MB")
        print(f"Trend:            {trend:+.1f} MB")
        print()

        if abs(trend) < 50:
            print("✓ STABLE: Memory appears bounded (trend < 50 MB)")
        elif trend > 0:
            print(f"⚠ WARNING: Memory trending upward ({trend:+.1f} MB)")
        else:
            print(f"✓ GOOD: Memory trending downward ({trend:+.1f} MB)")

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
