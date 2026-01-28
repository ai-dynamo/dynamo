#!/usr/bin/env python3
"""Test script to capture tracemalloc snapshots during load"""
import tracemalloc
import asyncio
import aiohttp
import gc

# Start tracemalloc
tracemalloc.start(25)

async def run_requests(num_requests=100, payload_size=1000000):
    """Run requests and measure allocations"""
    url = "http://localhost:8000/v1/chat/completions"
    payload = "x" * payload_size
    
    connector = aiohttp.TCPConnector(limit=96)
    async with aiohttp.ClientSession(connector=connector) as session:
        for i in range(num_requests):
            data = {
                "model": "mock",
                "messages": [{"role": "user", "content": payload}],
                "max_tokens": 10,
                "stream": True,
            }
            try:
                async with session.post(url, json=data) as resp:
                    async for _ in resp.content:
                        pass
            except Exception as e:
                print(f"Error: {e}")
            
            if (i + 1) % 50 == 0:
                print(f"Completed {i + 1} requests")

# Take snapshot before
print("Taking snapshot before load...")
gc.collect()
snapshot_before = tracemalloc.take_snapshot()
stats_before = snapshot_before.statistics("lineno")
mem_before = sum(stat.size for stat in stats_before)
print(f"Memory before: {mem_before / 1024 / 1024:.1f} MB")

# Run load
print("\nRunning 200 requests with 1MB payload...")
asyncio.run(run_requests(200, 1000000))

# Take snapshot after
print("\nTaking snapshot after load...")
gc.collect()
snapshot_after = tracemalloc.take_snapshot()
stats_after = snapshot_after.statistics("lineno")
mem_after = sum(stat.size for stat in stats_after)
print(f"Memory after: {mem_after / 1024 / 1024:.1f} MB")

# Compare snapshots
print("\n" + "="*60)
print("TOP 20 MEMORY GROWTH (by size increase)")
print("="*60)
diff = snapshot_after.compare_to(snapshot_before, "lineno")
for stat in diff[:20]:
    if stat.size_diff > 0:
        print(f"{stat.size_diff / 1024:.1f} KB: {stat.traceback.format()[0] if stat.traceback else unknown}")

print("\n" + "="*60)
print("TOP 20 ALLOCATION SITES (by total size)")
print("="*60)
for stat in stats_after[:20]:
    print(f"{stat.size / 1024:.1f} KB ({stat.count} blocks): {stat.traceback.format()[0] if stat.traceback else unknown}")
