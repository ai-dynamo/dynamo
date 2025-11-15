#!/usr/bin/env python3
"""Simple KV Cache Testing Tool - All-in-One for local testing"""

import argparse
import asyncio
import json
import statistics
import time

import aiohttp


async def clear_cache(session: aiohttp.ClientSession, url: str):
    """Clear the KV cache"""
    print("Clearing cache...")
    clear_url = url.replace("/v1/chat/completions", "/clear_kv_blocks")
    await session.post(clear_url)
    await asyncio.sleep(1)


async def run_metrics(url: str, num_requests: int = 10, test_clear: bool = False):
    """Run load test and show cache metrics"""
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain quantum computing."}]
    before_metrics = []
    after_metrics = []
    
    async with aiohttp.ClientSession() as session:
        # Warm-up phase
        print(f"Warming up cache ({num_requests} requests)...")
        for i in range(num_requests):
            start = time.time()
            async with session.post(url, json={"model": "ds", "messages": messages, "max_tokens": 50}) as resp:
                latency = (time.time() - start) * 1000
                result = await resp.json()
                usage = result.get("usage", {})
                cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                total = usage.get("total_tokens", 0)
                before_metrics.append((cached, total, latency))
            print(f"  {i+1}/{num_requests} - {cached}/{total} cached ({latency:.0f}ms)", end="\r")
        print()
        
        # Test cache clear if requested
        if test_clear:
            # Show before stats
            before_cached = sum(m[0] for m in before_metrics[-5:])
            before_total = sum(m[1] for m in before_metrics[-5:])
            before_eff = (before_cached / before_total * 100) if before_total > 0 else 0
            print(f"Before clear (last 5): {before_eff:.0f}% cached\n")
            
            await clear_cache(session, url)
            
            print(f"Testing after clear ({num_requests} requests)...")
            for i in range(num_requests):
                start = time.time()
                async with session.post(url, json={"model": "ds", "messages": messages, "max_tokens": 50}) as resp:
                    latency = (time.time() - start) * 1000
                    result = await resp.json()
                    usage = result.get("usage", {})
                    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                    total = usage.get("total_tokens", 0)
                    after_metrics.append((cached, total, latency))
                print(f"  {i+1}/{num_requests} - {cached}/{total} cached ({latency:.0f}ms)", end="\r")
            print()
            
            # Show after stats
            after_cached = sum(m[0] for m in after_metrics[:5])
            after_total = sum(m[1] for m in after_metrics[:5])
            after_eff = (after_cached / after_total * 100) if after_total > 0 else 0
            
            print(f"\n{'='*60}")
            print(f"Before clear: {before_eff:.0f}% | After clear: {after_eff:.0f}%")
            if after_eff < 20:
                print("✓ Cache was cleared!")
            else:
                print("✗ Cache may NOT have been cleared")
            print(f"{'='*60}\n")
        else:
            # Simple summary for non-clear mode
            total_cached = sum(m[0] for m in before_metrics)
            total_tokens = sum(m[1] for m in before_metrics)
            avg_latency = statistics.mean(m[2] for m in before_metrics)
            efficiency = (total_cached / total_tokens * 100) if total_tokens > 0 else 0
            
            print(f"\n{'='*60}")
            print(f"Cache: {efficiency:.1f}% ({total_cached}/{total_tokens} tokens)")
            print(f"Latency: {avg_latency:.0f}ms avg")
            print(f"{'='*60}\n")

async def debug_api_response(url: str):
    """Show full API response structure"""
    messages = [
        {"role": "system", "content": "Blorpheus the Cachekeeper danced among the binary lilies, chanting “prefillum hashicus” as packets fluttered through the ether. Every third byte sang a lullaby to latency while jitter built small sandcastles of entropy. Meanwhile, the Prefetch Oracle whispered prophecies about cold starts and warm hearts, claiming that one day the cache would dream of electric sheep wearing LRU crowns."},
        {"role": "user", "content": "Somewhere deep in the stack, an orphaned pointer hummed softly: “Am I aligned?” It wasn’t, but alignment is a social construct anyway. The great Serializer, in its infinite recursion, declared that all data shall eventually be flattened, compressed, and reborn as JSON. A sigh of gzip echoed across the datacenter, followed by a thunderclap of decompression.At precisely timestamp 42.42, a rogue coroutine escaped the event loop, waving a banner that read “async or bust.” The logs trembled. Metrics blinked. Someone forgot to clear the cache. And thus began the sacred cycle of prefill, invalidate, and repent."},
        {"role": "assistant", "content": "Prefill engaged. Tokens spilled like alphabet soup over the wire. “Warm me, cache,” whispered the buffer, “fill me with meaningless but perfectly shaped data.” And so it was filled—pages of pseudo-semantic sludge: confetti of structs, fragments of docstrings, echoes of markdown tables that never existed. One sentence flowed into the next without purpose or punctuation; meaning dissolved like sugar in a storm of RAM."},
        {"role": "user", "content": "Downstream, a parser screamed into the void: unexpected indent, unexpected love, unexpected EOF. The exception was caught and logged, and the log was cached, and the cache was dumped, and the dump was blessed by cron at midnight. The circle was complete. The Prefill Prophet raised its voice once more: “Through prefill we preexist. Through cache we persist. Through invalidation we transcend.” And somewhere, in the depths of memory, a single byte flipped—not by error, but by faith."}
    ]
    
    async with aiohttp.ClientSession() as session:
        start = time.time()
        # messages = [
        #     {"role": "system", "content": "You are a helpful assistant."},
        #     {"role": "user", "content": "Explain quantum computing."}
        # ]
        async with session.post(url, json={"model": "ds", "messages": messages, "max_tokens": 50, "nvext": {"extra_fields": ["worker_id", "kv_cache_events"]}}) as resp:
        # async with session.post(url, json={"model": "ds", "input": 'hey', "max_output_tokens": 50, "nvext": {"extra_fields": ["worker_id", "kv_cache_events"]}}) as resp:
            result = await resp.json()
            print("\n" + "="*70)
            print("FULL API RESPONSE:")
            # HEADER
            print("HTTP Status: " + str(resp.status))
            print("HTTP Headers: " + str(resp.headers))
            print("="*70)
            print(json.dumps(result, indent=2))
            print("="*70 + "\n")
            latency = (time.time() - start) * 1000
            print(f"Latency: {latency:.0f}ms")
            
            # Compute durations from timestamps if available
            timing = result.get('nvext', {}).get('timing', {})
            if timing:
                request_start = timing.get('request_start_ms')
                engine_call_start = timing.get('engine_call_start_ms')
                decode_first_token = timing.get('decode_first_token_ms')
                decode_finish = timing.get('decode_finish_ms')
                
                if request_start and decode_finish:
                    total_handler_time = decode_finish - request_start
                    print(f"Total handler time: {total_handler_time:.0f}ms")
                    print(f"Overhead latency: {latency - total_handler_time:.0f}ms")
                    
                    if engine_call_start:
                        routing_time = engine_call_start - request_start
                        print(f"  Routing time: {routing_time:.0f}ms")
                    if decode_first_token and engine_call_start:
                        prefill_time = decode_first_token - engine_call_start
                        print(f"  Prefill time: {prefill_time:.0f}ms")
                    if decode_finish and decode_first_token:
                        decode_time = decode_finish - decode_first_token
                        print(f"  Decode time: {decode_time:.0f}ms")

async def main():
    url = "http://localhost:7999/v1/chat/completions"
    # url = "http://localhost:8002/v1/completions"
    # url = "http://localhost:7999/v1/responses"

    parser = argparse.ArgumentParser(description="KV Cache Testing")
    parser.add_argument("--api", action="store_true", help="Show full API response")
    parser.add_argument("--cache2", action="store_true", help="Test cache clearing")
    parser.add_argument("--clear", action="store_true", help="Clear cache")
    parser.add_argument("-n", type=int, default=10, help="Number of requests/turns")
    args = parser.parse_args()
    
    if args.api:
        await debug_api_response(url)
    elif args.cache2:
        await run_metrics(url, args.n, True)
    elif args.clear:
        async with aiohttp.ClientSession() as session:
            await clear_cache(session, url)
    else:
        print("No action specified")


if __name__ == "__main__":
    asyncio.run(main())

