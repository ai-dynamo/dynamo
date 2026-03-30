#!/usr/bin/env python3
"""Test prefix cache hit vs miss by mutating reasoning content.

Cache hit: replay the conversation exactly as generated.
Cache miss: prepend a random char to the first reasoning segment of each
assistant turn, changing the token sequence and invalidating the prefix cache.
"""
import requests, json, time, copy, statistics, sys, argparse, uuid


def ttft_streaming(url, msgs, model, followup_msg):
    body = {
        'model': model,
        'max_tokens': 500,
        'stream': True,
        'messages': msgs + [followup_msg],
    }
    headers = {'Content-Type': 'application/json'}
    t0 = time.monotonic()
    r = requests.post(f"{url}/v1/chat/completions", json=body, headers=headers, stream=True, timeout=120)
    for line in r.iter_lines(decode_unicode=True):
        if line and line.startswith('data: ') and line != 'data: [DONE]':
            try:
                d = json.loads(line[6:])
                delta = d.get('choices', [{}])[0].get('delta', {})
                if delta.get('content') or delta.get('reasoning_content'):
                    r.close()
                    return (time.monotonic() - t0) * 1000
            except:
                pass
    r.close()
    return (time.monotonic() - t0) * 1000


def mutate_reasoning(messages):
    """Prepend a unique random string to the first reasoning segment of each assistant turn."""
    out = copy.deepcopy(messages)
    for msg in out:
        if msg.get('role') == 'assistant':
            rc = msg.get('reasoning_content')
            noise = uuid.uuid4().hex[:8] + " "
            if isinstance(rc, list) and rc:
                rc[0] = noise + rc[0]
            elif isinstance(rc, str) and rc:
                msg['reasoning_content'] = noise + rc
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://localhost:8000')
    parser.add_argument('--model', default='nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4')
    parser.add_argument('--trace-file', required=True)
    parser.add_argument('--runs', type=int, default=15)
    parser.add_argument('--jsonl', default=None)
    args = parser.parse_args()

    with open(args.trace_file) as f:
        trace = json.load(f)

    messages = trace['messages']
    print(f"Loaded {len(messages)} messages", file=sys.stderr)

    followup = {'role': 'user', 'content': 'Now rank all 30 cities by population, largest first.'}

    # Exact replay (should cache)
    msgs_exact = copy.deepcopy(messages)

    # Warm the exact variant
    print("Warming exact variant (5 requests)...", file=sys.stderr)
    for _ in range(5):
        ttft_streaming(args.url, msgs_exact, args.model, followup)
        time.sleep(0.3)

    time.sleep(1)

    print(f"\nMeasuring {args.runs} runs (alternating)...", file=sys.stderr)
    exact_times = []
    mutated_times = []

    for i in range(args.runs):
        # Exact replay (cache hit expected)
        e = ttft_streaming(args.url, msgs_exact, args.model, followup)
        time.sleep(0.2)

        # Mutated (cache miss expected — unique reasoning each time)
        msgs_mut = mutate_reasoning(messages)
        m = ttft_streaming(args.url, msgs_mut, args.model, followup)
        time.sleep(0.2)

        exact_times.append(e)
        mutated_times.append(m)
        print(f"  {i+1}: exact={e:.1f}ms  mutated={m:.1f}ms  delta={m-e:+.1f}ms", file=sys.stderr)

    print(f"\nExact:   mean={statistics.mean(exact_times):.1f}ms  stdev={statistics.stdev(exact_times):.1f}ms", file=sys.stderr)
    print(f"Mutated: mean={statistics.mean(mutated_times):.1f}ms  stdev={statistics.stdev(mutated_times):.1f}ms", file=sys.stderr)
    print(f"Delta:   {statistics.mean(mutated_times)-statistics.mean(exact_times):+.1f}ms", file=sys.stderr)

    if args.jsonl:
        with open(args.jsonl, 'w') as f:
            for i in range(args.runs):
                f.write(json.dumps({
                    'run': i+1,
                    'exact_ms': round(exact_times[i], 1),
                    'mutated_ms': round(mutated_times[i], 1),
                    'delta_ms': round(mutated_times[i] - exact_times[i], 1),
                }) + '\n')
        print(f"Saved to {args.jsonl}", file=sys.stderr)


if __name__ == '__main__':
    main()
