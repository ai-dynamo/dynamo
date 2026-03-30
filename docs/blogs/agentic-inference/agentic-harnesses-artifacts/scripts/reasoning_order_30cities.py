#!/usr/bin/env python3
"""Section 2: Reasoning order experiment with 30-city golden trace.

Takes the 62-message conversation from golden-trace-30cities.json,
creates correct (segmented) vs incorrect (flattened) reconstructions,
adds a follow-up, measures TTFT.
"""
import requests, json, time, copy, statistics, sys, argparse

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

    # Count reasoning
    total_reasoning = 0
    for msg in messages:
        if msg.get('role') == 'assistant':
            rc = msg.get('reasoning_content', '')
            if isinstance(rc, list):
                total_reasoning += sum(len(s) for s in rc)
            elif rc:
                total_reasoning += len(rc)
    print(f"Total reasoning chars: {total_reasoning}", file=sys.stderr)

    # Correct: keep segments as-is
    msgs_correct = copy.deepcopy(messages)

    # Incorrect: flatten all reasoning_content lists to single strings
    msgs_incorrect = copy.deepcopy(messages)
    for msg in msgs_incorrect:
        if msg.get('role') == 'assistant':
            rc = msg.get('reasoning_content')
            if isinstance(rc, list):
                msg['reasoning_content'] = '\n'.join(rc)

    followup = {'role': 'user', 'content': 'Now rank all 30 cities by population, largest first. Just give me the list.'}

    # Warm both
    print("Warming correct...", file=sys.stderr)
    for _ in range(3):
        ttft_streaming(args.url, msgs_correct, args.model, followup)
        time.sleep(0.3)

    print("Warming incorrect...", file=sys.stderr)
    for _ in range(3):
        ttft_streaming(args.url, msgs_incorrect, args.model, followup)
        time.sleep(0.3)

    time.sleep(1)

    # Measure
    print(f"\nMeasuring {args.runs} runs (alternating)...", file=sys.stderr)
    correct_times = []
    incorrect_times = []

    for i in range(args.runs):
        c = ttft_streaming(args.url, msgs_correct, args.model, followup)
        time.sleep(0.2)
        ic = ttft_streaming(args.url, msgs_incorrect, args.model, followup)
        time.sleep(0.2)
        correct_times.append(c)
        incorrect_times.append(ic)
        print(f"  {i+1}: correct={c:.1f}ms  incorrect={ic:.1f}ms  delta={ic-c:+.1f}ms", file=sys.stderr)

    print(f"\nCorrect:   mean={statistics.mean(correct_times):.1f}ms  stdev={statistics.stdev(correct_times):.1f}ms", file=sys.stderr)
    print(f"Incorrect: mean={statistics.mean(incorrect_times):.1f}ms  stdev={statistics.stdev(incorrect_times):.1f}ms", file=sys.stderr)
    print(f"Delta:     {statistics.mean(incorrect_times)-statistics.mean(correct_times):+.1f}ms", file=sys.stderr)

    if args.jsonl:
        with open(args.jsonl, 'w') as f:
            for i in range(args.runs):
                f.write(json.dumps({
                    'run': i+1,
                    'correct_ms': round(correct_times[i], 1),
                    'incorrect_ms': round(incorrect_times[i], 1),
                    'delta_ms': round(incorrect_times[i] - correct_times[i], 1),
                }) + '\n')
        print(f"Saved to {args.jsonl}", file=sys.stderr)


if __name__ == '__main__':
    main()
