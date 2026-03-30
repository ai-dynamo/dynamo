#!/usr/bin/env python3
"""Measure dispatch timing across a 30-city multi-turn tool-calling conversation.

For each turn: sends the conversation so far, measures:
- first_token_ms: first reasoning/content delta
- first_tool_info_ms: dispatch event or tool_delta (whichever comes first)
- finish_reason_ms: when finish_reason arrives (the OLD baseline)
- delta: how much earlier the harness knows about the tool call vs old buffered behavior

Runs the full 30-turn conversation once and records per-turn timings.
"""
import requests, json, time, sys, argparse


def stream_one_turn(url, model, messages, tools):
    """Send one streaming request, return timing dict."""
    body = {
        'model': model,
        'max_tokens': 2000,
        'stream': True,
        'messages': messages,
        'tools': tools,
        'tool_choice': 'auto',
    }
    headers = {'Content-Type': 'application/json'}
    t0 = time.monotonic()

    r = requests.post(f"{url}/v1/chat/completions", json=body, headers=headers, stream=True, timeout=60)

    timings = {
        'first_token_ms': None,
        'dispatch_ms': None,
        'tool_delta_ms': None,
        'first_tool_info_ms': None,
        'finish_reason_ms': None,
    }

    full_response = None
    event_type = None

    for line in r.iter_lines(decode_unicode=True):
        if not line:
            event_type = None
            continue

        now = (time.monotonic() - t0) * 1000

        if line.startswith('event: '):
            event_type = line[7:].strip()
            if event_type == 'tool_call_dispatch' and timings['dispatch_ms'] is None:
                timings['dispatch_ms'] = now
                if timings['first_tool_info_ms'] is None:
                    timings['first_tool_info_ms'] = now
            continue

        if line.startswith('data: '):
            raw = line[6:].strip()
            if raw == '[DONE]':
                break
            try:
                d = json.loads(raw)
            except json.JSONDecodeError:
                continue

            choices = d.get('choices', [])
            if not choices:
                continue

            delta = choices[0].get('delta', {})
            finish = choices[0].get('finish_reason')

            if timings['first_token_ms'] is None:
                if delta.get('content') or delta.get('reasoning_content'):
                    timings['first_token_ms'] = now

            tc = delta.get('tool_calls')
            if tc and timings['tool_delta_ms'] is None:
                timings['tool_delta_ms'] = now
                if timings['first_tool_info_ms'] is None:
                    timings['first_tool_info_ms'] = now

            if finish and timings['finish_reason_ms'] is None:
                timings['finish_reason_ms'] = now
                # Capture the full response message for conversation continuation
                # We need to reconstruct from the last chunk that has the message
                full_response = d

        event_type = None

    r.close()
    return timings


def run_conversation(url, model, cities):
    tools = [{
        'type': 'function',
        'function': {
            'name': 'echo',
            'description': 'Echo a city description to the user.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'city': {'type': 'string'},
                    'description': {'type': 'string'}
                },
                'required': ['city', 'description']
            }
        }
    }]

    messages = [
        {'role': 'system', 'content': 'You are a travel expert. For each city, think about what makes it special, then call the echo tool. Process one city at a time.'},
        {'role': 'user', 'content': f'Describe each city by calling echo: {", ".join(cities)}'},
    ]

    all_timings = []

    for turn in range(len(cities) + 5):  # extra turns for safety
        timings = stream_one_turn(url, model, messages, tools)
        timings['turn'] = turn + 1

        # Non-streaming request to get the full response for conversation continuation
        body = {
            'model': model,
            'max_tokens': 2000,
            'messages': messages,
            'tools': tools,
            'tool_choice': 'auto',
        }
        headers = {'Content-Type': 'application/json'}
        resp = requests.post(f"{url}/v1/chat/completions", json=body, headers=headers, timeout=60)
        data = resp.json()
        choice = data['choices'][0]
        msg = choice['message']
        finish = choice.get('finish_reason', '')

        tc = msg.get('tool_calls', [])
        if not tc or finish == 'stop':
            print(f"  Turn {turn+1}: model stopped (finish={finish})", file=sys.stderr)
            break

        # Extract city name
        try:
            args = json.loads(tc[0]['function']['arguments'])
            city = args.get('city', '?')
        except:
            city = '?'

        timings['city'] = city
        fr = timings['finish_reason_ms'] or 0
        fi = timings['first_tool_info_ms'] or 0
        info_before_done = fr - fi if fr and fi else 0
        timings['info_before_done_ms'] = round(info_before_done, 1)

        has_dispatch = timings['dispatch_ms'] is not None
        fi_str = f"{fi:.0f}" if fi else "?"
        fr_str = f"{fr:.0f}" if fr else "?"
        print(f"  Turn {turn+1} ({city}): first_info={fi_str}ms  done={fr_str}ms  delta={info_before_done:.1f}ms  {'dispatch' if has_dispatch else 'no dispatch'}", file=sys.stderr)

        all_timings.append(timings)

        # Append to conversation
        messages.append(msg)
        for t in tc:
            args_parsed = json.loads(t['function']['arguments'])
            result = f"Echoed: {args_parsed.get('city', '?')} - {args_parsed.get('description', '?')[:50]}..."
            messages.append({'role': 'tool', 'tool_call_id': t['id'], 'content': result})

        time.sleep(0.2)

    return all_timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://localhost:8000')
    parser.add_argument('--model', default='nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4')
    parser.add_argument('--label', default='unknown')
    parser.add_argument('--jsonl', default=None)
    args = parser.parse_args()

    cities = [
        'Tokyo', 'Paris', 'Cairo', 'Mumbai', 'São Paulo', 'Istanbul', 'Bangkok',
        'Lagos', 'Moscow', 'Beijing', 'London', 'New York', 'Sydney', 'Buenos Aires',
        'Nairobi', 'Seoul', 'Rome', 'Lima', 'Hanoi', 'Berlin', 'Marrakech',
        'Vancouver', 'Dubai', 'Singapore', 'Cape Town', 'Reykjavik', 'Havana',
        'Kathmandu', 'Athens', 'Stockholm'
    ]

    print(f"Label: {args.label}", file=sys.stderr)
    print(f"Running 30-city conversation...", file=sys.stderr)

    timings = run_conversation(args.url, args.model, cities)

    # Summary
    infos = [t['first_tool_info_ms'] for t in timings if t.get('first_tool_info_ms')]
    dones = [t['finish_reason_ms'] for t in timings if t.get('finish_reason_ms')]
    deltas = [t['info_before_done_ms'] for t in timings if t.get('info_before_done_ms')]

    print(f"\n=== Summary ({args.label}) ===", file=sys.stderr)
    print(f"Turns completed: {len(timings)}", file=sys.stderr)
    if deltas:
        print(f"Info before done: mean={sum(deltas)/len(deltas):.1f}ms per turn", file=sys.stderr)
        print(f"Cumulative earlier feedback: {sum(deltas):.0f}ms over {len(deltas)} turns", file=sys.stderr)

    has_dispatch = any(t.get('dispatch_ms') is not None for t in timings)
    print(f"Dispatch events: {'Yes' if has_dispatch else 'No'}", file=sys.stderr)

    if args.jsonl:
        with open(args.jsonl, 'w') as f:
            for t in timings:
                row = {k: round(v, 1) if isinstance(v, float) else v for k, v in t.items()}
                row['label'] = args.label
                f.write(json.dumps(row) + '\n')
        print(f"Saved to {args.jsonl}", file=sys.stderr)


if __name__ == '__main__':
    main()
