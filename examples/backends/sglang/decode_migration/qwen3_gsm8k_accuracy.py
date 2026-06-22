# SPDX-License-Identifier: Apache-2.0

"""Paired GSM8K accuracy check for Qwen3 decode migration."""

from __future__ import annotations

import argparse
import ast
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/test.jsonl"
)
INVALID = object()


def request_json(url: str, payload: dict, timeout: float = 900.0) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_ready(base_url: str, model: str, timeout: float = 600.0) -> None:
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=3) as response:
                models = {
                    item.get("id") for item in (json.load(response).get("data") or [])
                }
                if response.status == 200 and model in models:
                    return
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"Dynamo frontend did not become ready: {last_error}")


def load_dataset(path: Path) -> list[dict]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading GSM8K test data to {path}", flush=True)
        urllib.request.urlretrieve(GSM8K_URL, path)
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def get_one_example(item: dict, include_answer: bool) -> str:
    text = f"Question: {item['question']}\nAnswer:"
    if include_answer:
        text += f" {item['answer']}"
    return text


def build_prompt(lines: list[dict], index: int, num_shots: int) -> str:
    examples = "\n\n".join(
        get_one_example(item, include_answer=True) for item in lines[:num_shots]
    )
    question = get_one_example(lines[num_shots + index], include_answer=False)
    return (
        f"{examples}\n\n{question}\n"
        "Solve the problem carefully. End with the numeric answer."
    )


def answer_value(text: str):
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not numbers:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except (SyntaxError, ValueError):
        return INVALID


def source_nvext() -> dict:
    return {
        "routing_constraints": {"required_taints": ["decode/fast"]},
        "extra_fields": ["completion_token_ids"],
    }


def destination_nvext() -> dict:
    return {
        "routing_constraints": {"required_taints": ["decode/slow"]},
        "extra_fields": ["completion_token_ids"],
    }


def migration_nvext(think_end_token_id: int) -> dict:
    return {
        "decode_migration": {
            "source": {"required_taints": ["decode/fast"]},
            "destination": {"required_taints": ["decode/slow"]},
            "trigger": {"type": "token_id", "token_id": think_end_token_id},
        },
        "extra_fields": ["completion_token_ids"],
    }


def chat_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    nvext: dict,
) -> dict:
    return request_json(
        f"{base_url}/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "seed": 1234,
            "n": 1,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": True},
            "nvext": nvext,
        },
    )


def completion(
    base_url: str,
    model: str,
    prompt: list[int],
    max_tokens: int,
    nvext: dict,
) -> dict:
    return request_json(
        f"{base_url}/v1/completions",
        {
            "model": model,
            "prompt": prompt,
            "temperature": 0,
            "seed": 1234,
            "n": 1,
            "best_of": 1,
            "max_tokens": max_tokens,
            "nvext": nvext,
        },
    )


def response_token_ids(response: dict) -> list[int]:
    token_ids = (response.get("nvext") or {}).get("completion_token_ids")
    if not isinstance(token_ids, list) or not all(
        isinstance(token_id, int) for token_id in token_ids
    ):
        raise AssertionError(
            f"Response omitted nvext.completion_token_ids: {response!r}"
        )
    return token_ids


def response_fields(response: dict) -> tuple[str, str, list[int], str | None]:
    choice = response["choices"][0]
    message = choice["message"]
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    return content, reasoning, response_token_ids(response), choice.get("finish_reason")


def common_prefix_length(left: list[int], right: list[int]) -> int:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    return min(len(left), len(right))


def token_window(tokenizer, token_ids: list[int], center: int, radius: int = 8) -> dict:
    start = max(0, center - radius)
    end = min(len(token_ids), center + radius)
    window = token_ids[start:end]
    return {
        "range": [start, end],
        "token_ids": window,
        "text": tokenizer.decode(window, skip_special_tokens=False),
    }


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def latest_handoff_frontier(log_dir: Path, minimum_count: int) -> tuple[int, int]:
    text = strip_ansi((log_dir / "frontend.log").read_text(errors="replace"))
    matches = re.findall(
        r"decode migration source quiesced.*?"
        r"committed_len=(\d+).*?logical_len=(\d+)",
        text,
    )
    if len(matches) < minimum_count:
        raise AssertionError(
            f"Expected at least {minimum_count} handoff frontier logs, got "
            f"{len(matches)}"
        )
    committed_len, logical_len = matches[minimum_count - 1]
    return int(committed_len), int(logical_len)


def destination_replay(
    base_url: str,
    model: str,
    prompt_ids: list[int],
    migrated_ids: list[int],
    handoff_output_tokens: int,
    max_tokens: int,
) -> list[int]:
    prefix = migrated_ids[:handoff_output_tokens]
    remaining = max_tokens - len(prefix)
    if remaining <= 0:
        return prefix
    response = completion(
        base_url,
        model,
        prompt_ids + prefix,
        remaining,
        destination_nvext(),
    )
    return prefix + response_token_ids(response)


def log_count(log_dir: Path, text: str) -> int:
    return (log_dir / "frontend.log").read_text(errors="replace").count(text)


def wait_for_log_count(
    log_dir: Path, text: str, minimum: int, timeout: float = 120.0
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if log_count(log_dir, text) >= minimum:
            return
        time.sleep(0.25)
    raise AssertionError(f"Did not observe {minimum} occurrences of {text!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--num-questions", type=int, default=20)
    parser.add_argument(
        "--indices",
        help="Comma-separated zero-based candidate indices after the few-shot rows",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        help="Maximum dataset rows to try while collecting migrated samples",
    )
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--allowed-regressions", type=int, default=1)
    args = parser.parse_args()

    wait_ready(args.base_url, args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    think_end_ids = tokenizer.encode("</think>", add_special_tokens=False)
    if len(think_end_ids) != 1:
        raise AssertionError(f"Expected one </think> token, got {think_end_ids}")
    think_end_token_id = think_end_ids[0]

    data_path = args.data_path or (args.log_dir / "gsm8k-test.jsonl")
    lines = load_dataset(data_path)
    selected_indices = None
    if args.indices:
        selected_indices = [int(value) for value in args.indices.split(",")]
        if not selected_indices or any(index < 0 for index in selected_indices):
            raise ValueError("--indices must contain non-negative integers")
        args.num_questions = len(selected_indices)
    max_attempts = args.max_attempts or args.num_questions * 3
    if max_attempts < args.num_questions:
        raise ValueError("--max-attempts must be at least --num-questions")
    available_attempts = len(lines) - args.num_shots
    max_attempts = min(max_attempts, available_attempts)
    if max_attempts < args.num_questions:
        raise ValueError(
            f"GSM8K data has {available_attempts} candidate rows, "
            f"need at least {args.num_questions}"
        )

    baseline_correct = 0
    migrated_correct = 0
    answer_agreement = 0
    results = []
    skipped = []
    started = time.perf_counter()

    candidate_indices = (
        selected_indices if selected_indices is not None else range(max_attempts)
    )
    for index in candidate_indices:
        if index >= available_attempts:
            raise ValueError(
                f"Candidate index {index} exceeds available rows "
                f"({available_attempts})"
            )
        item = lines[args.num_shots + index]
        label = answer_value(item["answer"])
        if label is INVALID:
            raise AssertionError(f"Invalid GSM8K label at item {index}")
        prompt = build_prompt(lines, index, args.num_shots)
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        if not isinstance(prompt_ids, list):
            prompt_ids = prompt_ids["input_ids"]

        baseline = chat_completion(
            args.base_url,
            args.model,
            prompt,
            args.max_tokens,
            source_nvext(),
        )
        (
            baseline_text,
            baseline_reasoning,
            baseline_ids,
            baseline_finish_reason,
        ) = response_fields(baseline)
        baseline_answer = answer_value(baseline_text)

        commits_before = log_count(args.log_dir, "decode migration committed")
        frontiers_before = log_count(args.log_dir, "decode migration source quiesced")
        migrated = chat_completion(
            args.base_url,
            args.model,
            prompt,
            args.max_tokens,
            migration_nvext(think_end_token_id),
        )
        (
            migrated_text,
            migrated_reasoning,
            migrated_ids,
            migrated_finish_reason,
        ) = response_fields(migrated)
        if think_end_token_id not in migrated_ids:
            skipped_result = {
                "index": index,
                "reason": "no_think_end_before_generation_finished",
                "baseline_tokens": len(baseline_ids),
                "migrated_tokens": len(migrated_ids),
                "baseline_finish_reason": baseline_finish_reason,
                "migrated_finish_reason": migrated_finish_reason,
            }
            skipped.append(skipped_result)
            print(json.dumps(skipped_result), flush=True)
            continue

        # A request that reaches the trigger but does not commit is a migration
        # failure, not an ineligible benchmark sample.
        wait_for_log_count(
            args.log_dir,
            "decode migration committed",
            commits_before + 1,
        )
        committed_len, logical_len = latest_handoff_frontier(
            args.log_dir, frontiers_before + 1
        )
        prompt_len = len(prompt_ids)
        committed_output_tokens = committed_len - prompt_len
        handoff_output_tokens = logical_len - prompt_len
        if committed_output_tokens < 0 or handoff_output_tokens < 0:
            raise AssertionError(
                "Logged migration frontier precedes the reconstructed prompt: "
                f"prompt={prompt_len}, committed={committed_len}, "
                f"logical={logical_len}"
            )
        replay_ids = destination_replay(
            args.base_url,
            args.model,
            prompt_ids,
            migrated_ids,
            handoff_output_tokens,
            args.max_tokens,
        )
        migrated_answer = answer_value(migrated_text)

        baseline_is_correct = baseline_answer == label
        migrated_is_correct = migrated_answer == label
        baseline_correct += int(baseline_is_correct)
        migrated_correct += int(migrated_is_correct)
        answer_agreement += int(baseline_answer == migrated_answer)
        shared_prefix = common_prefix_length(baseline_ids, migrated_ids)
        replay_shared_prefix = common_prefix_length(migrated_ids, replay_ids)
        baseline_think_end_position = (
            baseline_ids.index(think_end_token_id)
            if think_end_token_id in baseline_ids
            else None
        )
        migrated_think_end_position = migrated_ids.index(think_end_token_id)
        result = {
            "index": index,
            "label": label,
            "baseline_answer": None if baseline_answer is INVALID else baseline_answer,
            "migrated_answer": None if migrated_answer is INVALID else migrated_answer,
            "baseline_correct": baseline_is_correct,
            "migrated_correct": migrated_is_correct,
            "baseline_tokens": len(baseline_ids),
            "migrated_tokens": len(migrated_ids),
            "baseline_finish_reason": baseline_finish_reason,
            "migrated_finish_reason": migrated_finish_reason,
            "common_prefix_tokens": shared_prefix,
            "baseline_think_end_position": baseline_think_end_position,
            "think_end_position": migrated_think_end_position,
            "prompt_tokens": prompt_len,
            "handoff_committed_tokens": committed_output_tokens,
            "handoff_logical_tokens": handoff_output_tokens,
            "migration_replay_common_prefix_tokens": replay_shared_prefix,
            "migration_replay_exact": replay_ids == migrated_ids,
            "baseline_divergence": {
                "baseline": token_window(tokenizer, baseline_ids, shared_prefix),
                "migrated": token_window(tokenizer, migrated_ids, shared_prefix),
            },
            "replay_divergence": {
                "migrated": token_window(tokenizer, migrated_ids, replay_shared_prefix),
                "replay": token_window(tokenizer, replay_ids, replay_shared_prefix),
            },
            "thinking_prefix_preserved": (
                baseline_think_end_position == migrated_think_end_position
                and shared_prefix > migrated_think_end_position
            ),
            "migrated_reasoning_characters": len(migrated_reasoning),
            "baseline_reasoning_characters": len(baseline_reasoning),
            "baseline_text_tail": baseline_text[-500:],
            "migrated_text_tail": migrated_text[-500:],
        }
        results.append(result)
        print(json.dumps(result), flush=True)
        if len(results) == args.num_questions:
            break

    if len(results) != args.num_questions:
        raise AssertionError(
            f"Collected {len(results)} migrated samples after {max_attempts} attempts; "
            f"need {args.num_questions}"
        )

    elapsed = time.perf_counter() - started
    migrated_samples = len(results)
    summary = {
        "status": "passed",
        "model": args.model,
        "questions": migrated_samples,
        "target_questions": args.num_questions,
        "attempted_questions": migrated_samples + len(skipped),
        "skipped_without_think_end": len(skipped),
        "think_end_token_id": think_end_token_id,
        "baseline_correct": baseline_correct,
        "migrated_correct": migrated_correct,
        "baseline_accuracy": baseline_correct / migrated_samples,
        "migrated_accuracy": migrated_correct / migrated_samples,
        "answer_agreement": answer_agreement / migrated_samples,
        "allowed_regressions": args.allowed_regressions,
        "elapsed_seconds": elapsed,
        "results": results,
        "skipped": skipped,
    }
    if migrated_correct < baseline_correct - args.allowed_regressions:
        summary["status"] = "failed"
    output_path = args.log_dir / "gsm8k_accuracy_results.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)

    for worker_log in ("fast.log", "slow.log"):
        text = (args.log_dir / worker_log).read_text(errors="replace")
        if "Scheduler hit an exception" in text:
            raise AssertionError(f"Scheduler crashed during GSM8K: {worker_log}")
        if "pool memory leak detected" in text:
            raise AssertionError(f"KV pool leak detected during GSM8K: {worker_log}")
    if summary["status"] != "passed":
        raise AssertionError(
            "Migration regressed more paired GSM8K answers than allowed: "
            f"baseline={baseline_correct}, migrated={migrated_correct}, "
            f"allowed={args.allowed_regressions}"
        )


if __name__ == "__main__":
    main()
