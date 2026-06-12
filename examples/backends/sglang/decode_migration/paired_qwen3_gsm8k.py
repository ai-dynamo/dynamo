#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Compare Qwen3 GSM8K accuracy with and without decode migration."""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import openai
from sglang.test.simple_eval_common import ChatCompletionSampler, make_report
from sglang.test.simple_eval_gsm8k import GSM8KEval, get_answer_value, get_one_example


class RecordingReasoningSampler(ChatCompletionSampler):
    """SGLang's chat sampler plus per-prompt reasoning metadata."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._response_records: dict[str, dict[str, Any]] = {}
        self._records_lock = threading.Lock()

    def __call__(self, message_list: list[dict[str, Any]]) -> str:
        request_messages = message_list
        if self.system_message:
            request_messages = [
                self._pack_message("system", self.system_message)
            ] + request_messages
        prompt = str(message_list[-1].get("content") or "")

        for trial in range(6):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=request_messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    reasoning_effort=self.reasoning_effort,
                    extra_body=self.extra_body,
                )
                choice = response.choices[0]
                message = choice.message
                content = message.content or ""
                reasoning = getattr(message, "reasoning_content", None) or ""
                if not reasoning and getattr(message, "model_extra", None):
                    reasoning = message.model_extra.get("reasoning_content") or ""
                usage = response.usage
                completion_tokens = (
                    usage.completion_tokens
                    if usage and usage.completion_tokens is not None
                    else 0
                )
                if completion_tokens:
                    self._completion_tokens.append(completion_tokens)
                record = {
                    "content": content,
                    "reasoning_content": reasoning,
                    "finish_reason": choice.finish_reason,
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": (
                        usage.prompt_tokens
                        if usage and usage.prompt_tokens is not None
                        else 0
                    ),
                }
                with self._records_lock:
                    self._response_records[prompt] = record
                return content
            except openai.BadRequestError as exc:
                print(f"Bad request: {exc}", flush=True)
                with self._records_lock:
                    self._response_records[prompt] = {
                        "content": "",
                        "reasoning_content": "",
                        "finish_reason": "bad_request",
                        "error": str(exc),
                    }
                return ""
            except Exception as exc:
                delay = 2**trial
                print(
                    f"Request failed; retry {trial + 1}/6 after {delay}s: {exc}",
                    flush=True,
                )
                time.sleep(delay)

        with self._records_lock:
            self._response_records[prompt] = {
                "content": "",
                "reasoning_content": "",
                "finish_reason": "retries_exhausted",
            }
        return ""

    def response_for(self, prompt: str) -> dict[str, Any]:
        with self._records_lock:
            return dict(self._response_records.get(prompt) or {})


def wait_ready(base_url: str, models: set[str], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=3) as response:
                available = {
                    item.get("id") for item in (json.load(response).get("data") or [])
                }
                if response.status == 200 and models <= available:
                    return
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(
        f"Frontend did not expose {sorted(models)} within {timeout}s: {last_error}"
    )


def file_offsets(paths: list[Path]) -> dict[Path, int]:
    return {path: path.stat().st_size if path.exists() else 0 for path in paths}


def read_since(path: Path, offset: int) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as file:
        file.seek(offset)
        return file.read().decode(errors="replace")


def run_gsm8k(
    *,
    base_url: str,
    model: str,
    num_examples: int,
    num_threads: int,
    max_tokens: int,
    num_shots: int,
    data_path: str | None,
    report_path: Path,
) -> dict[str, Any]:
    eval_obj = GSM8KEval(
        num_examples=num_examples,
        num_threads=num_threads,
        num_shots=num_shots,
        data_path=data_path,
    )
    sampler = RecordingReasoningSampler(
        base_url=f"{base_url}/v1",
        model=model,
        temperature=0,
        top_p=1,
        max_tokens=max_tokens,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )

    started = time.perf_counter()
    result = eval_obj(sampler)
    latency = time.perf_counter() - started
    report_path.write_text(make_report(result))

    records = []
    for index, (line, convo) in enumerate(zip(eval_obj._lines, result.convos)):
        prompt = eval_obj._few_shot_prompt + get_one_example(
            eval_obj._lines, index, include_answer=False
        )
        api_record = sampler.response_for(prompt)
        output = convo[-1]["content"]
        expected = get_answer_value(line["answer"])
        extracted = get_answer_value(output)
        records.append(
            {
                "index": index,
                "question": line["question"],
                "expected_answer": expected,
                "extracted_answer": extracted,
                "correct": extracted == expected,
                "output": output,
                "reasoning_content": api_record.get("reasoning_content", ""),
                "finish_reason": api_record.get("finish_reason"),
                "completion_tokens": api_record.get("completion_tokens", 0),
                "prompt_tokens": api_record.get("prompt_tokens", 0),
                "error": api_record.get("error"),
            }
        )

    completion_tokens = sum(int(item["completion_tokens"]) for item in records)
    finish_reasons = Counter(str(item["finish_reason"]) for item in records)
    return {
        "model": model,
        "score": result.score,
        "score_std": result.metrics.get("score:std"),
        "latency_seconds": latency,
        "completion_tokens": completion_tokens,
        "output_throughput": completion_tokens / latency if latency else None,
        "empty_outputs": sum(not item["output"] for item in records),
        "reasoning_nonempty": sum(bool(item["reasoning_content"]) for item in records),
        "reasoning_characters": sum(len(item["reasoning_content"]) for item in records),
        "finish_reasons": dict(finish_reasons),
        "records": records,
    }


def migration_events(text: str) -> dict[str, Any]:
    patterns = {
        "boundary_observed": (
            r"Observed migration token boundary request_id=([0-9a-f-]+)"
        ),
        "reserved": r"Reserved decode migration destination .*migration_id=([0-9a-f]+)",
        "armed": r"Armed decode migration destination .*migration_id=([0-9a-f]+)",
        "activated": (
            r"Finalized decode migration destination .*migration_id=([0-9a-f]+) "
            r"action=activate"
        ),
        "aborted": (
            r"Finalized decode migration destination .*migration_id=([0-9a-f]+) "
            r"action=abort"
        ),
        "transfer_completed": (
            r"Decode migration transfer completed .*migration_id=([0-9a-f]+)"
        ),
        "source_committed": (
            r"Finalized decode migration .*migration_id=([0-9a-f]+) action=commit"
        ),
        "source_resumed": (
            r"Finalized decode migration .*migration_id=([0-9a-f]+) action=resume"
        ),
        "source_cancelled": (
            r"Finalized decode migration .*migration_id=([0-9a-f]+) action=cancel"
        ),
    }
    ids = {
        name: sorted(set(re.findall(pattern, text)))
        for name, pattern in patterns.items()
    }
    return {
        "counts": {name: len(values) for name, values in ids.items()},
        "ids": ids,
    }


def paired_summary(
    baseline: dict[str, Any], migrated: dict[str, Any]
) -> dict[str, Any]:
    baseline_records = baseline["records"]
    migrated_records = migrated["records"]
    if len(baseline_records) != len(migrated_records):
        raise AssertionError("Baseline and migrated evaluations have different sizes")

    exact_outputs = 0
    exact_reasoning = 0
    answer_matches = 0
    baseline_only_correct = []
    migrated_only_correct = []
    output_mismatches = []
    for before, after in zip(baseline_records, migrated_records):
        index = before["index"]
        if before["output"] == after["output"]:
            exact_outputs += 1
        else:
            output_mismatches.append(index)
        if before["reasoning_content"] == after["reasoning_content"]:
            exact_reasoning += 1
        if before["extracted_answer"] == after["extracted_answer"]:
            answer_matches += 1
        if before["correct"] and not after["correct"]:
            baseline_only_correct.append(index)
        if after["correct"] and not before["correct"]:
            migrated_only_correct.append(index)

    total = len(baseline_records)
    completed_response_indices = [
        item["index"]
        for item in migrated_records
        if item["finish_reason"] == "stop" and item["output"]
    ]
    completed_index_set = set(completed_response_indices)
    baseline_subset_correct = sum(
        item["correct"]
        for item in baseline_records
        if item["index"] in completed_index_set
    )
    migrated_subset_correct = sum(
        item["correct"]
        for item in migrated_records
        if item["index"] in completed_index_set
    )
    subset_total = len(completed_response_indices)
    baseline_subset_score = (
        baseline_subset_correct / subset_total if subset_total else 0.0
    )
    migrated_subset_score = (
        migrated_subset_correct / subset_total if subset_total else 0.0
    )

    return {
        "num_examples": total,
        "baseline_score": baseline["score"],
        "migrated_score": migrated["score"],
        "score_delta": migrated["score"] - baseline["score"],
        "completed_response_subset": {
            "indices": completed_response_indices,
            "num_examples": subset_total,
            "baseline_score": baseline_subset_score,
            "migrated_score": migrated_subset_score,
            "score_delta": migrated_subset_score - baseline_subset_score,
        },
        "exact_output_matches": exact_outputs,
        "exact_output_match_rate": exact_outputs / total,
        "exact_reasoning_matches": exact_reasoning,
        "exact_reasoning_match_rate": exact_reasoning / total,
        "extracted_answer_matches": answer_matches,
        "extracted_answer_match_rate": answer_matches / total,
        "baseline_only_correct": baseline_only_correct,
        "migrated_only_correct": migrated_only_correct,
        "output_mismatch_indices": output_mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:28000")
    parser.add_argument("--baseline-model", default="decode-migration-baseline")
    parser.add_argument("--migrated-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--num-examples", type=int, default=200)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--migration-trigger-token-id", type=int, default=151668)
    parser.add_argument("--data-path")
    parser.add_argument("--ready-timeout", type=float, default=900)
    parser.add_argument("--minimum-migration-coverage", type=float, default=1.0)
    parser.add_argument("--minimum-reasoning-coverage", type=float, default=1.0)
    parser.add_argument("--max-score-regression", type=float, default=0.02)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()

    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    args.result_dir.mkdir(parents=True, exist_ok=True)
    wait_ready(
        args.base_url,
        {args.baseline_model, args.migrated_model},
        args.ready_timeout,
    )

    logs = [
        args.log_dir / "fast.log",
        args.log_dir / "slow.log",
        args.log_dir / "coordinator.log",
        args.log_dir / "coordinator-baseline.log",
    ]
    baseline_offsets = file_offsets(logs)
    print(f"Running baseline Qwen3 GSM8K: {args.baseline_model}", flush=True)
    baseline = run_gsm8k(
        base_url=args.base_url,
        model=args.baseline_model,
        num_examples=args.num_examples,
        num_threads=args.num_threads,
        max_tokens=args.max_tokens,
        num_shots=args.num_shots,
        data_path=args.data_path,
        report_path=args.result_dir / "baseline.html",
    )
    time.sleep(1)
    baseline_text = "".join(read_since(path, baseline_offsets[path]) for path in logs)
    baseline_events = migration_events(baseline_text)

    migrated_offsets = file_offsets(logs)
    print(f"Running migrated Qwen3 GSM8K: {args.migrated_model}", flush=True)
    migrated = run_gsm8k(
        base_url=args.base_url,
        model=args.migrated_model,
        num_examples=args.num_examples,
        num_threads=args.num_threads,
        max_tokens=args.max_tokens,
        num_shots=args.num_shots,
        data_path=args.data_path,
        report_path=args.result_dir / "migrated.html",
    )
    time.sleep(1)
    migrated_text = "".join(read_since(path, migrated_offsets[path]) for path in logs)
    migrated_events = migration_events(migrated_text)
    paired = paired_summary(baseline, migrated)

    proof_counts = migrated_events["counts"]
    transfer_coverage = (
        min(
            proof_counts["boundary_observed"],
            proof_counts["activated"],
            proof_counts["transfer_completed"],
            proof_counts["source_committed"],
        )
        / args.num_examples
    )
    baseline_reasoning_coverage = baseline["reasoning_nonempty"] / args.num_examples
    migrated_reasoning_coverage = migrated["reasoning_nonempty"] / args.num_examples

    summary = {
        "configuration": {
            "num_examples": args.num_examples,
            "num_threads": args.num_threads,
            "max_tokens": args.max_tokens,
            "num_shots": args.num_shots,
            "temperature": 0,
            "api": "chat_completions",
            "thinking_enabled": True,
            "migration_trigger_token_id": args.migration_trigger_token_id,
            "migration_trigger_token": "</think>",
        },
        "paired": paired,
        "baseline": {key: value for key, value in baseline.items() if key != "records"},
        "migrated": {key: value for key, value in migrated.items() if key != "records"},
        "migration": {
            "verified_transfer_coverage": transfer_coverage,
            "events": migrated_events,
            "baseline_events": baseline_events,
        },
        "reasoning": {
            "baseline_coverage": baseline_reasoning_coverage,
            "migrated_coverage": migrated_reasoning_coverage,
        },
    }

    (args.result_dir / "baseline.json").write_text(json.dumps(baseline, indent=2))
    (args.result_dir / "migrated.json").write_text(json.dumps(migrated, indent=2))
    (args.result_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

    unexpected_empty_outputs = [
        (label, item["index"], item["finish_reason"])
        for label, result in (("baseline", baseline), ("migrated", migrated))
        for item in result["records"]
        if not item["output"] and item["finish_reason"] != "length"
    ]
    if unexpected_empty_outputs:
        raise AssertionError(
            f"The benchmark contained unexpected empty responses: "
            f"{unexpected_empty_outputs}"
        )
    if baseline_events["counts"]["activated"]:
        raise AssertionError("The source-only baseline unexpectedly migrated requests")
    if transfer_coverage < args.minimum_migration_coverage:
        raise AssertionError(
            f"Verified migration coverage {transfer_coverage:.3%} is below "
            f"{args.minimum_migration_coverage:.3%}"
        )
    if min(baseline_reasoning_coverage, migrated_reasoning_coverage) < (
        args.minimum_reasoning_coverage
    ):
        raise AssertionError(
            "Reasoning coverage is below the required threshold: "
            f"baseline={baseline_reasoning_coverage:.3%}, "
            f"migrated={migrated_reasoning_coverage:.3%}"
        )
    score_delta = paired["score_delta"]
    if score_delta < -args.max_score_regression:
        raise AssertionError(
            "Migrated GSM8K score regressed by "
            f"{-score_delta:.3f}; limit is {args.max_score_regression:.3f}"
        )


if __name__ == "__main__":
    main()
