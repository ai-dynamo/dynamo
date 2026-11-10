# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a pinned corpus through the installed HTTP frontend on CPU."""

from __future__ import annotations

import copy
import os
import sys
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests

from tests.utils.constants import DynamoPortRange
from tests.utils.http_checks import model_registered
from tests.utils.managed_process import ManagedProcess
from tests.utils.port_utils import allocate_port, deallocate_port

from .contract import (
    Case,
    Corpus,
    ReplayError,
    canonical_message,
    digest,
    dumps,
    file_digest,
    judge_preprocessed,
    judge_response,
    load_corpus,
    loads,
    same_json,
    scheduled_events,
    verify_assets,
    write_json,
)

ANNOTATION_PREFIX = "frontend_replay:"
MODES = ("unary", "sse")
SCHEDULES = ("whole", "one_token", "fixed32", "recorded")


def prepare_assets(corpus_path: Path, model_dir: Path) -> dict[str, str]:
    corpus = load_corpus(corpus_path)
    model = corpus.profile["model"]
    model_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name, expected in model["assets"].items():
        destination = model_dir / name
        if destination.exists():
            if not destination.is_file() or file_digest(destination) != expected:
                raise ReplayError(f"existing model asset has the wrong hash: {name}")
            continue
        url = f"https://huggingface.co/{model['repo_id']}/resolve/{model['revision']}/{name}"
        headers = (
            {"Authorization": "Bearer " + os.environ["HF_TOKEN"]}
            if os.environ.get("HF_TOKEN")
            else {}
        )
        fd, temporary = tempfile.mkstemp(prefix=".asset-", dir=model_dir)
        try:
            with os.fdopen(fd, "wb") as output:
                try:
                    with requests.get(
                        url, headers=headers, stream=True, timeout=(10, 120)
                    ) as response:
                        response.raise_for_status()
                        for block in response.iter_content(1024 * 1024):
                            output.write(block)
                except requests.RequestException as exc:
                    raise ReplayError(f"asset download failed: {name}: {exc}") from exc
            if file_digest(Path(temporary)) != expected:
                raise ReplayError(f"downloaded model asset has the wrong hash: {name}")
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return verify_assets(corpus, model_dir)


def _sse_payloads(lines: Iterable[str]) -> list[dict[str, Any]]:
    data: list[str] = []
    payloads: list[dict[str, Any]] = []
    done = False
    for line in lines:
        if not isinstance(line, str):
            raise ReplayError("SSE input must be decoded UTF-8")
        if line.startswith(":"):
            continue
        if line == "":
            if not data:
                continue
            payload = "\n".join(data)
            data = []
            if done:
                raise ReplayError("SSE event after [DONE]")
            if payload == "[DONE]":
                done = True
            else:
                value = loads(payload)
                if not isinstance(value, dict) or "error" in value:
                    raise ReplayError("SSE contains an error or non-object payload")
                payloads.append(value)
        elif line.startswith("data:"):
            if done:
                raise ReplayError("SSE data after [DONE]")
            field = line[5:]
            data.append(field[1:] if field.startswith(" ") else field)
        elif line.startswith(("event:", "id:", "retry:")):
            continue
        else:
            raise ReplayError("invalid SSE field")
    if data or not done:
        raise ReplayError("SSE ended without a complete [DONE] event")
    return payloads


def assemble_sse(lines: Iterable[str]) -> dict[str, Any]:
    """Independent Chat Completions accumulator with terminal/order checks."""
    payloads = _sse_payloads(lines)
    content: list[str] = []
    reasoning: list[str] = []
    saw_content = False
    saw_reasoning = False
    calls: dict[int, dict[str, Any]] = {}
    finish = None
    usage = None
    identifier = None
    role = None
    for payload in payloads:
        if identifier is None:
            identifier = payload.get("id")
        elif payload.get("id") != identifier:
            raise ReplayError("response ID changed inside the stream")
        if payload.get("usage") is not None:
            if usage is not None:
                raise ReplayError("multiple usage records")
            usage = payload["usage"]
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) > 1:
            raise ReplayError("expected at most one streamed choice")
        if not choices:
            continue
        choice = choices[0]
        if choice.get("index") != 0:
            raise ReplayError("unexpected streamed choice index")
        delta = choice.get("delta")
        if not isinstance(delta, dict) or "reasoning" in delta:
            raise ReplayError("invalid delta or unexpected reasoning field")
        if finish is not None:
            raise ReplayError("choice delta after its finish reason")
        if delta.get("role") is not None:
            if delta["role"] != "assistant":
                raise ReplayError("unexpected streamed role")
            role = "assistant"
        for key, parts in (("content", content), ("reasoning_content", reasoning)):
            value = delta.get(key)
            if value is not None:
                if not isinstance(value, str):
                    raise ReplayError(f"{key} delta is not text")
                parts.append(value)
                if key == "content":
                    saw_content = True
                else:
                    saw_reasoning = True
        call_deltas = delta.get("tool_calls")
        if call_deltas is None:
            call_deltas = []
        if not isinstance(call_deltas, list):
            raise ReplayError("streamed tool_calls must be an array")
        for call in call_deltas:
            index = call.get("index")
            if type(index) is not int or index < 0:
                raise ReplayError("invalid tool-call index")
            target = calls.setdefault(
                index,
                {
                    "id": None,
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            if call.get("id") is not None:
                if target["id"] is not None and target["id"] != call["id"]:
                    raise ReplayError("tool-call ID changed inside the stream")
                target["id"] = call["id"]
            if call.get("type") not in (None, "function"):
                raise ReplayError("invalid streamed tool type")
            function = call.get("function") or {}
            for key in ("name", "arguments"):
                value = function.get(key)
                if value is not None:
                    if not isinstance(value, str):
                        raise ReplayError("streamed function fields must be text")
                    target["function"][key] += value
        if choice.get("finish_reason") is not None:
            finish = choice["finish_reason"]
    if finish not in ("stop", "tool_calls", "length") or role != "assistant":
        raise ReplayError("stream lacks an assistant role or supported terminal")
    if sorted(calls) != list(range(len(calls))):
        raise ReplayError("tool-call indexes are not contiguous")
    message = {
        "role": role,
        "content": "".join(content) if saw_content else None,
        "reasoning_content": "".join(reasoning) if saw_reasoning else None,
        "tool_calls": [calls[index] for index in sorted(calls)],
    }
    return {
        "http_status": 200,
        "message": canonical_message(message),
        "finish_reason": finish,
        "usage": usage,
    }


def normalize_unary(payload: Any) -> dict[str, Any]:
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("choices"), list)
        or len(payload["choices"]) != 1
    ):
        raise ReplayError("expected one unary choice")
    choice = payload["choices"][0]
    if choice.get("index") != 0 or choice.get("finish_reason") not in (
        "stop",
        "tool_calls",
        "length",
    ):
        raise ReplayError("unexpected unary choice or finish reason")
    return {
        "http_status": 200,
        "message": canonical_message(choice.get("message")),
        "finish_reason": choice["finish_reason"],
        "usage": payload.get("usage"),
    }


def _stable_ir(request: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(request)
    result.pop("mdc_sum", None)
    result["annotations"] = [
        item
        for item in result.get("annotations", [])
        if not isinstance(item, str) or not item.startswith(ANNOTATION_PREFIX)
    ]
    if isinstance(result.get("routing"), dict):
        for key in (
            "backend_instance_id",
            "prefill_worker_id",
            "decode_worker_id",
            "allowed_worker_ids",
        ):
            result["routing"].pop(key, None)
    return result


def _stable_response(response: dict[str, Any] | None) -> dict[str, Any] | None:
    if response is None:
        return None
    usage = response.get("usage")
    return {
        **response,
        "usage": {
            key: usage.get(key)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        if isinstance(usage, dict)
        else usage,
    }


def harness_identity() -> str:
    package = Path(__file__).parent
    helpers = package.parents[1] / "utils"
    paths = [
        package / name
        for name in ("contract.py", "runner.py", "worker.py", "__main__.py")
    ]
    paths += [
        helpers / name
        for name in (
            "managed_process.py",
            "port_utils.py",
            "http_checks.py",
            "constants.py",
            "output_paths.py",
        )
    ]
    return digest({path.name: file_digest(path) for path in paths})


@dataclass(frozen=True)
class RunOptions:
    corpus: Path
    model_dir: Path
    output_dir: Path
    limit: int | None = None
    schedules: tuple[str, ...] = ("whole", "one_token")
    modes: tuple[str, ...] = MODES
    concurrency: int = 4
    baseline: Path | None = None
    request_timeout: float = 60


def comparison_identity(
    corpus: Corpus, options: RunOptions, assets: dict[str, str]
) -> dict[str, Any]:
    return {
        "corpus_sha256": corpus.manifest_sha256,
        "profile_sha256": digest(corpus.profile),
        "assets": assets,
        "selected_ids_sha256": digest([case.identifier for case in corpus.cases]),
        "modes": sorted(options.modes),
        "schedules": sorted(options.schedules),
        "harness_sha256": harness_identity(),
        "policy": "frontend-replay-judgment/v1",
    }


def _request(
    case: Case, model_name: str, execution_id: str, schedule: str, mode: str
) -> tuple[dict[str, Any], list[str]]:
    request = copy.deepcopy(case.request)
    adapted = []
    for key in ("return_token_ids", "return_prompt_token_ids", "return_meta_info"):
        if key in request:
            request.pop(key)
            adapted.append(key)
    request["model"] = model_name
    request["stream"] = mode == "sse"
    nvext = request.setdefault("nvext", {})
    annotations = nvext.setdefault("annotations", [])
    if any(
        isinstance(item, str) and item.startswith(ANNOTATION_PREFIX)
        for item in annotations
    ):
        raise ReplayError("source request already contains a replay annotation")
    annotations.append(
        ANNOTATION_PREFIX
        + dumps(
            {
                "execution_id": execution_id,
                "case_id": case.identifier,
                "schedule": schedule,
            }
        )
    )
    if mode == "sse":
        request.setdefault("stream_options", {})["include_usage"] = True
    else:
        request.pop("stream_options", None)
    return request, adapted


def _exchange(
    case: Case,
    mode: str,
    schedule: str,
    corpus: Corpus,
    options: RunOptions,
    base_url: str,
) -> dict[str, Any]:
    execution_id = uuid.uuid4().hex
    request, adapted = _request(
        case, corpus.profile["model_name"], execution_id, schedule, mode
    )
    result: dict[str, Any] = {
        "key": [case.identifier, mode, schedule],
        "execution_id": execution_id,
        "source": case.source,
        "preprocessed": None,
        "response": None,
        "failures": [],
        "infra_errors": [],
        "response_judged": case.expected_response is not None,
    }
    raw: Any = None
    status = None
    started = time.monotonic()
    with requests.Session() as session:
        session.trust_env = False
        try:
            with session.post(
                base_url + "/v1/chat/completions",
                data=dumps(request).encode(),
                headers={"Content-Type": "application/json"},
                stream=mode == "sse",
                timeout=(5, options.request_timeout),
            ) as response:
                status = response.status_code
                if status != 200:
                    raw = response.text
                    result["failures"].append(
                        {
                            "boundary": "response",
                            "path": "/http_status",
                            "actual": status,
                            "expected": 200,
                        }
                    )
                else:
                    try:
                        if mode == "sse":
                            response.encoding = "utf-8"
                            raw = list(response.iter_lines(decode_unicode=True))
                            result["response"] = assemble_sse(raw)
                        else:
                            raw = response.text
                            result["response"] = normalize_unary(loads(raw))
                    except (ReplayError, KeyError, TypeError) as exc:
                        result["failures"].append(
                            {"boundary": "response", "path": "/wire", "error": str(exc)}
                        )
        except requests.RequestException as exc:
            result["infra_errors"].append({"stage": "http", "error": str(exc)})
    observation = options.output_dir / "worker" / f"{execution_id}.request.json"
    emission = options.output_dir / "worker" / f"{execution_id}.emission.json"
    if observation.is_file():
        result["preprocessed"] = loads(observation.read_bytes())
        result["failures"].extend(
            {"boundary": "preprocessed", **failure}
            for failure in judge_preprocessed(case, result["preprocessed"])
        )
    else:
        result["infra_errors"].append(
            {"stage": "worker", "error": "missing worker-ingress observation"}
        )
    if emission.is_file():
        delivered = loads(emission.read_bytes())
        expected_ids = [
            token
            for frame in scheduled_events(case, schedule)
            for token in frame["token_ids"]
        ]
        if (
            delivered.get("state") != "completed"
            or delivered.get("terminal_count") != 1
            or not same_json(delivered.get("token_ids"), expected_ids)
        ):
            result["infra_errors"].append(
                {
                    "stage": "worker",
                    "error": "worker did not emit the complete selected token sequence",
                }
            )
    else:
        result["infra_errors"].append(
            {"stage": "worker", "error": "missing emission receipt"}
        )
    if result["response"] is not None:
        result["failures"].extend(
            {"boundary": "response", **failure}
            for failure in judge_response(case, result["response"])
        )
    result["latency_ms"] = (time.monotonic() - started) * 1000
    result["status"] = (
        "infra_error"
        if result["infra_errors"]
        else "fail"
        if result["failures"]
        else "pass"
        if result["response_judged"]
        else "unjudged"
    )
    result["preprocessed_sha256"] = (
        digest(_stable_ir(result["preprocessed"]))
        if result["preprocessed"] is not None
        else None
    )
    result["response_sha256"] = (
        digest(_stable_response(result["response"]))
        if result["response"] is not None
        else None
    )
    write_json(
        options.output_dir / "wire" / f"{execution_id}.json",
        {
            "request": request,
            "adapted_capture_fields": adapted,
            "http_status": status,
            "response": raw,
        },
    )
    return result


def _baseline(
    directory: Path | None, identity: dict[str, Any]
) -> tuple[str, dict[str, dict[str, Any]]]:
    if directory is None or not (directory / "summary.json").is_file():
        return "baseline_unavailable", {}
    summary = loads((directory / "summary.json").read_bytes())
    if not same_json(summary.get("identity"), identity):
        return "baseline_not_comparable", {}
    path = directory / "observations.jsonl"
    if not path.is_file():
        return "baseline_unavailable", {}
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = loads(line)
            key = dumps(record["key"])
            if key in rows:
                raise ReplayError("duplicate baseline observation key")
            rows[key] = {
                name: record.get(name)
                for name in (
                    "status",
                    "preprocessed_sha256",
                    "response_sha256",
                    "response_judged",
                )
            }
    return "compared", rows


def compare_baseline(current: dict[str, Any], previous: dict[str, Any] | None) -> str:
    if previous is None:
        return "missing_observation"
    if previous["status"] == "infra_error" or current["status"] == "infra_error":
        return "infrastructure_unjudged"
    if previous["status"] == "fail" and current["status"] != "fail":
        return "fix"
    if previous["status"] != "fail" and current["status"] == "fail":
        return "new_failure"
    if previous["status"] == current["status"] == "fail":
        return "unchanged_failure"
    changed = any(
        previous.get(key) != current.get(key)
        for key in ("preprocessed_sha256", "response_sha256")
    )
    return "unjudged_change" if changed else "unchanged"


def _environment(output_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    root = Path(__file__).resolve().parents[3]
    env.update(
        PYTHONPATH=str(root),
        PYTHONNOUSERSITE="1",
        DYN_DISCOVERY_BACKEND="file",
        DYN_FILE_KV=str(output_dir / "discovery"),
        DYN_NAMESPACE="replay-" + uuid.uuid4().hex[:12],
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
    )
    env.pop("DYN_SYSTEM_PORT", None)
    for name in (
        "DYN_TCP_RPC_HOST",
        "DYN_TCP_RESPONSE_STREAM_HOST",
        "DYN_EVENT_PLANE_HOST",
    ):
        env[name] = "127.0.0.1"
    return env


def _write_junit(
    path: Path, rows: list[dict[str, Any]], infrastructure: list[str]
) -> None:
    suite = ET.Element(
        "testsuite",
        name="frontend-replay",
        tests=str(len(rows) + len(infrastructure)),
        failures=str(sum(row["status"] == "fail" for row in rows)),
        errors=str(
            sum(row["status"] == "infra_error" for row in rows) + len(infrastructure)
        ),
    )
    for row in rows:
        case = ET.SubElement(
            suite,
            "testcase",
            name=" / ".join(row["key"]),
            time=str(row.get("latency_ms", 0) / 1000),
        )
        if row["status"] in ("fail", "infra_error"):
            element = ET.SubElement(
                case, "error" if row["status"] == "infra_error" else "failure"
            )
            element.text = dumps(
                {"failures": row["failures"], "infra_errors": row["infra_errors"]}
            )
        elif row["status"] == "unjudged":
            ET.SubElement(case, "skipped", message="response has no independent golden")
    for index, error in enumerate(infrastructure):
        case = ET.SubElement(suite, "testcase", name=f"infrastructure / {index + 1}")
        ET.SubElement(case, "error", message="Replay incomplete").text = error
    path.write_bytes(ET.tostring(suite, encoding="utf-8", xml_declaration=True))
    path.chmod(0o600)


def run(options: RunOptions) -> tuple[int, dict[str, Any]]:
    if options.output_dir.exists() and any(options.output_dir.iterdir()):
        raise ReplayError("output directory must be new or empty")
    options.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (
        options.concurrency <= 0
        or not options.modes
        or not options.schedules
        or len(set(options.modes)) != len(options.modes)
        or len(set(options.schedules)) != len(options.schedules)
    ):
        raise ReplayError(
            "concurrency must be positive; modes/schedules must be nonempty and unique"
        )
    if set(options.modes) - set(MODES) or set(options.schedules) - set(SCHEDULES):
        raise ReplayError("unsupported mode or schedule")
    corpus = load_corpus(options.corpus, options.limit)
    assets = verify_assets(corpus, options.model_dir)
    identity = comparison_identity(corpus, options, assets)
    baseline_status, previous = _baseline(options.baseline, identity)
    rows: list[dict[str, Any]] = []
    invariants: dict[str, tuple[str | None, str | None]] = {}
    comparisons: dict[str, int] = {}
    port = allocate_port(DynamoPortRange.FRONTEND.value)
    env = _environment(options.output_dir.resolve())
    env["DYN_FILE_KV"] = str((options.output_dir / "discovery").resolve())
    Path(env["DYN_FILE_KV"]).mkdir(mode=0o700)
    endpoint = env["DYN_NAMESPACE"] + ".replay.generate"
    base_url = f"http://127.0.0.1:{port}"
    worker_command = [
        sys.executable,
        "-m",
        "tests.frontend.trace_replay.worker",
        "--corpus",
        str(corpus.manifest_path),
        "--model-dir",
        str(options.model_dir.resolve()),
        "--output-dir",
        str(options.output_dir.resolve()),
        "--endpoint",
        endpoint,
    ]
    if options.limit is not None:
        worker_command += ["--limit", str(options.limit)]
    frontend_command = [
        sys.executable,
        "-m",
        "dynamo.frontend",
        "--http-host",
        "127.0.0.1",
        "--http-port",
        str(port),
        "--dyn-chat-processor",
        "dynamo",
        "--reasoning-field-name",
        "reasoning_content",
        "--discovery-backend",
        "file",
        "--request-plane",
        "tcp",
        "--event-plane",
        "zmq",
        "--router-mode",
        "round-robin",
    ]
    build = None
    infrastructure = []
    observation_path = options.output_dir / "observations.jsonl"
    try:
        with ExitStack() as stack:
            ready = options.output_dir / "worker-ready.json"
            stack.enter_context(
                ManagedProcess(
                    command=worker_command,
                    env=env,
                    health_check_funcs=[lambda: ready.is_file()],
                    timeout=120,
                    terminate_all_matching_process_names=False,
                    log_dir=str((options.output_dir / "logs" / "worker").resolve()),
                    display_name="replay-worker",
                )
            )
            build = loads(ready.read_bytes())
            if (
                build.get("manifest_sha256") != corpus.manifest_sha256
                or build.get("selected_ids_sha256") != identity["selected_ids_sha256"]
            ):
                raise ReplayError("worker loaded a different corpus or case selection")
            stack.enter_context(
                ManagedProcess(
                    command=frontend_command,
                    env=env,
                    health_check_urls=[
                        (
                            base_url + "/v1/models",
                            lambda response: model_registered(
                                response, model=corpus.profile["model_name"]
                            ),
                        )
                    ],
                    timeout=120,
                    terminate_all_matching_process_names=False,
                    log_dir=str((options.output_dir / "logs" / "frontend").resolve()),
                    display_name="replay-frontend",
                )
            )
            tasks = iter(
                (case, mode, schedule)
                for case in corpus.cases
                for schedule in options.schedules
                for mode in options.modes
            )
            with observation_path.open(
                "x", encoding="utf-8"
            ) as observations, ThreadPoolExecutor(
                max_workers=options.concurrency
            ) as pool:
                observation_path.chmod(0o600)
                pending = set()
                for _ in range(options.concurrency):
                    task = next(tasks, None)
                    if task is not None:
                        pending.add(
                            pool.submit(_exchange, *task, corpus, options, base_url)
                        )
                while pending:
                    completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in completed:
                        row = future.result()
                        fingerprints = (
                            row["preprocessed_sha256"],
                            row["response_sha256"],
                        )
                        prior = invariants.get(row["key"][0])
                        if row["status"] != "infra_error":
                            if prior is None:
                                invariants[row["key"][0]] = fingerprints
                            elif prior != fingerprints:
                                row["failures"].append(
                                    {
                                        "boundary": "invariance",
                                        "path": "/delivery",
                                        "error": "same case changed under another mode or token grouping",
                                    }
                                )
                                row["status"] = "fail"
                        if baseline_status == "compared":
                            comparison = compare_baseline(
                                row, previous.get(dumps(row["key"]))
                            )
                            row["baseline"] = comparison
                            comparisons[comparison] = comparisons.get(comparison, 0) + 1
                        observations.write(dumps(row) + "\n")
                        observations.flush()
                        # Retain only compact result metadata; raw IR/wire live on disk.
                        rows.append(
                            {
                                key: row[key]
                                for key in (
                                    "key",
                                    "status",
                                    "failures",
                                    "infra_errors",
                                    "latency_ms",
                                    "response_judged",
                                )
                            }
                        )
                        task = next(tasks, None)
                        if task is not None:
                            pending.add(
                                pool.submit(_exchange, *task, corpus, options, base_url)
                            )
    except (OSError, RuntimeError, ReplayError, requests.RequestException) as exc:
        infrastructure.append(str(exc))
    finally:
        deallocate_port(port)
    planned = len(corpus.cases) * len(options.modes) * len(options.schedules)
    if len(rows) != planned:
        infrastructure.append(f"completed {len(rows)} of {planned} planned executions")
    counts = {
        status: sum(row["status"] == status for row in rows)
        for status in ("pass", "fail", "unjudged", "infra_error")
    }
    attention = comparisons.get("unjudged_change", 0) + comparisons.get(
        "missing_observation", 0
    )
    exit_code = (
        2
        if infrastructure or counts["infra_error"]
        else 1
        if counts["fail"] or attention
        else 0
    )
    status = (
        "infra_error"
        if exit_code == 2
        else "product_failure"
        if counts["fail"]
        else "review_required"
        if attention
        else "unjudged"
        if counts["unjudged"]
        else "pass"
    )
    summary = {
        "schema": "dynamo-frontend-replay-result/v1",
        "status": status,
        "exit_code": exit_code,
        "identity": identity,
        "build": build,
        "harness_revision": os.environ.get("DYN_REPLAY_HARNESS_REVISION"),
        "image_digest": os.environ.get("DYN_REPLAY_IMAGE_DIGEST"),
        "coverage": {
            "corpus_cases": corpus.total_cases,
            "selected_cases": len(corpus.cases),
            "executions_planned": planned,
            "executions_completed": len(rows),
            "response_judged": sum(row["response_judged"] for row in rows),
        },
        "counts": counts,
        "infrastructure_errors": infrastructure,
        "baseline": {"status": baseline_status, "counts": comparisons},
    }
    write_json(options.output_dir / "summary.json", summary)
    _write_junit(options.output_dir / "junit.xml", rows, infrastructure)
    return exit_code, summary
