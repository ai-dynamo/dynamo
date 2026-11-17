# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exercise the Parse example without a server, model package or Curator install."""
import importlib.util
import json
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pyarrow as pa
import pytest

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

SOURCE = Path(__file__).resolve().parents[1] / "curator/batch_client.py"
spec = importlib.util.spec_from_file_location("handoff_batch_client", SOURCE)
client_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client_module)


def completion():
    return {
        "id": "reply-1",
        "model": "parse",
        "choices": [{"finish_reason": "stop", "message": {"content": "decoded"}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    }


def result(custom_id):
    return {
        "custom_id": custom_id,
        "error": None,
        "response": {"status_code": 200, "body": completion()},
    }


def chunk(*ids):
    return client_module._input_chunks(
        {key: {"model": "parse"} for key in ids}, max_bytes=10000, max_lines=10
    )[0]


def transport_for(rows, *, counts=None, status_id="job-1", create_failure=False):
    calls = []
    polls = 0

    def handler(request):
        nonlocal polls
        calls.append((request.method, request.url.path))
        if request.url.path == "/v1/files":
            assert request.method == "POST"
            body = request.read()
            assert b'name="purpose"' in body and b"batch" in body
            assert b'"custom_id":"a"' in body
            return httpx.Response(200, json={"id": "input-1"})
        if request.url.path == "/v1/batches" and request.method == "POST":
            assert json.loads(request.read()) == {
                "input_file_id": "input-1",
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            }
            if create_failure:
                raise httpx.ReadTimeout("acknowledgement lost", request=request)
            return httpx.Response(200, json={"id": "job-1"})
        if request.url.path.endswith("/cancel"):
            return httpx.Response(200, json={"id": "job-1", "status": "cancelled"})
        if request.url.path == "/v1/batches/job-1":
            polls += 1
            if polls == 1:
                return httpx.Response(
                    200, json={"id": status_id, "status": "in_progress"}
                )
            return httpx.Response(
                200,
                json={
                    "id": status_id,
                    "status": "completed",
                    "output_file_id": "output-1",
                    "request_counts": (
                        counts
                        if counts is not None
                        else {"total": len(rows), "completed": len(rows), "failed": 0}
                    ),
                },
            )
        if request.url.path == "/v1/files/output-1/content":
            return httpx.Response(
                200, content="\n".join(json.dumps(row) for row in rows).encode()
            )
        raise AssertionError(str(request.url))

    return httpx.MockTransport(handler), calls


def test_completed_job_associates_reordered_output(monkeypatch):
    monkeypatch.setattr(client_module.time, "sleep", lambda _seconds: None)
    transport, calls = transport_for([result("b"), result("a")])
    with httpx.Client(transport=transport) as http:
        gateway = client_module.BatchClient("https://batch.test/v1", client=http)
        output = gateway.run([chunk("a", "b")])
    assert set(output) == {"a", "b"}
    assert gateway.requests_confirmed_submitted == 2
    assert calls == [
        ("POST", "/v1/files"),
        ("POST", "/v1/batches"),
        ("GET", "/v1/batches/job-1"),
        ("GET", "/v1/batches/job-1"),
        ("GET", "/v1/files/output-1/content"),
    ]


@pytest.mark.parametrize(
    "rows,counts,match",
    [
        ([result("a"), result("a")], None, "duplicate"),
        ([result("other")], None, "Unknown"),
        ([], None, "do not cover"),
        ([result("a")], {"total": 1, "completed": 0, "failed": 1}, "counts disagree"),
    ],
)
def test_terminal_results_must_match_submitted_ids(monkeypatch, rows, counts, match):
    monkeypatch.setattr(client_module.time, "sleep", lambda _seconds: None)
    transport, _ = transport_for(rows, counts=counts)
    with httpx.Client(transport=transport) as http:
        with pytest.raises(RuntimeError, match=match):
            client_module.BatchClient("https://batch.test/v1", client=http).run(
                [chunk("a")]
            )


def test_status_identity_must_match_job():
    transport, _ = transport_for([result("a")], status_id="foreign-job")
    with httpx.Client(transport=transport) as http:
        with pytest.raises(RuntimeError, match="different job ID"):
            client_module.BatchClient("https://batch.test/v1", client=http).run(
                [chunk("a")]
            )


def test_ambiguous_create_is_not_retried():
    transport, calls = transport_for([result("a")], create_failure=True)
    with httpx.Client(transport=transport) as http:
        gateway = client_module.BatchClient("https://batch.test/v1", client=http)
        with pytest.raises(RuntimeError, match="acknowledgement lost"):
            gateway.run([chunk("a")])
    assert calls == [("POST", "/v1/files"), ("POST", "/v1/batches")]
    assert gateway.request_submission_attempts == 1
    assert gateway.requests_confirmed_submitted == 0
    assert not gateway.jobs


def test_expired_deadline_sends_nothing():
    calls = []
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: calls.append(request))
    ) as http:
        with pytest.raises(TimeoutError):
            client_module.BatchClient(
                "https://batch.test/v1",
                client=http,
                absolute_deadline_monotonic=time.monotonic() - 1,
            )
    assert calls == []


@pytest.mark.parametrize("file_cap,body_cap", [(1, 10000), (10000, 10000)])
def test_actual_upload_caps_before_network(file_cap, body_cap):
    payload = chunk("a")
    if file_cap == 10000:
        file_cap = body_cap = len(payload)
    calls = []
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: calls.append(request))
    ) as http:
        gateway = client_module.BatchClient(
            "https://batch.test/v1",
            client=http,
            max_upload_file_bytes=file_cap,
            max_upload_body_bytes=body_cap,
        )
        with pytest.raises(RuntimeError, match="upload cap"):
            gateway.run([payload])
    assert calls == []


@pytest.mark.parametrize("change", ["model", "id", "finish_reason", "content", "usage"])
def test_response_validation_rejects_invalid_success(change):
    body = completion()
    if change == "model":
        body["model"] = "other"
    elif change == "id":
        body["id"] = ""
    elif change == "finish_reason":
        body["choices"][0]["finish_reason"] = "length"
    elif change == "content":
        body["choices"][0]["message"]["content"] = " "
    else:
        body["usage"]["total_tokens"] = 6
    with pytest.raises(ValueError):
        client_module._validate_response(body, "parse")


@dataclass
class FakeTask:
    data: object
    dataset_name: str = "synthetic"
    _metadata: dict = field(default_factory=dict)
    _stage_perf: dict = field(default_factory=dict)

    def to_pyarrow(self):
        return self.data


def test_zero_block_page_fails_and_preserves_good_peer(monkeypatch):
    tasks_module = types.ModuleType("nemo_curator.tasks")
    tasks_module.InterleavedBatch = FakeTask
    monkeypatch.setitem(sys.modules, "nemo_curator.tasks", tasks_module)
    monkeypatch.setattr(
        client_module, "_request", lambda _model, _image: {"model": "parse"}
    )
    monkeypatch.setattr(
        client_module,
        "_replay",
        lambda image, _model, _body: {
            "blocks": [] if image == b"empty" else [{"text": "kept"}]
        },
    )
    monkeypatch.setattr(client_module.time, "sleep", lambda _seconds: None)
    transport, _ = transport_for_inference()
    task = FakeTask(
        pa.Table.from_pylist(
            [
                {
                    "sample_id": "pdf",
                    "page_number": 0,
                    "binary_content": b"empty",
                    "materialize_error": None,
                },
                {
                    "sample_id": "pdf",
                    "page_number": 1,
                    "binary_content": b"good",
                    "materialize_error": None,
                },
            ]
        )
    )
    accounting = {}
    timings = {}
    with httpx.Client(transport=transport) as http:
        gateway = client_module.BatchClient("https://batch.test/v1", client=http)
        inferred = client_module.infer_tasks(
            [task],
            model_name="parse",
            gateway=gateway,
            accounting=accounting,
            timings=timings,
        )
    rows = inferred[0].to_pyarrow().to_pylist()
    assert rows[0]["parse_document"] is None
    assert "no decoded page blocks" in rows[0]["materialize_error"]
    assert json.loads(rows[1]["parse_document"])["blocks"] == [{"text": "kept"}]
    assert rows[1]["materialize_error"] is None
    assert set(accounting["rejected_responses"]) == {"page-0-0"}
    assert set(accounting["responses"]) == {"page-0-1"}
    assert timings["client_profile"] is gateway.profile
    assert timings["client_profile"]["runs_completed"] == 1
    assert all(
        timings[key] >= 0 for key in ("request_build_s", "gateway_s", "replay_s")
    )


def transport_for_inference():
    def handler(request):
        path = request.url.path
        if path == "/v1/files":
            return httpx.Response(200, json={"id": "input-1"})
        if path == "/v1/batches":
            return httpx.Response(200, json={"id": "job-1"})
        if path == "/v1/batches/job-1":
            return httpx.Response(
                200,
                json={
                    "id": "job-1",
                    "status": "completed",
                    "output_file_id": "output-1",
                    "request_counts": {"total": 2, "completed": 2, "failed": 0},
                },
            )
        if path == "/v1/files/output-1/content":
            return httpx.Response(
                200,
                content="\n".join(
                    json.dumps(result(key)) for key in ["page-0-1", "page-0-0"]
                ).encode(),
            )
        raise AssertionError(path)

    return httpx.MockTransport(handler), []


def test_zero_block_page_cannot_be_removed_from_coverage():
    rows = [
        {
            "sample_id": "pdf",
            "page_number": 0,
            "binary_content": b"image",
            "parse_document": '{"blocks": []}',
            "materialize_error": None,
        }
    ]
    for written in [set(), {("pdf", 0)}]:
        coverage = client_module.page_coverage(rows, written)
        assert not coverage["structural_page_coverage_complete"]
        assert len(coverage["empty_output_pages"]) == 1
        assert not coverage["image_grounding_accepted"]


def test_deadline_after_submission_leaves_remote_cleanup_unresolved(monkeypatch):
    clock = [100.0]
    calls = []
    monkeypatch.setattr(client_module.time, "monotonic", lambda: clock[0])

    def handler(request):
        calls.append((request.method, request.url.path))
        if request.url.path == "/v1/files":
            return httpx.Response(200, json={"id": "input-1"})
        if request.url.path == "/v1/batches":
            return httpx.Response(200, json={"id": "job-1"})
        if request.url.path == "/v1/batches/job-1":
            clock[0] = 111.0
            return httpx.Response(200, json={"id": "job-1", "status": "in_progress"})
        raise AssertionError("No request is permitted after the deadline")

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        gateway = client_module.BatchClient(
            "https://batch.test/v1", client=http, absolute_deadline_monotonic=110.0
        )
        with pytest.raises(RuntimeError, match="deadline expired"):
            gateway.run([chunk("a")])
    assert calls == [
        ("POST", "/v1/files"),
        ("POST", "/v1/batches"),
        ("GET", "/v1/batches/job-1"),
    ]
    assert gateway.jobs == {"job-1": "submitted"}
    assert gateway.requests_confirmed_submitted == 1


def test_client_profile_accumulates_jobs_polls_and_sleep(monkeypatch):
    clock = types.SimpleNamespace(now=0.0)
    events = []
    monkeypatch.setattr(
        client_module,
        "time",
        types.SimpleNamespace(
            perf_counter=lambda: clock.now,
            sleep=lambda seconds: setattr(clock, "now", clock.now + seconds),
        ),
    )
    jobs = {}

    def handler(request):
        clock.now += 0.25
        path = request.url.path
        if path == "/v1/files":
            return httpx.Response(200, json={"id": "input"})
        if path == "/v1/batches":
            job_id = f"job-{len(jobs)}"
            jobs[job_id] = 0
            return httpx.Response(200, json={"id": job_id})
        if path.startswith("/v1/batches/"):
            job_id = path.rsplit("/", 1)[1]
            jobs[job_id] += 1
            return httpx.Response(
                200,
                json={
                    "id": job_id,
                    "status": "in_progress" if jobs[job_id] == 1 else "completed",
                    "output_file_id": job_id,
                    "request_counts": {"total": 1, "completed": 1, "failed": 0},
                },
            )
        job_id = path.split("/")[-2]
        return httpx.Response(
            200, content=json.dumps(result("a" if job_id == "job-0" else "b"))
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        gateway = client_module.BatchClient(
            "https://batch.test/v1", client=http, poll_interval_s=2
        )
        monkeypatch.setattr(
            gateway, "_record", lambda event, **fields: events.append((event, fields))
        )
        gateway.run([chunk("a"), chunk("b")])
        phases = gateway.profile["phases"]
        assert phases["upload"] == {"elapsed_s": 0.5, "calls": 2, "failures": 0}
        assert phases["job_create"] == {"elapsed_s": 0.5, "calls": 2, "failures": 0}
        assert phases["status_http"] == {"elapsed_s": 1.0, "calls": 4, "failures": 0}
        assert phases["result_download"] == {
            "elapsed_s": 0.5,
            "calls": 2,
            "failures": 0,
        }
        assert phases["poll_sleep"] == {"elapsed_s": 2.0, "calls": 1, "failures": 0}
        assert gateway.profile["runs_completed"] == 1
        assert gateway.profile["runs_failed"] == 0
        assert not any("inference" in name or "queue" in name for name in phases)
        assert any(
            event == "client_phase" and fields.get("job_id") == "job-0"
            for event, fields in events
        )


@pytest.mark.parametrize("failure", ["download", "create"])
def test_failed_calls_still_count_and_profile_events_cannot_mask_error(
    monkeypatch, failure
):
    clock = types.SimpleNamespace(now=0.0)
    monkeypatch.setattr(
        client_module, "time", types.SimpleNamespace(perf_counter=lambda: clock.now)
    )

    def handler(request):
        clock.now += 1
        path = request.url.path
        if path == "/v1/files":
            return httpx.Response(200, json={"id": "input"})
        if path == "/v1/batches":
            if failure == "create":
                raise httpx.ReadTimeout("original failure", request=request)
            return httpx.Response(200, json={"id": "job"})
        if path == "/v1/batches/job":
            return httpx.Response(
                200,
                json={"id": "job", "status": "completed", "output_file_id": "output"},
            )
        raise httpx.ReadTimeout("original failure", request=request)

    def record(event, **_fields):
        if event == "client_phase":
            raise OSError("profile disk full")

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        gateway = client_module.BatchClient("https://batch.test/v1", client=http)
        monkeypatch.setattr(gateway, "_record", record)
        with pytest.raises(RuntimeError, match="original failure"):
            gateway.run([chunk("a")])
        counter = gateway.profile["phases"][
            "result_download" if failure == "download" else "job_create"
        ]
        assert counter == {"elapsed_s": 1.0, "calls": 1, "failures": 1}
        assert gateway.profile["runs_failed"] == 1
        assert gateway.profile["runs_completed"] == 0
        gateway.run([])
        assert gateway.profile["runs_started"] == 2
        assert gateway.profile["runs_completed"] == 1
        assert gateway.profile["runs_failed"] == 1
        assert counter["failures"] == 1


def test_local_pipeline_profile_includes_shutdown_after_failure():
    import ast

    source = SOURCE.with_name("batch.py").read_text()
    main = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    function = next(
        node
        for node in main.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_local_pipeline"
    )
    clock = types.SimpleNamespace(now=0.0)
    profiles = {}

    def advance(seconds):
        clock.now += seconds

    runtime = types.SimpleNamespace(
        is_initialized=lambda: False,
        init=lambda **_kwargs: advance(2),
        shutdown=lambda: advance(3),
    )
    namespace = {
        "time": types.SimpleNamespace(perf_counter=lambda: clock.now),
        "ray": runtime,
        "check_deadline": lambda _stage: None,
        "pipeline_profiles": profiles,
        "os": types.SimpleNamespace(environ={"PYTHONPATH": "/test"}),
        "RayDataExecutor": object,
    }
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]), "<driver profile>", "exec"
        ),
        namespace,
    )

    def run(_executor):
        advance(5)
        raise ValueError("pipeline failed")

    with pytest.raises(ValueError, match="pipeline failed"):
        namespace["run_local_pipeline"](types.SimpleNamespace(run=run), phase="prepare")
    assert profiles["prepare"] == {
        "pipeline_run_completed": False,
        "ray_init_s": 2.0,
        "pipeline_run_s": 5.0,
        "ray_shutdown_s": 3.0,
    }
