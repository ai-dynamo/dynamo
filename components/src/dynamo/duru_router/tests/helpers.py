# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared CPU fixtures and recorded-event replay helpers."""

import json
import os
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from dynamo.duru_router import scoring
from dynamo.duru_router.calibration.plan import METRIC
from dynamo.duru_router.events import open_feed


@pytest.fixture(autouse=True)
def memory_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("DURU_ACCOUNTING_TRANSPORT", "memory")
    monkeypatch.delenv("DURU_RECORD_EVENTS", raising=False)
    with tempfile.TemporaryDirectory(prefix="duru-test-") as directory:
        monkeypatch.setenv("DURU_IPC_DIR", directory)
        yield


def configuration(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            dict(
                schema_version=1,
                timing_metric=METRIC,
                contract={"model": "cpu"},
                max_prefill=8,
                prefill=dict(P=[1, 8], D=[0, 8], wall_ms=[[10, 10], [10, 10]]),
                decode=dict(grid=[dict(K=1, wall_ms=10), dict(K=16, wall_ms=10)]),
            )
        )
    )
    cfg = dict(
        ingress=str(tmp_path / "ingress/duru-ingress.jsonl"),
        ledgers=[str(tmp_path / "worker/compact-work.jsonl")],
        scores=str(tmp_path / "scores.json"),
        capacity=1,
        calibration=str(calibration),
        runtime_contract={"model": "cpu"},
        audit_dir=str(tmp_path / "audit"),
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    return path, cfg


class Tail:
    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.sequence = 0
        self.received_through = 0.0
        self.closed = False

    def read(self):
        if not self.path.exists():
            return []
        if self.path.stat().st_size < self.offset:
            raise ValueError(f"Journal truncated: {self.path}")
        result = []
        with self.path.open("rb") as f:
            f.seek(self.offset)
            while True:
                pos = f.tell()
                line = f.readline()
                if not line.endswith(b"\n"):
                    f.seek(pos)
                    break
                if self.closed:
                    raise ValueError(f"Event after journal footer: {self.path}")
                r = json.loads(line)
                if r["sequence"] != self.sequence + 1:
                    raise ValueError(
                        f"Journal gap: {self.path}, expected {self.sequence + 1}"
                    )
                self.closed = r["event"] == "footer"
                self.sequence += 1
                self.received_through = max(self.received_through, r["time"])
                result.append(r)
            self.offset = f.tell()
        return result

    def close(self):
        pass


def replay_accounting(config, stop=None, *, router_publish=None):
    with (
        patch.object(scoring, "EventStreamReader", Tail),
        patch.dict(os.environ, DURU_ACCOUNTING_TRANSPORT="memory"),
    ):
        scoring.run(config, stop, router_publish=router_publish)


def work_step(*, p=100, d=0, root="root", agent="agent", phase="prefill", **fields):
    return dict(
        accepted=True,
        root=root,
        agent=agent,
        phase=phase,
        p=p,
        d=d,
        k=p + d,
        generated_tokens=1,
        **fields,
    )


def arrival_record(
    time=100,
    request_id="req",
    program_id="native/session",
    *,
    root="root",
    agent="agent",
):
    return dict(
        time=time,
        event="request_arrived",
        request_id=request_id,
        program_id=program_id,
        prompt_tokens=8,
        root=root,
        agent=agent,
    )


@contextmanager
def event_feeds(config, ingress_rows, worker_rows, *, in_process):
    """Seed both event sources and close every producer, even if setup fails."""
    ingress = None
    with ExitStack() as stack:
        for logical, rows in [
            (config["ingress"], ingress_rows),
            (config["ledgers"][0], worker_rows),
        ]:
            path = Path(logical)
            if in_process and logical == config["ingress"]:
                ingress = scoring.Ingress(path.parent, in_process=True)
                feed = ingress.journal
            else:
                feed = open_feed(path.parent, name=path.stem)
            stack.callback(feed.close)
            for sequence, row in enumerate(rows, 1):
                feed.submit("record", dict(row, sequence=sequence))
        yield ingress
