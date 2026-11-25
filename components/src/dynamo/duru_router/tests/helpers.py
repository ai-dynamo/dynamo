# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def memory_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("DURU_ACCOUNTING_TRANSPORT", "memory")
    monkeypatch.delenv("DURU_RECORD_EVENTS", raising=False)
    with tempfile.TemporaryDirectory(prefix="duru-test-") as directory:
        monkeypatch.setenv("DURU_IPC_DIR", directory)
        yield


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
        assert self.path.stat().st_size >= self.offset, "Journal truncated"
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
                self.closed = r["event"] == "footer"
                self.sequence += 1
                assert r["sequence"] == self.sequence, (
                    "Journal gap",
                    self.path,
                    self.sequence,
                )
                self.received_through = max(self.received_through, r["time"])
                result.append(r)
            self.offset = f.tell()
        return result

    def close(self):
        pass


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
