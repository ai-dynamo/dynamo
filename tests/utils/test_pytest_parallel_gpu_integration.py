# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration coverage for GPU-parallel child output, without a GPU."""

from __future__ import annotations

import pytest

from tests.utils import pytest_parallel_gpu

pytestmark = [
    pytest.mark.integration,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.slow,
]


@pytest.mark.timeout(10)  # ~0.6s measured per case; headroom for child startup
@pytest.mark.parametrize(
    "summary_args",
    [[], ["-q"], ["--color=yes"]],
    ids=["fenced", "quiet", "color"],
)
def test_run_parallel_prints_output_of_pass_rescued_by_retry(
    tmp_path, monkeypatch, capsys, summary_args
):
    # A child that fails once and passes on an in-session retry exits 0. Its
    # buffered output must still reach the log, or the failed attempt is lost.
    counter = tmp_path / "attempts"
    test_file = tmp_path / "test_child.py"
    test_file.write_text(
        "import pathlib\n"
        "def test_fails_once():\n"
        f"    counter = pathlib.Path({str(counter)!r})\n"
        "    n = int(counter.read_text()) if counter.exists() else 0\n"
        "    counter.write_text(str(n + 1))\n"
        "    assert n > 0\n"
    )
    test_id = f"{test_file}::test_fails_once"
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    monkeypatch.setattr(pytest_parallel_gpu, "_JUNIT_DIR", str(tmp_path / "junit"))
    monkeypatch.setattr(
        pytest_parallel_gpu, "_JUNIT_COMBINED", str(tmp_path / "junit" / "combined.xml")
    )
    monkeypatch.setattr(
        pytest_parallel_gpu,
        "detect_gpus",
        lambda: [{"index": 0, "name": "fake", "total_mib": 1024}],
    )
    monkeypatch.setattr(pytest_parallel_gpu, "_get_gpu_used_gib", lambda gi=0: 0.0)
    real_sleep = pytest_parallel_gpu.time.sleep
    monkeypatch.setattr(
        pytest_parallel_gpu.time, "sleep", lambda s: real_sleep(min(s, 0.05))
    )

    rc = pytest_parallel_gpu.run_parallel(
        test_ids=[test_id],
        meta={test_id: {"profiled_vram_gib": 0.0, "timeout": 60}},
        max_vram_gib=0.0,
        num_slots=1,
        extra_pytest_args=[
            "-p",
            "no:cacheprovider",
            "-p",
            "no:ddtrace",
            "--reruns",
            "1",
            "--color=no",
            *summary_args,
        ],
    )

    err = pytest_parallel_gpu._ANSI_ESCAPE_RE.sub("", capsys.readouterr().err)
    assert rc == 0
    assert counter.read_text() == "2"
    assert "1 passed, 1 rerun" in err
    assert "PASSED (after in-session retry)" in err
