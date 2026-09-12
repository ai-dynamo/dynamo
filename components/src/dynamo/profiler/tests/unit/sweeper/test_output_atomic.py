# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.profiler.sweeper.output import atomic as atomic_module

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]


def test_interrupted_replace_keeps_previous_file(monkeypatch, tmp_path) -> None:
    target = tmp_path / "best.yaml"
    target.write_text("old\n")

    def interrupt(_source, _target):
        raise KeyboardInterrupt

    monkeypatch.setattr(atomic_module.os, "replace", interrupt)

    with pytest.raises(KeyboardInterrupt):
        atomic_module.replace_text(target, "new\n")

    assert target.read_text() == "old\n"
    assert list(tmp_path.glob(".*.tmp")) == []
