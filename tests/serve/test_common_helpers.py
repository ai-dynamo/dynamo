# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from tests.serve import common
from tests.utils.engine_process import EngineConfig

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def _config(*, spec: str = "") -> EngineConfig:
    env = {common.TEST_ONLY_PIP_ENV_KEY: spec} if spec else {}
    return EngineConfig(
        name="test",
        directory=".",
        marks=[],
        request_payloads=[],
        model="test",
        command=["true"],
        env=env,
    )


@pytest.fixture(autouse=True)
def clear_test_only_pip_targets():
    common._test_only_pip_targets.clear()
    yield
    common._test_only_pip_targets.clear()


def test_install_test_only_packages_uses_isolated_target(monkeypatch, tmp_path):
    target = tmp_path / "packages"
    calls = []
    monkeypatch.setattr(common.tempfile, "mkdtemp", lambda **_kwargs: str(target))
    monkeypatch.setattr(
        common.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append((cmd, kwargs)),
    )

    env = common._install_test_only_packages(
        _config(spec="decord2>=3.4.0,<4"), {"PYTHONPATH": "/existing"}
    )

    assert calls == [
        (
            [
                common.sys.executable,
                "-m",
                "pip",
                "install",
                "--target",
                str(target),
                "--no-deps",
                "decord2>=3.4.0,<4",
            ],
            {"check": True},
        )
    ]
    assert env["PYTHONPATH"] == f"{target}{common.os.pathsep}/existing"


def test_install_test_only_packages_reuses_successful_install(monkeypatch, tmp_path):
    target = tmp_path / "packages"
    calls = []
    monkeypatch.setattr(common.tempfile, "mkdtemp", lambda **_kwargs: str(target))
    monkeypatch.setattr(
        common.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append((cmd, kwargs)),
    )
    config = _config(spec="decord2>=3.4.0,<4")

    first = common._install_test_only_packages(config)
    second = common._install_test_only_packages(config)

    assert len(calls) == 1
    assert first["PYTHONPATH"] == str(target)
    assert second["PYTHONPATH"] == str(target)


def test_install_test_only_packages_removes_failed_target(monkeypatch, tmp_path):
    target = tmp_path / "packages"
    target.mkdir()
    marker = target / "partial-wheel"
    marker.touch()
    monkeypatch.setattr(common.tempfile, "mkdtemp", lambda **_kwargs: str(target))

    def fail_install(*_args, **_kwargs):
        raise RuntimeError("pip failed")

    monkeypatch.setattr(common.subprocess, "run", fail_install)

    with pytest.raises(RuntimeError, match="pip failed"):
        common._install_test_only_packages(_config(spec="decord2>=3.4.0,<4"))

    assert not Path(target).exists()
    assert not common._test_only_pip_targets
