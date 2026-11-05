# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
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
        lambda cmd, **kwargs: (
            calls.append((cmd, kwargs))
            or subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        ),
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
            {"capture_output": True, "text": True},
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
        lambda cmd, **kwargs: (
            calls.append((cmd, kwargs))
            or subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        ),
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


def _failing_pip(stdout="", stderr=""):
    """Stub a pip run that exits non-zero, honouring ``check`` like the real one.

    Honouring it matters: without it this stub would silently return a non-zero
    CompletedProcess to the pre-fix code, which passed ``check=True`` and never
    inspected the result -- so these tests would fail pre-fix for the wrong
    reason (nothing raised) instead of the right one (the reason pip failed is
    absent from what the caller receives).
    """

    def _run(cmd, **kwargs):
        if kwargs.get("check"):
            raise subprocess.CalledProcessError(1, cmd, output=stdout, stderr=stderr)
        return subprocess.CompletedProcess(cmd, 1, stdout=stdout, stderr=stderr)

    return _run


def test_pip_failure_carries_pip_output_to_the_caller(monkeypatch, tmp_path):
    """The reason pip failed must reach whoever catches this.

    With check=True and inherited streams, pytest swallowed pip's output and the
    caller saw only a CalledProcessError naming the argv -- so an index outage
    and a bad version pin were indistinguishable from the failure alone.
    """
    monkeypatch.setattr(
        common.tempfile, "mkdtemp", lambda **_kw: str(tmp_path / "pkgs")
    )
    monkeypatch.setattr(
        common.subprocess,
        "run",
        _failing_pip(stderr="ERROR: Could not find a version that satisfies decord2"),
    )

    with pytest.raises(RuntimeError) as excinfo:
        common._install_test_only_packages(_config(spec="decord2>=3.4.0,<4"))

    message = str(excinfo.value)
    assert "Could not find a version that satisfies decord2" in message
    assert "pip exit 1" in message
    assert "decord2>=3.4.0,<4" in message


def test_pip_failure_redacts_index_credentials(monkeypatch, tmp_path):
    """CI supplies PIP_INDEX_URL from a secret, so the captured output is a sink."""
    monkeypatch.setattr(
        common.tempfile, "mkdtemp", lambda **_kw: str(tmp_path / "pkgs")
    )
    monkeypatch.setattr(
        common.subprocess,
        "run",
        _failing_pip(
            stdout="Looking in indexes: https://ciuser:s3cr3t-token@example.com/simple\n",
            stderr="ERROR: nope",
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        common._install_test_only_packages(_config(spec="decord2>=3.4.0,<4"))

    message = str(excinfo.value)
    assert "s3cr3t-token" not in message
    assert "ciuser" not in message  # the user half can itself be the secret
    assert "****@example.com" in message  # the host stays diagnostic


def test_pip_failure_output_is_bounded(monkeypatch, tmp_path):
    """A backtracking resolve can print megabytes; the exception must not."""
    monkeypatch.setattr(
        common.tempfile, "mkdtemp", lambda **_kw: str(tmp_path / "pkgs")
    )
    monkeypatch.setattr(
        common.subprocess,
        "run",
        _failing_pip(stdout="B" * 500_000, stderr="ERROR: the actual reason"),
    )

    with pytest.raises(RuntimeError) as excinfo:
        common._install_test_only_packages(_config(spec="decord2>=3.4.0,<4"))

    message = str(excinfo.value)
    assert len(message) < common._PIP_OUTPUT_LIMIT + 500
    # Bounded head-and-tail, so the trailing ERROR line survives.
    assert "ERROR: the actual reason" in message


def test_pip_failure_redacts_a_token_only_index_url(monkeypatch, tmp_path):
    """A token-only index URL carries the whole secret before the ``@``.

    ``https://<token>@host/simple`` is valid and has no colon, so a pattern that
    matches only ``user:password@`` copies the token into the exception and from
    there into the CI log.
    """
    monkeypatch.setattr(
        common.tempfile, "mkdtemp", lambda **_kw: str(tmp_path / "pkgs")
    )
    monkeypatch.setattr(
        common.subprocess,
        "run",
        _failing_pip(
            stdout="Looking in indexes: https://ghp-s3cr3t-token@example.com/simple\n",
            stderr="ERROR: nope",
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        common._install_test_only_packages(_config(spec="decord2>=3.4.0,<4"))

    message = str(excinfo.value)
    assert "ghp-s3cr3t-token" not in message
    assert "****@example.com" in message


def test_plain_urls_are_left_alone(monkeypatch, tmp_path):
    """Control: a URL with no userinfo must survive untouched."""
    monkeypatch.setattr(
        common.tempfile, "mkdtemp", lambda **_kw: str(tmp_path / "pkgs")
    )
    monkeypatch.setattr(
        common.subprocess,
        "run",
        _failing_pip(
            stdout="Looking in indexes: https://example.com/simple\n",
            stderr="ERROR: see https://docs.example.com/help",
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        common._install_test_only_packages(_config(spec="decord2>=3.4.0,<4"))

    message = str(excinfo.value)
    assert "https://example.com/simple" in message
    assert "https://docs.example.com/help" in message
    assert "****" not in message
