# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the opt-in media-decoder runtime installer.

The pip install is fully mocked -- these tests never touch the network or the
real site-packages.
"""

from __future__ import annotations

import contextlib
import subprocess

import pytest

from dynamo.common.utils import media_decoders

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.fixture
def clean_env(monkeypatch):
    """Reset the three env vars, the process guard, and stub out the lock."""
    for key in (
        media_decoders.ENABLE_ENV,
        media_decoders.PACKAGES_ENV,
        media_decoders.PIP_ARGS_ENV,
    ):
        monkeypatch.delenv(key, raising=False)
    media_decoders._completed.clear()
    # Never take a real file lock or rewrite import caches in unit tests.
    monkeypatch.setattr(
        media_decoders, "_cross_process_lock", lambda: contextlib.nullcontext()
    )
    monkeypatch.setattr(media_decoders.importlib, "invalidate_caches", lambda: None)
    yield monkeypatch
    media_decoders._completed.clear()


def _record_pip(monkeypatch, *, fail: bool = False) -> list[list[str]]:
    """Replace subprocess.run with a recorder; return the list of commands."""
    calls: list[list[str]] = []

    def fake_run(cmd, check=False, **kwargs):
        calls.append(list(cmd))
        if fail:
            raise subprocess.CalledProcessError(1, cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(media_decoders.subprocess, "run", fake_run)
    return calls


def _set_available(monkeypatch, present) -> None:
    """Stub _module_available; `present` is a set of importable module names."""
    monkeypatch.setattr(media_decoders, "_module_available", lambda mod: mod in present)


def test_disabled_by_default_is_noop(clean_env):
    calls = _record_pip(clean_env)
    _set_available(clean_env, set())  # nothing installed
    media_decoders.maybe_install_media_decoders("vllm")
    assert calls == []


def test_enabled_but_already_present_skips_install(clean_env):
    clean_env.setenv(media_decoders.ENABLE_ENV, "1")
    calls = _record_pip(clean_env)
    _set_available(clean_env, {"cv2", "av"})
    media_decoders.maybe_install_media_decoders("vllm")
    assert calls == []


def test_vllm_installs_video_and_audio_carriers(clean_env):
    clean_env.setenv(media_decoders.ENABLE_ENV, "true")
    calls = _record_pip(clean_env)
    _set_available(clean_env, set())
    media_decoders.maybe_install_media_decoders("vllm")
    assert len(calls) == 1
    cmd = calls[0]
    assert "--break-system-packages" in cmd
    assert "opencv-python-headless" in cmd  # video carrier
    assert "av" in cmd  # audio (AAC) carrier
    # Dead-ends and cross-backend / encoder carriers must never appear.
    for banned in ("torchcodec", "pynvvideocodec", "decord2", "libx264"):
        assert banned not in cmd


def test_sglang_installs_only_decord(clean_env):
    clean_env.setenv(media_decoders.ENABLE_ENV, "yes")
    calls = _record_pip(clean_env)
    _set_available(clean_env, set())
    media_decoders.maybe_install_media_decoders("sglang")
    assert len(calls) == 1
    cmd = calls[0]
    assert "decord2" in cmd
    assert "opencv-python-headless" not in cmd
    assert "av" not in cmd


def test_installs_only_missing_modules(clean_env):
    clean_env.setenv(media_decoders.ENABLE_ENV, "1")
    calls = _record_pip(clean_env)
    _set_available(clean_env, {"cv2"})  # cv2 present, av missing
    media_decoders.maybe_install_media_decoders("vllm")
    assert len(calls) == 1
    cmd = calls[0]
    assert "av" in cmd
    assert "opencv-python-headless" not in cmd


def test_package_override_installs_verbatim(clean_env):
    clean_env.setenv(media_decoders.ENABLE_ENV, "1")
    clean_env.setenv(
        media_decoders.PACKAGES_ENV, "opencv-python-headless==4.10.0.84 av==12.0.0"
    )
    calls = _record_pip(clean_env)
    # Override bypasses the already-present short-circuit.
    _set_available(clean_env, {"cv2", "av"})
    media_decoders.maybe_install_media_decoders("vllm")
    assert len(calls) == 1
    cmd = calls[0]
    assert "opencv-python-headless==4.10.0.84" in cmd
    assert "av==12.0.0" in cmd


def test_extra_pip_args_are_appended(clean_env):
    clean_env.setenv(media_decoders.ENABLE_ENV, "1")
    clean_env.setenv(
        media_decoders.PIP_ARGS_ENV, "--index-url https://mirror/simple --no-deps"
    )
    calls = _record_pip(clean_env)
    _set_available(clean_env, set())
    media_decoders.maybe_install_media_decoders("sglang")
    cmd = calls[0]
    assert "--index-url" in cmd
    assert "https://mirror/simple" in cmd
    assert "--no-deps" in cmd


def test_install_failure_does_not_raise(clean_env):
    clean_env.setenv(media_decoders.ENABLE_ENV, "1")
    calls = _record_pip(clean_env, fail=True)
    _set_available(clean_env, set())
    # Must not propagate -- worker startup continues.
    media_decoders.maybe_install_media_decoders("vllm")
    assert len(calls) == 1  # install was attempted


def test_no_decode_backend_is_noop(clean_env):
    clean_env.setenv(media_decoders.ENABLE_ENV, "1")
    calls = _record_pip(clean_env)
    _set_available(clean_env, set())
    media_decoders.maybe_install_media_decoders("trtllm")
    assert calls == []


def test_unknown_backend_is_noop(clean_env):
    clean_env.setenv(media_decoders.ENABLE_ENV, "1")
    calls = _record_pip(clean_env)
    _set_available(clean_env, set())
    media_decoders.maybe_install_media_decoders("mystery")
    assert calls == []


def test_idempotent_second_call_does_not_reinstall(clean_env):
    clean_env.setenv(media_decoders.ENABLE_ENV, "1")
    calls = _record_pip(clean_env)
    _set_available(clean_env, set())
    media_decoders.maybe_install_media_decoders("sglang")
    media_decoders.maybe_install_media_decoders("sglang")
    assert len(calls) == 1  # second call short-circuits via the process guard


def test_malformed_pip_args_does_not_raise(clean_env):
    clean_env.setenv(media_decoders.ENABLE_ENV, "1")
    clean_env.setenv(media_decoders.PIP_ARGS_ENV, "--find-links '/opt/un balanced")
    calls = _record_pip(clean_env)
    _set_available(clean_env, set())
    # Unbalanced quotes must be caught, not crash worker startup.
    media_decoders.maybe_install_media_decoders("vllm")
    assert calls == []  # never reached the install


def test_failure_does_not_mark_complete_and_retries(clean_env):
    clean_env.setenv(media_decoders.ENABLE_ENV, "1")
    calls = _record_pip(clean_env, fail=True)
    _set_available(clean_env, set())
    media_decoders.maybe_install_media_decoders("vllm")
    assert "vllm" not in media_decoders._completed  # a failure is retryable
    media_decoders.maybe_install_media_decoders("vllm")
    assert len(calls) == 2  # retried on the next call


def test_default_install_uses_no_deps(clean_env):
    clean_env.setenv(media_decoders.ENABLE_ENV, "1")
    calls = _record_pip(clean_env)
    _set_available(clean_env, set())
    media_decoders.maybe_install_media_decoders("sglang")
    assert "--no-deps" in calls[0]  # default install must not perturb the stack


def test_override_install_keeps_deps(clean_env):
    clean_env.setenv(media_decoders.ENABLE_ENV, "1")
    clean_env.setenv(media_decoders.PACKAGES_ENV, "av==12.0.0")
    calls = _record_pip(clean_env)
    _set_available(clean_env, set())
    media_decoders.maybe_install_media_decoders("vllm")
    assert "av==12.0.0" in calls[0]
    assert "--no-deps" not in calls[0]  # operator-chosen specs resolve deps


def test_pip_install_is_bounded_by_timeout(clean_env):
    clean_env.setenv(media_decoders.ENABLE_ENV, "1")
    seen: dict = {}

    def fake_run(cmd, check=False, **kwargs):
        seen["cmd"] = list(cmd)
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0)

    clean_env.setattr(media_decoders.subprocess, "run", fake_run)
    _set_available(clean_env, set())
    media_decoders.maybe_install_media_decoders("sglang")
    assert seen["kwargs"].get("timeout") == 600  # default bound


def test_custom_timeout_override(clean_env):
    clean_env.setenv(media_decoders.ENABLE_ENV, "1")
    clean_env.setenv(media_decoders.TIMEOUT_ENV, "42")
    seen: dict = {}

    def fake_run(cmd, check=False, **kwargs):
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0)

    clean_env.setattr(media_decoders.subprocess, "run", fake_run)
    _set_available(clean_env, set())
    media_decoders.maybe_install_media_decoders("sglang")
    assert seen["kwargs"].get("timeout") == 42


def test_redact_masks_url_credentials():
    red = media_decoders._redact(
        "pip install --index-url https://user:tok@mirror/simple decord2"
    )
    assert "user:tok" not in red
    assert "https://***@mirror/simple" in red


def test_pending_subset_installs_only_still_missing(clean_env):
    clean_env.setenv(media_decoders.ENABLE_ENV, "1")
    calls = _record_pip(clean_env)
    # vLLM decoders are (cv2, av). First scan: both missing. Under the lock a
    # peer has meanwhile installed cv2, so only `av` should reach pip.
    seen = {"n": 0}

    def avail(mod):
        seen["n"] += 1
        # Calls 1-2 are the initial missing-scan (both absent). Calls 3+ are the
        # under-lock re-check and post-verify; cv2 now reads as present.
        return seen["n"] > 2 and mod == "cv2"

    clean_env.setattr(media_decoders, "_module_available", avail)
    media_decoders.maybe_install_media_decoders("vllm")
    assert len(calls) == 1
    assert "av" in calls[0]
    assert "opencv-python-headless" not in calls[0]
