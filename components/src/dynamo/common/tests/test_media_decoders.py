# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the explicit media-decoder installer.

The pip install is fully mocked -- these tests never touch the network or the
real site-packages.
"""

from __future__ import annotations

import contextlib
import subprocess
from pathlib import Path

import pytest

from dynamo.common.utils import media_decoders

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.fixture
def sandboxed(monkeypatch):
    """Stub out the lock, cache rewrite, and fresh-interpreter probe."""
    monkeypatch.setattr(
        media_decoders, "_cross_process_lock", lambda: contextlib.nullcontext()
    )
    monkeypatch.setattr(media_decoders.importlib, "invalidate_caches", lambda: None)
    # Route the fresh-interpreter probe through _module_available so one stub
    # controls both, and the probe never spawns a real python (which would
    # also collide with the subprocess.run recorders below).
    monkeypatch.setattr(
        media_decoders,
        "_modules_missing_fresh",
        lambda mods: [m for m in mods if not media_decoders._module_available(m)],
    )
    return monkeypatch


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


def _record_pip_kwargs(monkeypatch) -> list[dict]:
    """Replace subprocess.run with a kwargs recorder (for timeout checks)."""
    calls: list[dict] = []

    def fake_run(cmd, check=False, **kwargs):
        calls.append({"cmd": list(cmd), **kwargs})
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(media_decoders.subprocess, "run", fake_run)
    return calls


def _set_available(monkeypatch, present) -> None:
    """Stub _module_available; `present` is a set of importable module names."""
    monkeypatch.setattr(media_decoders, "_module_available", lambda mod: mod in present)


def _record_pip_and_mark(monkeypatch, present, *modules) -> list[list[str]]:
    """Recorder whose successful install makes `modules` importable.

    Mirrors reality: after pip exits 0 the module resolves, so the
    post-install verification in install_media_decoders passes.
    """
    calls: list[list[str]] = []

    def fake_run(cmd, check=False, **kwargs):
        calls.append(list(cmd))
        present.update(modules)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(media_decoders.subprocess, "run", fake_run)
    return calls


# ---------------------------------------------------------------------------
# Version bounds: the review requirement this design encodes.
# ---------------------------------------------------------------------------


def test_every_default_spec_has_lower_and_upper_bound():
    """Each validated spec pins a floor and caps the major version."""
    for decoders in media_decoders._BACKEND_DECODERS.values():
        for d in decoders:
            assert ">=" in d.spec, f"{d.package}: no validated lower bound: {d.spec}"
            assert ",<" in d.spec, f"{d.package}: no upper version cap: {d.spec}"
            assert d.spec.startswith(d.package), (
                f"spec {d.spec!r} does not start with its package name "
                f"{d.package!r}"
            )


def test_validated_specs_cover_every_backend_package():
    """VALIDATED_SPECS (the reuse surface for tests/docs) is complete."""
    for backend, decoders in media_decoders._BACKEND_DECODERS.items():
        for d in decoders:
            assert media_decoders.VALIDATED_SPECS.get(d.package) == d.spec, (
                f"{backend}: {d.package} spec missing or divergent in "
                "VALIDATED_SPECS"
            )


# ---------------------------------------------------------------------------
# Install behavior.
# ---------------------------------------------------------------------------


def test_already_present_installs_nothing(sandboxed):
    calls = _record_pip(sandboxed)
    _set_available(sandboxed, {"cv2", "av"})
    assert media_decoders.install_media_decoders("vllm") == []
    assert calls == []


def test_vllm_installs_bounded_video_and_audio_specs(sandboxed):
    present: set[str] = set()
    _set_available(sandboxed, present)
    calls = _record_pip_and_mark(sandboxed, present, "cv2", "av")
    installed = media_decoders.install_media_decoders("vllm")
    assert installed == [
        media_decoders.VALIDATED_SPECS["opencv-python-headless"],
        media_decoders.VALIDATED_SPECS["av"],
    ]
    (cmd,) = calls
    assert "--break-system-packages" in cmd
    for spec in installed:
        assert spec in cmd
    # Never installed: pynvvideocodec because the image already ships it as the
    # NVDEC path, and the rest because no vLLM decode path imports them.
    for banned in ("torchcodec", "pynvvideocodec", "decord2", "libx264"):
        assert not any(banned in part for part in cmd)


def test_sglang_installs_only_decord(sandboxed):
    present: set[str] = set()
    _set_available(sandboxed, present)
    calls = _record_pip_and_mark(sandboxed, present, "decord")
    installed = media_decoders.install_media_decoders("sglang")
    assert installed == [media_decoders.VALIDATED_SPECS["decord2"]]
    (cmd,) = calls
    assert not any("opencv" in part for part in cmd)


def test_trtllm_installs_opencv_only(sandboxed):
    present: set[str] = set()
    _set_available(sandboxed, present)
    calls = _record_pip_and_mark(sandboxed, present, "cv2")
    installed = media_decoders.install_media_decoders("trtllm")
    assert installed == [media_decoders.VALIDATED_SPECS["opencv-python-headless"]]
    (cmd,) = calls
    assert media_decoders.VALIDATED_SPECS["av"] not in cmd


def test_installs_only_missing_modules(sandboxed):
    present = {"cv2"}  # video carrier present, audio missing
    _set_available(sandboxed, present)
    calls = _record_pip_and_mark(sandboxed, present, "av")
    installed = media_decoders.install_media_decoders("vllm")
    assert installed == [media_decoders.VALIDATED_SPECS["av"]]
    (cmd,) = calls
    assert media_decoders.VALIDATED_SPECS["opencv-python-headless"] not in cmd


def test_default_install_uses_no_deps(sandboxed):
    present: set[str] = set()
    _set_available(sandboxed, present)
    calls = _record_pip_and_mark(sandboxed, present, "decord")
    media_decoders.install_media_decoders("sglang")
    (cmd,) = calls
    assert "--no-deps" in cmd


def test_unknown_backend_raises(sandboxed):
    _record_pip(sandboxed)
    with pytest.raises(ValueError, match="unknown backend"):
        media_decoders.install_media_decoders("mocker")


def test_package_override_installs_verbatim_with_deps(sandboxed):
    calls = _record_pip(sandboxed)
    _set_available(sandboxed, {"cv2", "av"})  # presence must not skip overrides
    installed = media_decoders.install_media_decoders(
        "vllm", packages=["opencv-python-headless==4.13.0.92", "custom-pkg"]
    )
    assert installed == ["opencv-python-headless==4.13.0.92", "custom-pkg"]
    (cmd,) = calls
    assert "--no-deps" not in cmd  # overrides resolve dependencies
    assert "custom-pkg" in cmd


def test_extra_pip_args_are_appended(sandboxed):
    present: set[str] = set()
    _set_available(sandboxed, present)
    calls = _record_pip_and_mark(sandboxed, present, "decord")
    media_decoders.install_media_decoders(
        "sglang", pip_args=["--no-index", "--find-links", "/wheels"]
    )
    (cmd,) = calls
    i = cmd.index("--no-index")
    assert cmd[i : i + 3] == ["--no-index", "--find-links", "/wheels"]


def test_install_failure_raises(sandboxed):
    """Explicit operator action: a broken install must be loud."""
    _record_pip(sandboxed, fail=True)
    _set_available(sandboxed, set())
    with pytest.raises(subprocess.CalledProcessError):
        media_decoders.install_media_decoders("sglang")


def test_module_missing_after_install_raises(sandboxed):
    """pip exiting 0 without producing an importable module is a failure."""
    _record_pip(sandboxed)  # exits 0 but the module never appears
    _set_available(sandboxed, set())
    with pytest.raises(RuntimeError, match="still not importable"):
        media_decoders.install_media_decoders("sglang")


def test_timeout_is_passed_to_pip(sandboxed):
    _set_available(sandboxed, set())
    calls = _record_pip_kwargs(sandboxed)
    with pytest.raises(RuntimeError):  # post-verify fails; timeout already recorded
        media_decoders.install_media_decoders("sglang", timeout_s=42)
    assert calls[0]["timeout"] == 42


def test_none_timeout_disables_bound(sandboxed):
    _set_available(sandboxed, set())
    calls = _record_pip_kwargs(sandboxed)
    with pytest.raises(RuntimeError):
        media_decoders.install_media_decoders("sglang", timeout_s=None)
    assert calls[0]["timeout"] is None


def test_dry_run_reports_without_installing(sandboxed):
    calls = _record_pip(sandboxed)
    _set_available(sandboxed, set())
    specs = media_decoders.install_media_decoders("vllm", dry_run=True)
    assert specs == [
        media_decoders.VALIDATED_SPECS["opencv-python-headless"],
        media_decoders.VALIDATED_SPECS["av"],
    ]
    assert calls == []


def test_pending_subset_installs_only_still_missing(sandboxed):
    """A racing process may install part of the set while we wait on the lock.

    The pre-lock check sees both vLLM carriers missing; by the post-lock
    re-check cv2 has appeared, so only the audio carrier should install.
    """
    present: set[str] = set()
    seen: list[str] = []

    def available(mod: str) -> bool:
        seen.append(mod)
        # cv2 "appears" (installed by the racing process) at its second check,
        # which is the post-lock re-check.
        if seen.count("cv2") >= 2:
            present.add("cv2")
        return mod in present

    sandboxed.setattr(media_decoders, "_module_available", available)
    calls = _record_pip_and_mark(sandboxed, present, "cv2", "av")
    installed = media_decoders.install_media_decoders("vllm")
    assert installed == [media_decoders.VALIDATED_SPECS["av"]]
    (cmd,) = calls
    assert media_decoders.VALIDATED_SPECS["opencv-python-headless"] not in cmd


def test_modules_missing_fresh_real_probe():
    """Unmocked: the fresh-interpreter probe distinguishes real modules.

    The probe exists because pip may install into a user-site directory
    created after the parent interpreter started; only a fresh interpreter
    (like the worker launched later) is guaranteed to see it. Exercise the
    real subprocess path here since every other test stubs it out.
    """
    missing = media_decoders._modules_missing_fresh(
        ["json", "definitely_not_a_module_xyz"]
    )
    assert missing == ["definitely_not_a_module_xyz"]
    assert media_decoders._modules_missing_fresh([]) == []


def test_redact_masks_url_credentials():
    line = "pip install --index-url https://user:secret@pypi.corp/simple pkg"
    masked = media_decoders._redact(line)
    assert "secret" not in masked
    assert "https://***@pypi.corp/simple" in masked


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def test_cli_dry_run_exits_zero_and_skips_pip(sandboxed):
    calls = _record_pip(sandboxed)
    _set_available(sandboxed, set())
    assert media_decoders.main(["vllm", "--dry-run"]) == 0
    assert calls == []


def test_cli_rejects_unknown_backend(sandboxed):
    with pytest.raises(SystemExit) as exc:
        media_decoders.main(["mocker"])
    assert exc.value.code == 2  # argparse usage error


def test_cli_malformed_pip_args_is_usage_error(sandboxed):
    calls = _record_pip(sandboxed)
    with pytest.raises(SystemExit) as exc:
        media_decoders.main(["vllm", "--pip-args", "'unclosed"])
    assert exc.value.code == 2
    assert calls == []


def test_cli_failure_exits_nonzero(sandboxed):
    _record_pip(sandboxed, fail=True)
    _set_available(sandboxed, set())
    assert media_decoders.main(["sglang"]) == 1


def test_cli_zero_timeout_disables_bound(sandboxed):
    _set_available(sandboxed, set())
    calls = _record_pip_kwargs(sandboxed)
    # pip is mocked and no module appears, so the run fails post-verify (exit 1)
    # -- the timeout kwarg it passed is what this test is about.
    assert media_decoders.main(["sglang", "--timeout-s", "0"]) == 1
    assert calls[0]["timeout"] is None


def test_cli_packages_override_reaches_pip(sandboxed):
    calls = _record_pip(sandboxed)
    _set_available(sandboxed, set())
    assert media_decoders.main(["vllm", "--packages", "av==18.0.0"]) == 0
    (cmd,) = calls
    assert "av==18.0.0" in cmd
    assert "--no-deps" not in cmd


def test_cli_pip_args_equals_form_single_flag(sandboxed):
    """--pip-args=--no-index must parse: a spaceless leading-dash value as a
    separate token is an argparse usage error, so the docs teach the = form.
    Found on a real image: the separate-token form exited 2, not 1."""
    present: set[str] = set()
    _set_available(sandboxed, present)
    calls = _record_pip_and_mark(sandboxed, present, "decord")
    assert media_decoders.main(["sglang", "--pip-args=--no-index"]) == 0
    (cmd,) = calls
    assert "--no-index" in cmd


def test_cli_pip_args_reach_pip(sandboxed):
    present: set[str] = set()
    _set_available(sandboxed, present)
    calls = _record_pip_and_mark(sandboxed, present, "decord")
    assert (
        media_decoders.main(["sglang", "--pip-args", "--no-index --find-links /wheels"])
        == 0
    )
    (cmd,) = calls
    assert "--no-index" in cmd and "/wheels" in cmd


# ---------------------------------------------------------------------------
# Nothing implicit: the env-var/startup pathway must not creep back.
# ---------------------------------------------------------------------------

_COMPONENTS_ROOT = Path(__file__).resolve().parents[3]


def test_no_production_code_invokes_the_installer():
    """The installer is operator-run only.

    Reviewers rejected implicit installation at worker startup (env-var gated
    hooks in entrypoints): an install that changes the container's codec
    surface has to be a visible, deliberate step. This sweep keeps any
    `__main__.py` from calling the installer and the retired env switch from
    coming back anywhere under components/src.
    """
    allowed = {
        Path("dynamo/common/utils/media_decoders.py"),  # the installer itself
        Path("dynamo/common/tests/test_media_decoders.py"),  # this test
    }
    offenders: list[str] = []
    for path in sorted(_COMPONENTS_ROOT.rglob("*.py")):
        rel = path.relative_to(_COMPONENTS_ROOT)
        if rel in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if "install_media_decoders" in text or "DYN_ENABLE_MEDIA_DECODERS" in text:
            offenders.append(str(rel))
    assert not offenders, (
        f"{offenders} reference the media-decoder installer; it must stay an "
        "explicit operator command, never wired into worker startup"
    )


def test_installer_module_has_no_env_switches():
    """The module reads no environment variables at all."""
    source = (_COMPONENTS_ROOT / "dynamo/common/utils/media_decoders.py").read_text(
        encoding="utf-8"
    )
    assert "os.environ" not in source and "getenv" not in source, (
        "media_decoders.py reads the environment; configuration belongs in "
        "CLI flags so the install stays explicit and self-describing"
    )
