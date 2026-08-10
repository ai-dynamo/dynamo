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

from dynamo.common.utils import install_media_decoders

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.fixture
def sandboxed(monkeypatch):
    """Stub out the lock, cache rewrite, and fresh-interpreter probe."""
    monkeypatch.setattr(
        install_media_decoders, "_cross_process_lock", lambda: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        install_media_decoders.importlib, "invalidate_caches", lambda: None
    )
    # Default probe stub: nothing importable. Tests refine it via
    # _set_available; stubbing keeps the probe from spawning a real python,
    # which would also collide with the subprocess.run recorders below.
    monkeypatch.setattr(
        install_media_decoders, "_modules_missing_fresh", lambda mods: list(mods)
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

    monkeypatch.setattr(install_media_decoders.subprocess, "run", fake_run)
    return calls


def _record_pip_kwargs(monkeypatch) -> list[dict]:
    """Replace subprocess.run with a kwargs recorder (for timeout checks)."""
    calls: list[dict] = []

    def fake_run(cmd, check=False, **kwargs):
        calls.append({"cmd": list(cmd), **kwargs})
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(install_media_decoders.subprocess, "run", fake_run)
    return calls


def _set_available(monkeypatch, present) -> None:
    """Stub the fresh-import probe; `present` = importable module names.

    `present` is read at call time, so a recorder that updates it after the
    mocked pip run makes post-install verification pass, mirroring reality.
    """
    monkeypatch.setattr(
        install_media_decoders,
        "_modules_missing_fresh",
        lambda mods: [m for m in mods if m not in present],
    )


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

    monkeypatch.setattr(install_media_decoders.subprocess, "run", fake_run)
    return calls


# ---------------------------------------------------------------------------
# Version bounds: the review requirement this design encodes.
# ---------------------------------------------------------------------------


def test_every_default_spec_has_lower_and_upper_bound():
    """Each validated spec pins a floor and caps the major version."""
    for decoders in install_media_decoders._BACKEND_DECODERS.values():
        for d in decoders:
            assert ">=" in d.spec, f"{d.package}: no validated lower bound: {d.spec}"
            assert ",<" in d.spec, f"{d.package}: no upper version cap: {d.spec}"
            assert d.spec.startswith(d.package), (
                f"spec {d.spec!r} does not start with its package name "
                f"{d.package!r}"
            )


def test_validated_specs_cover_every_backend_package():
    """VALIDATED_SPECS (the reuse surface for tests/docs) is complete."""
    for backend, decoders in install_media_decoders._BACKEND_DECODERS.items():
        for d in decoders:
            assert install_media_decoders.VALIDATED_SPECS.get(d.package) == d.spec, (
                f"{backend}: {d.package} spec missing or divergent in "
                "VALIDATED_SPECS"
            )


# ---------------------------------------------------------------------------
# Install behavior.
# ---------------------------------------------------------------------------


def test_already_present_installs_nothing(sandboxed):
    calls = _record_pip(sandboxed)
    _set_available(sandboxed, {"cv2", "av"})
    assert install_media_decoders.install_media_decoders("vllm") == []
    assert calls == []


def test_vllm_installs_bounded_video_and_audio_specs(sandboxed):
    present: set[str] = set()
    _set_available(sandboxed, present)
    calls = _record_pip_and_mark(sandboxed, present, "cv2", "av")
    installed = install_media_decoders.install_media_decoders("vllm")
    assert installed == [
        install_media_decoders.VALIDATED_SPECS["opencv-python-headless"],
        install_media_decoders.VALIDATED_SPECS["av"],
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
    installed = install_media_decoders.install_media_decoders("sglang")
    assert installed == [install_media_decoders.VALIDATED_SPECS["decord2"]]
    (cmd,) = calls
    assert not any("opencv" in part for part in cmd)


def test_trtllm_installs_opencv_only(sandboxed):
    present: set[str] = set()
    _set_available(sandboxed, present)
    calls = _record_pip_and_mark(sandboxed, present, "cv2")
    installed = install_media_decoders.install_media_decoders("trtllm")
    assert installed == [
        install_media_decoders.VALIDATED_SPECS["opencv-python-headless"]
    ]
    (cmd,) = calls
    assert install_media_decoders.VALIDATED_SPECS["av"] not in cmd


def test_installs_only_missing_modules(sandboxed):
    present = {"cv2"}  # video carrier present, audio missing
    _set_available(sandboxed, present)
    calls = _record_pip_and_mark(sandboxed, present, "av")
    installed = install_media_decoders.install_media_decoders("vllm")
    assert installed == [install_media_decoders.VALIDATED_SPECS["av"]]
    (cmd,) = calls
    assert install_media_decoders.VALIDATED_SPECS["opencv-python-headless"] not in cmd


def test_every_install_uses_no_deps(sandboxed):
    """--no-deps is unconditional now that custom specs are out of scope --
    the installer must never be able to shift the image's pinned stack."""
    present: set[str] = set()
    _set_available(sandboxed, present)
    calls = _record_pip_and_mark(sandboxed, present, "decord")
    install_media_decoders.install_media_decoders("sglang")
    (cmd,) = calls
    assert "--no-deps" in cmd


def test_unknown_backend_raises(sandboxed):
    _record_pip(sandboxed)
    with pytest.raises(ValueError, match="unknown backend"):
        install_media_decoders.install_media_decoders("mocker")


def test_extra_pip_args_are_appended(sandboxed):
    present: set[str] = set()
    _set_available(sandboxed, present)
    calls = _record_pip_and_mark(sandboxed, present, "decord")
    install_media_decoders.install_media_decoders(
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
        install_media_decoders.install_media_decoders("sglang")


def test_module_missing_after_install_raises(sandboxed):
    """pip exiting 0 without producing an importable module is a failure."""
    _record_pip(sandboxed)  # exits 0 but the module never appears
    _set_available(sandboxed, set())
    with pytest.raises(RuntimeError, match="still not importable"):
        install_media_decoders.install_media_decoders("sglang")


def test_timeout_is_passed_to_pip(sandboxed):
    _set_available(sandboxed, set())
    calls = _record_pip_kwargs(sandboxed)
    with pytest.raises(RuntimeError):  # post-verify fails; timeout already recorded
        install_media_decoders.install_media_decoders("sglang", timeout_s=42)
    assert calls[0]["timeout"] == 42


def test_none_timeout_disables_bound(sandboxed):
    _set_available(sandboxed, set())
    calls = _record_pip_kwargs(sandboxed)
    with pytest.raises(RuntimeError):
        install_media_decoders.install_media_decoders("sglang", timeout_s=None)
    assert calls[0]["timeout"] is None


def test_dry_run_reports_without_installing(sandboxed):
    calls = _record_pip(sandboxed)
    _set_available(sandboxed, set())
    specs = install_media_decoders.install_media_decoders("vllm", dry_run=True)
    assert specs == [
        install_media_decoders.VALIDATED_SPECS["opencv-python-headless"],
        install_media_decoders.VALIDATED_SPECS["av"],
    ]
    assert calls == []


def test_pending_subset_installs_only_still_missing(sandboxed):
    """A racing process may install part of the set while we wait on the lock.

    Probe round 1 (pre-check) sees both vLLM carriers missing; round 2
    (post-lock re-check) sees cv2 already installed by the racing process, so
    only the audio carrier installs; round 3 (post-verify) sees everything.
    """
    rounds = {"n": 0}

    def probe(mods):
        rounds["n"] += 1
        if rounds["n"] == 1:
            return list(mods)
        if rounds["n"] == 2:
            return [m for m in mods if m != "cv2"]
        return []

    sandboxed.setattr(install_media_decoders, "_modules_missing_fresh", probe)
    calls = _record_pip(sandboxed)
    installed = install_media_decoders.install_media_decoders("vllm")
    assert installed == [install_media_decoders.VALIDATED_SPECS["av"]]
    (cmd,) = calls
    assert install_media_decoders.VALIDATED_SPECS["opencv-python-headless"] not in cmd


def test_modules_missing_fresh_real_probe():
    """Unmocked: the fresh-interpreter probe distinguishes real modules.

    The probe exists because pip may install into a user-site directory
    created after the parent interpreter started; only a fresh interpreter
    (like the worker launched later) is guaranteed to see it. Exercise the
    real subprocess path here since every other test stubs it out.
    """
    missing = install_media_decoders._modules_missing_fresh(
        ["json", "definitely_not_a_module_xyz"]
    )
    assert missing == ["definitely_not_a_module_xyz"]
    assert install_media_decoders._modules_missing_fresh([]) == []


def test_probe_treats_present_but_broken_package_as_missing(tmp_path, monkeypatch):
    """A package whose files exist but whose import fails must count missing.

    This is the review finding: find_spec sees a broken wheel (e.g. deleted
    native libs) and the old probe skipped the install. The probe subprocess
    inherits PYTHONPATH, so plant a module that raises on import.
    """
    pkg = tmp_path / "brokenmod"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("raise RuntimeError('native libs gone')\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    assert install_media_decoders._modules_missing_fresh(["brokenmod"]) == ["brokenmod"]


def test_lock_refuses_symlink_and_preserves_target(tmp_path, monkeypatch):
    """A pre-planted symlink at the fixed lock path must not be followed.

    The old open(_LOCK_PATH, "w") truncated the symlink target; reproduced on
    a real runtime image before the fix. The lock must refuse the symlink
    (O_NOFOLLOW), leave the victim untouched, and fall back to lock-less
    operation rather than fail.
    """
    victim = tmp_path / "victim.txt"
    victim.write_text("SENTINEL")
    link = tmp_path / "lock"
    link.symlink_to(victim)
    monkeypatch.setattr(install_media_decoders, "_LOCK_PATH", link)
    with install_media_decoders._cross_process_lock():
        pass
    assert victim.read_text() == "SENTINEL"


def test_lock_normal_path_creates_private_file(tmp_path, monkeypatch):
    """Without an attacker, the lock file is created 0600 and usable."""
    lock = tmp_path / "lock"
    monkeypatch.setattr(install_media_decoders, "_LOCK_PATH", lock)
    with install_media_decoders._cross_process_lock():
        assert lock.exists()
        assert (lock.stat().st_mode & 0o777) == 0o600


def test_cli_error_log_redacts_credentialed_pip_args(sandboxed, caplog):
    """str(CalledProcessError) embeds the full pip command; the error log
    must mask userinfo from a credentialed --pip-args index URL."""
    _record_pip(sandboxed, fail=True)
    _set_available(sandboxed, set())
    rc = install_media_decoders.main(
        ["sglang", "--pip-args=--index-url https://ci:tok3nzz@pypi.corp/simple"]
    )
    assert rc == 1
    assert "tok3nzz" not in caplog.text
    assert "***@pypi.corp" in caplog.text


def test_redact_masks_url_credentials():
    line = "pip install --index-url https://user:secret@pypi.corp/simple pkg"
    masked = install_media_decoders._redact(line)
    assert "secret" not in masked
    assert "https://***@pypi.corp/simple" in masked


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def test_cli_dry_run_exits_zero_and_skips_pip(sandboxed):
    calls = _record_pip(sandboxed)
    _set_available(sandboxed, set())
    assert install_media_decoders.main(["vllm", "--dry-run"]) == 0
    assert calls == []


def test_cli_rejects_unknown_backend(sandboxed):
    with pytest.raises(SystemExit) as exc:
        install_media_decoders.main(["mocker"])
    assert exc.value.code == 2  # argparse usage error


def test_cli_malformed_pip_args_is_usage_error(sandboxed):
    calls = _record_pip(sandboxed)
    with pytest.raises(SystemExit) as exc:
        install_media_decoders.main(["vllm", "--pip-args", "'unclosed"])
    assert exc.value.code == 2
    assert calls == []


def test_cli_failure_exits_nonzero(sandboxed):
    _record_pip(sandboxed, fail=True)
    _set_available(sandboxed, set())
    assert install_media_decoders.main(["sglang"]) == 1


def test_cli_zero_timeout_disables_bound(sandboxed):
    _set_available(sandboxed, set())
    calls = _record_pip_kwargs(sandboxed)
    # pip is mocked and no module appears, so the run fails post-verify (exit 1)
    # -- the timeout kwarg it passed is what this test is about.
    assert install_media_decoders.main(["sglang", "--timeout-s", "0"]) == 1
    assert calls[0]["timeout"] is None


def test_cli_packages_flag_is_rejected(sandboxed):
    """--packages was removed per review: custom specs are plain pip's job.

    argparse must reject it so the retired customizability surface cannot
    silently come back.
    """
    calls = _record_pip(sandboxed)
    with pytest.raises(SystemExit) as exc:
        install_media_decoders.main(["vllm", "--packages", "custom-pkg"])
    assert exc.value.code == 2
    assert calls == []


def test_cli_pip_args_equals_form_single_flag(sandboxed):
    """--pip-args=--no-index must parse: a spaceless leading-dash value as a
    separate token is an argparse usage error, so the docs teach the = form.
    Found on a real image: the separate-token form exited 2, not 1."""
    present: set[str] = set()
    _set_available(sandboxed, present)
    calls = _record_pip_and_mark(sandboxed, present, "decord")
    assert install_media_decoders.main(["sglang", "--pip-args=--no-index"]) == 0
    (cmd,) = calls
    assert "--no-index" in cmd


def test_cli_pip_args_reach_pip(sandboxed):
    present: set[str] = set()
    _set_available(sandboxed, present)
    calls = _record_pip_and_mark(sandboxed, present, "decord")
    assert (
        install_media_decoders.main(
            ["sglang", "--pip-args", "--no-index --find-links /wheels"]
        )
        == 0
    )
    (cmd,) = calls
    assert "--no-index" in cmd and "/wheels" in cmd


# ---------------------------------------------------------------------------
# Nothing implicit: the env-var/startup pathway must not creep back.
# ---------------------------------------------------------------------------

# The dynamo package root (parents[2] = .../dynamo). parents[3] would be
# components/src in the repo but site-packages in an installed layout, where
# the sweep then walks every third-party package -- including files with
# Python 2 syntax that ast.parse cannot read.
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def test_no_entrypoint_references_the_installer():
    """The installer is operator-run only.

    Reviewers rejected implicit installation at worker startup (env-var gated
    hooks in entrypoints): an install that changes the container's codec
    surface has to be a visible, deliberate step. Entrypoints are where
    startup happens, so no `__main__.py` may reference the installer at all.

    Other production code MAY import its constants -- the actionable
    unsupported-codec errors single-source their version bounds from
    VALIDATED_SPECS -- so this sweep is scoped to entrypoints, and the two
    sweeps below cover the rest: nothing may CALL the installer, and the
    retired env switch must not come back anywhere.
    """
    offenders: list[str] = []
    for path in sorted(_PACKAGE_ROOT.rglob("__main__.py")):
        text = path.read_text(encoding="utf-8")
        if "install_media_decoders" in text or "DYN_ENABLE_MEDIA_DECODERS" in text:
            offenders.append(str(path.relative_to(_PACKAGE_ROOT)))
    assert not offenders, (
        f"{offenders} reference the media-decoder installer from an entrypoint; "
        "it must stay an explicit operator command, never wired into startup"
    )


def _source_calls_installer(source: str) -> bool:
    """AST-based detection of a call into the installer, aliases included.

    A literal `install_media_decoders(` grep misses
    `from ... import install_media_decoders as x; x()` and
    `import ...install_media_decoders as m; m.main()`. Walk the AST instead:
    collect every name the installer module (or its functions) is bound to,
    then flag any Call through one of those bindings.
    """
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Not parseable as this interpreter's Python (vendored/py2-era file);
        # it cannot be importing our installer through the import system.
        return False
    fn_aliases: set[str] = set()  # names bound to installer functions
    mod_aliases: set[str] = set()  # names bound to the installer module
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.endswith("install_media_decoders"):
                for a in node.names:
                    if a.name in ("install_media_decoders", "main"):
                        fn_aliases.add(a.asname or a.name)
            elif node.module.endswith("common.utils") or node.module == "utils":
                for a in node.names:
                    if a.name == "install_media_decoders":
                        mod_aliases.add(a.asname or a.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.endswith("install_media_decoders"):
                    mod_aliases.add(a.asname or a.name.split(".")[0])
    if not fn_aliases and not mod_aliases:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Name) and f.id in fn_aliases:
            return True
        if isinstance(f, ast.Attribute) and f.attr in (
            "install_media_decoders",
            "main",
        ):
            root = f.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in mod_aliases:
                return True
    return False


def test_no_production_code_calls_the_installer():
    """Importing constants is fine; invoking the install is not.

    Calls into the installer (through any import alias) and the retired
    DYN_ENABLE_MEDIA_DECODERS switch must appear nowhere outside the
    installer module and its test.
    """
    allowed = {
        Path("common/utils/install_media_decoders.py"),  # the installer itself
        Path("common/tests/test_install_media_decoders.py"),  # this test
    }
    offenders: list[str] = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        rel = path.relative_to(_PACKAGE_ROOT)
        if rel in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if "DYN_ENABLE_MEDIA_DECODERS" in text or _source_calls_installer(text):
            offenders.append(str(rel))
    assert not offenders, (
        f"{offenders} invoke the media-decoder installer (or resurrect its env "
        "switch); only an operator may run it"
    )


def test_call_detector_sees_through_aliases():
    """The detector must catch aliased calls and ignore constant imports."""
    calls = _source_calls_installer
    direct = "from dynamo.common.utils.install_media_decoders import install_media_decoders\ninstall_media_decoders('vllm')\n"
    aliased = "from dynamo.common.utils.install_media_decoders import install_media_decoders as x\nx('vllm')\n"
    mod_alias = (
        "import dynamo.common.utils.install_media_decoders as m\nm.main(['vllm'])\n"
    )
    from_pkg = "from dynamo.common.utils import install_media_decoders as imd\nimd.install_media_decoders('vllm')\n"
    constants_only = "from dynamo.common.utils.install_media_decoders import VALIDATED_SPECS\nprint(VALIDATED_SPECS)\n"
    assert calls(direct)
    assert calls(aliased)
    assert calls(mod_alias)
    assert calls(from_pkg)
    assert not calls(constants_only)


def test_installer_module_has_no_env_switches():
    """The module reads no environment variables at all."""
    source = (_PACKAGE_ROOT / "common/utils/install_media_decoders.py").read_text(
        encoding="utf-8"
    )
    assert "os.environ" not in source and "getenv" not in source, (
        "install_media_decoders.py reads the environment; configuration belongs in "
        "CLI flags so the install stays explicit and self-describing"
    )
