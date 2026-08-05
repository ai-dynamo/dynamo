# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit installer for backend media-decoder packages.

Dynamo's runtime images ship a deliberately narrow media stack: the in-tree
FFmpeg is built for VP8/VP9 video (VP9 output), and the wider backend decode
packages are not pre-installed. That keeps the distributed images small and
their input-format surface narrow by default.

Two input classes already work without this module. VP8/VP9 decode through the
in-tree FFmpeg, and H.264/H.265 decode on the GPU through NVDEC
(``common.multimodal.nvdec_decoder``), which every backend routes to by default.

This module covers what is left:

* compressed audio (AAC and friends), which NVDEC does not decode at all, and
* H.264/H.265 on hosts where NVDEC is unavailable -- a GPU with no video decode
  engine, or a container not granted the ``video`` driver capability.

Each backend decodes such input through a specific Python package whose wheel
bundles its own FFmpeg, so the support can be added by a plain ``pip install``
-- no image rebuild:

* vLLM video input    -> OpenCV (``cv2``),    package ``opencv-python-headless``
* vLLM audio input    -> PyAV (``av``),       package ``av``
* SGLang video input  -> decord (``decord``), package ``decord2``
* TRT-LLM video input -> OpenCV (``cv2``),    package ``opencv-python-headless``

Nothing here runs implicitly. There is no environment switch and no startup
hook: the operator runs the installer as a deliberate step --

    python -m dynamo.common.utils.install_media_decoders vllm

-- so a deployment that changes the image's codec surface says so where it can
be seen (a Dockerfile RUN line, a pod command, a runbook step), not in an
environment variable three layers of tooling away. Installing these packages
broadens the codec surface of the running container; review that against your
organization's distribution and security policies before rolling it out.

Each package installs at a version validated against Dynamo's multimodal test
suite (see the specs below): the lower bound is the exact validated version and
the upper bound excludes the next major, so a fresh install cannot silently
pick up an unvalidated release. This installer deliberately covers ONLY those
tested combinations; to pin different versions or install a custom subset, run
pip directly with the tool of your choice (see the docs for the equivalent
commands).

The default install runs with ``--no-deps`` so it cannot change the image's
pinned dependency stack (e.g. numpy under torch/vLLM); the carriers need only
numpy, which the backend already provides. The install is idempotent (skipped
when the module already imports), serialized across processes with a file
lock, and bounded by a timeout so a stalled index cannot hang forever. The
encode path (video output) is unaffected and stays on the in-tree FFmpeg. The
optional Rust frontend decoder links FFmpeg's compiled-in decoders and is
therefore not extended by an install here -- backend decode is.

For air-gapped or allowlisted-network hosts, point pip at a local wheelhouse:

    python -m dynamo.common.utils.install_media_decoders vllm \\
        --pip-args="--no-index --find-links /wheels"
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

try:
    import fcntl
except ImportError:  # non-POSIX platforms; locking becomes best-effort
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 600
_LOCK_PATH = Path(tempfile.gettempdir()) / "dynamo_media_decoders.lock"
# Mask URL userinfo (user:token@) before pip args reach the logs.
_CRED_RE = re.compile(r"(\w+://)[^/@\s]+@")


@dataclass(frozen=True)
class _Decoder:
    """A backend media-decoder pip package and the module it provides."""

    package: str  # pip distribution name
    module: str  # top-level import name, used for the already-present check
    spec: str  # pip requirement with validated lower bound + major-version cap
    kind: str  # "video" | "audio" (informational only)


# Version bounds: the lower bound is the version validated against Dynamo's
# multimodal serve tests (OpenCV and decord on GPU test runs; PyAV as shipped
# in the last image generation that carried it through the full suites). The
# upper bound excludes the next major so a fresh install cannot drift onto an
# unvalidated release -- opencv-python-headless 5.x already exists on PyPI and
# has not been validated. Raise a bound only after the multimodal suite passes
# against the new version.
_OPENCV = _Decoder(
    "opencv-python-headless", "cv2", "opencv-python-headless>=4.13.0.92,<5", "video"
)
_PYAV = _Decoder("av", "av", "av>=18.0.0,<19", "audio")
_DECORD = _Decoder("decord2", "decord", "decord2>=3.4.0,<4", "video")

# Validated pip specs by distribution name, for reuse by tests and docs so the
# bounds live in exactly one place.
VALIDATED_SPECS: dict[str, str] = {d.package: d.spec for d in (_OPENCV, _PYAV, _DECORD)}

# Only packages that sit on a real Dynamo decode execution path. Each wheel
# bundles its own FFmpeg, so a pip install adds software decode for input the
# image cannot otherwise handle, without rebuilding the in-tree FFmpeg.
#
# Excluded on purpose, for two different reasons:
#   * PyNvVideoCodec -- already shipped in every runtime image as the NVDEC
#     path, so H.264/H.265 decode needs no install. Listing it here would
#     reinstall what is already present.
#   * torchcodec, PyAV-on-SGLang, opencv-on-SGLang -- no Dynamo decode path
#     imports them, so installing them would add a carrier nothing calls.
_BACKEND_DECODERS: dict[str, tuple[_Decoder, ...]] = {
    "vllm": (_OPENCV, _PYAV),
    "sglang": (_DECORD,),
    # TRT-LLM decodes video_url input via tensorrt_llm.inputs -> _load_video_by_cv2
    # (OpenCV). It has no audio-input decode path today.
    #
    # The TRT-LLM images deliberately ship without opencv-python-headless, and
    # tests/dependencies/test_no_opencv.py guards that. Installing it puts it
    # back into the running container, so this entry only earns its keep on a
    # host where NVDEC cannot serve H.264/H.265.
    "trtllm": (_OPENCV,),
}


def _modules_missing_fresh(modules: Sequence[str]) -> list[str]:
    """Return the subset of `modules` a FRESH interpreter cannot IMPORT.

    Fresh interpreter, not this process: running as a non-root user, pip
    defaults to a user-site install, and a user-site directory created after
    the parent interpreter started is never added to its ``sys.path``
    (``site.py`` only does that at startup, and
    ``importlib.invalidate_caches()`` cannot add path entries). Verified on
    all three runtime images: the same-process check reported the install
    missing while a fresh process imported it fine. What matters
    operationally is the worker process launched after this command --
    which is exactly a fresh interpreter.

    Real import, not ``find_spec``: a package whose files are present but
    whose native libraries cannot load (a broken or partially removed wheel)
    has a spec and would be treated as installed, silently skipping the
    install and deferring the failure to request time. Importing in the
    probe subprocess keeps this process's module state untouched.
    """
    if not modules:
        return []
    probe = (
        "import importlib, sys\n"
        "missing = []\n"
        "for m in sys.argv[1:]:\n"
        "    try:\n"
        "        importlib.import_module(m)\n"
        "    except Exception:\n"
        "        missing.append(m)\n"
        "print(' '.join(missing))\n"
    )
    try:
        out = subprocess.run(
            [sys.executable, "-c", probe, *modules],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
    except Exception as exc:  # noqa: BLE001 - cannot verify => report missing
        logger.warning("fresh-interpreter verification failed to run: %s", exc)
        return list(modules)
    return out.stdout.split()


def _redact(text: str) -> str:
    """Mask URL userinfo (user:token@host) so pip args never leak secrets."""
    return _CRED_RE.sub(r"\1***@", text)


@contextlib.contextmanager
def _cross_process_lock() -> Iterator[None]:
    """Serialize installs across worker processes sharing one site-packages."""
    if fcntl is None:
        yield
        return
    lock_file = None
    acquired = False
    try:
        # The lock path is fixed and predictable inside a shared temp dir, so
        # never open it with something that follows symlinks or truncates:
        # a pre-planted symlink would redirect a truncate-on-open to an
        # arbitrary file owned by whoever runs the install (often root in a
        # Dockerfile RUN). O_NOFOLLOW refuses symlinks, the flag set carries
        # no O_TRUNC, and 0600 keeps the file private to its creator.
        fd = os.open(_LOCK_PATH, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        lock_file = os.fdopen(fd, "r+b", buffering=0)
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        acquired = True
    except OSError as exc:
        logger.debug("media-decoder install lock unavailable (%s); proceeding", exc)
    try:
        yield
    finally:
        if lock_file is not None:
            if acquired:
                with contextlib.suppress(OSError):
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()


def _pip_install(
    packages: Sequence[str],
    extra_args: Sequence[str],
    *,
    timeout_s: int | None,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--break-system-packages",  # runtime images use a PEP 668 system python
        "--no-input",
        # The carriers bundle their own FFmpeg and need only numpy, which the
        # backend already pins -- installing without deps avoids bumping the
        # image's pinned stack (e.g. numpy) out from under torch/vLLM.
        "--no-deps",
    ]
    cmd += [*extra_args, *packages]
    logger.info("Running: %s", _redact(" ".join(cmd)))
    subprocess.run(cmd, check=True, timeout=timeout_s)


def install_media_decoders(
    backend: str,
    *,
    pip_args: Sequence[str] = (),
    timeout_s: int | None = DEFAULT_TIMEOUT_S,
    dry_run: bool = False,
) -> list[str]:
    """Install `backend`'s media-decoder package(s); return the specs installed.

    Installs the backend's validated, version-bounded specs with ``--no-deps``,
    skipping any package whose module already imports. Deliberately covers only
    the tested combinations: for custom versions or subsets, run pip directly.

    Returns the list of pip specs actually installed (empty when everything was
    already present). ``dry_run`` returns what would install without running
    pip. Raises on failure -- this is a deliberate operator action, so a broken
    install must be loud, not a log line a worker scrolls past.
    """
    decoders = _BACKEND_DECODERS.get(backend)
    if decoders is None:
        raise ValueError(
            f"unknown backend {backend!r}; expected one of {sorted(_BACKEND_DECODERS)}"
        )
    # Install only what does not already import cleanly. The probe does a
    # real import in a fresh interpreter, so a present-but-broken package
    # counts as missing rather than being silently skipped.
    missing_names = set(_modules_missing_fresh([d.module for d in decoders]))
    missing = [d for d in decoders if d.module in missing_names]
    if not missing:
        logger.info(
            "media decoder package(s) already present for %s; nothing to do",
            backend,
        )
        return []
    specs = [d.spec for d in missing]
    verify_modules = [d.module for d in missing]

    if dry_run:
        logger.info(
            "dry run: would install for %s: %s", backend, _redact(" ".join(specs))
        )
        return specs

    logger.info(
        "installing media decoder package(s) for %s: %s",
        backend,
        _redact(" ".join(specs)),
    )
    with _cross_process_lock():
        # Another process may have installed while we waited on the lock.
        still = set(_modules_missing_fresh(verify_modules))
        pending = [
            (spec, mod) for spec, mod in zip(specs, verify_modules) if mod in still
        ]
        if not pending:
            logger.info(
                "media decoder package(s) already installed by another "
                "process; nothing to do"
            )
            return []
        specs = [spec for spec, _ in pending]
        verify_modules = [mod for _, mod in pending]
        _pip_install(specs, pip_args, timeout_s=timeout_s)

    importlib.invalidate_caches()
    still_missing = _modules_missing_fresh(verify_modules)
    if still_missing:
        raise RuntimeError(
            "media decoder module(s) still not importable after install: "
            + " ".join(still_missing)
            + ". If a package is present but broken, pip may have treated the "
            "requirement as already satisfied -- pip uninstall it and rerun, "
            "or pip install --force-reinstall the spec directly."
        )
    logger.info("media decoder package(s) ready: %s", _redact(" ".join(specs)))
    return specs


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``python -m dynamo.common.utils.install_media_decoders <backend>``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m dynamo.common.utils.install_media_decoders",
        description=(
            "Install a Dynamo backend's media-decoder package(s) at validated, "
            "version-bounded releases. Explicit by design: nothing installs "
            "unless an operator runs this."
        ),
    )
    parser.add_argument(
        "backend",
        choices=sorted(_BACKEND_DECODERS),
        help="backend whose media-decoder package(s) to install",
    )
    parser.add_argument(
        "--pip-args",
        default="",
        metavar="ARGS",
        help=(
            "extra arguments appended to `pip install`, shell-quoted as one "
            'string. Use the = form -- --pip-args="--no-index --find-links '
            '/wheels" -- so a value starting with a dash is not mistaken '
            "for an option (for air-gapped hosts)"
            "air-gapped hosts)"
        ),
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=DEFAULT_TIMEOUT_S,
        help=(
            f"pip timeout in seconds (default {DEFAULT_TIMEOUT_S}; "
            "0 or negative disables the timeout)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be installed and exit without running pip",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        extra_args = shlex.split(args.pip_args)
    except ValueError as exc:
        parser.error(f"--pip-args is not valid shell quoting: {exc}")

    timeout_s = args.timeout_s if args.timeout_s > 0 else None
    try:
        install_media_decoders(
            args.backend,
            pip_args=extra_args,
            timeout_s=timeout_s,
            dry_run=args.dry_run,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report and fail
        # Redact: CalledProcessError/TimeoutExpired stringify the full pip
        # command, which may carry a credentialed --pip-args index URL.
        logger.error(
            "media decoder install failed: %s. For offline/air-gapped hosts, "
            "point pip at a local wheelhouse, e.g. "
            "--pip-args='--no-index --find-links /wheels'.",
            _redact(str(exc)),
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
