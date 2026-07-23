# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in runtime installation of backend media-decoder packages.

Dynamo's runtime images ship a deliberately narrow media stack: the in-tree
FFmpeg is built for VP8/VP9 video (VP9 output), and the wider backend decode
packages are not pre-installed. That keeps the distributed images small and
their input-format surface narrow by default.

Some deployments need to accept a broader range of input video/audio formats
(for example H.264, H.265, or AAC). Each Dynamo backend decodes such input
through a specific Python package whose wheel bundles its own FFmpeg, so the
support can be added by a plain ``pip install`` -- no image rebuild:

* vLLM video input   -> OpenCV (``cv2``),   package ``opencv-python-headless``
* vLLM audio input   -> PyAV (``av``),       package ``av``
* SGLang video input -> decord (``decord``), package ``decord2``

When the operator opts in with ``DYN_ENABLE_MEDIA_DECODERS`` (off by default),
this module installs exactly the on-path package(s) for the running backend at
worker startup. Packages that some base images bundle but Dynamo never imports
(e.g. PyNvVideoCodec, torchcodec) are intentionally excluded -- installing them
would add a carrier with no code path.

The install is idempotent (skipped when the module already imports), serialized
across worker processes with a file lock, and never aborts worker startup on
failure. The encode path (video output) is unaffected and stays on the in-tree
FFmpeg. The optional Rust frontend decoder links FFmpeg's compiled-in decoders
and is therefore not extended by a runtime install -- backend decode is.

The default per-backend install runs with ``--no-deps`` so it cannot change the
image's pinned dependency stack (e.g. numpy under torch/vLLM); the carriers need
only numpy, which the backend already provides. pip is bounded by a timeout so a
stalled index cannot hang worker startup.

Advanced overrides:

* ``DYN_MEDIA_DECODER_PACKAGES`` -- whitespace-separated pip spec that replaces
  the per-backend defaults (for version pinning or a curated set). Installed
  with dependency resolution, so pin transitive versions if that matters.
* ``DYN_MEDIA_DECODER_PIP_ARGS`` -- extra args appended to ``pip install``
  (e.g. ``--find-links /wheels`` or ``--index-url ...`` for air-gapped hosts).
* ``DYN_MEDIA_DECODER_TIMEOUT_S`` -- pip timeout in seconds (default 600; 0
  disables it).
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from dynamo.common.utils.env import env_bool

try:
    import fcntl
except ImportError:  # non-POSIX platforms; locking becomes best-effort
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

ENABLE_ENV = "DYN_ENABLE_MEDIA_DECODERS"
PACKAGES_ENV = "DYN_MEDIA_DECODER_PACKAGES"
PIP_ARGS_ENV = "DYN_MEDIA_DECODER_PIP_ARGS"
TIMEOUT_ENV = "DYN_MEDIA_DECODER_TIMEOUT_S"

_DEFAULT_TIMEOUT_S = 600
_LOCK_PATH = Path(tempfile.gettempdir()) / "dynamo_media_decoders.lock"
# Mask URL userinfo (user:token@) before pip args reach the logs.
_CRED_RE = re.compile(r"(\w+://)[^/@\s]+@")


@dataclass(frozen=True)
class _Decoder:
    """A backend media-decoder pip package and the module it provides."""

    package: str  # pip install name
    module: str  # top-level import name, used for the already-present check
    kind: str  # "video" | "audio" (informational only)


# Only packages that sit on a real Dynamo decode execution path. Each wheel
# bundles its own FFmpeg, so a pip install restores H.264/H.265/AAC input decode
# without rebuilding the in-tree FFmpeg. Dead-end carriers some images bundle but
# Dynamo never imports (PyNvVideoCodec, torchcodec, PyAV-on-SGLang, opencv-on-
# TRT-LLM) are excluded on purpose.
_BACKEND_DECODERS: dict[str, tuple[_Decoder, ...]] = {
    "vllm": (
        _Decoder("opencv-python-headless", "cv2", "video"),
        _Decoder("av", "av", "audio"),
    ),
    "sglang": (_Decoder("decord2", "decord", "video"),),
}

# Backends that decode no compressed media input -- nothing to install.
_NO_DECODE_BACKENDS = frozenset({"trtllm"})

# Process-local guard so repeated calls in one interpreter are cheap no-ops.
_completed: set[str] = set()


def _module_available(module: str) -> bool:
    """Return True if `module` can be imported in this interpreter."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _redact(text: str) -> str:
    """Mask URL userinfo (user:token@host) so pip args never leak secrets."""
    return _CRED_RE.sub(r"\1***@", text)


def _install_timeout() -> int | None:
    """Bounded pip timeout in seconds. A 0/negative override disables it."""
    raw = os.environ.get(TIMEOUT_ENV, "").strip()
    if not raw:
        return _DEFAULT_TIMEOUT_S
    try:
        seconds = int(raw)
    except ValueError:
        logger.warning(
            "invalid %s=%r; using default %ss", TIMEOUT_ENV, raw, _DEFAULT_TIMEOUT_S
        )
        return _DEFAULT_TIMEOUT_S
    return seconds if seconds > 0 else None


@contextlib.contextmanager
def _cross_process_lock() -> Iterator[None]:
    """Serialize installs across worker processes sharing one site-packages."""
    if fcntl is None:
        yield
        return
    lock_file = None
    acquired = False
    try:
        lock_file = open(_LOCK_PATH, "w")  # noqa: SIM115 - closed in finally
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
    packages: list[str], extra_args: list[str], *, with_deps: bool
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--break-system-packages",  # runtime images use a PEP 668 system python
        "--no-input",
    ]
    if not with_deps:
        # Default carriers bundle their own FFmpeg and need only numpy, which the
        # backend already pins -- installing without deps avoids bumping the
        # image's pinned stack (e.g. numpy) out from under torch/vLLM.
        cmd.append("--no-deps")
    cmd += [*extra_args, *packages]
    logger.info("Running: %s", _redact(" ".join(cmd)))
    # Bounded so a stalled index cannot hang worker startup (or, via the lock,
    # every sibling worker on the host). TimeoutExpired is caught by the caller.
    subprocess.run(cmd, check=True, timeout=_install_timeout())


def maybe_install_media_decoders(backend: str) -> None:
    """Install this backend's media-decoder package(s) when opted in.

    No-op unless ``DYN_ENABLE_MEDIA_DECODERS`` is truthy. Safe to call more than
    once. Never raises: a failed install is logged and worker startup continues,
    so a media request that needs the missing format surfaces an actionable
    error at request time rather than crashing the worker at boot.
    """
    if not env_bool(ENABLE_ENV):
        return
    if backend in _completed:
        return

    override = os.environ.get(PACKAGES_ENV, "").strip()
    try:
        extra_args = shlex.split(os.environ.get(PIP_ARGS_ENV, ""))
    except ValueError as exc:
        logger.error(
            "Ignoring media decoder install: %s is malformed (%s). Fix its "
            "shell-style quoting and restart the worker.",
            PIP_ARGS_ENV,
            exc,
        )
        return

    if override:
        # Operator-specified specs (pinning / curated set). Trust them verbatim,
        # resolve dependencies, and let pip skip already-satisfied specs.
        packages = override.split()
        verify_modules: list[str] = []
        with_deps = True
    else:
        decoders = _BACKEND_DECODERS.get(backend)
        if not decoders:
            if backend in _NO_DECODE_BACKENDS:
                logger.debug(
                    "%s decodes no compressed media input; nothing to install",
                    backend,
                )
            else:
                logger.debug("no media decoders configured for backend %r", backend)
            _completed.add(backend)
            return
        # Install only what is not already importable (already-baked image, or a
        # second worker in the same interpreter).
        missing = [d for d in decoders if not _module_available(d.module)]
        if not missing:
            logger.debug(
                "media decoder backend(s) already present for %s; skipping", backend
            )
            _completed.add(backend)
            return
        packages = [d.package for d in missing]
        verify_modules = [d.module for d in missing]
        with_deps = False  # default carriers install --no-deps (see _pip_install)

    logger.info(
        "%s is set; installing media decoder backend(s) for broader video/audio "
        "input support (%s): %s",
        ENABLE_ENV,
        backend,
        _redact(" ".join(packages)),
    )
    try:
        with _cross_process_lock():
            if verify_modules:
                # Another worker may have installed while we waited on the lock.
                pending = [
                    pkg
                    for pkg, mod in zip(packages, verify_modules)
                    if not _module_available(mod)
                ]
                if not pending:
                    logger.info(
                        "media decoder backend(s) already installed by another "
                        "worker; skipping"
                    )
                    _completed.add(backend)
                    return
                packages = pending
            _pip_install(packages, extra_args, with_deps=with_deps)
    except Exception as exc:  # noqa: BLE001 - never break worker startup
        logger.error(
            "Failed to install media decoder backend(s) %s: %s. Requests needing "
            "these input formats will fail with an actionable error at request "
            "time. For offline/air-gapped clusters, pre-install the package(s) "
            "into the image or point %s at a local wheelhouse "
            "(e.g. '--find-links /wheels').",
            _redact(" ".join(packages)),
            exc,
            PIP_ARGS_ENV,
        )
        return

    importlib.invalidate_caches()
    _completed.add(backend)
    still_missing = [m for m in verify_modules if not _module_available(m)]
    if still_missing:
        logger.warning(
            "media decoder module(s) still not importable after install: %s",
            " ".join(still_missing),
        )
    else:
        logger.info("Media decoder backend(s) ready: %s", " ".join(packages))


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``python -m dynamo.common.utils.media_decoders <backend>``.

    Lets a launch script run the install before the worker starts, as an
    alternative to the in-process startup hook. No-op unless the env gate is set.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Install a Dynamo backend's media-decoder package(s) when "
            f"{ENABLE_ENV} is set (off by default). No-op otherwise."
        )
    )
    parser.add_argument(
        "backend",
        choices=sorted(set(_BACKEND_DECODERS) | _NO_DECODE_BACKENDS),
        help="backend whose media-decoder package(s) to install",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    maybe_install_media_decoders(args.backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
