# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actionable errors for media the shipped images cannot decode.

The runtime images deliberately omit the software media decoders (OpenCV,
PyAV, decord) and route H.264/H.265 video to NVDEC hardware decode instead.
Any other input codec needs one of those Python packages, so without an
explicit install the failure surfaces as a bare ``ModuleNotFoundError`` from
deep inside the backend -- no codec, no remedy, and in one observed case the
whole video payload embedded in the message. The builders here name the codec,
the missing package at its validated version bounds, the installer command,
and the hardware alternative, in one place so the three backends cannot drift.

The version bounds come from
:mod:`dynamo.common.utils.install_media_decoders` (the explicit installer);
importing its constants here is deliberate single-sourcing -- nothing in this
module installs anything.
"""

from __future__ import annotations

import importlib
import importlib.metadata

from dynamo.common.multimodal.nvdec_decoder import HW_ROUTED_CODECS, nvdec_available
from dynamo.common.utils.install_media_decoders import VALIDATED_SPECS, installer_covers

INSTALLER_CMD = "python -m dynamo.common.utils.install_media_decoders"


class MissingMediaDecoderError(RuntimeError):
    """A media request needs a decoder package the image does not ship.

    Deliberately not a ``ValueError``: the input may be perfectly valid media.
    The gap is deployment configuration, so handlers that map ``ValueError``
    to a client 4xx should not blame the request for it.
    """


def _carrier_present(module: str) -> bool:
    """Return whether the carrier imports; keep import errors out of diagnostics."""
    try:
        importlib.import_module(module)
    except Exception:  # noqa: BLE001 - any import failure means unusable here
        return False
    return True


def _needs_binary_wheel(backend: str, package: str) -> bool:
    """Whether this package must come from a wheel: vLLM's OpenCV sdist has no codecs."""
    return backend == "vllm" and package == "opencv-python-headless"


def _is_vllm_source_built_cv2(backend: str, package: str, module: str) -> bool:
    """Whether this is the codec-free OpenCV build shipped by vLLM."""
    return (
        _needs_binary_wheel(backend, package)
        and module == "cv2"
        and _carrier_present(module)
    )


def _installed_version(package: str) -> str | None:
    """The installed distribution version, or None when it has no metadata."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _install_hint(backend: str, package: str, module: str) -> str:
    """Use the installed OpenCV version for vLLM replacements, validated bounds otherwise."""
    if _is_vllm_source_built_cv2(backend, package, module):
        installed = _installed_version(package)
        if installed:
            hint = (
                "replace the shipped build with the binary wheel of the same "
                f"version: `pip install --no-deps --force-reinstall "
                f"--only-binary {package} '{package}=={installed}'`"
            )
        else:
            # cv2 omits the distribution packaging revision.
            hint = (
                "first determine the shipped cv2 version with "
                '`python -c "import cv2; print(cv2.__version__)"`, then replace '
                "it with the binary wheel built from that same version -- the "
                "distribution adds a packaging revision, so match on the prefix: "
                f"`pip install --no-deps --force-reinstall --only-binary {package} "
                f"'{package}==<cv2-version>.*'`"
            )
    else:
        only_binary = (
            f"--only-binary {package} " if _needs_binary_wheel(backend, package) else ""
        )
        hint = (
            "install the validated decoder with "
            f"`pip install --no-deps --force-reinstall "
            f"{only_binary}'{VALIDATED_SPECS[package]}'`"
        )
    if installer_covers(backend, package):
        hint += f" (or `{INSTALLER_CMD} {backend}`)"
    return hint


def _with_cause(message: str, cause: str | None) -> str:
    """Append the underlying decoder text so diagnostics survive the wrap.

    ``raise ... from exc`` preserves the cause for tracebacks, but handlers
    that ship only ``str(exc)`` to the client (HTTP error bodies) would drop
    it -- and the underlying reason is part of this error's contract.
    """
    if cause:
        return f"{message} (decoder reported: {cause})"
    return message


def video_decoder_missing(
    backend: str,
    package: str,
    module: str,
    codec: str | None,
    cause: str | None = None,
) -> MissingMediaDecoderError:
    """Build the error for a video whose decode path has no decoder.

    Two distinct situations produce it, and the remedy differs:

    * ``codec`` is H.264/H.265 but NVDEC is unavailable in this container --
      the primary fix is granting the ``video`` driver capability, not
      installing software.
    * any other codec -- NVDEC never decodes it, so the fix is the software
      decoder install or re-encoding the input to H.264/H.265.
    """
    codec_desc = f"codec '{codec}'" if codec else "an undetected codec"
    if codec in HW_ROUTED_CODECS and not nvdec_available():
        lead = (
            f"this video ({codec_desc}) normally decodes in hardware via NVDEC, "
            "but NVDEC is unavailable in this container. Grant the 'video' "
            "driver capability (NVIDIA_DRIVER_CAPABILITIES) to enable it, or "
        )
    elif _is_vllm_source_built_cv2(backend, package, module):
        lead = (
            f"this video ({codec_desc}) has no decoder in this image: shipped "
            "images decode only H.264/H.265 (in hardware, via NVDEC), and the "
            f"'{module}' they ship is built without a video backend. "
            "Re-encode the input to H.264/H.265, or "
        )
    else:
        lead = (
            f"this video ({codec_desc}) has no decoder in this image: shipped "
            "images decode only H.264/H.265 (in hardware, via NVDEC), and the "
            f"software decoder '{module}' is deliberately not installed. "
            "Re-encode the input to H.264/H.265, or "
        )
    return MissingMediaDecoderError(
        _with_cause(
            "Cannot decode video: "
            + lead
            + _install_hint(backend, package, module)
            + ".",
            cause,
        )
    )


def audio_decoder_missing(
    backend: str, cause: str | None = None
) -> MissingMediaDecoderError:
    """Build the error for audio input with no decoder in the image.

    NVDEC never decodes audio, so unlike video there is no hardware
    alternative -- the only remedy is the PyAV install.
    """
    return MissingMediaDecoderError(
        _with_cause(
            "Cannot decode audio: this input needs the PyAV decoder ('av'), which "
            "this image deliberately does not ship, and NVDEC does not decode "
            "audio. To enable audio input, " + _install_hint(backend, "av", "av") + ".",
            cause,
        )
    )
