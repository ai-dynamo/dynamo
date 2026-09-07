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

from dynamo.common.multimodal.nvdec_decoder import HW_ROUTED_CODECS, nvdec_available
from dynamo.common.utils.install_media_decoders import VALIDATED_SPECS

INSTALLER_CMD = "python -m dynamo.common.utils.install_media_decoders"


class MissingMediaDecoderError(RuntimeError):
    """A media request needs a decoder package the image does not ship.

    Deliberately not a ``ValueError``: the input may be perfectly valid media.
    The gap is deployment configuration, so handlers that map ``ValueError``
    to a client 4xx should not blame the request for it.
    """


def _install_hint(backend: str, package: str) -> str:
    spec = VALIDATED_SPECS[package]
    return (
        f"install the validated decoder with `pip install --no-deps '{spec}'` "
        f"(or `{INSTALLER_CMD} {backend}`)"
    )


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
    else:
        lead = (
            f"this video ({codec_desc}) has no decoder in this image: shipped "
            "images decode only H.264/H.265 (in hardware, via NVDEC), and the "
            f"software decoder '{module}' is deliberately not installed. "
            "Re-encode the input to H.264/H.265, or "
        )
    return MissingMediaDecoderError(
        _with_cause(
            "Cannot decode video: " + lead + _install_hint(backend, package) + ".",
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
            "audio. To enable audio input, " + _install_hint(backend, "av") + ".",
            cause,
        )
    )
