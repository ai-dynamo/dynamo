# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enumerate what a vendored libavcodec actually registers.

`scan_codecs.py` gates which media libraries may appear in an image. This answers
the different question of what a library that is *allowed* to be there actually
implements — needed to review a third-party waiver, since a bundled libavcodec
may or may not carry software decoders for the formats we exclude.

Neither obvious shortcut works. These libraries are stripped, so `nm` reports
nothing even for codecs that are certainly present; and `strings` gives false
positives, because a decoder's long name survives in the binary when only its
parser or bitstream filter is retained. The authoritative answer is what the
library registers at runtime, so walk `av_codec_iterate` and ask
`av_codec_is_decoder` per entry.

Usage:
    python3 enumerate_bundled_decoders.py <path-to-libavcodec.so>

Exit code is 1 when any excluded decoder is registered, so it can gate a check.
"""

from __future__ import annotations

import ctypes
import sys


class AVCodec(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("long_name", ctypes.c_char_p),
        ("type", ctypes.c_int),
        ("id", ctypes.c_int),
    ]


def main(path: str) -> int:
    lib = ctypes.CDLL(path)
    lib.av_codec_iterate.restype = ctypes.c_void_p
    lib.av_codec_iterate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.av_codec_is_decoder.restype = ctypes.c_int
    lib.av_codec_is_decoder.argtypes = [ctypes.c_void_p]

    opaque = ctypes.c_void_p(None)
    decoders: set[str] = set()
    encoders: set[str] = set()
    while True:
        c = lib.av_codec_iterate(ctypes.byref(opaque))
        if not c:
            break
        entry = ctypes.cast(c, ctypes.POINTER(AVCodec)).contents
        name = entry.name.decode()
        if lib.av_codec_is_decoder(c):
            decoders.add(name)
        else:
            encoders.add(name)

    print(f"library : {path}")
    print(f"registers: {len(decoders)} decoders, {len(encoders)} encoders\n")

    trimmed = ["h264", "hevc", "aac", "aac_fixed", "aac_latm"]
    kept = ["vp8", "vp9", "mjpeg", "av1"]

    print("  decoders we exclude from the software stack:")
    for name in trimmed:
        mark = "PRESENT" if name in decoders else "absent"
        print(f"    {name:<12} decoder: {mark}")
    print("\n  expected present (sanity check -- a broken probe fails here):")
    for name in kept:
        mark = "present" if name in decoders else "ABSENT"
        print(f"    {name:<12} decoder: {mark}")

    still_there = [n for n in trimmed if n in decoders]
    print()
    if still_there:
        print(f"VERDICT: registers excluded decoders: {', '.join(still_there)}")
        return 1
    print("VERDICT: none of the excluded decoders are registered")
    return 0


sys.exit(main(sys.argv[1]))
