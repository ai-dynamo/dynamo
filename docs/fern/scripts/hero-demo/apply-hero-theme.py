#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Embed a terminal theme into an asciicast v3 header (in place).
# Palette: Dynamo Glass — black and deep green with NVIDIA-green accents
# and neutral gray/off-white output inspired by the concept terminals below.
#
# Usage: apply-hero-theme.py <file.cast>
import json
import sys

THEME = {
    # Green marks actions and success. Neutral grays carry normal output,
    # metadata, and secondary narration so the demo stays crisp rather than
    # tinting every semantic role green.
    "fg": "#b8c0b9",
    "bg": "#071009",
    "palette": ":".join(
        [
            # normal: black red green yellow blue magenta cyan white
            "#273129",
            "#d6ddd7",
            "#8fd120",
            "#b8df7a",
            "#aab3ac",
            "#858f87",
            "#c3cbc4",
            "#e2e6e2",
            # bright
            "#667068",
            "#eef1ee",
            "#b8f36a",
            "#d0eba7",
            "#c4cbc5",
            "#aab2ab",
            "#d5dad6",
            "#f7f9f7",
        ]
    ),
}


def main(path: str) -> None:
    with open(path) as f:
        lines = f.readlines()
    header = json.loads(lines[0])
    header.setdefault("term", {})["theme"] = THEME
    lines[0] = json.dumps(header) + "\n"
    with open(path, "w") as f:
        f.writelines(lines)
    dur = sum(json.loads(line)[0] for line in lines[1:] if line.strip())
    print(f"themed {path}; duration ~{dur:.1f}s")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: apply-hero-theme.py <file.cast>")
    main(sys.argv[1])
