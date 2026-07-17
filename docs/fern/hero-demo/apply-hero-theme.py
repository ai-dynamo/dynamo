#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Embed a terminal theme into an asciicast v3 header (in place).
# Palette: GitHub Dark Default, but keeping GitHub "classic" yellow (#e3b341)
# at both the normal and bright slots.
#
# Usage: apply-hero-theme.py <file.cast>
import json
import sys

THEME = {
    # Plain terminal output uses the default fg → keep it a soft gray so it
    # reads as "normal output". Typed commands use WHITE (palette index 7),
    # which we pin to pure white below so commands pop brighter than output.
    "fg": "#8b949e",
    "bg": "#0d1117",
    "palette": ":".join([
        # normal: black red green yellow blue magenta cyan white
        "#484f58", "#ff7b72", "#3fb950", "#e3b341",
        "#58a6ff", "#bc8cff", "#39c5cf", "#ffffff",
        # bright
        "#6e7681", "#ffa198", "#56d364", "#e3b341",
        "#79c0ff", "#d2a8ff", "#56d4dd", "#ffffff",
    ]),
}


def main(path: str) -> None:
    with open(path) as f:
        lines = f.readlines()
    header = json.loads(lines[0])
    header.setdefault("term", {})["theme"] = THEME
    lines[0] = json.dumps(header) + "\n"
    with open(path, "w") as f:
        f.writelines(lines)
    dur = sum(json.loads(l)[0] for l in lines[1:] if l.strip())
    print(f"themed {path}; duration ~{dur:.1f}s")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: apply-hero-theme.py <file.cast>")
    main(sys.argv[1])
