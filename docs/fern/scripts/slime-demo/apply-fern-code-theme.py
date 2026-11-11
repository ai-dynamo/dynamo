#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Embed the dark "fern-code" palette into an asciicast v3 header (in place).
#
# The Slime Integration overview plays the cast with TerminalDemo's fern-code
# theme, which switches between light and dark with the site. This copy of the
# dark colors serves players without that theme, such as agg and asciinema.org.
# Keep it in step with the dark block of the fern-code theme in
# docs/fern/components/TerminalDemo.tsx. Each slot is the syntax role that
# slime-demo.sh names in its palette section. The verl Integration overview's
# cast uses the same slots, so ../verl-demo/record-verl-demo.sh runs this
# script too.
#
# Usage: apply-fern-code-theme.py <file.cast>
import json
import sys

THEME = {
    "fg": "#eeffff",
    "bg": "#0c0d0b",
    "palette": ":".join(
        [
            "#333333",  # 0  step header rules
            "#f07178",  # 1  Python attribute names
            "#c3e88d",  # 2  quoted strings; success
            "#ffcb6b",  # 3  command names
            "#82aaff",  # 4  shell builtins and Python function names
            "#c792ea",  # 5  declaration keywords
            "#89ddff",  # 6  operators and control keywords
            "#8b949e",  # 7  narration and comments
            "#6e7681",  # 8  progress and fast-forward notes
            "#f78c6c",  # 9  numbers in narration; in-progress status
            "#c3e88d",  # 10 unquoted arguments and flags
            "#eeffff",  # 11 Python parameters
            "#82aaff",  # 12 Python call arguments
            "#89ddff",  # 13 quote marks
            "#89ddff",  # 14 punctuation
            "#8cce37",  # 15 the prompt
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
        sys.exit("usage: apply-fern-code-theme.py <file.cast>")
    main(sys.argv[1])
