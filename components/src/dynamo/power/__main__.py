# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``python -m dynamo.power`` only points at the two real entry points."""

import sys

USAGE = (
    "dynamo.power has two commands:\n"
    "  python -m dynamo.power.node_agent --help   (EPD sensing + controller)\n"
    "  python -m dynamo.power.applier --help      (apply decisions to GPUs)\n"
)

if __name__ == "__main__":
    sys.stderr.write(USAGE)
    sys.exit(2)
