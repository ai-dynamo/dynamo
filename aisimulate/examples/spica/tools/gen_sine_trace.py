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

"""Generate a mooncake-format JSONL trace whose arrival rate follows a sine wave.

This utility supports the experimental Spica feature and may change with its workload schema.

The instantaneous request rate is ``base_rate + amplitude * sin(2*pi*t/period)``
(clamped at 0); ``round(rate)`` requests are placed evenly within each second.
isl/osl are constant, so the windowed ``num_req`` series is a clean sine — a
periodic load to test the predictors against a flat constant baseline.
"""

from __future__ import annotations

import argparse
import json
import math


def generate(
    out: str,
    duration_s: int,
    period_s: float,
    base_rate: float,
    amplitude: float,
    isl: int,
    osl: int,
) -> int:
    total = 0
    with open(out, "w", encoding="utf-8") as f:
        for sec in range(int(duration_s)):
            rate = base_rate + amplitude * math.sin(2 * math.pi * sec / period_s)
            count = max(0, round(rate))
            for k in range(count):
                ts_ms = int(sec * 1000 + k * 1000 // count)
                f.write(
                    json.dumps(
                        {"timestamp": ts_ms, "input_length": isl, "output_length": osl}
                    )
                    + "\n"
                )
                total += 1
    return total


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a sine-wave-load mooncake trace")
    p.add_argument("--out", required=True)
    p.add_argument("--duration-s", type=int, default=5400)
    p.add_argument("--period-s", type=float, default=900.0)
    p.add_argument("--base-rate", type=float, default=6.0, help="mean requests/sec")
    p.add_argument(
        "--amplitude", type=float, default=4.0, help="rate swing (requests/sec)"
    )
    p.add_argument("--isl", type=int, default=4000)
    p.add_argument("--osl", type=int, default=200)
    a = p.parse_args()
    n = generate(
        a.out, a.duration_s, a.period_s, a.base_rate, a.amplitude, a.isl, a.osl
    )
    print(
        f"wrote {n} requests to {a.out} (duration={a.duration_s}s period={a.period_s}s)"
    )


if __name__ == "__main__":
    main()
