# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Print one client's aiperf 0.13 config: AgentX traces at CLIENT_CONCURRENCY, warmup, then
settle + measure, without cache-bust (no KV cache anywhere in this campaign) and without
per-request records.

usage: aiperf_config.py URL[,URL...] ARTIFACT_DIR   (settings come from env.sh's variables)
"""

import json
import os
import subprocess
import sys


def env_sh():
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "env.sh")
    names = "MODEL MODEL_REVISION DATASET CLIENT_CONCURRENCY SETTLE_S MEASURE_S"
    out = subprocess.run(
        ["bash", "-c", f"source {script}; for n in {names}; do echo ${{!n}}; done"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    return dict(zip(names.split(), out))


def main():
    urls, artifacts = sys.argv[1].split(","), sys.argv[2]
    env = env_sh()
    concurrency = int(env["CLIENT_CONCURRENCY"])
    config = {
        "schemaVersion": "2.0",
        "randomSeed": 12345,
        "benchmark": {
            "model": env["MODEL"],
            "endpoint": {
                "urls": urls,
                "urlStrategy": "round_robin",
                "type": "chat",
                "streaming": True,
                "timeout": 21600.0,
                "useServerTokenCount": True,
                "extra": {"ignore_eos": True},
            },
            "dataset": {
                "type": "public",
                "dataset": env["DATASET"],
                "entries": 336,
                "sampling": "sequential",
                "ignoreTraceDelays": True,
            },
            "tokenizer": {"name": env["MODEL"], "revision": env["MODEL_REVISION"]},
            "warmup": {
                "type": "concurrency",
                "requests": 32,
                "concurrency": concurrency,
            },
            "profiling": {
                "type": "concurrency",
                # 10 s past the measure window, so its end sees load still running.
                "duration": int(env["SETTLE_S"]) + int(env["MEASURE_S"]) + 10,
                "concurrency": concurrency,
                "gracePeriod": 30,
            },
            "artifacts": {"dir": artifacts, "summary": ["json"], "records": False},
            "gpuTelemetry": {"enabled": False},
            "serverMetrics": {"enabled": False},
            "runtime": {"workers": 128, "recordProcessors": 16, "ui": "none"},
        },
    }
    json.dump(config, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
