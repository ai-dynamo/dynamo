#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path


def command(*args):
    result = subprocess.run(
        args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
    )
    return {
        "command": list(args),
        "exit_code": result.returncode,
        "output": result.stdout.strip(),
    }


def read_optional(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-binding", required=True)
    parser.add_argument("--numa-policy", required=True)
    parser.add_argument("--allocator", default="system")
    args = parser.parse_args()

    manifest = {
        "hostname": platform.node(),
        "kernel": platform.release(),
        "platform": platform.platform(),
        "cpu_binding": args.cpu_binding,
        "numa_policy": args.numa_policy,
        "allocator": args.allocator,
        "tokio_worker_threads": os.environ.get("TOKIO_WORKER_THREADS"),
        "event_threads": os.environ.get("EVENT_THREADS", "8"),
        "lscpu": command("lscpu", "-J"),
        "numactl": command("numactl", "--hardware"),
        "memory": command("free", "-b"),
        "rustc": command("rustc", "--version", "--verbose"),
        "governor": command(
            "bash",
            "-lc",
            "grep -h . /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort -u",
        ),
        "turbo_intel_no_turbo": read_optional(
            "/sys/devices/system/cpu/intel_pstate/no_turbo"
        ),
        "turbo_cpufreq_boost": read_optional("/sys/devices/system/cpu/cpufreq/boost"),
        "cpuset": command(
            "bash", "-lc", "grep '^Cpus_allowed_list:' /proc/self/status"
        ),
        "numa_status": command(
            "bash", "-lc", "grep '^Mems_allowed_list:' /proc/self/status"
        ),
    }
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
