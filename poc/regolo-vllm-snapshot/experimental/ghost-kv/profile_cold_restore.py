#!/usr/bin/env python3
"""Profile one cache-cold Ghost KV restore without changing checkpoint data."""

import argparse
import json
import os
import pathlib
import subprocess
import threading
import time


DEVICES = {"dm-0", "loop6", "sda"}


def read_text(path):
    return pathlib.Path(path).read_text(encoding="utf-8")


def diskstats():
    result = {}
    for line in read_text("/proc/diskstats").splitlines():
        fields = line.split()
        if len(fields) >= 11 and fields[2] in DEVICES:
            result[fields[2]] = {
                "reads_completed": int(fields[3]),
                "sectors_read": int(fields[5]),
                "read_ms": int(fields[6]),
                "writes_completed": int(fields[7]),
                "sectors_written": int(fields[9]),
                "write_ms": int(fields[10]),
            }
    return result


def meminfo():
    values = {}
    for line in read_text("/proc/meminfo").splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.split()[0]) * 1024
    return {key: values[key] for key in ("Cached", "MemAvailable", "Dirty", "Writeback")}


def cpu():
    values = [int(value) for value in read_text("/proc/stat").splitlines()[0].split()[1:]]
    return {"total_jiffies": sum(values), "idle_jiffies": values[3] + values[4]}


def gpu():
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,utilization.memory,memory.used", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout
    return [line.strip() for line in output.splitlines() if line.strip()]


def sample():
    return {"monotonic_s": time.monotonic(), "diskstats": diskstats(), "meminfo": meminfo(), "cpu": cpu(), "gpu": gpu()}


def cold_evict(root):
    files = sorted(path for path in root.rglob("*") if path.is_file() and not path.is_symlink())
    if not files:
        raise RuntimeError("no regular checkpoint files")
    total = 0
    for path in files:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            size = os.fstat(fd).st_size
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            total += size
        finally:
            os.close(fd)
    return {"files": len(files), "bytes": total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True, type=pathlib.Path)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--namespace", default="dynamo-snapshot-poc")
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    eviction = cold_evict(args.checkpoint_root)
    samples, stopped = [], threading.Event()

    def collect():
        while not stopped.is_set():
            samples.append(sample())
            stopped.wait(0.25)

    collector = threading.Thread(target=collect, daemon=True)
    started = time.monotonic()
    collector.start()
    command = ["./dist/snapshotctl", "restore", "--namespace", args.namespace,
               "--manifest", args.manifest, "--containers", "server",
               "--checkpoint-id", args.checkpoint_id]
    completed = subprocess.run(command, capture_output=True, text=True)
    pod_name = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))["metadata"]["name"]
    if completed.returncode == 0:
        # snapshotctl acknowledges asynchronously; the restore is complete only
        # when the agent has recorded its per-container terminal status.
        deadline = time.monotonic() + 180
        key = "nvidia.com/snapshot-restore-status.server"
        while time.monotonic() < deadline:
            pod = subprocess.run(
                ["kubectl", "-n", args.namespace, "get", "pod", pod_name, "-o", "json"],
                capture_output=True, text=True,
            )
            if pod.returncode:
                completed = subprocess.CompletedProcess(command, 1, completed.stdout, pod.stderr)
                break
            status = json.loads(pod.stdout).get("metadata", {}).get("annotations", {}).get(key)
            if status == "completed":
                break
            if status == "failed":
                completed = subprocess.CompletedProcess(command, 1, completed.stdout, pod.stdout)
                break
            time.sleep(0.25)
        else:
            completed = subprocess.CompletedProcess(command, 1, completed.stdout, "restore status timeout")
    stopped.set()
    collector.join()
    ended = time.monotonic()
    result = {
        "command": command,
        "cache_eviction": eviction,
        "duration_s": ended - started,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "samples": samples,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
