#!/usr/bin/env python3
"""Read-only environment inventory for V0; emits one JSON document."""

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys


def run(command, required=False):
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        if required:
            raise
        return {"available": False, "error": str(error)}
    value = {
        "available": result.returncode == 0,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "exit_code": result.returncode,
    }
    if required and result.returncode:
        raise RuntimeError(value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--pod", required=True)
    parser.add_argument("--container", default="server")
    parser.add_argument("--node", required=True)
    parser.add_argument("--pvc", default="snapshot-pvc")
    args = parser.parse_args()
    k = ["kubectl", "-n", args.namespace]
    inventory = {
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "commands_are_read_only": True,
        "tools": {name: shutil.which(name) for name in ("kubectl", "helm", "docker")},
        "kubernetes_version": run(["kubectl", "version", "-o", "json"]),
        "pod": run(k + ["get", "pod", args.pod, "-o", "json"]),
        "pod_events": run(
            k
            + [
                "get",
                "events",
                "--field-selector",
                f"involvedObject.name={args.pod}",
                "-o",
                "json",
            ]
        ),
        "node": run(["kubectl", "get", "node", args.node, "-o", "json"]),
        "pvc": run(k + ["get", "pvc", args.pvc, "-o", "json"]),
        "storage_class": run(["kubectl", "get", "storageclass", "-o", "json"]),
        "gpu_and_driver": run(
            k
            + [
                "exec",
                args.pod,
                "-c",
                args.container,
                "--",
                "nvidia-smi",
                "--query-gpu=name,driver_version,uuid,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "mounts": run(k + ["exec", args.pod, "-c", args.container, "--", "mount"]),
        "health": run(k + ["exec", args.pod, "-c", args.container, "--", "python3", "-c", "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health').status)"]),
    }
    json.dump(inventory, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
