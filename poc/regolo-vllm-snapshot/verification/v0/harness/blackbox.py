#!/usr/bin/env python3
"""Frozen black-box observer for one Kubernetes vLLM startup run."""

import argparse
import datetime as dt
import json
import pathlib
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request


def kubectl(namespace, *args, timeout=30):
    command = ["kubectl", "-n", namespace, *args]
    result = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{' '.join(command)} failed: {result.stderr.strip()}")
    return result.stdout


def utc_timestamp(value):
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def get_pod(namespace, pod):
    return json.loads(kubectl(namespace, "get", "pod", pod, "-o", "json"))


def ready_timestamp(pod):
    for condition in pod.get("status", {}).get("conditions", []):
        if condition.get("type") == "Ready" and condition.get("status") == "True":
            return condition.get("lastTransitionTime")
    return None


def urlopen_json(request, timeout=5):
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.read().decode("utf-8")


def wait_health(url, deadline):
    while time.time() < deadline:
        try:
            status, _ = urlopen_json(urllib.request.Request(url), timeout=2)
            if status == 200:
                return time.time()
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(0.25)
    raise TimeoutError("health endpoint did not return HTTP 200 before deadline")


def first_token(base_url, model, prompt, expected_regex, deadline):
    payload = json.dumps(
        {"model": model, "prompt": prompt, "max_tokens": 8, "temperature": 0, "stream": True}
    ).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    first_at = None
    text_parts = []
    timeout = max(1, deadline - time.time())
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"completion returned HTTP {response.status}")
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            choices = event.get("choices") or []
            token = choices[0].get("text", "") if choices else ""
            if token and first_at is None:
                first_at = time.time()
            text_parts.append(token)
    text = "".join(text_parts)
    return first_at, text, bool(first_at and re.search(expected_regex, text))


def gpu_memory_mib(namespace, pod, container):
    output = kubectl(
        namespace,
        "exec",
        pod,
        "-c",
        container,
        "--",
        "nvidia-smi",
        "--query-compute-apps=used_memory",
        "--format=csv,noheader,nounits",
    )
    values = [float(line.strip()) for line in output.splitlines() if line.strip()]
    return sum(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--pod", required=True)
    parser.add_argument("--container", default="server")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--block", required=True, type=int)
    parser.add_argument("--opaque-arm", required=True, choices=("A", "B"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--local-port", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--prompt", default="The answer to 1+1 is")
    parser.add_argument("--expected-regex", default=r"^\s*2")
    parser.add_argument("--events-out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    first_pod = get_pod(args.namespace, args.pod)
    created_at = first_pod["metadata"]["creationTimestamp"]
    created_epoch = utc_timestamp(created_at)
    deadline = created_epoch + args.timeout_seconds
    port_forward = subprocess.Popen(
        [
            "kubectl",
            "-n",
            args.namespace,
            "port-forward",
            f"pod/{args.pod}",
            f"{args.local_port}:8000",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    record = {
        "run_id": args.run_id,
        "block": args.block,
        "opaque_arm": args.opaque_arm,
        "pod_created_at": created_at,
        "excluded": False,
        "exclusion_reason": None,
        "cluster_incident_evidence": None,
    }
    try:
        ready_at = None
        while time.time() < deadline:
            ready_at = ready_timestamp(get_pod(args.namespace, args.pod))
            if ready_at:
                break
            time.sleep(1)
        if not ready_at:
            raise TimeoutError("Pod did not become Ready before deadline")
        health_epoch = wait_health(f"http://127.0.0.1:{args.local_port}/health", deadline)
        token_epoch, response_text, valid = first_token(
            f"http://127.0.0.1:{args.local_port}",
            args.model,
            args.prompt,
            args.expected_regex,
            deadline,
        )
        record.update(
            ready_s=utc_timestamp(ready_at) - created_epoch,
            http_200_s=health_epoch - created_epoch,
            first_token_s=token_epoch - created_epoch if token_epoch else None,
            gpu_memory_mib=gpu_memory_mib(args.namespace, args.pod, args.container),
            valid_response=valid,
            response_text=response_text,
        )
    except Exception as error:  # retained failure, never silently excluded
        record.update(valid_response=False, error=type(error).__name__ + ": " + str(error))
    finally:
        events = kubectl(
            args.namespace,
            "get",
            "events",
            "--field-selector",
            f"involvedObject.name={args.pod}",
            "-o",
            "json",
        )
        args.events_out.parent.mkdir(parents=True, exist_ok=True)
        args.events_out.write_text(events)
        port_forward.terminate()
        try:
            port_forward.wait(timeout=5)
        except subprocess.TimeoutExpired:
            port_forward.kill()

    json.dump(record, sys.stdout, sort_keys=True, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0 if record.get("valid_response") else 1


if __name__ == "__main__":
    raise SystemExit(main())
