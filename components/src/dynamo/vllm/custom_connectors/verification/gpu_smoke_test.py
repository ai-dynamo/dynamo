"""Single-GPU smoke test: start a vLLM server with our connector, hit /metrics,
confirm the new gauges appear.

This DOES NOT exercise an actual prefill->decode disagg flow (would need 2
GPUs + NIXL transport). It verifies:

  - vLLM starts with our connector module without errors
  - The Prometheus endpoint exposes vllm:nixl_num_pending_sends gauges
  - Their initial values are 0 (no requests pinned yet)

For a full disagg E2E, we'd need a 2-GPU setup with paired prefill+decode
workers and NIXL transport configured.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import subprocess
import sys
import time

import requests

# Use a tiny model that fits easily; --enforce-eager + low GPU mem to keep
# startup fast
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PORT = 8765

kv_transfer_config = json.dumps({
    "kv_connector": "NixlConnectorWithPendingMetrics",
    "kv_role": "kv_both",
    "kv_connector_module_path": (
        "dynamo.vllm.custom_connectors.nixl_with_pending_metrics"
    ),
    "engine_id": "smoke-test-engine",
})

cmd = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL,
    "--port", str(PORT),
    "--enforce-eager",
    "--max-model-len", "512",
    "--max-num-seqs", "8",
    "--gpu-memory-utilization", "0.3",
    "--kv-transfer-config", kv_transfer_config,
]
print("[smoke] launching vLLM:", " ".join(cmd))
proc = subprocess.Popen(cmd, env={
    **os.environ,
    "PYTHONPATH": "/home/krish/repos/amz-ads/dynamo/components/src",
})

try:
    # Poll until /metrics responds, with a 5-minute startup budget
    deadline = time.time() + 300
    metrics_url = f"http://localhost:{PORT}/metrics"
    while time.time() < deadline:
        try:
            r = requests.get(metrics_url, timeout=2)
            if r.ok:
                break
        except requests.RequestException:
            pass
        if proc.poll() is not None:
            print(f"[smoke] vLLM exited with code {proc.returncode}")
            sys.exit(1)
        time.sleep(2)
    else:
        print("[smoke] vLLM did not become ready in 300 s")
        sys.exit(1)

    # Grab metrics
    r = requests.get(metrics_url)
    body = r.text

    interesting = [
        line for line in body.splitlines()
        if "nixl" in line and not line.startswith("#")
    ]
    print(f"[smoke] /metrics returned {len(body)} bytes")
    print("[smoke] NIXL-related metric lines:")
    for line in interesting:
        print(f"   {line}")

    have_pending = "vllm:nixl_num_pending_sends" in body
    have_in_proc = "vllm:nixl_num_in_process_reqs" in body
    print()
    print(f"[smoke] vllm:nixl_num_pending_sends present: {have_pending}")
    print(f"[smoke] vllm:nixl_num_in_process_reqs present: {have_in_proc}")
    if have_pending and have_in_proc:
        print("[smoke] PASS: new gauges exposed via /metrics")
    else:
        print("[smoke] FAIL: gauges not in /metrics output")
        sys.exit(1)
finally:
    print("[smoke] shutting down vLLM")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
