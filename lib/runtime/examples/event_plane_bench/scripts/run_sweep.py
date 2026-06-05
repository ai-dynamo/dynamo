#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DIS-2172 event-plane NATS-vs-ZMQ sweep orchestrator.

For each (transport, n_workers) cell:
  1. start a frontend (round-robin) + one mocker process with --num-workers N
     (N independent event-plane publishers sharing one tokio runtime),
  2. wait until the model registers,
  3. start n_subs measurement subscribers per topic (event_plane_bench_sub),
  4. drive request load (loadgen.py) so the workers keep producing events,
  5. collect each subscriber's JSON (ns latency percentiles + seq gaps) plus a
     coarse established-TCP-connection count (proxy for ZMQ O(N*M) fan-out),
  6. tear everything down and cool off.

Everything is identical across transports EXCEPT DYN_EVENT_PLANE (and, for ZMQ,
NATS_SERVER is removed so the run is genuinely NATS-free — DGH-900).

Reuses the locally running nats(4222)/etcd(2379). The instrumented bindings
live in the temp venv built by the setup step.
"""
import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

VENV = "/tmp/dis2172-venv"
PY = f"{VENV}/bin/python"
SUB_BIN = "/home/zhongdaor/Workplace/dynamo/lib/runtime/examples/target/release/event_plane_bench_sub"
DIS2172 = "/home/zhongdaor/Workplace/dis2172-bench"
MODEL_PATH = "Qwen/Qwen2.5-0.5B"
HTTP_PORT = 8000


def wait_http_model(port: int, timeout: int = 60):
    url = f"http://localhost:{port}/v1/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                data = json.load(r).get("data", [])
                if data:
                    return data[0]["id"]
        except Exception:
            pass
        time.sleep(1)
    return None


def count_tcp_conns() -> int:
    """Coarse proxy for fan-out: count of established TCP connections."""
    try:
        out = subprocess.check_output(
            ["ss", "-tnH", "state", "established"], text=True
        )
        return len([ln for ln in out.splitlines() if ln.strip()])
    except Exception:
        return -1


def teardown(procs) -> None:
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass
    time.sleep(2)
    for p in procs:
        try:
            if p.poll() is None:
                p.kill()
        except Exception:
            pass
    time.sleep(2)


def run_one(transport, n_workers, n_subs, topics, duration, warmup, conc, outdir, idx, trial):
    ns = f"dis2172s{idx}"
    env = dict(os.environ)
    env["DYN_DISCOVERY_BACKEND"] = "etcd"
    bench_transport = "nats"
    if transport in ("zmq", "zmq-broker"):
        env["DYN_EVENT_PLANE"] = "zmq"
        env.pop("NATS_SERVER", None)  # DGH-900: keep ZMQ run NATS-free
        bench_transport = "zmq"  # bench_sub only knows nats|zmq; broker is auto via env
        if transport == "zmq-broker":
            # Brokered ZMQ: publishers→XSUB(5555), subscribers→XPUB(5556) via a proxy.
            # O(p+s) connections instead of direct mode's O(p×s).
            env["DYN_ZMQ_BROKER_URL"] = "xsub=tcp://127.0.0.1:5555 , xpub=tcp://127.0.0.1:5556"
    else:
        env["DYN_EVENT_PLANE"] = "nats"

    def log(name):
        return open(f"{outdir}/{name}_{idx}.log", "w")

    procs = []
    if transport == "zmq-broker":
        broker = subprocess.Popen(
            [PY, f"{DIS2172}/zmq_broker.py"], env=env,
            stdout=log("broker"), stderr=subprocess.STDOUT,
        )
        procs.append(broker)
        time.sleep(1)  # let the XSUB/XPUB proxy bind before pubs/subs connect
    fe = subprocess.Popen(
        [PY, "-m", "dynamo.frontend", "--http-port", str(HTTP_PORT),
         "--router-mode", "round-robin", "--namespace", ns],
        env=env, stdout=log("fe"), stderr=subprocess.STDOUT,
    )
    procs.append(fe)
    mk = subprocess.Popen(
        [PY, "-m", "dynamo.mocker", "--model-path", MODEL_PATH, "--model-name", "mock",
         "--endpoint", f"dyn://{ns}.backend.generate", "--num-workers", str(n_workers),
         "--event-plane", bench_transport, "--speedup-ratio", "10"],
        env=env, stdout=log("mk"), stderr=subprocess.STDOUT,
    )
    procs.append(mk)

    model = wait_http_model(HTTP_PORT, 90)
    if not model:
        teardown(procs)
        return [{"error": "model_not_registered", "transport": transport,
                 "n_workers": n_workers, "n_subs": n_subs, "topic": t} for t in topics]

    subs = []
    for topic in topics:
        for s in range(n_subs):
            outp = f"{outdir}/{transport}_N{n_workers}_M{n_subs}_t{trial}_{topic}_s{s}.json"
            senv = dict(env)
            ns_scoped = {"kv_metrics", "prefill_events"}  # namespace-scoped subjects
            senv.update(
                DYN_BENCH_NAMESPACE=ns, DYN_BENCH_COMPONENT="backend",
                DYN_BENCH_TOPIC=topic, DYN_BENCH_TRANSPORT=bench_transport,
                DYN_BENCH_SCOPE=("namespace" if topic in ns_scoped else "component"),
                DYN_BENCH_DURATION=str(duration), DYN_BENCH_WARMUP=str(warmup),
                DYN_BENCH_OUT=outp,
            )
            p = subprocess.Popen(
                [SUB_BIN], env=senv,
                stdout=open(f"{outdir}/sub_{idx}_{topic}_s{s}.log", "w"),
                stderr=subprocess.STDOUT,
            )
            subs.append((p, outp, topic, s))

    time.sleep(2)  # let subscribers connect/discover before load starts
    lg = subprocess.Popen(
        [PY, f"{DIS2172}/loadgen.py", "--port", str(HTTP_PORT), "--model", model,
         "--concurrency", str(conc), "--duration", str(duration + warmup + 4)],
        env=env, stdout=log("load"), stderr=subprocess.STDOUT,
    )

    # Sample fan-out mid-run (after warmup, while load is flowing).
    time.sleep(warmup + max(1, duration // 2))
    conns = count_tcp_conns()

    for (p, _outp, _topic, _s) in subs:
        try:
            p.wait(timeout=duration + warmup + 40)
        except subprocess.TimeoutExpired:
            p.kill()
    try:
        lg.wait(timeout=15)
    except subprocess.TimeoutExpired:
        lg.kill()

    rows = []
    for (p, outp, topic, s) in subs:
        try:
            with open(outp) as f:
                r = json.load(f)
        except Exception:
            r = {"error": "no_output", "topic": topic, "transport": transport}
        r.update(n_workers=n_workers, n_subs=n_subs, sub_idx=s,
                 tcp_conns_est=conns, model=model, trial=trial, mode=transport)
        rows.append(r)

    teardown(procs)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transports", default="nats,zmq")
    ap.add_argument("--workers", default="1,2,4,8,16,32,64,128")
    ap.add_argument("--subs", default="1", help="comma list of subscriber counts (M)")
    ap.add_argument("--trials", type=int, default=1, help="repeat each cell N times (Kyle-style)")
    ap.add_argument("--topics", default="kv-events,forward-pass-metrics")
    ap.add_argument("--duration", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--outdir", default=f"{DIS2172}/results")
    a = ap.parse_args()

    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    transports = a.transports.split(",")
    workers = [int(x) for x in a.workers.split(",")]
    subs_list = [int(x) for x in str(a.subs).split(",")]
    topics = a.topics.split(",")

    all_rows = []
    idx = 0
    for trial in range(1, a.trials + 1):
        for t in transports:
            for n in workers:
                for m in subs_list:
                    idx += 1
                    print(f"[sweep] trial={trial} {t} N={n} M={m} topics={topics} ...", flush=True)
                    rows = run_one(t, n, m, topics, a.duration, a.warmup,
                                   a.concurrency, a.outdir, idx, trial)
                    for r in rows:
                        lat = r.get("latency_ns", {})
                        print(f"    {r.get('topic'):>22} recv={r.get('received')} "
                              f"p50={lat.get('p50')}ns p99={lat.get('p99')}ns "
                              f"gaps={r.get('gaps')} drop={r.get('drop_rate')} "
                              f"conns={r.get('tcp_conns_est')} {r.get('error','')}", flush=True)
                    all_rows.extend(rows)
                    with open(f"{a.outdir}/summary.json", "w") as f:
                        json.dump(all_rows, f, indent=2)
                    time.sleep(3)

    print(f"[sweep] DONE -> {a.outdir}/summary.json ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
