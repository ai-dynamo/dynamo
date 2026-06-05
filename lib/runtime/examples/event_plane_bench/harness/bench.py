#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DIS-2172 unified event-plane benchmark: single-node OR multi-node, one command.

Sweeps  transport x workers(p) x subscribers(s) x topic  (with trials), drives
request load through the frontend so mocker workers emit real event-plane traffic,
and measures one-way publish->deliver latency (ns) + per-publisher seq-gap loss in
the `event_plane_bench_sub` subscribers. Writes one JSON per (cell, subscriber) and
an aggregate summary.json. Feed the output dir to report.py for plots + markdown.

Launchers (same sweep logic, different process placement + clock):
  local : every process on this host via subprocess. Clock = CLOCK_MONOTONIC
          (same-host, exact). Traffic is loopback (no NIC). Good for dev / quick A/B.
  slurm : processes spread across the job's allocated nodes via `srun --nodelist`.
          node[0] runs infra(etcd/nats)+frontend+subscribers+loadgen; the p workers
          are spread across ALL nodes (gpus_per_node mocker workers each), so their
          events cross the real NIC. Clock = CLOCK_REALTIME (REQUIRES PTP/NTP-synced
          nodes). Run this inside an salloc/sbatch allocation.

NVL72 target: --launcher slurm --gpus-per-node 18  (one mocker worker per GPU),
sweep --workers up to nodes*18 (72, 144, ...); --subs = number of router replicas.

Clock selection is passed to BOTH publisher (mocker, via DYN_EVENT_CLOCK) and
subscriber (event_plane_bench_sub, via DYN_BENCH_CLOCK) so a single build switches
at runtime. (Requires the bench_sub/mod.rs `--clock` support; see report/PR.)
"""
import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.environ.get("DYN_BENCH_PY", "/tmp/dis2172-venv/bin/python")
SUB_BIN = os.environ.get(
    "DYN_BENCH_SUB_BIN",
    "/home/zhongdaor/Workplace/dynamo/lib/runtime/examples/target/release/event_plane_bench_sub",
)
MODEL = os.environ.get("DYN_BENCH_MODEL", "Qwen/Qwen2.5-0.5B")
HTTP_PORT = 8000
NS_SCOPED_TOPICS = {"kv_metrics", "prefill_events"}  # namespace-scoped subjects


# --------------------------------------------------------------------------- #
# Launchers: the ONLY thing that differs between single-node and multi-node.
# --------------------------------------------------------------------------- #
class Launcher:
    name = "base"
    nodes = ["localhost"]

    def popen(self, argv, env, logpath, node_idx=0):
        raise NotImplementedError

    def addr(self, node_idx=0):
        """Routable address of a node (for etcd/nats/frontend endpoints)."""
        return "127.0.0.1"

    @property
    def nnodes(self):
        return len(self.nodes)


class LocalLauncher(Launcher):
    name = "local"
    nodes = ["localhost"]
    clock = "monotonic"

    def popen(self, argv, env, logpath, node_idx=0):
        return subprocess.Popen(argv, env=env, stdout=open(logpath, "w"),
                                stderr=subprocess.STDOUT)

    def addr(self, node_idx=0):
        return "127.0.0.1"


class SlurmLauncher(Launcher):
    name = "slurm"
    clock = "realtime"  # multi-host: monotonic is per-host; needs PTP-synced realtime

    def __init__(self):
        nl = os.environ.get("SLURM_JOB_NODELIST")
        if not nl:
            raise SystemExit("slurm launcher requires SLURM_JOB_NODELIST "
                             "(run inside salloc/sbatch).")
        self.nodes = subprocess.check_output(
            ["scontrol", "show", "hostnames", nl], text=True).split()
        print(f"[slurm] {len(self.nodes)} nodes: {self.nodes}", flush=True)

    def popen(self, argv, env, logpath, node_idx=0):
        node = self.nodes[node_idx % len(self.nodes)]
        srun = ["srun", "--nodes=1", "--ntasks=1", "--overlap", f"--nodelist={node}"]
        img = env.get("DYN_BENCH_IMAGE")
        if img:  # pyxis container mode (argv uses in-container paths: DYN_BENCH_PY / SUB_BIN)
            srun.append(f"--container-image={img}")
            mounts = env.get("DYN_BENCH_MOUNTS")
            if mounts:
                srun.append(f"--container-mounts={mounts}")
        srun += ["--"] + argv
        return subprocess.Popen(srun, env=env, stdout=open(logpath, "w"),
                                stderr=subprocess.STDOUT)

    def addr(self, node_idx=0):
        return self.nodes[node_idx]  # hostname, resolvable across the allocation


def make_launcher(name):
    return {"local": LocalLauncher, "slurm": SlurmLauncher}[name]()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def wait_model(frontend_addr, port, timeout=120):
    import urllib.request
    url = f"http://{frontend_addr}:{port}/v1/models"
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


def worker_distribution(p, nnodes, gpus_per_node):
    """Return list of (node_idx, num_workers) mocker placements summing to p."""
    if nnodes == 1:
        return [(0, p)]
    placements, remaining, node = [], p, 0
    per = gpus_per_node or ((p + nnodes - 1) // nnodes)
    while remaining > 0 and node < nnodes:
        n = min(per, remaining)
        placements.append((node, n))
        remaining -= n
        node += 1
    if remaining > 0:  # more workers than nodes*per: wrap (oversubscribe)
        i = 0
        while remaining > 0:
            placements.append((i % nnodes, min(per, remaining)))
            remaining -= per
            i += 1
    return placements


def teardown(procs):
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


# --------------------------------------------------------------------------- #
# One (transport, p, s, trial) cell
# --------------------------------------------------------------------------- #
def run_cell(lr, transport, p, s, topics, gpus_per_node, duration, warmup,
             conc, outdir, idx, trial, infra_addr, speedup_ratio):
    ns = f"dis2172_{idx}"
    fe_addr = lr.addr(0)
    env = dict(os.environ)
    env["DYN_DISCOVERY_BACKEND"] = "etcd"
    env["ETCD_ENDPOINTS"] = f"http://{infra_addr}:2379"
    env["DYN_EVENT_CLOCK"] = lr.clock          # publisher (mocker) clock
    bench_transport = "nats"
    if transport in ("zmq", "zmq-broker"):
        env["DYN_EVENT_PLANE"] = "zmq"
        env.pop("NATS_SERVER", None)
        bench_transport = "zmq"
        if transport == "zmq-broker":
            env["DYN_ZMQ_BROKER_URL"] = (
                f"xsub=tcp://{fe_addr}:5555 , xpub=tcp://{fe_addr}:5556")
    else:
        env["DYN_EVENT_PLANE"] = "nats"
        env["NATS_SERVER"] = f"nats://{infra_addr}:4222"

    def log(name):
        return f"{outdir}/{name}_{idx}.log"

    procs = []
    if transport == "zmq-broker":
        procs.append(lr.popen([PY, f"{HERE}/zmq_broker.py"], env, log("broker"), 0))
        time.sleep(1)

    # frontend on node0
    procs.append(lr.popen(
        [PY, "-m", "dynamo.frontend", "--http-port", str(HTTP_PORT),
         "--router-mode", "round-robin", "--namespace", ns],
        env, log("fe"), 0))

    # workers: one mocker process per node, num-workers = its share of p
    for (node_idx, nw) in worker_distribution(p, lr.nnodes, gpus_per_node):
        procs.append(lr.popen(
            [PY, "-m", "dynamo.mocker", "--model-path", MODEL, "--model-name", "mock",
             "--endpoint", f"dyn://{ns}.backend.generate", "--num-workers", str(nw),
             "--event-plane", bench_transport, "--speedup-ratio", speedup_ratio],
            env, log(f"mk_n{node_idx}"), node_idx))

    model = wait_model(fe_addr, HTTP_PORT, 120)
    if not model:
        teardown(procs)
        return [{"error": "model_not_registered", "mode": transport, "n_workers": p,
                 "n_subs": s, "topic": t, "trial": trial} for t in topics]

    # subscribers on node0 (the router/frontend node)
    subs = []
    for topic in topics:
        for si in range(s):
            outp = f"{outdir}/{transport}_p{p}_s{s}_t{trial}_{topic}_sub{si}.json"
            senv = dict(env)
            senv.update(
                DYN_BENCH_NAMESPACE=ns, DYN_BENCH_COMPONENT="backend",
                DYN_BENCH_TOPIC=topic, DYN_BENCH_TRANSPORT=bench_transport,
                DYN_BENCH_SCOPE=("namespace" if topic in NS_SCOPED_TOPICS else "component"),
                DYN_BENCH_CLOCK=lr.clock,          # subscriber clock (match publisher)
                DYN_BENCH_DURATION=str(duration), DYN_BENCH_WARMUP=str(warmup),
                DYN_BENCH_OUT=outp)
            subs.append((lr.popen([SUB_BIN], senv, f"{outdir}/sub_{idx}_{topic}_{si}.log", 0),
                         outp, topic, si))

    time.sleep(2)
    lg = lr.popen([PY, f"{HERE}/loadgen.py", "--port", str(HTTP_PORT), "--model", model,
                   "--concurrency", str(conc), "--duration", str(duration + warmup + 4)],
                  {**env, "LOADGEN_HOST": fe_addr}, log("load"), 0)

    for (pp, _o, _t, _s) in subs:
        try:
            pp.wait(timeout=duration + warmup + 60)
        except subprocess.TimeoutExpired:
            pp.kill()
    try:
        lg.wait(timeout=15)
    except subprocess.TimeoutExpired:
        lg.kill()

    rows = []
    for (pp, outp, topic, si) in subs:
        try:
            with open(outp) as f:
                r = json.load(f)
        except Exception:
            r = {"error": "no_output", "topic": topic}
        r.update(mode=transport, n_workers=p, n_subs=s, sub_idx=si, trial=trial,
                 launcher=lr.name, clock=lr.clock, nnodes=lr.nnodes)
        rows.append(r)
    teardown(procs)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--launcher", choices=["local", "slurm"], default="local")
    ap.add_argument("--transports", default="nats,zmq")
    ap.add_argument("--workers", default="1,4,16,64", help="p sweep (publishers)")
    ap.add_argument("--subs", default="1", help="s sweep (subscribers/router replicas)")
    ap.add_argument("--topics", default="kv-events,forward-pass-metrics")
    ap.add_argument("--gpus-per-node", type=int, default=0,
                    help="multi-node: mocker workers per node (e.g. 18 for NVL72 tray)")
    ap.add_argument("--duration", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--speedup-ratio", default="10",
                    help="mocker exec speedup; 0 = unthrottled (saturation test, no-PTP throughput mode)")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--infra-addr", default=None,
                    help="etcd/nats host (default: node0 address)")
    ap.add_argument("--out", default=f"{HERE}/results/run")
    a = ap.parse_args()

    lr = make_launcher(a.launcher)
    infra_addr = a.infra_addr or lr.addr(0)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    transports = a.transports.split(",")
    workers = [int(x) for x in a.workers.split(",")]
    subs = [int(x) for x in a.subs.split(",")]
    topics = a.topics.split(",")

    meta = dict(launcher=lr.name, nodes=lr.nodes, clock=lr.clock, infra_addr=infra_addr,
                transports=transports, workers=workers, subs=subs, topics=topics,
                gpus_per_node=a.gpus_per_node, duration=a.duration, trials=a.trials)
    with open(f"{a.out}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[bench] launcher={lr.name} nodes={lr.nnodes} clock={lr.clock} "
          f"infra={infra_addr}", flush=True)

    all_rows, idx = [], 0
    for trial in range(1, a.trials + 1):
        for t in transports:
            for p in workers:
                for s in subs:
                    idx += 1
                    print(f"[bench] trial={trial} {t} p={p} s={s} topics={topics}",
                          flush=True)
                    rows = run_cell(lr, t, p, s, topics, a.gpus_per_node, a.duration,
                                    a.warmup, a.concurrency, a.out, idx, trial, infra_addr,
                                    a.speedup_ratio)
                    for r in rows:
                        lat = r.get("latency_ns", {})
                        print(f"    {r.get('topic'):>22} p50={lat.get('p50')}ns "
                              f"gaps={r.get('gaps')} {r.get('error','')}", flush=True)
                    all_rows.extend(rows)
                    with open(f"{a.out}/summary.json", "w") as f:
                        json.dump(all_rows, f, indent=2)
                    time.sleep(3)

    print(f"[bench] DONE -> {a.out}/summary.json ({len(all_rows)} rows)\n"
          f"        next: python report.py --in {a.out}")


if __name__ == "__main__":
    main()
