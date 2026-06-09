#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DIS-2172 REAL end-to-end event-plane THROUGHPUT + LOSS benchmark.

Unlike the latency bench (which used a synthetic `event_plane_bench_sub`), this
harness instruments the THREE REAL event-plane consumers and measures how well
they actually CONSUME events under a real request-driven load:

  1. kv-events         -> consumed by the KV Router indexer (in `dynamo.frontend
                          --router-mode kv`). Counter at the indexer recv loop.
  2. forward-pass-metrics (FPM) -> consumed by the lightweight FPM receiver
                          (`python -m dynamo.common.recv_forward_pass_metrics
                          --mode throughput`). NOT the full Planner.
  3. active_sequences  -> router<->router replica sync. Only fires with s>=2
                          router replicas all started with --router-replica-sync.
                          Correctness-first: we report zero-gap + actual (low)
                          rate, NOT a throughput stress.

Each consumer is built with the receive-side counter (Rust `RecvCounter` /
Python `get_throughput_stats`), gated by DYN_BENCH_COUNT=1, emitting a per-window
JSON line (`{"dis2172_recv":"window",...}` / `"final"`) to its stderr. The
orchestrator scrapes those lines from each component's node-local log.

Topology per (transport, p, s) cell on node0 (+ workers spread across nodes):
  - infra: etcd + nats (env points every task at node0).
  - s kv-mode frontends on ports 8000..8000+s, each containing a KV Router that
    subscribes to kv-events (#1) and, when s>=2, active_sequences (#3) with
    --router-replica-sync.
  - 1 recv_forward_pass_metrics receiver (#2).
  - p mocker workers (publish kv-events + FPM) spread across the allocation.
  - loadgen processes that ROTATE requests across all s frontend ports so every
    replica routes traffic (-> active_sequences flows between routers).

Launchers: `local` (subprocess, loopback) and `slurm` (srun across the
allocation, optionally pyxis/enroot container). Reuses the proven launcher and
resource-sampler machinery from the latency bench.
"""
import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.environ.get("DYN_BENCH_PY", "/opt/dynamo/venv/bin/python")
MODEL = os.environ.get("DYN_BENCH_MODEL", "Qwen/Qwen2.5-0.5B")
HTTP_PORT_BASE = int(os.environ.get("DYN_BENCH_HTTP_PORT_BASE", "8000"))

# How long to wait for the frontend to register the model (cold pyxis extract +
# high mocker density can exceed the old 120s). Override DYN_BENCH_MODEL_WAIT.
MODEL_WAIT = int(os.environ.get("DYN_BENCH_MODEL_WAIT", "600"))
EXTRACT_PAD = int(os.environ.get("DYN_BENCH_EXTRACT_PAD", "220"))
# Per-window length for the receive-side counters (passed via DYN_BENCH_WINDOW_SECS).
WINDOW_SECS = int(os.environ.get("DYN_BENCH_WINDOW_SECS", "10"))
CONTAINER_NAME = os.environ.get("DYN_BENCH_CONTAINER_NAME", "dis2172e2e")

# Per-task stdout/stderr logs go NODE-LOCAL (not the NFS OUT dir). We scrape the
# per-window JSON counter lines from these logs at cell end.
LOGDIR = os.environ.get(
    "DYN_BENCH_LOGDIR",
    f"/tmp/dis2172-e2e-logs-{os.environ.get('SLURM_JOB_ID', os.getpid())}",
)

RESSAMPLE = os.environ.get("DYN_BENCH_RESSAMPLE", "1") not in ("0", "", "false")
RESSAMPLE_N = int(os.environ.get("DYN_BENCH_RESSAMPLE_N", "2"))
HOST_PY = os.environ.get("HOST_PY", "python3")


# --------------------------------------------------------------------------- #
# Launchers (process placement). Reused from the latency bench.
# --------------------------------------------------------------------------- #
class Launcher:
    name = "base"
    nodes = ["localhost"]

    def popen(self, argv, env, logpath, node_idx=0):
        raise NotImplementedError

    def popen_host(self, argv, env, logpath, node_idx=0):
        return self.popen(argv, env, logpath, node_idx)

    def addr(self, node_idx=0):
        return "127.0.0.1"

    @property
    def nnodes(self):
        return len(self.nodes)


class LocalLauncher(Launcher):
    name = "local"
    nodes = ["localhost"]

    def popen(self, argv, env, logpath, node_idx=0):
        return subprocess.Popen(argv, env=env, stdout=open(logpath, "w"),
                                stderr=subprocess.STDOUT)

    def addr(self, node_idx=0):
        return "127.0.0.1"


class SlurmLauncher(Launcher):
    name = "slurm"

    def __init__(self):
        nl = os.environ.get("SLURM_JOB_NODELIST")
        if not nl:
            raise SystemExit("slurm launcher requires SLURM_JOB_NODELIST "
                             "(run inside salloc/sbatch).")
        self.nodes = subprocess.check_output(
            ["scontrol", "show", "hostnames", nl], text=True).split()
        print(f"[slurm] {len(self.nodes)} nodes: {self.nodes}", flush=True)
        self._container_ready = False

    def prepare_containers(self, env):
        """Extract the image ONCE per node into a persistent pyxis named container."""
        img = env.get("DYN_BENCH_IMAGE")
        if not img or not CONTAINER_NAME:
            return
        for node in self.nodes:
            srun = ["srun", "--nodes=1", "--ntasks=1", "--overlap",
                    f"--nodelist={node}",
                    f"--container-image={img}",
                    f"--container-name={CONTAINER_NAME}"]
            mounts = env.get("DYN_BENCH_MOUNTS")
            if mounts:
                srun.append(f"--container-mounts={mounts}")
            srun += ["--", "true"]
            print(f"[slurm] extracting image into '{CONTAINER_NAME}' on {node} ...",
                  flush=True)
            t0 = time.time()
            rc = subprocess.call(srun, env=env)
            print(f"[slurm]   {node}: extract rc={rc} in {time.time()-t0:.0f}s",
                  flush=True)
        self._container_ready = True

    def popen(self, argv, env, logpath, node_idx=0):
        node = self.nodes[node_idx % len(self.nodes)]
        srun = ["srun", "--nodes=1", "--ntasks=1", "--overlap", f"--nodelist={node}"]
        img = env.get("DYN_BENCH_IMAGE")
        if img:
            if self._container_ready and CONTAINER_NAME:
                srun.append(f"--container-name={CONTAINER_NAME}")
            else:
                srun.append(f"--container-image={img}")
            mounts = env.get("DYN_BENCH_MOUNTS")
            if mounts:
                srun.append(f"--container-mounts={mounts}")
        srun += ["--"] + argv
        return subprocess.Popen(srun, env=env, stdout=open(logpath, "w"),
                                stderr=subprocess.STDOUT)

    def popen_host(self, argv, env, logpath, node_idx=0):
        node = self.nodes[node_idx % len(self.nodes)]
        srun = ["srun", "--nodes=1", "--ntasks=1", "--overlap",
                f"--nodelist={node}", "--"] + argv
        return subprocess.Popen(srun, env=env, stdout=open(logpath, "w"),
                                stderr=subprocess.STDOUT)

    def addr(self, node_idx=0):
        return self.nodes[node_idx]


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
    if remaining > 0:
        i = 0
        while remaining > 0:
            placements.append((i % nnodes, min(per, remaining)))
            remaining -= per
            i += 1
    return placements


_RECV_RE = re.compile(r'\{"dis2172_recv":"(window|final|window_start)".*?\}')


def res_snapshot(lr, transport, idx, snap_label):
    sampler = f"{HERE}/ressample.py"
    procs = []
    for node_idx in range(lr.nnodes):
        node = lr.nodes[node_idx] if hasattr(lr, "nodes") else "localhost"
        outp = f"{LOGDIR}/res_{idx}_{snap_label}_n{node_idx}.json"
        argv = [HOST_PY, sampler, "--out", outp, "--node", str(node),
                "--transport", transport, "--cell", str(idx), "--snap", snap_label]
        try:
            procs.append((lr.popen_host(argv, dict(os.environ),
                                        f"{LOGDIR}/ressampler_{idx}_{snap_label}_n{node_idx}.log",
                                        node_idx), outp, node))
        except Exception:
            pass
    for (pp, outp, node) in procs:
        try:
            pp.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                pp.kill()
            except Exception:
                pass
    snaps = []
    for node_idx in range(lr.nnodes):
        logp = f"{LOGDIR}/ressampler_{idx}_{snap_label}_n{node_idx}.log"
        try:
            with open(logp) as f:
                txt = f.read().strip()
            line = [ln for ln in txt.splitlines() if ln.strip().startswith("{")]
            if line:
                snaps.append(json.loads(line[-1]))
        except Exception:
            pass
    return snaps


def summarize_res(snaps):
    if not snaps:
        return {}
    classes = ("mocker", "frontend", "router", "recv_fpm", "nats",
               "zmq_broker", "loadgen")
    tcp_by_node = {}
    for s in snaps:
        n = s.get("node", "?")
        tcp = s.get("established_tcp")
        if tcp is None:
            continue
        tcp_by_node[n] = max(tcp_by_node.get(n, 0), tcp)
    est_tcp_total_peak = sum(tcp_by_node.values()) if tcp_by_node else None
    est_tcp_node_peak = max(tcp_by_node.values()) if tcp_by_node else None
    proc = {}
    for cls in classes:
        cpu_vals, rss_vals, fd_vals, nproc_vals = [], [], [], []
        for s in snaps:
            p = s.get("procs", {}).get(cls)
            if not p or not p.get("nproc"):
                continue
            cpu_vals.append(p.get("cpu_pct", 0.0))
            rss_vals.append(p.get("rss_kb", 0))
            fd_vals.append(p.get("fds", 0))
            nproc_vals.append(p.get("nproc", 0))
        if not nproc_vals:
            continue
        proc[cls] = {
            "cpu_pct_peak": round(max(cpu_vals), 1),
            "rss_mb_peak": round(max(rss_vals) / 1024.0, 1),
            "fds_peak": max(fd_vals),
            "nproc_peak": max(nproc_vals),
        }
    return {
        "established_tcp_total_peak": est_tcp_total_peak,
        "established_tcp_node_peak": est_tcp_node_peak,
        "established_tcp_by_node": tcp_by_node,
        "procs": proc,
        "n_snapshots": len(snaps),
    }


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
def run_cell(lr, transport, p, s, events, gpus_per_node, duration, conc,
             lg_procs, outdir, idx, trial, infra_addr, speedup_ratio):
    ns = f"dis2172e2e_{idx}"
    fe_addr = lr.addr(0)
    replica_sync = s >= 2  # active_sequences only fires across >=2 routers

    env = dict(os.environ)
    env["DYN_DISCOVERY_BACKEND"] = "etcd"
    env["ETCD_ENDPOINTS"] = f"http://{infra_addr}:2379"
    env["DYN_BENCH_COUNT"] = "1"               # enable receive-side counters
    env["DYN_BENCH_WINDOW_SECS"] = str(WINDOW_SECS)
    if transport == "zmq":
        env["DYN_EVENT_PLANE"] = "zmq"
        env.pop("NATS_SERVER", None)
    else:
        env["DYN_EVENT_PLANE"] = "nats"
        env["NATS_SERVER"] = f"nats://{infra_addr}:4222"
    bench_transport = transport

    def log(name):
        return f"{LOGDIR}/{name}_{idx}.log"

    procs = []

    # --- s kv-mode frontends (each contains a KV Router) on 8000..8000+s ---
    fe_ports = [HTTP_PORT_BASE + i for i in range(s)]
    fe_procs = []
    for i, port in enumerate(fe_ports):
        argv = [PY, "-m", "dynamo.frontend", "--http-port", str(port),
                "--router-mode", "kv", "--namespace", ns]
        if replica_sync:
            argv.append("--router-replica-sync")
        # A router process runs TWO RecvCounters (kv-events + active_sequences),
        # each with its own default site label, so we do NOT set
        # DYN_BENCH_COUNT_SITE here (that would override both to one label).
        # The scraper filters the log by the embedded "site" field.
        pp = lr.popen(argv, dict(env), log(f"fe{i}"), 0)
        procs.append(pp)
        fe_procs.append((pp, i, port))

    # --- p mocker workers spread across nodes ---
    for (node_idx, nw) in worker_distribution(p, lr.nnodes, gpus_per_node):
        procs.append(lr.popen(
            [PY, "-m", "dynamo.mocker", "--model-path", MODEL, "--model-name", "mock",
             "--endpoint", f"dyn://{ns}.backend.generate", "--num-workers", str(nw),
             "--event-plane", bench_transport, "--speedup-ratio", speedup_ratio],
            env, log(f"mk_n{node_idx}"), node_idx))

    # Wait for the FIRST frontend to register the model.
    model = wait_model(fe_addr, fe_ports[0], MODEL_WAIT)
    if not model:
        teardown(procs)
        return [{"error": "model_not_registered", "mode": transport, "n_workers": p,
                 "n_subs": s, "event": e, "trial": trial} for e in events]

    # --- FPM receiver (#2): lightweight standalone, throughput mode ---
    fpm_log = log("recv_fpm")
    fpm_proc = lr.popen(
        [PY, "-m", "dynamo.common.recv_forward_pass_metrics",
         "--namespace", ns, "--component", "backend", "--endpoint", "generate",
         "--mode", "throughput"],
        env, fpm_log, 0)
    procs.append(fpm_proc)

    time.sleep(2)

    # --- loadgen: rotate across ALL frontend ports so every replica routes ---
    per_proc = max(1, conc // lg_procs)
    lg_duration = duration + 4 + EXTRACT_PAD
    ports_csv = ",".join(str(pp) for pp in fe_ports)
    lgs = []
    for li in range(lg_procs):
        lgs.append(lr.popen(
            [PY, f"{HERE}/loadgen.py", "--ports", ports_csv, "--model", model,
             "--concurrency", str(per_proc), "--duration", str(lg_duration)],
            {**env, "LOADGEN_HOST": fe_addr}, log(f"load{li}"), li))

    # --- resource sampling during the measurement window (background thread) ---
    res_snaps = []
    res_thread = None
    if RESSAMPLE and RESSAMPLE_N > 0:
        import threading

        def _sampler_loop():
            time.sleep(min(WINDOW_SECS + 2, duration))
            span = max(duration - 2, 1)
            step = span / max(RESSAMPLE_N, 1)
            for k in range(RESSAMPLE_N):
                try:
                    res_snaps.extend(res_snapshot(lr, transport, idx, f"s{k}"))
                except Exception:
                    pass
                if k < RESSAMPLE_N - 1:
                    time.sleep(step)

        res_thread = threading.Thread(target=_sampler_loop, daemon=True)
        res_thread.start()

    # --- let the consumers run for the measurement duration, then stop ---
    # The counters emit per-window JSON to each component's log as they run.
    print(f"    [cell {idx}] running for {duration}s "
          f"(s={s} routers ports={fe_ports}, replica_sync={replica_sync}) ...",
          flush=True)
    time.sleep(duration)

    if res_thread is not None:
        res_thread.join(timeout=40)

    # Stop loadgen first (stops the event source), then the consumers (so they
    # flush their final counter line on cancellation), then everything.
    for lg in lgs:
        try:
            lg.terminate()
        except Exception:
            pass
    time.sleep(1)
    for lg in lgs:
        try:
            if lg.poll() is None:
                lg.kill()
        except Exception:
            pass

    # Graceful stop of consumers so the Rust RecvCounter::emit_final() / FPM
    # final line lands in the logs before we read them.
    teardown(procs)
    time.sleep(2)

    # --- scrape the receive-side counters from each component's log ---
    # kv-events (#1) and active_sequences (#3): both counted inside the s
    # frontend processes (one log each, two site labels per log).
    fe_counts = []
    for (_pp, i, port) in fe_procs:
        kv = scrape_counter_site(log(f"fe{i}"), "kv-events")
        aseq = scrape_counter_site(log(f"fe{i}"), "active_sequences")
        fe_counts.append({"replica": i, "port": port, "kv_events": kv,
                          "active_sequences": aseq})
    # FPM (#2): the receiver's stdout/stderr log.
    fpm_counts = scrape_counter_site(fpm_log, "fpm")

    res_summary = summarize_res(res_snaps) if RESSAMPLE else {}
    if RESSAMPLE:
        resp = f"{outdir}/{transport}_p{p}_s{s}_t{trial}_res.json"
        try:
            with open(resp, "w") as f:
                json.dump({"mode": transport, "n_workers": p, "n_subs": s,
                           "trial": trial, "nnodes": lr.nnodes,
                           "summary": res_summary, "snapshots": res_snaps}, f, indent=2)
        except Exception as e:
            print(f"    [res] write failed: {e}", flush=True)

    # Aggregate kv-events across replicas (each replica independently consumes
    # the full worker event stream, so we report per-replica + the mean).
    kv_rows = [c["kv_events"] for c in fe_counts]
    aseq_rows = [c["active_sequences"] for c in fe_counts]

    def agg(rows, key):
        vals = [r.get(key, 0) for r in rows if r]
        return vals

    row = {
        "mode": transport, "n_workers": p, "n_subs": s, "trial": trial,
        "launcher": lr.name, "nnodes": lr.nnodes, "replica_sync": replica_sync,
        "model": model,
        # Per-event-type results
        "kv_events": {
            "per_replica": kv_rows,
            "received_total": sum(agg(kv_rows, "received")),
            "gaps_total": sum(agg(kv_rows, "gaps")),
            "drop_rate_max": max(agg(kv_rows, "drop_rate") or [0.0]),
            "events_per_sec_mean": (
                sum(agg(kv_rows, "events_per_sec")) / len(kv_rows)
                if kv_rows else 0.0),
        },
        "fpm": fpm_counts,
        "active_sequences": {
            "per_replica": aseq_rows,
            "received_total": sum(agg(aseq_rows, "received")),
            "gaps_total": sum(agg(aseq_rows, "gaps")),
            "drop_rate_max": max(agg(aseq_rows, "drop_rate") or [0.0]),
            "events_per_sec_mean": (
                sum(agg(aseq_rows, "events_per_sec")) / len(aseq_rows)
                if aseq_rows else 0.0),
        },
    }
    if RESSAMPLE and res_summary:
        row["res"] = res_summary
    return [row]


def scrape_counter_site(logpath, site):
    """Scrape only the dis2172_recv lines whose site == `site` from a log.

    A single frontend process emits BOTH kv-events and active_sequences lines
    (two RecvCounters in one process), so we filter by the embedded site label.
    """
    windows = []
    final = None
    try:
        with open(logpath, errors="replace") as f:
            for line in f:
                m = _RECV_RE.search(line)
                if not m:
                    continue
                try:
                    obj = json.loads(m.group(0))
                except Exception:
                    continue
                if obj.get("site") != site:
                    continue
                if obj.get("dis2172_recv") == "window":
                    windows.append(obj)
                elif obj.get("dis2172_recv") == "final":
                    final = obj
    except FileNotFoundError:
        pass
    if final is None and windows:
        last = windows[-1]
        final = {"received": last.get("total_received", 0),
                 "gaps": last.get("total_gaps", 0),
                 "n_publishers": last.get("n_publishers", 0)}
    if final is None:
        final = {"received": 0, "gaps": 0, "n_publishers": 0}
    sent_est = final.get("received", 0) + final.get("gaps", 0)
    drop_rate = (final["gaps"] / sent_est) if sent_est > 0 else 0.0
    rates = [w.get("events_per_sec", 0.0) for w in windows]
    if len(rates) >= 3:
        rates = rates[1:-1]
    mean_rate = (sum(rates) / len(rates)) if rates else 0.0
    return {
        "received": final.get("received", 0),
        "gaps": final.get("gaps", 0),
        "sent_est": sent_est,
        "drop_rate": drop_rate,
        "n_publishers": final.get("n_publishers", 0),
        "n_windows": len(windows),
        "events_per_sec": mean_rate,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--launcher", choices=["local", "slurm"], default="local")
    ap.add_argument("--transports", default="nats,zmq")
    ap.add_argument("--workers", default="1,4,16,64", help="p sweep (publishers)")
    ap.add_argument("--subs", default="1,2,4,8",
                    help="s sweep = number of kv-mode router replicas "
                         "(s>=2 enables --router-replica-sync -> active_sequences)")
    ap.add_argument("--events", default="kv-events,fpm,active_sequences",
                    help="which event types to report (all flow regardless; this "
                         "is documentation only)")
    ap.add_argument("--gpus-per-node", type=int, default=0)
    ap.add_argument("--duration", type=int, default=40)
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--loadgen-procs", type=int, default=4)
    ap.add_argument("--speedup-ratio", default="10")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--infra-addr", default=None)
    ap.add_argument("--out", default=f"{HERE}/results/run")
    a = ap.parse_args()

    lr = make_launcher(a.launcher)
    if hasattr(lr, "prepare_containers"):
        lr.prepare_containers(dict(os.environ))
    infra_addr = a.infra_addr or lr.addr(0)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    Path(LOGDIR).mkdir(parents=True, exist_ok=True)
    print(f"[bench] per-task logs -> {LOGDIR} (node-local); data -> {a.out}", flush=True)

    transports = a.transports.split(",")
    workers = [int(x) for x in a.workers.split(",")]
    subs = [int(x) for x in a.subs.split(",")]
    events = a.events.split(",")

    meta = dict(launcher=lr.name, nodes=lr.nodes, infra_addr=infra_addr,
                transports=transports, workers=workers, subs=subs, events=events,
                gpus_per_node=a.gpus_per_node, duration=a.duration, trials=a.trials,
                concurrency=a.concurrency, loadgen_procs=a.loadgen_procs,
                window_secs=WINDOW_SECS, ressample=RESSAMPLE)
    with open(f"{a.out}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[bench] launcher={lr.name} nodes={lr.nnodes} infra={infra_addr}", flush=True)

    all_rows, idx = [], 0
    for trial in range(1, a.trials + 1):
        for t in transports:
            for p in workers:
                for s in subs:
                    idx += 1
                    print(f"[bench] trial={trial} {t} p={p} s={s}", flush=True)
                    rows = run_cell(lr, t, p, s, events, a.gpus_per_node, a.duration,
                                    a.concurrency, a.loadgen_procs, a.out, idx, trial,
                                    infra_addr, a.speedup_ratio)
                    for r in rows:
                        if r.get("error"):
                            print(f"    ERROR: {r['error']}", flush=True)
                            continue
                        kv = r["kv_events"]
                        fpm = r["fpm"]
                        aseq = r["active_sequences"]
                        print(f"    kv-events: recv={kv['received_total']} "
                              f"gaps={kv['gaps_total']} "
                              f"rate~{kv['events_per_sec_mean']:.1f}/s | "
                              f"fpm: recv={fpm['received']} gaps={fpm['gaps']} "
                              f"rate~{fpm['events_per_sec']:.1f}/s | "
                              f"active_seq: recv={aseq['received_total']} "
                              f"gaps={aseq['gaps_total']} "
                              f"rate~{aseq['events_per_sec_mean']:.2f}/s",
                              flush=True)
                    all_rows.extend(rows)
                    with open(f"{a.out}/summary.json", "w") as f:
                        json.dump(all_rows, f, indent=2)
                    time.sleep(3)

    print(f"[bench] DONE -> {a.out}/summary.json ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
