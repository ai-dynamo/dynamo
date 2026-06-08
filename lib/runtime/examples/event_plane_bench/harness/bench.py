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

# How long to wait for the frontend to register the model. The first srun into a
# pyxis container may have to extract the 24GB .sqsh rootfs, which can exceed the
# old hardcoded ~120s. At high mocker density (e.g. p=72 over few nodes -> 18
# mockers/node) the frontend also takes longer to see all workers register, so
# 300s was too tight and p=72-zmq timed out (DIS-2172). Default bumped to 600s;
# override with DYN_BENCH_MODEL_WAIT.
MODEL_WAIT = int(os.environ.get("DYN_BENCH_MODEL_WAIT", "600"))
# Extra slack (seconds) folded into the subscriber-wait timeout AND the loadgen
# --duration/wait, so a still-extracting container doesn't get its sub/loadgen
# killed before it has even started. Override with DYN_BENCH_EXTRACT_PAD.
EXTRACT_PAD = int(os.environ.get("DYN_BENCH_EXTRACT_PAD", "220"))
# Persistent pyxis container name: extract the image ONCE per node into this
# named container, then reuse it for every subsequent srun (avoids re-extracting
# the 24GB squashfs per task). Override / disable via DYN_BENCH_CONTAINER_NAME.
CONTAINER_NAME = os.environ.get("DYN_BENCH_CONTAINER_NAME", "dis2172bench")

# Per-task stdout/stderr logs (frontend/mocker/loadgen/subscriber). These are
# debug-only chatter (~1320 files for a full sweep) and MUST NOT land in the OUT
# dir, which lives on the (5G-quota) NFS home -- a full sweep overflows it and
# OSError(28) kills the summary.json write (DIS-2172 cell-83 crash). Write them
# to a NODE-LOCAL dir instead (the orchestrator opens them on node0's local fs);
# only summary.json + meta.json + the per-cell *_sub*.json DATA stay in OUT.
# Default /tmp/dis2172-logs-$SLURM_JOB_ID/, override with DYN_BENCH_LOGDIR.
LOGDIR = os.environ.get(
    "DYN_BENCH_LOGDIR",
    f"/tmp/dis2172-logs-{os.environ.get('SLURM_JOB_ID', os.getpid())}",
)

# CHANGE #3 (DIS-2172): event-plane RESOURCE-COST sampling. During each cell's
# measurement window we take a few lightweight per-node snapshots (established
# TCP count + per-process CPU/RSS/fd for mocker/sub/nats-server/frontend) via
# ressample.py run on the node HOST. This quantifies the ZMQ direct-mesh
# O(p×s) connection blow-up vs NATS O(p+s)-to-broker, and the nats-server
# broker CPU/RSS cost (active during NATS cells, ~idle during ZMQ cells).
# Gate with DYN_BENCH_RESSAMPLE=0 to disable; default ON. Number of snapshots
# per cell via DYN_BENCH_RESSAMPLE_N (default 2). The sampler is stdlib-only and
# runs with the HOST python (NOT the container venv) since it scans host /proc.
RESSAMPLE = os.environ.get("DYN_BENCH_RESSAMPLE", "1") not in ("0", "", "false")
RESSAMPLE_N = int(os.environ.get("DYN_BENCH_RESSAMPLE_N", "2"))
HOST_PY = os.environ.get("HOST_PY", "python3")


# --------------------------------------------------------------------------- #
# Launchers: the ONLY thing that differs between single-node and multi-node.
# --------------------------------------------------------------------------- #
class Launcher:
    name = "base"
    nodes = ["localhost"]

    def popen(self, argv, env, logpath, node_idx=0):
        raise NotImplementedError

    def popen_host(self, argv, env, logpath, node_idx=0):
        """Run argv on the node HOST (never inside the pyxis container).

        Used by the resource sampler so a host-level `ss`/`/proc` scan sees
        every process on the node (enroot runs in the host PID/net namespace)
        and counts ALL of the node's established TCP connections. Defaults to
        the same as popen() for launchers without a container layer (local).
        """
        return self.popen(argv, env, logpath, node_idx)

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
        # Whether the image has been extracted into the persistent named
        # container on each node yet (set by prepare_containers()).
        self._container_ready = False

    def prepare_containers(self, env):
        """Extract the image ONCE per node into a persistent pyxis named container.

        Pyxis re-extracts the (24GB) squashfs on EVERY `--container-image` srun.
        Instead we do a single `--container-image=<sqsh> --container-name=NAME`
        srun per node up front (extracts + names the rootfs), then every later
        task srun's with ONLY `--container-name=NAME` (no --container-image), so
        pyxis attaches to the already-extracted rootfs instead of re-extracting.
        Cuts per-cell wall-clock from ~6min toward ~1-2min.
        """
        img = env.get("DYN_BENCH_IMAGE")
        if not img or not CONTAINER_NAME:
            return  # bare-metal slurm (no pyxis) or persistence disabled
        for node in self.nodes:
            srun = ["srun", "--nodes=1", "--ntasks=1", "--overlap",
                    f"--nodelist={node}",
                    f"--container-image={img}",
                    f"--container-name={CONTAINER_NAME}"]
            mounts = env.get("DYN_BENCH_MOUNTS")
            if mounts:
                srun.append(f"--container-mounts={mounts}")
            # `true` just forces the create+extract; the named container's rootfs
            # persists on the node for subsequent --container-name attaches.
            srun += ["--", "true"]
            print(f"[slurm] extracting image into container '{CONTAINER_NAME}' "
                  f"on {node} ...", flush=True)
            t0 = time.time()
            rc = subprocess.call(srun, env=env)
            print(f"[slurm]   {node}: extract rc={rc} in {time.time()-t0:.0f}s",
                  flush=True)
        self._container_ready = True

    def popen(self, argv, env, logpath, node_idx=0):
        node = self.nodes[node_idx % len(self.nodes)]
        srun = ["srun", "--nodes=1", "--ntasks=1", "--overlap", f"--nodelist={node}"]
        img = env.get("DYN_BENCH_IMAGE")
        if img:  # pyxis container mode (argv uses in-container paths: DYN_BENCH_PY / SUB_BIN)
            if self._container_ready and CONTAINER_NAME:
                # Reuse the already-extracted rootfs (NO --container-image -> no
                # re-extract). Pyxis attaches to the persistent named container.
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
        """srun onto the node HOST with NO --container-* flags.

        The resource sampler must run on the host (not in pyxis) so its `ss`
        sees ALL of the node's established TCP and its /proc scan sees the
        enroot-launched mocker/sub/nats processes (host PID namespace). Uses
        the orchestrator's host python; ressample.py is stdlib-only.
        """
        node = self.nodes[node_idx % len(self.nodes)]
        srun = ["srun", "--nodes=1", "--ntasks=1", "--overlap",
                f"--nodelist={node}", "--"] + argv
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


def res_snapshot(lr, transport, idx, snap_label):
    """Fire ressample.py on EVERY node (host-level), wait, return list of snaps.

    The sampler also echoes its JSON snapshot to stdout; srun streams that back
    to node0's local LOGDIR (`--output`/Popen stdout), so the orchestrator reads
    the snapshots from those echoed logs even though the sampler's --out file
    lives on the (unreadable-from-here) remote node's local fs. Cheap: one ss +
    two /proc passes per node. Returns a list of per-node snapshot dicts
    (best-effort; missing/failed nodes are skipped).
    """
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
    snaps = []
    for (pp, outp, node) in procs:
        try:
            pp.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                pp.kill()
            except Exception:
                pass
    # The sampler echoes its JSON to stdout, which srun --output streams back to
    # node0's local LOGDIR regardless of which node it ran on -> read THAT (the
    # per-node --out file lives on the remote node's local fs, unreadable here).
    for node_idx in range(lr.nnodes):
        node = lr.nodes[node_idx] if hasattr(lr, "nodes") else "localhost"
        logp = f"{LOGDIR}/ressampler_{idx}_{snap_label}_n{node_idx}.log"
        try:
            with open(logp) as f:
                txt = f.read().strip()
            # last non-empty line is the JSON object
            line = [ln for ln in txt.splitlines() if ln.strip().startswith("{")]
            if line:
                snaps.append(json.loads(line[-1]))
        except Exception:
            pass
    return snaps


def summarize_res(snaps):
    """Fold a cell's per-node snapshots into compact summary stats.

    Returns dict with node-level totals/peaks so report.py / a reader can show
    the O(p×s) vs O(p+s) connection blow-up and the per-class CPU/RSS/fd cost
    without re-parsing every snapshot.
    """
    if not snaps:
        return {}
    classes = ("mocker", "sub", "nats", "frontend", "zmq_broker", "loadgen")
    # established_tcp: peak across snapshots, plus per-node breakdown.
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
            "rss_kb_peak": max(rss_vals),
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
def run_cell(lr, transport, p, s, topics, gpus_per_node, duration, warmup,
             conc, lg_procs, outdir, idx, trial, infra_addr, speedup_ratio):
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
        # Per-task stdout -> node-local LOGDIR (NOT the NFS OUT dir; see LOGDIR).
        return f"{LOGDIR}/{name}_{idx}.log"

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

    model = wait_model(fe_addr, HTTP_PORT, MODEL_WAIT)
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
            subs.append((lr.popen([SUB_BIN], senv, f"{LOGDIR}/sub_{idx}_{topic}_{si}.log", 0),
                         outp, topic, si))

    time.sleep(2)
    # A single asyncio loadgen process is CPU-bound and caps at ~100 req/s
    # (-> ~100 events/s) regardless of --concurrency, which is NOT the transport
    # limit. Fan out into `lg_procs` processes (each carrying conc/lg_procs
    # in-flight requests) spread across the allocation, so aggregate request
    # rate -- and thus the event-plane publish rate -- scales toward saturation
    # (DIS-2172 change B). The loadgen httpx pool is also sized to its
    # concurrency so >100 in-flight requests don't block on the default pool.
    per_proc = max(1, conc // lg_procs)
    lgs = []
    for li in range(lg_procs):
        lgs.append(lr.popen(
            [PY, f"{HERE}/loadgen.py", "--port", str(HTTP_PORT), "--model", model,
             "--concurrency", str(per_proc),
             "--duration", str(duration + warmup + 4 + EXTRACT_PAD)],
            {**env, "LOADGEN_HOST": fe_addr}, log(f"load{li}"), li))

    # CHANGE #3: take resource-cost snapshots DURING the measurement window in a
    # background thread, so the foreground sub.wait() is unperturbed. Snapshots
    # are spread across ~[warmup .. warmup+duration] (mid-window). Each fires a
    # cheap host-level ressample.py on every node. Collected after the subs join.
    res_snaps = []
    res_thread = None
    if RESSAMPLE and RESSAMPLE_N > 0:
        import threading

        def _sampler_loop():
            # Wait out warmup, then space N snapshots across the measure window.
            time.sleep(min(warmup + 2, duration + warmup))
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

    for (pp, _o, _t, _s) in subs:
        try:
            pp.wait(timeout=duration + warmup + 60 + EXTRACT_PAD)
        except subprocess.TimeoutExpired:
            pp.kill()
    if res_thread is not None:
        res_thread.join(timeout=40)
    # Subscribers have finished measuring; the loadgens are just the load source
    # (their --duration is padded by EXTRACT_PAD and they are almost always still
    # running here). Tear them all down at once -- do NOT wait 15s on each, which
    # would serialize to lg_procs*15s of dead time per cell.
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

    # CHANGE #3: fold the cell's resource snapshots into a compact summary, write
    # the per-cell *_res.json (raw snapshots + summary), and attach the summary
    # to every row so it lands in summary.json for report.py / readers.
    res_summary = summarize_res(res_snaps) if RESSAMPLE else {}
    if RESSAMPLE:
        resp = f"{outdir}/{transport}_p{p}_s{s}_t{trial}_res.json"
        try:
            with open(resp, "w") as f:
                json.dump({"mode": transport, "n_workers": p, "n_subs": s,
                           "trial": trial, "topics": topics, "nnodes": lr.nnodes,
                           "summary": res_summary, "snapshots": res_snaps}, f, indent=2)
        except Exception as e:
            print(f"    [res] write failed: {e}", flush=True)
        sm = res_summary
        print(f"    [res] est_tcp node_peak={sm.get('established_tcp_node_peak')} "
              f"total_peak={sm.get('established_tcp_total_peak')} "
              f"nats_rss_mb={sm.get('procs',{}).get('nats',{}).get('rss_mb_peak')} "
              f"nats_cpu%={sm.get('procs',{}).get('nats',{}).get('cpu_pct_peak')} "
              f"snaps={sm.get('n_snapshots')}", flush=True)

    rows = []
    for (pp, outp, topic, si) in subs:
        try:
            with open(outp) as f:
                r = json.load(f)
        except Exception:
            r = {"error": "no_output", "topic": topic}
        r.update(mode=transport, n_workers=p, n_subs=s, sub_idx=si, trial=trial,
                 launcher=lr.name, clock=lr.clock, nnodes=lr.nnodes)
        if RESSAMPLE and res_summary:
            r["res"] = res_summary
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
    ap.add_argument("--concurrency", type=int, default=256,
                    help="loadgen in-flight requests; HIGH (256) drives the "
                         "transport toward saturation (loadgen rate was the "
                         "bottleneck at low concurrency, masking ZMQ vs NATS)")
    ap.add_argument("--loadgen-procs", type=int, default=4,
                    help="parallel loadgen processes per cell, spread across "
                         "nodes. A single asyncio loadgen caps ~100 req/s "
                         "(CPU-bound), so fan out to scale event rate >>100/s; "
                         "concurrency is split across them (DIS-2172 change B)")
    ap.add_argument("--speedup-ratio", default="10",
                    help="mocker exec speedup; 0 = unthrottled (saturation test, no-PTP throughput mode)")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--infra-addr", default=None,
                    help="etcd/nats host (default: node0 address)")
    ap.add_argument("--out", default=f"{HERE}/results/run")
    a = ap.parse_args()

    lr = make_launcher(a.launcher)
    # Extract the image once per node into a persistent named container BEFORE
    # the sweep, so individual task srun's reuse it instead of re-extracting.
    if hasattr(lr, "prepare_containers"):
        lr.prepare_containers(dict(os.environ))
    infra_addr = a.infra_addr or lr.addr(0)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    # Per-task stdout/stderr go here (node-local), NOT into the NFS-home OUT dir.
    Path(LOGDIR).mkdir(parents=True, exist_ok=True)
    print(f"[bench] per-task logs -> {LOGDIR} (node-local); data + summary -> {a.out}",
          flush=True)
    transports = a.transports.split(",")
    workers = [int(x) for x in a.workers.split(",")]
    subs = [int(x) for x in a.subs.split(",")]
    topics = a.topics.split(",")

    meta = dict(launcher=lr.name, nodes=lr.nodes, clock=lr.clock, infra_addr=infra_addr,
                transports=transports, workers=workers, subs=subs, topics=topics,
                gpus_per_node=a.gpus_per_node, duration=a.duration, trials=a.trials,
                concurrency=a.concurrency, loadgen_procs=a.loadgen_procs,
                ressample=RESSAMPLE, ressample_n=RESSAMPLE_N)
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
                                    a.warmup, a.concurrency, a.loadgen_procs, a.out, idx,
                                    trial, infra_addr, a.speedup_ratio)
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
