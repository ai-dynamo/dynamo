#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Scale-up timing benchmark: cold start vs. CRIU checkpoint restore via DGDR.
#
# Deploys two DynamoGraphDeploymentRequests in parallel and measures decode worker
# scale-up time (container-start → Ready):
#
#   Cold DGDR:     plain cold start — checkpoint disabled via DGD service override so
#                  the operator skips checkpoint/restore regardless of cluster support.
#
#   Snapshot DGDR: CRIU checkpoint/restore — the profiler always injects checkpoint
#                  (mode=Auto) into generated DGDs.  The operator creates a
#                  DynamoCheckpoint CR from the first running pod; all subsequent
#                  scale-ups restore from it instead of cold-starting.  If no
#                  snapshot-agent is running the operator falls back to cold start.
#
# Usage:
#   python3 checkpoint_scale_timing.py --image nvcr.io/.../vllm-runtime:abc123
#
#   # Multiple TP sizes:
#   python3 checkpoint_scale_timing.py --tp-sizes 1,4 --image ...
#
# Required cluster resources:
#   PVC     hf-cache-pvc      HuggingFace model cache (created automatically if absent)
#   Secret  hf-token-secret   HUGGING_FACE_HUB_TOKEN
#   Secret  ngc-secret        Image pull secret

import argparse
import datetime
import json
import os
import shlex
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults (all overridable via env vars or CLI flags)
# ---------------------------------------------------------------------------

DEFAULTS = {
    "namespace": os.getenv("NAMESPACE", ""),  # empty → auto-detect
    "model": os.getenv("MODEL", "meta-llama/Meta-Llama-3.1-8B"),
    "tp_sizes": os.getenv("TP_SIZES", "1"),
    "gpu_count": int(os.getenv("GPU_COUNT", "1")),
    "max_model_len": int(
        os.getenv("MAX_MODEL_LEN", "0")
    ),  # 0 = auto (next_power_of_2(isl+osl))
    "isl": int(os.getenv("ISL", "3000")),
    "osl": int(os.getenv("OSL", "150")),
    "hf_cache_pvc": os.getenv("HF_CACHE_PVC", "hf-cache-pvc"),
    "hf_secret": os.getenv("HF_SECRET", "hf-token-secret"),
    "ngc_secret": os.getenv("NGC_SECRET", "ngc-secret"),
    "dgdr_name": os.getenv("DGDR_NAME", "scale-bench"),
    "runtime_class": os.getenv("RUNTIME_CLASS", ""),
    "search_strategy": os.getenv("SEARCH_STRATEGY", "rapid"),
    "profiling_timeout": int(os.getenv("PROFILING_TIMEOUT", "3600")),
    "deploy_timeout": int(os.getenv("DEPLOY_TIMEOUT", "900")),
    "scale_down_timeout": int(os.getenv("SCALE_DOWN_TIMEOUT", "600")),
    "prometheus_endpoint": os.getenv("PROMETHEUS_ENDPOINT", ""),
    "workdir": os.getenv("WORKDIR", ""),
    "hf_cache_pvc_size": os.getenv("HF_CACHE_PVC_SIZE", "300Gi"),
    "hf_cache_pvc_access_mode": os.getenv("HF_CACHE_PVC_ACCESS_MODE", "ReadWriteMany"),
    "hf_cache_storage_class": os.getenv("HF_CACHE_STORAGE_CLASS", "vast"),
    "nats_storage_class": os.getenv("NATS_STORAGE_CLASS", "vast"),
    "operator_image": os.getenv("OPERATOR_IMAGE", ""),
    "platform_release": os.getenv("PLATFORM_RELEASE", "dynamo-bench"),
    "operator_timeout": int(os.getenv("OPERATOR_TIMEOUT", "600")),
    "gpu_sku": os.getenv("GPU_SKU", ""),
    "gpus_per_node": int(os.getenv("GPUS_PER_NODE", "0")),
    "total_gpus": int(os.getenv("TOTAL_GPUS", "0")),
    "vram_mb": int(os.getenv("VRAM_MB", "0")),
    "snapshot_timeout": int(os.getenv("SNAPSHOT_TIMEOUT", "900")),
    "snapshot_agent_image": os.getenv(
        "SNAPSHOT_AGENT_IMAGE",
        "nvcr.io/nvidian/dynamo-dev/schwinns:snapshot-agent-59c8787696-parrestore-tp8-20260401-161420",
    ),
    "snapshot_release": os.getenv("SNAPSHOT_RELEASE", "dynamo-snapshot"),
    "snapshot_pvc_size": os.getenv("SNAPSHOT_PVC_SIZE", "1Ti"),
    "snapshot_pvc_storage_class": os.getenv("SNAPSHOT_PVC_STORAGE_CLASS", "vast"),
}

_SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent.parent.parent
_PLATFORM_CHART = _SCRIPT_DIR / "deploy" / "helm" / "charts" / "platform"
_SNAPSHOT_CHART = _SCRIPT_DIR / "deploy" / "helm" / "charts" / "snapshot"


# ---------------------------------------------------------------------------
# Logging — thread-aware (parallel mode routes to per-phase log files)
# ---------------------------------------------------------------------------

_thread_local = threading.local()


def _ts() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def _log_line(prefix: str, msg: str, err: bool = False):
    """Write a log line, routing to thread-local log file when in parallel mode."""
    line = f"{prefix}[{_ts()}] {msg}"
    logfile = getattr(_thread_local, "logfile", None)
    if logfile is not None:
        logfile.write(line + "\n")
        logfile.flush()
        # Update live-table status for this phase thread.
        status_cb = getattr(_thread_local, "status_cb", None)
        if status_cb is not None:
            status_cb(msg)
    else:
        print(line, file=sys.stderr if err else sys.stdout, flush=True)


def step(msg):
    _log_line("\n>>> ", msg)


def info(msg):
    _log_line("    ", msg)


def ok(msg):
    _log_line(" OK ", msg)


def warn(msg):
    _log_line(" !! ", msg, err=True)


def die(msg):
    print(f"[{_ts()}] FATAL: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Live status table (parallel benchmark mode, no external deps)
# ---------------------------------------------------------------------------


class _LiveTable:
    """ANSI live-updating status table for parallel benchmark phases.

    Each phase thread calls ``update(phase, msg)`` to update its status line.
    A background display thread redraws the table every second using cursor-up
    ANSI escapes.  The display is stopped via ``stop()`` which does a final draw.

    Output is flushed directly to sys.stdout; per-phase detailed logs go to
    individual log files (see ``run_parallel_for_tp``).
    """

    _ICON = {"waiting": " ", "running": "▶", "done": "✓", "error": "✗"}

    def __init__(self, phases: list[str]):
        self._phases = phases
        self._lock = threading.Lock()
        self._rows: dict[str, dict] = {
            p: {"state": "waiting", "msg": "queued", "elapsed": 0.0} for p in phases
        }
        self._start = time.monotonic()
        self._n_printed = 0
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ── public API ────────────────────────────────────────────────────────

    def update(self, phase: str, msg: str, state: str = "running"):
        with self._lock:
            r = self._rows.setdefault(phase, {})
            r["state"] = state
            r["msg"] = msg
            r["elapsed"] = time.monotonic() - self._start

    def done(self, phase: str, msg: str = "done"):
        self.update(phase, msg, state="done")

    def error(self, phase: str, msg: str):
        self.update(phase, msg, state="error")

    def stop(self):
        self._stop_evt.set()
        self._thread.join(timeout=3)
        self._draw(final=True)

    # ── internal ──────────────────────────────────────────────────────────

    def _draw(self, final: bool = False):
        with self._lock:
            rows = {p: dict(v) for p, v in self._rows.items()}
        elapsed_total = time.monotonic() - self._start
        m, s = divmod(int(elapsed_total), 60)
        lines = [
            f"\033[1m  ── BENCHMARK  T+{m:02d}:{s:02d} ──────────────────────────────────────────\033[0m",
        ]
        for ph in self._phases:
            r = rows.get(ph, {})
            icon = self._ICON.get(r.get("state", "waiting"), "?")
            msg = (r.get("msg") or "")[:46]
            el = r.get("elapsed", 0.0)
            # Colour: green=done, yellow=running, red=error, dim=waiting
            colour = ""
            if r.get("state") == "done":
                colour = "\033[32m"
            elif r.get("state") == "running":
                colour = "\033[33m"
            elif r.get("state") == "error":
                colour = "\033[31m"
            else:
                colour = "\033[2m"
            reset = "\033[0m"
            lines.append(f"  {colour}{icon} {ph:<12}  {msg:<46}  {el:>6.0f}s{reset}")
        lines.append("  \033[2m" + "─" * 68 + "\033[0m")

        out = sys.stdout
        if self._n_printed > 0:
            # Move cursor up to overwrite previous draw
            out.write(f"\033[{self._n_printed}A")
        for line in lines:
            out.write(f"\r\033[K{line}\n")
        if final:
            out.write("\n")
        out.flush()
        self._n_printed = len(lines)

    def _loop(self):
        while not self._stop_evt.wait(1.0):
            self._draw()


# ---------------------------------------------------------------------------
# kubectl helpers
# ---------------------------------------------------------------------------


def _kubectl(*args, check=True, capture=False):
    cmd = ["kubectl", *args]
    if capture:
        return subprocess.run(cmd, check=check, capture_output=True, text=True)
    return subprocess.run(cmd, check=check)


def kube_apply(path, ns):
    _kubectl("apply", "-f", path, "-n", ns)


def kube_delete_dgdr(name, ns):
    _kubectl("delete", "dgdr", name, "-n", ns, "--ignore-not-found=true", check=False)


def kube_get(resource, name, ns) -> dict:
    r = _kubectl(
        "get", resource, name, "-n", ns, "-o", "json", capture=True, check=False
    )
    return json.loads(r.stdout) if r.returncode == 0 else {}


def kube_list(resource, ns, label="") -> list:
    args = ["get", resource, "-n", ns, "-o", "json"]
    if label:
        args += ["-l", label]
    r = _kubectl(*args, capture=True, check=False)
    if r.returncode != 0:
        return []
    return json.loads(r.stdout).get("items", [])


def kube_logs(pod, ns, tail=50) -> str:
    r = _kubectl("logs", pod, "-n", ns, f"--tail={tail}", capture=True, check=False)
    return r.stdout if r.returncode == 0 else ""


def _dgd_patch_replicas(dgd_name: str, ns: str, svc_name: str, replicas: int) -> None:
    """Patch a DGD service's replica count."""
    patch = {"spec": {"services": {svc_name: {"replicas": replicas}}}}
    patch_path = Path(tempfile.gettempdir()) / "gms_replicas_patch.json"
    patch_path.write_text(json.dumps(patch))
    subprocess.run(
        [
            "kubectl",
            "patch",
            "dynamographdeployment",
            dgd_name,
            "-n",
            ns,
            "--type=merge",
            f"--patch-file={patch_path}",
        ],
        check=True,
    )
    info(f"DGD {dgd_name!r} service {svc_name!r} replicas → {replicas}")


def _wait_for_decode_pods_gone(ns: str, worker_label: str, timeout: int = 300) -> None:
    """Wait until no pods matching worker_label exist (Running or otherwise)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pods = kube_list("pods", ns, worker_label)
        if not pods:
            return
        names = [p["metadata"]["name"] for p in pods]
        info(f"Waiting for decode pods to terminate: {names}")
        time.sleep(5)
    raise RuntimeError(
        f"Timed out ({timeout}s) waiting for decode pods to terminate in namespace {ns!r}"
    )


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------


def detect_gpu_hardware() -> tuple[str, int, int, int]:
    """Auto-detect GPU hardware from cluster nodes.

    Returns (gpu_sku, gpus_per_node, total_gpus, vram_mb).
    """
    r = _kubectl(
        "get",
        "nodes",
        "-l",
        "nvidia.com/gpu.present=true",
        "-o",
        "jsonpath={range .items[*]}{.metadata.labels.nvidia\\.com/gpu\\.product}"
        " {.metadata.labels.nvidia\\.com/gpu\\.count}"
        " {.metadata.labels.nvidia\\.com/gpu\\.memory}\\n{end}",
        capture=True,
        check=False,
    )
    if r.returncode != 0 or not r.stdout.strip():
        warn(
            "Could not detect GPU hardware from nodes; pass --gpu-sku / --gpus-per-node / --vram-mb"
        )
        return "", 1, 1, 0

    total_gpus = 0
    gpu_sku = ""
    gpus_per_node = 0
    vram_mb = 0

    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        product, count_str, mem_str = parts[0], parts[1], parts[2]
        count = int(count_str) if count_str.isdigit() else 1
        mem = int(mem_str) if mem_str.isdigit() else 0
        total_gpus += count
        if not gpu_sku:
            p = product.upper()
            if "GB200" in p:
                gpu_sku = "gb200_sxm"
            elif "H200" in p:
                gpu_sku = "h200_sxm"
            elif "H100" in p:
                gpu_sku = "h100_sxm"
            elif "B200" in p:
                gpu_sku = "b200_sxm"
            elif "A100" in p:
                gpu_sku = "a100_sxm"
            elif "L40S" in p:
                gpu_sku = "l40s"
            else:
                warn(f"Unknown GPU product {product!r} — pass --gpu-sku explicitly")
                gpu_sku = product.lower().replace("-", "_").replace("nvidia_", "")
            gpus_per_node = count
            vram_mb = mem

    info(
        f"Auto-detected GPU hardware: sku={gpu_sku!r}  gpus_per_node={gpus_per_node}"
        f"  total={total_gpus}  vram={vram_mb}MiB"
    )
    return gpu_sku, gpus_per_node, total_gpus, vram_mb


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def cleanup_namespace(cfg: dict):
    """Remove all benchmark-related resources (run at start for a clean slate)."""
    ns = cfg["namespace"]
    release = cfg["platform_release"]

    step(f"Cleaning namespace {ns!r} ...")

    # 1. CRs first
    kube_delete_dgdr(cfg["dgdr_name"], ns)
    # Also clean up any stale -snap DGDR from older benchmark runs
    kube_delete_dgdr(f"{cfg['dgdr_name']}-snap", ns)
    for resource in (
        "dynamographdeployments",
        "dynamocheckpoints",
        "dynamocomponentdeployments",
    ):
        _kubectl(
            "delete",
            resource,
            "--all",
            "-n",
            ns,
            "--ignore-not-found=true",
            check=False,
        )
    _kubectl(
        "wait",
        "--for=delete",
        "dynamocomponentdeployments",
        "--all",
        "-n",
        ns,
        "--timeout=30s",
        check=False,
    )

    # 2. Force-delete dynamo worker pods
    _kubectl(
        "delete",
        "pods",
        "-n",
        ns,
        "-l",
        "nvidia.com/dynamo-graph-deployment-name",
        "--force",
        "--grace-period=0",
        "--ignore-not-found=true",
        check=False,
    )
    _kubectl(
        "wait",
        "--for=delete",
        "pod",
        "-l",
        "nvidia.com/dynamo-graph-deployment-name",
        "-n",
        ns,
        "--timeout=60s",
        check=False,
    )

    # 3a. Remove any stale manually-deployed snapshot-agent DaemonSet/ConfigMap that
    #     may have been left over from a previous non-Helm deployment.  The Helm chart
    #     uses a release-prefixed name (e.g. "dynamo-snapshot-agent"), so a plain
    #     "snapshot-agent" DaemonSet is always stale and safe to delete.
    _kubectl(
        "delete",
        "daemonset",
        "snapshot-agent",
        "-n",
        ns,
        "--ignore-not-found=true",
        check=False,
    )
    _kubectl(
        "delete",
        "configmap",
        "snapshot-agent-config",
        "-n",
        ns,
        "--ignore-not-found=true",
        check=False,
    )

    # 3. Helm releases (uninstalls operator + NATS + snapshot) — skipped with --keep-operator
    if not cfg.get("keep_operator"):
        for rel in (
            release,
            "dynamo-platform",
            cfg.get("snapshot_release", "dynamo-snapshot"),
        ):
            r = subprocess.run(
                ["helm", "uninstall", rel, "-n", ns, "--ignore-not-found"],
                capture_output=False,
            )
            if r.returncode != 0:
                warn(f"helm uninstall {rel!r} exited {r.returncode}")

    ok("Namespace clean")


# ---------------------------------------------------------------------------
# PVCs
# ---------------------------------------------------------------------------


def ensure_pvc(
    name: str, ns: str, size: str, access_mode: str, storage_class: str
) -> None:
    existing = kube_get("pvc", name, ns)
    if existing:
        phase = existing.get("status", {}).get("phase", "Unknown")
        info(f"PVC {name!r} already exists (phase={phase}) — skipping creation")
        return

    sc_line = f"\n  storageClassName: {storage_class}" if storage_class else ""
    manifest = textwrap.dedent(
        f"""\
        apiVersion: v1
        kind: PersistentVolumeClaim
        metadata:
          name: {name}
          namespace: {ns}
        spec:
          accessModes:
            - {access_mode}
          resources:
            requests:
              storage: {size}{sc_line}
    """
    )
    r = subprocess.run(
        ["kubectl", "apply", "-f", "-", "-n", ns],
        input=manifest,
        text=True,
        capture_output=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"Failed to create PVC {name!r}: {r.stderr.strip()}")
    ok(f"Created PVC {name!r}  ({size}  {access_mode})")


def ensure_pvcs(cfg: dict):
    step("Ensuring PVCs exist")
    ns = cfg["namespace"]
    ensure_pvc(
        cfg["hf_cache_pvc"],
        ns,
        cfg["hf_cache_pvc_size"],
        cfg["hf_cache_pvc_access_mode"],
        cfg["hf_cache_storage_class"],
    )


# ---------------------------------------------------------------------------
# Platform deployment
# ---------------------------------------------------------------------------


def _purge_operator_owned_resources(ns: str) -> None:
    """Delete resources created by the operator that Helm would try to adopt.

    The operator stamps resources with ``managed-by: dynamo-operator``.  If those
    resources already exist when Helm tries to install, Helm aborts with an
    ownership conflict.  Deleting them first lets Helm recreate them cleanly.
    """
    # Resource types known to be created by the operator during profiling / normal
    # operation that collide with Helm-managed resources in the platform chart.
    resource_types = [
        "serviceaccount",
        "role",
        "rolebinding",
        "clusterrole",
        "clusterrolebinding",
    ]
    for rtype in resource_types:
        _kubectl(
            "delete",
            rtype,
            "-l",
            "app.kubernetes.io/managed-by=dynamo-operator",
            "-n",
            ns,
            "--ignore-not-found=true",
            check=False,
        )


def deploy_platform(cfg: dict) -> str:
    ns = cfg["namespace"]
    release = cfg["platform_release"]
    operator_image = cfg["operator_image"]
    ngc_secret = cfg["ngc_secret"]
    nats_storage_class = cfg["nats_storage_class"]
    timeout = cfg["operator_timeout"]

    if not _PLATFORM_CHART.exists():
        raise RuntimeError(f"Platform chart not found at {_PLATFORM_CHART}.")

    # Delete any operator-owned resources that Helm would try to manage.
    # These are created by the operator during normal operation (e.g. profiling
    # jobs) and carry managed-by: dynamo-operator, which blocks Helm adoption.
    _purge_operator_owned_resources(ns)

    repo, tag = _parse_image(operator_image)

    step(f"Deploying Dynamo platform to namespace {ns!r} (release={release!r}) ...")
    # No --wait: the platform chart may include components that require GPUs and
    # will never become Ready.  We wait explicitly for the operator pod below.
    helm_cmd = [
        "helm",
        "upgrade",
        "--install",
        release,
        str(_PLATFORM_CHART),
        "--namespace",
        ns,
        "--create-namespace",
        "--set",
        f"dynamo-operator.controllerManager.manager.image.repository={repo}",
        "--set",
        f"dynamo-operator.controllerManager.manager.image.tag={tag}",
        "--set",
        "dynamo-operator.controllerManager.manager.image.pullPolicy=Always",
        "--set",
        "dynamo-operator.namespaceRestriction.enabled=true",
        "--set",
        "dynamo-operator.upgradeCRD=false",
        "--set",
        "dynamo-operator.gpuDiscovery.enabled=false",
        "--set",
        "dynamo-operator.checkpoint.enabled=true",
        "--set",
        "dynamo-operator.checkpoint.storage.pvc.pvcName=snapshot-pvc",
        "--set",
        "dynamo-operator.checkpoint.storage.pvc.basePath=/checkpoints",
    ]
    if ngc_secret:
        helm_cmd += ["--set", f"dynamo-operator.imagePullSecrets[0].name={ngc_secret}"]
    if nats_storage_class:
        helm_cmd += [
            "--set",
            f"nats.config.jetstream.fileStore.pvc.storageClassName={nats_storage_class}",
        ]

    result = subprocess.run(helm_cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"helm upgrade --install failed (exit {result.returncode})")
    ok(f"Platform chart installed: release={release!r}  namespace={ns!r}")

    _apply_crds()
    _wait_for_operator(ns, timeout)
    if cfg.get("snapshot_agent_image"):
        _deploy_snapshot_chart(
            ns,
            cfg["snapshot_agent_image"],
            ngc_secret,
            cfg["snapshot_release"],
            cfg["snapshot_pvc_size"],
            cfg["snapshot_pvc_storage_class"],
        )
    else:
        warn(
            "--snapshot-agent-image not set — snapshot-agent will NOT be deployed. "
            "Checkpoint restore requires the snapshot-agent DaemonSet to be running."
        )


def _deploy_snapshot_chart(
    ns: str,
    image: str,
    ngc_secret: str,
    release: str,
    pvc_size: str,
    storage_class: str,
) -> None:
    """Deploy the snapshot Helm chart (snapshot-pvc + snapshot-agent DaemonSet).

    The chart creates the checkpoint storage PVC and the snapshot-agent DaemonSet
    which handles CRIU checkpoint/restore on every GPU node.
    """
    if not _SNAPSHOT_CHART.exists():
        raise RuntimeError(f"Snapshot chart not found at {_SNAPSHOT_CHART}.")

    step(f"Deploying snapshot chart to namespace {ns!r} (release={release!r}) ...")

    # If snapshot-pvc survived a previous run (helm.sh/resource-policy=keep), adopt it
    # into the new Helm release so the install doesn't fail with an ownership conflict.
    existing_pvc = kube_get("pvc", "snapshot-pvc", ns)
    if existing_pvc:
        info("Adopting existing snapshot-pvc into Helm release ...")
        _kubectl(
            "label",
            "pvc",
            "snapshot-pvc",
            "-n",
            ns,
            "app.kubernetes.io/managed-by=Helm",
            "--overwrite",
            check=False,
        )
        _kubectl(
            "annotate",
            "pvc",
            "snapshot-pvc",
            "-n",
            ns,
            f"meta.helm.sh/release-name={release}",
            f"meta.helm.sh/release-namespace={ns}",
            "--overwrite",
            check=False,
        )

    repo, tag = _parse_image(image)
    helm_cmd = [
        "helm",
        "upgrade",
        "--install",
        release,
        str(_SNAPSHOT_CHART),
        "--namespace",
        ns,
        "--create-namespace",
        "--set",
        f"daemonset.image.repository={repo}",
        "--set",
        f"daemonset.image.tag={tag}",
        "--set",
        "daemonset.image.pullPolicy=Always",
        "--set",
        f"storage.pvc.size={pvc_size}",
        "--set",
        "storage.pvc.accessMode=ReadWriteMany",
    ]
    if storage_class:
        helm_cmd += ["--set", f"storage.pvc.storageClass={storage_class}"]
    if ngc_secret:
        helm_cmd += ["--set", f"daemonset.imagePullSecrets[0].name={ngc_secret}"]

    result = subprocess.run(helm_cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"helm upgrade --install snapshot chart failed (exit {result.returncode})"
        )
    ok(f"Snapshot chart installed: release={release!r}  namespace={ns!r}")

    # Derive DaemonSet name: Helm fullname logic — if release contains chart name "snapshot",
    # fullname = release; otherwise fullname = "{release}-snapshot". Then DS = "{fullname}-agent".
    ds_fullname = release if "snapshot" in release else f"{release}-snapshot"
    ds_name = f"{ds_fullname}-agent"

    # Wait for at least one agent pod to be Ready on a GPU node.
    result = _kubectl(
        "rollout",
        "status",
        f"daemonset/{ds_name}",
        "-n",
        ns,
        "--timeout=120s",
        check=False,
    )
    if result.returncode != 0:
        _kubectl(
            "get",
            "pods",
            "-l",
            f"app.kubernetes.io/instance={release}",
            "-n",
            ns,
            "-o",
            "wide",
            check=False,
        )
        warn(
            "snapshot-agent DaemonSet rollout did not complete within 120s — continuing anyway."
        )
    else:
        ok("snapshot-agent DaemonSet ready.")

    # Annotate the PVC so helm uninstall does not delete it.  The binaries we
    # stage below are expensive to copy (37 MB nsrestore + criu deps) and the
    # checkpoint data itself must survive cleanup between benchmark runs.
    _kubectl(
        "annotate",
        "pvc",
        "snapshot-pvc",
        "helm.sh/resource-policy=keep",
        "--overwrite",
        "-n",
        ns,
        check=False,
    )

    _stage_restore_binaries(ns, release, ds_name)


def _stage_restore_binaries(ns: str, release: str, ds_name: str) -> None:
    """Copy nsrestore, criu (+ libs), and cuda-checkpoint from an agent pod to the
    shared snapshot PVC at /checkpoints/.

    These binaries must be accessible from within the vLLM container's mount
    namespace during CRIU restore.  The PVC is mounted at /checkpoints/ in both
    the agent and the vLLM worker containers, making it the only path visible in
    both namespaces without modifying the vLLM image.

    Also updates snapshot-agent-config to point binaryPath / nsRestorePath / libDir
    at the PVC-staged paths so that new checkpoints are created with the correct
    paths baked into the manifest (avoiding the need to patch manifests later).
    """
    # Find any ready agent pod (retry for up to 60s in case rollout just completed).
    agent_pod = ""
    for _ in range(6):
        r = subprocess.run(
            [
                "kubectl",
                "get",
                "pods",
                "-n",
                ns,
                "-l",
                f"app.kubernetes.io/instance={release}",
                "--field-selector=status.phase=Running",
                "-o",
                "jsonpath={.items[0].metadata.name}",
            ],
            capture_output=True,
            text=True,
        )
        agent_pod = r.stdout.strip()
        if agent_pod:
            break
        time.sleep(10)
    if not agent_pod:
        warn("No running snapshot-agent pod found — skipping binary staging.")
        return

    # Idempotent: skip if all required binaries are already staged AND nsrestore
    # matches the agent's version (guards against stale binaries from a different agent image).
    chk = _kubectl(
        "exec",
        agent_pod,
        "-n",
        ns,
        "--",
        "sh",
        "-c",
        "test -x /checkpoints/nsrestore && "
        "test -x /checkpoints/criu && "
        "test -x /checkpoints/cuda-checkpoint && "
        "test -x /checkpoints/cuda-checkpoint-helper && "
        "[ \"$(md5sum /checkpoints/nsrestore | cut -d' ' -f1)\" = "
        "\"$(md5sum /usr/local/bin/nsrestore | cut -d' ' -f1)\" ]",
        check=False,
    )
    if chk.returncode == 0:
        ok(
            "Restore binaries already staged to PVC (nsrestore version matches) — skipping copy."
        )
        if _update_agent_config_paths(ns, release):
            # Config changed — restart agents so they pick up the new paths.
            _kubectl(
                "rollout", "restart", f"daemonset/{ds_name}", "-n", ns, check=False
            )
            _kubectl(
                "rollout",
                "status",
                f"daemonset/{ds_name}",
                "-n",
                ns,
                "--timeout=120s",
                check=False,
            )
            ok("snapshot-agent DaemonSet restarted with updated config.")
        return

    step("Staging restore binaries to snapshot PVC (/checkpoints/) ...")

    # nsrestore: statically-linked Go binary, copy directly.
    _kubectl(
        "exec",
        agent_pod,
        "-n",
        ns,
        "--",
        "sh",
        "-c",
        "cp /usr/local/bin/nsrestore /checkpoints/nsrestore && chmod +x /checkpoints/nsrestore",
    )

    # criu: dynamically linked — bundle all shared-library deps alongside it and
    # use a wrapper script that sets LD_LIBRARY_PATH before exec-ing the real binary.
    _kubectl(
        "exec",
        agent_pod,
        "-n",
        ns,
        "--",
        "sh",
        "-c",
        "cp /usr/local/sbin/criu /checkpoints/criu-bin && chmod +x /checkpoints/criu-bin",
    )
    _kubectl(
        "exec",
        agent_pod,
        "-n",
        ns,
        "--",
        "sh",
        "-c",
        "mkdir -p /checkpoints/lib && "
        "ldd /checkpoints/criu-bin | grep -oP '/[^ ]+\\.so[^ ]*' | "
        "xargs -I{} cp -L {} /checkpoints/lib/ 2>/dev/null || true",
    )
    wrapper = '#!/bin/bash\nLD_LIBRARY_PATH=/checkpoints/lib exec /checkpoints/criu-bin "$@"\n'
    _kubectl(
        "exec",
        agent_pod,
        "-n",
        ns,
        "--",
        "sh",
        "-c",
        f"printf {shlex.quote(wrapper)} > /checkpoints/criu && chmod +x /checkpoints/criu",
    )

    # CRIU CUDA plugin (only libc dependency — safe to use from inside container).
    _kubectl(
        "exec",
        agent_pod,
        "-n",
        ns,
        "--",
        "sh",
        "-c",
        "mkdir -p /checkpoints/criu-lib && "
        "cp /usr/local/lib/criu/cuda_plugin.so /checkpoints/criu-lib/",
    )

    # cuda-checkpoint + cuda-checkpoint-helper: stored on PVC so
    # _patch_checkpoint_rootfs_with_cuda_checkpoint can inject both into each
    # checkpoint's rootfs-diff.tar after the checkpoint is taken.
    _kubectl(
        "exec",
        agent_pod,
        "-n",
        ns,
        "--",
        "sh",
        "-c",
        "cp /usr/local/sbin/cuda-checkpoint /checkpoints/cuda-checkpoint && "
        "chmod +x /checkpoints/cuda-checkpoint && "
        "cp /usr/local/bin/cuda-checkpoint-helper /checkpoints/cuda-checkpoint-helper && "
        "chmod +x /checkpoints/cuda-checkpoint-helper",
    )

    ok("Restore binaries staged.")

    _update_agent_config_paths(ns, release)

    # Restart agents to pick up the updated config.
    _kubectl("rollout", "restart", f"daemonset/{ds_name}", "-n", ns, check=False)
    _kubectl(
        "rollout",
        "status",
        f"daemonset/{ds_name}",
        "-n",
        ns,
        "--timeout=120s",
        check=False,
    )
    ok("snapshot-agent DaemonSet restarted with updated config.")


def _update_agent_config_paths(ns: str, release: str) -> bool:
    """Returns True if the ConfigMap was actually modified."""
    """Patch the snapshot-agent ConfigMap to use PVC-staged binary paths.

    This ensures:
    - nsRestorePath → /checkpoints/nsrestore (used by the agent to launch nsrestore
      via nsenter into the container's namespace)
    - binaryPath → /checkpoints/criu (wrapper script; also baked into checkpoint
      manifests so restore uses the same path)
    - libDir → /checkpoints/criu-lib (for cuda_plugin.so; written to criu.conf
      during dump so restore finds the plugin automatically)

    The Helm chart names the ConfigMap "{release}-config" (e.g. dynamo-snapshot-config).
    """
    cm_name = f"{release}-config"
    r = subprocess.run(
        [
            "kubectl",
            "get",
            "configmap",
            cm_name,
            "-n",
            ns,
            "-o",
            "jsonpath={.data.config\\.yaml}",
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        warn(f"{cm_name} ConfigMap not found — skipping config patch.")
        return False

    config_yaml = r.stdout
    updated = config_yaml
    # Handle both quoted and unquoted variants emitted by different chart versions.
    replacements = [
        (
            'nsRestorePath: "/usr/local/bin/nsrestore"',
            "nsRestorePath: /checkpoints/nsrestore",
        ),
        (
            "nsRestorePath: /usr/local/bin/nsrestore",
            "nsRestorePath: /checkpoints/nsrestore",
        ),
        ('binaryPath: "/usr/local/sbin/criu"', "binaryPath: /checkpoints/criu"),
        ("binaryPath: /usr/local/sbin/criu", "binaryPath: /checkpoints/criu"),
        ('libDir: ""', "libDir: /checkpoints/criu-lib"),
    ]
    for old, new in replacements:
        updated = updated.replace(old, new)

    # Ensure storage.basePath is present (required by newer agent images).
    if "storage:" not in updated:
        updated = "storage:\n  basePath: /checkpoints\n\n" + updated

    if updated == config_yaml:
        info(f"{cm_name} already uses /checkpoints/ paths — no update needed.")
        return False

    patch = json.dumps({"data": {"config.yaml": updated}})
    subprocess.run(
        [
            "kubectl",
            "patch",
            "configmap",
            cm_name,
            "-n",
            ns,
            "--type=merge",
            "-p",
            patch,
        ],
        check=True,
    )
    ok(f"{cm_name} updated to use /checkpoints/ paths.")
    return True


def _patch_checkpoint_rootfs_with_cuda_checkpoint(
    ns: str, ckpt_name: str, snapshot_release: str = "dynamo-snapshot"
) -> None:
    """Inject restore binaries into a checkpoint's rootfs-diff.tar.

    The agent enters the target container's mount namespace and runs nsrestore
    from within that namespace. If the vllm-runtime image has a stale (schwinns)
    nsrestore baked in, it will fail on Janelle-format checkpoints. We inject the
    agent's nsrestore into rootfs-diff.tar so it overrides the image-baked version.

    Also injects cuda-checkpoint and cuda-checkpoint-helper, which nsrestore needs
    for the CUDA state restore step.

    All binaries are pre-staged to /checkpoints/ by _stage_restore_binaries.

    ckpt_name is the DynamoCheckpoint CR name, e.g. "checkpoint-3ae84b0cfcfd489e".
    The PVC directory uses just the hash portion after "checkpoint-".
    """
    r = subprocess.run(
        [
            "kubectl",
            "get",
            "pods",
            "-n",
            ns,
            "-l",
            f"app.kubernetes.io/instance={snapshot_release}",
            "--field-selector=status.phase=Running",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ],
        capture_output=True,
        text=True,
    )
    agent_pod = r.stdout.strip()
    if not agent_pod:
        warn("No running snapshot-agent pod — cannot patch checkpoint rootfs-diff.tar.")
        return

    # The PVC directory is named after the nvidia.com/snapshot-checkpoint-id label value
    # (which the script sets to "1"), not the CR hash. Find rootfs-diff.tar by searching.
    find_r = subprocess.run(
        [
            "kubectl",
            "exec",
            agent_pod,
            "-n",
            ns,
            "--",
            "sh",
            "-c",
            "find /checkpoints -name rootfs-diff.tar -maxdepth 4 2>/dev/null",
        ],
        capture_output=True,
        text=True,
    )
    tar_paths = [p.strip() for p in find_r.stdout.strip().splitlines() if p.strip()]
    if not tar_paths:
        warn("No rootfs-diff.tar found on PVC — cannot inject cuda binaries.")
        return

    ckpt_label = ckpt_name[:24]
    for tar_path in tar_paths:
        # Idempotent: skip if already injected (check for nsrestore).
        chk = _kubectl(
            "exec",
            agent_pod,
            "-n",
            ns,
            "--",
            "sh",
            "-c",
            f"tar -tf {tar_path} 2>/dev/null | grep -q 'usr/local/bin/nsrestore'",
            check=False,
        )
        if chk.returncode == 0:
            info(f"nsrestore already in {tar_path} — skipping.")
            continue

        step(f"Injecting restore binaries into {tar_path} ({ckpt_label}) ...")
        # Use extract→add→re-create rather than tar -rf (append) because -rf requires
        # seekable files and fails with "bad fd" on network-backed PVCs (NFS/Ceph).
        # Also injects nsrestore so the agent's version overrides any stale one in the
        # vllm-runtime image (schwinns vs Janelle format incompatibility).
        result = _kubectl(
            "exec",
            agent_pod,
            "-n",
            ns,
            "--",
            "sh",
            "-c",
            f"set -e && "
            f"rm -rf /tmp/ckpt-inject && "
            f"mkdir -p /tmp/ckpt-inject && "
            f"tar -xf {tar_path} -C /tmp/ckpt-inject 2>/dev/null || true && "
            f"mkdir -p /tmp/ckpt-inject/usr/local/sbin /tmp/ckpt-inject/usr/local/bin && "
            f"cp /checkpoints/nsrestore /tmp/ckpt-inject/usr/local/bin/nsrestore && "
            f"chmod +x /tmp/ckpt-inject/usr/local/bin/nsrestore && "
            f"cp /checkpoints/cuda-checkpoint /tmp/ckpt-inject/usr/local/sbin/cuda-checkpoint && "
            f"chmod +x /tmp/ckpt-inject/usr/local/sbin/cuda-checkpoint && "
            f"cp /checkpoints/cuda-checkpoint-helper /tmp/ckpt-inject/usr/local/bin/cuda-checkpoint-helper && "
            f"chmod +x /tmp/ckpt-inject/usr/local/bin/cuda-checkpoint-helper && "
            f"tar -cf {tar_path}.new -C /tmp/ckpt-inject . && "
            f"mv {tar_path}.new {tar_path} && "
            f"rm -rf /tmp/ckpt-inject",
            check=False,
        )
        if result.returncode != 0:
            warn(
                f"Failed to inject restore binaries into {tar_path} (exit {result.returncode})."
            )
        else:
            ok(f"Restore binaries injected into {tar_path}.")


def _apply_crds() -> None:
    """Apply all Dynamo CRDs from the platform chart's CRD directory."""
    crd_dir = _PLATFORM_CHART / "components" / "operator" / "crds"
    if not crd_dir.exists():
        warn(f"CRD directory not found at {crd_dir}, skipping CRD apply.")
        return
    step(f"Applying CRDs from {crd_dir} ...")
    # --server-side avoids the 262144-byte limit on the last-applied-configuration
    # annotation, which the large DGD CRD (embedded pod-spec OpenAPI schema) exceeds.
    # Retry up to 3 times — the API server occasionally returns "unexpected EOF" on
    # large CRDs due to a transient connection issue.
    cmd = ["kubectl", "apply", "--server-side", "--force-conflicts", "-f", str(crd_dir)]
    for attempt in range(1, 4):
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode == 0:
            ok("CRDs applied.")
            return
        if attempt < 3:
            warn(f"CRD apply failed (attempt {attempt}/3), retrying in 5s …")
            import time

            time.sleep(5)
    raise RuntimeError("kubectl apply CRDs failed after 3 attempts")


def _wait_for_operator(ns: str, timeout: int) -> None:
    """Wait for the dynamo-operator controller-manager pod to be Ready."""
    step(f"Waiting for dynamo-operator pod (timeout={timeout}s) ...")
    result = _kubectl(
        "wait",
        "pod",
        "-l",
        "app.kubernetes.io/name=dynamo-operator",
        "--for=condition=Ready",
        f"--timeout={timeout}s",
        "-n",
        ns,
        check=False,
    )
    if result.returncode != 0:
        # Print pod status to help diagnose (ImagePullBackOff, CrashLoopBackOff, etc.)
        _kubectl(
            "get",
            "pod",
            "-l",
            "app.kubernetes.io/name=dynamo-operator",
            "-n",
            ns,
            "-o",
            "wide",
            check=False,
        )
        raise RuntimeError(
            f"dynamo-operator pod did not become Ready within {timeout}s. "
            "Check image pull status above or re-run with --operator-timeout=900."
        )
    ok("Operator ready.")


def _derive_frontend_image(runtime_image: str) -> str:
    """Derive the dynamo-frontend image from a vllm-runtime image reference.

    Replaces the image name component with 'dynamo-frontend', preserving the
    registry path and tag.  Example:
        nvcr.io/nvidian/dynamo-dev/vllm-runtime:dyn-991.1
        → nvcr.io/nvidian/dynamo-dev/dynamo-frontend:dyn-991.1
    """
    slash_idx = runtime_image.rfind("/")
    prefix = runtime_image[: slash_idx + 1] if slash_idx >= 0 else ""
    suffix = runtime_image[slash_idx + 1 :]
    colon_idx = suffix.find(":")
    tag = suffix[colon_idx:] if colon_idx >= 0 else ""
    return f"{prefix}dynamo-frontend{tag}"


def _parse_image(image: str) -> tuple[str, str]:
    if ":" in image.split("/")[-1]:
        repo, tag = image.rsplit(":", 1)
    else:
        repo, tag = image, "latest"
    return repo, tag


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _parse_rfc3339(s: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))


def pod_is_ready(pod: dict) -> bool:
    for c in pod.get("status", {}).get("conditions", []):
        if c.get("type") == "Ready" and c.get("status") == "True":
            return True
    return False


def pod_start_to_ready_seconds(pod: dict) -> float | None:
    ready_at = None
    for c in pod.get("status", {}).get("conditions", []):
        if c.get("type") == "Ready" and c.get("status") == "True":
            ready_at = _parse_rfc3339(c["lastTransitionTime"])
            break
    started_at = None
    for cs in pod.get("status", {}).get("containerStatuses", []):
        t = cs.get("state", {}).get("running", {}).get("startedAt")
        if t:
            started_at = _parse_rfc3339(t)
            break
    if ready_at and started_at and ready_at >= started_at:
        return (ready_at - started_at).total_seconds()
    return None


# ---------------------------------------------------------------------------
# Wait helpers
# ---------------------------------------------------------------------------


def wait_dgdr_deployed(ns: str, name: str, timeout: int) -> dict:
    deadline = time.monotonic() + timeout
    last_phase = ""
    last_profiling = ""
    while time.monotonic() < deadline:
        dgdr = kube_get("dgdr", name, ns)
        phase = dgdr.get("status", {}).get("phase", "")
        profiling = dgdr.get("status", {}).get("profilingPhase", "")
        if phase != last_phase or profiling != last_profiling:
            extra = f" ({profiling})" if profiling else ""
            info(f"DGDR phase: {last_phase or '<unset>'} -> {phase}{extra}")
            last_phase = phase
            last_profiling = profiling
        if phase == "Deployed":
            return dgdr
        if phase == "Failed":
            msg = ""
            for c in dgdr.get("status", {}).get("conditions", []):
                if c.get("type") == "Succeeded":
                    msg = c.get("message", "")
            raise RuntimeError(f"DGDR failed: {msg}")
        time.sleep(15)
    raise TimeoutError(f"DGDR {name!r} did not reach Deployed within {timeout}s")


def wait_pod_ready(ns: str, label: str, timeout: int, poll=5) -> tuple[float, dict]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pods = kube_list("pods", ns, label)
        for pod in pods:
            if pod.get("status", {}).get("phase") == "Running" and pod_is_ready(pod):
                elapsed = pod_start_to_ready_seconds(pod) or 0.0
                return elapsed, pod
        time.sleep(poll)
    raise TimeoutError(f"No pod with label {label!r} became Ready within {timeout}s")


def wait_for_new_pod_ready(
    ns: str, worker_label: str, existing_pod_names: set, timeout: int, poll: int = 5
) -> tuple[float, dict]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for pod in kube_list("pods", ns, worker_label):
            if pod["metadata"]["name"] in existing_pod_names:
                continue
            if pod.get("status", {}).get("phase") == "Running" and pod_is_ready(pod):
                elapsed = pod_start_to_ready_seconds(pod) or 0.0
                return elapsed, pod
        time.sleep(poll)
    raise TimeoutError(
        f"No new pod beyond {existing_pod_names} matching {worker_label!r} "
        f"became Ready within {timeout}s"
    )


def wait_for_worker_count(
    ns: str, worker_label: str, target: int, timeout: int, poll: int = 10
) -> None:
    deadline = time.monotonic() + timeout
    last_count = -1
    while time.monotonic() < deadline:
        pods = kube_list("pods", ns, worker_label)
        ready = sum(1 for p in pods if pod_is_ready(p))
        if ready != last_count:
            info(f"Ready workers: {ready} (waiting for {target})")
            last_count = ready
        if ready == target:
            return
        time.sleep(poll)
    raise TimeoutError(
        f"Ready worker count did not reach {target} within {timeout}s "
        f"(last seen: {last_count})"
    )


# ---------------------------------------------------------------------------
# Service discovery helpers
# ---------------------------------------------------------------------------


def find_worker_service(dgd: dict) -> str:
    dgd_name = dgd.get("metadata", {}).get("name", "")
    services = dgd.get("spec", {}).get("services", {})
    for svc_name, svc_spec in services.items():
        if svc_spec.get("subComponentType") == "decode":
            return svc_name
    for svc_name, svc_spec in services.items():
        if svc_spec.get("componentType") == "worker":
            return svc_name
    raise RuntimeError(
        f"No worker/decode service found in DGD {dgd_name!r}. "
        f"Available: {list(services.keys())}"
    )


def find_prefill_service(dgd: dict) -> str | None:
    """Return the prefill service name, or None if the DGD has no prefill service."""
    services = dgd.get("spec", {}).get("services", {})
    for svc_name, svc_spec in services.items():
        if svc_spec.get("subComponentType") == "prefill":
            return svc_name
    return None


def find_frontend_service(dgd: dict) -> str:
    dgd_name = dgd["metadata"]["name"]
    services = dgd.get("spec", {}).get("services", {})
    for key, spec in services.items():
        if spec.get("componentType", "").lower() == "frontend":
            return f"{dgd_name}-{key.lower()}"
    raise RuntimeError(
        f"No service with componentType 'frontend' found in DGD {dgd_name!r}."
    )


def find_planner_pod(dgd_name: str, ns: str) -> str | None:
    for component in ("Planner", "planner"):
        label = (
            f"nvidia.com/dynamo-graph-deployment-name={dgd_name},"
            f"nvidia.com/dynamo-component={component}"
        )
        pods = kube_list("pods", ns, label)
        for pod in pods:
            if pod.get("status", {}).get("phase") == "Running":
                return pod["metadata"]["name"]
    return None


def detect_tp_from_dgd(dgd: dict) -> int:
    """Read the GPU count per decode replica from the DGD (= TP size)."""
    services = dgd.get("spec", {}).get("services", {})
    for svc_spec in services.values():
        if (
            svc_spec.get("subComponentType") == "decode"
            or svc_spec.get("componentType") == "worker"
        ):
            resources = svc_spec.get("resources", {})
            gpu_str = resources.get("limits", {}).get("gpu") or resources.get(
                "requests", {}
            ).get("gpu")
            if gpu_str:
                try:
                    return int(gpu_str)
                except ValueError:
                    pass
    return 1


# ---------------------------------------------------------------------------
# Load generation
# ---------------------------------------------------------------------------


def tail_planner_logs_background(
    dgd_name: str, ns: str, stop_event: threading.Event, interval: int = 30
) -> threading.Thread:
    keywords = (
        "scaling",
        "scale",
        "load",
        "active_decode",
        "num_req",
        "desired",
        "gms",
        "workers",
        "adjustment",
        "ITL",
        "TTFT",
    )

    def _tail():
        seen: set[str] = set()
        while not stop_event.is_set():
            stop_event.wait(interval)
            pod = find_planner_pod(dgd_name, ns)
            if not pod:
                continue
            logs = kube_logs(pod, ns, tail=80)
            new_lines = []
            for line in logs.splitlines():
                if line in seen:
                    continue
                seen.add(line)
                if any(kw.lower() in line.lower() for kw in keywords):
                    new_lines.append(line)
            if new_lines:
                info(f"[planner] {new_lines[-1]}")

    t = threading.Thread(target=_tail, daemon=True)
    t.start()
    return t


def _patch_replicas(ns: str, dgd_name: str, svc_name: str, replicas: int) -> None:
    patch = {"spec": {"services": {svc_name: {"replicas": replicas}}}}
    patch_path = Path(tempfile.gettempdir()) / "_dynamo_scale_patch.json"
    patch_path.write_text(json.dumps(patch))
    subprocess.run(
        [
            "kubectl",
            "patch",
            "dynamographdeployment",
            dgd_name,
            "-n",
            ns,
            "--type=merge",
            f"--patch-file={patch_path}",
        ],
        check=True,
    )
    info(f"DGD {dgd_name!r} service {svc_name!r} → replicas={replicas}")


def _extract_model_load_time(pod_name: str, ns: str) -> float | None:
    """Return the model-loading seconds from a vLLM pod's logs, or None if not found.

    Looks for vLLM's standard log line:
        Model loading took X.XX GiB memory and Y.YY seconds
    """
    import re

    logs = kube_logs(pod_name, ns, tail=200)
    for line in logs.splitlines():
        m = re.search(r"Model loading took .+ and ([\d.]+) seconds", line)
        if m:
            return float(m.group(1))
    return None


def _direct_scale_up(
    cfg: dict, svc_name: str, worker_label: str, dgd_name: str, label: str
) -> tuple[float, float | None]:
    """Scale DGD service to +1 replica by direct patch.

    Returns (container_ready_s, model_load_s) where model_load_s is extracted from
    the new pod's vLLM log line ("Model loading took ... Y.YY seconds"), or None if
    the line was not found.

    The Planner's load-based path returns None (no-op) when the model never saturates
    (slope=0, fallback x_sla=0), so a direct DGD patch is both safe and deterministic.
    """
    ns = cfg["namespace"]

    existing_pods = {
        p["metadata"]["name"]
        for p in kube_list("pods", ns, worker_label)
        if pod_is_ready(p)
    }
    target = len(existing_pods) + 1
    info(f"Existing Ready workers: {existing_pods} → patching to {target}")
    _patch_replicas(ns, dgd_name, svc_name, target)

    stop_event = threading.Event()
    tail_planner_logs_background(dgd_name, ns, stop_event)

    try:
        t0 = time.monotonic()
        info(
            f"Waiting up to {cfg['deploy_timeout']}s for a new decode worker to appear ..."
        )
        elapsed, pod = wait_for_new_pod_ready(
            ns, worker_label, existing_pods, cfg["deploy_timeout"]
        )
        wall = time.monotonic() - t0
        pod_name = pod["metadata"]["name"]
        model_load_s = _extract_model_load_time(pod_name, ns)
        ok(
            f"{label}: container→ready={elapsed:.1f}s  wall={wall:.1f}s  "
            f"model-load={model_load_s:.2f}s  pod={pod_name}"
            if model_load_s is not None
            else f"{label}: container→ready={elapsed:.1f}s  wall={wall:.1f}s  pod={pod_name}"
        )
        return elapsed, model_load_s
    finally:
        stop_event.set()


# ---------------------------------------------------------------------------
# DGDR YAML builder
# ---------------------------------------------------------------------------


def build_dgdr_yaml(cfg: dict, phase_mode: str = "auto") -> str:
    """Build a DGDR YAML manifest.

    Args:
        cfg: Configuration dict.
        phase_mode: Controls checkpoint behavior in the generated DGD:
            ``"cold"`` / ``"auto"``  — checkpoint disabled via DGD service override (plain cold start).
            ``"snap"``               — checkpoint Auto (profiler default; operator creates/restores CR).
    """
    model = cfg["model"]
    backend = "vllm"
    name = cfg["dgdr_name"]
    ns = cfg["namespace"]
    image = cfg[
        "image"
    ]  # vllm-runtime: used for spec.image, profiler, planner, workers
    frontend_image = _derive_frontend_image(
        image
    )  # dynamo-frontend: used for Frontend service only
    ngc_secret = cfg["ngc_secret"]
    hf_secret = cfg["hf_secret"]
    hf_cache_pvc = cfg["hf_cache_pvc"]
    strategy = cfg["search_strategy"]
    runtime_class = cfg["runtime_class"]
    gpu_sku = cfg["gpu_sku"]
    gpus_per_node = cfg["gpus_per_node"]
    total_gpus = cfg["total_gpus"]
    vram_mb = cfg["vram_mb"]
    prometheus_endpoint = cfg.get("prometheus_endpoint", "")
    force_tp = cfg.get("force_tp", 0)  # 0 = let profiler decide
    isl = cfg.get("isl", 3000)
    osl = cfg.get("osl", 150)
    max_model_len = cfg.get("max_model_len", 0)
    if max_model_len == 0:
        # Auto: smallest power of 2 that fits the benchmark workload
        _needed = isl + osl
        max_model_len = 1 << (_needed - 1).bit_length()

    runtime_class_line = (
        f"            runtimeClassName: {runtime_class}" if runtime_class else ""
    )

    planner_config: dict = {
        "mode": "decode",
        "backend": backend,
        "enable_throughput_scaling": bool(prometheus_endpoint),
        "enable_load_scaling": True,
        "no_correction": True,
        "min_endpoint": 1,
        "max_gpu_budget": 2,
        "load_adjustment_interval": 10,
        "load_min_observations": 3,
        "throughput_adjustment_interval": 60,
        "itl": 1.0,
    }
    if prometheus_endpoint:
        planner_config["metric_pulling_prometheus_endpoint"] = prometheus_endpoint

    hardware_section = ""
    if gpu_sku or gpus_per_node or total_gpus or vram_mb:
        hardware_section = f"""\
          hardware:
            gpuSku: "{gpu_sku}"
            numGpusPerNode: {gpus_per_node}
            totalGpus: {total_gpus}
            vramMb: {vram_mb}"""

    # When forcing TP=1, override GPU resources and add the vLLM TP arg.
    # The profiler may have generated a higher TP; these overrides take precedence
    # in the merged DGD.
    tp_resource_lines = ""
    tp_args_lines = ""
    if force_tp > 0:
        # These strings use the same absolute indentation as their surrounding
        # template lines so that the outer textwrap.dedent (which removes 8
        # spaces) produces correct relative YAML indentation.
        tp_resource_lines = (
            f"                    resources:\n"
            f"                      limits:\n"
            f'                        gpu: "{force_tp}"\n'
            f"                      requests:\n"
            f'                        gpu: "{force_tp}"\n'
        )
        tp_args_lines = (
            f"                        args:\n"
            f'                          - "--tensor-parallel-size"\n'
            f'                          - "{force_tp}"\n'
        )

    if max_model_len > 0:
        tp_args_lines = (
            (tp_args_lines if tp_args_lines else "                        args:\n")
            + '                          - "--max-model-len"\n'
            + f'                          - "{max_model_len}"\n'
        )

    # The profiler now always injects checkpoint (mode=Auto) into every DGD it
    # generates.  For the cold DGDR we override it off in the DGD service spec so
    # the operator treats it as a plain cold start.  The snap DGDR needs nothing
    # extra — checkpoint Auto is already injected by the profiler.
    _is_cold = phase_mode != "snap"
    checkpoint_disabled_override = (
        ("                    checkpoint:\n" "                      enabled: false\n")
        if _is_cold
        else ""
    )

    return textwrap.dedent(
        f"""\
        # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
        # SPDX-License-Identifier: Apache-2.0
        #
        # Auto-generated by checkpoint_scale_timing.py
        apiVersion: nvidia.com/v1beta1
        kind: DynamoGraphDeploymentRequest
        metadata:
          name: {name}
          namespace: {ns}
        spec:
          model: {model}
          backend: {backend}
          image: {image}
          searchStrategy: {strategy}
          autoApply: true
          sla:
            ttft: 500.0
            itl: 50.0
          workload:
            isl: {isl}
            osl: {osl}
{hardware_section}
          features:
            planner: {json.dumps(planner_config)}
          overrides:
            dgd:
              apiVersion: nvidia.com/v1alpha1
              kind: DynamoGraphDeployment
              spec:
                pvcs:
                  - name: {hf_cache_pvc}
                    create: false
                services:
                  Frontend:
                    extraPodSpec:
                      imagePullSecrets:
                        - name: {ngc_secret}
                      mainContainer:
                        image: {frontend_image}
                        env:
                          - name: DYN_ROUTER_MODE
                            value: kv
                  Planner:
                    extraPodSpec:
                      imagePullSecrets:
                        - name: {ngc_secret}
                      mainContainer:
                        image: {image}
                  VllmDecodeWorker:
                    replicas: 1
{checkpoint_disabled_override}{tp_resource_lines}                    envFromSecret: {hf_secret}
                    extraPodSpec:
                      {runtime_class_line}
                      imagePullSecrets:
                        - name: {ngc_secret}
                      tolerations:
                        - key: nvidia.com/gpu
                          operator: Exists
                          effect: NoSchedule
                      volumes:
                        - name: hf-cache
                          persistentVolumeClaim:
                            claimName: {hf_cache_pvc}
                      mainContainer:
{tp_args_lines}                        env:
                          - name: HF_HOME
                            value: /home/dynamo/.cache/huggingface
                        volumeMounts:
                          - name: hf-cache
                            mountPath: /home/dynamo/.cache/huggingface
                  VllmPrefillWorker:
                    envFromSecret: {hf_secret}
                    extraPodSpec:
                      {runtime_class_line}
                      imagePullSecrets:
                        - name: {ngc_secret}
                      tolerations:
                        - key: nvidia.com/gpu
                          operator: Exists
                          effect: NoSchedule
                      volumes:
                        - name: hf-cache
                          persistentVolumeClaim:
                            claimName: {hf_cache_pvc}
                      mainContainer:
{tp_args_lines}                        env:
                          - name: HF_HOME
                            value: /home/dynamo/.cache/huggingface
                        volumeMounts:
                          - name: hf-cache
                            mountPath: /home/dynamo/.cache/huggingface
        """
    )


# ---------------------------------------------------------------------------
# Namespace auto-detection
# ---------------------------------------------------------------------------


def detect_operator_namespace() -> str:
    r = _kubectl(
        "get",
        "pods",
        "--all-namespaces",
        "--field-selector=status.phase=Running",
        "-l",
        "app.kubernetes.io/name=dynamo-operator",
        "-o",
        "jsonpath={range .items[*]}{.metadata.namespace}\\n{end}",
        capture=True,
        check=False,
    )
    if r.returncode == 0:
        namespaces = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
        if namespaces:
            if len(namespaces) > 1:
                warn(
                    f"Multiple operator namespaces found: {namespaces}. Using {namespaces[0]!r}."
                )
            return namespaces[0]
    try:
        with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace") as f:
            return f.read().strip()
    except FileNotFoundError:
        pass
    warn("Could not auto-detect operator namespace — defaulting to 'default'.")
    return "default"


def detect_prometheus_endpoint() -> str:
    import urllib.request

    candidates = [
        ("monitoring", "prometheus-kube-prometheus-prometheus", 9090),
        ("monitoring", "prometheus-operated", 9090),
        ("monitoring", "kube-prometheus-stack-prometheus", 9090),
        ("prometheus", "prometheus-server", 80),
    ]
    for ns, svc, port in candidates:
        r = _kubectl("get", "svc", svc, "-n", ns, capture=True, check=False)
        if r.returncode != 0:
            continue
        url = f"http://{svc}.{ns}.svc.cluster.local:{port}"
        try:
            urllib.request.urlopen(f"{url}/-/ready", timeout=2)
            info(f"Auto-detected Prometheus (reachable): {url}")
            return url
        except Exception:
            pass
    return ""


# ---------------------------------------------------------------------------
# Benchmark scenarios
# ---------------------------------------------------------------------------


def _patch_checkpoint_job_id_label(ns: str, stop_event: threading.Event) -> None:
    """Background thread: add nvidia.com/snapshot-checkpoint-id=1 to checkpoint/restore pods.

    The new snapshot-agent image requires this label on both checkpoint source pods and
    restore target pods, but the operator does not set it automatically.
    Polls every 10s for pods missing the label and patches them.
    """
    labeled: set[str] = set()
    selectors = [
        "nvidia.com/snapshot-is-checkpoint-source=true",
        "nvidia.com/snapshot-is-restore-target=true",
    ]
    while not stop_event.is_set():
        try:
            for selector in selectors:
                r = subprocess.run(
                    [
                        "kubectl",
                        "get",
                        "pods",
                        "-n",
                        ns,
                        "-l",
                        selector,
                        "-o",
                        "json",
                    ],
                    capture_output=True,
                    text=True,
                )
                if r.returncode != 0:
                    continue
                items = json.loads(r.stdout).get("items", [])
                for pod in items:
                    pod_name = pod["metadata"]["name"]
                    if pod_name in labeled:
                        continue
                    labels_on_pod = pod.get("metadata", {}).get("labels", {})
                    if "nvidia.com/snapshot-checkpoint-id" not in labels_on_pod:
                        patch_r = subprocess.run(
                            [
                                "kubectl",
                                "label",
                                "pod",
                                pod_name,
                                "-n",
                                ns,
                                "nvidia.com/snapshot-checkpoint-id=1",
                                "--overwrite",
                            ],
                            capture_output=True,
                            text=True,
                        )
                        if patch_r.returncode == 0:
                            info(
                                f"Added nvidia.com/snapshot-checkpoint-id=1 to pod {pod_name}"
                            )
                            labeled.add(pod_name)
                        else:
                            warn(
                                f"Failed to label pod {pod_name}: {patch_r.stderr.strip()}"
                            )
        except Exception as e:
            warn(f"[checkpoint-id-labeler] {e}")
        stop_event.wait(10)


def wait_for_checkpoint_ready(
    dgd_name: str, ns: str, svc_name: str, timeout: int
) -> str:
    """Poll until the DynamoCheckpoint CR for svc_name is Ready. Returns checkpoint name.

    The DGD status.checkpoints[svc_name] gives us the checkpoint name once the operator
    creates the CR. We then poll the DynamoCheckpoint CR directly for phase=Ready, since
    the DGD status does not carry a 'ready' boolean.
    """
    deadline = time.monotonic() + timeout
    last_phase = ""
    ckpt_name = ""
    while time.monotonic() < deadline:
        dgd = kube_get("dynamographdeployment", dgd_name, ns)
        if dgd:
            ckpt_status = dgd.get("status", {}).get("checkpoints", {}).get(svc_name, {})
            ckpt_name = ckpt_status.get("checkpointName", "")
            if ckpt_name:
                # Checkpoint CR exists — check its phase directly.
                ckpt_cr = kube_get("dynamocheckpoint", ckpt_name, ns)
                phase = (ckpt_cr or {}).get("status", {}).get("phase", "")
                if phase != last_phase:
                    info(
                        f"DynamoCheckpoint {ckpt_name!r} phase: {last_phase or '<none>'} → {phase}"
                    )
                    last_phase = phase
                if phase == "Ready":
                    return ckpt_name
            else:
                info(
                    f"Waiting for operator to create DynamoCheckpoint for {svc_name!r} in {dgd_name!r}…"
                )
        time.sleep(10)

    raise TimeoutError(
        f"DynamoCheckpoint for {svc_name!r} in DGD {dgd_name!r} not Ready within {timeout}s"
        + (
            f" (last checkpoint: {ckpt_name!r})"
            if ckpt_name
            else " (no checkpoint created)"
        )
    )


def _discover_existing_dgdr(
    name: str,
    label: str,
    cfg: dict,
    table: "_LiveTable",
) -> dict:
    """Discover an already-deployed DGDR without applying or re-profiling.

    Used with --keep-deployments to skip Phase 1 and jump straight to scale-up.
    Raises RuntimeError if the DGDR is not in Deployed state.
    """
    ns = cfg["namespace"]
    dgdr_name = cfg["dgdr_name"]
    phase_mode = cfg.get("_phase_mode", "cold")

    table.update(label, "waiting for existing DGDR…")
    dgdr = kube_get("dgdr", dgdr_name, ns)
    if not dgdr:
        raise RuntimeError(
            f"DGDR {dgdr_name!r} not found in namespace {ns!r} (did you forget to deploy first?)"
        )

    # Wait for Deployed in case it's still profiling (e.g. Deploying after a prior run).
    dgdr = wait_dgdr_deployed(ns, dgdr_name, cfg["profiling_timeout"])

    dgd_name = dgdr.get("status", {}).get("dgdName", "")
    if not dgd_name:
        raise RuntimeError(f"status.dgdName is empty on DGDR {dgdr_name!r}")

    table.update(label, f"found → {dgd_name[:20]}")
    info(f"Reusing existing DGDR {dgdr_name!r} → DGD {dgd_name!r}")

    dgd = kube_get("dynamographdeployment", dgd_name, ns)
    if not dgd:
        raise RuntimeError(f"DGD {dgd_name!r} not found")

    svc_name = find_worker_service(dgd)
    tp = detect_tp_from_dgd(dgd)
    worker_label = (
        f"nvidia.com/dynamo-graph-deployment-name={dgd_name},"
        f"nvidia.com/dynamo-component={svc_name}"
    )

    return {
        "name": name,
        "label": label,
        "cfg": cfg,
        "dgdr_name": dgdr_name,
        "dgd_name": dgd_name,
        "dgd": dgd,
        "svc_name": svc_name,
        "tp": tp,
        "worker_label": worker_label,
        "phase_mode": phase_mode,
    }


def _deploy_dgdr(
    name: str,
    label: str,
    cfg: dict,
    workdir: Path,
    table: "_LiveTable",
) -> dict:
    """Apply a DGDR and wait for it to reach Deployed. Returns deployment info dict.

    Does NOT wait for checkpoint — callers handle that explicitly.
    """
    ns = cfg["namespace"]
    dgdr_name = cfg["dgdr_name"]
    phase_mode = cfg.get("_phase_mode", "cold")

    table.update(label, "building DGDR manifest…")
    yaml_text = build_dgdr_yaml(cfg, phase_mode=phase_mode)
    yaml_path = workdir / f"dgdr_{name}.yaml"
    yaml_path.write_text(yaml_text)

    table.update(label, "applying DGDR…")
    kube_apply(str(yaml_path), ns)

    table.update(label, "profiling model…")
    dgdr = wait_dgdr_deployed(ns, dgdr_name, cfg["profiling_timeout"])
    dgd_name = dgdr.get("status", {}).get("dgdName", "")
    if not dgd_name:
        raise RuntimeError(f"status.dgdName is empty on DGDR {dgdr_name!r}")

    table.update(label, f"deployed → {dgd_name[:20]}")
    dgd = kube_get("dynamographdeployment", dgd_name, ns)
    if not dgd:
        raise RuntimeError(f"DGD {dgd_name!r} not found")

    svc_name = find_worker_service(dgd)
    tp = detect_tp_from_dgd(dgd)
    worker_label = (
        f"nvidia.com/dynamo-graph-deployment-name={dgd_name},"
        f"nvidia.com/dynamo-component={svc_name}"
    )

    return {
        "name": name,
        "label": label,
        "cfg": cfg,
        "dgdr_name": dgdr_name,
        "dgd_name": dgd_name,
        "dgd": dgd,
        "svc_name": svc_name,
        "tp": tp,
        "worker_label": worker_label,
        "phase_mode": phase_mode,
    }


def _measure_scale_up_for_deployed(
    deployed_info: dict,
    table: "_LiveTable",
) -> dict:
    """Ensure 1 baseline worker is Ready, then measure the 1→2 scale-up time."""
    label = deployed_info["label"]
    cfg = deployed_info["cfg"]
    dgd_name = deployed_info["dgd_name"]
    svc_name = deployed_info["svc_name"]
    worker_label = deployed_info["worker_label"]
    ns = cfg["namespace"]

    # Ensure exactly 1 decode worker is running before measuring scale-up.
    # The planner may start at 0; patch to 1 and wait for it to be Ready.
    # With --keep-deployments there may be 2+ from a prior run — scale down to 1 first.
    # Workers may be crashlooping (0 Ready but 2 total) — use total pod count, not Ready count,
    # when waiting for scale-down to complete.
    all_pods = kube_list("pods", ns, worker_label)
    existing_ready = [p for p in all_pods if pod_is_ready(p)]
    if len(all_pods) > 1:
        table.update(label, "scaling down to 1 baseline worker…")
        _patch_replicas(ns, dgd_name, svc_name, 1)
        # Wait for extra pods to terminate (may be crashlooping → never Ready)
        scale_down_timeout = cfg.get("scale_down_timeout", 300)
        deadline = time.monotonic() + scale_down_timeout
        while time.monotonic() < deadline:
            all_pods = kube_list("pods", ns, worker_label)
            if len(all_pods) <= 1:
                break
            time.sleep(10)
        else:
            raise TimeoutError(
                f"Scale-down timed out: {len(all_pods)} pods still present after {scale_down_timeout}s"
            )
        existing_ready = [p for p in all_pods if pod_is_ready(p)]
    if len(existing_ready) < 1:
        table.update(label, "ensuring 1 baseline worker…")
        _patch_replicas(ns, dgd_name, svc_name, 1)
        existing_names: set[str] = {
            p["metadata"]["name"] for p in kube_list("pods", ns, worker_label)
        }
        wait_for_new_pod_ready(ns, worker_label, existing_names, cfg["deploy_timeout"])

    table.update(label, "scaling 1→2…")
    scale_up_s, model_load_s = _direct_scale_up(
        cfg, svc_name, worker_label, dgd_name, label
    )

    table.done(label, f"{scale_up_s:.0f}s container→ready")
    return {
        "name": deployed_info["name"],
        "label": label,
        "tp": deployed_info["tp"],
        "scale_up_s": scale_up_s,
        "model_load_s": model_load_s,
        "dgdr_name": deployed_info["dgdr_name"],
        "dgd_name": dgd_name,
    }


def run_all_scenarios(cfg: dict, args, workdir: Path) -> list[dict]:
    """Deploy cold + snapshot DGDRs and measure scale-up time on each.

    With unique DGD naming per DGDR (dyn-2543 fix), cold and snap produce distinct
    DGDs so their scale-ups are fully independent.

    Three-phase approach:

    Phase 1 (sequential): Deploy cold DGDR → wait for Deployed → deploy snap DGDR →
      wait for Deployed → wait for snap checkpoint to be Ready.
      Sequential profiling avoids two GPU-intensive profiling jobs contending.
      Waiting for checkpoint in Phase 1 means both scale-ups fire simultaneously
      from the same starting point in Phase 2.

    Phase 2 (parallel): Scale both DGDs 1→2 concurrently and measure time-to-Ready.
      Both fire at the same instant; cold measures weight-loading time while snap
      restores from the checkpoint created in Phase 1.
    """
    import traceback

    base_name = cfg["dgdr_name"]

    cold_cfg = dict(cfg)
    cold_cfg["_phase_mode"] = "cold"

    snap_cfg = dict(cfg)
    snap_cfg["dgdr_name"] = f"{base_name}-snap"
    snap_cfg["_phase_mode"] = "snap"

    scenarios: list[dict] = [
        {"name": "cold", "label": "Cold start (no checkpoint)", "cfg": cold_cfg},
        {
            "name": "snap",
            "label": "Snapshot restore (checkpoint.auto)",
            "cfg": snap_cfg,
        },
    ]

    labels = [s["label"] for s in scenarios]
    table = _LiveTable(labels)
    print()  # blank line before live table

    # Start background labeler for the full benchmark duration.
    # The new snapshot-agent requires nvidia.com/snapshot-checkpoint-id=1 on both
    # checkpoint source and restore target pods, but the operator doesn't set it.
    labeler_stop = threading.Event()
    labeler_thread = threading.Thread(
        target=_patch_checkpoint_job_id_label,
        args=(cfg["namespace"], labeler_stop),
        daemon=True,
    )
    labeler_thread.start()

    # ── Phase 1: Sequential deployment + checkpoint wait ─────────────────────
    # Deploy cold first, then snap.  Running both profiling jobs in parallel
    # causes GPU contention: cold's profiling job occupies the GPU while snap's
    # profiling job is stuck pending.
    # Waiting for snap's checkpoint here (before Phase 2) means both scale-ups
    # fire simultaneously — after both workers have been running for the same time.
    deployed: list[dict] = []

    keep_deployments = getattr(args, "keep_deployments", False)

    for s in scenarios:
        sc_workdir = workdir / s["name"]
        sc_workdir.mkdir(parents=True, exist_ok=True)
        try:
            if keep_deployments:
                dep = _discover_existing_dgdr(s["name"], s["label"], s["cfg"], table)
            else:
                dep = _deploy_dgdr(s["name"], s["label"], s["cfg"], sc_workdir, table)
            # For snap: wait for checkpoint to be ready before proceeding.
            if dep["phase_mode"] == "snap":
                table.update(dep["label"], "waiting for checkpoint…")
                try:
                    ckpt_name = wait_for_checkpoint_ready(
                        dep["dgd_name"],
                        dep["cfg"]["namespace"],
                        dep["svc_name"],
                        cfg["snapshot_timeout"],
                    )
                    table.update(dep["label"], f"checkpoint ready ({ckpt_name[:16]})")
                    info(f"Checkpoint ready: {ckpt_name!r}")
                    _patch_checkpoint_rootfs_with_cuda_checkpoint(
                        dep["cfg"]["namespace"],
                        ckpt_name,
                        dep["cfg"].get("snapshot_release", "dynamo-snapshot"),
                    )
                except TimeoutError as e:
                    warn(
                        f"[{dep['label']}] {e} — will scale up without checkpoint restore"
                    )
            deployed.append(dep)
        except Exception as exc:
            table.error(s["label"], f"deploy failed: {str(exc)[:40]}")
            warn(f"Scenario {s['name']!r} deploy failed: {exc}")
            traceback.print_exception(
                type(exc), exc, exc.__traceback__, file=sys.stderr
            )

    if not deployed:
        table.stop()
        labeler_stop.set()
        warn("All scenarios failed during deployment.")
        return []

    # ── Phase 2: Parallel scale-up measurement ───────────────────────────────
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=len(deployed)) as pool:
        futures = {
            pool.submit(_measure_scale_up_for_deployed, dep, table): dep
            for dep in deployed
        }

        for fut in as_completed(futures):
            dep = futures[fut]
            try:
                results.append(fut.result())
            except Exception as exc:
                table.error(dep["label"], f"scale-up failed: {str(exc)[:35]}")
                warn(f"Scenario {dep['name']!r} scale-up failed: {exc}")
                traceback.print_exception(
                    type(exc), exc, exc.__traceback__, file=sys.stderr
                )

    table.stop()
    labeler_stop.set()
    return results


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------


def print_results_table(results: list[dict], cfg: dict, workdir: Path) -> None:
    """Print a side-by-side comparison and save JSON summary."""
    if not results:
        warn("No results to display.")
        return

    cold = next((r for r in results if r["name"] == "cold"), None)
    snap = next((r for r in results if r["name"] == "snap"), None)

    print("\n" + "=" * 68)
    print("  SCALE-UP TIMING: COLD START vs SNAPSHOT RESTORE")
    print("=" * 68)
    print(f"  Model    : {cfg['model']}")
    if cold:
        print(f"  DGDR     : {cold['dgdr_name']}  TP={cold['tp']}")
    print("-" * 68)
    print(f"  {'Scenario':<40}  {'container→ready':>13}  {'model-load':>10}")
    print("-" * 68)

    def _row(r: dict) -> None:
        ml = r.get("model_load_s")
        ml_str = f"{ml:.1f}s" if ml is not None else "    n/a"
        print(f"  {r['label']:<40}  {r['scale_up_s']:>12.1f}s  {ml_str:>10}")

    for r in sorted(results, key=lambda x: {"cold": 0, "snap": 1}.get(x["name"], 9)):
        _row(r)

    if cold and snap and cold["scale_up_s"] > 0 and snap["scale_up_s"] > 0:
        print("-" * 68)
        ratio = cold["scale_up_s"] / snap["scale_up_s"]
        pct = (cold["scale_up_s"] - snap["scale_up_s"]) / cold["scale_up_s"] * 100
        print(f"  Speedup (cold → snapshot):  {ratio:.1f}x  ({pct:.0f}% faster)")

    print("=" * 68 + "\n")

    summary_path = workdir / "results_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    info(f"Summary saved to {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark scale-up time: cold start vs snapshot restore via DGDR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """            Always deploys two DGDRs in parallel and compares scale-up times:
              Cold DGDR:     {dgdr-name}       — baseline: no checkpoint
              Snapshot DGDR: {dgdr-name}-snap  — checkpoint Auto (profiler default)

            Both DGDRs use the same --image.  The profiler always injects
            checkpoint Auto into the DGD; the cold DGDR disables it via a
            DGD service override so it always cold-starts.

            Resources are left running after the benchmark completes for inspection.
            """
        ),
    )
    p.add_argument(
        "--image",
        default=os.getenv("IMAGE"),
        required=not os.getenv("IMAGE"),
        help="vllm-runtime image (used for spec.image/profiler, Planner, and workers). "
        "The dynamo-frontend image is derived automatically from the same tag.",
    )
    p.add_argument("--namespace", default=DEFAULTS["namespace"])
    p.add_argument("--model", default=DEFAULTS["model"])
    p.add_argument(
        "--tp-sizes",
        default=DEFAULTS["tp_sizes"],
        help="Comma-separated TP sizes (e.g. '1,4')",
    )
    p.add_argument("--gpu-count", type=int, default=DEFAULTS["gpu_count"])
    p.add_argument(
        "--max-model-len",
        type=int,
        default=DEFAULTS["max_model_len"],
        help="vLLM --max-model-len; 0 = auto (next power of 2 ≥ isl+osl)",
    )
    p.add_argument(
        "--isl",
        type=int,
        default=DEFAULTS["isl"],
        help="Input sequence length for profiling workload",
    )
    p.add_argument(
        "--osl",
        type=int,
        default=DEFAULTS["osl"],
        help="Output sequence length for profiling workload",
    )
    p.add_argument("--hf-cache-pvc", default=DEFAULTS["hf_cache_pvc"])
    p.add_argument("--hf-secret", default=DEFAULTS["hf_secret"])
    p.add_argument("--ngc-secret", default=DEFAULTS["ngc_secret"])
    p.add_argument("--dgdr-name", default=DEFAULTS["dgdr_name"])
    p.add_argument("--runtime-class", default=DEFAULTS["runtime_class"])
    p.add_argument(
        "--search-strategy",
        default=DEFAULTS["search_strategy"],
        choices=["rapid", "thorough"],
    )
    p.add_argument(
        "--profiling-timeout", type=int, default=DEFAULTS["profiling_timeout"]
    )
    p.add_argument("--deploy-timeout", type=int, default=DEFAULTS["deploy_timeout"])
    p.add_argument(
        "--snapshot-timeout",
        type=int,
        default=DEFAULTS["snapshot_timeout"],
        help="Max seconds to wait for checkpoint to be Ready before scaling up",
    )
    p.add_argument("--prometheus-endpoint", default=DEFAULTS["prometheus_endpoint"])
    p.add_argument("--workdir", default=DEFAULTS["workdir"])
    p.add_argument("--hf-cache-pvc-size", default=DEFAULTS["hf_cache_pvc_size"])
    p.add_argument(
        "--hf-cache-pvc-access-mode", default=DEFAULTS["hf_cache_pvc_access_mode"]
    )
    p.add_argument(
        "--hf-cache-storage-class", default=DEFAULTS["hf_cache_storage_class"]
    )
    p.add_argument("--nats-storage-class", default=DEFAULTS["nats_storage_class"])
    p.add_argument(
        "--operator-image",
        default=DEFAULTS["operator_image"],
        help="If set, deploy the Dynamo platform helm chart with this operator image",
    )
    p.add_argument(
        "--snapshot-agent-image",
        default=DEFAULTS["snapshot_agent_image"],
        help="snapshot-agent DaemonSet image (required for checkpoint restore to work)",
    )
    p.add_argument("--platform-release", default=DEFAULTS["platform_release"])
    p.add_argument("--operator-timeout", type=int, default=DEFAULTS["operator_timeout"])
    p.add_argument("--gpu-sku", default=DEFAULTS["gpu_sku"])
    p.add_argument("--gpus-per-node", type=int, default=DEFAULTS["gpus_per_node"])
    p.add_argument("--total-gpus", type=int, default=DEFAULTS["total_gpus"])
    p.add_argument("--vram-mb", type=int, default=DEFAULTS["vram_mb"])
    p.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Skip namespace cleanup at startup (reuse existing resources)",
    )
    p.add_argument(
        "--keep-operator",
        action="store_true",
        help="During cleanup, only remove DGDRs/DGDs — leave operator and NATS running",
    )
    p.add_argument(
        "--keep-deployments",
        action="store_true",
        help="Skip DGDR/DGD deployment and checkpoint wait; reuse existing DGDRs and "
        "go straight to scale-up measurement. Implies --skip-cleanup.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.operator_image:
        ns = args.namespace or os.getenv("NAMESPACE", "hannahz")
    elif args.namespace:
        ns = args.namespace
    else:
        step("Detecting operator namespace …")
        ns = detect_operator_namespace()

    if args.workdir:
        workdir = Path(args.workdir)
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        workdir = Path(tempfile.gettempdir()) / f"dynamo-scale-bench-{ts}"
    workdir.mkdir(parents=True, exist_ok=True)
    info(f"Working directory: {workdir}")

    tp_sizes = [int(x.strip()) for x in args.tp_sizes.split(",") if x.strip()]
    if not tp_sizes:
        die("--tp-sizes must be a non-empty comma-separated list of integers")

    cfg = {
        "image": args.image,
        "namespace": ns,
        "model": args.model,
        "gpu_count": args.gpu_count,
        "max_model_len": args.max_model_len,
        "isl": args.isl,
        "osl": args.osl,
        "hf_cache_pvc": args.hf_cache_pvc,
        "hf_secret": args.hf_secret,
        "ngc_secret": args.ngc_secret,
        "dgdr_name": args.dgdr_name,
        "runtime_class": args.runtime_class,
        "search_strategy": args.search_strategy,
        "profiling_timeout": args.profiling_timeout,
        "deploy_timeout": args.deploy_timeout,
        "scale_down_timeout": DEFAULTS["scale_down_timeout"],
        "prometheus_endpoint": args.prometheus_endpoint,
        "hf_cache_pvc_size": args.hf_cache_pvc_size,
        "hf_cache_pvc_access_mode": args.hf_cache_pvc_access_mode,
        "hf_cache_storage_class": args.hf_cache_storage_class,
        "nats_storage_class": args.nats_storage_class,
        "operator_image": args.operator_image,
        "keep_operator": args.keep_operator,
        "platform_release": args.platform_release,
        "operator_timeout": args.operator_timeout,
        "gpu_sku": args.gpu_sku,
        "gpus_per_node": args.gpus_per_node,
        "total_gpus": args.total_gpus,
        "vram_mb": args.vram_mb,
        "snapshot_timeout": args.snapshot_timeout,
        "snapshot_agent_image": args.snapshot_agent_image,
        "snapshot_release": DEFAULTS["snapshot_release"],
        "snapshot_pvc_size": DEFAULTS["snapshot_pvc_size"],
        "snapshot_pvc_storage_class": DEFAULTS["snapshot_pvc_storage_class"],
    }

    if not cfg["prometheus_endpoint"]:
        cfg["prometheus_endpoint"] = detect_prometheus_endpoint()
        if cfg["prometheus_endpoint"]:
            info(f"Auto-detected Prometheus: {cfg['prometheus_endpoint']}")

    r = subprocess.run(["kubectl", "get", "ns", ns], capture_output=True)
    if r.returncode != 0:
        die(f"Cannot access namespace {ns!r} — check KUBECONFIG")

    step("Starting scale-up timing benchmark")
    info(
        f"DGDR base name: {cfg['dgdr_name']}  model: {cfg['model']}  TP sizes: {tp_sizes}"
    )

    all_results: list[list[dict]] = []
    try:
        for tp in tp_sizes:
            step(f"=== TP={tp} ===")
            tp_cfg = dict(cfg)
            tp_cfg["force_tp"] = tp
            tp_cfg["gpu_count"] = tp
            if tp != tp_sizes[0]:
                tp_cfg["dgdr_name"] = f"{cfg['dgdr_name']}-tp{tp}"

            if not args.skip_cleanup and not args.keep_deployments:
                cleanup_namespace(tp_cfg)

            if not args.keep_deployments:
                if tp_cfg["operator_image"]:
                    deploy_platform(tp_cfg)
                ensure_pvcs(tp_cfg)

            if not tp_cfg["gpu_sku"] and not tp_cfg["gpus_per_node"]:
                sku, gpn, tg, vmb = detect_gpu_hardware()
                tp_cfg["gpu_sku"] = tp_cfg["gpu_sku"] or sku
                tp_cfg["gpus_per_node"] = tp_cfg["gpus_per_node"] or gpn
                tp_cfg["total_gpus"] = tp_cfg["total_gpus"] or tg
                tp_cfg["vram_mb"] = tp_cfg["vram_mb"] or vmb

            if not tp_cfg["image"]:
                die("--image is required (or set IMAGE env var)")

            tp_workdir = workdir / f"tp{tp}"
            tp_workdir.mkdir(parents=True, exist_ok=True)

            results = run_all_scenarios(tp_cfg, args, tp_workdir)
            all_results.append(results)
            print_results_table(results, tp_cfg, tp_workdir)

    except KeyboardInterrupt:
        warn("Interrupted")
        sys.exit(1)
    except (TimeoutError, RuntimeError) as e:
        die(str(e))


if __name__ == "__main__":
    main()
