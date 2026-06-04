"""In-pod per-process resource sampler for ``ResourcePoller``.

This is NOT imported by the framework — it is read as text and heredoc'd into
each target pod, then launched detached (``nohup python3 /tmp/respoller.py``).
It runs inside the worker container against that pod's own ``/proc`` + cgroup +
local NVML, so it sees per-process memory/CPU and (node-local) per-process GPU
that a remote client cannot.

Config via env (set by the launching event):
  RESPOLL_OUTDIR   output dir (defaults to $DYN_LOG_DIR, else /tmp/service_logs)
  RESPOLL_INTERVAL sample interval seconds (default 5)
  RESPOLL_INCLUDE  comma list from {mem,cpu,gpu} (default all)
  RESPOLL_STOP     stop-marker path (default /tmp/.respoller.stop)

Output (TSV, native values; %CPU and rates are differenced offline):
  <pod>.resources.agg.tsv    one row per GPU per tick (or one blank-GPU row):
      epoch_s pod working_set cgroup_current all_pids_rss pid1_rss
      gpu_idx gpu_util gpu_mem_used gpu_mem_total
  <pod>.resources.procs.tsv  one row per process per tick:
      epoch_s pid cmd rss utime stime gpu_mem gpu_sm
Every dimension is best-effort: a failure in one (e.g. pynvml absent) disables
just that dimension and leaves its columns as -1/blank.
"""
import glob
import os
import sys
import time

OUTDIR = (
    os.environ.get("RESPOLL_OUTDIR")
    or os.environ.get("DYN_LOG_DIR")
    or "/tmp/service_logs"
)
INTERVAL = float(os.environ.get("RESPOLL_INTERVAL") or "5")
INCLUDE = set((os.environ.get("RESPOLL_INCLUDE") or "mem,cpu,gpu").split(","))
STOP = os.environ.get("RESPOLL_STOP") or "/tmp/.respoller.stop"
POD = os.environ.get("HOSTNAME", "unknown")
CLK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100

try:
    os.makedirs(OUTDIR, exist_ok=True)
except OSError:
    pass
AGG = os.path.join(OUTDIR, POD + ".resources.agg.tsv")
PROCS = os.path.join(OUTDIR, POD + ".resources.procs.tsv")

# --- GPU (best effort; node-local NVML) ---
nvml = None
if "gpu" in INCLUDE:
    try:
        import pynvml

        pynvml.nvmlInit()
        nvml = pynvml
    except Exception as e:  # noqa: BLE001 - any failure → run without GPU
        sys.stderr.write("respoller: pynvml unavailable, gpu disabled: %s\n" % e)
        nvml = None


def read_cgroup():
    """(working_set, cgroup_current) bytes. working_set = current - inactive_file
    (what kubelet/cAdvisor use for OOMKill). cgroup v2 with v1 fallback."""
    try:
        if os.path.exists("/sys/fs/cgroup/memory.current"):  # v2
            cur = int(open("/sys/fs/cgroup/memory.current").read().strip())
            stat_path, key = "/sys/fs/cgroup/memory.stat", "inactive_file "
        else:  # v1
            cur = int(
                open("/sys/fs/cgroup/memory/memory.usage_in_bytes").read().strip()
            )
            stat_path, key = "/sys/fs/cgroup/memory/memory.stat", "total_inactive_file "
        inact = 0
        for ln in open(stat_path):
            if ln.startswith(key):
                inact = int(ln.split()[1])
                break
        return cur - inact, cur
    except Exception:  # noqa: BLE001
        return -1, -1


def read_procs():
    """list of (pid, cmd, rss_bytes, utime_jiffies, stime_jiffies) for every
    process in the pod. cmd from /proc/PID/cmdline (full proctitle — Worker_DP0
    vs _DP1; /proc/PID/status Name truncates at 15 chars)."""
    out = []
    want_cpu = "cpu" in INCLUDE
    for d in glob.glob("/proc/[0-9]*"):
        pid = d.rsplit("/", 1)[-1]
        try:
            rss = 0
            for ln in open(d + "/status"):
                if ln.startswith("VmRSS:"):
                    rss = int(ln.split()[1]) * 1024
                    break
            try:
                cmd = (
                    open(d + "/cmdline", "rb")
                    .read()
                    .replace(b"\x00", b" ")
                    .decode("utf-8", "replace")
                    .strip()
                )
            except Exception:  # noqa: BLE001
                cmd = ""
            if not cmd:
                try:
                    cmd = "[" + open(d + "/comm").read().strip() + "]"
                except Exception:  # noqa: BLE001
                    cmd = "?"
            ut = st = 0
            if want_cpu:
                s = open(d + "/stat").read()
                # fields after the (comm) group: utime=14th, stime=15th (1-indexed)
                parts = s[s.rfind(")") + 1 :].split()
                ut, st = int(parts[11]), int(parts[12])
            out.append((pid, cmd[:80], rss, ut, st))
        except Exception:  # noqa: BLE001 - process exited mid-read etc.
            continue
    return out


def gpu_device():
    """list of (idx, util_pct, mem_used, mem_total) per device."""
    rows = []
    if not nvml:
        return rows
    try:
        for i in range(nvml.nvmlDeviceGetCount()):
            h = nvml.nvmlDeviceGetHandleByIndex(i)
            try:
                gu = nvml.nvmlDeviceGetUtilizationRates(h).gpu
            except Exception:  # noqa: BLE001
                gu = -1
            try:
                m = nvml.nvmlDeviceGetMemoryInfo(h)
                mu, mt = m.used, m.total
            except Exception:  # noqa: BLE001
                mu = mt = -1
            rows.append((i, gu, mu, mt))
    except Exception:  # noqa: BLE001
        pass
    return rows


_last_ts = {}


def gpu_procs():
    """{pid: [gpu_mem_bytes, gpu_sm_pct]}. ComputeRunningProcesses (mem) is
    reliable; ProcessUtilization (sm) is best-effort and raises NOT_FOUND when
    the GPU has been idle — track lastSeenTimeStamp and treat empty as no
    recent activity."""
    res = {}
    if not nvml:
        return res
    try:
        for i in range(nvml.nvmlDeviceGetCount()):
            h = nvml.nvmlDeviceGetHandleByIndex(i)
            try:
                for p in nvml.nvmlDeviceGetComputeRunningProcesses(h):
                    mem = getattr(p, "usedGpuMemory", None)
                    res.setdefault(int(p.pid), [mem if mem is not None else -1, -1])
            except Exception:  # noqa: BLE001
                pass
            try:
                samples = nvml.nvmlDeviceGetProcessUtilization(h, _last_ts.get(i, 0))
                for s in samples or []:
                    _last_ts[i] = max(_last_ts.get(i, 0), int(s.timeStamp))
                    res.setdefault(int(s.pid), [-1, -1])
                    res[int(s.pid)][1] = s.smUtil
            except Exception:  # noqa: BLE001 - NOT_FOUND when idle
                pass
    except Exception:  # noqa: BLE001
        pass
    return res


def main():
    with open(AGG, "w") as f:
        f.write(
            "# ResourcePoller clk_tck=%d include=%s\n"
            % (CLK, ",".join(sorted(INCLUDE)))
        )
        f.write(
            "epoch_s\tpod\tworking_set\tcgroup_current\tall_pids_rss\tpid1_rss"
            "\tgpu_idx\tgpu_util\tgpu_mem_used\tgpu_mem_total\n"
        )
    with open(PROCS, "w") as f:
        f.write("# ResourcePoller clk_tck=%d\n" % CLK)
        f.write("epoch_s\tpid\tcmd\trss\tutime\tstime\tgpu_mem\tgpu_sm\n")
    try:
        os.unlink(STOP)
    except OSError:
        pass

    want_proc = ("mem" in INCLUDE) or ("cpu" in INCLUDE)
    while not os.path.exists(STOP):
        ts = "%.3f" % time.time()
        procs = read_procs() if want_proc else []
        ws, cur = read_cgroup() if "mem" in INCLUDE else (-1, -1)
        all_rss = sum(p[2] for p in procs) if procs else -1
        pid1_rss = next((p[2] for p in procs if p[0] == "1"), -1)
        gdev = gpu_device()
        gproc = gpu_procs()
        try:
            with open(AGG, "a") as f:
                if gdev:
                    for idx, gu, mu, mt in gdev:
                        f.write(
                            "%s\t%s\t%d\t%d\t%d\t%d\t%d\t%s\t%s\t%s\n"
                            % (ts, POD, ws, cur, all_rss, pid1_rss, idx, gu, mu, mt)
                        )
                else:
                    f.write(
                        "%s\t%s\t%d\t%d\t%d\t%d\t\t\t\t\n"
                        % (ts, POD, ws, cur, all_rss, pid1_rss)
                    )
            with open(PROCS, "a") as f:
                for pid, cmd, rss, ut, st in procs:
                    g = gproc.get(int(pid)) if pid.isdigit() else None
                    gm = g[0] if g else ""
                    gs = g[1] if g else ""
                    f.write(
                        "%s\t%s\t%s\t%d\t%d\t%d\t%s\t%s\n"
                        % (ts, pid, cmd, rss, ut, st, gm, gs)
                    )
        except Exception:  # noqa: BLE001
            pass
        time.sleep(INTERVAL)

    if nvml:
        try:
            nvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
