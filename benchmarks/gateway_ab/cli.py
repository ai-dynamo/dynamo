# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A/B benchmark: Dynamo Frontend (no EPP) vs Gateway + EPP in Dynamo runtime mode.

Run from the repository root::

    python3 -m benchmarks.gateway_ab --context <ctx> all

See README.md in this directory for the full procedure.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import logging
import pathlib
import subprocess
import sys
from typing import Any

from . import manifests, report
from .k8s_backend import K8sBackend
from .kube import KubeError, retry
from .loadgen import RESULTS_ROOT, LoadPoint, Workload, aiperf_command
from .local_backend import LocalBackend

log = logging.getLogger("gateway_ab")

EXIT_PREFLIGHT = 2
EXIT_REGRESSION = 3
# Relative to the working directory. Run from the repository root, it lands
# under benchmarks/results, which the repository already ignores.
DEFAULT_OUTPUT = "benchmarks/results/gateway_ab"


class Bench:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.cfg = manifests.BenchConfig(
            model=args.model,
            dynamo_version=args.dynamo_version,
            frontend_image=args.frontend_image or "",
            worker_image=args.worker_image or "",
            name_prefix=args.name_prefix,
            agg_workers=args.agg_workers,
            prefill_workers=args.prefill_workers,
            decode_workers=args.decode_workers,
            block_size=args.block_size,
            speedup_ratio=args.speedup_ratio,
            mocker_extra_args=tuple(args.mocker_arg or ()),
            router_cpu=args.router_cpu,
            worker_cpu=args.worker_cpu,
            sidecar_cpu=args.sidecar_cpu,
            loadgen_cpu=args.loadgen_cpu,
            hf_secret=args.hf_secret,
            image_pull_secret=args.image_pull_secret,
            gateway_name=args.gateway_name,
            gateway_namespace=args.gateway_namespace,
        )
        self.workload = Workload(
            # --concurrency switches to closed loop; otherwise open loop at
            # --request-rate or the Workload's default rates.
            concurrencies=tuple(args.concurrency or Workload.concurrencies),
            request_rates=(
                ()
                if args.concurrency
                else tuple(args.request_rate or Workload.request_rates)
            ),
            isl=args.isl,
            osl=args.osl,
            prefix_length=args.prefix_length,
            num_prefixes=args.num_prefixes,
            request_count=args.requests,
            warmup_count=args.warmup_requests,
            rounds=args.rounds,
            seed=args.seed,
            run_timeout_s=args.run_timeout,
        )
        self.topologies = tuple(args.topologies)
        if args.backend == "local":
            self.backend = LocalBackend(args, self.cfg)
        else:
            self.backend = K8sBackend(args, self.cfg)
        self.loadgen = self.backend.loadgen

    # -- backend delegation -------------------------------------------------

    def preflight(self) -> list[str]:
        return self.backend.preflight()

    def deploy_loadgen(self) -> None:
        self.backend.deploy_loadgen()

    def deploy_topology(self, topology: str) -> None:
        self.backend.deploy_topology(topology)
        self._smoke(topology)

    def teardown_topology(self, topology: str) -> None:
        self.backend.teardown_topology(topology)

    def teardown_all(self) -> None:
        self.backend.teardown_all()

    def url(self, topology: str, side: str) -> str:
        return self.backend.url(topology, side)

    def _smoke(self, topology: str) -> None:
        """One request per side through the full path, retried while it comes up."""
        for side in manifests.SIDES:
            url = self.url(topology, side)
            headers = manifests.request_headers(self.cfg, topology)
            try:
                latency = retry(
                    lambda: self.loadgen.smoke(url, self.cfg.model, headers),
                    attempts=60,
                    delay_s=5,
                    what=f"smoke request to {topology}/{side} at {url}",
                )
            except KubeError as e:
                raise KubeError(f"{e}\n{self.backend.diagnose(topology, side)}") from e
            log.info("Smoke %s/%s OK (%.0f ms)", topology, side, latency * 1000)

    # -- measure -----------------------------------------------------------

    def measure_topology(self, topology: str, results: pathlib.Path) -> int:
        """Run every (round, load point, side); return the failure count."""
        w = self.workload
        pending = []
        for rnd in range(w.rounds):
            for i, point in enumerate(w.points()):
                # Alternate which side goes first so slow drift (cache warmth,
                # neighbours, thermal) does not land on one side.
                order = manifests.SIDES if (rnd + i) % 2 == 0 else manifests.SIDES[::-1]
                for side in order:
                    out = (
                        results
                        / "runs"
                        / topology
                        / side
                        / point.label
                        / f"r{rnd}.json"
                    )
                    if _valid_run_file(out):
                        log.info("skip %s (already measured)", out.relative_to(results))
                    else:
                        pending.append((rnd, point, side, out))
        if not pending:
            return 0
        urls = {side: self.url(topology, side) for side in manifests.SIDES}
        headers = manifests.request_headers(self.cfg, topology)
        if not self.args.no_prime:
            for side in manifests.SIDES:
                self._prime(topology, side, urls[side], headers)
        failures = 0
        for rnd, point, side, out in pending:
            if not self._measure_point(
                topology, side, point, rnd, urls[side], headers, out
            ):
                failures += 1
        return failures

    def _prime(
        self, topology: str, side: str, url: str, headers: dict[str, str]
    ) -> None:
        """One unrecorded run at the highest load point before round 0.

        Without it the first measured point of each side pays for cold
        connection pools and lazily initialised code paths, and that cost
        lands on whichever side happens to run first. Its seed is disjoint
        from the measurement seeds, so it does not pre-warm their prefixes.
        """
        w = self.workload
        point = max(w.points())
        artifact_dir = f"{RESULTS_ROOT}/{topology}/{side}/prime"
        cmd = aiperf_command(
            model=self.cfg.model,
            url=url,
            headers=headers,
            workload=w,
            point=point,
            seed=w.seed + 10_000,
            artifact_dir=artifact_dir,
        )
        log.info(
            "priming %-6s %-8s %s (not recorded)", topology, side, point.describe()
        )
        try:
            self.loadgen.run(cmd, artifact_dir, w.run_timeout_s)
        except (KubeError, subprocess.TimeoutExpired) as e:
            log.warning("priming %s/%s failed: %s", topology, side, e)

    def _measure_point(
        self,
        topology: str,
        side: str,
        point: LoadPoint,
        rnd: int,
        url: str,
        headers: dict[str, str],
        out: pathlib.Path,
    ) -> bool:
        seed = self.workload.seed + rnd
        artifact_dir = f"{RESULTS_ROOT}/{topology}/{side}/{point.label}/r{rnd}"
        cmd = aiperf_command(
            model=self.cfg.model,
            url=url,
            headers=headers,
            workload=self.workload,
            point=point,
            seed=seed,
            artifact_dir=artifact_dir,
        )
        for attempt in (1, 2):
            started = _now()
            log.info(
                "round %d  %-6s %-8s %-10s attempt %d",
                rnd,
                topology,
                side,
                point.describe(),
                attempt,
            )
            try:
                result = self.loadgen.run(
                    cmd, artifact_dir, self.workload.run_timeout_s
                )
            except (KubeError, subprocess.TimeoutExpired) as e:
                log.warning("run failed: %s", e)
                continue
            result.update(
                topology=topology,
                side=side,
                load=point.label,
                round=rnd,
                seed=seed,
                url=url,
                expected_requests=self.workload.requests_for(point),
                started=started,
                finished=_now(),
            )
            # Written even when invalid, so the report can list it under
            # "Excluded runs"; a later resume measures it again.
            _write_atomic(out, json.dumps(result))
            if not report.is_valid(result):
                log.warning(
                    "  invalid: %d of %d requests recorded, %d errors",
                    result["records"],
                    result["expected_requests"],
                    result["errors"],
                )
                continue
            # A failure marker left by an earlier attempt or run no longer applies.
            out.with_suffix(".error").unlink(missing_ok=True)
            ttft = result["samples"]["time_to_first_token"]
            log.info(
                "  ok: %d requests, %d errors, TTFT p50 %.2f ms, %.0f output tok/s",
                result["records"],
                result["errors"],
                _median(ttft),
                result.get("output_token_throughput") or 0,
            )
            return True
        err = out.with_suffix(".error")
        err.parent.mkdir(parents=True, exist_ok=True)
        err.write_text(f"failed twice at {_now()}\n")
        return False

    # -- metadata ----------------------------------------------------------

    def _workload_meta(self) -> dict[str, Any]:
        # Round-tripped through JSON so tuples compare equal to the lists a
        # stored meta.json reads back as.
        return json.loads(
            json.dumps(
                {
                    k: v
                    for k, v in dataclasses.asdict(self.workload).items()
                    if k not in ("run_timeout_s", "rounds")
                }
            )
        )

    def write_meta(self, results: pathlib.Path) -> None:
        results.mkdir(parents=True, exist_ok=True)
        path = results / "meta.json"
        if path.exists():
            # Resuming. Mixing rounds measured under different settings
            # into one comparison would be silently wrong, so refuse.
            old = json.loads(path.read_text())
            if (
                old.get("workload") != self._workload_meta()
                or old.get("config")
                != json.loads(json.dumps(dataclasses.asdict(self.cfg)))
                or old.get("backend", "kubernetes") != self.backend.name
            ):
                raise KubeError(
                    f"{results} was measured with a different workload, deployment "
                    "config, or backend; use a new --results directory"
                )
            return
        meta: dict[str, Any] = {
            "started": _now(),
            **self.backend.environment(),
            "git_sha": _git_sha(),
            "frontend_image": self.cfg.frontend_image,
            "worker_image": self.cfg.worker_image,
            "aiperf_version": self.loadgen.aiperf_version(),
            "model": self.cfg.model,
            "workers": {
                "agg": self.cfg.agg_workers,
                "prefill": self.cfg.prefill_workers,
                "decode": self.cfg.decode_workers,
                "block_size": self.cfg.block_size,
                "speedup_ratio": self.cfg.speedup_ratio,
            },
            "workload": self._workload_meta(),
            "rounds": self.workload.rounds,
            "config": json.loads(json.dumps(dataclasses.asdict(self.cfg))),
        }
        path.write_text(json.dumps(meta, indent=2))


# -- commands --------------------------------------------------------------


def cmd_preflight(bench: Bench) -> int:
    problems = bench.preflight()
    if problems:
        for p in problems:
            log.error("preflight: %s", p)
        log.error("Install prerequisites with benchmarks/gateway_ab/setup_cluster.sh")
        return EXIT_PREFLIGHT
    log.info(
        "preflight OK (%s backend, %s)", bench.backend.name, bench.backend.describe()
    )
    return 0


def cmd_render(bench: Bench) -> int:
    out = pathlib.Path(bench.args.out)
    out.mkdir(parents=True, exist_ok=True)
    for obj in manifests.build_all(bench.cfg, bench.topologies):
        path = out / f"{obj['kind'].lower()}-{obj['metadata']['name']}.json"
        path.write_text(json.dumps(obj, indent=2) + "\n")
        print(path)
    return 0


def cmd_deploy(bench: Bench) -> int:
    rc = cmd_preflight(bench)
    if rc:
        return rc
    bench.deploy_loadgen()
    for topology in bench.topologies:
        bench.deploy_topology(topology)
    return 0


def cmd_run(bench: Bench) -> int:
    results = _results_dir(bench.args)
    bench.write_meta(results)
    failures = 0
    for topology in bench.topologies:
        failures += bench.measure_topology(topology, results)
    log.info("results in %s (%d failed points)", results, failures)
    rc = cmd_report(bench, results)
    return rc or (1 if failures else 0)


def cmd_report(bench: Bench, results: pathlib.Path | None = None) -> int:
    results = results or _results_dir(bench.args, must_exist=True)
    summary = report.build_summary(results, min_effect_pct=bench.args.min_effect)
    (results / "summary.json").write_text(json.dumps(summary, indent=2))
    md = report.render_markdown(summary)
    (results / "report.md").write_text(md + "\n")
    print(md)
    log.info("wrote %s and %s", results / "summary.json", results / "report.md")
    if bench.args.max_ttft_regression is not None:
        found = report.regressions(summary, bench.args.max_ttft_regression)
        for r in found:
            log.error("regression: %s", r)
        if found:
            return EXIT_REGRESSION
    # A side that failed outright leaves no comparison behind, and must not
    # read as a pass.
    gaps = report.incomplete(summary)
    for g in gaps:
        log.error("incomplete: %s", g)
    return 1 if gaps else 0


def cmd_teardown(bench: Bench) -> int:
    bench.teardown_all()
    return 0


def cmd_trend(args: argparse.Namespace) -> int:
    """Needs no cluster, so it runs before a Bench (and its kubectl) exists."""
    if not args.history:
        raise KubeError("trend needs --history <directory of result runs>")
    print(report.render_trend(pathlib.Path(args.history), quantile=args.quantile))
    return 0


def cmd_all(bench: Bench) -> int:
    rc = cmd_preflight(bench)
    if rc:
        return rc
    results = _results_dir(bench.args)
    bench.deploy_loadgen()
    bench.write_meta(results)
    failed = []
    try:
        for topology in bench.topologies:
            # One topology failing must not discard what the others measured.
            try:
                bench.deploy_topology(topology)
                if bench.measure_topology(topology, results):
                    failed.append(topology)
            except (KubeError, subprocess.TimeoutExpired) as e:
                log.error("%s failed: %s", topology, e)
                failed.append(topology)
            if not bench.args.keep:
                # Free the cluster for the next topology so its pods do not
                # compete with this one's leftovers.
                _best_effort(bench.teardown_topology, topology)
    finally:
        if not bench.args.keep:
            _best_effort(bench.teardown_all)
    rc = cmd_report(bench, results)
    if failed:
        log.error("incomplete topologies: %s", ", ".join(failed))
    return rc or (1 if failed else 0)


COMMANDS = {
    "preflight": cmd_preflight,
    "render": cmd_render,
    "deploy": cmd_deploy,
    "run": cmd_run,
    "report": cmd_report,
    "teardown": cmd_teardown,
    "trend": cmd_trend,
    "all": cmd_all,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python3 -m benchmarks.gateway_ab",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("command", choices=sorted(COMMANDS))
    p.add_argument("-v", "--verbose", action="store_true")

    b = p.add_argument_group("backend")
    b.add_argument(
        "--backend",
        choices=("kubernetes", "local"),
        default="kubernetes",
        help="kubernetes: DGDs behind a real Gateway; local: Docker on this host, "
        "no Kubernetes (see README)",
    )
    b.add_argument(
        "--epp-binary",
        help="local: EPP binary to run instead of the frontend image's /epp; it must "
        "support DYN_EPP_STATIC_ENDPOINT",
    )
    b.add_argument(
        "--envoy-image",
        default="envoyproxy/envoy:v1.35.3",
        help="local: Envoy image for the gateway",
    )

    k = p.add_argument_group("cluster (kubernetes backend)")
    k.add_argument("--context", help="kubectl context (default: current context)")
    k.add_argument("--namespace", default="gateway-ab")
    k.add_argument("--gateway-name", default="inference-gateway")
    k.add_argument("--gateway-namespace", default="agentgateway-system")
    k.add_argument("--gateway-url", help="override the in-cluster Gateway URL")
    k.add_argument("--hf-secret", help="Secret with HF_TOKEN, for gated models")
    k.add_argument("--image-pull-secret")
    k.add_argument("--ready-timeout", type=int, default=900, help="seconds per DGD")
    k.add_argument("--keep", action="store_true", help="all: leave deployments up")

    d = p.add_argument_group("deployment (identical on both sides)")
    d.add_argument(
        "--topologies",
        type=_csv(str, manifests.TOPOLOGIES),
        default=list(manifests.TOPOLOGIES),
    )
    d.add_argument("--model", default="Qwen/Qwen3-0.6B")
    d.add_argument("--dynamo-version", default="1.5.1")
    d.add_argument(
        "--frontend-image",
        help="Frontend/EPP image (default: dynamo-frontend:<version>)",
    )
    d.add_argument(
        "--worker-image", help="mocker image (default: dynamo-planner:<version>)"
    )
    d.add_argument("--name-prefix", default="ab")
    d.add_argument("--agg-workers", type=int, default=2)
    d.add_argument("--prefill-workers", type=int, default=2)
    d.add_argument("--decode-workers", type=int, default=2)
    d.add_argument("--block-size", type=int, default=16)
    d.add_argument("--speedup-ratio", type=float, default=1.0)
    d.add_argument(
        "--mocker-arg", action="append", help="extra mocker argument (repeatable)"
    )
    d.add_argument(
        "--router-cpu", default="2", help="CPU request for the Frontend and the EPP"
    )
    d.add_argument("--worker-cpu", default="1")
    d.add_argument("--sidecar-cpu", default="1")
    d.add_argument("--loadgen-cpu", default="4")

    w = p.add_argument_group("workload (identical on both sides)")
    w.add_argument(
        "--request-rate",
        type=_csv(_positive(float)),
        help="open-loop Poisson load points in requests/s (default: 6,14)",
    )
    w.add_argument(
        "--concurrency",
        type=_csv(_positive(int)),
        help="closed-loop load points instead of --request-rate, e.g. 8,32",
    )
    w.add_argument("--isl", type=int, default=1024)
    w.add_argument("--osl", type=int, default=128)
    w.add_argument("--prefix-length", type=int, default=512)
    w.add_argument("--num-prefixes", type=int, default=8)
    w.add_argument(
        "--requests",
        type=int,
        default=0,
        help="per point; 0 = about 40 s of arrivals (closed loop: max(200, 10 x c))",
    )
    w.add_argument(
        "--warmup-requests",
        type=int,
        default=0,
        help="0 = about 5 s of arrivals (closed loop: max(10, 2 x c))",
    )
    w.add_argument("--rounds", type=_positive(int), default=3)
    w.add_argument("--seed", type=int, default=42)
    w.add_argument(
        "--run-timeout", type=int, default=1800, help="seconds per aiperf run"
    )
    w.add_argument(
        "--no-prime",
        action="store_true",
        help="skip the unrecorded priming run per side before round 0",
    )

    r = p.add_argument_group("results")
    r.add_argument(
        "--results",
        help=f"results directory (default: {DEFAULT_OUTPUT}/<UTC time>)",
    )
    r.add_argument(
        "--out",
        default=f"{DEFAULT_OUTPUT}/manifests",
        help="render: output directory",
    )
    r.add_argument(
        "--min-effect",
        type=float,
        default=2.0,
        help="percent below which a delta is 'same'",
    )
    r.add_argument(
        "--max-ttft-regression",
        type=float,
        help=f"exit {EXIT_REGRESSION} if B's TTFT is worse than A by more than this percent",
    )
    r.add_argument(
        "--history", help="trend: directory whose subdirectories are result runs"
    )
    r.add_argument(
        "--quantile",
        default="p50",
        choices=("mean", "p50", "p90", "p99"),
        help="trend: TTFT statistic to tabulate",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.concurrency and args.request_rate:
        parser.error("--concurrency and --request-rate are mutually exclusive")
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        if args.command == "trend":
            return cmd_trend(args)
        return COMMANDS[args.command](Bench(args))
    except (KubeError, subprocess.TimeoutExpired) as e:
        log.error("%s", e)
        return 1


def _csv(kind, choices=None):
    def parse(text: str):
        items = [kind(x.strip()) for x in text.split(",") if x.strip()]
        if choices is not None:
            bad = [x for x in items if x not in choices]
            if bad:
                raise argparse.ArgumentTypeError(
                    f"invalid {bad}; choose from {choices}"
                )
        if not items:
            raise argparse.ArgumentTypeError("empty list")
        return items

    return parse


def _positive(kind):
    def parse(text: str):
        value = kind(text)
        if value <= 0:
            raise argparse.ArgumentTypeError(f"must be positive: {text}")
        return value

    return parse


def _results_dir(args: argparse.Namespace, must_exist: bool = False) -> pathlib.Path:
    if args.results:
        path = pathlib.Path(args.results)
    elif must_exist:
        raise KubeError("--results is required for this command")
    else:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = pathlib.Path(DEFAULT_OUTPUT) / stamp
        args.results = str(path)
    if must_exist and not path.exists():
        raise KubeError(f"results directory {path} does not exist")
    return path


def _best_effort(fn, *args) -> None:
    """Run a cleanup step without letting its failure hide earlier results."""
    try:
        fn(*args)
    except (KubeError, subprocess.TimeoutExpired) as e:
        log.error("cleanup failed: %s", e)


def _valid_run_file(path: pathlib.Path) -> bool:
    """Whether a run file exists and holds a valid measurement."""
    try:
        return report.is_valid(json.loads(path.read_text()))
    except (OSError, ValueError):
        return False


def _write_atomic(path: pathlib.Path, text: str) -> None:
    """Write via rename so an interrupted run never leaves a truncated file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=pathlib.Path(__file__).parent,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _median(values: list[float]) -> float:
    return report.stats.percentile(values, 50) if values else float("nan")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    sys.exit(main())
