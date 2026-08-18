#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run an Aggregated / colocated EPD Cartesian sweep with AIPerf."""

from __future__ import annotations

import argparse
import contextlib
import functools
import http.server
import itertools
import json
import math
import os
import shutil
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from make_payload import build_payload, load_tokenizer, write_dataset

SCRIPT_DIR = Path(__file__).resolve().parent
EXPECTED_AIPERF_VERSION = "0.10.0"
TOPOLOGY_ALIASES = {
    "aggregate": "aggregate",
    "agg": "aggregate",
    "epd": "epd",
}
BACKEND_LAUNCHERS = {
    "vllm": "run_vllm.sh",
    "sglang": "run_sglang.sh",
}


def decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


@dataclass(frozen=True)
class Cell:
    backend: str
    topology: str
    image_count: int
    image_token_budget: int
    isl: int
    osl: int
    qps: Decimal

    @property
    def service_key(self) -> tuple[str, str, int]:
        return self.backend, self.topology, self.image_token_budget

    @property
    def name(self) -> str:
        qps = decimal_text(self.qps).replace(".", "p")
        return f"img{self.image_count}-isl{self.isl}-osl{self.osl}-qps{qps}"


def _tokens(raw: Sequence[str]) -> list[str]:
    values = [part.strip() for item in raw for part in item.split(",")]
    if not values or any(not value for value in values):
        raise ValueError("sweep lists cannot contain empty values")
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate sweep value: {values}")
    return values


def _positive_ints(raw: Sequence[str], name: str, minimum: int = 1) -> list[int]:
    try:
        values = [int(value) for value in _tokens(raw)]
    except ValueError as error:
        raise ValueError(f"{name} values must be integers") from error
    if any(value < minimum for value in values):
        raise ValueError(f"{name} values must be >= {minimum}")
    return values


def build_cells(args: argparse.Namespace) -> list[Cell]:
    backends = [value.lower() for value in _tokens(args.backend)]
    if any(value not in {"vllm", "sglang"} for value in backends):
        raise ValueError("backend must be vllm or sglang")
    try:
        topologies = [
            TOPOLOGY_ALIASES[value.lower()] for value in _tokens(args.topology)
        ]
    except KeyError as error:
        raise ValueError("topology must be aggregate/epd (alias: agg)") from error
    if len(topologies) != len(set(topologies)):
        raise ValueError("duplicate topology after alias normalization")
    try:
        qps_values = [Decimal(value).normalize() for value in _tokens(args.qps)]
    except InvalidOperation as error:
        raise ValueError("QPS values must be decimal numbers") from error
    if any(not value.is_finite() or value <= 0 for value in qps_values):
        raise ValueError("QPS values must be finite and positive")

    counts = _positive_ints(args.image_count, "image-count")
    budgets = _positive_ints(args.image_token_budget, "image-token-budget", 64)
    isls = _positive_ints(args.isl, "isl")
    osls = _positive_ints(args.osl, "osl")
    return [
        Cell(backend, topology, count, budget, isl, osl, qps)
        for backend, topology, budget, count, isl, osl, qps in itertools.product(
            backends, topologies, budgets, counts, isls, osls, qps_values
        )
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Every sweep axis accepts one value, comma lists, or space lists.",
    )
    for option in (
        "backend",
        "topology",
        "image-count",
        "image-token-budget",
        "isl",
        "osl",
        "qps",
    ):
        parser.add_argument(f"--{option}", nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--tokenizer", help="defaults to --model")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--aiperf-bin", default=os.environ.get("AIPERF_BIN", "aiperf"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("DYN_HTTP_PORT", "8000"))
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


class ImageServer(http.server.ThreadingHTTPServer):
    request_queue_size = 128
    daemon_threads = True


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        del format, args


@contextlib.contextmanager
def image_origin(image_dir: Path) -> Iterator[str]:
    handler = functools.partial(QuietHandler, directory=str(image_dir))
    server = ImageServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def wait_for_model(
    endpoint: str, model: str, process: subprocess.Popen[str], timeout: int
) -> None:
    url = endpoint + "/v1/models"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"model launcher exited with {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if model in response.read().decode(errors="replace"):
                    return
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(2)
    raise TimeoutError(f"model {model!r} was not ready at {url} within {timeout}s")


def launcher_command(key: tuple[str, str, int], args: argparse.Namespace) -> list[str]:
    backend, topology, budget = key
    launcher = SCRIPT_DIR / "scripts" / BACKEND_LAUNCHERS[backend]
    return [
        "bash",
        str(launcher),
        topology,
        "--model",
        args.model,
        "--served-model-name",
        args.served_model_name,
        "--image-token-budget",
        str(budget),
    ]


@contextlib.contextmanager
def managed_service(
    key: tuple[str, str, int], args: argparse.Namespace, output_dir: Path
) -> Iterator[str]:
    command = launcher_command(key, args)
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(DYN_HTTP_PORT=str(args.port), DYN_LOG_DIR=str(model_dir))
    env.update(service_port_env(args.port))
    log_path = output_dir / "model-launcher.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        endpoint = f"http://127.0.0.1:{args.port}"
        try:
            wait_for_model(
                endpoint,
                args.served_model_name,
                process,
                int(os.environ.get("DYN_MODEL_READY_TIMEOUT_SECONDS", "3600")),
            )
            yield endpoint
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=10)


def measured_requests(qps: Decimal) -> int:
    return max(24, math.ceil(float(qps * 60)) + 1)


def service_port_env(http_port: int) -> dict[str, str]:
    """Derive one collision-free worker-port block from the HTTP port."""
    ports = {
        **{f"DYN_SYSTEM_PORT{i}": http_port + 80 + i for i in range(1, 4)},
        **{
            f"DYN_VLLM_NIXL_SIDE_CHANNEL_PORT{i}": http_port + 12096 + i
            for i in range(1, 4)
        },
        **{f"DYN_WORKER_PORT{i}": http_port + 22000 + i for i in range(1, 4)},
        **{f"DYN_NCCL_PORT{i}": http_port + 23000 + i for i in range(1, 4)},
    }
    if max(ports.values()) > 65535:
        raise ValueError("--port is too high to derive backend worker ports")
    return {name: str(port) for name, port in ports.items()}


def aiperf_command(
    *,
    args: argparse.Namespace,
    cell: Cell,
    endpoint: str,
    dataset: Path,
    artifact_dir: Path,
) -> list[str]:
    return [
        args.aiperf_bin,
        "profile",
        "--artifact-dir",
        str(artifact_dir),
        "--model",
        args.served_model_name,
        "--tokenizer",
        "builtin",
        "--endpoint-type",
        "chat",
        "--endpoint",
        "/v1/chat/completions",
        "--streaming",
        "--url",
        endpoint,
        "--input-file",
        str(dataset),
        "--custom-dataset-type",
        "raw_payload",
        "--use-server-token-count",
        "--export-level",
        "raw",
        "--request-rate",
        decimal_text(cell.qps),
        "--request-count",
        str(measured_requests(cell.qps)),
        "--arrival-pattern",
        "constant",
        "--warmup-arrival-pattern",
        "constant",
        "--concurrency",
        "64",
        "--workers-max",
        "64",
        "--record-processors",
        "8",
        "--warmup-request-count",
        "4",
        "--request-timeout-seconds",
        "600",
        "--random-seed",
        "202608160",
        "--dataset-sampling-strategy",
        "sequential",
        "--ui",
        "none",
    ]


def _metric(report: Mapping[str, Any], name: str, field: str = "avg") -> Any:
    value = report.get(name)
    return value.get(field) if isinstance(value, Mapping) else value


def validate_result(path: Path, cell: Cell) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    expected = measured_requests(cell.qps)
    errors = sum(
        int(item.get("count", 0)) if isinstance(item, Mapping) else 1
        for item in (report.get("error_summary") or [])
    )
    failures = []
    if _metric(report, "request_count") != expected:
        failures.append(
            f"request_count={_metric(report, 'request_count')} expected={expected}"
        )
    if errors:
        failures.append(f"errors={errors}")
    if report.get("was_cancelled"):
        failures.append("was_cancelled=true")
    if report.get("aiperf_version") != EXPECTED_AIPERF_VERSION:
        failures.append(f"aiperf_version={report.get('aiperf_version')!r}")
    for field in ("avg", "min", "max"):
        if _metric(report, "output_sequence_length", field) != cell.osl:
            failures.append(
                f"osl_{field}={_metric(report, 'output_sequence_length', field)}"
            )
    if failures:
        raise RuntimeError("AIPerf result gate failed: " + "; ".join(failures))
    return {
        **asdict(cell),
        "qps": decimal_text(cell.qps),
        "request_count": expected,
        "achieved_qps": _metric(report, "request_throughput"),
        "ttft_ms": _metric(report, "time_to_first_token"),
        "e2e_ms": _metric(report, "request_latency"),
    }


def run_cell(
    *,
    args: argparse.Namespace,
    cell: Cell,
    tokenizer: Any,
    image_url_root: str,
    endpoint: str,
    group_dir: Path,
) -> None:
    cell_dir = group_dir / cell.name
    report_path = cell_dir / "profile_export_aiperf.json"
    if cell_dir.exists():
        raise FileExistsError(f"cell output already exists: {cell_dir}")
    payload = build_payload(
        tokenizer=tokenizer,
        backend=cell.backend,
        model=args.served_model_name,
        image_dir=args.image_dir,
        image_url_root=image_url_root,
        image_count=cell.image_count,
        image_token_budget=cell.image_token_budget,
        isl=cell.isl,
        osl=cell.osl,
    )
    dataset = cell_dir / "input.jsonl"
    write_dataset(dataset, payload)
    command = aiperf_command(
        args=args,
        cell=cell,
        endpoint=endpoint,
        dataset=dataset,
        artifact_dir=cell_dir,
    )
    print(f"RUN  {cell.backend}/{cell.topology}/{cell.name}", flush=True)
    with (cell_dir / "aiperf.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    summary = validate_result(report_path, cell)
    (cell_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"PASS TTFT={summary['ttft_ms']} E2E={summary['e2e_ms']}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cells = build_cells(args)
    groups = [
        (key, list(items))
        for key, items in itertools.groupby(cells, key=lambda cell: cell.service_key)
    ]
    print(f"Planned {len(cells)} cells in {len(groups)} service groups")
    for cell in cells:
        print(
            f"  {cell.backend}/{cell.topology}/cap{cell.image_token_budget}/{cell.name}"
        )
    if args.dry_run:
        return 0

    if not args.image_dir.is_dir():
        raise FileNotFoundError(args.image_dir)
    if not 1 <= args.port <= 65_535:
        raise ValueError("port must be in 1..65535")
    if shutil.which(args.aiperf_bin) is None:
        raise FileNotFoundError(f"AIPerf executable not found: {args.aiperf_bin}")
    version = subprocess.run(
        [args.aiperf_bin, "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if version != EXPECTED_AIPERF_VERSION:
        raise RuntimeError(
            f"AIPerf version must be {EXPECTED_AIPERF_VERSION}, got {version!r}"
        )
    tokenizer = load_tokenizer(args.tokenizer or args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with image_origin(args.image_dir) as root:
        for key, group_cells in groups:
            backend, topology, budget = key
            group_dir = args.output_dir / f"{backend}-{topology}-cap{budget}"
            group_dir.mkdir(parents=True, exist_ok=True)
            with managed_service(key, args, group_dir) as endpoint:
                for cell in group_cells:
                    run_cell(
                        args=args,
                        cell=cell,
                        tokenizer=tokenizer,
                        image_url_root=root,
                        endpoint=endpoint,
                        group_dir=group_dir,
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
