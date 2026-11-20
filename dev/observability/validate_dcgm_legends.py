# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporary remote CI evidence requested on PR #17558; remove after recording it."""

import functools
import http.server
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CASES = [
    ("lowercase", {"hostname": "node-a"}, "node-a"),
    ("uppercase", {"Hostname": "node-a"}, "node-a"),
    ("both_same", {"hostname": "node-a", "Hostname": "node-a"}, "node-a"),
    ("both_different", {"hostname": "node-a", "Hostname": "node-old"}, "node-a"),
    ("empty_lowercase", {"hostname": "", "Hostname": "node-a"}, "node-a"),
    ("two_nodes_a", {"hostname": "node-a"}, "node-a"),
    ("two_nodes_b", {"hostname": "node-b"}, "node-b"),
]
METRICS = {
    "DCGM_FI_DEV_GPU_UTIL": 42,
    "DCGM_FI_DEV_MEM_COPY_UTIL": 7,
    "DCGM_FI_DEV_FB_USED": 8192,
    "DCGM_FI_PROF_NVLINK_TX_BYTES": 2000000000,
    "DCGM_FI_PROF_NVLINK_RX_BYTES": 3000000000,
}
EXPECTED_VALUES = [42, 7, 8, 5]
GRAFANA = (
    "https://raw.githubusercontent.com/grafana/grafana/v12.2.0/pkg/promlib/querydata/"
)


def download(url, path):
    with urllib.request.urlopen(url, timeout=60) as response:
        path.write_bytes(response.read())


def formatter(directory, rows):
    """Execute Grafana's unchanged backend functions with minimal data type stubs."""
    download(GRAFANA + "response.go", directory / "response.txt")
    download(GRAFANA + "request.go", directory / "request.txt")
    response = (directory / "response.txt").read_text()
    request = (directory / "request.txt").read_text()
    functions = []
    for name in ["metricNameFromLabels", "getName"]:
        match = re.search(r"(?ms)^func " + name + r"\(.*?^}\n", response)
        if match is None:
            raise ValueError(f"Grafana function {name} not found")
        functions.append(match.group())
    declarations = [
        re.search(r"(?m)^const legendFormatAuto = .*", request).group(),
        re.search(r"(?m)^var legendFormatRegexp = .*", request).group(),
    ]
    (directory / "go.mod").write_text("module legendcheck\n\ngo 1.23\n")
    for package, stub in [
        ("models", "type Query struct { Expr, LegendFormat string }"),
        ("data", "type Field struct { Labels map[string]string }"),
    ]:
        (directory / package).mkdir()
        (directory / package / "types.go").write_text(f"package {package}\n{stub}\n")
    main = """package main
import ("encoding/json"; "fmt"; "os"; "regexp"; "sort"; "strings"; "legendcheck/models"; "legendcheck/data")
type Row struct { Case, Panel, Legend, Expr, Want string; Labels map[string]string }
func main() {
    var rows []Row
    if err := json.NewDecoder(os.Stdin).Decode(&rows); err != nil { panic(err) }
    for _, row := range rows {
        got := getName(&models.Query{Expr: row.Expr, LegendFormat: row.Legend}, &data.Field{Labels: row.Labels})
        if got != row.Want { panic(fmt.Sprintf("%s/%s: got %q want %q", row.Case, row.Panel, got, row.Want)) }
        fmt.Printf("%s | %s | %s | %s\\n", row.Panel, row.Case, row.Labels["dcgm_host"], got)
    }
}
"""
    (directory / "main.go").write_text(main + "\n".join(declarations + functions))
    subprocess.run(
        ["go", "run", "."],
        cwd=directory,
        input=json.dumps(rows),
        text=True,
        check=True,
        timeout=120,
    )


class MetricsHandler(http.server.SimpleHTTPRequestHandler):
    def guess_type(self, path):
        return "text/plain; version=0.0.4"


def main():
    dev = json.loads(
        Path("dev/observability/grafana_dashboards/disagg-dashboard.json").read_text()
    )
    configmap = Path(
        "deploy/observability/grafana-disagg-dashboard-configmap.yaml"
    ).read_text()
    embedded = configmap.split("  disagg-dashboard.json: |-\n", 1)[1]
    embedded = "\n".join(line[4:] for line in embedded.splitlines())
    if json.loads(embedded) != dev:
        raise ValueError("The two complete dashboard definitions differ")
    targets = [
        (panel["title"], target)
        for panel in dev["panels"]
        for target in panel.get("targets", [])
        if "DCGM_" in target.get("expr", "")
    ]
    if len(targets) != 4:
        raise ValueError(f"Expected four DCGM targets, got {len(targets)}")
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        lines = []
        for case, labels, _ in CASES:
            labels = {"scenario": case, "gpu": "0", **labels}
            encoded = ",".join(
                f"{key}={json.dumps(value)}" for key, value in labels.items()
            )
            lines.extend(
                f"{metric}{{{encoded}}} {value}" for metric, value in METRICS.items()
            )
        (directory / "metrics").write_text("\n".join(lines) + "\n")
        server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0),
            functools.partial(MetricsHandler, directory=temporary),
        )
        threading.Thread(target=server.serve_forever, daemon=True).start()
        download(
            "https://github.com/prometheus/prometheus/releases/download/v3.4.1/prometheus-3.4.1.linux-amd64.tar.gz",
            directory / "prometheus.tar.gz",
        )
        with tarfile.open(directory / "prometheus.tar.gz") as archive:
            source = archive.extractfile("prometheus-3.4.1.linux-amd64/prometheus")
            with (directory / "prometheus").open("wb") as destination:
                shutil.copyfileobj(source, destination)
        (directory / "prometheus").chmod(0o755)
        (directory / "prometheus.yml").write_text(
            "global:\n  scrape_interval: 1s\nscrape_configs:\n"
            "  - job_name: dcgm\n    fallback_scrape_protocol: PrometheusText0.0.4\n    static_configs:\n"
            f"      - targets: ['127.0.0.1:{server.server_port}']\n"
        )
        # Prometheus chooses an available port; discover it from its startup log.
        with (directory / "prometheus.log").open("w") as log:
            process = subprocess.Popen(
                [
                    str(directory / "prometheus"),
                    "--config.file=" + str(directory / "prometheus.yml"),
                    "--storage.tsdb.path=" + str(directory / "tsdb"),
                    "--web.listen-address=127.0.0.1:0",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + 30
                while True:
                    match = re.search(
                        r"address=127\.0\.0\.1:(\d+)",
                        (directory / "prometheus.log").read_text(),
                    )
                    if match:
                        break
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError((directory / "prometheus.log").read_text())
                    time.sleep(0.2)
                api = "http://127.0.0.1:" + match.group(1) + "/api/v1/query?"
                rows = []
                for (panel, target), expected in zip(targets, EXPECTED_VALUES):
                    while True:
                        try:
                            with urllib.request.urlopen(
                                api + urllib.parse.urlencode({"query": target["expr"]}),
                                timeout=5,
                            ) as result:
                                payload = json.load(result)
                            if payload["status"] != "success":
                                raise ValueError(payload)
                            samples = payload["data"]["result"]
                            if len(samples) == len(CASES):
                                break
                        except urllib.error.URLError:
                            if process.poll() is not None:
                                raise RuntimeError(
                                    (directory / "prometheus.log").read_text()
                                )
                        if time.monotonic() > deadline:
                            targets_url = (
                                api.split("/api/v1/query?")[0] + "/api/v1/targets"
                            )
                            with urllib.request.urlopen(
                                targets_url, timeout=5
                            ) as result:
                                scrape_state = json.load(result)
                            raise TimeoutError(
                                f"Prometheus did not return all fixtures: {payload}; "
                                f"scrapes: {scrape_state}; log: "
                                + (directory / "prometheus.log").read_text()
                            )
                        time.sleep(0.2)
                    by_case = {
                        sample["metric"]["scenario"]: sample for sample in samples
                    }
                    for case, inputs, host in CASES:
                        sample = by_case[case]
                        labels = sample["metric"]
                        correct_value = float(sample["value"][1]) == expected
                        correct_host = labels.get("dcgm_host") == host
                        if not (correct_value and correct_host):
                            raise ValueError(
                                f"Wrong PromQL output for {panel}/{case}: {sample}"
                            )
                        if any(
                            labels.get(key, "") != value
                            for key, value in inputs.items()
                        ):
                            raise ValueError(f"PromQL changed input labels: {sample}")
                        rows.append(
                            {
                                "Case": case,
                                "Panel": panel,
                                "Labels": labels,
                                "Expr": target["expr"],
                                "Legend": target["legendFormat"],
                                "Want": "GPU 0 - " + host,
                            }
                        )
                print(
                    "Prometheus 3.4.1: 28 actual query results passed; values 42, 7, 8, 5 preserved.",
                    flush=True,
                )
                print(
                    "Complete dev JSON and ConfigMap dashboard definitions match.",
                    flush=True,
                )
                formatter(directory, rows)
            finally:
                process.terminate()
                process.wait(timeout=10)
                server.shutdown()


if __name__ == "__main__":
    main()
