# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import importlib.util
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = load("bench")
loadgen_run = load("loadgen_run")

DIGEST = "@sha256:" + "a" * 64
REGION = {
    "region": "us-east-2",
    "site": "aws-ohio",
    "short": "ohio",
    "clusterName": "ohio-cluster",
    "plannerImage": "planner" + DIGEST,
    "relayImage": "relay" + DIGEST,
    "controlImage": "control" + DIGEST,
}
CONFIG = {
    "namespace": "global-router-bench",
    "model": "Qwen/Qwen3-0.6B",
    "regions": [REGION, {**REGION, "region": "ap-northeast-1", "site": "aws-tokyo"}],
    "pools": [
        {"stem": "fast", "mockerArgs": ["--speedup-ratio", "5"]},
        {"stem": "slow", "mockerArgs": ["--speedup-ratio", "2"]},
    ],
    "workerReplicas": 8,
    "frontendCpu": "2",
    "routerReplicas": 2,
    "routerCpu": "2",
    # Any file stands in for the built shim.
    "nodelayShim": str(HERE / "nodelay.c"),
}


class BenchTest(unittest.TestCase):
    def test_backend_holds_both_pools_and_one_namespace(self):
        items = bench.backend_manifest(CONFIG, REGION)["items"]
        kinds = [item["kind"] for item in items]
        self.assertEqual(kinds.count("Namespace"), 1)
        workers = {
            item["metadata"]["name"]: item["spec"]
            for item in items
            if item["metadata"]["name"].endswith("-worker")
        }
        self.assertEqual(set(workers), {"fast-worker", "slow-worker"})
        for stem, spec in workers.items():
            pod = spec["template"]["spec"]
            self.assertEqual(spec["replicas"], 8)
            self.assertEqual(pod["nodeSelector"], bench.NODE_SELECTOR)
            self.assertEqual(pod["tolerations"], [bench.TOLERATION])
            ratio = "5" if stem == "fast-worker" else "2"
            self.assertEqual(
                pod["containers"][0]["args"][-2:], ["--speedup-ratio", ratio]
            )

    def test_every_gds_sees_every_regions_pools(self):
        pools = bench.routing_config(CONFIG)["pools"]
        self.assertEqual(
            {(p["site_id"], p["dgd_name"]) for p in pools},
            {(s, d) for s in ("aws-ohio", "aws-tokyo") for d in ("fast", "slow")},
        )

    def test_control_passes_the_policy_to_the_ingress(self):
        items = bench.control_manifest(CONFIG, REGION, "least-in-flight")["items"]
        ingress = next(
            item
            for item in items
            if item["kind"] == "Deployment"
            and item["metadata"]["name"] == "global-router"
        )
        container = ingress["spec"]["template"]["spec"]["containers"][0]
        self.assertEqual(container["args"][-2:], ["--router-policy", "least-in-flight"])
        self.assertEqual(container["resources"]["limits"]["cpu"], "2")
        self.assertEqual(ingress["spec"]["replicas"], 2)

    def test_only_http_servers_get_the_tcp_settings(self):
        for items, servers in (
            (
                bench.backend_manifest(CONFIG, REGION)["items"],
                {"fast-frontend", "slow-frontend"},
            ),
            (bench.control_manifest(CONFIG, REGION, None)["items"], {"global-router"}),
        ):
            (shim,) = [item for item in items if "binaryData" in item]
            self.assertEqual(
                base64.b64decode(shim["binaryData"]["libnodelay.so"]),
                Path(CONFIG["nodelayShim"]).read_bytes(),
            )
            for item in items:
                if item["kind"] != "Deployment":
                    continue
                pod = item["spec"]["template"]["spec"]
                tuned = item["metadata"]["name"] in servers
                env = {
                    e["name"]: e["value"] for e in pod["containers"][0].get("env", [])
                }
                self.assertEqual("LD_PRELOAD" in env, tuned, item["metadata"]["name"])
                self.assertEqual(
                    [c["name"] for c in pod.get("initContainers", [])],
                    ["tcp-sysctl"] if tuned else [],
                )

    def test_nodegroup_copies_the_gpu_groups(self):
        settings = {"instanceType": "m7i.4xlarge", "count": 2, "diskGiB": 100}
        gpu = {"nodeRole": "arn:role", "subnets": ["subnet-1"]}
        for data, expected in (
            ({"SecurityGroupIds": ["sg-1", "sg-2"]}, ["sg-1", "sg-2"]),
            ({"NetworkInterfaces": [{"Groups": ["sg-1", "sg-2"]}]}, ["sg-1", "sg-2"]),
            # No groups: EKS attaches the cluster security group to both.
            ({}, None),
        ):
            resources = bench.nodegroup_template(REGION, settings, gpu, data)[
                "Resources"
            ]
            launch = resources["LaunchTemplate"]["Properties"]["LaunchTemplateData"]
            self.assertEqual(launch.get("SecurityGroupIds"), expected)
            group = resources["Nodegroup"]["Properties"]
            self.assertEqual(group["NodeRole"], "arn:role")
            self.assertEqual(group["Labels"], bench.NODE_SELECTOR)


class WindowSummaryTest(unittest.TestCase):
    def record(self, start_s, end_s, ttft, error=None):
        return {
            "metadata": {
                "phase_kind": "profiling",
                "request_start_ns": int(start_s * 1e9),
                "request_end_ns": int(end_s * 1e9),
            },
            "metrics": {
                "time_to_first_token": {"value": ttft},
                "output_sequence_length": {"value": 10},
            },
            "error": error,
        }

    def test_counts_requests_finishing_inside_the_window(self):
        records = [
            self.record(100, 105, 1.0),  # before the window
            self.record(100, 115, 2.0),
            self.record(105, 125, 3.0),
            self.record(110, 140, 4.0),  # after the window
            self.record(110, 120, 9.0, error={"code": 503}),
        ]
        summary = loadgen_run.window_summary(records, start=110, end=130)
        self.assertEqual(summary["profiling_started"], 100)
        self.assertEqual(summary["completed_requests"], 2)
        self.assertEqual(summary["profiling_errors"], 1)
        self.assertEqual(summary["requests_per_second"], 0.1)
        self.assertEqual(summary["output_tokens_per_second"], 1.0)
        self.assertEqual(summary["ttft_ms"]["p50"], 3.0)


if __name__ == "__main__":
    unittest.main()
