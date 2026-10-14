# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise artifact retention without downloading models or starting GPU jobs."""

import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RECIPES = (
    "deepseek-v4-pro-agg",
    "deepseek-v4-pro-disagg",
    "kimi-k25-agg",
    "kimi-k25-disagg",
)


class RecipePerfArtifactsTest(unittest.TestCase):
    def setUp(self):
        workflow = yaml.safe_load(
            (ROOT / ".github/workflows/shared-recipe-perf.yml").read_text()
        )
        self.steps = workflow["jobs"]["perf"]["steps"]

    def lifecycle(self):
        render = next(
            step
            for step in self.steps
            if step.get("name") == "Render deployment and benchmark"
        )
        return render["env"]["PERF_LIFECYCLE"]

    def run_lifecycle(self, exit_code, collect=True):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            results = directory / "results"
            results.mkdir()
            (results / "profile.json").write_text('{"requests": 200}')
            (results / "inputs.json").write_text("sensitive prompts")
            env = dict(
                os.environ, AIPERF_OUTPUT_DIR=temporary, AIPERF_COLLECTION_TIMEOUT="2"
            )
            script = self.lifecycle() + f"\nexit {exit_code}\n"
            with subprocess.Popen(["bash", "-c", script], env=env) as process:
                try:
                    deadline = time.monotonic() + 5
                    while not (directory / "exit-code").exists():
                        self.assertIsNone(process.poll())
                        self.assertLess(time.monotonic(), deadline)
                        time.sleep(0.02)
                    self.assertEqual(
                        int((directory / "exit-code").read_text()), exit_code
                    )
                    self.assertIsNone(
                        process.poll(), "The pod must stay alive for collection"
                    )
                    self.assertTrue((results / "profile.json").exists())
                    self.assertFalse((results / "inputs.json").exists())
                    if collect:
                        (directory / "collected").touch()
                    result = process.wait(timeout=5)
                    self.assertEqual(result, exit_code if collect else exit_code or 1)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()

    def test_success_stays_alive_until_artifacts_are_collected(self):
        self.run_lifecycle(0)

    def test_failure_preserves_artifacts_and_exit_status(self):
        self.run_lifecycle(42)

    def test_collection_timeout_cannot_report_success(self):
        self.run_lifecycle(0, collect=False)

    def test_collection_timeout_preserves_benchmark_failure(self):
        self.run_lifecycle(42, collect=False)

    def test_workflow_exports_local_results_after_success_and_failure(self):
        run = next(
            step["run"] for step in self.steps if step.get("name") == "Run AIPerf"
        )
        collect = next(
            step["run"]
            for step in self.steps
            if step.get("name") == "Collect logs and perf artifacts"
        )
        for exit_code in (0, 42):
            with self.subTest(
                exit_code=exit_code
            ), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                output = directory / "output"
                artifacts = directory / "artifacts"
                (output / "results").mkdir(parents=True)
                artifacts.mkdir()
                (output / "results/profile.json").write_text("benchmark results")
                kubectl = directory / "kubectl"
                kubectl.write_text(
                    f"#!{sys.executable}\n"
                    "import json, subprocess, sys\n"
                    "args = sys.argv[1:]\n"
                    "if 'exec' in args:\n"
                    "    sys.exit(subprocess.run(args[args.index('--') + 1:]).returncode)\n"
                    "if 'get' in args and 'pods' in args:\n"
                    "    print(json.dumps({'items': [{'metadata': {'name': 'benchmark', "
                    "'creationTimestamp': '2026-01-01'}, 'status': {'phase': 'Running'}}]}))\n"
                    "elif 'get' in args and 'job' in args:\n"
                    "    print(json.dumps({'status': {'conditions': []}}))\n"
                )
                kubectl.chmod(0o755)
                env = dict(
                    os.environ,
                    PATH=str(directory) + os.pathsep + os.environ["PATH"],
                    AIPERF_OUTPUT_DIR=str(output),
                    AIPERF_COLLECTION_TIMEOUT="10",
                    ARTIFACTS=str(artifacts),
                    KUBE_CONTEXT="test",
                    NAMESPACE="test",
                    BENCHMARK="benchmark",
                    PERF_CONTAINER="bench",
                    PERF_POD="benchmark",
                    RECIPE="test",
                    RUN_KEY="test",
                    RESOURCE_KINDS="job",
                    DGD="test",
                    GITHUB_ENV=str(directory / "github-env"),
                )
                with subprocess.Popen(
                    ["bash", "-c", self.lifecycle() + f"\nexit {exit_code}"], env=env
                ) as process:
                    try:
                        deadline = time.monotonic() + 5
                        while not (output / "exit-code").exists():
                            self.assertLess(time.monotonic(), deadline)
                            time.sleep(0.02)
                        result = subprocess.run(
                            ["bash", "-e", "-o", "pipefail", "-c", run],
                            env=env,
                            timeout=5,
                        )
                        self.assertEqual(result.returncode, exit_code)
                        subprocess.run(
                            ["bash", "-e", "-o", "pipefail", "-c", collect],
                            env=env,
                            timeout=5,
                            check=True,
                        )
                        self.assertEqual(
                            (artifacts / "perf/profile.json").read_text(),
                            "benchmark results",
                        )
                        self.assertEqual(process.wait(timeout=5), exit_code)
                    finally:
                        if process.poll() is None:
                            process.kill()
                            process.wait()

    def test_every_recipe_writes_results_to_bounded_disk_storage(self):
        for recipe in RECIPES:
            with self.subTest(recipe=recipe):
                output = subprocess.check_output(
                    [
                        "kubectl",
                        "kustomize",
                        str(ROOT / ".github/ci" / recipe),
                        "--load-restrictor",
                        "LoadRestrictionsNone",
                    ],
                    text=True,
                )
                job = next(
                    item for item in yaml.safe_load_all(output) if item["kind"] == "Job"
                )
                spec = job["spec"]["template"]["spec"]
                container = spec["containers"][0]
                env = {item["name"]: item.get("value") for item in container["env"]}
                path = env.get("ARTIFACT_ROOT", env.get("ROOT_ARTIFACT_DIR"))
                mount = next(
                    mount
                    for mount in container["volumeMounts"]
                    if path.startswith(mount["mountPath"] + "/")
                )
                volume = next(
                    volume
                    for volume in spec["volumes"]
                    if volume["name"] == mount["name"]
                )
                self.assertIn("emptyDir", volume)
                self.assertNotEqual(volume["emptyDir"].get("medium"), "Memory")
                self.assertIn("sizeLimit", volume["emptyDir"])
                self.assertIn("ephemeral-storage", container["resources"]["requests"])
                self.assertIn("ephemeral-storage", container["resources"]["limits"])

    def test_upload_precedes_pod_cleanup(self):
        upload = next(
            i
            for i, step in enumerate(self.steps)
            if step.get("uses", "").startswith("actions/upload-artifact@")
        )
        cleanup = next(
            i
            for i, step in enumerate(self.steps)
            if step.get("name") == "Clean up recipe resources"
        )
        self.assertLess(upload, cleanup)

    def test_render_preserves_benchmark_script_and_timeout(self):
        render = next(
            step
            for step in self.steps
            if step.get("name") == "Render deployment and benchmark"
        )
        expression = re.search(
            r"yq -i '(with\(select\(\.kind == \"Job\" and .*?)' \"\$ARTIFACTS/rendered.yaml\"",
            render["run"],
            re.DOTALL,
        ).group(1)
        for recipe in RECIPES:
            with self.subTest(
                recipe=recipe
            ), tempfile.TemporaryDirectory() as temporary:
                output = subprocess.check_output(
                    [
                        "kubectl",
                        "kustomize",
                        str(ROOT / ".github/ci" / recipe),
                        "--load-restrictor",
                        "LoadRestrictionsNone",
                    ],
                    text=True,
                )
                original = next(
                    item for item in yaml.safe_load_all(output) if item["kind"] == "Job"
                )
                path = Path(temporary) / "rendered.yaml"
                path.write_text(output)
                env = dict(
                    os.environ,
                    BENCHMARK=original["metadata"]["name"],
                    PERF_LIFECYCLE=self.lifecycle(),
                )
                subprocess.run(
                    [os.environ.get("YQ", "yq"), "-i", expression, str(path)],
                    env=env,
                    check=True,
                )
                job = next(
                    item
                    for item in yaml.safe_load_all(path.read_text())
                    if item["kind"] == "Job"
                )
                container = job["spec"]["template"]["spec"]["containers"][0]
                self.assertEqual(
                    container["command"][4],
                    original["spec"]["template"]["spec"]["containers"][0]["command"][2],
                )
                variables = {
                    item["name"]: item.get("value") for item in container["env"]
                }
                self.assertEqual(
                    int(variables["AIPERF_BENCHMARK_TIMEOUT"]),
                    original["spec"]["activeDeadlineSeconds"],
                )
                self.assertGreater(
                    job["spec"]["activeDeadlineSeconds"],
                    int(variables["AIPERF_BENCHMARK_TIMEOUT"]),
                )
                subprocess.run(
                    ["bash", "-n", "-c", container["command"][2]], check=True
                )


if __name__ == "__main__":
    unittest.main()
