# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check GPU resource placement and rollout capacity."""

import unittest

import manifests


class GpuDeploymentTests(unittest.TestCase):
    def test_workers_fit_exactly_two_single_gpu_nodes_and_support_rolling_updates(self):
        image = "example.invalid/runtime@sha256:" + "a" * 64
        output = manifests.render("example", "example-site", image, image)
        workers = next(
            v
            for v in output["items"]
            if v["kind"] == "Deployment" and v["metadata"]["name"] == "qwen-gpu-worker"
        )
        pod = workers["spec"]["template"]["spec"]
        worker = pod["containers"][0]
        self.assertEqual(workers["spec"]["replicas"], 2)
        self.assertEqual(worker["resources"]["requests"]["nvidia.com/gpu"], "1")
        self.assertEqual(worker["resources"]["limits"]["nvidia.com/gpu"], "1")
        self.assertEqual(
            workers["spec"]["strategy"]["rollingUpdate"],
            {"maxSurge": 0, "maxUnavailable": 1},
        )
        self.assertEqual(worker["command"], ["python3", "-m", "dynamo.vllm"])
        self.assertTrue(
            any(
                v.get("hostPath", {})
                .get("path", "")
                .startswith("/var/lib/dynamo/model-cache/")
                for v in pod["volumes"]
            )
        )
        self.assertNotIn("dynamo.mocker", str(output))


if __name__ == "__main__":
    unittest.main()
