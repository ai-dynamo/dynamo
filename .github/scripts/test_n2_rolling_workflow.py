# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


class RollingWorkflowTest(unittest.TestCase):
    def test_rolling_pr_lane_builds_all_required_candidate_images(self):
        for change in ("sglang", "core", "frontend"):
            with self.subTest(change=change):
                workflow = yaml.safe_load(
                    (ROOT / ".github/workflows/pr.yaml").read_text()
                )
                outcomes = {}
                for name in (
                    "operator",
                    "frontend-build",
                    "sglang-build",
                    "frontend-copy-to-acr",
                    "sglang-copy-to-acr",
                    "rolling-upgrade-compatibility",
                ):
                    condition = workflow["jobs"][name]["if"]
                    condition = re.sub(
                        r"needs\.changed-files\.outputs\.(\w+) == 'true'",
                        lambda match: str(match[1] in (change, "run_deploy_tests")),
                        condition,
                    )
                    condition = re.sub(
                        r"needs\.([\w-]+)\.result == 'success'",
                        lambda match: str(outcomes.get(match[1], False)),
                        condition,
                    )
                    condition = (
                        condition.replace("always()", "True")
                        .replace("!cancelled()", "True")
                        .replace("&&", " and ")
                        .replace("||", " or ")
                    )
                    outcomes[name] = eval(
                        " ".join(condition.split()), {"__builtins__": {}}
                    )
                self.assertTrue(all(outcomes.values()), outcomes)


if __name__ == "__main__":
    unittest.main()
