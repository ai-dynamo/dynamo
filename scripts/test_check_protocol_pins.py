# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import unittest

from check_protocol_pins import check_pins


class ProtocolPinTests(unittest.TestCase):
    def setUp(self):
        self.context = {
            "vllm": {
                "cuda13.0": {"runtime_image_tag": "v0.30.0-ubuntu2404"},
                "cpu": {"runtime_image_tag": "v0.29.0"},
                "xpu": {"runtime_image_tag": "v0.29.0"},
                "vllm_omni_ref": "v0.30.0rc1",
            }
        }
        self.pins = {
            "versions": [
                {"version": "0.30.0", "commit": "a" * 40, "platforms": ["cuda13.0"]},
                {"version": "0.29.0", "commit": "b" * 40, "platforms": ["cpu", "xpu"]},
            ]
        }

    def test_all_shipped_platforms_are_checked(self):
        check_pins(self.context, self.pins)
        for platform in ["cuda13.0", "cpu", "xpu"]:
            context = copy.deepcopy(self.context)
            context["vllm"][platform]["runtime_image_tag"] = "v0.31.0"
            with self.assertRaisesRegex(ValueError, "framework-version bump"):
                check_pins(context, self.pins)

    def test_new_platform_requires_source_mapping(self):
        self.context["vllm"]["new_platform"] = {"runtime_image_tag": "v0.30.0"}
        with self.assertRaises(ValueError):
            check_pins(self.context, self.pins)

    def test_unrecognized_mutable_image_tag_fails(self):
        self.context["vllm"]["cpu"]["runtime_image_tag"] = "latest"
        with self.assertRaisesRegex(ValueError, "unrecognized"):
            check_pins(self.context, self.pins)

    def test_duplicate_mapping_and_mutable_source_fail(self):
        self.pins["versions"][0]["platforms"].append("cpu")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            check_pins(self.context, self.pins)
        self.pins["versions"][0]["commit"] = "main"
        with self.assertRaisesRegex(ValueError, "immutable"):
            check_pins(self.context, self.pins)


if __name__ == "__main__":
    unittest.main()
