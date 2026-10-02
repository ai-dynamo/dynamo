# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression checks for the runtime's packaged vllm-rs wrapper.

Run directly with Python; no vLLM install, GPU, or container build is required.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from jinja2 import Environment, StrictUndefined


class VllmRsWrapperTest(unittest.TestCase):
    def test_wrapper_preserves_binary_and_forwards_arguments(self):
        template = (
            Path(__file__).resolve().parents[2]
            / "templates"
            / "vllm_runtime.Dockerfile"
        ).read_text()
        block = template.split(
            "# Use the packaged binary to match the installed vLLM version.\n", 1
        )[1].split("\n\nUSER dynamo", 1)[0]
        block = block.removeprefix("RUN ").replace("\\\n", " ")

        for destination in ("absent", "symlink", "regular"):
            for allowlist in ("0", "1"):
                with self.subTest(destination=destination, allowlist=allowlist):
                    with tempfile.TemporaryDirectory() as directory:
                        root = Path(directory)
                        package = root / "package"
                        package.mkdir()
                        binary = package / "vllm-rs"
                        original = (
                            "#!/bin/sh\n"
                            "printf '%s\\n' \"${VLLM_PLUGINS-unset}\"\n"
                            "printf '<%s>\\n' \"$@\"\n"
                        )
                        binary.write_text(original)
                        binary.chmod(0o755)
                        link = root / "vllm-rs"
                        if destination == "symlink":
                            link.symlink_to(binary)
                        elif destination == "regular":
                            link.write_text("old entrypoint")

                        script = (
                            Environment(undefined=StrictUndefined)
                            .from_string(block)
                            .render(
                                python_executable="mock_python",
                                vllm_rs_allowlist=allowlist,
                                vllm_rs_link=str(link),
                                vllm_rs_plugins="modelexpress",
                                vllm_rs_required="1",
                            )
                        )
                        # Mock package discovery and Linux's timeout utility on
                        # macOS. The entire subprocess remains time-bounded.
                        setup = (
                            'mock_python() { printf "%s\\n" "$MOCK_PACKAGE"; }; '
                            'timeout() { shift; "$@"; }; '
                        )
                        env = dict(os.environ)
                        env.pop("VLLM_PLUGINS", None)
                        env["MOCK_PACKAGE"] = str(package)
                        env["PATH"] = f"{root}:{env['PATH']}"
                        subprocess.run(
                            ["sh", "-c", setup + script],
                            env=env,
                            check=True,
                            capture_output=True,
                            timeout=5,
                        )
                        self.assertEqual(binary.read_text(), original)
                        self.assertEqual(link.is_symlink(), allowlist == "0")
                        for plugins in (None, "", "custom"):
                            with self.subTest(plugins=plugins):
                                if plugins is None:
                                    env.pop("VLLM_PLUGINS", None)
                                else:
                                    env["VLLM_PLUGINS"] = plugins
                                result = subprocess.run(
                                    [str(link), "--help", "two words", ""],
                                    env=env,
                                    check=True,
                                    capture_output=True,
                                    text=True,
                                    timeout=5,
                                )
                                default = (
                                    "modelexpress" if allowlist == "1" else "unset"
                                )
                                expected = default if plugins is None else plugins
                                self.assertEqual(
                                    result.stdout,
                                    f"{expected}\n<--help>\n<two words>\n<>\n",
                                )
                        self.assertFalse(list(root.glob("vllm-rs.*")))


if __name__ == "__main__":
    unittest.main()
