# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[5]


class AggregatedLaunchTests(unittest.TestCase):
    def test_engine_receives_base_config_defaults_and_profiler_overrides(self):
        cases = [
            (None, False, "xgrammar"),
            ("max_batch_size: 1\n", False, "xgrammar"),
            ("guided_decoding_backend: guidance\n", False, "guidance"),
            ((ROOT / "lib/sidecar/trtllm/ci/whisper.yaml").read_text(), False, None),
            ("guided_decoding_backend: guidance\n", True, None),
        ]
        for config, disabled, expected in cases:
            with self.subTest(
                config=config, disabled=disabled
            ), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                wrapper = root / "python3"
                wrapper.write_text(
                    f"#!{sys.executable}\n"
                    "import json, os, signal, sys\n"
                    "from pathlib import Path\n"
                    "import yaml\n"
                    "args = sys.argv[1:]\n"
                    "if args[:1] == ['-c']:\n"
                    "    sys.exit(0)\n"
                    "if args[:2] == ['-m', 'tensorrt_llm.commands.serve']:\n"
                    "    config = Path(args[args.index('--extra_llm_api_options') + 1])\n"
                    "    Path(os.environ['CAPTURE']).write_text(json.dumps(yaml.safe_load(config.read_text())))\n"
                    "    sys.exit(0)\n"
                    "if args[:1] == ['-m'] or '--grpc-endpoint' in args:\n"
                    "    signal.pause()\n"
                    f"os.execv({sys.executable!r}, [{sys.executable!r}, *args])\n"
                )
                wrapper.chmod(0o755)
                capture = root / "captured.json"
                env = {
                    key: value
                    for key, value in os.environ.items()
                    if not key.startswith(("TRTLLM_", "DYN_", "_PROFILE_OVERRIDE_"))
                    and key not in ("AGG_ENGINE_ARGS", "MODEL")
                }
                env.update(
                    {
                        "PATH": f"{root}:{os.environ['PATH']}",
                        "TRTLLM_PYTHON": str(wrapper),
                        "DYNAMO_SIDECAR_BIN": str(wrapper),
                        "CAPTURE": str(capture),
                        "TMPDIR": str(root),
                        "_PROFILE_OVERRIDE_TRTLLM_MAX_TOTAL_TOKENS": "512",
                    }
                )
                if config is not None:
                    base = root / "base.yaml"
                    base.write_text(config)
                    env["AGG_ENGINE_ARGS"] = str(base)
                if disabled:
                    env["TRTLLM_DISABLE_GUIDED_DECODING"] = "1"
                process = subprocess.Popen(
                    ["bash", str(ROOT / "lib/sidecar/trtllm/launch/agg.sh")],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
                try:
                    stdout, stderr = process.communicate(timeout=10)
                finally:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5)
                self.assertEqual(process.returncode, 0, stdout + stderr)
                actual = json.loads(capture.read_text())
                self.assertEqual(actual["guided_decoding_backend"], expected)
                self.assertEqual(actual["kv_cache_config"]["max_tokens"], 512)
                if config is not None:
                    for key, value in yaml.safe_load(config).items():
                        if key == "kv_cache_config":
                            self.assertEqual(actual[key], {**value, "max_tokens": 512})
                        elif key != "guided_decoding_backend":
                            self.assertEqual(actual[key], value)


if __name__ == "__main__":
    unittest.main()
