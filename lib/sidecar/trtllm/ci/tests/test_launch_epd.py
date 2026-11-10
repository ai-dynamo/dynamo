# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU command-capture regressions, discovered by the sidecar unittest CI job."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
REVISION = "a" * 40

# These are inert command stubs, not engine lifecycle infrastructure. Each child
# captures its command before one child triggers failure/shutdown. The barrier
# ensures cleanup cannot race another child's capture.
COMMAND_STUB = """
import json
import os
import signal
import sys
import time
from pathlib import Path

args = sys.argv[1:]
if args[:1] == ['-c']:
    sys.exit(0)
if args[:1] == ['-']:
    os.execv(sys.executable, [sys.executable, *args])
capture = Path(os.environ['CAPTURE'])
record = {'args': args, 'pid': os.getpid(),
          'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
          'revision': os.environ.get('TRTLLM_OPENENGINE_ENCODER_MODEL_REVISION'),
          'endpoint': os.environ.get('TRTLLM_OPENENGINE_ENCODER_ENDPOINT'),
          'system_port': os.environ.get('DYN_SYSTEM_PORT')}
if '--extra_llm_api_options' in args:
    config = Path(args[args.index('--extra_llm_api_options') + 1])
    record['config'] = config.read_text()
    record['run_dir'] = str(config.parent)
def stopped(signum, frame):
    (capture / f'{os.getpid()}.stopped').touch()
    sys.exit(0)
signal.signal(signal.SIGTERM, stopped)
pending = capture / f'{os.getpid()}.pending'
pending.write_text(json.dumps(record))
pending.rename(capture / f'{os.getpid()}.json')
if '--port' in args and args[args.index('--port') + 1] == '31001':
    deadline = time.monotonic() + 5
    while len(list(capture.glob('*.json'))) < int(os.environ['EXPECTED_CHILDREN']):
        if time.monotonic() > deadline:
            sys.exit(99)
        time.sleep(0.01)
    if os.environ['SHUTDOWN'] == 'signal':
        os.kill(os.getppid(), signal.SIGTERM)
    else:
        sys.exit(23)
signal.pause()
"""

SNAPSHOT_STUB = """
import json
import os
from pathlib import Path

def snapshot_download(*, repo_id, revision):
    with Path(os.environ['SNAPSHOT_CALLS']).open('a') as output:
        output.write(json.dumps([repo_id, revision]) + '\\n')
    return os.environ['SNAPSHOT_PATH']
"""


class EpdLaunchTests(unittest.TestCase):
    def run_launcher(self, mode, overrides=None, *, shutdown="failure"):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            capture = directory / "capture"
            capture.mkdir()
            wrapper = directory / "python3"
            wrapper.write_text(f"#!{sys.executable}\n" + textwrap.dedent(COMMAND_STUB))
            wrapper.chmod(0o755)
            (directory / "huggingface_hub.py").write_text(
                textwrap.dedent(SNAPSHOT_STUB)
            )
            snapshot = directory / "snapshot with spaces" / REVISION
            snapshot.mkdir(parents=True)
            env = {
                key: value
                for key, value in os.environ.items()
                if not key.startswith(
                    (
                        "TRTLLM_",
                        "DYN_",
                        "ENCODER_",
                        "PD_",
                        "PREFILL_",
                        "DECODE_",
                        "MODEL",
                        "_PROFILE_OVERRIDE_",
                    )
                )
                and key not in ("EPD_MODE", "CUDA_VISIBLE_DEVICES", "PYTHONPATH")
            }
            env.update(
                PATH=f"{directory}:{os.environ['PATH']}",
                PYTHONPATH=str(directory),
                TRTLLM_PYTHON=str(wrapper),
                DYNAMO_SIDECAR_BIN=str(wrapper),
                MODEL="org/test-model",
                MODEL_REVISION=REVISION,
                TMPDIR=str(directory),
                CAPTURE=str(capture),
                SNAPSHOT_CALLS=str(directory / "snapshot-calls"),
                SNAPSHOT_PATH=str(snapshot),
                EXPECTED_CHILDREN="7" if mode == "e_p_d" else "5",
                SHUTDOWN=shutdown,
                DYN_ENGINE_GRPC_PORT1="31001",
                DYN_ENGINE_GRPC_PORT2="31011",
                DYN_ENGINE_GRPC_PORT3="31021",
                DYN_SYSTEM_PORT1="9081",
                DYN_SYSTEM_PORT2="9082",
                DYN_SYSTEM_PORT3="9083",
                _PROFILE_OVERRIDE_TRTLLM_MAX_TOTAL_TOKENS="512",
            )
            env.update(overrides or {})
            # A separate process group exercises the real launch-utils cleanup
            # without signalling the test runner. The backstop also covers a
            # broken launcher that never reaches wait_any_exit.
            with subprocess.Popen(
                [
                    "bash",
                    str(
                        ROOT / f"lib/sidecar/trtllm/launch/disagg_multimodal_{mode}.sh"
                    ),
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            ) as process:
                try:
                    stdout, stderr = process.communicate(timeout=10)
                    records = [
                        json.loads(path.read_text()) for path in capture.glob("*.json")
                    ]
                    for record in records:
                        if "run_dir" in record:
                            self.assertFalse(Path(record["run_dir"]).exists())
                        if (
                            "--port" not in record["args"]
                            or self.option(record, "--port") != "31001"
                            or shutdown == "signal"
                        ):
                            self.assertTrue(
                                (capture / f"{record['pid']}.stopped").exists()
                            )
                finally:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                result = process.returncode
            calls_path = directory / "snapshot-calls"
            calls = (
                [json.loads(line) for line in calls_path.read_text().splitlines()]
                if calls_path.exists()
                else []
            )
            return result, records, calls, str(snapshot), stdout + stderr

    @staticmethod
    def option(record, name):
        args = record["args"]
        return args[args.index(name) + 1]

    def test_pinned_source_gpu_allocation_and_role_commands(self):
        for mode in ("e_pd", "e_p_d"):
            for allocation in (None, "6,7", "GPU-allocated,MIG-other"):
                with self.subTest(mode=mode, allocation=allocation):
                    overrides = (
                        {}
                        if allocation is None
                        else {"CUDA_VISIBLE_DEVICES": allocation}
                    )
                    result, records, calls, snapshot, output = self.run_launcher(
                        mode, overrides
                    )
                    self.assertEqual(result, 23, output)
                    self.assertEqual(calls, [["org/test-model", REVISION]])
                    engines = [
                        record for record in records if "--port" in record["args"]
                    ]
                    sidecars = [
                        record
                        for record in records
                        if "--grpc-endpoint" in record["args"]
                    ]
                    expected = 3 if mode == "e_p_d" else 2
                    self.assertEqual(len(engines), expected)
                    self.assertEqual(len(sidecars), expected)
                    for engine in engines:
                        self.assertEqual(engine["args"][2], snapshot)
                        self.assertEqual(
                            self.option(engine, "--served_model_name"), "org/test-model"
                        )
                        self.assertEqual(engine["revision"], REVISION)
                        port = self.option(engine, "--port")
                        default = {
                            "31001": "0",
                            "31011": "1" if mode == "e_p_d" else "0",
                            "31021": "2",
                        }[port]
                        self.assertEqual(
                            engine["gpu"],
                            allocation.split(",")[0] if allocation else default,
                        )
                        if port != "31001":
                            config = json.loads(engine["config"])
                            self.assertTrue(config["disable_mm_encoder"])
                            self.assertEqual(
                                config["kv_cache_config"]["max_tokens"], 512
                            )
                            if mode == "e_p_d":
                                self.assertEqual(
                                    config["cache_transceiver_config"]["backend"],
                                    "NIXL",
                                )
                    for sidecar in sidecars:
                        self.assertEqual(self.option(sidecar, "--model-path"), snapshot)
                        self.assertEqual(
                            sidecar["system_port"],
                            {"31001": "9081", "31011": "9082", "31021": "9083"}[
                                self.option(sidecar, "--grpc-endpoint").rsplit(":", 1)[
                                    1
                                ]
                            ],
                        )
                    frontend = next(
                        record
                        for record in records
                        if "dynamo.frontend" in record["args"]
                    )
                    self.assertNotIn("--no-router-conditional-disagg", frontend["args"])
                    consumer = next(
                        record
                        for record in sidecars
                        if self.option(record, "--grpc-endpoint").endswith(":31011")
                    )
                    self.assertIn("--route-to-encoder", consumer["args"])
                    if mode == "e_p_d":
                        self.assertEqual(
                            self.option(consumer, "--disaggregation-mode"), "prefill"
                        )
                        decode = next(
                            record
                            for record in sidecars
                            if self.option(record, "--grpc-endpoint").endswith(":31021")
                        )
                        self.assertEqual(
                            self.option(decode, "--disaggregation-mode"), "decode"
                        )
                        self.assertNotIn("--route-to-encoder", decode["args"])

    def test_encoder_bind_connect_and_explicit_role_overrides(self):
        for mode in ("e_pd", "e_p_d"):
            for bind, connect, expected in (
                ("192.0.2.10", None, "192.0.2.10"),
                ("0.0.0.0", None, "127.0.0.1"),
                ("::", None, "[::1]"),
                ("0.0.0.0", "192.0.2.20", "192.0.2.20"),
            ):
                with self.subTest(mode=mode, bind=bind, connect=connect):
                    overrides = dict(
                        CUDA_VISIBLE_DEVICES="6,7",
                        ENCODER_GPU="7",
                        PD_GPU="8",
                        PREFILL_GPU="9",
                        DECODE_GPU="10",
                        ENCODER_BIND_HOST=bind,
                    )
                    if connect:
                        overrides["ENCODER_CONNECT_HOST"] = connect
                        overrides["ENCODER_ADVERTISE_HOST"] = "198.51.100.1"
                    result, records, _, _, output = self.run_launcher(mode, overrides)
                    self.assertEqual(result, 23, output)
                    engines = [
                        record for record in records if "--port" in record["args"]
                    ]
                    for engine in engines:
                        port = self.option(engine, "--port")
                        self.assertEqual(
                            engine["gpu"],
                            {
                                "31001": "7",
                                "31011": "9" if mode == "e_p_d" else "8",
                                "31021": "10",
                            }[port],
                        )
                        if port == "31001":
                            self.assertEqual(self.option(engine, "--host"), bind)
                            advertise = "198.51.100.1" if connect else expected
                            self.assertEqual(engine["endpoint"], f"{advertise}:31001")
                    encoder = next(
                        record for record in records if "encode" in record["args"]
                    )
                    self.assertEqual(
                        self.option(encoder, "--grpc-endpoint"), f"{expected}:31001"
                    )

    def test_signal_shutdown_cleans_children_and_configs(self):
        for mode in ("e_pd", "e_p_d"):
            with self.subTest(mode=mode):
                result, records, _, _, output = self.run_launcher(
                    mode, shutdown="signal"
                )
                self.assertEqual(result, 0, output)
                self.assertEqual(len(records), 7 if mode == "e_p_d" else 5)

    def test_rejects_empty_gpu_allocation_and_mutable_revision_before_launch(self):
        for overrides, expected in (
            ({"CUDA_VISIBLE_DEVICES": ""}, "CUDA_VISIBLE_DEVICES must name a GPU"),
            ({"MODEL_REVISION": "main"}, "full immutable checkpoint commit SHA"),
        ):
            with self.subTest(overrides=overrides):
                result, records, calls, _, output = self.run_launcher("e_pd", overrides)
                self.assertNotEqual(result, 0)
                self.assertEqual(records, [])
                self.assertEqual(calls, [])
                self.assertIn(expected, output)


if __name__ == "__main__":
    unittest.main()
