# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run inside the upstream pytest process; reuse vLLM's server lifecycle."""

import os
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

import requests
from constants import DynamoPortRange
from port_utils import reserved_ports

from tests.utils import RemoteOpenAIServerCustom, RemoteVLLMServer


def launch_dynamo(env_dict, model, args):
    """Exec the existing aggregate launcher inside vLLM's managed process group."""
    env = os.environ.copy()
    env.update(env_dict or {})
    worker_args = []
    args = iter(args)
    for arg in args:
        if arg == "--port":
            env["DYN_HTTP_PORT"] = next(args)
        elif arg == "--reasoning-parser":
            worker_args.extend(["--dyn-reasoning-parser", next(args)])
        else:
            worker_args.append(arg)
    os.execvpe(
        "bash",
        ["bash", env["DYNAMO_CHAT_LAUNCHER"], "--model", model, *worker_args],
        env,
    )


class RemoteDynamoServer(RemoteOpenAIServerCustom):
    """Replace only the launcher and readiness of the audited upstream fixtures."""

    def __init__(self, model, vllm_serve_args, **kwargs):
        """Start one isolated Dynamo deployment; retain upstream model/CLI handling.

        Selected fixtures supply model and engine arguments; cap the KV cache at 1 GiB.
        Each server reserves its own system port in the Rust i16 range.
        Upstream code owns child creation, timeout cleanup and process-group teardown.
        """
        self.model = model
        self.resources = ExitStack()
        try:
            discovery = self.resources.enter_context(
                tempfile.TemporaryDirectory(prefix="dynamo-upstream-")
            )
            system_port = self.resources.enter_context(
                reserved_ports(1, DynamoPortRange.FRONTEND)
            )[0]
            env = dict(kwargs.pop("env_dict", {}) or {})
            env.update(
                DYN_DISCOVERY_BACKEND="file",
                DYN_REQUEST_PLANE="tcp",
                DYN_FILE_KV=discovery,
                DYN_NAMESPACE=Path(discovery).name,
                DYN_SYSTEM_PORT=str(system_port),
            )
            super().__init__(
                model,
                [
                    *vllm_serve_args,
                    "--served-model-name",
                    model,
                    "--max-num-seqs",
                    "8",
                    "--kv-cache-memory-bytes",
                    "1073741824",
                ],
                launch_dynamo,
                env_dict=env,
                **kwargs,
            )
        except BaseException:
            self.resources.close()
            raise

    def _get_process_termination_timeout(self):
        """Use the generic upstream timeout; Dynamo has no vLLM API-server grace period."""
        return RemoteVLLMServer._get_process_termination_timeout(self)

    def _wait_for_server(self, *, url, timeout):
        """Wait for this model, not just an HTTP listener; fail on worker exit."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._poll() is not None:
                raise RuntimeError("Dynamo launcher exited before model registration")
            try:
                response = requests.get(self.url_for("v1/models"), timeout=2)
                if response.status_code == 200 and any(
                    item.get("id") == self.model
                    for item in response.json().get("data", [])
                ):
                    return
            except (requests.ConnectionError, requests.Timeout):
                pass  # Frontend has not bound its socket yet.
            time.sleep(0.5)
        raise TimeoutError(f"Dynamo did not register {self.model} within {timeout}s")

    def _shutdown(self):
        """Run upstream cleanup, then release this deployment's discovery state and port."""
        try:
            super()._shutdown()
        finally:
            self.resources.close()
