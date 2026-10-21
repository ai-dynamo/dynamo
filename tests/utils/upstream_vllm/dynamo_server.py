# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run inside the upstream pytest process; reuse vLLM's server lifecycle."""

import itertools
import json
import os
import tempfile
import time
from pathlib import Path

import httpx
import requests

from tests.utils import RemoteOpenAIServer, RemoteOpenAIServerCustom, RemoteVLLMServer

_exchanges = itertools.count()


async def record_response(response):
    """Capture the selected unary requests/responses without changing their JSON."""
    # shortcut: these four cases are unary; add a byte-stream tee before adding SSE.
    await response.aread()
    path = Path(os.environ["UPSTREAM_CHAT_OUTPUT"]) / f"exchange-{next(_exchanges):03d}"
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "request": json.loads(response.request.content),
                "status": response.status_code,
                "content_type": response.headers.get("content-type"),
            },
            indent=2,
        )
    )
    path.with_suffix(".body").write_bytes(response.content)


class RecordingClient:
    """Add evidence capture to the inherited upstream OpenAI client."""

    def get_async_client(self, **kwargs):
        """Return the native client with a recording HTTP transport; no retries."""
        kwargs["http_client"] = httpx.AsyncClient(
            timeout=120, trust_env=False, event_hooks={"response": [record_response]}
        )
        return super().get_async_client(**kwargs)


class ReferenceServer(RecordingClient, RemoteOpenAIServer):
    """Keep upstream startup and teardown, with unary evidence capture."""


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


class RemoteDynamoServer(RecordingClient, RemoteOpenAIServerCustom):
    """Replace only the launcher and readiness of the audited upstream fixtures."""

    def __init__(self, model, vllm_serve_args, **kwargs):
        """Start one isolated Dynamo deployment; retain upstream model/CLI handling.

        The runner supplies pinned model arguments for the four selected tests.
        The outer Dynamo fixture reserves DYN_SYSTEM_PORT in the Rust i16 range.
        Upstream code owns child creation, timeout cleanup and process-group teardown.
        """
        self.model = model
        self.discovery = tempfile.TemporaryDirectory(prefix="dynamo-upstream-")
        env = dict(kwargs.pop("env_dict", {}) or {})
        env.update(
            DYN_DISCOVERY_BACKEND="file",
            DYN_REQUEST_PLANE="tcp",
            DYN_FILE_KV=self.discovery.name,
            DYN_NAMESPACE=Path(self.discovery.name).name,
            DYN_SYSTEM_PORT=os.environ["DYN_SYSTEM_PORT"],
        )
        try:
            super().__init__(
                model, vllm_serve_args, launch_dynamo, env_dict=env, **kwargs
            )
        except BaseException:
            self.discovery.cleanup()
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
            except requests.ConnectionError:
                pass  # Frontend has not bound its socket yet.
            time.sleep(0.5)
        raise TimeoutError(f"Dynamo did not register {self.model} within {timeout}s")

    def _shutdown(self):
        """Run upstream process cleanup before removing this deployment's discovery state."""
        try:
            super()._shutdown()
        finally:
            self.discovery.cleanup()
