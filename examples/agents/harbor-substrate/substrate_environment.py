# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Harbor external environment backed by a pre-registered Substrate template.

The adapter talks to Substrate directly. Actor lifecycle and the egress policy
go through ``kubectl-ate`` with the runner's ServiceAccount token. Commands and
files go to the ``ate-env-guest`` daemon inside the actor, through Substrate's
router over mTLS with the runner Pod's certificate and the ``ate-target-actor``
routing header. The guest's gRPC services come from the env Python client.

The template must contain the selected task's built image and ate-env-guest.
This example supports Linux, root, one container, and public task networking.
CPU and memory limits are configured on the template.
"""

import asyncio
import collections
import contextlib
import json
import os
import re
import shlex
import signal
import tempfile
import time
import uuid
from pathlib import Path

import grpc
from ate_env import Client, EnvError
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentResourceCapabilities
from harbor.environments.tar_transfer import (
    extract_dir_from_file,
    pack_dir_to_file,
    remote_pack_command,
    remote_unpack_command,
)

# Routing header the Substrate router reads to select the actor.
TARGET_ACTOR_HEADER = "ate-target-actor"


class _ClientCallDetails(
    collections.namedtuple(
        "_ClientCallDetails",
        ["method", "timeout", "metadata", "credentials", "wait_for_ready"],
    ),
    grpc.aio.ClientCallDetails,
):
    pass


class _TargetActor:
    """Add the router's actor-selection header to a guest call.

    grpc.aio files an interceptor under the first call type it matches, so one
    object cannot serve all four. Each call type gets its own small class, and
    target_actor_interceptors() returns the set to pass to the channel.
    """

    def __init__(self, atespace: str, actor_id: str):
        self._target = f"{atespace}/{actor_id}"

    def _inject(self, details):
        metadata = list(details.metadata or [])
        metadata.append((TARGET_ACTOR_HEADER, self._target))
        return _ClientCallDetails(
            details.method,
            details.timeout,
            metadata,
            details.credentials,
            details.wait_for_ready,
        )


class _TargetActorUnaryUnary(_TargetActor, grpc.aio.UnaryUnaryClientInterceptor):
    async def intercept_unary_unary(self, continuation, details, request):
        return await continuation(self._inject(details), request)


class _TargetActorUnaryStream(_TargetActor, grpc.aio.UnaryStreamClientInterceptor):
    async def intercept_unary_stream(self, continuation, details, request):
        return await continuation(self._inject(details), request)


class _TargetActorStreamUnary(_TargetActor, grpc.aio.StreamUnaryClientInterceptor):
    async def intercept_stream_unary(self, continuation, details, request_iterator):
        return await continuation(self._inject(details), request_iterator)


class _TargetActorStreamStream(_TargetActor, grpc.aio.StreamStreamClientInterceptor):
    async def intercept_stream_stream(self, continuation, details, request_iterator):
        return await continuation(self._inject(details), request_iterator)


def target_actor_interceptors(atespace: str, actor_id: str) -> list:
    """Interceptors that route every call type to one actor."""
    return [
        cls(atespace, actor_id)
        for cls in (
            _TargetActorUnaryUnary,
            _TargetActorUnaryStream,
            _TargetActorStreamUnary,
            _TargetActorStreamStream,
        )
    ]


_PEM_BLOCK = re.compile(rb"-----BEGIN ([A-Z ]+)-----.*?-----END \1-----\r?\n?", re.S)


def split_credential_bundle(bundle: bytes) -> tuple[bytes, bytes]:
    """Return (certificate chain, private key) from a projected pod certificate bundle."""
    chain, key = [], None
    for match in _PEM_BLOCK.finditer(bundle):
        kind = match.group(1)
        if kind == b"CERTIFICATE":
            chain.append(match.group(0))
        elif kind.endswith(b"PRIVATE KEY"):
            key = match.group(0)
    if not chain or key is None:
        raise ValueError("credential bundle needs a certificate and a private key")
    return b"".join(chain), key


class SubstrateEnvironment(BaseEnvironment):
    def __init__(
        self,
        *args,
        kube_context: str,
        egress_policy: str,
        template: str | None = None,
        task_name: str | None = None,
        template_map: dict[str, str] | None = None,
        atespace: str = "ate-env",
        ate_cli: str = "kubectl-ate",
        ate_endpoint: str | None = None,
        ate_token_file: str | None = None,
        router_endpoint: str = "atenet-router.ate-system.svc:443",
        credential_bundle: str = "/run/substrate/credential-bundle.pem",
        trust_bundle: str = "/run/substrate/trust-bundle.pem",
        workdir: str = "/testbed",
        ready_timeout: float = 180,
        transfer_timeout: float = 600,
        uploads: dict[str, str] | None = None,
        **kwargs,
    ):
        self._template = template
        self._task_name = task_name
        self._template_map = template_map
        self._kube_context = kube_context
        self._egress_policy = Path(egress_policy).resolve()
        self._ate_cli = ate_cli
        self._ate_endpoint = ate_endpoint
        self._ate_token_file = ate_token_file
        self._router_endpoint = router_endpoint
        self._credential_bundle = Path(credential_bundle)
        self._trust_bundle = Path(trust_bundle)
        self._atespace = atespace
        self._workdir = workdir
        self._ready_timeout = ready_timeout
        self._transfer_timeout = transfer_timeout
        self._uploads = uploads or {}
        self._channel = None
        self._client = None
        self._env = None
        super().__init__(*args, **kwargs)
        prefix = re.sub(r"[^a-z0-9-]", "-", self.session_id.lower()).strip("-")
        self.actor_id = f"{prefix[:30] or 'harbor'}-{uuid.uuid4().hex}"

    @staticmethod
    def type() -> str:
        return "substrate"

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        # The provider selects an existing template; it cannot change its limits.
        return EnvironmentResourceCapabilities()

    def _validate_definition(self):
        if self._template_map is not None:
            if self._template is not None or self._task_name is not None:
                raise ValueError("Use template_map or template/task_name, not both")
            self._template = self._template_map.get(self.environment_name)
            self._task_name = self.environment_name
        if not self._template or not self._task_name:
            raise ValueError("A registered template is required for each task")
        if not self._kube_context or not self._egress_policy.is_file():
            raise ValueError(
                "An explicit Kubernetes context and egress policy file are required"
            )
        if not self._credential_bundle.is_file() or not self._trust_bundle.is_file():
            raise ValueError(
                "The runner needs its projected pod certificate and trust bundle"
            )
        if self.environment_name != self._task_name:
            raise ValueError("The ActorTemplate is prepared for a different task")
        if self._uses_compose:
            raise ValueError("This example does not support Docker Compose")
        if any(
            value is not None
            for value in (
                self._override_cpus,
                self._override_memory_mb,
                self._override_storage_mb,
            )
        ):
            raise ValueError("Set resource limits on the ActorTemplate, not Harbor")
        for mount in self._mounts:
            if mount["type"] != "bind" or mount["target"] not in {
                "/logs/agent",
                "/logs/verifier",
                "/logs/artifacts",
            }:
                raise ValueError("Remote actors cannot bind host paths; use uploads")

    @contextlib.contextmanager
    def _timing(self, phase: str, **details):
        started = time.time()
        clock = time.perf_counter()
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            record = {
                "phase": phase,
                "started_at_unix": started,
                "duration_sec": time.perf_counter() - clock,
                "succeeded": succeeded,
                **details,
            }
            path = self.trial_paths.trial_dir / "substrate-timing.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")

    # --- Substrate control plane, through kubectl-ate ---

    async def _ate(self, *args: str, timeout: float = 120) -> str:
        """Run one kubectl-ate command with the runner's token and return its stdout."""
        command = [self._ate_cli]
        if self._ate_endpoint:
            command.extend(["--endpoint", self._ate_endpoint])
        if self._ate_token_file:
            command.extend(["--token-file", self._ate_token_file])
        command.extend(["--context", self._kube_context, *args])
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            async with asyncio.timeout(timeout):
                stdout, stderr = await proc.communicate()
        finally:
            if proc.returncode is None:
                os.killpg(proc.pid, signal.SIGKILL)
                await proc.wait()
        if proc.returncode:
            raise RuntimeError(
                f"kubectl-ate {' '.join(args[:2])} failed: "
                f"{stderr.decode(errors='replace').strip()}"
            )
        return stdout.decode(errors="replace")

    # --- guest, through the Substrate router ---

    def _open_guest_channel(self) -> None:
        """Open the mTLS channel to the router that carries this actor's guest calls.

        The projected certificate rotates, so the files are read for every actor
        rather than once per process.
        """
        chain, key = split_credential_bundle(self._credential_bundle.read_bytes())
        credentials = grpc.ssl_channel_credentials(
            root_certificates=self._trust_bundle.read_bytes(),
            private_key=key,
            certificate_chain=chain,
        )
        self._channel = grpc.aio.secure_channel(
            self._router_endpoint,
            credentials,
            interceptors=target_actor_interceptors(self._atespace, self.actor_id),
        )
        self._client = Client(self._router_endpoint, channel=self._channel)
        self._env = self._client.env(self.actor_id, atespace=self._atespace)

    async def _close_guest_channel(self) -> None:
        if self._client is not None:
            await self._client.close()
        if self._channel is not None:
            await self._channel.close()
        self._channel = self._client = self._env = None

    async def start(self, force_build: bool) -> None:
        if force_build:
            raise ValueError("Build the task image and register its template first")
        receipt = {
            "actor_id": self.actor_id,
            "atespace": self._atespace,
            "template": self._template,
            "task": self.environment_name,
            "environment_id": self.environment_id,
            "harbor_session_id": self.session_id,
            "router": self._router_endpoint,
        }
        self.trial_paths.trial_dir.mkdir(parents=True, exist_ok=True)
        (self.trial_paths.trial_dir / "substrate.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        created = False
        try:
            async with asyncio.timeout(self._ready_timeout):
                with self._timing("create_actor"):
                    await self._ate(
                        "create",
                        "actor",
                        self.actor_id,
                        "-a",
                        self._atespace,
                        "--template",
                        self._template,
                    )
                    created = True
                with self._timing("egress_policy"):
                    await self._ate(
                        "create",
                        "egress-policy",
                        self.actor_id,
                        "-a",
                        self._atespace,
                        "-f",
                        str(self._egress_policy),
                    )
                self._open_guest_channel()
                with self._timing("first_guest_command"):
                    while True:
                        try:
                            # The first guest call makes the router wake the actor.
                            # The router answers Unavailable until the guest listens.
                            result = await self.exec("true", cwd="/", timeout_sec=5)
                            if result.return_code == 0:
                                break
                        except (EnvError, grpc.RpcError, TimeoutError):
                            pass
                        await asyncio.sleep(1)
            with self._timing("log_directories"):
                await self._checked("mkdir -p /logs/{agent,verifier,artifacts}")
            for source, destination in self._uploads.items():
                with self._timing("upload_directory", target=destination):
                    await self.upload_dir(source, destination)
            with self._timing("task_environment_upload"):
                await self._upload_environment_dir_after_start()
            self.logger.info("Substrate actor %s ready", self.actor_id)
        except BaseException:
            try:
                await self.stop(delete=created)
            except Exception:
                self.logger.exception(
                    "Failed to delete actor %s after startup failure", self.actor_id
                )
            raise

    async def stop(self, delete: bool):
        try:
            if delete:
                try:
                    async with asyncio.timeout(60):
                        await self._ate("delete", "actor", self.actor_id, "-a", self._atespace)
                except RuntimeError as exc:
                    if "notfound" not in str(exc).replace(" ", "").lower():
                        raise
            else:
                self.logger.info(
                    "Retaining actor %s in %s", self.actor_id, self._atespace
                )
        finally:
            await self._close_guest_channel()

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        if self._resolve_user(user) not in (None, "root", "0", 0):
            raise ValueError("This example supports root commands only")
        if self._env is None:
            raise RuntimeError("The Substrate environment has not started")
        proc = None
        try:
            async with asyncio.timeout(timeout_sec):
                proc = await self._env.start_process(
                    ["bash", "-lc", command],
                    cwd=cwd or self.task_env_config.workdir or self._workdir,
                    env=self._merge_env(env),
                    # Independent watchdog protects against a lost client connection.
                    timeout=(timeout_sec + 5) if timeout_sec is not None else 3600,
                )
                stdout, stderr = bytearray(), bytearray()
                exit_code = None
                async for chunk in proc.output(follow=True):
                    if chunk.stdout is not None:
                        stdout.extend(chunk.stdout)
                    if chunk.stderr is not None:
                        stderr.extend(chunk.stderr)
                    if chunk.exit is not None:
                        exit_code = chunk.exit.exit_code
                if exit_code is None:
                    raise RuntimeError("Process output ended without an exit status")
                return ExecResult(
                    stdout=stdout.decode(errors="replace"),
                    stderr=stderr.decode(errors="replace"),
                    return_code=exit_code,
                )
        except BaseException:
            if proc is not None:
                try:
                    async with asyncio.timeout(10):
                        await proc.kill()
                except Exception:
                    self.logger.exception("Could not kill cancelled guest command")
            raise

    async def _checked(self, command: str):
        result = await self.exec(command, timeout_sec=60)
        if result.return_code != 0:
            raise RuntimeError(f"Guest command failed: {result.stderr}")

    async def upload_file(self, source_path, target_path):
        source = Path(source_path)
        await self._checked(f"mkdir -p {shlex.quote(str(Path(target_path).parent))}")
        async with asyncio.timeout(self._transfer_timeout):
            with source.open("rb") as stream:
                await self._env.write_file(
                    target_path,
                    iter(lambda: stream.read(65536), b""),
                    mode=source.stat().st_mode & 0o777,
                )
        # File creation applies the guest's umask; chmod restores the source mode.
        await self._checked(
            f"chmod {source.stat().st_mode & 0o777:o} {shlex.quote(target_path)}"
        )

    async def download_file(self, source_path, target_path):
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        async with asyncio.timeout(self._transfer_timeout):
            with target.open("wb") as stream:
                async for chunk in self._env.read_file(source_path):
                    stream.write(chunk)

    async def upload_dir(self, source_dir, target_dir):
        remote = f"/tmp/harbor-{uuid.uuid4().hex}.tar.gz"
        extracted = False
        try:
            with tempfile.TemporaryDirectory() as temp:
                archive = Path(temp) / "tree.tar.gz"
                with self._timing("upload_pack", target=str(target_dir)):
                    await asyncio.to_thread(pack_dir_to_file, source_dir, archive)
                with self._timing(
                    "upload_transfer",
                    target=str(target_dir),
                    bytes=archive.stat().st_size,
                ):
                    # The archive is temporary data in an existing /tmp directory.
                    # Its entries carry the destination permissions, so no remote
                    # mkdir or chmod command is needed for the archive itself.
                    async with asyncio.timeout(self._transfer_timeout):
                        with archive.open("rb") as stream:
                            await self._env.write_file(
                                remote,
                                iter(lambda: stream.read(65536), b""),
                                mode=0o600,
                            )
                with self._timing("upload_extract", target=str(target_dir)):
                    await self._checked(
                        f"trap 'rm -f {remote}' EXIT; "
                        + remote_unpack_command(remote, target_dir)
                        + " --same-permissions --no-same-owner"
                    )
                    extracted = True
        finally:
            if not extracted:
                await self._checked(f"rm -f {shlex.quote(remote)}")

    async def download_dir(self, source_dir, target_dir):
        remote = f"/tmp/harbor-{uuid.uuid4().hex}.tar.gz"
        try:
            await self._checked(remote_pack_command(source_dir, remote))
            with tempfile.TemporaryDirectory() as temp:
                archive = Path(temp) / "tree.tar.gz"
                await self.download_file(remote, archive)
                await asyncio.to_thread(extract_dir_from_file, archive, target_dir)
        finally:
            await self._checked(f"rm -f {shlex.quote(remote)}")
