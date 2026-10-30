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

The template must contain the selected task's built image and ate-env-guest.
This example supports Linux, root, one container, and public task networking.
CPU and memory limits are configured on the template. The provider applies
the supplied egress policy to each actor.
"""

import asyncio
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

from ate_env import Client, EnvError, NotFoundError
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentResourceCapabilities
from harbor.environments.tar_transfer import (
    extract_dir_from_file,
    pack_dir_to_file,
    remote_pack_command,
    remote_unpack_command,
)


class SubstrateEnvironment(BaseEnvironment):
    def __init__(
        self,
        *args,
        endpoint: str,
        kube_context: str,
        egress_policy: str,
        template: str | None = None,
        task_name: str | None = None,
        template_map: dict[str, str] | None = None,
        atespace: str = "ate-env",
        ate_cli: str = "kubectl-ate",
        ate_endpoint: str | None = None,
        ate_token_file: str | None = None,
        workdir: str = "/testbed",
        ready_timeout: float = 180,
        transfer_timeout: float = 600,
        uploads: dict[str, str] | None = None,
        **kwargs,
    ):
        self._endpoint = endpoint
        self._template = template
        self._task_name = task_name
        self._template_map = template_map
        self._kube_context = kube_context
        self._egress_policy = Path(egress_policy).resolve()
        self._ate_cli = ate_cli
        self._ate_endpoint = ate_endpoint
        self._ate_token_file = ate_token_file
        self._atespace = atespace
        self._workdir = workdir
        self._ready_timeout = ready_timeout
        self._transfer_timeout = transfer_timeout
        self._uploads = uploads or {}
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
        }
        self.trial_paths.trial_dir.mkdir(parents=True, exist_ok=True)
        (self.trial_paths.trial_dir / "substrate.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        self._client = Client(self._endpoint)
        # Keep the handle before create so cleanup can address a timed-out create.
        self._env = self._client.env(self.actor_id, atespace=self._atespace)
        try:
            async with asyncio.timeout(self._ready_timeout):
                with self._timing("create_actor"):
                    await self._client.create(
                        self.actor_id,
                        atespace=self._atespace,
                        template_name=self._template,
                        template_atespace=self._atespace,
                    )
                with self._timing("egress_policy"):
                    await self._create_egress_policy()
                with self._timing("first_guest_command"):
                    while True:
                        try:
                            # Guest traffic activates the actor; state polling does not.
                            result = await self.exec("true", cwd="/", timeout_sec=5)
                            if result.return_code == 0:
                                break
                        except (EnvError, TimeoutError):
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
                await self.stop(delete=True)
            except Exception:
                self.logger.exception(
                    "Failed to delete actor %s after startup failure", self.actor_id
                )
            raise

    async def _create_egress_policy(self):
        # The env API has no policy RPC. Use the upstream control-plane CLI.
        cli_flags = []
        if self._ate_endpoint:
            cli_flags.extend(["--endpoint", self._ate_endpoint])
        if self._ate_token_file:
            cli_flags.extend(["--token-file", self._ate_token_file])
        proc = await asyncio.create_subprocess_exec(
            self._ate_cli,
            *cli_flags,
            "--context",
            self._kube_context,
            "create",
            "egress-policy",
            self.actor_id,
            "-a",
            self._atespace,
            "-f",
            str(self._egress_policy),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            _, stderr = await proc.communicate()
            if proc.returncode:
                raise RuntimeError(
                    f"Actor egress policy failed: {stderr.decode(errors='replace')}"
                )
        finally:
            if proc.returncode is None:
                os.killpg(proc.pid, signal.SIGKILL)
                await proc.wait()

    async def stop(self, delete: bool):
        try:
            if self._env is not None and delete:
                async with asyncio.timeout(60):
                    with contextlib.suppress(NotFoundError):
                        await self._env.delete()
            elif self._env is not None:
                self.logger.info(
                    "Retaining actor %s in %s", self.actor_id, self._atespace
                )
        finally:
            if self._client is not None:
                await self._client.close()
            self._client = self._env = None

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
