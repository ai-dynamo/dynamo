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

"""Integration checks against a local, real ate-env guest (no actor lifecycle).

Set ATE_ENV_GUEST_TARGET to a loopback guest daemon using workspace=/.
Only the explicit test commands execute; no agent or model is involved.
"""

import asyncio
import os
import shlex

import pytest
import pytest_asyncio
from ate_env import Client
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from substrate_environment import SubstrateEnvironment

pytestmark = [pytest.mark.integration, pytest.mark.gpu_0]


@pytest_asyncio.fixture
async def guest(tmp_path):
    target = os.environ.get("ATE_ENV_GUEST_TARGET")
    if not target:
        pytest.skip("ATE_ENV_GUEST_TARGET is required")
    policy = tmp_path / "egress.yaml"
    policy.write_text("rules: []\n")  # Never submitted in guest-only tests.
    environment = SubstrateEnvironment(
        endpoint=target,
        template="unused",
        task_name="fixture",
        kube_context="unused",
        egress_policy=str(policy),
        environment_dir=tmp_path,
        environment_name="fixture",
        session_id="fixture__" + "x" * 100,
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=EnvironmentConfig(),
        workdir=str(tmp_path),
        mounts=[
            {"type": "bind", "source": str(tmp_path / "agent"), "target": "/logs/agent"}
        ],
    )
    client = Client(target)
    environment._client = client
    environment._env = client.env("fixture")
    yield environment
    await client.close()


@pytest.mark.asyncio
async def test_bash_cwd_env_and_exit_status(guest, tmp_path):
    script = tmp_path / "source.sh"
    script.write_text("export FROM_SOURCE=loaded\n")
    with guest.scoped_exec_env({"CHECK": "overlay"}):
        result = await guest.exec(
            "set -o pipefail; source ./source.sh; "
            'printf "%s %s %s" "$FROM_SOURCE" "$CHECK" "$PWD"; '
            "echo error >&2; exit 7",
            env={"CHECK": "command"},
        )
    assert result.return_code == 7
    assert result.stdout == f"loaded overlay {tmp_path}"
    assert result.stderr == "error\n"
    assert (await guest.exec("false | true; exit ${PIPESTATUS[0]}")).return_code == 1
    assert len(guest.actor_id) <= 63


@pytest.mark.asyncio
async def test_directory_roundtrip_preserves_files_modes_links(guest, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "empty").mkdir()
    (source / "payload").write_bytes(bytes(range(256)) * 513)
    (source / "script").write_text("#!/bin/bash\nexit 0\n")
    (source / "script").chmod(0o755)
    (source / "link").symlink_to("script")
    remote = str(tmp_path / "remote directory")
    await guest.upload_dir(source, remote)
    target = tmp_path / "download"
    await guest.download_dir(remote, target)
    assert (target / "payload").read_bytes() == (source / "payload").read_bytes()
    assert (target / "script").stat().st_mode & 0o111 == 0o111
    assert (target / "link").is_symlink()
    assert (target / "empty").is_dir()
    single = str(tmp_path / "single-script")
    await guest.upload_file(source / "script", single)
    assert (await guest.exec(f"stat -c %a {shlex.quote(single)}")).stdout == "755\n"


@pytest.mark.asyncio
async def test_timeout_kills_process_group(guest, tmp_path):
    marker = tmp_path / "must-not-exist"
    with pytest.raises(TimeoutError):
        await guest.exec(
            f"(sleep 1; touch {shlex.quote(str(marker))}) & wait",
            timeout_sec=0.2,
        )
    await asyncio.sleep(1.1)
    assert not marker.exists()
    assert (await guest.exec("echo still-alive")).stdout == "still-alive\n"


@pytest.mark.asyncio
async def test_cancellation_kills_process_group(guest, tmp_path):
    marker = tmp_path / "cancelled-child"
    started = tmp_path / "started"
    task = asyncio.create_task(
        guest.exec(
            f"touch {shlex.quote(str(started))}; "
            f"(sleep 1; touch {shlex.quote(str(marker))}) & wait"
        )
    )
    async with asyncio.timeout(5):
        while not started.exists():
            await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(1.1)
    assert not marker.exists()


@pytest.mark.asyncio
async def test_parallel_commands_and_unsupported_user(guest):
    results = await asyncio.gather(*(guest.exec(f"echo {i}") for i in range(4)))
    assert [r.stdout for r in results] == [f"{i}\n" for i in range(4)]
    with pytest.raises(ValueError, match="root"):
        await guest.exec("id", user="nobody")
