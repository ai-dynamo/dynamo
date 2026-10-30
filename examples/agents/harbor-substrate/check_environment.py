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

"""Create two independent actors to check a prepared image or golden template."""

import argparse
import asyncio
import contextlib
import json
import time
import uuid

from ate_env import Client, EnvError, NotFoundError


async def check(args) -> None:
    client = Client(args.endpoint)
    try:
        for _ in range(2):
            actor = "prepare-" + uuid.uuid4().hex
            handle = client.env(actor, atespace=args.atespace)
            started = time.perf_counter()
            try:
                async with asyncio.timeout(args.timeout):
                    await client.create(
                        actor,
                        atespace=args.atespace,
                        template_name=args.template,
                        template_atespace=args.atespace,
                    )
                    while True:
                        try:
                            async with asyncio.timeout(5):
                                probe = await handle.start_process(["true"], timeout=5)
                                async for chunk in probe.output(follow=True):
                                    if (
                                        chunk.exit is not None
                                        and chunk.exit.exit_code != 0
                                    ):
                                        raise RuntimeError(
                                            "Guest readiness probe failed"
                                        )
                            break
                        except (EnvError, TimeoutError):
                            await asyncio.sleep(1)
                    process = await handle.start_process(
                        [
                            "bash",
                            "-lc",
                            "set -e; test ! -e /tmp/harbor-clone-marker; "
                            "touch /tmp/harbor-clone-marker; "
                            'cd /testbed; test -z "$(git status --porcelain --untracked-files=no)"; '
                            "git rev-parse HEAD; node --version; pi --version; "
                            "cd /root/pi-dynamo-provider; sha256sum -c /opt/pi-provider.sha256",
                        ],
                        timeout=args.timeout,
                    )
                    exit_code = None
                    async for chunk in process.output(follow=True):
                        if chunk.stdout:
                            print(chunk.stdout.decode(errors="replace"), end="")
                        if chunk.stderr:
                            print(chunk.stderr.decode(errors="replace"), end="")
                        if chunk.exit is not None:
                            exit_code = chunk.exit.exit_code
                    if exit_code != 0:
                        raise RuntimeError(
                            f"Prepared environment check failed: {exit_code}"
                        )
                    print(
                        json.dumps(
                            {
                                "actor": actor,
                                "template": args.template,
                                "elapsed_sec": time.perf_counter() - started,
                            }
                        )
                    )
            finally:
                async with asyncio.timeout(60):
                    with contextlib.suppress(NotFoundError):
                        await handle.delete()
    finally:
        await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True)
    parser.add_argument("--endpoint", default="ate-env-api.ate-env.svc:7777")
    parser.add_argument("--atespace", default="ate-env")
    parser.add_argument("--timeout", type=int, default=1200)
    asyncio.run(check(parser.parse_args()))


if __name__ == "__main__":
    main()
