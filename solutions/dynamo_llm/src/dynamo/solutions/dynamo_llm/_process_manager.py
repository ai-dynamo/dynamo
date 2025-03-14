# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass
class Command:
    args: list[str]
    name: str = ""
    call: bool = False
    swallow_output: bool = False
    environment: dict[str, str] = None
    delay: int = 0


class ProcessManager:
    def __init__(self, args, unknown_args, sleep_between_launch=0.5):
        self._processes = []
        self._dry_run = args.dry_run
        self._commands = []
        self._sleep = sleep_between_launch

    def add_commands(self, commands: list[Command]):
        if not commands:
            return
        for command in commands:
            self.add_command(command)

    def add_command(self, command: Command):
        self._commands.append(command)

    def start(self):
        for command in self._commands:
            if not command or not command.args:
                continue

            starting = "Starting" if not command.call else "Calling"

            print("----------------------------------")
            print(f"{starting} {command.name} Process")
            print()
            if command.environment:
                print(f"\t {command.environment}")
            print(f"\t {' '.join(command.args)}")
            print("----------------------------------")

            if not self._dry_run:
                env = os.environ.copy()
                if command.environment:
                    env.update(command.environment)

                if command.call:
                    result = subprocess.call(command.args, env=env)
                    if result != 0:
                        sys.exit(result)
                else:
                    input_ = subprocess.DEVNULL
                    output_ = subprocess.DEVNULL
                    if command.swallow_output:
                        output_ = subprocess.DEVNULL
                    else:
                        output_ = subprocess.PIPE
                    self._processes.append(
                        subprocess.Popen(
                            command.args, stdin=input_, stdout=output_, env=env
                        )
                    )
                    input_ = self._processes[-1].stdout

                    sed_cmd = ["sed", f"s/^/[{command.name}] /"]
                    self._processes.append(
                        subprocess.Popen(sed_cmd, stdin=input_, env=env)
                    )
                    input_.close()

                    delay = max(command.delay, self._sleep)
                    time.sleep(delay)

    def __enter__(self):
        def handler(signum, frame):
            exit_code = 0
            print(f"Handling signal {signum}")

            # for process in self._processes:
            #     print(f"\tterminating: process {process}")
            #     try:
            #         process.terminate()
            #         process.kill()
            #     except Exception:
            #         pass

            sys.exit(exit_code)

        signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)

        for sig in signals:
            try:
                signal.signal(sig, handler)
            except Exception:
                pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Exiting ... ")
        for process in self._processes:
            print(f"\tterminating: process {process}")
            try:
                process.terminate()
                process.kill()
            except Exception:
                pass
