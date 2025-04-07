#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
import subprocess
import sys


def run_executable(executable_name, args=None, capture_output=True, text=True):
    """
    Runs an executable located in the cli/bin directory.

    Parameters:
        executable_name (str): Name of the executable file.
        args (list): List of arguments to pass to the executable.
        capture_output (bool): Whether to capture stdout and stderr.
        text (bool): If True, returns output as string; otherwise bytes.

    Returns:
        subprocess.CompletedProcess: The result of the executed command.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bin_dir = os.path.join(script_dir, "bin")

    # Construct full path to executable
    executable_path = os.path.join(bin_dir, executable_name)

    # Check if executable exists
    if not os.path.isfile(executable_path):
        raise FileNotFoundError(
            f"Executable '{executable_name}' not found in {bin_dir}"
        )

    # Prepare command
    command = [executable_path]
    if args:
        command = [executable_path] + args
    else:
        command = [executable_path]

    # Run the command using subprocess.run()
    result = subprocess.run(command, capture_output=capture_output, text=text)

    return result


def dynamo_run(args=None):
    """
    Run the dynamo-run executable with the provided arguments.
    If no args provided, passes through sys.argv[1:] to the executable.
    """
    if args is None:
        args = sys.argv[1:]

    # Run with capture_output=False to allow direct stdout/stderr streaming
    result = run_executable("dynamo-run", args=args, capture_output=False)
    return result.returncode


def llmctl(args=None):
    """
    Run the llmctl executable with the provided arguments.
    If no args provided, passes through sys.argv[1:] to the executable.
    """
    if args is None:
        args = sys.argv[1:]
    result = run_executable("llmctl", args=args, capture_output=False)
    return result.returncode


def http(args=None):
    """
    Run the http executable with the provided arguments.
    If no args provided, passes through sys.argv[1:] to the executable.
    """
    if args is None:
        args = sys.argv[1:]
    result = run_executable("http", args=args, capture_output=False)
    return result.returncode


def metrics(args=None):
    """
    Run the metrics executable with the provided arguments.
    If no args provided, passes through sys.argv[1:] to the executable.
    """
    if args is None:
        args = sys.argv[1:]
    result = run_executable("metrics", args=args, capture_output=False)
    return result.returncode
