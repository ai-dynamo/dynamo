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

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Data generation and analysis tools for benchmarking",
        prog="datagen",
    )

    # Add subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Create the parser for the "analyze" command
    subparsers.add_parser("analyze", help="Analyze data")

    # Create the parser for the "synthesize" command
    subparsers.add_parser("synthesize", help="Synthesize data")

    args, remaining = parser.parse_known_args()

    if args.command == "analyze":
        # Import and run the analyzer main
        from prefix_data_generator import prefix_analyzer

        sys.argv = [sys.argv[0]] + remaining
        prefix_analyzer.main()
    elif args.command == "synthesize":
        # Import and run the synthesizer main
        from prefix_data_generator import synthesizer

        sys.argv = [sys.argv[0]] + remaining
        synthesizer.main()


if __name__ == "__main__":
    main()
