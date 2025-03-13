#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -e

# Function to print usage
print_usage() {
    echo "Usage: $0 --original-ref <original_tag_or_branch> --fork-repo <fork_repo_url> --fork-ref <fork_tag_or_branch> --output <patch_output_path>"
    echo
    echo "Arguments:"
    echo "  --original-ref    The tag or branch name from the original vllm-project/vllm repo"
    echo "  --fork-repo   The URL of the forked repository"
    echo "  --fork-ref    The tag or branch name from the forked repository"
    echo "  --output      Path where the generated patch file should be saved"
    echo
    echo "Example:"
    echo "  $0 --original-ref v0.2.0 --fork-repo https://github.com/user/vllm.git --fork-ref feature-branch --output ./my-patch.diff"
    exit 1
}

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --original-ref)
            ORIGINAL_REF="$2"
            shift 2
            ;;
        --fork-repo)
            FORK_REPO="$2"
            shift 2
            ;;
        --fork-ref)
            FORK_REF="$2"
            shift 2
            ;;
        --output)
            PATCH_OUTPUT="$2"
            shift 2
            ;;
        *)
            print_usage
            ;;
    esac
done

# Check if all required arguments are provided
if [ -z "$ORIGINAL_REF" ] || [ -z "$FORK_REPO" ] || [ -z "$FORK_REF" ] || [ -z "$PATCH_OUTPUT" ]; then
    print_usage
fi

# Convert patch output path to absolute path if it's relative
if [[ ! "$PATCH_OUTPUT" = /* ]]; then
    PATCH_OUTPUT="$(pwd)/${PATCH_OUTPUT}"
fi

TEMP_DIR=$(mktemp -d)

# Clean up temp directory on script exit
trap 'rm -rf "$TEMP_DIR"' EXIT

# Clone original vLLM to a temp directory
git clone https://github.com/vllm-project/vllm.git "$TEMP_DIR/original_vllm"

cd "$TEMP_DIR/original_vllm"

git remote add fork "$FORK_REPO"
git fetch fork "$FORK_REF"
git diff "$ORIGINAL_REF" fork/"$FORK_REF" --  . ':(exclude)*.png' > "$PATCH_OUTPUT"

echo "Patch created successfully: $PATCH_OUTPUT"