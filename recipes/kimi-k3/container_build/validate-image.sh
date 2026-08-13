#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

recipe_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
image="${1:?usage: $0 <image@sha256:digest>}"
output_dir="${OUTPUT_DIR:-$recipe_dir/out}"

if [[ "$image" != *@sha256:* ]]; then
    echo "validation requires an immutable image@sha256 reference" >&2
    exit 2
fi
if [[ "$(uname -m)" != aarch64 ]]; then
    echo "validation must run on a native aarch64 host" >&2
    exit 2
fi

mkdir -p "$output_dir"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
docker pull "$image"

# Pass the validator through stdin rather than a shared-filesystem bind mount.
docker run --rm --pull=always --gpus all --entrypoint python3 -i "$image" - \
    < "$recipe_dir/validate_gpu.py" \
    2>&1 | tee "$output_dir/gpu-validation.log"
