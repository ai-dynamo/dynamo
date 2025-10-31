# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -x

# nvidia-smi -lgc 1965" ### This way doesn't work on GB200-NVL. bug 5069775

hostname
nvidia-smi

MAX_GPU_CLOCK=$(nvidia-smi -q -d CLOCK | grep -m 1 -A 1 Max | awk '/Graphics/ {print $3}')
MAX_MEM_CLOCK=$(nvidia-smi -q -d CLOCK | grep -m 1 -A 4 Max | awk '/Memory/ {print $3}')

echo "Setting application clock to Mem Clock: $MAX_MEM_CLOCK and GPU Clock: $MAX_GPU_CLOCK."

sudo nvidia-smi -rgc
sudo nvidia-smi -ac $MAX_MEM_CLOCK,$MAX_GPU_CLOCK