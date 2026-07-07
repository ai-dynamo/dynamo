#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cat << 'EOF'

DEPRECATED: Ray Backend
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This script demonstrates the Ray backend for Elastic Expert Parallelism (Elastic EP).
The Ray backend is no longer recommended and is maintained for backward compatibility only.

RECOMMENDED APPROACH:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For data parallel and expert parallelism deployments, use the PyTorch multiprocessing
(mp) backend instead. This is the officially recommended approach and provides better
integration with vLLM's distributed execution model.

Use this example instead:
   bash launch/dep.sh

This provides MP-based data parallelism with expert parallelism support and works
seamlessly with Dynamo's distributed deployment infrastructure.

For expert parallelism and dynamic scaling features, see:
   - docs/backends/vllm/vllm-examples.md
   - launch/dep.sh

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ray Backend Notes (Legacy):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If you need to use the Ray backend for advanced use cases:

1. Ensure Ray is installed:
   pip install "ray>=2.55.0"

2. Refer to the vLLM documentation for Ray-based distributed execution:
   https://docs.vllm.ai/en/latest/serving/distributed_serving.html

3. Contact NVIDIA support for guidance on Ray backend deployments.

EOF

exit 1
