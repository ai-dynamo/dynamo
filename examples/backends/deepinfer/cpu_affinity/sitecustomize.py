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

# Put this file's directory first on the DeepInfer worker's PYTHONPATH (launch_agg.sh does).
#
# deepinfer.utils.bind_cpu pins each rank to an even slice of *all* physical cores for
# *all* physical GPUs, assuming an exclusive node. When the container's cgroup allows only
# some cores (a Kubernetes pod), psutil rejects the slice ("CPU number 0 is not eligible").
# This remaps any slice that falls outside the allowed set onto the rank's share of the
# allowed CPUs. Set DEEPINFER_LOCAL_NGPUS to the GPUs in the container.
import os

_ALLOWED_KEY = "DEEPINFER_ALLOWED_CPUS"
# Capture before any rank binds itself; child ranks inherit it through the env.
if _ALLOWED_KEY not in os.environ:
    os.environ[_ALLOWED_KEY] = ",".join(map(str, sorted(os.sched_getaffinity(0))))


def _install() -> None:
    try:
        import deep_cluster.topology as topology
        import deep_cluster.topology.bind as bind
    except Exception:
        return
    original = bind.set_cpu_affinity

    def set_cpu_affinity(cpus):
        cpus = list(cpus)
        allowed = [int(c) for c in os.environ[_ALLOWED_KEY].split(",")]
        if cpus and not set(cpus) <= set(allowed):
            num_ranks = int(os.environ.get("DEEPINFER_LOCAL_NGPUS", "1"))
            rank = (cpus[0] // len(cpus)) % num_ranks
            per_rank = max(1, len(allowed) // num_ranks)
            cpus = allowed[rank * per_rank : (rank + 1) * per_rank] or allowed
        original(cpus)

    bind.set_cpu_affinity = set_cpu_affinity
    topology.set_cpu_affinity = set_cpu_affinity


_install()
