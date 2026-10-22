# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

if "PYTHONHASHSEED" not in os.environ:
    os.environ["PYTHONHASHSEED"] = "0"

if __name__ == "__main__":
    from dynamo.common.snapshot.restore_context import maybe_run_restore_standby_mode

    # In restore mode (SNAPSHOT_RESTORE_STANDBY=1), before importing TRT-LLM,
    # write selected restore-time env vars to snapshot-control/restore-context.json
    # and exec `sleep infinity` without initializing CUDA or backend/runtime state.
    maybe_run_restore_standby_mode()

    from dynamo.common.snapshot.lifecycle import is_snapshot_enabled

    if is_snapshot_enabled():
        # Open MPI's UCX transport enables TCP keepalive between ranks. Snapshot
        # stops network traffic before it freezes each rank, so unanswered probes
        # make the kernel drop those connections and the restored engine fails
        # its first collective. The ob1 TCP transport sets no keepalive. Open MPI
        # starts when TRT-LLM is imported, so set this first.
        os.environ.setdefault("OMPI_MCA_pml", "ob1")
        os.environ.setdefault("OMPI_MCA_btl", "tcp,self")

    from dynamo.trtllm.main import main

    main()
