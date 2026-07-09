# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

from dynkv.admission import run_admission
from dynkv.admission_aof import run_admission_aof_rewrite
from dynkv.admission_replication import run_admission_replication
from dynkv.aof import run_aof_rewrite
from dynkv.core_suite import run
from dynkv.gc_bounded import run_bounded_gc
from dynkv.gc_oversized import run_oversized_chunked_gc
from dynkv.gc_persistence import run_chunked_gc_persistence
from dynkv.gc_replication import run_gc_replication
from dynkv.fuzz import run_malformed_wire_fuzz
from dynkv.leases import run_worker_owner_leases
from dynkv.limits import run_wire_limits
from dynkv.registration import run_multi_rank_registration
from dynkv.replication import run_replication


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--module", required=True)
    args = parser.parse_args()
    run(args.server, args.module)
    run_malformed_wire_fuzz(args.server, args.module)
    run_aof_rewrite(args.server, args.module)
    run_replication(args.server, args.module)
    run_multi_rank_registration(args.server, args.module)
    run_wire_limits(args.server, args.module)
    run_worker_owner_leases(args.server, args.module)
    run_bounded_gc(args.server, args.module)
    run_chunked_gc_persistence(args.server, args.module)
    run_oversized_chunked_gc(args.server, args.module)
    run_gc_replication(args.server, args.module)
    run_admission(args.server, args.module)
    run_admission_aof_rewrite(args.server, args.module)
    run_admission_replication(args.server, args.module)
