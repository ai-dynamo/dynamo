#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Check what UCX transport NIXL selects for CPU-to-CPU same-node transfers.
#
# Reads the UCX "self cfg" line (loopback transport selection) and interprets it.
# Note: "self cfg" shows what transports UCX picked for THIS process's loopback.
# The actual inter-process path (frontend→backend) may differ; use
# UCX_LOG_LEVEL=debug to capture "key cfg" lines between two live processes.
#
# Usage:
#   bash check_ucx_transport.sh                         # default UCX settings
#   UCX_MM_ERROR_HANDLING=y bash check_ucx_transport.sh # Mikhail's sysv fix
#   UCX_TLS=cma,self bash check_ucx_transport.sh        # force CMA only

set -euo pipefail

echo "=== UCX transport check for NIXL CPU-to-CPU same-node ==="
echo

echo "--- Active UCX env overrides ---"
env | grep '^UCX_' | sed 's/^/  /' || echo "  (none)"
echo

echo "--- Available transports on this node ---"
if command -v ucx_info &>/dev/null; then
    ucx_info -d 2>/dev/null | grep -i "Transport" | awk '{print $NF}' | sort -u | sed 's/^/  /'
else
    echo "  (ucx_info not in PATH)"
fi
echo

# Write probe to a temp file to avoid heredoc quoting issues
PROBE=$(mktemp /tmp/nixl_probe_XXXXXX.py)
RAW=$(mktemp /tmp/ucx_raw_XXXXXX.txt)
trap 'rm -f "${PROBE}" "${RAW}" /tmp/ucx_raw_peer.txt 2>/dev/null' EXIT

cat > "${PROBE}" <<'EOF'
import asyncio, torch, sys
import dynamo.nixl_connect as nc

async def main():
    c = nc.Connector()
    await c.initialize()
    buf = torch.zeros(4096, dtype=torch.uint8)
    d = nc.Descriptor(buf)
    await c.create_readable(d)

asyncio.run(main())
EOF

echo "--- UCX self cfg (UCX_LOG_LEVEL=info) ---"
UCX_LOG_LEVEL=info python3 "${PROBE}" > "${RAW}" 2>&1 || true
grep -iE 'self cfg' "${RAW}" | head -3 | sed 's/^/  /' || echo "  (not found — see note below)"
echo

echo "--- Interpretation ---"
python3 - "${RAW}" <<'PYEOF'
import re, sys

with open(sys.argv[1]) as f:
    raw = f.read()

m = re.search(r'self cfg#\d+\s+(.*)', raw)
if not m:
    print("  self cfg line not found.")
    print("  Possible reason: UCX_LOG_LEVEL=info is not being picked up.")
    print("  Try: UCX_LOG_LEVEL=debug python3 -c 'import dynamo.nixl_connect' 2>&1 | grep -iE 'cfg#|transport'")
    sys.exit(0)

cfg = m.group(1)
c = cfg.lower()
print(f"  UCX self cfg:  {cfg[:120]}")
print()

# Classify transport
if re.search(r'rc_mlx5|ud_mlx5|dc_mlx5|rc_verbs|ud_verbs', c):
    tl   = 'InfiniBand RDMA (rc_mlx5 / ud_mlx5)'
    perf = 'Excellent for cross-node. For same-node, CMA is typically faster.'
    note = (
        "  NOTE: IB selected. On same-node transfers, CMA/SysV avoids the HCA entirely.\n"
        "  To force CMA: UCX_TLS=cma,self bash check_ucx_transport.sh\n"
        "  To benchmark: SINGLE_GPU=true bash bench_mm_transfer.sh nixl"
    )
elif 'cma' in c:
    tl   = 'CMA (cross-memory attach, kernel process_vm_readv)'
    perf = 'Fast same-node (~5-10 GB/s). Good for NIXL on this machine.'
    note = ''
elif 'sysv' in c:
    tl   = 'SysV shared memory'
    perf = 'Fast same-node (~5-10 GB/s).'
    note = ''
elif 'posix' in c:
    tl   = 'POSIX shared memory'
    perf = 'Fast same-node (~5-10 GB/s).'
    note = ''
elif 'tcp' in c:
    tl   = 'TCP (software loopback)'
    perf = 'Slow for large payloads (~100-500 MB/s, ~20-50ms for 3.6 MB).'
    note = (
        "  ACTION: TCP detected — shared-memory transport not active.\n"
        "  On nodes where sysv exists but fails with 'no peer failure handler':\n"
        "    UCX_MM_ERROR_HANDLING=y bash check_ucx_transport.sh\n"
        "  On nodes without CMA/XPMEM/KNEM, set this env var before running\n"
        "  bench_mm_transfer.sh or the production stack."
    )
else:
    tl   = '(unrecognized — inspect self cfg line above)'
    perf = '(unknown)'
    note = ''

print(f"  Transport : {tl}")
print(f"  Expected  : {perf}")
if note:
    print()
    print(note)
PYEOF

echo
echo "--- How to check the actual inter-process (frontend→backend) path ---"
echo "  UCX_LOG_LEVEL=debug captures 'key cfg' lines showing the peer transport."
echo "  Run with NIXL stack active and grep the log:"
echo "    grep -i 'key cfg' /tmp/bench_mm_transfer/stack_nixl.log | head -5"
echo "  Or add UCX_LOG_LEVEL=debug to the launch script env and rerun bench_mm_transfer.sh."
