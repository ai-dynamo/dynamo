#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Sweep the publish modes for the pure two-process GMS byte-sharing probe.
# Fresh server per mode (persist-on-abort). Writer and reader are separate processes.
set -u
export PYTHONPATH=/tmp/gmsoverride:${PYTHONPATH:-}

for mode in none sync unmap commit; do
    echo "================ MODE: $mode ================"
    SOCK=/tmp/gmsipc_$mode.sock; AF=/tmp/gmsipc_$mode.json; SRVLOG=/tmp/gmsipc_srv_$mode.log
    rm -f "$SOCK" "$AF" "$SRVLOG"
    python3 -m gpu_memory_service --device 0 --tag kv_cache --persist-on-abort \
        --socket-path "$SOCK" > "$SRVLOG" 2>&1 &
    SPID=$!
    for i in $(seq 1 30); do grep -q "Server started" "$SRVLOG" 2>/dev/null && break; sleep 1; done
    python3 /tmp/gms_ipc_test.py writer "$SOCK" "$mode" "$AF" 2>&1 || echo "[A] writer errored"
    python3 /tmp/gms_ipc_test.py reader "$SOCK" "$AF" 2>&1 || echo "[B] reader errored"
    kill -9 "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
    sleep 1
    echo ""
done
echo "==== SUMMARY ===="
grep -H "RESULT" /dev/null 2>/dev/null
