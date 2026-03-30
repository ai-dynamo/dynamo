#!/bin/bash
# Monitor a running sweep on a compute node.
# Usage: bash monitor_sweep.sh <compute-node> [log-file] [interval-seconds]
#
# Examples:
#   bash monitor_sweep.sh gb-nvl-059-compute03
#   bash monitor_sweep.sh gb-nvl-059-compute03 /tmp/srun_sweep.log 120

set -euo pipefail

NODE="${1:?Usage: monitor_sweep.sh <compute-node> [log-file] [interval]}"
LOG="${2:-/tmp/srun_sweep.log}"
INTERVAL="${3:-60}"

while true; do
    echo ""
    echo "========================================"
    date "+%Y-%m-%d %H:%M:%S"
    echo "========================================"

    ssh -o ConnectTimeout=5 "$NODE" "
        echo \"Screen:\" && screen -ls 2>/dev/null || echo '  (none)'
        echo \"\"
        DONE=\$(grep -c 'aiperf request_rate=.*done' $LOG 2>/dev/null || echo 0)
        TOTAL=\$(grep -c 'Config:' $LOG 2>/dev/null || echo 0)
        SKIP=\$(grep -c 'SKIP' $LOG 2>/dev/null || echo 0)
        echo \"Progress: \$DONE done, \$SKIP skipped, \$TOTAL attempted\"
        echo \"\"
        echo \"Current:\"
        grep 'Config:\|Input:' $LOG 2>/dev/null | tail -2
        echo \"\"
        ERRORS=\$(grep -i 'CalledProcessError\|exit code [^0]\|FAILED' $LOG 2>/dev/null | tail -3)
        if [ -n \"\$ERRORS\" ]; then
            echo \"Errors:\"
            echo \"\$ERRORS\"
            echo \"\"
        fi
        echo \"Last output:\"
        tail -5 $LOG 2>/dev/null
    " 2>&1 || echo "  SSH failed"

    sleep "$INTERVAL"
done
