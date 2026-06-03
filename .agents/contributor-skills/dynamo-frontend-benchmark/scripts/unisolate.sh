#!/usr/bin/env bash
# Revert bench/isolate.sh: give all cores back to every slice, restore IRQ
# balancing, and tear down the frontend's bench.slice.
#   sudo bash bench/unisolate.sh
# (A reboot also clears everything, since isolate.sh used --runtime.)
set -uo pipefail
[[ $EUID -eq 0 ]] || { echo "ERROR: run as root (sudo bash bench/unisolate.sh)"; exit 1; }

echo "[unisolate] stopping any isolated frontend (bench.slice) ..."
systemctl stop bench.slice 2>/dev/null || true

echo "[unisolate] restoring slices to all cpus ..."
# `revert --runtime` drops the transient drop-ins; fall back to explicit 0-23.
if ! systemctl revert --runtime system.slice user.slice init.scope 2>/dev/null; then
    for unit in system.slice user.slice init.scope; do
        systemctl set-property --runtime "$unit" AllowedCPUs=0-23 2>/dev/null || true
    done
fi

echo "[unisolate] restoring IRQ affinity + irqbalance ..."
for f in /proc/irq/*/smp_affinity_list; do echo 0-23 > "$f" 2>/dev/null || true; done
systemctl start irqbalance 2>/dev/null || true

echo "[unisolate] verify — effective cpus (want 0-23):"
for unit in system.slice user.slice; do
    echo "  $unit: $(cat /sys/fs/cgroup/$unit/cpuset.cpus.effective 2>/dev/null)"
done
echo "[unisolate] done."
