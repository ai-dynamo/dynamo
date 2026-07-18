#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu

iterations=${1:-6000}
i=0
while [ "$i" -lt "$iterations" ]; do
    wall_ns=$(date -u +%s%N)
    utc=$(date -u +%FT%T.%NZ)
    uptime=$(cut -d' ' -f1 /host/proc/uptime)
    printf 'BEGIN\t%s\t%s\t%s\n' "$wall_ns" "$uptime" "$utc"
    awk '$1 == "cpu" { print "STAT", $0 }' /host/proc/stat
    awk '
        $1 == "MemAvailable:" || $1 == "Cached:" ||
        $1 == "Dirty:" || $1 == "Writeback:" {
            print "MEM", $0
        }
    ' /host/proc/meminfo
    awk '{ print "PSI", "cpu", $0 }' /host/proc/pressure/cpu
    awk '{ print "PSI", "io", $0 }' /host/proc/pressure/io
    awk '{ print "PSI", "memory", $0 }' /host/proc/pressure/memory
    awk '
        $3 == "nvme2n1" || $3 == "nvme4n1" || $3 == "nvme5n1" ||
        $3 == "nvme6n1" || $3 == "nvme7n1" || $3 == "nvme8n1" ||
        $3 == "nvme9n1" {
            print "DISK", $0
        }
    ' /host/proc/diskstats
    awk '
        {
            gsub(":", "", $1)
            if ($1 == "bond0" || $1 == "ens9f0np0" || $1 == "ens9f1np1")
                print "NET", $0
        }
    ' /host/proc/net/dev
    printf 'END\t%s\n' "$wall_ns"
    i=$((i + 1))
    sleep 0.2
done
