#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Analyze a folded-stack file from perf (on-CPU) or offcputime-bpfcc (off-CPU).

Usage:
    python3 analyze_folded.py <file.folded> [--offcpu]

Folded format (one line per stack):  frame;frame;...;leaf  <count>
 - perf on-CPU (stackcollapse-perf.pl): count = samples; leaf = where CPU was spent.
 - offcputime-bpfcc -df: count = microseconds blocked; a literal "-" frame
   separates the USER stack (root->leaf) from the KERNEL stack. The innermost
   USER frame (just before "-") is what called into the blocking syscall.

On-CPU: prints top self-time (exclusive) leaves -> where compute goes.
Off-CPU (--offcpu): prints top innermost-user-frames + a primitive breakdown
   (futex/park vs epoll/network vs lock vs rayon) -> what threads wait on.
"""
import re
import sys
from collections import defaultdict


def clean(fr):
    fr = re.sub(r"::h[0-9a-f]{16}.*", "", fr)
    fr = re.sub(r"\.llvm\..*", "", fr)
    for a, b in [
        ("$LT$", "<"),
        ("$GT$", ">"),
        ("$u20$", " "),
        ("$C$", ","),
        ("$u7b$", "{"),
        ("$u7d$", "}"),
        ("..", "::"),
    ]:
        fr = fr.replace(a, b)
    return fr[:100]


def load(path):
    stacks = []
    for line in open(path):
        line = line.rstrip("\n")
        i = line.rfind(" ")
        if i < 0:
            continue
        try:
            v = int(line[i + 1 :])
        except ValueError:
            continue
        stacks.append((line[:i].split(";"), v))
    return stacks


def oncpu(stacks):
    tot = sum(v for _, v in stacks)
    agg = defaultdict(int)
    for frames, v in stacks:
        agg[clean(frames[-1])] += v
    print(f"total samples={tot}; top self-time (exclusive) leaves:")
    for fr, v in sorted(agg.items(), key=lambda x: -x[1])[:20]:
        print(f"{100*v/tot:5.1f}%  {fr}")


def offcpu(stacks):
    tot = sum(v for _, v in stacks)
    by_frame = defaultdict(int)
    by_cat = defaultdict(int)
    for frames, v in stacks:
        k = frames.index("-") if "-" in frames else len(frames)
        iu = clean(frames[k - 1]) if k > 0 else clean(frames[-1])
        by_frame[iu] += v
        s = ";".join(frames).lower()
        leaf = frames[k - 1].lower() if k > 0 else ""
        if "rayon" in s:
            cat = "rayon pool (idle/spin)"
        elif "epoll" in leaf or "epoll" in s and "io" in s:
            cat = "epoll / network wait"
        elif "mutex" in s or "rwlock" in s or "parking_lot" in s or "__lll_lock" in s:
            cat = "LOCK (mutex/rwlock/glibc)"
        elif "mpsc" in s or "channel" in s or "notify" in s or "semaphore" in s:
            cat = "channel/notify/semaphore"
        elif "syscall" in leaf or "futex" in s:
            cat = "park/idle (futex)"
        else:
            cat = "other"
        by_cat[cat] += v
    print(f"total off-CPU={tot/1e6:.0f} core-s; innermost USER frame before blocking:")
    for fr, v in sorted(by_frame.items(), key=lambda x: -x[1])[:18]:
        print(f"{100*v/tot:5.1f}%  {v/1e6:7.1f}s  {fr}")
    print("\nby primitive category:")
    for c, v in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f"{100*v/tot:5.1f}%  {c}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    stacks = load(sys.argv[1])
    if not stacks:
        print("no parseable stacks (empty/failed capture?)")
        sys.exit(1)
    (offcpu if "--offcpu" in sys.argv else oncpu)(stacks)
