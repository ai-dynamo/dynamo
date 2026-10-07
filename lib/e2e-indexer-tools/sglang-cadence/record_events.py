#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record every SGLang KV-event ZMQ multipart message with its receive time (msgpack stream)."""
import argparse
import signal
import time

import msgpack
import zmq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="tcp://127.0.0.1:5557")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.RCVHWM, 0)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.connect(a.endpoint)
    stop = {"v": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("v", True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("v", True))
    n = 0
    with open(a.out, "wb") as f:
        packer = msgpack.Packer()
        while not stop["v"]:
            if not sub.poll(200):
                continue
            frames = sub.recv_multipart()
            f.write(packer.pack([time.time(), frames]))
            n += 1
            if n % 1000 == 0:
                f.flush()
    print(f"recorded {n} messages")


if __name__ == "__main__":
    main()
