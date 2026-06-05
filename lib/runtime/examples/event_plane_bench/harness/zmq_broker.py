#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal ZMQ XSUB/XPUB broker for DIS-2172 brokered-ZMQ event-plane test.

dynamo's event plane already supports broker mode (DYN_ZMQ_BROKER_URL); it just
needs an XSUB/XPUB proxy to point at. Publishers connect to XSUB (:5555),
subscribers connect to XPUB (:5556). This collapses the brokerless O(p×s)
direct mesh into O(p+s) connections (at the cost of one extra hop + a central
process). Run one of these per benchmark cell.
"""
import zmq


def main() -> None:
    ctx = zmq.Context(1)
    xsub = ctx.socket(zmq.XSUB)
    xsub.bind("tcp://0.0.0.0:5555")  # publishers' PUB connect here
    xpub = ctx.socket(zmq.XPUB)
    xpub.bind("tcp://0.0.0.0:5556")  # subscribers' SUB connect here
    try:
        zmq.proxy(xsub, xpub)
    except KeyboardInterrupt:
        pass
    finally:
        xsub.close(0)
        xpub.close(0)
        ctx.term()


if __name__ == "__main__":
    main()
