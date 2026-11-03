# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Paths and settings shared by the stress-campaign scripts. Source it.
# Export CAMPAIGN and NIC before running anything; README.md lists the rest.

# A directory every node reads and writes, holding bench/ (this directory's parent),
# src/, builds/, results/ and control/.
export CAMPAIGN=${CAMPAIGN:?export CAMPAIGN, a directory shared by every node}
BENCH=$CAMPAIGN/bench
# The interface the nodes talk to each other over.
export NIC=${NIC:?export NIC, the network interface between the nodes}

# Rust builds: codegen flags come only from the repository's .cargo/config.toml.
# Put cargo and protoc on PATH (or set CARGO_HOME, RUSTUP_HOME and PROTOC) before sourcing.
unset RUSTFLAGS CARGO_ENCODED_RUSTFLAGS RUSTC_WRAPPER CARGO_BUILD_RUSTFLAGS

# aiperf 0.13.0. The client nodes run it offline, from caches filled beforehand.
AIPERF=${AIPERF:-aiperf}
AIPERF_HF_HOME=${AIPERF_HF_HOME:-$CAMPAIGN/cache/huggingface}
AIPERF_CACHE=${AIPERF_CACHE:-$CAMPAIGN/cache/aiperf}
MODEL=Qwen/Qwen3-0.6B
MODEL_REVISION=c1899de289a04d12100db370d81485cdf75e47ca
DATASET=semianalysis_cc_traces_weka_with_subagents_060526

POOLS=16
POOL_PORT_BASE=9000
FRONTEND_PORT=8000
# Each client node's aiperf: half the campaign's 8192 concurrency.
CLIENT_CONCURRENCY=4096
# aiperf can't process per-request records at these rates (each holds ~870 chunk
# timestamps; the backlog fills a client node's memory in about two minutes at 10k req/s),
# so the phase is short and aiperf only generates load: the frontend's /metrics measure it.
SETTLE_S=15
MEASURE_S=45
