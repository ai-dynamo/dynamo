# Velo response comparison on Tyche

The adapter is selected with `DYN_RESPONSE_PLANE=velo` and
`DYN_VELO_RESPONSE_TRANSPORT=tcp|ucx`. UCX requires the `velo-ucx` build feature.
TCP remains the default response plane. QUIC is unchanged.

The companion Velo change adds native graceful stop, ticket cancellation and
bounded UCX send admission. Dynamo polls the Velo stream directly. It preserves
its payload codec, prologue and completion checks. A shared process service owns
the Velo transports; a worker owns only its service reference and request streams.

The harness extends `tyche-main-profile-20260923`. The workload template preserves
2,048 speedup-10 workers in 12 processes, two NUMA-bound frontends, AgentX's 336
traces, seed 12345 and concurrency 6,144. Each run starts new services, uses 32
warmups, a 30-second ramp and a 120-second measurement, then drains and exports.

Before the final campaign, refresh Dynamo main, apply the integration, pin the
Velo commit, and build one UCX-enabled binary. Freeze revisions, all three Cargo locks,
features, build flags, source overlays and binary hashes for all modes. The ZMQ
socket-capacity overlay is benchmark-only and must be identical for all modes.

Run one discard per mode, then four measured rounds in this balanced order:

| Round | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 1 | TCP | QUIC | Velo RDMA | Velo TCP |
| 2 | QUIC | Velo TCP | TCP | Velo RDMA |
| 3 | Velo TCP | Velo RDMA | QUIC | TCP |
| 4 | Velo RDMA | TCP | Velo TCP | QUIC |

TCP and QUIC use the current Ethernet network. Select RDMA explicitly with
`UCX_TLS=rc_mlx5,ud_mlx5,self` and a verified `UCX_NET_DEVICES` port. Preserve UCX
lane logs and hardware counter changes to prove the selected path. Do not accept
silent TCP fallback. Hardware tests and the response saturation check must pass
before the full matrix.

`rootcause_telemetry.py` records Ethernet hardware counters, offloads, IRQs,
softirqs, host network counters and RDMA port counters. Keep wire packet counts
separate from host aggregation and completion counts. InfiniBand `port_rcv_data`
and `port_xmit_data` count four-byte units. Packet-size buckets are reported only
when the hardware exposes them. Keep response-only traffic separate from total
frontend traffic where counters permit. Record fabric, speed, MTU, PCI/NUMA
placement and actual process memory placement.

Require complete exports, zero unexpected request errors and 2,048 active KV
sources on both frontends. Keep failed artifacts. Take profiles outside clean
runs. Report client saturation and different fabrics as comparison limits.
Report packet load separately from throughput and tail latency. The provisional
limits versus Dynamo TCP are 5% throughput or p99 latency regression and 10%
serving CPU per request regression. Also compare every metric with QUIC.

No performance result is available yet. No default change is part of this work.
