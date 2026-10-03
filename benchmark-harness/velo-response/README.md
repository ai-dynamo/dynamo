# Velo response comparison on Tyche

The [completed September 23 comparison](report/README.md) includes all 16 measured
runs, packet and end-to-end results, retained errors, source pins, and build hashes.

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

Require complete exports and 2,048 active KV sources on both frontends. Small
error counts remain in the directional comparison and are reported for every
run. The harness accepts error fractions up to 0.01%; it does not hide those
records or their error types. Keep failed artifacts. Take profiles outside timing
runs. Report client saturation and different fabrics as comparison limits.
Report packet load separately from throughput and tail latency. The provisional
limits versus Dynamo TCP are 5% throughput or p99 latency regression and 10%
serving CPU per request regression. Also compare every metric with QUIC.

The scripts use the existing Tyche cache, model, dataset, AIPerf and Python
environment paths in `configs/template.json`. `prepare_remote.py` installs the
established ZMQ capacity overlay and the Python launcher once, before building.
It requires the prior `dynamo-tyche-hf-runtime-m2048-20260917` source directory.
Retain the resulting overlay patch and hashes with the report.

Stage this worktree at `$ROOT/src/dynamo`, where `ROOT` is the path in the
template. Copy `harness/` to `$ROOT/harness` and the template to `$ROOT/configs`.
Run builds through `srun` in a Tyche allocation. Run `verify_hardware.py` once on
each allocated node. Analysis also checks each process's UCX active-message
lane against its selected RDMA device. After verifying the two RDMA
ports on the allocated nodes, the campaign sequence is:

```bash
python3 "$ROOT/harness/make_configs.py" --root "$ROOT" \
  --dynamo "$DYNAMO_REV" --velo "$VELO_REV" \
  --rdma-device mlx5_0:1 --rdma-device-numa1 mlx5_4:1
bash "$ROOT/harness/build_grace.sh"
python3 "$ROOT/harness/freeze.py" --root "$ROOT" \
  --main "$MAIN_REV" --velo "$VELO_REV"
bash "$ROOT/harness/run_condition.sh" smoke-velo-tcp
bash "$ROOT/harness/run_condition.sh" smoke-velo-rdma
bash "$ROOT/harness/run_condition.sh" preflight-velo-rdma
bash "$ROOT/harness/controller.sh"
python3 "$ROOT/harness/summarize_environment.py"
python3 "$ROOT/harness/report.py"
```

The allocation keeper writes `$ROOT/control/job-id`. `freeze.py --verify`
checks the source revision, overlay, lockfiles and binary before every run.
Use new run labels after a failed attempt; do not replace its artifacts.
The saturation preflight uses concurrency 8,192 and per-process worker metrics.
It is separate from the four-way timing matrix.

`run_condition.sh` stops before a new run when less than 30 minutes remain.
For a continuation, preserve `manifests/nodes.txt` as
`manifests/campaign-original-nodes.txt`, queue the same nodes on a normal
partition, and save the new job ID in `control/continuation-job-id`.
After the controller exits with status 75 and the previous run is complete,
release the idle allocation. When the new keeper writes its job ID, run
`harness/resume_continuation.sh`. It checks the node order, devices, link speeds,
MTU and frozen build, then resumes only unfinished conditions. Analysis uses
the allocation ID saved for each run and includes all 16 measured runs.

No default change is part of this work.
