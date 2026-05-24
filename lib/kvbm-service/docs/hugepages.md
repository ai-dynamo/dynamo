# Hugepages for the kvbm-service host-memory pool

The host-memory pool (`HostMemoryPool`) allocates one pinned slab per host-CPU
NUMA node and tries to back each slab with hugepages. This page is the
operator-facing reference: why hugepages matter, how to reserve them, what
the service does when they're missing, and how to read what landed where.

## Why hugepages

The pool keeps tens to hundreds of GiB of pinned host memory live for the
duration of a service process. Every device → host transfer touches the TLB.
At 4 KiB pages a 64 GiB slab is 16 million PTEs; at 2 MiB pages it's 32,768.
TLB pressure dominates the cost of small transfers on hot paths.

2 MiB pages are the practical sweet spot. 1 GiB pages reduce TLB pressure
further but are out of scope for this milestone — neither the config surface
nor the allocator currently asks for them.

## Allocation tiers

`PoolConfig::hugepage_mode` (default `BestEffort`) controls how each slab
tries to land hugepages. The actual outcome is recorded per slab as
`HugepageTier` and surfaced in `/v1/pool` and the
`kvbm_service_pool_bytes_total{node, tier}` gauge.

| mode          | tier ladder per slab                                          | failure mode                               |
|---------------|---------------------------------------------------------------|--------------------------------------------|
| `Disabled`    | `mmap(MAP_PRIVATE \| MAP_ANON)`                               | only on raw allocation failure             |
| `BestEffort`  | `MAP_HUGETLB` → anon + `MADV_HUGEPAGE` → plain anon           | only if all three fail                     |
| `Required`    | `MAP_HUGETLB` only                                            | service refuses to start on shortfall      |

`BestEffort` is the default because it keeps the service bootable while still
making the perf cliff visible — operators see `tier=thp` or `tier=none` in
`/v1/pool` instead of a silent fallback.

## Reserve hugepages on the host

Pick **one** of the following. The boot-time path is more reliable because
hugepages allocate before any process has fragmented memory.

### Boot-time (recommended)

Add to the kernel command line and reboot:

```
default_hugepagesz=2M hugepagesz=2M hugepages=N
```

`N` is the number of 2 MiB pages. For `S` GiB of hugetlb capacity,
`N = S * 512`.

### Runtime

System-wide:

```sh
sudo sysctl -w vm.nr_hugepages=N
```

Per-NUMA-node (preferred on multi-socket boxes — see Grace section):

```sh
echo N | sudo tee /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages
echo N | sudo tee /sys/devices/system/node/node1/hugepages/hugepages-2048kB/nr_hugepages
```

### Verify

```sh
cat /proc/meminfo | grep -i huge
ls /sys/kernel/mm/hugepages/
for n in /sys/devices/system/node/node*/hugepages/hugepages-2048kB/nr_hugepages; do
  echo -n "$n: "; cat "$n"
done
```

The same data flows into `inspect_resources` (`cargo run -p dynamo-memory
--bin inspect_resources`) and into the service's `/v1/pool` snapshot at
runtime.

## Containers

`MAP_HUGETLB` and `cuMemHostRegister` need the right caps and a hugetlbfs
mount.

| concern                     | configuration                                            |
|-----------------------------|----------------------------------------------------------|
| `mlock`/lock budget         | `--ulimit memlock=-1` (or a generous explicit value)     |
| capability for HUGETLB      | `--cap-add IPC_LOCK`                                     |
| hugetlbfs available         | mount `/dev/hugepages` (Docker mounts it by default)     |
| Kubernetes                  | set `resources.limits["hugepages-2Mi"]` on the pod       |

If the service runs under cgroup v2 with a hugetlb controller cap, the cap
is enforced — the kernel returns `ENOMEM` and the slab falls through the
tier ladder (or, in `Required` mode, the service refuses to start).

## Grace / GB200 specifics

The host-memory pool only targets CPU-bearing NUMA nodes. On a 4-GPU GB200
tray that's the two Grace nodes (0 and 1); the four HBM nodes (2, 10, 18,
26) are owned by the GPUs and out of scope. Run `inspect_resources` to
confirm the classification before you size the pool.

NVIDIA-recommended kernel knobs for the host nodes:

- `kernel.numa_balancing=0` — autonuma fights the pool's `mbind` placement.
- `init_on_alloc=0` — avoids a large-alloc latency cliff on Grace.
- 64K base page kernel is NVIDIA's default for Grace. Transparent
  hugepages at 2 MiB (mTHP) require kernel ≥ 6.9; explicit hugetlbfs at
  2 MiB always works.

Per-node hugepage reservation matters more on GB200 than on x86: once Grace
memory fragments, the kernel can fail to find contiguous huge pages on the
node the pool asked for. Use the per-node sysfs path when reserving:

```sh
echo $N | sudo tee /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages
echo $N | sudo tee /sys/devices/system/node/node1/hugepages/hugepages-2048kB/nr_hugepages
```

## Observability

The service surfaces what it actually got in two places.

### `/v1/pool`

```json
{
  "instance_id": "9b…",
  "leased": false,
  "total_bytes": 870440960000,
  "slabs": [
    {
      "numa_node": 0,
      "size_bytes": 435220480000,
      "hugepage_tier": {"Explicit": {"page_size": 2097152}},
      "agent_name": "kvbm-svc:9b…:n0"
    },
    {
      "numa_node": 1,
      "size_bytes": 435220480000,
      "hugepage_tier": "Thp",
      "agent_name": "kvbm-svc:9b…:n1"
    }
  ],
  "hugepage_default_size_bytes": 2097152,
  "thp_enabled": "madvise"
}
```

`hugepage_tier` is one of:

- `{"Explicit": {"page_size": 2097152}}` — `MAP_HUGETLB` succeeded.
- `"Thp"` — fell back to `madvise(MADV_HUGEPAGE)`; kernel may or may not
  have actually promoted the pages.
- `"None"` — plain anon mmap; no hugepage backing.

### Prometheus

```
kvbm_service_pool_bytes_total{node="0", tier="explicit"} 4.3522e+11
kvbm_service_pool_bytes_total{node="1", tier="thp"}       4.3522e+11
kvbm_service_pool_slabs_total{tier="explicit"} 1
kvbm_service_pool_slabs_total{tier="thp"}       1
```

### Logs

`HostMemoryPool::new` always logs a summary line at INFO with the per-node
tier; misconfiguration triggers WARNs:

- `PoolSizing::Ratio(r) > 0.95`
- `requested_per_node > 0.9 * MemTotal` (per node)
- `pool covers < 50% of total host memory` (info-level when sizing is
  explicit and the box is heavily under-utilized)

## Troubleshooting

`tier=None` everywhere with `BestEffort`:
- `cat /proc/meminfo | grep Huge` → `HugePages_Total: 0`. Reserve hugepages
  (see above) and restart.

`tier=Thp` only:
- The hugetlb pool is exhausted or not reserved on the right node. Check
  `/sys/devices/system/node/nodeX/hugepages/hugepages-2048kB/free_hugepages`
  for each host-memory node.

Service fails to start with `hugepages required but unavailable`:
- `hugepage_mode = Required` plus insufficient pool. Either reserve more
  pages, or relax to `BestEffort` if a perf cliff is acceptable.

`/v1/pool` returns 503:
- The service was started via `KvbmService::start` or
  `start_with_container` (no pool). Use `KvbmService::start_with_pool` to
  build one.
