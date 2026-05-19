# Dynamo Observability

For detailed documentation on Observability (Prometheus metrics, tracing, and logging), please refer to [docs/observability/](../../docs/observability/).

## Dynamo Local Resource Monitor (Prometheus Exporter)

[`dynamo_local_resource_monitor.py`](dynamo_local_resource_monitor.py) is a Dynamo-specific Prometheus exporter that tracks per-process resource usage (VRAM, GPU utilization, PCIe bandwidth, CPU, disk I/O, network I/O) for Dynamo inference processes — labeled by model name, process identity, and PID.

### Why this and not the existing observability tools?

Design context lives in [DEP #9065](https://github.com/ai-dynamo/dynamo/issues/9065). Short version:

- **DCGM** scrapes at 1 s and is per-device — it can't attribute VRAM/PCIe to a specific PID, and 1 s smears the 10–200 ms events (cudaMalloc defrag passes, PCIe weight bursts, torch-compile stalls) we need to see during engine startup.
- **Dynamo's request-lifecycle Prometheus metrics** scrape every 1–15 s and only fire after a request lands, so they're blind to startup-phase failures.
- **Generic per-PID exporters** (e.g. `node_exporter`, `process-exporter`) don't walk the Dynamo subprocess lineage (`pytest → python → bash → python`), so one test shows up as four unrelated rows instead of one logical worker.
- **Nsight Systems** is a 980 MB profile-then-analyze tool — right for one-off deep dives, wrong for an always-on CI signal.
- **aiperf** is client-side load + latency, not server-side per-process telemetry, and runs *after* the engine is up.

This exporter samples at 200 ms (PCIe internally at 10/s), groups subprocesses into one logical worker per engine, and labels each per-engine PID with framework + model + test name (e.g. `VLLM::EngineCore, Qwen3-0.6B, test_serve_deployment[aggregated_unified]`) — so Grafana renders parallel engines bin-packed onto one GPU as separate side-by-side series instead of one summed line. That's what makes flaky one-of-N-tests failures diagnosable.

The Prometheus instance that scrapes this exporter is gated behind the `resource-monitor` Docker Compose profile, so the main observability stack comes up without it. To enable it, pass `--profile resource-monitor`:

```bash
docker compose --profile resource-monitor -f deploy/docker-observability.yml up -d
```

A plain `docker compose -f deploy/docker-observability.yml up -d` will skip the `dynamo-resource-monitor` service entirely.

**Run the exporter on the host machine** (not inside a container):

```bash
pip install psutil nvidia-ml-py prometheus-client
python3 dynamo_local_resource_monitor.py --host 0.0.0.0 --port 8051
```

Then verify metrics at `http://<host>:8051/metrics`.

If any dependencies are missing, the script prints the exact `pip install` command needed and exits.

> **Firewall note:** If your host runs UFW (or similar), you must allow port 8051 from the Docker Compose bridge network for `dynamo-resource-monitor` to scrape the exporter:
>
> ```bash
> BRIDGE_IF="br-$(docker network inspect deploy_server --format '{{.Id}}' | cut -c1-12)"
> sudo ufw allow in on "$BRIDGE_IF" to any port 8051 proto tcp
> ```
>
> Verify the scrape target is healthy with `curl -s http://localhost:9091/api/v1/targets | jq '.data.activeTargets[] | {job, health}'`.
