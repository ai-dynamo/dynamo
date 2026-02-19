# GPU Failover POC: Decisions

## Decision 1: K8s Layout

**Intra-pod with shared GPU access via DRA.**

Primary engine, shadow engine, and GMS run as containers within a single pod. GPUs are shared across containers using a DRA `ResourceClaimTemplate` where all containers reference the same claim ([reference example](https://github.com/NVIDIA/k8s-dra-driver-gpu/blob/8effb048f94b3f18338a6f93527352cda82ee385/demo/specs/quickstart/v1/gpu-test2.yaml)). UDS socket sharing between containers uses an `emptyDir` volume.

### GMS Container

- All GMS processes are bundled into a single sidecar container: `devices × {weights, kv_cache}` processes per container.
- Any child process death causes the container to exit (`wait -n` pattern), triggering a kubelet restart.
- A `startupProbe` gates on all GMS sockets being ready before kubelet unblocks engine containers. Uses sidecar init container (`restartPolicy: Always`, K8s 1.29+).

### Engine Restart on GMS Failure

- Engines must restart when GMS fails. Options under evaluation:
  - Application-level handler: engine detects GMS connection loss (broken pipe on UDS) and self-terminates.
  - Canary liveness probe on engine containers checking GMS socket availability (less reliable if GMS restarts faster than probe interval).
- **TODO**: Verify that engines reliably restart on GMS failure in practice.
