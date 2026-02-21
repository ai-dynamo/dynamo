# vLLM Multiprocessing (mp) Backend Support

## Context

vLLM now supports native multiprocessing for multi-node deployments without requiring Ray. The operator currently hardcodes Ray for all multi-node TP/PP vLLM deployments. We want to:

- Default new deployments to `mp` (multiprocessing)
- Preserve `ray` for existing deployments when the operator is upgraded
- Allow users to explicitly switch between `ray` and `mp`

## Approach: Operator Version Annotation + Explicit Override

Use a **general-purpose operator version annotation** stamped on CREATE by a mutating webhook, plus an optional **explicit override annotation** for direct user control.

### Annotations

- **`nvidia.com/dynamo-operator-version`** (general mechanism): Stamped automatically by the mutating webhook on CREATE. Records which operator version created the DGD. Enables version-gated behavior changes for this and future features.
- **`nvidia.com/vllm-distributed-executor-backend`** (explicit override, optional): User-set annotation. Values: `mp` or `ray`. Takes priority over version-based defaulting when present.

### Why this design

- **General mechanism**: The version annotation handles this use case and any future version-dependent behavior changes -- no new annotations needed per feature.
- **Simple webhook**: Only stamps the version on CREATE; does nothing on UPDATE.
- **No CRD change**: Annotations are free-form metadata.
- **Backward compatible by default**: Absent version annotation means pre-upgrade DGD; the controller defaults to legacy behavior (Ray).
- **User control**: The explicit override annotation lets users opt in or out regardless of version.

### Decision logic

```
1. Check nvidia.com/vllm-distributed-executor-backend annotation
   -> "mp": use multiprocessing
   -> "ray": use Ray
   -> absent: fall through to step 2

2. Check nvidia.com/dynamo-operator-version annotation
   -> absent: pre-upgrade DGD -> use Ray (backward compat)
   -> version >= 0.9.0 (mp threshold): use multiprocessing
   -> version < 0.9.0: use Ray
```

## Key Behavioral Differences: Ray vs mp

**Ray mode** (current):

- Leader: starts a Ray head node, then runs vLLM with `--distributed-executor-backend ray`
- Worker: only runs a Ray agent (`ray start --block`) -- does not run vLLM directly
- vLLM on the leader spawns Ray actors on workers; workers have no HTTP server, no probes

**mp mode** (new):

- Leader: runs the original vLLM command with injected flags: `--nnodes`, `--node-rank 0`, `--master-addr`, `--master-port`, `--distributed-executor-backend mp`
- Worker: runs the same vLLM command with `--headless` (no API server), `--node-rank <rank>`, and the same coordination flags
- All nodes run vLLM directly; no Ray dependency; worker probes are still removed (no HTTP server due to `--headless`)


## Migration Matrix

| Scenario | What happens |
|---|---|
| **New DGD (operator >= 0.8.0)** | Webhook stamps version -> mp by default |
| **New DGD + explicit `ray` override** | Override wins -> Ray |
| **Existing DGD (pre-upgrade, never touched)** | No version annotation -> Ray (no change) |
| **Existing DGD (edited after upgrade)** | UPDATE doesn't stamp version -> Ray (no change) |
| **Existing DGD + explicit `mp` override** | Override wins -> mp (user opts in) |

