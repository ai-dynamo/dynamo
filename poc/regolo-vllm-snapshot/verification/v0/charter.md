# Test charter: vLLM cold start versus Dynamo Snapshot restore

Protocol version: `V0.1`

## Objective and null hypothesis

Determine whether restoring one single-GPU vLLM container on the same NVIDIA
L40S materially reduces startup latency without changing response validity or
GPU memory consumption. The null hypothesis is that restore does not improve
cold-start latency.

This is a black-box test. Observations are limited to Kubernetes objects and
events, the vLLM `/health` endpoint, its OpenAI-compatible API, and GPU metrics
reported by NVIDIA tooling. No customer traffic or customer prompt is allowed.

## Frozen design

- Exactly 10 eligible cold runs and 10 eligible restore runs.
- Paired blocks execute on the same pinned L40S node.
- Treatment order within each block and the opaque A/B key are randomized from
  integer seed `20260811`.
- Aggregation uses opaque A/B labels. The key is revealed only after the
  blinded summary has been written.
- Primary latency is Pod creation timestamp to receipt of the first valid token.
- Secondary latencies are Pod creation to Ready and Pod creation to first HTTP
  200 from `/health`.
- A run times out 30 minutes after Pod creation. A timeout is a retained failure.
- Application, checkpoint, restore, response, and compatibility failures are
  retained. They are never outliers and are never excluded.
- A run may be excluded only when raw Kubernetes events demonstrate a cluster
  incident unrelated to the candidate. Exclusion requires a non-empty evidence
  reference. When one run is excluded, the entire paired block is excluded and
  the complete block is repeated with new run identifiers.
- No statistical or IQR-based outlier removal is permitted.

## Compatibility identity

The compatibility hash is SHA-256 of canonical JSON containing exactly:

1. candidate image digest;
2. model revision;
3. GPU product;
4. NVIDIA driver version;
5. original command;
6. original arguments;
7. normalized PodSpec.

Mutable image tags are invalid inputs. A candidate image change invalidates all
prior V1 results.

## Metrics and formulas

All timestamps are UTC RFC 3339 with fractional seconds.

- `ready_s = ready_at - pod_created_at`
- `http_200_s = http_200_at - pod_created_at`
- `first_token_s = first_token_at - pod_created_at`
- `speedup = median(cold first_token_s) / median(restore first_token_s)`
- `gpu_delta = abs(restore_gpu_mib - paired_cold_gpu_mib) / paired_cold_gpu_mib`
- `restore_success_rate = successful_restore_runs / 10`
- `break_even_restores = checkpoint_duration_s / (median_cold_first_token_s - median_restore_first_token_s)`

Break-even is infinite when the denominator is zero or negative. Checkpoint
duration and size are reported separately and never folded into startup time.

## Decision

- **Go:** 10/10 restores succeed, every response is valid, every paired restore
  GPU reading is within +/-5% of cold, and median first-token restore is at
  least 3x faster.
- **Optimize:** all correctness and memory requirements pass but speedup is
  below 3x. Investigate PVC throughput and checkpoint size without changing
  thresholds.
- **No-Go:** any restore crash/failure, invalid response, incompatibility,
  secret/data retention risk, or GPU memory deviation above 5%.
- **Protocol invalid:** fewer than 10 eligible complete blocks, compatibility
  mismatch, unpaired exclusion, missing raw evidence, or unblinding before the
  blinded summary is sealed.

## Security

Use an empty KV cache at checkpoint time. The readiness probe is not a prompt.
Use only the fixed synthetic validation prompt after restore. Credentials must
be short-lived Kubernetes Secrets, the 1 TiB PVC must be encrypted by its
storage class, and no credential may be embedded in an image or checkpoint.
