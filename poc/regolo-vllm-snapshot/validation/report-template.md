# Dynamo Snapshot PoC validation report

Date: 2026-08-13 UTC
Status: **EXECUTED — GO**

## Answer

For `openai/gpt-oss-20b` revision
`6cee5e81ee83917806bbde320786a8fb61efebee`, restoring the same-node,
single-L40S vLLM process from a Dynamo checkpoint reduced median Pod creation
to first valid token from **129.729 s** to **36.267 s**. The measured median
speedup is **3.577x**, above the frozen 3x Go threshold.

All 10 restores completed, all 20 synthetic responses were valid, and every
run reported 38,464 MiB of GPU memory. The maximum paired GPU-memory delta was
0%. The V0.1 decision is **Go** for this pinned PoC configuration.

## Frozen protocol and identity

- Verification version: V0.1
- Dynamo: v1.3.0, commit `8ce9e22f11576402102ea9d8b8e46233f5430a0d`
- Seed: `20260811`
- Design: 10 paired blinded blocks, one cold and one restore run per block
- V0 manifest SHA-256: `d01079551a7a33118ccf79d7ae5f4e7b3bbacc39e3724d106db0e43051855359`
- I1 manifest SHA-256: `8769f34e1778416934a36987068b8cd15ae7be2d70ea6ba5f0d9bcc230fb3259`
- Run-plan SHA-256: `24e89334bbcb27428954d4523ef1280937ea0d3e56c6d100d5d9846bdb40cf04`
- Raw-results SHA-256: `b55e89421e759c06a3be0a57a6030b7e12445662d01988efc566e2ed848b91f6`
- Compatibility SHA-256: `a42c07d50e863d43838bcf0ec3c07c544324579f3df80cc08047191838e1e805`
- Kubernetes checkpoint locator: `h-a42c07d50e863d43838bcf0ec3c07c544324579f3df80cc08047191838e1e`
- Source image: `docker.io/vllm/vllm-openai@sha256:c2f3b1b964e47809b722b5e75b61b1e7b39a50f70388cf2bf2418f16a9f31da2`
- Candidate image: `docker.io/library/regolo-vllm-snapshot@sha256:84e626a76456827946ada12120fd6842ae7eefc4b2a4005663bab137385f030a`

The blinded summary was written before the A/B key was opened. Its SHA-256 is
`0e9951e1d24bd0e1cd7b12aaef3bed86b87aa925a9f14894fb5aba88dee29ee6`,
and its checksum manifest verifies successfully.

## Environment

- Node: `ec213103`, Ubuntu 22.04, kernel `5.15.0-139-generic`
- Kubernetes: K3s v1.34.10; CRI: `containerd://2.2.5-k3s2`
- GPU: one NVIDIA L40S, UUID `GPU-fa5675b0-0607-2841-4cbd-f7167ca4deb2`
- Driver: NVIDIA open driver 580.178.04
- Model: `openai/gpt-oss-20b`, revision pinned above
- Command: `vllm serve openai/gpt-oss-20b --revision 6cee5e81ee83917806bbde320786a8fb61efebee --host 0.0.0.0 --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.85`
- Readiness: HTTP `GET /health`; validation uses a fixed synthetic prompt
- Model cache: the same pre-populated host path is mounted for cold and restore
- Storage: 1 TiB RWO local LUKS2 PVC, AES-XTS-plain64, key retained only in tmpfs
- Snapshot agent: immutable v1.3.0 image, node-pinned, namespace-scoped RBAC

## Results

| Metric | Cold | Restore | Requirement |
| --- | ---: | ---: | --- |
| Eligible runs | 10 | 10 | 10 + 10 |
| Median Pod to Ready | 127.000 s | 27.500 s | reported |
| Median Pod to HTTP 200 | 128.279 s | 28.479 s | reported |
| Median Pod to first token | 129.729 s | 36.267 s | restore >=3x faster for Go |
| First-token range | 124.217–154.493 s | 31.205–143.663 s | no outlier removal |
| Valid responses | 10/10 | 10/10 | 20/20 total |
| GPU memory | 38,464 MiB | 38,464 MiB | each pair within +/-5% |
| Restore success | n/a | 10/10 | 10/10 |

- Median speedup: **3.577105525960319x**
- Checkpoint duration: **428.342 s** end-to-end
- Checkpoint size: **43,886,233,878 bytes** (43.89 GB decimal)
- Break-even: **4.583 restores**, or approximately five restores
- Exclusions: **0**
- Retained failures: **0**

The restore distribution has one retained 143.663 s tail observation. With 10
samples, nearest-rank p95 equals the maximum, so restore p95 is also 143.663 s.
This does not violate V0.1 because the median passes, correctness and memory
requirements pass, and the protocol forbids statistical outlier removal. It is
a production-readiness caveat and should be investigated with a larger sample.

An independent implementation recomputed all decision metrics directly from
the raw JSONL and A/B key without importing the PoC analysis library. Every
value matched `final-report.json` exactly.

## Failures, exclusions, and deviations

There were no V1 failures or exclusions and no missing manifests or raw event
dumps. All 20 run IDs were unique and matched the registered plan; each of the
10 blocks contained exactly one A and one B run.

Recorded environment and implementation deviations:

1. The available Regolo configuration used a mutable vLLM image reference; the
   source was resolved and pinned to the immutable digest above.
2. The model revision was derived from the available Regolo configuration and
   pinned before the experiment.
3. The NVIDIA driver was upgraded from 570.211.01 to 580.178.04; V0 was rerun.
4. K3s containerd socket/root paths required a runtime-only Helm post-renderer.
5. The full compatibility hash is retained as identity; its deterministic
   63-character mapping is used only as the Kubernetes checkpoint locator.
6. The pinned agent's `cuda-checkpoint` lacks `--launch-job`; the documented
   single-GPU `--disable-cuda-checkpoint-job-file` mode was used. Multi-GPU is
   outside scope.
7. The PoC uses a local file-backed LUKS2 volume rather than production
   StorageClass/KMS integration.
8. Checkpoint duration includes about 94 s of scheduler wait, making break-even
   conservative.
9. The functional smoke restore included GPU scheduling delay and was not used
   in V1 statistics.
10. Earlier numerically positive V1 data were invalidated after discovering an
    auto-mounted service-account token in the prior checkpoint source. The
    entire V0/I1/V1 sequence was repeated with token mounting disabled.

## Security, resilience, and retention

- `automountServiceAccountToken: false` was enforced and audited on every V1
  manifest; no projected Kubernetes API token volume was present.
- The checkpoint source received only `GET /health`; no inference prompt or
  customer traffic populated its KV cache before checkpointing.
- The checkpoint manifest and admitted source Pod passed the pre-checkpoint
  audit with zero findings. After V1, the corrected fail-closed audit scanned
  the retained candidate's Docker configuration and complete history with zero
  findings. Containerd's manifest for the immutable candidate digest names the
  same config digest as the audited Docker image ID; this binding and the audit
  result are retained in `report/candidate-secret-audit.json`. The final
  manifest scan also passed for all 20 V1 Pods.
- The compatibility hash and deterministic checkpoint locator matched the
  campaign and I1 checkpoint metadata before V1 ran.
- The encrypted PVC remains Bound and the ephemeral LUKS key remains outside
  the repository in `/run` with mode `0600`.
- The namespace, PVC, 43.9 GB checkpoint, candidate image, and raw evidence
  remain retained. No destructive cleanup was performed because a retention
  period has not yet been approved.

This is a same-node, single-GPU PoC result. It does not establish multi-GPU,
cross-node, autoscaling, customer-traffic, long-soak, or production KMS
behavior.

## Evidence

The sealed runtime evidence is retained outside Git at:

- `/root/snapshot/dynamo-snapshot-runtime/artifacts/v0-driver580-v3`
- `/root/snapshot/dynamo-snapshot-runtime/artifacts/i1-driver580-v3`
- `/root/snapshot/dynamo-snapshot-runtime/artifacts/v1-driver580-v3`

V1 contains all 20 sanitized JSONL records, 20 rendered Pod manifests, 20 raw
Kubernetes event dumps, the registered plan and seed, the checksummed blinded
summary, and the final report. The LUKS key and 43.9 GB checkpoint binary are
not stored in Git. A reviewable subset containing the plan, JSONL measurements,
blinded summary/checksum, final JSON, and its own SHA-256 manifest is committed
under `validation/evidence`.
