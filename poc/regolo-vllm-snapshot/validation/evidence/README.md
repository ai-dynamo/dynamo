# Sanitized V1 evidence

This directory contains the small, reviewable subset of the 2026-08-13 V1
evidence needed to cross-check the published aggregates and decision:

- the preregistered opaque run plan;
- all 20 JSONL measurement records;
- the summary written and checksummed before unblinding;
- the final unblinded decision report; and
- the post-V1 fail-closed candidate image audit and digest binding.

`SHA256SUMS` seals these files. Verify it from this directory with:

```bash
sha256sum -c SHA256SUMS
```

The unblinding-key file is intentionally not committed. The complete sealed
runtime evidence, including 20 Pod manifests and 20 raw Kubernetes event dumps,
is retained under `/root/snapshot/dynamo-snapshot-runtime/artifacts/v1-driver580-v3`.
No LUKS key, Kubernetes credential, container image, or 43.9 GB checkpoint data
is stored here.
