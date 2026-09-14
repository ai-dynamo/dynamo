<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Compiled Results

`workloads/compile_run.py` writes one directory per source run. A compiled
directory identifies its raw run, transformation command, schemas, units,
missing-data handling, and exclusions. Raw artifacts remain unchanged.

`workloads/compile_planner_gym_batch_impact.py` writes a campaign directory
with `summary.json` and `README.md`. `summary.json` uses schema version `1.0`,
`analysis_kind=planner-gym-batch-impact-campaign`, and one of these statuses:

- `valid`: every campaign and data-quality invariant passed;
- `invalid-inputs-present`: output was written for inspection, but one or more
  runs or cross-run invariants failed.

The top-level schema contains:

- `campaign`: input/valid/invalid counts, pairing method, minimum overlap,
  canonical online workload, Batch workload, and serving signatures, per-arm
  Async and control-plane signatures, the native Planner signature, and
  canonical image identities;
- `runs`: one normalized record per supplied attempt, including arm, ordinal,
  validity reasons, online metrics, optional Batch metrics, serving/Async/
  Planner/control-plane signatures, runtime evidence, and source provenance;
- `arms`: per-arm counts plus mean, standard deviation, coefficient of
  variation, minimum, and maximum for reportable metrics;
- `comparisons`: chronological ordinal-paired treatment effects and direction
  checks;
- `data_quality`: campaign-level issue codes and details;
- `stopping_assessment`: the configured coefficient-of-variation threshold,
  instability reasons, achieved counts, target repetitions, and whether more
  balanced repetitions are required;
- `provenance`: compiler path/checksum, ordered input directories, and any
  explicit image-ID attestations used to restore sanitized observations.

## Runtime identity contract

For every run, the compiler reads the start and end Kubernetes image
inventories. It requires one clean, Ready frontend, worker, and Async image at
both boundaries, plus one Planner image for a native-Planner arm. The pod UID,
name, image reference, and observed imageID must remain stable across the
online interval. When that observed ID was sanitized, the same explicit
attestation supplies the resolved identity at both boundaries.

Each normalized identity contains:

- `image_ref`: the configured container image reference;
- `image_id`: the resolved identity included in signatures;
- `observed_image_id`: the exact value retained in the raw artifact;
- `image_id_attested`: whether `image_id` was restored from an explicit
  attestation because artifact sanitization replaced the observed value.

The compiler signs the frontend and worker together as the serving identity
and requires that signature to match across all arms. It signs Async in every
run, Planner in native-Planner runs, and their combined control-plane identity.
Async and control-plane signatures must be internally consistent within each
arm; the Planner signature must be consistent across the native-Planner arm.
`canonical_deployment_signature_sha256` remains a compatibility alias for the
shared serving signature.

A sanitized selected imageID is not treated as an identity. Supply its
independently preserved resolved value with a repeatable
`--image-id-attestation IMAGE_REF=RESOLVED_IMAGE_ID` argument. Compilation
fails closed when a selected sanitized image has no attestation, or when an
attestation is malformed, duplicated, conflicting, or unused because its
image reference did not match a sanitized selected image. The output records
every accepted attestation and distinguishes attested identities from values
observed directly by Kubernetes. One image-reference-to-ID attestation can bind
historical sanitized records to a known digest, but it cannot independently
detect per-run digest drift hidden by those records. Current captures preserve
standard Docker, containerd, and CRI-O SHA-256 runtime IDs directly so their
cross-run consistency is observed rather than attested.

The compiler accepts only compatible run schema, source identity, runtime
identity, workload signature, and arm contracts. The Batch workload signature
compares the source checksum, selected range, model, temperature, output-token
limit, and fixed request shape between stock and native-Planner runs. It
deliberately excludes the normalized output checksum and submitted IDs because
the harness assigns different `custom_id` values to those arms. Supply all
attempts in chronological order. Do not edit `summary.json` or remove an
invalid run to make the status pass; correct the source problem, run a new
balanced repetition when the stopping assessment requires one, and compile the
full attempt set into a new output directory.
