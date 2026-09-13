<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# WorkerSet Admission Contract

A WorkerSet identifies a namespace, component, endpoint, model, and worker type.
Deployment generations belong to separate WorkerSets: versioned namespaces may
serve the same model concurrently through independent routing pipelines. A
rolling update that creates a new WorkerSet does not replace the configuration
of an existing set.

## Compatibility Identity

Use `ModelDeploymentCard::mdcsum()` after existing discovery-boundary
normalization and tokenizer overrides, before pipeline construction. Capture
that checksum as `mdc_checksum` and use it throughout admission, construction,
and status tracking. Keep the checksum algorithm and metadata-cache identity
unchanged; do not introduce another normalized materialization hash or move
its exclusions into `mdcsum()`. The independent LoRA projection fingerprint is
separate from WorkerSet compatibility.

Checksum equality is intentionally stricter than equivalent serving behavior.
Different advertised router settings, absent versus explicit defaults, or
different `source_path` values may reject a newcomer even if its effective
serving configuration would match. Preserve supported wire decoding and the
N-2 compatibility shims in the parent instructions. Legacy cards join when
existing boundary normalization yields matching MDC checksums; equivalent
representations receive no additional same-set compatibility exemption.

## Reservation and Succession

The first valid card observed by a frontend reserves the WorkerSet's
configuration immediately, including while construction is queued, in progress,
or retrying after failure. Matching workers join that incumbent cohort. A larger
competing cohort cannot displace it. Rejected workers remain tracked for removal
and reconsideration, but they and their adapters must not change incumbent
admissions, serving publications, routing settings, applicable adapter
projections, construction, or retry deadlines.

Retain cohorts in first-observation order while they have members. Duplicate
events and reconciliation snapshots do not change that order. Only after every
incumbent worker disappears may the oldest remaining cohort become eligible
for a fresh pipeline. A cohort that empties and later returns joins the end.

On succession, clear the old admissions, cancel construction, withdraw the
committed group, and retire its admission channel. The successor receives a
fresh channel and globally unique build generation. Fence late results with
both the selected checksum and build generation; retained old clients cannot
acquire successor worker IDs, even when the endpoint or checksum is reused.
Log each newly rejected cohort at `ERROR` with the model, WorkerSet, incumbent
and rejected checksums, and a representative instance. Suppress unchanged
repeats and report remaining disagreements when the incumbent changes.

## Routing and Readiness

Every discovery-managed frontend routing hop, including prefill and encoder,
must use the committed target's admission receiver and selected card. Intersect
endpoint discovery with admitted worker IDs. Resolve advertised routing mode,
KV block size, and other configuration from that selected card; do not relist
arbitrary endpoint cards. Identify a binding by group and successful build
generation, so succession rebuilds it even when the endpoint is unchanged.
Compatible membership changes update admissions without rebuilding bindings.
Public explicit-endpoint constructors retain a separate legacy adapter path;
discovery-managed topology must never fall back to unrestricted discovery.

Selection is frontend-local. Frontends that observe conflicting registrations
in different orders may choose different incumbents; there is no shared
election or cross-frontend agreement guarantee. This is an intentional
availability tradeoff: serving each frontend's admitted incumbent is preferable
to serving none. Different local winners are expected behavior, not a defect
to correct with fail-closed withdrawal or a shared election.

Frontend readiness reflects its committed membership. The KV DC Relay evaluates
discovery independently and can remain conservative while a frontend serves its
incumbent. The shared readiness evaluator guarantees equal answers for equivalent
input units, not for these different views of conflicting registrations.
