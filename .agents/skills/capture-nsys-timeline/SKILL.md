---
name: capture-nsys-timeline
description: >-
  Captures Nsight Systems (nsys) timelines from chosen components of a Dynamo DynamoGraphDeployment on Kubernetes
  with NVIDIA Nsight Operator, using kubectl and Nsight's nsight_operator.py CLI, and verifies that every expected pod
  produced a report. Use when someone asks to profile, trace, or record a timeline of a Dynamo frontend, router,
  aggregated worker, or prefill or decode worker on a cluster that runs Nsight Operator. Use deploy-dynamo-recipe for
  bring-up and troubleshoot-dynamo for failed deployments.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - kubernetes
    - profiling
    - nsight-systems
---

# Capture Nsys Timeline

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Purpose

Record `nsys` timelines of the Dynamo components a user names, in one capture window, and prove that every expected
pod produced a report. Nsight Operator adds `nsys` to selected pods when they are created; this skill chooses the
selectors, writes the per-test profile config, runs the window with Nsight's CLI, and checks completeness. It does
not wrap or replace `nsight_operator.py`.

Read `docs/fern/pages/kubernetes/operations/nsight-systems-timelines.mdx` first. It holds the profile-config
template, the label table, the per-component `nsys` arguments and NVTX settings, and the pitfalls. This skill does
not repeat them.

## Prerequisites

- Nsight Operator installed by a cluster admin, and a test namespace with a coordinator and a gateway (auto-provisioned
  or created by the namespace admin). For S3 reports, a `NsightCloudStorageConfig` in the namespace.
- `kubectl` rights in the namespace to create `NsightOperatorProfileConfig` and `DynamoGraphDeployment` objects, list
  pods, and port-forward.
- `nsight_operator.py` from the NGC "Nsight Operator Resources" of the same release as the operator.
- Never print, copy, or log secrets. `autoconfigure` writes the gateway's storage key to `~/.nsight-cloud.conf`;
  where the user allows it, use `configure` and read reports with the user's own credentials instead.

## Required Inputs

- Kube context and namespace.
- DGD name. Use a test DGD; ask before applying, editing, or restarting any DGD.
- The components to record, for example `frontend` or `decode`. Ask if they are not given.
- Warm-up delay and window length. Keep the window at 300 seconds or less.
- The storage config name for S3, or in-cluster storage.

## Instructions

### 1. Preflight

- Pin `--context` and `-n` on every `kubectl` call, and bound every wait with a timeout.
- Check which profile-config fields the API server accepts:
  `kubectl explain nsightoperatorprofileconfig.spec.injectionRules`. Nsight Operator 26.3 has `podObjectSelector`,
  `nsightTags`, and `processIncludePatterns` on each rule; 26.2 has `objectSelector` or `matchConditions` on the rule
  and `nsightTag` and `injectionIncludePatterns` on the tool config. Use the names the server accepts.
- List existing profile configs in the namespace. Do not reuse or overwrite another test's config, tool-config, or rule
  names, and never write a config named `dynamo-profile-config`.
- Run `python3 nsight_operator.py --tag <dgd> status` to see whether the tag already holds a session. A tag holds one
  session at a time.

### 2. Write The Profile Config

Start from the page's template. Then:

- Add one rule per requested component. Select on `nvidia.com/dynamo-graph-deployment-name: <dgd>` plus the component
  label from the page's table, for example `nvidia.com/dynamo-component-type: decode`. To record only decode workers,
  write only the decode rule.
- Check the real labels first with `kubectl get pods -l nvidia.com/dynamo-graph-deployment-name=<dgd> --show-labels`
  when pods exist. Component types are lowercase; `Frontend` is a component name, not a type.
- Put the same literal tag, the DGD name, on every rule. Use per-test names for the config, the tool configs, and the
  rules.
- Write process patterns with `[.]`, never a backslash.
- Leave out `--pytorch=autograd-nvtx`, `--gpu-metrics`, and `--trace-fork-before-exec`.
- Validate with `kubectl apply --dry-run=server -f <file>`, then apply it.

### 3. Get Fresh Pods

Only pods created after the config exists are profiled.

- New DGD: apply the config first, then hand the DGD to `deploy-dynamo-recipe` (ask before applying it).
- Running DGD: ask before restarting its pods. Restarting GPU workers can stall on a full cluster.
- Wait until every selected pod is Ready. The first profiled pod in a namespace can wait while Nsight's services start.

### 4. Confirm Injection

- Derive the expected pods from the DGD and the selectors (`kubectl get pods -l ...`), never from the agents that
  happened to register.
- `python3 nsight_operator.py --tag <dgd> status` must list one collection agent per expected pod. Ignore
  `otlp-converter-*` entries; they are Nsight's OpenTelemetry converter.
- If a pod is missing, follow Nsight's sidecar injection checks, check the pod's init containers and its
  `nsight-profile.nvidia.com/profiles` annotation, and set `logOutput` on the rule for the process hook's log. Hand
  pod failures to `troubleshoot-dynamo`.

### 5. Capture One Window

- Start the load first: the user's own tool, or `run-aiperf-benchmark`.
- Run:

  ```bash
  python3 nsight_operator.py --tag <dgd> session-begin --title <dgd>
  python3 nsight_operator.py --tag <dgd> profiler-start --delay <warm-up seconds> --duration <seconds>
  ```

- Never open windows with `kubectl exec ... nsys start` in injected pods; those reports are not uploaded.

### 6. Collect And Verify

- Poll `python3 nsight_operator.py --tag <dgd> ls` in a bounded loop until every expected pod has a report. Reports
  appear only after the delay, the window, report generation, and the upload, so allow several minutes.
- Run `session-end`, then `download -o <dir>`, or read the reports from the bucket with the user's credentials.
- Check completeness: every expected pod has a report, and every GPU worker report contains kernel rows
  (`nsys stats --report cuda_gpu_kern_sum <file>`). Report a missing pod or a GPU report without kernels as an
  incomplete capture; do not present partial results as complete.
- Only then delete the test's profile config, and the DGD if the user created it only for this test.

### 7. Hand Off Analysis

Use `nsys stats`, `nsys recipe`, or the agent skill that ships with recent Nsight Systems releases for analysis. Run
heavy analysis where the reports are, not on shared login hosts.

## Output Contract

Report:

- the namespace, the DGD, and the components and pods captured;
- the tag, the session, and the window start time and length;
- the report location for every expected pod, and the completeness verdict;
- the `nsys` arguments and NVTX variables used;
- anything skipped or not verified.

## Limitations

- A tag holds one session at a time, so one test per tag.
- Selection is per component; all replicas of a component share the tag.
- Not yet tested: the 26.3 profile config on a live operator, upload to external S3, a coordinator window on a GPU
  worker, workers with more than one GPU, SGLang and TensorRT-LLM workers, and Dynamo sidecar mode.

## References

- `docs/fern/pages/kubernetes/operations/nsight-systems-timelines.mdx`: template, labels, settings, and pitfalls.
- `docs/fern/pages/reference/kubernetes-api/dynamo-component-deployment.mdx`: the "Pod Labels" section.
- [Nsight Operator documentation](https://docs.nvidia.com/nsight-operator/), including the multitenant quickstart and
  the CLI reference.
