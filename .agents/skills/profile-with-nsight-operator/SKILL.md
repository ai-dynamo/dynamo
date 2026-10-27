---
name: profile-with-nsight-operator
description: >-
  Profiles chosen components of a Dynamo DynamoGraphDeployment on Kubernetes with Nsight Systems (nsys) through NVIDIA
  Nsight Operator, using kubectl, Nsight's Dynamo helper nsight_operator_dynamo.py, and Nsight's nsight_operator.py CLI,
  and verifies that every expected pod produced a report. Use when someone asks to profile, trace, or run nsys on a
  Dynamo frontend, router, aggregated worker, or prefill or decode worker on a cluster that runs Nsight Operator. Use
  deploy-dynamo-recipe for bring-up and troubleshoot-dynamo for failed deployments.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - kubernetes
    - profiling
    - nsight-systems
---

# Profile with Nsight Operator

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Purpose

Profile the Dynamo components a user names with `nsys`, in one capture window, and prove that every expected pod
produced a report. Nsight Operator adds `nsys` to selected pods when they are created; this skill chooses the
selectors, configures profiling with Nsight's Dynamo helper or its own profile config, runs the window with Nsight's
CLI, and checks completeness. It does not wrap or replace `nsight_operator_dynamo.py` or `nsight_operator.py`.

Read `docs/fern/pages/kubernetes/operations/nsight-operator-profiling.mdx` first. It holds the profile-config
template, the label table, the per-component `nsys` arguments and NVTX settings, and the pitfalls. This skill does
not repeat them.

## Prerequisites

- Nsight Operator installed by a cluster admin, and a test namespace with a coordinator and a gateway (auto-provisioned
  or created by the namespace admin). For S3 reports, a `NsightCloudStorageConfig` in the namespace.
- `kubectl` rights in the namespace to create `NsightOperatorProfileConfig` and `DynamoGraphDeployment` objects, list
  pods, and port-forward.
- `nsight_operator.py`, and `nsight_operator_dynamo.py` for the helper path, from the NGC "Nsight Operator Resources"
  of the same release as the operator. The helper writes that release's profile-config field names.
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

- Pin `--context` and `-n` on every `kubectl` call, as the commands below do, and bound every wait with a timeout.
- `nsight_operator_dynamo.py` and `nsight_operator.py` have no `--context` option; they use the kubeconfig's current
  context. Before each call to either script, check that `kubectl config current-context` prints `<context>`. If it
  does not, stop and ask the user: do not switch the shared kubeconfig's context on your own. With the user's approval,
  run the scripts with `KUBECONFIG` set to a kubeconfig whose current context is `<context>`.
- Check which profile-config fields the API server accepts:
  `kubectl --context <context> -n <namespace> explain nsightoperatorprofileconfig.spec.injectionRules`.
  Nsight Operator 26.3 has `podObjectSelector`, `nsightTags`, and `processIncludePatterns` on each rule; 26.2 has
  `objectSelector` or `matchConditions` on the rule and `nsightTag` and `injectionIncludePatterns` on the tool config.
  Use the names the server accepts.
- List existing profile configs in the namespace. Do not reuse or overwrite another test's config, tool-config, or rule
  names. `dynamo-profile-config` belongs to Nsight's helper; never write it by hand.
- Run `python3 nsight_operator.py --tag <tag> status` to see whether the tag already holds a session. A tag holds one
  session at a time. `<tag>` is `default` on the helper path and the DGD name on the own-config path.

### 2. Configure Profiling

Choose one path and tell the user which one and why.

#### 2a. Nsight's Dynamo Helper

Use the helper when the operator runs with auto-provisioning, in-cluster storage is enough, and no one else profiles in
the namespace with the helper. Its `configure` command applies one profile config named `dynamo-profile-config`, with
a coordinator tool config, on the `default` tag. A second `configure` in the same namespace replaces it.

```bash
python3 nsight_operator_dynamo.py -n <namespace> configure \
    --deployment-name <dgd> \
    --components <type> [<type> ...] \
    --nsys-profile-args "-t cuda,nvtx,osrt --python-sampling=true"
```

- `-n` goes before the subcommand.
- Pass lowercase component types, such as `frontend`, `worker`, `prefill`, or `decode`. The helper compares them with
  `nvidia.com/dynamo-component-type`, so `Frontend` or `Backend` selects no pod.
- Always pass `--deployment-name`. Without it the helper selects every Dynamo pod in the namespace. It matches pods
  whose name contains the string, so check that no other DGD's name contains it.
- Always pass `--nsys-profile-args`. The default adds `--pytorch=autograd-nvtx`.
- Never pass `--gpu-metrics`; it applies a DaemonSet whose pods run privileged. Never run the helper's `install` or
  `uninstall`; they install or remove the operator for the whole cluster, which is the cluster admin's job.
- Read the profile config the helper prints, then continue at step 3 with `<tag>` = `default`.

#### 2b. Your Own Profile Config

Use your own config for S3 reports, a tag per DGD, process patterns per component, or when the helper's config would
collide with someone else's. Start from the page's template. Then:

- Check the real labels first, when pods exist:
  `kubectl --context <context> -n <namespace> get pods -l nvidia.com/dynamo-graph-deployment-name=<dgd> --show-labels`.
  Component types are lowercase; `Frontend` is a component name, not a type.
- Add one rule per requested component. Select on `nvidia.com/dynamo-graph-deployment-name: <dgd>` plus the component
  label from the page's table that the observed labels support:
  - a component type, for example `nvidia.com/dynamo-component-type: decode`. It matches every component of that type.
  - `nvidia.com/dynamo-sub-component-type: prefill` or `decode` for workers that kept type `worker` across an operator
    upgrade.
  - `nvidia.com/dynamo-component: <component name>`, with the name's case kept, to select one named component when
    several share a type, or a component without a type, such as a standalone router.
- To record only decode workers, write only the decode rule.
- Put the same literal tag, the DGD name, on every rule. Use per-test names for the config, the tool configs, and the
  rules.
- Write process patterns with `[.]`, never a backslash.
- Leave out `--pytorch=autograd-nvtx`, `--gpu-metrics`, and `--trace-fork-before-exec`.
- Validate with `kubectl --context <context> -n <namespace> apply --dry-run=server -f <file>`, then apply it with the
  same flags. Use `<tag>` = the DGD name in later steps.

### 3. Get Fresh Pods

Only pods created after the config exists are profiled.

- New DGD: apply the config first, then hand the DGD to `deploy-dynamo-recipe` (ask before applying it).
- Running DGD: ask before restarting its pods. Restarting GPU workers can stall on a full cluster.
- Wait until every selected pod is Ready. The first profiled pod in a namespace can wait while Nsight's services start.

### 4. Confirm Injection

- Derive the expected pods from the DGD and the selectors
  (`kubectl --context <context> -n <namespace> get pods -l ...`), never from the agents that happened to register.
- `python3 nsight_operator.py --tag <tag> status` must list one collection agent per expected pod. Ignore
  `otlp-converter-*` entries; they are Nsight's OpenTelemetry converter.
- If a pod is missing, follow Nsight's sidecar injection checks, check the pod's init containers and its
  `nsight-profile.nvidia.com/profiles` annotation, and set `logOutput` on the rule for the process hook's log. Hand
  pod failures to `troubleshoot-dynamo`.

### 5. Capture One Window

- Start the load first: the user's own tool, or `run-aiperf-benchmark`.
- Run:

  ```bash
  python3 nsight_operator.py --tag <tag> session-begin --title <dgd>
  python3 nsight_operator.py --tag <tag> profiler-start --delay <warm-up seconds> --duration <seconds>
  ```

- Use `nsight_operator.py` windows by default. A window started with `kubectl exec ... nsys start` in one injected pod
  also works, but its report stays only inside that pod's `main` container: it is not uploaded, `ls` does not list it,
  and it is lost if the container restarts. Use it only when the user asks for one exact pod, and copy the report out
  as the docs page shows.

### 6. Collect And Verify

- Reports appear only after the delay, the window, report generation, and the upload. Set a polling deadline of the
  delay plus the window plus 10 minutes, unless the user gives another one, and tell the user the deadline.
- Poll `python3 nsight_operator.py --tag <tag> ls` until every expected pod has a report or the deadline passes.
- Download while the session is still active, because `ls` and `download` use the tag's active session unless
  `--session <UUID>` is passed. Then end the session, also when the deadline passed:

  ```bash
  python3 nsight_operator.py --tag <tag> download -o <dir>
  python3 nsight_operator.py --tag <tag> session-end
  ```

  Instead of `download`, you can read the reports from the bucket with the user's credentials.
- Check completeness: every expected pod has a report, and every GPU worker report contains kernel rows
  (`nsys stats --report cuda_gpu_kern_sum <file>`). Report a missing pod or a GPU report without kernels as an
  incomplete capture; do not present partial results as complete.
- If the deadline passed, check `status` and injection for each missing pod, and report the capture as incomplete
  with the reports you have.
- Only then remove the profile config, and the DGD if the user created it only for this test. On the helper path,
  check the current context, then run `python3 nsight_operator_dynamo.py -n <namespace> unconfigure`. On the
  own-config path, run
  `kubectl --context <context> -n <namespace> delete nsightoperatorprofileconfig <config name>`.

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

- `docs/fern/pages/kubernetes/operations/nsight-operator-profiling.mdx`: the helper, template, labels, settings, and
  pitfalls.
- [Nsight Operator documentation](https://docs.nvidia.com/nsight-operator/), including the multitenant quickstart and
  the CLI reference.
