---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Contact Router
subtitle: Group-aware admission for RL rollouts that bounds weight-version staleness
---

**Experimental.** Contact Router is a builtin router plugin for RL rollout serving. It starts from the [ThunderAgent Program Scheduler](thunderagent-program-scheduler.md) (same pause/resume and placement) and adds admission by data group, so raising client concurrency queues more groups instead of producing more stale, discarded rollouts.

## Model

- A **data group** is the rollouts of one prompt that train together. Every request of a rollout carries its group and group size in headers, plus the weight version the harness is on.
- A group is stamped with the current weight version when its **first** program starts. All its programs keep that version.
- The trainer discards a group once `current_version - group_version` exceeds its lag limit, so any group admitted too early is wasted work.
- Groups that have not started wait in the queue with **no version**, hold no KV and do not age.

## Admission

A new group starts only if all of its programs fit: `running programs of started groups + group size <= limit`. Programs of started groups always pass. The limit is the smaller of:

- **KV limit** (`kv_admission`): `pause_threshold * total KV tokens / average final context`, where the average is an exponential average (weight 0.05) of each finished program's final context. Using final rather than current context avoids admitting everything while contexts are still short.
- **Age limit** (`target_age`): starts at `initial_programs`. When a group's last program finishes, `age = current_version - group_version` and the limit moves by `age_gain * (target_age - age) * group_size`, never below one group. Updates wait until the current version exceeds `age_delay_versions`, so warm-up groups (which finish while the version barely moves) do not push the limit up.

A good `initial_programs` is `target_age * trajectories_per_trainer_step / f`, where `f` is the mean of each group's slowest rollout time over the mean rollout time (about 1.2 for 8-rollout agentic groups). Set `age_delay_versions` to the trainer's lag limit.

## Configuration

```yaml
request_classifier:
  type: contact_router
  parameters:
    kv_admission: true
    target_age: 5
    initial_programs: 520
    age_gain: 0.1
    age_delay_versions: 8
worker_selection:
  aggregated: contact_router
  instances:
    - name: contact_router
      type: contact_router
```

Start the frontend as for ThunderAgent, with `--router-policy-config contact_router.yaml` and soft session affinity.

Request headers:

| Header | Meaning |
|---|---|
| `X-Dynamo-Session-ID` | One rollout (program). |
| `x-dynamo-group` | Data group of the rollout. |
| `x-dynamo-group-size` | Rollouts in the group (default 1). |
| `x-dynamo-weight-version` | Weight version the harness is on; the router tracks the highest seen. Header name set by `version_header`. |
| `x-dynamo-env-s` | Expected environment seconds of the task (priority only). |

Parameters beyond ThunderAgent's:

| Parameter | Default | Meaning |
|---|---|---|
| `kv_admission` | `false` | Gate new groups on the KV limit. |
| `target_age` | unset | Target group age in versions; enables the age limit. |
| `initial_programs` | `256` | Starting age limit. |
| `age_gain` | `0.1` | Programs per group program per version of age error. |
| `age_delay_versions` | `0` | Do not move the age limit until the current version exceeds this. |
| `version_header` | `x-dynamo-weight-version` | Header carrying the weight version. |
| `priority` | unset | `{wv, left, env_s}` weights; pauses the lowest and resumes/admits the highest `wv * group_version + left * programs_left + env_s * env_seconds` first. |
