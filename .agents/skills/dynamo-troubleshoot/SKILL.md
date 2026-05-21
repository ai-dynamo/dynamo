---
name: dynamo-troubleshoot
description: >-
  Diagnose and remediate runtime issues in a deployed Dynamo workload.
  Triage symptoms, inspect operator logs and Custom Resource status,
  match against known-issue signatures, and apply fixes or escalate
  with a complete evidence bundle. Use when a Dynamo worker is
  crashlooping, /v1/models is empty, the Planner is stuck, KV transfer
  falls back to TCP, the conversion webhook is failing, or any other
  day-2 problem on a running DGD or DGDR.
version: 1.2.0
author: NVIDIA
tags:
  - dynamo
  - troubleshooting
  - day-2
  - operations
  - kubernetes
tools:
  - Shell
  - Read
  - Write
---
<!--
Progressive Disclosure:
- Level 1 (YAML front matter): Trigger matching on "Dynamo crashlooping", "stuck Planner", "empty /v1/models", "DGD Failed", "operator error"
- Level 2 (this file): 4-phase day-2 workflow (Triage → Inspect → Diagnose → Remediate)
- Level 3: references/ for symptom-to-cause signature library, debug commands, log patterns
           scripts/  for the day-2 evidence collector

Scope: diagnosing and fixing a Dynamo workload that is failing or misbehaving at runtime
Out of scope: initial deployment (see dynamo-deploy), planning (see dynamo-plan),
              quantization (see dynamo-optimize), benchmarking (see dynamo-benchmark),
              cluster-level install (see dynamo-install, deferred).

Phase shape:
  This skill is day-2. The four phases substitute for the standard
  Pre-Check / Install / Validate / Deploy shape, but keep the same
  contract: four phases, strict ordering, never skip.
-->

# Dynamo Troubleshoot (Day-2 Operations)

A failing Dynamo workload: worker `CrashLoopBackOff`, `/v1/models`
empty after a Ready DGD, Planner stuck at `num_workers=1`, conversion
webhook timeouts, KV transfer falling back to TCP. Walk the four
phases; do not skip.

## Workflow

Strict 4-phase workflow. Always follow in order. Never skip a phase.

```
Phase 1: Triage     → identify the affected surface and the user-visible symptom
Phase 2: Inspect    → collect the artifacts (status, logs, events, describe)
Phase 3: Diagnose   → match against the signature library; rule in / rule out
Phase 4: Remediate  → apply the fix, or escalate with a complete evidence bundle
```

---

## Command Safety

Most troubleshooting is inspection. A few common remediations are
DESTRUCTIVE (deleting CRDs, restarting workers); the user must approve
each.

### DESTRUCTIVE — always require explicit confirmation

| Command Pattern | Risk |
|---|---|
| `kubectl delete pod <pod> -n <ns>` (single pod) | Pod restart. Causes a brief inference traffic blip if the deployment has only one replica. |
| `kubectl delete dgd <name>` | Removes the entire deployment. **Per, the DGDR does not own the DGD**; the DGDR remains. Inference stops until a new DGD is created. |
| `kubectl rollout restart deploy/<operator>` | Rolls the operator. Active reconciliations pause briefly; in-flight DGDR phase transitions may be delayed. |
| `kubectl delete crd dynamographdeployments.nvidia.com` | Removes the CRD and ALL DGDs cluster-wide. Never use as a troubleshooting step; this is operator-removal territory. |

### MUTATING — require confirmation with explanation

| Command Pattern | Impact |
|---|---|
| `kubectl patch dgd <name> -p '...'` | Changes the live spec. Operator may roll workers. |
| `kubectl apply -f <fixed-manifest>` | Edits-then-applies; depending on the field, can trigger a roll. |
| `kubectl scale --replicas=N` (on a deployment Dynamo doesn't manage) | Only valid for non-Dynamo deployments; do not use on DGD-owned workloads (use `kubectl patch dgd` instead). |

### SAFE — no confirmation needed

```
kubectl get dgd/dgdr/dgdsa/dcd/dynamomodel
kubectl describe dgd <name>
kubectl get events --sort-by='.lastTimestamp'
kubectl logs <pod>
kubectl logs <pod> --previous
kubectl get pods -n <ns> -o wide
kubectl top pods -n <ns>
kubectl get apiservice | grep nvidia.com
helm status dynamo-platform
helm get values dynamo-platform
```

---

## Phase 1: Triage

**Goal.** Identify the affected surface (which Dynamo Custom Resource,
which backend, which pod) and the user-visible symptom in one sentence.

**Inputs needed**:

| Input | How to get |
|---|---|
| Namespace | From the user's environment |
| Resource name | `kubectl get dgd -A` or `kubectl get dgdr -A` to find the affected one |
| Symptom | User report: "inference returns 500", "/v1/models empty", "worker crashloops" |
| Recent changes | Was a release bumped? Was the platform upgraded? Was a manifest edited? |

**Decision:** Multiple Dynamo resources are in flight. Which one is
failing? Provide the namespace and DGD/DGDR name before Phase 2 runs
commands against it.

Wait for explicit identification.

**Verification gate.** Single resource named (e.g.
`DGD/my-model in dynamo-system`); user-visible symptom captured.

---

## Phase 2: Inspect

**Goal.** Collect the canonical day-2 artifacts. Do not interpret yet —
just gather.

Run the bundled collector:

```bash
bash scripts/collect-evidence.sh -k dgd -r <name> -n <ns> -o /tmp/<name>-evidence
```

The script captures (per):

| Artifact | File |
|---|---|
| Resource YAML | `<name>.yaml` |
| Status conditions | `conditions.txt` |
| Pod listing | `pods.txt` |
| Per-pod describe | `describe-<pod>.txt` |
| Per-pod logs (current + previous) | `logs-<pod>.txt`, `logs-<pod>-previous.txt` |
| Recent events | `events.txt` |
| Operator logs (last 1000 lines) | `operator.log` |
| Helm status | `helm-status.txt` |

The script is a passive collector — no `kubectl delete`, no `kubectl
patch`, no `kubectl apply`. Safe to run repeatedly.

**Decision (optional):** If you also want the platform-wide context
(other DGDs in the cluster, CRD versions, conversion webhook state),
run:

```bash
bash ../dynamo-deploy/scripts/verify-platform.sh -n <ns>
```

That cross-skill call is the `dynamo-deploy` post-install verifier
applied as a diagnostic. Same script, different context.

**Verification gate.** Evidence bundle in `/tmp/<name>-evidence/`
contains the full set of artifacts.

---

## Phase 3: Diagnose

**Goal.** Match the collected artifacts against the signature library
in [references/symptom-signatures.md](references/symptom-signatures.md).

Common starting points:

| Symptom | First signature to check |
|---|---|
| `/v1/models` empty for >2 min after DGD Ready | D3 (registration window) and D2 (HF token on Frontend) |
| Worker `CrashLoopBackOff` immediately | D5 (disagg KV transfer config), D4 (chat template), pull errors |
| Worker stays Pending, no events about scheduling | Grove gang scheduling, KAI scheduler readiness, GPU resource exhaustion |
| Operator logs show "conversion webhook timeout" | Webhook certs, the network path operator → webhook service |
| Planner stuck at `num_workers=1` | Latency-mode Grove pod-gang roll issue (DYN-2879) |
| KV transfer "falls back to TCP" message | RDMA / EFA misconfiguration; UCX path |
| DGDR profiling Job `OOMKilled` | Image too large for default ephemeral-storage; bump request |
| `helm install dynamo-platform` 404 on OCI | D1 (Helm v4 OCI registry) |

For each candidate signature:

1. Read the signature in [references/symptom-signatures.md](references/symptom-signatures.md).
2. Match the **Symptom** and **Root cause** lines against the evidence bundle.
3. If matched, jump to Phase 4 with the signature's **Fix** field.
4. If not matched, work down the list. If no signature matches, the
   issue is novel — proceed to Phase 4 escalation.

**Decision:** A signature is matched (`<DN>` or `<custom>`). The fix
will `<verb>` the cluster state in this way: `<consequence>`. Proceed?

Wait for explicit yes before any DESTRUCTIVE/MUTATING action.

**Verification gate.** One of:
- A signature is identified with confidence (Symptom + Root cause both match).
- No signature matches; preparing escalation.

---

## Phase 4: Remediate or Escalate

### 4.A Remediate

Apply the fix from the matched signature. After the fix:

```bash
# Re-run the collector to capture the post-fix state.
bash scripts/collect-evidence.sh -k dgd -r <name> -n <ns> -o /tmp/<name>-post-fix

# Diff the conditions before/after.
diff /tmp/<name>-evidence/conditions.txt /tmp/<name>-post-fix/conditions.txt

# Confirm the user-visible symptom is gone.
curl http://localhost:8000/v1/models | python3 -m json.tool   # or whatever the original probe was
```

**Verification gate.** The user-visible symptom from Phase 1 no longer
reproduces.

### 4.B Escalate

When no signature matches, file an NVBug with the evidence bundle.

Required NVBug fields:

| Field | From |
|---|---|
| Synopsis | Phase 1 user-visible symptom, one line |
| Severity | P0 for inference-down; P1 for degraded; P2 for non-blocking |
| Found in | Dynamo release line (read `helm get values dynamo-platform` for the chart appVersion) |
| Description | Phase 1 inputs + Phase 2 evidence bundle path |
| Attachments | tarball of `/tmp/<name>-evidence/` |

```bash
# Make the evidence bundle attachable.
tar -czf /tmp/<name>-evidence.tar.gz -C /tmp <name>-evidence/
```

Also: open a parallel Linear DYN-* issue in the corresponding team (
DIS for prefill/decode/router, DEP for operator/platform, LLM for
backends, OPS for containers/CI). Link the NVBug.

**Verification gate.** NVBug filed with the evidence bundle; Linear
DYN-* issue opened and linked.

---

## Refusal Conditions

- The namespace is production-protected and the proposed fix is a
  DESTRUCTIVE command — refuse and surface as a change-management
  request instead.
- The diagnosis is uncertain (multiple signatures partially match) and
  the proposed fix has side effects on adjacent workloads — refuse and
  collect more evidence first.
- The issue is in a sibling skill's scope (e.g. an install problem) —
  refuse and hand off to the sibling skill.

---

## Cross-Skill Referencing

| Need | Sibling |
|---|---|
| Re-deploy after a destructive remediation | [dynamo-deploy](../dynamo-deploy/SKILL.md) |
| Validate planning assumptions after a Planner issue | [dynamo-plan](../dynamo-plan/SKILL.md) |
| Measure performance regression caused by an issue | [dynamo-benchmark](../dynamo-benchmark/SKILL.md) |
| Platform-level health is suspect | Use `dynamo-deploy`'s `scripts/verify-platform.sh` |

---

## Prerequisites

Phase 1 (the workflow's first phase) verifies the full readiness state. The short list:

- A failing or misbehaving Dynamo resource (DGD, DGDR, DCD, DGDSA) identified by name and namespace.
- `kubectl` read access to the namespace.
- Optional: `helm` access for platform-level diagnostics.

---

## Limitations

What this skill does NOT cover:

- Does NOT fix issues that require operator code changes; those escalate to NVBug + Linear DYN-*.
- Does NOT cover initial install issues. Install-time problems route through `dynamo-deploy` Phase 1 / the deferred `dynamo-install` skill.
- Does NOT modify production-protected namespaces without explicit confirmation (per the skill's Refusal Conditions).
- Does NOT replace the on-call rotation. Critical incidents follow the team's incident-response process; this skill collects evidence to feed that process.

See `## Cross-Skill Referencing` for the appropriate sibling skill in each case.

---

## Troubleshooting

When a step in this workflow fails:

- **Per-skill known patterns** are catalogued in [references/known-issues.md](references/known-issues.md). Walk that list first.
- **Cross-skill day-2 patterns** (worker crashloops, conversion-webhook timeouts, KV-transfer fallback, gateway 500s) are owned by [dynamo-troubleshoot](../dynamo-troubleshoot/SKILL.md). The 4-phase day-2 workflow there (Triage → Inspect → Diagnose → Remediate) applies.
- **Per-release bugs** live in the active QA tracker (per the skill's `version` field). Pull the tracker view for the matching release line.

---

## Available Scripts

| Script | Purpose | Arguments |
|---|---|---|
| `scripts/collect-evidence.sh` | Passive day-2 evidence collector (status, logs, events, operator log) | `-k <kind> -r <name> [-n <ns>] -o <out-dir>` |

Each script implements the `pass/fail/warn` () or `check()` () pattern and exits non-zero on any failure. Output is structured for agent consumption.

Invocation via the [agentskills.io](https://agentskills.io/) `run_script` protocol:

```python
run_script("scripts/collect-evidence.sh", args=["-d", dgd_name, "-n", namespace])
```

Equivalent direct invocation:

```bash
bash scripts/collect-evidence.sh -d <dgd> -n <ns>
```

---

## References and Scripts

- [references/symptom-signatures.md](references/symptom-signatures.md) —
  Symptom-to-cause signature library. Each entry uses the strict
  6-element shape from.
- [references/debug-commands.md](references/debug-commands.md) — Reference
  kubectl / helm / curl commands the skill uses, grouped by surface.
- [references/known-issues.md](references/known-issues.md) — Pointer
  back to the stable issue patterns in `dynamo-deploy`'s
  `known-issues.md`; this skill's own signatures library is the more
  comprehensive surface for day-2.
- [scripts/collect-evidence.sh](scripts/collect-evidence.sh) — Passive
  evidence collector. PASS/FAIL/WARN per.
