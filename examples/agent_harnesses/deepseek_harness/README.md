<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek Harness with Dynamo

This package-only recipe runs the published DeepSeek Harness (DSH) headless profile against Dynamo's OpenAI-compatible Chat Completions endpoint. It preserves DSH-native root-session and compaction metadata on the model-hidden transport, captures redacted protocol evidence, and can send an explicit terminal signal to the native ThunderAgent router after normal or interrupted shutdown.

## Frozen tuple

| Artifact | Revision |
| --- | --- |
| Dynamo base | `a6261680a974ca7c74dcf49592a7376d7de99380` |
| Dynamo DSH normalization | `1b373f91a451d6a242aafd8851390f8ffdf4c3dc` plus the package-only cleanup in this branch |
| Complete recipe revision | `org.opencontainers.image.revision` image label, set from the clean checkout commit at build time |
| Published DSH client | `@deepseek-ai/dsh@0.1.0-rc.8` |
| Client base image | `node:24.10.0-bookworm-slim@sha256:b8d2197aff9129d16c801a3e3e1b2a873c4946480f5a310f38056df2268c38d9` |
| Image dependency graph | [`client/pnpm-lock.yaml`](client/pnpm-lock.yaml), installed by pnpm `11.7.0` with `--frozen-lockfile` |

No DSH source checkout or source patch is part of this supported path. Immediate parent lineage is not an acceptance criterion because the published package does not emit it.

## Run one headless task

Run the driver on a POSIX host with Linux `/proc` or `/bin/ps`. Set `DYNAMO_API_KEY` only when the endpoint requires it. The driver never reads an ambient `DEEPSEEK_API_KEY`; it selects `DYNAMO_API_KEY`, projects that value into the variable expected by DSH, passes a small allowlist of noncredential environment variables, and otherwise uses `dummy`. Use `--api-key-env NAME` only when explicitly selecting another credential variable. The driver creates an isolated temporary home and generated model profile, binds its relay only to loopback, and deletes the temporary home after the run.

```bash
node .agents/skills/dynamo-agent-harness/scripts/drive_deepseek_harness.mjs --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-0.6B --task 'Inspect this workspace with tools, report one verified fact, and do not modify files.' --capture evidence/dsh-headless.jsonl
```

The capture path uses exclusive creation. Choose a new path for each run. Add `--overwrite-capture` only when replacing that exact evidence file is intentional.

The package is pinned in the driver. A successful trace has a `request` record with `x-deepseek-harness-session-id`, a hashed `x-deepseek-harness-user-id`, the request body used by Dynamo to derive `input_trigger`, and a corresponding `response` status. Compaction requests additionally carry exact `x-deepseek-harness-compact: 1` and normalize to `agent_context.compaction`.

## Why this recipe is one-shot

This is a choice of DSH surface, not a Dynamo session limitation. The published `dsh --profile headless "task"` command is explicitly implemented as a one-task application: it creates one fresh persisted Agent, submits one top-level task, waits until the Agent is idle, flushes the session, prints the final assistant message, requests application exit, and terminates the process.

One top-level task can still produce several model calls, tool-result turns, compaction requests, and subagent activity. Those requests retain stable DSH session identity for the lifetime of the task. “One-shot” means the host process accepts no second top-level prompt after the answer; it does not mean one HTTP request or an ephemeral model context.

Keeping `DSH_HOME` preserves session artifacts but does not turn the headless command into an interactive server. A persistent automation path should be a separate recipe built from the published `@deepseek-ai/dsh-acp` package: keep one ACP stdio connection alive, call `session/new` once, then send repeated `session/prompt` requests for that session. DSH ACP currently owns sessions for the life of the connection and supports repeated prompts, but does not support loading or resuming a session after the connection exits. Human-interactive persistence belongs to DSH's TUI or Web profiles, not this CPU batch Job.

## Dynamo 1.3 compatibility bridge

The native DSH mapping in this branch is newer than the pinned Dynamo 1.3 runtime used by the shared qualification deployment. Add `--canonicalize-dynamo-headers` only for that older endpoint. The relay preserves the native DSH header and also copies its session value to `x-dynamo-session-id`. This restores identity and affinity on Dynamo 1.3, but it cannot add native DSH compaction normalization to that older server. Leave the option off when the server includes the native mapping so that mapping remains the contract under test.

```bash
node .agents/skills/dynamo-agent-harness/scripts/drive_deepseek_harness.mjs --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-0.6B --task 'Inspect this workspace with tools and report one verified fact.' --canonicalize-dynamo-headers --capture evidence/dsh-dynamo-1-3.jsonl
```

## ThunderAgent terminal cleanup

Add `--session-final` only when the endpoint is the native ThunderAgent frontend. The relay records every observed DSH session, waits for DSH to flush and exit, then sends one model-hidden `x-dynamo-session-final: true` request per session. It does the same after SIGINT or SIGTERM, bounds each terminal call to five seconds by default, and exits nonzero if Dynamo rejects cleanup or if no DSH session reached Dynamo. SIGINT and SIGTERM terminate the complete tracked Corepack/pnpm/DSH process tree, including detached descendants observed before shutdown, and retain conventional exit codes `130` and `143` after the bounded drain.

```bash
node .agents/skills/dynamo-agent-harness/scripts/drive_deepseek_harness.mjs --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-0.6B --task 'Run a tool and summarize its result.' --session-final --capture evidence/dsh-thunderagent.jsonl
```

Do not use `--session-final` with a stock KV frontend. Stock KV has no program lifecycle to close, and a generic Chat Completions frontend could treat the terminal envelope as ordinary model work. ThunderAgent consumes it at the router before model forwarding.

## Kubernetes client job

The checked-in ConfigMap targets the project-owned `agent-well-lit-stock-frontend` service, serves `Qwen/Qwen3-0.6B`, and enables the Dynamo 1.3 compatibility bridge. The manifests create only a CPU client in `anish-agent-well-lit-path`; they do not deploy or claim a GPU backend. Build from a clean checkout so the image label identifies the exact recipe commit. Set `DSH_CLIENT_TAG` to a writable registry tag before running these commands.

```bash
: "${DSH_CLIENT_TAG:?Set DSH_CLIENT_TAG to a writable registry tag}"
if [ -n "$(git status --porcelain)" ]; then echo "build from a clean checkout" >&2; exit 1; fi
DSH_RECIPE_COMMIT="$(git rev-parse HEAD)"
docker build --build-arg DYNAMO_RECIPE_COMMIT="$DSH_RECIPE_COMMIT" -f examples/agent_harnesses/deepseek_harness/Dockerfile.client -t "$DSH_CLIENT_TAG" .
docker push "$DSH_CLIENT_TAG"
```

Resolve the pushed tag to an immutable `name@sha256:digest` reference and export it as `DSH_CLIENT_IMAGE`. Do not apply the checked-in Job directly: its local-only image name is a render sentinel. The command below rejects mutable references, copies the manifests to a temporary directory, and replaces that sentinel without modifying the checkout.

```bash
: "${DSH_CLIENT_IMAGE:?Set DSH_CLIENT_IMAGE to the pushed name@sha256 digest}"
case "$DSH_CLIENT_IMAGE" in *@sha256:*) ;; *) echo "DSH_CLIENT_IMAGE must be immutable" >&2; exit 1;; esac
DSH_RENDER_DIR="$(mktemp -d)"
cp -R examples/agent_harnesses/deepseek_harness/kubernetes/. "$DSH_RENDER_DIR/"
python3 - "$DSH_RENDER_DIR/job.yaml" "$DSH_CLIENT_IMAGE" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
sentinel = "image: dsh-dynamo-client:0.1.0-rc.8"
replacement = f"image: {sys.argv[2]}"
source = path.read_text()
if source.count(sentinel) != 1:
    raise SystemExit("expected exactly one checked-in client image sentinel")
path.write_text(source.replace(sentinel, replacement))
PY
```

Create the namespace and ConfigMap, then create the API secret from the shell only when the endpoint requires authentication. The secret value never enters a manifest or the repository. Apply the rendered Job after its dependencies exist.

```bash
kubectl apply -f "$DSH_RENDER_DIR/namespace.yaml"
kubectl apply -f "$DSH_RENDER_DIR/configmap.yaml"
if [ -n "${DYNAMO_API_KEY:-}" ]; then
  kubectl -n anish-agent-well-lit-path create secret generic dsh-dynamo-api-key --from-literal=api-key="$DYNAMO_API_KEY" --dry-run=client -o yaml | kubectl apply -f -
fi
kubectl apply --dry-run=server -f "$DSH_RENDER_DIR/job.yaml"
kubectl apply -f "$DSH_RENDER_DIR/job.yaml"
kubectl -n anish-agent-well-lit-path wait --for=condition=Ready pod -l job-name=dsh-dynamo-client --timeout=10m
DSH_POD="$(kubectl -n anish-agent-well-lit-path get pod -l job-name=dsh-dynamo-client -o jsonpath='{.items[0].metadata.name}')"
kubectl -n anish-agent-well-lit-path logs -f "$DSH_POD" -c dsh
kubectl -n anish-agent-well-lit-path wait --for=condition=complete job/dsh-dynamo-client --timeout=30m
kubectl -n anish-agent-well-lit-path cp "$DSH_POD:/evidence/dsh-request-trace.jsonl" ./dsh-request-trace.jsonl -c dsh
```

The image installs the published package from the frozen lockfile and passes its installed CLI entrypoint through `--dsh-bin`; this option does not imply a source build. Set `DSH_SESSION_FINAL` to `true` only in a ConfigMap targeting a ThunderAgent service. Against a server containing the native DSH mapping, set `DSH_CANONICALIZE_DYNAMO_HEADERS` to `false` before creating the Job.

Remove only the project-owned client objects after copying evidence. Keep the namespace when it also contains the shared Dynamo backend.

```bash
kubectl -n anish-agent-well-lit-path delete job dsh-dynamo-client --ignore-not-found
kubectl -n anish-agent-well-lit-path delete configmap dsh-dynamo-client --ignore-not-found
kubectl -n anish-agent-well-lit-path delete secret dsh-dynamo-api-key --ignore-not-found
rm -rf "$DSH_RENDER_DIR"
```

Before applying anything on NScale, recalculate cluster allocation, confirm the namespace is still absent or project-owned, and keep all well-lit-path workloads below two nodes and 16 GPUs in aggregate. Never pin a node, copy another workload's DRA claim, tolerate a reserved taint, or mutate node labels or taints.

## Evidence and acceptance

Run the local compatibility checks with:

```bash
DSH_PACKAGE_SMOKE=1 node --test .agents/skills/dynamo-agent-harness/scripts/test_drive_deepseek_harness.mjs
cargo test -p dynamo-llm --no-default-features agent_context_from_deepseek_harness_headers_preserves_compaction
```

For a real tool run, retain the relay JSONL, Dynamo request trace, DSH stdout and stderr, Kubernetes manifests, image digests, model identity, frontend or router logs, and worker telemetry. Match the relay's native session value to Dynamo's normalized `agent_context.session_id`; for ThunderAgent, also prove the final request was handled without model forwarding and that no program remains live.

## Security and ownership boundary

The relay reads only `DYNAMO_API_KEY` by default, projects the selected value to DSH as `DEEPSEEK_API_KEY`, and never writes it to evidence. The DSH process receives an isolated home plus an allowlist containing executable search path, locale, terminal, temporary-directory, timezone, and certificate settings; unrelated parent credentials and package-manager cache locations are not inherited. The relay records model request bodies in plaintext, hashes the stable anonymous DSH user ID, and creates the trace with owner-only permissions and exclusive creation by default. Treat the resulting file as sensitive because prompts, tool results, paths, and source excerpts can still be present. DSH owns its profiles, persistence, tools, sandbox, credentials, subagent behavior, and future native lifecycle hooks. Dynamo owns protocol normalization, affinity and routing, request tracing, and ThunderAgent program lifecycle. The relay is an integration shim, not a second DSH runtime or a public Web gateway.

## Known limitations

- This supported path carries root-session and compaction metadata only; immediate parent lineage is intentionally out of scope.
- A direct local run uses a top-level-pinned `pnpm dlx` package and can resolve transitive dependencies again. Use the container's frozen lockfile for reproducible qualification evidence.
- The headless product surface accepts one top-level task. Persistent multi-turn DSH requires a separately packaged ACP host and is not claimed by this recipe.
- The terminal hook observes sessions at the relay. A DSH child that never reaches the model endpoint cannot be discovered or finalized by the shim.
- Input triggers are derived by Dynamo from the Chat Completions body, not emitted as a DSH-specific header. Validate `user_message` and `tool_result` in Dynamo request traces.
- The relay is loopback-only and intentionally minimal; it is not a multi-tenant authentication, rate-limit, or policy boundary.
- The Dynamo 1.3 compatibility bridge preserves identity and affinity but cannot backport native DSH compaction normalization.
- Windows is not qualified because the relay requires POSIX process-tree signaling.

## Future persistent path

The next DSH milestone should compose the published `@deepseek-ai/dsh-acp` server with the same pinned model, sandbox, and persistence plugins, then add an ACP host driver that keeps one connection open for multiple prompts and translates connection teardown into bounded ThunderAgent finalization. This is integration and packaging work in Dynamo; it does not require a DSH source patch.
