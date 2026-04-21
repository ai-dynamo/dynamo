# SKILLS.md — Operations cookbook & agent skills for ai-dynamo/dynamo

Two audiences, two sections. The **Operations cookbook** (top) is for anyone
deploying or operating Dynamo. The **Contributor skills** (bottom) is for
Dynamo maintainers and the AI agents helping them.

If you're an agent (Claude Code, Cursor, Devin), start at
[`CLAUDE.md`](CLAUDE.md) for routing.

---

## Operations cookbook (for users)

> **Audience:** anyone running Dynamo. Copy-pasteable procedures, no Claude
> Code or `gh` setup required.

### User skills (Claude Code / Cursor)

If you're driving Dynamo through an agent CLI, invoke these skill modules
instead of the inline procedures below.

| Skill | Purpose |
|---|---|
| [`quickstart`](.claude/skills/quickstart/SKILL.md) | Walk a recipe from zero to a smoke-tested deployment -- recipe pick, namespace + HF secret, storage class, model cache, deploy, ready-wait, port-forward, smoke test. |
| [`verify-cluster`](.claude/skills/verify-cluster/SKILL.md) | Preflight: CRDs, default storage class, GPU driver, image-pull, GPU operator. Returns go/no-go with remediation. |
| [`inspect-pods`](.claude/skills/inspect-pods/SKILL.md) | Pod status + logs + describe + events + frontend port-forward, with the right Dynamo pod selector. |
| [`troubleshoot`](.claude/skills/troubleshoot/SKILL.md) | Match a failure symptom to a ranked cause + fix from `docs/troubleshooting.md`, with live cluster evidence. |

The procedures below remain as a fallback for users without an agent CLI.

### Build a local image for Kimi-k2.5 (or any top-of-tree use case)

```bash
./container/build.sh --framework vllm --tag my-tag
docker push <your-registry>/ai-dynamo/vllm-runtime:my-tag
yq -i '(.spec.services[].extraPodSpec.mainContainer.image) |= sub("nvcr\.io/nvidia/ai-dynamo/(.+):my-tag", "<your-registry>/ai-dynamo/$1:my-tag")' recipes/kimi-k2.5/vllm/agg/deploy.yaml
```

### Verify Dynamo CRDs before applying recipes

```bash
kubectl get crd | grep dynamo
# Expect three: dynamocomponentdeployments, dynamographdeployments, dynamographdeploymentrequests
```

### Swap storage class for your cluster

```bash
STORAGE_CLASS=$(kubectl get storageclass -o jsonpath='{.items[?(@.metadata.annotations.storageclass\.kubernetes\.io/is-default-class=="true")].metadata.name}')
yq -i "(.. | select(has(\"storageClassName\")).storageClassName) = \"$STORAGE_CLASS\"" deploy.yaml
```

### Enable observability export on a running deployment

```bash
kubectl set env deployment/<dynamo-frontend> \
  OTEL_EXPORT_ENABLED=true \
  OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://tempo:4317 \
  OTEL_EXPORTER_OTLP_LOGS_ENDPOINT=http://loki-otlp:4317 \
  DYN_SYSTEM_PORT=8081
```

### Confirm driver matches the image you pulled

```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
# Cross-check against the Feature Support Matrix in docs/backends/trtllm/README.md
```

---

## Contributor skills (for maintainers)

> **Audience:** Dynamo maintainers and AI agents (Claude Code, Cursor) helping
> them. Each entry is a runnable skill module under
> [`.claude/skills/`](.claude/skills/) with its own `SKILL.md` and supporting
> files.

| Skill | Purpose |
|---|---|
| [`debug-session`](.claude/skills/debug-session/SKILL.md) | Start a structured debugging session with a worklog file for an issue in the Dynamo ecosystem. |
| [`dep-create`](.claude/skills/dep-create/SKILL.md) | Create a new Dynamo Enhancement Proposal (DEP) as a GitHub Issue on `ai-dynamo/dynamo`. |
| [`dep-status`](.claude/skills/dep-status/SKILL.md) | List DEP issues with status, area, PIC, and approval state; find DEPs related to a topic. |
| [`dep-update`](.claude/skills/dep-update/SKILL.md) | Update DEP status through its lifecycle — triage, review, approve, defer, or close. |
| [`dynamo-docs`](.claude/skills/dynamo-docs/SKILL.md) | Maintain the Dynamo Fern docs site — add, update, move, or remove pages. |
| [`gh-issue-bug`](.claude/skills/gh-issue-bug/SKILL.md) | File a well-structured bug report against `ai-dynamo/dynamo` using conversation context. |
| [`pr-monitor`](.claude/skills/pr-monitor/SKILL.md) | Check CI status, analyze failures, and explain skips for a Dynamo PR. |
| [`tool-parser-generator`](.claude/skills/tool-parser-generator/SKILL.md) | Generate optimized tool-call parsers from HuggingFace model chat templates. |

### Adding a contributor skill

1. Add a markdown file under `.claude/skills/<skill-name>/SKILL.md` with frontmatter:
   ```yaml
   ---
   name: <skill-name>
   description: <one-line description>
   ---
   ```
2. Append an entry to the "Contributor skills" table above.
3. Commit both.
