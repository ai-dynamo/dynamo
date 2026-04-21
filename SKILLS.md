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

> **Heads up:** the cookbook entries above are inline procedures, not invocable
> skill modules. Promoting them to runnable `.claude/skills/<name>/SKILL.md`
> entries is tracked in [issue #8456](https://github.com/ai-dynamo/dynamo/issues/8456).
> Until then, agents should follow the inline steps as the source of truth.

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
