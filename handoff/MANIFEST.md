# Dynamo production Kubernetes handoff manifest

Created: 2026-05-13

## Source Repositories

| Name | Remote | Role |
|---|---|---|
| `origin` | `https://github.com/ai-blaise/dynamo-prod-k8s` | Fork and write target |
| `upstream` | `https://github.com/ai-dynamo/dynamo` | Read-only Dynamo upstream |
| `infrastructure` | `https://github.com/ai-blaise/infrastructure` | Deployment helper scripts and environment-specific operations |

Last checked before adding this handoff kit:

| Repository | Branch | Commit |
|---|---|---|
| `ai-blaise/dynamo-prod-k8s` | `main` | `3f4fbf3b87ded3d1593abaf3c1d5003bf9ce5000` |
| `ai-dynamo/dynamo` | `main` | `f60d1d8e5351ab7fc3979220cbd0c48ae80578eb` |
| `ai-blaise/infrastructure` | `main` | `039fb27705f880a394f393628bd1718b416031b3` |

Do not rely on these hashes as current. Re-run `git ls-remote` before every
merge or deployment phase.

## Protected Custom Surfaces

These are fork-owned and must be preserved across upstream merges:

- `deploy/production/**`
- `deploy/production/addons/**`
- `deploy/production/examples/deepseek-v32-reap-sglang.yaml`
- `deploy/production/docs/smg-integration.md`
- `deploy/production/runbooks/**`
- `deploy/production/policies/**`
- `tests/deploy/test_deepseek_reap_manifest_contract.py`
- `tests/deploy/test_smg_boundary_contract.py`
- `components/src/dynamo/frontend/frontend_args.py`
- `components/src/dynamo/frontend/main.py`
- `lib/llm/src/model_card.rs`
- `docs/components/frontend/Tokenizer.md`
- `docs/components/frontend/configuration.md`
- `docs/features/tokenizer/README.md`

`deploy/production/**` is not upstream Dynamo content. Treat it as custom
production infrastructure for this fork.

## Production Add-ons

The production stack documents and wires these add-ons through GitOps instead of
bundling them into the Dynamo Helm chart:

- Actions Runner Controller
- External Secrets Operator
- Falco
- Fluentd
- GPU Operator
- Grafana Loki
- Grove
- KAI Scheduler
- KEDA
- kube-no-trouble
- LeaderWorkerSet
- OpenTelemetry
- Parca
- Prometheus Operator / kube-prometheus
- SMG
- Trivy Operator
- Velero
- Volcano

SMG is HTTP-only in this fork's production profile. It does not provide
tokenization, detokenization, reasoning parsing, tool-call parsing, multimodal
processing, MCP orchestration, or durable chat history.

## Tokenizer Contract

FastOkenS is the default frontend tokenizer path for this fork.

- CLI default: `--tokenizer fastokens`
- Env var passed to Rust: `DYN_TOKENIZER=fastokens`
- HuggingFace opt-out: `--tokenizer default` or `DYN_TOKENIZER=default`
- TikToken-format tokenizers are unchanged.
- If FastOkenS cannot load a supported BPE `tokenizer.json`, Dynamo falls back
  to HuggingFace without dropping requests.

When upstream changes tokenizer docs or frontend config, reconcile those changes
with this fork default before committing.

## Deployment Target

All deployment and production verification work should run on:

- VM: `instance-20260415-161450`
- Zone: `asia-south1-b`
- Project: `blaise-478114`

The second `asia-sout1-b` node may be used only if the primary VM is unavailable.

## Standard Verification

Run these locally for quick feedback, then repeat the meaningful checks on the
VM before treating an implementation phase as complete.

```bash
PYTHONPATH=components/src python -m pytest \
  tests/deploy/test_deepseek_reap_manifest_contract.py \
  tests/deploy/test_smg_boundary_contract.py

python - <<'PY'
from pathlib import Path
import yaml
count = 0
for path in Path("deploy/production").rglob("*.yaml"):
    with path.open() as handle:
        list(yaml.safe_load_all(handle))
    count += 1
print(f"parsed {count} production YAML files")
PY

bash -n deploy/pre-deployment/pre-deployment-check.sh tests/smg-roundtrip.sh
kubectl kustomize deploy/production/gitops >/tmp/dynamo-prod-k8s-kustomize.yaml
cargo fmt --check --package dynamo-tokenizers --package dynamo-llm --package dynamo-kv-router
cargo test -p dynamo-tokenizers fastokens
```

For merge-only phases, also verify:

```bash
git diff --cached -- deploy/production
git log -1 --oneline --parents
git ls-remote https://github.com/ai-blaise/dynamo-prod-k8s.git refs/heads/main
git ls-remote https://github.com/ai-dynamo/dynamo.git refs/heads/main
```

## Merge Rules

1. Merge upstream Dynamo into this fork in a way that keeps custom production
   integrations intact.
2. Prefer composing upstream changes with fork changes over choosing one side
   wholesale.
3. Never delete `deploy/production/**` during conflict resolution.
4. Preserve FastOkenS as the default tokenizer unless the user explicitly
   changes that production contract.
5. Keep third-party add-ons externally managed through Argo CD/GitOps.
6. Do not introduce generated comments, broad defensive wrappers, or casts that
   are inconsistent with nearby code.
7. Commit and push to `ai-blaise/dynamo-prod-k8s` after implementation phases.

## Model Deployment Context

Primary production test model:

- `cerebras/DeepSeek-V3.2-REAP-345B-A37B`
- Inference engine: SGLang
- Deployment shape: full DeepSeek-style TP + DP, using the SGLang DeepSeek V3.2
  launch guidance and the fork's production manifest.

Use `ai-blaise/infrastructure` for environment-specific deployment help.
