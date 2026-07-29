# Dynamo OpenEngine sidecar

This crate generates its Rust client and server bindings from the OpenEngine schema at immutable source commit `d09a7313b3af2fbcd9b17aa4d31c509207ab51db`. It does not consume a language-specific OpenEngine package or check generated bindings into Dynamo.

The `dynamo-openengine-sidecar` binary is intentionally engine-neutral. The same artifact discovers and serves TRT-LLM, vLLM, and SGLang endpoints without engine-name dispatch. `--expected-engine` and `--expected-schema-release` are optional deployment assertions; compatible schema revision ranges are negotiated independently of the release assertion.

`ModelInfo.model_id` is the canonical model and tokenizer source used by Dynamo preprocessing. The sidecar registers the primary served name plus aliases and forwards context-first handoff data without interpreting engine-specific attributes. Client-created bootstrap endpoint, room, and handoff identifiers are carried losslessly under `attributes_struct["openengine.client_bootstrap.v1"]`; a prefill connector's routable `local_endpoints` entry enables Dynamo's concurrent bootstrap path.

Launch and DGD examples live under each engine's `examples/backends/<engine>` tree. In Kubernetes, mount the same tokenizer/model cache into the CPU sidecar and GPU engine containers, and mount one shared LoRA cache across P/D pods.

For local development, set `OPENENGINE_PROTO_ROOT` to either the OpenEngine checkout root or its `proto` directory. The build verifies that a Git checkout is exactly at the pinned source commit and rejects dirty or untracked files under `proto`.

For a metadata-free schema export, also set `OPENENGINE_SCHEMA_RELEASE` to either the pinned 40-character lowercase source commit or the corresponding immutable 32-character lowercase BSR commit. That exact identity is embedded in the sidecar. Moving labels are rejected.

```bash
OPENENGINE_PROTO_ROOT=/home/connorc/sidecar/openengine-trtllm cargo build -p dynamo-openengine-sidecar

buf export buf.build/openengine/openengine:<immutable-bsr-commit> --output /tmp/openengine-schema
OPENENGINE_PROTO_ROOT=/tmp/openengine-schema \
OPENENGINE_SCHEMA_RELEASE=<immutable-bsr-commit> \
cargo build -p dynamo-openengine-sidecar
```

Alternatively, if `buf` is installed, `OPENENGINE_BSR_MODULE=buf.build/openengine/openengine:<immutable-bsr-commit>` exports that exact module into Cargo's build directory before generation. The reference must contain a 32-character lowercase commit, not a label.

The current sidecar worktree defaults to the requested local sibling checkout when no source variable is set. Release and CI builds should consume an immutable BSR commit.
