# OpenEngine sidecar

> [!WARNING]
> **Experimental.** This initial sidecar targets OpenEngine v0.1.0 and the TensorRT-LLM OpenEngine server. Its supported request surface is intentionally small.

`dynamo-openengine-sidecar` connects a Dynamo worker to an out-of-process OpenEngine gRPC server. It uses the shared sidecar worker and transport arguments, discovers the server role and model through `GetServerInfo` and `GetModelInfo`, and supports aggregated generation plus context-first prefill/decode handoff.

## Run

Install TensorRT-LLM's optional OpenEngine bindings, then start an aggregated server:

```bash
python -m pip install --extra-index-url https://buf.build/gen/python \
  "tensorrt_llm[openengine]"

trtllm-serve <model> \
  --grpc \
  --grpc-protocol openengine \
  --host 127.0.0.1 \
  --port 50051
```

Start the Dynamo worker:

```bash
dynamo-openengine-sidecar \
  --grpc-endpoint 127.0.0.1:50051
```

The sidecar discovers the canonical model source and served name from TensorRT-LLM. All standard worker and gRPC transport options come from `dynamo_sidecar_common::SidecarArgs`.

For disaggregated serving, start TensorRT-LLM with `--server_role context` or `--server_role generation`, then run one sidecar beside each server. The OpenEngine server role is authoritative; the inherited `--disaggregation-mode` option is not used.

## Initial scope

- OpenEngine v0.1.0 `Generate`, `GetServerInfo`, and `GetModelInfo`
- Token-ID input, one output sequence, sampling, stopping, output logprobs, priority, and DP-rank metadata
- Aggregated and context-first prefill/decode serving
- Opaque `KvSessionRef` preservation through Dynamo's prefill handoff

The initial TensorRT-LLM server leaves the other OpenEngine control RPCs unimplemented. This sidecar therefore does not call health, load, abort, LoRA, or KV-event RPCs. Multimodal input, guided decoding, prompt logprobs, LoRA selection, RL controls, encode workers, beam search, and `n > 1` are rejected locally.

The vendored schema provenance is recorded in [`proto/README.md`](proto/README.md).
