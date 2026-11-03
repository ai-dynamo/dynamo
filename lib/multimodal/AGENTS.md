# lib/multimodal

`dynamo-multimodal` is the multimodal identity both ends of the wire compute: the per-block and per-request metadata (`BlockExtraInfo`, `RequestExtraInfo`), the pad and identifier rules, and the vLLM extra-key marker. Engines publish with it and the router matches with it; keep the known-answer tests next to each rule.

- Depends on (Dynamo crates): none.
- Must not depend on: the runtime, tokio, any transport (zmq, axum, hyper, reqwest, prometheus), any router crate (it sits below `dynamo-kv-hashing` and `dynamo-kv-events`).
- Hosts import this crate directly.
- Check: `bash scripts/router_layering_check.sh` (and `--host`).
- Design: `lib/router/README.md`.
