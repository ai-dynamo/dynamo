# lib/model-card

`dynamo-model-card` owns the model vocabulary every Dynamo component shares: the `ModelDeploymentCard` and its artifacts (`CheckedFile`, tokenizer and prompt-formatter kinds, the card checksum), `ModelRuntimeConfig`, `LocalModel` and its builder, `ModelType`/`ModelInput`/`EndpointType`, the hub loader, request templates, the frontend configuration groups a `LocalModel` carries, and the configuration a card serializes (`RouterConfig`, `LoadThresholdConfig`, the media decoder and fetcher settings, tensor model metadata, and the parser family names a card may select). It is data, defaults, validation, and loading; it serves no requests.

- Depends on (Dynamo crates): `dynamo-runtime`, `dynamo-kv-hints`, `dynamo-kv-indexer`, `dynamo-kv-router`, `dynamo-lora`, `dynamo-router-protocol`, `dynamo-scheduler`, `dynamo-session-affinity`, `dynamo-slot-tracker`, and the published `dynamo-tokenizers`, `dynamo-renderer`, `dynamo-parsers`, `dynamo-parsers-v2`.
- Must not depend on: `dynamo-llm` or any crate above it (bindings, sidecars, backend-common, mocker servers). Request and response types, preprocessing, decoding, HTTP and gRPC services stay in `dynamo-llm`; a function that needs them belongs there and takes the configuration as an argument.
- A card field changes the wire every frontend and worker reads from discovery: keep serde compatibility tests next to the type, and read `src/local_model/AGENTS.md` before adding to `ModelRuntimeConfig`.
- Hosts import this crate directly; `dynamo-llm` does not re-export it.
- Tests read the sample models under `lib/llm/tests/data/sample-models`.
- Check: `cargo test -p dynamo-model-card`, and `cargo tree -p dynamo-model-card | grep dynamo-llm` must print nothing.
