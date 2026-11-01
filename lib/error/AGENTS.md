# lib/error

- This crate is the error contract: `DynamoError`, `ErrorClass`, `ErrorReason`, `Diagnostic`, `PublicDetails`, the builder, and the reason catalog. Classification semantics follow DEP #14354: the component closest to the cause classifies once; boundaries render the class.
- Depend on serde and serde_json only. Never add `dynamo-runtime`, any router crate, tokio, anyhow, thiserror, tracing, axum, reqwest, zmq, or prometheus; the gate rejects them.
- Every public item is documented; `public-api.txt` is the committed surface (`scripts/router_public_api_check.sh --update` after an intended change).
- A new reason belongs in the catalog with its class; a status code is a function of the class, never of the reason.
- Run `scripts/router_layering_check.sh` before committing.
