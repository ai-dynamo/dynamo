# lib/error

`dynamo-error` is the error type every Dynamo component classifies failures with (`DynamoError`, `ErrorClass`, `ErrorReason`, `PublicDetails`; DEP #14354). Classify once at the component closest to the cause; a status code follows from the class, never from the reason.

## Rules

- **Adding an `ErrorReason`.** Add the key to `ErrorReason::catalog_class` under exactly one normalized class. `ErrorReason::new` rejects a key that is not there, and a class and reason that disagree fail closed to `Internal/runtime.invalid_error` (`unknown_catalog_reason_fails_closed`, `malformed_semantic_reason_fails_closed`). A key is lowercase ASCII, digits, and `._-`, at most `ErrorReason::MAX_BYTES`. Don't build one from dynamic text: reasons are metric labels.
- **Adding an `ErrorClass` variant.** Add its arm to `as_str`, `normalized`, and `ErrorReason::for_class`, and a round-trip test beside `semantic_metadata_roundtrips`. A class a reader does not know deserializes as `Internal` (`unknown_class_fails_closed_during_deserialization`).
- **Choosing an HTTP status.** Only in the `http` module, behind the `http` feature. `http::http_action_for_class` maps the normalized class to a status. Don't pick a status from a reason, and don't add a status field to `DynamoError`.
- **Putting text on an error.** `Diagnostic` is for logs and traces only; never copy it into a response or a metric label. Only `PublicDetails` may reach a client, and it does not sanitize, so fill it only where the value is known to be client-safe (#18133).
- **Adding a field to `PublicDetails` or `DynamoError`.** They cross the wire between versions. Use `#[serde(default, skip_serializing_if = "Option::is_none")]` (`context_length_details_accept_payloads_without_prompt_tokens`).
- **Adding a dependency.** The crate depends on `serde` and `serde_json` only, and every component links it. The optional `http` feature adds `http` and `thiserror`; do not add dependencies or non-HTTP code behind it.

## Test

```bash
cargo test -p dynamo-error --lib --all-features
```
