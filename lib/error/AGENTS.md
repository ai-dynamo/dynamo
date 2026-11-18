# lib/error

`dynamo-error` is the error type every Dynamo component classifies failures with (`DynamoError`, `ErrorClass`, `ErrorReason`, `PublicDetails`; DEP #14354). Classify once at the component closest to the cause; a status code follows from the class, never from the reason.

The `http` feature adds the `http` module (class-to-status policy, overload status, sanitized response vocabulary, `HttpError`), the `http` crate for `StatusCode` and `thiserror` for `HttpError`; nothing else may be added behind it.
