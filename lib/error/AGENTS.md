# lib/error

`dynamo-error` is the error type every Dynamo component classifies failures with (`DynamoError`, `ErrorClass`, `ErrorReason`, `PublicDetails`; DEP #14354). Classify once at the component closest to the cause; a status code follows from the class, never from the reason.
