# lib/kv-events

`dynamo-kv-events` owns the KV cache event types engines publish, the `RouterEvent` envelope, the engine ZMQ codec, the replay cursor, and the residency vocabulary. Changing a wire shape changes what engines and sidecars publish, so keep the serde compatibility tests next to the type.
