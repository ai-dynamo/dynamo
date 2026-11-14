# Hybrid lookup A/B benchmark

Build each snapshot with identical flags, preserving each executable before
building the other snapshot. Force an indexer package rebuild when sharing a
Cargo target between archived source snapshots.

```sh
CARGO_PROFILE_DEV_OPT_LEVEL=2 CARGO_PROFILE_DEV_DEBUG=0 \
  CARGO_PROFILE_DEV_INCREMENTAL=false \
  cargo build -p dynamo-kv-indexer --example hybrid_lookup --locked --offline
python3 lib/router/kv-indexer/benches/run_hybrid_lookup_ab.py \
  --baseline-bin /path/to/baseline-bin --final-bin /path/to/final-bin \
  --output /path/to/new-results --duration-ms 1000 --repeats 5
```

The standard-library-only runner alternates fresh processes and validates
all fixture scores inside the executable. It produces raw JSONL and median
summary JSON. The larger experiment report and source snapshots stay outside
the repository.
