# Verification V0 (frozen before implementation)

This directory is the black-box acceptance contract for the Dynamo Snapshot
proof of concept. It is deliberately separate from `implementation/`.

The files covered by `SHA256SUMS` must not change during I1 or V1. To revise
the protocol, record approval, increment the verification version, and create
a new checksum manifest before changing implementation code.

Run the contract tests from the project root with:

```bash
python3 -m unittest discover -s verification/v0/tests -v
```

`artifacts/` is not part of the frozen suite. It holds environment-specific
inventory, raw events and JSONL measurements produced by V0/V1 runs.
