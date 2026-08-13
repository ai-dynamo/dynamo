# V2 proposal status

This directory is separate from the frozen V0.1 protocol and immutable V1
evidence. `protocol.draft.json` is a proposal, not authorization to measure or
change the live environment.

Execution remains blocked until the user explicitly approves the draft. V2-A
then instruments the existing supported non-GMS path. V2-B requires another
approval and may evaluate exactly one optimization only after V2-A attributes
the bottleneck. GMS is not eligible while Dynamo v1.3.0 documents GMS plus
Snapshot as disabled.

Every run uses the `v2-` prefix, seed `20260814`, new run identifiers, and new
artifact directories. Nothing in V0, I1, or V1 may be modified or overwritten.
