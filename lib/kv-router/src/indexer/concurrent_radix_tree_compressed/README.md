# Concurrent Radix Tree Compressed

This indexer is optimized around two common KV-cache store patterns. They put
opposite pressure on the node locking strategy, so changes to node shape
bookkeeping should keep both in mind.

## Prefill Fanout

Many workers may share a prefix and then store different prompt suffixes under
the same parent. The hot operation is:

```text
shared prefix parent
many different workers
many different first child hashes
insert new compressed child nodes
```

The desired behavior is to let independent child inserts proceed concurrently.
Child insertion should avoid exclusive node-wide shape ownership when the parent
edge is stable. It should only serialize against heavy edge-shape changes that
would change what the child map means, such as splitting or extending the parent
edge.

## Decode Extension

During decode, a worker often appends small batches to the tail of its own
compressed edge. The hot operation is:

```text
one worker
one leaf compressed node
parent hash is current tail
no children
append more blocks to the edge
```

The desired behavior is a fast uncontended commit. This path should avoid child
allocation and child-map insertion. It can tolerate node-wide structural
serialization better than prefill fanout because the common case is effectively
one writer per decode chain.

## Locking Implication

These patterns should not be forced through one undifferentiated structural
writer path. A useful split is:

```text
light structural update = attach a child under the current stable edge tail
heavy structural update = extend/split/move children/change child depth
```

Light child insertion wants concurrency across independent child keys. Heavy
edge-shape updates need exclusive structural ownership because they change the
semantic depth of children.
