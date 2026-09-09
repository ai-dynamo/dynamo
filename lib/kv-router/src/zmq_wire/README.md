# ZMQ KV Event Wire Parsing

This module decodes msgpack KV events emitted by engine-side ZMQ publishers.
It supports both tagged map/object events and Python `msgspec` tagged
`array_like=True` events.

## Map Events

Map/object events are parsed by field name. Field order does not matter, and
optional fields may be omitted independently.

Rust test fixtures must use `rmp_serde::to_vec_named` or
`Serializer::with_struct_map()` to produce this shape. Plain
`rmp_serde::to_vec` encodes structs positionally.

## Positional Events

Array-like events are positional tuples. The parser treats each event as a
fixed positional prefix followed by an optional metadata tail.

`BlockStored` fixed prefix:

```text
0 tag
1 block_hashes
2 parent_block_hash
3 token_ids
4 block_size
5 old lora_id slot
6 medium
7 lora_name (vLLM string) or BlockStoredMetadata (SGLang map)
8 extra_keys
```

`BlockStored` optional tail:

```text
block_mm_infos
group_idx
kv_cache_spec_kind
kv_cache_spec_sliding_window
```

`BlockRemoved` fixed prefix:

```text
0 tag
1 block_hashes
2 medium
```

`BlockRemoved` optional tail:

```text
group_idx
kv_cache_spec_kind
kv_cache_spec_sliding_window
```

For positional events, later fields require placeholders for all earlier
positional slots. For example, a `BlockStored` event that wants to carry
`group_idx` must also include positions 5 through 8, even if they are `None`.
The event may terminate early only when no later fields are present.

The tail is parsed by type where possible. Because `group_idx` and
`kv_cache_spec_sliding_window` are both `u32`, numeric tail fields are
order-sensitive: the first numeric tail value is `group_idx`, and the second is
`kv_cache_spec_sliding_window`.

## Producer Notes

vLLM's `BlockStored` tuple includes the vLLM-compatible fixed prefix before
the metadata tail, so `group_idx` and cache metadata are parsed from tail
positions.

SGLang emits the vLLM-compatible prefix through `medium` (positions 0 to 6)
and does not emit cache-group metadata. An unsalted event terminates at
position 6 and parses because the tuple ends early.

Since SGLang 0.5.18 (sgl-project/sglang#30827), a stored node that carries a
`cache_salt` is emitted as `BlockStoredWithMetadata`: the same prefix plus a
`BlockStoredMetadata` map at position 7 with the key `cache_salt`.
sgl-project/sglang#37482 adds a `session_id` key to that map. The parser tells
this map apart from vLLM's `lora_name` string by type, maps `cache_salt` to the
cache namespace, and ignores keys it does not model. SGLang never emits
`extra_keys` or the tail fields, so the map is the last element.
