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
7 lora_name
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

SGLang workers should always publish with `"format": "dynamo"` in
`--kv-events-config`, with or without LoRA. `dynamo.sglang` adds it when no
format is set and the installed SGLang supports it; an SGLang server started
another way must set it. That format follows this wire format: map events
parsed by name, with `lora_name` on `BlockStored`. Other SGLang shapes are
legacy and parsed only for compatibility: the default format, and the
positional `BlockStored` of releases before 0.5.20, which ends at `lora_id` and
parses because the tuple terminates early.

SGLang builds with routing-label support add the optional named field
`mm_routing_spans` to `BlockStored` in the `dynamo` format. It is a list aligned
with `block_hashes`; each block contains `[start, length, label]` spans with
block-relative token offsets and lengths. The label echoes Dynamo's hexadecimal
`u64` media hash. For example, `[[], [[0, 4, "0123456789abcdef"]]]` labels the
first four tokens of the second block. The field adds no positional slot.
For sources flagged with `engine_owned_mm_identity`, events without usable
labels retain multimodal exclusion.

SGLang's `dynamo` format can also report `request_cache_salt`, the request
namespace supplied to the engine, including one set by a proxy. Selection is by field
presence: a string selects that namespace, explicit `null` selects unsalted
routing, and only an omitted field falls back to `cache_salt` and the legacy
namespace rules. Empty strings are unsalted. LoRA identity remains independent.
For example, `{"cache_salt": "sglang-mm-v1:...", "request_cache_salt": null}`
must hash routing tokens without the internal media salt. This adds no
positional slot. SGLang omits the field when it equals `cache_salt`.

For `engine_owned_mm_identity` sources, all indexed blocks receive canonical
routing IDs, including text-only requests and text before media. The internal
media salt gives identical text prefixes different engine IDs; translating only
media blocks would leave those prefixes aliased during eviction. A continuation
whose parent mapping is unknown is excluded until its chain can be reconstructed.
