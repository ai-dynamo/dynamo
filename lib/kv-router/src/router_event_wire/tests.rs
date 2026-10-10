// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::Serialize;

use super::*;
use crate::identity::{
    CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, PoolId, RoutingScopeId, StableDpSlotId,
};
use crate::protocols::BlockMmObjectInfo;

fn serde_decode(bytes: &[u8]) -> Result<Vec<RouterEvent>, rmp_serde::decode::Error> {
    rmp_serde::from_slice(bytes)
}

fn walk(bytes: &[u8]) -> Option<Vec<RouterEvent>> {
    Walker::new(bytes).router_events()
}

/// The typed walker must decode `bytes` itself, without the serde fallback,
/// to exactly the serde value.
fn assert_walks(bytes: &[u8]) -> Vec<RouterEvent> {
    let expected = serde_decode(bytes).expect("serde must accept the payload");
    let walked = walk(bytes).expect("typed walker must handle the payload");
    assert_eq!(walked, expected);
    walked
}

/// The public decoder must match serde exactly, and the walker must never
/// accept a payload serde rejects or decode one differently. Returns whether
/// serde accepts the payload.
fn assert_matches_serde(bytes: &[u8]) -> bool {
    let expected = serde_decode(bytes);
    if let Some(walked) = walk(bytes) {
        assert_eq!(
            Some(&walked),
            expected.as_ref().ok(),
            "walker diverged from serde on {bytes:02x?}"
        );
    }
    let decoded = decode_router_event_batch(bytes);
    assert_eq!(format!("{decoded:?}"), format!("{expected:?}"));
    expected.is_ok()
}

fn cache_owner() -> CacheOwnerId {
    CacheOwnerId::new(
        PoolId::new(
            IndexerDomainId::new(
                CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
                RoutingScopeId::new([2; 16], IdentitySource::DefaultDerived),
            ),
            DcId::new(3),
        ),
        StableDpSlotId::new([4; 16], IdentitySource::Explicit),
    )
}

fn block(block_hash: u64, tokens_hash: u64) -> KvCacheStoredBlockData {
    KvCacheStoredBlockData {
        block_hash: ExternalSequenceBlockHash(block_hash),
        tokens_hash: LocalBlockHash(tokens_hash),
        mm_extra_info: None,
    }
}

fn kv_event(event_id: u64, data: KvCacheEventData, dp_rank: u32) -> KvCacheEvent {
    KvCacheEvent {
        event_id,
        data,
        dp_rank,
    }
}

fn stored(
    parent_hash: Option<u64>,
    start_position: Option<u32>,
    blocks: Vec<KvCacheStoredBlockData>,
) -> KvCacheEventData {
    KvCacheEventData::Stored(KvCacheStoreData {
        parent_hash: parent_hash.map(ExternalSequenceBlockHash),
        start_position,
        blocks,
    })
}

fn removed(block_hashes: &[u64]) -> KvCacheEventData {
    KvCacheEventData::Removed(KvCacheRemoveData {
        block_hashes: block_hashes
            .iter()
            .copied()
            .map(ExternalSequenceBlockHash)
            .collect(),
    })
}

/// Integers straddling every MessagePack unsigned width boundary.
const WIDTHS: [u64; 10] = [
    0,
    127,
    128,
    255,
    256,
    65_535,
    65_536,
    u32::MAX as u64,
    u32::MAX as u64 + 1,
    u64::MAX,
];

/// One batch covering every event variant, storage tier, wire residency
/// domain, and optional field both present and absent.
fn every_variant_batch() -> Vec<RouterEvent> {
    let text_blocks = WIDTHS
        .iter()
        .zip(WIDTHS.iter().rev())
        .map(|(&block_hash, &tokens_hash)| block(block_hash, tokens_hash))
        .collect();
    let mut mm_block = block(42, 43);
    mm_block.mm_extra_info = Some(BlockExtraInfo {
        mm_objects: vec![
            BlockMmObjectInfo {
                mm_hash: u64::MAX,
                offsets: vec![(0, 4), (9, 16)],
            },
            BlockMmObjectInfo {
                mm_hash: 7,
                offsets: Vec::new(),
            },
        ],
    });

    let mut legacy_cleared = RouterEvent::with_storage_tier(
        11,
        kv_event(4, KvCacheEventData::Cleared, 0),
        StorageTier::External,
    );
    legacy_cleared.residency_domain = WireResidencyDomain::Missing;
    let mut future_domain = RouterEvent::new(12, kv_event(5, stored(None, None, Vec::new()), 0));
    future_domain.residency_domain = WireResidencyDomain::Unknown("future_owner".into());
    let mut invalid_domain = RouterEvent::new(13, kv_event(6, removed(&[]), 0));
    invalid_domain.residency_domain = WireResidencyDomain::Invalid;

    vec![
        RouterEvent::new(
            7,
            kv_event(1, stored(Some(u64::MAX), Some(u32::MAX), text_blocks), 0),
        ),
        RouterEvent::with_storage_tier(
            u64::MAX,
            kv_event(u64::MAX, stored(None, None, vec![mm_block, block(1, 2)]), 3),
            StorageTier::HostPinned,
        )
        .with_session_id("session-\u{3b1}"),
        RouterEvent::with_cache_owner(
            9,
            kv_event(3, removed(&WIDTHS), u32::MAX),
            StorageTier::Disk,
            cache_owner(),
        ),
        RouterEvent::with_residency_domain(
            10,
            kv_event(7, KvCacheEventData::Cleared, 1),
            StorageTier::Device,
            ResidencyDomain::CacheOwner,
        )
        .with_state_source(cache_owner()),
        legacy_cleared,
        future_domain,
        invalid_domain,
    ]
}

#[test]
fn current_encoder_round_trips_every_variant_through_typed_walker() {
    let batch = every_variant_batch();
    let bytes = rmp_serde::to_vec_named(&batch).unwrap();
    assert_eq!(assert_walks(&bytes), batch);
    assert_eq!(decode_router_event_batch(&bytes).unwrap(), batch);
    assert!(walk(&rmp_serde::to_vec_named(&Vec::<RouterEvent>::new()).unwrap()).is_some());
}

/// Wire bytes the current publisher emits for a stored, removed, and cleared
/// event. Pinned so encoder drift is caught here rather than in production.
const GOLDEN_BATCH: &str = concat!(
    "9384a9776f726b65725f696407ac73746f726167655f74696572a6646576696365b07265",
    "736964656e63795f646f6d61696ea6776f726b6572a56576656e7483a86576656e745f69",
    "6401a46461746181a673746f72656483ab706172656e745f68617368cd0100ae73746172",
    "745f706f736974696f6e10a6626c6f636b739283aa626c6f636b5f68617368cf01234567",
    "89abcdefab746f6b656e735f6861736811ad6d6d5f65787472615f696e666fc083aa626c",
    "6f636b5f686173680cab746f6b656e735f6861736816ad6d6d5f65787472615f696e666f",
    "c0a764705f72616e6b0084a9776f726b65725f696407ac73746f726167655f74696572ab",
    "686f73745f70696e6e6564b07265736964656e63795f646f6d61696ea6776f726b6572a5",
    "6576656e7483a86576656e745f696402a46461746181a772656d6f76656481ac626c6f63",
    "6b5f6861736865739201cf0123456789abcdefa764705f72616e6b0185a9776f726b6572",
    "5f696407ac73746f726167655f74696572a6646576696365b07265736964656e63795f64",
    "6f6d61696ea6776f726b6572a56576656e7483a86576656e745f696403a464617461a763",
    "6c6561726564a764705f72616e6b00aa73657373696f6e5f6964a5732d303031",
);

fn golden_batch() -> Vec<RouterEvent> {
    vec![
        RouterEvent::new(
            7,
            kv_event(
                1,
                stored(
                    Some(256),
                    Some(16),
                    vec![block(0x0123_4567_89ab_cdef, 17), block(12, 22)],
                ),
                0,
            ),
        ),
        RouterEvent::with_storage_tier(
            7,
            kv_event(2, removed(&[1, 0x0123_4567_89ab_cdef]), 1),
            StorageTier::HostPinned,
        ),
        RouterEvent::new(7, kv_event(3, KvCacheEventData::Cleared, 0)).with_session_id("s-001"),
    ]
}

fn hex(encoded: &str) -> Vec<u8> {
    (0..encoded.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(&encoded[index..index + 2], 16).unwrap())
        .collect()
}

#[test]
fn golden_publisher_bytes_decode_through_typed_walker() {
    let golden = hex(GOLDEN_BATCH);
    assert_eq!(rmp_serde::to_vec_named(&golden_batch()).unwrap(), golden);
    assert_eq!(assert_walks(&golden), golden_batch());
}

// Field subsets emitted by earlier publishers, before `dp_rank`,
// `mm_extra_info`, `storage_tier`, `start_position`, `residency_domain`,
// `state_source`, and `session_id` existed. Field order differs from the
// current types on purpose.
#[derive(Serialize)]
struct V0RouterEvent {
    event: V0KvCacheEvent,
    worker_id: u64,
}

#[derive(Serialize)]
struct V0KvCacheEvent {
    data: V0KvCacheEventData,
    event_id: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum V0KvCacheEventData {
    Stored(V0KvCacheStoreData),
    Removed(KvCacheRemoveData),
    Cleared,
}

#[derive(Serialize)]
struct V0KvCacheStoreData {
    blocks: Vec<V0KvCacheStoredBlockData>,
    parent_hash: Option<u64>,
}

#[derive(Serialize)]
struct V0KvCacheStoredBlockData {
    tokens_hash: u64,
    block_hash: u64,
}

#[test]
fn legacy_publisher_field_subsets_decode_through_typed_walker() {
    let legacy = vec![
        V0RouterEvent {
            worker_id: 5,
            event: V0KvCacheEvent {
                event_id: 1,
                data: V0KvCacheEventData::Stored(V0KvCacheStoreData {
                    parent_hash: Some(3),
                    blocks: vec![V0KvCacheStoredBlockData {
                        block_hash: 4,
                        tokens_hash: 40,
                    }],
                }),
            },
        },
        V0RouterEvent {
            worker_id: 5,
            event: V0KvCacheEvent {
                event_id: 2,
                data: V0KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: vec![ExternalSequenceBlockHash(4)],
                }),
            },
        },
        V0RouterEvent {
            worker_id: 5,
            event: V0KvCacheEvent {
                event_id: 3,
                data: V0KvCacheEventData::Cleared,
            },
        },
    ];
    let decoded = assert_walks(&rmp_serde::to_vec_named(&legacy).unwrap());

    let legacy_event = |event_id, data| {
        let mut event = RouterEvent::new(5, kv_event(event_id, data, 0));
        event.residency_domain = WireResidencyDomain::Missing;
        event
    };
    assert_eq!(
        decoded,
        vec![
            legacy_event(1, stored(Some(3), None, vec![block(4, 40)])),
            legacy_event(2, removed(&[4])),
            legacy_event(3, KvCacheEventData::Cleared),
        ]
    );
}

// Minimal MessagePack builders for shapes serde-derived encoders never emit.
fn enc<T: Serialize>(value: &T) -> Vec<u8> {
    rmp_serde::to_vec_named(value).unwrap()
}

fn map(entries: Vec<(&str, Vec<u8>)>) -> Vec<u8> {
    let mut out = vec![0x80 | u8::try_from(entries.len()).unwrap()];
    for (key, value) in entries {
        out.extend(enc(&key));
        out.extend(value);
    }
    out
}

fn array(items: Vec<Vec<u8>>) -> Vec<u8> {
    let mut out = vec![0x90 | u8::try_from(items.len()).unwrap()];
    out.extend(items.into_iter().flatten());
    out
}

fn block_map(extra: Vec<(&str, Vec<u8>)>) -> Vec<u8> {
    let mut entries = vec![("block_hash", enc(&1u64)), ("tokens_hash", enc(&2u64))];
    entries.extend(extra);
    map(entries)
}

fn store_batch(blocks: Vec<u8>) -> Vec<u8> {
    let data = map(vec![("stored", map(vec![("blocks", blocks)]))]);
    let event = map(vec![("event_id", enc(&1u64)), ("data", data)]);
    array(vec![map(vec![("worker_id", enc(&7u64)), ("event", event)])])
}

fn event_batch(router_event_extra: Vec<(&str, Vec<u8>)>, data: Vec<u8>) -> Vec<u8> {
    let mut entries = vec![
        ("worker_id", enc(&7u64)),
        ("event", map(vec![("event_id", enc(&1u64)), ("data", data)])),
    ];
    entries.extend(router_event_extra);
    array(vec![map(entries)])
}

fn cleared() -> Vec<u8> {
    enc(&"cleared")
}

#[test]
fn non_canonical_encodings_decode_like_serde() {
    let signed = |marker: u8, bytes: &[u8]| [&[marker], bytes].concat();
    let walked = [
        // Signed and widened integer encodings serde accepts for u64 fields.
        store_batch(array(vec![map(vec![
            ("block_hash", signed(0xd3, &5i64.to_be_bytes())),
            ("tokens_hash", signed(0xd0, &[0x7f])),
        ])])),
        store_batch(array(vec![map(vec![
            ("block_hash", signed(0xd2, &9i32.to_be_bytes())),
            ("tokens_hash", signed(0xcf, &3u64.to_be_bytes())),
        ])])),
        // Wide map and array headers for small lengths.
        store_batch(
            [
                &[0xdc, 0x00, 0x01][..],
                &[0xde, 0x00, 0x02],
                &enc(&"block_hash"),
                &[1],
                &enc(&"tokens_hash"),
                &[2],
            ]
            .concat(),
        ),
        // Unknown fields with nested values at every struct level.
        store_batch(array(vec![block_map(vec![
            (
                "future",
                map(vec![("x", array(vec![enc(&1u8), enc(&"y")]))]),
            ),
            ("bin", vec![0xc4, 0x02, 0xff, 0xfe]),
            ("ext", vec![0xd4, 0x01, 0x05]),
        ])])),
        event_batch(
            vec![("future", array(vec![enc(&-1i64), enc(&1.5f64)]))],
            cleared(),
        ),
        // Positional structs, which serde-derived visitors also accept.
        event_batch(
            Vec::new(),
            map(vec![(
                "stored",
                array(vec![
                    enc(&()),
                    enc(&3u32),
                    array(vec![array(vec![enc(&1u64), enc(&2u64), enc(&())])]),
                ]),
            )]),
        ),
        array(vec![map(vec![
            ("worker_id", enc(&7u64)),
            ("event", array(vec![enc(&1u64), cleared()])),
        ])]),
        // Byte-string sequences and enum spellings rmp-serde tolerates.
        event_batch(
            Vec::new(),
            map(vec![(
                "removed",
                map(vec![("block_hashes", vec![0xc4, 0x02, 0x01, 0x02])]),
            )]),
        ),
        event_batch(
            vec![("storage_tier", map(vec![("disk", enc(&()))]))],
            cleared(),
        ),
        event_batch(Vec::new(), map(vec![("cleared", enc(&()))])),
        // Tolerant residency domains.
        event_batch(vec![("residency_domain", enc(&3u8))], cleared()),
        event_batch(vec![("residency_domain", enc(&"other"))], cleared()),
        event_batch(
            vec![("residency_domain", vec![0xa2, 0xff, 0xfe])],
            cleared(),
        ),
        // Trailing bytes are ignored by `from_slice`.
        [
            store_batch(array(vec![block_map(Vec::new())])),
            vec![0xc1, 0xff],
        ]
        .concat(),
    ];
    for bytes in walked {
        assert_walks(&bytes);
        assert!(assert_matches_serde(&bytes));
    }

    // Byte and field-index keys, which only the full serde fallback handles.
    let event = map(vec![("event_id", enc(&1u64)), ("data", cleared())]);
    let fallback_only = [
        array(vec![
            [
                &[0x82, 0xc4, 0x09][..],
                b"worker_id",
                &[7],
                &enc(&"event"),
                &event,
            ]
            .concat(),
        ]),
        array(vec![
            [&[0x82, 0x00, 0x07][..], &enc(&"event"), &event].concat(),
        ]),
    ];
    for bytes in fallback_only {
        assert!(walk(&bytes).is_none());
        assert!(
            assert_matches_serde(&bytes),
            "serde must accept {bytes:02x?}"
        );
    }
}

/// Captures the field or variant names a serde-derived type declares,
/// aliases included, without decoding anything.
struct NameProbe<'a>(&'a mut &'static [&'static str]);

impl<'de> serde::Deserializer<'de> for NameProbe<'_> {
    type Error = serde::de::value::Error;

    fn deserialize_any<V: serde::de::Visitor<'de>>(self, _: V) -> Result<V::Value, Self::Error> {
        Err(serde::de::Error::custom("not a derived struct or enum"))
    }

    fn deserialize_struct<V: serde::de::Visitor<'de>>(
        self,
        _: &'static str,
        fields: &'static [&'static str],
        _: V,
    ) -> Result<V::Value, Self::Error> {
        *self.0 = fields;
        Err(serde::de::Error::custom("probe"))
    }

    fn deserialize_enum<V: serde::de::Visitor<'de>>(
        self,
        _: &'static str,
        variants: &'static [&'static str],
        _: V,
    ) -> Result<V::Value, Self::Error> {
        *self.0 = variants;
        Err(serde::de::Error::custom("probe"))
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
        bytes byte_buf option unit unit_struct newtype_struct seq tuple
        tuple_struct map identifier ignored_any
    }
}

fn serde_names<T: DeserializeOwned>() -> &'static [&'static str] {
    let mut names: &'static [&'static str] = &[];
    let _ = T::deserialize(NameProbe(&mut names));
    names
}

/// The walker matches field and variant names as raw bytes and skips any
/// other key, so a rename or alias on a payload type must be mirrored there.
/// Round-trip tests catch renames but not aliases, which an encoder never
/// emits; without this check an aliased optional field would silently decode
/// as absent.
#[test]
fn walker_names_match_serde_derived_names() {
    let expected: [(&[&str], &[&str]); 7] = [
        (
            serde_names::<RouterEvent>(),
            &[
                "worker_id",
                "storage_tier",
                "residency_domain",
                "event",
                "state_source",
                "session_id",
            ],
        ),
        (
            serde_names::<KvCacheEvent>(),
            &["event_id", "data", "dp_rank"],
        ),
        (
            serde_names::<KvCacheEventData>(),
            &["stored", "removed", "cleared"],
        ),
        (
            serde_names::<KvCacheStoreData>(),
            &["parent_hash", "start_position", "blocks"],
        ),
        (
            serde_names::<KvCacheStoredBlockData>(),
            &["block_hash", "tokens_hash", "mm_extra_info"],
        ),
        (serde_names::<KvCacheRemoveData>(), &["block_hashes"]),
        (
            serde_names::<StorageTier>(),
            &["device", "host_pinned", "disk", "external"],
        ),
    ];
    for (declared, walked) in expected {
        assert_eq!(declared, walked);
    }
}

#[test]
fn rejected_payloads_return_serde_errors() {
    let rejected = [
        Vec::new(),
        vec![0xc1],
        map(Vec::new()),
        store_batch(array(vec![block_map(vec![("block_hash", enc(&3u64))])])),
        store_batch(array(vec![map(vec![("block_hash", enc(&1u64))])])),
        store_batch(array(vec![map(vec![
            ("block_hash", enc(&-1i64)),
            ("tokens_hash", enc(&2u64)),
        ])])),
        store_batch(array(vec![map(vec![
            ("block_hash", enc(&1.0f64)),
            ("tokens_hash", enc(&2u64)),
        ])])),
        store_batch(array(vec![block_map(vec![("mm_extra_info", enc(&1u8))])])),
        store_batch(enc(&"blocks")),
        event_batch(Vec::new(), enc(&"stored")),
        event_batch(Vec::new(), map(vec![("evicted", map(Vec::new()))])),
        event_batch(vec![("storage_tier", enc(&"tape"))], cleared()),
        event_batch(vec![("storage_tier", enc(&()))], cleared()),
        event_batch(vec![("session_id", vec![0xa2, 0xff, 0xfe])], cleared()),
        event_batch(vec![("state_source", enc(&"owner"))], cleared()),
        event_batch(vec![("worker_id", enc(&8u64))], cleared()),
        array(vec![map(vec![
            ("worker_id", enc(&7u64)),
            (
                "event",
                map(vec![
                    ("event_id", enc(&1u64)),
                    ("data", cleared()),
                    ("dp_rank", enc(&(u64::from(u32::MAX) + 1))),
                ]),
            ),
        ])]),
    ];
    for bytes in rejected {
        assert!(
            !assert_matches_serde(&bytes),
            "serde must reject {bytes:02x?}"
        );
    }
}

#[test]
fn truncated_and_mutated_payloads_match_serde() {
    let bytes = rmp_serde::to_vec_named(&every_variant_batch()).unwrap();
    for end in 0..bytes.len() {
        assert_matches_serde(&bytes[..end]);
    }
    let replacements = [
        0x00, 0x01, 0x7f, 0x80, 0x81, 0x90, 0x91, 0xa0, 0xa6, 0xc0, 0xc1, 0xc2, 0xc4, 0xcc, 0xcf,
        0xd0, 0xd3, 0xd9, 0xdc, 0xde, 0xdf, 0xe0, 0xff,
    ];
    let mut mutated = bytes.clone();
    for index in 0..bytes.len() {
        for replacement in replacements.into_iter().chain([bytes[index] ^ 0x01]) {
            mutated[index] = replacement;
            assert_matches_serde(&mutated);
        }
        mutated[index] = bytes[index];
    }
}

#[test]
fn delegated_values_keep_serde_depth_limit() {
    // Budget at a block field value: 1024 minus the batch array, router event,
    // KV event, store data, blocks array, and block maps. The enum variant map
    // is not charged.
    const BLOCK_FIELD_DEPTH: usize = 1018;
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(|| {
            let mut outcomes = Vec::new();
            for nesting in BLOCK_FIELD_DEPTH - 2..=BLOCK_FIELD_DEPTH + 1 {
                let mut nested = vec![0x91; nesting - 1];
                nested.push(0x90);
                let bytes = store_batch(array(vec![block_map(vec![("deep", nested)])]));
                let accepted = assert_matches_serde(&bytes);
                assert_eq!(walk(&bytes).is_some(), accepted, "nesting {nesting}");
                outcomes.push(accepted);
            }
            assert_eq!(outcomes, [true, true, false, false]);
        })
        .unwrap()
        .join()
        .unwrap();
}
