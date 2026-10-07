// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed MessagePack decoding for event-plane `Vec<RouterEvent>` batches.
//!
//! Publishers encode batches with `rmp_serde::to_vec_named`, so every struct is
//! a string-keyed map. Through serde, rmp-serde validates every map key as
//! UTF-8 and routes every struct, key, and integer through its generic
//! marker dispatch, which is most of the decode cost of multi-block store
//! events: each stored block carries three keys.
//!
//! [`decode_router_event_batch`] walks that canonical named-map shape
//! directly. A value in any other shape is handed to rmp-serde at the same
//! nesting depth, and anything else the walker does not recognize, including
//! every malformed payload, falls back to a full `rmp_serde::from_slice`. The
//! result is therefore identical to `rmp_serde::from_slice::<Vec<RouterEvent>>`
//! for every input, including its errors.
//!
//! The walker hardcodes the payload types' serde field and variant names, so
//! a rename or alias there must be mirrored here; the tests pin both sets.

use serde::de::{DeserializeOwned, IgnoredAny};

use crate::identity::CacheOwnerId;
use crate::protocols::{
    BlockExtraInfo, ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData,
    KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash, ResidencyDomain, RouterEvent,
    StorageTier, WireResidencyDomain,
};

/// rmp-serde's default nesting budget for `from_slice`.
const MAX_DEPTH: u16 = 1024;
/// Upper bound on speculative preallocation, matching serde's cautious size hint.
const MAX_PREALLOC_BYTES: usize = 1024 * 1024;

const NIL: u8 = 0xc0;
const FIXMAP_1: u8 = 0x81;

/// Decode a MessagePack `Vec<RouterEvent>` event-plane payload.
///
/// Accepts exactly the inputs `rmp_serde::from_slice::<Vec<RouterEvent>>`
/// accepts and returns the same values and errors.
pub fn decode_router_event_batch(
    payload: &[u8],
) -> Result<Vec<RouterEvent>, rmp_serde::decode::Error> {
    if let Some(events) = Walker::new(payload).router_events() {
        return Ok(events);
    }
    rmp_serde::from_slice(payload)
}

/// Remaining rmp-serde depth budget after entering one array or map.
///
/// rmp-serde decrements before descending and rejects a budget of zero.
fn enter(depth: u16) -> Option<u16> {
    depth.checked_sub(1).filter(|&depth| depth != 0)
}

/// Store a struct field, rejecting duplicates as serde-derived structs do.
fn fill<T>(slot: &mut Option<T>, value: T) -> Option<()> {
    if slot.is_some() {
        return None;
    }
    *slot = Some(value);
    Some(())
}

fn prealloc<T>(len: usize, remaining_bytes: usize) -> Vec<T> {
    let max_elems = MAX_PREALLOC_BYTES / std::mem::size_of::<T>().max(1);
    Vec::with_capacity(len.min(remaining_bytes).min(max_elems))
}

/// Cursor over a MessagePack buffer.
///
/// Every method returns `None` when the next value is not in the expected
/// shape. Header readers leave the cursor untouched on `None` so the caller
/// can hand the value to rmp-serde; any other `None` aborts the walk.
struct Walker<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Walker<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    fn peek(&self) -> Option<u8> {
        self.buf.get(self.pos).copied()
    }

    fn be<const N: usize>(&self, offset: usize) -> Option<[u8; N]> {
        let start = self.pos.checked_add(offset)?;
        self.buf.get(start..start.checked_add(N)?)?.try_into().ok()
    }

    /// Consume a `header`-byte prefix followed by `len` payload bytes.
    fn take(&mut self, header: usize, len: usize) -> Option<&'a [u8]> {
        let start = self.pos.checked_add(header)?;
        let end = start.checked_add(len)?;
        let bytes = self.buf.get(start..end)?;
        self.pos = end;
        Some(bytes)
    }

    fn skip_nil(&mut self) -> bool {
        if self.peek() != Some(NIL) {
            return false;
        }
        self.pos += 1;
        true
    }

    fn map_len(&mut self) -> Option<usize> {
        let (header, len) = match self.peek()? {
            marker @ 0x80..=0x8f => (1, usize::from(marker & 0x0f)),
            0xde => (3, usize::from(u16::from_be_bytes(self.be(1)?))),
            0xdf => (5, u32::from_be_bytes(self.be(1)?) as usize),
            _ => return None,
        };
        self.pos += header;
        Some(len)
    }

    fn array_len(&mut self) -> Option<usize> {
        let (header, len) = match self.peek()? {
            marker @ 0x90..=0x9f => (1, usize::from(marker & 0x0f)),
            0xdc => (3, usize::from(u16::from_be_bytes(self.be(1)?))),
            0xdd => (5, u32::from_be_bytes(self.be(1)?) as usize),
            _ => return None,
        };
        self.pos += header;
        Some(len)
    }

    /// Raw bytes of a MessagePack string, without UTF-8 validation.
    ///
    /// Field and variant names are ASCII, so comparing raw bytes matches
    /// rmp-serde, which treats an invalid UTF-8 key as an unknown byte key.
    fn str_bytes(&mut self) -> Option<&'a [u8]> {
        match self.peek()? {
            marker @ 0xa0..=0xbf => self.take(1, usize::from(marker & 0x1f)),
            0xd9 => self.take(2, usize::from(self.be::<1>(1)?[0])),
            0xda => self.take(3, usize::from(u16::from_be_bytes(self.be(1)?))),
            0xdb => self.take(5, u32::from_be_bytes(self.be(1)?) as usize),
            _ => None,
        }
    }

    /// An integer in any MessagePack encoding that serde accepts as `u64`.
    fn uint(&mut self) -> Option<u64> {
        let marker = self.peek()?;
        let (header, value) = match marker {
            0x00..=0x7f => (0, u64::from(marker)),
            0xcc => (1, u64::from(self.be::<1>(1)?[0])),
            0xcd => (2, u64::from(u16::from_be_bytes(self.be(1)?))),
            0xce => (4, u64::from(u32::from_be_bytes(self.be(1)?))),
            0xcf => (8, u64::from_be_bytes(self.be(1)?)),
            0xd0 => (1, u64::try_from(i8::from_be_bytes(self.be(1)?)).ok()?),
            0xd1 => (2, u64::try_from(i16::from_be_bytes(self.be(1)?)).ok()?),
            0xd2 => (4, u64::try_from(i32::from_be_bytes(self.be(1)?)).ok()?),
            0xd3 => (8, u64::try_from(i64::from_be_bytes(self.be(1)?)).ok()?),
            _ => return None,
        };
        self.pos += 1 + header;
        Some(value)
    }

    fn u32(&mut self) -> Option<u32> {
        u32::try_from(self.uint()?).ok()
    }

    /// Decode the value at the cursor with rmp-serde under the depth budget
    /// the full decoder would have at this position.
    fn delegate<T: DeserializeOwned>(&mut self, depth: u16) -> Option<T> {
        let mut rest = &self.buf[self.pos..];
        let before = rest.len();
        let mut deserializer = rmp_serde::Deserializer::new(&mut rest);
        deserializer.set_max_depth(depth.into());
        let value = T::deserialize(&mut deserializer).ok()?;
        drop(deserializer);
        self.pos += before - rest.len();
        Some(value)
    }

    fn skip_value(&mut self, depth: u16) -> Option<()> {
        self.delegate::<IgnoredAny>(depth).map(drop)
    }

    fn nullable<T: DeserializeOwned>(&mut self, depth: u16) -> Option<Option<T>> {
        if self.skip_nil() {
            return Some(None);
        }
        self.delegate(depth)
    }

    fn router_events(&mut self) -> Option<Vec<RouterEvent>> {
        let len = self.array_len()?;
        let depth = enter(MAX_DEPTH)?;
        let mut events = prealloc(len, self.remaining());
        for _ in 0..len {
            events.push(self.router_event(depth)?);
        }
        Some(events)
    }

    fn router_event(&mut self, depth: u16) -> Option<RouterEvent> {
        let Some(len) = self.map_len() else {
            return self.delegate(depth);
        };
        let depth = enter(depth)?;
        let mut worker_id = None;
        let mut storage_tier = None;
        let mut residency_domain = None;
        let mut event = None;
        let mut state_source = None;
        let mut session_id = None;
        for _ in 0..len {
            match self.str_bytes()? {
                b"worker_id" => fill(&mut worker_id, self.uint()?)?,
                b"storage_tier" => fill(&mut storage_tier, self.storage_tier(depth)?)?,
                b"residency_domain" => fill(&mut residency_domain, self.residency_domain(depth)?)?,
                b"event" => fill(&mut event, self.kv_cache_event(depth)?)?,
                b"state_source" => fill(&mut state_source, self.nullable::<CacheOwnerId>(depth)?)?,
                b"session_id" => fill(&mut session_id, self.session_id(depth)?)?,
                _ => self.skip_value(depth)?,
            }
        }
        Some(RouterEvent {
            worker_id: worker_id?,
            storage_tier: storage_tier.unwrap_or_default(),
            residency_domain: residency_domain.unwrap_or_default(),
            event: event?,
            state_source: state_source.flatten(),
            session_id: session_id.flatten(),
        })
    }

    fn storage_tier(&mut self, depth: u16) -> Option<StorageTier> {
        let start = self.pos;
        let tier = match self.str_bytes() {
            Some(b"device") => StorageTier::Device,
            Some(b"host_pinned") => StorageTier::HostPinned,
            Some(b"disk") => StorageTier::Disk,
            Some(b"external") => StorageTier::External,
            _ => {
                self.pos = start;
                return self.delegate(depth);
            }
        };
        Some(tier)
    }

    fn residency_domain(&mut self, depth: u16) -> Option<WireResidencyDomain> {
        let start = self.pos;
        let domain = match self.str_bytes() {
            Some(b"worker") => ResidencyDomain::Worker,
            Some(b"cache_owner") => ResidencyDomain::CacheOwner,
            _ => {
                self.pos = start;
                return self.delegate(depth);
            }
        };
        Some(WireResidencyDomain::Known(domain))
    }

    fn session_id(&mut self, depth: u16) -> Option<Option<String>> {
        let start = self.pos;
        if let Some(Ok(session_id)) = self.str_bytes().map(std::str::from_utf8) {
            return Some(Some(session_id.to_owned()));
        }
        self.pos = start;
        self.nullable(depth)
    }

    fn kv_cache_event(&mut self, depth: u16) -> Option<KvCacheEvent> {
        let Some(len) = self.map_len() else {
            return self.delegate(depth);
        };
        let depth = enter(depth)?;
        let mut event_id = None;
        let mut data = None;
        let mut dp_rank = None;
        for _ in 0..len {
            match self.str_bytes()? {
                b"event_id" => fill(&mut event_id, self.uint()?)?,
                b"data" => fill(&mut data, self.event_data(depth)?)?,
                b"dp_rank" => fill(&mut dp_rank, self.u32()?)?,
                _ => self.skip_value(depth)?,
            }
        }
        Some(KvCacheEvent {
            event_id: event_id?,
            data: data?,
            dp_rank: dp_rank.unwrap_or_default(),
        })
    }

    /// Externally tagged enum: `{"stored": {..}}`, `{"removed": {..}}`, or `"cleared"`.
    ///
    /// rmp-serde does not charge depth for the single-entry variant map.
    fn event_data(&mut self, depth: u16) -> Option<KvCacheEventData> {
        let start = self.pos;
        if self.peek()? == FIXMAP_1 {
            self.pos += 1;
            match self.str_bytes() {
                Some(b"stored") => return self.store_data(depth).map(KvCacheEventData::Stored),
                Some(b"removed") => return self.remove_data(depth).map(KvCacheEventData::Removed),
                _ => {}
            }
        } else if matches!(self.str_bytes(), Some(b"cleared")) {
            return Some(KvCacheEventData::Cleared);
        }
        self.pos = start;
        self.delegate(depth)
    }

    fn store_data(&mut self, depth: u16) -> Option<KvCacheStoreData> {
        let Some(len) = self.map_len() else {
            return self.delegate(depth);
        };
        let depth = enter(depth)?;
        let mut parent_hash = None;
        let mut start_position = None;
        let mut blocks = None;
        for _ in 0..len {
            match self.str_bytes()? {
                b"parent_hash" => {
                    let value = if self.skip_nil() {
                        None
                    } else {
                        Some(ExternalSequenceBlockHash(self.uint()?))
                    };
                    fill(&mut parent_hash, value)?
                }
                b"start_position" => {
                    let value = if self.skip_nil() {
                        None
                    } else {
                        Some(self.u32()?)
                    };
                    fill(&mut start_position, value)?
                }
                b"blocks" => fill(&mut blocks, self.stored_blocks(depth)?)?,
                _ => self.skip_value(depth)?,
            }
        }
        Some(KvCacheStoreData {
            parent_hash: parent_hash.flatten(),
            start_position: start_position.flatten(),
            blocks: blocks?,
        })
    }

    fn stored_blocks(&mut self, depth: u16) -> Option<Vec<KvCacheStoredBlockData>> {
        let Some(len) = self.array_len() else {
            return self.delegate(depth);
        };
        let depth = enter(depth)?;
        let mut blocks = prealloc(len, self.remaining());
        for _ in 0..len {
            blocks.push(self.stored_block(depth)?);
        }
        Some(blocks)
    }

    fn stored_block(&mut self, depth: u16) -> Option<KvCacheStoredBlockData> {
        let Some(len) = self.map_len() else {
            return self.delegate(depth);
        };
        let depth = enter(depth)?;
        let mut block_hash = None;
        let mut tokens_hash = None;
        let mut mm_extra_info = None;
        for _ in 0..len {
            match self.str_bytes()? {
                b"block_hash" => fill(&mut block_hash, self.uint()?)?,
                b"tokens_hash" => fill(&mut tokens_hash, self.uint()?)?,
                b"mm_extra_info" => {
                    fill(&mut mm_extra_info, self.nullable::<BlockExtraInfo>(depth)?)?
                }
                _ => self.skip_value(depth)?,
            }
        }
        Some(KvCacheStoredBlockData {
            block_hash: ExternalSequenceBlockHash(block_hash?),
            tokens_hash: LocalBlockHash(tokens_hash?),
            mm_extra_info: mm_extra_info.flatten(),
        })
    }

    fn remove_data(&mut self, depth: u16) -> Option<KvCacheRemoveData> {
        let Some(len) = self.map_len() else {
            return self.delegate(depth);
        };
        let depth = enter(depth)?;
        let mut block_hashes = None;
        for _ in 0..len {
            match self.str_bytes()? {
                b"block_hashes" => fill(&mut block_hashes, self.block_hashes(depth)?)?,
                _ => self.skip_value(depth)?,
            }
        }
        Some(KvCacheRemoveData {
            block_hashes: block_hashes?,
        })
    }

    fn block_hashes(&mut self, depth: u16) -> Option<Vec<ExternalSequenceBlockHash>> {
        let Some(len) = self.array_len() else {
            return self.delegate(depth);
        };
        enter(depth)?;
        let mut hashes = prealloc(len, self.remaining());
        for _ in 0..len {
            hashes.push(ExternalSequenceBlockHash(self.uint()?));
        }
        Some(hashes)
    }
}

#[cfg(test)]
mod tests;
