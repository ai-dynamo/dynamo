// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Raw engine lists -> production publisher envelopes, and envelopes -> wire `RouterEvent`s.

use anyhow::{Result, bail, ensure};
use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash, ResidencyDomain, RouterEvent, StorageTier,
};
use dynamo_llm::kv_router::publisher::OfflinePublisherPipeline;
use serde::Serialize;

use crate::plan::salt_hash;
use crate::stream::{BaseStream, EventLists, EventRef};

/// A base after the production publisher pipeline: every list is one event-plane envelope and
/// outbound event IDs run contiguously from 1 through the warm-up and then the timed section.
#[derive(Debug, Clone, Default)]
pub struct ProcessedBase {
    pub warmup: EventLists,
    pub timed: EventLists,
}

impl ProcessedBase {
    pub fn warmup_first_event_id(&self) -> u64 {
        1
    }

    pub fn timed_first_event_id(&self) -> u64 {
        1 + self.warmup.events() as u64
    }
}

/// Events and blocks a run of lists writes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct WriteTotals {
    pub events: u64,
    pub stored_blocks: u64,
    pub removed_blocks: u64,
}

impl WriteTotals {
    pub fn write_blocks(&self) -> u64 {
        self.stored_blocks + self.removed_blocks
    }

    pub fn add(&mut self, other: Self) {
        self.events += other.events;
        self.stored_blocks += other.stored_blocks;
        self.removed_blocks += other.removed_blocks;
    }

    /// Removed over stored blocks; 0 when nothing is stored.
    pub fn remove_ratio(&self) -> f64 {
        if self.stored_blocks == 0 {
            return 0.0;
        }
        self.removed_blocks as f64 / self.stored_blocks as f64
    }

    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "events": self.events,
            "stored_blocks": self.stored_blocks,
            "removed_blocks": self.removed_blocks,
            "write_blocks": self.write_blocks(),
        })
    }
}

/// Totals of the lists `[0, n)` for each `n` in `prefixes`, in one pass over the section.
pub fn prefix_totals(lists: &EventLists, prefixes: &[usize]) -> Vec<WriteTotals> {
    let mut order: Vec<usize> = (0..prefixes.len()).collect();
    order.sort_unstable_by_key(|&index| prefixes[index]);
    let mut out = vec![WriteTotals::default(); prefixes.len()];
    let mut running = WriteTotals::default();
    let mut done = 0;
    for index in order {
        let end = prefixes[index].min(lists.lists());
        for list in done..end {
            for event in lists.list_events(list) {
                running.events += 1;
                match lists.event(event) {
                    EventRef::Store { pairs, .. } => {
                        running.stored_blocks += pairs.len() as u64 / 2
                    }
                    EventRef::Remove { blocks } => running.removed_blocks += blocks.len() as u64,
                    EventRef::Clear => {}
                }
            }
        }
        done = done.max(end);
        out[index] = running;
    }
    out
}

/// Run a base's raw engine lists through [`OfflinePublisherPipeline`] (dedup, coalescing,
/// event-ID assignment, envelope split), exactly as a live worker's publisher would.
pub async fn process_base(raw: &BaseStream) -> Result<ProcessedBase> {
    let mut pipeline = OfflinePublisherPipeline::new(0);
    let mut next_event_id = 1;
    let warmup = process_section(&mut pipeline, &raw.warmup, &mut next_event_id).await?;
    let timed = process_section(&mut pipeline, &raw.timed, &mut next_event_id).await?;
    Ok(ProcessedBase { warmup, timed })
}

async fn process_section(
    pipeline: &mut OfflinePublisherPipeline,
    raw: &EventLists,
    next_event_id: &mut u64,
) -> Result<EventLists> {
    let mut processed = EventLists::default();
    for list in 0..raw.lists() {
        let ts_us = raw.list_ts_us[list];
        let events = raw
            .list_events(list)
            .map(|index| {
                (
                    kv_event(raw.event(index), 0, 0, |hash| hash),
                    StorageTier::Device,
                )
            })
            .collect();
        for envelope in pipeline.process_list(events).await {
            for event in envelope {
                ensure!(
                    event.event.event_id == *next_event_id,
                    "publisher pipeline emitted event {} where {} was expected",
                    event.event.event_id,
                    next_event_id
                );
                *next_event_id += 1;
                push_event(&mut processed, &event.event.data)?;
            }
            processed.close_list(ts_us);
        }
    }
    Ok(processed)
}

fn push_event(lists: &mut EventLists, data: &KvCacheEventData) -> Result<()> {
    match data {
        KvCacheEventData::Stored(store) => {
            if store.start_position.is_some()
                || store
                    .blocks
                    .iter()
                    .any(|block| block.mm_extra_info.is_some())
            {
                bail!("positional or multimodal stores are not supported");
            }
            lists.push_store(
                store.parent_hash.map(|hash| hash.0),
                store
                    .blocks
                    .iter()
                    .map(|block| (block.block_hash.0, block.tokens_hash.0)),
            );
        }
        KvCacheEventData::Removed(remove) => {
            lists.push_remove(remove.block_hashes.iter().map(|hash| hash.0))
        }
        KvCacheEventData::Cleared => lists.push_clear(),
    }
    Ok(())
}

fn kv_event(
    event: EventRef<'_>,
    event_id: u64,
    dp_rank: u32,
    map: impl Fn(u64) -> u64,
) -> KvCacheEvent {
    let data = match event {
        EventRef::Store { parent, pairs } => KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash: parent.map(|hash| ExternalSequenceBlockHash(map(hash))),
            start_position: None,
            blocks: pairs
                .chunks_exact(2)
                .map(|pair| KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(map(pair[0])),
                    tokens_hash: LocalBlockHash(map(pair[1])),
                    mm_extra_info: None,
                })
                .collect(),
        }),
        EventRef::Remove { blocks } => KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: blocks
                .iter()
                .map(|&hash| ExternalSequenceBlockHash(map(hash)))
                .collect(),
        }),
        EventRef::Clear => KvCacheEventData::Cleared,
    };
    KvCacheEvent {
        event_id,
        data,
        dp_rank,
    }
}

/// One envelope's wire payload for a phantom: salted hashes, the phantom's worker ID, and the
/// same `RouterEvent` shape the production publisher's `emit` builds.
pub fn envelope_events(
    lists: &EventLists,
    list: usize,
    section_first_event_id: u64,
    worker_id: u64,
    salt_key: u64,
) -> Vec<RouterEvent> {
    lists
        .list_events(list)
        .map(|index| {
            let event = kv_event(
                lists.event(index),
                section_first_event_id + index as u64,
                0,
                |hash| salt_hash(hash, salt_key),
            );
            RouterEvent::with_residency_domain(
                worker_id,
                event,
                StorageTier::Device,
                ResidencyDomain::Worker,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn processing_coalesces_lists_and_numbers_through_both_sections() {
        let mut raw = BaseStream::default();
        raw.warmup.push_store(None, [(1, 11)]);
        raw.warmup.push_store(Some(1), [(2, 12)]);
        raw.warmup.close_list(0);
        raw.timed.push_remove([2]);
        raw.timed.close_list(100);
        raw.timed.push_store(Some(1), [(3, 13)]);
        raw.timed.close_list(200);

        let processed = process_base(&raw).await.unwrap();
        assert_eq!(processed.warmup.events(), 1, "a chained list coalesces");
        assert_eq!(processed.timed_first_event_id(), 2);
        assert_eq!(processed.timed.list_ts_us, vec![100, 200]);

        let events = envelope_events(&processed.timed, 1, processed.timed_first_event_id(), 77, 5);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].worker_id, 77);
        assert_eq!(events[0].event.event_id, 3);
        let KvCacheEventData::Stored(store) = &events[0].event.data else {
            panic!("expected a store");
        };
        assert_eq!(
            store.parent_hash,
            Some(ExternalSequenceBlockHash(salt_hash(1, 5)))
        );
        assert_eq!(
            store.blocks[0].tokens_hash,
            LocalBlockHash(salt_hash(13, 5))
        );
    }

    #[test]
    fn prefix_totals_cover_unsorted_prefixes_in_one_pass() {
        let mut lists = EventLists::default();
        lists.push_store(None, [(1, 1), (2, 2)]);
        lists.close_list(10);
        lists.push_remove([1]);
        lists.push_store(Some(2), [(3, 3)]);
        lists.close_list(20);
        lists.push_clear();
        lists.close_list(30);

        let totals = prefix_totals(&lists, &[3, 0, 1, 9]);
        assert_eq!(totals[1], WriteTotals::default());
        assert_eq!(
            totals[2],
            WriteTotals {
                events: 1,
                stored_blocks: 2,
                removed_blocks: 0,
            }
        );
        let all = WriteTotals {
            events: 4,
            stored_blocks: 3,
            removed_blocks: 1,
        };
        assert_eq!((totals[0], totals[3]), (all, all));
        assert_eq!(all.write_blocks(), 4);
    }
}
