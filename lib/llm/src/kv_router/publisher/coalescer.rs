// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail};
use dynamo_kv_router::protocols::{
    BlockExtraInfo, BlockHashOptions, BlockMmObjectInfo, DpRank, ExternalSequenceBlockHash,
    KvCacheEventData, KvCacheRemoveData, KvCacheStoreBlockInput, KvCacheStoreData,
    KvCacheStoreInput, KvCacheStoredBlockData, StorageTier, compute_block_hash_for_seq,
};

#[derive(Debug, Clone)]
struct SourceNode {
    parent: Option<ExternalSequenceBlockHash>,
    depth: usize,
    source_position: Option<u32>,
    block: KvCacheStoredBlockData,
    input: KvCacheStoreBlockInput,
    present_tiers: HashSet<StorageTier>,
    children: HashSet<ExternalSequenceBlockHash>,
    superblocks: HashSet<ExternalSequenceBlockHash>,
}

#[derive(Debug, Clone)]
struct Superblock {
    parent: Option<ExternalSequenceBlockHash>,
    start_position: Option<u32>,
    depth: usize,
    members: Vec<ExternalSequenceBlockHash>,
    block: KvCacheStoredBlockData,
    active_tiers: HashSet<StorageTier>,
}

#[derive(Debug, Default)]
struct RankState {
    nodes: HashMap<ExternalSequenceBlockHash, SourceNode>,
    superblocks: HashMap<ExternalSequenceBlockHash, Superblock>,
}

/// Coalesces engine-sized source blocks into larger router-visible blocks.
///
/// Source events have already been batched and source-hash removals have
/// already passed through the publisher's refcounting deduplicator. The
/// coalescer therefore treats repeated stores as idempotent and any removal as
/// the final removal of that source placement.
#[derive(Debug)]
pub(super) struct EventCoalescer {
    source_block_size: u32,
    target_block_size: u32,
    ratio: usize,
    ranks: HashMap<DpRank, RankState>,
}

impl EventCoalescer {
    pub(super) fn new(source_block_size: u32, target_block_size: u32) -> Result<Self> {
        if source_block_size == 0 {
            bail!("KV event source block size must be greater than zero");
        }
        if target_block_size < source_block_size {
            bail!(
                "KV event coalescing block size ({target_block_size}) cannot be smaller than the source block size ({source_block_size})"
            );
        }
        if target_block_size % source_block_size != 0 {
            bail!(
                "KV event coalescing block size ({target_block_size}) must be a multiple of the source block size ({source_block_size})"
            );
        }
        Ok(Self {
            source_block_size,
            target_block_size,
            ratio: (target_block_size / source_block_size) as usize,
            ranks: HashMap::new(),
        })
    }

    pub(super) fn enabled(&self) -> bool {
        self.ratio > 1
    }

    pub(super) fn process(
        &mut self,
        dp_rank: DpRank,
        tier: StorageTier,
        data: KvCacheEventData,
        store_input: Option<KvCacheStoreInput>,
    ) -> Vec<KvCacheEventData> {
        if !self.enabled() {
            return vec![data];
        }

        match data {
            KvCacheEventData::Stored(store) => {
                let Some(input) = store_input else {
                    tracing::warn!(
                        dp_rank,
                        ?tier,
                        "Dropping KV store event: coalescing requires source token context"
                    );
                    return self.fail_closed_store(dp_rank, tier, &store);
                };
                self.process_store(dp_rank, tier, store, input)
            }
            KvCacheEventData::Removed(remove) => self.process_remove(dp_rank, tier, remove),
            KvCacheEventData::Cleared => {
                self.ranks.clear();
                vec![KvCacheEventData::Cleared]
            }
        }
    }

    fn process_store(
        &mut self,
        dp_rank: DpRank,
        tier: StorageTier,
        store: KvCacheStoreData,
        input: KvCacheStoreInput,
    ) -> Vec<KvCacheEventData> {
        if store.blocks.len() != input.blocks.len() {
            tracing::warn!(
                dp_rank,
                ?tier,
                blocks = store.blocks.len(),
                contexts = input.blocks.len(),
                "Dropping KV store event: block/token context length mismatch"
            );
            return self.fail_closed_store(dp_rank, tier, &store);
        }

        let rank = self.ranks.entry(dp_rank).or_default();
        let KvCacheStoreData {
            parent_hash,
            start_position,
            blocks,
        } = store;
        let mut parent = parent_hash;
        let mut candidates = HashSet::new();
        let mut chain_valid = true;
        let mut failed_hashes = Vec::new();

        let pairs: Vec<_> = blocks.into_iter().zip(input.blocks.into_iter()).collect();
        for (index, (block, block_input)) in pairs.iter().cloned().enumerate() {
            if !chain_valid {
                break;
            }
            let source_position =
                start_position.and_then(|position| position.checked_add(index as u32));
            let depth = if let Some(parent_hash) = parent {
                let Some(parent_node) = rank.nodes.get(&parent_hash) else {
                    tracing::warn!(
                        dp_rank,
                        ?tier,
                        block_hash = block.block_hash.0,
                        parent_hash = parent_hash.0,
                        "Dropping KV store chain: coalescer does not know the parent block"
                    );
                    failed_hashes.extend(pairs[index..].iter().map(|(block, _)| block.block_hash));
                    break;
                };
                parent_node.depth + 1
            } else if let Some(position) = source_position {
                position as usize + 1
            } else {
                1
            };

            let block_hash = block.block_hash;
            if let Some(existing) = rank.nodes.get_mut(&block_hash) {
                if existing.parent != parent
                    || existing.depth != depth
                    || existing.source_position != source_position
                    || existing.block != block
                    || existing.input != block_input
                {
                    tracing::warn!(
                        dp_rank,
                        ?tier,
                        block_hash = block_hash.0,
                        "Dropping conflicting duplicate KV source block"
                    );
                    chain_valid = false;
                    failed_hashes.extend(pairs[index..].iter().map(|(block, _)| block.block_hash));
                    continue;
                }
                existing.present_tiers.insert(tier);
                candidates.extend(existing.superblocks.iter().copied());
            } else {
                rank.nodes.insert(
                    block_hash,
                    SourceNode {
                        parent,
                        depth,
                        source_position,
                        block,
                        input: block_input,
                        present_tiers: HashSet::from([tier]),
                        children: HashSet::new(),
                        superblocks: HashSet::new(),
                    },
                );
                if let Some(parent_hash) = parent
                    && let Some(parent_node) = rank.nodes.get_mut(&parent_hash)
                {
                    parent_node.children.insert(block_hash);
                }
            }

            if depth % self.ratio == 0 && !rank.superblocks.contains_key(&block_hash) {
                match build_superblock(
                    rank,
                    block_hash,
                    self.source_block_size,
                    self.target_block_size,
                    self.ratio,
                ) {
                    Ok(superblock) => {
                        for member in &superblock.members {
                            if let Some(node) = rank.nodes.get_mut(member) {
                                node.superblocks.insert(block_hash);
                            }
                        }
                        rank.superblocks.insert(block_hash, superblock);
                    }
                    Err(error) => {
                        tracing::warn!(
                            dp_rank,
                            ?tier,
                            block_hash = block_hash.0,
                            %error,
                            "Failed to build coalesced KV block"
                        );
                        failed_hashes.push(block_hash);
                    }
                }
            }
            if let Some(node) = rank.nodes.get(&block_hash) {
                candidates.extend(node.superblocks.iter().copied());
            }
            parent = Some(block_hash);
        }

        let failed_removal = deactivate_sources(rank, tier, &failed_hashes);

        let mut candidates: Vec<_> = candidates.into_iter().collect();
        candidates.sort_unstable_by_key(|hash| {
            (
                rank.superblocks
                    .get(hash)
                    .map(|superblock| superblock.depth)
                    .unwrap_or(usize::MAX),
                hash.0,
            )
        });

        let mut output = Vec::new();
        if let Some(removal) = failed_removal {
            output.push(KvCacheEventData::Removed(removal));
        }
        for endpoint in candidates {
            let should_activate = rank.superblocks.get(&endpoint).is_some_and(|superblock| {
                !superblock.active_tiers.contains(&tier)
                    && superblock.members.iter().all(|member| {
                        rank.nodes
                            .get(member)
                            .is_some_and(|node| node.present_tiers.contains(&tier))
                    })
            });
            if !should_activate {
                continue;
            }
            let superblock = rank.superblocks.get_mut(&endpoint).unwrap();
            superblock.active_tiers.insert(tier);
            output.push(KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: superblock.parent,
                start_position: superblock.start_position,
                blocks: vec![superblock.block.clone()],
            }));
        }
        output
    }

    fn fail_closed_store(
        &mut self,
        dp_rank: DpRank,
        tier: StorageTier,
        store: &KvCacheStoreData,
    ) -> Vec<KvCacheEventData> {
        let Some(rank) = self.ranks.get_mut(&dp_rank) else {
            return Vec::new();
        };
        let hashes: Vec<_> = store.blocks.iter().map(|block| block.block_hash).collect();
        deactivate_sources(rank, tier, &hashes)
            .map(KvCacheEventData::Removed)
            .into_iter()
            .collect()
    }

    fn process_remove(
        &mut self,
        dp_rank: DpRank,
        tier: StorageTier,
        remove: KvCacheRemoveData,
    ) -> Vec<KvCacheEventData> {
        let Some(rank) = self.ranks.get_mut(&dp_rank) else {
            return Vec::new();
        };
        let mut removed_superblocks = HashSet::new();
        let mut gc_candidates = Vec::new();

        for source_hash in remove.block_hashes {
            let memberships = match rank.nodes.get_mut(&source_hash) {
                Some(node) => {
                    if !node.present_tiers.remove(&tier) {
                        continue;
                    }
                    node.superblocks.iter().copied().collect::<Vec<_>>()
                }
                None => continue,
            };
            for endpoint in memberships {
                if rank
                    .superblocks
                    .get_mut(&endpoint)
                    .is_some_and(|superblock| superblock.active_tiers.remove(&tier))
                {
                    removed_superblocks.insert(endpoint);
                }
            }
            gc_candidates.push(source_hash);
        }

        for source_hash in gc_candidates {
            gc_from(rank, source_hash);
        }

        if removed_superblocks.is_empty() {
            Vec::new()
        } else {
            let mut block_hashes: Vec<_> = removed_superblocks.into_iter().collect();
            block_hashes.sort_unstable_by_key(|hash| hash.0);
            vec![KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes,
            })]
        }
    }
}

fn deactivate_sources(
    rank: &mut RankState,
    tier: StorageTier,
    source_hashes: &[ExternalSequenceBlockHash],
) -> Option<KvCacheRemoveData> {
    let mut removed_superblocks = HashSet::new();
    for source_hash in source_hashes {
        let memberships = match rank.nodes.get_mut(source_hash) {
            Some(node) => {
                node.present_tiers.remove(&tier);
                node.superblocks.iter().copied().collect::<Vec<_>>()
            }
            None => continue,
        };
        for endpoint in memberships {
            if rank
                .superblocks
                .get_mut(&endpoint)
                .is_some_and(|superblock| superblock.active_tiers.remove(&tier))
            {
                removed_superblocks.insert(endpoint);
            }
        }
    }
    if removed_superblocks.is_empty() {
        None
    } else {
        let mut block_hashes: Vec<_> = removed_superblocks.into_iter().collect();
        block_hashes.sort_unstable_by_key(|hash| hash.0);
        Some(KvCacheRemoveData { block_hashes })
    }
}

fn build_superblock(
    rank: &RankState,
    endpoint: ExternalSequenceBlockHash,
    source_block_size: u32,
    target_block_size: u32,
    ratio: usize,
) -> Result<Superblock> {
    let mut members = Vec::with_capacity(ratio);
    let mut cursor = Some(endpoint);
    for _ in 0..ratio {
        let Some(hash) = cursor else {
            bail!("source chain ended before a complete superblock was available");
        };
        let Some(node) = rank.nodes.get(&hash) else {
            bail!("source chain contains an unknown block");
        };
        members.push(hash);
        cursor = node.parent;
    }
    members.reverse();

    let nodes: Vec<_> = members
        .iter()
        .map(|hash| rank.nodes.get(hash).unwrap())
        .collect();
    let first = nodes[0];
    let lora_name = first.input.lora_name.as_deref();
    let cache_namespace = first.input.cache_namespace.as_deref();
    let is_eagle = first.input.is_eagle;
    if nodes.iter().any(|node| {
        node.input.lora_name.as_deref() != lora_name
            || node.input.cache_namespace.as_deref() != cache_namespace
            || node.input.is_eagle != is_eagle
    }) {
        bail!("source blocks in one superblock have incompatible hashing context");
    }

    let expected = source_block_size as usize + usize::from(is_eagle);
    if nodes
        .iter()
        .any(|node| node.input.token_ids.len() != expected)
    {
        bail!("source block token count does not match the configured source block size");
    }

    let mut tokens = Vec::with_capacity(target_block_size as usize + usize::from(is_eagle));
    for (index, node) in nodes.iter().enumerate() {
        if is_eagle && index > 0 {
            if tokens.last() != node.input.token_ids.first() {
                bail!("adjacent EAGLE source blocks do not share their boundary token");
            }
            tokens.extend_from_slice(&node.input.token_ids[1..]);
        } else {
            tokens.extend_from_slice(&node.input.token_ids);
        }
    }

    let hash_mm_extra_info = merge_mm_info(
        nodes
            .iter()
            .map(|node| node.input.hash_mm_extra_info.as_ref()),
        source_block_size as usize,
    );
    let block_mm_infos = hash_mm_extra_info
        .as_ref()
        .map(|info| vec![Some(info.clone())]);
    let hashes = compute_block_hash_for_seq(
        &tokens,
        target_block_size,
        BlockHashOptions {
            block_mm_infos: block_mm_infos.as_deref(),
            lora_name,
            cache_namespace,
            is_eagle: Some(is_eagle),
        },
    );
    let Some(tokens_hash) = hashes.first().copied() else {
        bail!("coalesced token sequence did not produce a target block hash");
    };

    let published_mm_extra_info = merge_mm_info(
        nodes.iter().map(|node| node.block.mm_extra_info.as_ref()),
        source_block_size as usize,
    );
    let start_position = first
        .source_position
        .map(|position| position / ratio as u32);

    Ok(Superblock {
        parent: cursor,
        start_position,
        depth: nodes.last().unwrap().depth,
        members,
        block: KvCacheStoredBlockData {
            block_hash: endpoint,
            tokens_hash,
            mm_extra_info: published_mm_extra_info,
        },
        active_tiers: HashSet::new(),
    })
}

fn merge_mm_info<'a>(
    infos: impl Iterator<Item = Option<&'a BlockExtraInfo>>,
    source_block_size: usize,
) -> Option<BlockExtraInfo> {
    let mut merged: Vec<BlockMmObjectInfo> = Vec::new();
    for (block_index, info) in infos.enumerate() {
        let Some(info) = info else {
            continue;
        };
        let offset = block_index * source_block_size;
        for object in &info.mm_objects {
            let shifted: Vec<_> = object
                .offsets
                .iter()
                .map(|(start, end)| (start + offset, end + offset))
                .collect();
            if let Some(existing) = merged
                .iter_mut()
                .find(|candidate| candidate.mm_hash == object.mm_hash)
            {
                existing.offsets.extend(shifted);
            } else {
                merged.push(BlockMmObjectInfo {
                    mm_hash: object.mm_hash,
                    offsets: shifted,
                });
            }
        }
    }
    (!merged.is_empty()).then_some(BlockExtraInfo { mm_objects: merged })
}

fn gc_from(rank: &mut RankState, mut source_hash: ExternalSequenceBlockHash) {
    loop {
        let eligible = rank
            .nodes
            .get(&source_hash)
            .is_some_and(|node| node.present_tiers.is_empty() && node.children.is_empty());
        if !eligible {
            return;
        }
        let node = rank.nodes.remove(&source_hash).unwrap();
        for endpoint in node.superblocks {
            let removable = rank
                .superblocks
                .get(&endpoint)
                .is_some_and(|superblock| superblock.active_tiers.is_empty());
            if !removable {
                continue;
            }
            if let Some(superblock) = rank.superblocks.remove(&endpoint) {
                for member in superblock.members {
                    if let Some(member_node) = rank.nodes.get_mut(&member) {
                        member_node.superblocks.remove(&endpoint);
                    }
                }
            }
        }
        let Some(parent) = node.parent else {
            return;
        };
        if let Some(parent_node) = rank.nodes.get_mut(&parent) {
            parent_node.children.remove(&source_hash);
        }
        source_hash = parent;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::protocols::{KvCacheStoreBlockInput, LocalBlockHash};

    fn block(hash: u64, token: u32) -> (KvCacheStoredBlockData, KvCacheStoreBlockInput) {
        (
            KvCacheStoredBlockData {
                block_hash: ExternalSequenceBlockHash(hash),
                tokens_hash: LocalBlockHash(token as u64),
                mm_extra_info: None,
            },
            KvCacheStoreBlockInput {
                token_ids: vec![token],
                lora_name: None,
                cache_namespace: None,
                is_eagle: false,
                hash_mm_extra_info: None,
            },
        )
    }

    fn block_with_tokens(
        hash: u64,
        token_ids: Vec<u32>,
        is_eagle: bool,
    ) -> (KvCacheStoredBlockData, KvCacheStoreBlockInput) {
        (
            KvCacheStoredBlockData {
                block_hash: ExternalSequenceBlockHash(hash),
                tokens_hash: LocalBlockHash(hash),
                mm_extra_info: None,
            },
            KvCacheStoreBlockInput {
                token_ids,
                lora_name: None,
                cache_namespace: None,
                is_eagle,
                hash_mm_extra_info: None,
            },
        )
    }

    fn store(parent: Option<u64>, values: &[(u64, u32)]) -> (KvCacheEventData, KvCacheStoreInput) {
        let (blocks, inputs): (Vec<_>, Vec<_>) = values
            .iter()
            .map(|(hash, token)| block(*hash, *token))
            .unzip();
        (
            KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: parent.map(ExternalSequenceBlockHash),
                start_position: None,
                blocks,
            }),
            KvCacheStoreInput { blocks: inputs },
        )
    }

    fn stored_hashes(output: &[KvCacheEventData]) -> Vec<u64> {
        output
            .iter()
            .filter_map(|data| match data {
                KvCacheEventData::Stored(store) => Some(store.blocks[0].block_hash.0),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn emits_when_group_completes_and_removes_on_any_member() {
        let mut coalescer = EventCoalescer::new(1, 4).unwrap();
        let (data, input) = store(None, &[(1, 10), (2, 20), (3, 30)]);
        assert!(
            coalescer
                .process(0, StorageTier::Device, data, Some(input))
                .is_empty()
        );
        let (data, input) = store(Some(3), &[(4, 40)]);
        let output = coalescer.process(0, StorageTier::Device, data, Some(input));
        assert_eq!(stored_hashes(&output), vec![4]);

        let output = coalescer.process(
            0,
            StorageTier::Device,
            KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![ExternalSequenceBlockHash(2)],
            }),
            None,
        );
        assert_eq!(
            output,
            vec![KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![ExternalSequenceBlockHash(4)]
            })]
        );
    }

    #[test]
    fn restoring_member_reemits_superblock() {
        let mut coalescer = EventCoalescer::new(1, 4).unwrap();
        let (data, input) = store(None, &[(1, 10), (2, 20), (3, 30), (4, 40)]);
        assert_eq!(
            stored_hashes(&coalescer.process(0, StorageTier::Device, data, Some(input))),
            vec![4]
        );
        coalescer.process(
            0,
            StorageTier::Device,
            KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![ExternalSequenceBlockHash(2)],
            }),
            None,
        );
        let (data, input) = store(Some(1), &[(2, 20)]);
        assert_eq!(
            stored_hashes(&coalescer.process(0, StorageTier::Device, data, Some(input))),
            vec![4]
        );
    }

    #[test]
    fn shared_prefix_removal_invalidates_both_branches() {
        let mut coalescer = EventCoalescer::new(1, 4).unwrap();
        let (data, input) = store(None, &[(1, 10), (2, 20), (3, 30), (4, 40)]);
        coalescer.process(0, StorageTier::Device, data, Some(input));
        let (data, input) = store(Some(2), &[(5, 50), (6, 60)]);
        assert_eq!(
            stored_hashes(&coalescer.process(0, StorageTier::Device, data, Some(input))),
            vec![6]
        );

        let output = coalescer.process(
            0,
            StorageTier::Device,
            KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![ExternalSequenceBlockHash(1)],
            }),
            None,
        );
        let KvCacheEventData::Removed(remove) = &output[0] else {
            panic!("expected remove")
        };
        assert_eq!(
            remove.block_hashes,
            vec![ExternalSequenceBlockHash(4), ExternalSequenceBlockHash(6)]
        );
    }

    #[test]
    fn tiers_are_independent() {
        let mut coalescer = EventCoalescer::new(1, 2).unwrap();
        let (data, input) = store(None, &[(1, 10), (2, 20)]);
        coalescer.process(0, StorageTier::Device, data.clone(), Some(input.clone()));
        coalescer.process(
            0,
            StorageTier::HostPinned,
            data.clone(),
            Some(input.clone()),
        );
        coalescer.process(0, StorageTier::External, data, Some(input));

        let output = coalescer.process(
            0,
            StorageTier::Device,
            KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![ExternalSequenceBlockHash(1)],
            }),
            None,
        );
        assert_eq!(output.len(), 1);
        assert!(
            coalescer.ranks[&0].superblocks[&ExternalSequenceBlockHash(2)]
                .active_tiers
                .contains(&StorageTier::HostPinned)
        );
        assert!(
            coalescer.ranks[&0].superblocks[&ExternalSequenceBlockHash(2)]
                .active_tiers
                .contains(&StorageTier::External)
        );
    }

    #[test]
    fn several_member_removals_emit_one_superblock_removal() {
        let mut coalescer = EventCoalescer::new(1, 4).unwrap();
        let (data, input) = store(None, &[(1, 10), (2, 20), (3, 30), (4, 40)]);
        coalescer.process(0, StorageTier::Device, data, Some(input));

        let output = coalescer.process(
            0,
            StorageTier::Device,
            KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![
                    ExternalSequenceBlockHash(1),
                    ExternalSequenceBlockHash(3),
                    ExternalSequenceBlockHash(4),
                ],
            }),
            None,
        );
        assert_eq!(
            output,
            vec![KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![ExternalSequenceBlockHash(4)]
            })]
        );
    }

    #[test]
    fn incomplete_group_removal_and_clear_emit_no_spurious_events() {
        let mut coalescer = EventCoalescer::new(1, 4).unwrap();
        let (data, input) = store(None, &[(1, 10), (2, 20)]);
        assert!(
            coalescer
                .process(0, StorageTier::Device, data, Some(input))
                .is_empty()
        );
        assert!(
            coalescer
                .process(
                    0,
                    StorageTier::Device,
                    KvCacheEventData::Removed(KvCacheRemoveData {
                        block_hashes: vec![ExternalSequenceBlockHash(2)],
                    }),
                    None,
                )
                .is_empty()
        );
        assert_eq!(
            coalescer.process(0, StorageTier::Device, KvCacheEventData::Cleared, None),
            vec![KvCacheEventData::Cleared]
        );
        assert!(coalescer.ranks.is_empty());
    }

    #[test]
    fn missing_context_withdraws_an_active_superblock() {
        let mut coalescer = EventCoalescer::new(1, 4).unwrap();
        let (data, input) = store(None, &[(1, 10), (2, 20), (3, 30), (4, 40)]);
        coalescer.process(0, StorageTier::Device, data.clone(), Some(input));

        let output = coalescer.process(0, StorageTier::Device, data, None);
        assert_eq!(
            output,
            vec![KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![ExternalSequenceBlockHash(4)]
            })]
        );
    }

    #[test]
    fn computes_target_hash_from_combined_eagle_tokens() {
        let mut coalescer = EventCoalescer::new(2, 4).unwrap();
        let (first, mut first_input) = block_with_tokens(1, vec![10, 20, 30], true);
        let (second, mut second_input) = block_with_tokens(2, vec![30, 40, 50], true);
        for input in [&mut first_input, &mut second_input] {
            input.lora_name = Some("adapter".to_string());
            input.cache_namespace = Some("salt".to_string());
        }
        let output = coalescer.process(
            0,
            StorageTier::Device,
            KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![first, second],
            }),
            Some(KvCacheStoreInput {
                blocks: vec![first_input, second_input],
            }),
        );
        let KvCacheEventData::Stored(store) = &output[0] else {
            panic!("expected store")
        };
        let expected = compute_block_hash_for_seq(
            &[10, 20, 30, 40, 50],
            4,
            BlockHashOptions {
                lora_name: Some("adapter"),
                cache_namespace: Some("salt"),
                is_eagle: Some(true),
                ..BlockHashOptions::default()
            },
        )[0];
        assert_eq!(store.blocks[0].tokens_hash, expected);
    }

    #[test]
    fn emits_several_superblocks_from_one_source_batch() {
        let mut coalescer = EventCoalescer::new(1, 2).unwrap();
        let (data, input) = store(None, &[(1, 10), (2, 20), (3, 30), (4, 40)]);
        let output = coalescer.process(0, StorageTier::Device, data, Some(input));
        assert_eq!(stored_hashes(&output), vec![2, 4]);
        let KvCacheEventData::Stored(second) = &output[1] else {
            panic!("expected second store")
        };
        assert_eq!(second.parent_hash, Some(ExternalSequenceBlockHash(2)));
    }

    #[test]
    fn dp_ranks_are_independent() {
        let mut coalescer = EventCoalescer::new(1, 2).unwrap();
        let (data, input) = store(None, &[(1, 10), (2, 20)]);
        coalescer.process(0, StorageTier::Device, data.clone(), Some(input.clone()));
        coalescer.process(1, StorageTier::Device, data, Some(input));
        coalescer.process(
            0,
            StorageTier::Device,
            KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![ExternalSequenceBlockHash(1)],
            }),
            None,
        );
        assert!(
            coalescer.ranks[&1].superblocks[&ExternalSequenceBlockHash(2)]
                .active_tiers
                .contains(&StorageTier::Device)
        );
    }

    #[test]
    fn merges_multimodal_metadata_at_target_offsets() {
        let mut coalescer = EventCoalescer::new(2, 4).unwrap();
        let mm_info = BlockExtraInfo {
            mm_objects: vec![BlockMmObjectInfo {
                mm_hash: 99,
                offsets: vec![(0, 1)],
            }],
        };
        let (mut first, mut first_input) = block_with_tokens(1, vec![10, 20], false);
        let (mut second, mut second_input) = block_with_tokens(2, vec![30, 40], false);
        first.mm_extra_info = Some(mm_info.clone());
        second.mm_extra_info = Some(mm_info.clone());
        first_input.hash_mm_extra_info = Some(mm_info.clone());
        second_input.hash_mm_extra_info = Some(mm_info);

        let output = coalescer.process(
            0,
            StorageTier::Device,
            KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![first, second],
            }),
            Some(KvCacheStoreInput {
                blocks: vec![first_input, second_input],
            }),
        );
        let KvCacheEventData::Stored(store) = &output[0] else {
            panic!("expected store")
        };
        let merged = store.blocks[0].mm_extra_info.as_ref().unwrap();
        assert_eq!(merged.mm_objects[0].offsets, vec![(0, 1), (2, 3)]);
    }
}
