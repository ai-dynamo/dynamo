// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

impl ConcurrentRadixTreeCompressed {
    // ------------------------------------------------------------------
    // Tree dump
    // ------------------------------------------------------------------

    pub(super) fn dump_tree_as_events(&self) -> Vec<RouterEvent> {
        tracing::debug!("Dumping concurrent radix tree as events");

        let mut events = Vec::new();
        let mut event_id = 0u64;
        let mut queue = VecDeque::new();

        for child_node in self.root.live_children() {
            queue.push_back((child_node, None::<ExternalSequenceBlockHash>));
        }

        Self::append_dump_events_from_queue(&mut events, &mut event_id, queue);

        let mut anchor_queue = VecDeque::new();
        for anchor in self.anchor_nodes.iter() {
            let anchor_id = *anchor.key();
            for child_node in anchor.value().live_children() {
                anchor_queue.push_back((child_node, Some(anchor_id)));
            }
        }
        Self::append_dump_events_from_queue(&mut events, &mut event_id, anchor_queue);

        events
    }

    fn append_dump_events_from_queue(
        events: &mut Vec<RouterEvent>,
        event_id: &mut u64,
        mut queue: VecDeque<(SharedNode, Option<ExternalSequenceBlockHash>)>,
    ) {
        while let Some((start_node, parent_hash)) = queue.pop_front() {
            let mut merged_edge: Vec<(LocalBlockHash, ExternalSequenceBlockHash)> = Vec::new();
            let mut current = start_node;

            loop {
                let snapshot = current.dump_snapshot();

                if !snapshot.has_any_workers && snapshot.children_empty {
                    break;
                }

                merged_edge.extend_from_slice(&snapshot.edge);

                // Merge condition: this node is a pure passthrough that can be
                // collapsed with its single child. Requires identical worker sets
                // and no partial-coverage cutoffs on either side.
                if snapshot.can_merge {
                    let next = snapshot.live_children[0].clone();
                    current = next;
                    continue;
                }

                if merged_edge.is_empty() {
                    break;
                }

                let full_blocks: Vec<KvCacheStoredBlockData> = merged_edge
                    .iter()
                    .map(|&(local, ext)| KvCacheStoredBlockData {
                        tokens_hash: local,
                        block_hash: ext,
                        mm_extra_info: None,
                    })
                    .collect();
                let last_ext = merged_edge.last().unwrap().1;

                for worker in snapshot.full_edge_workers {
                    events.push(RouterEvent::new(
                        worker.worker_id,
                        KvCacheEvent {
                            event_id: *event_id,
                            data: KvCacheEventData::Stored(KvCacheStoreData {
                                parent_hash,
                                start_position: None,
                                blocks: full_blocks.clone(),
                            }),
                            dp_rank: worker.dp_rank,
                        },
                    ));
                    *event_id += 1;
                }
                for (worker, cutoff) in snapshot.worker_cutoffs {
                    events.push(RouterEvent::new(
                        worker.worker_id,
                        KvCacheEvent {
                            event_id: *event_id,
                            data: KvCacheEventData::Stored(KvCacheStoreData {
                                parent_hash,
                                start_position: None,
                                blocks: full_blocks[..cutoff].to_vec(),
                            }),
                            dp_rank: worker.dp_rank,
                        },
                    ));
                    *event_id += 1;
                }

                for child in snapshot.live_children {
                    queue.push_back((child, Some(last_ext)));
                }

                break;
            }
        }
    }
}
