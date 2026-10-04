// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::indexer::compressed_radix::append_dump_events;

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
            queue.push_back(DumpStart {
                node: child_node,
                parent_hash: None,
                parent_ranks: None,
            });
        }

        self.append_dump_events_from_queue(&mut events, &mut event_id, queue);

        let mut anchor_queue = VecDeque::new();
        for anchor in self.anchor_nodes.iter() {
            let anchor_id = *anchor.key();
            let snapshot = {
                let guard = crossbeam_epoch::pin();
                anchor.value().dump_snapshot(self.slots.table(&guard))
            };
            let parent_ranks = Arc::new(FxHashSet::from_iter(snapshot.full_edge_workers));
            for child_node in snapshot.live_children {
                anchor_queue.push_back(DumpStart {
                    node: child_node,
                    parent_hash: Some(anchor_id),
                    parent_ranks: Some(parent_ranks.clone()),
                });
            }
        }
        self.append_dump_events_from_queue(&mut events, &mut event_id, anchor_queue);

        events
    }

    fn append_dump_events_from_queue(
        &self,
        events: &mut Vec<RouterEvent>,
        event_id: &mut u64,
        mut queue: VecDeque<DumpStart>,
    ) {
        while let Some(start) = queue.pop_front() {
            let mut merged_edge: Vec<(LocalBlockHash, ExternalSequenceBlockHash)> = Vec::new();
            let mut current = start.node;
            // One pin and slot table per merged chain: the merge compares raw slots, so
            // every node of the chain must map them to ranks the same way.
            let guard = crossbeam_epoch::pin();
            let table = self.slots.table(&guard);

            loop {
                let mut snapshot = current.dump_snapshot(table);

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

                // Like a reader, credit a rank below a parent only if it covers the
                // parent's whole edge. This also keeps a node unlinked after it was
                // queued, whose stale bits a recycled slot now maps to another rank,
                // from crediting that rank.
                if let Some(parent_ranks) = &start.parent_ranks {
                    snapshot
                        .full_edge_workers
                        .retain(|worker| parent_ranks.contains(worker));
                    snapshot
                        .worker_cutoffs
                        .retain(|(worker, _)| parent_ranks.contains(worker));
                }

                let last_ext = merged_edge.last().unwrap().1;

                append_dump_events(
                    events,
                    event_id,
                    start.parent_hash,
                    &merged_edge,
                    &snapshot.full_edge_workers,
                    &snapshot.worker_cutoffs,
                );

                if snapshot.full_edge_workers.is_empty() {
                    break;
                }
                let parent_ranks = Arc::new(FxHashSet::from_iter(snapshot.full_edge_workers));
                for child in snapshot.live_children {
                    queue.push_back(DumpStart {
                        node: child,
                        parent_hash: Some(last_ext),
                        parent_ranks: Some(parent_ranks.clone()),
                    });
                }

                break;
            }
        }
    }
}

struct DumpStart {
    node: SharedNode,
    parent_hash: Option<ExternalSequenceBlockHash>,
    /// Ranks covering the parent's whole edge; `None` below the root.
    parent_ranks: Option<Arc<FxHashSet<WorkerWithDpRank>>>,
}
