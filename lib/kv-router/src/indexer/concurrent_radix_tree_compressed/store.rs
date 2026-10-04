// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

struct SplitLookupFinish<'a> {
    split: SplitLookupData,
    prefix_node: &'a SharedNode,
    prefix_blocks: &'a [KvCacheStoredBlockData],
    tail_blocks: &'a [KvCacheStoredBlockData],
    tail_node: &'a SharedNode,
}

struct StoreParentCursor<'a> {
    parent: &'a SharedNode,
    parent_is_anchor: bool,
    last_ext_hash: Option<ExternalSequenceBlockHash>,
}

impl ConcurrentRadixTreeCompressed {
    // ------------------------------------------------------------------
    // apply_stored
    // ------------------------------------------------------------------

    #[cfg_attr(feature = "profile", inline(never))]
    pub(super) fn apply_stored(
        &self,
        lookup: &mut LaneLookup,
        worker: WorkerWithDpRank,
        op: KvCacheStoreData,
        id: u64,
        counters: Option<&PreBoundEventCounters>,
        guard: &Guard,
    ) -> Result<(), KvCacheEventError> {
        let slot = self.slots.acquire(worker, guard).inspect_err(|_| {
            tracing::warn!(
                worker_id = worker.worker_id.to_string(),
                dp_rank = worker.dp_rank,
                id,
                "No free coverage slot; skipping store operation"
            );
        })?;
        let worker = EventWorker {
            rank: worker,
            slot,
            table: self.slots.table(guard),
        };
        lookup.add_worker(worker.rank);

        let parent_resolution = self.resolve_store_parent(lookup, worker, &op, id)?;
        let outcome = self.insert_for_resolved_parent(lookup, worker, parent_resolution, &op)?;
        self.record_store_outcome(&outcome, counters);

        Ok(())
    }

    fn resolve_store_parent(
        &self,
        lookup: &mut LaneLookup,
        worker: EventWorker<'_>,
        op: &KvCacheStoreData,
        id: u64,
    ) -> Result<StoreParentResolution, KvCacheEventError> {
        let Some(parent_hash) = op.parent_hash else {
            return Ok(StoreParentResolution::InsertFrom {
                parent: self.root.clone(),
                parent_is_anchor: false,
            });
        };

        self.resolve_explicit_store_parent(lookup, worker, parent_hash, op, id)
    }

    fn resolve_explicit_store_parent(
        &self,
        lookup: &mut LaneLookup,
        worker: EventWorker<'_>,
        parent_hash: ExternalSequenceBlockHash,
        op: &KvCacheStoreData,
        id: u64,
    ) -> Result<StoreParentResolution, KvCacheEventError> {
        let mut node = self.lookup_store_parent_node(lookup, worker, parent_hash, None, op, id)?;
        loop {
            // NOTE(perf): Combining coverage rejection and edge planning into
            // one state snapshot regressed throughput. Keep these phases separate.
            match node.parent_coverage(worker.slot, parent_hash) {
                ParentCoverage::Covered => {}
                ParentCoverage::Uncovered(uncovered) => {
                    return Err(self.reject_uncovered_store_parent(
                        lookup,
                        worker,
                        parent_hash,
                        uncovered,
                        id,
                    ));
                }
                // The lookup entry is stale; continue from the node the hash moved to,
                // not from the entry, which repair leaves alone if the worker no longer
                // covers the parent there.
                ParentCoverage::Missing => {
                    node = self.lookup_store_parent_node(
                        lookup,
                        worker,
                        parent_hash,
                        Some(&node),
                        op,
                        id,
                    )?;
                    continue;
                }
            }

            // A split after the coverage check surfaces as `Missing` on the next pass.
            let Some(plan) = node.plan_store_parent_edge(parent_hash, &op.blocks) else {
                continue;
            };
            let edge_action = node.apply_store_parent_edge_plan(worker.slot, plan, &op.blocks);

            match edge_action {
                ParentEdgeAction::Stale => continue,
                ParentEdgeAction::ReuseExistingEdge { coverage_changed } => {
                    return Ok(StoreParentResolution::ReusedExistingEdge {
                        node,
                        coverage_changed,
                    });
                }
                ParentEdgeAction::InsertFromParent(split_data) => {
                    if let Some(split) = split_data {
                        self.apply_split_lookup(lookup, worker.table, &node, split);
                    }
                    return Ok(StoreParentResolution::InsertFrom {
                        parent_is_anchor: self.is_anchor_node(parent_hash, &node),
                        parent: node,
                    });
                }
            }
        }
    }

    /// Find the store parent through the worker's lookup, or, when `stale` is the
    /// node a locked check found missing `parent_hash`, through lookup repair.
    /// Falls back to branch anchors.
    fn lookup_store_parent_node(
        &self,
        lookup: &mut LaneLookup,
        worker: EventWorker<'_>,
        parent_hash: ExternalSequenceBlockHash,
        stale: Option<&SharedNode>,
        op: &KvCacheStoreData,
        id: u64,
    ) -> Result<SharedNode, KvCacheEventError> {
        let node = match stale {
            None => lookup.node(worker.rank, parent_hash),
            Some(stale) => self.repair_stale(
                lookup,
                worker.table,
                stale,
                parent_hash,
                LookupRepairDirection::TowardTail,
            ),
        };
        if let Some(node) = node {
            return Ok(node);
        }

        if let Some(node) = self.resolve_anchor_lookup(lookup, worker, parent_hash) {
            return Ok(node);
        }

        tracing::warn!(
            worker_id = worker.rank.worker_id.to_string(),
            dp_rank = worker.rank.dp_rank,
            id,
            parent_hash = ?op.parent_hash,
            num_blocks = op.blocks.len(),
            "Failed to find parent block; skipping store operation"
        );
        Err(KvCacheEventError::ParentBlockNotFound)
    }

    fn reject_uncovered_store_parent(
        &self,
        lookup: &mut LaneLookup,
        worker: EventWorker<'_>,
        parent_hash: ExternalSequenceBlockHash,
        uncovered: UncoveredParent,
        id: u64,
    ) -> KvCacheEventError {
        tracing::warn!(
            worker_id = worker.rank.worker_id.to_string(),
            dp_rank = worker.rank.dp_rank,
            id,
            parent_hash = ?parent_hash,
            pos = uncovered.pos,
            cutoff = uncovered.cutoff,
            "Stale parent: worker no longer covers parent_hash; rejecting store"
        );

        lookup.remove(worker.rank, parent_hash);
        self.release_hash(worker.rank, parent_hash);
        KvCacheEventError::ParentBlockNotFound
    }

    #[cfg_attr(feature = "profile", inline(never))]
    fn insert_for_resolved_parent(
        &self,
        lookup: &mut LaneLookup,
        worker: EventWorker<'_>,
        parent_resolution: StoreParentResolution,
        op: &KvCacheStoreData,
    ) -> Result<StoreInsertOutcome, KvCacheEventError> {
        let outcome = match parent_resolution {
            StoreParentResolution::InsertFrom {
                parent,
                parent_is_anchor,
            } => {
                return self.insert_blocks_from(
                    lookup,
                    worker,
                    &parent,
                    parent_is_anchor,
                    op.parent_hash,
                    &op.blocks,
                );
            }
            StoreParentResolution::ReusedExistingEdge {
                node,
                coverage_changed,
            } => self.finish_with_lookup_update(
                lookup,
                worker,
                &op.blocks,
                &node,
                !op.blocks.is_empty() && !coverage_changed,
            ),
        };
        Ok(outcome)
    }

    fn finish_with_lookup_update(
        &self,
        lookup: &mut LaneLookup,
        worker: EventWorker<'_>,
        blocks: &[KvCacheStoredBlockData],
        node: &SharedNode,
        duplicate_store: bool,
    ) -> StoreInsertOutcome {
        let lookup_changed = self.update_lookup_for_blocks(worker.rank, lookup, blocks, node);

        StoreInsertOutcome {
            duplicate_store: duplicate_store && !lookup_changed,
        }
    }

    fn finish_after_split_lookup(
        &self,
        lookup: &mut LaneLookup,
        worker: EventWorker<'_>,
        finish: SplitLookupFinish<'_>,
    ) -> StoreInsertOutcome {
        self.apply_split_lookup(lookup, worker.table, finish.prefix_node, finish.split);

        self.update_lookup_for_blocks(
            worker.rank,
            lookup,
            finish.prefix_blocks,
            finish.prefix_node,
        );
        self.update_lookup_for_blocks(worker.rank, lookup, finish.tail_blocks, finish.tail_node);

        StoreInsertOutcome {
            duplicate_store: false,
        }
    }

    fn record_store_outcome(
        &self,
        outcome: &StoreInsertOutcome,
        counters: Option<&PreBoundEventCounters>,
    ) {
        if outcome.duplicate_store
            && let Some(counters) = counters
        {
            counters.inc_warning(EventWarningKind::DuplicateStore);
        }
    }

    fn child_or_insert_remaining(
        &self,
        lookup: &mut LaneLookup,
        worker: EventWorker<'_>,
        cursor: StoreParentCursor<'_>,
        remaining: &[KvCacheStoredBlockData],
    ) -> Result<StoreInsertStep, KvCacheEventError> {
        debug_assert!(!remaining.is_empty());

        let first_local = remaining[0].tokens_hash;
        let plan = cursor
            .parent
            .child_lookup_plan(cursor.last_ext_hash, first_local);

        let shape_version = match plan {
            ParentChildPlan::StaleParent { hash } => {
                let Some(resolved) =
                    self.resolve_lookup(lookup, worker, hash, LookupRepairDirection::TowardTail)
                else {
                    tracing::warn!(
                        worker_id = worker.rank.worker_id.to_string(),
                        dp_rank = worker.rank.dp_rank,
                        parent_hash = ?hash,
                        num_blocks = remaining.len(),
                        "Stale parent hash could not be resolved; rejecting store"
                    );
                    return Err(KvCacheEventError::ParentBlockNotFound);
                };

                return Ok(StoreInsertStep::RetryParent {
                    parent_is_anchor: self.is_anchor_node(hash, &resolved),
                    parent: resolved,
                });
            }
            ParentChildPlan::Descend(child) => return Ok(StoreInsertStep::Descend(child)),
            ParentChildPlan::InteriorParent { shape_version }
            | ParentChildPlan::MissingChild { shape_version } => shape_version,
        };

        if let Some(parent_hash) = cursor.last_ext_hash
            && !cursor.parent_is_anchor
        {
            // Parent hashes can point inside a compressed edge. Before attaching a
            // child, try to reuse that existing suffix or split at the parent.
            if let Some(edge_plan) = cursor.parent.plan_store_parent_edge(parent_hash, remaining) {
                let edge_action =
                    cursor
                        .parent
                        .apply_store_parent_edge_plan(worker.slot, edge_plan, remaining);
                match edge_action {
                    ParentEdgeAction::Stale => {
                        return Ok(StoreInsertStep::RetryParent {
                            parent: cursor.parent.clone(),
                            parent_is_anchor: cursor.parent_is_anchor,
                        });
                    }
                    ParentEdgeAction::ReuseExistingEdge { coverage_changed } => {
                        return Ok(StoreInsertStep::Done(self.finish_with_lookup_update(
                            lookup,
                            worker,
                            remaining,
                            cursor.parent,
                            !remaining.is_empty() && !coverage_changed,
                        )));
                    }
                    ParentEdgeAction::InsertFromParent(Some(split)) => {
                        self.apply_split_lookup(lookup, worker.table, cursor.parent, split);
                        return Ok(StoreInsertStep::RetryParent {
                            parent: cursor.parent.clone(),
                            parent_is_anchor: cursor.parent_is_anchor,
                        });
                    }
                    ParentEdgeAction::InsertFromParent(None) => {}
                }
            }
        }

        if let Some(parent_hash) = cursor.last_ext_hash
            && !cursor.parent_is_anchor
        {
            // Keep the decode-extension happy path as a direct commit attempt:
            // the node method revalidates the shape and appends only if the
            // parent is still the covered tail leaf.
            match cursor.parent.try_extend_leaf_with_version(
                worker.slot,
                parent_hash,
                remaining,
                shape_version,
            ) {
                None => {
                    return Ok(StoreInsertStep::RetryParent {
                        parent: cursor.parent.clone(),
                        parent_is_anchor: cursor.parent_is_anchor,
                    });
                }
                Some(true) => {
                    return Ok(StoreInsertStep::Done(self.finish_with_lookup_update(
                        lookup,
                        worker,
                        remaining,
                        cursor.parent,
                        false,
                    )));
                }
                Some(false) => {}
            }
        }

        let new_node = Arc::new(Node::from_blocks_for_slot(remaining, worker.slot));
        if let Some(parent_hash) = cursor.last_ext_hash
            && !cursor.parent_is_anchor
        {
            // Allocation can race with another writer leaving the parent
            // extendable. Recheck extension once before publishing a child.
            match cursor.parent.try_extend_leaf_with_version(
                worker.slot,
                parent_hash,
                remaining,
                shape_version,
            ) {
                None => {
                    return Ok(StoreInsertStep::RetryParent {
                        parent: cursor.parent.clone(),
                        parent_is_anchor: cursor.parent_is_anchor,
                    });
                }
                Some(true) => {
                    return Ok(StoreInsertStep::Done(self.finish_with_lookup_update(
                        lookup,
                        worker,
                        remaining,
                        cursor.parent,
                        false,
                    )));
                }
                Some(false) => {}
            }
        }

        match cursor
            .parent
            .insert_child_if_still_missing(first_local, new_node, shape_version)
        {
            InsertChildOutcome::Stale => Ok(StoreInsertStep::RetryParent {
                parent: cursor.parent.clone(),
                parent_is_anchor: cursor.parent_is_anchor,
            }),
            InsertChildOutcome::Existing(child) => Ok(StoreInsertStep::Descend(child)),
            InsertChildOutcome::Inserted(new_node) => Ok(StoreInsertStep::Done(
                self.finish_with_lookup_update(lookup, worker, remaining, &new_node, false),
            )),
        }
    }

    fn insert_into_child(
        &self,
        lookup: &mut LaneLookup,
        worker: EventWorker<'_>,
        child: &SharedNode,
        remaining: &[KvCacheStoredBlockData],
        duplicate_store: bool,
    ) -> ChildInsertStep {
        loop {
            let scan = child.scan_store_prefix(remaining);
            let mut duplicate_store = duplicate_store;

            debug_assert!(
                scan.match_len >= 1,
                "first hash must match since child was found by it"
            );

            if scan.match_len < scan.edge_len {
                // A partial match either means this worker only covers a prefix
                // of the compressed edge, or the store diverges and must split
                // the edge.
                if scan.match_len == remaining.len() {
                    let Some(coverage_changed) = child.cover_prefix_with_version(
                        worker.slot,
                        scan.match_len,
                        scan.shape_version,
                    ) else {
                        continue;
                    };
                    return ChildInsertStep::Done(self.finish_with_lookup_update(
                        lookup,
                        worker,
                        remaining,
                        child,
                        duplicate_store && !remaining.is_empty() && !coverage_changed,
                    ));
                }

                let tail = &remaining[scan.match_len..];
                debug_assert!(!tail.is_empty());

                let tail_candidate = Arc::new(Node::from_blocks_for_slot(tail, worker.slot));
                let SplitStoreOutcome::Done { split, tail_node } = child.split_for_store_tail(
                    worker.slot,
                    scan.match_len,
                    tail[0].tokens_hash,
                    tail_candidate,
                    scan.shape_version,
                ) else {
                    continue;
                };
                return ChildInsertStep::Done(self.finish_after_split_lookup(
                    lookup,
                    worker,
                    SplitLookupFinish {
                        split,
                        prefix_node: child,
                        prefix_blocks: &remaining[..scan.match_len],
                        tail_blocks: tail,
                        tail_node: &tail_node,
                    },
                ));
            }

            // The whole child edge matched. Mark this worker as covering it,
            // update lookup for that edge, then continue with the unmatched
            // suffix.
            let Some(promoted) =
                child.promote_to_full_with_version(worker.slot, scan.shape_version)
            else {
                continue;
            };
            if promoted {
                duplicate_store = false;
            }

            let lookup_changed = self.update_lookup_for_blocks(
                worker.rank,
                lookup,
                &remaining[..scan.edge_len],
                child,
            );
            if lookup_changed {
                duplicate_store = false;
            }

            return ChildInsertStep::Descend {
                edge_len: scan.edge_len,
                last_ext_hash: remaining[scan.edge_len - 1].block_hash,
                duplicate_store,
            };
        }
    }

    #[cfg_attr(feature = "profile", inline(never))]
    fn insert_blocks_from(
        &self,
        lookup: &mut LaneLookup,
        worker: EventWorker<'_>,
        parent: &SharedNode,
        parent_is_anchor: bool,
        seed_hash: Option<ExternalSequenceBlockHash>,
        blocks: &[KvCacheStoredBlockData],
    ) -> Result<StoreInsertOutcome, KvCacheEventError> {
        let mut current_parent = parent.clone();
        let mut current_parent_is_anchor = parent_is_anchor;
        let mut remaining = blocks;
        let mut duplicate_store = !blocks.is_empty();
        // Track the last ExternalSequenceBlockHash we matched to detect if
        // `current_parent` was split by a concurrent thread between iterations.
        // A split shortens `current_parent`'s edge and moves our last-matched
        // hash into a new suffix child. The child lookup plan detects this under
        // a read lock, then mutation paths revalidate the node shape version.
        //
        // Seeded with parent_hash so the very first iteration detects a split
        // that occurred after apply_stored released its write lock but before
        // we acquired ours here.
        let mut last_ext_hash: Option<ExternalSequenceBlockHash> = seed_hash;

        while !remaining.is_empty() {
            let child = match self.child_or_insert_remaining(
                lookup,
                worker,
                StoreParentCursor {
                    parent: &current_parent,
                    parent_is_anchor: current_parent_is_anchor,
                    last_ext_hash,
                },
                remaining,
            )? {
                StoreInsertStep::RetryParent {
                    parent,
                    parent_is_anchor,
                } => {
                    current_parent = parent;
                    current_parent_is_anchor = parent_is_anchor;
                    continue;
                }
                StoreInsertStep::Descend(child) => child,
                StoreInsertStep::Done(outcome) => return Ok(outcome),
            };

            match self.insert_into_child(lookup, worker, &child, remaining, duplicate_store) {
                ChildInsertStep::Done(outcome) => return Ok(outcome),
                ChildInsertStep::Descend {
                    edge_len,
                    last_ext_hash: matched_hash,
                    duplicate_store: duplicate,
                } => {
                    duplicate_store = duplicate;
                    last_ext_hash = Some(matched_hash);
                    remaining = &remaining[edge_len..];
                    current_parent = child;
                    current_parent_is_anchor = false;
                }
            }
        }

        Ok(StoreInsertOutcome { duplicate_store })
    }

    #[cfg(test)]
    pub(super) fn insert_blocks_from_for_test(
        &self,
        lookup: &mut LaneLookup,
        worker: WorkerWithDpRank,
        parent: &SharedNode,
        seed_hash: ExternalSequenceBlockHash,
        blocks: &[KvCacheStoredBlockData],
    ) -> Result<StoreInsertOutcome, KvCacheEventError> {
        let guard = crossbeam_epoch::pin();
        let worker = self.event_worker_for_test(worker, &guard);
        self.insert_blocks_from(lookup, worker, parent, false, Some(seed_hash), blocks)
    }
}
