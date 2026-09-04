// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LoRA-aware candidate narrowing shared by every selection host.

use std::collections::HashSet;

use crate::protocols::{WorkerId, WorkerWithDpRank};

/// Host-owned knowledge of which workers can serve a LoRA adapter.
///
/// The frontend implements this over its LoRA routing table and state tracker;
/// a runtime-free host may back it with a static map. Implementations return a
/// subset of `available`, ordered arbitrarily. Returning `available` unchanged
/// means "no LoRA-specific preference".
pub trait LoraWorkerFilter: Send + Sync {
    fn filter_worker_ids_for_lora(&self, lora_name: &str, available: &[WorkerId]) -> Vec<WorkerId>;
}

/// Narrow `allowed_worker_ids` to the workers `filter` reports for `lora_name`,
/// strictly within the existing candidate universe.
///
/// - Without a LoRA name the caller's allow-set is returned untouched.
/// - The universe is the caller's allow-set when present, else `all_workers`.
/// - A pinned worker inside the universe is always retained, even if the filter
///   dropped it, so the pin still wins for cache correctness. A pin outside the
///   universe is not re-added: the caller's constraint stands.
/// - An empty narrowing falls back to the caller's allow-set rather than
///   producing an unroutable request.
pub fn narrow_allowed_worker_ids_by_lora(
    filter: &dyn LoraWorkerFilter,
    lora_name: Option<&str>,
    allowed_worker_ids: Option<HashSet<WorkerId>>,
    pinned_worker: Option<&WorkerWithDpRank>,
    all_workers: impl FnOnce() -> Vec<WorkerId>,
) -> Option<HashSet<WorkerId>> {
    let Some(lora_name) = lora_name else {
        return allowed_worker_ids;
    };
    let base: Vec<WorkerId> = match &allowed_worker_ids {
        Some(allowed) => allowed.iter().copied().collect(),
        None => all_workers(),
    };
    if base.is_empty() {
        return allowed_worker_ids;
    }
    let mut narrowed: HashSet<WorkerId> = filter
        .filter_worker_ids_for_lora(lora_name, &base)
        .into_iter()
        .collect();
    if let Some(pinned) = pinned_worker
        && base.contains(&pinned.worker_id)
    {
        narrowed.insert(pinned.worker_id);
    }
    if narrowed.is_empty() {
        return allowed_worker_ids;
    }
    Some(narrowed)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct OnlyWorker(WorkerId);

    impl LoraWorkerFilter for OnlyWorker {
        fn filter_worker_ids_for_lora(
            &self,
            _lora_name: &str,
            available: &[WorkerId],
        ) -> Vec<WorkerId> {
            available
                .iter()
                .copied()
                .filter(|id| *id == self.0)
                .collect()
        }
    }

    fn narrow(
        filter: &dyn LoraWorkerFilter,
        lora_name: Option<&str>,
        allowed: Option<HashSet<WorkerId>>,
        pinned: Option<WorkerWithDpRank>,
    ) -> Option<HashSet<WorkerId>> {
        narrow_allowed_worker_ids_by_lora(filter, lora_name, allowed, pinned.as_ref(), || {
            vec![1, 2, 3]
        })
    }

    #[test]
    fn no_lora_leaves_allow_set_untouched() {
        let allowed = Some(HashSet::from([1, 2]));
        assert_eq!(narrow(&OnlyWorker(3), None, allowed.clone(), None), allowed);
        assert_eq!(narrow(&OnlyWorker(3), None, None, None), None);
    }

    #[test]
    fn narrows_within_all_workers_or_allow_set() {
        assert_eq!(
            narrow(&OnlyWorker(2), Some("a"), None, None),
            Some(HashSet::from([2]))
        );
        // Filter target outside the caller's allow-set: nothing survives, so
        // the caller's set is preserved rather than widened.
        let allowed = Some(HashSet::from([1, 3]));
        assert_eq!(
            narrow(&OnlyWorker(2), Some("a"), allowed.clone(), None),
            allowed
        );
    }

    #[test]
    fn pinned_worker_is_retained_only_inside_the_universe() {
        let pinned = WorkerWithDpRank::new(1, 0);
        assert_eq!(
            narrow(&OnlyWorker(2), Some("a"), None, Some(pinned)),
            Some(HashSet::from([1, 2]))
        );
        let allowed = Some(HashSet::from([2, 3]));
        assert_eq!(
            narrow(&OnlyWorker(2), Some("a"), allowed, Some(pinned)),
            Some(HashSet::from([2]))
        );
    }

    #[test]
    fn empty_universe_is_a_noop() {
        assert_eq!(
            narrow_allowed_worker_ids_by_lora(&OnlyWorker(1), Some("a"), None, None, Vec::new),
            None
        );
    }
}
