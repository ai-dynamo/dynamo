// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::{
    protocols::{ExternalSequenceBlockHash, WorkerWithDpRank},
    SessionPrefixIndexer,
};
use serde_json::json;

fn hashes(values: &[u64]) -> Vec<ExternalSequenceBlockHash> {
    values
        .iter()
        .copied()
        .map(ExternalSequenceBlockHash)
        .collect()
}

fn lineage(
    index: &SessionPrefixIndexer,
    session: &str,
    worker: WorkerWithDpRank,
) -> Vec<Vec<ExternalSequenceBlockHash>> {
    let mut lineages = index
        .get_session_block_lineage(session, worker, None)
        .expect("an unanchored lineage query cannot fail");
    lineages.sort();
    lineages
}

fn raw(lineages: &[Vec<ExternalSequenceBlockHash>]) -> Vec<Vec<u64>> {
    lineages
        .iter()
        .map(|path| path.iter().map(|hash| hash.0).collect())
        .collect()
}

fn main() {
    let worker_1 = WorkerWithDpRank::new(1, 0);
    let worker_2 = WorkerWithDpRank::new(2, 0);
    let chain = hashes(&[1, 2, 3]);
    let index = SessionPrefixIndexer::new();

    index
        .update_session_from_stored_blocks("session-1", worker_1, None, &chain)
        .unwrap();
    let session_1 = lineage(&index, "session-1", worker_1);
    assert_eq!(session_1, vec![chain.clone()]);

    assert!(index
        .update_session_from_match("session-2", worker_1, chain[2])
        .unwrap());
    let session_2 = lineage(&index, "session-2", worker_1);
    assert_eq!(session_2, vec![chain.clone()]);
    assert_eq!(
        index.node_count(),
        3,
        "cache reuse must not duplicate nodes"
    );

    index
        .update_session_from_match("session-3", worker_1, chain[2])
        .unwrap();
    index
        .update_session_from_match("session-3", worker_2, chain[2])
        .unwrap();
    assert_eq!(
        index.update_session_from_removed_blocks(worker_2, &[chain[2]]),
        1
    );
    let worker_1_after_worker_2_removal = lineage(&index, "session-3", worker_1);
    let worker_2_after_removal = lineage(&index, "session-3", worker_2);
    assert_eq!(worker_1_after_worker_2_removal, vec![chain.clone()]);
    assert_eq!(worker_2_after_removal, vec![chain[..2].to_vec()]);

    assert_eq!(
        index.update_session_from_removed_blocks(worker_1, &chain[1..]),
        3,
        "all sessions on worker 1 must recede past the removed suffix"
    );
    let session_1_after_batched_removal = lineage(&index, "session-1", worker_1);
    assert_eq!(session_1_after_batched_removal, vec![chain[..1].to_vec()]);

    assert_eq!(index.clear_worker_frontiers(worker_2), 1);
    assert!(lineage(&index, "session-3", worker_2).is_empty());
    assert_eq!(
        lineage(&index, "session-3", worker_1),
        vec![chain[..1].to_vec()]
    );

    let cross_worker = SessionPrefixIndexer::new();
    cross_worker
        .update_session_from_stored_blocks("session-a", worker_1, None, &chain)
        .unwrap();
    cross_worker
        .update_session_from_stored_blocks("session-b", worker_2, None, &chain)
        .unwrap();
    assert_eq!(
        cross_worker.update_session_from_removed_blocks(worker_1, &chain),
        1
    );
    let session_a_worker_1 = lineage(&cross_worker, "session-a", worker_1);
    let session_b_worker_2 = lineage(&cross_worker, "session-b", worker_2);
    assert!(session_a_worker_1.is_empty());
    assert_eq!(session_b_worker_2, vec![chain.clone()]);

    let same_worker = SessionPrefixIndexer::new();
    for session in ["session-a", "session-b"] {
        same_worker
            .update_session_from_stored_blocks(session, worker_1, None, &chain)
            .unwrap();
    }
    assert_eq!(
        same_worker.update_session_from_removed_blocks(worker_1, &chain),
        2
    );
    let same_worker_session_a = lineage(&same_worker, "session-a", worker_1);
    let same_worker_session_b = lineage(&same_worker, "session-b", worker_1);
    assert!(same_worker_session_a.is_empty());
    assert!(same_worker_session_b.is_empty());

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "linear_lineage": {
                "passed": true,
                "session_1_worker_1": raw(&session_1),
            },
            "cache_hit_association": {
                "passed": true,
                "session_2_worker_1": raw(&session_2),
                "logical_node_count": index.node_count(),
            },
            "worker_qualified_removal": {
                "passed": true,
                "session_3_worker_1": raw(&worker_1_after_worker_2_removal),
                "session_3_worker_2": raw(&worker_2_after_removal),
            },
            "batched_suffix_removal": {
                "passed": true,
                "session_1_worker_1": raw(&session_1_after_batched_removal),
            },
            "worker_clear": {
                "passed": true,
                "session_3_worker_1": raw(&lineage(&index, "session-3", worker_1)),
                "session_3_worker_2": raw(&lineage(&index, "session-3", worker_2)),
            },
            "cross_worker_shared_hash_removal": {
                "passed": true,
                "session_a_worker_1": raw(&session_a_worker_1),
                "session_b_worker_2": raw(&session_b_worker_2),
            },
            "same_worker_shared_hash_removal": {
                "passed": true,
                "session_a_worker_1": raw(&same_worker_session_a),
                "session_b_worker_1": raw(&same_worker_session_b),
            },
        }))
        .unwrap()
    );
}
