// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cross-TP routing helper.
//!
//! [`route_local_to_remote`] maps local worker ranks onto remote worker ranks
//! when the two sides run different tensor-parallel degrees. It is consumed by
//! the layout-compatibility checks (`super::layout_compat`) so cross-leader
//! metadata import validates the correct per-rank pairs.

/// Routing strategy: which local ranks receive from which remote ranks.
///
/// This function determines how metadata/transfers are routed when
/// the local and remote TP sizes differ.
///
/// # Examples
/// - TP=4 local, TP=4 remote: 1:1 mapping (rank 0→0, 1→1, 2→2, 3→3)
/// - TP=4 local, TP=2 remote: 0→0, 1→0, 2→1, 3→1 (many-to-one)
/// - TP=2 local, TP=4 remote: 0→\[0,1\], 1→\[2,3\] (one-to-many)
pub fn route_local_to_remote(
    local_rank: usize,
    local_count: usize,
    remote_count: usize,
) -> Vec<usize> {
    if local_count == remote_count {
        // 1:1 mapping
        vec![local_rank]
    } else if local_count > remote_count {
        // Many local → few remote: multiple locals share a remote
        vec![local_rank % remote_count]
    } else {
        // Few local → many remote: each local gets multiple remotes
        let remotes_per_local = remote_count / local_count;
        let start = local_rank * remotes_per_local;
        // Last local rank absorbs any remainder from non-divisible ratios
        let end = if local_rank == local_count - 1 {
            remote_count
        } else {
            start + remotes_per_local
        };
        (start..end).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_route_1_to_1() {
        // Same TP size
        assert_eq!(route_local_to_remote(0, 4, 4), vec![0]);
        assert_eq!(route_local_to_remote(1, 4, 4), vec![1]);
        assert_eq!(route_local_to_remote(2, 4, 4), vec![2]);
        assert_eq!(route_local_to_remote(3, 4, 4), vec![3]);
    }

    #[test]
    fn test_route_many_to_one() {
        // Local TP=4, Remote TP=2
        assert_eq!(route_local_to_remote(0, 4, 2), vec![0]);
        assert_eq!(route_local_to_remote(1, 4, 2), vec![1]);
        assert_eq!(route_local_to_remote(2, 4, 2), vec![0]);
        assert_eq!(route_local_to_remote(3, 4, 2), vec![1]);
    }

    #[test]
    fn test_route_one_to_many() {
        // Local TP=2, Remote TP=4
        assert_eq!(route_local_to_remote(0, 2, 4), vec![0, 1]);
        assert_eq!(route_local_to_remote(1, 2, 4), vec![2, 3]);
    }

    #[test]
    fn test_route_4_to_8() {
        // Local TP=4, Remote TP=8
        assert_eq!(route_local_to_remote(0, 4, 8), vec![0, 1]);
        assert_eq!(route_local_to_remote(1, 4, 8), vec![2, 3]);
        assert_eq!(route_local_to_remote(2, 4, 8), vec![4, 5]);
        assert_eq!(route_local_to_remote(3, 4, 8), vec![6, 7]);
    }

    #[test]
    fn test_route_non_divisible_remainder() {
        // Local TP=2, Remote TP=5: last local rank absorbs remainder
        assert_eq!(route_local_to_remote(0, 2, 5), vec![0, 1]);
        assert_eq!(route_local_to_remote(1, 2, 5), vec![2, 3, 4]);

        // Local TP=3, Remote TP=7: last rank gets extras
        assert_eq!(route_local_to_remote(0, 3, 7), vec![0, 1]);
        assert_eq!(route_local_to_remote(1, 3, 7), vec![2, 3]);
        assert_eq!(route_local_to_remote(2, 3, 7), vec![4, 5, 6]);
    }
}
