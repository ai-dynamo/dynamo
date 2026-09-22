// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::local_data_parallel_range;

// Regression: ambiguous, out-of-bounds, or overflowing ownership can advertise
// unreachable DP ranks; validate the metadata before worker registration.
#[test]
fn local_dp_ownership_requires_a_valid_unambiguous_range() {
    for (global, start, local, expected) in [(2, 0, 0, 0..2), (8, 0, 4, 0..4), (8, 4, 4, 4..8)] {
        assert_eq!(
            local_data_parallel_range(global, start, local).unwrap(),
            expected
        );
    }
    for (global, start, local) in [
        (0, 0, 0),
        (8, 4, 0),
        (8, 0, 9),
        (8, 4, 5),
        (u32::MAX, u32::MAX - 1, 4),
    ] {
        assert!(
            local_data_parallel_range(global, start, local).is_err(),
            "invalid range: {start} + {local} of {global}"
        );
    }
}
