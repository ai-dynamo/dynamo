// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::MAX_VERIFICATION_WINDOW;

const MAX_LANES: usize = 16;
const MAX_VERIFICATION_GROUPS: usize = MAX_LANES * MAX_VERIFICATION_WINDOW;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ProbeGroup {
    position: usize,
    lanes: u16,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct FallbackStats {
    pub(super) left_edge_lanes: u64,
    pub(super) activated_lanes: u64,
    pub(super) probe_calls: u64,
    pub(super) lane_probes: u64,
    pub(super) provenance_skips: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PrefixSearchResult<const D: usize> {
    pub(super) depths: [u32; D],
    pub(super) fallback: FallbackStats,
}

#[cfg(any(not(feature = "metrics"), test))]
pub(super) fn find_prefix_depths<const D: usize>(
    sequence_len: usize,
    initial_mask: u16,
    verification_window: usize,
    prefetch: impl FnMut(usize),
    probe: impl FnMut(usize) -> u16,
) -> [u32; D] {
    find_prefix_depths_impl::<D, true>(
        sequence_len,
        initial_mask,
        verification_window,
        prefetch,
        probe,
    )
    .depths
}

#[cfg(feature = "metrics")]
pub(super) fn find_prefix_depths_with_stats<const D: usize>(
    sequence_len: usize,
    initial_mask: u16,
    verification_window: usize,
    prefetch: impl FnMut(usize),
    probe: impl FnMut(usize) -> u16,
) -> PrefixSearchResult<D> {
    find_prefix_depths_impl::<D, true>(
        sequence_len,
        initial_mask,
        verification_window,
        prefetch,
        probe,
    )
}

#[cfg(test)]
pub(super) fn fixed_window_prefix_depths<const D: usize>(
    sequence_len: usize,
    initial_mask: u16,
    verification_window: usize,
    prefetch: impl FnMut(usize),
    probe: impl FnMut(usize) -> u16,
) -> [u32; D] {
    find_prefix_depths_impl::<D, false>(
        sequence_len,
        initial_mask,
        verification_window,
        prefetch,
        probe,
    )
    .depths
}

#[cfg(test)]
pub(super) fn find_prefix_depths_with_test_stats<const D: usize>(
    sequence_len: usize,
    initial_mask: u16,
    verification_window: usize,
    prefetch: impl FnMut(usize),
    probe: impl FnMut(usize) -> u16,
) -> PrefixSearchResult<D> {
    find_prefix_depths_impl::<D, true>(
        sequence_len,
        initial_mask,
        verification_window,
        prefetch,
        probe,
    )
}

fn find_prefix_depths_impl<const D: usize, const FALLBACK: bool>(
    sequence_len: usize,
    initial_mask: u16,
    verification_window: usize,
    mut prefetch: impl FnMut(usize),
    mut probe: impl FnMut(usize) -> u16,
) -> PrefixSearchResult<D> {
    let mut depths = [0u32; D];
    let mut fallback = FallbackStats::default();
    if sequence_len == 0 {
        return PrefixSearchResult { depths, fallback };
    }

    debug_assert!((1..=MAX_LANES).contains(&D));
    debug_assert!((1..=MAX_VERIFICATION_WINDOW).contains(&verification_window));
    let configured_mask = lane_mask::<D>();
    let active = initial_mask & configured_mask & probe(0);
    if active == 0 {
        return PrefixSearchResult { depths, fallback };
    }

    let mut lower = [0usize; D];
    let mut lower_predecessor = [0usize; D];
    let mut upper = [sequence_len; D];
    let mut sampling = active;
    let mut position = 1usize;
    if position < sequence_len {
        prefetch(position);
    }

    while position < sequence_len && sampling != 0 {
        if let Some(next) = position.checked_mul(2).filter(|&next| next < sequence_len) {
            prefetch(next);
        }
        let hits = probe(position);
        let hit_lanes = sampling & hits;
        let miss_lanes = sampling & !hits;
        for lane in 0..D {
            let lane_bit = 1u16 << lane;
            if hit_lanes & lane_bit != 0 {
                lower_predecessor[lane] = lower[lane];
                lower[lane] = position;
            } else if miss_lanes & lane_bit != 0 {
                upper[lane] = position;
            }
        }
        sampling &= hits;

        let Some(next) = position.checked_mul(2) else {
            break;
        };
        position = next;
    }

    loop {
        let mut groups = [ProbeGroup::default(); MAX_LANES];
        let mut group_count = 0usize;
        for lane in 0..D {
            let lane_bit = 1u16 << lane;
            if active & lane_bit == 0 || upper[lane] - lower[lane] <= 1 {
                continue;
            }
            let midpoint = lower[lane] + (upper[lane] - lower[lane]) / 2;
            add_group(&mut groups, &mut group_count, midpoint, lane_bit);
        }
        if group_count == 0 {
            break;
        }

        for group in &groups[..group_count] {
            prefetch(group.position);
        }
        for group in &groups[..group_count] {
            let hits = probe(group.position);
            for lane in 0..D {
                let lane_bit = 1u16 << lane;
                if group.lanes & lane_bit == 0 {
                    continue;
                }
                if hits & lane_bit != 0 {
                    lower_predecessor[lane] = lower[lane];
                    lower[lane] = group.position;
                } else {
                    upper[lane] = group.position;
                }
            }
        }
    }

    for (lane, (depth, &upper_bound)) in depths.iter_mut().zip(&upper).enumerate() {
        if active & (1u16 << lane) != 0 {
            *depth = upper_bound.min(u32::MAX as usize) as u32;
        }
    }

    let mut verification_groups = [ProbeGroup::default(); MAX_VERIFICATION_GROUPS];
    let mut verification_group_count = 0usize;
    let mut verification_start = [0usize; D];
    for (lane, &depth) in upper.iter().enumerate() {
        let lane_bit = 1u16 << lane;
        if active & lane_bit == 0 {
            continue;
        }
        let start = depth.saturating_sub(verification_window);
        verification_start[lane] = start;
        for verify_position in start..depth {
            add_group(
                &mut verification_groups,
                &mut verification_group_count,
                verify_position,
                lane_bit,
            );
        }
    }
    verification_groups[..verification_group_count].sort_unstable_by_key(|group| group.position);

    let mut verifying = active;
    let mut fallback_lanes = 0u16;
    for group in &verification_groups[..verification_group_count] {
        let participants = group.lanes & verifying;
        if participants == 0 {
            continue;
        }
        prefetch(group.position);
        let misses = participants & !probe(group.position);
        for (lane, depth) in depths.iter_mut().enumerate() {
            let lane_bit = 1u16 << lane;
            if misses & lane_bit != 0 {
                *depth = group.position.min(u32::MAX as usize) as u32;
                if group.position == verification_start[lane] {
                    fallback_lanes |= lane_bit;
                }
            }
        }
        verifying &= !misses;
    }

    if !FALLBACK {
        return PrefixSearchResult { depths, fallback };
    }
    fallback.left_edge_lanes = u64::from(fallback_lanes.count_ones());

    // A miss at the window's left edge can expose a false terminal branch. Scan only
    // the previously unexamined gap after the terminal lower bound's predecessor.
    let mut fallback_next = [0usize; D];
    let mut fallback_end = [0usize; D];
    for lane in 0..D {
        let lane_bit = 1u16 << lane;
        if fallback_lanes & lane_bit == 0 {
            continue;
        }
        let start = lower_predecessor[lane].saturating_add(1);
        let end = verification_start[lane];
        if lower_predecessor[lane] >= end {
            fallback.provenance_skips += 1;
            fallback_lanes &= !lane_bit;
            continue;
        }
        if start == end {
            fallback_lanes &= !lane_bit;
            continue;
        }
        fallback_next[lane] = start;
        fallback_end[lane] = end;
    }
    fallback.activated_lanes = u64::from(fallback_lanes.count_ones());
    if fallback_lanes == 0 {
        return PrefixSearchResult { depths, fallback };
    }

    while fallback_lanes != 0 {
        let position = (0..D)
            .filter(|&lane| fallback_lanes & (1u16 << lane) != 0)
            .map(|lane| fallback_next[lane])
            .min()
            .expect("at least one fallback lane");
        let mut participants = 0u16;
        for (lane, next) in fallback_next.iter().enumerate() {
            let lane_bit = 1u16 << lane;
            if fallback_lanes & lane_bit != 0 && *next == position {
                participants |= lane_bit;
            }
        }

        prefetch(position);
        let misses = participants & !probe(position);
        fallback.probe_calls += 1;
        fallback.lane_probes += u64::from(participants.count_ones());
        for (lane, depth) in depths.iter_mut().enumerate() {
            let lane_bit = 1u16 << lane;
            if participants & lane_bit == 0 {
                continue;
            }
            if misses & lane_bit != 0 {
                *depth = position.min(u32::MAX as usize) as u32;
                fallback_lanes &= !lane_bit;
                continue;
            }
            fallback_next[lane] += 1;
            if fallback_next[lane] == fallback_end[lane] {
                fallback_lanes &= !lane_bit;
            }
        }
    }

    PrefixSearchResult { depths, fallback }
}

#[cfg(any(test, feature = "bench"))]
#[allow(dead_code)]
pub(super) fn linear_prefix_depths<const D: usize>(
    sequence_len: usize,
    initial_mask: u16,
    mut probe: impl FnMut(usize) -> u16,
) -> [u32; D] {
    let mut depths = [0u32; D];
    let mut active = initial_mask & lane_mask::<D>();
    for position in 0..sequence_len {
        if active == 0 {
            break;
        }
        active &= probe(position);
        for (lane, depth) in depths.iter_mut().enumerate() {
            if active & (1u16 << lane) != 0 {
                *depth = (position + 1).min(u32::MAX as usize) as u32;
            }
        }
    }
    depths
}

fn add_group<const N: usize>(
    groups: &mut [ProbeGroup; N],
    group_count: &mut usize,
    position: usize,
    lane: u16,
) {
    if let Some(group) = groups[..*group_count]
        .iter_mut()
        .find(|group| group.position == position)
    {
        group.lanes |= lane;
        return;
    }

    debug_assert!(*group_count < N);
    groups[*group_count] = ProbeGroup {
        position,
        lanes: lane,
    };
    *group_count += 1;
}

const fn lane_mask<const D: usize>() -> u16 {
    if D >= u16::BITS as usize {
        u16::MAX
    } else {
        (1u16 << D) - 1
    }
}
