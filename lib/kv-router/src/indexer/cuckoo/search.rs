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

pub(super) fn find_prefix_depths<const D: usize>(
    sequence_len: usize,
    initial_mask: u16,
    verification_window: usize,
    mut prefetch: impl FnMut(usize),
    mut probe: impl FnMut(usize) -> u16,
) -> [u32; D] {
    let mut depths = [0u32; D];
    if sequence_len == 0 {
        return depths;
    }

    debug_assert!((1..=MAX_LANES).contains(&D));
    debug_assert!((1..=MAX_VERIFICATION_WINDOW).contains(&verification_window));
    let configured_mask = lane_mask::<D>();
    let active = initial_mask & configured_mask & probe(0);
    if active == 0 {
        return depths;
    }

    let mut lower = [0usize; D];
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
    for (lane, &depth) in upper.iter().enumerate() {
        let lane_bit = 1u16 << lane;
        if active & lane_bit == 0 {
            continue;
        }
        for verify_position in depth.saturating_sub(verification_window)..depth {
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
            }
        }
        verifying &= !misses;
    }

    depths
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
