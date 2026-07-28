// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
use std::cell::RefCell;

use super::MAX_VERIFICATION_WINDOW;

// NOTE: Capacity pressure may leave stable holes and therefore non-monotone prefix evidence
// (hit, miss, later hit). Search is advisory and every exponential, binary, verification, and
// fallback loop remains structurally bounded by the input length/window. Do not add query retry,
// a seqlock, or consumer fencing to repair producer capacity omissions.

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

impl FallbackStats {
    #[cfg(test)]
    fn merge(&mut self, other: Self) {
        self.left_edge_lanes += other.left_edge_lanes;
        self.activated_lanes += other.activated_lanes;
        self.probe_calls += other.probe_calls;
        self.lane_probes += other.lane_probes;
        self.provenance_skips += other.provenance_skips;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PrefixSearchResult<const D: usize> {
    pub(super) depths: [u32; D],
    pub(super) fallback: FallbackStats,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PrefixSearchState<const D: usize> {
    active: u16,
    lower: [usize; D],
    lower_predecessor: [usize; D],
    upper: [usize; D],
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SearchPhase {
    Initial,
    Exponential,
    Binary,
    Verification,
    Fallback,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SearchTraceKind {
    Prefetch,
    Probe,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SearchTraceEvent {
    pub(super) kind: SearchTraceKind,
    pub(super) phase: SearchPhase,
    pub(super) position: usize,
    pub(super) lanes: u16,
}

#[cfg(test)]
thread_local! {
    static SEARCH_TRACE: RefCell<Option<Vec<SearchTraceEvent>>> = const { RefCell::new(None) };
}

#[cfg(test)]
fn record_trace(kind: SearchTraceKind, phase: SearchPhase, position: usize, lanes: u16) {
    SEARCH_TRACE.with_borrow_mut(|trace| {
        if let Some(trace) = trace {
            trace.push(SearchTraceEvent {
                kind,
                phase,
                position,
                lanes,
            });
        }
    });
}

pub(super) fn find_prefix_depths<const D: usize>(
    sequence_len: usize,
    initial_mask: u16,
    verification_window: usize,
    prefetch: impl FnMut(usize),
    probe: impl FnMut(usize) -> u16,
) -> [u32; D] {
    find_prefix_depths_impl::<D, true, false>(
        sequence_len,
        initial_mask,
        verification_window,
        prefetch,
        probe,
    )
    .depths
}

#[cfg(test)]
pub(super) fn find_max_depth_matches<const D: usize>(
    sequence_len: usize,
    initial_mask: u16,
    verification_window: usize,
    prefetch: impl FnMut(usize),
    probe: impl FnMut(usize) -> u16,
) -> [u32; D] {
    find_max_depth_matches_impl::<D, true, false>(
        sequence_len,
        initial_mask,
        verification_window,
        prefetch,
        probe,
    )
    .depths
}

#[cfg(test)]
pub(super) fn fixed_window_prefix_depths<const D: usize>(
    sequence_len: usize,
    initial_mask: u16,
    verification_window: usize,
    prefetch: impl FnMut(usize),
    probe: impl FnMut(usize) -> u16,
) -> [u32; D] {
    find_prefix_depths_impl::<D, false, false>(
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
    find_prefix_depths_impl::<D, true, false>(
        sequence_len,
        initial_mask,
        verification_window,
        prefetch,
        probe,
    )
}

#[cfg(test)]
pub(super) fn find_prefix_depths_with_test_trace<const D: usize>(
    sequence_len: usize,
    initial_mask: u16,
    verification_window: usize,
    prefetch: impl FnMut(usize),
    probe: impl FnMut(usize) -> u16,
) -> (PrefixSearchResult<D>, Vec<SearchTraceEvent>) {
    trace_search(|| {
        find_prefix_depths_impl::<D, true, true>(
            sequence_len,
            initial_mask,
            verification_window,
            prefetch,
            probe,
        )
    })
}

#[cfg(test)]
pub(super) fn find_max_depth_matches_with_test_trace<const D: usize>(
    sequence_len: usize,
    initial_mask: u16,
    verification_window: usize,
    prefetch: impl FnMut(usize),
    probe: impl FnMut(usize) -> u16,
) -> (PrefixSearchResult<D>, Vec<SearchTraceEvent>) {
    trace_search(|| {
        find_max_depth_matches_impl::<D, true, true>(
            sequence_len,
            initial_mask,
            verification_window,
            prefetch,
            probe,
        )
    })
}

#[cfg(test)]
pub(super) fn refine_binary_level_with_test_trace<const D: usize>(
    lower: [usize; D],
    upper: [usize; D],
    selected: u16,
    eligible: u16,
    probe: impl FnMut(usize) -> u16,
) -> ([usize; D], [usize; D], Vec<SearchTraceEvent>) {
    let mut state = PrefixSearchState {
        active: eligible,
        lower,
        lower_predecessor: [0; D],
        upper,
    };
    let (_, trace) = trace_search(|| {
        let mut probe = probe;
        assert!(refine_binary_level::<D, true>(
            &mut state,
            selected,
            eligible,
            &mut |_| {},
            &mut probe,
        ));
        empty_result::<D>()
    });
    (state.lower, state.upper, trace)
}

#[cfg(test)]
fn trace_search<const D: usize>(
    search: impl FnOnce() -> PrefixSearchResult<D>,
) -> (PrefixSearchResult<D>, Vec<SearchTraceEvent>) {
    SEARCH_TRACE.with_borrow_mut(|trace| {
        assert!(trace.is_none(), "nested CKF search tracing is unsupported");
        *trace = Some(Vec::new());
    });
    let result = search();
    let trace = SEARCH_TRACE.with_borrow_mut(|trace| {
        trace
            .take()
            .expect("CKF search trace must remain active through the search")
    });
    (result, trace)
}

fn find_prefix_depths_impl<const D: usize, const FALLBACK: bool, const TRACE: bool>(
    sequence_len: usize,
    initial_mask: u16,
    verification_window: usize,
    mut prefetch: impl FnMut(usize),
    mut probe: impl FnMut(usize) -> u16,
) -> PrefixSearchResult<D> {
    if sequence_len == 0 {
        return empty_result();
    }

    debug_assert!((1..=MAX_LANES).contains(&D));
    debug_assert!((1..=MAX_VERIFICATION_WINDOW).contains(&verification_window));
    let mut state =
        initialize_search::<D, TRACE>(sequence_len, initial_mask, &mut prefetch, &mut probe);
    if state.active == 0 {
        return empty_result();
    }

    let active = state.active;
    while refine_binary_level::<D, TRACE>(&mut state, active, active, &mut prefetch, &mut probe) {}

    finalize_lanes::<D, FALLBACK, TRACE>(
        &state,
        active,
        verification_window,
        &mut prefetch,
        &mut probe,
    )
}

#[cfg(test)]
fn find_max_depth_matches_impl<const D: usize, const FALLBACK: bool, const TRACE: bool>(
    sequence_len: usize,
    initial_mask: u16,
    verification_window: usize,
    mut prefetch: impl FnMut(usize),
    mut probe: impl FnMut(usize) -> u16,
) -> PrefixSearchResult<D> {
    if sequence_len == 0 {
        return empty_result();
    }

    debug_assert!((1..=MAX_LANES).contains(&D));
    debug_assert!((1..=MAX_VERIFICATION_WINDOW).contains(&verification_window));
    let mut state =
        initialize_search::<D, TRACE>(sequence_len, initial_mask, &mut prefetch, &mut probe);
    if state.active == 0 {
        return empty_result();
    }
    // Maximum-only search is not universally faster: equal ceilings provide no frontier to
    // prune. Reuse the full scheduler for that case; see PR #11435 for workload benchmarks.
    if active_lanes_share_ceiling(&state) {
        let active = state.active;
        while refine_binary_level::<D, TRACE>(&mut state, active, active, &mut prefetch, &mut probe)
        {
        }
        return project_max_depths(finalize_lanes::<D, FALLBACK, TRACE>(
            &state,
            active,
            verification_window,
            &mut prefetch,
            &mut probe,
        ));
    }
    let mut unresolved = state.active;
    let mut best_depth = 0u32;
    let mut winners = 0u16;
    let mut fallback = FallbackStats::default();

    while unresolved != 0 {
        for lane in 0..D {
            let lane_bit = 1u16 << lane;
            if unresolved & lane_bit != 0 && score_bound(state.upper[lane]) < best_depth {
                unresolved &= !lane_bit;
            }
        }
        if unresolved == 0 {
            break;
        }

        let frontier_ceiling = (0..D)
            .filter(|&lane| unresolved & (1u16 << lane) != 0)
            .map(|lane| score_bound(state.upper[lane]))
            .max()
            .expect("at least one unresolved CKF lane");
        let mut frontier = 0u16;
        let mut terminal = 0u16;
        for lane in 0..D {
            let lane_bit = 1u16 << lane;
            if unresolved & lane_bit == 0 || score_bound(state.upper[lane]) != frontier_ceiling {
                continue;
            }
            frontier |= lane_bit;
            if state.upper[lane] - state.lower[lane] <= 1 {
                terminal |= lane_bit;
            }
        }

        if terminal != 0 {
            let result = finalize_lanes::<D, FALLBACK, TRACE>(
                &state,
                terminal,
                verification_window,
                &mut prefetch,
                &mut probe,
            );
            fallback.merge(result.fallback);
            for (lane, &depth) in result.depths.iter().enumerate() {
                let lane_bit = 1u16 << lane;
                if terminal & lane_bit == 0 || depth == 0 {
                    continue;
                }
                if depth > best_depth {
                    best_depth = depth;
                    winners = lane_bit;
                } else if depth == best_depth {
                    winners |= lane_bit;
                }
            }
            unresolved &= !terminal;
            continue;
        }

        let refined = refine_binary_level::<D, TRACE>(
            &mut state,
            frontier,
            unresolved,
            &mut prefetch,
            &mut probe,
        );
        debug_assert!(refined, "nonterminal CKF frontier must have a midpoint");
    }

    let mut depths = [0u32; D];
    if best_depth != 0 {
        for (lane, depth) in depths.iter_mut().enumerate() {
            if winners & (1u16 << lane) != 0 {
                *depth = best_depth;
            }
        }
    }
    PrefixSearchResult { depths, fallback }
}

#[cfg(test)]
fn active_lanes_share_ceiling<const D: usize>(state: &PrefixSearchState<D>) -> bool {
    let mut ceiling = None;
    for lane in 0..D {
        if state.active & (1u16 << lane) == 0 {
            continue;
        }
        let lane_ceiling = score_bound(state.upper[lane]);
        if ceiling.is_some_and(|ceiling| ceiling != lane_ceiling) {
            return false;
        }
        ceiling = Some(lane_ceiling);
    }
    true
}

#[cfg(test)]
fn project_max_depths<const D: usize>(mut result: PrefixSearchResult<D>) -> PrefixSearchResult<D> {
    let best_depth = result.depths.iter().copied().max().unwrap_or(0);
    for depth in &mut result.depths {
        if *depth != best_depth {
            *depth = 0;
        }
    }
    result
}

fn initialize_search<const D: usize, const TRACE: bool>(
    sequence_len: usize,
    initial_mask: u16,
    prefetch: &mut impl FnMut(usize),
    probe: &mut impl FnMut(usize) -> u16,
) -> PrefixSearchState<D> {
    let configured = initial_mask & lane_mask::<D>();
    #[cfg(test)]
    if TRACE {
        record_trace(SearchTraceKind::Probe, SearchPhase::Initial, 0, configured);
    }
    let active = configured & probe(0);
    let mut state = PrefixSearchState {
        active,
        lower: [0; D],
        lower_predecessor: [0; D],
        upper: [sequence_len; D],
    };
    if active == 0 {
        return state;
    }

    let mut sampling = active;
    let mut position = 1usize;
    if position < sequence_len {
        #[cfg(test)]
        if TRACE {
            record_trace(
                SearchTraceKind::Prefetch,
                SearchPhase::Exponential,
                position,
                sampling,
            );
        }
        prefetch(position);
    }

    while position < sequence_len && sampling != 0 {
        if let Some(next) = position.checked_mul(2).filter(|&next| next < sequence_len) {
            #[cfg(test)]
            if TRACE {
                record_trace(
                    SearchTraceKind::Prefetch,
                    SearchPhase::Exponential,
                    next,
                    sampling,
                );
            }
            prefetch(next);
        }
        #[cfg(test)]
        if TRACE {
            record_trace(
                SearchTraceKind::Probe,
                SearchPhase::Exponential,
                position,
                sampling,
            );
        }
        let hits = probe(position);
        let hit_lanes = sampling & hits;
        let miss_lanes = sampling & !hits;
        for lane in 0..D {
            let lane_bit = 1u16 << lane;
            if hit_lanes & lane_bit != 0 {
                state.lower_predecessor[lane] = state.lower[lane];
                state.lower[lane] = position;
            } else if miss_lanes & lane_bit != 0 {
                state.upper[lane] = position;
            }
        }
        sampling &= hits;

        let Some(next) = position.checked_mul(2) else {
            break;
        };
        position = next;
    }

    state
}

fn refine_binary_level<const D: usize, const TRACE: bool>(
    state: &mut PrefixSearchState<D>,
    selected: u16,
    eligible: u16,
    prefetch: &mut impl FnMut(usize),
    probe: &mut impl FnMut(usize) -> u16,
) -> bool {
    let mut groups = [ProbeGroup::default(); MAX_LANES];
    let mut group_count = 0usize;
    for lane in 0..D {
        let lane_bit = 1u16 << lane;
        if eligible & lane_bit == 0 || state.upper[lane] - state.lower[lane] <= 1 {
            continue;
        }
        let midpoint = state.lower[lane] + (state.upper[lane] - state.lower[lane]) / 2;
        add_group(&mut groups, &mut group_count, midpoint, lane_bit);
    }

    let has_selected_group = groups[..group_count]
        .iter()
        .any(|group| group.lanes & selected != 0);
    if !has_selected_group {
        return false;
    }

    for group in groups[..group_count]
        .iter()
        .filter(|group| group.lanes & selected != 0)
    {
        #[cfg(test)]
        if TRACE {
            record_trace(
                SearchTraceKind::Prefetch,
                SearchPhase::Binary,
                group.position,
                group.lanes,
            );
        }
        prefetch(group.position);
    }
    for group in groups[..group_count]
        .iter()
        .filter(|group| group.lanes & selected != 0)
    {
        #[cfg(test)]
        if TRACE {
            record_trace(
                SearchTraceKind::Probe,
                SearchPhase::Binary,
                group.position,
                group.lanes,
            );
        }
        let hits = probe(group.position);
        for lane in 0..D {
            let lane_bit = 1u16 << lane;
            if group.lanes & lane_bit == 0 {
                continue;
            }
            if hits & lane_bit != 0 {
                state.lower_predecessor[lane] = state.lower[lane];
                state.lower[lane] = group.position;
            } else {
                state.upper[lane] = group.position;
            }
        }
    }
    true
}

fn finalize_lanes<const D: usize, const FALLBACK: bool, const TRACE: bool>(
    state: &PrefixSearchState<D>,
    lanes: u16,
    verification_window: usize,
    prefetch: &mut impl FnMut(usize),
    probe: &mut impl FnMut(usize) -> u16,
) -> PrefixSearchResult<D> {
    let mut depths = [0u32; D];
    let mut fallback = FallbackStats::default();
    for (lane, depth) in depths.iter_mut().enumerate() {
        if lanes & (1u16 << lane) != 0 {
            *depth = score_bound(state.upper[lane]);
        }
    }

    let mut verification_groups = [ProbeGroup::default(); MAX_VERIFICATION_GROUPS];
    let mut verification_group_count = 0usize;
    let mut verification_start = [0usize; D];
    for (lane, &depth) in state.upper.iter().enumerate() {
        let lane_bit = 1u16 << lane;
        if lanes & lane_bit == 0 {
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

    let mut verifying = lanes;
    let mut fallback_lanes = 0u16;
    for group in &verification_groups[..verification_group_count] {
        let participants = group.lanes & verifying;
        if participants == 0 {
            continue;
        }
        #[cfg(test)]
        if TRACE {
            record_trace(
                SearchTraceKind::Prefetch,
                SearchPhase::Verification,
                group.position,
                participants,
            );
        }
        prefetch(group.position);
        #[cfg(test)]
        if TRACE {
            record_trace(
                SearchTraceKind::Probe,
                SearchPhase::Verification,
                group.position,
                participants,
            );
        }
        let misses = participants & !probe(group.position);
        for (lane, depth) in depths.iter_mut().enumerate() {
            let lane_bit = 1u16 << lane;
            if misses & lane_bit != 0 {
                *depth = score_bound(group.position);
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
        let start = state.lower_predecessor[lane].saturating_add(1);
        let end = verification_start[lane];
        if state.lower_predecessor[lane] >= end {
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

        #[cfg(test)]
        if TRACE {
            record_trace(
                SearchTraceKind::Prefetch,
                SearchPhase::Fallback,
                position,
                participants,
            );
        }
        prefetch(position);
        #[cfg(test)]
        if TRACE {
            record_trace(
                SearchTraceKind::Probe,
                SearchPhase::Fallback,
                position,
                participants,
            );
        }
        let misses = participants & !probe(position);
        fallback.probe_calls += 1;
        fallback.lane_probes += u64::from(participants.count_ones());
        for (lane, depth) in depths.iter_mut().enumerate() {
            let lane_bit = 1u16 << lane;
            if participants & lane_bit == 0 {
                continue;
            }
            if misses & lane_bit != 0 {
                *depth = score_bound(position);
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
                *depth = score_bound(position + 1);
            }
        }
    }
    depths
}

fn empty_result<const D: usize>() -> PrefixSearchResult<D> {
    PrefixSearchResult {
        depths: [0; D],
        fallback: FallbackStats::default(),
    }
}

fn score_bound(position: usize) -> u32 {
    position.min(u32::MAX as usize) as u32
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

#[cfg(test)]
mod capacity_hole_tests {
    use super::*;

    #[test]
    fn stable_non_monotone_capacity_holes_remain_bounded() {
        const SEQUENCE_LEN: usize = 64;
        let (result, trace) = find_prefix_depths_with_test_trace::<1>(
            SEQUENCE_LEN,
            1,
            2,
            |_| {},
            |position| u16::from(matches!(position, 0 | 4 | 16 | 32)),
        );

        assert!(usize::try_from(result.depths[0]).unwrap() <= SEQUENCE_LEN);
        assert!(trace.iter().all(|event| event.position < SEQUENCE_LEN));
        let probe_count = trace
            .iter()
            .filter(|event| event.kind == SearchTraceKind::Probe)
            .count();
        assert!(probe_count <= SEQUENCE_LEN * 2 + MAX_VERIFICATION_WINDOW * 2);
    }
}
