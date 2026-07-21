// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// Shared cursor state for monotonically increasing event streams.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CursorState {
    #[default]
    Initial,
    Live(u64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CursorObservation {
    Initial { got: u64 },
    Contiguous { got: u64 },
    Gap { expected: u64, got: u64 },
    Stale { got: u64, last_applied: Option<u64> },
}

impl CursorState {
    #[must_use]
    pub fn last_applied_id(self) -> Option<u64> {
        match self {
            CursorState::Initial => None,
            CursorState::Live(id) => Some(id),
        }
    }

    #[must_use]
    pub fn observe(self, got: u64) -> CursorObservation {
        match self {
            CursorState::Initial => CursorObservation::Initial { got },
            CursorState::Live(last) if got <= last => CursorObservation::Stale {
                got,
                last_applied: Some(last),
            },
            CursorState::Live(last) if got == last + 1 => CursorObservation::Contiguous { got },
            CursorState::Live(last) => CursorObservation::Gap {
                expected: last + 1,
                got,
            },
        }
    }

    #[must_use]
    pub fn advance_to(self, id: u64) -> Self {
        let _ = self;
        CursorState::Live(id)
    }
}

#[cfg(test)]
mod tests {
    use super::{CursorObservation, CursorState};

    #[test]
    fn initial_observation_preserves_first_id() {
        assert_eq!(
            CursorState::Initial.observe(0),
            CursorObservation::Initial { got: 0 }
        );
        assert_eq!(
            CursorState::Initial.observe(5),
            CursorObservation::Initial { got: 5 }
        );
    }

    #[test]
    fn live_observation_detects_contiguous_gap_and_stale_ids() {
        assert_eq!(
            CursorState::Live(10).observe(11),
            CursorObservation::Contiguous { got: 11 }
        );
        assert_eq!(
            CursorState::Live(10).observe(15),
            CursorObservation::Gap {
                expected: 11,
                got: 15,
            }
        );
        assert_eq!(
            CursorState::Live(10).observe(10),
            CursorObservation::Stale {
                got: 10,
                last_applied: Some(10),
            }
        );
        assert_eq!(
            CursorState::Live(10).observe(9),
            CursorObservation::Stale {
                got: 9,
                last_applied: Some(10),
            }
        );
    }
}
