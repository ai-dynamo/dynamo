// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod occupancy;
mod picker;
mod types;

pub(crate) use occupancy::get_or_create_routing_occupancy_state;
pub use occupancy::{
    OccupancyPolicy, OccupancyReservation, OccupancySelection, RoutingOccupancyState,
};
pub(crate) use picker::RoutePicker;
pub use types::RouteTarget;
pub(crate) use types::{
    AdmissionKind, CandidateView, RouteCandidate, RouteContext, RouteDecision, RouteDevice,
    RoutePolicy,
};
