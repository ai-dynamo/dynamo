// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod coordinator;
mod push_router;
mod trace;

pub(crate) use coordinator::{AffinityAcquire, affinity_id};
pub use coordinator::{AffinityCoordinator, AffinityTarget, explicit_target};
pub use push_router::SessionAffinityPushRouter;

pub const MAX_SESSION_AFFINITY_TTL_SECS: u64 = 31_536_000;
pub const MAX_SESSION_AFFINITY_ENTRIES: usize = 65_536;
pub const MAX_SESSION_AFFINITY_ID_BYTES: usize = 256;

pub type LlmResponse =
    crate::types::Annotated<crate::protocols::common::llm_backend::LLMEngineOutput>;

#[cfg(test)]
mod tests;
