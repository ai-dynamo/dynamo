// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod coordinator;
mod push_router;

pub(crate) use coordinator::affinity_id;
pub use coordinator::{
    AffinityAcquire, AffinityCoordinator, AffinityInitialization, AffinityLease, AffinityTarget,
    explicit_target,
};
pub use push_router::SessionAffinityPushRouter;

pub type LlmResponse =
    crate::types::Annotated<crate::protocols::common::llm_backend::LLMEngineOutput>;

#[cfg(test)]
mod tests;
