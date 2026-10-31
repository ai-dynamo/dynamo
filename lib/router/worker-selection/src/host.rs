// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What an in-tree host names from the worker-selection stage beyond the plugin contract it
//! reaches through `dynamo_kv_router::plugins::prelude`: the configuration slices, the
//! conditional-disaggregation decision inputs, the LoRA filter, and the trace types it renders.
//! Every item is re-exported from its owning module; this list is the host contract (hosts
//! inventory v2, section 3).

pub use crate::conditional_disagg::{
    ConditionalDisaggConfig, ConditionalDisaggDecisionInput, ConditionalDisaggPolicyKind,
};
pub use crate::error::WorkerSelectionPolicyError;
pub use crate::lora_filter::LoraWorkerFilter;
pub use crate::policy::WorkerSelectionPolicy;
pub use crate::runtime::RuntimePolicyError;
pub use crate::selection_config::{SelectionConfig, SharedCacheType};
pub use crate::shared_cache::SharedCacheHits;
pub use crate::trace::{SelectionTrace, SelectionTraceSpec};
