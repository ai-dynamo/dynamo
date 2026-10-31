// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What an in-tree host names from the admission stage: the classifier runtime it drives, the
//! queue rejection it renders, the queue policy and profiles it configures, and the queue
//! snapshot replay reads. Every item is re-exported from its owning module; this list is the
//! host contract (hosts inventory v2, section 3).

pub use crate::classifier::{AbortCause, RequestLifecycle, RequestProgress, WorkerView};
pub use crate::placement::WorkerPlacement;
pub use crate::profile::{PolicyClassConfig, PolicyProfile, RouterQueuePolicy};
pub use crate::queue::{PolicyQueue, QueueRejection, QueueSnapshot};

// What the frontend host (lib/llm) names beyond the lists above.
pub use crate::queue::QueueLimitKind;
