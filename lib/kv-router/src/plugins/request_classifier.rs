// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request classifier plugin contract: classify a request into a policy class before it
//! reaches the scheduler's queue.
//!
//! The stage lives in `dynamo-admission`; this module re-exports it at the paths plugins and
//! hosts already name and adds what only the composed router knows: the host's instance
//! config and the registry.

mod config;
mod registry;

pub use dynamo_admission::classifier::{
    AbortCause, ClassificationError, ClassificationOverrides, ClassifierError, ClassifyEvent,
    ClassifyFuture, ClassifyRequest, NoWorkers, RequestClassifier, RequestClassifierContext,
    RequestClassifierRuntime, RequestLifecycle, RequestProgress, RequestProgressUpdater,
    WorkerCacheView, WorkerLoadView, WorkerView,
};

pub(crate) use config::RawRequestClassifierConfig;
pub use config::RequestClassifierConfig;
pub(crate) use registry::RequestClassifierRegistry;
pub use registry::{
    RequestClassifierFactory, RequestClassifierParameters, RequestClassifierProvider,
    RequestClassifierProviderError, RequestClassifierRegistryError,
};
