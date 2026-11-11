// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Admit, hold, or reject now: the Dynamo router's admission stage.
//!
//! The kernel classifies a request through [`RequestClassifierRuntime`] (which runs the host's
//! [`RequestClassifier`] plugin under panic capture and shutdown), resolves its policy class
//! against a [`PolicyProfile`], and holds it in a [`PolicyQueue`] until a worker has room or a
//! limit rejects it with a typed [`QueueRejection`]. [`RequestLifecycle`] is the per-request
//! state machine that feeds the plugin its events. Everything a plugin names lives here: the
//! [`RequestClassifier`] trait, [`ClassifyRequest`], [`ClassifyEvent`], the [`WorkerView`] it
//! reads workers through, and [`RequestProgress`]. The YAML policy file, the plugin registry,
//! and the gate that judges "all eligible workers full" over the slot projection stay in
//! `lib/router/kv-router`, which implements [`WorkerView`] over worker selection's rows.
//!
//! Kernel-facing: the `ClassifyRequest` constructors, `ClassificationOverrides`, the
//! `PolicyQueue` scheduling methods (`extract_next`, `settle`, `take_expired`), and
//! `RequestClassifierRuntime` are public for the kernel in another crate; hosts and plugins do
//! not call them. `QueueMetadata` is also what offline replay enqueues with.

pub mod classifier;
pub mod host;
pub mod placement;
pub mod profile;
pub mod queue;

pub use classifier::{
    AbortCause, ClassificationError, ClassificationOverrides, ClassifierError, ClassifyEvent,
    ClassifyFuture, ClassifyRequest, NoWorkers, RequestClassifier, RequestClassifierContext,
    RequestClassifierRuntime, RequestLifecycle, RequestProgress, RequestProgressUpdater,
    WorkerCacheView, WorkerLoadView, WorkerView,
};
pub use placement::WorkerPlacement;
pub use profile::{
    AdmissionConfigError, PolicyClassConfig, PolicyProfile, RouterQueuePolicy,
    SYNTHETIC_POLICY_CLASS, UnknownPolicyClassError,
};
pub use queue::{
    DispatchCharge, PolicyQueue, PolicyQueueEntry, PolicyQueueStats, QueueLimitKind, QueueMetadata,
    QueueRejection, QueueSnapshot,
};
