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
//! `lib/kv-router`, which implements [`WorkerView`] over worker selection's rows.
//!
//! Kernel-facing: the `ClassifyRequest` constructors, `ClassificationOverrides`,
//! `QueueMetadata`, the `PolicyQueue` scheduling methods, and `RequestClassifierRuntime` are
//! public for the kernel in another crate; hosts and plugins do not call them.

pub mod classifier;
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
    AdmissionConfigError, FamilyBucketTable, PolicyClassConfig, PolicyProfile, RouterQueuePolicy,
    SYNTHETIC_POLICY_CLASS,
};
pub use queue::{
    PolicyQueue, PolicyQueueEntry, PolicyQueueStats, QueueLimitKind, QueueMetadata, QueueRejection,
    QueueSnapshot,
};
