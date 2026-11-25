// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Admit, hold, or reject now: the Dynamo router's admission stage.
//!
//! The kernel classifies a request through [`RequestClassifierRuntime`] (which runs the host's
//! [`RequestClassifier`] plugin under panic capture and shutdown), resolves its policy class
//! against a [`PolicyProfile`], and holds it in a [`PolicyQueue`] until a worker has room or a
//! limit rejects it with a typed [`QueueRejection`]. [`RequestLifecycle`] is the per-request
//! state machine that feeds the plugin its events. Everything a plugin names lives here: the
//! [`RequestClassifier`] trait, [`ClassifyRequest`], [`ClassifyEvent`], and
//! [`RequestProgress`]. A classifier reads workers through worker selection's
//! `WorkerCandidates`, the view a filter or scorer reads, so the two plugin kinds share one
//! set of worker inputs. The YAML policy file, the plugin registry, and the gate that judges
//! "all eligible workers full" over the slot projection stay in `lib/router/kv-router`.
//!
//! Kernel-facing: `host::ClassifyRequestHostExt` (building a `ClassifyRequest` and reading
//! back what the classifier set), `ClassificationOverrides`, the `PolicyQueue` scheduling
//! methods (`extract_next`, `settle`, `take_expired`), and `RequestClassifierRuntime` are
//! public for the kernel in another crate; hosts and plugins do not call them. `QueueMetadata` is also what offline replay enqueues with.

mod classifier;
pub mod host;
mod lane_key;
mod profile;
mod queue;

pub use classifier::{
    AbortCause, ClassificationError, ClassificationOverrides, ClassifierError, ClassifyEvent,
    ClassifyRequest, DispatchDecision, DispatchGate, DispatchHead, DispatchRecheck,
    RequestClassifier, RequestClassifierContext, RequestClassifierRuntime, RequestLifecycle,
    RequestProgress, RequestProgressUpdater,
};
pub use lane_key::LaneKey;
pub use profile::{
    AdmissionConfigError, PolicyClassConfig, PolicyProfile, RouterQueuePolicy,
    SYNTHETIC_POLICY_CLASS, UnknownPolicyClassError,
};
pub use queue::{
    BlockReason, DispatchCharge, LaneCheck, PolicyQueue, PolicyQueueEntry, PolicyQueueStats,
    QueueLane, QueueLimitKind, QueueMetadata, QueueRejection, QueueSnapshot, RecheckScope,
};
