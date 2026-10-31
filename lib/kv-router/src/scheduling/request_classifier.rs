// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The classifier runtime and request lifecycle: live in the admission stage.

pub use dynamo_admission::classifier::{
    AbortCause, ClassificationError, ClassificationOverrides, ClassifierError, ClassifyEvent,
    ClassifyFuture, ClassifyRequest, RequestClassifier, RequestClassifierContext,
    RequestClassifierRuntime, RequestLifecycle,
};
