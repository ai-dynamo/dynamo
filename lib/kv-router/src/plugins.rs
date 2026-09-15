// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Registration and startup resolution of statically linked router plugins.

mod registry;

pub use registry::{
    DYN_ROUTER_DECODE_POLICY, DYN_ROUTER_PREFILL_POLICY, DYN_ROUTER_WORKER_SELECTION_POLICY,
    RouterPluginRegistry, WorkerSelectionPolicyParameters, WorkerSelectionPolicyProvider,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistry,
    WorkerSelectionPolicyRegistryError,
};

use crate::WorkerSelectionPolicyFactory;
use crate::scheduling::{RequestClassifierFactory, RequestClassifierRegistryError};

/// Configured factories shared across router construction, with fresh instances per router.
#[derive(Clone, Default)]
pub struct RouterPlugins {
    worker_selection: Option<WorkerSelectionPolicyFactory>,
    request_classifier: Option<RequestClassifierFactory>,
}

impl RouterPlugins {
    pub fn with_worker_selection(mut self, factory: WorkerSelectionPolicyFactory) -> Self {
        self.worker_selection = Some(factory);
        self
    }

    pub fn with_request_classifier(mut self, factory: RequestClassifierFactory) -> Self {
        self.request_classifier = Some(factory);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.worker_selection.is_none() && self.request_classifier.is_none()
    }

    pub fn worker_selection(&self) -> Option<&WorkerSelectionPolicyFactory> {
        self.worker_selection.as_ref()
    }

    pub fn request_classifier(&self) -> Option<&RequestClassifierFactory> {
        self.request_classifier.as_ref()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RouterPluginRegistryError {
    #[error(transparent)]
    WorkerSelection(#[from] WorkerSelectionPolicyRegistryError),
    #[error(transparent)]
    RequestClassifier(#[from] RequestClassifierRegistryError),
}
