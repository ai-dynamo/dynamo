// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for request status.

use dynamo_kvbm::v2::integrations::scheduler::RequestStatus;
use pyo3::prelude::*;

/// Python wrapper for RequestStatus.
///
/// This wraps the real `RequestStatus` from `dynamo_kvbm::v2::integrations::scheduler`.
///
/// Example:
///     status = RequestStatus.finished_stopped()
///     scheduler.finish_requests(["req-1"], status)
#[pyclass(name = "RequestStatus")]
#[derive(Clone)]
pub struct PyRequestStatus {
    pub(crate) inner: RequestStatus,
}

#[pymethods]
impl PyRequestStatus {
    /// Create a Waiting status.
    #[staticmethod]
    pub fn waiting() -> Self {
        Self {
            inner: RequestStatus::Waiting,
        }
    }

    /// Create a Running status.
    #[staticmethod]
    pub fn running() -> Self {
        Self {
            inner: RequestStatus::Running,
        }
    }

    /// Create a Preempted status.
    #[staticmethod]
    pub fn preempted() -> Self {
        Self {
            inner: RequestStatus::Preempted,
        }
    }

    /// Create a FinishedStopped status.
    #[staticmethod]
    pub fn finished_stopped() -> Self {
        Self {
            inner: RequestStatus::FinishedStopped,
        }
    }

    /// Create a FinishedAborted status.
    #[staticmethod]
    pub fn finished_aborted() -> Self {
        Self {
            inner: RequestStatus::FinishedAborted,
        }
    }

    /// Create a FinishedLengthCapped status.
    #[staticmethod]
    pub fn finished_length_capped() -> Self {
        Self {
            inner: RequestStatus::FinishedLengthCapped,
        }
    }

    /// Check if the status is a finished state.
    pub fn is_finished(&self) -> bool {
        self.inner.is_finished()
    }

    fn __repr__(&self) -> String {
        format!("RequestStatus::{:?}", self.inner)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}
