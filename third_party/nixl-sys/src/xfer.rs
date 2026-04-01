// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::*;

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum XferOp {
    Read = 0,
    Write = 1,
}

/// Methods used for estimating transfer costs
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum CostMethod {
    AnalyticalBackend = 0,
    Unknown = 1,
}

impl From<u32> for CostMethod {
    fn from(value: u32) -> Self {
        match value {
            0 => CostMethod::AnalyticalBackend,
            _ => CostMethod::Unknown,
        }
    }
}

/// A handle to a transfer request
pub struct XferRequest {
    inner: NonNull<bindings::nixl_capi_xfer_req_s>,
    agent: Arc<RwLock<AgentInner>>,
}

impl XferRequest {
    pub(crate) fn new(
        inner: NonNull<bindings::nixl_capi_xfer_req_s>,
        agent: Arc<RwLock<AgentInner>>,
    ) -> Self {
        Self { inner, agent }
    }

    pub(crate) fn handle(&self) -> *mut bindings::nixl_capi_xfer_req_s {
        self.inner.as_ptr()
    }

    /// Gets telemetry data for this transfer request
    ///
    /// # Returns
    /// Transfer telemetry data containing timing and performance metrics
    ///
    /// # Errors
    /// * `NoTelemetry`  - If telemetry is not enabled or transfer is not complete
    /// * `InvalidParam` - If the request handle is invalid
    /// * `BackendError` - If there was an error retrieving telemetry data
    pub fn get_telemetry(&self) -> Result<XferTelemetry, NixlError> {
        tracing::trace!("Getting transfer telemetry from request");
        let mut telemetry = bindings::nixl_capi_xfer_telemetry_s {
            start_time_us: 0,
            post_duration_us: 0,
            xfer_duration_us: 0,
            total_bytes: 0,
            desc_count: 0,
        };

        let status = unsafe {
            nixl_capi_get_xfer_telemetry(
                self.agent.write().unwrap().handle.as_ptr(),
                self.handle(),
                &mut telemetry,
            )
        };

        match status {
            NIXL_CAPI_SUCCESS => {
                tracing::trace!("Successfully retrieved transfer telemetry from request");
                Ok(XferTelemetry {
                    start_time_us: telemetry.start_time_us,
                    post_duration_us: telemetry.post_duration_us,
                    xfer_duration_us: telemetry.xfer_duration_us,
                    total_bytes: telemetry.total_bytes,
                    desc_count: telemetry.desc_count,
                })
            },
            NIXL_CAPI_IN_PROG => {
                tracing::error!(error = "transfer_not_complete", "Transfer not complete");
                Err(NixlError::NoTelemetry)
            },
            NIXL_CAPI_ERROR_NO_TELEMETRY => {
                tracing::error!(error = "telemetry_not_enabled", "Telemetry not enabled");
                Err(NixlError::NoTelemetry)
            },
            _ => {
                tracing::error!(error = "backend_error", "Failed to get transfer telemetry from request");
                Err(NixlError::BackendError)
            }
        }
    }
}

// SAFETY: XferRequest can be sent between threads safely
unsafe impl Send for XferRequest {}
// SAFETY: XferRequest can be shared between threads safely
unsafe impl Sync for XferRequest {}

impl Drop for XferRequest {
    fn drop(&mut self) {
        unsafe {
            bindings::nixl_capi_release_xfer_req(
                self.agent.write().unwrap().handle.as_ptr(),
                self.inner.as_ptr(),
            );

            bindings::nixl_capi_destroy_xfer_req(self.inner.as_ptr());
        }
    }
}
