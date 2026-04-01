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

pub struct XferDlistHandle {
    inner: *mut bindings::nixl_capi_xfer_dlist_handle_s,
    agent: NonNull<bindings::nixl_capi_agent_s>
}

impl XferDlistHandle {
    pub fn new(inner: *mut bindings::nixl_capi_xfer_dlist_handle_s,
                      agent: NonNull<bindings::nixl_capi_agent_s>) -> Self {
        Self { inner, agent }
    }

    pub fn handle(&self) -> *mut bindings::nixl_capi_xfer_dlist_handle_s {
        self.inner
    }
}

impl Drop for XferDlistHandle {
    fn drop(&mut self) {
        unsafe {
            nixl_capi_release_xfer_dlist_handle(self.agent.as_ptr(),
                                               self.handle());
        }
    }
}