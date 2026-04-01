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

mod query;
mod reg;
mod sync_manager;
mod xfer;
mod xfer_dlist_handle;

pub use query::{QueryResponse, QueryResponseIterator, QueryResponseList};
pub use reg::{RegDescList, RegDescriptor};
pub use sync_manager::{BackendSyncable, SyncManager};
pub use xfer::{XferDescList, XferDescriptor};
pub use xfer_dlist_handle::XferDlistHandle;

/// Memory types supported by NIXL
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemType {
    Dram,
    Vram,
    Block,
    Object,
    File,
    Unknown,
}

impl From<nixl_capi_mem_type_t> for MemType {
    fn from(mem_type: nixl_capi_mem_type_t) -> Self {
        match mem_type {
            0 => MemType::Dram,
            1 => MemType::Vram,
            2 => MemType::Block,
            3 => MemType::Object,
            4 => MemType::File,
            _ => MemType::Unknown,
        }
    }
}

impl fmt::Display for MemType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // SAFETY: We know the memory type is valid and the string will be available
        let mut str_ptr = ptr::null();
        unsafe {
            let mem_type = match self {
                MemType::Dram => 0,
                MemType::Vram => 1,
                MemType::Block => 2,
                MemType::Object => 3,
                MemType::File => 4,
                MemType::Unknown => 5,
            };
            nixl_capi_mem_type_to_string(mem_type, &mut str_ptr);
            let c_str = CStr::from_ptr(str_ptr);
            write!(f, "{}", c_str.to_str().unwrap())
        }
    }
}
