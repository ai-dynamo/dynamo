// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#[cfg(feature = "mistralrs")]
pub mod mistralrs;

#[cfg(feature = "sglang")]
pub mod sglang;

#[cfg(feature = "llamacpp")]
pub mod llamacpp;

#[cfg(feature = "vllm")]
pub mod vllm;

#[cfg(feature = "trtllm")]
pub mod trtllm;

#[cfg(feature = "python")]
pub mod python;

#[derive(Debug, Clone)]
pub struct MultiNodeConfig {
    /// How many nodes / hosts we are using
    pub num_nodes: u32,
    /// Unique consecutive integer to identify this node
    pub node_rank: u32,
    /// host:port of head / control node
    pub leader_addr: String,
}

impl Default for MultiNodeConfig {
    fn default() -> Self {
        MultiNodeConfig {
            num_nodes: 1,
            node_rank: 0,
            leader_addr: "".to_string(),
        }
    }
}

#[cfg(any(feature = "sglang", feature = "vllm", feature = "python"))]
use pyo3::prelude::*;

/// On Mac embedded Python interpreters do not pick up the virtual env.
#[cfg(all(
    target_os = "macos",
    any(feature = "sglang", feature = "vllm", feature = "python")
))]
fn fix_venv(venv: String, py: pyo3::Python<'_>) -> anyhow::Result<()> {
    let version_info = py.version_info();
    let sys: PyObject = py.import("sys")?.into();
    let sys_path = sys.getattr(py, "path")?;
    let venv_path = format!(
        "{venv}/lib/python{}.{}/site-packages",
        version_info.major, version_info.minor
    );
    // TODO: This should go _before_ the site-packages
    sys_path.call_method1(py, "append", (venv_path,))?;
    Ok(())
}

#[cfg(all(
    target_os = "linux",
    any(feature = "sglang", feature = "vllm", feature = "python")
))]
fn fix_venv(_venv: String, _py: Python<'_>) {}
