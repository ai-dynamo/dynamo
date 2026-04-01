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
use crate::descriptors::{QueryResponseList, RegDescList};
use crate::bindings::{
    nixl_capi_agent_config_s as nixl_capi_agent_config_t,
    nixl_capi_thread_sync_t, nixl_capi_create_configured_agent};

impl From<ThreadSync> for nixl_capi_thread_sync_t {
    fn from(value: ThreadSync) -> Self {
        match value {
            ThreadSync::None => crate::bindings::nixl_capi_thread_sync_t_NIXL_CAPI_THREAD_SYNC_NONE,
            ThreadSync::Strict => crate::bindings::nixl_capi_thread_sync_t_NIXL_CAPI_THREAD_SYNC_STRICT,
            ThreadSync::Rw => crate::bindings::nixl_capi_thread_sync_t_NIXL_CAPI_THREAD_SYNC_RW,
            ThreadSync::Default => crate::bindings::nixl_capi_thread_sync_t_NIXL_CAPI_THREAD_SYNC_DEFAULT,
        }
    }
}

/// A NIXL agent that can create backends and manage memory
#[derive(Debug, Clone)]
pub struct Agent {
    inner: Arc<RwLock<AgentInner>>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum XferStatus {
    Success,
    InProgress,
}

impl XferStatus {
    pub fn is_success(&self) -> bool {
        return *self == XferStatus::Success;
    }
}

impl Agent {
    /// Creates a new agent with the given name
    pub fn new(name: &str) -> Result<Self, NixlError> {
        tracing::trace!(agent.name = %name, "Creating new NIXL agent");
        let c_name = CString::new(name)?;
        let mut agent = ptr::null_mut();
        let status = unsafe { nixl_capi_create_agent(c_name.as_ptr(), &mut agent) };

        match status {
            NIXL_CAPI_SUCCESS => {
                // SAFETY: If status is NIXL_CAPI_SUCCESS, agent is non-null
                let handle = unsafe { NonNull::new_unchecked(agent) };
                tracing::trace!(agent.name = %name, "Successfully created NIXL agent");
                Ok(Self {
                    inner: Arc::new(RwLock::new(AgentInner::new(handle, name.to_string()))),
                })
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => {
                tracing::error!(agent.name = %name, error = "invalid_param", "Failed to create NIXL agent");
                Err(NixlError::InvalidParam)
            }
            _ => {
                tracing::error!(agent.name = %name, error = "backend_error", "Failed to create NIXL agent");
                Err(NixlError::BackendError)
            }
        }
    }

    /// Creates a new agent with the given configuration
    pub fn new_configured(name: &str, cfg: &AgentConfig) -> Result<Self, NixlError> {
        tracing::trace!(agent.name = %name, "Creating configured NIXL agent");
        let c_name = CString::new(name)?;

        // Prepare C ABI config
        let mut c_cfg = nixl_capi_agent_config_t {
            enable_prog_thread: cfg.enable_prog_thread,
            enable_listen_thread: cfg.enable_listen_thread,
            listen_port: cfg.listen_port,
            thread_sync: cfg.thread_sync.into(),
            num_workers: cfg.num_workers,
            pthr_delay_us: cfg.pthr_delay_us,
            lthr_delay_us: cfg.lthr_delay_us,
            capture_telemetry: cfg.capture_telemetry,
        };

        let mut agent = ptr::null_mut();
        let status = unsafe {
            nixl_capi_create_configured_agent(c_name.as_ptr(), &mut c_cfg, &mut agent)
        };

        match status {
            NIXL_CAPI_SUCCESS => {
                // SAFETY: If status is NIXL_CAPI_SUCCESS, agent is non-null
                let handle = unsafe { NonNull::new_unchecked(agent) };
                tracing::trace!(agent.name = %name, "Successfully created configured NIXL agent");
                Ok(Self {
                    inner: Arc::new(RwLock::new(AgentInner::new(handle, name.to_string()))),
                })
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => {
                tracing::error!(agent.name = %name, error = "invalid_param", "Failed to create configured NIXL agent");
                Err(NixlError::InvalidParam)
            }
            _ => {
                tracing::error!(agent.name = %name, error = "backend_error", "Failed to create configured NIXL agent");
                Err(NixlError::BackendError)
            }
        }
    }

    /// Gets the name of the agent
    pub fn name(&self) -> String {
        self.inner.read().unwrap().name.clone()
    }

    /// Gets the list of available plugins
    pub fn get_available_plugins(&self) -> Result<utils::StringList, NixlError> {
        tracing::trace!("Getting available NIXL plugins");
        let mut plugins = ptr::null_mut();

        // SAFETY: self.inner is guaranteed to be valid by NonNull
        let status = unsafe {
            nixl_capi_get_available_plugins(
                self.inner.write().unwrap().handle.as_ptr(),
                &mut plugins,
            )
        };

        match status {
            0 => {
                // SAFETY: If status is 0, plugins was successfully created and is non-null
                let inner = unsafe { NonNull::new_unchecked(plugins) };
                tracing::trace!("Successfully retrieved NIXL plugins");
                Ok(utils::StringList::new(inner))
            }
            -1 => {
                tracing::error!(error = "invalid_param", "Failed to get NIXL plugins");
                Err(NixlError::InvalidParam)
            }
            _ => {
                tracing::error!(error = "backend_error", "Failed to get NIXL plugins");
                Err(NixlError::BackendError)
            }
        }
    }

    /// Gets the parameters for a plugin
    ///
    /// # Arguments
    /// * `plugin_name` - The name of the plugin
    ///
    /// # Returns
    /// The plugin's memory list and parameters
    ///
    /// # Errors
    /// Returns a NixlError if:
    /// * The plugin name contains interior nul bytes
    /// * The operation fails
    pub fn get_plugin_params(
        &self,
        plugin_name: &str,
    ) -> Result<(MemList, utils::Params), NixlError> {
        let plugin_name = CString::new(plugin_name)?;
        let mut mems = ptr::null_mut();
        let mut params = ptr::null_mut();

        // SAFETY: self.inner is guaranteed to be valid by NonNull
        let status = unsafe {
            nixl_capi_get_plugin_params(
                self.inner.read().unwrap().handle.as_ptr(),
                plugin_name.as_ptr(),
                &mut mems,
                &mut params,
            )
        };

        match status {
            0 => {
                // SAFETY: If status is 0, both pointers were successfully created and are non-null
                let mems_inner = unsafe { NonNull::new_unchecked(mems) };
                let params_inner = unsafe { NonNull::new_unchecked(params) };
                Ok((
                    MemList { inner: mems_inner },
                    utils::Params::new(params_inner),
                ))
            }
            -1 => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Creates a new backend for the given plugin using the provided parameters
    pub fn create_backend(
        &self,
        plugin: &str,
        params: &utils::Params,
    ) -> Result<Backend, NixlError> {
        tracing::trace!(plugin.name = %plugin, "Creating new NIXL backend");
        let c_plugin = CString::new(plugin).map_err(|_| NixlError::InvalidParam)?;
        let name = c_plugin.to_string_lossy().to_string();
        let mut backend = ptr::null_mut();
        let status = unsafe {
            nixl_capi_create_backend(
                self.inner.write().unwrap().handle.as_ptr(),
                c_plugin.as_ptr(),
                params.handle(),
                &mut backend,
            )
        };

        match status {
            NIXL_CAPI_SUCCESS => {
                let backend_handle = NonNull::new(backend).ok_or(NixlError::BackendError)?;
                self.inner
                    .write()
                    .unwrap()
                    .backends
                    .insert(name.clone(), backend_handle);
                tracing::trace!(plugin.name = %plugin, "Successfully created NIXL backend");
                Ok(Backend {
                    inner: backend_handle,
                })
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => {
                tracing::error!(plugin.name = %plugin, error = "invalid_param", "Failed to create NIXL backend");
                Err(NixlError::InvalidParam)
            }
            _ => {
                tracing::error!(plugin.name = %plugin, error = "backend_error", "Failed to create NIXL backend");
                Err(NixlError::BackendError)
            }
        }
    }

    /// Gets a backend by name
    pub fn get_backend(&self, name: &str) -> Option<Backend> {
        self.inner
            .read()
            .unwrap()
            .get_backend(name)
            .map(|backend| Backend { inner: backend })
    }

    /// Gets the parameters and memory types for a backend after initialization
    pub fn get_backend_params(
        &self,
        backend: &Backend,
    ) -> Result<(MemList, utils::Params), NixlError> {
        let mut mem_list = ptr::null_mut();
        let mut params = ptr::null_mut();

        let status = unsafe {
            nixl_capi_get_backend_params(
                self.inner.read().unwrap().handle.as_ptr(),
                backend.inner.as_ptr(),
                &mut mem_list,
                &mut params,
            )
        };

        if status != NIXL_CAPI_SUCCESS {
            return Err(NixlError::BackendError);
        }

        // SAFETY: If status is NIXL_CAPI_SUCCESS, both pointers are non-null
        unsafe {
            Ok((
                MemList {
                    inner: NonNull::new_unchecked(mem_list),
                },
                utils::Params::new(NonNull::new_unchecked(params)),
            ))
        }
    }

    /// Registers a memory descriptor with the agent
    ///
    /// # Arguments
    /// * `descriptor` - The memory descriptor to register
    /// * `opt_args` - Optional arguments for the registration
    pub fn register_memory(
        &self,
        descriptor: &impl NixlDescriptor,
        opt_args: Option<&OptArgs>,
    ) -> Result<RegistrationHandle, NixlError> {
        let mut reg_dlist = RegDescList::new(descriptor.mem_type())?;
        unsafe {
            reg_dlist.add_storage_desc(descriptor)?;

            nixl_capi_register_mem(
                self.inner.write().unwrap().handle.as_ptr(),
                reg_dlist.handle(),
                opt_args.map_or(std::ptr::null_mut(), |args| args.inner.as_ptr()),
            );
        }
        Ok(RegistrationHandle {
            agent: Some(self.inner.clone()),
            ptr: unsafe { descriptor.as_ptr() } as usize,
            size: descriptor.size(),
            dev_id: descriptor.device_id(),
            mem_type: descriptor.mem_type(),
        })
    }

    /// Query information about memory/storage
    ///
    /// # Arguments
    /// * `descs` - Registration descriptor list to query
    /// * `opt_args` - Optional arguments specifying backends
    ///
    /// # Returns
    /// A list of query responses, where each response may contain parameters
    /// describing the memory/storage characteristics.
    pub fn query_mem(
        &self,
        descs: &RegDescList,
        opt_args: Option<&OptArgs>,
    ) -> Result<QueryResponseList, NixlError> {
        let resp = QueryResponseList::new()?;

        let status = {
            let inner_guard = self.inner.write().unwrap();
            unsafe {
                nixl_capi_query_mem(
                    inner_guard.handle.as_ptr(),
                    descs.handle(),
                    resp.handle(),
                    opt_args.map_or(std::ptr::null_mut(), |args| args.inner.as_ptr()),
                )
            }
        };

        match status {
            NIXL_CAPI_SUCCESS => Ok(resp),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Gets the local metadata for this agent as a byte array
    pub fn get_local_md(&self) -> Result<Vec<u8>, NixlError> {
        tracing::trace!("Getting local metadata");
        let mut data = std::ptr::null_mut();
        let mut len = 0;

        let status = unsafe {
            nixl_capi_get_local_md(
                self.inner.write().unwrap().handle.as_ptr(),
                &mut data as *mut *mut _,
                &mut len,
            )
        };

        let data = data as *const u8;

        if data.is_null() {
            tracing::trace!(
                error = "invalid_data_pointer",
                "Failed to get local metadata"
            );
            return Err(NixlError::InvalidDataPointer);
        }

        match status {
            NIXL_CAPI_SUCCESS => {
                let bytes = unsafe {
                    let slice = std::slice::from_raw_parts(data, len);
                    let vec = slice.to_vec();
                    libc::free(data as *mut libc::c_void);
                    vec
                };
                tracing::trace!(metadata.size = len, "Successfully retrieved local metadata");
                Ok(bytes)
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => {
                tracing::error!(error = "invalid_param", "Failed to get local metadata");
                Err(NixlError::InvalidParam)
            }
            _ => {
                tracing::error!(error = "backend_error", "Failed to get local metadata");
                Err(NixlError::BackendError)
            }
        }
    }

    /// Gets the local partial metadata as a byte array
    ///
    /// # Arguments
    /// * `descs` - Registration descriptor list to get metadata for
    /// * `opt_args` - Optional arguments for getting metadata
    ///
    /// # Returns
    /// A byte array containing the local partial metadata
    ///
    pub fn get_local_partial_md(&self, descs: &RegDescList, opt_args: Option<&OptArgs>) -> Result<Vec<u8>, NixlError> {
        tracing::trace!("Getting local partial metadata");
        let mut data = std::ptr::null_mut();
        let mut len: usize = 0;
        let inner_guard = self.inner.write().unwrap();

        let status = unsafe {
            nixl_capi_get_local_partial_md(
                inner_guard.handle.as_ptr(),
                descs.handle(),
                &mut data as *mut *mut _,
                &mut len,
                opt_args.map_or(std::ptr::null_mut(), |args| args.inner.as_ptr()),
            )
        };
        match status {
            NIXL_CAPI_SUCCESS => {
                let bytes = unsafe {
                    let slice = std::slice::from_raw_parts(data as *const u8, len);
                    let vec = slice.to_vec();
                    libc::free(data as *mut libc::c_void);
                    vec
                };
                tracing::trace!(metadata.size = len, "Successfully retrieved local partial metadata");
                Ok(bytes)
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => {
                tracing::error!(error = "invalid_param", "Failed to get local partial metadata");
                Err(NixlError::InvalidParam)
            }
            _ => {
                tracing::error!(error = "backend_error", "Failed to get local partial metadata");
                Err(NixlError::BackendError)
            }
        }
    }

    /// Loads remote metadata from a byte slice
    pub fn load_remote_md(&self, metadata: &[u8]) -> Result<String, NixlError> {
        tracing::trace!(metadata.size = metadata.len(), "Loading remote metadata");
        let mut agent_name = std::ptr::null_mut();

        let status = unsafe {
            nixl_capi_load_remote_md(
                self.inner.write().unwrap().handle.as_ptr(),
                metadata.as_ptr() as *const std::ffi::c_void,
                metadata.len(),
                &mut agent_name,
            )
        };

        match status {
            NIXL_CAPI_SUCCESS => {
                let name = unsafe {
                    let c_str = std::ffi::CStr::from_ptr(agent_name);
                    let s = c_str.to_str().unwrap().to_string();
                    libc::free(agent_name as *mut libc::c_void);
                    s
                };
                self.inner.write().unwrap().remotes.insert(name.clone());
                tracing::trace!(remote.agent = %name, "Successfully loaded remote metadata");
                Ok(name)
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => {
                tracing::error!(error = "invalid_param", "Failed to load remote metadata");
                Err(NixlError::InvalidParam)
            }
            _ => {
                tracing::error!(error = "backend_error", "Failed to load remote metadata");
                Err(NixlError::BackendError)
            }
        }
    }

    pub fn make_connection(&self, remote_agent: &str, opt_args: Option<&OptArgs>) -> Result<(), NixlError> {
        let remote_agent = CString::new(remote_agent)?;
        let inner_guard = self.inner.write().unwrap();

        let status = unsafe {
            nixl_capi_agent_make_connection(
                inner_guard.handle.as_ptr(),
                remote_agent.as_ptr(),
                opt_args.map_or(std::ptr::null_mut(), |args| args.inner.as_ptr()),
            )
        };

        match status {
            NIXL_CAPI_SUCCESS => Ok(()),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    pub fn prepare_xfer_dlist(
        &self,
        agent_name: &str,
        descs: &XferDescList,
        opt_args: Option<&OptArgs>,
    ) -> Result<XferDlistHandle, NixlError> {
        let c_agent_name = CString::new(agent_name)?;
        let mut dlist_hndl = std::ptr::null_mut();
        let inner_guard = self.inner.read().unwrap();

        let status = unsafe {
            nixl_capi_prep_xfer_dlist(
                inner_guard.handle.as_ptr(),
                c_agent_name.as_ptr(),
                descs.handle(),
                &mut dlist_hndl,
                opt_args.map_or(std::ptr::null_mut(), |args| args.inner.as_ptr()),
            )
        };

        match status {
            NIXL_CAPI_SUCCESS => Ok(XferDlistHandle::new(dlist_hndl, inner_guard.handle)),
            _ => Err(NixlError::BackendError),
        }
    }

    pub fn make_xfer_req(&self, operation: XferOp,
                         local_descs: &XferDlistHandle, local_indices: &[i32],
                         remote_descs: &XferDlistHandle, remote_indices: &[i32],
                         opt_args: Option<&OptArgs>) -> Result<XferRequest, NixlError> {
        let mut req = std::ptr::null_mut();
        let inner_guard = self.inner.read().unwrap();

        let status = unsafe {
            nixl_capi_make_xfer_req(
                inner_guard.handle.as_ptr(),
                operation as bindings::nixl_capi_xfer_op_t,
                local_descs.handle(),
                local_indices.as_ptr(),
                local_indices.len() as usize,
                remote_descs.handle(),
                remote_indices.as_ptr(),
                remote_indices.len() as usize,
                &mut req,
                opt_args.map_or(std::ptr::null_mut(), |args| args.inner.as_ptr())
            )
        };

        match status {
            NIXL_CAPI_SUCCESS => Ok(XferRequest::new(NonNull::new(req)
                .ok_or(NixlError::FailedToCreateXferRequest)?,
                self.inner.clone(),
            )),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Check if remote metadata for a specific agent is available
    ///
    /// This function checks if the metadata for the specified remote agent has been
    /// loaded and if specific descriptors can be found in the metadata.
    ///
    /// # Arguments
    /// * `remote_agent` - Name of the remote agent to check
    /// * `descs` - Optional descriptor list to check against the remote metadata.
    ///            If None, only checks if any metadata exists for the agent.
    ///
    /// # Returns
    /// `true` if the remote agent's metadata is available (and descriptors are found if provided),
    /// `false` otherwise
    pub fn check_remote_metadata(&self, remote_agent: &str, descs: Option<&XferDescList>) -> bool {
        tracing::trace!(remote_agent = %remote_agent, "Checking remote metadata");

        let c_remote_name = match CString::new(remote_agent) {
            Ok(name) => name,
            Err(_) => {
                tracing::trace!(
                    error = "invalid_param",
                    remote_agent = %remote_agent,
                    "Invalid remote agent name"
                );
                return false;
            }
        };

        let status = unsafe {
            bindings::nixl_capi_check_remote_md(
                self.inner.read().unwrap().handle.as_ptr(),
                c_remote_name.as_ptr(),
                descs.map_or(std::ptr::null_mut(), |d| d.as_ptr()),
            )
        };

        match status {
            NIXL_CAPI_SUCCESS => {
                tracing::trace!(remote_agent = %remote_agent, "Remote metadata is available");
                true
            }
            _ => {
                tracing::trace!(remote_agent = %remote_agent, "Remote metadata is not available");
                false
            }
        }
    }

    /// Invalidates a remote metadata for this agent
    pub fn invalidate_remote_md(&self, remote_agent: &str) -> Result<(), NixlError> {
        self.inner
            .write()
            .unwrap()
            .invalidate_remote_md(remote_agent)
    }

    /// Invalidates all remote metadata for this agent
    pub fn invalidate_all_remotes(&self) -> Result<(), NixlError> {
        self.inner.write().unwrap().invalidate_all_remotes()
    }

    /// Send this agent's metadata to etcdAdd commentMore actions
    ///
    /// This enables other agents to discover this agent's metadata via etcd.
    ///
    /// # Arguments
    /// * `opt_args` - Optional arguments for sending metadata
    pub fn send_local_md(&self, opt_args: Option<&OptArgs>) -> Result<(), NixlError> {
        tracing::trace!("Sending local metadata to etcd");
        let inner_guard = self.inner.write().unwrap();
        let status = unsafe {
            bindings::nixl_capi_send_local_md(
                inner_guard.handle.as_ptr(),
                opt_args.map_or(std::ptr::null_mut(), |args| args.inner.as_ptr()),
            )
        };

        match status {
            NIXL_CAPI_SUCCESS => {
                tracing::trace!("Successfully sent local metadata to etcd");
                Ok(())
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => {
                tracing::error!(
                    error = "invalid_param",
                    "Failed to send local metadata to etcd"
                );
                Err(NixlError::InvalidParam)
            }
            _ => {
                tracing::error!(
                    error = "backend_error",
                    "Failed to send local metadata to etcd"
                );
                Err(NixlError::BackendError)
            }
        }
    }

    /// Send this agent's partial metadata
    ///
    /// # Arguments
    /// * `descs` - Registration descriptor list to send
    /// * `opt_args` - Optional arguments for sending metadata
    pub fn send_local_partial_md(&self, descs: &RegDescList, opt_args: Option<&OptArgs>) -> Result<(), NixlError> {
        tracing::trace!("Sending local partial metadata to etcd");
        let inner_guard = self.inner.write().unwrap();
        let status = unsafe {
            nixl_capi_send_local_partial_md(
                inner_guard.handle.as_ptr(),
                descs.handle(),
                opt_args.map_or(std::ptr::null_mut(), |args| args.inner.as_ptr()),
            )
        };
        match status {
            NIXL_CAPI_SUCCESS => {
                tracing::trace!("Successfully sent local partial metadata to etcd");
                Ok(())
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => {
                tracing::error!(error = "invalid_param", "Failed to send local partial metadata to etcd");
                Err(NixlError::InvalidParam)
            }
            _ => Err(NixlError::BackendError)
        }
    }


    /// Fetch a remote agent's metadata from etcd
    ///
    /// Once fetched, the metadata will be loaded and cached locally, enabling
    /// communication with the remote agent.
    ///
    /// # Arguments
    /// * `remote_name` - Name of the remote agent to fetch metadata for
    /// * `opt_args` - Optional arguments for fetching metadata
    pub fn fetch_remote_md(
        &self,
        remote_name: &str,
        opt_args: Option<&OptArgs>,
    ) -> Result<(), NixlError> {
        tracing::trace!(remote_agent = %remote_name, "Fetching remote metadata from etcd");

        let c_remote_name = CString::new(remote_name)?;
        let inner_guard = self.inner.write().unwrap();

        let status = unsafe {
            bindings::nixl_capi_fetch_remote_md(
                inner_guard.handle.as_ptr(),
                c_remote_name.as_ptr(),
                opt_args.map_or(std::ptr::null_mut(), |args| args.inner.as_ptr()),
            )
        };

        match status {
            NIXL_CAPI_SUCCESS => {
                self.inner
                    .write()
                    .unwrap()
                    .remotes
                    .insert(remote_name.to_string());
                tracing::trace!(remote_agent = %remote_name, "Successfully fetched remote metadata from etcd");
                Ok(())
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => {
                tracing::error!(error = "invalid_param", remote_agent = %remote_name, "Failed to fetch remote metadata from etcd");
                Err(NixlError::InvalidParam)
            }
            _ => {
                tracing::error!(error = "backend_error", remote_agent = %remote_name, "Failed to fetch remote metadata from etcd");
                Err(NixlError::BackendError)
            }
        }
    }

    /// Invalidate this agent's metadata in etcd
    ///
    /// This signals to other agents that this agent's metadata is no longer valid.
    ///
    /// # Arguments
    /// * `opt_args` - Optional arguments for invalidating metadata
    pub fn invalidate_local_md(&self, opt_args: Option<&OptArgs>) -> Result<(), NixlError> {
        tracing::trace!("Invalidating local metadata in etcd");
        let inner_guard = self.inner.write().unwrap();
        let status = unsafe {
            bindings::nixl_capi_invalidate_local_md(
                inner_guard.handle.as_ptr(),
                opt_args.map_or(std::ptr::null_mut(), |args| args.inner.as_ptr()),
            )
        };

        match status {
            NIXL_CAPI_SUCCESS => {
                tracing::trace!("Successfully invalidated local metadata in etcd");
                Ok(())
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => {
                tracing::error!(
                    error = "invalid_param",
                    "Failed to invalidate local metadata in etcd"
                );
                Err(NixlError::InvalidParam)
            }
            _ => {
                tracing::error!(
                    error = "backend_error",
                    "Failed to invalidate local metadata in etcd"
                );
                Err(NixlError::BackendError)
            }
        }
    }

    /// Send a notification to a remote agent
    ///
    /// # Arguments
    /// * `remote_agent` - Name of the remote agent to send notification to
    /// * `message` - The notification message to send
    /// * `backend` - Optional backend to use for sending the notification
    ///
    /// # Returns
    /// `Ok(())` if the notification was sent successfully
    pub fn send_notification(
        &self,
        remote_agent: &str,
        message: &[u8],
        backend: Option<&Backend>,
    ) -> Result<(), NixlError> {
        tracing::trace!(remote_agent = %remote_agent, "Sending notification");

        let c_remote_name = CString::new(remote_agent)?;
        let inner_guard = self.inner.write().unwrap();

        let opt_args = if backend.is_some() {
            let mut args = OptArgs::new()?;
            if let Some(b) = backend {
                args.add_backend(b)?;
            }
            Some(args)
        } else {
            None
        };

        let status = unsafe {
            nixl_capi_gen_notif(
                inner_guard.handle.as_ptr(),
                c_remote_name.as_ptr(),
                message.as_ptr() as *const std::ffi::c_void,
                message.len(),
                opt_args
                    .as_ref()
                    .map_or(std::ptr::null_mut(), |args| args.inner.as_ptr()),
            )
        };

        match status {
            NIXL_CAPI_SUCCESS => {
                tracing::trace!(remote_agent = %remote_agent, "Successfully sent notification");
                Ok(())
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => {
                tracing::error!(error = "invalid_param", remote_agent = %remote_agent, "Failed to send notification");
                Err(NixlError::InvalidParam)
            }
            _ => {
                tracing::error!(error = "backend_error", remote_agent = %remote_agent, "Failed to send notification");
                Err(NixlError::BackendError)
            }
        }
    }

    /// Creates a transfer request between local and remote descriptors
    ///
    /// # Arguments
    /// * `operation` - The transfer operation (read or write)
    /// * `local_descs` - The local descriptor list
    /// * `remote_descs` - The remote descriptor list
    /// * `remote_agent` - The name of the remote agent
    /// * `opt_args` - Optional arguments for the transfer
    ///
    /// # Returns
    /// A handle to the transfer request
    ///
    /// # Errors
    /// Returns a NixlError if the operation fails
    pub fn create_xfer_req(
        &self,
        operation: XferOp,
        local_descs: &XferDescList,
        remote_descs: &XferDescList,
        remote_agent: &str,
        opt_args: Option<&OptArgs>,
    ) -> Result<XferRequest, NixlError> {
        let remote_agent = CString::new(remote_agent)?;
        let mut req = std::ptr::null_mut();

        // SAFETY: All pointers are guaranteed to be valid
        let status = unsafe {
            bindings::nixl_capi_create_xfer_req(
                self.inner.read().unwrap().handle.as_ptr(),
                operation as bindings::nixl_capi_xfer_op_t,
                local_descs.handle(),
                remote_descs.handle(),
                remote_agent.as_ptr(),
                &mut req,
                opt_args.map_or(std::ptr::null_mut(), |args| args.inner.as_ptr()),
            )
        };

        match status {
            NIXL_CAPI_SUCCESS => {
                // SAFETY: If status is NIXL_CAPI_SUCCESS, req is guaranteed to be non-null
                let inner = NonNull::new(req).ok_or(NixlError::FailedToCreateXferRequest)?;
                Ok(XferRequest::new(inner, self.inner.clone()))
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::FailedToCreateXferRequest),
        }
    }

    /// Estimates the cost of a transfer request
    ///
    /// # Arguments
    /// * `req` - Transfer request handle
    /// * `opt_args` - Optional arguments for the estimation
    ///
    /// # Returns
    /// A tuple containing (duration in microseconds, error margin in microseconds, cost method)
    ///
    /// # Errors
    /// Returns a NixlError if the operation fails
    pub fn estimate_xfer_cost(
        &self,
        req: &XferRequest,
        opt_args: Option<&OptArgs>,
    ) -> Result<(i64, i64, CostMethod), NixlError> {
        let mut duration_us: i64 = 0;
        let mut err_margin_us: i64 = 0;
        let mut method: u32 = 0;

        let status = unsafe {
            nixl_capi_estimate_xfer_cost(
                self.inner.write().unwrap().handle.as_ptr(),
                req.handle(),
                opt_args.map_or(ptr::null_mut(), |args| args.inner.as_ptr()),
                &mut duration_us,
                &mut err_margin_us,
                &mut method as *mut u32 as *mut bindings::nixl_capi_cost_t,
            )
        };

        match status {
            NIXL_CAPI_SUCCESS => Ok((duration_us, err_margin_us, CostMethod::from(method))),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Posts a transfer request to initiate a transfer
    ///
    /// After this, the transfer state can be checked asynchronously until completion.
    /// For small transfers that complete within the call, the function returns `Ok(false)`.
    /// Otherwise, it returns `Ok(true)` to indicate the transfer is in progress.
    ///
    /// # Arguments
    /// * `req` - Transfer request handle obtained from `create_xfer_req`
    /// * `opt_args` - Optional arguments for the transfer request
    ///
    /// # Returns
    /// * `Ok(false)` - If the transfer completed immediately
    /// * `Ok(true)` - If the transfer is in progress
    /// * `Err` - If there was an error posting the transfer request
    pub fn post_xfer_req(
        &self,
        req: &XferRequest,
        opt_args: Option<&OptArgs>,
    ) -> Result<bool, NixlError> {
        tracing::trace!("Posting transfer request");
        let status = unsafe {
            nixl_capi_post_xfer_req(
                self.inner.write().unwrap().handle.as_ptr(),
                req.handle(),
                opt_args.map_or(ptr::null_mut(), |args| args.inner.as_ptr()),
            )
        };

        match status {
            NIXL_CAPI_SUCCESS => {
                tracing::trace!(
                    status = "completed",
                    "Transfer request completed immediately"
                );
                Ok(false)
            }
            NIXL_CAPI_IN_PROG => {
                tracing::trace!(status = "in_progress", "Transfer request in progress");
                Ok(true)
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => {
                tracing::error!(error = "invalid_param", "Failed to post transfer request");
                Err(NixlError::InvalidParam)
            }
            _ => {
                tracing::error!(error = "backend_error", "Failed to post transfer request");
                Err(NixlError::BackendError)
            }
        }
    }

    /// Checks the status of a transfer request
    ///
    /// Returns `Ok(true)` if the transfer is still in progress, `Ok(false)` if it completed successfully.
    ///
    /// # Arguments
    /// * `req` - Transfer request handle after `post_xfer_req`
    pub fn get_xfer_status(&self, req: &XferRequest) -> Result<XferStatus, NixlError> {
        let status = unsafe {
            nixl_capi_get_xfer_status(self.inner.write().unwrap().handle.as_ptr(), req.handle())
        };

        match status {
            NIXL_CAPI_SUCCESS => Ok(XferStatus::Success), // Transfer completed
            NIXL_CAPI_IN_PROG => Ok(XferStatus::InProgress),  // Transfer in progress
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Queries the backend for a transfer request
    ///
    /// # Arguments
    /// * `req` - Transfer request handle after `post_xfer_req`
    ///
    /// # Returns
    /// A handle to the backend used for the transfer
    ///
    /// # Errors
    /// Returns a NixlError if the operation fails
    pub fn query_xfer_backend(&self, req: &XferRequest) -> Result<Backend, NixlError> {
        let mut backend = std::ptr::null_mut();
        let inner_guard = self.inner.write().unwrap();
        let status = unsafe {
            nixl_capi_query_xfer_backend(
                inner_guard.handle.as_ptr(),
                req.handle(),
                &mut backend
            )
        };
        match status {
            NIXL_CAPI_SUCCESS => {
                Ok(Backend{ inner: NonNull::new(backend).ok_or(NixlError::FailedToCreateBackend)? })
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }


    /// Gets notifications from other agents
    ///
    /// # Arguments
    /// * `notifs` - Notification map to populate with notifications
    /// * `opt_args` - Optional arguments to filter notifications by backend
    pub fn get_notifications(
        &self,
        notifs: &mut NotificationMap,
        opt_args: Option<&OptArgs>,
    ) -> Result<(), NixlError> {
        tracing::trace!("Getting notifications");
        let status = unsafe {
            nixl_capi_get_notifs(
                self.inner.write().unwrap().handle.as_ptr(),
                notifs.inner.as_ptr(),
                opt_args.map_or(ptr::null_mut(), |args| args.inner.as_ptr()),
            )
        };

        match status {
            NIXL_CAPI_SUCCESS => {
                tracing::trace!("Successfully retrieved notifications");
                Ok(())
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => {
                tracing::error!(error = "invalid_param", "Failed to get notifications");
                Err(NixlError::InvalidParam)
            }
            _ => {
                tracing::error!(error = "backend_error", "Failed to get notifications");
                Err(NixlError::BackendError)
            }
        }
    }
}

/// Inner state for an agent that manages the raw pointer
#[derive(Debug)]
pub(crate) struct AgentInner {
    pub(crate) name: String,
    pub(crate) handle: NonNull<bindings::nixl_capi_agent_s>,
    pub(crate) backends: HashMap<String, NonNull<bindings::nixl_capi_backend_s>>,
    pub(crate) remotes: HashSet<String>,
}

#[derive(Clone, Copy, Debug)]
pub enum ThreadSync {
    None,
    Strict,
    Rw,
    Default,
}

#[derive(Clone, Debug)]
pub struct AgentConfig {
    pub enable_prog_thread: bool,
    pub enable_listen_thread: bool,
    pub listen_port: i32,
    pub thread_sync: ThreadSync,
    pub num_workers: u32,
    pub pthr_delay_us: u64,
    pub lthr_delay_us: u64,
    pub capture_telemetry: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            enable_prog_thread: true,
            enable_listen_thread: false,
            listen_port: 0,
            thread_sync: ThreadSync::None,
            num_workers: 1,
            pthr_delay_us: 0,
            lthr_delay_us: 100_000,
            capture_telemetry: false,
        }
    }
}

unsafe impl Send for AgentInner {}
unsafe impl Sync for AgentInner {}

impl AgentInner {
    fn new(handle: NonNull<bindings::nixl_capi_agent_s>, name: String) -> Self {
        Self {
            name,
            handle,
            backends: HashMap::new(),
            remotes: HashSet::new(),
        }
    }

    fn get_backend(&self, name: &str) -> Option<NonNull<bindings::nixl_capi_backend_s>> {
        self.backends.get(name).cloned()
    }

    fn invalidate_remote_md_cstr(&self, remote_agent: &str) -> Result<(), NixlError> {
        let remote_agent = CString::new(remote_agent).map_err(|_| NixlError::InvalidParam)?;
        unsafe {
            nixl_capi_invalidate_remote_md(self.handle.as_ptr(), remote_agent.as_ptr().cast());
        }
        Ok(())
    }

    fn invalidate_remote_md(&mut self, remote_agent: &str) -> Result<(), NixlError> {
        if self.remotes.remove(remote_agent) {
            self.invalidate_remote_md_cstr(remote_agent)?;
        } else {
            return Err(NixlError::InvalidParam);
        }
        Ok(())
    }

    fn invalidate_all_remotes(&mut self) -> Result<(), NixlError> {
        let remotes: Vec<_> = self.remotes.drain().collect();
        for remote in remotes {
            self.invalidate_remote_md_cstr(&remote)?;
        }
        Ok(())
    }
}

impl Drop for AgentInner {
    fn drop(&mut self) {
        tracing::trace!("Dropping NIXL agent");
        unsafe {
            // invalidate all remotes
            for remote in self.remotes.iter() {
                tracing::trace!(remote.agent = %remote, "Invalidating remote agent");
                if let Err(error) = self.invalidate_remote_md_cstr(remote) {
                    tracing::warn!(remote.agent = %remote, ?error, "Failed to invalidate remote agent");
                }
            }

            // destroy all backends
            for backend in self.backends.values() {
                tracing::trace!("Destroying backend");
                nixl_capi_destroy_backend(backend.as_ptr());
            }

            nixl_capi_destroy_agent(self.handle.as_ptr());
        }
        tracing::trace!("NIXL agent dropped");
    }
}
