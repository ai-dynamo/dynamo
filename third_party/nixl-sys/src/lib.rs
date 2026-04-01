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

//! Raw FFI bindings to the NIXL library
//!
//! This crate provides low-level bindings to the NIXL C++ library.
//! It is not meant to be used directly, but rather through the higher-level
//! `nixl` crate.

use libc::uintptr_t;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, CString};
use std::fmt;
use std::marker::PhantomData;
use std::ptr;
use std::ptr::NonNull;
use std::sync::{Arc, RwLock};
use thiserror::Error;

// Include the generated bindings
mod bindings {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(dead_code)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

// Re-export types from the included bindings
use bindings::{
    nixl_capi_create_agent, nixl_capi_create_backend, nixl_capi_create_notif_map,
    nixl_capi_create_opt_args, nixl_capi_create_reg_dlist, nixl_capi_create_xfer_dlist,
    nixl_capi_deregister_mem, nixl_capi_destroy_agent, nixl_capi_destroy_backend,
    nixl_capi_destroy_mem_list, nixl_capi_destroy_notif_map, nixl_capi_destroy_opt_args,
    nixl_capi_destroy_params, nixl_capi_destroy_reg_dlist, nixl_capi_destroy_string_list,
    nixl_capi_destroy_xfer_dlist, nixl_capi_get_available_plugins, nixl_capi_get_backend_params,
    nixl_capi_get_local_md, nixl_capi_get_notifs, nixl_capi_get_plugin_params,
    nixl_capi_get_xfer_status, nixl_capi_invalidate_remote_md, nixl_capi_load_remote_md,
    nixl_capi_mem_list_get, nixl_capi_mem_list_is_empty, nixl_capi_mem_list_size,
    nixl_capi_mem_type_t, nixl_capi_mem_type_to_string, nixl_capi_notif_map_clear,
    nixl_capi_notif_map_get_agent_at, nixl_capi_notif_map_get_notif,
    nixl_capi_notif_map_get_notifs_size, nixl_capi_notif_map_size, nixl_capi_opt_args_add_backend,
    nixl_capi_opt_args_get_has_notif, nixl_capi_opt_args_get_notif_msg,
    nixl_capi_opt_args_get_skip_desc_merge, nixl_capi_opt_args_set_has_notif,
    nixl_capi_opt_args_set_notif_msg, nixl_capi_opt_args_set_skip_desc_merge,
    nixl_capi_params_create_iterator, nixl_capi_params_destroy_iterator, nixl_capi_params_is_empty,
    nixl_capi_params_iterator_next, nixl_capi_post_xfer_req, nixl_capi_reg_dlist_add_desc,
    nixl_capi_reg_dlist_clear, nixl_capi_register_mem, nixl_capi_string_list_get,
    nixl_capi_string_list_size, nixl_capi_xfer_dlist_add_desc, nixl_capi_xfer_dlist_clear,
    nixl_capi_agent_make_connection,
    nixl_capi_reg_dlist_print, nixl_capi_xfer_dlist_print, nixl_capi_gen_notif, nixl_capi_estimate_xfer_cost,
    nixl_capi_query_mem, nixl_capi_create_query_resp_list, nixl_capi_destroy_query_resp_list,
    nixl_capi_query_resp_list_size, nixl_capi_query_resp_list_has_value,
    nixl_capi_query_resp_list_get_params, nixl_capi_prep_xfer_dlist, nixl_capi_release_xfer_dlist_handle,
    nixl_capi_make_xfer_req, nixl_capi_get_local_partial_md,
    nixl_capi_send_local_partial_md, nixl_capi_query_xfer_backend, nixl_capi_opt_args_set_ip_addr,
    nixl_capi_opt_args_set_port, nixl_capi_get_xfer_telemetry,
    nixl_capi_create_params, nixl_capi_params_add, nixl_capi_is_stub
};

// Re-export status codes
pub use bindings::{
    nixl_capi_status_t_NIXL_CAPI_ERROR_BACKEND as NIXL_CAPI_ERROR_BACKEND,
    nixl_capi_status_t_NIXL_CAPI_ERROR_INVALID_PARAM as NIXL_CAPI_ERROR_INVALID_PARAM,
    nixl_capi_status_t_NIXL_CAPI_IN_PROG as NIXL_CAPI_IN_PROG,
    nixl_capi_status_t_NIXL_CAPI_SUCCESS as NIXL_CAPI_SUCCESS,
    nixl_capi_status_t_NIXL_CAPI_ERROR_NO_TELEMETRY as NIXL_CAPI_ERROR_NO_TELEMETRY
};

mod agent;
mod descriptors;
mod notify;
mod utils;
mod xfer;

pub use agent::*;
pub use descriptors::*;
pub use notify::*;
pub use utils::*;
pub use xfer::*;

/// Errors that can occur when using NIXL
#[derive(Error, Debug)]
pub enum NixlError {
    #[error("Invalid parameter provided to NIXL")]
    InvalidParam,
    #[error("Backend error occurred")]
    BackendError,
    #[error("Failed to create CString from input: {0}")]
    StringConversionError(#[from] std::ffi::NulError),
    #[error("Index out of bounds")]
    IndexOutOfBounds,
    #[error("Invalid data pointer")]
    InvalidDataPointer,
    #[error("Failed to create XferRequest")]
    FailedToCreateXferRequest,
    #[error("Failed to create registration descriptor list")]
    RegDescListCreationFailed,
    #[error("Failed to add registration descriptor")]
    RegDescAddFailed,
    #[error("Failed to create XferDlistHandle")]
    FailedToCreateXferDlistHandle,
    #[error("Failed to create backend")]
    FailedToCreateBackend,
    #[error("Telemetry is not enabled or transfer is not complete")]
    NoTelemetry,
}

/// A safe wrapper around NIXL memory list
pub struct MemList {
    inner: NonNull<bindings::nixl_capi_mem_list_s>,
}

impl Drop for MemList {
    fn drop(&mut self) {
        // SAFETY: self.inner is guaranteed to be valid by NonNull
        unsafe {
            nixl_capi_destroy_mem_list(self.inner.as_ptr());
        }
    }
}

#[derive(Debug)]
pub struct RegistrationHandle {
    agent: Option<Arc<RwLock<AgentInner>>>,
    ptr: usize,
    size: usize,
    dev_id: u64,
    mem_type: MemType,
}

impl RegistrationHandle {
    pub fn agent_name(&self) -> Option<String> {
        self.agent
            .as_ref()
            .map(|agent| agent.read().unwrap().name.clone())
    }

    pub fn deregister(&mut self) -> Result<(), NixlError> {
        if let Some(agent) = self.agent.take() {
            tracing::trace!(
                ptr = self.ptr,
                size = self.size,
                dev_id = self.dev_id,
                mem_type = ?self.mem_type,
                "Deregistering memory"
            );
            let mut reg_dlist = RegDescList::new(self.mem_type)?;
            unsafe {
                reg_dlist.add_desc(self.ptr, self.size, self.dev_id);
                let _opt_args = OptArgs::new().unwrap();
                nixl_capi_deregister_mem(
                    agent.write().unwrap().handle.as_ptr(),
                    reg_dlist.handle(),
                    _opt_args.inner.as_ptr(),
                );
            }
            tracing::trace!("Memory deregistered successfully");
        }
        Ok(())
    }
}

impl Drop for RegistrationHandle {
    fn drop(&mut self) {
        tracing::trace!(
            ptr = self.ptr,
            size = self.size,
            dev_id = self.dev_id,
            mem_type = ?self.mem_type,
            "Dropping registration handle"
        );
        if let Err(e) = self.deregister() {
            tracing::debug!(error = ?e, "Failed to deregister memory");
        }
    }
}

/// A NIXL backend that can be used for data transfer
#[derive(Debug)]
pub struct Backend {
    inner: NonNull<bindings::nixl_capi_backend_s>,
}

unsafe impl Send for Backend {}
unsafe impl Sync for Backend {}

/// Transfer telemetry data wrapper
#[derive(Debug)]
pub struct XferTelemetry {
    /// Start time in microseconds since epoch
    pub start_time_us: u64,
    /// Post operation duration in microseconds
    pub post_duration_us: u64,
    /// Transfer duration in microseconds
    pub xfer_duration_us: u64,
    /// Total bytes transferred
    pub total_bytes: u64,
    /// Number of descriptors
    pub desc_count: u64,
}

impl XferTelemetry {
    /// Get the start time as a Duration since Unix epoch
    pub fn start_time(&self) -> std::time::Duration {
        std::time::Duration::from_micros(self.start_time_us)
    }

    /// Get the post operation duration
    pub fn post_duration(&self) -> std::time::Duration {
        std::time::Duration::from_micros(self.post_duration_us)
    }

    /// Get the transfer duration
    pub fn xfer_duration(&self) -> std::time::Duration {
        std::time::Duration::from_micros(self.xfer_duration_us)
    }

    /// Get the total duration (post + transfer)
    pub fn total_duration(&self) -> std::time::Duration {
        std::time::Duration::from_micros(self.post_duration_us + self.xfer_duration_us)
    }

    /// Calculate the transfer rate in bytes per second
    pub fn transfer_rate_bps(&self) -> f64 {
        if self.xfer_duration_us == 0 {
            0.0
        } else {
            (self.total_bytes as f64) / (self.xfer_duration_us as f64 / 1_000_000.0)
        }
    }
}

/// A safe wrapper around NIXL optional arguments
pub struct OptArgs {
    inner: NonNull<bindings::nixl_capi_opt_args_s>,
}

impl OptArgs {
    /// Creates a new empty optional arguments struct
    pub fn new() -> Result<Self, NixlError> {
        let mut args = ptr::null_mut();

        let status = unsafe { nixl_capi_create_opt_args(&mut args) };

        match status {
            0 => {
                // SAFETY: If status is 0, args was successfully created and is non-null
                let inner = unsafe { NonNull::new_unchecked(args) };
                Ok(Self { inner })
            }
            -1 => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Add a backend to the optional arguments
    pub fn add_backend(&mut self, backend: &Backend) -> Result<(), NixlError> {
        let status =
            unsafe { nixl_capi_opt_args_add_backend(self.inner.as_ptr(), backend.inner.as_ptr()) };
        match status {
            NIXL_CAPI_SUCCESS => Ok(()),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Set the notification message
    pub fn set_notification_message(&mut self, message: &[u8]) -> Result<(), NixlError> {
        let status = unsafe {
            nixl_capi_opt_args_set_notif_msg(
                self.inner.as_ptr(),
                message.as_ptr() as *const _,
                message.len(),
            )
        };
        match status {
            NIXL_CAPI_SUCCESS => Ok(()),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Get the notification message
    pub fn get_notification_message(&self) -> Result<Vec<u8>, NixlError> {
        let mut data = ptr::null_mut();
        let mut len = 0;
        let status =
            unsafe { nixl_capi_opt_args_get_notif_msg(self.inner.as_ptr(), &mut data, &mut len) };

        match status {
            NIXL_CAPI_SUCCESS => {
                if data.is_null() {
                    Ok(Vec::new())
                } else {
                    // SAFETY: If status is 0 and data is not null, it points to valid memory of size len
                    let message = unsafe {
                        let slice = std::slice::from_raw_parts(data as *const u8, len);
                        let vec = slice.to_vec();
                        libc::free(data as *mut _);
                        vec
                    };
                    Ok(message)
                }
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Set whether notification is enabled
    pub fn set_has_notification(&mut self, has_notification: bool) -> Result<(), NixlError> {
        let status =
            unsafe { nixl_capi_opt_args_set_has_notif(self.inner.as_ptr(), has_notification) };
        match status {
            NIXL_CAPI_SUCCESS => Ok(()),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Get whether notification is enabled
    pub fn has_notification(&self) -> Result<bool, NixlError> {
        let mut has_notification = false;
        let status =
            unsafe { nixl_capi_opt_args_get_has_notif(self.inner.as_ptr(), &mut has_notification) };
        match status {
            NIXL_CAPI_SUCCESS => Ok(has_notification),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Set whether to skip descriptor merging
    pub fn set_skip_descriptor_merge(&mut self, skip_merge: bool) -> Result<(), NixlError> {
        let status =
            unsafe { nixl_capi_opt_args_set_skip_desc_merge(self.inner.as_ptr(), skip_merge) };
        match status {
            NIXL_CAPI_SUCCESS => Ok(()),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Get whether descriptor merging is skipped
    pub fn skip_descriptor_merge(&self) -> Result<bool, NixlError> {
        let mut skip_merge = false;
        let status =
            unsafe { nixl_capi_opt_args_get_skip_desc_merge(self.inner.as_ptr(), &mut skip_merge) };
        match status {
            NIXL_CAPI_SUCCESS => Ok(skip_merge),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Set the IP address
    /// used in sendLocalMD, fetchRemoteMD, invalidateLocalMD, sendLocalPartialMD.
    pub fn set_ip_addr(&mut self, ip_addr: &str) -> Result<(), NixlError> {
        let c_str = CString::new(ip_addr).expect("Failed to convert string to CString");
        let status = unsafe { nixl_capi_opt_args_set_ip_addr(self.inner.as_ptr(), c_str.as_ptr()) };
        match status {
            NIXL_CAPI_SUCCESS => Ok(()),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Set the port
    /// used in sendLocalMD, fetchRemoteMD, invalidateLocalMD, sendLocalPartialMD.
    pub fn set_port(&mut self, port: u16) -> Result<(), NixlError> {
        let status = unsafe { nixl_capi_opt_args_set_port(self.inner.as_ptr(), port) };
        match status {
            NIXL_CAPI_SUCCESS => Ok(()),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }
}

impl Drop for OptArgs {
    fn drop(&mut self) {
        tracing::trace!("Dropping optional arguments");
        unsafe {
            nixl_capi_destroy_opt_args(self.inner.as_ptr());
        }
        tracing::trace!("Optional arguments dropped");
    }
}

impl MemList {
    /// Returns true if the memory list is empty
    pub fn is_empty(&self) -> Result<bool, NixlError> {
        let mut is_empty = false;

        // SAFETY: self.inner is guaranteed to be valid by NonNull
        let status = unsafe { nixl_capi_mem_list_is_empty(self.inner.as_ptr(), &mut is_empty) };

        match status {
            0 => Ok(is_empty),
            -1 => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns the number of memory types in the list
    pub fn len(&self) -> Result<usize, NixlError> {
        let mut size = 0;

        // SAFETY: self.inner is guaranteed to be valid by NonNull
        let status = unsafe { nixl_capi_mem_list_size(self.inner.as_ptr(), &mut size) };

        match status {
            0 => Ok(size),
            -1 => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns the memory type at the given index
    pub fn get(&self, index: usize) -> Result<MemType, NixlError> {
        let mut mem_type = 0;

        // SAFETY: self.inner is guaranteed to be valid by NonNull
        let status = unsafe { nixl_capi_mem_list_get(self.inner.as_ptr(), index, &mut mem_type) };

        match status {
            0 => Ok(MemType::from(mem_type)),
            -1 => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns an iterator over the memory types
    pub fn iter(&self) -> MemListIterator<'_> {
        MemListIterator {
            list: self,
            index: 0,
            length: self.len().unwrap_or(0),
        }
    }
}

/// An iterator over memory types in a MemList
pub struct MemListIterator<'a> {
    list: &'a MemList,
    index: usize,
    length: usize,
}

impl Iterator for MemListIterator<'_> {
    type Item = Result<MemType, NixlError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.length {
            None
        } else {
            let result = self.list.get(self.index);
            self.index += 1;
            Some(result)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.length - self.index;
        (remaining, Some(remaining))
    }
}

/// A trait for storage types that can be used with NIXL
pub trait MemoryRegion: std::fmt::Debug + Send + Sync {
    /// Get a raw pointer to the storage
    ///
    /// # Safety
    /// The caller must ensure:
    /// - The pointer is not used after the storage is dropped
    /// - Access patterns respect the storage's thread safety model
    unsafe fn as_ptr(&self) -> *const u8;

    /// Returns the total size of the storage in bytes
    fn size(&self) -> usize;
}

/// A trait for types that can be added to NIXL descriptor lists
pub trait NixlDescriptor: MemoryRegion {
    /// Get the memory type for this descriptor
    fn mem_type(&self) -> MemType;

    /// Get the device ID for this memory region
    fn device_id(&self) -> u64;
}

/// A trait for types that can be registered with NIXL
pub trait NixlRegistration: NixlDescriptor {
    fn register(&mut self, agent: &Agent, opt_args: Option<&OptArgs>) -> Result<(), NixlError>;
}

/// System memory storage implementation using a Vec<u8>
#[derive(Debug)]
pub struct SystemStorage {
    data: Vec<u8>,
    handle: Option<RegistrationHandle>,
}

impl SystemStorage {
    /// Create a new system storage with the given size
    pub fn new(size: usize) -> Result<Self, NixlError> {
        let data = vec![0; size];
        Ok(Self { data, handle: None })
    }

    /// Fill the storage with a specific byte value
    pub fn memset(&mut self, value: u8) {
        self.data.fill(value);
    }

    /// Get a slice of the underlying data
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }
}

impl MemoryRegion for SystemStorage {
    fn size(&self) -> usize {
        self.data.len()
    }

    unsafe fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }
}

impl NixlDescriptor for SystemStorage {
    fn mem_type(&self) -> MemType {
        MemType::Dram
    }

    fn device_id(&self) -> u64 {
        0
    }
}

impl NixlRegistration for SystemStorage {
    fn register(&mut self, agent: &Agent, opt_args: Option<&OptArgs>) -> Result<(), NixlError> {
        let handle = agent.register_memory(self, opt_args)?;
        self.handle = Some(handle);
        Ok(())
    }
}

pub fn is_stub() -> bool {
    unsafe { nixl_capi_is_stub() }
}
