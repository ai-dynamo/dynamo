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

use std::{
    collections::HashMap,
    ffi::{CStr, CString},
    os::raw::c_char, // Added for *const c_char
    ptr::{self, NonNull},
};

/// A safe wrapper around NIXL notification map
pub struct NotificationMap {
    pub(crate) inner: NonNull<bindings::nixl_capi_notif_map_s>,
}

impl NotificationMap {
    /// Creates a new empty notification map
    pub fn new() -> Result<Self, NixlError> {
        let mut map = ptr::null_mut();
        let status = unsafe { nixl_capi_create_notif_map(&mut map) };
        match status {
            NIXL_CAPI_SUCCESS => {
                // SAFETY: If status is NIXL_CAPI_SUCCESS, map is non-null
                let inner = unsafe { NonNull::new_unchecked(map) };
                Ok(Self { inner })
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns the number of agents that have notifications
    pub fn len(&self) -> Result<usize, NixlError> {
        let mut size = 0;
        let status = unsafe { nixl_capi_notif_map_size(self.inner.as_ptr(), &mut size) };
        match status {
            NIXL_CAPI_SUCCESS => Ok(size),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns true if there are no notifications
    pub fn is_empty(&self) -> Result<bool, NixlError> {
        Ok(self.len()? == 0)
    }

    /// Returns an iterator over the agent names that have notifications
    pub fn agents(&self) -> NotificationMapAgentIterator<'_> {
        NotificationMapAgentIterator {
            map: self,
            index: 0,
            length: self.len().unwrap_or(0),
        }
    }

    /// Returns the number of notifications for a given agent
    pub fn get_notifications_size(&self, agent_name: &str) -> Result<usize, NixlError> {
        let mut size = 0;
        let c_name = CString::new(agent_name).map_err(|_| NixlError::InvalidParam)?;
        let status = unsafe {
            nixl_capi_notif_map_get_notifs_size(self.inner.as_ptr(), c_name.as_ptr(), &mut size)
        };
        match status {
            NIXL_CAPI_SUCCESS => Ok(size),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns an iterator over the notifications for a given agent
    pub fn get_notifications(
        &self,
        agent_name: &str,
    ) -> Result<NotificationIterator<'_>, NixlError> {
        let size = self.get_notifications_size(agent_name)?;
        Ok(NotificationIterator {
            map: self,
            agent_name: agent_name.to_string(),
            index: 0,
            length: size,
        })
    }

    /// Returns a specific notification for a given agent as raw bytes
    pub fn get_notification_bytes(
        &self,
        agent_name: &str,
        index: usize,
    ) -> Result<Vec<u8>, NixlError> {
        let c_name = CString::new(agent_name).map_err(|_| NixlError::InvalidParam)?;
        let mut data: *const u8 = ptr::null();
        let mut len = 0;
        let status = unsafe {
            nixl_capi_notif_map_get_notif(
                self.inner.as_ptr(),
                c_name.as_ptr(),
                index,
                &mut data as *mut *const _ as *mut *const std::ffi::c_void,
                &mut len,
            )
        };
        match status {
            NIXL_CAPI_SUCCESS => {
                if data.is_null() {
                    Ok(Vec::new())
                } else {
                    // SAFETY: If status is NIXL_CAPI_SUCCESS, data points to valid memory of size len
                    // This data is owned by the C side and is valid until the map is cleared or modified.
                    let bytes = unsafe {
                        let slice = std::slice::from_raw_parts(data, len);
                        slice.to_vec()
                    };
                    Ok(bytes)
                }
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Takes all notifications from the map, converting them to Strings,
    /// and clears the underlying C map for reuse.
    ///
    /// Returns a HashMap where keys are agent names and values are vectors of
    /// notification strings for that agent.
    ///
    /// If a notification\'s byte data is not valid UTF-8, this method will
    /// return an error (`NixlError::BackendError` in current impl, ideally a specific UTF-8 error).
    pub fn take_notifs(&mut self) -> Result<HashMap<String, Vec<String>>, NixlError> {
        let mut all_notifications = HashMap::new();
        let num_agents = self.len()?;

        for agent_idx in 0..num_agents {
            let mut c_agent_name_ptr: *const c_char = ptr::null();
            let status_agent_name = unsafe {
                nixl_capi_notif_map_get_agent_at(
                    self.inner.as_ptr(),
                    agent_idx,
                    &mut c_agent_name_ptr,
                )
            };

            if status_agent_name != NIXL_CAPI_SUCCESS {
                // This case should ideally not happen if num_agents is correct
                // and map is consistent.
                return Err(if status_agent_name == NIXL_CAPI_ERROR_INVALID_PARAM {
                    NixlError::InvalidParam
                } else {
                    NixlError::BackendError
                });
            }

            if c_agent_name_ptr.is_null() {
                // Should not happen if get_agent_at succeeded.
                return Err(NixlError::BackendError);
            }

            let agent_name_cstr = unsafe { CStr::from_ptr(c_agent_name_ptr) };
            let agent_name_string = agent_name_cstr
                .to_str()
                .map_err(|_| NixlError::InvalidParam)? // Map UTF-8 error on agent name to InvalidParam
                .to_owned();

            let mut num_notifs_for_agent = 0;
            let status_notif_size = unsafe {
                nixl_capi_notif_map_get_notifs_size(
                    self.inner.as_ptr(),
                    c_agent_name_ptr, // Use the C string directly
                    &mut num_notifs_for_agent,
                )
            };

            if status_notif_size != NIXL_CAPI_SUCCESS {
                return Err(if status_notif_size == NIXL_CAPI_ERROR_INVALID_PARAM {
                    NixlError::InvalidParam
                } else {
                    NixlError::BackendError
                });
            }

            let mut agent_specific_notifications = Vec::with_capacity(num_notifs_for_agent);

            for notif_idx in 0..num_notifs_for_agent {
                let mut data_ptr: *const std::ffi::c_void = ptr::null();
                let mut data_len: usize = 0;

                let status_notif_data = unsafe {
                    nixl_capi_notif_map_get_notif(
                        self.inner.as_ptr(),
                        c_agent_name_ptr, // Use the C string directly
                        notif_idx,
                        &mut data_ptr,
                        &mut data_len,
                    )
                };

                if status_notif_data != NIXL_CAPI_SUCCESS {
                    return Err(if status_notif_data == NIXL_CAPI_ERROR_INVALID_PARAM {
                        NixlError::InvalidParam
                    } else {
                        NixlError::BackendError
                    });
                }

                let notification_bytes = if data_ptr.is_null() || data_len == 0 {
                    Vec::new()
                } else {
                    // SAFETY: Pointer and length are from a successful C API call.
                    // Data is valid until map is cleared/modified. We copy it immediately.
                    unsafe { std::slice::from_raw_parts(data_ptr as *const u8, data_len) }.to_vec()
                };

                // Attempt to convert Vec<u8> to String
                let notification_string =
                    String::from_utf8(notification_bytes).map_err(|_e| NixlError::BackendError)?; // FIXME: Ideally, a specific UTF-8 error variant in NixlError (e.g., InvalidNotificationEncoding)

                agent_specific_notifications.push(notification_string);
            }
            all_notifications.insert(agent_name_string, agent_specific_notifications);
        }

        // After successfully extracting all data, clear the C map
        let clear_status = unsafe { nixl_capi_notif_map_clear(self.inner.as_ptr()) };
        match clear_status {
            NIXL_CAPI_SUCCESS => Ok(all_notifications),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam), // Should not happen if self.inner is valid
            _ => Err(NixlError::BackendError),
        }
    }
}

/// An iterator over agent names in a NotificationMap
pub struct NotificationMapAgentIterator<'a> {
    map: &'a NotificationMap,
    index: usize,
    length: usize,
}

impl<'a> Iterator for NotificationMapAgentIterator<'a> {
    type Item = Result<&'a str, NixlError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.length {
            None
        } else {
            let mut agent_name = ptr::null();
            let status = unsafe {
                nixl_capi_notif_map_get_agent_at(
                    self.map.inner.as_ptr(),
                    self.index,
                    &mut agent_name,
                )
            };
            self.index += 1;
            match status {
                NIXL_CAPI_SUCCESS => {
                    // SAFETY: If status is NIXL_CAPI_SUCCESS, agent_name points to a valid C string
                    let name = unsafe { CStr::from_ptr(agent_name) };
                    Some(name.to_str().map_err(|_| NixlError::InvalidParam))
                }
                NIXL_CAPI_ERROR_INVALID_PARAM => Some(Err(NixlError::InvalidParam)),
                _ => Some(Err(NixlError::BackendError)),
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.length - self.index;
        (remaining, Some(remaining))
    }
}

/// An iterator over notifications for a specific agent
pub struct NotificationIterator<'a> {
    map: &'a NotificationMap,
    agent_name: String,
    index: usize,
    length: usize,
}

impl Iterator for NotificationIterator<'_> {
    type Item = Result<Vec<u8>, NixlError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.length {
            None
        } else {
            let result = self
                .map
                .get_notification_bytes(&self.agent_name, self.index);
            self.index += 1;
            Some(result)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.length - self.index;
        (remaining, Some(remaining))
    }
}

impl Drop for NotificationMap {
    fn drop(&mut self) {
        tracing::trace!("Dropping notification map");
        unsafe {
            nixl_capi_destroy_notif_map(self.inner.as_ptr());
        }
        tracing::trace!("Notification map dropped");
    }
}
