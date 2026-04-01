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
use crate::Params;

/// A safe wrapper around a NIXL query response list
pub struct QueryResponseList {
    inner: NonNull<bindings::nixl_capi_query_resp_list_s>,
}

/// Represents a single query response which may or may not contain parameters
pub struct QueryResponse<'a> {
    list: &'a QueryResponseList,
    index: usize,
}

impl QueryResponseList {
    /// Creates a new empty query response list
    pub fn new() -> Result<Self, NixlError> {
        let mut list = ptr::null_mut();
        let status = unsafe { nixl_capi_create_query_resp_list(&mut list) };

        match status {
            NIXL_CAPI_SUCCESS => {
                // SAFETY: If status is NIXL_CAPI_SUCCESS, list is non-null
                let inner = unsafe { NonNull::new_unchecked(list) };
                Ok(Self { inner })
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns the number of responses in the list
    pub fn len(&self) -> Result<usize, NixlError> {
        let mut size = 0;
        let status = unsafe { nixl_capi_query_resp_list_size(self.inner.as_ptr(), &mut size) };

        match status {
            NIXL_CAPI_SUCCESS => Ok(size),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns true if the list is empty
    pub fn is_empty(&self) -> Result<bool, NixlError> {
        Ok(self.len()? == 0)
    }

    /// Gets a query response at the given index
    pub fn get(&self, index: usize) -> Result<QueryResponse<'_>, NixlError> {
        let size = self.len()?;
        if index >= size {
            return Err(NixlError::InvalidParam);
        }

        Ok(QueryResponse { list: self, index })
    }

    /// Returns an iterator
    pub fn iter(&self) -> Result<QueryResponseIterator<'_>, NixlError> {
        Ok(QueryResponseIterator {
            list: self,
            index: 0,
            len: self.len()?,
        })
    }

    pub(crate) fn handle(&self) -> *mut bindings::nixl_capi_query_resp_list_s {
        self.inner.as_ptr()
    }
}

impl<'a> QueryResponse<'a> {
    /// Returns true if this response contains parameters
    pub fn has_value(&self) -> Result<bool, NixlError> {
        let mut has_value = false;
        let status = unsafe {
            nixl_capi_query_resp_list_has_value(
                self.list.inner.as_ptr(),
                self.index,
                &mut has_value,
            )
        };

        match status {
            NIXL_CAPI_SUCCESS => Ok(has_value),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Gets the parameters if this response has a value
    pub fn get_params(&self) -> Result<Option<Params>, NixlError> {
        if !self.has_value()? {
            return Ok(None);
        }

        let mut params = ptr::null_mut();
        let status = unsafe {
            nixl_capi_query_resp_list_get_params(self.list.inner.as_ptr(), self.index, &mut params)
        };

        match status {
            NIXL_CAPI_SUCCESS => {
                // SAFETY: If status is NIXL_CAPI_SUCCESS, params is non-null
                let inner = unsafe { NonNull::new_unchecked(params) };
                Ok(Some(Params::new(inner)))
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }
}

/// An iterator over query responses
pub struct QueryResponseIterator<'a> {
    list: &'a QueryResponseList,
    index: usize,
    len: usize,
}

impl<'a> Iterator for QueryResponseIterator<'a> {
    type Item = QueryResponse<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.len {
            None
        } else {
            let response = QueryResponse {
                list: self.list,
                index: self.index,
            };
            self.index += 1;
            Some(response)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for QueryResponseIterator<'a> {
    fn len(&self) -> usize {
        self.len - self.index
    }
}

impl Drop for QueryResponseList {
    fn drop(&mut self) {
        // SAFETY: self.inner is guaranteed to be valid by NonNull
        unsafe {
            nixl_capi_destroy_query_resp_list(self.inner.as_ptr());
        }
    }
}
