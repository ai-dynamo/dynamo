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

/// A safe wrapper around a list of strings from NIXL
pub struct StringList {
    inner: NonNull<bindings::nixl_capi_string_list_s>,
}

impl StringList {
    pub(crate) fn new(inner: NonNull<bindings::nixl_capi_string_list_s>) -> Self {
        Self { inner }
    }

    /// Returns the number of strings in the list
    pub fn len(&self) -> Result<usize, NixlError> {
        let mut size = 0;
        let status = unsafe { nixl_capi_string_list_size(self.inner.as_ptr(), &mut size) };

        match status {
            0 => Ok(size),
            -1 => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns true if the list contains no strings
    pub fn is_empty(&self) -> Result<bool, NixlError> {
        Ok(self.len()? == 0)
    }

    /// Returns the string at the given index
    pub fn get(&self, index: usize) -> Result<&str, NixlError> {
        let mut str_ptr = ptr::null();
        let status = unsafe { nixl_capi_string_list_get(self.inner.as_ptr(), index, &mut str_ptr) };

        match status {
            0 => {
                // SAFETY: If status is 0, str_ptr points to a valid null-terminated string
                let c_str = unsafe { CStr::from_ptr(str_ptr) };
                Ok(c_str.to_str().unwrap()) // Safe because NIXL strings are valid UTF-8
            }
            -1 => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns an iterator over the strings in the list
    pub fn iter(&self) -> StringListIterator<'_> {
        StringListIterator {
            list: self,
            index: 0,
            length: self.len().unwrap_or(0),
        }
    }
}

impl Drop for StringList {
    fn drop(&mut self) {
        // SAFETY: self.inner is guaranteed to be valid by NonNull
        unsafe {
            nixl_capi_destroy_string_list(self.inner.as_ptr());
        }
    }
}

/// An iterator over the strings in a StringList
pub struct StringListIterator<'a> {
    list: &'a StringList,
    index: usize,
    length: usize,
}

impl<'a> Iterator for StringListIterator<'a> {
    type Item = Result<&'a str, NixlError>;

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
