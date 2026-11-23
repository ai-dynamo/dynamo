// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// Finds the end of a leading JSON object or array across successive UTF-8 fragments.
/// Callers select the container opener. This tracks framing, not full JSON validity.
#[derive(Debug, Default)]
pub(crate) struct JsonContainerBoundary {
    depth: usize,
    in_string: bool,
    escaped: bool,
}

impl JsonContainerBoundary {
    /// Byte offset immediately after the root closer in this fragment.
    pub(crate) fn end_in(&mut self, fragment: &str) -> Option<usize> {
        self.scan(fragment, None)
    }

    /// An unquoted '<' cannot be JSON. Return it to the native grammar even
    /// when an invalid or unfinished container never supplies its closing bracket.
    pub(crate) fn end_before_markup(&mut self, fragment: &str) -> Option<usize> {
        self.scan(fragment, Some(b'<'))
    }

    fn scan(&mut self, fragment: &str, stop_before: Option<u8>) -> Option<usize> {
        for (offset, byte) in fragment.bytes().enumerate() {
            if self.in_string {
                if self.escaped {
                    self.escaped = false;
                } else if byte == b'\\' {
                    self.escaped = true;
                } else if byte == b'"' {
                    self.in_string = false;
                }
            } else if stop_before == Some(byte) {
                return Some(offset);
            } else {
                match byte {
                    b'"' => self.in_string = true,
                    b'{' | b'[' => self.depth += 1,
                    b'}' | b']' => {
                        self.depth -= 1;
                        if self.depth == 0 {
                            return Some(offset + 1);
                        }
                    }
                    _ => {}
                }
            }
        }
        None
    }
}
