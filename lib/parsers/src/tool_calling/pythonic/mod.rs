// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod pythonic_parser;

pub use super::{config, response};
pub use pythonic_parser::{detect_tool_call_start_pythonic, try_tool_call_parse_pythonic};

pub fn find_tool_call_end_position_pythonic(chunk: &str) -> usize {
    chunk.len()
}
