// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod glm47_parser;
mod kimi_k2_parser;
mod parser;

pub use super::response;
pub use glm47_parser::{
    detect_tool_call_start_glm47, find_tool_call_end_position_glm47, try_tool_call_parse_glm47,
};
pub use kimi_k2_parser::{
    detect_tool_call_start_kimi_k2, find_tool_call_end_position_kimi_k2,
    try_tool_call_parse_kimi_k2,
};
pub use parser::{
    detect_tool_call_start_xml, find_tool_call_end_position_xml, try_tool_call_parse_xml,
};
