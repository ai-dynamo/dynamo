// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod parser;
mod glm47_parser;

pub use super::response;
pub use parser::{
    detect_tool_call_start_xml, find_tool_call_end_position_xml, try_tool_call_parse_xml,
};
pub use glm47_parser::{
    detect_tool_call_start_glm47, find_tool_call_end_position_glm47, try_tool_call_parse_glm47,
};
