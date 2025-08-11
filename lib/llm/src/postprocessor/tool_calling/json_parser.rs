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




fn extract_json_content(&self, message: &str) -> &str {
	let trimmed = message.trim();
	
	// Handle list wrapper tokens
	if let (Some(start), Some(end)) = (&self.config.parallel_tool_start_token, &self.config.parallel_tool_end_token) {
		if trimmed.starts_with(start) && trimmed.ends_with(end) {
			return &trimmed[start.len()..trimmed.len() - end.len()];
		}
	}
	
	// Handle individual call wrapper tokens
	if let (Some(start), Some(end)) = (&self.config.call_start_token, &self.config.call_end_token) {
		if trimmed.starts_with(start) && trimmed.ends_with(end) {
			return &trimmed[start.len()..trimmed.len() - end.len()];
		}
	}
	
	// Handle special prefixes like <|python_tag|>
	if let Some(stripped) = trimmed.strip_prefix("<|python_tag|>") {
		return stripped;
	}
	
	trimmed
}

fn parse_single_call(&self, json_content: &str) -> anyhow::Result<Option<ToolCallResponse>> {
	// Try parsing as single function call
	if let Ok(function_call) = self.parse_function_call(json_content)? {
		return Ok(Some(function_call));
	}
	
	// Try parsing as list and take the last one
	if let Ok(mut calls) = self.parse_function_calls_list(json_content)? {
		if let Some(last_call) = calls.pop() {
			return Ok(Some(last_call));
		}
	}
	
	Ok(None)
}

fn parse_function_call(&self, json_content: &str) -> anyhow::Result<ToolCallResponse> {
	let use_parameters = self.config.use_parameters_field.unwrap_or(false);
	
	if use_parameters {
		// Try with "parameters" field
		if let Ok(deser) = serde_json::from_str::<HashMap<String, Value>>(json_content) {
			if let (Some(name), Some(params)) = (deser.get("name"), deser.get("parameters")) {
				if let (Some(name_str), Some(params_obj)) = (name.as_str(), params.as_object()) {
					return Ok(ToolCallResponse {
						id: format!("call-{}", Uuid::new_v4()),
						tp: ToolCallType::Function,
						function: CalledFunction {
							name: name_str.to_string(),
							arguments: serde_json::to_string(params_obj)?,
						},
					});
				}
			}
		}
	} else {
		// Try with "arguments" field
		if let Ok(deser) = serde_json::from_str::<HashMap<String, Value>>(json_content) {
			if let (Some(name), Some(args)) = (deser.get("name"), deser.get("arguments")) {
				if let (Some(name_str), Some(args_obj)) = (name.as_str(), args.as_object()) {
					return Ok(ToolCallResponse {
						id: format!("call-{}", Uuid::new_v4()),
						tp: ToolCallType::Function,
						function: CalledFunction {
							name: name_str.to_string(),
							arguments: serde_json::to_string(args_obj)?,
						},
					});
				}
			}
		}
	}
	
	anyhow::bail!("Failed to parse function call from JSON")
}

fn parse_function_calls_list(&self, json_content: &str) -> anyhow::Result<Vec<ToolCallResponse>> {
	let use_parameters = self.config.use_parameters_field.unwrap_or(false);
	
	if use_parameters {
		// Try parsing as list with "parameters" field
		if let Ok(list) = serde_json::from_str::<Vec<HashMap<String, Value>>>(json_content) {
			return list.into_iter()
				.filter_map(|item| {
					if let (Some(name), Some(params)) = (item.get("name"), item.get("parameters")) {
						if let (Some(name_str), Some(params_obj)) = (name.as_str(), params.as_object()) {
							return Some(Ok(ToolCallResponse {
								id: format!("call-{}", Uuid::new_v4()),
								tp: ToolCallType::Function,
								function: CalledFunction {
									name: name_str.to_string(),
									arguments: serde_json::to_string(params_obj).ok()?,
								},
							}));
						}
					}
					None
				})
				.collect();
		}
	} else {
		// Try parsing as list with "arguments" field
		if let Ok(list) = serde_json::from_str::<Vec<HashMap<String, Value>>>(json_content) {
			return list.into_iter()
				.filter_map(|item| {
					if let (Some(name), Some(args)) = (item.get("name"), item.get("arguments")) {
						if let (Some(name_str), Some(args_obj)) = (name.as_str(), args.as_object()) {
							return Some(Ok(ToolCallResponse {
								id: format!("call-{}", Uuid::new_v4()),
								tp: ToolCallType::Function,
								function: CalledFunction {
									name: name_str.to_string(),
									arguments: serde_json::to_string(args_obj).ok()?,
								},
							}));
						}
					}
					None
				})
				.collect();
		}
	}
	
	anyhow::bail!("Failed to parse function calls list from JSON")
}