// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native template metadata and custom/client-search protocol adaptation.

use super::super::tools::custom_input_or_raw;
use super::*;

#[derive(Clone, Debug, Default)]
pub struct ResponsesToolContext {
    pub tools: Vec<RequestTool>,
    pub declared_tools: Vec<RequestTool>,
    pub(super) native: bool,
    pub names: ToolNameMap,
    pub(super) legacy_history: Option<Vec<ChatCompletionRequestMessage>>,
    pub(super) phase_history: Option<(
        Vec<ChatCompletionRequestMessage>,
        HashMap<usize, MessagePhase>,
    )>,
    pub history_phases: HashMap<usize, MessagePhase>,
    pub(super) history_tool_names: HashMap<String, Vec<String>>,
    pub(super) tool_choice_name: Option<String>,
    pub(super) selected_tools: Option<HashSet<(Option<String>, String)>>,
}

fn search_call_id(id: Option<&str>) -> anyhow::Result<&str> {
    id.filter(|id| !id.is_empty()).ok_or_else(|| {
        ResponsesConversionError::InvalidArgument(
            "Client tool search requires a nonempty call_id".into(),
        )
        .into()
    })
}

pub(crate) fn effective_response_tools(
    request: &NvCreateResponse,
) -> anyhow::Result<Vec<RequestTool>> {
    let mut tools = request.inner.tools.clone().unwrap_or_default();
    let client_search = tools.iter().any(|tool| matches!(&tool.inner, Tool::ToolSearch(search) if search.execution == Some(ToolSearchExecutionType::Client)));
    if tools.iter().any(|tool| matches!(&tool.inner, Tool::ToolSearch(search) if search.execution != Some(ToolSearchExecutionType::Client))) {
        return Err(ResponsesConversionError::UnsupportedContent("Server-executed tool search is not supported".into()).into());
    }
    let search_count = tools
        .iter()
        .filter(|tool| matches!(&tool.inner, Tool::ToolSearch(_)))
        .count();
    if search_count > 1
        || client_search
            && tools.iter().any(|tool| {
                matches!(&tool.inner, Tool::Function(f) if f.name == "tool_search")
                    || matches!(&tool.inner, Tool::Custom(f) if f.name == "tool_search")
            })
    {
        return Err(ResponsesConversionError::InvalidArgument(
            "Client tool search has an ambiguous tool_search definition".into(),
        )
        .into());
    }
    if matches!(
        request.inner.tool_choice,
        Some(RequestToolChoiceParam::ToolSearch)
    ) && !client_search
    {
        return Err(ResponsesConversionError::InvalidArgument(
            "tool_choice=tool_search requires a client tool_search definition".into(),
        )
        .into());
    }
    let mut calls = std::collections::HashSet::new();
    let mut seen_calls = std::collections::HashSet::new();
    if let InputParam::Items(items) = &request.inner.input {
        for item in items {
            match item {
                InputItem::Item(Item::ToolSearchCall(call)) => {
                    if !client_search || call.execution == Some(ToolSearchExecutionType::Server) {
                        return Err(ResponsesConversionError::UnsupportedContent(
                            "Only configured client-executed tool search can be replayed".into(),
                        )
                        .into());
                    }
                    let id = search_call_id(call.call_id.as_deref())?;
                    if !call.arguments.is_object() || !seen_calls.insert(id.to_owned()) {
                        return Err(ResponsesConversionError::InvalidArgument(
                            "Tool search requires object arguments and unique call_id values"
                                .into(),
                        )
                        .into());
                    }
                    calls.insert(id.to_owned());
                }
                InputItem::Item(Item::ToolSearchOutput(result)) => {
                    let id = search_call_id(result.call_id.as_deref())?;
                    if result.execution == Some(ToolSearchExecutionType::Server)
                        || !calls.remove(id)
                    {
                        return Err(ResponsesConversionError::InvalidArgument(
                            "Client tool search output must match a preceding client call".into(),
                        )
                        .into());
                    }
                    for loaded in &result.tools {
                        match &loaded.inner {
                            Tool::Function(f) if f.name != "tool_search" => {
                                tools.retain(|tool| !matches!(&tool.inner, Tool::Function(old) if old.name == f.name) && !matches!(&tool.inner, Tool::Custom(old) if old.name == f.name));
                                tools.push(loaded.clone());
                            }
                            Tool::Custom(f) if f.name != "tool_search" => {
                                tools.retain(|tool| !matches!(&tool.inner, Tool::Function(old) if old.name == f.name) && !matches!(&tool.inner, Tool::Custom(old) if old.name == f.name));
                                tools.push(loaded.clone());
                            }
                            Tool::Namespace(ns) => {
                                if let Some(existing) = tools.iter_mut().find(|tool| matches!(&tool.inner, Tool::Namespace(old) if old.name == ns.name)) {
                                    let Tool::Namespace(old) = &mut existing.inner else { unreachable!() };
                                    for member in &ns.tools {
                                        let name = namespace_member_name(member);
                                        old.tools.retain(|old| namespace_member_name(old) != name);
                                        old.tools.push(member.clone());
                                    }
                                } else { tools.push(loaded.clone()); }
                            }
                            _ => return Err(ResponsesConversionError::InvalidArgument("Client tool search results must contain function, custom, or namespace tools".into()).into()),
                        }
                    }
                }
                _ => {}
            }
        }
    }
    validate_custom_formats(&tools)?;
    Ok(tools)
}

fn validate_custom_formats(tools: &[RequestTool]) -> anyhow::Result<()> {
    // Custom freeform text is supported; constrained grammars need a backend
    // enforcement path and must not be silently advertised as honored.
    for tool in tools {
        let formats: Vec<_> = match &tool.inner {
            Tool::Custom(custom) => vec![&custom.format],
            Tool::Namespace(ns) => ns
                .tools
                .iter()
                .filter_map(|member| match member {
                    NamespaceToolParamTool::Custom(custom) => Some(&custom.format),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        };
        for format in formats {
            if serde_json::to_value(format)?["type"] != "text" {
                return Err(ResponsesConversionError::UnsupportedContent(
                    "Custom tool grammar formats are not supported".into(),
                )
                .into());
            }
        }
    }
    Ok(())
}

fn namespace_member_name(member: &NamespaceToolParamTool) -> &str {
    match member {
        NamespaceToolParamTool::Function(f) => &f.name,
        NamespaceToolParamTool::Custom(f) => &f.name,
    }
}

pub(super) fn qualified_name(name: &str, namespace: Option<&str>) -> String {
    match namespace {
        Some(ns) => format!("{ns}.{name}"),
        None => name.to_owned(),
    }
}

fn convert_template_tools(
    tools: &[RequestTool],
    native: bool,
) -> anyhow::Result<Vec<ChatCompletionTool>> {
    let inner: Vec<_> = tools.iter().map(|tool| tool.inner.clone()).collect();
    let names = ToolNameMap::new(&inner, None);
    let mut converted = Vec::new();
    let mut origins = HashMap::<String, Option<String>>::new();
    let mut push = |name: &str,
                    description: &Option<String>,
                    parameters: Option<serde_json::Value>,
                    strict: Option<bool>,
                    namespace: Option<&str>|
     -> anyhow::Result<()> {
        let registered = if native {
            qualified_name(name, namespace)
        } else {
            names.encode(namespace, name)
        };
        if let Some(previous) = origins.get(&registered) {
            if previous.as_deref() != namespace {
                return Err(ResponsesConversionError::InvalidArgument(
                    "Responses function tool names are ambiguous after namespace flattening".into(),
                )
                .into());
            }
        } else {
            origins.insert(registered.clone(), namespace.map(str::to_owned));
        }
        converted.push(ChatCompletionTool {
            custom: None,
            r#type: ChatCompletionToolType::Function,
            defer_loading: None,
            function: FunctionObject {
                name: registered,
                description: description.clone(),
                parameters,
                strict,
                defer_loading: None,
            },
        });
        Ok(())
    };
    for tool in tools {
        match &tool.inner {
            Tool::Function(f) => push(
                &f.name,
                &f.description,
                f.parameters.clone(),
                f.strict,
                None,
            )?,
            Tool::Custom(f) => push(
                &f.name,
                &f.description,
                Some(custom_tool_parameters()),
                None,
                None,
            )?,
            Tool::Namespace(ns) => {
                for member in &ns.tools {
                    match member {
                        NamespaceToolParamTool::Function(f) => push(
                            &f.name,
                            &f.description,
                            f.parameters.clone(),
                            f.strict,
                            Some(&ns.name),
                        )?,
                        NamespaceToolParamTool::Custom(f) => push(
                            &f.name,
                            &f.description,
                            Some(custom_tool_parameters()),
                            None,
                            Some(&ns.name),
                        )?,
                    }
                }
            }
            Tool::ToolSearch(search) => {
                if let Some(value) = client_tool_search_function(search) {
                    let f = &value["function"];
                    push(
                        "tool_search",
                        &f["description"].as_str().map(str::to_owned),
                        Some(f["parameters"].clone()),
                        None,
                        None,
                    )?;
                }
            }
            _ => return unsupported_tool(&tool.inner, "tools"),
        }
    }
    Ok(converted)
}

impl NvCreateChatCompletionRequest {
    pub(crate) fn validated_effective_tools(
        &self,
    ) -> anyhow::Result<std::borrow::Cow<'_, [ChatCompletionTool]>> {
        if self.responses_tools.is_some() {
            self.validate_responses_tool_definitions()?;
            Ok(std::borrow::Cow::Borrowed(
                self.inner.tools.as_deref().unwrap_or_default(),
            ))
        } else {
            crate::protocols::openai::validate::validated_effective_tools(&self.inner)
        }
    }

    /// Qualified names are an internal Responses representation. Validate
    /// caller-supplied namespace and leaf names independently so the ordinary
    /// Chat Completions name rules remain unchanged.
    pub(crate) fn validate_responses_tool_definitions(&self) -> anyhow::Result<()> {
        use crate::protocols::openai::validate::validate_tools;
        let Some(context) = &self.responses_tools else {
            return validate_tools(&self.inner.tools.as_deref());
        };
        let mut definitions = Vec::new();
        for tool in &context.tools {
            if let Tool::Namespace(namespace) = &tool.inner {
                let name = ChatCompletionTool {
                    custom: None,
                    r#type: ChatCompletionToolType::Function,
                    defer_loading: None,
                    function: FunctionObject {
                        name: namespace.name.clone(),
                        ..Default::default()
                    },
                };
                validate_tools(&Some(std::slice::from_ref(&name)))?;
                for member in &namespace.tools {
                    let leaf = ChatCompletionTool {
                        custom: None,
                        r#type: ChatCompletionToolType::Function,
                        defer_loading: None,
                        function: FunctionObject {
                            name: namespace_member_name(member).to_owned(),
                            ..Default::default()
                        },
                    };
                    validate_tools(&Some(std::slice::from_ref(&leaf)))?;
                }
            }
            definitions.extend(convert_template_tools(std::slice::from_ref(tool), false)?);
        }
        validate_tools(&Some(definitions.as_slice()))
    }

    /// Phase replay is a model capability, independent of namespace rendering.
    pub(crate) fn normalize_responses_history_phases(&mut self, enabled: bool) {
        if let Some(context) = &mut self.responses_tools
            && let Some((messages, phases)) = &context.phase_history
        {
            if enabled {
                self.inner.messages = messages.clone();
                context.history_phases = phases.clone();
            } else if let Some(messages) = &context.legacy_history {
                self.inner.messages = messages.clone();
                context.history_phases.clear();
            }
        }
    }

    pub(crate) fn normalize_responses_tools(&mut self, native: bool) -> anyhow::Result<()> {
        if !native {
            self.normalize_responses_history_phases(false);
        }
        if let Some(context) = &mut self.responses_tools {
            context.native = native;
        }
        let Some(context) = &self.responses_tools else {
            return Ok(());
        };
        let inner: Vec<_> = context
            .tools
            .iter()
            .map(|tool| tool.inner.clone())
            .collect();
        let names = if native {
            context.names.native()
        } else {
            context.names.clone()
        };
        let mut tools = super::convert_tools(&inner, &names)?;
        if let Some(selected) = &context.selected_tools {
            tools.retain(|tool| {
                let (namespace, name) = names.decode(&tool.function.name);
                selected.contains(&(namespace.map(str::to_owned), name.to_owned()))
            });
        }
        self.inner.tools = Some(tools).filter(|tools| !tools.is_empty());
        let resolve = |name: &str| -> String {
            let (namespace, leaf) = context.names.decode(name);
            if native {
                names.encode_history(namespace, leaf)
            } else {
                names.encode(namespace, leaf)
            }
        };
        let mut occurrences: HashMap<&str, usize> = HashMap::new();
        for message in &mut self.inner.messages {
            if let ChatCompletionRequestMessage::Assistant(message) = message {
                for call in message.tool_calls.iter_mut().flatten() {
                    let occurrence = occurrences.entry(&call.id).or_default();
                    let original = context
                        .history_tool_names
                        .get(&call.id)
                        .and_then(|names| names.get(*occurrence))
                        .unwrap_or(&call.function.name);
                    call.function.name = resolve(original);
                    *occurrence += 1;
                }
            }
        }
        if let Some(ChatCompletionToolChoiceOption::Named(choice)) = &mut self.inner.tool_choice {
            let original = context
                .tool_choice_name
                .as_deref()
                .unwrap_or(&choice.function.name);
            choice.function.name = resolve(original);
        }
        Ok(())
    }

    /// Serialize an in-process request for a Python chat processor. The model's
    /// configured parsers opt into phase replay; metadata stays off worker wires.
    pub fn python_chat_processor_request_json(
        &self,
        runtime_config: &crate::local_model::runtime_config::ModelRuntimeConfig,
    ) -> Result<serde_json::Value, serde_json::Error> {
        let mut value = serde_json::to_value(self)?;
        if let Some(context) = &self.responses_tools {
            if let Some(template_tools) = self.native_template_tools() {
                // This catalog is renderer-only: Python chooses it after resolving
                // the active template. Keep worker serialization unchanged and
                // switch engine names, selected choice and history as one view.
                let mut native = self.clone();
                native.normalize_responses_history_phases(
                    runtime_config.preserves_response_message_phases(),
                );
                native
                    .normalize_responses_tools(true)
                    .map_err(serde::ser::Error::custom)?;
                let native = serde_json::to_value(native)?;
                // Match the production Rust preprocessor's Spark-only policy.
                // Other families retain the full native catalog for named choices.
                let render_named_tool_eagerly =
                    super::super::chat_completions::unified_parser::configured_family(
                        runtime_config.tool_call_parser.as_deref(),
                        runtime_config.reasoning_parser.as_deref(),
                    ) == Some("muse_spark");
                value["responses_native_tools"] = serde_json::json!({
                    "render_named_tool_eagerly": render_named_tool_eagerly,
                    "template_tools": template_tools,
                    "tools": native["tools"],
                    "tool_choice": native["tool_choice"],
                    "messages": native["messages"],
                });
            }
            let history = if runtime_config.preserves_response_message_phases() {
                context.phase_history.as_ref()
            } else {
                None
            };
            if let Some((messages, history_phases)) = history {
                value["messages"] = serde_json::to_value(messages)?;
                // Assistant ordinals survive multimodal user-message splitting
                // and system-message consolidation; absolute indices do not.
                let phases: Vec<Option<MessagePhase>> = messages
                    .iter()
                    .enumerate()
                    .filter(|(_, message)| {
                        matches!(message, ChatCompletionRequestMessage::Assistant(_))
                    })
                    .map(|(index, _)| history_phases.get(&index).cloned())
                    .collect();
                value["responses_history_phases"] = serde_json::to_value(phases)?;
            } else if let Some(messages) = &context.legacy_history {
                value["messages"] = serde_json::to_value(messages)?;
            }
        }
        Ok(value)
    }

    pub(crate) fn template_messages_json(&self) -> serde_json::Value {
        let mut messages = self.inner.function_template_messages();
        if let Some(context) = &self.responses_tools {
            for (index, phase) in &context.history_phases {
                if let Some(message) = messages.get_mut(*index) {
                    message["recipient"] = serde_json::json!(match phase {
                        MessagePhase::Commentary => "commentary",
                        MessagePhase::FinalAnswer => "user",
                    });
                }
            }
        }
        messages
    }

    fn allowed_template_tools(&self) -> Option<Vec<RequestTool>> {
        let context = self.responses_tools.as_ref()?;
        let Some(selected) = &context.selected_tools else {
            return Some(context.declared_tools.clone());
        };
        Some(
            context
                .declared_tools
                .iter()
                .filter_map(|tool| {
                    let mut tool = tool.clone();
                    let keep = match &mut tool.inner {
                        Tool::Namespace(namespace) => {
                            namespace.tools.retain(|member| {
                                selected.contains(&(
                                    Some(namespace.name.clone()),
                                    namespace_member_name(member).to_owned(),
                                ))
                            });
                            !namespace.tools.is_empty()
                        }
                        Tool::Function(f) => selected.contains(&(None, f.name.clone())),
                        Tool::Custom(f) => selected.contains(&(None, f.name.clone())),
                        Tool::ToolSearch(_) => selected.contains(&(None, "tool_search".to_owned())),
                        _ => false,
                    };
                    keep.then_some(tool)
                })
                .collect(),
        )
    }

    pub(crate) fn declared_template_tools(&self) -> Option<serde_json::Value> {
        let context = self.responses_tools.as_ref()?;
        let declared: Vec<_> = self
            .allowed_template_tools()?
            .into_iter()
            .map(|tool| tool.inner)
            .collect();
        let names = if context.native {
            context.names.native()
        } else {
            context.names.clone()
        };
        let tools = super::convert_tools(&declared, &names).expect("validated declared registry");
        Some(serde_json::to_value(tools).expect("typed tools serialize"))
    }

    pub(crate) fn native_template_tools(&self) -> Option<minijinja::Value> {
        let tools = self.allowed_template_tools()?;
        let wants_native = tools.iter().any(|tool| match &tool.inner {
            Tool::Namespace(_) | Tool::ToolSearch(_) => true,
            Tool::Function(f) => f.defer_loading == Some(true),
            Tool::Custom(f) => f.defer_loading == Some(true),
            _ => false,
        });
        wants_native
            .then(|| minijinja::Value::from_serialize(native_tools_for_chat_template(&tools)))
    }
}

impl ResponseParams {
    pub(super) fn is_client_search(&self, name: &str) -> bool {
        let names = self.tool_name_map();
        let (namespace, leaf) = names.decode(name);
        namespace.is_none() && leaf == "tool_search"
            && self.tools.iter().flatten().any(|tool| matches!(tool, Tool::ToolSearch(search) if search.execution == Some(ToolSearchExecutionType::Client)))
    }

    pub(super) fn tool_identity(&self, emitted: &str) -> (String, Option<String>, bool) {
        let names = self.tool_name_map();
        let (decoded_namespace, decoded_name) = names.decode(emitted);
        let resolved = qualified_name(decoded_name, decoded_namespace);
        let emitted = resolved.as_str();
        let tools = if self.loaded_tools.is_empty() {
            self.tools.as_deref().unwrap_or_default()
        } else {
            self.loaded_tools.as_slice()
        };
        // Match qualified names before considering a legacy bare member name.
        for tool in tools {
            if let Tool::Namespace(ns) = tool {
                for member in &ns.tools {
                    let (name, custom) = match member {
                        NamespaceToolParamTool::Function(f) => (&f.name, false),
                        NamespaceToolParamTool::Custom(f) => (&f.name, true),
                    };
                    if qualified_name(name, Some(&ns.name)) == emitted {
                        return (name.clone(), Some(ns.name.clone()), custom);
                    }
                }
            }
        }
        for tool in tools {
            match tool {
                Tool::Function(f) if f.name == emitted => return (emitted.into(), None, false),
                Tool::Custom(f) if f.name == emitted => return (emitted.into(), None, true),
                _ => {}
            }
        }
        let mut found = None;
        for tool in tools {
            if let Tool::Namespace(ns) = tool {
                for member in &ns.tools {
                    let (name, custom) = match member {
                        NamespaceToolParamTool::Function(f) => (&f.name, false),
                        NamespaceToolParamTool::Custom(f) => (&f.name, true),
                    };
                    if name == emitted {
                        if found.is_some() {
                            return (emitted.into(), None, false);
                        }
                        found = Some((name.clone(), Some(ns.name.clone()), custom));
                    }
                }
            }
        }
        found.unwrap_or_else(|| {
            (
                decoded_name.to_owned(),
                decoded_namespace.map(str::to_owned),
                false,
            )
        })
    }

    pub(super) fn tool_output(
        &self,
        item_id: String,
        call_id: String,
        emitted: &str,
        arguments: &str,
        status: OutputStatus,
    ) -> anyhow::Result<OutputItem> {
        if self.is_client_search(emitted) {
            let arguments = match serde_json::from_str::<serde_json::Value>(arguments) {
                Ok(value) if value.is_object() => value,
                _ if status == OutputStatus::Incomplete
                    || (arguments.is_empty() && status == OutputStatus::InProgress) =>
                {
                    // A truncated search has no executable argument object.
                    serde_json::json!({})
                }
                Ok(_) => anyhow::bail!("Client tool search arguments must be an object"),
                Err(error) => return Err(error.into()),
            };
            return Ok(serde_json::from_value(
                serde_json::json!({"type":"tool_search_call","id":item_id,"call_id":call_id,"execution":"client","arguments":arguments,"status":status}),
            )?);
        }
        let (name, namespace, custom) = self.tool_identity(emitted);
        if custom {
            let input = if (arguments.is_empty() && status == OutputStatus::InProgress)
                || (status == OutputStatus::Incomplete
                    && serde_json::from_str::<serde_json::Value>(arguments).is_err())
            {
                // Match Chat: a token limit must not expose a partial adapter wrapper.
                String::new()
            } else {
                // Completed malformed output keeps the existing raw-text fallback.
                custom_input_or_raw(arguments)
            };
            // The custom item has no status field; the response signals truncation.
            // Construct this non_exhaustive type through its wire representation.
            Ok(serde_json::from_value(
                serde_json::json!({"type":"custom_tool_call", "id":item_id,
                "call_id":call_id,"name":name,"namespace":namespace,"input":input}),
            )?)
        } else {
            Ok(OutputItem::FunctionCall(FunctionToolCall {
                caller: None,
                r#async: None,
                id: Some(item_id),
                call_id,
                name,
                namespace,
                arguments: arguments.to_owned(),
                status: Some(status),
            }))
        }
    }

    pub(super) fn echoed_tools(&self) -> Option<Vec<RequestTool>> {
        let mut tools = self.template_tools.clone()?;
        for tool in &mut tools {
            if let Tool::Function(function) = &mut tool.inner {
                function.strict.get_or_insert(true);
            }
            if let Tool::Namespace(ns) = &mut tool.inner {
                for member in &mut ns.tools {
                    if let NamespaceToolParamTool::Function(function) = member {
                        function.strict.get_or_insert(true);
                    }
                }
            }
        }
        Some(tools)
    }
}

/// Normalize new input item kinds at the API boundary, then reuse main's
/// message converter for media, reasoning, refusal and tool-result semantics.
pub(super) fn normalize_template_input(item: &InputItem) -> anyhow::Result<InputItem> {
    let InputItem::Item(inner) = item else {
        return Ok(item.clone());
    };
    let mut value = serde_json::to_value(item)?;
    match inner {
        Item::CustomToolCall(call) => {
            value["type"] = serde_json::json!("function_call");
            value["arguments"] = serde_json::json!(
                serde_json::json!({CUSTOM_TOOL_INPUT_ARG: call.input}).to_string()
            );
        }
        Item::CustomToolCallOutput(_) => {
            value["type"] = serde_json::json!("function_call_output");
        }
        Item::ToolSearchCall(call) => {
            value["type"] = serde_json::json!("function_call");
            value["name"] = serde_json::json!("tool_search");
            value["call_id"] = serde_json::json!(search_call_id(call.call_id.as_deref())?);
            value["arguments"] = serde_json::json!(call.arguments.to_string());
        }
        Item::ToolSearchOutput(result) => {
            value = serde_json::json!({"type":"function_call_output", "call_id":search_call_id(result.call_id.as_deref())?, "output":serde_json::to_string(&result.tools)?});
        }
        _ => return Ok(item.clone()),
    }
    Ok(serde_json::from_value(value)?)
}

pub(super) fn convert_input_items_with_phases(
    items: &[InputItem],
    names: &ToolNameMap,
) -> anyhow::Result<(
    Vec<ChatCompletionRequestMessage>,
    HashMap<usize, MessagePhase>,
)> {
    let mut messages = Vec::new();
    let mut phases = HashMap::new();
    let mut pending = Vec::new();
    for item in items {
        let phase = match item {
            InputItem::Item(Item::Message(MessageItem::Output(message))) => message.phase,
            InputItem::EasyMessage(message) if message.role == ResponseRole::Assistant => {
                message.phase
            }
            _ => None,
        };
        if let Some(phase) = phase {
            messages.extend(convert_input_items_to_messages(&pending, names)?);
            pending.clear();
            let current = convert_input_items_to_messages(std::slice::from_ref(item), names)?;
            for (offset, message) in current.iter().enumerate() {
                if matches!(message, ChatCompletionRequestMessage::Assistant(_)) {
                    phases.insert(messages.len() + offset, phase);
                }
            }
            messages.extend(current);
        } else {
            pending.push(normalize_template_input(item)?);
        }
    }
    messages.extend(convert_input_items_to_messages(&pending, names)?);
    Ok((messages, phases))
}
