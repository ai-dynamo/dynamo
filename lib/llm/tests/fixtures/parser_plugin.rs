// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Build independently as a cdylib against the frontend-crates SDK.

use dynamo_parsers_v2::unified::plugin::{
    GuidanceChoice, GuidanceRequest, Hooks, Manifest, PromptRequest,
};
use dynamo_parsers_v2::{
    Tool, ToolCallDelta, UnifiedParser, UnifiedParserEvent, UnifiedParserOutput,
};

const TOOL: &str = "get_weather";
const ARGUMENTS: &str = r#"{"city":"Paris"}"#;
const GENERATION: &str = "R:look\nT:before\nC:weather\nT:after\n";

#[derive(Default)]
struct Lines {
    pending: String,
}

impl UnifiedParser for Lines {
    fn initialize_request(
        &mut self,
        init: dynamo_parsers_v2::UnifiedParserInit,
    ) -> anyhow::Result<()> {
        if init.starting_state == dynamo_parsers_v2::UnifiedParserStartingState::Reasoning {
            self.pending.push_str("R:");
        }
        Ok(())
    }

    fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> anyhow::Result<()> {
        self.pending.push_str(delta);
        while let Some((line, rest)) = self.pending.split_once('\n') {
            let event = match line {
                "C:weather" => UnifiedParserEvent::ToolCall(ToolCallDelta {
                    tool_index: 0,
                    name: Some(TOOL.into()),
                    arguments: ARGUMENTS.into(),
                    complete: true,
                }),
                _ if line.starts_with("R:") => UnifiedParserEvent::Reasoning(line[2..].into()),
                _ if line.starts_with("T:") => UnifiedParserEvent::Text(line[2..].into()),
                _ => anyhow::bail!("unexpected fixture record"),
            };
            output.events.push(event);
            self.pending = rest.to_owned();
        }
        Ok(())
    }

    fn finish(&mut self) -> anyhow::Result<UnifiedParserOutput> {
        anyhow::ensure!(self.pending.is_empty(), "unfinished fixture record");
        Ok(UnifiedParserOutput::default())
    }

    fn reset(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

fn manifest() -> Manifest {
    Manifest {
        family: "dis3562_lines".into(),
        aliases: vec!["dis3562_alias".into()],
        preserve_special_tokens: false,
        native_completion_authoritative: false,
        required_tokens: Vec::new(),
        structural_tag: true,
        inspect_prompt: true,
    }
}

fn create(tools: &[Tool]) -> anyhow::Result<Box<dyn UnifiedParser>> {
    anyhow::ensure!(
        tools.iter().any(|tool| tool.name == TOOL),
        "missing fixture tool"
    );
    Ok(Box::new(Lines::default()))
}

fn guide(request: GuidanceRequest) -> anyhow::Result<Option<serde_json::Value>> {
    Ok(matches!(request.choice, GuidanceChoice::Named(name) if name == TOOL).then(|| {
        serde_json::json!({"type":"structural_tag","format":{"type":"const_string","value":GENERATION}})
    }))
}

fn inspect(
    request: PromptRequest,
) -> anyhow::Result<dynamo_parsers_v2::unified::prefill::ParserPromptState> {
    use dynamo_parsers_v2::UnifiedParserStartingState;
    let prompt = request.prompt.trim_end();
    let reasoning = prompt.ends_with("R:");
    let response = prompt.ends_with("T:");
    Ok(dynamo_parsers_v2::unified::prefill::ParserPromptState {
        reasoning,
        explicit_reasoning: reasoning,
        starting_state: response.then_some(UnifiedParserStartingState::Response),
        prefix: response.then(|| "T:".into()),
        ..Default::default()
    })
}

dynamo_parsers_v2::export_unified_parser_plugin!(
    manifest,
    create,
    hooks = Hooks {
        guide: Some(guide),
        inspect_prompt: Some(inspect),
    }
);
