// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Checks that the OpenEngine server serves the expected role and a model the
//! sidecar can register, and maps what it reports onto Dynamo's registration.

use std::collections::HashMap;
use std::sync::Arc;

use dynamo_backend_common::{
    DisaggregationMode, DynamoError, EngineConfig, LlmRegistration, WorkerConfig,
};
use dynamo_runtime::metrics::prometheus_names::labels;

use crate::args::Args;
use crate::client::{UNIDENTIFIED_PEER, invalid_argument};
use crate::pb;

/// The OpenEngine contract revision this sidecar implements.
const CLIENT_REVISION: u32 = 1;

/// The engine's name in messages and as the gRPC peer: the name it reports,
/// or a neutral one when it reports none.
pub(crate) fn engine_label(engine_name: &str) -> &str {
    if engine_name.is_empty() {
        UNIDENTIFIED_PEER
    } else {
        engine_name
    }
}

/// What the sidecar learned from the engine at startup.
#[derive(Clone, Debug)]
pub(crate) struct DiscoveredEngine {
    /// The `ServerInfo.engine_name` the engine reports, trimmed. Any name is
    /// served; it only names the engine in logs and messages.
    pub(crate) engine_name: Arc<str>,
    /// Logged only. Never registered: WorkerSet admission compares model card
    /// checksums, and per-process values would split one deployment.
    pub(crate) engine_version: String,
    /// Logged only, for the same reason.
    pub(crate) instance_id: String,
    pub(crate) role: pb::EngineRole,
    pub(crate) served_model_name: String,
    pub(crate) served_model_aliases: Vec<String>,
    pub(crate) context_length: u32,
    pub(crate) kv_block_size: Option<u32>,
    pub(crate) total_kv_blocks: Option<u64>,
    pub(crate) max_running_requests: Option<u64>,
    pub(crate) max_batched_tokens: Option<u64>,
    pub(crate) tool_call_parser: Option<String>,
    pub(crate) reasoning_parser: Option<String>,
}

impl DiscoveredEngine {
    /// Fails closed on any server this sidecar cannot serve correctly: an
    /// incompatible contract revision, the wrong role, or a model it cannot
    /// register.
    pub(crate) fn check(
        server: pb::ServerInfo,
        model: pb::ModelInfo,
        mode: DisaggregationMode,
    ) -> Result<Self, DynamoError> {
        let engine_name = server.engine_name.trim();
        let engine = engine_label(engine_name);
        if server.minimum_client_revision > CLIENT_REVISION {
            return Err(invalid_argument(format!(
                "{engine} requires OpenEngine client revision {} or newer; this sidecar \
                 implements revision {CLIENT_REVISION}",
                server.minimum_client_revision
            )));
        }

        let role =
            pb::EngineRole::try_from(server.engine_role).unwrap_or(pb::EngineRole::Unspecified);
        let expected = expected_role(mode)?;
        if role != expected {
            return Err(invalid_argument(format!(
                "{engine} reports engine_role {}, but --disaggregation-mode {mode} needs {}",
                role_name(role),
                role_name(expected)
            )));
        }

        let context_length = model
            .max_context_length
            .filter(|length| *length > 0)
            .ok_or_else(|| {
                invalid_argument(format!(
                    "{engine} reports no positive max_context_length; the worker has no context \
                     window to register"
                ))
            })?;
        let served_model_name = model.served_model_name.trim().to_owned();
        if served_model_name.is_empty() {
            return Err(invalid_argument(format!(
                "{engine} reports an empty served_model_name"
            )));
        }

        let capacity = server.capacity.unwrap_or_default();
        let kv_block_size = capacity.kv_block_size.filter(|size| *size > 0);
        Ok(Self {
            engine_name: engine_name.into(),
            engine_version: server.engine_version,
            instance_id: server.instance_id,
            role,
            served_model_name,
            served_model_aliases: model
                .served_model_aliases
                .into_iter()
                .map(|alias| alias.trim().to_owned())
                .filter(|alias| !alias.is_empty())
                .collect(),
            context_length,
            kv_block_size,
            // Counted in `kv_block_size` blocks, so meaningless without one.
            total_kv_blocks: kv_block_size.and(capacity.total_kv_blocks.filter(|count| *count > 0)),
            max_running_requests: capacity.max_running_requests.filter(|count| *count > 0),
            max_batched_tokens: capacity.max_batched_tokens.filter(|count| *count > 0),
            tool_call_parser: non_empty(model.tool_call_parser),
            reasoning_parser: non_empty(model.reasoning_parser),
        })
    }

    /// Checks the served model name the deployment expects, when it gives one,
    /// against the one the engine reports. The sidecar registers the engine's
    /// name, so a mismatch would serve the model under a name the deployment
    /// did not ask for.
    pub(crate) fn check_served_name(&self, expected: Option<&str>) -> Result<(), DynamoError> {
        let Some(expected) = expected.map(str::trim) else {
            return Ok(());
        };
        if expected == self.served_model_name {
            return Ok(());
        }
        Err(invalid_argument(format!(
            "{} reports served_model_name {:?}, but --served-model-name is {expected:?}",
            engine_label(&self.engine_name),
            self.served_model_name
        )))
    }

    /// The model is the served name too, so the worker's metrics carry it; the
    /// tokenizer source is `WorkerConfig.model_name`.
    pub(crate) fn engine_config(&self) -> EngineConfig {
        let runtime_data = HashMap::from([(
            "grpc_service".to_string(),
            serde_json::Value::String("openengine.v1.Inference".to_string()),
        )]);
        EngineConfig {
            model: self.served_model_name.clone(),
            served_model_name: Some(self.served_model_name.clone()),
            model_aliases: self.served_model_aliases.clone(),
            runtime_data,
            llm: Some(LlmRegistration {
                context_length: Some(self.context_length),
                kv_cache_block_size: self.kv_block_size,
                total_kv_blocks: self.total_kv_blocks,
                max_num_seqs: self.max_running_requests,
                max_num_batched_tokens: self.max_batched_tokens,
                data_parallel_size: Some(1),
                ..Default::default()
            }),
        }
    }

    /// The model path stays the tokenizer source; the served name is left to
    /// `EngineConfig`, which carries the engine's, and labels the metrics.
    pub(crate) fn worker_config(&self, args: Args) -> WorkerConfig {
        let common = args.sidecar.common;
        let mode = common.disaggregation_mode;
        WorkerConfig {
            runtime: common.runtime,
            namespace: common.namespace,
            // A decode worker registers under the decode role's component so
            // the frontend's prefill router finds it; an aggregated worker
            // keeps the operator-configured one.
            component: if mode.is_decode() {
                mode.discovery_component().to_string()
            } else {
                common.component
            },
            endpoint: common.endpoint,
            endpoint_types: common.endpoint_types,
            custom_jinja_template: common.custom_jinja_template,
            model_name: args.model_path,
            served_model_name: None,
            // An explicit Dynamo parser wins over the one the engine advertises.
            tool_call_parser: common
                .dyn_tool_call_parser
                .or_else(|| self.tool_call_parser.clone()),
            reasoning_parser: common
                .dyn_reasoning_parser
                .or_else(|| self.reasoning_parser.clone()),
            exclude_tools_when_tool_choice_none: common.exclude_tools_when_tool_choice_none,
            // An LPX engine publishes no KV events and hosts no KV indexer.
            enable_kv_routing: false,
            enable_local_indexer: false,
            disaggregation_mode: mode,
            route_to_encoder: false,
            enable_rl: common.enable_rl,
            metrics_labels: [labels::MODEL, labels::MODEL_NAME]
                .map(|label| (label.to_string(), self.served_model_name.clone()))
                .into(),
            ..Default::default()
        }
    }
}

/// The engine role a sidecar in `mode` serves; the LPX sidecar serves agg and
/// decode only.
pub(crate) fn expected_role(mode: DisaggregationMode) -> Result<pb::EngineRole, DynamoError> {
    match mode {
        DisaggregationMode::Aggregated => Ok(pb::EngineRole::Aggregated),
        DisaggregationMode::Decode => Ok(pb::EngineRole::Decode),
        DisaggregationMode::Prefill | DisaggregationMode::Encode => Err(invalid_argument(format!(
            "the LPX sidecar does not serve --disaggregation-mode {mode}"
        ))),
    }
}

pub(crate) fn role_name(role: pb::EngineRole) -> &'static str {
    match role {
        pb::EngineRole::Unspecified => "UNSPECIFIED",
        pb::EngineRole::Aggregated => "AGGREGATED",
        pb::EngineRole::Prefill => "PREFILL",
        pb::EngineRole::Decode => "DECODE",
    }
}

fn non_empty(value: String) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_owned())
}
