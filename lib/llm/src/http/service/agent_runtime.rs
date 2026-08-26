// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::{Arc, LazyLock};

use agent_rt_sandbox::{
    HttpSandboxProvider, HttpSandboxProviderConfig, HttpSandboxProviderError, SandboxFailurePolicy,
    SandboxProviderExecutor, SandboxToolError, SandboxToolExecutorConfig,
};
use axum::body::to_bytes;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use dynamo_agent_rt::{
    AgentRuntime, AgentRuntimeError, AgentStreamRuntimeError, AgentToolRuntimeError,
    AnthropicMaterializationError, AnthropicMessages, AnthropicRequestMaterializer,
    AnthropicRequestSelector, AnthropicStreamEventError, AnthropicStreamEventInterpreter,
    AnthropicToolLoopAdapter, AnthropicUuidGenerator, AuthorizationScope,
    Blake3ToolIdempotencyKeys, BoxFuture, CanonicalJsonFingerprinter, ConfiguredToolRouter,
    IdempotencyKey, InferenceFuture, InferenceIntent, InferenceInvoker, InferenceOutput,
    InferenceRequest, MaterializationError, ModelStepKind, OpenAiResponses,
    PolicyAnthropicOutputInterpreter, PolicyResponsesOutputInterpreter, ResponseId,
    ResponsesRequestMaterializer, ResponsesStreamEventInterpreter, ResponsesToolAdapterError,
    ResponsesToolLoopAdapter, RoutedAnthropicOutcomeError, RoutedAnthropicOutcomePolicy,
    RoutedResponsesOutcomeError, RoutedResponsesOutcomePolicy, RunStreamResult, RunTurn,
    RunTurnResult, RuntimeAuthorization, RuntimeLimits, RuntimeSelectionContext, RuntimeSelector,
    SystemClock, ToolExecutionFailure, ToolExecutionRequest, ToolExecutionResult, ToolExecutor,
    ToolFailureDisposition, ToolFailurePolicy, ToolRoute, ToolRouter, ToolRunError, ToolRunner,
    TurnState, UuidGenerator,
};
use dynamo_agent_rt_store::{DuckDbStore, DuckDbStoreError, StoreInvariantError};
use dynamo_agent_tools::{
    BraveWebSearchError, BraveWebSearchExecutor, BraveWebSearchFailurePolicy, BraveWebSearchProfile,
};
use dynamo_protocols::types::anthropic::{
    AnthropicCreateMessageRequest, AnthropicDelta, AnthropicErrorResponse,
    AnthropicMessageDeltaBody, AnthropicMessageResponse, AnthropicResponseContentBlock,
    AnthropicStreamEvent,
};
use dynamo_protocols::types::responses::{
    CreateResponse, ResponseCompletedEvent, ResponseCreatedEvent, ResponseInProgressEvent,
    ResponseOutputItemDoneEvent, ResponseStreamEvent, Status, Tool,
};
use dynamo_runtime::pipeline::{AsyncEngineContextProvider, Context};
use futures::{Stream, StreamExt, stream};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use thiserror::Error;

use super::disconnect::create_connection_monitor;
use super::metrics::{CancellationLabels, Endpoint};
use super::{anthropic, metadata, openai, service_v2};
use crate::protocols::agents::{agent_context_header_values, session_affinity_header_value};
use crate::protocols::common::extensions::{
    AGENT_CONTEXT_CONTEXT_KEY, AgentContext, HEADER_DATA_PARALLEL_RANK_ALIAS, HEADER_DP_RANK,
    HEADER_DP_RANK_ALIAS, HEADER_PREFILL_DP_RANK, HEADER_PREFILL_DP_RANK_ALIAS,
    HEADER_PREFILL_INSTANCE_ID, HEADER_PREFILL_INSTANCE_ID_ALIAS, HEADER_REQUEST_PRIORITY,
    HEADER_REQUEST_STRICT_PRIORITY, HEADER_TENANT_ID, HEADER_WORKER_INSTANCE_ID,
    HEADER_WORKER_INSTANCE_ID_ALIAS, NvExt, SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId,
};
use crate::protocols::common::input_trigger::{
    classify_anthropic_request, classify_response_request,
};
use crate::protocols::openai::responses::stream_converter::ResponseEventSerializer;
use crate::protocols::openai::responses::{NvCreateResponse, NvResponse, ResponseParams};
use crate::request_template::{RequestTemplate, resolve_request_model};

static ENABLED: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("DYN_ENABLE_AGENT_RT_POC")
        .ok()
        .and_then(|value| value.parse::<bool>().ok())
        .unwrap_or(false)
});

const AUTH_MODE_ENV: &str = "DYN_AGENT_RT_AUTH_MODE";
const LOCAL_TENANT_ENV: &str = "DYN_AGENT_RT_LOCAL_TENANT_ID";
const LOCAL_PRINCIPAL_ENV: &str = "DYN_AGENT_RT_LOCAL_PRINCIPAL_ID";
const TRUSTED_PROXY_TOKEN_ENV: &str = "DYN_AGENT_RT_TRUSTED_PROXY_TOKEN";
const PERMITTED_CONNECTORS_ENV: &str = "DYN_AGENT_RT_PERMITTED_CONNECTORS";
const AUTH_HEADER: &str = "x-dynamo-agent-rt-auth";
const TENANT_HEADER: &str = "x-dynamo-tenant-id";
const PRINCIPAL_HEADER: &str = "x-dynamo-principal-id";
const DUCKDB_PATH_ENV: &str = "DYN_AGENT_RT_DUCKDB_PATH";
const STATEFUL_ANTHROPIC_ENV: &str = "DYN_AGENT_RT_STATEFUL_ANTHROPIC";
const WEB_SEARCH_API_KEY_ENV: &str = "BRAVE_SEARCH_API_KEY";
const WEB_SEARCH_TOOL_NAME_ENV: &str = "DYN_AGENT_RT_WEB_SEARCH_TOOL_NAME";
const SANDBOX_ENDPOINT_ENV: &str = "DYN_AGENT_RT_SANDBOX_ENDPOINT";
const SANDBOX_TOKEN_ENV: &str = "DYN_AGENT_RT_SANDBOX_TOKEN";
const SANDBOX_ALLOW_HTTP_ENV: &str = "DYN_AGENT_RT_SANDBOX_ALLOW_HTTP";
const SANDBOX_TOOL_NAME_ENV: &str = "DYN_AGENT_RT_SANDBOX_TOOL_NAME";
const SANDBOX_PROFILE_ENV: &str = "DYN_AGENT_RT_SANDBOX_PROFILE";

static AUTH_CONFIG: LazyLock<Result<AgentRuntimeAuthConfig, String>> =
    LazyLock::new(AgentRuntimeAuthConfig::from_environment);

pub(super) fn enabled() -> bool {
    *ENABLED
}

struct AgentRuntimeAuthConfig {
    mode: AgentRuntimeAuthMode,
    permitted_connectors: BTreeSet<String>,
}

enum AgentRuntimeAuthMode {
    Local(AuthorizationScope),
    TrustedProxy { token_sha256: [u8; 32] },
}

#[derive(Debug, Error, PartialEq, Eq)]
enum IngressAuthorizationError {
    #[error("trusted proxy authentication failed")]
    Unauthorized,
    #[error("trusted proxy identity is missing or invalid")]
    InvalidIdentity,
}

impl AgentRuntimeAuthConfig {
    fn from_environment() -> Result<Self, String> {
        Self::from_lookup(|name| std::env::var(name).ok())
    }

    fn from_lookup(mut lookup: impl FnMut(&str) -> Option<String>) -> Result<Self, String> {
        let mode = match lookup(AUTH_MODE_ENV).as_deref() {
            Some("local") => {
                let tenant_id = lookup(LOCAL_TENANT_ENV).unwrap_or_else(|| "local".to_owned());
                let principal_id =
                    lookup(LOCAL_PRINCIPAL_ENV).unwrap_or_else(|| "local".to_owned());
                validate_scope_component(&tenant_id)
                    .map_err(|error| format!("invalid {LOCAL_TENANT_ENV}: {error}"))?;
                validate_scope_component(&principal_id)
                    .map_err(|error| format!("invalid {LOCAL_PRINCIPAL_ENV}: {error}"))?;
                AgentRuntimeAuthMode::Local(AuthorizationScope {
                    tenant_id,
                    principal_id,
                })
            }
            Some("trusted_proxy") => {
                let token = lookup(TRUSTED_PROXY_TOKEN_ENV).ok_or_else(|| {
                    format!("{TRUSTED_PROXY_TOKEN_ENV} is required in trusted_proxy mode")
                })?;
                if !(32..=1024).contains(&token.len()) {
                    return Err(format!(
                        "{TRUSTED_PROXY_TOKEN_ENV} must contain 32 to 1024 bytes"
                    ));
                }
                AgentRuntimeAuthMode::TrustedProxy {
                    token_sha256: Sha256::digest(token.as_bytes()).into(),
                }
            }
            Some(mode) => {
                return Err(format!(
                    "unsupported {AUTH_MODE_ENV} value {mode:?}; expected local or trusted_proxy"
                ));
            }
            None => return Err(format!("{AUTH_MODE_ENV} must be set")),
        };
        let permitted_connectors = lookup(PERMITTED_CONNECTORS_ENV)
            .map(|value| parse_connector_allowlist(&value))
            .transpose()?
            .unwrap_or_default();
        Ok(Self {
            mode,
            permitted_connectors,
        })
    }

    fn authorize(
        &self,
        headers: &HeaderMap,
    ) -> Result<RuntimeAuthorization, IngressAuthorizationError> {
        let scope = match &self.mode {
            AgentRuntimeAuthMode::Local(scope) => scope.clone(),
            AgentRuntimeAuthMode::TrustedProxy { token_sha256 } => {
                let provided_token = header_value(headers, AUTH_HEADER)
                    .ok_or(IngressAuthorizationError::Unauthorized)?;
                let provided_sha256: [u8; 32] = Sha256::digest(provided_token.as_bytes()).into();
                if !bool::from(token_sha256.ct_eq(&provided_sha256)) {
                    return Err(IngressAuthorizationError::Unauthorized);
                }
                let tenant_id = header_value(headers, TENANT_HEADER)
                    .filter(|value| validate_scope_component(value).is_ok())
                    .ok_or(IngressAuthorizationError::InvalidIdentity)?;
                let principal_id = header_value(headers, PRINCIPAL_HEADER)
                    .filter(|value| validate_scope_component(value).is_ok())
                    .ok_or(IngressAuthorizationError::InvalidIdentity)?;
                AuthorizationScope {
                    tenant_id,
                    principal_id,
                }
            }
        };
        Ok(RuntimeAuthorization {
            scope,
            permitted_connectors: self.permitted_connectors.clone(),
            limits: RuntimeLimits::default(),
        })
    }
}

fn validate_scope_component(value: &str) -> Result<(), &'static str> {
    if !(1..=128).contains(&value.len()) {
        return Err("must contain 1 to 128 bytes");
    }
    if !value.bytes().all(|byte| {
        byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':' | b'/' | b'@')
    }) {
        return Err("contains unsupported characters");
    }
    Ok(())
}

fn parse_connector_allowlist(value: &str) -> Result<BTreeSet<String>, String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|connector| !connector.is_empty())
        .map(|connector| {
            validate_scope_component(connector)
                .map_err(|error| format!("invalid connector {connector:?}: {error}"))?;
            Ok(connector.to_owned())
        })
        .collect()
}

fn ingress_authorization(
    headers: &HeaderMap,
) -> Result<RuntimeAuthorization, openai::ErrorResponse> {
    let config = AUTH_CONFIG.as_ref().map_err(|error| {
        tracing::error!(%error, "agent runtime ingress authorization configuration is invalid");
        openai::ErrorMessage::service_unavailable_with_body(
            "Agent runtime ingress is not configured".to_owned(),
        )
    })?;
    config.authorize(headers).map_err(|error| {
        tracing::warn!(%error, "agent runtime ingress authorization rejected request");
        match error {
            IngressAuthorizationError::Unauthorized => openai::ErrorMessage::agent_runtime_error(
                StatusCode::UNAUTHORIZED,
                "Agent runtime authentication failed",
            ),
            IngressAuthorizationError::InvalidIdentity => {
                openai::ErrorMessage::agent_runtime_error(
                    StatusCode::BAD_REQUEST,
                    "Trusted proxy identity is missing or invalid",
                )
            }
        }
    })
}

type ResponsesStore = DuckDbStore<OpenAiResponses>;

#[derive(Clone, Default)]
struct RuntimeToolRouter {
    web_search: ConfiguredToolRouter,
    sandbox: ConfiguredToolRouter,
}

impl ToolRouter for RuntimeToolRouter {
    fn route(&self, tool_name: &str) -> Option<ToolRoute> {
        self.web_search
            .route(tool_name)
            .or_else(|| self.sandbox.route(tool_name))
    }
}

type ResponsesOutcomePolicy = RoutedResponsesOutcomePolicy<RuntimeToolRouter>;
type ResponsesOutputInterpreter = PolicyResponsesOutputInterpreter<ResponsesOutcomePolicy>;
type ResponsesAgentRuntimeCore = AgentRuntime<
    OpenAiResponses,
    ResponsesStore,
    ResponsesRequestMaterializer,
    CanonicalJsonFingerprinter,
    DynamoResponsesInvoker,
    ResponsesOutputInterpreter,
    UuidGenerator,
    SystemClock,
>;
type ResponsesToolAdapter = ResponsesToolLoopAdapter<RuntimeToolRouter>;
type ResponsesToolRunner = ToolRunner<
    ResponsesStore,
    RuntimeToolExecutor,
    Blake3ToolIdempotencyKeys,
    RuntimeToolFailurePolicy,
>;

type AnthropicStore = DuckDbStore<AnthropicMessages>;
type AnthropicOutcomePolicy = RoutedAnthropicOutcomePolicy<RuntimeToolRouter>;
type AnthropicOutputInterpreter = PolicyAnthropicOutputInterpreter<AnthropicOutcomePolicy>;
type AnthropicAgentRuntimeCore = AgentRuntime<
    AnthropicMessages,
    AnthropicStore,
    AnthropicRequestMaterializer,
    CanonicalJsonFingerprinter,
    DynamoAnthropicInvoker,
    AnthropicOutputInterpreter,
    AnthropicUuidGenerator,
    SystemClock,
>;
type AnthropicToolAdapter = AnthropicToolLoopAdapter<RuntimeToolRouter>;
type AnthropicToolRunner = ToolRunner<
    AnthropicStore,
    RuntimeToolExecutor,
    Blake3ToolIdempotencyKeys,
    RuntimeToolFailurePolicy,
>;

struct RuntimeToolExecutor {
    web_search: BraveWebSearchExecutor,
    sandbox: Option<SandboxProviderExecutor<HttpSandboxProvider>>,
}

#[derive(Debug, Error)]
enum RuntimeToolExecutorError {
    #[error("web search execution failed: {0}")]
    WebSearch(#[source] BraveWebSearchError),
    #[error("sandbox execution failed: {0}")]
    Sandbox(#[source] SandboxToolError<HttpSandboxProviderError>),
    #[error("tool connector is not configured: {0}")]
    UnsupportedConnector(String),
}

impl ToolExecutor for RuntimeToolExecutor {
    type Error = RuntimeToolExecutorError;

    fn execute(
        &self,
        request: ToolExecutionRequest,
    ) -> BoxFuture<'_, Result<ToolExecutionResult, Self::Error>> {
        Box::pin(async move {
            match request.connector.as_str() {
                "web_search" => self
                    .web_search
                    .execute(request)
                    .await
                    .map_err(RuntimeToolExecutorError::WebSearch),
                "sandbox" => self
                    .sandbox
                    .as_ref()
                    .ok_or_else(|| {
                        RuntimeToolExecutorError::UnsupportedConnector("sandbox".to_owned())
                    })?
                    .execute(request)
                    .await
                    .map_err(RuntimeToolExecutorError::Sandbox),
                connector => Err(RuntimeToolExecutorError::UnsupportedConnector(
                    connector.to_owned(),
                )),
            }
        })
    }

    fn lookup<'a>(
        &'a self,
        request: &'a ToolExecutionRequest,
    ) -> BoxFuture<'a, Result<Option<ToolExecutionResult>, Self::Error>> {
        Box::pin(async move {
            match request.connector.as_str() {
                "web_search" => self
                    .web_search
                    .lookup(request)
                    .await
                    .map_err(RuntimeToolExecutorError::WebSearch),
                "sandbox" => self
                    .sandbox
                    .as_ref()
                    .ok_or_else(|| {
                        RuntimeToolExecutorError::UnsupportedConnector("sandbox".to_owned())
                    })?
                    .lookup(request)
                    .await
                    .map_err(RuntimeToolExecutorError::Sandbox),
                connector => Err(RuntimeToolExecutorError::UnsupportedConnector(
                    connector.to_owned(),
                )),
            }
        })
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct RuntimeToolFailurePolicy;

impl ToolFailurePolicy<RuntimeToolExecutorError> for RuntimeToolFailurePolicy {
    fn classify(&self, error: &RuntimeToolExecutorError) -> ToolFailureDisposition {
        match error {
            RuntimeToolExecutorError::WebSearch(error) => {
                BraveWebSearchFailurePolicy.classify(error)
            }
            RuntimeToolExecutorError::Sandbox(error) => SandboxFailurePolicy.classify(error),
            RuntimeToolExecutorError::UnsupportedConnector(_) => {
                ToolFailureDisposition::Failed(ToolExecutionFailure {
                    code: "unsupported_connector".to_owned(),
                    message: error.to_string(),
                    retryable: false,
                })
            }
        }
    }
}

pub(super) struct ResponsesAgentRuntime {
    runtime: Arc<ResponsesAgentRuntimeCore>,
    router: RuntimeToolRouter,
    tool_adapter: Arc<ResponsesToolAdapter>,
    tool_runner: Arc<ResponsesToolRunner>,
}

pub(super) struct AnthropicAgentRuntime {
    runtime: Arc<AnthropicAgentRuntimeCore>,
    router: RuntimeToolRouter,
    tool_adapter: Arc<AnthropicToolAdapter>,
    tool_runner: Arc<AnthropicToolRunner>,
    requires_durable_state: bool,
}

impl AnthropicAgentRuntime {
    pub(super) fn requires_runtime(&self, request: &AnthropicCreateMessageRequest) -> bool {
        anthropic_request_requires_runtime(request, &self.router, self.requires_durable_state)
    }

    fn request_uses_runtime_tools(&self, request: &AnthropicCreateMessageRequest) -> bool {
        request.tools.as_ref().is_some_and(|tools| {
            tools
                .iter()
                .any(|tool| self.router.route(&tool.name).is_some())
        })
    }
}

fn anthropic_request_requires_runtime(
    request: &AnthropicCreateMessageRequest,
    router: &impl ToolRouter,
    requires_durable_state: bool,
) -> bool {
    AnthropicRequestSelector.requires_runtime(
        request,
        RuntimeSelectionContext {
            has_runtime_owned_tools: request
                .tools
                .as_ref()
                .is_some_and(|tools| tools.iter().any(|tool| router.route(&tool.name).is_some())),
            requires_durable_state,
        },
    )
}

impl ResponsesAgentRuntime {
    fn request_uses_runtime_tools(&self, request: &CreateResponse) -> bool {
        request_uses_runtime_tools(request, &self.router)
    }
}

fn request_uses_runtime_tools(request: &CreateResponse, router: &impl ToolRouter) -> bool {
    request.tools.as_ref().is_some_and(|tools| {
        tools.iter().any(|tool| {
            matches!(tool, Tool::Function(function) if router.route(&function.name).is_some())
        })
    })
}

type ResponsesRuntimeError = AgentRuntimeError<
    DuckDbStoreError,
    MaterializationError<Infallible>,
    serde_json::Error,
    DynamoResponsesInvocationError,
    RoutedResponsesOutcomeError,
>;

type ResponsesStreamRuntimeError =
    AgentStreamRuntimeError<ResponsesRuntimeError, Infallible, DuckDbStoreError>;

type RuntimeToolRunError = ToolRunError<DuckDbStoreError, RuntimeToolExecutorError>;
type ResponsesToolRuntimeError = AgentToolRuntimeError<
    ResponsesRuntimeError,
    ResponsesToolAdapterError,
    RuntimeToolRunError,
    DuckDbStoreError,
>;
type ResponsesToolStreamRuntimeError = AgentToolRuntimeError<
    ResponsesStreamRuntimeError,
    ResponsesToolAdapterError,
    RuntimeToolRunError,
    DuckDbStoreError,
>;

type AnthropicRuntimeError = AgentRuntimeError<
    DuckDbStoreError,
    AnthropicMaterializationError,
    serde_json::Error,
    DynamoAnthropicInvocationError,
    RoutedAnthropicOutcomeError,
>;
type AnthropicStreamRuntimeError =
    AgentStreamRuntimeError<AnthropicRuntimeError, AnthropicStreamEventError, DuckDbStoreError>;
type AnthropicToolRuntimeError =
    AgentToolRuntimeError<AnthropicRuntimeError, Infallible, RuntimeToolRunError, DuckDbStoreError>;
type AnthropicToolStreamRuntimeError = AgentToolRuntimeError<
    AnthropicStreamRuntimeError,
    Infallible,
    RuntimeToolRunError,
    DuckDbStoreError,
>;

#[derive(Debug, Error)]
pub(super) enum AgentRuntimeInitError {
    #[error("failed to initialize agent runtime DuckDB store: {0}")]
    Store(#[from] DuckDbStoreError),
    #[error("invalid agent runtime deployment configuration: {0}")]
    Configuration(String),
}

pub(super) fn new_responses_runtime() -> Result<Arc<ResponsesAgentRuntime>, AgentRuntimeInitError> {
    let store = match std::env::var_os(DUCKDB_PATH_ENV) {
        Some(path) if !path.is_empty() => ResponsesStore::open(path),
        _ => {
            if enabled() {
                tracing::warn!(
                    env = DUCKDB_PATH_ENV,
                    "agent runtime is using an in-memory DuckDB store; set the path for restart durability"
                );
            }
            ResponsesStore::open_in_memory()
        }
    }?;
    let (web_search_router, web_search) = web_search_components()?;
    let (sandbox_router, sandbox) = sandbox_components()?;
    let router = RuntimeToolRouter {
        web_search: web_search_router,
        sandbox: sandbox_router,
    };
    let executor = RuntimeToolExecutor {
        web_search,
        sandbox,
    };
    let output_interpreter =
        PolicyResponsesOutputInterpreter::new(RoutedResponsesOutcomePolicy::new(router.clone()));
    let runtime = Arc::new(AgentRuntime::new(
        store.clone(),
        ResponsesRequestMaterializer::default(),
        CanonicalJsonFingerprinter,
        DynamoResponsesInvoker,
        output_interpreter,
        UuidGenerator,
        SystemClock,
    ));
    Ok(Arc::new(ResponsesAgentRuntime {
        runtime,
        router: router.clone(),
        tool_adapter: Arc::new(ResponsesToolLoopAdapter::new(router)),
        tool_runner: Arc::new(ToolRunner::new(
            store,
            executor,
            Blake3ToolIdempotencyKeys,
            RuntimeToolFailurePolicy,
        )),
    }))
}

pub(super) fn new_anthropic_runtime() -> Result<Arc<AnthropicAgentRuntime>, AgentRuntimeInitError> {
    let store = match std::env::var_os(DUCKDB_PATH_ENV) {
        Some(path) if !path.is_empty() => AnthropicStore::open(path),
        _ => AnthropicStore::open_in_memory(),
    }?;
    let (web_search_router, web_search) = web_search_components()?;
    let (sandbox_router, sandbox) = sandbox_components()?;
    let router = RuntimeToolRouter {
        web_search: web_search_router,
        sandbox: sandbox_router,
    };
    let executor = RuntimeToolExecutor {
        web_search,
        sandbox,
    };
    let output_interpreter =
        PolicyAnthropicOutputInterpreter::new(RoutedAnthropicOutcomePolicy::new(router.clone()));
    let runtime = Arc::new(AgentRuntime::new(
        store.clone(),
        AnthropicRequestMaterializer,
        CanonicalJsonFingerprinter,
        DynamoAnthropicInvoker,
        output_interpreter,
        AnthropicUuidGenerator,
        SystemClock,
    ));
    let requires_durable_state = optional_environment(STATEFUL_ANTHROPIC_ENV)?
        .map(|value| {
            value.parse::<bool>().map_err(|_| {
                AgentRuntimeInitError::Configuration(format!(
                    "{STATEFUL_ANTHROPIC_ENV} must be true or false"
                ))
            })
        })
        .transpose()?
        .unwrap_or(false);
    Ok(Arc::new(AnthropicAgentRuntime {
        runtime,
        router: router.clone(),
        tool_adapter: Arc::new(AnthropicToolLoopAdapter::new(router)),
        tool_runner: Arc::new(ToolRunner::new(
            store,
            executor,
            Blake3ToolIdempotencyKeys,
            RuntimeToolFailurePolicy,
        )),
        requires_durable_state,
    }))
}

fn optional_environment(name: &'static str) -> Result<Option<String>, AgentRuntimeInitError> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(AgentRuntimeInitError::Configuration(
            format!("{name} must contain UTF-8"),
        )),
    }
}

fn web_search_components()
-> Result<(ConfiguredToolRouter, BraveWebSearchExecutor), AgentRuntimeInitError> {
    let api_key = optional_environment(WEB_SEARCH_API_KEY_ENV)?;
    let tool_name = optional_environment(WEB_SEARCH_TOOL_NAME_ENV)?;
    web_search_components_from_config(api_key, tool_name)
}

fn web_search_components_from_config(
    api_key: Option<String>,
    tool_name: Option<String>,
) -> Result<(ConfiguredToolRouter, BraveWebSearchExecutor), AgentRuntimeInitError> {
    let Some(api_key) = api_key else {
        return Ok((
            ConfiguredToolRouter::default(),
            BraveWebSearchExecutor::new([])
                .map_err(|error| AgentRuntimeInitError::Configuration(error.to_string()))?,
        ));
    };
    let tool_name = tool_name.unwrap_or_else(|| "web_search".to_owned());
    validate_scope_component(&tool_name).map_err(|error| {
        AgentRuntimeInitError::Configuration(format!("invalid {WEB_SEARCH_TOOL_NAME_ENV}: {error}"))
    })?;
    let profile_name = "brave_default".to_owned();
    let profile = BraveWebSearchProfile::new(api_key).map_err(|error| {
        AgentRuntimeInitError::Configuration(format!(
            "invalid web-search deployment configuration: {error}"
        ))
    })?;
    let router = ConfiguredToolRouter::new([(
        tool_name,
        ToolRoute::new("web_search", "search").with_profile(profile_name.clone()),
    )]);
    let executor = BraveWebSearchExecutor::new([(profile_name, profile)]).map_err(|error| {
        AgentRuntimeInitError::Configuration(format!(
            "failed to initialize web-search executor: {error}"
        ))
    })?;
    Ok((router, executor))
}

fn sandbox_components() -> Result<
    (
        ConfiguredToolRouter,
        Option<SandboxProviderExecutor<HttpSandboxProvider>>,
    ),
    AgentRuntimeInitError,
> {
    let endpoint = optional_environment(SANDBOX_ENDPOINT_ENV)?;
    let token = optional_environment(SANDBOX_TOKEN_ENV)?;
    let allow_http = optional_environment(SANDBOX_ALLOW_HTTP_ENV)?
        .map(|value| {
            value.parse::<bool>().map_err(|_| {
                AgentRuntimeInitError::Configuration(format!(
                    "{SANDBOX_ALLOW_HTTP_ENV} must be true or false"
                ))
            })
        })
        .transpose()?
        .unwrap_or(false);
    let tool_name = optional_environment(SANDBOX_TOOL_NAME_ENV)?;
    let profile = optional_environment(SANDBOX_PROFILE_ENV)?;
    sandbox_components_from_config(endpoint, token, allow_http, tool_name, profile)
}

fn sandbox_components_from_config(
    endpoint: Option<String>,
    token: Option<String>,
    allow_http: bool,
    tool_name: Option<String>,
    profile: Option<String>,
) -> Result<
    (
        ConfiguredToolRouter,
        Option<SandboxProviderExecutor<HttpSandboxProvider>>,
    ),
    AgentRuntimeInitError,
> {
    let Some(endpoint) = endpoint else {
        return Ok((ConfiguredToolRouter::default(), None));
    };
    let token = token.ok_or_else(|| {
        AgentRuntimeInitError::Configuration(format!(
            "{SANDBOX_TOKEN_ENV} is required when sandbox execution is enabled"
        ))
    })?;
    let tool_name = tool_name.unwrap_or_else(|| "python".to_owned());
    let profile = profile.unwrap_or_else(|| "python-deny-egress".to_owned());
    validate_scope_component(&tool_name).map_err(|error| {
        AgentRuntimeInitError::Configuration(format!("invalid {SANDBOX_TOOL_NAME_ENV}: {error}"))
    })?;
    validate_scope_component(&profile).map_err(|error| {
        AgentRuntimeInitError::Configuration(format!("invalid {SANDBOX_PROFILE_ENV}: {error}"))
    })?;
    let provider = HttpSandboxProvider::new(HttpSandboxProviderConfig {
        endpoint,
        bearer_token: token,
        allow_http,
        ..HttpSandboxProviderConfig::default()
    })
    .map_err(|error| AgentRuntimeInitError::Configuration(error.to_string()))?;
    let router = ConfiguredToolRouter::new([(
        tool_name,
        ToolRoute::new("sandbox", "python").with_profile(profile),
    )]);
    let executor = SandboxProviderExecutor::new(provider, SandboxToolExecutorConfig::default())
        .map_err(|error| AgentRuntimeInitError::Configuration(error.to_string()))?;
    Ok((router, Some(executor)))
}

/// Dynamo-owned, explicitly filtered ingress data forwarded across model steps.
///
/// Authorization credentials, caller-supplied identity, routing headers, and
/// lifecycle-finalization hints are intentionally absent. The carrier is
/// ephemeral and is never written to agent runtime checkpoints.
#[derive(Clone)]
struct DynamoInvocationCarrier {
    metadata: BTreeMap<String, String>,
    agent_context: Option<crate::protocols::agents::AgentContextHeaderValues>,
    session_affinity: Option<SessionAffinityId>,
    trace_request_id: Option<String>,
}

impl DynamoInvocationCarrier {
    fn from_headers(headers: &HeaderMap) -> Result<Self, metadata::MetadataHeaderError> {
        let metadata = metadata::extract_metadata_from_http(headers)?;
        let agent_context = agent_context_header_values(headers).map(|mut values| {
            // Agent-runtime turns do not own Dynamo session lifecycle. In
            // particular, an external session-final hint must not evict KV
            // state on every internal model step.
            values.session_final = None;
            values
        });
        let session_affinity = session_affinity_header_value(headers).map(SessionAffinityId::new);
        let trace_request_id = crate::request_trace::is_enabled()
            .then(|| header_value(headers, "x-request-id"))
            .flatten();
        Ok(Self {
            metadata,
            agent_context,
            session_affinity,
            trace_request_id,
        })
    }

    fn context(&self, request: NvCreateResponse, request_id: String) -> Context<NvCreateResponse> {
        let input_trigger = classify_response_request(&request);
        let mut request = Context::with_id_and_metadata(request, request_id, self.metadata.clone());
        if let Some(trace_request_id) = &self.trace_request_id {
            request.insert(
                crate::request_trace::X_REQUEST_ID_CONTEXT_KEY,
                trace_request_id.clone(),
            );
        }
        if let Some(values) = &self.agent_context {
            let mut agent_context = AgentContext::from(values.clone());
            agent_context.input_trigger = Some(input_trigger);
            request.insert(AGENT_CONTEXT_CONTEXT_KEY, agent_context);
        }
        if let Some(session_affinity) = &self.session_affinity {
            request.insert(SESSION_AFFINITY_CONTEXT_KEY, session_affinity.clone());
        }
        request
    }

    fn anthropic_context(
        &self,
        request: AnthropicCreateMessageRequest,
        request_id: String,
    ) -> Context<AnthropicCreateMessageRequest> {
        let input_trigger = classify_anthropic_request(&request);
        let mut request = Context::with_id_and_metadata(request, request_id, self.metadata.clone());
        if let Some(trace_request_id) = &self.trace_request_id {
            request.insert(
                crate::request_trace::X_REQUEST_ID_CONTEXT_KEY,
                trace_request_id.clone(),
            );
        }
        if let Some(values) = &self.agent_context {
            let mut agent_context = AgentContext::from(values.clone());
            agent_context.input_trigger = Some(input_trigger);
            request.insert(AGENT_CONTEXT_CONTEXT_KEY, agent_context);
        }
        if let Some(session_affinity) = &self.session_affinity {
            request.insert(SESSION_AFFINITY_CONTEXT_KEY, session_affinity.clone());
        }
        request
    }
}

/// Ephemeral Dynamo ingress state forwarded across model steps.
#[derive(Clone)]
pub(super) struct DynamoResponsesContext {
    state: Arc<service_v2::State>,
    template: Option<RequestTemplate>,
    request_id: String,
    carrier: DynamoInvocationCarrier,
    nvext: Option<NvExt>,
}

#[derive(Clone)]
pub(super) struct DynamoAnthropicContext {
    state: Arc<service_v2::State>,
    template: Option<RequestTemplate>,
    routing_headers: HeaderMap,
    request_id: String,
    carrier: DynamoInvocationCarrier,
}

fn filtered_routing_headers(headers: &HeaderMap) -> HeaderMap {
    const ROUTING_HEADERS: [&str; 11] = [
        HEADER_WORKER_INSTANCE_ID,
        HEADER_WORKER_INSTANCE_ID_ALIAS,
        HEADER_PREFILL_INSTANCE_ID,
        HEADER_PREFILL_INSTANCE_ID_ALIAS,
        HEADER_DP_RANK,
        HEADER_DP_RANK_ALIAS,
        HEADER_DATA_PARALLEL_RANK_ALIAS,
        HEADER_PREFILL_DP_RANK,
        HEADER_PREFILL_DP_RANK_ALIAS,
        HEADER_REQUEST_PRIORITY,
        HEADER_REQUEST_STRICT_PRIORITY,
    ];
    let mut filtered = HeaderMap::new();
    for name in ROUTING_HEADERS {
        if let Some(value) = headers.get(name) {
            filtered.insert(name, value.clone());
        }
    }
    if let Some(value) = headers.get(HEADER_TENANT_ID) {
        filtered.insert(HEADER_TENANT_ID, value.clone());
    }
    filtered
}

impl DynamoResponsesContext {
    fn new(
        state: Arc<service_v2::State>,
        template: Option<RequestTemplate>,
        request_id: String,
        carrier: DynamoInvocationCarrier,
        nvext: Option<NvExt>,
    ) -> Self {
        Self {
            state,
            template,
            request_id,
            carrier,
            nvext,
        }
    }
}

pub(super) async fn handle_responses(
    state: Arc<service_v2::State>,
    template: Option<RequestTemplate>,
    headers: HeaderMap,
    request: NvCreateResponse,
) -> Result<Response, openai::ErrorResponse> {
    let streaming = request.inner.stream.unwrap_or(false);
    let store = request.inner.store.unwrap_or(false);
    let response_params = ResponseParams::from_create_response(&request.inner);
    let parent_response_id = request
        .inner
        .previous_response_id
        .as_deref()
        .map(ResponseId::from);
    let request_id = openai::get_or_create_request_id(&headers);
    let carrier = DynamoInvocationCarrier::from_headers(&headers)
        .map_err(|error| openai::ErrorMessage::request_headers_too_large(&error.to_string()))?;
    let idempotency_key = header_value(&headers, "idempotency-key")
        .or_else(|| header_value(&headers, "x-idempotency-key"))
        .unwrap_or_else(|| request_id.clone());
    let authorization = ingress_authorization(&headers)?;
    let agent_runtime = state.responses_agent_runtime();
    let stage_runtime_tool_rounds =
        streaming && agent_runtime.request_uses_runtime_tools(&request.inner);
    let invocation_context =
        DynamoResponsesContext::new(state.clone(), template, request_id, carrier, request.nvext);
    let step_kind = if parent_response_id.is_some() {
        ModelStepKind::ClientToolContinuation
    } else {
        ModelStepKind::Initial
    };
    let command = RunTurn {
        request: request.inner,
        parent_response_id,
        authorization,
        idempotency_key: IdempotencyKey::from(idempotency_key),
        invocation_context,
        inference_intent: InferenceIntent { step_kind },
        lease_duration_millis: 120_000,
    };

    if streaming {
        let stream_interpreter = if stage_runtime_tool_rounds {
            ResponsesStreamEventInterpreter::stage_runtime_tool_rounds()
        } else {
            ResponsesStreamEventInterpreter::default()
        };
        let result = agent_runtime
            .runtime
            .clone()
            .run_stream_with_tools(
                command,
                stream_interpreter,
                agent_runtime.tool_adapter.clone(),
                agent_runtime.tool_runner.clone(),
            )
            .await
            .map_err(tool_stream_runtime_error_response)?;
        let stream: AgentResponsesStream = match result {
            RunStreamResult::Live(stream) => {
                Box::pin(stream.map(|event| event.map_err(axum::Error::new)))
            }
            RunStreamResult::Existing(record) => {
                let response = existing_response(record)?;
                Box::pin(stream::iter(
                    committed_response_events(response).into_iter().map(Ok),
                ))
            }
        };
        return Ok(sse_response(
            stream,
            ResponseEventSerializer::new(&response_params),
            state.sse_keep_alive_for_response(stage_runtime_tool_rounds),
        ));
    }

    let result = agent_runtime
        .runtime
        .run_unary_with_tools(
            command,
            agent_runtime.tool_adapter.as_ref(),
            agent_runtime.tool_runner.as_ref(),
        )
        .await
        .map_err(tool_runtime_error_response)?;
    let response = match result {
        RunTurnResult::Committed { record, .. } => record.response.ok_or_else(|| {
            openai::ErrorMessage::internal_server_error(
                "Agent runtime committed a turn without a response",
            )
        })?,
        RunTurnResult::Existing(record) => existing_response(record)?,
    };
    let response = NvResponse {
        inner: response,
        nvext: None,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        store,
    };

    Ok(axum::Json(response).into_response())
}

pub(super) async fn handle_anthropic(
    state: Arc<service_v2::State>,
    template: Option<RequestTemplate>,
    headers: HeaderMap,
    request: AnthropicCreateMessageRequest,
) -> Result<Response, Response> {
    let streaming = request.stream;
    let request_id = openai::get_or_create_request_id(&headers);
    let carrier = DynamoInvocationCarrier::from_headers(&headers).map_err(|error| {
        anthropic::anthropic_error(
            StatusCode::REQUEST_HEADER_FIELDS_TOO_LARGE,
            "invalid_request_error",
            &error.to_string(),
        )
    })?;
    let idempotency_key = header_value(&headers, "idempotency-key")
        .or_else(|| header_value(&headers, "x-idempotency-key"))
        .unwrap_or_else(|| request_id.clone());
    let authorization = ingress_authorization(&headers).map_err(openai_error_to_anthropic)?;
    let agent_runtime = state.anthropic_agent_runtime();
    let stage_runtime_tool_rounds = streaming && agent_runtime.request_uses_runtime_tools(&request);
    let invocation_context = DynamoAnthropicContext {
        state: state.clone(),
        template,
        routing_headers: filtered_routing_headers(&headers),
        request_id,
        carrier,
    };
    let command = RunTurn {
        request,
        parent_response_id: None,
        authorization,
        idempotency_key: IdempotencyKey::from(idempotency_key),
        invocation_context,
        inference_intent: InferenceIntent {
            step_kind: ModelStepKind::Initial,
        },
        lease_duration_millis: 120_000,
    };

    if streaming {
        let stream_interpreter = if stage_runtime_tool_rounds {
            AnthropicStreamEventInterpreter::stage_runtime_tool_rounds()
        } else {
            AnthropicStreamEventInterpreter::default()
        };
        let result = agent_runtime
            .runtime
            .clone()
            .run_stream_with_tools(
                command,
                stream_interpreter,
                agent_runtime.tool_adapter.clone(),
                agent_runtime.tool_runner.clone(),
            )
            .await
            .map_err(anthropic_tool_stream_error_response)?;
        let stream: AgentAnthropicStream = match result {
            RunStreamResult::Live(stream) => {
                Box::pin(stream.map(|event| event.map_err(axum::Error::new)))
            }
            RunStreamResult::Existing(record) => {
                let response =
                    existing_anthropic_response(record).map_err(anthropic_replay_error_response)?;
                Box::pin(stream::iter(
                    committed_anthropic_events(response).into_iter().map(Ok),
                ))
            }
        };
        return Ok(anthropic_sse_response(
            stream,
            state.sse_keep_alive_for_response(stage_runtime_tool_rounds),
        ));
    }

    let result = agent_runtime
        .runtime
        .run_unary_with_tools(
            command,
            agent_runtime.tool_adapter.as_ref(),
            agent_runtime.tool_runner.as_ref(),
        )
        .await
        .map_err(anthropic_tool_error_response)?;
    let response = match result {
        RunTurnResult::Committed { record, .. } => record.response.ok_or_else(|| {
            anthropic_public_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Agent runtime committed a turn without a response",
                "missing committed Anthropic response",
            )
        })?,
        RunTurnResult::Existing(record) => {
            existing_anthropic_response(record).map_err(anthropic_replay_error_response)?
        }
    };
    Ok(axum::Json(response).into_response())
}

fn openai_error_to_anthropic(error: openai::ErrorResponse) -> Response {
    anthropic::anthropic_error(error.0, "api_error", error.1.message())
}

fn existing_anthropic_response(
    record: Box<dynamo_agent_rt::CheckpointRecord<AnthropicMessages>>,
) -> Result<AnthropicMessageResponse, (StatusCode, &'static str, &'static str)> {
    match record.state {
        TurnState::Completed | TurnState::AwaitingClientToolOutput => record.response.ok_or((
            StatusCode::INTERNAL_SERVER_ERROR,
            "Agent runtime replayable turn has no response",
            "missing replayable Anthropic response",
        )),
        TurnState::InFlight | TurnState::ToolStarted => Err((
            StatusCode::CONFLICT,
            "Agent runtime turn is still in progress",
            "active Anthropic turn replay",
        )),
        TurnState::OutcomeUnknown => Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "Agent runtime turn outcome is unknown",
            "Anthropic turn outcome unknown",
        )),
        TurnState::Failed => Err((
            StatusCode::CONFLICT,
            "Agent runtime turn previously failed",
            "failed Anthropic turn replay",
        )),
    }
}

fn anthropic_replay_error_response(
    (status, message, details): (StatusCode, &'static str, &'static str),
) -> Response {
    anthropic_public_error(status, message, details)
}

type AgentAnthropicStream =
    Pin<Box<dyn Stream<Item = Result<AnthropicStreamEvent, axum::Error>> + Send>>;

fn anthropic_sse_response(
    stream: AgentAnthropicStream,
    keep_alive: Option<std::time::Duration>,
) -> Response {
    let stream = stream.map(|event| {
        event.and_then(|event| {
            crate::protocols::anthropic::stream_converter::make_sse_event(&event)
                .map_err(axum::Error::new)
        })
    });
    let mut response = Sse::new(stream);
    if let Some(keep_alive) = keep_alive {
        response = response.keep_alive(KeepAlive::default().interval(keep_alive));
    }
    response.into_response()
}

fn committed_anthropic_events(response: AnthropicMessageResponse) -> Vec<AnthropicStreamEvent> {
    let mut initial = response.clone();
    initial.content.clear();
    initial.stop_reason = None;
    initial.stop_sequence = None;
    initial.usage.output_tokens = 0;
    let mut events = Vec::with_capacity(response.content.len().saturating_mul(3) + 3);
    events.push(AnthropicStreamEvent::MessageStart { message: initial });
    for (index, block) in response.content.iter().enumerate() {
        let index = index as u32;
        match block {
            AnthropicResponseContentBlock::Thinking {
                thinking,
                signature,
            } => {
                events.push(AnthropicStreamEvent::ContentBlockStart {
                    index,
                    content_block: AnthropicResponseContentBlock::Thinking {
                        thinking: String::new(),
                        signature: String::new(),
                    },
                });
                if !thinking.is_empty() {
                    events.push(AnthropicStreamEvent::ContentBlockDelta {
                        index,
                        delta: AnthropicDelta::ThinkingDelta {
                            thinking: thinking.clone(),
                        },
                        usage: Some(response.usage.clone()),
                    });
                }
                if !signature.is_empty() {
                    events.push(AnthropicStreamEvent::ContentBlockDelta {
                        index,
                        delta: AnthropicDelta::SignatureDelta {
                            signature: signature.clone(),
                        },
                        usage: Some(response.usage.clone()),
                    });
                }
            }
            AnthropicResponseContentBlock::Text { text, citations } => {
                events.push(AnthropicStreamEvent::ContentBlockStart {
                    index,
                    content_block: AnthropicResponseContentBlock::Text {
                        text: String::new(),
                        citations: None,
                    },
                });
                if !text.is_empty() {
                    events.push(AnthropicStreamEvent::ContentBlockDelta {
                        index,
                        delta: AnthropicDelta::TextDelta { text: text.clone() },
                        usage: Some(response.usage.clone()),
                    });
                }
                for citation in citations.iter().flatten() {
                    events.push(AnthropicStreamEvent::ContentBlockDelta {
                        index,
                        delta: AnthropicDelta::CitationsDelta {
                            citation: citation.clone(),
                        },
                        usage: Some(response.usage.clone()),
                    });
                }
            }
            AnthropicResponseContentBlock::ToolUse { id, name, input } => {
                events.push(AnthropicStreamEvent::ContentBlockStart {
                    index,
                    content_block: AnthropicResponseContentBlock::ToolUse {
                        id: id.clone(),
                        name: name.clone(),
                        input: serde_json::json!({}),
                    },
                });
                events.push(AnthropicStreamEvent::ContentBlockDelta {
                    index,
                    delta: AnthropicDelta::InputJsonDelta {
                        partial_json: input.to_string(),
                    },
                    usage: Some(response.usage.clone()),
                });
            }
            block => events.push(AnthropicStreamEvent::ContentBlockStart {
                index,
                content_block: block.clone(),
            }),
        }
        events.push(AnthropicStreamEvent::ContentBlockStop { index });
    }
    events.push(AnthropicStreamEvent::MessageDelta {
        delta: AnthropicMessageDeltaBody {
            stop_reason: response.stop_reason.clone(),
            stop_sequence: response.stop_sequence.clone(),
        },
        usage: response.usage,
    });
    events.push(AnthropicStreamEvent::MessageStop {});
    events
}

fn anthropic_tool_stream_error_response(error: AnthropicToolStreamRuntimeError) -> Response {
    match error {
        AgentToolRuntimeError::Runtime(error) => anthropic_stream_runtime_error_response(error),
        AgentToolRuntimeError::Adapter {
            error,
            checkpoint_error: _,
        } => match error {},
        AgentToolRuntimeError::ToolBatch {
            errors,
            checkpoint_error,
            ..
        } => anthropic_public_error(
            checkpoint_failure_status(
                &checkpoint_error,
                errors
                    .iter()
                    .map(tool_run_error_status)
                    .max_by_key(|status| status_severity(*status))
                    .unwrap_or(StatusCode::BAD_GATEWAY),
            ),
            "Server-side tool execution failed",
            "Anthropic server-side tool batch failed",
        ),
        AgentToolRuntimeError::MissingCalls { checkpoint_error } => anthropic_public_error(
            checkpoint_failure_status(&checkpoint_error, StatusCode::INTERNAL_SERVER_ERROR),
            "Agent runtime could not resolve the server-tool call",
            "Anthropic runtime tool calls missing",
        ),
        AgentToolRuntimeError::ToolRoundLimit {
            checkpoint_error, ..
        }
        | AgentToolRuntimeError::ParallelToolLimit {
            checkpoint_error, ..
        } => anthropic_public_error(
            checkpoint_failure_status(&checkpoint_error, StatusCode::BAD_GATEWAY),
            "Server-side tool execution exceeded deployment limits",
            "Anthropic runtime tool limit exceeded",
        ),
        AgentToolRuntimeError::MissingLease | AgentToolRuntimeError::MissingResponse => {
            anthropic_public_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Agent runtime lost server-tool turn state",
                "Anthropic runtime tool state missing",
            )
        }
    }
}

fn anthropic_tool_error_response(error: AnthropicToolRuntimeError) -> Response {
    match error {
        AgentToolRuntimeError::Runtime(error) => anthropic_runtime_error_response(error),
        AgentToolRuntimeError::Adapter {
            error,
            checkpoint_error: _,
        } => match error {},
        AgentToolRuntimeError::ToolBatch {
            errors,
            checkpoint_error,
            ..
        } => anthropic_public_error(
            checkpoint_failure_status(
                &checkpoint_error,
                errors
                    .iter()
                    .map(tool_run_error_status)
                    .max_by_key(|status| status_severity(*status))
                    .unwrap_or(StatusCode::BAD_GATEWAY),
            ),
            "Server-side tool execution failed",
            "Anthropic server-side tool batch failed",
        ),
        AgentToolRuntimeError::MissingCalls { checkpoint_error } => anthropic_public_error(
            checkpoint_failure_status(&checkpoint_error, StatusCode::INTERNAL_SERVER_ERROR),
            "Agent runtime could not resolve the server-tool call",
            "Anthropic runtime tool calls missing",
        ),
        AgentToolRuntimeError::ToolRoundLimit {
            checkpoint_error, ..
        }
        | AgentToolRuntimeError::ParallelToolLimit {
            checkpoint_error, ..
        } => anthropic_public_error(
            checkpoint_failure_status(&checkpoint_error, StatusCode::BAD_GATEWAY),
            "Server-side tool execution exceeded deployment limits",
            "Anthropic runtime tool limit exceeded",
        ),
        AgentToolRuntimeError::MissingLease | AgentToolRuntimeError::MissingResponse => {
            anthropic_public_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Agent runtime lost server-tool turn state",
                "Anthropic runtime tool state missing",
            )
        }
    }
}

fn anthropic_stream_runtime_error_response(error: AnthropicStreamRuntimeError) -> Response {
    match error {
        AgentStreamRuntimeError::Runtime(error) => anthropic_runtime_error_response(error),
        AgentStreamRuntimeError::ExpectedStreaming { checkpoint_error } => anthropic_public_error(
            checkpoint_failure_status(&checkpoint_error, StatusCode::BAD_GATEWAY),
            "Inference backend did not return a response stream",
            "Anthropic inference returned unary output for a streaming request",
        ),
        AgentStreamRuntimeError::Interpreter {
            error,
            checkpoint_error,
        } => anthropic_public_error(
            checkpoint_failure_status(&checkpoint_error, StatusCode::BAD_GATEWAY),
            "Inference backend returned an invalid Anthropic stream",
            error,
        ),
        AgentStreamRuntimeError::StagedEventEncoding {
            error,
            checkpoint_error,
        } => anthropic_public_error(
            checkpoint_failure_status(&checkpoint_error, StatusCode::BAD_GATEWAY),
            "Inference backend returned an invalid stream event",
            error,
        ),
        AgentStreamRuntimeError::StagedEventLimit {
            limit_bytes,
            checkpoint_error,
        } => anthropic_public_error(
            checkpoint_failure_status(&checkpoint_error, StatusCode::BAD_GATEWAY),
            "Inference stream exceeded the runtime staging limit",
            format_args!("staged Anthropic stream exceeded {limit_bytes} bytes"),
        ),
        AgentStreamRuntimeError::MissingTerminal { checkpoint_error } => anthropic_public_error(
            checkpoint_failure_status(&checkpoint_error, StatusCode::BAD_GATEWAY),
            "Inference stream ended without a terminal response",
            "Anthropic stream ended before message_stop",
        ),
    }
}

fn anthropic_runtime_error_response(error: AnthropicRuntimeError) -> Response {
    match error {
        AgentRuntimeError::Store(error) => {
            let response = store_error_response(error);
            openai_error_to_anthropic(response)
        }
        AgentRuntimeError::Materialize(error) => anthropic_public_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Agent runtime could not materialize the Anthropic turn",
            error,
        ),
        AgentRuntimeError::Fingerprint(error) => anthropic_public_error(
            StatusCode::BAD_REQUEST,
            "Request could not be fingerprinted",
            error,
        ),
        AgentRuntimeError::LeaseDeadlineOverflow => anthropic_public_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Agent runtime could not create the turn",
            "lease deadline overflow",
        ),
        AgentRuntimeError::Inference {
            error,
            checkpoint_error,
        } => {
            if checkpoint_error.is_some() {
                return anthropic_public_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Agent runtime could not durably record the failed turn",
                    format_args!(
                        "Anthropic inference failed: {error}; failed-state commit: {checkpoint_error:?}"
                    ),
                );
            }
            anthropic_invocation_error_response(error)
        }
        AgentRuntimeError::StreamingUnsupported { checkpoint_error } => anthropic_public_error(
            checkpoint_failure_status(&checkpoint_error, StatusCode::BAD_GATEWAY),
            "Inference backend returned an incompatible response",
            "Anthropic unary runtime received streaming output",
        ),
        AgentRuntimeError::Output {
            error,
            checkpoint_error,
        } => anthropic_public_error(
            checkpoint_failure_status(&checkpoint_error, StatusCode::BAD_GATEWAY),
            "Inference backend returned an invalid response",
            error,
        ),
        AgentRuntimeError::InvalidOutputState {
            state,
            checkpoint_error,
        } => anthropic_public_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Agent runtime produced an invalid turn transition",
            format_args!(
                "invalid Anthropic output state {state:?}; failed-state commit: {checkpoint_error:?}"
            ),
        ),
    }
}

fn anthropic_invocation_error_response(error: DynamoAnthropicInvocationError) -> Response {
    match error {
        DynamoAnthropicInvocationError::Dynamo { status, message } => {
            anthropic_public_error(status, &message, "Dynamo Anthropic invocation failed")
        }
        DynamoAnthropicInvocationError::Body(error) => anthropic_public_error(
            StatusCode::BAD_GATEWAY,
            "Failed to read the inference backend response",
            error,
        ),
        DynamoAnthropicInvocationError::Decode(error) => anthropic_public_error(
            StatusCode::BAD_GATEWAY,
            "Inference backend returned an invalid response",
            error,
        ),
    }
}

fn anthropic_public_error(
    status: StatusCode,
    public_message: &str,
    details: impl std::fmt::Display,
) -> Response {
    if status.is_server_error() {
        tracing::error!(%status, %details, "Anthropic agent runtime request failed");
    } else {
        tracing::debug!(%status, %details, "Anthropic agent runtime request rejected");
    }
    anthropic::anthropic_error(status, "api_error", public_message)
}

fn existing_response(
    record: Box<dynamo_agent_rt::CheckpointRecord<OpenAiResponses>>,
) -> Result<dynamo_protocols::types::responses::Response, openai::ErrorResponse> {
    match record.state {
        TurnState::Completed | TurnState::AwaitingClientToolOutput => {
            record.response.ok_or_else(|| {
                openai::ErrorMessage::internal_server_error(
                    "Agent runtime replayable turn has no response",
                )
            })
        }
        TurnState::InFlight | TurnState::ToolStarted => {
            Err(openai::ErrorMessage::agent_runtime_error(
                StatusCode::CONFLICT,
                "Agent runtime turn is still in progress",
            ))
        }
        TurnState::OutcomeUnknown => Err(openai::ErrorMessage::agent_runtime_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "Agent runtime turn outcome is unknown",
        )),
        TurnState::Failed => Err(openai::ErrorMessage::agent_runtime_error(
            StatusCode::CONFLICT,
            "Agent runtime turn previously failed",
        )),
    }
}

fn header_value(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
}

fn tool_stream_runtime_error_response(
    error: ResponsesToolStreamRuntimeError,
) -> openai::ErrorResponse {
    tool_error_response(error, stream_runtime_error_response)
}

fn tool_runtime_error_response(error: ResponsesToolRuntimeError) -> openai::ErrorResponse {
    tool_error_response(error, runtime_error_response)
}

fn tool_error_response<R>(
    error: AgentToolRuntimeError<
        R,
        ResponsesToolAdapterError,
        RuntimeToolRunError,
        DuckDbStoreError,
    >,
    runtime_error: impl FnOnce(R) -> openai::ErrorResponse,
) -> openai::ErrorResponse
where
    R: std::error::Error + Send + Sync + 'static,
{
    let details = error.to_string();
    let (status, message) = match error {
        AgentToolRuntimeError::Runtime(error) => return runtime_error(error),
        AgentToolRuntimeError::Adapter {
            checkpoint_error, ..
        } => (
            checkpoint_failure_status(&checkpoint_error, StatusCode::BAD_GATEWAY),
            "Inference backend returned an invalid server-tool call",
        ),
        AgentToolRuntimeError::ToolBatch {
            errors,
            checkpoint_error,
            ..
        } => (
            checkpoint_failure_status(
                &checkpoint_error,
                errors
                    .iter()
                    .map(tool_run_error_status)
                    .max_by_key(|status| status_severity(*status))
                    .unwrap_or(StatusCode::BAD_GATEWAY),
            ),
            "Server-side tool execution failed",
        ),
        AgentToolRuntimeError::MissingCalls { checkpoint_error } => (
            checkpoint_failure_status(&checkpoint_error, StatusCode::INTERNAL_SERVER_ERROR),
            "Agent runtime could not resolve the server-tool call",
        ),
        AgentToolRuntimeError::ToolRoundLimit {
            checkpoint_error, ..
        }
        | AgentToolRuntimeError::ParallelToolLimit {
            checkpoint_error, ..
        } => (
            checkpoint_failure_status(&checkpoint_error, StatusCode::BAD_GATEWAY),
            "Server-side tool execution exceeded deployment limits",
        ),
        AgentToolRuntimeError::MissingLease | AgentToolRuntimeError::MissingResponse => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Agent runtime lost server-tool turn state",
        ),
    };
    agent_runtime_public_error(status, message, details)
}

fn checkpoint_failure_status(
    checkpoint_error: &Option<DuckDbStoreError>,
    otherwise: StatusCode,
) -> StatusCode {
    if checkpoint_error.is_some() {
        StatusCode::SERVICE_UNAVAILABLE
    } else {
        otherwise
    }
}

fn tool_run_error_status(error: &RuntimeToolRunError) -> StatusCode {
    match error {
        ToolRunError::UnauthorizedConnector(_) => StatusCode::FORBIDDEN,
        ToolRunError::Journal(_)
        | ToolRunError::RecoveryLookup { .. }
        | ToolRunError::ExecutionTimedOut { .. }
        | ToolRunError::RecoveryTimedOut { .. }
        | ToolRunError::OutcomeUnknown { .. }
        | ToolRunError::CorruptJournal
        | ToolRunError::JournalAfterExecution(_) => StatusCode::SERVICE_UNAVAILABLE,
        ToolRunError::Executor {
            error,
            outcome_unknown,
            journal_error,
        } => {
            if *outcome_unknown || journal_error.is_some() {
                StatusCode::SERVICE_UNAVAILABLE
            } else {
                tool_executor_error_status(error)
            }
        }
        ToolRunError::PersistedFailure(failure) => {
            if failure.retryable {
                StatusCode::SERVICE_UNAVAILABLE
            } else {
                StatusCode::BAD_GATEWAY
            }
        }
        ToolRunError::OutputTooLarge { journal_error, .. } => {
            if journal_error.is_some() {
                StatusCode::SERVICE_UNAVAILABLE
            } else {
                StatusCode::BAD_GATEWAY
            }
        }
    }
}

fn tool_executor_error_status(error: &RuntimeToolExecutorError) -> StatusCode {
    match error {
        RuntimeToolExecutorError::WebSearch(error) => web_search_error_status(error),
        RuntimeToolExecutorError::Sandbox(error) => sandbox_error_status(error),
        RuntimeToolExecutorError::UnsupportedConnector(_) => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

fn web_search_error_status(error: &BraveWebSearchError) -> StatusCode {
    match error {
        BraveWebSearchError::Timeout
        | BraveWebSearchError::Transport(_)
        | BraveWebSearchError::ExecutorClosed => StatusCode::SERVICE_UNAVAILABLE,
        BraveWebSearchError::ProviderStatus(status) if *status == 429 || *status >= 500 => {
            StatusCode::SERVICE_UNAVAILABLE
        }
        BraveWebSearchError::UnsupportedRoute { .. } | BraveWebSearchError::UnknownProfile(_) => {
            StatusCode::INTERNAL_SERVER_ERROR
        }
        BraveWebSearchError::InvalidArguments(_)
        | BraveWebSearchError::ProviderStatus(_)
        | BraveWebSearchError::ResponseTooLarge { .. }
        | BraveWebSearchError::Decode(_)
        | BraveWebSearchError::Normalize(_) => StatusCode::BAD_GATEWAY,
    }
}

fn sandbox_error_status(error: &SandboxToolError<HttpSandboxProviderError>) -> StatusCode {
    match error {
        SandboxToolError::UnsupportedOperation(_) | SandboxToolError::IdentityMismatch => {
            StatusCode::INTERNAL_SERVER_ERROR
        }
        SandboxToolError::InvalidArguments(_)
        | SandboxToolError::EmptyCommand
        | SandboxToolError::KnownFailure { .. } => StatusCode::BAD_GATEWAY,
        SandboxToolError::Provider(_)
        | SandboxToolError::OutcomeUnknown
        | SandboxToolError::WaitTimedOut(_) => StatusCode::SERVICE_UNAVAILABLE,
    }
}

fn status_severity(status: StatusCode) -> u8 {
    match status {
        StatusCode::INTERNAL_SERVER_ERROR => 5,
        StatusCode::SERVICE_UNAVAILABLE => 4,
        StatusCode::BAD_GATEWAY => 3,
        StatusCode::FORBIDDEN => 2,
        _ => 1,
    }
}

fn stream_runtime_error_response(error: ResponsesStreamRuntimeError) -> openai::ErrorResponse {
    match error {
        AgentStreamRuntimeError::Runtime(error) => runtime_error_response(error),
        AgentStreamRuntimeError::ExpectedStreaming { checkpoint_error } => {
            if checkpoint_error.is_some() {
                return agent_runtime_public_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Agent runtime could not durably record the failed turn",
                    format_args!(
                        "streaming response type mismatch; failed-state commit: {checkpoint_error:?}"
                    ),
                );
            }
            agent_runtime_public_error(
                StatusCode::BAD_GATEWAY,
                "Inference backend did not return a response stream",
                "inference backend returned unary output for a streaming request",
            )
        }
        AgentStreamRuntimeError::Interpreter { error, .. } => match error {},
        AgentStreamRuntimeError::StagedEventEncoding {
            error,
            checkpoint_error,
        } => agent_runtime_public_error(
            checkpoint_failure_status(&checkpoint_error, StatusCode::BAD_GATEWAY),
            "Inference backend returned an invalid stream event",
            error,
        ),
        AgentStreamRuntimeError::StagedEventLimit {
            limit_bytes,
            checkpoint_error,
        } => agent_runtime_public_error(
            checkpoint_failure_status(&checkpoint_error, StatusCode::BAD_GATEWAY),
            "Inference stream exceeded the runtime staging limit",
            format_args!("staged stream exceeded {limit_bytes} bytes"),
        ),
        AgentStreamRuntimeError::MissingTerminal { checkpoint_error } => {
            let status = if checkpoint_error.is_some() {
                StatusCode::SERVICE_UNAVAILABLE
            } else {
                StatusCode::BAD_GATEWAY
            };
            agent_runtime_public_error(
                status,
                "Inference stream ended without a terminal response",
                format_args!("missing terminal event; failed-state commit: {checkpoint_error:?}"),
            )
        }
    }
}

fn runtime_error_response(error: ResponsesRuntimeError) -> openai::ErrorResponse {
    match error {
        AgentRuntimeError::Store(error) => store_error_response(error),
        AgentRuntimeError::Materialize(error) => materialization_error_response(error),
        AgentRuntimeError::Fingerprint(error) => agent_runtime_public_error(
            StatusCode::BAD_REQUEST,
            "Request could not be fingerprinted",
            error,
        ),
        AgentRuntimeError::LeaseDeadlineOverflow => agent_runtime_public_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Agent runtime could not create the turn",
            "lease deadline overflow",
        ),
        AgentRuntimeError::Inference {
            error,
            checkpoint_error,
        } => {
            if checkpoint_error.is_some() {
                return agent_runtime_public_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Agent runtime could not durably record the failed turn",
                    format_args!(
                        "inference failed: {error}; failed-state commit: {checkpoint_error:?}"
                    ),
                );
            }
            invocation_error_response(error)
        }
        AgentRuntimeError::StreamingUnsupported { checkpoint_error } => {
            let status = if checkpoint_error.is_some() {
                StatusCode::SERVICE_UNAVAILABLE
            } else {
                StatusCode::BAD_GATEWAY
            };
            agent_runtime_public_error(
                status,
                "Inference backend returned an incompatible response",
                format_args!(
                    "unary runtime received streaming output; failed-state commit: {checkpoint_error:?}"
                ),
            )
        }
        AgentRuntimeError::Output {
            error,
            checkpoint_error,
        } => {
            let status = if checkpoint_error.is_some() {
                StatusCode::SERVICE_UNAVAILABLE
            } else {
                StatusCode::BAD_GATEWAY
            };
            agent_runtime_public_error(
                status,
                "Inference backend returned an invalid response",
                format_args!(
                    "output interpretation failed: {error}; failed-state commit: {checkpoint_error:?}"
                ),
            )
        }
        AgentRuntimeError::InvalidOutputState {
            state,
            checkpoint_error,
        } => agent_runtime_public_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Agent runtime produced an invalid turn transition",
            format_args!(
                "invalid output state {state:?}; failed-state commit: {checkpoint_error:?}"
            ),
        ),
    }
}

fn store_error_response(error: DuckDbStoreError) -> openai::ErrorResponse {
    let (status, message) = match &error {
        DuckDbStoreError::Invariant(StoreInvariantError::NotFound) => (
            StatusCode::NOT_FOUND,
            "Previous response was not found or is not accessible",
        ),
        DuckDbStoreError::Invariant(StoreInvariantError::IdempotencyConflict) => (
            StatusCode::CONFLICT,
            "Idempotency key was already used for a different request",
        ),
        DuckDbStoreError::Invariant(
            StoreInvariantError::ResponseAlreadyExists(_)
            | StoreInvariantError::ParentNotReplayable(_)
            | StoreInvariantError::LeaseNotFound
            | StoreInvariantError::LeaseMismatch
            | StoreInvariantError::LeaseExpired
            | StoreInvariantError::VersionConflict
            | StoreInvariantError::ToolAlreadyFinished(_),
        ) => (
            StatusCode::CONFLICT,
            "Agent turn changed concurrently; retry with the same idempotency key",
        ),
        DuckDbStoreError::Invariant(
            StoreInvariantError::InvalidLeaseDeadline
            | StoreInvariantError::LeaseDeadlineNotExtended
            | StoreInvariantError::InvalidTransition { .. }
            | StoreInvariantError::VersionOverflow
            | StoreInvariantError::Corrupt,
        )
        | DuckDbStoreError::Json(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Agent checkpoint store failed",
        ),
        DuckDbStoreError::Database(_)
        | DuckDbStoreError::Poisoned
        | DuckDbStoreError::Closed
        | DuckDbStoreError::Join(_) => (
            StatusCode::SERVICE_UNAVAILABLE,
            "Agent checkpoint store is unavailable",
        ),
    };
    agent_runtime_public_error(status, message, error)
}

fn materialization_error_response(
    error: MaterializationError<Infallible>,
) -> openai::ErrorResponse {
    let (status, message) = match &error {
        MaterializationError::MissingChain | MaterializationError::ScopeMismatch(_) => (
            StatusCode::NOT_FOUND,
            "Previous response was not found or is not accessible",
        ),
        MaterializationError::NonReplayable { .. } => (
            StatusCode::CONFLICT,
            "Previous response is not ready for continuation",
        ),
        MaterializationError::ParentMismatch { .. } => (
            StatusCode::BAD_REQUEST,
            "Continuation does not match the requested previous response",
        ),
        MaterializationError::UnexpectedChain | MaterializationError::BrokenChain(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Agent checkpoint chain is invalid",
        ),
        MaterializationError::Policy(error) => match *error {},
    };
    agent_runtime_public_error(status, message, error)
}

fn invocation_error_response(error: DynamoResponsesInvocationError) -> openai::ErrorResponse {
    match error {
        DynamoResponsesInvocationError::Dynamo { status, message } => {
            agent_runtime_public_error(status, &message, "Dynamo Responses invocation failed")
        }
        DynamoResponsesInvocationError::Body(error) => agent_runtime_public_error(
            StatusCode::BAD_GATEWAY,
            "Failed to read the inference backend response",
            error,
        ),
        DynamoResponsesInvocationError::Decode(error) => agent_runtime_public_error(
            StatusCode::BAD_GATEWAY,
            "Inference backend returned an invalid response",
            error,
        ),
    }
}

fn agent_runtime_public_error(
    status: StatusCode,
    public_message: &str,
    details: impl std::fmt::Display,
) -> openai::ErrorResponse {
    if status.is_server_error() {
        tracing::error!(%status, %details, "agent runtime request failed");
    } else {
        tracing::debug!(%status, %details, "agent runtime request rejected");
    }
    openai::ErrorMessage::agent_runtime_error(status, public_message)
}

fn committed_response_events(
    response: dynamo_protocols::types::responses::Response,
) -> Vec<ResponseStreamEvent> {
    let mut initial = response.clone();
    initial.status = Status::InProgress;
    initial.output.clear();
    let mut sequence_number = 0_u64;
    let mut events = Vec::with_capacity(response.output.len() + 3);
    events.push(ResponseStreamEvent::ResponseCreated(ResponseCreatedEvent {
        sequence_number,
        response: initial.clone(),
    }));
    sequence_number += 1;
    events.push(ResponseStreamEvent::ResponseInProgress(
        ResponseInProgressEvent {
            sequence_number,
            response: initial,
        },
    ));
    sequence_number += 1;
    for (output_index, item) in response.output.iter().cloned().enumerate() {
        events.push(ResponseStreamEvent::ResponseOutputItemDone(
            ResponseOutputItemDoneEvent {
                sequence_number,
                output_index: output_index as u32,
                item,
            },
        ));
        sequence_number += 1;
    }
    events.push(ResponseStreamEvent::ResponseCompleted(
        ResponseCompletedEvent {
            sequence_number,
            response,
        },
    ));
    events
}

type AgentResponsesStream =
    std::pin::Pin<Box<dyn Stream<Item = Result<ResponseStreamEvent, axum::Error>> + Send>>;

fn sse_response(
    stream: AgentResponsesStream,
    serializer: ResponseEventSerializer,
    keep_alive: Option<std::time::Duration>,
) -> Response {
    let stream = async_stream::stream! {
        tokio::pin!(stream);
        while let Some(event) = stream.next().await {
            let event = match event {
                Ok(event) => event,
                Err(error) => {
                    yield Err(error);
                    return;
                }
            };
            yield serializer.serialize(&event).map_err(axum::Error::new);
        }
        yield Ok::<Event, axum::Error>(Event::default().data("[DONE]"));
    };
    let mut response = Sse::new(stream);
    if let Some(keep_alive) = keep_alive {
        response = response.keep_alive(KeepAlive::default().interval(keep_alive));
    }
    response.into_response()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use axum::body::to_bytes;
    use axum::http::{HeaderMap, StatusCode};

    use dynamo_agent_rt::{
        AgentRuntimeError, AnthropicStreamEventInterpreter, AuthorizationScope, CheckpointRecord,
        CheckpointVersion, ConfiguredToolRouter, IdempotencyKey, MaterializationError,
        ModelStepKind, OutputIdentity, RequestFingerprint, ResponseId, StreamEventAction,
        StreamEventInterpreter, ToolRoute, ToolRouter, TurnState,
    };
    use dynamo_agent_rt_store::{DuckDbStoreError, StoreInvariantError};
    use dynamo_protocols::types::anthropic::{
        AnthropicCreateMessageRequest, AnthropicResponseContentBlock, AnthropicStopReason,
        AnthropicStreamEvent, AnthropicUsage,
    };
    use dynamo_protocols::types::responses::CreateResponse;
    use futures::stream;

    use crate::protocols::common::extensions::{
        AGENT_CONTEXT_CONTEXT_KEY, AgentContext, InputTrigger, SESSION_AFFINITY_CONTEXT_KEY,
        SessionAffinityId,
    };
    use crate::protocols::openai::responses::ResponseParams;
    use crate::protocols::openai::responses::stream_converter::ResponseEventSerializer;

    use super::{
        AUTH_HEADER, AUTH_MODE_ENV, AgentRuntimeAuthConfig, AnthropicStore,
        DynamoInvocationCarrier, DynamoResponsesInvocationError, IngressAuthorizationError,
        LOCAL_PRINCIPAL_ENV, LOCAL_TENANT_ENV, PERMITTED_CONNECTORS_ENV, PRINCIPAL_HEADER,
        ResponsesRuntimeError, ResponsesStore, TENANT_HEADER, TRUSTED_PROXY_TOKEN_ENV,
        anthropic_request_requires_runtime, committed_anthropic_events, committed_response_events,
        existing_response, filtered_routing_headers, request_uses_runtime_tools,
        runtime_error_response, sandbox_components_from_config, sse_response,
        web_search_components_from_config,
    };

    #[test]
    fn stream_staging_is_selected_only_for_declared_runtime_tools() {
        let router =
            ConfiguredToolRouter::new([("python".to_owned(), ToolRoute::new("sandbox", "python"))]);
        let request = |tool_name: &str| -> CreateResponse {
            serde_json::from_value(serde_json::json!({
                "model": "model",
                "input": "hello",
                "tools": [{
                    "type": "function",
                    "name": tool_name,
                    "parameters": {"type": "object"}
                }]
            }))
            .unwrap()
        };

        assert!(request_uses_runtime_tools(&request("python"), &router));
        assert!(!request_uses_runtime_tools(
            &request("client_shell"),
            &router
        ));
        assert!(!request_uses_runtime_tools(
            &CreateResponse::default(),
            &router
        ));
    }

    #[test]
    fn anthropic_runtime_selection_preserves_client_tool_passthrough() {
        let router =
            ConfiguredToolRouter::new([("web".to_owned(), ToolRoute::new("web_search", "search"))]);
        let request = |tool_name: &str| -> AnthropicCreateMessageRequest {
            serde_json::from_value(serde_json::json!({
                "model": "model",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [{
                    "name": tool_name,
                    "description": "test",
                    "input_schema": {"type": "object"}
                }]
            }))
            .unwrap()
        };

        assert!(!anthropic_request_requires_runtime(
            &request("client_shell"),
            &router,
            false
        ));
        assert!(anthropic_request_requires_runtime(
            &request("web"),
            &router,
            false
        ));
        assert!(anthropic_request_requires_runtime(
            &request("client_shell"),
            &router,
            true
        ));
    }

    #[test]
    fn anthropic_replay_stream_reconstructs_the_committed_response() {
        let response = dynamo_protocols::types::anthropic::AnthropicMessageResponse {
            id: "msg_public".to_owned(),
            object_type: "message".to_owned(),
            role: "assistant".to_owned(),
            content: vec![
                AnthropicResponseContentBlock::Text {
                    text: "hello".to_owned(),
                    citations: None,
                },
                AnthropicResponseContentBlock::ToolUse {
                    id: "tool_1".to_owned(),
                    name: "client_shell".to_owned(),
                    input: serde_json::json!({"command": "pwd"}),
                },
            ],
            model: "model".to_owned(),
            stop_reason: Some(AnthropicStopReason::ToolUse),
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: 5,
                output_tokens: 3,
                ..Default::default()
            },
        };
        let events = committed_anthropic_events(response.clone());
        assert!(
            matches!(events.first(), Some(AnthropicStreamEvent::MessageStart { message }) if message.id == "msg_public")
        );
        assert!(matches!(
            events.last(),
            Some(AnthropicStreamEvent::MessageStop {})
        ));

        let identity = OutputIdentity {
            response_id: ResponseId::from("msg_public"),
            parent_response_id: None,
        };
        let mut interpreter = AnthropicStreamEventInterpreter::default();
        interpreter.begin_step(ModelStepKind::Initial);
        let mut reconstructed = None;
        for event in events {
            if let StreamEventAction::Terminal { response, .. } =
                interpreter.observe(event, &identity).unwrap()
            {
                reconstructed = Some(response);
            }
        }
        let reconstructed = reconstructed.expect("message_stop reconstructs the response");
        assert_eq!(
            serde_json::to_value(reconstructed).unwrap(),
            serde_json::to_value(response).unwrap()
        );
    }

    #[test]
    fn anthropic_internal_steps_receive_only_routing_headers() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer secret".parse().unwrap());
        headers.insert("x-dynamo-agent-rt-auth", "runtime-secret".parse().unwrap());
        headers.insert("x-dynamo-tenant-id", "tenant-auth".parse().unwrap());
        headers.insert("x-tenant-id", "tenant-cache-salt".parse().unwrap());
        headers.insert("x-dynamo-request-priority", "7".parse().unwrap());

        let filtered = filtered_routing_headers(&headers);
        assert_eq!(filtered["x-tenant-id"], "tenant-cache-salt");
        assert_eq!(filtered["x-dynamo-request-priority"], "7");
        assert!(!filtered.contains_key("authorization"));
        assert!(!filtered.contains_key("x-dynamo-agent-rt-auth"));
        assert!(!filtered.contains_key("x-dynamo-tenant-id"));
    }

    #[test]
    fn duckdb_file_can_host_responses_and_anthropic_protocols() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("agent-runtime.duckdb");
        let responses = ResponsesStore::open(&path).expect("open Responses store");
        let anthropic = AnthropicStore::open(&path).expect("open Anthropic store");
        drop(anthropic);
        drop(responses);
    }

    #[test]
    fn web_search_route_exists_only_with_deployment_credentials() {
        let (router, _) =
            web_search_components_from_config(None, None).expect("disabled configuration is valid");
        assert!(router.route("web_search").is_none());

        let (router, _) = web_search_components_from_config(
            Some("deployment-secret".to_owned()),
            Some("search_the_web".to_owned()),
        )
        .expect("web-search configuration is valid");
        let route = router.route("search_the_web").expect("route configured");
        assert_eq!(route.connector, "web_search");
        assert_eq!(route.operation, "search");
        assert_eq!(route.profile, "brave_default");
        assert!(router.route("web_search").is_none());
    }

    #[test]
    fn sandbox_route_requires_an_authenticated_deployment_endpoint() {
        let (router, executor) = sandbox_components_from_config(None, None, false, None, None)
            .expect("disabled sandbox configuration is valid");
        assert!(router.route("python").is_none());
        assert!(executor.is_none());

        let missing_token = sandbox_components_from_config(
            Some("https://sandbox.example.com".to_owned()),
            None,
            false,
            None,
            None,
        );
        assert!(matches!(missing_token, Err(error) if error.to_string().contains("SANDBOX_TOKEN")));

        let (router, executor) = sandbox_components_from_config(
            Some("http://127.0.0.1:8090".to_owned()),
            Some("0123456789abcdef0123456789abcdef".to_owned()),
            true,
            Some("run_python".to_owned()),
            Some("python-deny-egress".to_owned()),
        )
        .expect("explicit local HTTP sandbox configuration is valid");
        let route = router.route("run_python").expect("route configured");
        assert_eq!(route.connector, "sandbox");
        assert_eq!(route.operation, "python");
        assert_eq!(route.profile, "python-deny-egress");
        assert!(executor.is_some());
    }

    #[test]
    fn active_runtime_tool_turn_is_not_replayed_as_complete() {
        let record = Box::new(CheckpointRecord {
            response_id: ResponseId::from("resp-1"),
            parent_response_id: None,
            scope: AuthorizationScope {
                tenant_id: "tenant".to_owned(),
                principal_id: "principal".to_owned(),
            },
            idempotency_key: IdempotencyKey::from("idem-1"),
            request_fingerprint: RequestFingerprint::new([0; 32]),
            state: TurnState::ToolStarted,
            version: CheckpointVersion(1),
            request: CreateResponse::default(),
            output_items: Vec::new(),
            response: None,
        });

        assert_eq!(
            existing_response(record).unwrap_err().0,
            StatusCode::CONFLICT
        );
    }

    #[test]
    fn runtime_errors_have_stable_non_leaking_http_statuses() {
        let missing: ResponsesRuntimeError =
            AgentRuntimeError::Store(DuckDbStoreError::Invariant(StoreInvariantError::NotFound));
        let response = runtime_error_response(missing);
        assert_eq!(response.0, StatusCode::NOT_FOUND);
        assert_eq!(
            response.1.message(),
            "Previous response was not found or is not accessible"
        );

        let conflict: ResponsesRuntimeError = AgentRuntimeError::Store(
            DuckDbStoreError::Invariant(StoreInvariantError::IdempotencyConflict),
        );
        assert_eq!(runtime_error_response(conflict).0, StatusCode::CONFLICT);

        let cross_scope: ResponsesRuntimeError = AgentRuntimeError::Materialize(
            MaterializationError::ScopeMismatch(ResponseId::from("resp-private")),
        );
        assert_eq!(runtime_error_response(cross_scope).0, StatusCode::NOT_FOUND);

        let not_replayable: ResponsesRuntimeError =
            AgentRuntimeError::Materialize(MaterializationError::NonReplayable {
                response_id: ResponseId::from("resp-running"),
                state: TurnState::InFlight,
            });
        assert_eq!(
            runtime_error_response(not_replayable).0,
            StatusCode::CONFLICT
        );

        let downstream_overload: ResponsesRuntimeError = AgentRuntimeError::Inference {
            error: DynamoResponsesInvocationError::Dynamo {
                status: StatusCode::TOO_MANY_REQUESTS,
                message: "backend overloaded".to_owned(),
            },
            checkpoint_error: None,
        };
        assert_eq!(
            runtime_error_response(downstream_overload).0,
            StatusCode::TOO_MANY_REQUESTS
        );

        let durability_unknown: ResponsesRuntimeError = AgentRuntimeError::Inference {
            error: DynamoResponsesInvocationError::Dynamo {
                status: StatusCode::BAD_REQUEST,
                message: "bad request".to_owned(),
            },
            checkpoint_error: Some(DuckDbStoreError::Poisoned),
        };
        assert_eq!(
            runtime_error_response(durability_unknown).0,
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    fn auth_config(entries: &[(&str, &str)]) -> Result<AgentRuntimeAuthConfig, String> {
        let environment: HashMap<&str, &str> = entries.iter().copied().collect();
        AgentRuntimeAuthConfig::from_lookup(|name| environment.get(name).map(ToString::to_string))
    }

    #[test]
    fn local_authorization_ignores_caller_identity() {
        let config = auth_config(&[
            (AUTH_MODE_ENV, "local"),
            (LOCAL_TENANT_ENV, "configured-tenant"),
            (LOCAL_PRINCIPAL_ENV, "configured-principal"),
            (PERMITTED_CONNECTORS_ENV, "web_search,sandbox"),
        ])
        .unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(TENANT_HEADER, "spoofed-tenant".parse().unwrap());
        headers.insert(PRINCIPAL_HEADER, "spoofed-principal".parse().unwrap());

        let authorization = config.authorize(&headers).unwrap();

        assert_eq!(authorization.scope.tenant_id, "configured-tenant");
        assert_eq!(authorization.scope.principal_id, "configured-principal");
        assert_eq!(
            authorization.permitted_connectors,
            ["sandbox".to_owned(), "web_search".to_owned()].into()
        );
    }

    #[test]
    fn trusted_proxy_requires_secret_before_accepting_identity() {
        let token = "a-trusted-proxy-secret-with-32-bytes-minimum";
        let config = auth_config(&[
            (AUTH_MODE_ENV, "trusted_proxy"),
            (TRUSTED_PROXY_TOKEN_ENV, token),
        ])
        .unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(TENANT_HEADER, "tenant-a".parse().unwrap());
        headers.insert(PRINCIPAL_HEADER, "principal-a".parse().unwrap());

        assert_eq!(
            config.authorize(&headers).unwrap_err(),
            IngressAuthorizationError::Unauthorized
        );
        headers.insert(AUTH_HEADER, "wrong-secret".parse().unwrap());
        assert_eq!(
            config.authorize(&headers).unwrap_err(),
            IngressAuthorizationError::Unauthorized
        );
        headers.insert(AUTH_HEADER, token.parse().unwrap());
        let authorization = config.authorize(&headers).unwrap();
        assert_eq!(authorization.scope.tenant_id, "tenant-a");
        assert_eq!(authorization.scope.principal_id, "principal-a");
    }

    #[test]
    fn authorization_configuration_is_fail_closed() {
        assert!(
            auth_config(&[])
                .err()
                .expect("missing mode rejected")
                .contains(AUTH_MODE_ENV)
        );
        assert!(
            auth_config(&[(AUTH_MODE_ENV, "trusted_proxy")])
                .err()
                .expect("missing token rejected")
                .contains(TRUSTED_PROXY_TOKEN_ENV)
        );
        assert!(
            auth_config(&[
                (AUTH_MODE_ENV, "trusted_proxy"),
                (TRUSTED_PROXY_TOKEN_ENV, "too-short")
            ])
            .err()
            .expect("short token rejected")
            .contains("32 to 1024")
        );
    }

    #[test]
    fn invocation_carrier_forwards_only_typed_ingress_data() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer client-secret".parse().unwrap());
        headers.insert("x-dynamo-tenant-id", "spoofed-tenant".parse().unwrap());
        headers.insert("thread-id", "thread-123".parse().unwrap());
        headers.insert("x-codex-parent-thread-id", "thread-parent".parse().unwrap());
        headers.insert("x-dynamo-session-final", "true".parse().unwrap());
        headers.insert("x-dynamo-meta-policy-class", "latency".parse().unwrap());
        headers.insert(
            "x-dynamo-meta-authorization",
            "Bearer nested-secret".parse().unwrap(),
        );

        let carrier = DynamoInvocationCarrier::from_headers(&headers).unwrap();
        assert_eq!(
            carrier.metadata.get("policy-class").map(String::as_str),
            Some("latency")
        );
        assert!(!carrier.metadata.contains_key("authorization"));
        assert_eq!(
            carrier
                .agent_context
                .as_ref()
                .map(|context| context.session_id.as_str()),
            Some("thread-123")
        );
        assert_eq!(
            carrier
                .agent_context
                .as_ref()
                .and_then(|context| context.parent_session_id.as_deref()),
            Some("thread-parent")
        );
        assert_eq!(
            carrier
                .agent_context
                .as_ref()
                .and_then(|context| context.session_final),
            None
        );
        assert_eq!(
            carrier
                .session_affinity
                .as_ref()
                .map(SessionAffinityId::as_str),
            Some("thread-123")
        );

        let request = serde_json::from_value(serde_json::json!({
            "model": "model",
            "input": "hello"
        }))
        .unwrap();
        let context = carrier.context(request, "request-1".to_string());
        assert_eq!(
            context.metadata().get("policy-class").map(String::as_str),
            Some("latency")
        );
        let agent_context = context
            .get::<AgentContext>(AGENT_CONTEXT_CONTEXT_KEY)
            .expect("agent context attached");
        assert_eq!(agent_context.session_final, None);
        assert_eq!(agent_context.kv_hints, None);
        assert_eq!(agent_context.input_trigger, Some(InputTrigger::UserMessage));
        let affinity = context
            .get::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY)
            .expect("session affinity attached");
        assert_eq!(affinity.as_str(), "thread-123");
    }

    #[tokio::test]
    async fn completed_response_is_exposed_as_codex_compatible_sse() {
        let inner = serde_json::from_value(serde_json::json!({
            "created_at": 1,
            "id": "resp_public",
            "model": "model",
            "object": "response",
            "output": [{
                "type": "message",
                "id": "msg-1",
                "role": "assistant",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "text": "hello",
                    "annotations": [],
                    "logprobs": null
                }]
            }],
            "status": "completed"
        }))
        .unwrap();
        let stream = Box::pin(stream::iter(
            committed_response_events(inner).into_iter().map(Ok),
        ));
        let response = sse_response(
            stream,
            ResponseEventSerializer::new(&ResponseParams::default()),
            None,
        );

        assert_eq!(response.headers()["content-type"], "text/event-stream");
        let body = to_bytes(response.into_body(), 1 << 20).await.unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        let output_done = body.find("event: response.output_item.done").unwrap();
        let completed = body.find("event: response.completed").unwrap();
        assert!(output_done < completed);
        assert!(body.contains("resp_public"));
        assert!(body.contains("hello"));
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct DynamoResponsesInvoker;

#[derive(Debug, Error)]
pub(super) enum DynamoResponsesInvocationError {
    #[error("Dynamo Responses invocation failed ({status}): {message}")]
    Dynamo { status: StatusCode, message: String },
    #[error("failed to read Dynamo Responses body: {0}")]
    Body(#[from] axum::Error),
    #[error("failed to decode Dynamo Responses body: {0}")]
    Decode(#[from] serde_json::Error),
}

impl InferenceInvoker<OpenAiResponses> for DynamoResponsesInvoker {
    type Context = DynamoResponsesContext;
    type Error = DynamoResponsesInvocationError;

    fn invoke<'a>(
        &'a self,
        request: &'a InferenceRequest<OpenAiResponses, Self::Context>,
    ) -> InferenceFuture<'a, OpenAiResponses, Self::Error> {
        Box::pin(async move {
            let inner = request.request.clone();
            let streaming = inner.stream.unwrap_or(false);
            let model = resolve_request_model(
                inner.model.as_deref().unwrap_or(""),
                request.context.template.as_ref(),
            )
            .to_owned();
            let pipeline_request = request.context.carrier.context(
                NvCreateResponse {
                    inner,
                    nvext: request.context.nvext.clone(),
                },
                request.context.request_id.clone(),
            );
            let engine_context = pipeline_request.context();
            let labels = CancellationLabels {
                model: request
                    .context
                    .state
                    .manager()
                    .metric_model_for(&model)
                    .to_owned(),
                endpoint: Endpoint::Responses.to_string(),
                request_type: if streaming {
                    "agent_runtime_stream"
                } else {
                    "agent_runtime_unary"
                }
                .to_owned(),
            };
            let (mut connection_handle, stream_handle) = create_connection_monitor(
                engine_context,
                Some(request.context.state.metrics_clone()),
                labels,
            )
            .await;

            if streaming {
                let stream = openai::responses_native_stream(
                    request.context.state.clone(),
                    request.context.template.clone(),
                    pipeline_request,
                    stream_handle,
                )
                .await
                .map_err(|(status, message)| {
                    DynamoResponsesInvocationError::Dynamo {
                        status,
                        message: message.message().to_owned(),
                    }
                })?;
                connection_handle.disarm();
                return Ok(InferenceOutput::Streaming(Box::pin(stream.map(Ok))));
            }

            let response = openai::responses(
                request.context.state.clone(),
                request.context.template.clone(),
                pipeline_request,
                stream_handle,
            )
            .await
            .map_err(|(status, message)| DynamoResponsesInvocationError::Dynamo {
                status,
                message: message.message().to_owned(),
            })?;
            connection_handle.disarm();

            if !response.status().is_success() {
                return Err(DynamoResponsesInvocationError::Dynamo {
                    status: response.status(),
                    message: "Dynamo Responses invocation failed".to_owned(),
                });
            }
            let body = to_bytes(response.into_body(), openai::get_body_limit()).await?;
            let response: NvResponse = serde_json::from_slice(&body)?;
            Ok(InferenceOutput::Unary(Box::new(response.inner)))
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct DynamoAnthropicInvoker;

#[derive(Debug, Error)]
pub(super) enum DynamoAnthropicInvocationError {
    #[error("Dynamo Anthropic invocation failed ({status}): {message}")]
    Dynamo { status: StatusCode, message: String },
    #[error("failed to read Dynamo Anthropic body: {0}")]
    Body(#[from] axum::Error),
    #[error("failed to decode Dynamo Anthropic body: {0}")]
    Decode(#[from] serde_json::Error),
}

async fn anthropic_invocation_error(response: Response) -> DynamoAnthropicInvocationError {
    let status = response.status();
    let message = match to_bytes(response.into_body(), openai::get_body_limit()).await {
        Ok(body) => serde_json::from_slice::<AnthropicErrorResponse>(&body)
            .map(|error| error.error.message)
            .unwrap_or_else(|_| "Dynamo Anthropic invocation failed".to_owned()),
        Err(error) => {
            tracing::error!(%status, error = %error, "failed to read Dynamo Anthropic error body");
            "Dynamo Anthropic invocation failed".to_owned()
        }
    };
    DynamoAnthropicInvocationError::Dynamo { status, message }
}

impl InferenceInvoker<AnthropicMessages> for DynamoAnthropicInvoker {
    type Context = DynamoAnthropicContext;
    type Error = DynamoAnthropicInvocationError;

    fn invoke<'a>(
        &'a self,
        request: &'a InferenceRequest<AnthropicMessages, Self::Context>,
    ) -> InferenceFuture<'a, AnthropicMessages, Self::Error> {
        Box::pin(async move {
            let inner = request.request.clone();
            let streaming = inner.stream;
            let model =
                resolve_request_model(&inner.model, request.context.template.as_ref()).to_owned();
            let pipeline_request = request
                .context
                .carrier
                .anthropic_context(inner, request.context.request_id.clone());
            let engine_context = pipeline_request.context();
            let metric_model = request
                .context
                .state
                .manager()
                .metric_model_for(&model)
                .to_owned();
            let inflight_guard = request.context.state.metrics_clone().create_inflight_guard(
                &metric_model,
                Endpoint::AnthropicMessages,
                streaming,
                &request.context.request_id,
            );
            let labels = CancellationLabels {
                model: metric_model,
                endpoint: Endpoint::AnthropicMessages.to_string(),
                request_type: if streaming {
                    "agent_runtime_stream"
                } else {
                    "agent_runtime_unary"
                }
                .to_owned(),
            };
            let (mut connection_handle, stream_handle) = create_connection_monitor(
                engine_context,
                Some(request.context.state.metrics_clone()),
                labels,
            )
            .await;

            if streaming {
                let stream = match anthropic::anthropic_messages_native_stream(
                    request.context.state.clone(),
                    request.context.template.clone(),
                    pipeline_request,
                    request.context.routing_headers.clone(),
                    stream_handle,
                    inflight_guard,
                )
                .await
                {
                    Ok(stream) => stream,
                    Err(response) => return Err(anthropic_invocation_error(response).await),
                };
                connection_handle.disarm();
                return Ok(InferenceOutput::Streaming(Box::pin(stream.map(Ok))));
            }

            let response = match anthropic::anthropic_messages(
                request.context.state.clone(),
                request.context.template.clone(),
                pipeline_request,
                request.context.routing_headers.clone(),
                stream_handle,
                inflight_guard,
            )
            .await
            {
                Ok(response) => response,
                Err(response) => return Err(anthropic_invocation_error(response).await),
            };
            connection_handle.disarm();

            if !response.status().is_success() {
                return Err(anthropic_invocation_error(response).await);
            }
            let body = to_bytes(response.into_body(), openai::get_body_limit()).await?;
            let response: AnthropicMessageResponse = serde_json::from_slice(&body)?;
            Ok(InferenceOutput::Unary(Box::new(response)))
        })
    }
}
