// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo backend that serves one vLLM prefill and one TileRT decode server
//! as a single worker.
//!
//! TileRT's prefill pushes the prompt KV to a decode server that the request
//! itself names, so the decode side has to be known before prefill starts.
//! Dynamo's prefill router picks the prefill first and the decode afterwards;
//! pairing the two engines behind one worker sidesteps that ordering.

use std::collections::HashMap;
use std::future::Future;
use std::num::NonZeroUsize;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use dynamo_backend_common::{
    AsyncEngineContext, BackendError, CompletionUsage, DisaggregationMode, DynamoError,
    EngineConfig, GenerateContext, LLMEngine, LLMEngineOutput, LLMEngineOutputExt, LlmRegistration,
    PreprocessedRequest, PromptTokensDetails, RuntimeConfig, WorkerConfig, usage,
};
use dynamo_sidecar_common::{
    EngineBootstrapResult, GrpcEndpoint, GrpcTransportConfig, connection_timeout, engine_shutdown,
    invalid_argument, startup_deadline, stream_incomplete,
};
use futures::StreamExt;
use futures::stream::BoxStream;
use tokio::sync::{OnceCell, OwnedSemaphorePermit, Semaphore};
use tokio::time::{Instant, timeout, timeout_at};
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

use crate::args::Args;
use crate::decode::{DecodeBody, DecodeCapabilities, DecodeClient, DecodeEvent, DecodeFinish};
use crate::error::{backend_error, is_kind, is_transport_failure};
use crate::prefill::{self, PrefillClient, PrefillRequest};

const ALREADY_STARTED: &str = "TileRT sidecar has already started";
const PROBE_MAX_TOKENS: u32 = 4;
/// Token batches follow each other within milliseconds, so a decode that
/// sends nothing for this long has stalled.
const DECODE_IDLE_TIMEOUT: Duration = Duration::from_secs(60);
/// How long shutdown waits for an in-flight request to release its KV.
const CLEANUP_DRAIN: Duration = Duration::from_secs(10);
/// How long a cancelled request waits for the decode server to claim its KV.
/// KV normally lands within milliseconds of the prefill's answer.
const CLAIM_GRACE: Duration = Duration::from_secs(1);
const DECODE_HEALTH_INTERVAL: Duration = Duration::from_secs(5);
const DECODE_HEALTH_FAILURES: u32 = 3;

/// A request's sampling settings, resolved once so the prefill and the decode
/// sample alike.
#[derive(Clone, Debug)]
pub(crate) struct RequestPlan {
    pub max_tokens: u32,
    pub temperature: f32,
    pub seed: Option<i64>,
    pub ignore_eos: bool,
    pub repetition_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
}

impl RequestPlan {
    /// Refuses what the decode stage cannot honour before either engine does
    /// any work. vLLM would apply such a setting to the first token and the
    /// decode would ignore it for the rest of the reply.
    pub(crate) fn new(
        request: &PreprocessedRequest,
        context_length: u32,
        capabilities: DecodeCapabilities,
    ) -> Result<Self, DynamoError> {
        if let Some(what) = unsupported(request) {
            return Err(invalid_argument(format!(
                "{what} is not supported by the TileRT sidecar"
            )));
        }
        let sampling = &request.sampling_options;
        let repetition_penalty = sampling
            .repetition_penalty
            .filter(|penalty| *penalty != 1.0);
        let presence_penalty = sampling.presence_penalty.filter(|penalty| *penalty != 0.0);
        if (repetition_penalty.is_some() || presence_penalty.is_some()) && !capabilities.penalties {
            return Err(invalid_argument(
                "this TileRT decode server does not support repetition or presence penalties",
            ));
        }
        let ignore_eos = request.stop_conditions.ignore_eos.unwrap_or(false);
        if ignore_eos && !capabilities.ignore_eos {
            return Err(invalid_argument(
                "this TileRT decode server does not support ignore_eos",
            ));
        }

        let prompt_tokens = u32::try_from(request.token_ids.len()).unwrap_or(u32::MAX);
        if prompt_tokens == 0 {
            return Err(invalid_argument("the request has no prompt tokens"));
        }
        let room = context_length.saturating_sub(prompt_tokens);
        if room == 0 {
            return Err(invalid_argument(format!(
                "the prompt has {prompt_tokens} tokens and the context window is {context_length}"
            )));
        }
        let max_tokens = request.stop_conditions.max_tokens.unwrap_or(room).min(room);
        if max_tokens == 0 {
            return Err(invalid_argument("max_tokens must be at least 1"));
        }
        Ok(Self {
            max_tokens,
            temperature: sampling.temperature.unwrap_or(1.0),
            seed: sampling.seed,
            ignore_eos,
            repetition_penalty,
            presence_penalty,
        })
    }

    pub(crate) fn prefill_request(
        &self,
        rid: String,
        token_ids: Vec<u32>,
        decode_kv_host: &str,
        decode_ctrl_port: u16,
    ) -> PrefillRequest {
        PrefillRequest {
            request_id: rid,
            token_ids,
            temperature: self.temperature,
            seed: self.seed,
            ignore_eos: self.ignore_eos,
            repetition_penalty: self.repetition_penalty,
            presence_penalty: self.presence_penalty,
            decode_kv_host: decode_kv_host.to_string(),
            decode_ctrl_port,
        }
    }

    /// The `sampling` object of a `/pd/decode` body.
    pub(crate) fn decode_sampling(&self) -> serde_json::Map<String, serde_json::Value> {
        let mut sampling = serde_json::Map::new();
        sampling.insert("temperature".into(), f64::from(self.temperature).into());
        sampling.insert("top_p".into(), 1.0.into());
        sampling.insert("top_k".into(), 0.into());
        if let Some(penalty) = self.repetition_penalty {
            sampling.insert("repetition_penalty".into(), f64::from(penalty).into());
        }
        if let Some(penalty) = self.presence_penalty {
            sampling.insert("presence_penalty".into(), f64::from(penalty).into());
        }
        if let Some(seed) = self.seed {
            sampling.insert("seed".into(), seed.into());
        }
        if self.ignore_eos {
            sampling.insert("ignore_eos".into(), true.into());
        }
        sampling
    }
}

/// The first request setting that no TileRT decode server honours, if any.
fn unsupported(request: &PreprocessedRequest) -> Option<&'static str> {
    let sampling = &request.sampling_options;
    let stop = &request.stop_conditions;
    let output = &request.output_options;
    [
        (
            output.logprobs.is_some() || output.prompt_logprobs.is_some(),
            "logprobs",
        ),
        (sampling.n.is_some_and(|n| n != 1), "n other than 1"),
        (
            sampling.best_of.is_some_and(|best_of| best_of != 1),
            "best_of other than 1",
        ),
        (sampling.use_beam_search == Some(true), "beam search"),
        (sampling.guided_decoding.is_some(), "structured output"),
        (
            sampling.top_p.is_some_and(|top_p| top_p != 1.0),
            "top_p other than 1",
        ),
        (
            sampling.top_k.is_some_and(|top_k| !matches!(top_k, -1 | 0)),
            "top_k",
        ),
        (sampling.min_p.is_some_and(|min_p| min_p != 0.0), "min_p"),
        (
            sampling
                .frequency_penalty
                .is_some_and(|penalty| penalty != 0.0),
            "frequency_penalty",
        ),
        (
            sampling
                .length_penalty
                .is_some_and(|penalty| penalty != 1.0),
            "length_penalty",
        ),
        (
            sampling
                .logit_bias
                .as_ref()
                .is_some_and(|bias| !bias.is_empty()),
            "logit_bias",
        ),
        (
            stop.min_tokens.is_some_and(|min_tokens| min_tokens > 0),
            "min_tokens",
        ),
        (
            stop.max_thinking_tokens.is_some(),
            "a thinking token budget",
        ),
        (
            request.multi_modal_data.is_some() || request.prompt_embeds.is_some(),
            "multimodal or embedded input",
        ),
    ]
    .into_iter()
    .find_map(|(requested, what)| requested.then_some(what))
}

/// TileRT derives the id it files a request's KV under from vLLM's internal
/// request id by stripping `cmpl-`/`chatcmpl-` prefixes and short numeric or
/// hex suffixes. A dash-free 32-hex id survives that unchanged.
fn new_request_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

fn request_usage(
    prompt_tokens: u32,
    completion_tokens: u32,
    cached: Option<u32>,
) -> CompletionUsage {
    let mut completion_usage = usage(prompt_tokens, completion_tokens);
    completion_usage.prompt_tokens_details = cached.map(|cached_tokens| PromptTokensDetails {
        cached_tokens: Some(cached_tokens.min(prompt_tokens)),
        ..Default::default()
    });
    completion_usage
}

fn millis(since: Instant) -> f64 {
    (since.elapsed().as_secs_f64() * 10_000.0).round() / 10.0
}

/// What is left to undo on the engines if a request ends now.
#[derive(Clone, Copy)]
enum Pending {
    /// vLLM is running the prefill and may push its KV.
    Prefill,
    /// The prefill is over; its KV may sit on the decode server, unclaimed.
    Kv,
    /// `/pd/decode` is in flight and may or may not have claimed the KV.
    Decode,
    /// The decode server has answered; it cleans up after itself.
    Nothing,
}

/// Holds the worker's slot for one request. When the request ends before the
/// decode server has taken it over, the slot stays taken until both engines
/// are clean, so the next request cannot meet this one's KV. It acts on drop
/// because the framework may drop a stream without cancelling it.
struct Handoff {
    prefill: Arc<PrefillClient>,
    decode: DecodeClient,
    rid: String,
    pending: Pending,
    permit: Option<OwnedSemaphorePermit>,
}

impl Drop for Handoff {
    fn drop(&mut self) {
        let (Some(permit), pending) = (self.permit.take(), self.pending) else {
            return;
        };
        if matches!(pending, Pending::Nothing) {
            return;
        }
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let (prefill, decode) = (self.prefill.clone(), self.decode.clone());
        let rid = std::mem::take(&mut self.rid);
        let clean = async move {
            match pending {
                Pending::Prefill => {
                    prefill.abort(&rid).await;
                    decode.release(&rid).await;
                }
                Pending::Kv => {
                    decode.release(&rid).await;
                }
                Pending::Decode => {
                    decode.cancel(&rid).await;
                    // The abandoned `/pd/decode` may still claim KV that lands
                    // while the release waits for it.
                    if !decode.release(&rid).await {
                        decode.cancel(&rid).await;
                    }
                }
                Pending::Nothing => {}
            }
            drop(permit);
        };
        runtime.spawn(clean.in_current_span());
    }
}

pub struct TilertSidecarEngine {
    grpc_endpoint: GrpcEndpoint,
    transport: GrpcTransportConfig,
    decode: DecodeClient,
    decode_kv_host: String,
    decode_ctrl_port: u16,
    model_path: String,
    served_model_name: Option<String>,
    context_length: u32,
    kv_transfer_timeout: Duration,
    capabilities: DecodeCapabilities,
    prefill: OnceCell<Arc<PrefillClient>>,
    /// The decode server holds one request's KV at a time.
    slot: Arc<Semaphore>,
    shutdown: CancellationToken,
    lost: CancellationToken,
    lost_reason: Arc<OnceLock<String>>,
}

impl TilertSidecarEngine {
    /// Parses the command line without connecting; the engines are contacted
    /// once the shared runner has started its probes.
    pub fn from_cli() -> Result<
        (
            RuntimeConfig,
            impl Future<Output = EngineBootstrapResult<Self>>,
        ),
        DynamoError,
    > {
        let args = <Args as clap::Parser>::parse();
        args.validate()?;
        Ok((args.sidecar.common.runtime.clone(), Self::bootstrap(args)))
    }

    /// Waits for both engines, checks the KV handoff with one short request,
    /// and builds the worker registration. It holds one probe connection
    /// through the wait and drops it here; `start` opens the configured pool.
    pub(crate) async fn bootstrap(args: Args) -> EngineBootstrapResult<Self> {
        let endpoint = args.sidecar.grpc_endpoint.clone();
        let transport = args.sidecar.grpc.config();
        let deadline = startup_deadline(transport.startup_deadline)?;
        let decode = DecodeClient::new(args.decode_http_endpoint.clone())?;
        let decode_kv_host = args.decode_kv_host();
        let kv_transfer_timeout = Duration::from_secs(args.kv_transfer_timeout_secs);
        tracing::info!(
            prefill = %endpoint,
            decode = %args.decode_http_endpoint,
            startup_deadline = ?transport.startup_deadline,
            "waiting for the vLLM prefill and the TileRT decode server"
        );
        let probe_transport = GrpcTransportConfig {
            connections: NonZeroUsize::MIN,
            ..transport
        };
        let prefill = PrefillClient::connect(&endpoint, probe_transport, deadline).await?;
        decode
            .wait_healthy(deadline, transport.retry_interval)
            .await?;
        let capabilities = decode.capabilities().await?;
        tracing::info!(?capabilities, "TileRT decode capabilities");

        if args.probe_token_ids.is_empty() {
            tracing::warn!(
                "no --probe-token-ids: the KV handoff is not checked before registration"
            );
        } else {
            let probe = Probe {
                prefill: &prefill,
                decode: &decode,
                decode_kv_host: &decode_kv_host,
                decode_ctrl_port: args.decode_ctrl_port,
                kv_transfer_timeout,
            };
            timeout_at(deadline, probe.run(args.probe_token_ids.clone()))
                .await
                .map_err(|_| {
                    connection_timeout("the startup probe exceeded the startup deadline")
                })??;
        }
        drop(prefill);

        let common = args.sidecar.common;
        let engine = Self {
            grpc_endpoint: endpoint,
            transport,
            decode,
            decode_kv_host,
            decode_ctrl_port: args.decode_ctrl_port,
            model_path: args.model_path.clone(),
            served_model_name: args.served_model_name.clone(),
            context_length: args.context_length,
            kv_transfer_timeout,
            capabilities,
            prefill: OnceCell::new(),
            slot: Arc::new(Semaphore::new(1)),
            shutdown: CancellationToken::new(),
            lost: CancellationToken::new(),
            lost_reason: Arc::default(),
        };
        let config = WorkerConfig {
            runtime: common.runtime,
            namespace: common.namespace,
            component: common.component,
            endpoint: common.endpoint,
            endpoint_types: common.endpoint_types,
            custom_jinja_template: common.custom_jinja_template,
            model_name: args.model_path,
            served_model_name: args.served_model_name,
            tool_call_parser: common.dyn_tool_call_parser,
            reasoning_parser: common.dyn_reasoning_parser,
            exclude_tools_when_tool_choice_none: common.exclude_tools_when_tool_choice_none,
            enable_kv_routing: false,
            disaggregation_mode: DisaggregationMode::Aggregated,
            route_to_encoder: false,
            enable_rl: common.enable_rl,
            default_thinking_mode: args.default_thinking_mode,
            ..Default::default()
        };
        Ok((engine, config))
    }
}

/// One short request through both engines.
struct Probe<'a> {
    prefill: &'a PrefillClient,
    decode: &'a DecodeClient,
    decode_kv_host: &'a str,
    decode_ctrl_port: u16,
    kv_transfer_timeout: Duration,
}

impl Probe<'_> {
    async fn run(&self, token_ids: Vec<u32>) -> Result<(), DynamoError> {
        let rid = new_request_id();
        let prompt_tokens = token_ids.len();
        let started = Instant::now();
        let outcome = self
            .prefill
            .prefill(PrefillRequest {
                request_id: rid.clone(),
                token_ids,
                temperature: 1.0,
                seed: None,
                ignore_eos: false,
                repetition_penalty: None,
                presence_penalty: None,
                decode_kv_host: self.decode_kv_host.to_string(),
                decode_ctrl_port: self.decode_ctrl_port,
            })
            .await?;
        let prefill_ms = millis(started);
        let Some(first_token) = outcome.first_token.or(outcome.stop_token) else {
            self.decode.release(&rid).await;
            return Err(backend_error(
                BackendError::Unknown,
                "startup probe: vLLM returned neither a first token nor a stop token",
            ));
        };
        let decoded = self
            .decode
            .decode_blocking(DecodeBody {
                rid: rid.clone(),
                first_token_id: first_token,
                max_tokens: PROBE_MAX_TOKENS,
                sampling: serde_json::Map::new(),
                timeout_s: self.kv_transfer_timeout.as_secs_f64(),
                stream: false,
            })
            .await;
        let (tokens, phases) = match decoded {
            Ok(decoded) => decoded,
            Err(error) if is_kind(&error, BackendError::ConnectionTimeout) => {
                return Err(connection_timeout(format!(
                    "startup probe: the prefill finished but no KV for request {rid} reached the \
                     decode server at {}:{} within {:?}. Check that vLLM runs TileRT's KV \
                     connector, that this address is reachable from the prefill node, and that \
                     nothing else is using this decode server.",
                    self.decode_kv_host, self.decode_ctrl_port, self.kv_transfer_timeout
                )));
            }
            Err(error) => return Err(error),
        };
        tracing::info!(
            prompt_tokens,
            prefill_ms,
            decoded_tokens = tokens.len(),
            phases = %phases.unwrap_or_default(),
            "TileRT startup probe passed"
        );
        Ok(())
    }
}

#[async_trait]
impl LLMEngine for TilertSidecarEngine {
    async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
        if self.prefill.initialized() {
            return Err(engine_shutdown(ALREADY_STARTED));
        }
        // Both engines answered during the bootstrap, so this is a reconnect
        // to a server that is already serving.
        let deadline = startup_deadline(self.transport.startup_deadline)?;
        let prefill = PrefillClient::connect(&self.grpc_endpoint, self.transport, deadline).await?;
        let vllm_health = prefill.health_channel();
        self.prefill
            .set(Arc::new(prefill))
            .map_err(|_| engine_shutdown(ALREADY_STARTED))?;

        let (stop, lost) = (self.shutdown.child_token(), self.lost.clone());
        let reason = Arc::clone(&self.lost_reason);
        let decode = self.decode.clone();
        let open_timeout = self.transport.connect_attempt_timeout;
        tokio::spawn(async move {
            let why = tokio::select! {
                _ = stop.cancelled() => return,
                why = prefill::watch_health(vllm_health, open_timeout) => format!("vLLM prefill: {why}"),
                why = decode.watch_health(DECODE_HEALTH_INTERVAL, DECODE_HEALTH_FAILURES) => {
                    format!("TileRT decode: {why}")
                }
            };
            let _ = reason.set(why);
            lost.cancel();
        });

        let mut runtime_data = HashMap::new();
        runtime_data.insert(
            "engine".to_string(),
            serde_json::Value::String("tilert".to_string()),
        );
        Ok(EngineConfig {
            model: self.model_path.clone(),
            served_model_name: self.served_model_name.clone(),
            model_aliases: Vec::new(),
            runtime_data,
            llm: Some(LlmRegistration {
                context_length: Some(self.context_length),
                max_num_seqs: Some(1),
                ..Default::default()
            }),
        })
    }

    async fn generate(
        &self,
        request: PreprocessedRequest,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        let prefill = self
            .prefill
            .get()
            .cloned()
            .ok_or_else(|| engine_shutdown("TileRT sidecar is not started"))?;
        let plan = RequestPlan::new(&request, self.context_length, self.capabilities)?;
        let token_ids = Arc::unwrap_or_clone(request.token_ids);
        let prompt_len = u32::try_from(token_ids.len()).unwrap_or(u32::MAX);
        let decode = self.decode.clone();
        let slot = self.slot.clone();
        let decode_kv_host = self.decode_kv_host.clone();
        let decode_ctrl_port = self.decode_ctrl_port;
        let kv_transfer_timeout = self.kv_transfer_timeout;
        let shutdown = self.shutdown.clone();
        let stopped_ctx = ctx.inner_arc();
        let dynamo_request_id = ctx.id().to_string();

        Ok(Box::pin(async_stream::stream! {
            let mut stopped = Box::pin(async move { stopped_ctx.stopped().await });
            let mut shutting_down = Box::pin(async move { shutdown.cancelled_owned().await });
            let queued = Instant::now();

            let permit = tokio::select! {
                biased;
                _ = &mut stopped => None,
                _ = &mut shutting_down => None,
                permit = slot.acquire_owned() => permit.ok(),
            };
            let Some(permit) = permit else {
                yield Ok(LLMEngineOutput::cancelled().with_usage(usage(prompt_len, 0)));
                return;
            };
            let queue_ms = millis(queued);

            let rid = new_request_id();
            let mut handoff = Handoff {
                prefill: prefill.clone(),
                decode: decode.clone(),
                rid: rid.clone(),
                pending: Pending::Prefill,
                permit: Some(permit),
            };
            tracing::debug!(
                request_id = %dynamo_request_id,
                rid,
                prompt_tokens = prompt_len,
                "TileRT request started"
            );
            let prefill_started = Instant::now();
            let prefill_call = prefill.prefill(plan.prefill_request(
                rid.clone(),
                token_ids,
                &decode_kv_host,
                decode_ctrl_port,
            ));
            let outcome = tokio::select! {
                biased;
                _ = &mut stopped => None,
                _ = &mut shutting_down => None,
                outcome = prefill_call => Some(outcome),
            };
            let outcome = match outcome {
                None => {
                    yield Ok(LLMEngineOutput::cancelled().with_usage(usage(prompt_len, 0)));
                    return;
                }
                Some(Err(error)) => {
                    // vLLM refuses an invalid request before it runs it.
                    handoff.pending = if is_kind(&error, BackendError::InvalidArgument) {
                        Pending::Nothing
                    } else {
                        Pending::Kv
                    };
                    yield Err(error);
                    return;
                }
                Some(Ok(outcome)) => outcome,
            };
            handoff.pending = Pending::Kv;
            let prefill_ms = millis(prefill_started);
            let prompt_tokens = outcome.prompt_tokens.unwrap_or(prompt_len);
            let cached_tokens = outcome.cached_tokens;

            // When the first token is EOS, vLLM returns no token. Handing the
            // decode server the stop token makes it finish at once and
            // release the KV.
            let Some(first_token) = outcome.first_token.or(outcome.stop_token) else {
                yield Err(backend_error(
                    BackendError::Unknown,
                    "vLLM finished the prefill without a first token or a stop token",
                ));
                return;
            };

            let decode_started = Instant::now();
            handoff.pending = Pending::Decode;
            let mut decode_call = Box::pin(decode.decode_stream(DecodeBody {
                rid: rid.clone(),
                first_token_id: first_token,
                max_tokens: plan.max_tokens,
                sampling: plan.decode_sampling(),
                timeout_s: kv_transfer_timeout.as_secs_f64(),
                stream: true,
            }));
            let opened = tokio::select! {
                biased;
                _ = &mut stopped => None,
                _ = &mut shutting_down => None,
                opened = &mut decode_call => Some(opened),
            };
            let Some(opened) = opened else {
                // The server can cancel a decode it has claimed, not one that
                // still waits for its KV, so give the claim a moment.
                match timeout(CLAIM_GRACE, &mut decode_call).await {
                    Ok(Ok(_events)) => {
                        decode.cancel(&rid).await;
                        handoff.pending = Pending::Nothing;
                    }
                    Ok(Err(error)) if !is_transport_failure(&error) => {
                        handoff.pending = Pending::Nothing;
                    }
                    Ok(Err(_)) | Err(_) => {}
                }
                yield Ok(LLMEngineOutput::cancelled()
                    .with_usage(request_usage(prompt_tokens, 0, cached_tokens)));
                return;
            };
            let events = match opened {
                Ok(events) => events,
                Err(error) => {
                    // A refusal means the server has let the KV go; a
                    // transport failure does not.
                    if !is_transport_failure(&error) {
                        handoff.pending = Pending::Nothing;
                    }
                    yield Err(error);
                    return;
                }
            };
            handoff.pending = Pending::Nothing;
            let mut events = Box::pin(events);

            let mut completion_tokens: u32 = 0;
            let mut first_line_ms = None;
            loop {
                let event = tokio::select! {
                    biased;
                    _ = &mut stopped => None,
                    _ = &mut shutting_down => None,
                    event = timeout(DECODE_IDLE_TIMEOUT, events.next()) => Some(event),
                };
                let Some(event) = event else {
                    decode.cancel(&rid).await;
                    yield Ok(LLMEngineOutput::cancelled()
                        .with_usage(request_usage(prompt_tokens, completion_tokens, cached_tokens)));
                    break;
                };
                match event {
                    Ok(Some(Ok(DecodeEvent::Tokens(token_ids)))) => {
                        if token_ids.is_empty() {
                            continue;
                        }
                        first_line_ms.get_or_insert_with(|| millis(decode_started));
                        completion_tokens = completion_tokens
                            .saturating_add(u32::try_from(token_ids.len()).unwrap_or(u32::MAX));
                        yield Ok(LLMEngineOutput {
                            token_ids,
                            ..Default::default()
                        });
                    }
                    Ok(Some(Ok(DecodeEvent::Done { finish, declared_tokens, phases }))) => {
                        if declared_tokens.is_some_and(|declared| declared != u64::from(completion_tokens)) {
                            // Not `StreamIncomplete`: every token was already
                            // delivered, so the request must not be retried.
                            yield Err(backend_error(
                                BackendError::Unknown,
                                format!(
                                    "TileRT decode declared {} tokens and sent {completion_tokens}",
                                    declared_tokens.unwrap_or_default()
                                ),
                            ));
                            break;
                        }
                        tracing::debug!(
                            request_id = %dynamo_request_id,
                            rid,
                            prompt_tokens,
                            cached_tokens = cached_tokens.unwrap_or_default(),
                            completion_tokens,
                            queue_ms,
                            prefill_ms,
                            first_line_ms = first_line_ms.unwrap_or_default(),
                            decode_ms = millis(decode_started),
                            finish = ?finish,
                            phases = %phases.unwrap_or_default(),
                            "TileRT request finished"
                        );
                        let terminal = match finish {
                            DecodeFinish::Stop => LLMEngineOutput::stop(),
                            DecodeFinish::Length => LLMEngineOutput::length(),
                            DecodeFinish::Cancelled => LLMEngineOutput::cancelled(),
                        };
                        yield Ok(terminal
                            .with_usage(request_usage(prompt_tokens, completion_tokens, cached_tokens)));
                        break;
                    }
                    Ok(Some(Err(error))) => {
                        yield Err(error);
                        break;
                    }
                    Ok(None) => {
                        yield Err(stream_incomplete(
                            "TileRT decode stream ended without a terminal message",
                        ));
                        break;
                    }
                    Err(_) => {
                        decode.cancel(&rid).await;
                        yield Err(backend_error(
                            BackendError::ResponseTimeout,
                            format!("TileRT decode sent nothing for {DECODE_IDLE_TIMEOUT:?}"),
                        ));
                        break;
                    }
                }
            }
        }))
    }

    async fn abort(&self, _ctx: Arc<dyn AsyncEngineContext>) {
        // Cancellation is handled inside the request stream, which knows which
        // engine the request is in.
    }

    async fn engine_lost(&self) -> DynamoError {
        self.lost.cancelled().await;
        let reason = self.lost_reason.get().map_or("unknown", String::as_str);
        engine_shutdown(format!("TileRT engine lost: {reason}"))
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        self.shutdown.cancel();
        // A request that ended early keeps the slot until its KV is released;
        // leaving sooner would strand that KV on the decode server.
        if timeout(CLEANUP_DRAIN, self.slot.acquire()).await.is_err() {
            tracing::warn!(
                "a request was still releasing its KV after {CLEANUP_DRAIN:?}; shutting down anyway"
            );
        }
        Ok(())
    }
}
