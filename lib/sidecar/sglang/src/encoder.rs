// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native encoder service discovery and the Dynamo encode terminal.

use std::time::Duration;

use dynamo_backend_common::{
    DynamoError, EngineConfig, GenerateContext, LLMEngineOutput, MultimodalData,
    PreprocessedRequest,
};
use dynamo_sidecar_common::{
    DEFAULT_MAX_GRPC_MESSAGE_SIZE, GrpcChannelPool, GrpcEndpoint, GrpcTransportConfig,
};
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};
use tokio::sync::OnceCell;
use tokio::time::{Instant, timeout, timeout_at};
use tokio_util::sync::CancellationToken;
use tonic::transport::Channel;

use crate::client;
use crate::discovery::EngineInfo;
use crate::proto::encoder as pb;
use crate::protocol;

type EncoderClient = pb::sglang_encoder_client::SglangEncoderClient<Channel>;

pub(crate) enum NativeBackend {
    Language(Box<EngineInfo>),
    Encoder(pb::GetEncoderInfoResponse),
}

impl From<EngineInfo> for NativeBackend {
    fn from(value: EngineInfo) -> Self {
        Self::Language(Box::new(value))
    }
}

pub(crate) fn client(channel: Channel) -> EncoderClient {
    EncoderClient::new(channel)
        .max_encoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
        .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
}

/// UNIMPLEMENTED identifies an endpoint without the staged encoder service.
/// Any other failure is an encoder discovery error, including invalid topology.
pub(crate) async fn discover(
    rpc: &mut EncoderClient,
    deadline: Instant,
) -> Result<Option<pb::GetEncoderInfoResponse>, DynamoError> {
    let result = timeout_at(deadline, rpc.get_encoder_info(pb::GetEncoderInfoRequest {}))
        .await
        .map_err(|_| client::connection_timeout("GetEncoderInfo timed out"))?;
    match result {
        Ok(info) => Ok(Some(info.into_inner())),
        Err(status) if status.code() == tonic::Code::Unimplemented => Ok(None),
        Err(status) => Err(client::status_to_dynamo("GetEncoderInfo", status)),
    }
}

#[derive(Deserialize, Serialize)]
struct Handoff {
    peer: String,
    #[serde(flatten)]
    transfer: pb::StagedImageTransfer,
}

impl Handoff {
    fn validate(&self, request_id: &str, count: usize) -> Result<(), DynamoError> {
        let t = &self.transfer;
        if t.version != 1
            || request_id.is_empty()
            || t.request_id != request_id
            || !is_lower_hex(&t.transfer_id, 32)
            || t.part_id != format!("{}_local_part_0", t.transfer_id)
            || t.part_idx != 0
            || t.num_parts != 1
            || count == 0
            || t.image_count as usize != count
            || t.images.len() != count
            || t.images
                .iter()
                .any(|i| i.t == 0 || i.h == 0 || i.w == 0 || !is_lower_hex(&i.identity, 64))
        {
            return Err(client::invalid_request(
                "invalid SGLang encoder_result version, identity, part, or image layout",
            ));
        }
        if advertised_peer(&self.peer)? != self.peer {
            return Err(client::invalid_request(
                "invalid encoder peer in encoder_result",
            ));
        }
        Ok(())
    }
}

fn is_lower_hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn advertised_peer(raw: &str) -> Result<String, DynamoError> {
    let endpoint = GrpcEndpoint::parse(raw, "--encoder-advertise-endpoint")?;
    if !endpoint.as_str().starts_with("http://") {
        return Err(client::invalid_arg(
            "an encoder must advertise a TCP endpoint reachable by its consumers",
        ));
    }
    if matches!(endpoint.authority_host(), "0.0.0.0" | "[::]" | "*") {
        return Err(client::invalid_arg(
            "cannot advertise a wildcard encoder address",
        ));
    }
    let port = url::Url::parse(endpoint.as_str())
        .ok()
        .and_then(|u| u.port_or_known_default())
        .filter(|port| *port != 0)
        .ok_or_else(|| client::invalid_arg("invalid encoder port"))?;
    // GrpcEndpoint normalizes the default HTTP port away. The native Python
    // and Rust gRPC clients expect an explicit host:port on the transfer.
    Ok(format!("grpc://{}:{port}", endpoint.authority_host()))
}

fn images(request: &PreprocessedRequest) -> Result<&[MultimodalData], DynamoError> {
    let Some(media) = &request.multi_modal_data else {
        return Ok(&[]);
    };
    if media
        .iter()
        .any(|(key, items)| key != "image_url" && !items.is_empty())
    {
        return Err(client::invalid_request(
            "staged SGLang encoders accept images only",
        ));
    }
    Ok(media.get("image_url").map_or(&[], Vec::as_slice))
}

pub(crate) fn validate_consumer(
    request: &PreprocessedRequest,
    request_id: &str,
    required: bool,
) -> Result<(), DynamoError> {
    if let Some(result) = &request.encoder_result {
        let handoff = Handoff::deserialize(result)
            .map_err(|e| client::invalid_arg(format!("invalid SGLang encoder_result: {e}")))?;
        // The native consumer counts placeholders in the original token IDs.
        // Raw images need not accompany the descriptor. If Dynamo retained
        // the media for routing/retries, check the two counts agree here too.
        let count = match &request.multi_modal_data {
            Some(_) => images(request)?.len(),
            None => handoff.transfer.image_count as usize,
        };
        handoff.validate(request_id, count)?;
    } else if required
        && (request
            .multi_modal_data
            .iter()
            .flat_map(|m| m.values())
            .any(|v| !v.is_empty())
            || request
                .extra_args
                .as_ref()
                .and_then(|v| v.get(dynamo_sglang_generate::PAYLOAD_KEY))
                .is_some_and(|body| {
                    ["image_data", "video_data", "audio_data"]
                        .iter()
                        .any(|key| {
                            body.get(*key).is_some_and(|v| {
                                !v.is_null() && v.as_array().is_none_or(|v| !v.is_empty())
                            })
                        })
                }))
    {
        // EncoderRouter also reaches here when no E is live or its RPC failed.
        // Under the required topology this must never issue a second encode.
        return Err(client::invalid_request(
            "image request requires a successful Dynamo encoder_result",
        ));
    }
    Ok(())
}

pub(crate) struct Encoder {
    info: pb::GetEncoderInfoResponse,
    peer: String,
    pool: OnceCell<GrpcChannelPool>,
}

impl Encoder {
    pub(crate) fn new(
        info: pb::GetEncoderInfoResponse,
        endpoint: &GrpcEndpoint,
        advertised: Option<&str>,
    ) -> Result<Self, DynamoError> {
        if info.model_path.trim().is_empty() {
            return Err(client::protocol_error(
                "GetEncoderInfo returned an empty model_path",
            ));
        }
        Ok(Self {
            info,
            peer: advertised_peer(advertised.unwrap_or(endpoint.as_str()))?,
            pool: OnceCell::new(),
        })
    }

    pub(crate) fn tokenizer(&self) -> &str {
        if self.info.tokenizer_path.is_empty() {
            &self.info.model_path
        } else {
            &self.info.tokenizer_path
        }
    }

    pub(crate) fn served_name(&self) -> Option<String> {
        (!self.info.served_model_name.is_empty()).then(|| self.info.served_model_name.clone())
    }

    pub(crate) async fn start(
        &self,
        endpoint: &GrpcEndpoint,
        transport: &GrpcTransportConfig,
    ) -> Result<EngineConfig, DynamoError> {
        let deadline = Instant::now() + transport.startup_deadline;
        let pool = client::connect_pool(endpoint, *transport, deadline, false).await?;
        let mut rpc = client(pool.next_channel());
        let info = discover(&mut rpc, deadline).await?;
        if info.as_ref() != Some(&self.info) {
            return Err(client::invalid_arg(
                "native encoder identity changed during startup",
            ));
        }
        self.pool
            .set(pool)
            .map_err(|_| client::engine_shutdown("encoder already started"))?;
        Ok(EngineConfig {
            model: self.info.model_path.clone(),
            served_model_name: self.served_name(),
            llm: Some(Default::default()),
            ..Default::default()
        })
    }

    fn rpc(&self) -> Result<EncoderClient, DynamoError> {
        self.pool
            .get()
            .map(|pool| client(pool.next_channel()))
            .ok_or_else(|| client::engine_shutdown("encoder has not started"))
    }

    pub(crate) fn watch_serving(
        &self,
        attempt_timeout: Duration,
    ) -> Result<impl Future<Output = String> + Send + 'static, DynamoError> {
        let rpc = self.rpc()?;
        let expected = self.info.clone();
        Ok(client::watch_metadata(
            "GetEncoderInfo",
            attempt_timeout,
            move || {
                let mut rpc = rpc.clone();
                let expected = expected.clone();
                async move {
                    let reply = rpc.get_encoder_info(pb::GetEncoderInfoRequest {}).await?;
                    if reply.into_inner() == expected {
                        Ok(())
                    } else {
                        Err(tonic::Status::failed_precondition(
                            "native encoder identity changed",
                        ))
                    }
                }
            },
        ))
    }

    pub(crate) fn stage(
        &self,
        request: PreprocessedRequest,
        ctx: GenerateContext,
        stop: CancellationToken,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        if images(&request)?.is_empty() || request.encoder_result.is_some() {
            return Err(client::invalid_request(
                "SGLang Stage requires unencoded images",
            ));
        }
        protocol::validate_request(&request)?;
        if request.media_io_kwargs.is_some() {
            return Err(client::invalid_request(
                "SGLang Stage cannot apply media_io_kwargs",
            ));
        }
        let mm_items = request
            .multi_modal_data
            .and_then(|mut media| media.remove("image_url"))
            .unwrap_or_default()
            .into_iter()
            .map(protocol::into_media_source)
            .collect::<Result<Vec<_>, _>>()?;
        let count = mm_items.len();
        let request_id = ctx.id().to_string();
        let message = pb::StageRequest {
            request_id: request_id.clone(),
            mm_items,
        };
        let mut rpc = self.rpc()?;
        let peer = self.peer.clone();
        Ok(Box::pin(async_stream::try_stream! {
            let response = tokio::select! {
                biased;
                _ = stop.cancelled() => None,
                _ = ctx.stopped() => None,
                res = rpc.stage(message) => Some(res),
            };
            let Some(response) = response else {
                yield LLMEngineOutput::cancelled();
                return;
            };
            let transfer = response.map_err(|e| client::status_to_dynamo("Stage", e))?
                .into_inner().transfer.ok_or_else(|| client::protocol_error("Stage returned no transfer"))?;
            let handoff = Handoff { peer, transfer };
            if let Err(error) = handoff.validate(&request_id, count) {
                let _ = timeout(Duration::from_secs(2), rpc.release(pb::ReleaseRequest {
                    transfer_id: handoff.transfer.transfer_id.clone(),
                })).await;
                Err(error)?;
            }
            tracing::debug!(
                peer = %handoff.peer,
                request_id = %request_id,
                part_id = %handoff.transfer.part_id,
                image_count = count,
                "native SGLang encoder staged images"
            );
            let result = serde_json::to_value(&handoff).map_err(|e| client::protocol_error(e.to_string()))?;
            let serde_json::Value::Object(object) = result else {
                unreachable!("the encoder handoff is a struct")
            };
            yield LLMEngineOutput::encode_terminal(object);
        }))
    }
}
