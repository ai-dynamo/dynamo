// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native encoder service discovery and the Dynamo encode terminal.

use std::sync::Arc;
use std::time::Duration;

use dynamo_backend_common::{
    DynamoError, EngineConfig, GenerateContext, LLMEngineOutput, PreprocessedRequest,
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
use crate::staged_media::{self, StagedMedia};

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
    fn validate(&self, request_id: &str, images: usize, videos: usize) -> Result<(), DynamoError> {
        let t = &self.transfer;
        let expected_version = if videos == 0 { 1 } else { 2 };
        if t.version != expected_version
            || request_id.is_empty()
            || t.request_id != request_id
            || !is_lower_hex(&t.transfer_id, 32)
            || t.part_id != format!("{}_local_part_0", t.transfer_id)
            || t.part_idx != 0
            || t.num_parts != 1
            || (images == 0 && videos == 0)
            || t.image_count as usize != images
            || t.images.len() != images
            || t.videos.len() != videos
            || !t.videos.iter().all(valid_video_layout)
            || t.images
                .iter()
                .any(|i| i.t == 0 || i.h == 0 || i.w == 0 || !is_lower_hex(&i.identity, 64))
        {
            return Err(client::invalid_request(
                "invalid SGLang encoder_result version, identity, part, or media layout",
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

fn valid_video_layout(layout: &pb::VideoLayout) -> bool {
    let frames = layout.t as usize;
    frames > 0
        && layout.h > 0
        && layout.w > 0
        && (layout.identities.len() == 1 || layout.identities.len() == frames)
        && layout
            .identities
            .iter()
            .all(|value| is_lower_hex(value, 64))
        && (layout.timestamps.is_empty() || layout.timestamps.len() == frames)
        && layout
            .timestamps
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
        && layout.timestamps.is_sorted()
        && layout
            .second_per_grid_ts
            .is_none_or(|value| value.is_finite() && value >= 0.0)
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

/// Count original source items, including one item per video regardless of its
/// temporal grid size. Each source list preserves its own placeholder order.
fn media_counts(request: &PreprocessedRequest) -> Result<(usize, usize), DynamoError> {
    let Some(media) = &request.multi_modal_data else {
        return Ok((0, 0));
    };
    if media
        .iter()
        .any(|(key, items)| !matches!(key.as_str(), "image_url" | "video_url") && !items.is_empty())
    {
        return Err(client::invalid_request(
            "staged SGLang encoders accept images and videos only",
        ));
    }
    Ok((
        media.get("image_url").map_or(0, Vec::len),
        media.get("video_url").map_or(0, Vec::len),
    ))
}

pub(crate) fn validate_consumer(
    request: &PreprocessedRequest,
    request_id: &str,
    required: bool,
    supports_staged_video: bool,
) -> Result<(), DynamoError> {
    if let Some(result) = &request.encoder_result {
        let handoff = Handoff::deserialize(result)
            .map_err(|e| client::invalid_arg(format!("invalid SGLang encoder_result: {e}")))?;
        // The native consumer counts placeholders in the original token IDs.
        // When Dynamo retains sources for retries, also check both modality counts.
        let (images, videos) = match &request.multi_modal_data {
            Some(_) => media_counts(request)?,
            None => (
                handoff.transfer.image_count as usize,
                handoff.transfer.videos.len(),
            ),
        };
        if videos > 0 && !supports_staged_video {
            return Err(client::invalid_request(
                "the selected SGLang language worker does not support staged video",
            ));
        }
        handoff.validate(request_id, images, videos)?;
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
            "multimodal request requires a successful Dynamo encoder_result",
        ));
    }
    Ok(())
}

pub(crate) fn stage_request(
    request: PreprocessedRequest,
    request_id: String,
    supports_video: bool,
) -> Result<pb::StageRequest, DynamoError> {
    let (images, videos) = media_counts(&request)?;
    if (images == 0 && videos == 0) || request.encoder_result.is_some() {
        return Err(client::invalid_request(
            "SGLang Stage requires unencoded media",
        ));
    }
    if videos > 0 && !supports_video {
        return Err(client::invalid_request(
            "the selected SGLang encoder does not support staged video",
        ));
    }
    protocol::validate_request(&request)?;
    if request.media_io_kwargs.is_some() {
        return Err(client::invalid_request(
            "SGLang Stage cannot apply media_io_kwargs",
        ));
    }
    let mut media = request.multi_modal_data.unwrap_or_default();
    let mut sources = |key: &str| {
        media
            .remove(key)
            .unwrap_or_default()
            .into_iter()
            .map(protocol::into_media_source)
            .collect::<Result<Vec<_>, _>>()
    };
    Ok(pb::StageRequest {
        request_id,
        mm_items: sources("image_url")?,
        video_urls: sources("video_url")?,
    })
}

pub(crate) struct Encoder {
    info: pb::GetEncoderInfoResponse,
    peer: String,
    pool: OnceCell<GrpcChannelPool>,
    media: Option<Arc<StagedMedia>>,
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
        let media = info
            .staged_media_policy
            .as_ref()
            .map(StagedMedia::new)
            .transpose()?
            .map(Arc::new);
        Ok(Self {
            info,
            media,
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
        let mut config = EngineConfig {
            model: self.info.model_path.clone(),
            served_model_name: self.served_name(),
            llm: Some(Default::default()),
            ..Default::default()
        };
        if self.info.supports_staged_video {
            config.runtime_data.insert(
                dynamo_llm::local_model::runtime_config::SGLANG_STAGED_VIDEO_V2.into(),
                true.into(),
            );
        }
        Ok(config)
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
        let request_id = ctx.id().to_string();
        let mut message =
            stage_request(request, request_id.clone(), self.info.supports_staged_video)?;
        let images = message.mm_items.len();
        let videos = message.video_urls.len();
        let mut rpc = self.rpc()?;
        let peer = self.peer.clone();
        let media = self.media.clone();
        Ok(Box::pin(async_stream::try_stream! {
            let response = tokio::select! {
                biased;
                _ = stop.cancelled() => None,
                _ = ctx.stopped() => None,
                res = async {
                    staged_media::prepare(&mut message, media.as_deref()).await?;
                    rpc.stage(message).await.map_err(|e| client::status_to_dynamo("Stage", e))
                } => Some(res),
            };
            let Some(response) = response else {
                yield LLMEngineOutput::cancelled();
                return;
            };
            let transfer = response?
                .into_inner().transfer.ok_or_else(|| client::protocol_error("Stage returned no transfer"))?;
            let handoff = Handoff { peer, transfer };
            if let Err(error) = handoff.validate(&request_id, images, videos) {
                let _ = timeout(Duration::from_secs(2), rpc.release(pb::ReleaseRequest {
                    transfer_id: handoff.transfer.transfer_id.clone(),
                })).await;
                Err(error)?;
            }
            tracing::debug!(
                peer = %handoff.peer,
                request_id = %request_id,
                part_id = %handoff.transfer.part_id,
                image_count = images,
                video_count = videos,
                "native SGLang encoder staged media"
            );
            let result = serde_json::to_value(&handoff).map_err(|e| client::protocol_error(e.to_string()))?;
            let serde_json::Value::Object(object) = result else {
                unreachable!("the encoder handoff is a struct")
            };
            yield LLMEngineOutput::encode_terminal(object);
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v1_handoff_json_is_unchanged_for_legacy_consumers() {
        let descriptor = serde_json::json!({
            "version": 1, "peer": "grpc://e:9100", "request_id": "r",
            "transfer_id":"10".repeat(16), "part_id":format!("{}_local_part_0","10".repeat(16)),
            "num_parts":1,"part_idx":0,"image_count":1,"images":[{"t":1,"h":4,"w":4,"identity":"c".repeat(64)}]
        });
        let handoff: Handoff = serde_json::from_value(descriptor.clone()).unwrap();
        handoff.validate("r", 1, 0).unwrap();
        assert_eq!(serde_json::to_value(handoff).unwrap(), descriptor);
    }
}
