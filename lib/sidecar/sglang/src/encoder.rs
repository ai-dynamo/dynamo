// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native encoder service discovery and the Dynamo encode terminal.

use std::sync::Arc;
use std::time::Duration;

use dynamo_backend_common::{
    DynamoError, EngineConfig, GenerateContext, LLMEngineOutput, PreprocessedRequest,
};
use dynamo_llm::kv_router::encoder_router::ENCODER_RELEASE;
use dynamo_llm::local_model::runtime_config::{
    ENCODER_HANDOFF_CAPABILITIES, ENCODER_HANDOFF_CONTRACT, ENCODER_HANDOFF_TRANSFER_MODE,
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

/// The same advertisement governs source admission and descriptor validation
/// on E and on every language worker. Audio uses its own contract so a visual
/// request can still use the older descriptors on an audio-capable worker.
#[derive(Default, Serialize)]
pub(crate) struct StagedCapabilities {
    pub(crate) images: bool,
    pub(crate) video: bool,
    pub(crate) native_model_type: String,
    pub(crate) audio_model_type: String,
    #[serde(skip)]
    pub(crate) handoff_key_id: String,
    #[serde(skip)]
    pub(crate) layout_id: String,
    // Only E and P/PD participate in embedding transfer.
    #[serde(skip)]
    pub(crate) transfer_backend: Option<&'static str>,
}

pub(crate) fn require_staged_layout_id(id: &str) -> Result<(), DynamoError> {
    if id.is_empty() {
        return Err(client::protocol_error(
            "staged SGLang worker must report its descriptor layout ID",
        ));
    }
    Ok(())
}

pub(crate) fn staged_transfer_backend(backend: &str) -> Result<&'static str, DynamoError> {
    match backend {
        "nixl" => Ok("nixl"),
        "zmq_to_scheduler" => Ok("zmq_to_scheduler"),
        _ => Err(client::protocol_error(format!(
            "staged SGLang worker must report its resolved encoder transfer backend (nixl or zmq_to_scheduler), got {backend:?}"
        ))),
    }
}

impl StagedCapabilities {
    pub(crate) fn advertise(&self, config: &mut EngineConfig) {
        config.runtime_data.insert(
            ENCODER_HANDOFF_CONTRACT.into(),
            serde_json::json!({ "sglang": { "layout_id": self.layout_id, "handoff_key_id": self.handoff_key_id } }),
        );
        config
            .runtime_data
            .insert(ENCODER_HANDOFF_CAPABILITIES.into(), serde_json::json!(self));
        if let Some(backend) = self.transfer_backend {
            config
                .runtime_data
                .insert(ENCODER_HANDOFF_TRANSFER_MODE.into(), backend.into());
        }
    }

    fn model_for(&self, [images, videos, audios]: [usize; 3]) -> Result<&str, DynamoError> {
        if images > 0 && !self.images {
            return Err(client::invalid_request(
                "the selected SGLang worker does not support staged images",
            ));
        }
        if videos > 0 && !self.video {
            return Err(client::invalid_request(
                "the selected SGLang worker does not support staged video",
            ));
        }
        if audios == 0 {
            return Ok(&self.native_model_type);
        }
        if self.audio_model_type.is_empty() {
            return Err(client::invalid_request(
                "the selected SGLang worker does not support staged audio",
            ));
        }
        Ok(&self.audio_model_type)
    }
}

#[derive(Deserialize, Serialize)]
struct Handoff {
    peer: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    instance_id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    signature: String,
    #[serde(flatten)]
    transfer: pb::StagedImageTransfer,
}

impl Handoff {
    fn validate(
        &self,
        request_id: &str,
        [images, videos, audios]: [usize; 3],
        native_model_type: &str,
    ) -> Result<(), DynamoError> {
        let t = &self.transfer;
        if !matches_layout_contract(t, videos, audios, native_model_type)
            || request_id.is_empty()
            || t.request_id != request_id
            || !is_lower_hex(&t.transfer_id, 32)
            || t.part_id != format!("{}_local_part_0", t.transfer_id)
            || t.part_idx != 0
            || t.num_parts != 1
            || (images == 0 && videos == 0 && audios == 0)
            || t.image_count as usize != images
            || t.images.len() != images
            || t.videos.len() != videos
            || t.audios.len() != audios
            || !t.videos.iter().all(|v| valid_video_layout(v, t.version))
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

// E and the selected language worker both validate against their own native
// service's advertisement. An absent model type means that peer speaks v1/v2.
fn matches_layout_contract(
    transfer: &pb::StagedImageTransfer,
    videos: usize,
    audios: usize,
    native_model_type: &str,
) -> bool {
    if transfer.native_model_type != native_model_type {
        return false;
    }
    if native_model_type.is_empty() {
        return audios == 0
            && transfer.version == (if videos == 0 { 1 } else { 2 })
            && transfer.embedding_width == 0
            && transfer
                .images
                .iter()
                .all(|layout| layout.embedding_rows == 0 && layout.original_size.is_empty())
            && transfer
                .videos
                .iter()
                .all(|layout| layout.embedding_rows == 0);
    }
    transfer.version == (if audios == 0 { 3 } else { 4 })
        && transfer.audios.iter().all(|layout| {
            is_lower_hex(&layout.identity, 64)
                && layout.feature_length > 0
                && layout.embedding_rows > 0
                && layout.embedding_width > 0
                && layout.embedding_width <= transfer.embedding_width
                && (!transfer.images.is_empty()
                    || !transfer.videos.is_empty()
                    || layout.embedding_width == transfer.embedding_width)
        })
        && transfer.embedding_width > 0
        && transfer.images.iter().all(|layout| {
            layout.embedding_rows > 0
                && (layout.original_size.is_empty()
                    || matches!(layout.original_size.as_slice(), [w, h] if *w > 0 && *h > 0))
        })
        && transfer
            .videos
            .iter()
            .all(|layout| layout.embedding_rows > 0)
}

fn valid_video_layout(layout: &pb::VideoLayout, version: u32) -> bool {
    let temporal_grid = layout.t as usize;
    // The v3 processor may compress several temporal patches into one frame
    // embedding. Its timestamps and per-frame identities refer to those groups.
    let groups = if version >= 3 && !layout.timestamps.is_empty() {
        layout.timestamps.len()
    } else {
        temporal_grid
    };
    groups > 0
        && temporal_grid > 0
        && temporal_grid.is_multiple_of(groups)
        && layout.h > 0
        && layout.w > 0
        && (layout.identities.len() == 1 || layout.identities.len() == groups)
        && layout
            .identities
            .iter()
            .all(|value| is_lower_hex(value, 64))
        && (layout.timestamps.is_empty() || layout.timestamps.len() == groups)
        && layout
            .timestamps
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
        && layout.timestamps.is_sorted()
        && layout
            .second_per_grid_ts
            .is_none_or(|value| value.is_finite() && value > 0.0)
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
fn media_counts(request: &PreprocessedRequest) -> Result<[usize; 3], DynamoError> {
    let Some(media) = &request.multi_modal_data else {
        return Ok([0; 3]);
    };
    if media.iter().any(|(key, items)| {
        !matches!(key.as_str(), "image_url" | "video_url" | "audio_url") && !items.is_empty()
    }) {
        return Err(client::invalid_request(
            "staged SGLang encoders require image, video, or audio URLs",
        ));
    }
    Ok(["image_url", "video_url", "audio_url"].map(|key| media.get(key).map_or(0, Vec::len)))
}

pub(crate) fn validate_consumer(
    request: &PreprocessedRequest,
    request_id: &str,
    required: bool,
    capabilities: &StagedCapabilities,
) -> Result<(), DynamoError> {
    if let Some(result) = &request.encoder_result {
        let handoff = Handoff::deserialize(result)
            .map_err(|e| client::invalid_arg(format!("invalid SGLang encoder_result: {e}")))?;
        if !capabilities.handoff_key_id.is_empty()
            && !(is_lower_hex(&handoff.signature, 64) && is_lower_hex(&handoff.instance_id, 32))
        {
            return Err(client::invalid_request(
                "missing or malformed signed encoder receipt",
            ));
        }
        // The native consumer counts placeholders in the original token IDs.
        // When Dynamo retains sources for retries, also check each modality's count.
        let counts = match &request.multi_modal_data {
            Some(_) => media_counts(request)?,
            None => [
                handoff.transfer.image_count as usize,
                handoff.transfer.videos.len(),
                handoff.transfer.audios.len(),
            ],
        };
        handoff.validate(request_id, counts, capabilities.model_for(counts)?)?;
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
        // A caller that bypasses the encoder router still needs a staged result.
        capabilities.model_for(media_counts(request)?)?;
        for source in request
            .multi_modal_data
            .iter()
            .flat_map(|m| m.values())
            .flatten()
        {
            protocol::media_source(source)?;
        }
        return Err(client::cannot_connect(
            "multimodal request requires a successful Dynamo encoder_result",
        ));
    }
    Ok(())
}

pub(crate) fn stage_request(
    request: PreprocessedRequest,
    request_id: String,
    capabilities: &StagedCapabilities,
) -> Result<pb::StageRequest, DynamoError> {
    let counts = media_counts(&request)?;
    if counts == [0; 3] || request.encoder_result.is_some() {
        return Err(client::invalid_request(
            "SGLang Stage requires unencoded media",
        ));
    }
    capabilities.model_for(counts)?;
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
        audio_urls: sources("audio_url")?,
    })
}

pub(crate) struct Encoder {
    info: pb::GetEncoderInfoResponse,
    capabilities: StagedCapabilities,
    peer: String,
    pool: OnceCell<GrpcChannelPool>,
    stage_timeout: Duration,
    media: Option<Arc<StagedMedia>>,
}

impl Encoder {
    pub(crate) fn new(
        info: pb::GetEncoderInfoResponse,
        endpoint: &GrpcEndpoint,
        advertised: Option<&str>,
        stage_timeout: Duration,
    ) -> Result<Self, DynamoError> {
        if info.model_path.trim().is_empty() {
            return Err(client::protocol_error(
                "GetEncoderInfo returned an empty model_path",
            ));
        }
        if info.instance_id.len() != 16 {
            return Err(client::protocol_error(
                "GetEncoderInfo must identify the staged encoder with a 16-byte instance_id",
            ));
        }
        require_staged_layout_id(&info.staged_layout_id)?;
        let peer = if info.handoff_key_id.is_empty() {
            advertised_peer(advertised.unwrap_or(endpoint.as_str()))?
        } else {
            let peer = advertised_peer(&info.handoff_peer)?;
            if !is_lower_hex(&info.handoff_key_id, 64)
                || peer != info.handoff_peer
                || advertised
                    .map(advertised_peer)
                    .transpose()?
                    .is_some_and(|explicit| explicit != peer)
            {
                return Err(client::invalid_arg(
                    "encoder handoff key or advertised signing endpoint mismatch",
                ));
            }
            peer
        };
        let media = info
            .staged_media_policy
            .as_ref()
            .map(StagedMedia::new)
            .transpose()?
            .map(Arc::new);
        Ok(Self {
            capabilities: StagedCapabilities {
                images: info.supports_staged_images.unwrap_or(true),
                video: info.supports_staged_video,
                native_model_type: info.native_model_type.clone(),
                audio_model_type: info.staged_audio_model_type.clone(),
                handoff_key_id: info.handoff_key_id.clone(),
                layout_id: info.staged_layout_id.clone(),
                transfer_backend: Some(staged_transfer_backend(&info.staged_transfer_backend)?),
            },
            info,
            media,
            peer,
            pool: OnceCell::new(),
            stage_timeout,
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
        self.capabilities.advertise(&mut config);
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
        if let Some(original_id) = ctx.metadata().get(ENCODER_RELEASE) {
            let handoff = request
                .encoder_result
                .as_ref()
                .and_then(|value| Handoff::deserialize(value).ok())
                .ok_or_else(|| {
                    client::invalid_request("encoder release requires a staged result")
                })?;
            // The original worker and its existing native channel are the only
            // release targets. Never dial a peer supplied in the receipt.
            if original_id.is_empty()
                || handoff.transfer.request_id != *original_id
                || handoff.peer != self.peer
                || !is_lower_hex(&handoff.transfer.transfer_id, 32)
                || media_counts(&request)? != [0; 3]
                || !request.token_ids.is_empty()
                || (!self.capabilities.handoff_key_id.is_empty()
                    && handoff.instance_id
                        != self
                            .info
                            .instance_id
                            .iter()
                            .map(|b| format!("{b:02x}"))
                            .collect::<String>())
            {
                return Err(client::invalid_request(
                    "encoder release does not match its original stage",
                ));
            }
            let mut rpc = self.rpc()?;
            let transfer_id = handoff.transfer.transfer_id;
            let expected = self.info.instance_id.clone();
            return Ok(Box::pin(async_stream::try_stream! {
                let ack = client::rpc_with_deadline("Release", Instant::now() + Duration::from_secs(2), rpc.release(pb::ReleaseRequest { transfer_id })).await?;
                if ack.into_inner().instance_id != expected {
                    Err(client::protocol_error("Release came from a different encoder instance"))?;
                }
                yield LLMEngineOutput::stop();
            }));
        }
        let request_id = ctx.id().to_string();
        let mut message = stage_request(request, request_id.clone(), &self.capabilities)?;
        let images = message.mm_items.len();
        let videos = message.video_urls.len();
        let audios = message.audio_urls.len();
        let counts = [images, videos, audios];
        let mut rpc = self.rpc()?;
        let peer = self.peer.clone();
        let instance_id = self.info.instance_id.clone();
        let authenticated = !self.capabilities.handoff_key_id.is_empty();
        let native_model_type = self.capabilities.model_for(counts)?.to_owned();
        let media = self.media.clone();
        let stage_timeout = self.stage_timeout;
        Ok(Box::pin(async_stream::try_stream! {
            let response = tokio::select! {
                biased;
                _ = stop.cancelled() => None,
                _ = ctx.stopped() => None,
                res = async {
                    staged_media::prepare(&mut message, media.as_deref()).await?;
                    let mut request = tonic::Request::new(message);
                    request.set_timeout(stage_timeout);
                    client::rpc_with_deadline("Stage", Instant::now() + stage_timeout, rpc.stage(request)).await
                } => Some(res),
            };
            let Some(response) = response else {
                yield LLMEngineOutput::cancelled();
                return;
            };
            let response = response?.into_inner();
            let transfer = response.transfer.ok_or_else(|| client::invalid_stage_reply("Stage returned no transfer"))?;
            let handoff = Handoff {
                peer, transfer, signature: response.handoff_mac,
                instance_id: if authenticated {
                    instance_id.iter().map(|b| format!("{b:02x}")).collect()
                } else { String::new() },
            };
            let validated = (|| {
                if response.instance_id != instance_id {
                    return Err(client::invalid_stage_reply("Stage came from a different encoder instance"));
                }
                if authenticated && !is_lower_hex(&handoff.signature, 64) {
                    return Err(client::invalid_stage_reply("Stage returned no valid encoder handoff MAC"));
                }
                handoff.validate(&request_id, counts, &native_model_type).map_err(|e| {
                    client::invalid_stage_reply(format!("Stage returned an invalid handoff: {e}"))
                })?;
                serde_json::to_value(&handoff).map_err(|e| client::protocol_error(e.to_string()))
            })();
            if validated.is_err() || stop.is_cancelled() || ctx.is_stopped() {
                let _ = timeout(Duration::from_secs(2), rpc.release(pb::ReleaseRequest {
                    transfer_id: handoff.transfer.transfer_id.clone(),
                })).await;
                if stop.is_cancelled() || ctx.is_stopped() {
                    yield LLMEngineOutput::cancelled();
                    return;
                }
            }
            let result = validated?;
            tracing::debug!(
                peer = %handoff.peer,
                request_id = %request_id,
                part_id = %handoff.transfer.part_id,
                image_count = images,
                video_count = videos,
                audio_count = audios,
                "native SGLang encoder staged media"
            );
            let serde_json::Value::Object(object) = result else {
                unreachable!("the encoder handoff is a struct")
            };
            let mut terminal = LLMEngineOutput::encode_terminal(object);
            terminal.extra_args = Some(Box::new(serde_json::json!({ (ENCODER_RELEASE): true })));
            yield terminal;
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_registration_requires_an_instance_id() {
        let endpoint = GrpcEndpoint::parse("127.0.0.1:9100", "encoder").unwrap();
        for size in [0, 15, 17] {
            let info = pb::GetEncoderInfoResponse {
                model_path: "vision".into(),
                instance_id: vec![1; size],
                ..Default::default()
            };
            assert!(Encoder::new(info, &endpoint, None, Duration::from_secs(1)).is_err());
        }
    }

    #[test]
    fn signed_encoder_advertises_its_peer_and_key_in_the_worker_contract() {
        let local = GrpcEndpoint::parse("127.0.0.1:9100", "encoder").unwrap();
        let key = "ab".repeat(32);
        let info = pb::GetEncoderInfoResponse {
            model_path: "vision".into(),
            instance_id: vec![1; 16],
            handoff_peer: "grpc://worker:9100".into(),
            handoff_key_id: key.clone(),
            staged_transfer_backend: "nixl".into(),
            staged_layout_id: "vision-layout".into(),
            ..Default::default()
        };
        let connect =
            |info, advertised| Encoder::new(info, &local, advertised, Duration::from_secs(1));
        let encoder = connect(info.clone(), Some("worker:9100")).unwrap();
        assert_eq!(encoder.peer, "grpc://worker:9100");
        let mut config = EngineConfig::default();
        encoder.capabilities.advertise(&mut config);
        assert_eq!(
            config.runtime_data[ENCODER_HANDOFF_CONTRACT]["sglang"]["handoff_key_id"],
            key
        );
        assert_eq!(config.runtime_data[ENCODER_HANDOFF_TRANSFER_MODE], "nixl");
        assert!(
            connect(
                pb::GetEncoderInfoResponse {
                    staged_layout_id: String::new(),
                    ..info.clone()
                },
                None
            )
            .is_err()
        );
        let video_encoder = connect(
            pb::GetEncoderInfoResponse {
                supports_staged_video: true,
                ..info.clone()
            },
            None,
        )
        .unwrap();
        let mut with_video = EngineConfig::default();
        video_encoder.capabilities.advertise(&mut with_video);
        assert_eq!(
            config.runtime_data[ENCODER_HANDOFF_CONTRACT],
            with_video.runtime_data[ENCODER_HANDOFF_CONTRACT]
        );
        assert_ne!(
            config.runtime_data[ENCODER_HANDOFF_CAPABILITIES],
            with_video.runtime_data[ENCODER_HANDOFF_CAPABILITIES]
        );
        assert!(encoder.capabilities.model_for([1, 0, 0]).is_ok());
        assert!(encoder.capabilities.model_for([0, 1, 0]).is_err());
        for backend in ["", "auto", "zmq_to_tokenizer"] {
            let incompatible = pb::GetEncoderInfoResponse {
                staged_transfer_backend: backend.into(),
                ..info.clone()
            };
            assert!(connect(incompatible, None).is_err());
        }
        assert!(connect(info.clone(), Some("other:9100")).is_err());
        assert!(
            connect(
                pb::GetEncoderInfoResponse {
                    handoff_key_id: "bad".into(),
                    ..info
                },
                None
            )
            .is_err()
        );
    }

    #[test]
    fn v1_handoff_json_is_unchanged_for_legacy_consumers() {
        let descriptor = serde_json::json!({
            "version": 1, "peer": "grpc://e:9100", "request_id": "r",
            "transfer_id":"10".repeat(16), "part_id":format!("{}_local_part_0","10".repeat(16)),
            "num_parts":1,"part_idx":0,"image_count":1,"images":[{"t":1,"h":4,"w":4,"identity":"c".repeat(64)}]
        });
        let handoff: Handoff = serde_json::from_value(descriptor.clone()).unwrap();
        handoff.validate("r", [1, 0, 0], "").unwrap();
        assert_eq!(serde_json::to_value(handoff).unwrap(), descriptor);
    }

    #[test]
    fn native_handoff_checks_the_advertised_model_and_item_shapes() {
        let key = "10".repeat(16);
        let descriptor = serde_json::json!({
            "version": 3, "peer": "grpc://e:9100", "request_id": "r",
            "transfer_id": key, "part_id": format!("{key}_local_part_0"),
            "part_idx": 0, "num_parts": 1, "image_count": 1,
            "native_model_type": "kimi_vl", "embedding_width": 1536,
            "images": [{
                "t": 1, "h": 4, "w": 6, "identity": "bb".repeat(32),
                "embedding_rows": 6, "original_size": [96, 64]
            }]
        });
        let handoff: Handoff = serde_json::from_value(descriptor.clone()).unwrap();
        handoff.validate("r", [1, 0, 0], "kimi_vl").unwrap();
        assert_eq!(serde_json::to_value(&handoff).unwrap(), descriptor);
        assert!(handoff.validate("r", [1, 0, 0], "").is_err());
        assert!(handoff.validate("r", [1, 0, 0], "glm4v").is_err());
        for (field, value) in [
            ("/version", serde_json::json!(1)),
            ("/embedding_width", serde_json::json!(0)),
            ("/images/0/embedding_rows", serde_json::json!(0)),
            ("/images/0/original_size", serde_json::json!([96])),
            ("/images/0/original_size", serde_json::json!([0, 64])),
        ] {
            let mut malformed = descriptor.clone();
            *malformed.pointer_mut(field).unwrap() = value;
            let handoff: Handoff = serde_json::from_value(malformed).unwrap();
            assert!(
                handoff.validate("r", [1, 0, 0], "kimi_vl").is_err(),
                "{field}"
            );
        }
    }
}
