// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    Arc, Edge, OnceLock, PipelineError, Service, Sink, Source, async_trait, private::Token,
};
use crate::pipeline::{PipelineIO, ServiceEngine};

mod base;
mod pipeline;
mod segment;

pub(crate) struct SinkEdge<Resp: PipelineIO> {
    edge: OnceLock<Edge<Resp>>,
}

pub struct ServiceBackend<Req: PipelineIO, Resp: PipelineIO> {
    engine: ServiceEngine<Req, Resp>,
    inner: SinkEdge<Resp>,
}

// todo - use a once lock of a TransportEngine
pub struct SegmentSink<Req: PipelineIO, Resp: PipelineIO> {
    engine: OnceLock<ServiceEngine<Req, Resp>>,
    inner: SinkEdge<Resp>,
}

#[allow(dead_code)]
pub struct EgressPort<Req: PipelineIO, Resp: PipelineIO> {
    engine: Service<Req, Resp>,
}

// impl<Resp: PipelineIO> SegmentSink<Req, Resp> {
//     pub connect(&self)
// }

// impl<Req, Resp> EgressPort<Req, Resp>
// where
//     Req: PipelineIO + Serialize,
//     Resp: for<'de> Deserialize<'de> + DataType,
// {
// }

// #[async_trait]
// impl<Req, Resp> AsyncEngine<Context<Req>, Annotated<Resp>> for EgressPort<Req, Resp>
// where
//     Req: PipelineIO + Serialize,
//     Resp: for<'de> Deserialize<'de> + DataType,
// {
//     async fn generate(&self, request: Context<Req>) -> Result<Resp, GenerateError> {
//         // when publish our request, we need to publish it with a subject
//         // we will use a trait in the future
//         let tx_subject = "tx-model-subject".to_string();

//         let rx_subject = "rx-model-subject".to_string();

//         // make a response channel
//         let (bytes_tx, bytes_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(16);

//         // register the bytes_tx sender with the response subject
//         // let bytes_stream = self.response_subscriber.register(rx_subject, bytes_tx);

//         // ask network impl for a Sender to the cancellation channel

//         let request = request
//             .try_map(|req| bincode::serialize(&req))
//             .map_err(|e| {
//                 GenerateError(format!(
//                     "Failed to serialize request in egress port: {}",
//                     e.to_string()
//                 ))
//             })?;

//         let (data, context) = request.transfer(());

//         let stream_ctx = Arc::new(StreamContext::from(context));

//         let shutdown_ctx = stream_ctx.clone();

//         let (live_tx, live_rx) = tokio::sync::oneshot::channel::<()>();

//         let byte_stream = ReceiverStream::new(bytes_rx);

//         let decoded = byte_stream
//             // decode the response
//             .map(move |item| {
//                 bincode::deserialize::<Annotated<Resp>>(&item)
//                     .expect("failed to deserialize response")
//             })
//             .scan(Some(live_tx), move |live_tx, item| {
//                 match item {
//                     Annotated::End => {
//                         // this essentially drops the channel
//                         let _ = live_tx.take();
//                     }
//                     _ => {}
//                 }
//                 futures::future::ready(Some(item))
//             });

//         return Ok(ResponseStream::new(Box::pin(decoded), stream_ctx));
//     }
// }
