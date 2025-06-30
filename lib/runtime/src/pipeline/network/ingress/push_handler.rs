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

use super::*;
use serde::{Deserialize, Serialize};

#[async_trait]
impl<T: Data, U: Data> PushWorkHandler for Ingress<SingleIn<T>, ManyOut<U>>
where
    T: Data + for<'de> Deserialize<'de> + std::fmt::Debug,
    U: Data + Serialize + std::fmt::Debug,
{
    async fn handle_payload(&self, payload: Bytes) -> Result<(), PipelineError> {
        // decode the control message and the request
        let msg = TwoPartCodec::default()
            .decode_message(payload)?
            .into_message_type();

        // we must have a header and a body
        // it will be held by this closure as a Some(permit)
        let (control_msg, request) = match msg {
            TwoPartMessageType::HeaderAndData(header, data) => {
                tracing::trace!(
                    "received two part message with ctrl: {} bytes, data: {} bytes",
                    header.len(),
                    data.len()
                );
                let control_msg: RequestControlMessage = match serde_json::from_slice(&header) {
                    Ok(cm) => cm,
                    Err(err) => {
                        let json_str = String::from_utf8_lossy(&header);
                        return Err(PipelineError::DeserializationError(
                            format!("Failed deserializing to RequestControlMessage. err={err}, json_str={json_str}"),
                        ));
                    }
                };
                let request: T = serde_json::from_slice(&data)?;
                (control_msg, request)
            }
            _ => {
                return Err(PipelineError::Generic(String::from("Unexpected message from work queue; unable extract a TwoPartMessage with a header and data")));
            }
        };

        // extend request with context
        tracing::trace!("received control message: {:?}", control_msg);
        tracing::trace!("received request: {:?}", request);
        let request: context::Context<T> = Context::with_id(request, control_msg.id);

        // todo - eventually have a handler class which will returned an abstracted object, but for now,
        // we only support tcp here, so we can just unwrap the connection info
        tracing::trace!("creating tcp response stream");
        let mut publisher = tcp::client::TcpClient::create_response_stream(
            request.context(),
            control_msg.connection_info,
        )
        .await
        .map_err(|e| {
            PipelineError::Generic(format!("Failed to create response stream: {:?}", e,))
        })?;

        tracing::trace!("calling generate");
        let stream = self
            .segment
            .get()
            .expect("segment not set")
            .generate(request)
            .await
            .map_err(PipelineError::GenerateError);

        // the prolouge is sent to the client to indicate that the stream is ready to receive data
        // or if the generate call failed, the error is sent to the client
        let mut stream = match stream {
            Ok(stream) => {
                tracing::trace!("Successfully generated response stream; sending prologue");
                let _result = publisher.send_prologue(None).await;
                stream
            }
            Err(e) => {
                tracing::error!("Failed to generate response stream: {:?}", e);
                let _result = publisher.send_prologue(Some(e.to_string())).await;
                Err(e)?
            }
        };

        let context = stream.context();

        while let Some(resp) = stream.next().await {
            tracing::trace!("Sending response: {:?}", resp);
            let resp_bytes = serde_json::to_vec(&resp)
                .expect("fatal error: invalid response object - this should never happen");
            if (publisher.send(resp_bytes.into()).await).is_err() {
                tracing::error!("Failed to publish response for stream {}", context.id());
                context.stop_generating();
                break;
            }
        }

        Ok(())
    }
}
