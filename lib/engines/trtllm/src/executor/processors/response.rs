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

use std::thread;
use tokio::sync::mpsc;

use super::*;
use crate::executor::ResponseQueues;

pub struct ResponseProcessor {
    handle: thread::JoinHandle<()>,
}

impl ResponseProcessor {
    pub fn new(state: ProcessorState, response_queues: ResponseQueues) -> Self {
        let handle = std::thread::spawn(move || {
            process_responses(state, response_queues);
        });
        ResponseProcessor { handle }
    }

    /// Block and wait for the response processor to finish
    pub fn join(self) -> thread::Result<()> {
        self.handle.join()
    }
}

#[derive(Debug, thiserror::Error)]
enum ResponseError {
    #[error("Response queue dropped; possible client disconnect")]
    ResponseQueueDropped,

    #[error("Response channel closed; possible client disconnect")]
    ChannelClosed,

    #[error("Response channel full; backpress detected in response stream")]
    ChannelFull,

    #[error("Invalid response: no error or result found")]
    InvalidResponse,

    /// Error indicating that TensorRT LLM returned an error
    /// This also indicates that the request was not successful and no further responses
    /// will be sent for this request
    #[error("TensorRT LLM Engine Error: {0}")]
    EngineError(String),

    #[error("Completed successfully")]
    RequestComplete,
}

fn process_responses(state: ProcessorState, response_queues: ResponseQueues) {
    loop {
        // this blocks the thread until the response is ready or the server is shutdown
        let message = state
            .executor
            .await_responses()
            .expect("Failed to await responses");

        // check shutdown condition
        if message.shutdown {
            tracing::info!("Server shutdown detected");
            break;
        }

        // process responses - hold the lock while we iterate to avoid any contention
        // grabbing and releasing it for each response
        let mut queues = response_queues.lock().unwrap();

        for output in message.responses {
            let request_id = output.request_id;
            let client_id = output.client_id.expect("client_id is missing");
            let tx = queues.get(&client_id);

            match try_send(tx, output) {
                Ok(_) => {}
                Err(e) => {
                    tracing::trace!(client_id, "processing response: {}", e);
                    match e {
                        ResponseError::InvalidResponse => {
                            // this would likely be a bug on the server; we expect the oneof to be set
                            tracing::warn!(client_id, "Invalid response; No action required");
                        }
                        ResponseError::EngineError(_) => {
                            // no need to cancel, the server will not send any more responses
                            queues.remove(&client_id);
                        }
                        ResponseError::ChannelFull => {
                            // critical error
                            tracing::error!(
                                client_id,
                                "Alert: backpressure detected in response stream"
                            );
                            state.executor.cancel_request(request_id);
                            queues.remove(&client_id);
                        }
                        ResponseError::ChannelClosed => {
                            // the first indication the client has disconnected
                            state.executor.cancel_request(request_id);
                            queues.remove(&client_id);
                        }
                        ResponseError::ResponseQueueDropped => {
                            // if we get a response for a dropped queue, we need to cancel the request
                            state.executor.cancel_request(request_id);
                        }
                        ResponseError::RequestComplete => {
                            // no need to cancel, the server will not send any more responses
                            queues.remove(&client_id);
                        }
                    }
                }
            }
        }
    }
}

fn try_send(
    tx: Option<&mpsc::Sender<Result<protocols::Output>>>,
    response: protocols::Response,
) -> Result<(), ResponseError> {
    let mut rc = Ok(());

    let tx = tx.ok_or(ResponseError::ResponseQueueDropped)?;

    let result = match (response.output, response.error_msg) {
        (Some(output), None) => {
            if output.is_final {
                rc = Err(ResponseError::RequestComplete);
            }
            Ok(output)
        }
        (None, Some(e)) => {
            rc = Err(ResponseError::EngineError(e.clone()));
            Err(ResponseError::EngineError(e.clone()))
        }
        (None, None) => return Err(ResponseError::InvalidResponse),
        (Some(_), Some(_)) => return Err(ResponseError::InvalidResponse),
    };

    match tx.try_send(result.map_err(|e| e.into())) {
        Ok(_) => {}
        Err(e) => match e {
            mpsc::error::TrySendError::Closed(_) => {
                return Err(ResponseError::ChannelClosed);
            }
            mpsc::error::TrySendError::Full(_) => {
                return Err(ResponseError::ChannelFull);
            }
        },
    }

    rc
}
