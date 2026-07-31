// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::engine::AsyncEngineContextProvider;

use super::*;

impl<In: PipelineIO, Out: PipelineIO> Default for Frontend<In, Out> {
    fn default() -> Self {
        Self {
            edge: OnceLock::new(),
            sinks: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl<In: PipelineIO, Out: PipelineIO> Source<In> for Frontend<In, Out> {
    async fn on_next(&self, data: In, _: private::Token) -> Result<(), Error> {
        self.edge
            .get()
            .ok_or(PipelineError::NoEdge)?
            .write(data)
            .await
    }

    fn set_edge(&self, edge: Edge<In>, _: private::Token) -> Result<(), PipelineError> {
        self.edge
            .set(edge)
            .map_err(|_| PipelineError::EdgeAlreadySet)?;
        Ok(())
    }
}

#[async_trait]
impl<In: PipelineIO, Out: PipelineIO + AsyncEngineContextProvider> Sink<Out> for Frontend<In, Out> {
    async fn on_data(&self, data: Out, _: private::Token) -> Result<(), Error> {
        let ctx = data.context();

        let mut sinks = self.sinks.lock().unwrap();
        let pending = sinks
            .remove(ctx.id())
            .ok_or(PipelineError::DetachedStreamReceiver)
            .inspect_err(|_| {
                ctx.stop_generating();
            })?;
        drop(sinks);

        Ok(pending
            .sender
            .send(data)
            .map_err(|_| PipelineError::DetachedStreamReceiver)
            .inspect_err(|_| {
                ctx.stop_generating();
            })?)
    }
}

#[async_trait]
impl<In: PipelineIO + Sync, Out: PipelineIO> AsyncEngine<In, Out, Error> for Frontend<In, Out> {
    async fn generate(&self, request: In) -> Result<Out, Error> {
        let (tx, rx) = oneshot::channel::<Out>();
        let request_id = request.id().to_string();
        let registration = Arc::new(());
        {
            let mut sinks = self.sinks.lock().unwrap();
            sinks.insert(
                request_id.clone(),
                PendingResponse {
                    registration: registration.clone(),
                    sender: tx,
                },
            );
        }
        // A response removes this entry in `on_data`. The guard handles every earlier
        // exit, including a downstream error or cancellation, and its token prevents an
        // older call from removing a newer registration which reused the same request ID.
        let _registration = ResponseRegistration {
            request_id,
            registration,
            pending: self.sinks.clone(),
        };
        self.on_next(request, private::Token {}).await?;
        Ok(rx.await.map_err(|_| PipelineError::DetachedStreamSender)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{ManyOut, SingleIn, error::PipelineErrorExt};

    #[tokio::test]
    async fn test_frontend_no_edge() {
        let source = Frontend::<SingleIn<()>, ManyOut<()>>::default();
        let error = source
            .generate(().into())
            .await
            .unwrap_err()
            .try_into_pipeline_error()
            .unwrap();

        match error {
            PipelineError::NoEdge => (),
            _ => panic!("Expected NoEdge error"),
        }
        assert!(source.sinks.lock().unwrap().is_empty());

        let result = source
            .on_next(().into(), private::Token)
            .await
            .unwrap_err()
            .try_into_pipeline_error()
            .unwrap();

        match result {
            PipelineError::NoEdge => (),
            _ => panic!("Expected NoEdge error"),
        }
    }

    #[test]
    fn stale_registration_does_not_remove_replacement() {
        let pending = Arc::new(Mutex::new(HashMap::new()));
        let old_registration = Arc::new(());
        let new_registration = Arc::new(());
        let (sender, _receiver) = oneshot::channel::<ManyOut<()>>();
        pending.lock().unwrap().insert(
            "reused-request-id".to_string(),
            PendingResponse {
                registration: new_registration,
                sender,
            },
        );

        drop(ResponseRegistration {
            request_id: "reused-request-id".to_string(),
            registration: old_registration,
            pending: pending.clone(),
        });

        assert!(pending.lock().unwrap().contains_key("reused-request-id"));
    }
}
