// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::hash_map::Entry;

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
        let tx = sinks
            .remove(ctx.id())
            .ok_or(PipelineError::DetachedStreamReceiver)
            .inspect_err(|_| {
                ctx.stop_generating();
            })?;
        drop(sinks);

        Ok(tx
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
        let id = request.id().to_string();
        {
            // The response is routed back by request id, so a second in-flight request with the
            // same id would steal or orphan the first caller's stream. Refuse it instead.
            let mut sinks = self.sinks.lock().unwrap();
            match sinks.entry(id.clone()) {
                Entry::Occupied(_) => return Err(PipelineError::DuplicateRequestId(id).into()),
                Entry::Vacant(slot) => {
                    slot.insert(tx);
                }
            }
        }
        if let Err(err) = self.on_next(request, private::Token {}).await {
            // Do not leave a stale sender behind; it would block a later request with this id.
            self.sinks.lock().unwrap().remove(&id);
            return Err(err);
        }
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
}
