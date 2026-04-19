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
            let mut sinks = self.sinks.lock().unwrap();
            sinks.insert(id.clone(), tx);
        }
        if let Err(e) = self.on_next(request, private::Token {}).await {
            // Clean up the orphaned sender to prevent unbounded HashMap growth
            let mut sinks = self.sinks.lock().unwrap();
            sinks.remove(&id);
            return Err(e);
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

    #[tokio::test]
    async fn test_generate_failure_cleans_up_sinks() {
        // When generate() fails because on_next() errors (no edge linked),
        // the oneshot sender should be removed from the sinks HashMap.
        // Otherwise orphaned entries accumulate as a memory leak.
        let source = Frontend::<SingleIn<()>, ManyOut<()>>::default();

        // generate() should fail with NoEdge
        let err = source.generate(().into()).await.unwrap_err();
        assert!(err.downcast_ref::<PipelineError>().is_some());

        // The sinks HashMap should be empty — the orphaned sender must be cleaned up
        let sinks = source.sinks.lock().unwrap();
        assert!(
            sinks.is_empty(),
            "sinks HashMap should be empty after generate() failure, but has {} entries",
            sinks.len()
        );
    }

    #[tokio::test]
    async fn test_repeated_generate_failures_do_not_accumulate() {
        // Repeated generate() failures should not cause the sinks HashMap to grow.
        let source = Frontend::<SingleIn<()>, ManyOut<()>>::default();

        for _ in 0..100 {
            let _ = source.generate(().into()).await;
        }

        let sinks = source.sinks.lock().unwrap();
        assert!(
            sinks.is_empty(),
            "sinks HashMap should be empty after 100 failed generate() calls, but has {} entries",
            sinks.len()
        );
    }
}
