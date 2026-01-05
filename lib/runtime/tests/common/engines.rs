// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]

use std::{future::Future, pin::Pin, sync::Arc};

use async_trait::async_trait;
use futures::Stream;
use tokio::sync::mpsc;

use dynamo_runtime::engine::{
    AsyncEngine, AsyncEngineContext, AsyncEngineContextProvider, AsyncEngineStream,
    Data as DataType, Engine, EngineStream,
};

use dynamo_runtime::pipeline::{
    Error, ManyOut, SingleIn,
    context::{Context, StreamContext},
};

pub type AsyncFn<T, U> = dyn Fn(T) -> Pin<Box<dyn Future<Output = U> + Send>> + Send + Sync;

#[derive(Clone)]
// Define a struct that holds an async closure
pub struct AsyncProcessor<T, U> {
    func: Arc<AsyncFn<T, U>>,
}

impl<T, U> AsyncProcessor<T, U>
where
    T: Send + 'static,
    U: Send + 'static,
{
    // Define a `new` method that captures the already pinned async block
    pub fn new<F, Fut>(f: F) -> Self
    where
        F: Fn(T) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = U> + Send + 'static,
    {
        // Wrap the closure in Arc and Box it for internal management
        AsyncProcessor {
            func: Arc::new(move |input: T| Box::pin(f(input))),
        }
    }

    // Method to execute the captured async function
    pub async fn process(&self, input: T) -> U {
        (self.func)(input).await
    }
}

#[derive(Debug, Clone)]
pub struct ResponseSource<T: Send + Sync + 'static> {
    tx: mpsc::Sender<T>,
    ctx: StreamContext,
}

impl<T: Send + Sync + 'static> ResponseSource<T> {
    fn new(tx: mpsc::Sender<T>, ctx: StreamContext) -> Self {
        ResponseSource { tx, ctx }
    }

    /// Emit a response to the stream
    pub async fn emit(&self, data: T) -> Result<(), ()> {
        self.tx.send(data).await.map_err(|_| ())
    }

    /// Check if a stop has been requested
    pub fn stop_requested(&self) -> bool {
        self.ctx.is_stopped()
    }

    /// Yield control until a stop is requested
    /// This is useful in a tokio::select! block
    pub async fn stopped(&self) {
        self.ctx.stopped().await;
    }
}

pub type AsyncGenerator<Req, Resp> = AsyncProcessor<(Req, ResponseSource<Resp>), ()>;

pub struct ReceiverStream<Resp: DataType> {
    receiver: tokio::sync::mpsc::Receiver<Resp>,
    context: Arc<dyn AsyncEngineContext>,
}

impl<Resp: DataType> ReceiverStream<Resp> {
    pub fn new(
        receiver: tokio::sync::mpsc::Receiver<Resp>,
        context: Arc<dyn AsyncEngineContext>,
    ) -> Self {
        Self { receiver, context }
    }
}

impl<Resp: DataType> Stream for ReceiverStream<Resp> {
    type Item = Resp;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        // if self.context.stop_issued() {
        //     return std::task::Poll::Ready(None);
        // }

        // Pinning the receiver to safely call poll_recv
        Pin::new(&mut self.receiver).poll_recv(cx)
    }
}

impl<Resp: DataType> std::fmt::Debug for ReceiverStream<Resp> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReceiverStream")
            .field("context", &self.context)
            .finish()
    }
}

impl<Resp: DataType> AsyncEngineStream<Resp> for ReceiverStream<Resp> {}

impl<Resp: DataType> AsyncEngineContextProvider for ReceiverStream<Resp> {
    fn context(&self) -> Arc<dyn AsyncEngineContext> {
        self.context.clone()
    }
}

pub struct LlmdbaEngine<Req: DataType, Resp: DataType> {
    lambda: Arc<AsyncGenerator<Req, Resp>>,
}

impl<Req: DataType, Resp: DataType> LlmdbaEngine<Req, Resp> {
    fn new(lambda: AsyncGenerator<Req, Resp>) -> Self {
        LlmdbaEngine {
            lambda: Arc::new(lambda),
        }
    }

    pub fn from_generator(
        generator: AsyncGenerator<Req, Resp>,
    ) -> Engine<SingleIn<Req>, ManyOut<Resp>, Error> {
        Arc::new(LlmdbaEngine::new(generator))
    }
}

#[async_trait]
impl<Req: DataType, Resp: DataType> AsyncEngine<SingleIn<Req>, ManyOut<Resp>, Error>
    for LlmdbaEngine<Req, Resp>
{
    async fn generate(&self, request: Context<Req>) -> Result<EngineStream<Resp>, Error> {
        let (tx, rx) = mpsc::channel::<Resp>(1);
        let (req, ctx) = request.transfer(());
        let ctx: StreamContext = ctx.into();
        let s = ResponseSource::new(tx, ctx.clone());

        let lambda = self.lambda.clone();
        let _handle = tokio::spawn(async move { lambda.process((req, s)).await });

        let ctx = Arc::new(ctx);
        let stream = ReceiverStream::<Resp>::new(rx, ctx);
        let stream = Box::pin(stream);
        Ok(stream)
    }
}

#[cfg(test)]
mod tests {

    use futures::StreamExt;

    use super::*;

    #[tokio::test]
    async fn test_async_processor() {
        let processor = AsyncProcessor::new(move |x: i32| {
            async move {
                // Simulate some async work
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                format!("Processed value: {}", x)
            }
        });

        // Use the processor to run the async closure
        let result = processor.process(42).await;
        println!("{}", result); // Output: Processed value: 42

        let result2 = processor.process(100).await;
        println!("{}", result2); // Output: Processed value: 100
    }

    #[tokio::test]
    async fn test_generator() {
        let generator = AsyncGenerator::<String, String>::new(|(req, stream)| async move {
            let chars = req.chars().collect::<Vec<char>>();
            for c in chars {
                match stream.emit(c.to_string()).await {
                    Ok(_) => {}
                    Err(_) => break,
                }
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
        });

        let engine = LlmdbaEngine::new(generator);

        let mut stream = engine.generate("test".to_string().into()).await.unwrap();

        let mut counter = 0;
        while let Some(_output) = stream.next().await {
            counter += 1;
        }

        assert_eq!(counter, 4);
    }
}
