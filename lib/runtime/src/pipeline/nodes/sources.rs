// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::pipeline::{AsyncEngine, PipelineIO};

mod base;
mod common;

pub struct Frontend<In: PipelineIO, Out: PipelineIO> {
    edge: OnceLock<Edge<In>>,
    sinks: Arc<Mutex<HashMap<String, oneshot::Sender<Out>>>>,
}

/// A [`ServiceFrontend`] is the interface for an [`AsyncEngine<SingleIn<Context<In>>, ManyOut<Annotated<Out>>, Error>`]
pub struct ServiceFrontend<In: PipelineIO, Out: PipelineIO> {
    inner: Frontend<In, Out>,
}

pub struct SegmentSource<In: PipelineIO, Out: PipelineIO> {
    inner: Frontend<In, Out>,
}

// impl<In: DataType, Out: PipelineIO> Frontend<In, Out> {
//     pub fn new() -> Arc<Self> {
//         Arc::new(Self {
//             edge: OnceLock::new(),
//             sinks: Arc::new(Mutex::new(HashMap::new())),
//         })
//     }
// }

// impl<In: DataType, Out: PipelineIO> SegmentSource<In, Out> {
//     pub fn new() -> Arc<Self> {
//         Arc::new(Self {
//             edge: OnceLock::new(),
//             sinks: Arc::new(Mutex::new(HashMap::new())),
//         })
//     }
// }

// #[async_trait]
// impl<In: DataType, Out: PipelineIO> Source<Context<In>> for Frontend<In, Out> {
//     async fn on_next(&self, data: Context<In>, _: private::Token) -> Result<(), PipelineError> {
//         self.edge
//             .get()
//             .ok_or(PipelineError::NoEdge)?
//             .write(data)
//             .await
//     }

//     fn set_edge(
//         &self,
//         edge: Edge<Context<In>>>,
//         _: private::Token,
//     ) -> Result<(), PipelineError> {
//         self.edge
//             .set(edge)
//             .map_err(|_| PipelineError::EdgeAlreadySet)?;
//         Ok(())
//     }
// }

// #[async_trait]
// impl<In: DataType, Out: PipelineIO> Sink<PipelineStream<Out>> for Frontend<In, Out> {
//     async fn on_data(
//         &self,
//         data: PipelineStream<Out>,
//         _: private::Token,
//     ) -> Result<(), PipelineError> {
//         let context = data.context();

//         let mut sinks = self.sinks.lock().unwrap();
//         let tx = sinks
//             .remove(context.id())
//             .ok_or(PipelineError::DetachedStreamReceiver)
//             .map_err(|e| {
//                 data.context().stop_generating();
//                 e
//             })?;
//         drop(sinks);

//         let ctx = data.context();
//         tx.send(data)
//             .map_err(|_| PipelineError::DetachedStreamReceiver)
//             .map_err(|e| {
//                 ctx.stop_generating();
//                 e
//             })
//     }
// }

// impl<In: DataType, Out: PipelineIO> Link<Context<In>> for Frontend<In, Out> {
//     fn link<S: Sink<Context<In>> + 'static>(&self, sink: Arc<S>) -> Result<Arc<S>, PipelineError> {
//         let edge = Edge::new(sink.clone());
//         self.set_edge(edge.into(), private::Token {})?;
//         Ok(sink)
//     }
// }

// #[async_trait]
// impl<In: DataType, Out: PipelineIO> AsyncEngine<Context<In>, Annotated<Out>, PipelineError>
//     for Frontend<In, Out>
// {
//     async fn generate(&self, request: Context<In>) -> Result<PipelineStream<Out>, PipelineError> {
//         let (tx, rx) = oneshot::channel::<PipelineStream<Out>>();
//         {
//             let mut sinks = self.sinks.lock().unwrap();
//             sinks.insert(request.id().to_string(), tx);
//         }
//         self.on_next(request, private::Token {}).await?;
//         rx.await.map_err(|_| PipelineError::DetachedStreamSender)
//     }
// }

// // SegmentSource

// #[async_trait]
// impl<In: DataType, Out: PipelineIO> Source<Context<In>> for SegmentSource<In, Out> {
//     async fn on_next(&self, data: Context<In>, _: private::Token) -> Result<(), PipelineError> {
//         self.edge
//             .get()
//             .ok_or(PipelineError::NoEdge)?
//             .write(data)
//             .await
//     }

//     fn set_edge(
//         &self,
//         edge: Edge<Context<In>>>,
//         _: private::Token,
//     ) -> Result<(), PipelineError> {
//         self.edge
//             .set(edge)
//             .map_err(|_| PipelineError::EdgeAlreadySet)?;
//         Ok(())
//     }
// }

// #[async_trait]
// impl<In: DataType, Out: PipelineIO> Sink<PipelineStream<Out>> for SegmentSource<In, Out> {
//     async fn on_data(
//         &self,
//         data: PipelineStream<Out>,
//         _: private::Token,
//     ) -> Result<(), PipelineError> {
//         let context = data.context();

//         let mut sinks = self.sinks.lock().unwrap();
//         let tx = sinks
//             .remove(context.id())
//             .ok_or(PipelineError::DetachedStreamReceiver)
//             .map_err(|e| {
//                 data.context().stop_generating();
//                 e
//             })?;
//         drop(sinks);

//         let ctx = data.context();
//         tx.send(data)
//             .map_err(|_| PipelineError::DetachedStreamReceiver)
//             .map_err(|e| {
//                 ctx.stop_generating();
//                 e
//             })
//     }
// }

// impl<In: DataType, Out: PipelineIO> Link<Context<In>> for SegmentSource<In, Out> {
//     fn link<S: Sink<Context<In>> + 'static>(&self, sink: Arc<S>) -> Result<Arc<S>, PipelineError> {
//         let edge = Edge::new(sink.clone());
//         self.set_edge(edge.into(), private::Token {})?;
//         Ok(sink)
//     }
// }

// #[async_trait]
// impl<In: DataType, Out: PipelineIO> AsyncEngine<Context<In>, Annotated<Out>, PipelineError>
//     for SegmentSource<In, Out>
// {
//     async fn generate(&self, request: Context<In>) -> Result<PipelineStream<Out>, PipelineError> {
//         let (tx, rx) = oneshot::channel::<PipelineStream<Out>>();
//         {
//             let mut sinks = self.sinks.lock().unwrap();
//             sinks.insert(request.id().to_string(), tx);
//         }
//         self.on_next(request, private::Token {}).await?;
//         rx.await.map_err(|_| PipelineError::DetachedStreamSender)
//     }
// }

// #[cfg(test)]

// mod tests {
//     use super::*;

//     #[tokio::test]
//     async fn test_pipeline_source_no_edge() {
//         let source = Frontend::<(), ()>::new();
//         let stream = source.generate(().into()).await;
//         match stream {
//             Err(PipelineError::NoEdge) => (),
//             _ => panic!("Expected NoEdge error"),
//         }
//     }
// }

// pub struct IngressPort<In, Out: PipelineIO> {
//     edge: OnceLock<ServiceEngine<In, Out>>,
// }

// impl<In, Out> IngressPort<In, Out>
// where
//     In: for<'de> Deserialize<'de> + DataType,
//     Out: PipelineIO + Serialize,
// {
//     pub fn new() -> Arc<Self> {
//         Arc::new(IngressPort {
//             edge: OnceLock::new(),
//         })
//     }
// }

// #[async_trait]
// impl<In, Out> AsyncEngine<Context<Vec<u8>>, Vec<u8>> for IngressPort<In, Out>
// where
//     In: for<'de> Deserialize<'de> + DataType,
//     Out: PipelineIO + Serialize,
// {
//     async fn generate(
//         &self,
//         request: Context<Vec<u8>>,
//     ) -> Result<EngineStream<Vec<u8>>, PipelineError> {
//         // Deserialize request
//         let request = request.try_map(|bytes| {
//             bincode::deserialize::<In>(&bytes)
//                 .map_err(|err| PipelineError(format!("Failed to deserialize request: {}", err)))
//         })?;

//         // Forward request to edge
//         let stream = self
//             .edge
//             .get()
//             .ok_or(PipelineError("No engine to forward request to".to_string()))?
//             .generate(request)
//             .await?;

//         // Serialize response stream

//         let stream =
//             stream.map(|resp| bincode::serialize(&resp).expect("Failed to serialize response"));

//         Err(PipelineError(format!("Not implemented")))
//     }
// }

// fn convert_stream<T, U>(
//     stream: impl Stream<Item = ServerStream<T>> + Send + 'static,
//     ctx: Arc<dyn AsyncEngineContext>,
//     transform: Arc<dyn Fn(T) -> Result<U, StreamError> + Send + Sync>,
// ) -> Pin<Box<dyn Stream<Item = ServerStream<U>> + Send>>
// where
//     T: Send + 'static,
//     U: Send + 'static,
// {
//     Box::pin(stream.flat_map(move |item| {
//         let ctx = ctx.clone();
//         let transform = transform.clone();
//         match item {
//             ServerStream::Data(data) => match transform(data) {
//                 Ok(transformed) => futures::stream::iter(vec![ServerStream::Data(transformed)]),
//                 Err(e) => {
//                     // Trigger cancellation and propagate the error, followed by Sentinel
//                     ctx.stop_generating();
//                     futures::stream::iter(vec![ServerStream::Error(e), ServerStream::Sentinel])
//                 }
//             },
//             other => futures::stream::iter(vec![other]),
//         }
//     })
//     // Use take_while to stop processing when encountering the Sentinel
//     .take_while(|item| futures::future::ready(!matches!(item, ServerStream::Sentinel))))
// }
