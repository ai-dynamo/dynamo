// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]

use anyhow::Error;
use futures::{StreamExt, stream};
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Duration};

use dynamo_runtime::engine::ResponseStream;
use dynamo_runtime::pipeline::{
    AsyncEngine,
    Data,
    Event,
    ManyOut,
    Operator,
    ServiceBackend,
    ServiceEngine,
    ServiceFrontend,
    SingleIn,
    *, // TODO remove the star
};

mod common;
use common::engines::{AsyncGenerator, LlmdbaEngine as LambdaEngine};
use common::mock;

/// The [`super::engine::ResponseStream`] is annotated with the following types.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Annotated<T: Data> {
    /// The primary data which expected to be returned.
    Data(T),

    /// An actionable [`Event`] that can be handled.
    Event(Event),

    /// Additional information or metadata produced by the pipeline.
    Comment(String),

    /// An error produced by the pipeline. Multiple errors can be produced.
    Error(String),

    /// A sentinel value to indicate the end of the stream. This should not be emitted publicly.
    /// The implementation should be able to do the equivalent of a `.take_while` and trigger a
    /// stop if detected.
    End,
}

/// An [`Operator`] is used when you want to transform both the input and output of a pipeline.
/// In this case, our operator will perform the preprocessing step, but also add an annotation
/// to the output stream
struct PreprocesOperator {}

#[async_trait::async_trait]
impl
    Operator<
        SingleIn<String>,
        ManyOut<Annotated<String>>,
        SingleIn<String>,
        ManyOut<Annotated<String>>,
    > for PreprocesOperator
{
    async fn generate(
        &self,
        req: SingleIn<String>,
        next: Arc<dyn AsyncEngine<SingleIn<String>, ManyOut<Annotated<String>>, Error>>,
    ) -> Result<ManyOut<Annotated<String>>, Error> {
        // capture some details about the request
        let prepend = vec![Annotated::<String>::Comment(format!(
            "PreprocessOperator: {:?}",
            req
        ))];

        // we will append the result of this to the response stream via a chain
        let prepend_stream = stream::iter(prepend);

        // modify the request
        let req = req.map(|x| format!("{} from operator", x));

        // issue the preprocessed request to the next engine
        let stream = next.generate(req).await?;

        // capture the context of the response stream
        let ctx = stream.context();

        // chain the prepend stream to the response stream
        Ok(ResponseStream::new(
            Box::pin(prepend_stream.chain(stream)),
            ctx,
        ))
    }
}

fn make_backend_engine() -> ServiceEngine<SingleIn<String>, ManyOut<Annotated<String>>> {
    LambdaEngine::from_generator(AsyncGenerator::<String, Annotated<String>>::new(
        |(req, stream)| async move {
            let chars = req.chars().collect::<Vec<char>>();
            for c in chars {
                match stream.emit(Annotated::Data(c.to_string())).await {
                    Ok(_) => {}
                    Err(_) => return,
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        },
    ))
}

#[tokio::test]
async fn test_service_source_sink() {
    let source = ServiceFrontend::<SingleIn<String>, ManyOut<Annotated<String>>>::new();
    let sink = ServiceBackend::from_engine(make_backend_engine());

    let service = source.link(sink).unwrap().link(source).unwrap();

    let mut stream = service.generate("test".to_string().into()).await.unwrap();

    let mut counter = 0;
    while let Some(_output) = stream.next().await {
        counter += 1;
    }

    assert_eq!(counter, 4);
}

fn make_preprocessor() -> Arc<PipelineNode<SingleIn<String>, SingleIn<String>>> {
    PipelineNode::<SingleIn<String>, SingleIn<String>>::new(Box::new(|req| {
        Ok(req.map(|x| format!("{} world", x)))
    }))
}

#[allow(clippy::type_complexity)]
fn make_postprocessor() -> Arc<PipelineNode<ManyOut<Annotated<String>>, ManyOut<Annotated<String>>>>
{
    PipelineNode::<ManyOut<Annotated<String>>, ManyOut<Annotated<String>>>::new(Box::new(|req| {
        let ctx = req.context();
        let double_stream = req.flat_map(|x| {
            let x1 = x.clone();
            let x2 = x;
            stream::iter(vec![x1, x2])
        });
        Ok(ResponseStream::new(Box::pin(double_stream), ctx))
    }))
}

// Node 0:
// [frontend] -------[pre processor]-----> [backend]
// [frontend] <----- [post processor] ---- [backend]
fn make_service()
-> Result<ServiceEngine<SingleIn<String>, ManyOut<Annotated<String>>>, PipelineError> {
    // Frontend - Callable interface
    let frontend = ServiceFrontend::<SingleIn<String>, ManyOut<Annotated<String>>>::new();

    // Mimics processing the prompt and tokenization
    let preprocess = make_preprocessor();

    // Mimics decoding; shows we can use any type of stream operation,
    // e.g. map, flat_map, fold, scan, etc. to transform the response stream
    let postprocess = make_postprocessor();

    // Mimics backend streaming by emitting each character of the input string
    let backend = ServiceBackend::from_engine(make_backend_engine());

    // LLM Pipelines are build by linking the frontend to the backend for input handling
    // then linking from the backend to the frontend for the output handling
    let service = frontend
        .link(preprocess)?
        .link(backend)?
        .link(postprocess)?
        .link(frontend)?;

    Ok(service)
}

#[tokio::test]
async fn test_service_source_node_sink() {
    let service = make_service().unwrap();

    let mut stream = service.generate("test".to_string().into()).await.unwrap();

    let mut counter = 0;
    while let Some(_output) = stream.next().await {
        counter += 1;
    }

    assert_eq!(counter, 20);
}

// Put the post process on node 0, but the preprocessor and the compute on node1
// Node 0:
// [frontend] ---------------------------> [segment_sink]
// [frontend] <----- [post processor] ---- [segment_sink]
//
// Node 1:
// [segment_source] ---- [preprocessor] ---> [backend]
// [segment_source] <----------------------- [backend]
#[tokio::test]
#[ignore = "Blocked by AsyncEngineStream trait missing Sync supertrait"]
#[expect(unused_variables)]
async fn test_disaggregated_service() {
    println!("Running test_disaggregated_service");

    // Node 0
    let frontend = ServiceFrontend::<SingleIn<String>, ManyOut<Annotated<String>>>::new();
    let postprocessor = make_postprocessor();
    let end_node_0 = SegmentSink::<SingleIn<String>, ManyOut<Annotated<String>>>::new();
    let node0_service = frontend
        .link(end_node_0.clone())
        .unwrap()
        .link(postprocessor)
        .unwrap()
        .link(frontend)
        .unwrap();

    // Node 1
    let start_node1 = SegmentSource::<SingleIn<String>, ManyOut<Annotated<String>>>::new();
    let preprocessor = make_preprocessor();
    let backend = ServiceBackend::from_engine(make_backend_engine());
    let node1_service = start_node1
        .link(preprocessor)
        .unwrap()
        .link(backend)
        .unwrap()
        .link(start_node1.clone())
        .unwrap();

    let opts = mock::MockNetworkOptions::default();
    let (egress, ingress) = mock::MockNetworkTransport::<
        SingleIn<String>,
        ManyOut<Annotated<String>>,
    >::new_egress_ingress(opts);

    // BLOCKED: Cannot attach egress because Engine<T,U,E> = Arc<dyn AsyncEngine<...>>
    // but AsyncEngineStream cannot be Sync (by design), preventing trait object creation
    // end_node_0.attach(egress).unwrap();
    // Commented out since attach is blocked
    // ingress.segment(node1_service).unwrap();
    // tokio::spawn(ingress.execute());
    // let mut stream = node0_service
    //     .generate("test".to_string().into())
    //     .await
    //     .unwrap();
    // let mut counter = 0;
    // while let Some(_output) = stream.next().await {
    //     counter += 1;
    // }
    // assert_eq!(counter, 20);

    println!(
        "Test blocked: SegmentSink::attach requires Arc<dyn AsyncEngine> but AsyncEngineStream cannot be Sync"
    );
}

// Node 0:
// [frontend] --> [pre processor] --> [operator] ----------------------> [backend]
// [frontend] <---------------------- [operator] <--[post processor] <-- [backend]
fn make_service_with_operator()
-> Result<ServiceEngine<SingleIn<String>, ManyOut<Annotated<String>>>, PipelineError> {
    // Frontend - Callable interface
    let frontend = ServiceFrontend::<SingleIn<String>, ManyOut<Annotated<String>>>::new();

    // Mimics processing the prompt and tokenization
    let preprocess = make_preprocessor();

    // Mimics decoding; shows we can use any type of stream operation,
    // e.g. map, flat_map, fold, scan, etc. to transform the response stream
    let postprocess = make_postprocessor();

    // Mimics backend streaming by emitting each character of the input string
    let backend = ServiceBackend::from_engine(make_backend_engine());

    let operator = PipelineOperator::new(Arc::new(PreprocesOperator {}));

    // LLM Pipelines are build by linking the frontend to the backend for input handling
    // then linking from the backend to the frontend for the output handling
    let service = frontend
        .link(preprocess)?
        .link(operator.forward_edge())?
        .link(backend)?
        .link(postprocess)?
        .link(operator.backward_edge())?
        .link(frontend)?;

    Ok(service)
}

#[tokio::test]
async fn test_service_source_node_sink_with_operator() {
    let service = make_service_with_operator().unwrap();

    let mut stream = service.generate("test".to_string().into()).await.unwrap();

    let mut counter = 0;
    let mut annotations_counter = 0;
    while let Some(output) = stream.next().await {
        match output {
            Annotated::Data(_) => counter += 1,
            Annotated::Comment(_) => annotations_counter += 1,
            _ => {}
        }
    }

    assert_eq!(annotations_counter, 1);
    assert_eq!(counter, 48);
}
