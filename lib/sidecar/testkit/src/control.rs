// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use futures::{StreamExt, stream::BoxStream};
use tokio::sync::{Notify, watch};

/// Implement on a local adapter, keeping native protobuf and transport types out of the testkit.
pub trait Protocol: Send + Sync + 'static {
    type Request: Clone + Send + Sync + 'static;
    type Response: Clone + Send + Sync + 'static;
    type Error: Send + 'static;

    fn request_id(request: &Self::Request) -> &str;
    fn record_tokens(response: &Self::Response, tokens: &mut Vec<u32>) -> bool;
    fn is_terminal(response: &Self::Response) -> bool;
    fn injected_error(message: &'static str) -> Self::Error;
}

#[derive(Clone, Copy, Default)]
pub enum OpenAction {
    #[default]
    Continue,
    Fail,
    Hold,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamPoint {
    /// One-based count of native responses containing output tokens.
    TokenResponse(usize),
    Terminal,
}

#[derive(Clone, Copy)]
pub enum StreamAction {
    Continue,
    Close,
    Fail,
    ReplayFirst,
}

#[derive(Clone, Copy)]
pub struct StreamFault {
    pub at: StreamPoint,
    pub action: StreamAction,
    pub pause: bool,
}

#[derive(Clone, Copy, Default)]
pub struct RequestPlan {
    pub open: OpenAction,
    pub stream: Option<StreamFault>,
}

#[derive(Clone, Copy, Debug)]
pub enum Event {
    Received,
    Checkpoint,
    Dropped,
}

#[derive(Clone, Copy, Default)]
struct Progress {
    received: bool,
    checkpoint: bool,
    dropped: bool,
}

impl Progress {
    fn reached(self, event: Event) -> bool {
        match event {
            Event::Received => self.received,
            Event::Checkpoint => self.checkpoint,
            Event::Dropped => self.dropped,
        }
    }
}

struct Observed<P: Protocol> {
    request: Option<P::Request>,
    responses: Vec<P::Response>,
    tokens: Vec<u32>,
}

struct RequestState<P: Protocol> {
    plan: RequestPlan,
    observed: Mutex<Observed<P>>,
    progress: watch::Sender<Progress>,
    release: Notify,
}

pub struct RequestHandle<P: Protocol>(Arc<RequestState<P>>);

impl<P: Protocol> Clone for RequestHandle<P> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<P: Protocol> RequestHandle<P> {
    fn new(plan: RequestPlan) -> Self {
        if let Some(StreamFault {
            at: StreamPoint::TokenResponse(count),
            ..
        }) = plan.stream
        {
            assert!(count > 0, "token response checkpoints are one-based");
        }
        Self(Arc::new(RequestState {
            plan,
            observed: Mutex::new(Observed {
                request: None,
                responses: Vec::new(),
                tokens: Vec::new(),
            }),
            progress: watch::channel(Progress::default()).0,
            release: Notify::new(),
        }))
    }

    pub fn reached(&self, event: Event) -> bool {
        self.0.progress.borrow().reached(event)
    }

    pub async fn wait(&self, event: Event) {
        self.0
            .progress
            .subscribe()
            .wait_for(|progress| progress.reached(event))
            .await
            .unwrap();
    }

    pub fn release(&self) {
        self.0.release.notify_one();
    }

    pub fn tokens(&self) -> Vec<u32> {
        self.0.observed.lock().unwrap().tokens.clone()
    }

    pub fn native_request(&self) -> Option<P::Request> {
        self.0.observed.lock().unwrap().request.clone()
    }

    pub fn native_responses(&self) -> Vec<P::Response> {
        self.0.observed.lock().unwrap().responses.clone()
    }
}

pub struct Controller<P: Protocol> {
    requests: Arc<Mutex<HashMap<String, RequestHandle<P>>>>,
}

impl<P: Protocol> Default for Controller<P> {
    fn default() -> Self {
        Self {
            requests: Arc::default(),
        }
    }
}

impl<P: Protocol> Clone for Controller<P> {
    fn clone(&self) -> Self {
        Self {
            requests: Arc::clone(&self.requests),
        }
    }
}

impl<P: Protocol> Controller<P> {
    pub fn request(&self, request_id: &str, plan: RequestPlan) -> RequestHandle<P> {
        let handle = RequestHandle::new(plan);
        let mut requests = self.requests.lock().unwrap();
        assert!(
            !requests.contains_key(request_id),
            "request already registered: {request_id}"
        );
        requests.insert(request_id.to_owned(), handle.clone());
        handle
    }

    pub async fn open(&self, request: &P::Request) -> Result<OpenedRequest<P>, P::Error> {
        let handle = self
            .requests
            .lock()
            .unwrap()
            .entry(P::request_id(request).to_owned())
            .or_insert_with(|| RequestHandle::new(RequestPlan::default()))
            .clone();
        {
            let mut observed = handle.0.observed.lock().unwrap();
            assert!(
                observed.request.is_none(),
                "request ID submitted more than once"
            );
            observed.request = Some(request.clone());
        }
        let guard = RequestGuard(handle.clone());
        handle
            .0
            .progress
            .send_modify(|progress| progress.received = true);
        match handle.0.plan.open {
            OpenAction::Continue => {}
            OpenAction::Fail => return Err(P::injected_error("injected open failure")),
            OpenAction::Hold => handle.0.release.notified().await,
        }
        Ok(OpenedRequest { handle, guard })
    }
}

struct RequestGuard<P: Protocol>(RequestHandle<P>);

impl<P: Protocol> Drop for RequestGuard<P> {
    fn drop(&mut self) {
        self.0
            .0
            .progress
            .send_modify(|progress| progress.dropped = true);
    }
}

pub struct OpenedRequest<P: Protocol> {
    handle: RequestHandle<P>,
    guard: RequestGuard<P>,
}

impl<P: Protocol> OpenedRequest<P> {
    pub fn wrap(
        self,
        mut source: BoxStream<'static, Result<P::Response, P::Error>>,
    ) -> BoxStream<'static, Result<P::Response, P::Error>> {
        Box::pin(async_stream::try_stream! {
            let _guard = self.guard;
            let handle = self.handle;
            let mut first = None;
            let mut token_responses = 0;
            while let Some(response) = source.next().await {
                let response = response?;
                let has_tokens = {
                    let mut observed = handle.0.observed.lock().unwrap();
                    observed.responses.push(response.clone());
                    P::record_tokens(&response, &mut observed.tokens)
                };
                if has_tokens {
                    token_responses += 1;
                    if first.is_none() {
                        first = Some(response.clone());
                    }
                }
                let terminal = P::is_terminal(&response);
                yield response;
                if let Some(fault) = handle.0.plan.stream {
                    let matches = match fault.at {
                        StreamPoint::TokenResponse(count) => has_tokens && count == token_responses,
                        StreamPoint::Terminal => terminal,
                    };
                    if matches {
                        handle.0.progress.send_modify(|progress| progress.checkpoint = true);
                        if fault.pause {
                            handle.0.release.notified().await;
                        }
                        match fault.action {
                            StreamAction::Continue => {}
                            StreamAction::Close => return,
                            StreamAction::Fail => Err(P::injected_error("injected read failure"))?,
                            StreamAction::ReplayFirst => {
                                yield first.take().expect("Mocker generated tokens");
                                futures::future::pending::<()>().await;
                            }
                        }
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::{poll, stream};

    struct TestProtocol;

    #[derive(Clone, Debug, PartialEq, Eq)]
    enum Response {
        Token(u32),
        Terminal,
    }

    impl Protocol for TestProtocol {
        type Request = &'static str;
        type Response = Response;
        type Error = &'static str;

        fn request_id(request: &Self::Request) -> &str {
            request
        }

        fn record_tokens(response: &Self::Response, tokens: &mut Vec<u32>) -> bool {
            if let Response::Token(token) = response {
                tokens.push(*token);
                true
            } else {
                false
            }
        }

        fn is_terminal(response: &Self::Response) -> bool {
            matches!(response, Response::Terminal)
        }

        fn injected_error(message: &'static str) -> Self::Error {
            message
        }
    }

    #[tokio::test]
    async fn replay_first_after_terminal_keeps_stream_open() {
        let control = Controller::<TestProtocol>::default();
        control.request(
            "replay",
            RequestPlan {
                stream: Some(StreamFault {
                    at: StreamPoint::Terminal,
                    action: StreamAction::ReplayFirst,
                    pause: false,
                }),
                ..Default::default()
            },
        );
        let source = stream::iter([Ok(Response::Token(7)), Ok(Response::Terminal)]).boxed();
        let mut stream = control.open(&"replay").await.unwrap().wrap(source);

        crate::bounded("replayed responses", async {
            for expected in [Response::Token(7), Response::Terminal, Response::Token(7)] {
                assert_eq!(stream.next().await, Some(Ok(expected)));
            }
        })
        .await;
        assert!(poll!(stream.next()).is_pending());
    }
}
