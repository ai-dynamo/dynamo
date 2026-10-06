// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::io;
use std::sync::{Arc, Mutex};

use dynamo_sidecar_testkit::control::{Controller, Protocol};
use dynamo_sidecar_testkit::server::TestServer;
use futures::StreamExt;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::sync::{Notify, watch};

pub struct Fixture {
    server: TestServer,
    aborted: watch::Receiver<Vec<String>>,
    held: Arc<Mutex<HashMap<String, Arc<ResponseGate>>>>,
}

pub struct ResponseGate {
    empty_sent: watch::Sender<bool>,
    release: Notify,
    responses: Vec<Value>,
}

impl ResponseGate {
    pub async fn wait_empty_response(&self) {
        dynamo_sidecar_testkit::bounded(
            "HTTP headers and empty response",
            self.empty_sent.subscribe().wait_for(|sent| *sent),
        )
        .await
        .unwrap();
    }

    pub fn release(&self) {
        self.release.notify_one();
    }
}

impl Fixture {
    pub async fn start(control: Controller<Adapter>) -> Self {
        let (abort_tx, aborted) = watch::channel(Vec::new());
        let held = Arc::new(Mutex::new(HashMap::new()));
        let server_held = Arc::clone(&held);
        let server = TestServer::start(move |listener, _shutdown| async move {
            let mut connections = tokio::task::JoinSet::new();
            loop {
                tokio::select! {
                    accepted = listener.accept() => {
                        let (socket, _) = accepted?;
                        connections.spawn(serve(socket, control.clone(), abort_tx.clone(), Arc::clone(&server_held)));
                    }
                    result = connections.join_next(), if !connections.is_empty() => {
                        result.unwrap()??;
                    }
                }
            }
        })
        .await
        .unwrap();
        Self {
            server,
            aborted,
            held,
        }
    }

    pub fn hold_responses(&self, request_id: &str, responses: Vec<Value>) -> Arc<ResponseGate> {
        let gate = Arc::new(ResponseGate {
            empty_sent: watch::channel(false).0,
            release: Notify::new(),
            responses,
        });
        assert!(
            self.held
                .lock()
                .unwrap()
                .insert(request_id.into(), Arc::clone(&gate))
                .is_none()
        );
        gate
    }

    pub async fn wait_aborted(&self, request_id: &str) {
        let mut aborted = self.aborted.clone();
        dynamo_sidecar_testkit::bounded(
            "native HTTP abort request",
            aborted.wait_for(|ids| ids.iter().any(|id| id == request_id)),
        )
        .await
        .unwrap();
    }

    pub fn aborted_requests(&self) -> Vec<String> {
        self.aborted.borrow().clone()
    }

    pub fn port(&self) -> u16 {
        self.server
            .endpoint()
            .rsplit_once(':')
            .unwrap()
            .1
            .parse()
            .unwrap()
    }

    pub async fn shutdown(&mut self) {
        self.server.shutdown().await.unwrap();
    }
}

async fn serve(
    socket: TcpStream,
    control: Controller<Adapter>,
    aborted: watch::Sender<Vec<String>>,
    held: Arc<Mutex<HashMap<String, Arc<ResponseGate>>>>,
) -> anyhow::Result<()> {
    let mut reader = BufReader::new(socket);
    let mut line = String::new();
    if reader.read_line(&mut line).await? == 0 {
        return Ok(());
    }
    let is_health = line.starts_with("GET /health ");
    let is_abort = line.starts_with("POST /abort_request ");
    assert!(
        is_health || is_abort || line.starts_with("POST /generate "),
        "{line}"
    );
    let mut length = 0;
    loop {
        line.clear();
        if reader.read_line(&mut line).await? == 0 {
            return Ok(());
        }
        if line == "\r\n" {
            break;
        }
        if let Some((name, value)) = line.split_once(':')
            && name.eq_ignore_ascii_case("content-length")
        {
            length = value.trim().parse()?;
        }
    }
    if is_health {
        reader
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            .await?;
        return Ok(());
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body).await?;
    let request: Value = serde_json::from_slice(&body)?;
    if is_abort {
        let id = Adapter::request_id(&request);
        assert_eq!(request, json!({"rid": id, "abort_all": false}));
        aborted.send_modify(|ids| ids.push(id.to_string()));
        reader
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            .await?;
        return Ok(());
    }
    let (mut reader, mut writer) = reader.into_inner().into_split();
    let gate = held.lock().unwrap().remove(Adapter::request_id(&request));
    let mut eof = [0];
    let mut has_terminal = false;
    tokio::select! {
        received = reader.read(&mut eof) => {
            assert_eq!(received?, 0, "unexpected bytes after /generate body");
            assert!(
                has_terminal || aborted.borrow().iter().any(|id| id == Adapter::request_id(&request)),
                "unfinished generation disconnected before its abort request"
            );
        }
        result = async {
            let opened = control.open(&request).await?;
            writer.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n").await?;
            let responses = if let Some(gate) = gate {
                let empty = json!({"output_ids": [], "text": "", "meta_info": {"finish_reason": null}});
                writer.write_all(format!("data: {empty}\n\n").as_bytes()).await?;
                gate.empty_sent.send_replace(true);
                gate.release.notified().await;
                gate.responses.clone()
            } else {
                responses(Adapter::request_id(&request)).into()
            };
            let source = futures::stream::iter(responses.into_iter().map(Ok));
            let mut stream = opened.wrap(Box::pin(source));
            while let Some(response) = stream.next().await {
                let response = response?;
                has_terminal = Adapter::is_terminal(&response);
                writer.write_all(format!("data: {response}\n\n").as_bytes()).await?;
            }
            Ok::<_, io::Error>(())
        } => result?,
    }
    Ok(())
}

pub fn responses(request_id: &str) -> [Value; 2] {
    [
        json!({"output_ids": [101], "text": "first", "meta_info": {"id": request_id, "finish_reason": null}}),
        json!({"output_ids": [102], "text": "last", "meta_info": {"id": request_id, "finish_reason": {"type": "length"}}}),
    ]
}

pub struct Adapter;

impl Protocol for Adapter {
    type Request = Value;
    type Response = Value;
    type Error = io::Error;

    fn request_id(request: &Value) -> &str {
        request["rid"].as_str().unwrap()
    }

    fn record_tokens(response: &Value, tokens: &mut Vec<u32>) -> bool {
        let ids = response["output_ids"].as_array().unwrap();
        tokens.extend(
            ids.iter()
                .map(|id| u32::try_from(id.as_u64().unwrap()).unwrap()),
        );
        !ids.is_empty()
    }

    fn is_terminal(response: &Value) -> bool {
        !response["meta_info"]["finish_reason"].is_null()
    }

    fn injected_error(message: &'static str) -> io::Error {
        io::Error::other(message)
    }

    fn invalid_argument_error(message: &'static str) -> io::Error {
        io::Error::new(io::ErrorKind::InvalidInput, message)
    }
}
