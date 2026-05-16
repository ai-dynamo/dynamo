// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{env, net::SocketAddr, time::Duration};

use anyhow::{Context, Result};
use dynamo_protocols::types::realtime::{
    RealtimeClientEvent, RealtimeResponseStatus, RealtimeServerEvent,
    RealtimeServerEventResponseAudioDelta, RealtimeServerEventResponseAudioDone,
    RealtimeServerEventResponseCreated, RealtimeServerEventResponseDone,
    RealtimeServerEventSessionUpdated,
};
use dynamo_runtime::protocols::annotated::Annotated;
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::{TcpStream, tcp::OwnedWriteHalf},
};
use voice_agent_realtime::{
    ECHO_AUDIO_DELTA_CHUNK_LEN, annotated_event, echo_response, unsupported_event,
};

#[derive(Debug)]
struct Args {
    connect: SocketAddr,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut connect = env::var("DYN_REALTIME_FRONTEND_BACKEND_ADDR")
            .unwrap_or_else(|_| "127.0.0.1:8081".to_string());

        let mut argv = env::args().skip(1);
        while let Some(arg) = argv.next() {
            match arg.as_str() {
                "--connect" => connect = argv.next().context("--connect requires a value")?,
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => anyhow::bail!("unknown argument: {other}"),
            }
        }

        Ok(Self {
            connect: connect
                .parse()
                .with_context(|| format!("invalid --connect address: {connect}"))?,
        })
    }
}

fn print_help() {
    println!(
        "Run a local Dynamo realtime backend\n\n\
         Usage: cargo run --manifest-path examples/voice_agent/realtime/Cargo.toml \
         --bin voice-agent-realtime-backend -- [OPTIONS]\n\n\
         Options:\n\
           --connect <HOST:PORT>  Frontend backend-bridge address [default: 127.0.0.1:8081]\n\
           -h, --help             Print help"
    );
}

async fn write_chunk(
    writer: &mut OwnedWriteHalf,
    chunk: &Annotated<RealtimeServerEvent>,
) -> Result<()> {
    let payload = serde_json::to_string(chunk).context("failed to serialize response chunk")?;
    writer
        .write_all(payload.as_bytes())
        .await
        .context("failed to write response chunk")?;
    writer
        .write_all(b"\n")
        .await
        .context("failed to write response frame delimiter")?;
    writer
        .flush()
        .await
        .context("failed to flush response chunk")
}

async fn handle_event(
    writer: &mut OwnedWriteHalf,
    event: RealtimeClientEvent,
    session_id: &str,
    frame: &mut u64,
) -> Result<()> {
    match event {
        RealtimeClientEvent::SessionUpdate(req) => {
            *frame += 1;
            write_chunk(
                writer,
                &annotated_event(
                    *frame,
                    RealtimeServerEvent::SessionUpdated(RealtimeServerEventSessionUpdated {
                        event_id: format!("event_{session_id}_{frame}"),
                        session: req.session,
                    }),
                ),
            )
            .await
        }
        RealtimeClientEvent::InputAudioBufferAppend(req) => {
            let response_id = format!("resp_{session_id}_{frame}");
            let item_id = format!("item_{session_id}_{frame}");

            *frame += 1;
            write_chunk(
                writer,
                &annotated_event(
                    *frame,
                    RealtimeServerEvent::ResponseCreated(RealtimeServerEventResponseCreated {
                        event_id: format!("event_{session_id}_{frame}"),
                        response: echo_response(&response_id, RealtimeResponseStatus::InProgress),
                    }),
                ),
            )
            .await?;

            let audio = req.audio.as_str();
            let mut start = 0;
            while start < audio.len() {
                let mut end = (start + ECHO_AUDIO_DELTA_CHUNK_LEN).min(audio.len());
                while !audio.is_char_boundary(end) {
                    end -= 1;
                }
                *frame += 1;
                write_chunk(
                    writer,
                    &annotated_event(
                        *frame,
                        RealtimeServerEvent::ResponseOutputAudioDelta(
                            RealtimeServerEventResponseAudioDelta {
                                event_id: format!("event_{session_id}_{frame}"),
                                response_id: response_id.clone(),
                                item_id: item_id.clone(),
                                output_index: 0,
                                content_index: 0,
                                delta: audio[start..end].to_string(),
                            },
                        ),
                    ),
                )
                .await?;
                start = end;
            }

            *frame += 1;
            write_chunk(
                writer,
                &annotated_event(
                    *frame,
                    RealtimeServerEvent::ResponseOutputAudioDone(
                        RealtimeServerEventResponseAudioDone {
                            event_id: format!("event_{session_id}_{frame}"),
                            response_id: response_id.clone(),
                            item_id: item_id.clone(),
                            output_index: 0,
                            content_index: 0,
                        },
                    ),
                ),
            )
            .await?;

            *frame += 1;
            write_chunk(
                writer,
                &annotated_event(
                    *frame,
                    RealtimeServerEvent::ResponseDone(RealtimeServerEventResponseDone {
                        event_id: format!("event_{session_id}_{frame}"),
                        response: echo_response(&response_id, RealtimeResponseStatus::Completed),
                    }),
                ),
            )
            .await
        }
        other => {
            *frame += 1;
            write_chunk(writer, &unsupported_event(*frame, session_id, &other)).await
        }
    }
}

async fn serve_connection(stream: TcpStream) -> Result<()> {
    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);
    let mut line = String::new();
    let session_id = "voice-agent-backend";
    let mut frame = 0u64;

    loop {
        line.clear();
        let read = reader
            .read_line(&mut line)
            .await
            .context("failed to read request frame")?;
        if read == 0 {
            return Ok(());
        }

        let event: RealtimeClientEvent =
            serde_json::from_str(line.trim_end()).context("failed to parse request frame")?;
        handle_event(&mut write_half, event, session_id, &mut frame).await?;
    }
}

async fn run_backend(args: Args) -> Result<()> {
    loop {
        match TcpStream::connect(args.connect).await {
            Ok(stream) => {
                println!("Connected realtime backend to tcp://{}", args.connect);
                if let Err(err) = serve_connection(stream).await {
                    eprintln!("backend connection failed: {err:#}");
                } else {
                    eprintln!("backend connection closed");
                }
            }
            Err(err) => {
                eprintln!(
                    "waiting for frontend bridge at tcp://{}: {err}",
                    args.connect
                );
            }
        }

        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()))
        .init();

    let args = Args::parse()?;
    tokio::select! {
        result = run_backend(args) => result,
        result = tokio::signal::ctrl_c() => {
            result.context("failed to listen for Ctrl-C")?;
            Ok(())
        }
    }
}
