// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{env, net::SocketAddr, time::Duration};

use anyhow::{Context, Result};
use dynamo_llm::{
    engines::TOKEN_ECHO_DELAY,
    protocols::openai::chat_completions::{
        NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
    },
};
use dynamo_protocols::types::FinishReason;
use dynamo_runtime::protocols::annotated::Annotated;
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::{TcpStream, tcp::OwnedWriteHalf},
};
use voice_agent_realtime::last_user_text;

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
    chunk: &Annotated<NvCreateChatCompletionStreamResponse>,
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

async fn handle_request(
    writer: &mut OwnedWriteHalf,
    req: NvCreateChatCompletionRequest,
    request_index: u64,
) -> Result<()> {
    let prompt = last_user_text(&req, request_index);
    let mut deltas = req.response_generator(format!("backend-{request_index}"));
    let mut id = 1u64;

    for c in prompt.chars() {
        tokio::time::sleep(*TOKEN_ECHO_DELAY).await;
        let response = deltas.create_choice(0, Some(c.to_string()), None, None);
        write_chunk(
            writer,
            &Annotated {
                id: Some(id.to_string()),
                data: Some(response),
                event: None,
                comment: None,
                error: None,
            },
        )
        .await?;
        id += 1;
    }

    let response = deltas.create_choice(0, None, Some(FinishReason::Stop), None);
    write_chunk(
        writer,
        &Annotated {
            id: Some(id.to_string()),
            data: Some(response),
            event: None,
            comment: None,
            error: None,
        },
    )
    .await
}

async fn serve_connection(stream: TcpStream) -> Result<()> {
    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);
    let mut line = String::new();
    let mut request_index = 0u64;

    loop {
        line.clear();
        let read = reader
            .read_line(&mut line)
            .await
            .context("failed to read request frame")?;
        if read == 0 {
            return Ok(());
        }

        request_index += 1;
        let req: NvCreateChatCompletionRequest =
            serde_json::from_str(line.trim_end()).context("failed to parse request frame")?;
        handle_request(&mut write_half, req, request_index).await?;
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
