// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use dynamo_bench::coding::common::expand_user_path;
use dynamo_bench::coding::mooncake::{MooncakeJsonlWriter, MooncakeRow, RollingHashIdMapper};
use flate2::read::MultiGzDecoder;
use serde::Deserialize;
use serde_json::Value;

#[derive(Parser, Debug)]
#[command(name = "agent_trace_to_mooncake")]
#[command(about = "Convert Dynamo agent trace JSONL/JSONL.GZ records to Mooncake replay JSONL")]
struct Args {
    #[arg(long, action = clap::ArgAction::Append, required = true, num_args = 1..)]
    input_path: Vec<String>,

    #[arg(long)]
    output_file: String,
}

#[derive(Debug, Clone, Deserialize)]
struct AgentTraceRecord {
    event_type: String,
    event_time_unix_ms: u64,
    #[serde(default)]
    request: Option<AgentRequestMetrics>,
}

#[derive(Debug, Clone, Deserialize)]
struct AgentRequestMetrics {
    request_id: String,
    #[serde(default)]
    output_tokens: Option<u64>,
    #[serde(default)]
    request_received_ms: Option<u64>,
    #[serde(default)]
    total_time_ms: Option<f64>,
    #[serde(default)]
    replay: Option<AgentReplayMetrics>,
}

#[derive(Debug, Clone, Deserialize)]
struct AgentReplayMetrics {
    trace_block_size: usize,
    input_length: usize,
    input_sequence_hashes: Vec<u64>,
}

#[derive(Debug, Clone)]
struct RequestEntry {
    start_ms: i64,
    end_ms: i64,
    request: AgentRequestMetrics,
    replay: AgentReplayMetrics,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let input_paths = args
        .input_path
        .iter()
        .map(|path| expand_user_path(path))
        .collect::<Vec<_>>();
    let output_path = expand_user_path(&args.output_file);

    let requests = load_agent_trace(&input_paths)?;
    let (trace_block_size, rows) = build_mooncake_rows(requests)?;
    let mut writer = MooncakeJsonlWriter::create(&output_path, None)?;

    for row in &rows {
        writer.write_row(row)?;
    }

    let stats = writer.finish()?;
    println!(
        "Wrote {} Mooncake rows to {}",
        stats.row_count,
        output_path.display()
    );
    println!("Trace block size: {trace_block_size}");
    Ok(())
}

fn load_agent_trace(paths: &[PathBuf]) -> Result<Vec<RequestEntry>> {
    let mut requests = Vec::new();

    for path in paths {
        let reader = open_trace_reader(path)?;
        for (line_index, line) in reader.lines().enumerate() {
            let line = line
                .with_context(|| format!("failed to read {}:{}", path.display(), line_index + 1))?;
            if line.trim().is_empty() {
                continue;
            }
            let Some(record) = parse_trace_record(&line).with_context(|| {
                format!("failed to parse {}:{}", path.display(), line_index + 1)
            })?
            else {
                continue;
            };
            if record.event_type == "request_end" {
                requests.push(request_entry(record).with_context(|| {
                    format!(
                        "invalid request_end at {}:{}",
                        path.display(),
                        line_index + 1
                    )
                })?);
            }
        }
    }

    if requests.is_empty() {
        bail!("no request_end records with replay fields found");
    }

    Ok(requests)
}

fn open_trace_reader(path: &Path) -> Result<Box<dyn BufRead>> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let reader: Box<dyn Read> = if path.extension().and_then(|ext| ext.to_str()) == Some("gz") {
        Box::new(MultiGzDecoder::new(file))
    } else {
        Box::new(file)
    };
    Ok(Box::new(BufReader::new(reader)))
}

fn parse_trace_record(line: &str) -> Result<Option<AgentTraceRecord>> {
    let value: Value = serde_json::from_str(line)?;
    let event = value.get("event").unwrap_or(&value);
    if !event.is_object() {
        return Ok(None);
    }
    Ok(Some(serde_json::from_value(event.clone())?))
}

fn request_entry(record: AgentTraceRecord) -> Result<RequestEntry> {
    let request = record
        .request
        .ok_or_else(|| anyhow!("request_end record is missing request payload"))?;
    let replay = request
        .replay
        .clone()
        .ok_or_else(|| anyhow!("request payload is missing replay metrics"))?;
    if replay.trace_block_size == 0 {
        bail!("request replay trace_block_size must be greater than 0");
    }
    if replay.input_sequence_hashes.len() * replay.trace_block_size < replay.input_length {
        bail!(
            "input_length {} exceeds replay hash capacity {}",
            replay.input_length,
            replay.input_sequence_hashes.len() * replay.trace_block_size
        );
    }

    let (start_ms, end_ms) = request_times(record.event_time_unix_ms, &request);
    Ok(RequestEntry {
        start_ms,
        end_ms,
        request,
        replay,
    })
}

fn request_times(event_time_unix_ms: u64, request: &AgentRequestMetrics) -> (i64, i64) {
    let total_ms = request
        .total_time_ms
        .map(|value| value.max(0.0).round() as u64)
        .unwrap_or_else(|| {
            event_time_unix_ms
                .saturating_sub(request.request_received_ms.unwrap_or(event_time_unix_ms))
        });
    let end_ms = request
        .request_received_ms
        .map(|start| start.saturating_add(total_ms))
        .unwrap_or(event_time_unix_ms);
    let start_ms = request
        .request_received_ms
        .unwrap_or_else(|| event_time_unix_ms.saturating_sub(total_ms));
    (saturating_i64(start_ms), saturating_i64(end_ms))
}

fn build_mooncake_rows(mut requests: Vec<RequestEntry>) -> Result<(usize, Vec<MooncakeRow>)> {
    let global_start_ms = requests
        .iter()
        .map(|request| request.start_ms)
        .min()
        .ok_or_else(|| anyhow!("no request records to convert"))?;
    let trace_block_size = requests[0].replay.trace_block_size;
    for request in &requests {
        if request.replay.trace_block_size != trace_block_size {
            bail!(
                "mixed replay trace_block_size values are not supported: {} and {}",
                trace_block_size,
                request.replay.trace_block_size
            );
        }
    }

    requests.sort_by(|left, right| {
        (left.start_ms, left.end_ms, &left.request.request_id).cmp(&(
            right.start_ms,
            right.end_ms,
            &right.request.request_id,
        ))
    });

    let mut mapper = RollingHashIdMapper::new(trace_block_size);
    let mut rows = Vec::new();
    for request in requests {
        let hash_ids = mapper.ids_for_sequence_hashes(&request.replay.input_sequence_hashes);
        let output_length = request.request.output_tokens.ok_or_else(|| {
            anyhow!(
                "request {} is missing output length",
                request.request.request_id
            )
        })?;
        rows.push(MooncakeRow {
            session_id: None,
            input_length: request.replay.input_length,
            output_length: usize::try_from(output_length)
                .context("output length does not fit in usize")?,
            hash_ids,
            timestamp: Some(request.start_ms - global_start_ms),
            delay: None,
        });
    }

    Ok((trace_block_size, rows))
}

fn saturating_i64(value: u64) -> i64 {
    value.min(i64::MAX as u64) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(
        request_id: &str,
        start_ms: i64,
        end_ms: i64,
        sequence_hashes: Vec<u64>,
    ) -> RequestEntry {
        RequestEntry {
            start_ms,
            end_ms,
            request: AgentRequestMetrics {
                request_id: request_id.to_string(),
                output_tokens: Some(5),
                request_received_ms: Some(start_ms as u64),
                total_time_ms: Some((end_ms - start_ms) as f64),
                replay: None,
            },
            replay: AgentReplayMetrics {
                trace_block_size: 2,
                input_length: sequence_hashes.len() * 2,
                input_sequence_hashes: sequence_hashes,
            },
        }
    }

    #[test]
    fn converter_preserves_absolute_timestamps_per_request() {
        let requests = vec![
            request("req-a", 1_000, 1_100, vec![11, 22]),
            request("req-b", 1_500, 1_600, vec![11, 33]),
        ];

        let (_, entries) = build_mooncake_rows(requests).unwrap();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].timestamp, Some(0));
        assert_eq!(entries[0].delay, None);
        assert_eq!(entries[1].timestamp, Some(500));
        assert_eq!(entries[1].delay, None);
        assert_eq!(entries[0].session_id, None);
        assert_eq!(entries[1].session_id, None);
        assert_eq!(entries[0].hash_ids[0], entries[1].hash_ids[0]);
    }

    #[test]
    fn converter_preserves_parallel_start_times_as_independent_rows() {
        let requests = vec![
            request("req-a", 1_000, 1_500, vec![11]),
            request("req-b", 1_000, 1_700, vec![22]),
        ];

        let (_, entries) = build_mooncake_rows(requests).unwrap();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].timestamp, Some(0));
        assert_eq!(entries[1].timestamp, Some(0));
        assert_eq!(entries[0].delay, None);
        assert_eq!(entries[1].delay, None);
        assert_eq!(entries[0].session_id, None);
        assert_eq!(entries[1].session_id, None);
    }

    #[test]
    fn request_times_uses_event_time_when_total_duration_is_missing() {
        let request = AgentRequestMetrics {
            request_id: "req".to_string(),
            output_tokens: Some(1),
            request_received_ms: Some(1_000),
            total_time_ms: None,
            replay: None,
        };

        assert_eq!(request_times(1_250, &request), (1_000, 1_250));
    }
}
