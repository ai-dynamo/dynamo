// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The replay corpus: lanes replay a trace's plays, the mocker simulates them
//! offline, and the result is compacted into each request's timing.
//!
//! A lane is one closed-loop conversation slot, like one unit of aiperf
//! concurrency: it replays a play's requests in the order they started, one at a
//! time (a Weka play's subagents run in series with it), and starts another play
//! when one ends. Every copy of a play gets disjoint hash IDs, the way aiperf's
//! cache-bust prefixes make repeated plays unique.
//!
//! Only timing comes from the mocker: when each request arrives, produces its first
//! output, and completes. Where its KV blocks live is decided at replay by the router
//! (see `kv_cache`), so the lanes are simply pinned to engine ranks here. The mocker
//! runs a window past the measured window so that window requests finish and
//! are recorded; requests arriving after the window are dropped.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result, bail, ensure};
use dynamo_bench::kv_router_common::progress::make_progress_bar;
use dynamo_mocker::common::protocols::MockerConfig;
use dynamo_mocker::loadgen::{SessionTrace, Trace, TurnTrace, load_weka_agentic_rows};
use dynamo_mocker::replay::{
    ReplayRouterMode, ReplayTerminalStatus, SlaThresholds,
    simulate_loaded_trace_with_router_mode_and_options,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum TraceFormat {
    /// A Weka agentic corpus directory, such as the AgentX traces.
    Weka,
    /// A Mooncake JSONL trace.
    Mooncake,
}

/// Everything that shapes the corpus; the defaults are the `agentx` workload. Two
/// runs with equal parameters and the same trace replay the same requests.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, clap::Args)]
pub struct WorkloadParams {
    #[arg(long, value_enum, default_value = "weka")]
    pub trace_format: TraceFormat,
    /// Tokens per trace hash ID. Weka traces carry their own; Mooncake needs it given.
    #[arg(long)]
    pub trace_block_size: Option<usize>,
    /// Concurrent conversation slots, like aiperf concurrency.
    #[arg(long, default_value_t = 6144)]
    pub lanes: usize,
    #[arg(long, default_value_t = 2048)]
    pub workers: usize,
    /// DP ranks per worker; each rank is one engine.
    #[arg(long, default_value_t = 1)]
    pub dp_size: u32,
    #[arg(long, default_value_t = 16)]
    pub block_size: u32,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub ignore_trace_delays: bool,
    /// Trace time replayed before the measured window.
    #[arg(long, default_value_t = 30_000)]
    pub warmup_ms: u64,
    /// Trace time measured.
    #[arg(long, default_value_t = 90_000)]
    pub window_ms: u64,
    /// Per-rank KV capacity in blocks, for both the mocker's timing and the replay's
    /// caches: about 1M tokens, so caches evict.
    #[arg(long, default_value_t = 65_536)]
    pub num_gpu_blocks: usize,
    /// Engine batch limits, from the DeepSeek-V4.1 vLLM agentic recipe.
    #[arg(long, default_value_t = 16_384)]
    pub max_num_batched_tokens: usize,
    #[arg(long, default_value_t = 1_024)]
    pub max_num_seqs: usize,
    #[arg(long, default_value_t = 10.0)]
    pub engine_speedup: f64,
}

impl WorkloadParams {
    pub fn ranks(&self) -> usize {
        self.workers * self.dp_size as usize
    }
}

/// One conversation from the trace, with hash IDs local to the play.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Play {
    /// When the play starts, relative to the trace's start; zero with `ignore_trace_delays`.
    pub start_ms: f64,
    pub turns: Vec<PlayTurn>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlayTurn {
    pub input_length: u32,
    pub output_length: u32,
    /// Recorded gap after the previous turn; zero with `ignore_trace_delays`.
    pub delay_ms: f64,
    pub hash_ids: Vec<u32>,
}

/// One replay of a play on a lane, with its hash IDs offset by `base`.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct PlayCopy {
    pub play: u32,
    pub base: u32,
}

/// Turns a closed-loop lane chains plays up to, more than a measured window reaches.
const LANE_TURNS: usize = 2048;

/// One request's timing, in trace microseconds.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ReplayRequest {
    pub arrival_us: u64,
    pub first_output_us: Option<u64>,
    pub completed_us: Option<u64>,
    pub lane: u32,
    /// Which of the lane's play copies, and the turn within that play.
    pub copy: u32,
    pub turn: u32,
    pub input_length: u32,
    pub output_length: u32,
}

/// What the mocker left out of the corpus.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Omitted {
    /// Requests the engine refused or failed.
    pub rejected: usize,
    /// Window requests still running when the mocker stopped, which it does not
    /// record; their lanes end there.
    pub unfinished: usize,
}

#[derive(Serialize, Deserialize)]
pub struct Corpus {
    pub params: WorkloadParams,
    pub trace_block_size: usize,
    pub plays: Vec<Play>,
    /// The plays each lane replays back to back.
    pub lanes: Vec<Vec<PlayCopy>>,
    /// Sorted by arrival.
    pub requests: Vec<ReplayRequest>,
    pub omitted: Omitted,
    /// Computed from the contents, never saved.
    #[serde(skip)]
    pub digest: String,
}

impl Corpus {
    /// The prompt tokens `request` sends, built exactly as the mocker built them.
    pub fn prompt_tokens(&self, request: &ReplayRequest) -> Result<Vec<u32>> {
        let copy = self.lanes[request.lane as usize][request.copy as usize];
        TurnTrace {
            input_length: request.input_length as usize,
            hash_ids: self.plays[copy.play as usize].turns[request.turn as usize]
                .hash_ids
                .iter()
                .map(|id| id + copy.base)
                .collect(),
            ..TurnTrace::default()
        }
        .synthesize_tokens(self.trace_block_size)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut file = std::io::BufWriter::new(
            std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?,
        );
        rmp_serde::encode::write(&mut file, self).context("write corpus")?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let file = std::io::BufReader::new(
            std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?,
        );
        let mut corpus: Self = rmp_serde::from_read(file).context("read corpus")?;
        corpus.digest = digest(&corpus)?;
        Ok(corpus)
    }

    /// Requests per second the corpus generates inside the window, in trace time.
    pub fn window_rate(&self) -> Result<f64> {
        let params = &self.params;
        let start = params.warmup_ms * 1_000;
        let end = start + params.window_ms * 1_000;
        let count = self
            .requests
            .iter()
            .filter(|request| (start..end).contains(&request.arrival_us))
            .count();
        ensure!(
            count > 0,
            "the corpus has no requests in its measured window"
        );
        Ok(count as f64 * 1_000.0 / params.window_ms as f64)
    }
}

/// Load the trace and generate the corpus with `threads` concurrent mocker runs.
pub fn generate(trace: &Path, params: &WorkloadParams, threads: usize) -> Result<Corpus> {
    let (trace_block_size, mut plays, hash_scope) = load_plays(trace, params)?;
    ensure!(!plays.is_empty(), "the trace has no usable plays");
    if params.ignore_trace_delays {
        for play in &mut plays {
            play.start_ms = 0.0;
            for turn in &mut play.turns {
                turn.delay_ms = 0.0;
            }
        }
    }
    ensure!(params.lanes > 0, "--lanes must be positive");
    let lanes = lane_copies(&plays, params.lanes, hash_scope)?;
    let ranks = params.ranks();
    ensure!(ranks > 0, "--workers and --dp-size must be positive");

    let simulator = Simulator {
        params,
        trace_block_size,
        plays: &plays,
        lanes: &lanes,
    };
    let next_rank = AtomicUsize::new(0);
    let outputs = Mutex::new((0..ranks).map(|_| None).collect::<Vec<_>>());
    let progress = make_progress_bar(Some(ranks as u64));
    std::thread::scope(|scope| -> Result<()> {
        let workers: Vec<_> = (0..threads.max(1).min(ranks))
            .map(|_| {
                scope.spawn(|| -> Result<()> {
                    loop {
                        let rank = next_rank.fetch_add(1, Ordering::Relaxed);
                        if rank >= ranks {
                            return Ok(());
                        }
                        let output = simulator
                            .simulate_rank(rank)
                            .with_context(|| format!("engine rank {rank}"))?;
                        outputs.lock().unwrap()[rank] = Some(output);
                        progress.inc(1);
                    }
                })
            })
            .collect();
        for worker in workers {
            worker.join().expect("corpus generator panicked")?;
        }
        Ok(())
    })?;
    progress.finish_and_clear();

    let mut requests = Vec::new();
    let mut omitted = Omitted::default();
    for (rank_requests, rank_omitted) in outputs.into_inner().unwrap().into_iter().flatten() {
        requests.extend(rank_requests);
        omitted.rejected += rank_omitted.rejected;
        omitted.unfinished += rank_omitted.unfinished;
    }
    requests.sort_by_key(|request| (request.arrival_us, request.lane, request.copy, request.turn));

    let mut corpus = Corpus {
        params: params.clone(),
        trace_block_size,
        plays,
        lanes,
        requests,
        omitted,
        digest: String::new(),
    };
    corpus.digest = digest(&corpus)?;
    Ok(corpus)
}

/// Whether hash IDs are local to each play (Weka) or shared across the trace (Mooncake).
#[derive(Clone, Copy)]
enum HashScope {
    PerPlay,
    Shared { span: u32 },
}

fn load_plays(trace: &Path, params: &WorkloadParams) -> Result<(usize, Vec<Play>, HashScope)> {
    match params.trace_format {
        TraceFormat::Weka => {
            let (summary, rows) = load_weka_agentic_rows(trace)
                .with_context(|| format!("load Weka trace {}", trace.display()))?;
            let block_size = summary.header.block_size;
            if let Some(expected) = params.trace_block_size {
                ensure!(
                    expected == block_size,
                    "--trace-block-size {expected} does not match the trace's {block_size}"
                );
            }
            // Group by play in corpus order; each play's requests in start order.
            let mut order: Vec<String> = Vec::new();
            let mut by_play: HashMap<String, Vec<_>> = HashMap::new();
            for row in rows {
                by_play
                    .entry(row.play_id.clone())
                    .or_insert_with(|| {
                        order.push(row.play_id.clone());
                        Vec::new()
                    })
                    .push(row);
            }
            let mut plays = Vec::with_capacity(order.len());
            for play_id in order {
                let mut rows = by_play.remove(&play_id).unwrap();
                rows.sort_by(|a, b| a.not_before_ms.total_cmp(&b.not_before_ms));
                let mut ids: HashMap<u64, u32> = HashMap::new();
                let mut turns = Vec::with_capacity(rows.len());
                let mut previous_ms: Option<f64> = None;
                for row in rows {
                    let (Some(input), Some(output), Some(hash_ids)) =
                        (row.input_length, row.output_length, row.hash_ids)
                    else {
                        continue;
                    };
                    let delay_ms =
                        previous_ms.map_or(0.0, |previous| (row.not_before_ms - previous).max(0.0));
                    previous_ms = Some(row.not_before_ms);
                    let hash_ids = hash_ids
                        .into_iter()
                        .map(|id| {
                            let next = ids.len() as u32;
                            *ids.entry(id).or_insert(next)
                        })
                        .collect();
                    turns.push(PlayTurn {
                        input_length: u32::try_from(input)?,
                        output_length: u32::try_from(output.max(1))?,
                        delay_ms,
                        hash_ids,
                    });
                }
                if !turns.is_empty() {
                    plays.push(Play {
                        start_ms: 0.0,
                        turns,
                    });
                }
            }
            Ok((block_size, plays, HashScope::PerPlay))
        }
        TraceFormat::Mooncake => {
            let Some(block_size) = params.trace_block_size else {
                bail!("--trace-block-size is required for Mooncake traces");
            };
            // Arrivals relative to the trace's first, as the measured window is.
            let trace = Trace::from_mooncake(trace, block_size)
                .and_then(Trace::normalize_session_starts)
                .with_context(|| format!("load Mooncake trace {}", trace.display()))?;
            let mut span = 0u32;
            let plays = trace
                .sessions
                .into_iter()
                .map(|session| {
                    let turns = session
                        .turns
                        .into_iter()
                        .map(|turn| {
                            span = turn
                                .hash_ids
                                .iter()
                                .fold(span, |span, &id| span.max(id + 1));
                            Ok(PlayTurn {
                                input_length: u32::try_from(turn.input_length)?,
                                output_length: u32::try_from(turn.max_output_tokens.max(1))?,
                                delay_ms: turn.delay_after_previous_ms,
                                hash_ids: turn.hash_ids,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    Ok(Play {
                        start_ms: session.first_arrival_timestamp_ms.unwrap_or(0.0),
                        turns,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok((block_size, plays, HashScope::Shared { span }))
        }
    }
}

/// The plays each lane replays back to back, as an aiperf slot takes the next
/// conversation. With per-play hash IDs every copy gets its own range; with shared
/// IDs each pass over the trace does, so plays keep the prefixes they share.
fn lane_copies(plays: &[Play], lanes: usize, scope: HashScope) -> Result<Vec<Vec<PlayCopy>>> {
    let spans: Vec<u64> = plays
        .iter()
        .map(|play| {
            play.turns
                .iter()
                .flat_map(|turn| turn.hash_ids.iter())
                .map(|&id| u64::from(id) + 1)
                .max()
                .unwrap_or(0)
        })
        .collect();
    let mut next = 0u64;
    (0..lanes)
        .map(|lane| {
            let mut copies = Vec::new();
            let mut turns = 0;
            while turns < LANE_TURNS {
                let n = lane + copies.len() * lanes;
                let play = n % plays.len();
                let base = match scope {
                    HashScope::PerPlay => {
                        next += spans[play];
                        next - spans[play]
                    }
                    HashScope::Shared { span } => (n / plays.len()) as u64 * u64::from(span),
                };
                u32::try_from(base + spans[play]).context("lanes exceed the u32 hash-ID space")?;
                copies.push(PlayCopy {
                    play: play as u32,
                    base: base as u32,
                });
                turns += plays[play].turns.len();
            }
            Ok(copies)
        })
        .collect()
}

struct Simulator<'a> {
    params: &'a WorkloadParams,
    trace_block_size: usize,
    plays: &'a [Play],
    lanes: &'a [Vec<PlayCopy>],
}

impl Simulator<'_> {
    /// Simulate the lanes pinned to `rank`: their requests up to the window's end,
    /// and what was left out.
    fn simulate_rank(&self, rank: usize) -> Result<(Vec<ReplayRequest>, Omitted)> {
        let params = self.params;
        let window_end_ms = (params.warmup_ms + params.window_ms) as f64;
        let lanes: Vec<usize> = (rank..self.lanes.len()).step_by(params.ranks()).collect();
        // Each lane's turns in session order, as (copy, turn within its play).
        let positions: Vec<Vec<(u32, u32)>> = lanes
            .iter()
            .map(|&lane| {
                self.lanes[lane]
                    .iter()
                    .enumerate()
                    .flat_map(|(copy, play_copy)| {
                        let turns = self.plays[play_copy.play as usize].turns.len() as u32;
                        (0..turns).map(move |turn| (copy as u32, turn))
                    })
                    .collect()
            })
            .collect();
        let sessions: Vec<SessionTrace> = lanes
            .iter()
            .map(|&lane| {
                let copies = &self.lanes[lane];
                let turns = copies
                    .iter()
                    .flat_map(|play_copy| {
                        self.plays[play_copy.play as usize]
                            .turns
                            .iter()
                            .map(move |turn| TurnTrace {
                                input_length: turn.input_length as usize,
                                max_output_tokens: turn.output_length as usize,
                                hash_ids: turn
                                    .hash_ids
                                    .iter()
                                    .map(|id| id + play_copy.base)
                                    .collect(),
                                delay_after_previous_ms: turn.delay_ms,
                                ..TurnTrace::default()
                            })
                    })
                    .collect();
                SessionTrace {
                    session_id: lane.to_string(),
                    first_arrival_timestamp_ms: Some(self.plays[copies[0].play as usize].start_ms),
                    turns,
                }
            })
            .collect();
        // The mocker starts each simulation at its earliest session.
        let Some(offset_ms) = sessions
            .iter()
            .filter_map(|session| session.first_arrival_timestamp_ms)
            .min_by(f64::total_cmp)
        else {
            return Ok((Vec::new(), Omitted::default()));
        };
        let stop_ms = window_end_ms + params.window_ms as f64 - offset_ms;
        if stop_ms <= 0.0 {
            return Ok((Vec::new(), Omitted::default()));
        }

        let engine = MockerConfig::from_value(serde_json::json!({
            "engine": {
                "num_gpu_blocks": params.num_gpu_blocks,
                "block_size": params.block_size,
                "speedup_ratio": params.engine_speedup,
                // The replay's caches always reuse prefixes: KV routing needs them.
                "enable_prefix_caching": true,
                "max_num_batched_tokens": params.max_num_batched_tokens,
                "max_num_seqs": params.max_num_seqs,
            }
        }))?;
        let report = simulate_loaded_trace_with_router_mode_and_options(
            engine,
            None,
            None,
            Trace {
                block_size: self.trace_block_size,
                sessions,
            },
            1,
            1.0,
            ReplayRouterMode::RoundRobin,
            true,
            Some(stop_ms),
            SlaThresholds::default(),
        )?;

        // Back to trace time, in microseconds.
        let micros = |local_ms: f64| ((local_ms + offset_ms) * 1_000.0) as u64;
        let window_end_us = (window_end_ms * 1_000.0) as u64;
        let mut requests = Vec::with_capacity(report.per_request.len());
        let mut omitted = Omitted::default();
        // The last recorded turn of each lane, and when it completed.
        let mut last: Vec<Option<(usize, u64)>> = vec![None; lanes.len()];
        for record in &report.per_request {
            let index = record
                .session_id
                .as_deref()
                .and_then(|id| id.parse::<usize>().ok())
                .and_then(|lane| lanes.iter().position(|&pinned| pinned == lane))
                .context("the mocker recorded a request with no lane")?;
            let position = record
                .turn_index
                .context("the mocker recorded no turn index")?;
            let (copy, turn) = positions[index][position];
            let arrival_us = micros(record.arrival_time_ms);
            let completed_us = micros(record.terminal_time_ms);
            if last[index].is_none_or(|(previous, _)| position > previous) {
                last[index] = Some((position, completed_us));
            }
            if record.terminal_status != ReplayTerminalStatus::Completed {
                omitted.rejected += 1;
                continue;
            }
            if arrival_us >= window_end_us {
                continue;
            }
            requests.push(ReplayRequest {
                arrival_us,
                first_output_us: record.first_token_ms.map(micros),
                completed_us: Some(completed_us),
                lane: lanes[index] as u32,
                copy,
                turn,
                input_length: u32::try_from(record.input_length)?,
                output_length: u32::try_from(record.requested_output_length)?,
            });
        }
        // A lane whose next turn was due inside the window, but has no record, was
        // still running when the mocker stopped.
        for (index, last) in last.iter().enumerate() {
            let next = last.map_or(0, |(position, _)| position + 1);
            let Some(&(copy, turn)) = positions[index].get(next) else {
                // The lane ran out of chained turns. Inside the window it would idle
                // from here, and the load would fall below `--lanes` unseen.
                ensure!(
                    last.is_none_or(|(_, completed_us)| completed_us >= window_end_us),
                    "lane {} finished all {} of its chained turns before the window ended; \
                     shorten --window-ms or lower --engine-speedup",
                    lanes[index],
                    positions[index].len()
                );
                continue;
            };
            let copies = &self.lanes[lanes[index]];
            let due_us = match last {
                Some((_, completed_us)) => {
                    let play = &self.plays[copies[copy as usize].play as usize];
                    completed_us + (play.turns[turn as usize].delay_ms * 1_000.0) as u64
                }
                None => (self.plays[copies[0].play as usize].start_ms * 1_000.0) as u64,
            };
            if due_us < window_end_us {
                omitted.unfinished += 1;
            }
        }
        Ok((requests, omitted))
    }
}

/// A digest over everything replayed, so A/B arms can confirm they ran the same corpus.
fn digest(corpus: &Corpus) -> Result<String> {
    // Serialized straight into the hasher, without a second copy of the corpus.
    let mut hasher = std::io::BufWriter::new(blake3::Hasher::new());
    let replayed = (
        &corpus.params,
        corpus.trace_block_size,
        &corpus.plays,
        &corpus.lanes,
        &corpus.requests,
    );
    rmp_serde::encode::write(&mut hasher, &replayed).context("hash the corpus")?;
    let hasher = hasher
        .into_inner()
        .map_err(std::io::IntoInnerError::into_error)?;
    Ok(hasher.finalize().to_hex().to_string())
}
