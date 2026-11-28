// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Offline replay of the classifier's native model with one history setting per process.

// Include the production model without exposing the predictor or a fit hook to plugins.
#[allow(dead_code)]
#[path = "../src/output_length/predictor.rs"]
mod predictor;
#[allow(dead_code)]
#[path = "../src/output_length/trees.rs"]
mod trees;

use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::time::Instant;

use predictor::{Prediction, Predictor, RIDGE_REFIT, TREES_REFIT, backend_priority};
use serde::Serialize;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Clone, Copy)]
enum History {
    On,
    Off,
}

impl History {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "history" => Ok(Self::On),
            "no-history" => Ok(Self::Off),
            _ => Err(invalid("expected history or no-history").into()),
        }
    }

    fn key(self, session: u64) -> Option<u64> {
        matches!(self, Self::On).then_some(session)
    }
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

struct Pending {
    model: String,
    session: Option<u64>,
    isl: usize,
    prediction: Prediction,
}

#[derive(Default)]
struct Timings {
    prediction: Vec<u64>,
    feedback: Vec<u64>,
    eviction: Vec<u64>,
}

#[derive(Serialize)]
struct Distribution {
    samples: usize,
    p50_ns: u64,
    p99_ns: u64,
    max_ns: u64,
}

fn distribution(mut values: Vec<u64>) -> Option<Distribution> {
    values.sort_unstable();
    let percentile = |p: usize| values[(values.len() * p).div_ceil(100) - 1];
    Some(Distribution {
        samples: values.len(),
        max_ns: *values.last()?,
        p50_ns: percentile(50),
        p99_ns: percentile(99),
    })
}

#[cfg(target_os = "linux")]
fn thread_cpu_ns() -> Option<u64> {
    let mut time = std::mem::MaybeUninit::<libc::timespec>::uninit();
    // A successful call initializes the timespec.
    if unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, time.as_mut_ptr()) } != 0 {
        return None;
    }
    let time = unsafe { time.assume_init() };
    Some(time.tv_sec as u64 * 1_000_000_000 + time.tv_nsec as u64)
}

#[cfg(target_os = "linux")]
fn memory_kib() -> (Option<u64>, Option<u64>) {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return (None, None);
    };
    let value = |key| {
        status
            .lines()
            .find_map(|line| line.strip_prefix(key))?
            .split_ascii_whitespace()
            .next()?
            .parse()
            .ok()
    };
    (value("VmRSS:"), value("VmHWM:"))
}

#[cfg(not(target_os = "linux"))]
fn thread_cpu_ns() -> Option<u64> {
    None
}

#[cfg(not(target_os = "linux"))]
fn memory_kib() -> (Option<u64>, Option<u64>) {
    (None, None)
}

fn measured<T>(samples: &mut Vec<u64>, operation: impl FnOnce() -> T) -> T {
    let begin = thread_cpu_ns();
    let result = operation();
    if let Some((begin, end)) = begin.zip(thread_cpu_ns()) {
        samples.push(end.saturating_sub(begin));
    }
    result
}

#[derive(Serialize)]
struct Metrics {
    predictions: usize,
    feedback_events: usize,
    retained_sessions: usize,
    session_evictions: Option<usize>,
    process_rss_kib: Option<u64>,
    process_peak_rss_kib: Option<u64>,
    prediction_thread_cpu: Option<Distribution>,
    feedback_thread_cpu: Option<Distribution>,
    eviction_feedback_thread_cpu: Option<Distribution>,
}

impl Timings {
    fn finish(
        self,
        predictions: usize,
        feedback_events: usize,
        sessions: usize,
        evictions: Option<usize>,
    ) -> Metrics {
        let (process_rss_kib, process_peak_rss_kib) = memory_kib();
        Metrics {
            predictions,
            feedback_events,
            retained_sessions: sessions,
            session_evictions: evictions,
            process_rss_kib,
            process_peak_rss_kib,
            prediction_thread_cpu: distribution(self.prediction),
            feedback_thread_cpu: distribution(self.feedback),
            eviction_feedback_thread_cpu: distribution(self.eviction),
        }
    }
}

fn replay(events: impl BufRead, mut output: impl Write, history: History) -> Result<Metrics> {
    writeln!(output, "id\tpredicted_tokens\tbackend_priority\tsource")?;
    let start = Instant::now();
    let mut now = start;
    let mut held_out = false;
    let mut models = BTreeMap::<String, Predictor>::new();
    let mut pending = HashMap::<u64, Pending>::new();
    let mut timings = Timings::default();
    let (mut predictions, mut feedback_events) = (0, 0);
    for (line_no, line) in events.lines().enumerate() {
        let line = line?;
        let fields: Vec<_> = line.split('\t').collect();
        match fields.as_slice() {
            ["FIT"] if !held_out && !models.is_empty() && pending.is_empty() => {
                // Give the shipped predictor a refit boundary after training. Hold the logical
                // clock here for scoring so fitted ridge and tree models stay fixed.
                now = start + RIDGE_REFIT.max(TREES_REFIT);
                for model in models.values_mut() {
                    model.predict(None, 1, now);
                }
                for model in models.values_mut() {
                    model.install_tree_fit();
                }
                held_out = true;
            }
            ["P", id, model, session, isl] => {
                let id = id.parse()?;
                let session = history.key(session.parse()?);
                let isl = isl.parse()?;
                let predictor = models
                    .entry((*model).to_owned())
                    .or_insert_with(Predictor::new);
                let prediction = if held_out {
                    measured(&mut timings.prediction, || {
                        predictor.predict(session, isl, now)
                    })
                } else {
                    predictor.predict(session, isl, now)
                };
                if held_out {
                    predictions += 1;
                    writeln!(
                        output,
                        "{id}\t{:.10}\t{}\t{:?}",
                        prediction.tokens,
                        backend_priority(prediction.tokens),
                        prediction.source
                    )?;
                }
                if pending
                    .insert(
                        id,
                        Pending {
                            model: (*model).to_owned(),
                            session,
                            isl,
                            prediction,
                        },
                    )
                    .is_some()
                {
                    return Err(invalid(format!(
                        "line {}: duplicate pending request {id}",
                        line_no + 1
                    ))
                    .into());
                }
            }
            ["C", id, output_tokens] => {
                let id = id.parse()?;
                let request = pending
                    .remove(&id)
                    .ok_or_else(|| invalid(format!("completion without request {id}")))?;
                let out = output_tokens.parse()?;
                let model = models
                    .get_mut(&request.model)
                    .ok_or_else(|| invalid("missing predictor for feedback"))?;
                let mut learn = || {
                    model.learn(
                        request.session,
                        request.isl,
                        out,
                        &request.prediction.features,
                        now,
                    )
                };
                if held_out {
                    feedback_events += 1;
                    measured(&mut timings.feedback, learn);
                } else {
                    learn();
                }
            }
            _ => return Err(invalid(format!("line {}: invalid replay event", line_no + 1)).into()),
        }
    }
    if !held_out || predictions == 0 || !pending.is_empty() {
        return Err(invalid("incomplete training or held-out replay").into());
    }
    output.flush()?;
    let sessions = models.values().map(|model| model.stats().sessions).sum();
    Ok(timings.finish(predictions, feedback_events, sessions, None))
}

fn churn(requests: u64, history: History) -> Result<Metrics> {
    let now = Instant::now();
    let mut model = Predictor::new();
    let mut timings = Timings::default();
    let mut evictions = 0;
    for session in 0..requests {
        let key = history.key(session);
        let prediction = measured(&mut timings.prediction, || model.predict(key, 256, now));
        let before = model.stats().sessions;
        let samples = timings.feedback.len();
        measured(&mut timings.feedback, || {
            model.learn(key, 256, 64, &prediction.features, now)
        });
        if model.stats().sessions < before {
            evictions += 1;
            timings
                .eviction
                .extend_from_slice(&timings.feedback[samples..]);
        }
    }
    let requests = usize::try_from(requests)?;
    Ok(timings.finish(requests, requests, model.stats().sessions, Some(evictions)))
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args()
        .skip(1)
        .filter(|arg| arg != "--bench" && arg != "--test")
        .collect();
    let (metrics, path) = match args.as_slice() {
        [mode, events, history, output, metrics] if mode == "replay" => {
            let history = History::parse(history)?;
            let events = BufReader::new(File::open(events)?);
            let output = BufWriter::new(File::create_new(output)?);
            (replay(events, output, history)?, metrics)
        }
        [mode, requests, history, metrics] if mode == "churn" => {
            (churn(requests.parse()?, History::parse(history)?)?, metrics)
        }
        args if args.is_empty() || args == ["--help"] => {
            println!("output_length_replay replay EVENTS history|no-history PREDICTIONS METRICS");
            println!("output_length_replay churn REQUESTS history|no-history METRICS");
            return Ok(());
        }
        _ => return Err(invalid("invalid output_length_replay arguments, see --help").into()),
    };
    let mut writer = BufWriter::new(File::create_new(path)?);
    serde_json::to_writer_pretty(&mut writer, &metrics)?;
    writeln!(writer)?;
    writer.flush()?;
    Ok(())
}
