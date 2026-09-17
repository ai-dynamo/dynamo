// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reads one shadow tap topic and keeps running totals.
//!
//! It stands in for a shadow deployment: it proves that the tap delivers every
//! request, in the form the workers see it, to a process outside the frontend.
//! For a request-only tap it counts prompt tokens. For a joined tap it also
//! counts output tokens and outcomes and tracks arrival gaps.
//!
//! The runtime is configured from the same `DYN_*` environment as the
//! frontend (`DYN_DISCOVERY_BACKEND`, `DYN_EVENT_PLANE`, ...).

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::Parser;
use dynamo_llm::shadow::{ShadowEnvelope, ShadowOutcome};
use dynamo_runtime::transports::event_plane::EventSubscriber;
use dynamo_runtime::{DistributedRuntime, Runtime, Worker, logging};
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(about = "Count what a Dynamo shadow tap publishes")]
struct Args {
    /// Topic of the tap to read.
    #[arg(long)]
    topic: String,
    /// Namespace the tap publishes in.
    #[arg(long, default_value = "dynamo")]
    namespace: String,
    /// Seconds between progress lines.
    #[arg(long, default_value_t = 1.0)]
    report_interval: f64,
    /// Rewrite this file with the totals, as JSON, at every report.
    #[arg(long)]
    summary_path: Option<PathBuf>,
}

#[derive(Debug, Default, Serialize)]
struct Totals {
    records: u64,
    prompt_tokens: u64,
    output_tokens: u64,
    responses: u64,
    complete: u64,
    cancelled: u64,
    error: u64,
    /// Records the tap dropped because its queue was full.
    tap_gaps: u64,
    /// Records the event plane lost between the tap and this process.
    transport_gaps: u64,
    arrival_span_ms: f64,
}

/// A contiguous run of sequence numbers, counted without assuming order.
/// Concurrent requests reach the tap queue slightly out of `seq` order, so
/// "expected next" arithmetic reports gaps that are not there.
#[derive(Default)]
struct SeqRange {
    first: Option<u64>,
    last: u64,
    seen: u64,
}

impl SeqRange {
    fn observe(&mut self, seq: u64) {
        let first = *self.first.get_or_insert(seq);
        self.first = Some(first.min(seq));
        self.last = self.last.max(seq);
        self.seen += 1;
    }

    /// A subscriber that joins late starts its range at the first record it
    /// sees, so earlier records do not count as missing.
    fn missing(&self) -> u64 {
        self.first
            .map_or(0, |first| (self.last - first + 1).saturating_sub(self.seen))
    }
}

/// What is known about one publisher, which is one tap in one frontend.
#[derive(Default)]
struct PublisherState {
    tap_seq: SeqRange,
    transport_seq: SeqRange,
    first_arrival_ns: u64,
    last_arrival_ns: u64,
}

impl Totals {
    fn record(
        &mut self,
        state: &mut PublisherState,
        transport_seq: u64,
        envelope: &ShadowEnvelope,
    ) {
        self.records += 1;
        self.prompt_tokens += envelope.request.token_ids.len() as u64;
        state.tap_seq.observe(envelope.seq);
        state.transport_seq.observe(transport_seq);

        if state.first_arrival_ns == 0 {
            state.first_arrival_ns = envelope.arrival_unix_ns;
        }
        state.first_arrival_ns = state.first_arrival_ns.min(envelope.arrival_unix_ns);
        state.last_arrival_ns = state.last_arrival_ns.max(envelope.arrival_unix_ns);
        let span = (state.last_arrival_ns - state.first_arrival_ns) as f64 / 1e6;
        self.arrival_span_ms = self.arrival_span_ms.max(span);

        if let Some(response) = &envelope.response {
            self.responses += 1;
            self.output_tokens += response.output_tokens;
            match response.outcome {
                ShadowOutcome::Complete => self.complete += 1,
                ShadowOutcome::Cancelled => self.cancelled += 1,
                ShadowOutcome::Error => self.error += 1,
            }
        }
    }

    fn count_gaps(&mut self, publishers: &HashMap<u64, PublisherState>) {
        self.tap_gaps = publishers
            .values()
            .map(|state| state.tap_seq.missing())
            .sum();
        self.transport_gaps = publishers
            .values()
            .map(|state| state.transport_seq.missing())
            .sum();
    }
}

fn main() -> Result<()> {
    logging::init();
    let worker = Worker::from_settings()?;
    worker.execute(app)
}

async fn app(runtime: Runtime) -> Result<()> {
    let args = Args::parse();
    let drt = DistributedRuntime::from_settings(runtime.clone()).await?;
    let namespace = drt.namespace(args.namespace.clone())?;
    let mut subscriber = EventSubscriber::for_namespace(&namespace, args.topic.clone())
        .await?
        .typed::<ShadowEnvelope>();
    println!(
        "subscribed namespace={} topic={}",
        args.namespace, args.topic
    );

    let mut totals = Totals::default();
    let mut publishers: HashMap<u64, PublisherState> = HashMap::new();
    let mut report = tokio::time::interval(Duration::from_secs_f64(args.report_interval));
    let mut last = (Instant::now(), 0u64, 0u64);
    let shutdown = runtime.primary_token();

    loop {
        tokio::select! {
            _ = shutdown.cancelled() => break,
            _ = report.tick() => {
                if totals.records != last.1 {
                    totals.count_gaps(&publishers);
                    let seconds = last.0.elapsed().as_secs_f64();
                    println!(
                        "records={} prompt_tokens={} output_tokens={} ({:.0} req/s, {:.0} prompt tok/s) tap_gaps={} transport_gaps={}",
                        totals.records,
                        totals.prompt_tokens,
                        totals.output_tokens,
                        (totals.records - last.1) as f64 / seconds,
                        (totals.prompt_tokens - last.2) as f64 / seconds,
                        totals.tap_gaps,
                        totals.transport_gaps,
                    );
                    write_summary(&args, &totals)?;
                }
                last = (Instant::now(), totals.records, totals.prompt_tokens);
            }
            event = subscriber.next() => match event {
                Some(Ok((transport, envelope))) => {
                    let state = publishers.entry(transport.publisher_id).or_default();
                    totals.record(state, transport.sequence, &envelope);
                }
                Some(Err(error)) => eprintln!("undecodable record: {error}"),
                None => break,
            },
        }
    }

    totals.count_gaps(&publishers);
    write_summary(&args, &totals)?;
    println!("{}", serde_json::to_string_pretty(&totals)?);
    Ok(())
}

fn write_summary(args: &Args, totals: &Totals) -> Result<()> {
    let Some(path) = &args.summary_path else {
        return Ok(());
    };
    let partial = path.with_extension("tmp");
    let mut file = std::fs::File::create(&partial)?;
    file.write_all(&serde_json::to_vec_pretty(totals)?)?;
    std::fs::rename(partial, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::SeqRange;

    #[test]
    fn reordered_sequence_numbers_are_not_gaps() {
        let mut range = SeqRange::default();
        for seq in [4, 6, 5, 7] {
            range.observe(seq);
        }
        assert_eq!(range.missing(), 0);
        range.observe(10);
        assert_eq!(range.missing(), 2);
    }
}
