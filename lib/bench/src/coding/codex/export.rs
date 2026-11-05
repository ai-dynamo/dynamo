// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Codex request-trace export: request ordering, cross-thread causality, and replay rows.

use crate::coding::codex::parser::{CodexTool, RolloutThread, SkipReason, parse_rollout_file};
use crate::coding::common::anonymized_session_id;
use crate::coding::replay::{
    AgentContext, HARNESS_EVENT_SOURCE, PrefixPool, REQUEST_TRACE_SCHEMA, ReplayBase,
    ReplayDependency, ReplayFields, RequestEndEvent, RequestFields, ToolEvent, ToolFields,
    TraceLine, prefix_pool_key, synthetic_stream_seed, usage_shaped_tokens,
};
use anyhow::{Context, Result, bail};
use dynamo_data_gen::{AgenticDependencyRelation, AgenticDependencyTrigger};
use rustc_hash::FxHashMap;
use serde::Serialize;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Clone, Copy)]
pub struct ExportConfig {
    pub block_size: usize,
    pub preserve_session_ids: bool,
}

/// What the export kept, dropped, and linked, for judging how faithful a replay can be.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExportReport {
    pub files: usize,
    pub skipped_without_ordinals: usize,
    pub skipped_without_usage: usize,
    pub duplicate_threads: usize,
    pub threads: usize,
    pub requests: usize,
    pub compactions: usize,
    pub tools: usize,
    pub preempted_responses: usize,
    pub aborted_responses: usize,
    pub untracked_responses: usize,
    pub unmatched_compactions: usize,
    pub task_edges: usize,
    pub wait_edges: usize,
    pub reattributed_calls: usize,
    pub unlinked_waits: usize,
    pub undelivered_tasks: usize,
    pub unresolved_agents: usize,
    pub cache_prefix_blocks: usize,
    pub forked_prefix_blocks: usize,
    pub pooled_prefix_blocks: usize,
}

impl ExportReport {
    pub fn render(&self) -> String {
        format!(
            "Files: rollouts={} skipped_without_ordinals={} skipped_without_usage={} duplicate_threads={}\n\
             Requests: threads={} requests={} compactions={} tools={} cache_prefix_blocks={} forked_prefix_blocks={} pooled_prefix_blocks={}\n\
             Causality: task_edges={} wait_edges={} reattributed_calls={} unlinked_waits={} undelivered_tasks={} unresolved_agents={}\n\
             Limitations: synthetic_kv_hashes={} dropped_preempted_responses={} dropped_aborted_responses={} dropped_untracked_responses={} unmatched_compactions={}",
            self.files,
            self.skipped_without_ordinals,
            self.skipped_without_usage,
            self.duplicate_threads,
            self.threads,
            self.requests,
            self.compactions,
            self.tools,
            self.cache_prefix_blocks,
            self.forked_prefix_blocks,
            self.pooled_prefix_blocks,
            self.task_edges,
            self.wait_edges,
            self.reattributed_calls,
            self.unlinked_waits,
            self.undelivered_tasks,
            self.unresolved_agents,
            self.requests,
            self.preempted_responses,
            self.aborted_responses,
            self.untracked_responses,
            self.unmatched_compactions,
        )
    }
}

/// Parses rollout files on `workers` threads, keeping one copy of each Codex thread.
///
/// Codex moves rollouts into `archived_sessions` while it runs, so one thread can be discovered
/// twice; the copy with more recorded requests wins.
pub fn load_threads(
    paths: &[PathBuf],
    workers: usize,
    report: &mut ExportReport,
) -> Result<Vec<RolloutThread>> {
    let next = AtomicUsize::new(0);
    let mut parsed = std::thread::scope(|scope| {
        let handles = (0..workers.max(1))
            .map(|_| {
                scope.spawn(|| {
                    let mut local = Vec::new();
                    while let Some(path) = paths.get(next.fetch_add(1, Ordering::Relaxed)) {
                        local.push((path, parse_rollout_file(path)));
                    }
                    local
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("rollout parser panicked"))
            .collect::<Vec<_>>()
    });
    parsed.sort_by(|left, right| left.0.cmp(right.0));

    report.files = paths.len();
    let mut by_thread: FxHashMap<String, RolloutThread> = FxHashMap::default();
    for (_, outcome) in parsed {
        let thread = match outcome? {
            Ok(thread) => thread,
            Err(SkipReason::NoOrdinals) => {
                report.skipped_without_ordinals += 1;
                continue;
            }
            Err(SkipReason::NoUsageRecords) => {
                report.skipped_without_usage += 1;
                continue;
            }
        };
        if let Some(existing) = by_thread.get(&thread.thread_id) {
            report.duplicate_threads += 1;
            if existing.requests.len() >= thread.requests.len() {
                continue;
            }
        }
        by_thread.insert(thread.thread_id.clone(), thread);
    }
    let mut threads = by_thread.into_values().collect::<Vec<_>>();
    threads.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(threads)
}

/// A message queued for a thread, with the request whose completion queued it.
#[derive(Debug, Clone, Copy)]
struct Queued {
    queued_at_ms: i64,
    sender: (usize, usize),
    kind: QueuedKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QueuedKind {
    /// `spawn_agent`: the recipient's first turn.
    Spawn,
    /// `followup_task`: a new task that starts a turn.
    Task,
    /// `send_message` or a child's final answer: only queued.
    Message,
}

/// An edge from a request to the request it waited on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Edge {
    sender: (usize, usize),
    relation: AgenticDependencyRelation,
}

/// Cross-thread edges by thread and request, and the request that spawned each thread.
struct Links {
    edges: Vec<Vec<Vec<Edge>>>,
    spawned_by: FxHashMap<usize, (usize, usize)>,
}

/// The response that emitted a tool call. A call in a response Codex never recorded is
/// attributed to the latest earlier recorded request of the same thread.
fn call_source(
    thread: &RolloutThread,
    tool: &CodexTool,
    report: &mut ExportReport,
) -> Option<usize> {
    if let Some(source) = tool.source_request {
        return Some(source);
    }
    let source = thread
        .requests
        .iter()
        .rposition(|request| request.start_ms <= tool.started_at_ms)?;
    report.reattributed_calls += 1;
    Some(source)
}

fn link_threads(threads: &[RolloutThread], report: &mut ExportReport) -> Links {
    let thread_index = threads
        .iter()
        .enumerate()
        .map(|(index, thread)| (thread.thread_id.as_str(), index))
        .collect::<FxHashMap<_, _>>();
    let mut inboxes = vec![Vec::<Queued>::new(); threads.len()];
    for (sender, thread) in threads.iter().enumerate() {
        let tools_by_call = thread
            .tools
            .iter()
            .map(|tool| (tool.call_id.as_str(), tool))
            .collect::<FxHashMap<_, _>>();
        for send in &thread.sends {
            let Some(&recipient) = thread_index.get(send.recipient_thread_id.as_str()) else {
                report.unresolved_agents += 1;
                continue;
            };
            let Some(tool) = tools_by_call.get(send.call_id.as_str()) else {
                continue;
            };
            let kind = match tool.name.as_str() {
                "spawn_agent" => QueuedKind::Spawn,
                "followup_task" => QueuedKind::Task,
                "send_message" => QueuedKind::Message,
                _ => continue,
            };
            let Some(source) = call_source(thread, tool, report) else {
                continue;
            };
            inboxes[recipient].push(Queued {
                queued_at_ms: send.queued_at_ms,
                sender: (sender, source),
                kind,
            });
        }
        // A finished child queues its final answer for the parent.
        for completion in &thread.child_completions {
            let Some(&child) = thread_index.get(completion.child_thread_id.as_str()) else {
                report.unresolved_agents += 1;
                continue;
            };
            let Some(last) = threads[child]
                .requests
                .iter()
                .rposition(|request| request.end_ms <= completion.queued_at_ms)
            else {
                continue;
            };
            inboxes[sender].push(Queued {
                queued_at_ms: completion.queued_at_ms,
                sender: (child, last),
                kind: QueuedKind::Message,
            });
        }
    }
    for inbox in &mut inboxes {
        inbox.sort_by_key(|queued| queued.queued_at_ms);
    }

    let mut edges = threads
        .iter()
        .map(|thread| vec![Vec::new(); thread.requests.len()])
        .collect::<Vec<_>>();
    let mut spawned_by = FxHashMap::default();
    let mut push = |recipient: usize, request: usize, edge: Edge| {
        let request_edges: &mut Vec<Edge> = &mut edges[recipient][request];
        if request_edges.contains(&edge) {
            return false;
        }
        request_edges.push(edge);
        true
    };

    for (recipient, inbox) in inboxes.iter().enumerate() {
        let thread = &threads[recipient];
        for queued in inbox
            .iter()
            .filter(|queued| queued.kind != QueuedKind::Message)
        {
            // A spawn starts the child's first turn. The child may stamp that turn's input in
            // the same millisecond as the parent's spawn record, so target it directly.
            let target = if queued.kind == QueuedKind::Spawn {
                spawned_by.entry(recipient).or_insert(queued.sender);
                Some(0)
            } else {
                thread
                    .requests
                    .iter()
                    .position(|request| request.start_ms >= queued.queued_at_ms)
            };
            let Some(target) = target else {
                report.undelivered_tasks += 1;
                continue;
            };
            let from_parent = thread.parent_thread_id.as_deref()
                == Some(threads[queued.sender.0].thread_id.as_str());
            let relation = if from_parent {
                AgenticDependencyRelation::Spawn
            } else {
                AgenticDependencyRelation::Join
            };
            let edge = Edge {
                sender: queued.sender,
                relation,
            };
            report.task_edges += usize::from(push(recipient, target, edge));
        }
        // A wait ends on the first message queued after its request started, or times out. A
        // wait with no visible message either timed out or waited on a sender outside the export.
        for tool in thread.tools.iter().filter(|tool| tool.name == "wait_agent") {
            let Some(ended_at_ms) = tool.ended_at_ms else {
                continue;
            };
            let Some(source) = call_source(thread, tool, report) else {
                continue;
            };
            let window_start_ms = thread.requests[source].start_ms;
            let Some(queued) = inbox
                .iter()
                .find(|queued| (window_start_ms..=ended_at_ms).contains(&queued.queued_at_ms))
            else {
                report.unlinked_waits += 1;
                continue;
            };
            let Some(consumer) = thread
                .requests
                .iter()
                .position(|request| request.start_ms >= ended_at_ms)
            else {
                continue;
            };
            let edge = Edge {
                sender: queued.sender,
                relation: AgenticDependencyRelation::Join,
            };
            report.wait_edges += usize::from(push(recipient, consumer, edge));
        }
    }
    Links { edges, spawned_by }
}

#[derive(Serialize)]
struct NoToolMetadata;

/// Writes one Dynamo request-trace row per recorded response, followed by its tool rows.
pub fn write_request_trace(
    output_path: &Path,
    threads: &[RolloutThread],
    config: ExportConfig,
    report: &mut ExportReport,
) -> Result<()> {
    if config.block_size == 0 {
        bail!("block_size must be greater than 0");
    }
    let export_ids = threads
        .iter()
        .map(|thread| {
            if config.preserve_session_ids {
                thread.thread_id.clone()
            } else {
                anonymized_session_id(&thread.thread_id)
            }
        })
        .collect::<Vec<_>>();
    let request_id =
        |(thread, request): (usize, usize)| format!("codex:{}:{request}", export_ids[thread]);
    let thread_index = threads
        .iter()
        .enumerate()
        .map(|(index, thread)| (thread.thread_id.as_str(), index))
        .collect::<FxHashMap<_, _>>();

    let links = link_threads(threads, report);
    // A forked child's first request inherits the context of the request that spawned it.
    let mut fork_sources = FxHashMap::default();
    let mut fork_children = FxHashMap::<(usize, usize), usize>::default();
    for (child, thread) in threads.iter().enumerate() {
        let Some(&source) = links.spawned_by.get(&child) else {
            continue;
        };
        if thread.forked_from_id.as_deref() == Some(threads[source.0].thread_id.as_str()) {
            fork_sources.insert(child, source);
            *fork_children.entry(source).or_default() += 1;
        }
    }

    let mut order = threads
        .iter()
        .enumerate()
        .flat_map(|(thread, rollout)| {
            (0..rollout.requests.len()).map(move |request| (thread, request))
        })
        .collect::<Vec<_>>();
    order.sort_by_key(|&(thread, request)| {
        let request_row = &threads[thread].requests[request];
        (request_row.start_ms, request_row.end_ms, thread, request)
    });
    let Some(&(first_thread, first_request)) = order.first() else {
        bail!("no Codex requests to export");
    };
    let trace_start_ms = threads[first_thread].requests[first_request].start_ms;

    let tools_by_request = threads
        .iter()
        .map(|thread| {
            let mut grouped = vec![Vec::new(); thread.requests.len()];
            for tool in &thread.tools {
                if let (Some(source), Some(_)) = (tool.source_request, tool.ended_at_ms) {
                    grouped[source].push(tool);
                }
            }
            grouped
        })
        .collect::<Vec<_>>();
    let seeds = export_ids
        .iter()
        .map(|id| synthetic_stream_seed(id))
        .collect::<Vec<_>>();

    let mut output = BufWriter::new(
        File::create(output_path)
            .with_context(|| format!("failed to create {}", output_path.display()))?,
    );
    let mut prefix_pool = PrefixPool::default();
    let mut previous_bases: Vec<Option<ReplayBase>> = threads.iter().map(|_| None).collect();
    let mut fork_bases: FxHashMap<(usize, usize), ReplayBase> = FxHashMap::default();
    for (thread_index_value, request_index) in order {
        let thread = &threads[thread_index_value];
        let request = &thread.requests[request_index];
        let input_length = usize::try_from(request.usage.input_tokens)?;
        let cached_length = usize::try_from(request.usage.cached_input_tokens)?;

        let fork_base = previous_bases[thread_index_value]
            .is_none()
            .then(|| fork_sources.get(&thread_index_value))
            .flatten()
            .and_then(|source| {
                let remaining = fork_children.get_mut(source)?;
                *remaining -= 1;
                if *remaining == 0 {
                    fork_bases.remove(source)
                } else {
                    fork_bases.get(source).cloned()
                }
            });
        let (source, blocks_counter) = match (&previous_bases[thread_index_value], &fork_base) {
            (Some(base), _) => (Some(base), &mut report.cache_prefix_blocks),
            (None, Some(base)) => (Some(base), &mut report.forked_prefix_blocks),
            (None, None) if cached_length > 0 => {
                // A fork whose spawning request is outside the export shares its inherited
                // context with its siblings, not with every thread in the working directory.
                let key = match &thread.forked_from_id {
                    Some(parent) => prefix_pool_key("codex-fork", parent, &thread.cwd),
                    None => prefix_pool_key("codex", &request.model, &thread.cwd),
                };
                (
                    Some(prefix_pool.prefix(&key, cached_length, config.block_size)?),
                    &mut report.pooled_prefix_blocks,
                )
            }
            (None, None) => (None, &mut report.cache_prefix_blocks),
        };
        let (tokens, shared_tokens) = usage_shaped_tokens(
            seeds[thread_index_value],
            request_index,
            input_length,
            cached_length,
            source.map(|base| base.tokens.as_slice()),
        );
        *blocks_counter += shared_tokens / config.block_size;
        let replay = ReplayBase::derive(source, shared_tokens, tokens, config.block_size)?;

        let id = request_id((thread_index_value, request_index));
        let dependencies = links.edges[thread_index_value][request_index]
            .iter()
            .map(|edge| ReplayDependency {
                request_id: request_id(edge.sender),
                relation: edge.relation,
                trigger: AgenticDependencyTrigger::Completion,
            })
            .collect::<Vec<_>>();
        let parent_session_id = thread
            .parent_thread_id
            .as_deref()
            .and_then(|parent| thread_index.get(parent))
            .map(|parent| export_ids[*parent].as_str());
        let agent_context = AgentContext {
            session_id: &export_ids[thread_index_value],
            parent_session_id,
        };
        let row = TraceLine {
            timestamp: nonnegative_ms(request.end_ms - trace_start_ms),
            event: RequestEndEvent {
                schema: REQUEST_TRACE_SCHEMA,
                event_type: "request_end",
                event_time_unix_ms: nonnegative_ms(request.end_ms),
                event_source: HARNESS_EVENT_SOURCE,
                agent_context: &agent_context,
                request: RequestFields {
                    request_id: &id,
                    model: &request.model,
                    input_tokens: replay.tokens.len(),
                    output_tokens: usize::try_from(request.usage.output_tokens)?,
                    request_received_ms: nonnegative_ms(request.start_ms),
                    total_time_ms: (request.end_ms - request.start_ms).max(0) as f64,
                    replay: ReplayFields {
                        trace_block_size: config.block_size,
                        input_length: replay.tokens.len(),
                        input_sequence_hashes: &replay.hashes,
                        dependencies: &dependencies,
                    },
                    cached_tokens: Some(cached_length),
                    claude: None,
                },
            },
        };
        write_json_line(&mut output, &row)?;
        for tool in &tools_by_request[thread_index_value][request_index] {
            let ended_at_ms = tool.ended_at_ms.expect("grouped tools have ended");
            let tool_row = TraceLine {
                timestamp: nonnegative_ms(ended_at_ms - trace_start_ms),
                event: ToolEvent {
                    schema: REQUEST_TRACE_SCHEMA,
                    event_type: "tool_end",
                    event_time_unix_ms: nonnegative_ms(ended_at_ms),
                    event_source: HARNESS_EVENT_SOURCE,
                    agent_context: &agent_context,
                    tool: ToolFields::<NoToolMetadata> {
                        tool_call_id: &tool.call_id,
                        tool_class: &tool.name,
                        claude: None,
                        started_at_unix_ms: nonnegative_ms(tool.started_at_ms),
                        ended_at_unix_ms: nonnegative_ms(ended_at_ms),
                        duration_ms: (ended_at_ms - tool.started_at_ms).max(0) as f64,
                        status: "succeeded",
                        output_bytes: None,
                        error_type: None,
                    },
                },
            };
            write_json_line(&mut output, &tool_row)?;
            report.tools += 1;
        }
        report.requests += 1;
        report.compactions += usize::from(request.compaction);

        if fork_children.contains_key(&(thread_index_value, request_index)) {
            fork_bases.insert((thread_index_value, request_index), replay.clone());
        }
        previous_bases[thread_index_value] =
            (request_index + 1 < thread.requests.len()).then_some(replay);
    }
    output.flush()?;

    report.threads = threads.len();
    for thread in threads {
        report.preempted_responses += thread.unrecorded.preempted;
        report.aborted_responses += thread.unrecorded.aborted;
        report.untracked_responses += thread.unrecorded.untracked;
        report.unmatched_compactions += thread.unmatched_compactions;
    }
    Ok(())
}

fn write_json_line(writer: &mut impl Write, value: &impl Serialize) -> Result<()> {
    serde_json::to_writer(&mut *writer, value)?;
    writer.write_all(b"\n")?;
    Ok(())
}

fn nonnegative_ms(value: i64) -> u64 {
    value.max(0) as u64
}

#[cfg(test)]
mod tests {
    use super::{ExportConfig, ExportReport, load_threads, write_request_trace};
    use crate::coding::codex::parser::tests::{line, rollout};
    use dynamo_data_gen::AgenticDependencyRelation;
    use dynamo_data_gen::request_trace::agentic::lower_agentic_mooncake_rows;
    use dynamo_data_gen::request_trace::load::load_request_trace_records;
    use serde_json::{Value, json};
    use tempfile::TempDir;

    fn usage(second: u32, response_id: &str, input: u64, cached: u64, output: u64) -> Value {
        line(
            second,
            "token_usage_record",
            json!({"response_id": response_id, "usage": {"input_tokens": input, "cached_input_tokens": cached, "output_tokens": output}}),
        )
    }

    fn activity(second: u32, id: &str, kind: &str, thread: &str) -> Value {
        line(
            second,
            "event_msg",
            json!({"type": "item_completed", "item": {"type": "SubAgentActivity", "id": id, "kind": kind, "agent_thread_id": thread}}),
        )
    }

    fn call(second: u32, call_id: &str, name: &str) -> Value {
        line(
            second,
            "response_item",
            json!({"type": "function_call", "call_id": call_id, "name": name}),
        )
    }

    fn output(second: u32, call_id: &str) -> Value {
        line(
            second,
            "response_item",
            json!({"type": "function_call_output", "call_id": call_id}),
        )
    }

    fn export(rollouts: &[&tempfile::NamedTempFile]) -> (ExportReport, Vec<Value>) {
        let mut report = ExportReport::default();
        let paths = rollouts
            .iter()
            .map(|rollout| rollout.path().to_path_buf())
            .collect::<Vec<_>>();
        let threads = load_threads(&paths, 2, &mut report).unwrap();
        let output_dir = TempDir::new().unwrap();
        let output_path = output_dir.path().join("codex.jsonl");
        let config = ExportConfig {
            block_size: 64,
            preserve_session_ids: true,
        };
        write_request_trace(&output_path, &threads, config, &mut report).unwrap();
        let rows = std::fs::read_to_string(&output_path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .filter(|row: &Value| row["event"]["event_type"] == "request_end")
            .collect();
        (report, rows)
    }

    fn forked_child(id: &str, spawned_at: u32) -> tempfile::NamedTempFile {
        rollout(&[
            line(
                spawned_at,
                "session_meta",
                json!({"id": id, "parent_thread_id": "parent", "forked_from_id": "parent", "subagent_history_start_ordinal": 2}),
            ),
            line(spawned_at, "session_meta", json!({"id": "parent"})),
            line(
                spawned_at + 1,
                "response_item",
                json!({"type": "agent_message", "author": "/root"}),
            ),
            line(
                spawned_at + 2,
                "response_item",
                json!({"type": "reasoning"}),
            ),
            usage(spawned_at + 3, "resp-c1", 1_200, 1_000, 30),
        ])
    }

    #[test]
    fn forks_spawned_by_one_request_each_inherit_its_context() {
        let parent = rollout(&[
            line(0, "session_meta", json!({"id": "parent"})),
            line(
                1,
                "response_item",
                json!({"type": "message", "role": "user"}),
            ),
            call(2, "call-a", "spawn_agent"),
            call(2, "call-b", "spawn_agent"),
            usage(3, "resp-p1", 1_000, 0, 10),
            activity(4, "call-a", "started", "child-a"),
            activity(4, "call-b", "started", "child-b"),
        ]);
        let (report, _) = export(&[
            &parent,
            &forked_child("child-a", 4),
            &forked_child("child-b", 4),
        ]);
        assert_eq!(report.task_edges, 2);
        assert_eq!(report.forked_prefix_blocks, 2 * (1_000 / 64));
    }

    #[test]
    fn waits_link_only_messages_queued_while_waiting() {
        // The child finishes before the parent's waiting request starts, so its answer arrives
        // with that request and cannot be what the wait returned on.
        let parent = rollout(&[
            line(0, "session_meta", json!({"id": "parent"})),
            line(
                1,
                "response_item",
                json!({"type": "message", "role": "user"}),
            ),
            call(2, "call-spawn", "spawn_agent"),
            usage(3, "resp-p1", 1_000, 0, 10),
            activity(4, "call-spawn", "started", "child"),
            output(4, "call-spawn"),
            activity(20, "subagent-completed-turn", "completed", "child"),
            line(
                30,
                "response_item",
                json!({"type": "message", "role": "user"}),
            ),
            call(31, "call-wait", "wait_agent"),
            usage(32, "resp-p2", 1_100, 1_000, 5),
            output(70, "call-wait"),
            line(71, "response_item", json!({"type": "reasoning"})),
            usage(72, "resp-p3", 1_200, 1_100, 5),
        ]);
        let child = rollout(&[
            line(
                4,
                "session_meta",
                json!({"id": "child", "parent_thread_id": "parent"}),
            ),
            line(
                5,
                "response_item",
                json!({"type": "agent_message", "author": "/root"}),
            ),
            line(6, "response_item", json!({"type": "reasoning"})),
            usage(10, "resp-c1", 500, 0, 30),
        ]);
        let (report, rows) = export(&[&parent, &child]);
        assert_eq!((report.wait_edges, report.unlinked_waits), (0, 1));
        let consumer = rows
            .iter()
            .find(|row| row["event"]["request"]["request_id"] == "codex:parent:2")
            .unwrap();
        assert!(consumer["event"]["request"]["replay"]["dependencies"].is_null());
    }

    #[test]
    fn messages_and_calls_in_unrecorded_responses_still_link() {
        let parent = rollout(&[
            line(0, "session_meta", json!({"id": "parent"})),
            line(
                1,
                "response_item",
                json!({"type": "message", "role": "user"}),
            ),
            line(2, "response_item", json!({"type": "reasoning"})),
            usage(3, "resp-p1", 1_000, 0, 10),
            // This response is cut off by a queued message, so its spawn has no usage record.
            call(5, "call-spawn", "spawn_agent"),
            line(
                6,
                "response_item",
                json!({"type": "agent_message", "author": "/root/other"}),
            ),
            activity(6, "call-spawn", "started", "child"),
            call(8, "call-message", "send_message"),
            usage(9, "resp-p2", 1_100, 1_000, 5),
            activity(9, "call-message", "interacted", "child"),
        ]);
        let child = rollout(&[
            line(
                6,
                "session_meta",
                json!({"id": "child", "parent_thread_id": "missing-root"}),
            ),
            line(
                7,
                "response_item",
                json!({"type": "agent_message", "author": "/root"}),
            ),
            line(8, "response_item", json!({"type": "reasoning"})),
            usage(10, "resp-c1", 500, 0, 30),
            line(
                11,
                "response_item",
                json!({"type": "agent_message", "author": "/root"}),
            ),
            line(12, "response_item", json!({"type": "reasoning"})),
            usage(13, "resp-c2", 600, 500, 30),
        ]);
        let (report, rows) = export(&[&parent, &child]);
        assert_eq!(report.reattributed_calls, 1);
        // The spawn links; the queued message starts no turn.
        assert_eq!(report.task_edges, 1);
        let child_rows = rows
            .iter()
            .filter(|row| row["event"]["agent_context"]["session_id"] == "child")
            .collect::<Vec<_>>();
        assert_eq!(
            child_rows[0]["event"]["request"]["replay"]["dependencies"][0]["request_id"],
            "codex:parent:0"
        );
        assert!(child_rows[1]["event"]["request"]["replay"]["dependencies"].is_null());
        // The parent named by the child is not in the export, so no parent session is claimed.
        assert!(child_rows[0]["event"]["agent_context"]["parent_session_id"].is_null());
    }

    #[test]
    fn spawned_fork_and_wait_become_explicit_edges() {
        let parent = rollout(&[
            line(0, "session_meta", json!({"id": "parent", "cwd": "/repo"})),
            line(0, "turn_context", json!({"model": "gpt"})),
            line(
                1,
                "response_item",
                json!({"type": "message", "role": "user"}),
            ),
            call(2, "call-spawn", "spawn_agent"),
            usage(3, "resp-p1", 1_000, 0, 10),
            activity(4, "call-spawn", "started", "child"),
            output(4, "call-spawn"),
            call(5, "call-wait", "wait_agent"),
            usage(6, "resp-p2", 1_100, 1_000, 5),
            activity(20, "subagent-completed-turn", "completed", "child"),
            output(20, "call-wait"),
            line(
                22,
                "response_item",
                json!({"type": "message", "role": "assistant"}),
            ),
            usage(23, "resp-p3", 1_300, 1_100, 20),
        ]);
        let child = rollout(&[
            line(
                4,
                "session_meta",
                json!({"id": "child", "parent_thread_id": "parent", "forked_from_id": "parent", "subagent_history_start_ordinal": 3, "cwd": "/repo"}),
            ),
            line(4, "session_meta", json!({"id": "parent"})),
            line(
                4,
                "response_item",
                json!({"type": "message", "role": "user"}),
            ),
            line(5, "turn_context", json!({"model": "gpt"})),
            line(
                5,
                "response_item",
                json!({"type": "agent_message", "author": "/root"}),
            ),
            line(7, "response_item", json!({"type": "reasoning"})),
            usage(10, "resp-c1", 1_200, 1_000, 30),
            line(
                15,
                "response_item",
                json!({"type": "message", "role": "assistant"}),
            ),
            usage(18, "resp-c2", 1_300, 1_200, 40),
        ]);

        let mut report = ExportReport::default();
        let threads = load_threads(
            &[parent.path().to_path_buf(), child.path().to_path_buf()],
            2,
            &mut report,
        )
        .unwrap();
        let output_dir = TempDir::new().unwrap();
        let output_path = output_dir.path().join("codex.jsonl");
        write_request_trace(
            &output_path,
            &threads,
            ExportConfig {
                block_size: 64,
                preserve_session_ids: true,
            },
            &mut report,
        )
        .unwrap();
        assert_eq!((report.task_edges, report.wait_edges), (1, 1));
        assert_eq!(report.forked_prefix_blocks, 1_000 / 64);

        let hashes = |request_id: &str| -> Vec<u64> {
            std::fs::read_to_string(&output_path)
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str::<Value>(line).unwrap())
                .find(|row| row["event"]["request"]["request_id"] == request_id)
                .map(|row| {
                    serde_json::from_value(
                        row["event"]["request"]["replay"]["input_sequence_hashes"].clone(),
                    )
                    .unwrap()
                })
                .unwrap()
        };
        // The forked child's first request reuses the context of the request that spawned it.
        assert_eq!(
            hashes("codex:child:0")[..15],
            hashes("codex:parent:0")[..15]
        );

        let loaded = load_request_trace_records(&[output_path]).unwrap();
        let mut rows = Vec::new();
        lower_agentic_mooncake_rows(loaded, |_, row| {
            rows.push(row);
            Ok(())
        })
        .unwrap();
        let edges = |request_id: &str| {
            rows.iter()
                .find(|row| row.request_id == request_id)
                .unwrap()
                .dependencies
                .iter()
                .map(|edge| (edge.request_id.clone(), edge.relation))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            edges("codex:child:0"),
            [(
                "codex:parent:0".to_string(),
                AgenticDependencyRelation::Spawn
            )]
        );
        assert_eq!(
            edges("codex:parent:2"),
            [
                ("codex:child:1".to_string(), AgenticDependencyRelation::Join),
                (
                    "codex:parent:1".to_string(),
                    AgenticDependencyRelation::Sequence
                ),
            ]
        );
    }
}
