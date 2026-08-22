// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Local Weka/AgentX trace ingestion for typed agentic replay.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;

use crate::{
    AGENTIC_MOONCAKE_SCHEMA, AGENTIC_MOONCAKE_VERSION, AgenticDependency,
    AgenticDependencyRelation, AgenticDependencyTrigger, AgenticHashIdScope, AgenticMooncakeHeader,
    AgenticMooncakeRow, AgenticSourceProvenance,
};

const JOIN_EPSILON_SECONDS: f64 = 1e-6;
const SEAM_MAX_GAP_SECONDS: f64 = 3600.0;
const SEAM_MIN_OVERLAP_RATIO: f64 = 0.5;
const NANOSECONDS_PER_SECOND: f64 = 1_000_000_000.0;
const NANOSECONDS_PER_MILLISECOND: f64 = 1_000_000.0;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WekaImportSummary {
    pub header: AgenticMooncakeHeader,
    pub files: usize,
    pub plays: usize,
    pub requests: usize,
    pub raw_zero_outputs: usize,
}

/// A preflighted local Weka corpus.
///
/// Opening performs deterministic traversal, rejects symlinks and mixed block
/// sizes, and computes the raw corpus digest before any row can be emitted.
pub struct WekaImporter {
    root: PathBuf,
    files: Vec<WekaFile>,
    header: AgenticMooncakeHeader,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SubagentMode {
    Blocking,
    Background,
}

struct WekaFile {
    path: PathBuf,
    relative_path: String,
    digest: blake3::Hash,
}

impl WekaImporter {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let (root, files) = collect_source_files(path)?;
        if files.is_empty() {
            bail!("Weka source {} contains no JSON files", path.display());
        }

        let mut corpus_hasher = blake3::Hasher::new();
        corpus_hasher.update(b"dynamo-weka-corpus-v1\0");
        let mut block_size = None;
        let mut corpus_model = None;
        let mut preflighted = Vec::with_capacity(files.len());
        for (file_path, relative_path) in files {
            let bytes = read_source_bytes(&file_path)?;
            update_corpus_digest(&mut corpus_hasher, &relative_path, &bytes);
            let digest = blake3::hash(&bytes);
            let trace = parse_trace(&file_path, &bytes)?;
            validate_trace_header(&trace, &relative_path)?;
            let model = trace_request_model(&trace, &relative_path)?;
            match block_size {
                Some(expected) if expected != trace.block_size => bail!(
                    "Weka corpus mixes block sizes: {} has {}, expected {}",
                    relative_path,
                    trace.block_size,
                    expected
                ),
                None => block_size = Some(trace.block_size),
                _ => {}
            }
            match corpus_model.as_deref() {
                Some(expected) if expected != model => bail!(
                    "Weka corpus mixes request models: {} has {:?}, expected {:?}",
                    relative_path,
                    model,
                    expected
                ),
                None => corpus_model = Some(model.to_string()),
                _ => {}
            }
            preflighted.push(WekaFile {
                path: file_path,
                relative_path,
                digest,
            });
        }
        let digest = corpus_hasher.finalize().to_hex().to_string();

        Ok(Self {
            root,
            files: preflighted,
            header: AgenticMooncakeHeader {
                schema: AGENTIC_MOONCAKE_SCHEMA.to_string(),
                version: AGENTIC_MOONCAKE_VERSION,
                block_size: block_size.expect("non-empty corpus has a block size"),
                hash_id_scope: AgenticHashIdScope::Local,
                source: AgenticSourceProvenance {
                    format: "weka".to_string(),
                    digest,
                },
            },
        })
    }

    pub fn header(&self) -> &AgenticMooncakeHeader {
        &self.header
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Lower one source file at a time and move rows into the caller's sink.
    pub fn for_each_row<F>(&self, mut emit: F) -> Result<WekaImportSummary>
    where
        F: FnMut(AgenticMooncakeRow) -> Result<()>,
    {
        let mut requests = 0;
        let mut raw_zero_outputs = 0;
        for file in &self.files {
            let bytes = read_source_bytes(&file.path)?;
            let digest = blake3::hash(&bytes);
            if digest != file.digest {
                bail!(
                    "Weka source {} changed after preflight: expected {}, found {}",
                    file.relative_path,
                    file.digest.to_hex(),
                    digest.to_hex()
                );
            }
            let trace = parse_trace(&file.path, &bytes)?;
            let lowered = lower_trace(&trace, &file.relative_path)?;
            raw_zero_outputs += lowered.raw_zero_outputs;
            requests += lowered.rows.len();
            for row in lowered.rows {
                emit(row)?;
            }
        }
        Ok(WekaImportSummary {
            header: self.header.clone(),
            files: self.files.len(),
            plays: self.files.len(),
            requests,
            raw_zero_outputs,
        })
    }

    pub fn collect_rows(&self) -> Result<(WekaImportSummary, Vec<AgenticMooncakeRow>)> {
        let mut rows = Vec::new();
        let summary = self.for_each_row(|row| {
            rows.push(row);
            Ok(())
        })?;
        Ok((summary, rows))
    }
}

pub fn stream_weka_agentic_rows<F>(path: impl AsRef<Path>, emit: F) -> Result<WekaImportSummary>
where
    F: FnMut(AgenticMooncakeRow) -> Result<()>,
{
    WekaImporter::open(path)?.for_each_row(emit)
}

pub fn load_weka_agentic_rows(
    path: impl AsRef<Path>,
) -> Result<(WekaImportSummary, Vec<AgenticMooncakeRow>)> {
    WekaImporter::open(path)?.collect_rows()
}

#[derive(Debug, Deserialize)]
struct WekaTrace {
    id: String,
    #[serde(default)]
    models: Vec<String>,
    block_size: usize,
    hash_id_scope: String,
    #[serde(default)]
    tool_tokens: usize,
    #[serde(default)]
    system_tokens: usize,
    requests: Vec<WekaEntry>,
    #[serde(default)]
    totals: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum WekaEntry {
    #[serde(rename = "n")]
    Normal(WekaRequest),
    #[serde(rename = "s")]
    Streaming(WekaRequest),
    #[serde(rename = "subagent")]
    Subagent(WekaSubagent),
}

#[derive(Debug, Clone, Deserialize)]
struct WekaRequest {
    t: f64,
    model: String,
    #[serde(rename = "in")]
    input_length: usize,
    #[serde(rename = "out")]
    output_length: usize,
    #[serde(default)]
    hash_ids: Vec<u64>,
    #[serde(default)]
    input_types: Vec<String>,
    #[serde(default)]
    output_types: Vec<String>,
    #[serde(default)]
    stop: String,
    #[serde(default)]
    api_time: Option<f64>,
    #[serde(default)]
    think_time: Option<f64>,
    #[serde(default)]
    ttft: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct WekaSubagent {
    t: f64,
    agent_id: String,
    subagent_type: String,
    #[serde(default)]
    duration_ms: Option<i64>,
    #[serde(default)]
    total_tokens: Option<usize>,
    #[serde(default)]
    tool_use_count: Option<usize>,
    status: String,
    requests: Vec<WekaInnerEntry>,
    #[serde(default)]
    models: Vec<String>,
    #[serde(default)]
    tool_tokens: usize,
    #[serde(default)]
    system_tokens: usize,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum WekaInnerEntry {
    #[serde(rename = "n")]
    Normal(WekaRequest),
    #[serde(rename = "s")]
    Streaming(WekaRequest),
}

impl WekaEntry {
    fn request(&self) -> Option<&WekaRequest> {
        match self {
            Self::Normal(request) | Self::Streaming(request) => Some(request),
            Self::Subagent(_) => None,
        }
    }
}

#[derive(Clone)]
struct IndexedRequest {
    source_id: String,
    source_order: usize,
    request: WekaRequest,
}

#[derive(Clone)]
struct Fork {
    parent_chain: Option<usize>,
    fork_source_id: Option<String>,
    depth: usize,
    fork_time: f64,
}

#[derive(Default)]
struct Chain {
    requests: Vec<IndexedRequest>,
    fork: Option<Fork>,
    spliced_into: Option<usize>,
    tail_source_id: Option<String>,
    tail_hashes: Vec<u64>,
    tail_end: f64,
    tail_model: String,
}

struct ChainDetection {
    chains: Vec<Chain>,
    main_index: usize,
    worker_indices: Vec<usize>,
}

struct Stream {
    session_id: String,
    requests: Vec<IndexedRequest>,
    fork_source_id: Option<String>,
    scope_id: String,
}

struct LoweredTrace {
    rows: Vec<AgenticMooncakeRow>,
    raw_zero_outputs: usize,
}

fn lower_trace(trace: &WekaTrace, relative_path: &str) -> Result<LoweredTrace> {
    validate_trace_header(trace, relative_path)?;
    let namespace = namespace(relative_path);
    let play_id = format!("{namespace}:play:{}", trace.id);
    let mut top_level = Vec::new();
    let mut explicit = Vec::new();
    let mut raw_zero_outputs = 0;

    for (outer_index, entry) in trace.requests.iter().enumerate() {
        if let Some(request) = entry.request() {
            validate_request(request, relative_path)?;
            raw_zero_outputs += usize::from(request.output_length == 0);
            top_level.push(IndexedRequest {
                source_id: format!("outer:{outer_index}"),
                source_order: outer_index,
                request: request.clone(),
            });
        } else if let WekaEntry::Subagent(subagent) = entry {
            explicit.push((outer_index, subagent));
        }
    }
    if top_level.is_empty() {
        bail!("Weka trace {} has no parent requests", relative_path);
    }

    let (preamble, detection_input) = split_preamble(top_level);
    let detection = detect_chains(detection_input);
    let mut streams = streams_from_detection(&namespace, &play_id, "root", &detection, preamble);
    let root_stream_index = streams
        .iter()
        .position(|stream| stream.session_id.ends_with(":root"))
        .expect("root stream is present");
    let parent_stream_count = streams.len();
    let mut join_markers = Vec::<(String, Vec<String>)>::new();

    for (outer_index, subagent) in explicit {
        let mode = validate_subagent(subagent, relative_path)?;
        let mut owner_candidates = streams[..parent_stream_count]
            .iter()
            .enumerate()
            .flat_map(|(stream_index, stream)| {
                stream
                    .requests
                    .iter()
                    .filter(move |request| request.source_order < outer_index)
                    .map(move |request| (stream_index, request))
            })
            .collect::<Vec<_>>();
        owner_candidates.sort_by_key(|(_, request)| request.source_order);
        let Some((owner_stream_index, owner_request)) = owner_candidates.pop() else {
            bail!(
                "Weka trace {} has subagent {} at outer index {} without a preceding parent request",
                relative_path,
                subagent.agent_id,
                outer_index
            );
        };
        if owner_candidates
            .last()
            .is_some_and(|(_, candidate)| candidate.source_order == owner_request.source_order)
        {
            bail!(
                "Weka trace {} has ambiguous parent-stream ownership for subagent {} at outer index {}",
                relative_path,
                subagent.agent_id,
                outer_index
            );
        }
        let spawn_source_id = owner_request.source_id.clone();

        let mut inner = Vec::with_capacity(subagent.requests.len());
        for (inner_index, entry) in subagent.requests.iter().enumerate() {
            let mut request = match entry {
                WekaInnerEntry::Normal(request) | WekaInnerEntry::Streaming(request) => {
                    request.clone()
                }
            };
            if request.t + JOIN_EPSILON_SECONDS < subagent.t {
                request.t += subagent.t;
            }
            validate_request(&request, relative_path)?;
            raw_zero_outputs += usize::from(request.output_length == 0);
            inner.push(IndexedRequest {
                source_id: format!("outer:{outer_index}:inner:{inner_index}"),
                source_order: inner_index,
                request,
            });
        }
        if inner.is_empty() {
            continue;
        }

        let (preamble, detection_input) = split_preamble(inner);
        let child_detection = detect_chains(detection_input);
        let child_prefix = format!("subagent:{outer_index}:{}", subagent.agent_id);
        let mut child_streams = streams_from_detection(
            &namespace,
            &play_id,
            &child_prefix,
            &child_detection,
            preamble,
        );
        let child_root = child_streams
            .iter_mut()
            .find(|stream| stream.session_id.ends_with(&format!(":{child_prefix}")))
            .expect("subagent root stream is present");
        child_root.fork_source_id = Some(spawn_source_id);
        child_root.scope_id = child_root.session_id.clone();
        let scope_id = child_root.scope_id.clone();
        for stream in &mut child_streams {
            stream.scope_id = scope_id.clone();
        }

        let join_source_id = if mode == SubagentMode::Blocking {
            let child_end = subagent_end(subagent);
            streams[owner_stream_index]
                .requests
                .iter()
                .filter(|request| {
                    request.source_order > outer_index
                        && request.request.t + JOIN_EPSILON_SECONDS >= child_end
                })
                .min_by_key(|request| request.source_order)
                .map(|request| request.source_id.clone())
        } else {
            None
        };

        let child_stream_indices = streams.len()..(streams.len() + child_streams.len());
        streams.extend(child_streams);
        if let Some(join_source_id) = join_source_id {
            let terminal_sources = child_stream_indices
                .filter_map(|index| streams[index].requests.last())
                .map(|request| request.source_id.clone())
                .collect::<Vec<_>>();
            if !terminal_sources.is_empty() {
                join_markers.push((join_source_id, terminal_sources));
            }
        }
    }

    let root_time = streams
        .iter()
        .flat_map(|stream| stream.requests.iter())
        .map(|request| request.request.t)
        .fold(f64::INFINITY, f64::min);
    let mut row_by_source = HashMap::new();
    let mut request_by_source = HashMap::new();
    for stream in &streams {
        for request in &stream.requests {
            row_by_source.insert(
                request.source_id.clone(),
                format!("{namespace}:request:{}", request.source_id),
            );
            request_by_source.insert(request.source_id.clone(), request);
        }
    }

    let root_source = streams[root_stream_index]
        .requests
        .first()
        .expect("root stream is non-empty")
        .source_id
        .clone();
    let mut dependencies: HashMap<String, Vec<AgenticDependency>> = HashMap::new();
    for stream in &streams {
        for window in stream.requests.windows(2) {
            let predecessor = &window[0];
            let request = &window[1];
            push_dependency(
                dependencies.entry(request.source_id.clone()).or_default(),
                AgenticDependency {
                    request_id: row_by_source[&predecessor.source_id].clone(),
                    trigger: AgenticDependencyTrigger::Completion,
                    delay_ms: seconds_to_milliseconds(
                        request.request.t - request_end(&predecessor.request),
                    ),
                    relation: AgenticDependencyRelation::Sequence,
                },
            );
        }
    }

    for stream in &streams {
        let Some(first) = stream.requests.first() else {
            continue;
        };
        let Some(spawn_source_id) = stream.fork_source_id.as_ref() else {
            continue;
        };
        let parent = streams
            .iter()
            .flat_map(|stream| stream.requests.iter())
            .find(|request| &request.source_id == spawn_source_id)
            .ok_or_else(|| anyhow!("spawn source {spawn_source_id} is missing"))?;
        let (trigger, delay_ms) = if first.request.t < request_end(&parent.request) {
            (
                AgenticDependencyTrigger::Dispatch,
                seconds_to_milliseconds(first.request.t - parent.request.t),
            )
        } else {
            (
                AgenticDependencyTrigger::Completion,
                seconds_to_milliseconds(first.request.t - request_end(&parent.request)),
            )
        };
        push_dependency(
            dependencies.entry(first.source_id.clone()).or_default(),
            AgenticDependency {
                request_id: row_by_source[spawn_source_id].clone(),
                trigger,
                delay_ms,
                relation: AgenticDependencyRelation::Spawn,
            },
        );
    }

    install_cross_stream_frontiers(&streams, &row_by_source, &mut dependencies);

    for (target_source, terminal_sources) in join_markers {
        for terminal_source in terminal_sources {
            let target = request_by_source[&target_source];
            let terminal = request_by_source[&terminal_source];
            push_dependency(
                dependencies.entry(target_source.clone()).or_default(),
                AgenticDependency {
                    request_id: row_by_source[&terminal_source].clone(),
                    trigger: AgenticDependencyTrigger::Completion,
                    delay_ms: seconds_to_milliseconds(
                        target.request.t - request_end(&terminal.request),
                    ),
                    relation: AgenticDependencyRelation::Join,
                },
            );
        }
    }

    // A disjoint detected stream has no natural fork. Anchor it to the play
    // root's dispatch so the validated graph has one causal root without
    // inventing a completion barrier.
    for stream in &streams {
        let Some(first) = stream.requests.first() else {
            continue;
        };
        if first.source_id == root_source
            || dependencies
                .get(&first.source_id)
                .is_some_and(|edges| !edges.is_empty())
        {
            continue;
        }
        push_dependency(
            dependencies.entry(first.source_id.clone()).or_default(),
            AgenticDependency {
                request_id: row_by_source[&root_source].clone(),
                trigger: AgenticDependencyTrigger::Dispatch,
                delay_ms: seconds_to_milliseconds(
                    first.request.t - streams[root_stream_index].requests[0].request.t,
                ),
                relation: AgenticDependencyRelation::Spawn,
            },
        );
    }

    let mut rows = Vec::new();
    let mut used_hashes = HashMap::new();
    for stream in streams {
        for request in stream.requests {
            let request_id = row_by_source[&request.source_id].clone();
            let hash_ids = normalized_hashes(
                relative_path,
                &request_id,
                &request.request,
                trace.block_size,
                &mut used_hashes,
            )?;
            rows.push(AgenticMooncakeRow {
                request_id,
                play_id: play_id.clone(),
                session_id: stream.session_id.clone(),
                model: request.request.model.clone(),
                input_length: Some(request.request.input_length),
                output_length: Some(request.request.output_length.max(1)),
                output_token_ids: None,
                hash_ids: Some(hash_ids),
                not_before_ms: seconds_to_milliseconds(request.request.t - root_time),
                priority: None,
                strict_priority: None,
                policy_class: None,
                dependencies: dependencies.remove(&request.source_id).unwrap_or_default(),
            });
        }
    }
    rows.sort_by(|left, right| left.request_id.cmp(&right.request_id));
    if rows.len()
        != trace
            .requests
            .iter()
            .map(|entry| match entry {
                WekaEntry::Normal(_) | WekaEntry::Streaming(_) => 1,
                WekaEntry::Subagent(subagent) => subagent.requests.len(),
            })
            .sum::<usize>()
    {
        bail!(
            "Weka trace {} did not lower every source request (orphan subagents are unsupported)",
            relative_path
        );
    }
    Ok(LoweredTrace {
        rows,
        raw_zero_outputs,
    })
}

fn streams_from_detection(
    namespace: &str,
    _play_id: &str,
    prefix: &str,
    detection: &ChainDetection,
    mut preamble: Vec<IndexedRequest>,
) -> Vec<Stream> {
    let mut live = Vec::with_capacity(1 + detection.worker_indices.len());
    let mut main_requests = detection.chains[detection.main_index].requests.clone();
    main_requests.append(&mut preamble);
    main_requests.sort_by(request_order);
    live.push(Stream {
        session_id: format!("{namespace}:session:{prefix}"),
        requests: main_requests,
        fork_source_id: detection.chains[detection.main_index]
            .fork
            .as_ref()
            .and_then(|fork| fork.fork_source_id.clone()),
        scope_id: format!("{namespace}:scope:{prefix}"),
    });
    for (worker_index, chain_index) in detection.worker_indices.iter().copied().enumerate() {
        let chain = &detection.chains[chain_index];
        live.push(Stream {
            session_id: format!("{namespace}:session:{prefix}:worker:{worker_index}"),
            requests: chain.requests.clone(),
            fork_source_id: chain
                .fork
                .as_ref()
                .and_then(|fork| fork.fork_source_id.clone()),
            scope_id: format!("{namespace}:scope:{prefix}"),
        });
    }
    live
}

fn install_cross_stream_frontiers(
    streams: &[Stream],
    row_by_source: &HashMap<String, String>,
    dependencies: &mut HashMap<String, Vec<AgenticDependency>>,
) {
    let mut by_scope: BTreeMap<&str, Vec<&Stream>> = BTreeMap::new();
    for stream in streams {
        by_scope.entry(&stream.scope_id).or_default().push(stream);
    }
    for scoped_streams in by_scope.values() {
        for target_stream in scoped_streams {
            for target in &target_stream.requests {
                let mut frontier = Vec::new();
                for other_stream in scoped_streams {
                    if std::ptr::eq(*other_stream, *target_stream) {
                        continue;
                    }
                    let latest = other_stream
                        .requests
                        .iter()
                        .filter(|candidate| {
                            candidate.request.t < target.request.t
                                && request_end(&candidate.request)
                                    <= target.request.t + JOIN_EPSILON_SECONDS
                        })
                        .max_by(|left, right| {
                            request_end(&left.request)
                                .total_cmp(&request_end(&right.request))
                                .then(left.request.t.total_cmp(&right.request.t))
                                .then(left.source_id.cmp(&right.source_id))
                        });
                    if let Some(latest) = latest {
                        frontier.push(latest);
                    }
                }
                let pruned = frontier
                    .iter()
                    .filter(|candidate| {
                        !frontier.iter().any(|later| {
                            candidate.request.t < later.request.t
                                && request_end(&candidate.request)
                                    <= later.request.t + JOIN_EPSILON_SECONDS
                        })
                    })
                    .copied()
                    .collect::<Vec<_>>();
                for predecessor in pruned {
                    push_dependency(
                        dependencies.entry(target.source_id.clone()).or_default(),
                        AgenticDependency {
                            request_id: row_by_source[&predecessor.source_id].clone(),
                            trigger: AgenticDependencyTrigger::Completion,
                            delay_ms: 0.0,
                            relation: AgenticDependencyRelation::ReplayBarrier,
                        },
                    );
                }
            }
        }
    }
}

fn detect_chains(mut requests: Vec<IndexedRequest>) -> ChainDetection {
    requests.sort_by(request_order);
    let mut chains = Vec::<Chain>::new();
    let mut chain_by_source = HashMap::<String, usize>::new();
    let mut forks_by_source = HashMap::<String, Vec<usize>>::new();
    let mut request_by_source = HashMap::<String, WekaRequest>::new();

    for indexed in requests {
        request_by_source.insert(indexed.source_id.clone(), indexed.request.clone());
        if indexed.request.hash_ids.is_empty() {
            if chains.is_empty() {
                chains.push(Chain::default());
            }
            chain_by_source.insert(indexed.source_id.clone(), 0);
            chains[0].requests.push(indexed);
            continue;
        }
        if chains.is_empty() {
            chains.push(Chain::default());
            append_chain(0, &mut chains[0], &mut chain_by_source, indexed);
            continue;
        }
        if let Some(target) = extension_target(&chains, &indexed.request) {
            append_chain(target, &mut chains[target], &mut chain_by_source, indexed);
            continue;
        }
        if chains.iter().all(|chain| chain.tail_hashes.is_empty()) {
            append_chain(0, &mut chains[0], &mut chain_by_source, indexed);
            continue;
        }
        let (parent_chain, depth) = max_lcp_chain(&chains, &indexed.request.hash_ids);
        let fork_source_id = parent_chain.and_then(|parent| chains[parent].tail_source_id.clone());
        let chain_index = chains.len();
        chains.push(Chain {
            fork: Some(Fork {
                parent_chain,
                fork_source_id: fork_source_id.clone(),
                depth,
                fork_time: indexed.request.t,
            }),
            ..Default::default()
        });
        append_chain(
            chain_index,
            &mut chains[chain_index],
            &mut chain_by_source,
            indexed,
        );
        if let Some(source_id) = fork_source_id.filter(|_| depth > 0) {
            forks_by_source
                .entry(source_id)
                .or_default()
                .push(chain_index);
        }
    }

    resolve_seams(
        &mut chains,
        &mut forks_by_source,
        &mut chain_by_source,
        &request_by_source,
    );
    let mut aliases = HashMap::new();
    for (index, chain) in chains.iter().enumerate() {
        if let Some(owner) = chain.spliced_into {
            aliases.insert(index, owner);
        }
    }
    let resolve = |mut index: usize| {
        while let Some(owner) = aliases.get(&index) {
            index = *owner;
        }
        index
    };
    let main_index = chains
        .iter()
        .enumerate()
        .filter(|(_, chain)| chain.spliced_into.is_none())
        .min_by(|(_, left), (_, right)| request_order(&left.requests[0], &right.requests[0]))
        .map(|(index, _)| resolve(index))
        .unwrap_or(0);
    let mut worker_indices = chains
        .iter()
        .enumerate()
        .filter_map(|(index, chain)| {
            (chain.spliced_into.is_none() && index != main_index).then_some(index)
        })
        .collect::<Vec<_>>();
    worker_indices.sort_by(|left, right| {
        request_order(&chains[*left].requests[0], &chains[*right].requests[0])
    });
    ChainDetection {
        chains,
        main_index,
        worker_indices,
    }
}

fn append_chain(
    chain_index: usize,
    chain: &mut Chain,
    chain_by_source: &mut HashMap<String, usize>,
    indexed: IndexedRequest,
) {
    chain_by_source.insert(indexed.source_id.clone(), chain_index);
    if !indexed.request.hash_ids.is_empty() {
        chain.tail_source_id = Some(indexed.source_id.clone());
        chain.tail_hashes.clone_from(&indexed.request.hash_ids);
        chain.tail_end = request_end(&indexed.request);
        chain.tail_model.clone_from(&indexed.request.model);
    }
    chain.requests.push(indexed);
}

fn extension_target(chains: &[Chain], request: &WekaRequest) -> Option<usize> {
    let mut best = None;
    let mut best_len = 0;
    for (index, chain) in chains.iter().enumerate() {
        let tail_len = chain.tail_hashes.len();
        if tail_len == 0
            || tail_len > request.hash_ids.len()
            || tail_len <= best_len
            || chain.tail_model != request.model
            || chain.tail_end > request.t + JOIN_EPSILON_SECONDS
            || chain.tail_hashes != request.hash_ids[..tail_len]
        {
            continue;
        }
        best = Some(index);
        best_len = tail_len;
    }
    best
}

fn max_lcp_chain(chains: &[Chain], hashes: &[u64]) -> (Option<usize>, usize) {
    let mut best = None;
    let mut best_key = (0, 0);
    for (index, chain) in chains.iter().enumerate() {
        let depth = lcp(&chain.tail_hashes, hashes);
        let key = (depth, chain.tail_hashes.len());
        if depth > 0 && key > best_key {
            best = Some(index);
            best_key = key;
        }
    }
    (best, best_key.0)
}

fn resolve_seams(
    chains: &mut [Chain],
    forks_by_source: &mut HashMap<String, Vec<usize>>,
    chain_by_source: &mut HashMap<String, usize>,
    request_by_source: &HashMap<String, WekaRequest>,
) {
    let mut keys = forks_by_source.keys().cloned().collect::<BTreeSet<_>>();
    let mut aliases = HashMap::<usize, usize>::new();
    let mut processed = HashSet::new();
    while let Some(source_id) = keys.pop_first() {
        if !processed.insert(source_id.clone()) {
            continue;
        }
        let mut owner = chain_by_source[&source_id];
        while let Some(next) = aliases.get(&owner) {
            owner = *next;
        }
        if chains[owner].tail_source_id.as_deref() != Some(source_id.as_str()) {
            continue;
        }
        let tail = &request_by_source[&source_id];
        let registered = forks_by_source[&source_id]
            .iter()
            .copied()
            .filter(|index| chains[*index].spliced_into.is_none())
            .collect::<Vec<_>>();
        let elected = registered
            .iter()
            .copied()
            .filter(|index| seam_eligible(&chains[*index], tail))
            .max_by(|left, right| {
                let left_fork = chains[*left].fork.as_ref().unwrap();
                let right_fork = chains[*right].fork.as_ref().unwrap();
                left_fork
                    .depth
                    .cmp(&right_fork.depth)
                    .then_with(|| right_fork.fork_time.total_cmp(&left_fork.fork_time))
                    .then_with(|| right.cmp(left))
            });
        let Some(elected) = elected else {
            continue;
        };
        let moved = std::mem::take(&mut chains[elected].requests);
        for request in &moved {
            chain_by_source.insert(request.source_id.clone(), owner);
        }
        chains[owner].requests.extend(moved);
        chains[owner].tail_source_id = chains[elected].tail_source_id.clone();
        chains[owner].tail_hashes = chains[elected].tail_hashes.clone();
        chains[owner].tail_end = chains[elected].tail_end;
        chains[owner].tail_model = chains[elected].tail_model.clone();
        chains[elected].spliced_into = Some(owner);
        aliases.insert(elected, owner);

        let Some(new_tail_source) = chains[owner].tail_source_id.clone() else {
            continue;
        };
        let new_tail_hashes = chains[owner].tail_hashes.clone();
        let new_tail = &request_by_source[&new_tail_source];
        for candidate in registered {
            if candidate == elected || chains[candidate].spliced_into.is_some() {
                continue;
            }
            let candidate_first = &chains[candidate].requests[0].request;
            if new_tail.t > candidate_first.t + JOIN_EPSILON_SECONDS {
                continue;
            }
            let depth = lcp(&new_tail_hashes, &candidate_first.hash_ids);
            if depth == 0 {
                continue;
            }
            let fork = chains[candidate].fork.as_mut().unwrap();
            fork.parent_chain = Some(owner);
            fork.fork_source_id = Some(new_tail_source.clone());
            fork.depth = depth;
            forks_by_source
                .entry(new_tail_source.clone())
                .or_default()
                .push(candidate);
            processed.remove(&new_tail_source);
            keys.insert(new_tail_source.clone());
        }
    }
}

fn seam_eligible(chain: &Chain, tail: &WekaRequest) -> bool {
    let Some(fork) = chain.fork.as_ref() else {
        return false;
    };
    if fork.depth == 0 || chain.requests.is_empty() {
        return false;
    }
    let first = &chain.requests[0].request;
    let gap = first.t - request_end(tail);
    let overlap = fork.depth as f64 / tail.hash_ids.len().max(1) as f64;
    request_end(tail) <= first.t + JOIN_EPSILON_SECONDS
        && first.model == tail.model
        && !(gap > SEAM_MAX_GAP_SECONDS && overlap < SEAM_MIN_OVERLAP_RATIO)
}

fn split_preamble(mut requests: Vec<IndexedRequest>) -> (Vec<IndexedRequest>, Vec<IndexedRequest>) {
    requests.sort_by(request_order);
    if requests.len() < 2 || requests[0].request.hash_ids.is_empty() {
        return (Vec::new(), requests);
    }
    let first = &requests[0].request;
    if requests[1..]
        .iter()
        .any(|other| lcp(&first.hash_ids, &other.request.hash_ids) > 0)
    {
        return (Vec::new(), requests);
    }
    if first.output_length > 64 {
        let other_hashes = requests[1..]
            .iter()
            .flat_map(|request| request.request.hash_ids.iter().copied())
            .collect::<HashSet<_>>();
        if first
            .hash_ids
            .iter()
            .any(|hash| other_hashes.contains(hash))
        {
            return (Vec::new(), requests);
        }
    }
    let first = requests.remove(0);
    (vec![first], requests)
}

fn normalized_hashes(
    relative_path: &str,
    request_id: &str,
    request: &WekaRequest,
    block_size: usize,
    used: &mut HashMap<u64, String>,
) -> Result<Vec<u64>> {
    let full_blocks = request.input_length / block_size;
    let has_partial = !request.input_length.is_multiple_of(block_size);
    let mut result = Vec::with_capacity(full_blocks + usize::from(has_partial));
    for block_index in 0..full_blocks {
        let identity = request.hash_ids.get(block_index).map_or_else(
            || format!("private:missing:{request_id}:{block_index}"),
            |hash| format!("source:{relative_path}:{hash}"),
        );
        result.push(unique_hash("weka-full-block", &identity, used)?);
    }
    if has_partial {
        result.push(unique_hash(
            "weka-partial-tail",
            &format!("{request_id}:{full_blocks}:{}", request.input_length),
            used,
        )?);
    }
    Ok(result)
}

fn unique_hash(domain: &str, identity: &str, used: &mut HashMap<u64, String>) -> Result<u64> {
    for nonce in 0_u32..=u32::MAX {
        let digest = blake3::hash(format!("{domain}\0{identity}\0{nonce}").as_bytes());
        let value = u64::from_le_bytes(digest.as_bytes()[..8].try_into().unwrap());
        match used.get(&value) {
            Some(existing) if existing != identity => continue,
            Some(_) => return Ok(value),
            None => {
                used.insert(value, identity.to_string());
                return Ok(value);
            }
        }
    }
    bail!("could not allocate a collision-free hash for {identity}")
}

fn push_dependency(values: &mut Vec<AgenticDependency>, dependency: AgenticDependency) {
    if !values.iter().any(|existing| {
        existing.request_id == dependency.request_id
            && existing.trigger == dependency.trigger
            && existing.relation == dependency.relation
    }) {
        values.push(dependency);
    }
}

fn validate_trace_header(trace: &WekaTrace, relative_path: &str) -> Result<()> {
    if trace.id.trim().is_empty() {
        bail!("Weka trace {} has an empty id", relative_path);
    }
    if trace.block_size == 0 {
        bail!("Weka trace {} has zero block_size", relative_path);
    }
    if trace.hash_id_scope != "local" {
        bail!(
            "Weka trace {} has unsupported hash_id_scope {:?}; expected local",
            relative_path,
            trace.hash_id_scope
        );
    }
    let _ = (
        &trace.models,
        trace.tool_tokens,
        trace.system_tokens,
        &trace.totals,
    );
    Ok(())
}

fn validate_subagent(subagent: &WekaSubagent, relative_path: &str) -> Result<SubagentMode> {
    if !subagent.t.is_finite() || subagent.t < 0.0 {
        bail!(
            "Weka trace {} has invalid subagent timestamp",
            relative_path
        );
    }
    if subagent.agent_id.trim().is_empty() {
        bail!(
            "Weka trace {} has an empty subagent agent_id",
            relative_path
        );
    }
    let mode = match subagent.status.as_str() {
        "completed" => SubagentMode::Blocking,
        "async_launched" => SubagentMode::Background,
        status => bail!(
            "Weka trace {} has unsupported non-success subagent status {:?} for {}",
            relative_path,
            status,
            subagent.agent_id
        ),
    };
    if mode == SubagentMode::Blocking && subagent.requests.is_empty() {
        bail!(
            "Weka trace {} has blocking subagent {} with no replayable requests; external waits are not modeled",
            relative_path,
            subagent.agent_id
        );
    }
    let _ = (
        &subagent.subagent_type,
        subagent.total_tokens,
        subagent.tool_use_count,
        &subagent.models,
        subagent.tool_tokens,
        subagent.system_tokens,
    );
    Ok(mode)
}

fn trace_request_model<'a>(trace: &'a WekaTrace, relative_path: &str) -> Result<&'a str> {
    let mut models = BTreeSet::new();
    for entry in &trace.requests {
        match entry {
            WekaEntry::Normal(request) | WekaEntry::Streaming(request) => {
                models.insert(request.model.as_str());
            }
            WekaEntry::Subagent(subagent) => {
                for entry in &subagent.requests {
                    let request = match entry {
                        WekaInnerEntry::Normal(request) | WekaInnerEntry::Streaming(request) => {
                            request
                        }
                    };
                    models.insert(request.model.as_str());
                }
            }
        }
    }
    if models.len() != 1 {
        bail!(
            "Weka trace {} must contain exactly one request model, found {}",
            relative_path,
            models.len()
        );
    }
    let model = models.into_iter().next().expect("one request model");
    if model.trim().is_empty() {
        bail!("Weka trace {} has an empty request model", relative_path);
    }
    Ok(model)
}

fn validate_request(request: &WekaRequest, relative_path: &str) -> Result<()> {
    if !request.t.is_finite() || request.t < 0.0 {
        bail!("Weka trace {} has invalid request timestamp", relative_path);
    }
    if request.model.trim().is_empty() {
        bail!("Weka trace {} has an empty request model", relative_path);
    }
    if request.input_length == 0 {
        bail!("Weka trace {} has a zero-length request", relative_path);
    }
    if request.api_time.is_some_and(|value| !value.is_finite()) {
        bail!("Weka trace {} has non-finite api_time", relative_path);
    }
    let _ = (
        &request.input_types,
        &request.output_types,
        &request.stop,
        request.think_time,
        request.ttft,
    );
    Ok(())
}

fn subagent_end(subagent: &WekaSubagent) -> f64 {
    if let Some(duration_ms) = subagent.duration_ms {
        return subagent.t + (duration_ms.max(0) as f64 / 1000.0);
    }
    subagent
        .requests
        .iter()
        .map(|entry| match entry {
            WekaInnerEntry::Normal(request) | WekaInnerEntry::Streaming(request) => {
                let mut request = request.clone();
                if request.t + JOIN_EPSILON_SECONDS < subagent.t {
                    request.t += subagent.t;
                }
                request_end(&request)
            }
        })
        .fold(subagent.t, f64::max)
}

fn request_end(request: &WekaRequest) -> f64 {
    request.t + request.api_time.unwrap_or(0.0).max(0.0)
}

fn seconds_to_milliseconds(seconds: f64) -> f64 {
    (seconds.max(0.0) * NANOSECONDS_PER_SECOND).round() / NANOSECONDS_PER_MILLISECOND
}

fn request_order(left: &IndexedRequest, right: &IndexedRequest) -> std::cmp::Ordering {
    left.request
        .t
        .total_cmp(&right.request.t)
        .then(left.source_order.cmp(&right.source_order))
        .then(left.source_id.cmp(&right.source_id))
}

fn lcp(left: &[u64], right: &[u64]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

fn namespace(relative_path: &str) -> String {
    let digest = blake3::hash(relative_path.as_bytes()).to_hex().to_string();
    format!("weka:{}", &digest[..16])
}

fn read_source_bytes(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))
}

fn parse_trace(path: &Path, bytes: &[u8]) -> Result<WekaTrace> {
    serde_json::from_slice(bytes)
        .with_context(|| format!("failed to parse Weka trace {}", path.display()))
}

fn collect_source_files(path: &Path) -> Result<(PathBuf, Vec<(PathBuf, String)>)> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("failed to stat Weka source {}", path.display()))?;
    if metadata.file_type().is_symlink() {
        bail!("Weka source may not be a symlink: {}", path.display());
    }
    if metadata.is_file() {
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let name = path
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or_else(|| anyhow!("Weka path is not valid UTF-8: {}", path.display()))?;
        return Ok((
            parent.to_path_buf(),
            vec![(path.to_path_buf(), name.to_string())],
        ));
    }
    if !metadata.is_dir() {
        bail!(
            "Weka source is neither a file nor directory: {}",
            path.display()
        );
    }
    let mut files = Vec::new();
    visit_directory(path, path, &mut files)?;
    files.sort_by(|left, right| left.1.cmp(&right.1));
    Ok((path.to_path_buf(), files))
}

fn visit_directory(
    root: &Path,
    directory: &Path,
    files: &mut Vec<(PathBuf, String)>,
) -> Result<()> {
    let mut entries = std::fs::read_dir(directory)
        .with_context(|| format!("failed to read Weka directory {}", directory.display()))?
        .collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        let metadata = std::fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() {
            bail!("Weka corpus may not contain symlinks: {}", path.display());
        }
        if metadata.is_dir() {
            visit_directory(root, &path, files)?;
            continue;
        }
        if !metadata.is_file() || path.extension().and_then(|value| value.to_str()) != Some("json")
        {
            continue;
        }
        let relative = path.strip_prefix(root).expect("visited path is below root");
        let relative = normalize_relative_path(relative)?;
        files.push((path, relative));
    }
    Ok(())
}

fn normalize_relative_path(path: &Path) -> Result<String> {
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(value) => parts.push(
                value
                    .to_str()
                    .ok_or_else(|| anyhow!("Weka path is not valid UTF-8: {}", path.display()))?,
            ),
            _ => bail!("Weka relative path is not normalized: {}", path.display()),
        }
    }
    Ok(parts.join("/"))
}

fn update_corpus_digest(hasher: &mut blake3::Hasher, relative_path: &str, bytes: &[u8]) {
    let relative_path = relative_path.as_bytes();
    hasher.update(&(relative_path.len() as u64).to_le_bytes());
    hasher.update(relative_path);
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn request(t: f64, input: usize, output: usize, hashes: &[u64]) -> serde_json::Value {
        serde_json::json!({
            "t": t,
            "type": "s",
            "model": "model",
            "in": input,
            "out": output,
            "hash_ids": hashes,
        })
    }

    fn write_trace(path: &Path, requests: serde_json::Value) {
        let trace = serde_json::json!({
            "id": "play",
            "models": ["model"],
            "block_size": 4,
            "hash_id_scope": "local",
            "requests": requests,
        });
        std::fs::write(path, serde_json::to_vec(&trace).unwrap()).unwrap();
    }

    #[test]
    fn explicit_overlap_join_and_background_lower_to_typed_edges() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.json");
        write_trace(
            &path,
            serde_json::json!([
                {"t":0.0,"type":"s","model":"model","in":8,"out":1,"hash_ids":[1,2],"api_time":1.0},
                {"t":0.2,"type":"subagent","agent_id":"a","subagent_type":"Explore","duration_ms":400,"status":"completed","requests":[
                    {"t":0.3,"type":"s","model":"model","in":6,"out":1,"hash_ids":[3,4],"api_time":0.2}
                ],"models":["model"]},
                {"t":1.0,"type":"s","model":"model","in":9,"out":1,"hash_ids":[1,2,5]},
                {"t":1.1,"type":"subagent","agent_id":"bg","subagent_type":"Explore","status":"async_launched","requests":[
                    {"t":1.2,"type":"s","model":"model","in":4,"out":0,"hash_ids":[9],"api_time":0.1}
                ],"models":["model"]},
                {"t":1.5,"type":"s","model":"model","in":13,"out":1,"hash_ids":[1,2,5,6]}
            ]),
        );

        let (summary, rows) = load_weka_agentic_rows(&path).unwrap();
        assert_eq!(summary.requests, 5);
        assert_eq!(summary.raw_zero_outputs, 1);
        let child = rows
            .iter()
            .find(|row| row.request_id.ends_with("outer:1:inner:0"))
            .unwrap();
        assert!(child.dependencies.iter().any(|edge| {
            edge.trigger == AgenticDependencyTrigger::Dispatch
                && edge.relation == AgenticDependencyRelation::Spawn
        }));
        let consumer = rows
            .iter()
            .find(|row| row.request_id.ends_with("outer:2"))
            .unwrap();
        assert!(
            consumer.dependencies.iter().any(|edge| {
                edge.request_id == child.request_id
                    && edge.relation == AgenticDependencyRelation::Join
                    && (edge.delay_ms - 500.0).abs() < 1e-6
            }),
            "lowered rows: {rows:#?}"
        );
        let background = rows
            .iter()
            .find(|row| row.request_id.ends_with("outer:3:inner:0"))
            .unwrap();
        assert_eq!(background.output_length, Some(1));
        assert!(!rows.iter().any(|row| {
            row.dependencies.iter().any(|edge| {
                edge.request_id == background.request_id
                    && edge.relation == AgenticDependencyRelation::Join
            })
        }));
    }

    #[test]
    fn post_completion_spawn_and_equality_join_preserve_recorded_timing() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.json");
        write_trace(
            &path,
            serde_json::json!([
                {"t":0.0,"type":"s","model":"model","in":8,"out":1,"hash_ids":[1,2],"api_time":0.5},
                {"t":0.8,"type":"subagent","agent_id":"a","subagent_type":"Explore","duration_ms":400,"status":"completed","requests":[
                    {"t":0.9,"type":"s","model":"model","in":8,"out":1,"hash_ids":[3,4],"api_time":0.2},
                    {"t":1.15,"type":"s","model":"model","in":12,"out":1,"hash_ids":[3,4,5],"api_time":0.05}
                ],"models":["model"]},
                {"t":1.2,"type":"s","model":"model","in":12,"out":1,"hash_ids":[1,2,6]}
            ]),
        );

        let (_, rows) = load_weka_agentic_rows(&path).unwrap();
        let parent = rows
            .iter()
            .find(|row| row.request_id.ends_with("outer:0"))
            .unwrap();
        let first_child = rows
            .iter()
            .find(|row| row.request_id.ends_with("outer:1:inner:0"))
            .unwrap();
        let second_child = rows
            .iter()
            .find(|row| row.request_id.ends_with("outer:1:inner:1"))
            .unwrap();
        let consumer = rows
            .iter()
            .find(|row| row.request_id.ends_with("outer:2"))
            .unwrap();

        assert!(first_child.dependencies.iter().any(|edge| {
            edge.request_id == parent.request_id
                && edge.trigger == AgenticDependencyTrigger::Completion
                && edge.relation == AgenticDependencyRelation::Spawn
                && (edge.delay_ms - 400.0).abs() < 1e-6
        }));
        assert!(second_child.dependencies.iter().any(|edge| {
            edge.request_id == first_child.request_id
                && edge.trigger == AgenticDependencyTrigger::Completion
                && edge.relation == AgenticDependencyRelation::Sequence
                && (edge.delay_ms - 50.0).abs() < 1e-6
        }));
        assert!(consumer.dependencies.iter().any(|edge| {
            edge.request_id == second_child.request_id
                && edge.trigger == AgenticDependencyTrigger::Completion
                && edge.relation == AgenticDependencyRelation::Join
        }));
    }

    #[test]
    fn flattened_hash_fork_becomes_a_dispatch_spawned_child_stream() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.json");
        write_trace(
            &path,
            serde_json::json!([
                {"t":0.0,"type":"s","model":"model","in":8,"out":1,"hash_ids":[1,2],"api_time":2.0},
                {"t":0.5,"type":"s","model":"model","in":8,"out":1,"hash_ids":[1,3],"api_time":0.2},
                {"t":0.8,"type":"s","model":"model","in":12,"out":1,"hash_ids":[1,3,4],"api_time":0.2},
                {"t":2.0,"type":"s","model":"model","in":12,"out":1,"hash_ids":[1,2,5]}
            ]),
        );

        let (_, rows) = load_weka_agentic_rows(&path).unwrap();
        let parent = rows
            .iter()
            .find(|row| row.request_id.ends_with("outer:0"))
            .unwrap();
        let child = rows
            .iter()
            .find(|row| row.request_id.ends_with("outer:1"))
            .unwrap();
        let child_next = rows
            .iter()
            .find(|row| row.request_id.ends_with("outer:2"))
            .unwrap();
        let main_next = rows
            .iter()
            .find(|row| row.request_id.ends_with("outer:3"))
            .unwrap();

        assert_ne!(parent.session_id, child.session_id);
        assert_eq!(child.session_id, child_next.session_id);
        assert!(child.dependencies.iter().any(|edge| {
            edge.request_id == parent.request_id
                && edge.trigger == AgenticDependencyTrigger::Dispatch
                && edge.relation == AgenticDependencyRelation::Spawn
                && (edge.delay_ms - 500.0).abs() < 1e-6
        }));
        assert!(child_next.dependencies.iter().any(|edge| {
            edge.request_id == child.request_id
                && edge.trigger == AgenticDependencyTrigger::Completion
                && edge.relation == AgenticDependencyRelation::Sequence
                && (edge.delay_ms - 100.0).abs() < 1e-6
        }));
        assert!(main_next.dependencies.iter().any(|edge| {
            edge.request_id == child_next.request_id
                && edge.trigger == AgenticDependencyTrigger::Completion
                && edge.relation == AgenticDependencyRelation::ReplayBarrier
        }));
    }

    #[test]
    fn explicit_subagent_spawn_and_join_use_one_parent_stream() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.json");
        write_trace(
            &path,
            serde_json::json!([
                {"t":0.0,"type":"s","model":"model","in":8,"out":1,"hash_ids":[1,2],"api_time":1.0},
                {"t":0.2,"type":"s","model":"model","in":8,"out":1,"hash_ids":[1,3],"api_time":0.2},
                {"t":0.25,"type":"subagent","agent_id":"owned","subagent_type":"Explore","duration_ms":500,"status":"completed","requests":[
                    {"t":0.3,"type":"s","model":"model","in":4,"out":1,"hash_ids":[9],"api_time":0.1}
                ],"models":["model"]},
                {"t":1.0,"type":"s","model":"model","in":12,"out":1,"hash_ids":[1,2,4]},
                {"t":1.2,"type":"s","model":"model","in":12,"out":1,"hash_ids":[1,3,5]}
            ]),
        );

        let (_, rows) = load_weka_agentic_rows(&path).unwrap();
        let owner = rows
            .iter()
            .find(|row| row.request_id.ends_with("outer:1"))
            .unwrap();
        let child = rows
            .iter()
            .find(|row| row.request_id.ends_with("outer:2:inner:0"))
            .unwrap();
        let other_stream_continuation = rows
            .iter()
            .find(|row| row.request_id.ends_with("outer:3"))
            .unwrap();
        let owner_stream_continuation = rows
            .iter()
            .find(|row| row.request_id.ends_with("outer:4"))
            .unwrap();

        assert!(child.dependencies.iter().any(|edge| {
            edge.request_id == owner.request_id
                && edge.trigger == AgenticDependencyTrigger::Dispatch
                && edge.relation == AgenticDependencyRelation::Spawn
        }));
        assert!(!other_stream_continuation.dependencies.iter().any(|edge| {
            edge.request_id == child.request_id && edge.relation == AgenticDependencyRelation::Join
        }));
        assert!(owner_stream_continuation.dependencies.iter().any(|edge| {
            edge.request_id == child.request_id
                && edge.trigger == AgenticDependencyTrigger::Completion
                && edge.relation == AgenticDependencyRelation::Join
        }));
    }

    #[test]
    fn local_hashes_are_namespaced_and_partial_tails_are_private() {
        let directory = tempdir().unwrap();
        write_trace(
            &directory.path().join("a.json"),
            serde_json::json!([request(0.0, 6, 1, &[7, 8])]),
        );
        write_trace(
            &directory.path().join("b.json"),
            serde_json::json!([request(0.0, 6, 1, &[7, 8])]),
        );
        let (_, rows) = load_weka_agentic_rows(directory.path()).unwrap();
        assert_eq!(rows.len(), 2);
        assert_ne!(rows[0].hash_ids, rows[1].hash_ids);
        assert_eq!(rows[0].hash_ids.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn extra_hashes_are_truncated_and_missing_blocks_are_request_private() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.json");
        write_trace(
            &path,
            serde_json::json!([request(0.0, 8, 1, &[7, 8, 999]), request(1.0, 12, 1, &[7])]),
        );

        let (_, rows) = load_weka_agentic_rows(&path).unwrap();
        let first = rows
            .iter()
            .find(|row| row.request_id.ends_with("outer:0"))
            .unwrap()
            .hash_ids
            .as_ref()
            .unwrap();
        let second = rows
            .iter()
            .find(|row| row.request_id.ends_with("outer:1"))
            .unwrap()
            .hash_ids
            .as_ref()
            .unwrap();

        assert_eq!(first.len(), 2);
        assert_eq!(second.len(), 3);
        assert_eq!(first[0], second[0]);
        assert_ne!(first[1], second[1]);
        assert_ne!(second[1], second[2]);
    }

    #[test]
    fn corpus_preflight_is_deterministic_and_rejects_mixed_block_sizes() {
        let first = tempdir().unwrap();
        let second = tempdir().unwrap();
        for directory in [&first, &second] {
            std::fs::create_dir(directory.path().join("nested")).unwrap();
        }
        write_trace(
            &first.path().join("nested/b.json"),
            serde_json::json!([request(0.0, 4, 1, &[2])]),
        );
        write_trace(
            &first.path().join("a.json"),
            serde_json::json!([request(0.0, 4, 1, &[1])]),
        );
        write_trace(
            &second.path().join("a.json"),
            serde_json::json!([request(0.0, 4, 1, &[1])]),
        );
        write_trace(
            &second.path().join("nested/b.json"),
            serde_json::json!([request(0.0, 4, 1, &[2])]),
        );
        assert_eq!(
            WekaImporter::open(first.path())
                .unwrap()
                .header()
                .source
                .digest,
            WekaImporter::open(second.path())
                .unwrap()
                .header()
                .source
                .digest
        );

        let mixed_path = second.path().join("nested/b.json");
        let mut mixed: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&mixed_path).unwrap()).unwrap();
        mixed["block_size"] = 8.into();
        std::fs::write(&mixed_path, serde_json::to_vec(&mixed).unwrap()).unwrap();
        let error = WekaImporter::open(second.path())
            .err()
            .expect("mixed blocks");
        assert!(error.to_string().contains("mixes block sizes"), "{error:#}");
    }

    #[test]
    fn emission_rejects_source_changes_after_preflight() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.json");
        write_trace(&path, serde_json::json!([request(0.0, 4, 1, &[1])]));
        let importer = WekaImporter::open(&path).unwrap();

        write_trace(&path, serde_json::json!([request(0.0, 4, 1, &[2])]));
        let mut emitted = Vec::new();
        let error = importer
            .for_each_row(|row| {
                emitted.push(row);
                Ok(())
            })
            .unwrap_err();

        assert!(emitted.is_empty());
        assert!(
            error
                .to_string()
                .contains("Weka source trace.json changed after preflight"),
            "{error:#}"
        );
    }

    #[test]
    fn corpus_preflight_rejects_mixed_request_models() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.json");
        write_trace(
            &path,
            serde_json::json!([
                request(0.0, 4, 1, &[1]),
                {"t":1.0,"type":"s","model":"other-model","in":4,"out":1,"hash_ids":[2]}
            ]),
        );

        let error = WekaImporter::open(&path)
            .err()
            .expect("mixed models must fail preflight");
        assert!(
            error
                .to_string()
                .contains("must contain exactly one request model"),
            "{error:#}"
        );
    }

    #[test]
    fn subagent_status_controls_background_and_blocking_validation() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.json");
        let cases = [
            (
                "failed",
                serde_json::json!([request(0.0, 4, 1, &[1]), {
                    "t":0.1,"type":"subagent","agent_id":"failed","subagent_type":"Explore","status":"failed","requests":[
                        {"t":0.2,"type":"s","model":"model","in":4,"out":1,"hash_ids":[2]}
                    ],"models":["model"]
                }]),
                "unsupported non-success subagent status",
            ),
            (
                "blocking empty",
                serde_json::json!([request(0.0, 4, 1, &[1]), {
                    "t":0.1,"type":"subagent","agent_id":"empty","subagent_type":"Explore","status":"completed","requests":[],"models":["model"]
                }]),
                "external waits are not modeled",
            ),
        ];
        for (name, requests, expected) in cases {
            write_trace(&path, requests);
            let error = load_weka_agentic_rows(&path).expect_err(name);
            assert!(error.to_string().contains(expected), "{name}: {error:#}");
        }

        write_trace(
            &path,
            serde_json::json!([
                request(0.0, 4, 1, &[1]),
                {"t":0.1,"type":"subagent","agent_id":"background","subagent_type":"Explore","status":"async_launched","requests":[],"models":[]},
                request(1.0, 8, 1, &[1,2])
            ]),
        );
        let (summary, rows) = load_weka_agentic_rows(&path).unwrap();
        assert_eq!(summary.requests, 2);
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn subagent_without_a_preceding_parent_is_rejected() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trace.json");
        write_trace(
            &path,
            serde_json::json!([
                {"t":0.0,"type":"subagent","agent_id":"orphan","subagent_type":"Explore","status":"completed","requests":[
                    {"t":0.0,"type":"s","model":"model","in":4,"out":1,"hash_ids":[9]}
                ],"models":["model"]},
                request(1.0, 4, 1, &[1])
            ]),
        );

        let error = load_weka_agentic_rows(&path).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("subagent orphan at outer index 0 without a preceding parent request"),
            "{error:#}"
        );
    }
}
