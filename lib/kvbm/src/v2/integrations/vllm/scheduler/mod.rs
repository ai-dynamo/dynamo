// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! v1 Scheduler (Rust) – high-level sketch and scaffolding
//!
//! This module provides a Rust-native scheduler interface and data models that
//! mirror the vLLM v1 scheduler concepts, with deliberate deviations in the
//! control flow to enable:
//! - a staging thread that performs expensive pre-checks and block matches
//! - a ready-to-execute queue for the current forward pass
//! - parallel processing of per-request outputs from the model runner
//!
//! The implementation below focuses on clear, strongly typed Rust APIs and
//! the separation of Python boundary conversions from the hot-path logic.
//! Many functions are left as stubs with detailed TODO notes.
//!
//! PERFORMANCE NOTES:
//! - Uses Arc<Mutex<RequestState>> to allow concurrent access to different requests
//! - Avoids cloning Arc where possible (just borrow references)
//! - Only clones strings when building output structs
//! - Mutexes are per-request, so different requests can be accessed concurrently
//! - Rust's borrow checker prevents multiple mutable access to HashMap values simultaneously
//!   (even if just mutating the values, not the map structure), hence the mutex pattern
//! - Consider rayon::scope for parallel processing in update_from_output when performance is critical

// pub mod leader;
// pub mod comms;
pub mod worker;

use super::KvbmVllmConfig;

#[allow(dead_code)]
mod next {

    use derive_getters::Dissolve;
    use tokio::sync::OwnedSemaphorePermit;

    use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
    use std::hash::{Hash, Hasher};
    use std::sync::{Arc, Mutex};

    // NOTE: Keep this module self-contained and compilation-safe.
    // We avoid depending on other internal modules until integration time.

    /// Unique identifier for a request.
    pub type RequestId = String;

    /// Zero-based index of a client (used to route outputs back to the origin).
    pub type ClientId = i32;

    /// Logical block identifier in the KV cache.
    pub type BlockId = u32;

    /// Unix monotonic timestamp in seconds.
    pub type MonotonicTs = f64;

    /// Small wrapper to indicate the configured LoRA limits.
    ///
    /// vLLM enforces a maximum number of concurrent LoRA adapters for a step.
    /// When the limit is reached, additional requests using different LoRA IDs are
    /// deferred for the current step. We mirror that accounting here.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct LoraLimits {
        pub max_loras: Option<usize>,
    }

    /// Constants extracted from vLLM configs (scheduler + cache + parallel + spec).
    ///
    /// Initialize from vLLM:
    /// - `scheduler_config.max_num_seqs`, `max_num_batched_tokens`, `max_model_len`
    /// - `cache_config.block_size`, `cache_config.num_gpu_blocks (> 0)`
    /// - `parallel_config.pipeline_parallel_size (> 1 => PP enabled)`
    /// - `speculative_config.num_speculative_tokens` (if enabled)
    /// - `include_finished_set`, `log_stats`
    #[derive(Clone, Debug)]
    pub struct SchedulerConstants {
        /// Size of a KV block in tokens. vLLM: `cache_config.block_size`.
        /// Used for ceil-div computations of memory and for offload planning.
        pub block_size: usize,

        /// Maximum sequence length (prompt + output) supported by model.
        /// vLLM: `scheduler_config.max_model_len`. Used to clamp scheduled tokens.
        pub max_model_len: usize,
        /// Maximum number of concurrently running sequences. vLLM: `max_num_seqs`.
        pub max_num_seqs: usize,
        /// Global per-step token budget across requests. vLLM: `max_num_batched_tokens`.
        pub max_num_batched_tokens: usize,

        /// Cap for prefill chunk size used to bound latency. vLLM may split long
        /// prefills using this threshold. If set, we min(num_new_tokens, threshold).
        pub long_prefill_token_threshold: Option<usize>,

        /// Whether prefill can be chunked even if it would exceed the remaining
        /// token budget. If false, such requests are deferred. Mirrors vLLM logic.
        pub chunked_prefill_enabled: bool,
        /// When true, multimodal encoder inputs are not chunked; each item must
        /// be scheduled whole or postponed. Mirrors vLLM `disable_chunked_mm_input`.
        pub disable_chunked_mm_input: bool,

        /// Whether pipeline parallelism is enabled (PP > 1). If true, scheduler
        /// may need to return `new_token_ids` back to the first-stage worker.
        pub pipeline_parallel: bool,

        /// Include finished request IDs in outputs for efficient lifetime tracking
        /// in multi-engine setups. Mirrors vLLM option.
        pub include_finished_set: bool,
        /// Emit stats records each step (prefix cache, spec decoding, counters).
        pub log_stats: bool,

        /// Whether speculative decoding is enabled and the number of speculative
        /// tokens (lookahead) per request per step.
        pub use_spec_decode: bool,
        pub num_spec_tokens: usize,

        /// LoRA concurrency limits for each step.
        pub lora_limits: LoraLimits,
        /// Total device KV blocks available to allocate (from vLLM cache config).
        pub total_gpu_blocks: usize,
    }

    impl SchedulerConstants {
        pub fn token_budget(&self) -> usize {
            self.max_num_batched_tokens
        }
    }

    /// Status of a request – mirrors `vllm.v1.request.RequestStatus`.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    pub enum RequestStatus {
        /// The request is queued and waiting to be scheduled.
        Waiting,
        /// Waiting for FSM compilation for structured output (guided decoding).
        WaitingForFsm,
        /// Waiting for remote KV transfers to complete before resuming.
        WaitingForKvLoad,
        /// Currently scheduled/running in this or a previous step.
        Running,
        /// Temporarily descheduled due to resource pressure (candidate for resume).
        Preempted,
        /// Terminal states (anything after PREEMPTED is considered finished):
        FinishedStopped,
        FinishedLengthCapped,
        FinishedAborted,
        FinishedIgnored,
    }

    impl RequestStatus {
        pub fn is_finished(self) -> bool {
            self > RequestStatus::Preempted
        }
    }

    /// Minimal LoRA descriptor required by the scheduler for accounting.
    #[derive(Clone, Debug, Default)]
    pub struct LoraRequestLight {
        pub lora_int_id: i64,
    }

    /// Placeholder for structured-output support.
    #[derive(Clone, Debug, Default)]
    pub struct StructuredOutputLight {
        pub enabled: bool,
    }

    /// Placeholder for multimodal positions (encoder inputs).
    #[derive(Clone, Debug, Default)]
    pub struct MmPosition {
        pub offset: usize,
        pub length: usize,
    }

    /// State tracked per request inside the scheduler.
    #[derive(Clone, Debug)]
    pub struct RequestState {
        /// Unique request id (string).
        pub request_id: RequestId,

        /// Originating client index; used to route outputs to the right frontend.
        pub client_id: ClientId,

        /// Larger means higher priority (used for preemption policies).
        pub priority: i32,

        /// Current status; transitions are driven by schedule and runner outputs.
        pub status: RequestStatus,

        /// Request-specific EOS token (can differ per LoRA). Used for stop checks.
        pub eos_token_id: Option<i32>,

        /// LoRA context for accounting (see `LoraLimits`).
        pub lora: Option<LoraRequestLight>,

        /// Salt Hash
        pub salt_hash: Option<u64>,

        /// Whether structured decoding is enabled; integrates with grammar bitmasks.
        pub structured_output: StructuredOutputLight,

        /// Monotonic arrival time; used in tie-breaking and stats.
        pub arrival_time: MonotonicTs,

        /// Prompt tokens received at request creation.
        pub prompt_token_ids: Vec<i32>,
        /// Generated tokens so far (grows over steps).
        pub output_token_ids: Vec<i32>,
        /// Concatenation of prompt + output (read-only view in vLLM).
        pub all_token_ids: Vec<i32>,

        /// Used by async scheduling to pre-reserve positions.
        pub num_output_placeholders: usize,
        /// Draft tokens for speculative decoding (to be validated next step).
        pub spec_token_ids: Vec<i32>,
        /// Tokens that have been computed (prefix-cached + executed this step).
        /// vLLM advances this AFTER schedule() and may adjust in update_from_output
        /// to account for rejected speculative tokens.
        pub num_computed_tokens: usize,
        /// Number of tokens served by prefix cache (>= 0 once known).
        pub num_cached_tokens: i32,
        /// Indicator of corrupted outputs (NaNs in logits); > 0 means corrupted.
        pub num_nans_in_logits: i32,

        /// Positions and lengths of multimodal encoder inputs in the token stream.
        pub mm_positions: Vec<MmPosition>,
        /// Whether the request has encoder inputs (e.g., images for LMMs).
        pub has_encoder_inputs: bool,

        /// Request-specific salt used in block hashing. In vLLM this is
        /// `Request.cache_salt`; we use it as the KV block hashing salt (aka
        /// salt_hash) when creating connector slots or computing block hashes.
        pub cache_salt: Option<String>,
        /// Maximum allowed output tokens for this request. In vLLM derived from
        /// `sampling_params.max_tokens` or set to 1 for pooling models.
        /// Used for worst-case projections.
        pub max_tokens: usize,
    }

    impl RequestState {
        pub fn num_tokens(&self) -> usize {
            self.all_token_ids.len()
        }

        pub fn num_tokens_with_spec(&self) -> usize {
            self.all_token_ids.len() + self.spec_token_ids.len()
        }

        pub fn use_structured_output(&self) -> bool {
            self.structured_output.enabled
        }
    }

    /// Data for requests scheduled for the first time in a step.
    #[derive(Clone, Debug, Dissolve)]
    pub struct NewRequestData {
        /// Request id for the new (first-time scheduled) request.
        pub request_id: RequestId,
        /// Prompt tokens to be cached by workers so we don't resend every step.
        pub prompt_token_ids: Vec<i32>,
        /// New block ids allocated this step per KV cache group.
        pub block_ids: Vec<Vec<BlockId>>, // per cache group
        /// Value of `num_computed_tokens` after schedule() for the request.
        pub num_computed_tokens: usize,
        /// Optional LoRA metadata to cache on workers.
        pub lora: Option<LoraRequestLight>,
        /// Hashing Salt
        pub salt_hash: Option<u64>,
    }

    /// Data for requests that were seen before; we send only incremental info.
    #[derive(Clone, Debug, Default)]
    pub struct CachedRequestData {
        /// Request id for a request that has been seen in previous steps.
        pub request_id: RequestId,
        /// If true, indicates this request was preempted and has just resumed,
        /// thus the provided block ids should replace, not append.
        pub resumed_from_preemption: bool,
        /// When pipeline parallelism is enabled, sampled token ids to return to
        /// the first-stage worker; otherwise left empty.
        pub new_token_ids: Vec<i32>,
        /// New block ids allocated this step per KV cache group.
        pub new_block_ids: Vec<Vec<BlockId>>, // per cache group
        /// `num_computed_tokens` before applying the tokens scheduled this step.
        pub num_computed_tokens: usize,
    }

    /// Batch-scoped output from the scheduler.
    #[derive(Clone, Debug, Default)]
    pub struct SchedulerOutput {
        /// New requests scheduled for the first time this step.
        pub new_requests: Vec<NewRequestData>,
        /// Previously-seen requests with incremental diffs for this step.
        pub cached_requests: Vec<CachedRequestData>,

        /// Per-request tokens scheduled this step (before spec rejections).
        pub num_scheduled_tokens: HashMap<RequestId, usize>,
        /// Sum of all scheduled tokens.
        pub total_num_scheduled_tokens: usize,

        /// If present, the speculative draft tokens scheduled for validation.
        pub scheduled_spec_decode_tokens: HashMap<RequestId, Vec<i32>>,
        /// Encoder input indices to process in this step per request.
        pub scheduled_encoder_inputs: HashMap<RequestId, Vec<usize>>, // indices into mm inputs
        /// Number of common prefix blocks across requests per KV group (for cascade attention).
        pub num_common_prefix_blocks: Vec<usize>,

        /// Requests that finished between previous and current steps; used by
        /// workers to free per-request cached state.
        pub finished_req_ids: BTreeSet<RequestId>,
        /// Pairs of (req_id, encoder_input_index) to free from encoder caches.
        pub free_encoder_input_ids: Vec<(RequestId, usize)>,

        /// Mapping from req_id to its index in the batch for grammar bitmask slicing.
        pub structured_output_request_ids: HashMap<RequestId, usize>,
    }

    /// Per-request output returned to the engine frontend.
    #[derive(Clone, Debug, Default)]
    pub struct EngineCoreOutput {
        /// Id of the request.
        pub request_id: RequestId,
        /// Newly generated token ids for this step.
        pub new_token_ids: Vec<i32>,
        pub new_logprobs: Option<()>, // TODO: add logprobs tensors/lists when needed
        pub new_prompt_logprobs_tensors: Option<()>,
        pub pooling_output: Option<()>, // TODO: add tensor handle when needed
        pub finish_reason: Option<i32>, // Map to FinishReason codes
        pub stop_reason: Option<i32>,
        pub events: Option<Vec<()>>, // TODO
        /// Connector-specific metadata to instruct workers on KV transfer.
        pub kv_transfer_params: Option<serde_json::Value>,
        /// The number of tokens served from prefix cache (for stats/clients).
        pub num_cached_tokens: i32,
    }

    /// Grouped outputs for a client index.
    #[derive(Clone, Debug, Default)]
    pub struct EngineCoreOutputs {
        /// Outputs for all requests that originated from this client.
        pub outputs: Vec<EngineCoreOutput>,
        /// Optional set of requests that finished since last step (multi-engine).
        pub finished_requests: Option<BTreeSet<RequestId>>,
    }

    /// Owned, Rust-native representation of `vllm.v1.outputs.ModelRunnerOutput`.
    #[derive(Clone, Debug, Default)]
    pub struct OwnedModelRunnerOutput {
        /// Request ids in the same order as outputs (batch order).
        pub req_ids: Vec<RequestId>,
        /// Inverse map from req_id to its index in arrays.
        pub req_id_to_index: HashMap<RequestId, usize>,
        /// Generated token ids per request for this step (variable length per req).
        pub sampled_token_ids: Vec<Vec<i32>>, // per-request variable length
        /// Speculative draft tokens per request (None if spec decoding disabled).
        pub spec_token_ids: Option<Vec<Vec<i32>>>,
        /// Optional sampled logprobs (lists or tensors) — converted to Rust-friendly form.
        pub logprobs: Option<()>, // TODO
        /// Prompt logprobs per request id (None for non-prefill steps).
        pub prompt_logprobs_dict: HashMap<RequestId, Option<()>>, // TODO
        /// Optional pooling outputs per request (embeddings for pooler models).
        pub pooler_output: Vec<Option<()>>, // TODO
        /// From connector output: req_ids that finished receiving remote KVs.
        pub kv_connector_finished_recving: Option<BTreeSet<RequestId>>,
        /// From connector output: req_ids that finished sending and can be freed.
        pub kv_connector_finished_sending: Option<BTreeSet<RequestId>>,
        /// Number of NaNs observed in logits per request (detect corruption).
        pub num_nans_in_logits: Option<HashMap<RequestId, i32>>,
    }

    impl OwnedModelRunnerOutput {
        pub fn is_empty(&self) -> bool {
            self.req_ids.is_empty()
        }
    }

    /// Scheduling policy – limited to priority and FCFS for now.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum SchedulingPolicy {
        Priority,
        Fcfs,
    }

    /// Interface for (pluggable) KV cache manager used by the scheduler.
    ///
    /// Note: This trait is designed to be called from the scheduler's main thread.
    /// If you need concurrent access from multiple threads, consider wrapping the
    /// implementation in Arc<Mutex<>> or using interior mutability patterns.
    pub trait KvCacheManager: Send + Sync {
        /// Try to allocate device slots for `num_new_tokens` (plus lookahead if any).
        ///
        /// Called by the scheduler during `schedule()` for both running requests
        /// and newly staged requests. If allocation fails (returns None), the
        /// scheduler may attempt preemption (or skip) and try again later.
        fn allocate_slots(
            &self,
            request: &RequestState,
            num_new_tokens: usize,
            num_lookahead_tokens: usize,
        ) -> Option<Vec<Vec<BlockId>>>;

        /// Free all resources for a finished request.
        ///
        /// Called when a request transitions to any finished status or when the
        /// connector indicates GPU blocks can be reclaimed.
        fn free(&self, request: &RequestState);

        /// Called to update prefix cache stats or hashes when a request is freed.
        fn free_block_hashes(&self, _request: &RequestState) {}

        /// Compute common-prefix blocks across the batch; sized by kv cache groups.
        ///
        /// Called at the end of `schedule()` to inform potential cascade attention
        /// optimizations downstream.
        fn get_num_common_prefix_blocks(
            &self,
            _sample: &RequestState,
            _batch_size: usize,
        ) -> Vec<usize> {
            vec![]
        }

        /// For new requests: get locally computed blocks/tokens (prefix cache hits).
        ///
        /// Typically used by a staging thread to determine how many tokens can be
        /// served from local cache and to initialize `num_computed_tokens`.
        fn get_computed_blocks(&self, _request: &RequestState) -> (Vec<Vec<BlockId>>, usize) {
            (vec![], 0)
        }

        /// Return block ids for a given request (all groups concatenated if needed).
        ///
        /// Used when finalizing a request to free resources or by connectors to
        /// identify which blocks are tied to the request.
        fn get_block_ids(&self, _request_id: &RequestId) -> (Vec<BlockId>,);

        /// Cache blocks up to `num_computed_tokens` for a request (prefix cache).
        ///
        /// Used after remote KV reception to mark blocks cacheable for future reuse.
        fn cache_blocks(&self, _request: &RequestState, _num_computed_tokens: usize) {}
    }

    // KV blocks can be loaded async or sync

    // KV blocks loads can be managed by the leader or workers
    // if managed by the workers, then

    /// The method of loading KV blocks
    pub enum KvLoadType {
        LeaderAsync,
        LeaderSync,
        WorkerAsync,
        WorkerSync,
    }

    pub struct ConnectorMatchResult {
        /// The number of tokens from the start of the sequences that are already
        /// present and provided by the caller
        pub num_computed_tokens: usize,

        /// The number of tokens that are matched by the connectors data sources.
        /// These tokens represent the range in the input sequence
        /// `(num_computed_tokens..num_connector_tokens)`
        pub num_connector_tokens: usize,

        /// The method used to load KV blocks
        pub load_type: KvLoadType,
    }

    /// Optional KV connector interface used for P/D disaggregation and offloading.
    ///
    /// Note: This trait is designed to be called from the scheduler's main thread.
    /// If you need concurrent access from multiple threads, consider wrapping the
    /// implementation in Arc<Mutex<>> or using interior mutability patterns.
    pub trait KvConnector: Send + Sync {
        /// Determine how many tokens match external caches and whether to load
        /// remote KVs asynchronously. In vLLM, called for waiting requests during
        /// scheduling; in this design, expected to run in a staging thread.
        fn get_matched_connector_tokens(
            &self,
            request: &RequestState,
            num_computed_tokens: usize,
        ) -> (usize, /*load_kv_async*/ bool);

        /// Called after device allocation to update connector-internal state and
        /// possibly trigger onboarding of remote tokens into the allocated blocks.
        fn update_state_after_alloc(
            &self,
            request: &RequestState,
            all_block_ids_for_step: &[Vec<BlockId>],
            num_external_computed_tokens: usize,
        );

        /// Build opaque connector metadata to be attached to the scheduler output
        /// and consumed by workers to execute KV transfers.
        fn build_connector_metadata(&self, _output: &SchedulerOutput) -> Option<Vec<u8>> {
            None
        }

        /// Notify the connector that a request has finished. Returns whether block
        /// freeing must be delayed (true) and optional kv_transfer_params to attach
        /// to the client's output.
        fn request_finished(
            &self,
            _request: &RequestState,
            _block_ids: &[BlockId],
        ) -> (/*delay_free*/ bool, Option<serde_json::Value>) {
            (false, None)
        }

        /// Update connector state with finished send/recv notifications coming from
        /// the worker processes for the previous step.
        fn update_connector_output(
            &self,
            _finished_recving: &BTreeSet<RequestId>,
            _finished_sending: &BTreeSet<RequestId>,
        ) {
        }
    }

    /// Structured output manager – used to build grammar bitmasks and advance FSMs.
    pub trait StructuredOutputManager: Send + Sync {
        /// Build grammar bitmask for structured decoding for the entire batch,
        /// slicing per request based on `structured_output_request_ids`.
        fn grammar_bitmask(
            &self,
            _requests: &HashMap<RequestId, Arc<Mutex<RequestState>>>,
            structured_output_request_ids: &HashMap<RequestId, usize>,
            scheduled_spec_decode_tokens: &HashMap<RequestId, Vec<i32>>,
        ) -> Option<Vec<i32>>; // placeholder for NDArray

        /// Whether the structured output FSM should advance based on new tokens.
        fn should_advance(&self, _request: &RequestState) -> bool {
            false
        }
    }

    /// The main scheduler trait – mirrors vLLM's `SchedulerInterface` with Rust-native models.
    pub trait Scheduler: Send + Sync {
        /// Plan a single scheduling step (one forward pass worth of work), subject
        /// to token and sequence budgets. Returns a `SchedulerOutput` with new and
        /// cached request diffs for workers to prepare inputs and execute.
        fn schedule(&mut self) -> SchedulerOutput;

        /// Consume model runner outputs to update internal state, produce client
        /// outputs, and free finished requests. Must handle speculative token
        /// rejections and connector updates.
        fn update_from_output(
            &mut self,
            scheduler_output: &SchedulerOutput,
            model_runner_output: &OwnedModelRunnerOutput,
        ) -> HashMap<ClientId, EngineCoreOutputs>;

        /// Enqueue a new request into the scheduler. A staging thread may process
        /// it before `schedule()` moves it to the running set.
        fn add_request(&mut self, request: RequestState);

        /// External finish (abort/stop) for one or more requests. Frees resources
        /// and generates finished events on next step.
        fn finish_requests(&mut self, request_ids: &[RequestId], finished_status: RequestStatus);

        /// Returns number of running and waiting requests.
        fn get_request_counts(&self) -> (usize, usize);

        /// Whether there are finished requests to be returned in next outputs.
        fn has_finished_requests(&self) -> bool;

        fn reset_prefix_cache(&mut self) -> bool {
            false
        }

        fn shutdown(&mut self) {}

        fn get_kv_connector(&self) -> Option<Arc<dyn KvConnector>> {
            None
        }
    }

    /// Internal ready-to-execute queue item.
    #[derive(Clone, Debug)]
    #[allow(dead_code)]
    struct ReadyItem {
        request_id: RequestId,
        // Additional bookkeeping for staged data could be added here.
    }

    /// A basic FCFS/priority queue for waiting requests.
    #[derive(Default)]
    #[allow(dead_code)]
    struct RequestQueue {
        fcfs: VecDeque<RequestId>,
    }

    #[allow(dead_code)]
    impl RequestQueue {
        fn add_request(&mut self, req_id: RequestId) {
            self.fcfs.push_back(req_id);
        }
        fn peek_request(&self) -> Option<&RequestId> {
            self.fcfs.front()
        }
        fn pop_request(&mut self) -> Option<RequestId> {
            self.fcfs.pop_front()
        }
        fn remove_requests(&mut self, to_remove: &HashSet<RequestId>) {
            self.fcfs.retain(|r| !to_remove.contains(r));
        }
        fn is_empty(&self) -> bool {
            self.fcfs.is_empty()
        }
    }

    /// Our reference scheduler implementation.
    ///
    /// Performance notes:
    /// - Uses Arc<Mutex<RequestState>> to allow concurrent access to different requests
    /// - Avoids cloning Arc where possible (just borrow references)
    /// - Only clones strings when building output structs
    /// - Mutexes are per-request, so different requests can be accessed concurrently
    /// - This pattern is necessary because Rust cannot statically guarantee that
    ///   multiple mutable references to HashMap values are safe (values might have
    ///   internal references to each other or the map)
    pub struct RustScheduler {
        #[allow(dead_code)]
        consts: SchedulerConstants,

        #[allow(dead_code)]
        kv_cache: Arc<dyn KvCacheManager>,
        #[allow(dead_code)]
        structured: Option<Arc<dyn StructuredOutputManager>>,
        #[allow(dead_code)]
        connector: Option<Arc<dyn KvConnector>>,

        #[allow(dead_code)]
        policy: SchedulingPolicy,

        #[allow(dead_code)]
        requests: HashMap<RequestId, Arc<Mutex<RequestState>>>,
        #[allow(dead_code)]
        waiting: RequestQueue,
        #[allow(dead_code)]
        running: Vec<RequestId>,
        #[allow(dead_code)]
        ready_to_execute: VecDeque<ReadyItem>,

        #[allow(dead_code)]
        finished_req_ids: BTreeSet<RequestId>,
        #[allow(dead_code)]
        finished_req_ids_dict: Option<HashMap<ClientId, BTreeSet<RequestId>>>,
        /// Secondary budget for preparing requests (separate from main scheduling budget).
        #[allow(dead_code)]
        preparation_budget: PreparationBudget,
        /// Preparation state for each request.
        #[allow(dead_code)]
        preparation_states: HashMap<RequestId, RequestPreparationState>,
    }

    // impl RustScheduler {
    //     pub fn new(
    //         consts: SchedulerConstants,
    //         kv_cache: Arc<dyn KvCacheManager>,
    //         structured: Option<Arc<dyn StructuredOutputManager>>,
    //         connector: Option<Arc<dyn KvConnector>>,
    //         policy: SchedulingPolicy,
    //     ) -> Self {
    //         let finished_req_ids_dict = if consts.include_finished_set {
    //             Some(HashMap::new())
    //         } else {
    //             None
    //         };
    //         let total_gpu_blocks = consts.total_gpu_blocks;
    //         Self {
    //             consts,
    //             kv_cache,
    //             structured,
    //             connector,
    //             policy,
    //             requests: HashMap::new(),
    //             waiting: RequestQueue::default(),
    //             running: Vec::new(),
    //             ready_to_execute: VecDeque::new(),
    //             finished_req_ids: BTreeSet::new(),
    //             finished_req_ids_dict,
    //             preparation_budget: PreparationBudget::new(total_gpu_blocks / 4), // Use 25% of GPU blocks for preparation
    //             preparation_states: HashMap::new(),
    //         }
    //     }

    //     fn advance_num_computed_tokens(&mut self, output: &SchedulerOutput) {
    //         for (req_id, n) in &output.num_scheduled_tokens {
    //             if let Some(req) = self.requests.get(req_id) {
    //                 let mut req = req.lock().unwrap();
    //                 req.num_computed_tokens = req.num_computed_tokens.saturating_add(*n);
    //                 // TODO: encoder-freeing logic can be placed here (after num_computed_tokens updates)
    //             }
    //         }
    //         self.finished_req_ids.clear();
    //     }

    //     /// Free request resources and emit finished ids. Connector may delay GPU
    //     /// block freeing if remote copy-out is pending; in that case, the worker
    //     /// will later notify via `update_from_output` connector outputs.
    //     fn free_request(&mut self, request_id: &RequestId) -> Option<serde_json::Value> {
    //         // Free KV blocks and hashes; record finished ids.
    //         if let Some(req) = self.requests.remove(request_id) {
    //             let req_guard = req.lock().unwrap();
    //             let block_ids_tuple = self.kv_cache.get_block_ids(&req_guard.request_id);
    //             let (delay_free, kv_meta) = if let Some(conn) = &self.connector {
    //                 conn.request_finished(&req_guard, &block_ids_tuple.0)
    //             } else {
    //                 (false, None)
    //             };
    //             if !delay_free {
    //                 self.kv_cache.free(&req_guard);
    //                 self.kv_cache.free_block_hashes(&req_guard);
    //             }
    //             self.finished_req_ids.insert(request_id.clone());
    //             if let Some(map) = &mut self.finished_req_ids_dict {
    //                 map.entry(req_guard.client_id)
    //                     .or_default()
    //                     .insert(request_id.clone());
    //             }
    //             return kv_meta;
    //         }
    //         None
    //     }

    //     // -------------------------------
    //     // Worst-Case Projection Utilities
    //     // -------------------------------

    //     /// Projection parameters controlling modeling assumptions.
    //     /// - `tokens_per_pass_per_request`: use 1 to model decoding; higher for chunked prefill.
    //     /// - `consider_only_running`: if true, ignore waiting requests in projections.
    //     pub fn projection_params_default(&self) -> ProjectionParams {
    //         ProjectionParams {
    //             tokens_per_pass_per_request: 1,
    //             consider_only_running: true,
    //         }
    //     }

    //     /// Compute worst-case projections:
    //     /// - `wc_until_first_complete`: min passes to finish among active requests
    //     /// - `wc_until_block_starvation`: first pass when total blocks > capacity
    //     /// - `predicted_blocks_per_pass`: total blocks trajectory across horizon
    //     pub fn compute_worst_case_projection(&self, params: &ProjectionParams) -> WorstCaseProjection {
    //         let mut per_request: Vec<RequestProjection> = Vec::new();
    //         let block_size = self.consts.block_size;

    //         let iter_ids: Vec<_> = if params.consider_only_running {
    //             self.running.iter().collect()
    //         } else {
    //             self.requests.keys().collect()
    //         };

    //         for req_id in iter_ids {
    //             if let Some(req_arc) = self.requests.get(req_id) {
    //                 let req = req_arc.lock().unwrap();
    //                 let current_tokens = req.num_tokens();
    //                 let current_blocks = div_ceil(current_tokens, block_size);
    //                 let remaining_tokens = req.max_tokens.saturating_sub(req.output_token_ids.len());
    //                 let steps_until_completion = remaining_tokens; // 1 token per pass
    //                 per_request.push(RequestProjection {
    //                     request_id: req.request_id.clone(),
    //                     current_tokens,
    //                     current_blocks,
    //                     remaining_tokens,
    //                     steps_until_completion,
    //                 });
    //             }
    //         }

    //         let wc_until_first_complete = per_request
    //             .iter()
    //             .map(|r| r.steps_until_completion)
    //             .min()
    //             .unwrap_or(0);

    //         let mut predicted_blocks_per_pass: Vec<usize> = Vec::new();
    //         let mut wc_until_block_starvation: Option<usize> = None;
    //         let total_capacity = self.consts.total_gpu_blocks;

    //         let horizon = wc_until_first_complete.max(1);
    //         for k in 0..=horizon {
    //             let mut total_blocks = 0usize;
    //             for r in &per_request {
    //                 let incr = (k as usize)
    //                     .saturating_mul(params.tokens_per_pass_per_request)
    //                     .min(r.remaining_tokens);
    //                 let future_tokens = r.current_tokens + incr;
    //                 total_blocks = total_blocks.saturating_add(div_ceil(future_tokens, block_size));
    //             }
    //             predicted_blocks_per_pass.push(total_blocks);
    //             if wc_until_block_starvation.is_none() && total_blocks > total_capacity {
    //                 wc_until_block_starvation = Some(k);
    //             }
    //         }

    //         WorstCaseProjection {
    //             per_request,
    //             wc_until_first_complete,
    //             wc_until_block_starvation,
    //             predicted_blocks_per_pass,
    //         }
    //     }

    //     /// Plan which requests to offload to host to free a target number of blocks.
    //     /// Greedy heuristic: pick the requests with the fewest blocks first.
    //     pub fn plan_offload_for_deficit(&self, deficit_blocks: usize) -> OffloadPlan {
    //         let block_size = self.consts.block_size;
    //         let mut candidates: Vec<(RequestId, usize)> = Vec::new();

    //         for (req_id, req_arc) in &self.requests {
    //             let req = req_arc.lock().unwrap();
    //             let current_blocks = div_ceil(req.num_tokens(), block_size);
    //             if current_blocks > 0 {
    //                 candidates.push((req_id.clone(), current_blocks));
    //             }
    //         }

    //         candidates.sort_by_key(|(_, blocks)| *blocks);

    //         let mut selected: Vec<OffloadCandidate> = Vec::new();
    //         let mut freed = 0usize;
    //         for (req_id, blocks) in candidates {
    //             selected.push(OffloadCandidate {
    //                 request_id: req_id,
    //                 blocks,
    //             });
    //             freed = freed.saturating_add(blocks);
    //             if freed >= deficit_blocks {
    //                 break;
    //             }
    //         }

    //         OffloadPlan {
    //             selected,
    //             total_blocks_freed: freed,
    //             satisfied: freed >= deficit_blocks,
    //         }
    //     }

    //     // -------------------------------
    //     // Request Preparation & Staging
    //     // -------------------------------

    //     /// Start preparing a request by checking external blocks and allocating GPU blocks.
    //     /// This is called from the staging thread when a request is ready to be prepared.
    //     pub fn start_request_preparation(&mut self, request_id: &RequestId) -> bool {
    //         let Some(req_arc) = self.requests.get(request_id) else {
    //             return false;
    //         };

    //         // Get current request state
    //         let req_guard = req_arc.lock().unwrap();
    //         let prompt_tokens = req_guard.prompt_token_ids.len();
    //         let max_output_tokens = req_guard.max_tokens;
    //         let total_tokens = prompt_tokens + max_output_tokens;
    //         let total_blocks_needed = div_ceil(total_tokens, self.consts.block_size);

    //         // Check GPU cache for prefix matches
    //         let (block_ids, num_matched_gpu_blocks) = self.kv_cache.get_computed_blocks(&req_guard);
    //         let gpu_matched_tokens = num_matched_gpu_blocks * self.consts.block_size;

    //         // Check external sources for additional matches
    //         let external_matched_tokens = if let Some(conn) = &self.connector {
    //             conn.get_matched_connector_tokens(&req_guard, gpu_matched_tokens)
    //         } else {
    //             (0, false)
    //         };
    //         let external_matched_blocks = div_ceil(external_matched_tokens, self.consts.block_size);

    //         // Calculate remaining blocks needed
    //         let remaining_blocks =
    //             total_blocks_needed.saturating_sub(gpu_matched_blocks + external_matched_blocks);

    //         // Check if we can allocate the remaining blocks within preparation budget
    //         if !self.preparation_budget.can_allocate(remaining_blocks) {
    //             // Not enough budget - pause preparation
    //             self.preparation_states.insert(
    //                 request_id.clone(),
    //                 RequestPreparationState {
    //                     status: RequestPreparationStatus::Paused,
    //                     external_blocks: ExternalBlockInfo {
    //                         available_blocks: external_matched_blocks,
    //                         stage_to_host_first: false, // TODO: get from connector
    //                         blocks_on_host: 0,
    //                     },
    //                     allocated_gpu_blocks: vec![],
    //                     total_blocks_needed,
    //                     gpu_matched_blocks,
    //                     external_matched_blocks,
    //                     remaining_blocks_needed: remaining_blocks,
    //                 },
    //             );
    //             return false;
    //         }

    //         // Allocate GPU blocks for the remaining tokens
    //         let allocated_blocks = self.kv_cache.allocate_slots(
    //             &req_guard,
    //             remaining_blocks,
    //             0, // No lookahead tokens during preparation
    //         );

    //         let Some(allocated_blocks) = allocated_blocks else {
    //             // Failed to allocate - pause preparation
    //             self.preparation_states.insert(
    //                 request_id.clone(),
    //                 RequestPreparationState {
    //                     status: RequestPreparationStatus::Paused,
    //                     external_blocks: ExternalBlockInfo {
    //                         available_blocks: external_matched_blocks,
    //                         stage_to_host_first: false,
    //                         blocks_on_host: 0,
    //                     },
    //                     allocated_gpu_blocks: vec![],
    //                     total_blocks_needed,
    //                     gpu_matched_blocks,
    //                     external_matched_blocks,
    //                     remaining_blocks_needed: remaining_blocks,
    //                 },
    //             );
    //             return false;
    //         };

    //         // Update preparation budget - flatten the nested block structure
    //         let total_blocks: usize = allocated_blocks.iter().map(|group| group.len()).sum();
    //         self.preparation_budget.allocate(total_blocks);

    //         // Create preparation state
    //         let prep_state = RequestPreparationState {
    //             status: RequestPreparationStatus::Preparing,
    //             external_blocks: ExternalBlockInfo {
    //                 available_blocks: external_matched_blocks,
    //                 stage_to_host_first: false, // TODO: get from connector
    //                 blocks_on_host: 0,
    //             },
    //             allocated_gpu_blocks: allocated_blocks.into_iter().flatten().collect(),
    //             total_blocks_needed,
    //             gpu_matched_blocks,
    //             external_matched_blocks,
    //             remaining_blocks_needed: remaining_blocks,
    //         };

    //         self.preparation_states
    //             .insert(request_id.clone(), prep_state);
    //         true
    //     }

    //     /// Complete request preparation by transferring external blocks to device.
    //     /// This is called when external block transfer is complete.
    //     pub fn complete_request_preparation(&mut self, request_id: &RequestId) -> bool {
    //         let Some(prep_state) = self.preparation_states.get_mut(request_id) else {
    //             return false;
    //         };

    //         if prep_state.status != RequestPreparationStatus::Preparing {
    //             return false;
    //         }

    //         // Update request state with computed tokens
    //         if let Some(req_arc) = self.requests.get(request_id) {
    //             let mut req_guard = req_arc.lock().unwrap();
    //             req_guard.num_computed_tokens = req_guard.num_computed_tokens.saturating_add(
    //                 (prep_state.gpu_matched_blocks + prep_state.external_matched_blocks)
    //                     * self.consts.block_size,
    //             );
    //         }

    //         // Move request to ready queue
    //         prep_state.status = RequestPreparationStatus::Ready;
    //         self.ready_to_execute.push_back(ReadyItem {
    //             request_id: request_id.clone(),
    //         });

    //         // Free preparation budget
    //         self.preparation_budget
    //             .free(prep_state.allocated_gpu_blocks.len());

    //         true
    //     }

    //     /// Handle connector staging to host first (when connector wants to stage everything to host).
    //     pub fn handle_host_staging(&mut self, request_id: &RequestId, blocks_on_host: usize) -> bool {
    //         let Some(prep_state) = self.preparation_states.get_mut(request_id) else {
    //             return false;
    //         };

    //         // Update external block info to reflect host staging
    //         prep_state.external_blocks.stage_to_host_first = true;
    //         prep_state.external_blocks.blocks_on_host = blocks_on_host;

    //         // If we have enough blocks on host, we can complete preparation
    //         if prep_state.external_blocks.blocks_on_host >= prep_state.remaining_blocks_needed {
    //             self.complete_request_preparation(request_id)
    //         } else {
    //             // Still waiting for more blocks to be staged to host
    //             true
    //         }
    //     }

    //     /// Resume preparation of a paused request when budget becomes available.
    //     pub fn resume_preparation(&mut self, request_id: &RequestId) -> bool {
    //         let Some(prep_state) = self.preparation_states.get(request_id) else {
    //             return false;
    //         };

    //         // If we have enough budget, try to start preparation again
    //         if self
    //             .preparation_budget
    //             .can_allocate(prep_state.remaining_blocks_needed)
    //         {
    //             self.start_request_preparation(request_id)
    //         } else {
    //             false
    //         }
    //     }

    //     /// Get current preparation status for a request.
    //     pub fn get_preparation_status(
    //         &self,
    //         request_id: &RequestId,
    //     ) -> Option<&RequestPreparationStatus> {
    //         self.preparation_states.get(request_id).map(|s| &s.status)
    //     }

    //     /// Get preparation budget status.
    //     pub fn get_preparation_budget(&self) -> &PreparationBudget {
    //         &self.preparation_budget
    //     }
    // }

    // impl Scheduler for RustScheduler {
    //     /// See `Scheduler::schedule` for semantics. This implementation deviates
    //     /// from vLLM by focusing only on currently running and ready-to-execute
    //     /// requests; expensive matching is expected in a staging thread.
    //     fn schedule(&mut self) -> SchedulerOutput {
    //         let mut output = SchedulerOutput::default();
    //         let mut token_budget = self.consts.token_budget();

    //         // 1) Schedule running requests up to budget.
    //         let mut req_index = 0usize;
    //         while req_index < self.running.len() && token_budget > 0 {
    //             let req_id = &self.running[req_index];
    //             let req_arc = match self.requests.get(req_id) {
    //                 Some(a) => a, // Just borrow, don't clone the Arc
    //                 None => {
    //                     req_index += 1;
    //                     continue;
    //                 }
    //             };
    //             let req_guard = req_arc.lock().unwrap();

    //             // Compute how many tokens to schedule this step.
    //             let mut num_new_tokens = req_guard
    //                 .num_tokens_with_spec()
    //                 .saturating_add(req_guard.num_output_placeholders)
    //                 .saturating_sub(req_guard.num_computed_tokens);

    //             if let Some(thresh) = self.consts.long_prefill_token_threshold {
    //                 if thresh > 0 && num_new_tokens > thresh {
    //                     num_new_tokens = thresh;
    //                 }
    //             }
    //             num_new_tokens = num_new_tokens.min(token_budget);
    //             // Keep within model length constraints.
    //             num_new_tokens = num_new_tokens.min(
    //                 self.consts
    //                     .max_model_len
    //                     .saturating_sub(1 + req_guard.num_computed_tokens),
    //             );

    //             if num_new_tokens == 0 {
    //                 req_index += 1;
    //                 continue;
    //             }

    //             // Try to allocate blocks (with potential lookahead for spec decoding).
    //             let new_blocks = self.kv_cache.allocate_slots(
    //                 &req_guard,
    //                 num_new_tokens,
    //                 if self.consts.use_spec_decode {
    //                     self.consts.num_spec_tokens
    //                 } else {
    //                     0
    //                 },
    //             );
    //             if new_blocks.is_none() {
    //                 // NOTE: Preemption strategy can go here. For now, skip.
    //                 req_index += 1;
    //                 continue;
    //             }
    //             let new_blocks = new_blocks.unwrap();

    //             // Record scheduling.
    //             output.cached_requests.push(CachedRequestData {
    //                 request_id: req_guard.request_id.clone(),
    //                 resumed_from_preemption: false,
    //                 new_token_ids: if self.consts.pipeline_parallel {
    //                     // when PP>1 we may need to ship token ids
    //                     let start = req_guard.num_computed_tokens;
    //                     let end = start + num_new_tokens;
    //                     req_guard.all_token_ids[start..end].to_vec()
    //                 } else {
    //                     vec![]
    //                 },
    //                 new_block_ids: new_blocks.clone(),
    //                 num_computed_tokens: req_guard.num_computed_tokens,
    //             });
    //             output
    //                 .num_scheduled_tokens
    //                 .insert(req_guard.request_id.clone(), num_new_tokens);
    //             token_budget = token_budget.saturating_sub(num_new_tokens);

    //             // If using a connector, inform it about allocation results.
    //             if let Some(conn) = &self.connector {
    //                 conn.update_state_after_alloc(&req_guard, &new_blocks, /*num_external*/ 0);
    //             }

    //             req_index += 1;
    //         }

    //         // 2) Pick from ready_to_execute (staged) and schedule new ones.
    //         while token_budget > 0 {
    //             let Some(ready) = self.ready_to_execute.pop_front() else {
    //                 break;
    //             };
    //             let Some(req_arc) = self.requests.get(&ready.request_id) else {
    //                 continue;
    //             };
    //             let req_guard = req_arc.lock().unwrap();

    //             // For staged requests, `num_computed_tokens` may already reflect local/external matches.
    //             let mut num_new_tokens = req_guard
    //                 .num_tokens()
    //                 .saturating_sub(req_guard.num_computed_tokens);
    //             if let Some(thresh) = self.consts.long_prefill_token_threshold {
    //                 if thresh > 0 && num_new_tokens > thresh {
    //                     num_new_tokens = thresh;
    //                 }
    //             }

    //             if !self.consts.chunked_prefill_enabled && num_new_tokens > token_budget {
    //                 continue;
    //             }

    //             num_new_tokens = num_new_tokens.min(token_budget);
    //             if num_new_tokens == 0 {
    //                 continue;
    //             }

    //             let new_blocks = self.kv_cache.allocate_slots(
    //                 &req_guard,
    //                 num_new_tokens,
    //                 if self.consts.use_spec_decode {
    //                     self.consts.num_spec_tokens
    //                 } else {
    //                     0
    //                 },
    //             );
    //             let Some(new_blocks) = new_blocks else {
    //                 continue;
    //             };

    //             output.new_requests.push(NewRequestData {
    //                 request_id: req_guard.request_id.clone(),
    //                 prompt_token_ids: req_guard.prompt_token_ids.clone(),
    //                 block_ids: new_blocks.clone(),
    //                 num_computed_tokens: req_guard.num_computed_tokens,
    //                 lora: req_guard.lora.clone(),
    //             });
    //             output
    //                 .num_scheduled_tokens
    //                 .insert(req_guard.request_id.clone(), num_new_tokens);
    //             token_budget = token_budget.saturating_sub(num_new_tokens);

    //             // Move to running.
    //             self.running.push(req_guard.request_id.clone());
    //         }

    //         // 3) Compute common prefix stats.
    //         if let Some(first) = self.running.first().and_then(|id| self.requests.get(id)) {
    //             let req = first.lock().unwrap();
    //             output.num_common_prefix_blocks = self
    //                 .kv_cache
    //                 .get_num_common_prefix_blocks(&req, self.running.len());
    //         }

    //         // 4) Connector metadata, if any.
    //         if let Some(conn) = &self.connector {
    //             let _md = conn.build_connector_metadata(&output);
    //             // TODO: attach to output when needed by bindings
    //         }

    //         // 5) Advance in-request counters and clear finished list for next tick.
    //         output.total_num_scheduled_tokens = output.num_scheduled_tokens.values().copied().sum();
    //         self.advance_num_computed_tokens(&output);

    //         output
    //     }

    //     /// See `Scheduler::update_from_output` for semantics. Adjusts for spec
    //     /// rejections, emits outputs grouped by client, and advances structured
    //     /// decoding if applicable.
    //     fn update_from_output(
    //         &mut self,
    //         scheduler_output: &SchedulerOutput,
    //         model_runner_output: &OwnedModelRunnerOutput,
    //     ) -> HashMap<ClientId, EngineCoreOutputs> {
    //         let mut outputs_by_client: HashMap<ClientId, EngineCoreOutputs> = HashMap::new();

    //         // Fast return if nothing was scheduled.
    //         if scheduler_output.num_scheduled_tokens.is_empty() || model_runner_output.is_empty() {
    //             return outputs_by_client;
    //         }

    //         // PERF: Parallelize per-request processing of outputs. We avoid holding the
    //         // GIL here because `OwnedModelRunnerOutput` is a Rust-owned view.
    //         // For simplicity, use a scoped iterator without rayon to keep the sketch self-contained.
    //         //
    //         // TODO: When performance is critical, consider using rayon::scope for parallel processing:
    //         // rayon::scope(|s| {
    //         //     for (req_id, num_tokens_scheduled) in &scheduler_output.num_scheduled_tokens {
    //         //         let req_arc = self.requests.get(req_id).cloned();
    //         //         s.spawn(move |_| { /* process request */ });
    //         //     }
    //         // });
    //         for (req_id, num_tokens_scheduled) in &scheduler_output.num_scheduled_tokens {
    //             if *num_tokens_scheduled == 0 {
    //                 continue;
    //             }

    //             let Some(req_arc) = self.requests.get(req_id) else {
    //                 continue;
    //             };
    //             let mut req = req_arc.lock().unwrap();
    //             let Some(&req_index) = model_runner_output.req_id_to_index.get(req_id) else {
    //                 continue;
    //             };

    //             let mut generated_token_ids = model_runner_output
    //                 .sampled_token_ids
    //                 .get(req_index)
    //                 .cloned()
    //                 .unwrap_or_default();

    //             // Speculative decoding adjustment – reduce computed tokens by rejected drafts.
    //             if let Some(spec_lists) = &model_runner_output.spec_token_ids {
    //                 if let Some(scheduled_spec) =
    //                     scheduler_output.scheduled_spec_decode_tokens.get(req_id)
    //                 {
    //                     let accepted = generated_token_ids.len().saturating_sub(1);
    //                     let rejected = scheduled_spec.len() + 1 - generated_token_ids.len();
    //                     if rejected > 0 {
    //                         req.num_computed_tokens = req.num_computed_tokens.saturating_sub(rejected);
    //                     }
    //                     // Replace drafts with next-step drafts if needed.
    //                     req.spec_token_ids = spec_lists.get(req_index).cloned().unwrap_or_default();
    //                     let _ = accepted; // reserved for stats
    //                 }
    //             }

    //             // Append newly generated tokens and check stop conditions.
    //             let mut stopped = false;
    //             if !generated_token_ids.is_empty() {
    //                 for (idx, tok) in generated_token_ids.iter().copied().enumerate() {
    //                     req.output_token_ids.push(tok);
    //                     req.all_token_ids.push(tok);
    //                     // TODO: implement stop check using max_model_len / eos / sampling params
    //                     let _num_new = idx + 1;
    //                 }
    //             }

    //             // Pooler output / prompt logprobs / events / kv_transfer_params can be attached here.
    //             let kv_transfer_params = if stopped {
    //                 self.free_request(req_id)
    //             } else {
    //                 None
    //             };

    //             let eco = EngineCoreOutput {
    //                 request_id: req_id.clone(),
    //                 new_token_ids: generated_token_ids,
    //                 finish_reason: if stopped { Some(0) } else { None },
    //                 stop_reason: req.stop_reason(),
    //                 kv_transfer_params,
    //                 num_cached_tokens: req.num_cached_tokens,
    //                 ..Default::default()
    //             };

    //             outputs_by_client
    //                 .entry(req.client_id)
    //                 .or_default()
    //                 .outputs
    //                 .push(eco);
    //         }

    //         // Update connector status for finished KV transfers.
    //         if let Some(conn) = &self.connector {
    //             let finished_recving = model_runner_output
    //                 .kv_connector_finished_recving
    //                 .clone()
    //                 .unwrap_or_default();
    //             let finished_sending = model_runner_output
    //                 .kv_connector_finished_sending
    //                 .clone()
    //                 .unwrap_or_default();
    //             conn.update_connector_output(&finished_recving, &finished_sending);
    //             for req_id in finished_sending {
    //                 // Safe to free blocks for requests done sending.
    //                 let _ = self.free_request(&req_id);
    //             }
    //         }

    //         // Attach finished request sets to one of the client groups.
    //         if let Some(map) = &mut self.finished_req_ids_dict {
    //             if let Some((_client, group)) = map.iter_mut().next() {
    //                 if let Some((_k, v)) = outputs_by_client.iter_mut().next() {
    //                     v.finished_requests = Some(group.clone());
    //                 }
    //                 group.clear();
    //             }
    //         }

    //         outputs_by_client
    //     }

    //     fn add_request(&mut self, request: RequestState) {
    //         let req_id = request.request_id.clone();
    //         self.waiting.add_request(req_id.clone());
    //         self.requests.insert(req_id, Arc::new(Mutex::new(request)));
    //     }

    //     fn finish_requests(&mut self, request_ids: &[RequestId], finished_status: RequestStatus) {
    //         let ids: HashSet<_> = request_ids.iter().cloned().collect();
    //         self.waiting.remove_requests(&ids);
    //         self.running.retain(|r| !ids.contains(r));

    //         for req_id in request_ids {
    //             if let Some(req) = self.requests.get(req_id).cloned() {
    //                 let mut req = req.lock().unwrap();
    //                 req.status = finished_status;
    //             }
    //             let _ = self.free_request(req_id);
    //         }
    //     }

    //     fn get_request_counts(&self) -> (usize, usize) {
    //         (self.running.len(), self.waiting.fcfs.len())
    //     }

    //     fn has_finished_requests(&self) -> bool {
    //         !self.finished_req_ids.is_empty()
    //     }
    // }

    // impl RequestState {
    //     fn stop_reason(&self) -> Option<i32> {
    //         // TODO: encode stop reason similar to vLLM (stop string, length, abort)
    //         None
    //     }
    // }

    // ==========================
    // Staging Thread Sketch (TODO)
    // ==========================
    //
    // Design:
    // - A dedicated staging thread receives new requests and performs:
    //   * local prefix-cache matches via KvCacheManager.get_computed_blocks
    //   * external matches via KvConnector.get_matched_connector_tokens
    //   * updates request.num_computed_tokens accordingly
    //   * enqueues ReadyItem into ready_to_execute when prepared
    // - Communication: std::sync::mpsc or tokio::mpsc; configurable batch size
    // - Safety: staging must never mutate fields used by schedule() without holding
    //   the same request mutex; use a small immutable message to the scheduler
    //   instead (request_id + prepared state deltas).
    // - This sketch leaves concrete code to integration.

    // ---------------------------------
    // Projection & Offload Data Models
    // ---------------------------------

    /// Integer ceil division helper.
    #[allow(dead_code)]
    fn div_ceil(n: usize, d: usize) -> usize {
        n.div_ceil(d)
    }

    /// Parameters controlling projection behavior.
    #[derive(Clone, Debug)]
    pub struct ProjectionParams {
        /// Tokens each request advances per pass (1 for decode). Larger to approximate chunked prefill.
        pub tokens_per_pass_per_request: usize,
        /// Whether to restrict projections to currently running requests only.
        pub consider_only_running: bool,
    }

    /// Per-request projection snapshot used to build worst-case aggregates.
    #[derive(Clone, Debug)]
    pub struct RequestProjection {
        pub request_id: RequestId,
        pub current_tokens: usize,
        pub current_blocks: usize,
        pub remaining_tokens: usize,
        pub steps_until_completion: usize,
    }

    /// Aggregate projection results.
    #[derive(Clone, Debug)]
    pub struct WorstCaseProjection {
        /// Per-request current counts and remaining work.
        pub per_request: Vec<RequestProjection>,
        /// In worst-case, upper bound on number of passes until the first request completes.
        pub wc_until_first_complete: usize,
        /// First pass index at which blocks would exceed capacity, if ever.
        pub wc_until_block_starvation: Option<usize>,
        /// Total blocks predicted per pass over the examined horizon.
        pub predicted_blocks_per_pass: Vec<usize>,
    }

    /// Offload plan proposal to avoid or resolve a predicted starvation.
    #[derive(Clone, Debug)]
    pub struct OffloadPlan {
        pub selected: Vec<OffloadCandidate>,
        pub total_blocks_freed: usize,
        pub satisfied: bool,
    }

    /// Request chosen for offloading and the approximate blocks freed if fully moved to host.
    #[derive(Clone, Debug)]
    pub struct OffloadCandidate {
        pub request_id: RequestId,
        pub blocks: usize,
    }

    // ---------------------------------
    // Request Preparation & Staging
    // ---------------------------------

    /// Status of a request during the preparation/staging phase.
    #[derive(Clone, Debug, PartialEq)]
    pub enum RequestPreparationStatus {
        /// Request is waiting to be prepared (not yet staged).
        Waiting,
        /// Request is being prepared - external blocks are being transferred to device.
        Preparing,
        /// Request is prepared and ready to execute (moved to ready_to_execute queue).
        Ready,
        /// Request preparation was paused due to insufficient GPU blocks.
        Paused,
    }

    /// Information about external blocks that need to be transferred to device.
    #[derive(Clone, Debug)]
    pub struct ExternalBlockInfo {
        /// Number of blocks available from external sources (host, remote).
        pub available_blocks: usize,
        /// Whether the connector wants to stage everything to host first.
        pub stage_to_host_first: bool,
        /// If staging to host first, how many blocks are currently on host.
        pub blocks_on_host: usize,
    }

    /// Request preparation state tracking.
    #[derive(Clone, Debug)]
    pub struct RequestPreparationState {
        /// Current preparation status.
        pub status: RequestPreparationStatus,
        /// External block information from connector.
        pub external_blocks: ExternalBlockInfo,
        /// GPU blocks allocated for this request during preparation.
        pub allocated_gpu_blocks: Vec<BlockId>,
        /// Total blocks needed for the full request (prompt + max output).
        pub total_blocks_needed: usize,
        /// Blocks already matched from GPU cache.
        pub gpu_matched_blocks: usize,
        /// Blocks already matched from external sources.
        pub external_matched_blocks: usize,
        /// Remaining blocks that need to be allocated.
        pub remaining_blocks_needed: usize,
    }

    /// Secondary budget for preparing requests (separate from main scheduling budget).
    #[derive(Clone, Debug)]
    pub struct PreparationBudget {
        /// Maximum GPU blocks that can be allocated for request preparation.
        pub max_gpu_blocks: usize,
        /// Currently allocated blocks for preparation.
        pub allocated_blocks: usize,
        /// Whether we're currently in a preparation phase.
        pub is_preparing: bool,
    }

    impl PreparationBudget {
        pub fn new(max_gpu_blocks: usize) -> Self {
            Self {
                max_gpu_blocks,
                allocated_blocks: 0,
                is_preparing: false,
            }
        }

        pub fn can_allocate(&self, blocks: usize) -> bool {
            self.allocated_blocks + blocks <= self.max_gpu_blocks
        }

        pub fn allocate(&mut self, blocks: usize) -> bool {
            if self.can_allocate(blocks) {
                self.allocated_blocks += blocks;
                true
            } else {
                false
            }
        }

        pub fn free(&mut self, blocks: usize) {
            self.allocated_blocks = self.allocated_blocks.saturating_sub(blocks);
        }
    }

    pub struct RequestPreloadState {}

    #[allow(dead_code)]
    impl RequestPreloadState {
        async fn prepare_request(&mut self, _request: RequestState) {
            // match against gpu blocks

            // match against host/disk blocks

            // determine if prefill should be computed locally or offloaded
        }

        async fn onboard_locally_stored_blocks(&mut self, _request: RequestState) {
            unimplemented!()
        }

        async fn onboard_remotely_stored_blocks(&mut self, _request: RequestState) {
            unimplemented!()
        }

        async fn onboard_remotely_computed_blocks(&mut self, _request: RequestState) {
            // if remote prefill, then ensure ensure gpu blocks are available
            // in host memory, then release the gpu blocks

            // acquire cpu blocks for the remote instance to write

            // prepare src and dst descriptors
            // the remote prefill worker will pull kv blocks from "src" descriptors
            // the remote prefill worker will push kv blocks to the "dst" descriptors

            // issue remote prefill request and await its completion
        }
    }

    pub struct SchedulerState {}

    // ========================================================================
    // Rust Scheduler State for tracking requests in parallel with vLLM
    // ========================================================================

    /// Compute a deterministic u64 hash from a string cache_salt.
    /// This ensures consistent hashing between Python and Rust.
    pub fn compute_salt_hash(cache_salt: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        cache_salt.hash(&mut hasher);
        hasher.finish()
    }

    /// Initial request data available at add_request time.
    /// This is a simplified version that only contains data we have
    /// before any scheduling decisions are made.
    #[derive(Clone, Debug)]
    pub struct InitialRequestData {
        pub request_id: String,
        pub prompt_token_ids: Vec<i32>,
        pub salt_hash: Option<u64>,
        pub lora_int_id: Option<i64>,
        pub priority: i32,
        pub arrival_time: f64,
    }

    /// The Rust scheduler state that tracks requests in parallel with vLLM.
    /// This allows us to build up Rust-side state while vLLM continues to
    /// drive the actual scheduling decisions.
    #[derive(Debug)]
    pub struct RustSchedulerState {
        /// Map of request_id to initial request data
        requests: Arc<Mutex<HashMap<String, InitialRequestData>>>,
    }

    impl RustSchedulerState {
        /// Create a new empty scheduler state.
        pub fn new() -> Self {
            Self {
                requests: Arc::new(Mutex::new(HashMap::new())),
            }
        }

        /// Add a new request to the scheduler state.
        /// Called when DynamoScheduler.add_request() is invoked.
        pub fn add_request(
            &self,
            request_id: String,
            prompt_token_ids: Vec<i32>,
            cache_salt: Option<String>,
            lora_int_id: Option<i64>,
            priority: i32,
            arrival_time: f64,
        ) -> Result<(), String> {
            // Convert cache_salt string to u64 hash
            let salt_hash = cache_salt.as_ref().map(|s| compute_salt_hash(s));

            let request_data = InitialRequestData {
                request_id: request_id.clone(),
                prompt_token_ids,
                salt_hash,
                lora_int_id,
                priority,
                arrival_time,
            };

            let mut requests = self.requests.lock().unwrap();
            requests.insert(request_id.clone(), request_data);

            // Log for debugging
            println!(
                "Rust scheduler: Added request {} with {} prompt tokens",
                request_id,
                requests.get(&request_id).unwrap().prompt_token_ids.len()
            );

            Ok(())
        }

        /// Mark requests as finished without removing them.
        /// The actual removal happens when the scheduler reports them as finished.
        /// This is called when finish_requests is invoked externally (e.g., client disconnect).
        pub fn mark_as_finished(&self, request_ids: Vec<String>) -> Result<(), String> {
            // TODO: When we track request states, update the state to finished here.
            // For now, this is a no-op as we don't track states yet.
            // The request will be removed when update_from_output reports it as finished.
            for req_id in &request_ids {
                println!(
                    "Rust scheduler: Marked request {} as finished (no-op for now)",
                    req_id
                );
            }
            Ok(())
        }

        /// Remove finished requests from the scheduler state.
        /// This should only be called when the scheduler reports requests as finished
        /// via scheduler_output.finished_req_ids in update_from_output.
        pub fn remove_finished_requests(&self, request_ids: Vec<String>) -> Result<(), String> {
            let mut requests = self.requests.lock().unwrap();

            for req_id in &request_ids {
                if requests.remove(req_id).is_some() {
                    println!("Rust scheduler: Removed finished request {}", req_id);
                }
            }

            Ok(())
        }

        /// Get the current number of tracked requests.
        pub fn num_requests(&self) -> usize {
            self.requests.lock().unwrap().len()
        }

        /// Check if a request is being tracked.
        pub fn has_request(&self, request_id: &str) -> bool {
            self.requests.lock().unwrap().contains_key(request_id)
        }

        /// Get all currently tracked request IDs.
        pub fn get_request_ids(&self) -> Vec<String> {
            self.requests.lock().unwrap().keys().cloned().collect()
        }
    }

    impl Default for RustSchedulerState {
        fn default() -> Self {
            Self::new()
        }
    }

    #[allow(dead_code)]
    pub struct LogicalBlock {
        block_id: usize,
        permit: OwnedSemaphorePermit,
    }

    #[allow(dead_code)]
    enum BlockState {
        Mutable,
        Registering,
        Immutable,
    }

    #[allow(dead_code)]
    pub struct Block<const ID: u128> {
        block_id: u64,
    }

    #[allow(dead_code)]
    impl<const ID: u128> Block<ID> {
        fn block_id(&self) -> u64 {
            self.block_id
        }

        fn block_set_id() -> u8 {
            todo!("split the u128 into two u64s, return the first u8 of the second u64")
        }

        fn instance_id() -> u64 {
            todo!("return the first u64 bits of the u128")
        }
    }
}
