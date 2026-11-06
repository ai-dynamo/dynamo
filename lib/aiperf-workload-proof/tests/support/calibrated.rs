// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Preparatory calibrated GPU control for the native AIPerf-hosted Replayer.
//!
//! This proves *provider wiring*, not a full-scenario equivalence claim. Per
//! role (prefill and decode), the canonical H100 BF16 op-level silicon timing
//! model and the native KV-cache geometry from `new_with_geometry` are the one
//! timing source for both the native Replayer engines and the real Dynamo KV
//! Router launch config. The router's `MockerConfig` carries the same resolved
//! FPM config as its role and is given a canonical AIS callback, never a
//! `PerfModel::Fixed` placeholder.
//!
//! Scope limit: the shared `run_fixture` branch still uses AIPerf's builtin
//! tokenizer. This control therefore says nothing about tokenizer equivalence
//! for the configured model; the pinned model tokenizer and model metadata are
//! acquired by main before any full-scenario claim. Model identity is explicit
//! via `AISIM_MODEL` (default `Qwen/Qwen3-8B`) and construction fails, rather
//! than falling back, when the checked-in op-level silicon tables cannot serve
//! that identity.

#![cfg(feature = "calibrated-proof")]

use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
    rc::Rc,
    sync::Arc,
};

use aiperf_clock_runtime::cancellation::RunCancellation;
use aisimulate_core::engine::{
    CanonicalTimingModel, EngineConfig, EngineLaunchConfig, TimingModel, TimingModelConfig,
    TransferTimingMode,
};
use aisimulate_core::replay::{
    ReplayCaptureOptions, ReplayDeterminism, ReplayEngineConfig, ReplayEngineFactory, ReplayReport,
    ReplayRoleConfig, ReplayRuntimeInput, ReplaySpec, Replayer,
};
use aisimulate_core::{
    BackendKind, DataType, DatabaseMode,
    perfmodel::{
        EstimationMode, ForwardPassFallbackPolicy, ForwardPassPerfModelConfig,
        ForwardPassWorkerType,
        memory::{
            EstimateSource, KvCacheEstimateOptions, KvCacheEstimateRequest, KvCacheMemoryFraction,
        },
    },
};
use anyhow::{Context, Result, ensure};
use dynamo_mocker::common::perf_model::{AisCallback, PerfModel};
use dynamo_mocker::placement::{KvReplayComposition, MockerConfig};
use serde_json::{Value, json};

// The attaching test target owns the fixture module so it is loaded once.
use crate::run_fixture;

pub(crate) const DEFAULT_MODEL: &str = "Qwen/Qwen3-8B";
pub(crate) const SYSTEM: &str = "h100_sxm";
pub(crate) const LONG_CONTEXT_MODEL: &str = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4";

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) enum HardwareSelection {
    H100Sxm,
    B200Sxm,
}

impl HardwareSelection {
    pub(crate) fn system(self) -> &'static str {
        match self {
            Self::H100Sxm => SYSTEM,
            Self::B200Sxm => "b200_sxm",
        }
    }

    fn pins(self) -> (&'static str, &'static str, usize, u64) {
        match self {
            Self::H100Sxm => (
                HARDWARE_YAML_SHA256,
                CALIBRATION_MANIFEST_SHA256,
                CALIBRATION_FILE_COUNT,
                CALIBRATION_TOTAL_BYTES,
            ),
            // Frozen from the canonical AISim checkout's git-tracked assets;
            // identical sorted relative-path manifest algorithm for both GPUs.
            Self::B200Sxm => (
                "2cf8081b7a78da88cc332d4fdaf4259ba7b47f48397f9c100bbf7206e5245cfd",
                "73134c5fe1874b0deefe8242f4bbf4c6046fdeb661f269c594e4ebcc090c5add",
                321,
                37_311_596,
            ),
        }
    }
}

/// Explicit arithmetic choice, independent of a checkpoint ID's suffix. This
/// proof does not claim that BF16 timing measures NVFP4 checkpoint execution.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) enum ProofPrecision {
    Bfloat16,
    Fp8WeightsBfloat16AttentionKv,
}

impl ProofPrecision {
    fn weight_dtype(self) -> &'static str {
        self.quant_mode()
    }

    fn quant_mode(self) -> &'static str {
        match self {
            Self::Bfloat16 => "bfloat16",
            Self::Fp8WeightsBfloat16AttentionKv => "fp8",
        }
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) enum PrefillMode {
    Unchunked,
    Chunked { max_num_batched_tokens: usize },
}

/// Complete serving identity shared by canonical timing, physical geometry,
/// replay and router. PP1/attention-DP1 remain fixed proof assumptions.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RoleIdentity<'a> {
    pub(crate) model: &'a str,
    pub(crate) systems: &'a Path,
    pub(crate) hardware: HardwareSelection,
    pub(crate) precision: ProofPrecision,
    pub(crate) role: ForwardPassWorkerType,
    pub(crate) tensor_parallel_size: usize,
    pub(crate) max_num_seqs: usize,
    pub(crate) prefill: PrefillMode,
}

pub(crate) const BACKEND_VERSION: &str = "0.24.0";
pub(crate) const IDENTITY_SEED: [u8; 32] = [7; 32];
pub(crate) const PREFILL_WORKERS: usize = 2;
pub(crate) const DECODE_WORKERS: usize = 2;
/// Matches the fixture's authored hash/block size and the existing native
/// hosted proof. A different block size would make request KV identities and
/// router block accounting disagree.
pub(crate) const BLOCK_SIZE: usize = 8;
pub(crate) const MAX_NUM_SEQS: usize = 8;
pub(crate) const MAX_NUM_BATCHED_TOKENS: usize = 4096;
pub(crate) const FIXED_PREFILL_MS: f64 = 1.0;
pub(crate) const FIXED_DECODE_MS: f64 = 2.0;

pub(crate) fn configured_model() -> (String, &'static str) {
    match std::env::var("AISIM_MODEL") {
        Ok(model) => (model, "AISIM_MODEL"),
        Err(std::env::VarError::NotPresent) => (DEFAULT_MODEL.to_owned(), "default"),
        Err(error) => panic!("AISIM_MODEL is not readable: {error}"),
    }
}

pub(crate) fn systems_root() -> Result<PathBuf> {
    let root = std::env::var_os("AISIM_SYSTEMS_ROOT").context(
        "AISIM_SYSTEMS_ROOT must identify the pinned checked-in systems assets; no skip or synthetic fallback",
    )?;
    let root = PathBuf::from(root);
    ensure!(
        root.is_dir(),
        "canonical systems root is required at {}",
        root.display()
    );
    Ok(root)
}

/// Explicit topology assumption for this 2P2D control.
///
/// A `ReplaySpec` names abstract prefill/decode worker counts, not host
/// placement. Four 1-GPU workers would fit inside one 8-GPU H100 node, where
/// NVLink's `intra_node_bw` would apply, but the spec cannot prove that
/// placement. This control therefore chooses the cross-node case explicitly:
/// the hardware spec's **inter-node** per-GPU, single-direction rate. That is
/// the conservative choice and a hardware-model link assumption, not a
/// measured network claim for a deployment.
const LINK_TOPOLOGY: &str = "inter_node";

/// Minimal view of the AISim-owned `systems/<system>.yaml` hardware contract.
///
/// `aisim-core` parses the full schema into `SystemSpec`, but that type is not
/// reachable through a public path (`perfmodel::common` is crate-private), so
/// this control reads the same canonical file directly and deserializes only
/// the node fields it consumes. `serde` ignores the rest of the document.
#[derive(Debug, serde::Deserialize)]
struct HardwareSystemYaml {
    node: HardwareNodeYaml,
}

#[derive(Debug, serde::Deserialize)]
struct HardwareNodeYaml {
    num_gpus_per_node: u32,
    /// Inter-node bandwidth, bytes/s per GPU, single direction.
    inter_node_bw: f64,
    /// Point-to-point latency, seconds.
    #[serde(default)]
    p2p_latency: f64,
}

use sha2::{Digest, Sha256};
use std::io::Read;

/// SHA-256 of the exact `systems/h100_sxm.yaml` git blob at the pinned
/// AISimulate revision, frozen from `git cat-file blob HEAD:.../h100_sxm.yaml`.
const HARDWARE_YAML_SHA256: &str =
    "bbbe72e97e0271364ec047d94bade25d03f0ef9c43981b0a81cd86ae4fe1f039";
/// SHA-256 over the complete `data/h100_sxm` subtree, not only the op tables
/// this control consumes. See `calibration_inventory` for the line format.
const CALIBRATION_MANIFEST_SHA256: &str =
    "3282e67f31ba7b914fae069e819299988ef8e06143677418ff3065756e9448c9";
const CALIBRATION_FILE_COUNT: usize = 328;
const CALIBRATION_TOTAL_BYTES: u64 = 28_392_736;
const MAX_HARDWARE_YAML_BYTES: u64 = 1 << 20;
const MAX_CALIBRATION_FILE_BYTES: u64 = 1 << 30;
const MAX_MODEL_CONFIG_BYTES: u64 = 1 << 20;
const HASH_CHUNK_BYTES: usize = 64 * 1024;

/// Hardware link provenance plus the exact content identity and rates this
/// control uses. The hashes cover the whole canonical calibration root, so a
/// path alone can never stand in for the bytes that produced the timings.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub(crate) struct HardwareLink {
    pub(crate) topology: &'static str,
    pub(crate) system: &'static str,
    pub(crate) yaml: PathBuf,
    pub(crate) yaml_sha256: String,
    pub(crate) calibration_root: PathBuf,
    pub(crate) calibration_manifest_sha256: String,
    pub(crate) calibration_file_count: usize,
    pub(crate) calibration_total_bytes: u64,
    /// Git HEAD at the assets root, when a checkout is available. Integrity is
    /// enforced by the content hashes; this is a supplementary source identity.
    pub(crate) source_revision: Option<String>,
    pub(crate) num_gpus_per_node: u32,
    pub(crate) bandwidth_bytes_per_s: f64,
    /// Seconds.
    pub(crate) p2p_latency_s: f64,
}

impl HardwareLink {
    /// Decimal GB/s (`1 GB = 1e9 bytes`), the unit AISim's replay transfer
    /// model consumes.
    pub(crate) fn bandwidth_gbps(&self) -> f64 {
        self.bandwidth_bytes_per_s / 1e9
    }

    /// Hardware point-to-point latency in milliseconds.
    pub(crate) fn p2p_latency_ms(&self) -> f64 {
        self.p2p_latency_s * 1000.0
    }
}

/// Per-role KV-transfer configuration derived from native memory geometry and
/// one hardware link, kept beside the timing model so both cannot drift.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub(crate) struct KvTransferConfig {
    pub(crate) bytes_per_token: usize,
    pub(crate) bandwidth_gbps: f64,
    pub(crate) link: HardwareLink,
}

/// Read one calibration asset with an explicit symlink rejection and byte
/// bound. `AISIM_SYSTEMS_ROOT` is a trusted test path, but the artifact read
/// still refuses to follow links out of the owned canonical root.
fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("inspect calibration asset {}", path.display()))?;
    ensure!(
        !metadata.file_type().is_symlink() && metadata.file_type().is_file(),
        "calibration asset {} must be a regular file; symlinks and special files are rejected",
        path.display()
    );
    ensure!(
        metadata.len() <= limit,
        "calibration asset {} is {} bytes, above the {limit}-byte bound",
        path.display(),
        metadata.len()
    );
    let file = fs::File::open(path)
        .with_context(|| format!("open calibration asset {}", path.display()))?;
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .with_context(|| format!("read calibration asset {}", path.display()))?;
    ensure!(
        bytes.len() as u64 <= limit,
        "calibration asset {} exceeded the {limit}-byte bound while reading",
        path.display()
    );
    Ok(bytes)
}

/// Stream one regular file into SHA-256 without loading it whole.
fn hash_file_bounded(path: &Path, limit: u64) -> Result<(String, u64)> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("inspect calibration asset {}", path.display()))?;
    ensure!(
        !metadata.file_type().is_symlink() && metadata.file_type().is_file(),
        "calibration asset {} must be a regular file; symlinks and special files are rejected",
        path.display()
    );
    ensure!(
        metadata.len() <= limit,
        "calibration asset {} is {} bytes, above the {limit}-byte bound",
        path.display(),
        metadata.len()
    );
    let mut file = fs::File::open(path)
        .with_context(|| format!("open calibration asset {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; HASH_CHUNK_BYTES];
    let mut total = 0u64;
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("stream calibration asset {}", path.display()))?;
        if read == 0 {
            break;
        }
        total = total
            .checked_add(read as u64)
            .context("calibration asset size overflow")?;
        ensure!(
            total <= limit,
            "calibration asset {} exceeded the {limit}-byte bound while reading",
            path.display()
        );
        hasher.update(&buffer[..read]);
    }
    Ok((format!("{:x}", hasher.finalize()), total))
}

fn collect_manifest_files(
    root: &Path,
    dir: &Path,
    files: &mut Vec<(String, PathBuf)>,
) -> Result<()> {
    for entry in fs::read_dir(dir)
        .with_context(|| format!("read calibration directory {}", dir.display()))?
    {
        let entry = entry.with_context(|| format!("read entry in {}", dir.display()))?;
        let path = entry.path();
        let file_type = entry
            .file_type()
            .with_context(|| format!("inspect calibration entry {}", path.display()))?;
        ensure!(
            !file_type.is_symlink(),
            "calibration inventory {} contains a symlink; refusing to follow it",
            path.display()
        );
        if file_type.is_dir() {
            collect_manifest_files(root, &path, files)?;
        } else {
            ensure!(
                file_type.is_file(),
                "calibration inventory {} contains a non-regular entry",
                path.display()
            );
            let relative = path
                .strip_prefix(root)
                .with_context(|| format!("relativize calibration asset {}", path.display()))?
                .components()
                .map(|component| component.as_os_str().to_string_lossy())
                .collect::<Vec<_>>()
                .join("/");
            files.push((relative, path));
        }
    }
    Ok(())
}

/// Hash the complete canonical `data/<system>` subtree, sorted by relative
/// path, as `sha256(file) + "  " + relative_path + "\n"` lines. Every regular
/// file is included; symlinks and special files are rejected. The aggregate
/// is compared with the literal frozen from the pinned AISimulate git object.
fn calibration_inventory(
    systems: &Path,
    hardware: HardwareSelection,
) -> Result<(PathBuf, String, usize, u64)> {
    let (_, expected_manifest, expected_count, expected_bytes) = hardware.pins();
    let root = systems.join("data").join(hardware.system());
    let metadata = fs::symlink_metadata(&root)
        .with_context(|| format!("inspect calibration root {}", root.display()))?;
    ensure!(
        !metadata.file_type().is_symlink() && metadata.is_dir(),
        "calibration root {} must be a real directory; symlinks are rejected",
        root.display()
    );
    let mut files = Vec::new();
    collect_manifest_files(&root, &root, &mut files)?;
    files.sort_by(|left, right| left.0.cmp(&right.0));

    let mut aggregate = Sha256::new();
    let mut total_bytes = 0u64;
    for (relative, path) in &files {
        let (hash, bytes) = hash_file_bounded(path, MAX_CALIBRATION_FILE_BYTES)?;
        total_bytes = total_bytes
            .checked_add(bytes)
            .context("calibration subtree size overflow")?;
        aggregate.update(hash.as_bytes());
        aggregate.update(b"  ");
        aggregate.update(relative.as_bytes());
        aggregate.update(b"\n");
    }
    let manifest_sha256 = format!("{:x}", aggregate.finalize());
    ensure!(
        files.len() == expected_count
            && total_bytes == expected_bytes
            && manifest_sha256 == expected_manifest,
        "whole calibration root {} changed: {} files/{} bytes/manifest {} vs pinned {}/{} bytes/{}",
        root.display(),
        files.len(),
        total_bytes,
        manifest_sha256,
        expected_count,
        expected_bytes,
        expected_manifest
    );
    Ok((root, manifest_sha256, files.len(), total_bytes))
}

/// Supplementary source identity. The content hashes above, not this revision,
/// are what prove which timing bytes the control consumed.
fn source_revision(systems: &Path) -> Option<String> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(systems)
        .args(["rev-parse", "--verify", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let revision = String::from_utf8(output.stdout).ok()?;
    let revision = revision.trim();
    (revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .then(|| revision.to_owned())
}

/// Canonical maximum context length from the pinned model config, not an
/// engine default. `AISIM_SYSTEMS_ROOT` is `<core>/systems`, so the bundled
/// model configs are its sibling `model_configs` directory.
fn canonical_model_config(model: &str, systems: &Path) -> Result<ModelConfigEvidence> {
    let config = systems
        .parent()
        .context("AISIM_SYSTEMS_ROOT must live beside the canonical model_configs directory")?
        .join("model_configs")
        .join(format!("{}_config.json", model.replace('/', "--")));
    let bytes = read_bounded(&config, MAX_MODEL_CONFIG_BYTES)?;
    let value: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse canonical model config {}", config.display()))?;
    let max = value
        .get("max_position_embeddings")
        .or_else(|| value.get("max_seq_len"))
        .and_then(Value::as_u64)
        .with_context(|| {
            format!(
                "{} does not declare a canonical max_position_embeddings/max_seq_len",
                config.display()
            )
        })?;
    ensure!(max > 0, "{} max context must be positive", config.display());
    let max_model_len = usize::try_from(max).context("canonical model context exceeds usize")?;
    let sha256 = format!("{:x}", Sha256::digest(&bytes));
    if model == DEFAULT_MODEL {
        ensure!(
            sha256 == "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
            "canonical Qwen3-8B model config changed: {sha256}"
        );
    }
    if model == LONG_CONTEXT_MODEL {
        ensure!(
            sha256 == "3440a8570e178cfc8a44e0877d48d6bc959cceed89a8e3a9afa7fa45790dea4e",
            "canonical Nemotron Lightning model config changed: {sha256}"
        );
    }
    let recurrent_layers = value["layers_block_type"].as_array().map_or(0, |layers| {
        layers
            .iter()
            .filter(|layer| matches!(layer.as_str(), Some("mamba" | "linear_attention")))
            .count()
    });
    let recurrent_metadata = json!({
        "model_type": value["model_type"],
        "layers_block_type": value["layers_block_type"],
        "mamba_num_heads": value["mamba_num_heads"],
        "mamba_head_dim": value["mamba_head_dim"],
        "mamba_ssm_cache_dtype": value["mamba_ssm_cache_dtype"],
        "ssm_state_size": value["ssm_state_size"],
        "conv_kernel": value["conv_kernel"],
        "n_groups": value["n_groups"],
    });
    Ok(ModelConfigEvidence {
        path: config,
        sha256,
        bytes: bytes.len(),
        max_model_len,
        recurrent_layers,
        recurrent_metadata,
    })
}

/// Exact model metadata consumed for physical capacity and context limits.
#[derive(Debug, serde::Serialize)]
pub(crate) struct ModelConfigEvidence {
    pub(crate) path: PathBuf,
    pub(crate) sha256: String,
    pub(crate) bytes: usize,
    pub(crate) max_model_len: usize,
    pub(crate) recurrent_layers: usize,
    pub(crate) recurrent_metadata: Value,
}

fn hardware_link(systems: &Path, hardware: HardwareSelection) -> Result<HardwareLink> {
    let system = hardware.system();
    let (expected_yaml, _, _, _) = hardware.pins();
    let yaml = systems.join(format!("{system}.yaml"));
    let bytes = read_bounded(&yaml, MAX_HARDWARE_YAML_BYTES)?;
    let yaml_sha256 = format!("{:x}", Sha256::digest(&bytes));
    ensure!(
        yaml_sha256 == expected_yaml,
        "{} content SHA-256 {yaml_sha256} does not match the pinned {expected_yaml}",
        yaml.display()
    );
    let text =
        std::str::from_utf8(&bytes).with_context(|| format!("{} must be UTF-8", yaml.display()))?;
    let parsed: HardwareSystemYaml = serde_yaml::from_str(text)
        .with_context(|| format!("parse AISim hardware spec {}", yaml.display()))?;
    let (
        calibration_root,
        calibration_manifest_sha256,
        calibration_file_count,
        calibration_total_bytes,
    ) = calibration_inventory(systems, hardware)?;
    let node = parsed.node;
    ensure!(
        node.num_gpus_per_node > 0,
        "{} must describe a non-empty node",
        yaml.display()
    );
    ensure!(
        node.inter_node_bw.is_finite() && node.inter_node_bw > 0.0,
        "{} inter_node_bw must be finite and positive",
        yaml.display()
    );
    ensure!(
        node.p2p_latency.is_finite() && node.p2p_latency > 0.0,
        "{} p2p_latency must be finite and positive for a physical handoff",
        yaml.display()
    );
    Ok(HardwareLink {
        topology: LINK_TOPOLOGY,
        system,
        yaml,
        yaml_sha256,
        calibration_root,
        calibration_manifest_sha256,
        calibration_file_count,
        calibration_total_bytes,
        source_revision: source_revision(systems),
        num_gpus_per_node: node.num_gpus_per_node,
        bandwidth_bytes_per_s: node.inter_node_bw,
        p2p_latency_s: node.p2p_latency,
    })
}

fn memory_role(role: ForwardPassWorkerType) -> &'static str {
    match role {
        ForwardPassWorkerType::Prefill => "prefill",
        ForwardPassWorkerType::Decode => "decode",
        _ => "aggregated",
    }
}

/// A fully resolved, immutable per-role canonical FPM config. Callers must not
/// mutate it after construction: the timing model, native memory geometry and
/// router launch config all derive from this exact identity.
fn role_config(identity: RoleIdentity<'_>) -> Result<ForwardPassPerfModelConfig> {
    let mut config = ForwardPassPerfModelConfig::new(
        identity.model,
        identity.hardware.system(),
        BackendKind::Vllm,
        identity.role,
    );
    config.backend_version = Some(BACKEND_VERSION.into());
    config.tp = u32::try_from(identity.tensor_parallel_size)?;
    config.pp = 1;
    config.attention_dp = 1;
    config.gemm_quant_mode = Some(identity.precision.quant_mode().into());
    config.moe_quant_mode = Some(identity.precision.quant_mode().into());
    config.fmha_quant_mode = Some("bfloat16".into());
    config.kvcache_quant_mode = Some("bfloat16".into());
    config.estimation_mode = EstimationMode::OpLevel;
    config.fallback_policy = ForwardPassFallbackPolicy::Deny;
    config.database_mode = DatabaseMode::Silicon;
    config.strict_provenance = true;
    config.systems_paths = vec![identity.systems.to_path_buf()];
    config.estimator_config.correction.enabled = false;
    Ok(config)
}

pub(crate) struct RoleSetup {
    pub(crate) timing: Arc<CanonicalTimingModel>,
    pub(crate) tensor_parallel_size: u32,
    /// Recurrent-state modeling limitations are separate from attention KV.
    pub(crate) state_geometry: Value,
    pub(crate) rank: EngineConfig,
    pub(crate) router: MockerConfig,
    pub(crate) diagnostics: Value,
    pub(crate) geometry: Value,
    pub(crate) model_config: ModelConfigEvidence,
    /// Physical KV payload and hardware link used for the P/D handoff.
    pub(crate) transfer: KvTransferConfig,
}

/// Router-side adapter over the exact canonical estimator used by the engine.
/// The owning Mocker boundary has no canonical FPM constructor of its own: an
/// external AIS provider is materialized as a process-local `AisCallback`, so
/// the host must install the real model here instead of leaving Polynomial or
/// choosing `PerfModel::Fixed`.
struct CanonicalAisCallback {
    timing: Arc<CanonicalTimingModel>,
}

impl AisCallback for CanonicalAisCallback {
    fn predict_prefill(&self, batch_size: usize, isl: usize, prefix: usize) -> Result<f64> {
        self.timing.predict_prefill_ms(batch_size, isl, prefix)
    }

    fn predict_decode(
        &self,
        batch_size: usize,
        context_length: usize,
        _output_tokens: usize,
    ) -> Result<f64> {
        let total_kv_tokens = batch_size
            .checked_mul(context_length)
            .context("decode KV overflow")?;
        self.timing
            .predict_decode_ms(batch_size, total_kv_tokens, context_length, total_kv_tokens)
    }
}

pub(crate) fn canonical_role(identity: RoleIdentity<'_>) -> Result<RoleSetup> {
    let RoleIdentity {
        model,
        systems,
        hardware,
        precision,
        role,
        tensor_parallel_size,
        max_num_seqs,
        prefill,
    } = identity;
    ensure!(
        max_num_seqs > 0 && tensor_parallel_size > 0,
        "role sequence and TP limits must be positive"
    );
    let model_config = canonical_model_config(model, systems)?;
    // An unchunked FIFO head needs a complete model-admissible pass. Chunked
    // serving instead sizes activations to its real per-pass budget, without
    // changing the model's native context maximum.
    let (max_num_batched_tokens, enable_chunked_prefill) = match prefill {
        PrefillMode::Unchunked => (
            MAX_NUM_BATCHED_TOKENS.max(model_config.max_model_len),
            false,
        ),
        PrefillMode::Chunked {
            max_num_batched_tokens,
        } => {
            ensure!(
                max_num_batched_tokens > 0,
                "chunked prefill budget must be positive"
            );
            (max_num_batched_tokens, true)
        }
    };
    let system = hardware.system();
    let config = role_config(identity)?;
    let memory = KvCacheEstimateRequest {
        engine: serde_json::from_value(json!({
            "schema_version": aisimulate_core::perfmodel::ENGINE_CONFIG_SCHEMA_VERSION,
            "model_name": model,
            "system_name": system,
            "systems_path": systems,
            "backend": BackendKind::Vllm,
            "backend_version": BACKEND_VERSION,
            "forward_model": "op_level",
            "tp_size": tensor_parallel_size,
            "pp_size": 1,
            "attention_dp_size": 1,
            "kv_block_size": BLOCK_SIZE,
            "weight_dtype": precision.weight_dtype(),
            "moe_dtype": precision.weight_dtype(),
            "activation_dtype": DataType::Bfloat16,
            "kv_cache_dtype": DataType::Bfloat16,
            "strict_provenance": true,
            "database_mode": DatabaseMode::Silicon,
            // Canonical geometry rejects a disaggregated role that silently
            // inherits the aggregated worker default.
            "extra": {"worker_type": memory_role(role)},
        }))
        .context("complete matching native memory engine identity")?,
        max_num_tokens: u32::try_from(max_num_batched_tokens)
            .context("serving per-pass token budget must fit u32")?,
        max_batch_size: u32::try_from(max_num_seqs)?,
        kv_cache_memory_fraction: KvCacheMemoryFraction::OfTotal(0.9),
        gpu_memory_capacity_bytes_override: None,
        cuda_graph_reserved_bytes: 0,
        tolerance_fraction: Some(0.05),
        options: KvCacheEstimateOptions {
            allow_naive_fallback: false,
            allow_hf_config_download: false,
            naive_kv_reservation: 0.8,
        },
    };
    let (timing, memory) =
        CanonicalTimingModel::new_with_geometry(config, memory).with_context(|| {
            format!(
                "calibrated {role:?} timing/memory for {model} on {system} \
                 vllm {BACKEND_VERSION}; checked-in op-level silicon tables must \
                 serve this identity and deny fallback"
            )
        })?;

    let provenance = timing
        .provenance()
        .context("canonical construction omitted resolved provenance")?;
    ensure!(
        provenance.selected_estimation_mode == EstimationMode::OpLevel,
        "canonical {role:?} timing did not select op-level estimation"
    );
    ensure!(
        provenance.config.database_mode == DatabaseMode::Silicon,
        "canonical {role:?} timing did not select the silicon database"
    );
    ensure!(
        provenance.config.fallback_policy == ForwardPassFallbackPolicy::Deny,
        "canonical {role:?} timing must deny fallback"
    );
    ensure!(
        provenance.selection_failures.is_empty(),
        "no alternate estimator may qualify the {role:?} proof"
    );
    ensure!(
        memory.source == EstimateSource::Native,
        "canonical {role:?} geometry requires the native memory model"
    );
    ensure!(
        memory.total_kv_size_tokens > 0 && memory.kv_size_per_token_bytes > 0,
        "canonical {role:?} geometry is empty"
    );

    // The handoff carries exactly the native per-token KV geometry the rank
    // already uses for capacity, over one explicit hardware link. Nothing here
    // is a fixed transfer constant.
    let link = hardware_link(systems, hardware)?;
    ensure!(
        tensor_parallel_size <= link.num_gpus_per_node as usize,
        "proof TP size exceeds the selected hardware node"
    );
    let bytes_per_token = usize::try_from(memory.kv_size_per_token_bytes)
        .context("native KV geometry exceeds usize")?;
    ensure!(
        bytes_per_token > 0,
        "native {role:?} geometry must yield positive KV transfer bytes per token"
    );
    let bandwidth_gbps = link.bandwidth_gbps();
    ensure!(
        bandwidth_gbps.is_finite() && bandwidth_gbps > 0.0,
        "{} inter-node link must yield positive decimal GB/s",
        link.yaml.display()
    );

    let diagnostics = serde_json::to_value(timing.diagnostics())?;
    ensure!(
        diagnostics["source"] == json!("aic"),
        "calibrated {role:?} timing must be served by the AIC provider: {diagnostics}"
    );

    let adjusted = memory
        .tolerance_adjusted
        .as_ref()
        .context("explicit tolerance geometry")?;
    let num_gpu_blocks = usize::try_from(adjusted.total_kv_size_tokens)? / BLOCK_SIZE;
    ensure!(
        num_gpu_blocks > 0,
        "native {role:?} geometry yields no KV blocks"
    );

    let timing_config = serde_json::to_value(&provenance.config)?;
    // Full GPU modeling must not treat `None` as unlimited: the pinned model
    // config's native context maximum is the rank's prompt-plus-output limit.
    let max_model_len = model_config.max_model_len;
    let rank = EngineConfig {
        // External AIS timing is priced by the role factory supplied below; the
        // rank still names the exact resolved config so the engine and router
        // cannot silently diverge.
        timing_model: TimingModelConfig::External {
            provider: "aic".into(),
            config: timing_config.clone(),
        },
        max_num_seqs,
        max_num_batched_tokens,
        max_model_len: Some(max_model_len),
        block_size: BLOCK_SIZE,
        num_gpu_blocks,
        kv_cache_bytes_per_token: Some(bytes_per_token),
        kv_transfer_bytes_per_token: Some(bytes_per_token),
        kv_transfer_bandwidth: Some(bandwidth_gbps),
        // Transfer the complete prompt footprint; the causal assertion below
        // prices exactly this token count, never a partial cached prefix.
        kv_transfer_timing_mode: TransferTimingMode::FullPrompt,
        enable_prefix_caching: true,
        enable_chunked_prefill,
        async_scheduling: Some(false),
        worker_type: match role {
            ForwardPassWorkerType::Prefill => aisimulate_core::engine::WorkerType::Prefill,
            ForwardPassWorkerType::Decode => aisimulate_core::engine::WorkerType::Decode,
            ForwardPassWorkerType::Aggregated => aisimulate_core::engine::WorkerType::Aggregated,
        },
        ..Default::default()
    };
    let router =
        canonical_router_config(&rank, tensor_parallel_size, &timing_config, timing.clone())?;

    let state_geometry = if model_config.recurrent_layers > 0 {
        json!({
            "required": true,
            "supported": false,
            "full_physical_gpu_modeling": false,
            "recurrent_layers": model_config.recurrent_layers,
            "metadata": model_config.recurrent_metadata,
            "reason": "canonical estimate_state_cache supports Kimi K3/DeepSeek V4 only; native Nemotron token KV explicitly excludes fixed per-request Mamba state; no state capacity or transfer size is invented",
        })
    } else {
        json!({"required": false, "supported": true, "reason": "pinned model config declares no recurrent layers"})
    };
    Ok(RoleSetup {
        timing,
        tensor_parallel_size: u32::try_from(tensor_parallel_size)?,
        state_geometry,
        rank,
        router,
        diagnostics,
        geometry: serde_json::to_value(&memory)?,
        model_config,
        transfer: KvTransferConfig {
            bytes_per_token,
            bandwidth_gbps,
            link,
        },
    })
}

/// Build the router launch config through its owning parser and require that it
/// carries the same canonical FPM config. The parser maps external AIS timing
/// to a callback slot; install the real canonical estimator there and fail if
/// the provider or the resulting slot is not what this proof requires.
fn canonical_router_config(
    rank: &EngineConfig,
    tensor_parallel_size: usize,
    timing_config: &Value,
    timing: Arc<CanonicalTimingModel>,
) -> Result<MockerConfig> {
    let launch = EngineLaunchConfig {
        engine: rank.clone(),
        dp_size: 1,
        tensor_parallel_size,
        num_gpu_blocks_is_explicit: true,
        ..Default::default()
    };
    let mut router = MockerConfig::from_value(json!({"engine": launch}))
        .context("canonical engine rank must be a valid Mocker launch config")?;
    ensure!(
        router.tensor_parallel_size == tensor_parallel_size,
        "router must preserve canonical TP identity"
    );
    ensure!(
        router.ais_perf_config() == Some(timing_config),
        "router MockerConfig must carry the same resolved canonical FPM config as the role engine"
    );
    ensure!(
        !matches!(router.perf_model.as_ref(), PerfModel::Fixed { .. }),
        "canonical external timing must never be priced by PerfModel::Fixed"
    );
    router.perf_model = Arc::new(PerfModel::from_ais_callback(Arc::new(
        CanonicalAisCallback {
            timing: timing.clone(),
        },
    )));
    Ok(router)
}

fn native_run(frame: &str, model: &str) -> Result<aiperf_config::model::BenchmarkRun> {
    let mut run = aiperf_protocol::runner_v2::decode_run_request_v2(frame.as_bytes())
        .context("native strict codec")?
        .run;
    let item = run
        .cfg
        .models
        .as_mut()
        .context("fixture must name models")?
        .items
        .first_mut()
        .context("fixture must name at least one model")?;
    item.name = model.to_owned();
    Ok(run)
}

pub(crate) fn role_descriptor(setup: &RoleSetup) -> ReplayRoleConfig {
    ReplayRoleConfig {
        dp_size: 1,
        tensor_parallel_size: setup.tensor_parallel_size,
        prefill_context_parallel_size: 1,
        num_gpu_blocks_is_explicit: Some(true),
        rank: setup.rank.clone(),
    }
}

/// Disaggregated 2P2D spec with separate per-role geometry. Requests are
/// supplied by the hosted participant, not by the serializable spec.
pub(crate) fn spec(prefill: &RoleSetup, decode: &RoleSetup) -> Result<ReplaySpec> {
    ensure!(
        prefill.state_geometry["supported"] == true && decode.state_geometry["supported"] == true,
        "full physical replay requires supported recurrent-state geometry; prefill={} decode={}",
        prefill.state_geometry,
        decode.state_geometry
    );
    ensure!(
        prefill.tensor_parallel_size == decode.tensor_parallel_size,
        "proof prefill/decode TP topology must match"
    );
    ensure!(
        prefill.transfer.link == decode.transfer.link,
        "prefill and decode must share one hardware link assumption"
    );
    ensure!(
        prefill.transfer.bytes_per_token > 0 && decode.transfer.bytes_per_token > 0,
        "every disaggregated role needs positive native KV transfer bytes per token"
    );
    ensure!(
        prefill.transfer.bytes_per_token == decode.transfer.bytes_per_token,
        "prefill and decode must price the same native KV transfer bytes per token: {} vs {}",
        prefill.transfer.bytes_per_token,
        decode.transfer.bytes_per_token
    );
    ensure!(
        prefill.transfer.bandwidth_gbps > 0.0 && decode.transfer.bandwidth_gbps > 0.0,
        "every disaggregated role needs a positive physical transfer bandwidth"
    );
    ensure!(
        prefill.transfer.bandwidth_gbps == decode.transfer.bandwidth_gbps,
        "prefill and decode must price the same transfer bandwidth: {} vs {}",
        prefill.transfer.bandwidth_gbps,
        decode.transfer.bandwidth_gbps
    );
    // AISim applies `handoff_latency_ms` only when a role cannot derive a delay
    // from `kv_transfer_bytes_per_token` + `kv_transfer_bandwidth`. Both roles
    // supply that pair, so the bandwidth-priced serialization is authoritative
    // and this hardware p2p fallback is not added on top (no double count).
    let handoff_latency_ms = prefill.transfer.link.p2p_latency_ms();
    ensure!(
        handoff_latency_ms.is_finite() && handoff_latency_ms > 0.0,
        "hardware p2p latency must produce a positive finite handoff fallback"
    );
    serde_json::from_value(json!({
        "version": 1,
        "topology": {
            "kind": "disaggregated",
            "prefill": {"initial_workers": PREFILL_WORKERS},
            "decode": {"initial_workers": DECODE_WORKERS},
            "handoff_latency_ms": handoff_latency_ms
        },
        "engine": ReplayEngineConfig {
            dp_size: 1,
            tensor_parallel_size: prefill.tensor_parallel_size,
            prefill_context_parallel_size: 1,
            num_gpu_blocks_is_explicit: Some(true),
            rank: prefill.rank.clone(),
            prefill: Some(role_descriptor(prefill)),
            decode: Some(role_descriptor(decode)),
            ..Default::default()
        },
        "adapters": {"placement": {"provider": "dynamo_kv_router"}},
        "requests": []
    }))
    .context("strict disaggregated ReplaySpec")
}

/// Host the production AIPerf plans on the native Replayer event loop with the
/// real Dynamo KV Router composition and the per-role canonical factories.
pub(crate) fn host(
    run: &aiperf_config::model::BenchmarkRun,
    prefill: &RoleSetup,
    decode: &RoleSetup,
) -> Result<(ReplayReport, Value)> {
    let (participant, handle) = aisim_aiperf_workload_runner::participant_adapter::native_hosted(
        run,
        aisim_aiperf_workload_runner::participant_adapter::NativeInputProjection::Text,
        RunCancellation::new(),
        Rc::new(aiperf_timing::RecordingPhaseObserver::default()),
        Rc::new(aiperf_rng::identity::DeterministicIdentitySource::new(
            IDENTITY_SEED,
        )),
    )
    .context("shared native lowering hosts the production plans")?;
    let prefill_timing: Arc<dyn TimingModel> = prefill.timing.clone();
    let decode_timing: Arc<dyn TimingModel> = decode.timing.clone();
    let composition = KvReplayComposition::disaggregated(
        prefill.router.clone(),
        decode.router.clone(),
        PREFILL_WORKERS,
        DECODE_WORKERS,
        None,
        None,
        None,
    );
    let report = Replayer::with_composition(
        spec(prefill, decode)?,
        ReplayEngineFactory::with_optional_role_timing_models(
            Some(prefill_timing),
            Some(decode_timing),
        ),
        composition,
    )
    .context("real Dynamo KV composition accepts the disaggregated calibrated spec")?
    .with_runtime_input(ReplayRuntimeInput::External(Box::new(participant)))
    .with_capture_options(ReplayCaptureOptions {
        capture_per_request: true,
        determinism: ReplayDeterminism::CanonicalV1,
        ..Default::default()
    })
    .run()
    .context("native Replayer event loop hosts calibrated AIPerf over the real KV composition")?;
    Ok((report, handle.into_report().context("AIPerf phase report")?))
}

fn per_request_records(report: &ReplayReport) -> Vec<Value> {
    serde_json::to_value(&report.per_request)
        .expect("engine per-request records")
        .as_array()
        .expect("engine per-request records")
        .clone()
}

fn assert_calibrated_branch(report: &ReplayReport, hosted: &Value, prefill: &RoleSetup) {
    let rows = hosted["requests"].as_array().expect("hosted requests");
    assert_eq!(
        rows.len(),
        4,
        "the seeded strict frame has two parent and two child requests"
    );
    for row in rows {
        assert_eq!(row["status"], "settled");
        assert_eq!(row["terminal"], "completed");
        assert_eq!(row["output_token_ids"].as_array().expect("tokens").len(), 2);
    }
    let correlations: BTreeSet<_> = rows
        .iter()
        .map(|row| {
            row["request"]["correlation_id"]
                .as_str()
                .expect("correlation id")
        })
        .collect();
    assert_eq!(
        correlations.len(),
        2,
        "parent and child are distinct sessions"
    );

    let records = per_request_records(report);
    assert_eq!(records.len(), 4, "engine must retain every request");
    for record in &records {
        assert_eq!(record["terminal_status"], "completed");
        let history = record["routing_history"]
            .as_array()
            .expect("routing history");
        let pools: Vec<&str> = history
            .iter()
            .map(|route| route["pool"].as_str().expect("route pool"))
            .collect();
        assert_eq!(
            pools.iter().filter(|pool| **pool == "prefill").count(),
            1,
            "each request is placed on exactly one prefill worker: {history:?}"
        );
        assert_eq!(
            pools.iter().filter(|pool| **pool == "decode").count(),
            1,
            "each request is placed on exactly one decode worker: {history:?}"
        );
        assert!(
            record["prefill_worker_idx"]
                .as_u64()
                .expect("prefill worker")
                < PREFILL_WORKERS as u64
        );
        assert!(
            record["decode_worker_idx"].as_u64().expect("decode worker") < DECODE_WORKERS as u64
        );
        let ttft = record["ttft_ms"].as_f64().expect("ttft");
        assert!(ttft > 0.0, "calibrated TTFT must be positive: {ttft}");
        assert!(
            record["terminal_time_ms"].as_f64().expect("terminal")
                > record["arrival_time_ms"].as_f64().expect("arrival"),
            "calibrated terminal must follow arrival"
        );

        // The handoff is a real transfer, not an instant metadata hop: the
        // source must hold KV, the destination must reserve and then activate,
        // and activation must follow source-held by positive modeled time.
        let prefill_complete_ms = record["prefill_complete_ms"]
            .as_f64()
            .expect("disaggregated prefill completion timestamp");
        let source_held_ms = record["source_held_ms"]
            .as_f64()
            .expect("disaggregated source-held timestamp");
        let destination_reserved_ms = record["destination_reserved_ms"]
            .as_f64()
            .expect("disaggregated destination-reservation timestamp");
        let destination_activated_ms = record["destination_activated_ms"]
            .as_f64()
            .expect("disaggregated destination-activation timestamp");
        assert!(
            source_held_ms > 0.0 && destination_activated_ms > 0.0,
            "handoff timestamps must be positive: source_held={source_held_ms} \
             destination_activated={destination_activated_ms}"
        );
        assert!(
            prefill_complete_ms <= source_held_ms,
            "source cannot hold KV before prefill completes"
        );
        assert!(
            source_held_ms <= destination_reserved_ms,
            "vLLM source-first handoff must reserve the destination after the source held"
        );
        assert!(
            destination_reserved_ms <= destination_activated_ms,
            "destination activation cannot precede its reservation"
        );
        let transfer_ms = destination_activated_ms - source_held_ms;
        assert!(
            transfer_ms > 0.0,
            "physical KV transfer must take positive modeled time, got {transfer_ms}ms"
        );
        let input_length = record["input_length"]
            .as_u64()
            .expect("per-request input length");
        let expected_serialization_ms = input_length as f64
            * prefill.transfer.bytes_per_token as f64
            / (prefill.transfer.bandwidth_gbps * 1e9)
            * 1000.0;
        assert!(
            transfer_ms + 1e-6 >= expected_serialization_ms,
            "handoff {transfer_ms}ms is shorter than the physical \
             {expected_serialization_ms}ms serialization of {input_length} tokens at \
             {} bytes/token over the {} {} link",
            prefill.transfer.bytes_per_token,
            prefill.transfer.link.topology,
            prefill.transfer.link.system
        );
    }
}

/// Positive causal control: the calibrated model is shape dependent on both
/// roles and cannot coincide with the fixed `1ms/2ms` per-pass constants for
/// every probed shape. This is the cheap substitute for an unavailable native
/// per-pass trace; the full run's raw per-request TTFTs are printed alongside.
fn assert_not_fixed(prefill: &RoleSetup, decode: &RoleSetup) -> Result<(f64, f64, f64, f64)> {
    let prefill_small = prefill.timing.predict_prefill_ms(1, 16, 0)?;
    let prefill_large = prefill.timing.predict_prefill_ms(1, 2048, 0)?;
    ensure!(
        prefill_small > 0.0 && prefill_large > 0.0,
        "calibrated prefill timings must be positive"
    );
    ensure!(
        prefill_small != prefill_large,
        "calibrated prefill must vary with prompt shape, not stay a fixed per-pass constant"
    );
    ensure!(
        prefill_small != FIXED_PREFILL_MS || prefill_large != FIXED_PREFILL_MS,
        "calibrated prefill is indistinguishable from the fixed {FIXED_PREFILL_MS}ms control"
    );

    let decode_small = decode.timing.predict_decode_ms(1, 32, 32, 32)?;
    let decode_large = decode.timing.predict_decode_ms(1, 4096, 4096, 4096)?;
    ensure!(
        decode_small > 0.0 && decode_large > 0.0,
        "calibrated decode timings must be positive"
    );
    ensure!(
        decode_small != decode_large,
        "calibrated decode must vary with context, not stay a fixed per-pass constant"
    );
    ensure!(
        decode_small != FIXED_DECODE_MS || decode_large != FIXED_DECODE_MS,
        "calibrated decode is indistinguishable from the fixed {FIXED_DECODE_MS}ms control"
    );
    Ok((prefill_small, prefill_large, decode_small, decode_large))
}

/// Calibrated H100 FPM control body. Shared by the standalone
/// `calibrated_native_hosted` test target and the full-scenario E2E, so the
/// canonical role/spec/host path exists exactly once.
pub(crate) fn calibrated_control() -> Result<()> {
    let (model, model_source) = configured_model();
    ensure!(!model.trim().is_empty(), "AISIM_MODEL must not be empty");
    let systems = systems_root()?;

    let identity = RoleIdentity {
        model: &model,
        systems: &systems,
        hardware: HardwareSelection::H100Sxm,
        precision: ProofPrecision::Bfloat16,
        role: ForwardPassWorkerType::Prefill,
        tensor_parallel_size: 1,
        max_num_seqs: MAX_NUM_SEQS,
        prefill: PrefillMode::Unchunked,
    };
    let prefill = canonical_role(identity)
        .context("prefill canonical op-level timing and native geometry")?;
    let decode = canonical_role(RoleIdentity {
        role: ForwardPassWorkerType::Decode,
        ..identity
    })
    .context("decode canonical op-level timing and native geometry")?;
    let (prefill_small, prefill_large, decode_small, decode_large) =
        assert_not_fixed(&prefill, &decode).context("fixed-vs-calibrated causal control")?;

    // The control must actually price a physical handoff: nonzero native KV
    // payload on both roles, a positive hardware link, and one shared
    // topology assumption with a positive p2p fallback.
    ensure!(
        prefill.transfer.bytes_per_token > 0 && decode.transfer.bytes_per_token > 0,
        "native KV geometry must yield positive transfer bytes per token on every role"
    );
    ensure!(
        prefill.transfer.bandwidth_gbps > 0.0 && decode.transfer.bandwidth_gbps > 0.0,
        "the hardware link must yield positive transfer bandwidth on every role"
    );
    ensure!(
        prefill.transfer.link == decode.transfer.link
            && prefill.transfer.link.p2p_latency_ms() > 0.0,
        "disaggregated roles must share one positive hardware handoff link"
    );

    let directory = tempfile::tempdir().context("temporary fixture directory")?;
    let run = native_run(&run_fixture::branch_request(directory.path()), &model)
        .context("native strict run with the configured model identity")?;
    let (report, hosted) = host(&run, &prefill, &decode).context("calibrated native proof")?;
    assert_calibrated_branch(&report, &hosted, &prefill);

    let mut ttfts: Vec<f64> = per_request_records(&report)
        .iter()
        .map(|record| record["ttft_ms"].as_f64().expect("ttft"))
        .collect();
    ttfts.sort_by(f64::total_cmp);
    let ttft_nonconstant = ttfts
        .windows(2)
        .any(|pair| pair[0].to_bits() != pair[1].to_bits());

    println!(
        "CALIBRATED_NATIVE_HOSTED_RAW {}",
        serde_json::to_string_pretty(&json!({
            "proof": "calibrated-native-hosted-h100-op-level-bf16-silicon-deny-fallback",
            "scope": "preparatory GPU provider control; not a full-scenario or tokenizer-equivalence claim",
            "model": model,
            "model_source": model_source,
            "max_model_len_note": "rank.max_model_len derives from the pinned model config max_position_embeddings; unbounded context is not claimed",
            "system": SYSTEM,
            "backend": "vllm",
            "backend_version": BACKEND_VERSION,
            "precision": "bfloat16",
            "hardware_kv_transfer": {
                "topology_assumption": prefill.transfer.link.topology,
                "system": prefill.transfer.link.system,
                "hardware_spec": prefill.transfer.link.yaml.display().to_string(),
                "hardware_yaml_sha256": prefill.transfer.link.yaml_sha256.as_str(),
                "hardware_yaml_sha256_expected": HARDWARE_YAML_SHA256,
                "num_gpus_per_node": prefill.transfer.link.num_gpus_per_node,
                "inter_node_bandwidth_bytes_per_s": prefill.transfer.link.bandwidth_bytes_per_s,
                "inter_node_bandwidth_decimal_gb_s": prefill.transfer.bandwidth_gbps,
                "p2p_latency_s": prefill.transfer.link.p2p_latency_s,
                "p2p_latency_ms": prefill.transfer.link.p2p_latency_ms(),
                "handoff_latency_ms": prefill.transfer.link.p2p_latency_ms(),
                "handoff_latency_use": "fallback only when a role lacks bytes-per-token or positive bandwidth; never added to the bandwidth-priced serialization",
                "bytes_per_token": {
                    "prefill": prefill.transfer.bytes_per_token,
                    "decode": decode.transfer.bytes_per_token,
                },
                "note": "inter-node hardware link assumption for abstract 2P2D workers; not a measured deployment network claim",
            },
            "calibration_inventory": {
                "scope": "complete canonical data/h100_sxm subtree; all regular files, not only the op tables this control consumes",
                "root": prefill.transfer.link.calibration_root.display().to_string(),
                "manifest_algorithm": "SHA-256 over sorted '<file-sha256>  <relative-path>\\n' lines, streamed; symlinks and special files rejected",
                "manifest_sha256": prefill.transfer.link.calibration_manifest_sha256.as_str(),
                "manifest_sha256_expected": CALIBRATION_MANIFEST_SHA256,
                "file_count": prefill.transfer.link.calibration_file_count,
                "total_bytes": prefill.transfer.link.calibration_total_bytes,
                "source_revision": prefill.transfer.link.source_revision.as_deref(),
                "source_revision_note": "git HEAD when a checkout is available; content hashes, not revision identity, enforce integrity",
                "op_table_inventory": "per-role canonical timing diagnostics below; no separate owner-exposed op-table source-file list was available",
            },
            "tokenizer": serde_json::to_value(&run.cfg.tokenizer).expect("tokenizer descriptor"),
            "tokenizer_note": "fixture builtin tokenizer; main acquires the pinned model tokenizer before a full-scenario claim",
            "roles": {
                "prefill": {
                    "timing": prefill.diagnostics,
                    "native_kv_geometry": prefill.geometry,
                    "model_config": prefill.model_config,
                    "kv_transfer": {
                        "bytes_per_token": prefill.transfer.bytes_per_token,
                        "bandwidth_decimal_gb_s": prefill.transfer.bandwidth_gbps,
                    },
                    "router_perf_model": "canonical_ais_callback",
                    "replay_rank": prefill.rank,
                },
                "decode": {
                    "timing": decode.diagnostics,
                    "native_kv_geometry": decode.geometry,
                    "model_config": decode.model_config,
                    "kv_transfer": {
                        "bytes_per_token": decode.transfer.bytes_per_token,
                        "bandwidth_decimal_gb_s": decode.transfer.bandwidth_gbps,
                    },
                    "router_perf_model": "canonical_ais_callback",
                    "replay_rank": decode.rank,
                },
            },
            "causal_control": {
                "fixed": {"prefill_ms": FIXED_PREFILL_MS, "decode_ms": FIXED_DECODE_MS},
                "calibrated": {
                    "prefill_16_ms": prefill_small,
                    "prefill_2048_ms": prefill_large,
                    "decode_32_ms": decode_small,
                    "decode_4096_ms": decode_large,
                },
            },
            "actual_ttft_ms": ttfts,
            "actual_ttft_nonconstant": ttft_nonconstant,
            "native_engine_report": report,
            "hosted_report": hosted,
        }))
        .expect("calibrated native artifact")
    );

    Ok(())
}
