// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Preparatory calibrated GPU control for the native AIPerf-hosted Replayer.
//!
//! This proves *provider wiring*, not a full-scenario equivalence claim. Per
//! role (prefill and decode), the canonical H100 BF16 op-level silicon timing
//! model and capacity configuration from the canonical compiler are the one
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
use aisimulate_core::engine::{EngineConfig, TimingModel, TimingModelConfig};
use aisimulate_core::replay::{
    PreparedReplay, ReplayCaptureOptions, ReplayDeterminism, ReplayEngineConfig,
    ReplayEngineFactory, ReplayReport, ReplayRoleConfig, ReplayRuntimeInput, ReplaySpec, Replayer,
    WorkerStage,
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

/// Per-role KV-transfer configuration resolved by the owning compiler and
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

pub(crate) struct RoleSetup {
    pub(crate) timing: Arc<dyn TimingModel>,
    descriptor: ReplayRoleConfig,
    pub(crate) state_geometry: Value,
    pub(crate) rank: EngineConfig,
    pub(crate) router: MockerConfig,
    pub(crate) diagnostics: Value,
    pub(crate) geometry: Value,
    pub(crate) model_config: ModelConfigEvidence,
    pub(crate) transfer: KvTransferConfig,
    pub(crate) factory: ReplayEngineFactory,
    physical_spec: ReplaySpec,
}

/// Configuration policy belongs entirely to AISimulate's public prediction
/// schema, compiler, Runner materializer and existing physical role factory.
/// This proof only binds the resulting model instance to Dynamo's placement API.
pub(crate) fn roles_from_config(config: Value) -> Result<(RoleSetup, RoleSetup)> {
    let prepared = PreparedReplay::from_prediction_config(&serde_json::to_string(&config)?)?;
    let engine: ReplayEngineConfig = serde_json::from_value(prepared.spec.engine.clone())?;
    ensure!(
        matches!(
            prepared.spec.topology,
            aisimulate_core::replay::ReplayTopology::Disaggregated { .. }
        ),
        "this composition control requires a disaggregated physical configuration"
    );
    let role_setup = |stage| -> Result<RoleSetup> {
        let descriptor = engine.role(stage);
        let timing = prepared
            .factory
            .timing_model(stage)
            .context("canonical role did not materialize a timing model")?;
        let TimingModelConfig::External { provider, config } = &descriptor.rank.timing_model else {
            anyhow::bail!("calibrated control requires canonical AIS timing");
        };
        ensure!(
            provider == "aic",
            "calibrated control requires the owning AIS provider"
        );
        let model = config["model"]
            .as_str()
            .context("resolved canonical model identity")?;
        let system = config["system"]
            .as_str()
            .context("resolved canonical hardware identity")?;
        let hardware = match system {
            "h100_sxm" => HardwareSelection::H100Sxm,
            "b200_sxm" => HardwareSelection::B200Sxm,
            _ => anyhow::bail!("no pinned proof asset inventory for {system}"),
        };
        let systems = config["systems_paths"]
            .as_array()
            .filter(|paths| paths.len() == 1)
            .and_then(|paths| paths[0].as_str())
            .map(PathBuf::from)
            .context("proof requires one explicit canonical systems root")?;
        let model_config = canonical_model_config(model, &systems)?;
        let link = hardware_link(&systems, hardware)?;
        let launch = descriptor.launch_config(None)?;
        let router = canonical_router_config(&launch, timing.clone())?;
        let bytes_per_token = descriptor
            .rank
            .kv_transfer_bytes_per_token
            .context("owning compiler did not resolve KV transfer footprint")?;
        let bandwidth_gbps = descriptor
            .rank
            .kv_transfer_bandwidth
            .context("physical configuration must author a transfer link")?;
        ensure!(
            bandwidth_gbps == link.bandwidth_gbps(),
            "configured transfer link must match this control's pinned inter-node hardware assumption"
        );
        let role_name = match stage {
            WorkerStage::Prefill => "prefill",
            WorkerStage::Decode => "decode",
            WorkerStage::Aggregated => "aggregated",
        };
        let state_geometry = json!({
            "required": model_config.recurrent_layers > 0,
            "supported": model_config.recurrent_layers == 0,
            "recurrent_layers": model_config.recurrent_layers,
            "note": "proof inventory only; no recurrent-state capacity or payload is invented",
        });
        let diagnostics = json!({"source": provider, "resolved_config": config,
            "timing_evidence_available": timing.evidence_summary().is_some(),
            "timing_evidence_note": "availability only; no operation-level evidence is serialized by this control"});
        Ok(RoleSetup {
            timing,
            descriptor: descriptor.clone(),
            state_geometry,
            rank: descriptor.rank,
            router,
            diagnostics,
            geometry: prepared.memory_diagnostics[role_name].clone(),
            model_config,
            transfer: KvTransferConfig {
                bytes_per_token,
                bandwidth_gbps,
                link,
            },
            factory: prepared.factory.clone(),
            physical_spec: prepared.spec.clone(),
        })
    };
    Ok((
        role_setup(WorkerStage::Prefill)?,
        role_setup(WorkerStage::Decode)?,
    ))
}

/// A small authored AISim configuration, not a second physical configuration
/// implementation. The same canonical parser and factory serve the full target.
pub(crate) fn small_control_roles() -> Result<(RoleSetup, RoleSetup)> {
    let (model, _) = configured_model();
    let systems = systems_root()?;
    let link = hardware_link(&systems, HardwareSelection::H100Sxm)?;
    let worker = json!({
        "parallelism": {"replicas": PREFILL_WORKERS, "tensor": 1, "pipeline": 1,
            "attention_data": 1, "moe_tensor": 1, "moe_expert": 1},
        "scheduler": {"max_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "max_sequences": MAX_NUM_SEQS, "async_scheduling": false},
        "kv_cache": {"block_size": BLOCK_SIZE, "prefix_caching": true,
            "capacity": {"type": "default", "memory_fraction": 0.9}},
        "timing": {"type": "default"},
    });
    roles_from_config(json!({"engine": {
        "mode": "disaggregated", "model": model, "hardware": SYSTEM,
        "backend": "vllm", "backend_version": BACKEND_VERSION,
        "enable_chunked_prefill": true,
        "gemm_quant_mode": "bfloat16", "moe_quant_mode": "bfloat16",
        "kvcache_quant_mode": "bfloat16", "fmha_quant_mode": "bfloat16",
        "estimation_mode": "op_level", "fallback_policy": "deny", "database_mode": "SILICON",
        "systems_paths": [systems], "estimator_config": {"correction": {"enabled": false}},
        "kv_transfer": {"bandwidth_gb_per_second": link.bandwidth_gbps(), "timing_mode": "full_prompt"},
        "workers": {"prefill": worker, "decode": worker},
    }}))
}

/// Router-side adapter over the exact canonical estimator used by the engine.
/// The owning Mocker boundary has no canonical FPM constructor of its own: an
/// external AIS provider is materialized as a process-local `AisCallback`, so
/// the host must install the real model here instead of leaving Polynomial or
/// choosing `PerfModel::Fixed`.
struct CanonicalAisCallback {
    timing: Arc<dyn TimingModel>,
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

/// Build the router launch config through its owning parser and require that it
/// carries the same canonical FPM config. The parser maps external AIS timing
/// to a callback slot; install the real canonical estimator there and fail if
/// the provider or the resulting slot is not what this proof requires.
fn canonical_router_config(
    launch: &aisimulate_core::engine::EngineLaunchConfig,
    timing: Arc<dyn TimingModel>,
) -> Result<MockerConfig> {
    let mut router = MockerConfig::from_value(serde_json::to_value(launch)?)
        .context("owner-normalized physical launch must be accepted by Dynamo")?;
    ensure!(
        router.engine == *launch,
        "Dynamo must preserve the complete owner-normalized launch identity"
    );
    ensure!(
        !matches!(router.perf_model.as_ref(), PerfModel::Fixed { .. }),
        "canonical timing must not be replaced with fixed timing"
    );
    router.perf_model = Arc::new(PerfModel::from_ais_callback(Arc::new(
        CanonicalAisCallback { timing },
    )));
    Ok(router)
}

#[test]
fn owner_launch_normalization_preserves_role_allocation_and_callback_identity() -> Result<()> {
    struct PlacementProbe;
    impl TimingModel for PlacementProbe {
        fn predict_prefill_ms(&self, batch: usize, tokens: usize, prefix: usize) -> Result<f64> {
            Ok((batch + tokens + prefix) as f64)
        }
        fn predict_decode_ms(
            &self,
            batch: usize,
            kv: usize,
            context: usize,
            total: usize,
        ) -> Result<f64> {
            Ok((batch + kv + context + total) as f64)
        }
    }
    for (stage, role_name) in [
        (WorkerStage::Prefill, "prefill"),
        (WorkerStage::Decode, "decode"),
    ] {
        let config: ReplayEngineConfig = serde_json::from_value(json!({
            "tensor_parallel_size": 2, "num_gpu_blocks_is_explicit": false,
            "rank": {"num_gpu_blocks": 37, "block_size": 8,
                "kv_transfer_bytes_per_token": 44928, "kv_transfer_bandwidth": 50.0,
                "timing_model": {"type": "external", "provider": "aic", "config": {
                    "model": DEFAULT_MODEL, "system": SYSTEM, "backend": "vllm",
                    "worker_type": role_name, "tp": 2, "pp": 1, "attention_dp": 1,
                    "gpu_memory_utilization": 0.83, "cuda_graph_reserved_bytes": 128,
                }}}
        }))?;
        let role = config.role(stage);
        let launch = role.launch_config(Some(0.25))?;
        assert_eq!(launch.engine.worker_type, role.rank.worker_type);
        assert_eq!(launch.gpu_memory_utilization, Some(0.83));
        assert_eq!(launch.cuda_graph_reserved_bytes, Some(128));
        assert!(!launch.num_gpu_blocks_is_explicit);
        assert_eq!(launch.num_gpu_blocks, 37);
        assert_eq!(launch.kv_transfer_bytes_per_token, Some(44928));
        assert_eq!(launch.tensor_parallel_size, 2);
        assert_eq!(launch.startup_time, Some(0.25));
        let timing: Arc<dyn TimingModel> = Arc::new(PlacementProbe);
        let router = canonical_router_config(&launch, timing.clone())?;
        let identity = router
            .ais_perf_config()
            .context("canonical timing identity")?;
        assert_eq!(identity["worker_type"], role_name);
        assert_eq!(identity["tp"], 2);
        assert!(identity.get("gpu_memory_utilization").is_none());
        assert_eq!(
            router.perf_model.predict_prefill_time(2, 16, 8)?,
            timing.predict_prefill_ms(2, 8, 8)?
        );
        assert_eq!(
            router.perf_model.predict_decode_time(2, 64, 32, 64)?,
            timing.predict_decode_ms(2, 64, 32, 64)?
        );
    }
    Ok(())
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
    let mut descriptor = setup.descriptor.clone();
    descriptor.rank = setup.rank.clone();
    descriptor
}

/// Disaggregated 2P2D spec with separate per-role geometry. Requests are
/// supplied by the hosted participant, not by the serializable spec.
pub(crate) fn spec(prefill: &RoleSetup, decode: &RoleSetup) -> Result<ReplaySpec> {
    ensure!(
        prefill.state_geometry["supported"] == true && decode.state_geometry["supported"] == true,
        "this calibrated control does not establish recurrent-state modeling"
    );
    let mut spec = prefill.physical_spec.clone();
    let mut engine: ReplayEngineConfig = serde_json::from_value(spec.engine)?;
    // Focused rejection controls may deliberately reduce the physical context
    // limit. All ordinary configuration was already lowered by its owner.
    engine.prefill = Some(role_descriptor(prefill));
    engine.decode = Some(role_descriptor(decode));
    spec.engine = serde_json::to_value(engine)?;
    spec.adapters.placement.provider = "dynamo_kv_router".into();
    spec.requests.clear();
    Ok(spec)
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
    let composition = KvReplayComposition::disaggregated(
        prefill.router.clone(),
        decode.router.clone(),
        PREFILL_WORKERS,
        DECODE_WORKERS,
        None,
        None,
        None,
    );
    let report =
        Replayer::with_composition(spec(prefill, decode)?, prefill.factory.clone(), composition)
            .context("real Dynamo KV composition accepts the disaggregated calibrated spec")?
            .with_runtime_input(ReplayRuntimeInput::External(Box::new(participant)))
            .with_capture_options(ReplayCaptureOptions {
                capture_per_request: true,
                determinism: ReplayDeterminism::CanonicalV1,
                ..Default::default()
            })
            .run()
            .context(
                "native Replayer event loop hosts calibrated AIPerf over the real KV composition",
            )?;
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
    let (prefill, decode) =
        small_control_roles().context("canonical AISim configuration and physical factory")?;
    let (prefill_small, prefill_large, decode_small, decode_large) =
        assert_not_fixed(&prefill, &decode).context("fixed-vs-calibrated causal control")?;

    // The control must actually price a physical handoff: a compiler-resolved
    // payload on both roles and one shared positive hardware link. Capacity's
    // native memory geometry is not relabelled as the transfer payload.
    ensure!(
        prefill.transfer.bytes_per_token > 0 && decode.transfer.bytes_per_token > 0,
        "compiler must resolve positive transfer bytes per token on every role"
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
            "max_model_len_note": "rank.max_model_len is resolved by the canonical AISim compiler; unbounded context is not claimed",
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
                "configured_topology": prefill.physical_spec.topology,
                "handoff_latency_use": "the owning configuration's fallback only; the hardware p2p latency above is not added to bandwidth-priced serialization",
                "payload_source": "owning compiler transfer estimate; distinct from native capacity geometry",
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
                    "capacity_diagnostics": prefill.geometry,
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
                    "capacity_diagnostics": decode.geometry,
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
