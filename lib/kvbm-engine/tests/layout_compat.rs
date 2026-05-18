// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cross-leader layout compatibility tests for the operational vs
//! universal block-layout modes.
//!
//! Built per the reproducer-first rule: each `universal_rejects_*` and
//! `operational_rejects_*` test must fail before `check_import_compat`
//! is implemented and pass after.
//!
//! Tests use synthetic `SerializedLayout`s constructed via
//! `SerializedLayout::pack` so they don't need a real NIXL agent or
//! GPU — the compat check inspects shape/metadata only.

use std::ops::Range;

use dynamo_memory::StorageKind;
use dynamo_memory::nixl::MemType;
use kvbm_common::LogicalLayoutHandle;
use kvbm_common::{BlockLayoutMode, KvBlockLayout, KvDim};
use kvbm_engine::leader::layout_compat::check_import_compat;
use kvbm_physical::layout::{
    BlockFormat, FullyContiguousDetails, LayoutConfig, LayoutDescriptor, LayoutTypeDetails,
    NixlMetadata,
};
use kvbm_physical::manager::{
    LayoutHandle, LogicalLayoutDescriptor, ParallelismDescriptor, SerializedLayout, WorkerAddress,
};

// ---------------------------------------------------------------------------
// Synthetic worker builders
// ---------------------------------------------------------------------------

/// Shape parameters that survive into the canonical descriptor. Tests use
/// this struct as a recipe and tweak one field at a time to drive
/// each rejection scenario.
#[derive(Clone, Copy)]
struct CanonicalRecipe {
    num_layers_total: usize,
    num_heads_total: usize,
    outer_dim: usize,
    page_size: usize,
    head_dim: usize,
    dtype_width_bytes: usize,
}

impl Default for CanonicalRecipe {
    fn default() -> Self {
        Self {
            num_layers_total: 32,
            num_heads_total: 64,
            outer_dim: 2,
            page_size: 16,
            head_dim: 64,
            dtype_width_bytes: 2,
        }
    }
}

/// Build one worker's `SerializedLayout` with the recipe sharded across
/// `tp_size` workers (heads divided evenly).
fn build_worker(
    recipe: CanonicalRecipe,
    kv_layout: KvBlockLayout,
    tp_size: usize,
    rank: usize,
    worker_id: u64,
) -> SerializedLayout {
    assert!(tp_size >= 1);
    assert!(rank < tp_size);
    assert!(
        recipe.num_heads_total.is_multiple_of(tp_size),
        "test recipe: num_heads_total={} must be divisible by tp_size={}",
        recipe.num_heads_total,
        tp_size,
    );

    let per_worker_num_heads = recipe.num_heads_total / tp_size;
    let inner_dim = per_worker_num_heads * recipe.head_dim;

    let cfg = LayoutConfig::builder()
        .num_blocks(4)
        .num_layers(recipe.num_layers_total)
        .outer_dim(recipe.outer_dim)
        .page_size(recipe.page_size)
        .inner_dim(inner_dim)
        .dtype_width_bytes(recipe.dtype_width_bytes)
        .num_heads(Some(per_worker_num_heads))
        .build()
        .unwrap();

    let layout_descriptor = LayoutDescriptor {
        version: LayoutDescriptor::CURRENT_VERSION,
        layout_config: cfg,
        location: StorageKind::System,
        nixl_metadata: NixlMetadata::new(format!("agent-{worker_id}"), MemType::Dram, 0),
        memory_descriptors: vec![],
        layout_type_details: LayoutTypeDetails::FullyContiguous(FullyContiguousDetails {
            block_format: BlockFormat::Operational,
            kv_block_layout: kv_layout,
        }),
    };

    let parallelism = ParallelismDescriptor {
        tp_size,
        pp_size: 1,
        rank,
        shard_axis: KvDim::HeadCount,
        global_extents: vec![
            (KvDim::Layer, recipe.num_layers_total),
            (KvDim::HeadCount, recipe.num_heads_total),
            (KvDim::HeadSize, recipe.head_dim),
            (KvDim::Page, recipe.page_size),
        ],
        layer_ownership: Range {
            start: 0,
            end: recipe.num_layers_total,
        },
    };

    let logical = LogicalLayoutDescriptor::new(
        LayoutHandle::new(worker_id, 0),
        LogicalLayoutHandle::G2,
        layout_descriptor,
    );

    SerializedLayout::pack(
        WorkerAddress::new(worker_id, format!("agent-{worker_id}")),
        Vec::new(),
        vec![logical],
        Some(parallelism),
    )
    .unwrap()
}

/// Build a TP=`tp_size` leader as a vec of per-worker `SerializedLayout`s.
fn build_leader(
    recipe: CanonicalRecipe,
    kv_layout: KvBlockLayout,
    tp_size: usize,
    worker_id_base: u64,
) -> Vec<SerializedLayout> {
    (0..tp_size)
        .map(|rank| {
            build_worker(
                recipe,
                kv_layout,
                tp_size,
                rank,
                worker_id_base + rank as u64,
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Operational mode
// ---------------------------------------------------------------------------

#[test]
fn operational_accepts_identical_layouts() {
    let recipe = CanonicalRecipe::default();
    let local = build_leader(recipe, KvBlockLayout::OperationalNHD, 4, 100);
    let remote = build_leader(recipe, KvBlockLayout::OperationalNHD, 4, 200);
    check_import_compat(BlockLayoutMode::Operational, &local, &remote, None)
        .expect("identical layouts must pass operational compat");
}

#[test]
fn operational_rejects_different_kv_block_layout() {
    let recipe = CanonicalRecipe::default();
    let local = build_leader(recipe, KvBlockLayout::OperationalNHD, 4, 100);
    let remote = build_leader(recipe, KvBlockLayout::OperationalHND, 4, 200);
    let err = check_import_compat(BlockLayoutMode::Operational, &local, &remote, None)
        .expect_err("different KvBlockLayout must fail operational compat");
    let s = format!("{err}");
    assert!(
        s.to_lowercase().contains("kvblocklayout") || s.contains("OperationalHND"),
        "expected layout-name in error, got: {s}"
    );
}

#[test]
fn operational_rejects_different_num_heads_per_worker() {
    let recipe = CanonicalRecipe::default();
    let local = build_leader(recipe, KvBlockLayout::OperationalNHD, 4, 100);
    // Different TP at remote (TP=8 ⇒ per-worker num_heads=8) → operational rejects.
    let remote = build_leader(recipe, KvBlockLayout::OperationalNHD, 8, 200);
    check_import_compat(BlockLayoutMode::Operational, &local, &remote, None)
        .expect_err("different per-worker num_heads must fail operational compat");
}

#[test]
fn operational_rejects_different_canonical_num_heads() {
    let local_recipe = CanonicalRecipe::default();
    let mut remote_recipe = local_recipe;
    remote_recipe.num_heads_total = 48; // matched per-worker only if tp differs
    let local = build_leader(local_recipe, KvBlockLayout::OperationalNHD, 4, 100);
    let remote = build_leader(remote_recipe, KvBlockLayout::OperationalNHD, 4, 200);
    check_import_compat(BlockLayoutMode::Operational, &local, &remote, None)
        .expect_err("different num_heads must fail operational compat");
}

// ---------------------------------------------------------------------------
// Universal mode
// ---------------------------------------------------------------------------

#[test]
fn universal_accepts_identical_layouts() {
    let recipe = CanonicalRecipe::default();
    let local = build_leader(recipe, KvBlockLayout::OperationalNHD, 4, 100);
    let remote = build_leader(recipe, KvBlockLayout::OperationalNHD, 4, 200);
    check_import_compat(BlockLayoutMode::Universal, &local, &remote, None)
        .expect("identical layouts must pass universal compat");
}

#[test]
fn universal_accepts_different_per_worker_permutation() {
    let recipe = CanonicalRecipe::default();
    let local = build_leader(recipe, KvBlockLayout::OperationalNHD, 4, 100);
    let remote = build_leader(recipe, KvBlockLayout::OperationalHND, 4, 200);
    check_import_compat(BlockLayoutMode::Universal, &local, &remote, None)
        .expect("universal must accept different per-worker permutation when canonicals match");
}

#[test]
fn universal_accepts_different_tp() {
    let recipe = CanonicalRecipe::default(); // num_heads_total=64
    let local = build_leader(recipe, KvBlockLayout::OperationalNHD, 4, 100); // 16 heads/worker
    let remote = build_leader(recipe, KvBlockLayout::OperationalNHD, 8, 200); // 8 heads/worker
    check_import_compat(BlockLayoutMode::Universal, &local, &remote, None)
        .expect("universal must accept different TP when canonical num_heads_total matches");
}

#[test]
fn universal_rejects_different_canonical_num_heads() {
    let local_recipe = CanonicalRecipe::default(); // num_heads_total=64
    let mut remote_recipe = local_recipe;
    remote_recipe.num_heads_total = 48; // canonical mismatch
    let local = build_leader(local_recipe, KvBlockLayout::OperationalNHD, 4, 100);
    let remote = build_leader(remote_recipe, KvBlockLayout::OperationalNHD, 4, 200);
    let err = check_import_compat(BlockLayoutMode::Universal, &local, &remote, None)
        .expect_err("different canonical num_heads must fail universal compat");
    let s = format!("{err}");
    assert!(
        s.contains("num_heads_total") && s.contains("local=64") && s.contains("remote=48"),
        "expected named field + values, got: {s}"
    );
}

#[test]
fn universal_rejects_different_head_dim() {
    let local_recipe = CanonicalRecipe::default();
    let mut remote_recipe = local_recipe;
    remote_recipe.head_dim = 128;
    let local = build_leader(local_recipe, KvBlockLayout::OperationalNHD, 4, 100);
    let remote = build_leader(remote_recipe, KvBlockLayout::OperationalNHD, 4, 200);
    let err = check_import_compat(BlockLayoutMode::Universal, &local, &remote, None)
        .expect_err("different canonical head_dim must fail universal compat");
    assert!(format!("{err}").contains("head_dim"));
}

#[test]
fn universal_rejects_different_dtype_width() {
    let local_recipe = CanonicalRecipe::default();
    let mut remote_recipe = local_recipe;
    remote_recipe.dtype_width_bytes = 4;
    let local = build_leader(local_recipe, KvBlockLayout::OperationalNHD, 4, 100);
    let remote = build_leader(remote_recipe, KvBlockLayout::OperationalNHD, 4, 200);
    let err = check_import_compat(BlockLayoutMode::Universal, &local, &remote, None)
        .expect_err("different canonical dtype_width must fail universal compat");
    assert!(format!("{err}").contains("dtype_width_bytes"));
}

#[test]
fn universal_rejects_different_page_size() {
    let local_recipe = CanonicalRecipe::default();
    let mut remote_recipe = local_recipe;
    remote_recipe.page_size = 32;
    let local = build_leader(local_recipe, KvBlockLayout::OperationalNHD, 4, 100);
    let remote = build_leader(remote_recipe, KvBlockLayout::OperationalNHD, 4, 200);
    let err = check_import_compat(BlockLayoutMode::Universal, &local, &remote, None)
        .expect_err("different canonical page_size must fail universal compat");
    assert!(format!("{err}").contains("page_size"));
}

#[test]
fn universal_rejects_unknown_local() {
    let recipe = CanonicalRecipe::default();
    let local = build_leader(recipe, KvBlockLayout::Unknown, 4, 100);
    let remote = build_leader(recipe, KvBlockLayout::OperationalNHD, 4, 200);
    check_import_compat(BlockLayoutMode::Universal, &local, &remote, None)
        .expect_err("local Unknown layout must fail universal compat");
}

#[test]
fn universal_rejects_unknown_remote() {
    let recipe = CanonicalRecipe::default();
    let local = build_leader(recipe, KvBlockLayout::OperationalNHD, 4, 100);
    let remote = build_leader(recipe, KvBlockLayout::Unknown, 4, 200);
    check_import_compat(BlockLayoutMode::Universal, &local, &remote, None)
        .expect_err("remote Unknown layout must fail universal compat");
}

// ---------------------------------------------------------------------------
// Empty / edge
// ---------------------------------------------------------------------------

#[test]
fn empty_sides_short_circuit() {
    let res = check_import_compat(BlockLayoutMode::Operational, &[], &[], None);
    assert!(res.is_ok(), "empty local + empty remote is a no-op");
}

// ---------------------------------------------------------------------------
// build_layout_compat_payload_with_template — template-priority contract
// ---------------------------------------------------------------------------
//
// The connector's hub-register path holds raw (unstamped) worker
// metadata + a freshly-built `ParallelismTemplate`. These tests pin the
// contract that the template wins over missing `ParallelismDescriptor`
// in universal mode, and that the no-template path still rejects.

/// Build a worker whose `SerializedLayout` carries no
/// `ParallelismDescriptor` (legacy / pre-stamp shape).
fn build_unstamped_worker(
    recipe: CanonicalRecipe,
    kv_layout: KvBlockLayout,
    worker_id: u64,
) -> SerializedLayout {
    let cfg = LayoutConfig::builder()
        .num_blocks(4)
        .num_layers(recipe.num_layers_total)
        .outer_dim(recipe.outer_dim)
        .page_size(recipe.page_size)
        .inner_dim(recipe.num_heads_total * recipe.head_dim)
        .dtype_width_bytes(recipe.dtype_width_bytes)
        .num_heads(Some(recipe.num_heads_total))
        .build()
        .unwrap();

    let layout_descriptor = LayoutDescriptor {
        version: LayoutDescriptor::CURRENT_VERSION,
        layout_config: cfg,
        location: StorageKind::System,
        nixl_metadata: NixlMetadata::new(format!("agent-{worker_id}"), MemType::Dram, 0),
        memory_descriptors: vec![],
        layout_type_details: LayoutTypeDetails::FullyContiguous(FullyContiguousDetails {
            block_format: BlockFormat::Operational,
            kv_block_layout: kv_layout,
        }),
    };

    let logical = LogicalLayoutDescriptor::new(
        LayoutHandle::new(worker_id, 0),
        LogicalLayoutHandle::G2,
        layout_descriptor,
    );

    // The `None` here is the point — no ParallelismDescriptor.
    SerializedLayout::pack(
        WorkerAddress::new(worker_id, format!("agent-{worker_id}")),
        Vec::new(),
        vec![logical],
        None,
    )
    .unwrap()
}

/// Build a stamped worker that exports both G1 (operational) and G2
/// (universal) — the post-c3 connector layout. Used to assert the
/// payload picks G2's KvBlockLayout (the transfer-canonical tier),
/// not G1's.
fn build_g1_g2_worker(
    recipe: CanonicalRecipe,
    g1_kv_layout: KvBlockLayout,
    g2_kv_layout: KvBlockLayout,
    worker_id: u64,
) -> SerializedLayout {
    let cfg = LayoutConfig::builder()
        .num_blocks(4)
        .num_layers(recipe.num_layers_total)
        .outer_dim(recipe.outer_dim)
        .page_size(recipe.page_size)
        .inner_dim(recipe.num_heads_total * recipe.head_dim)
        .dtype_width_bytes(recipe.dtype_width_bytes)
        .num_heads(Some(recipe.num_heads_total))
        .build()
        .unwrap();

    let make_descriptor = |kv: KvBlockLayout| LayoutDescriptor {
        version: LayoutDescriptor::CURRENT_VERSION,
        layout_config: cfg.clone(),
        location: StorageKind::System,
        nixl_metadata: NixlMetadata::new(format!("agent-{worker_id}"), MemType::Dram, 0),
        memory_descriptors: vec![],
        layout_type_details: LayoutTypeDetails::FullyContiguous(FullyContiguousDetails {
            block_format: BlockFormat::Operational,
            kv_block_layout: kv,
        }),
    };

    let g1 = LogicalLayoutDescriptor::new(
        LayoutHandle::new(worker_id, 0),
        LogicalLayoutHandle::G1,
        make_descriptor(g1_kv_layout),
    );
    let g2 = LogicalLayoutDescriptor::new(
        LayoutHandle::new(worker_id, 1),
        LogicalLayoutHandle::G2,
        make_descriptor(g2_kv_layout),
    );

    SerializedLayout::pack(
        WorkerAddress::new(worker_id, format!("agent-{worker_id}")),
        Vec::new(),
        vec![g1, g2],
        Some(ParallelismDescriptor {
            tp_size: 1,
            pp_size: 1,
            rank: 0,
            shard_axis: KvDim::HeadCount,
            global_extents: vec![
                (KvDim::Layer, recipe.num_layers_total),
                (KvDim::HeadCount, recipe.num_heads_total),
            ],
            layer_ownership: 0..recipe.num_layers_total,
        }),
    )
    .unwrap()
}

/// c3 reproducer: when the connector emits both G1 (operational) and
/// G2 (universal) layouts, the layout_compat payload must report G2's
/// layout (the transfer-canonical tier), not G1's. Pre-c3 the builder
/// picks `first_layout` which is G1 (registered first), so the
/// per_worker_layout falsely shows OperationalNHD instead of Universal.
#[test]
fn layout_compat_payload_reports_g2_layout_in_universal_mode() {
    let recipe = CanonicalRecipe::default();
    let worker = build_g1_g2_worker(
        recipe,
        KvBlockLayout::OperationalNHD,
        KvBlockLayout::Universal,
        7,
    );

    let payload = kvbm_engine::leader::layout_compat::build_layout_compat_payload(
        BlockLayoutMode::Universal,
        &worker,
    )
    .expect("worker carries both G1 and G2; universal payload should succeed");

    assert_eq!(
        payload.per_worker_layout,
        KvBlockLayout::Universal,
        "Universal-mode payload must report G2's layout (transfer-canonical), \
         not G1's; got {:?}",
        payload.per_worker_layout
    );
}

/// c3 stop-time review (Codex): the transfer-canonical helper picks G2
/// for *wire reporting*, but the Universal-mode label gate must walk
/// every tier — otherwise a vLLM probe failure that leaves G1 as
/// `KvBlockLayout::Unknown` slips through because G2 is always
/// `Universal` by construction in Universal mode. The fused permute
/// kernel needs to know G1's actual axis order (NHD vs HND); an
/// Unknown G1 would crash or produce garbage at transfer time.
#[test]
fn universal_check_rejects_unknown_g1_even_when_g2_universal() {
    let recipe = CanonicalRecipe::default();
    let local = build_g1_g2_worker(
        recipe,
        KvBlockLayout::OperationalNHD,
        KvBlockLayout::Universal,
        100,
    );
    let remote = build_g1_g2_worker(
        recipe,
        KvBlockLayout::Unknown, // ← the smuggled-through bug
        KvBlockLayout::Universal,
        200,
    );

    let err = check_import_compat(
        BlockLayoutMode::Universal,
        std::slice::from_ref(&local),
        std::slice::from_ref(&remote),
        None,
    )
    .expect_err(
        "Universal mode must reject a peer whose G1 is Unknown even when \
         G2 is Universal — the permute kernel needs G1's labeled axis order",
    );
    let msg = format!("{err}").to_lowercase();
    assert!(
        msg.contains("unknown") || msg.contains("labeled"),
        "rejection should name the Unknown layout; got: {err}",
    );
}

#[test]
fn template_satisfies_universal_when_metadata_unstamped() {
    let recipe = CanonicalRecipe::default();
    let raw = build_unstamped_worker(recipe, KvBlockLayout::Universal, 42);

    let layout_cfg = LayoutConfig::builder()
        .num_blocks(4)
        .num_layers(recipe.num_layers_total)
        .outer_dim(recipe.outer_dim)
        .page_size(recipe.page_size)
        .inner_dim(recipe.num_heads_total * recipe.head_dim)
        .dtype_width_bytes(recipe.dtype_width_bytes)
        .num_heads(Some(recipe.num_heads_total))
        .build()
        .unwrap();
    let template = kvbm_engine::leader::parallelism::ParallelismTemplate::from_layout_config(
        &layout_cfg,
        kvbm_config::ParallelismMode::TensorParallel,
        1,
    )
    .unwrap();

    let payload = kvbm_engine::leader::layout_compat::build_layout_compat_payload_with_template(
        BlockLayoutMode::Universal,
        &raw,
        Some(&template),
    )
    .expect("template should satisfy universal mode despite unstamped metadata");

    let expected_canonical = template
        .canonical_block_shape()
        .expect("template recipe must derive a canonical shape");
    assert_eq!(
        payload.canonical,
        Some(expected_canonical),
        "template canonical must win when metadata is unstamped",
    );
    assert_eq!(payload.tp_size, template.tp_size);
    assert_eq!(payload.pp_size, template.pp_size);
    assert_eq!(payload.mode, BlockLayoutMode::Universal);
}

#[test]
fn no_template_universal_rejects_unstamped_metadata() {
    let recipe = CanonicalRecipe::default();
    let raw = build_unstamped_worker(recipe, KvBlockLayout::Universal, 42);

    let err = kvbm_engine::leader::layout_compat::build_layout_compat_payload_with_template(
        BlockLayoutMode::Universal,
        &raw,
        None,
    )
    .expect_err("universal mode must reject unstamped metadata when no template is supplied");
    let m = format!("{err}");
    assert!(
        m.contains("ParallelismDescriptor") && m.contains("ParallelismTemplate"),
        "error must explicitly name both fallbacks (got: {m})",
    );
}

#[test]
fn no_template_operational_tolerates_unstamped_metadata() {
    // The operational defence-in-depth path runs without a template and
    // must degrade gracefully: payload built with `canonical = None`,
    // `tp_size = 1`, `pp_size = 1`.
    let recipe = CanonicalRecipe::default();
    let raw = build_unstamped_worker(recipe, KvBlockLayout::OperationalNHD, 42);

    let payload = kvbm_engine::leader::layout_compat::build_layout_compat_payload_with_template(
        BlockLayoutMode::Operational,
        &raw,
        None,
    )
    .expect("operational mode must tolerate unstamped metadata without a template");
    assert!(payload.canonical.is_none());
    assert_eq!(payload.tp_size, 1);
    assert_eq!(payload.pp_size, 1);
}
