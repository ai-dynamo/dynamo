// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Phantom-stream format.
//!
//! A stream directory holds `manifest.json` plus one binary file per *base* worker. A base is
//! one worker timeline captured offline (the phase-1 AgentX harness): its raw engine KV events,
//! split into an untimed warm-up prefix and a timed section, plus the lookups (request block
//! hashes) of the timed section. Events are grouped into *lists*, one per engine publish call
//! (one scheduler pass), which is the unit the production KV publisher batches.
//!
//! Binary layout (little endian): magic `DPHSTRM1`, then three sections (warm-up events, timed
//! events, queries), each prefixed by its byte length so readers can skip it. Every array is a
//! `u64` element count followed by the elements.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::ops::Range;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

pub const SCHEMA: &str = "dynamo-e2e.phantom-streams.v1";
pub const MANIFEST_FILE: &str = "manifest.json";
const MAGIC: &[u8; 8] = b"DPHSTRM1";

pub const KIND_STORE_ROOT: u8 = 0;
pub const KIND_STORE_CHILD: u8 = 1;
pub const KIND_REMOVE: u8 = 2;
pub const KIND_CLEAR: u8 = 3;

/// Events grouped into lists, stored as flat arrays.
///
/// A store's hashes are `(block_hash, tokens_hash)` pairs; a remove's are block hashes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EventLists {
    pub list_ts_us: Vec<u64>,
    /// Exclusive end index into the event arrays, per list.
    pub list_end: Vec<u64>,
    pub kind: Vec<u8>,
    /// Parent sequence hash of a child store; 0 otherwise.
    pub parent: Vec<u64>,
    /// Exclusive end index into `hashes`, per event.
    pub hash_end: Vec<u64>,
    pub hashes: Vec<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventRef<'a> {
    Store {
        parent: Option<u64>,
        /// `(block_hash, tokens_hash)` pairs, flattened.
        pairs: &'a [u64],
    },
    Remove {
        blocks: &'a [u64],
    },
    Clear,
}

impl EventLists {
    pub fn lists(&self) -> usize {
        self.list_end.len()
    }

    pub fn events(&self) -> usize {
        self.kind.len()
    }

    pub fn list_events(&self, list: usize) -> Range<usize> {
        let start = if list == 0 {
            0
        } else {
            self.list_end[list - 1] as usize
        };
        start..self.list_end[list] as usize
    }

    pub fn event(&self, index: usize) -> EventRef<'_> {
        let start = if index == 0 {
            0
        } else {
            self.hash_end[index - 1] as usize
        };
        let hashes = &self.hashes[start..self.hash_end[index] as usize];
        match self.kind[index] {
            KIND_STORE_ROOT => EventRef::Store {
                parent: None,
                pairs: hashes,
            },
            KIND_STORE_CHILD => EventRef::Store {
                parent: Some(self.parent[index]),
                pairs: hashes,
            },
            KIND_REMOVE => EventRef::Remove { blocks: hashes },
            _ => EventRef::Clear,
        }
    }

    /// Blocks an event writes (stored or removed).
    pub fn event_blocks(&self, index: usize) -> u64 {
        match self.event(index) {
            EventRef::Store { pairs, .. } => pairs.len() as u64 / 2,
            EventRef::Remove { blocks } => blocks.len() as u64,
            EventRef::Clear => 0,
        }
    }

    pub fn push_store(&mut self, parent: Option<u64>, pairs: impl IntoIterator<Item = (u64, u64)>) {
        self.kind.push(match parent {
            None => KIND_STORE_ROOT,
            Some(_) => KIND_STORE_CHILD,
        });
        self.parent.push(parent.unwrap_or(0));
        for (block_hash, tokens_hash) in pairs {
            self.hashes.push(block_hash);
            self.hashes.push(tokens_hash);
        }
        self.hash_end.push(self.hashes.len() as u64);
    }

    pub fn push_remove(&mut self, blocks: impl IntoIterator<Item = u64>) {
        self.kind.push(KIND_REMOVE);
        self.parent.push(0);
        self.hashes.extend(blocks);
        self.hash_end.push(self.hashes.len() as u64);
    }

    pub fn push_clear(&mut self) {
        self.kind.push(KIND_CLEAR);
        self.parent.push(0);
        self.hash_end.push(self.hashes.len() as u64);
    }

    /// Close the list of events pushed since the previous close; empty lists are dropped.
    pub fn close_list(&mut self, ts_us: u64) {
        let start = self.list_end.last().copied().unwrap_or(0);
        if self.kind.len() as u64 == start {
            return;
        }
        self.list_end.push(self.kind.len() as u64);
        self.list_ts_us.push(ts_us);
    }

    pub fn totals(&self) -> SectionTotals {
        let mut totals = SectionTotals {
            lists: self.lists() as u64,
            events: self.events() as u64,
            ..SectionTotals::default()
        };
        for index in 0..self.events() {
            match self.event(index) {
                EventRef::Store { pairs, .. } => totals.stored_blocks += pairs.len() as u64 / 2,
                EventRef::Remove { blocks } => totals.removed_blocks += blocks.len() as u64,
                EventRef::Clear => totals.cleared += 1,
            }
        }
        totals
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.list_ts_us.len() == self.list_end.len(),
            "list arrays disagree"
        );
        let events = self.kind.len();
        ensure!(
            self.parent.len() == events && self.hash_end.len() == events,
            "event arrays disagree"
        );
        ensure!(
            self.list_end.windows(2).all(|pair| pair[0] < pair[1])
                && self.list_end.first().is_none_or(|&end| end > 0)
                && self.list_end.last().is_none_or(|&end| end == events as u64),
            "list ends are not strictly increasing over all events"
        );
        ensure!(
            self.list_ts_us.windows(2).all(|pair| pair[0] <= pair[1]),
            "list timestamps decrease"
        );
        let mut start = 0u64;
        for (index, (&end, &kind)) in self.hash_end.iter().zip(&self.kind).enumerate() {
            ensure!(
                end >= start && end <= self.hashes.len() as u64,
                "event {index} hash range is invalid"
            );
            match kind {
                KIND_STORE_ROOT | KIND_STORE_CHILD => {
                    ensure!(
                        (end - start).is_multiple_of(2) && end > start,
                        "store {index} is malformed"
                    )
                }
                KIND_REMOVE => ensure!(end > start, "remove {index} is empty"),
                KIND_CLEAR => ensure!(end == start, "clear {index} has hashes"),
                other => bail!("event {index} has unknown kind {other}"),
            }
            start = end;
        }
        ensure!(start == self.hashes.len() as u64, "trailing hashes");
        Ok(())
    }

    fn byte_len(&self) -> u64 {
        array_bytes(self.list_ts_us.len(), 8)
            + array_bytes(self.list_end.len(), 8)
            + array_bytes(self.kind.len(), 1)
            + array_bytes(self.parent.len(), 8)
            + array_bytes(self.hash_end.len(), 8)
            + array_bytes(self.hashes.len(), 8)
    }

    fn write(&self, out: &mut impl Write) -> Result<()> {
        out.write_all(&self.byte_len().to_le_bytes())?;
        write_u64s(out, &self.list_ts_us)?;
        write_u64s(out, &self.list_end)?;
        write_u8s(out, &self.kind)?;
        write_u64s(out, &self.parent)?;
        write_u64s(out, &self.hash_end)?;
        write_u64s(out, &self.hashes)
    }

    fn read(input: &mut impl Read) -> Result<Self> {
        let _byte_len = read_u64(input)?;
        let lists = Self {
            list_ts_us: read_u64s(input)?,
            list_end: read_u64s(input)?,
            kind: read_u8s(input)?,
            parent: read_u64s(input)?,
            hash_end: read_u64s(input)?,
            hashes: read_u64s(input)?,
        };
        lists.validate()?;
        Ok(lists)
    }
}

/// Timed lookups: one block-hash sequence per request.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Queries {
    pub ts_us: Vec<u64>,
    pub end: Vec<u64>,
    pub hashes: Vec<u64>,
}

impl Queries {
    pub fn len(&self) -> usize {
        self.ts_us.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ts_us.is_empty()
    }

    pub fn push(&mut self, ts_us: u64, hashes: impl IntoIterator<Item = u64>) {
        self.hashes.extend(hashes);
        self.end.push(self.hashes.len() as u64);
        self.ts_us.push(ts_us);
    }

    pub fn get(&self, index: usize) -> (u64, &[u64]) {
        let start = if index == 0 {
            0
        } else {
            self.end[index - 1] as usize
        };
        (
            self.ts_us[index],
            &self.hashes[start..self.end[index] as usize],
        )
    }

    fn validate(&self) -> Result<()> {
        ensure!(self.ts_us.len() == self.end.len(), "query arrays disagree");
        ensure!(
            self.end.windows(2).all(|pair| pair[0] <= pair[1])
                && self
                    .end
                    .last()
                    .is_none_or(|&end| end == self.hashes.len() as u64),
            "query ranges are invalid"
        );
        ensure!(
            self.ts_us.windows(2).all(|pair| pair[0] <= pair[1]),
            "query timestamps decrease"
        );
        Ok(())
    }

    fn byte_len(&self) -> u64 {
        array_bytes(self.ts_us.len(), 8)
            + array_bytes(self.end.len(), 8)
            + array_bytes(self.hashes.len(), 8)
    }

    fn write(&self, out: &mut impl Write) -> Result<()> {
        out.write_all(&self.byte_len().to_le_bytes())?;
        write_u64s(out, &self.ts_us)?;
        write_u64s(out, &self.end)?;
        write_u64s(out, &self.hashes)
    }

    fn read(input: &mut impl Read) -> Result<Self> {
        let _byte_len = read_u64(input)?;
        let queries = Self {
            ts_us: read_u64s(input)?,
            end: read_u64s(input)?,
            hashes: read_u64s(input)?,
        };
        queries.validate()?;
        Ok(queries)
    }
}

/// One base worker's streams.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BaseStream {
    /// Untimed warm-up prefix (list timestamps are 0).
    pub warmup: EventLists,
    pub timed: EventLists,
    pub queries: Queries,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SectionTotals {
    pub lists: u64,
    pub events: u64,
    pub stored_blocks: u64,
    pub removed_blocks: u64,
    pub cleared: u64,
}

impl SectionTotals {
    pub fn write_blocks(&self) -> u64 {
        self.stored_blocks + self.removed_blocks
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BaseInfo {
    pub file: String,
    pub warmup: SectionTotals,
    pub timed: SectionTotals,
    pub queries: u64,
    pub query_blocks: u64,
    pub first_ts_us: Option<u64>,
    pub last_ts_us: Option<u64>,
}

impl BaseInfo {
    pub fn describe(file: String, stream: &BaseStream) -> Self {
        let first = [
            stream.timed.list_ts_us.first(),
            stream.queries.ts_us.first(),
        ]
        .into_iter()
        .flatten()
        .min()
        .copied();
        let last = [stream.timed.list_ts_us.last(), stream.queries.ts_us.last()]
            .into_iter()
            .flatten()
            .max()
            .copied();
        Self {
            file,
            warmup: stream.warmup.totals(),
            timed: stream.timed.totals(),
            queries: stream.queries.len() as u64,
            query_blocks: stream.queries.hashes.len() as u64,
            first_ts_us: first,
            last_ts_us: last,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub schema: String,
    pub block_size: u32,
    /// Timed-section origin and end (virtual microseconds) over every base.
    pub t0_us: u64,
    pub t1_us: u64,
    pub bases: Vec<BaseInfo>,
    /// Free-form provenance (capture configuration, corpus digests).
    pub provenance: serde_json::Value,
}

impl Manifest {
    pub fn new(block_size: u32, bases: Vec<BaseInfo>, provenance: serde_json::Value) -> Self {
        let t0_us = bases
            .iter()
            .filter_map(|base| base.first_ts_us)
            .min()
            .unwrap_or(0);
        let t1_us = bases
            .iter()
            .filter_map(|base| base.last_ts_us)
            .max()
            .unwrap_or(t0_us);
        Self {
            schema: SCHEMA.to_string(),
            block_size,
            t0_us,
            t1_us,
            bases,
            provenance,
        }
    }

    pub fn span_us(&self) -> u64 {
        self.t1_us.saturating_sub(self.t0_us).max(1)
    }

    pub fn read(dir: &Path) -> Result<Self> {
        let path = dir.join(MANIFEST_FILE);
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let manifest: Self =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        ensure!(
            manifest.schema == SCHEMA,
            "{} has schema {}, expected {SCHEMA}",
            path.display(),
            manifest.schema
        );
        ensure!(
            !manifest.bases.is_empty(),
            "{} lists no bases",
            path.display()
        );
        Ok(manifest)
    }

    pub fn write(&self, dir: &Path) -> Result<()> {
        let path = dir.join(MANIFEST_FILE);
        std::fs::write(&path, serde_json::to_string_pretty(self)?)
            .with_context(|| format!("writing {}", path.display()))
    }
}

/// Default floor for removed/stored blocks over a timed window: below it the capture's KV was
/// still filling, so the stream under-represents eviction (removes never reached steady state).
pub const DEFAULT_MIN_TIMED_REMOVE_RATIO: f64 = 0.8;

/// One base's warm-up and timed write mix.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EvictionRow {
    pub base: usize,
    pub warmup_stored_blocks: u64,
    pub warmup_removed_blocks: u64,
    /// Blocks still resident after the warm-up (stored minus removed) over the capture's KV
    /// capacity, when the manifest records it. Near 1 means eviction started before the timed
    /// section.
    pub warmup_resident_fraction: Option<f64>,
    pub timed_stored_blocks: u64,
    pub timed_removed_blocks: u64,
    pub timed_remove_ratio: f64,
}

/// Per-stream eviction over the warm-up and timed sections, and the streams whose timed section
/// removes fewer than `min_timed_remove_ratio` of the blocks it stores.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EvictionReport {
    pub min_timed_remove_ratio: f64,
    /// Per-worker KV capacity in blocks of the capture (`provenance.key.num_gpu_blocks`).
    pub capacity_blocks: Option<u64>,
    pub timed_remove_ratio: f64,
    pub bases_below_min: Vec<usize>,
    pub rows: Vec<EvictionRow>,
}

impl EvictionReport {
    pub fn steady(&self) -> bool {
        self.bases_below_min.is_empty()
    }
}

fn ratio(removed: u64, stored: u64) -> f64 {
    if stored == 0 {
        return 0.0;
    }
    removed as f64 / stored as f64
}

pub fn eviction_report(manifest: &Manifest, min_timed_remove_ratio: f64) -> EvictionReport {
    let capacity_blocks = manifest
        .provenance
        .pointer("/key/num_gpu_blocks")
        .and_then(serde_json::Value::as_u64)
        .filter(|&blocks| blocks > 0);
    let rows: Vec<EvictionRow> = manifest
        .bases
        .iter()
        .enumerate()
        .map(|(base, info)| EvictionRow {
            base,
            warmup_stored_blocks: info.warmup.stored_blocks,
            warmup_removed_blocks: info.warmup.removed_blocks,
            warmup_resident_fraction: capacity_blocks.map(|capacity| {
                info.warmup
                    .stored_blocks
                    .saturating_sub(info.warmup.removed_blocks) as f64
                    / capacity as f64
            }),
            timed_stored_blocks: info.timed.stored_blocks,
            timed_removed_blocks: info.timed.removed_blocks,
            timed_remove_ratio: ratio(info.timed.removed_blocks, info.timed.stored_blocks),
        })
        .collect();
    let bases_below_min = rows
        .iter()
        .filter(|row| row.timed_remove_ratio < min_timed_remove_ratio)
        .map(|row| row.base)
        .collect();
    let (removed, stored) = rows.iter().fold((0, 0), |(removed, stored), row| {
        (
            removed + row.timed_removed_blocks,
            stored + row.timed_stored_blocks,
        )
    });
    EvictionReport {
        min_timed_remove_ratio,
        capacity_blocks,
        timed_remove_ratio: ratio(removed, stored),
        bases_below_min,
        rows,
    }
}

pub fn base_file_name(base: usize) -> String {
    format!("base-{base:05}.bin")
}

/// Which sections [`read_base`] decodes; skipped sections stay empty.
#[derive(Debug, Clone, Copy)]
pub struct Sections {
    pub events: bool,
    pub queries: bool,
}

pub fn write_base(path: &Path, stream: &BaseStream) -> Result<()> {
    let tmp = path.with_extension("tmp");
    let file = File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
    let mut out = BufWriter::with_capacity(1 << 20, file);
    out.write_all(MAGIC)?;
    stream.warmup.write(&mut out)?;
    stream.timed.write(&mut out)?;
    stream.queries.write(&mut out)?;
    out.into_inner()
        .map_err(|error| error.into_error())?
        .sync_all()?;
    std::fs::rename(&tmp, path).with_context(|| format!("renaming to {}", path.display()))
}

pub fn read_base(path: &Path, sections: Sections) -> Result<BaseStream> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut input = BufReader::with_capacity(1 << 20, file);
    let mut magic = [0u8; 8];
    input.read_exact(&mut magic)?;
    ensure!(
        &magic == MAGIC,
        "{} is not a phantom stream",
        path.display()
    );
    let mut stream = BaseStream::default();
    for target in [0, 1] {
        if !sections.events {
            let len = read_u64(&mut input)?;
            input.seek(SeekFrom::Current(len as i64))?;
            continue;
        }
        let lists =
            EventLists::read(&mut input).with_context(|| format!("reading {}", path.display()))?;
        if target == 0 {
            stream.warmup = lists;
        } else {
            stream.timed = lists;
        }
    }
    if sections.queries {
        stream.queries =
            Queries::read(&mut input).with_context(|| format!("reading {}", path.display()))?;
    }
    Ok(stream)
}

fn array_bytes(len: usize, element: u64) -> u64 {
    8 + len as u64 * element
}

fn write_u64s(out: &mut impl Write, values: &[u64]) -> Result<()> {
    out.write_all(&(values.len() as u64).to_le_bytes())?;
    for value in values {
        out.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn write_u8s(out: &mut impl Write, values: &[u8]) -> Result<()> {
    out.write_all(&(values.len() as u64).to_le_bytes())?;
    out.write_all(values)?;
    Ok(())
}

fn read_u64(input: &mut impl Read) -> Result<u64> {
    let mut bytes = [0u8; 8];
    input.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_u64s(input: &mut impl Read) -> Result<Vec<u64>> {
    let len = usize::try_from(read_u64(input)?)?;
    let mut values = Vec::with_capacity(len);
    let mut buffer = vec![0u8; 8 * 8192];
    let mut remaining = len;
    while remaining > 0 {
        let chunk = remaining.min(8192);
        let bytes = &mut buffer[..chunk * 8];
        input.read_exact(bytes)?;
        values.extend(
            bytes
                .chunks_exact(8)
                .map(|word| u64::from_le_bytes(word.try_into().expect("8-byte chunk"))),
        );
        remaining -= chunk;
    }
    Ok(values)
}

fn read_u8s(input: &mut impl Read) -> Result<Vec<u8>> {
    let len = usize::try_from(read_u64(input)?)?;
    let mut values = vec![0u8; len];
    input.read_exact(&mut values)?;
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> BaseStream {
        let mut stream = BaseStream::default();
        stream.warmup.push_store(None, [(1, 1), (2, 2)]);
        stream.warmup.close_list(0);
        stream.timed.push_store(Some(2), [(3, 3)]);
        stream.timed.push_remove([1]);
        stream.timed.close_list(10);
        stream.timed.close_list(11);
        stream.timed.push_clear();
        stream.timed.close_list(20);
        stream.queries.push(9, [1, 2, 3]);
        stream.queries.push(19, []);
        stream
    }

    #[test]
    fn round_trips_and_skips_sections() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(base_file_name(3));
        let stream = sample();
        write_base(&path, &stream).unwrap();

        let all = read_base(
            &path,
            Sections {
                events: true,
                queries: true,
            },
        )
        .unwrap();
        assert_eq!(all, stream);
        assert_eq!(all.timed.lists(), 2, "the empty list was dropped");
        assert_eq!(
            all.timed.event(0),
            EventRef::Store {
                parent: Some(2),
                pairs: &[3, 3]
            }
        );
        assert_eq!(all.timed.event(2), EventRef::Clear);

        let queries_only = read_base(
            &path,
            Sections {
                events: false,
                queries: true,
            },
        )
        .unwrap();
        assert_eq!(queries_only.queries, stream.queries);
        assert_eq!(queries_only.timed, EventLists::default());

        let info = BaseInfo::describe(base_file_name(3), &stream);
        assert_eq!(info.timed.stored_blocks, 1);
        assert_eq!(info.timed.removed_blocks, 1);
        assert_eq!(info.timed.cleared, 1);
        assert_eq!((info.first_ts_us, info.last_ts_us), (Some(9), Some(20)));
    }

    #[test]
    fn eviction_report_flags_streams_still_filling() {
        let base = |warmup: (u64, u64), timed: (u64, u64)| BaseInfo {
            warmup: SectionTotals {
                stored_blocks: warmup.0,
                removed_blocks: warmup.1,
                ..SectionTotals::default()
            },
            timed: SectionTotals {
                stored_blocks: timed.0,
                removed_blocks: timed.1,
                ..SectionTotals::default()
            },
            ..BaseInfo::default()
        };
        let manifest = Manifest::new(
            1,
            vec![base((1200, 200), (100, 95)), base((300, 0), (100, 10))],
            serde_json::json!({ "key": { "num_gpu_blocks": 1000 } }),
        );
        let report = eviction_report(&manifest, DEFAULT_MIN_TIMED_REMOVE_RATIO);
        assert_eq!(report.capacity_blocks, Some(1000));
        assert_eq!(report.bases_below_min, vec![1]);
        assert!(!report.steady());
        assert_eq!(report.rows[0].warmup_resident_fraction, Some(1.0));
        assert_eq!(report.rows[1].warmup_resident_fraction, Some(0.3));
        assert!((report.timed_remove_ratio - 0.525).abs() < 1e-12);
    }

    #[test]
    fn rejects_corrupt_ranges() {
        let mut stream = sample();
        stream.timed.list_end[0] = 5;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.bin");
        write_base(&path, &stream).unwrap();
        assert!(
            read_base(
                &path,
                Sections {
                    events: true,
                    queries: false
                }
            )
            .is_err()
        );
    }
}
