// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Store fingerprints in page-sized copy-on-write chunks so clones stay cheap
//! and sparse edits only duplicate the pages they touch.

use std::sync::Arc;

/// Use 64 KiB pages so buckets never cross a page boundary and sparse edits
/// copy far less than a whole filter.
const PAGE_SLOTS: usize = 32_768;

#[derive(Clone)]
struct Page {
    slots: [u16; PAGE_SLOTS],
}

#[derive(Clone)]
pub(super) struct BucketPages {
    pages: Vec<Arc<Page>>,
    len: usize,
}

impl BucketPages {
    pub(super) fn zeroed(len: usize) -> Self {
        let pages = (0..len.div_ceil(PAGE_SLOTS))
            .map(|_| {
                Arc::new(Page {
                    slots: [0; PAGE_SLOTS],
                })
            })
            .collect();
        Self { pages, len }
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub(super) fn get(&self, index: usize) -> u16 {
        self.pages[index / PAGE_SLOTS].slots[index % PAGE_SLOTS]
    }

    #[inline]
    pub(super) fn set(&mut self, index: usize, value: u16) {
        let page = Arc::make_mut(&mut self.pages[index / PAGE_SLOTS]);
        page.slots[index % PAGE_SLOTS] = value;
    }

    pub(super) fn read_array<const N: usize>(&self, start: usize) -> [u16; N] {
        debug_assert!(start.is_multiple_of(N));
        let page_offset = start % PAGE_SLOTS;
        self.pages[start / PAGE_SLOTS].slots[page_offset..page_offset + N]
            .try_into()
            .expect("bucket stays within one page")
    }

    #[inline]
    pub(super) fn contains_in_array<const N: usize>(&self, start: usize, value: u16) -> bool {
        let page_offset = start % PAGE_SLOTS;
        self.pages[start / PAGE_SLOTS].slots[page_offset..page_offset + N].contains(&value)
    }

    pub(super) fn write_array<const N: usize>(&mut self, start: usize, values: &[u16; N]) {
        debug_assert!(start.is_multiple_of(N));
        let page_offset = start % PAGE_SLOTS;
        Arc::make_mut(&mut self.pages[start / PAGE_SLOTS]).slots[page_offset..page_offset + N]
            .copy_from_slice(values);
    }

    pub(super) fn to_vec(&self) -> Vec<u16> {
        let mut flat = Vec::with_capacity(self.len);
        let mut remaining = self.len;
        for page in &self.pages {
            let count = remaining.min(PAGE_SLOTS);
            flat.extend_from_slice(&page.slots[..count]);
            remaining -= count;
        }
        flat
    }

    /// Stream contiguous slots directly so snapshot encoding does not need a
    /// flat temporary buffer.
    pub(super) fn append_le_bytes(&self, out: &mut Vec<u8>, start: usize, end: usize) {
        let mut cursor = start;
        while cursor < end {
            let page_index = cursor / PAGE_SLOTS;
            let page_offset = cursor % PAGE_SLOTS;
            let page_limit = self.len.min((page_index + 1) * PAGE_SLOTS) - page_index * PAGE_SLOTS;
            let count = (end - cursor).min(page_limit - page_offset);
            let slots = &self.pages[page_index].slots[page_offset..page_offset + count];
            #[cfg(target_endian = "little")]
            out.extend_from_slice(bytemuck::cast_slice(slots));
            #[cfg(target_endian = "big")]
            for &fingerprint in slots {
                out.extend_from_slice(&fingerprint.to_le_bytes());
            }
            cursor += count;
        }
    }

    /// Restore slots directly into the page-backed storage without flattening
    /// first.
    pub(super) fn write_le_bytes(&mut self, start: usize, src: &[u8]) {
        debug_assert_eq!(src.len() % 2, 0);
        let mut slot_cursor = start;
        let mut byte_cursor = 0;
        while byte_cursor < src.len() {
            let page_index = slot_cursor / PAGE_SLOTS;
            let page_offset = slot_cursor % PAGE_SLOTS;
            let page_limit = self.len.min((page_index + 1) * PAGE_SLOTS) - page_index * PAGE_SLOTS;
            let slots = (src.len() - byte_cursor)
                .div_ceil(2)
                .min(page_limit - page_offset);
            let bytes = slots * 2;
            let destination = &mut Arc::make_mut(&mut self.pages[page_index]).slots
                [page_offset..page_offset + slots];
            #[cfg(target_endian = "little")]
            bytemuck::cast_slice_mut::<u16, u8>(destination)
                .copy_from_slice(&src[byte_cursor..byte_cursor + bytes]);
            #[cfg(target_endian = "big")]
            for (slot, chunk) in destination
                .iter_mut()
                .zip(src[byte_cursor..byte_cursor + bytes].chunks_exact(2))
            {
                *slot = u16::from_le_bytes([chunk[0], chunk[1]]);
            }
            slot_cursor += slots;
            byte_cursor += bytes;
        }
    }
}
