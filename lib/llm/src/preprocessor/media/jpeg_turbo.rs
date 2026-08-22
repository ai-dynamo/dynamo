// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime bindings for libjpeg-turbo's TurboJPEG decode API.
//!
//! The shared library is loaded with `dlopen` instead of linked at build time so
//! Dynamo keeps working in images that do not install libturbojpeg. Callers can
//! use this as an opt-in fast path and fall back to the pure-Rust `image`
//! decoder when the library is unavailable.

#![allow(unsafe_code)]

use std::{
    os::raw::{c_int, c_uchar, c_ulong, c_void},
    sync::OnceLock,
};

use anyhow::{Result, bail};
use libloading::{Library, Symbol};

type TjHandle = *mut c_void;
const TJCS_GRAY: c_int = 2;
const TJPF_RGB: c_int = 0;
const TJPF_GRAY: c_int = 6;

type TjInitDecompress = unsafe extern "C" fn() -> TjHandle;
type TjDecompressHeader3 = unsafe extern "C" fn(
    TjHandle,
    *const c_uchar,
    c_ulong,
    *mut c_int,
    *mut c_int,
    *mut c_int,
    *mut c_int,
) -> c_int;
type TjDecompress2 = unsafe extern "C" fn(
    TjHandle,
    *const c_uchar,
    c_ulong,
    *mut c_uchar,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
) -> c_int;
type TjDestroy = unsafe extern "C" fn(TjHandle) -> c_int;

struct TurboJpeg {
    _lib: Library,
    init: TjInitDecompress,
    header: TjDecompressHeader3,
    decompress: TjDecompress2,
    destroy: TjDestroy,
}

// The resolved symbols are immutable C entry points. The Library is retained
// for the process lifetime by OnceLock.
unsafe impl Send for TurboJpeg {}
unsafe impl Sync for TurboJpeg {}

#[derive(Debug)]
pub(crate) struct DecodedJpeg {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) channels: usize,
    pub(crate) data: Vec<u8>,
}

fn load_turbojpeg() -> Option<TurboJpeg> {
    const CANDIDATES: &[&str] = &[
        "libturbojpeg.so.0",
        "libturbojpeg.so",
        "libturbojpeg.0.dylib",
        "libturbojpeg.dylib",
    ];

    // SAFETY: Load the documented TurboJPEG shared library by soname and only
    // resolve the four symbols below.
    let lib = CANDIDATES
        .iter()
        .find_map(|name| unsafe { Library::new(name) }.ok())?;

    // SAFETY: The symbol signatures match the TurboJPEG C API. We copy bare
    // function pointers while keeping `lib` alive in TurboJpeg.
    let (init, header, decompress, destroy) = unsafe {
        let init: Symbol<TjInitDecompress> = lib.get(b"tjInitDecompress\0").ok()?;
        let header: Symbol<TjDecompressHeader3> = lib.get(b"tjDecompressHeader3\0").ok()?;
        let decompress: Symbol<TjDecompress2> = lib.get(b"tjDecompress2\0").ok()?;
        let destroy: Symbol<TjDestroy> = lib.get(b"tjDestroy\0").ok()?;
        (*init, *header, *decompress, *destroy)
    };

    Some(TurboJpeg {
        _lib: lib,
        init,
        header,
        decompress,
        destroy,
    })
}

fn turbojpeg() -> Option<&'static TurboJpeg> {
    static TJ: OnceLock<Option<TurboJpeg>> = OnceLock::new();
    TJ.get_or_init(load_turbojpeg).as_ref()
}

pub(crate) fn available() -> bool {
    turbojpeg().is_some()
}

pub(crate) fn is_jpeg(bytes: &[u8]) -> bool {
    bytes.len() >= 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF
}

/// Decode JPEG bytes to HWC UINT8 using libjpeg-turbo defaults.
///
/// Returns `Ok(None)` for non-JPEG inputs, missing libturbojpeg, or decoder
/// failures that the caller should pass to the standard `image` fallback.
/// Configured resource limits return `Err` because falling back should not
/// bypass them.
pub(crate) fn decode_jpeg(
    bytes: &[u8],
    max_width: Option<u32>,
    max_height: Option<u32>,
    max_alloc: Option<u64>,
) -> Result<Option<DecodedJpeg>> {
    if !is_jpeg(bytes) {
        return Ok(None);
    }
    let Some(tj) = turbojpeg() else {
        return Ok(None);
    };

    // SAFETY:
    // - The handle is null-checked and destroyed on every exit path below.
    // - `bytes` is a valid immutable input buffer for the duration of each C call.
    // - `buf` is allocated to exactly width * height * channels after checked
    //   arithmetic and configured allocation limits; the selected TurboJPEG
    //   pixel format with pitch 0 writes that layout.
    unsafe {
        let handle = (tj.init)();
        if handle.is_null() {
            return Ok(None);
        }

        let (mut w, mut h, mut subsamp, mut colorspace) = (0_i32, 0_i32, 0_i32, 0_i32);
        let hdr = (tj.header)(
            handle,
            bytes.as_ptr(),
            bytes.len() as c_ulong,
            &mut w,
            &mut h,
            &mut subsamp,
            &mut colorspace,
        );
        if hdr != 0 || w <= 0 || h <= 0 {
            (tj.destroy)(handle);
            return Ok(None);
        }

        let (width, height) = (w as u32, h as u32);
        if max_width.is_some_and(|limit| width > limit)
            || max_height.is_some_and(|limit| height > limit)
        {
            (tj.destroy)(handle);
            bail!("Image dimensions exceed configured limits: {width}x{height}");
        }

        let (pixel_format, channels) = if colorspace == TJCS_GRAY {
            (TJPF_GRAY, 1_usize)
        } else {
            (TJPF_RGB, 3_usize)
        };
        let nbytes = match u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|p| p.checked_mul(channels as u64))
        {
            Some(n) => n,
            None => {
                (tj.destroy)(handle);
                bail!("Image allocation size overflow for dimensions: {width}x{height}");
            }
        };
        if let Some(limit) = max_alloc
            && nbytes > limit
        {
            (tj.destroy)(handle);
            bail!("Image allocation {nbytes} bytes exceeds configured limit {limit} bytes");
        }

        let nbytes = match usize::try_from(nbytes) {
            Ok(n) => n,
            Err(_) => {
                (tj.destroy)(handle);
                bail!("Image allocation size does not fit in usize: {nbytes} bytes");
            }
        };
        let mut buf = Vec::new();
        if let Err(err) = buf.try_reserve_exact(nbytes) {
            (tj.destroy)(handle);
            bail!("Image allocation {nbytes} bytes could not be reserved: {err:?}");
        }
        buf.resize(nbytes, 0);
        let rc = (tj.decompress)(
            handle,
            bytes.as_ptr(),
            bytes.len() as c_ulong,
            buf.as_mut_ptr(),
            w,
            0,
            h,
            pixel_format,
            0,
        );
        (tj.destroy)(handle);
        if rc != 0 {
            return Ok(None);
        }

        Ok(Some(DecodedJpeg {
            width,
            height,
            channels,
            data: buf,
        }))
    }
}
