//! Runtime (dlopen) binding to libjpeg-turbo's TurboJPEG API for JPEG decode.
//!
//! PIL/Pillow (and therefore vLLM) decode JPEGs with libjpeg-turbo using its
//! default options: accurate (islow) integer IDCT and "fancy" (bilinear) chroma
//! upsampling. The pure-Rust `image`/`zune-jpeg` decoder differs by a few levels
//! per pixel, which the vision encoder amplifies into a large embedding shift,
//! making TokenSpeed's multimodal accuracy diverge from vLLM. Decoding through
//! libjpeg-turbo with the same defaults makes SMG's pixel values match vLLM's.
//!
//! We load libturbojpeg at RUNTIME via `dlopen` rather than linking it, so the
//! crate (and every consumer — including the Go/Python bindings and CI builds
//! that don't ship libturbojpeg) compiles on any platform with no build script
//! and no link-time dependency. Where the shared library is present (the serving
//! image), decode goes through it for PIL parity; where it's absent,
//! `decode_jpeg_rgb` returns `None` and the caller falls back to the pure-Rust
//! decoder. Default flags (0) select accurate DCT + fancy upsampling, matching
//! Pillow.
//!
//! This module is the crate's only FFI surface, so it locally overrides the
//! workspace-wide `unsafe_code = "deny"` for the C bindings.
#![allow(unsafe_code)]

use std::{
    os::raw::{c_int, c_uchar, c_ulong, c_void},
    sync::OnceLock,
};

use image::{DynamicImage, RgbImage};
use libloading::{Library, Symbol};

type TjHandle = *mut c_void;
const TJPF_RGB: c_int = 0;

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

/// Resolved TurboJPEG entry points. Holds the loaded `Library` so the function
/// pointers stay valid for the process lifetime.
struct TurboJpeg {
    _lib: Library,
    init: TjInitDecompress,
    header: TjDecompressHeader3,
    decompress: TjDecompress2,
    destroy: TjDestroy,
}

// The function pointers are plain C entry points with no shared mutable state;
// the library handle is kept alive for the process and never mutated.
unsafe impl Send for TurboJpeg {}
unsafe impl Sync for TurboJpeg {}

fn load_turbojpeg() -> Option<TurboJpeg> {
    // Try the runtime soname first (shipped by the runtime package), then the
    // dev symlink and common macOS names.
    const CANDIDATES: &[&str] = &[
        "libturbojpeg.so.0",
        "libturbojpeg.so",
        "libturbojpeg.0.dylib",
        "libturbojpeg.dylib",
    ];
    // SAFETY: loading a system shared library by name; we only resolve the four
    // documented TurboJPEG symbols below and keep the handle for their lifetime.
    let lib = CANDIDATES
        .iter()
        .find_map(|name| unsafe { Library::new(name) }.ok())?;
    // SAFETY: each symbol is resolved against the just-loaded library with the
    // signature documented by the TurboJPEG API. We copy the bare function
    // pointers out (dropping the borrowing `Symbol`s) and keep `lib` alive in
    // the returned struct, so the pointers remain valid.
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

/// Process-wide cached TurboJPEG binding, or `None` if the library is absent.
fn turbojpeg() -> Option<&'static TurboJpeg> {
    static TJ: OnceLock<Option<TurboJpeg>> = OnceLock::new();
    TJ.get_or_init(load_turbojpeg).as_ref()
}

/// True if `bytes` start with the JPEG SOI marker.
pub fn is_jpeg(bytes: &[u8]) -> bool {
    bytes.len() >= 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF
}

/// Decode a JPEG to an RGB8 `DynamicImage` via libjpeg-turbo (PIL-compatible
/// defaults). Returns `None` on any failure — including libturbojpeg not being
/// available at runtime — so the caller can fall back to the pure-Rust decoder.
pub fn decode_jpeg_rgb(bytes: &[u8]) -> Option<DynamicImage> {
    if !is_jpeg(bytes) {
        return None;
    }
    let tj = turbojpeg()?;
    // SAFETY:
    // - `tj.init` returns a handle that is null-checked before use; every early
    //   return below calls `tj.destroy(handle)` first, so the handle is freed
    //   exactly once and never used after destruction.
    // - `bytes` is a live `&[u8]`; its ptr/len describe a valid immutable region
    //   for the duration of the calls (libjpeg-turbo only reads it).
    // - `buf` is an owned `Vec<u8>` sized to exactly `w*h*3` (overflow-checked)
    //   from the header dimensions, and is the sole alias passed to the decoder;
    //   with `pitch=0` (=> `w*3`) and `TJPF_RGB` the decoder writes at most
    //   `w*h*3` bytes, so no out-of-bounds write occurs. The image is built only
    //   after `rc == 0` confirms a successful, complete write.
    unsafe {
        let handle = (tj.init)();
        if handle.is_null() {
            return None;
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
            return None;
        }
        let (wu, hu) = (w as usize, h as usize);
        // Guard against absurd dimensions before allocating.
        let nbytes = match wu.checked_mul(hu).and_then(|p| p.checked_mul(3)) {
            Some(n) => n,
            None => {
                (tj.destroy)(handle);
                return None;
            }
        };
        let mut buf = vec![0_u8; nbytes];
        let rc = (tj.decompress)(
            handle,
            bytes.as_ptr(),
            bytes.len() as c_ulong,
            buf.as_mut_ptr(),
            w,
            0, // pitch = 0 -> width * pixelsize
            h,
            TJPF_RGB,
            0, // default flags: accurate IDCT + fancy upsampling (matches Pillow)
        );
        (tj.destroy)(handle);
        if rc != 0 {
            return None;
        }
        RgbImage::from_raw(w as u32, h as u32, buf).map(DynamicImage::ImageRgb8)
    }
}
