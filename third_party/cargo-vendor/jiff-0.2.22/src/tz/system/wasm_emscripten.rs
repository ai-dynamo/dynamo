use core::ffi::{c_char, CStr};
use core::ptr::NonNull;

use crate::{
    tz::{TimeZone, TimeZoneDatabase},
    util::constant::unwrapr,
};

extern "C" {
    /// https://emscripten.org/docs/api_reference/emscripten.h.html#c.emscripten_run_script_string
    fn emscripten_run_script_string(script: *const c_char) -> *mut c_char;
}

pub(super) fn get(db: &TimeZoneDatabase) -> Option<TimeZone> {
    const SCRIPT: &CStr = unwrapr!(
        CStr::from_bytes_with_nul(
            "Intl.DateTimeFormat().resolvedOptions().timeZone\0".as_bytes(),
        ),
        "not a valid C string literal"
    );

    // SAFETY: SCRIPT is a valid pointer to a null-terminated string. The
    // result will be a pointer to a null-terminated C string, or null if the
    // script failed to run. The returned value is valid until the next call to
    // `emscripten_run_script_string` in the same thread, so it is valid for
    // the duration of this function.
    let script_result =
        unsafe { emscripten_run_script_string(SCRIPT.as_ptr()) };
    let script_ptr = match NonNull::new(script_result) {
        Some(script_cstr) => script_cstr,
        None => {
            trace!(
                "executing JavaScript to discover IANA time zone \
                 identifier returned a null pointer",
            );
            return None;
        }
    };
    // SAFETY: OK because of the contract of `emscripten_run_script_string`.
    // We assume that when it returns a non-null pointer (which is checked
    // above) that it is a valid C string.
    let script_cstr = unsafe { CStr::from_ptr(script_ptr.as_ptr()) };
    let name = match script_cstr.to_str().ok() {
        Some(name) => name,
        None => {
            trace!(
                "executing JavaScript to discover IANA time zone \
                 identifier returned invalid UTF-8: {script_cstr}",
                script_cstr = escape::Bytes(script_cstr.to_bytes()),
            );
            return None;
        }
    };
    let tz = match db.get(name) {
        Ok(tz) => tz,
        Err(_err) => {
            trace!(
                "got {name:?} as time zone name, \
                 but failed to find time zone with that name in \
                 zoneinfo database {db:?}: {_err}",
            );
            return None;
        }
    };
    Some(tz)
}

pub(super) fn read(_db: &TimeZoneDatabase, path: &str) -> Option<TimeZone> {
    match super::read_unnamed_tzif_file(path) {
        Ok(tz) => Some(tz),
        Err(_err) => {
            trace!("failed to read {path} as unnamed time zone: {_err}");
            None
        }
    }
}
