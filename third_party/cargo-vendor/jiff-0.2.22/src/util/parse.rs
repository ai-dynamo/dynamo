use crate::error::util::{ParseFractionError, ParseIntError};

/// Parses an `i64` number from the beginning to the end of the given slice of
/// ASCII digit characters.
///
/// If any byte in the given slice is not `[0-9]`, then this returns an error.
/// Similarly, if the number parsed does not fit into a `i64`, then this
/// returns an error. Notably, this routine does not permit parsing a negative
/// integer. (We use `i64` because everything in this crate uses signed
/// integers, and because a higher level routine might want to parse the sign
/// and then apply it to the result of this routine.)
#[cfg_attr(feature = "perf-inline", inline(always))]
pub(crate) fn i64(bytes: &[u8]) -> Result<i64, ParseIntError> {
    if bytes.is_empty() {
        return Err(ParseIntError::NoDigitsFound);
    }
    let mut n: i64 = 0;
    for &byte in bytes {
        if !(b'0' <= byte && byte <= b'9') {
            return Err(ParseIntError::InvalidDigit(byte));
        }
        let digit = i64::from(byte - b'0');
        n = n
            .checked_mul(10)
            .and_then(|n| n.checked_add(digit))
            .ok_or(ParseIntError::TooBig)?;
    }
    Ok(n)
}

/// Parsed an optional `u64` that is a prefix of `bytes`.
///
/// If no digits (`[0-9]`) were found at the beginning of `bytes`, then `None`
/// is returned.
///
/// Note that this is safe to call on untrusted input. It will not attempt
/// to consume more input than could possibly fit into a parsed integer.
///
/// Since this returns a `u64`, it is possible that an integer that cannot
/// fit into an `i64` is returned. Callers should handle this. (Indeed,
/// `DurationUnits` handles this case.)
///
/// # Errors
///
/// When the parsed integer cannot fit into a `u64`.
#[cfg_attr(feature = "perf-inline", inline(always))]
pub(crate) fn u64_prefix(
    bytes: &[u8],
) -> Result<(Option<u64>, &[u8]), ParseIntError> {
    // Discovered via `u64::MAX.to_string().len()`.
    const MAX_U64_DIGITS: usize = 20;

    let mut digit_count = 0;
    let mut n: u64 = 0;
    while digit_count <= MAX_U64_DIGITS {
        let Some(&byte) = bytes.get(digit_count) else { break };
        if !byte.is_ascii_digit() {
            break;
        }
        digit_count += 1;
        // OK because we confirmed `byte` is an ASCII digit.
        let digit = u64::from(byte - b'0');
        n = n
            .checked_mul(10)
            .and_then(|n| n.checked_add(digit))
            .ok_or(ParseIntError::TooBig)?;
    }
    if digit_count == 0 {
        return Ok((None, bytes));
    }
    Ok((Some(n), &bytes[digit_count..]))
}

/// Parses a `u32` fractional number from the beginning to the end of the given
/// slice of ASCII digit characters.
///
/// The fraction's maximum precision is always 9 digits. The returned integer
/// will always be in units of `10^{max_precision}`. For example, this
/// will parse a fractional amount of seconds with a maximum precision of
/// nanoseconds.
///
/// If any byte in the given slice is not `[0-9]`, then this returns an error.
/// Notably, this routine does not permit parsing a negative integer.
pub(crate) fn fraction(bytes: &[u8]) -> Result<u32, ParseFractionError> {
    if bytes.is_empty() {
        return Err(ParseFractionError::NoDigitsFound);
    } else if bytes.len() > ParseFractionError::MAX_PRECISION {
        return Err(ParseFractionError::TooManyDigits);
    }
    let mut n: u32 = 0;
    for &byte in bytes {
        let digit = match byte.checked_sub(b'0') {
            None => {
                return Err(ParseFractionError::InvalidDigit(byte));
            }
            Some(digit) if digit > 9 => {
                return Err(ParseFractionError::InvalidDigit(byte));
            }
            Some(digit) => {
                debug_assert!((0..=9).contains(&digit));
                u32::from(digit)
            }
        };
        n = n
            .checked_mul(10)
            .and_then(|n| n.checked_add(digit))
            .ok_or_else(|| ParseFractionError::TooBig)?;
    }
    for _ in bytes.len()..ParseFractionError::MAX_PRECISION {
        n = n.checked_mul(10).ok_or_else(|| ParseFractionError::TooBig)?;
    }
    Ok(n)
}

/// Parses an `OsStr` into a `&str` when `&[u8]` isn't easily available.
///
/// This is effectively `OsStr::to_str`, but with a slightly better error
/// message.
#[cfg(any(feature = "tz-system", feature = "tzdb-zoneinfo"))]
pub(crate) fn os_str_utf8<'o, O>(
    os_str: &'o O,
) -> Result<&'o str, crate::error::util::OsStrUtf8Error>
where
    O: ?Sized + AsRef<std::ffi::OsStr>,
{
    let os_str = os_str.as_ref();
    os_str
        .to_str()
        .ok_or_else(|| crate::error::util::OsStrUtf8Error::from(os_str))
}

/// Parses an `OsStr` into a `&str` when `&[u8]` isn't easily available.
///
/// The main difference between this and `OsStr::to_str` is that this will
/// be a zero-cost conversion on Unix platforms to `&[u8]`. On Windows, this
/// will do UTF-8 validation and return an error if it's invalid UTF-8.
#[cfg(feature = "tz-system")]
pub(crate) fn os_str_bytes<'o, O>(
    os_str: &'o O,
) -> Result<&'o [u8], crate::error::util::OsStrUtf8Error>
where
    O: ?Sized + AsRef<std::ffi::OsStr>,
{
    let os_str = os_str.as_ref();
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        Ok(os_str.as_bytes())
    }
    #[cfg(not(unix))]
    {
        // It is suspect that we're doing UTF-8 validation and then throwing
        // away the fact that we did UTF-8 validation. So this could lead
        // to an extra UTF-8 check if the caller ultimately needs UTF-8. If
        // that's important, we can add a new API that returns a `&str`. But it
        // probably won't matter because an `OsStr` in this crate is usually
        // just an environment variable.
        Ok(os_str_utf8(os_str)?.as_bytes())
    }
}

/// Splits the given input into two slices at the given position.
///
/// If the position is greater than the length of the slice given, then this
/// returns `None`.
#[cfg_attr(feature = "perf-inline", inline(always))]
pub(crate) fn split(input: &[u8], at: usize) -> Option<(&[u8], &[u8])> {
    if at > input.len() {
        None
    } else {
        Some(input.split_at(at))
    }
}

/// Returns a function that converts two slices to an offset.
///
/// It takes the starting point as input and returns a function that, when
/// given an ending point (greater than or equal to the starting point), then
/// the corresponding pointers are subtracted and an offset relative to the
/// starting point is returned.
///
/// This is useful as a helper function in parsing routines that use slices
/// but want to report offsets.
///
/// # Panics
///
/// This may panic if the ending point is not a suffix slice of `start`.
pub(crate) fn offseter<'a>(
    start: &'a [u8],
) -> impl Fn(&'a [u8]) -> usize + 'a {
    move |end| (end.as_ptr() as usize) - (start.as_ptr() as usize)
}

/// Returns a function that converts two slices to the slice between them.
///
/// This takes a starting point as input and returns a function that, when
/// given an ending point (greater than or equal to the starting point), it
/// returns a slice beginning at the starting point and ending just at the
/// ending point.
///
/// This is useful as a helper function in parsing routines.
///
/// # Panics
///
/// This may panic if the ending point is not a suffix slice of `start`.
pub(crate) fn slicer<'a>(
    start: &'a [u8],
) -> impl Fn(&'a [u8]) -> &'a [u8] + 'a {
    let mkoffset = offseter(start);
    move |end| {
        let offset = mkoffset(end);
        &start[..offset]
    }
}
