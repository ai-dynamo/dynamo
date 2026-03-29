use crate::{error, util::escape};

pub(crate) mod friendly;
pub(crate) mod offset;
pub(crate) mod rfc2822;
pub(crate) mod rfc9557;
pub(crate) mod strtime;
pub(crate) mod temporal;
pub(crate) mod util;

#[derive(Clone, Debug)]
pub(crate) enum Error {
    HybridDurationEmpty,
    HybridDurationPrefix {
        sign: u8,
    },
    IntoFull {
        #[cfg(feature = "alloc")]
        value: alloc::boxed::Box<str>,
        #[cfg(feature = "alloc")]
        unparsed: alloc::boxed::Box<[u8]>,
    },
    StdFmtWriteAdapter,
}

impl Error {
    pub(crate) fn into_full_error(
        _value: &dyn core::fmt::Display,
        _unparsed: &[u8],
    ) -> Error {
        Error::IntoFull {
            #[cfg(feature = "alloc")]
            value: alloc::string::ToString::to_string(_value).into(),
            #[cfg(feature = "alloc")]
            unparsed: _unparsed.into(),
        }
    }
}

impl error::IntoError for Error {
    fn into_error(self) -> error::Error {
        self.into()
    }
}

impl From<Error> for error::Error {
    #[cold]
    #[inline(never)]
    fn from(err: Error) -> error::Error {
        error::ErrorKind::Fmt(err).into()
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::Error::*;

        match *self {
            HybridDurationEmpty => f.write_str(
                "an empty string is not a valid duration in either \
                 the ISO 8601 format or Jiff's \"friendly\" format",
            ),
            HybridDurationPrefix { sign } => write!(
                f,
                "found nothing after sign `{sign}`, \
                 which is not a valid duration in either \
                 the ISO 8601 format or Jiff's \"friendly\" format",
                sign = escape::Byte(sign),
            ),
            #[cfg(not(feature = "alloc"))]
            IntoFull { .. } => f.write_str(
                "parsed value, but unparsed input remains \
                 (expected no unparsed input)",
            ),
            #[cfg(feature = "alloc")]
            IntoFull { ref value, ref unparsed } => write!(
                f,
                "parsed value '{value}', but unparsed input {unparsed:?} \
                 remains (expected no unparsed input)",
                unparsed = escape::Bytes(unparsed),
            ),
            StdFmtWriteAdapter => {
                f.write_str("an error occurred when formatting an argument")
            }
        }
    }
}
