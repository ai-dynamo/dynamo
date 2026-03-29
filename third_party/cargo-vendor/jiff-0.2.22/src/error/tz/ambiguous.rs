use crate::{
    error,
    tz::{Offset, TimeZone},
};

#[derive(Clone, Debug)]
pub(crate) enum Error {
    BecauseFold { before: Offset, after: Offset },
    BecauseGap { before: Offset, after: Offset },
    InTimeZone { tz: TimeZone },
}

impl From<Error> for error::Error {
    #[cold]
    #[inline(never)]
    fn from(err: Error) -> error::Error {
        error::ErrorKind::TzAmbiguous(err).into()
    }
}

impl error::IntoError for Error {
    fn into_error(self) -> error::Error {
        self.into()
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::Error::*;

        match *self {
            BecauseFold { before, after } => write!(
                f,
                "datetime is ambiguous since it falls into a \
                 fold between offsets {before} and {after}",
            ),
            BecauseGap { before, after } => write!(
                f,
                "datetime is ambiguous since it falls into a \
                 gap between offsets {before} and {after}",
            ),
            InTimeZone { ref tz } => write!(
                f,
                "error converting datetime to instant in time zone {tz}",
                tz = tz.diagnostic_name(),
            ),
        }
    }
}
