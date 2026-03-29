use crate::error;

#[derive(Clone, Debug)]
pub(crate) enum Error {
    ConvertNonFixed {
        kind: &'static str,
    },
    #[cfg(not(feature = "tz-system"))]
    FailedSystem,
}

impl From<Error> for error::Error {
    #[cold]
    #[inline(never)]
    fn from(err: Error) -> error::Error {
        error::ErrorKind::TzTimeZone(err).into()
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
            ConvertNonFixed { kind } => write!(
                f,
                "cannot convert non-fixed {kind} time zone to offset \
                 without a timestamp or civil datetime",
            ),
            #[cfg(not(feature = "tz-system"))]
            FailedSystem => f.write_str("failed to get system time zone"),
        }
    }
}
