use crate::error;

#[derive(Clone, Debug)]
pub(crate) enum Error {
    ColonPrefixInvalidUtf8,
    InvalidPosixTz,
}

impl From<Error> for error::Error {
    #[cold]
    #[inline(never)]
    fn from(err: Error) -> error::Error {
        error::ErrorKind::TzPosix(err).into()
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
            ColonPrefixInvalidUtf8 => f.write_str(
                "POSIX time zone string with a `:` prefix \
                 contains invalid UTF-8",
            ),
            InvalidPosixTz => f.write_str("invalid POSIX time zone string"),
        }
    }
}
