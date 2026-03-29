use crate::error;

#[derive(Clone, Debug)]
pub(crate) enum Error {
    OverflowAddDuration,
    OverflowAddSpan,
}

impl From<Error> for error::Error {
    #[cold]
    #[inline(never)]
    fn from(err: Error) -> error::Error {
        error::ErrorKind::Timestamp(err).into()
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
            OverflowAddDuration => {
                f.write_str("adding duration overflowed timestamp")
            }
            OverflowAddSpan => f.write_str("adding span overflowed timestamp"),
        }
    }
}
