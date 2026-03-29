use crate::error;

#[derive(Clone, Debug)]
pub(crate) enum Error {
    FailedNegateUnsignedDuration,
    RangeUnsignedDuration,
}

impl From<Error> for error::Error {
    #[cold]
    #[inline(never)]
    fn from(err: Error) -> error::Error {
        error::ErrorKind::Duration(err).into()
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
            FailedNegateUnsignedDuration => {
                f.write_str("failed to negate unsigned duration")
            }
            RangeUnsignedDuration => {
                f.write_str("unsigned duration exceeds Jiff's limits")
            }
        }
    }
}
