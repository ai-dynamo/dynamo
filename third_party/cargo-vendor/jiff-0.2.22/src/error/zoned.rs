use crate::{error, Unit};

#[derive(Clone, Debug)]
pub(crate) enum Error {
    AddDateTime,
    AddDays,
    AddTimestamp,
    ConvertDateTimeToTimestamp,
    ConvertIntermediateDatetime,
    FailedLengthOfDay,
    FailedSpanNanoseconds,
    FailedStartOfDay,
    MismatchTimeZoneUntil { largest: Unit },
}

impl From<Error> for error::Error {
    #[cold]
    #[inline(never)]
    fn from(err: Error) -> error::Error {
        error::ErrorKind::Zoned(err).into()
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
            AddDateTime => f.write_str(
                "failed to add span to datetime from zoned datetime",
            ),
            AddDays => {
                f.write_str("failed to add days to date in zoned datetime")
            }
            AddTimestamp => f.write_str(
                "failed to add span to timestamp from zoned datetime",
            ),
            ConvertDateTimeToTimestamp => {
                f.write_str("failed to convert civil datetime to timestamp")
            }
            ConvertIntermediateDatetime => f.write_str(
                "failed to convert intermediate datetime \
                 to zoned timestamp",
            ),
            FailedLengthOfDay => f.write_str(
                "failed to add 1 day to zoned datetime to find length of day",
            ),
            FailedSpanNanoseconds => f.write_str(
                "failed to compute span in nanoseconds \
                 between zoned datetimes",
            ),
            FailedStartOfDay => {
                f.write_str("failed to find start of day for zoned datetime")
            }
            MismatchTimeZoneUntil { largest } => write!(
                f,
                "computing the span between zoned datetimes, with \
                 {largest} units, requires that the time zones are \
                 equivalent, but the zoned datetimes have distinct \
                 time zones",
                largest = largest.singular(),
            ),
        }
    }
}
