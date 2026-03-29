use crate::{error, Unit};

#[derive(Clone, Debug)]
pub(crate) enum Error {
    ConvertDateTimeToTimestamp,
    ConvertNanoseconds { unit: Unit },
    ConvertNegative,
    ConvertSpanToSignedDuration,
    FailedSpanBetweenDateTimes { unit: Unit },
    FailedSpanBetweenZonedDateTimes { unit: Unit },
    OptionLargest,
    OptionLargestInSpan,
    OptionSmallest,
    ToDurationCivil,
    ToDurationDaysAre24Hours,
    ToDurationZoned,
}

impl From<Error> for error::Error {
    #[cold]
    #[inline(never)]
    fn from(err: Error) -> error::Error {
        error::ErrorKind::Span(err).into()
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
            ConvertDateTimeToTimestamp => f.write_str(
                "failed to interpret datetime in UTC \
                 in order to convert it to a timestamp",
            ),
            ConvertNanoseconds { unit } => write!(
                f,
                "failed to convert rounded nanoseconds \
                 to span for largest unit set to '{unit}'",
                unit = unit.plural(),
            ),
            ConvertNegative => f.write_str(
                "cannot convert negative span \
                 to unsigned `std::time::Duration`",
            ),
            ConvertSpanToSignedDuration => f.write_str(
                "failed to convert span to duration without relative datetime \
                 (must use `jiff::Span::to_duration` instead)",
            ),
            FailedSpanBetweenDateTimes { unit } => write!(
                f,
                "failed to get span between datetimes \
                 with largest unit set to '{unit}'",
                unit = unit.plural(),
            ),
            FailedSpanBetweenZonedDateTimes { unit } => write!(
                f,
                "failed to get span between zoned datetimes \
                 with largest unit set to '{unit}'",
                unit = unit.plural(),
            ),
            OptionLargest => {
                f.write_str("error with `largest` rounding option")
            }
            OptionLargestInSpan => {
                f.write_str("error with largest unit in span to be rounded")
            }
            OptionSmallest => {
                f.write_str("error with `smallest` rounding option")
            }
            ToDurationCivil => f.write_str(
                "could not compute normalized relative span \
                 from civil datetime",
            ),
            ToDurationDaysAre24Hours => f.write_str(
                "could not compute normalized relative span \
                 when all days are assumed to be 24 hours",
            ),
            ToDurationZoned => f.write_str(
                "could not compute normalized relative span \
                 from zoned datetime",
            ),
        }
    }
}
