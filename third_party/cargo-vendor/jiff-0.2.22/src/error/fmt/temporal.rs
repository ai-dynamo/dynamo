use crate::{error, tz::Offset, util::escape};

#[derive(Clone, Debug)]
pub(crate) enum Error {
    #[cfg(not(feature = "alloc"))]
    AllocPosixTimeZone,
    AmbiguousTimeMonthDay,
    AmbiguousTimeYearMonth,
    CivilDateTimeZulu,
    ConvertDateTimeToTimestamp {
        offset: Offset,
    },
    EmptyTimeZone,
    ExpectedDateDesignatorFoundByte {
        byte: u8,
    },
    ExpectedDateDesignatorFoundEndOfInput,
    ExpectedDurationDesignatorFoundByte {
        byte: u8,
    },
    ExpectedDurationDesignatorFoundEndOfInput,
    ExpectedFourDigitYear,
    ExpectedNoSeparator,
    ExpectedOneDigitWeekday,
    ExpectedSeparatorFoundByte {
        byte: u8,
    },
    ExpectedSeparatorFoundEndOfInput,
    ExpectedSixDigitYear,
    ExpectedTimeDesignator,
    ExpectedTimeDesignatorFoundByte {
        byte: u8,
    },
    ExpectedTimeDesignatorFoundEndOfInput,
    ExpectedTimeUnits,
    ExpectedTwoDigitDay,
    ExpectedTwoDigitHour,
    ExpectedTwoDigitMinute,
    ExpectedTwoDigitMonth,
    ExpectedTwoDigitSecond,
    ExpectedTwoDigitWeekNumber,
    ExpectedWeekPrefixFoundByte {
        byte: u8,
    },
    ExpectedWeekPrefixFoundEndOfInput,
    FailedDayInDate,
    FailedDayInMonthDay,
    FailedFractionalSecondInTime,
    FailedHourInTime,
    FailedMinuteInTime,
    FailedMonthInDate,
    FailedMonthInMonthDay,
    FailedMonthInYearMonth,
    FailedOffsetNumeric,
    FailedSecondInTime,
    FailedSeparatorAfterMonth,
    FailedSeparatorAfterWeekNumber,
    FailedSeparatorAfterYear,
    FailedTzdbLookup,
    FailedWeekNumberInDate,
    FailedWeekNumberPrefixInDate,
    FailedWeekdayInDate,
    FailedYearInDate,
    FailedYearInYearMonth,
    InvalidDate,
    InvalidMonthDay,
    InvalidTimeZoneUtf8,
    InvalidWeekDate,
    InvalidYearMonth,
    InvalidYearZero,
    MissingOffsetInTimestamp,
    MissingTimeInDate,
    MissingTimeInTimestamp,
    MissingTimeZoneAnnotation,
    ParseDayTwoDigit,
    ParseHourTwoDigit,
    ParseMinuteTwoDigit,
    ParseMonthTwoDigit,
    ParseSecondTwoDigit,
    ParseWeekNumberTwoDigit,
    ParseWeekdayOneDigit,
    ParseYearFourDigit,
    ParseYearSixDigit,
    // This is the only error for formatting a Temporal value. And
    // actually, it's not even part of Temporal, but just lives in that
    // module (for convenience reasons).
    PrintTimeZoneFailure,
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
        error::ErrorKind::FmtTemporal(err).into()
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::Error::*;

        match *self {
            #[cfg(not(feature = "alloc"))]
            AllocPosixTimeZone => f.write_str(
                "cannot parsed time zones other than IANA time zone \
                 identifiers or fixed offsets \
                 without the `alloc` crate feature enabled for `jiff`",
            ),
            AmbiguousTimeMonthDay => {
                f.write_str("parsed time is ambiguous with a month-day date")
            }
            AmbiguousTimeYearMonth => {
                f.write_str("parsed time is ambiguous with a year-month date")
            }
            CivilDateTimeZulu => f.write_str(
                "cannot parse civil date/time from string with a Zulu \
                 offset, parse as a `jiff::Timestamp` first \
                 and convert to a civil date/time instead",
            ),
            ConvertDateTimeToTimestamp { offset } => write!(
                f,
                "failed to convert civil datetime to timestamp \
                 with offset {offset}",
            ),
            EmptyTimeZone => {
                f.write_str("an empty string is not a valid time zone")
            }
            ExpectedDateDesignatorFoundByte { byte } => write!(
                f,
                "expected to find date unit designator suffix \
                 (`Y`, `M`, `W` or `D`), but found `{byte}` instead",
                byte = escape::Byte(byte),
            ),
            ExpectedDateDesignatorFoundEndOfInput => f.write_str(
                "expected to find date unit designator suffix \
                 (`Y`, `M`, `W` or `D`), but found end of input",
            ),
            ExpectedDurationDesignatorFoundByte { byte } => write!(
                f,
                "expected to find duration beginning with `P` or `p`, \
                 but found `{byte}` instead",
                byte = escape::Byte(byte),
            ),
            ExpectedDurationDesignatorFoundEndOfInput => f.write_str(
                "expected to find duration beginning with `P` or `p`, \
                 but found end of input",
            ),
            ExpectedFourDigitYear => f.write_str(
                "expected four digit year (or leading sign for \
                 six digit year), but found end of input",
            ),
            ExpectedNoSeparator => f.write_str(
                "expected no separator since none was \
                 found after the year, but found a `-` separator",
            ),
            ExpectedOneDigitWeekday => f.write_str(
                "expected one digit weekday, but found end of input",
            ),
            ExpectedSeparatorFoundByte { byte } => write!(
                f,
                "expected `-` separator, but found `{byte}`",
                byte = escape::Byte(byte),
            ),
            ExpectedSeparatorFoundEndOfInput => {
                f.write_str("expected `-` separator, but found end of input")
            }
            ExpectedSixDigitYear => f.write_str(
                "expected six digit year (because of a leading sign), \
                 but found end of input",
            ),
            ExpectedTimeDesignator => f.write_str(
                "parsing ISO 8601 duration in this context requires \
                 that the duration contain a time component and no \
                 components of days or greater",
            ),
            ExpectedTimeDesignatorFoundByte { byte } => write!(
                f,
                "expected to find time unit designator suffix \
                 (`H`, `M` or `S`), but found `{byte}` instead",
                byte = escape::Byte(byte),
            ),
            ExpectedTimeDesignatorFoundEndOfInput => f.write_str(
                "expected to find time unit designator suffix \
                 (`H`, `M` or `S`), but found end of input",
            ),
            ExpectedTimeUnits => f.write_str(
                "found a time designator (`T` or `t`) in an ISO 8601 \
                 duration string, but did not find any time units",
            ),
            ExpectedTwoDigitDay => {
                f.write_str("expected two digit day, but found end of input")
            }
            ExpectedTwoDigitHour => {
                f.write_str("expected two digit hour, but found end of input")
            }
            ExpectedTwoDigitMinute => f.write_str(
                "expected two digit minute, but found end of input",
            ),
            ExpectedTwoDigitMonth => {
                f.write_str("expected two digit month, but found end of input")
            }
            ExpectedTwoDigitSecond => f.write_str(
                "expected two digit second, but found end of input",
            ),
            ExpectedTwoDigitWeekNumber => f.write_str(
                "expected two digit week number, but found end of input",
            ),
            ExpectedWeekPrefixFoundByte { byte } => write!(
                f,
                "expected `W` or `w`, but found `{byte}` instead",
                byte = escape::Byte(byte),
            ),
            ExpectedWeekPrefixFoundEndOfInput => {
                f.write_str("expected `W` or `w`, but found end of input")
            }
            FailedDayInDate => f.write_str("failed to parse day in date"),
            FailedDayInMonthDay => {
                f.write_str("failed to payse day in month-day")
            }
            FailedFractionalSecondInTime => {
                f.write_str("failed to parse fractional seconds in time")
            }
            FailedHourInTime => f.write_str("failed to parse hour in time"),
            FailedMinuteInTime => {
                f.write_str("failed to parse minute in time")
            }
            FailedMonthInDate => f.write_str("failed to parse month in date"),
            FailedMonthInMonthDay => {
                f.write_str("failed to parse month in month-day")
            }
            FailedMonthInYearMonth => {
                f.write_str("failed to parse month in year-month")
            }
            FailedOffsetNumeric => f.write_str(
                "offset successfully parsed, \
                 but failed to convert to numeric `jiff::tz::Offset`",
            ),
            FailedSecondInTime => {
                f.write_str("failed to parse second in time")
            }
            FailedSeparatorAfterMonth => {
                f.write_str("failed to parse separator after month")
            }
            FailedSeparatorAfterWeekNumber => {
                f.write_str("failed to parse separator after week number")
            }
            FailedSeparatorAfterYear => {
                f.write_str("failed to parse separator after year")
            }
            FailedTzdbLookup => f.write_str(
                "parsed apparent IANA time zone identifier, \
                 but the tzdb lookup failed",
            ),
            FailedWeekNumberInDate => {
                f.write_str("failed to parse week number in date")
            }
            FailedWeekNumberPrefixInDate => {
                f.write_str("failed to parse week number prefix in date")
            }
            FailedWeekdayInDate => {
                f.write_str("failed to parse weekday in date")
            }
            FailedYearInDate => f.write_str("failed to parse year in date"),
            FailedYearInYearMonth => {
                f.write_str("failed to parse year in year-month")
            }
            InvalidDate => f.write_str("parsed date is not valid"),
            InvalidMonthDay => f.write_str("parsed month-day is not valid"),
            InvalidTimeZoneUtf8 => f.write_str(
                "found plausible IANA time zone identifier, \
                 but it is not valid UTF-8",
            ),
            InvalidWeekDate => f.write_str("parsed week date is not valid"),
            InvalidYearMonth => f.write_str("parsed year-month is not valid"),
            InvalidYearZero => f.write_str(
                "year zero must be written without a sign or a \
                 positive sign, but not a negative sign",
            ),
            MissingOffsetInTimestamp => f.write_str(
                "failed to find offset component, \
                 which is required for parsing a timestamp",
            ),
            MissingTimeInDate => f.write_str(
                "successfully parsed date, but no time component was found",
            ),
            MissingTimeInTimestamp => f.write_str(
                "failed to find time component, \
                 which is required for parsing a timestamp",
            ),
            MissingTimeZoneAnnotation => f.write_str(
                "failed to find time zone annotation in square brackets, \
                 which is required for parsing a zoned datetime",
            ),
            ParseDayTwoDigit => {
                f.write_str("failed to parse two digit integer as day")
            }
            ParseHourTwoDigit => {
                f.write_str("failed to parse two digit integer as hour")
            }
            ParseMinuteTwoDigit => {
                f.write_str("failed to parse two digit integer as minute")
            }
            ParseMonthTwoDigit => {
                f.write_str("failed to parse two digit integer as month")
            }
            ParseSecondTwoDigit => {
                f.write_str("failed to parse two digit integer as second")
            }
            ParseWeekNumberTwoDigit => {
                f.write_str("failed to parse two digit integer as week number")
            }
            ParseWeekdayOneDigit => {
                f.write_str("failed to parse one digit integer as weekday")
            }
            ParseYearFourDigit => {
                f.write_str("failed to parse four digit integer as year")
            }
            ParseYearSixDigit => {
                f.write_str("failed to parse six digit integer as year")
            }
            PrintTimeZoneFailure => f.write_str(
                "time zones without IANA identifiers that aren't either \
                 fixed offsets or a POSIX time zone can't be serialized \
                 (this typically occurs when this is a system time zone \
                  derived from `/etc/localtime` on Unix systems that \
                  isn't symlinked to an entry in `/usr/share/zoneinfo`)",
            ),
        }
    }
}
