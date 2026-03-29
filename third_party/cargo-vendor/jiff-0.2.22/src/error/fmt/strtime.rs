use crate::{civil::Weekday, error, tz::Offset, util::escape};

#[derive(Clone, Debug)]
pub(crate) enum Error {
    ColonCount {
        directive: u8,
    },
    DirectiveFailure {
        directive: u8,
        colons: u8,
    },
    DirectiveFailureDot {
        directive: u8,
    },
    ExpectedDirectiveAfterColons,
    ExpectedDirectiveAfterFlag {
        flag: u8,
    },
    ExpectedDirectiveAfterWidth,
    FailedStrftime,
    FailedStrptime,
    FailedWidth,
    InvalidDate,
    InvalidISOWeekDate,
    InvalidWeekdayMonday {
        got: Weekday,
    },
    InvalidWeekdaySunday {
        got: Weekday,
    },
    MismatchOffset {
        parsed: Offset,
        got: Offset,
    },
    MismatchWeekday {
        parsed: Weekday,
        got: Weekday,
    },
    MissingTimeHourForFractional,
    MissingTimeHourForMinute,
    MissingTimeHourForSecond,
    MissingTimeMinuteForFractional,
    MissingTimeMinuteForSecond,
    MissingTimeSecondForFractional,
    RangeTimestamp,
    RangeWidth,
    RequiredDateForDateTime,
    RequiredDateTimeForTimestamp,
    RequiredDateTimeForZoned,
    RequiredOffsetForTimestamp,
    RequiredSomeDayForDate,
    RequiredTimeForDateTime,
    RequiredYearForDate,
    UnconsumedStrptime {
        #[cfg(feature = "alloc")]
        remaining: alloc::boxed::Box<[u8]>,
    },
    UnexpectedEndAfterDot,
    UnexpectedEndAfterPercent,
    UnknownDirectiveAfterDot {
        directive: u8,
    },
    UnknownDirective {
        directive: u8,
    },
    ZonedOffsetOrTz,
}

impl Error {
    pub(crate) fn unconsumed(_remaining: &[u8]) -> Error {
        Error::UnconsumedStrptime {
            #[cfg(feature = "alloc")]
            remaining: _remaining.into(),
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
        error::ErrorKind::FmtStrtime(err).into()
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::Error::*;

        match *self {
            ColonCount { directive } => write!(
                f,
                "invalid number of `:` in `%{directive}` directive",
                directive = escape::Byte(directive),
            ),
            DirectiveFailure { directive, colons } => write!(
                f,
                "%{colons}{directive} failed",
                colons = escape::RepeatByte { byte: b':', count: colons },
                directive = escape::Byte(directive),
            ),
            DirectiveFailureDot { directive } => write!(
                f,
                "%.{directive} failed",
                directive = escape::Byte(directive),
            ),
            ExpectedDirectiveAfterColons => f.write_str(
                "expected to find specifier directive after colons, \
                 but found end of format string",
            ),
            ExpectedDirectiveAfterFlag { flag } => write!(
                f,
                "expected to find specifier directive after flag \
                 `{flag}`, but found end of format string",
                flag = escape::Byte(flag),
            ),
            ExpectedDirectiveAfterWidth => f.write_str(
                "expected to find specifier directive after parsed width, \
                 but found end of format string",
            ),
            FailedStrftime => f.write_str("strftime formatting failed"),
            FailedStrptime => f.write_str("strptime parsing failed"),
            FailedWidth => {
                f.write_str("failed to parse conversion specifier width")
            }
            InvalidDate => f.write_str("invalid date"),
            InvalidISOWeekDate => f.write_str("invalid ISO 8601 week date"),
            InvalidWeekdayMonday { got } => write!(
                f,
                "weekday `{got:?}` is not valid for \
                 Monday based week number",
            ),
            InvalidWeekdaySunday { got } => write!(
                f,
                "weekday `{got:?}` is not valid for \
                 Sunday based week number",
            ),
            MissingTimeHourForFractional => f.write_str(
                "parsing format did not include hour directive, \
                 but did include fractional second directive (cannot have \
                 smaller time units with bigger time units missing)",
            ),
            MissingTimeHourForMinute => f.write_str(
                "parsing format did not include hour directive, \
                 but did include minute directive (cannot have \
                 smaller time units with bigger time units missing)",
            ),
            MissingTimeHourForSecond => f.write_str(
                "parsing format did not include hour directive, \
                 but did include second directive (cannot have \
                 smaller time units with bigger time units missing)",
            ),
            MissingTimeMinuteForFractional => f.write_str(
                "parsing format did not include minute directive, \
                 but did include fractional second directive (cannot have \
                 smaller time units with bigger time units missing)",
            ),
            MissingTimeMinuteForSecond => f.write_str(
                "parsing format did not include minute directive, \
                 but did include second directive (cannot have \
                 smaller time units with bigger time units missing)",
            ),
            MissingTimeSecondForFractional => f.write_str(
                "parsing format did not include second directive, \
                 but did include fractional second directive (cannot have \
                 smaller time units with bigger time units missing)",
            ),
            MismatchOffset { parsed, got } => write!(
                f,
                "parsed time zone offset `{parsed}`, but \
                 offset from timestamp and time zone is `{got}`",
            ),
            MismatchWeekday { parsed, got } => write!(
                f,
                "parsed weekday `{parsed:?}` does not match \
                 weekday `{got:?}` from parsed date",
            ),
            RangeTimestamp => f.write_str(
                "parsed datetime and offset, \
                 but combining them into a zoned datetime \
                 is outside Jiff's supported timestamp range",
            ),
            RangeWidth => write!(
                f,
                "parsed width is too big, max is {max}",
                max = u8::MAX
            ),
            RequiredDateForDateTime => {
                f.write_str("date required to parse datetime")
            }
            RequiredDateTimeForTimestamp => {
                f.write_str("datetime required to parse timestamp")
            }
            RequiredDateTimeForZoned => {
                f.write_str("datetime required to parse zoned datetime")
            }
            RequiredOffsetForTimestamp => {
                f.write_str("offset required to parse timestamp")
            }
            RequiredSomeDayForDate => f.write_str(
                "a month/day, day-of-year or week date must be \
                 present to create a date, but none were found",
            ),
            RequiredTimeForDateTime => {
                f.write_str("time required to parse datetime")
            }
            RequiredYearForDate => f.write_str("year required to parse date"),
            #[cfg(feature = "alloc")]
            UnconsumedStrptime { ref remaining } => write!(
                f,
                "strptime expects to consume the entire input, but \
                 `{remaining}` remains unparsed",
                remaining = escape::Bytes(remaining),
            ),
            #[cfg(not(feature = "alloc"))]
            UnconsumedStrptime {} => f.write_str(
                "strptime expects to consume the entire input, but \
                 there is unparsed input remaining",
            ),
            UnexpectedEndAfterDot => f.write_str(
                "invalid format string, expected directive after `%.`",
            ),
            UnexpectedEndAfterPercent => f.write_str(
                "invalid format string, expected byte after `%`, \
                 but found end of format string",
            ),
            UnknownDirective { directive } => write!(
                f,
                "found unrecognized specifier directive `{directive}`",
                directive = escape::Byte(directive),
            ),
            UnknownDirectiveAfterDot { directive } => write!(
                f,
                "found unrecognized specifier directive `{directive}` \
                 following `%.`",
                directive = escape::Byte(directive),
            ),
            ZonedOffsetOrTz => f.write_str(
                "either offset (from `%z`) or IANA time zone identifier \
                 (from `%Q`) is required for parsing zoned datetime",
            ),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum ParseError {
    ExpectedAmPm,
    ExpectedAmPmTooShort,
    ExpectedIanaTz,
    ExpectedIanaTzEndOfInput,
    ExpectedMonthAbbreviation,
    ExpectedMonthAbbreviationTooShort,
    ExpectedWeekdayAbbreviation,
    ExpectedWeekdayAbbreviationTooShort,
    ExpectedChoice {
        available: &'static [&'static [u8]],
    },
    ExpectedFractionalDigit,
    ExpectedMatchLiteralByte {
        expected: u8,
        got: u8,
    },
    ExpectedMatchLiteralEndOfInput {
        expected: u8,
    },
    ExpectedNonEmpty {
        directive: u8,
    },
    #[cfg(not(feature = "alloc"))]
    NotAllowedAlloc {
        directive: u8,
        colons: u8,
    },
    NotAllowedLocaleClockTime,
    NotAllowedLocaleDate,
    NotAllowedLocaleDateAndTime,
    NotAllowedLocaleTwelveHourClockTime,
    NotAllowedTimeZoneAbbreviation,
    ParseDay,
    ParseDayOfYear,
    ParseCentury,
    ParseFractionalSeconds,
    ParseHour,
    ParseIsoWeekNumber,
    ParseIsoWeekYear,
    ParseIsoWeekYearTwoDigit,
    ParseMinute,
    ParseMondayWeekNumber,
    ParseMonth,
    ParseSecond,
    ParseSundayWeekNumber,
    ParseTimestamp,
    ParseWeekdayNumber,
    ParseYear,
    ParseYearTwoDigit,
    UnknownMonthName,
    UnknownWeekdayAbbreviation,
}

impl error::IntoError for ParseError {
    fn into_error(self) -> error::Error {
        self.into()
    }
}

impl From<ParseError> for error::Error {
    #[cold]
    #[inline(never)]
    fn from(err: ParseError) -> error::Error {
        error::ErrorKind::FmtStrtimeParse(err).into()
    }
}

impl core::fmt::Display for ParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::ParseError::*;

        match *self {
            ExpectedAmPm => f.write_str("expected to find `AM` or `PM`"),
            ExpectedAmPmTooShort => f.write_str(
                "expected to find `AM` or `PM`, \
                 but the remaining input is too short \
                 to contain one",
            ),
            ExpectedIanaTz => f.write_str(
                "expected to find the start of an IANA time zone \
                 identifier name or component",
            ),
            ExpectedIanaTzEndOfInput => f.write_str(
                "expected to find the start of an IANA time zone \
                 identifier name or component, \
                 but found end of input instead",
            ),
            ExpectedMonthAbbreviation => {
                f.write_str("expected to find month name abbreviation")
            }
            ExpectedMonthAbbreviationTooShort => f.write_str(
                "expected to find month name abbreviation, \
                 but the remaining input is too short \
                 to contain one",
            ),
            ExpectedWeekdayAbbreviation => {
                f.write_str("expected to find weekday abbreviation")
            }
            ExpectedWeekdayAbbreviationTooShort => f.write_str(
                "expected to find weekday abbreviation, \
                 but the remaining input is too short \
                 to contain one",
            ),
            ExpectedChoice { available } => {
                f.write_str(
                    "failed to find expected value, available choices are: ",
                )?;
                for (i, choice) in available.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{}", escape::Bytes(choice))?;
                }
                Ok(())
            }
            ExpectedFractionalDigit => f.write_str(
                "expected at least one fractional decimal digit, \
                 but did not find any",
            ),
            ExpectedMatchLiteralByte { expected, got } => write!(
                f,
                "expected to match literal byte `{expected}` from \
                 format string, but found byte `{got}` in input",
                expected = escape::Byte(expected),
                got = escape::Byte(got),
            ),
            ExpectedMatchLiteralEndOfInput { expected } => write!(
                f,
                "expected to match literal byte `{expected}` from \
                 format string, but found end of input",
                expected = escape::Byte(expected),
            ),
            ExpectedNonEmpty { directive } => write!(
                f,
                "expected non-empty input for directive `%{directive}`, \
                 but found end of input",
                directive = escape::Byte(directive),
            ),
            #[cfg(not(feature = "alloc"))]
            NotAllowedAlloc { directive, colons } => write!(
                f,
                "cannot parse `%{colons}{directive}` \
                 without Jiff's `alloc` feature enabled",
                colons = escape::RepeatByte { byte: b':', count: colons },
                directive = escape::Byte(directive),
            ),
            NotAllowedLocaleClockTime => {
                f.write_str("parsing locale clock time is not allowed")
            }
            NotAllowedLocaleDate => {
                f.write_str("parsing locale date is not allowed")
            }
            NotAllowedLocaleDateAndTime => {
                f.write_str("parsing locale date and time is not allowed")
            }
            NotAllowedLocaleTwelveHourClockTime => {
                f.write_str("parsing locale 12-hour clock time is not allowed")
            }
            NotAllowedTimeZoneAbbreviation => {
                f.write_str("parsing time zone abbreviation is not allowed")
            }
            ParseCentury => {
                f.write_str("failed to parse year number for century")
            }
            ParseDay => f.write_str("failed to parse day number"),
            ParseDayOfYear => {
                f.write_str("failed to parse day of year number")
            }
            ParseFractionalSeconds => f.write_str(
                "failed to parse fractional second component \
                 (up to 9 digits, nanosecond precision)",
            ),
            ParseHour => f.write_str("failed to parse hour number"),
            ParseMinute => f.write_str("failed to parse minute number"),
            ParseWeekdayNumber => {
                f.write_str("failed to parse weekday number")
            }
            ParseIsoWeekNumber => {
                f.write_str("failed to parse ISO 8601 week number")
            }
            ParseIsoWeekYear => {
                f.write_str("failed to parse ISO 8601 week year")
            }
            ParseIsoWeekYearTwoDigit => {
                f.write_str("failed to parse 2-digit ISO 8601 week year")
            }
            ParseMondayWeekNumber => {
                f.write_str("failed to parse Monday-based week number")
            }
            ParseMonth => f.write_str("failed to parse month number"),
            ParseSecond => f.write_str("failed to parse second number"),
            ParseSundayWeekNumber => {
                f.write_str("failed to parse Sunday-based week number")
            }
            ParseTimestamp => {
                f.write_str("failed to parse Unix timestamp (in seconds)")
            }
            ParseYear => f.write_str("failed to parse year"),
            ParseYearTwoDigit => f.write_str("failed to parse 2-digit year"),
            UnknownMonthName => f.write_str("unrecognized month name"),
            UnknownWeekdayAbbreviation => {
                f.write_str("unrecognized weekday abbreviation")
            }
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum FormatError {
    RequiresDate,
    RequiresInstant,
    RequiresOffset,
    RequiresTime,
    RequiresTimeZone,
    RequiresTimeZoneOrOffset,
    InvalidUtf8,
    ZeroPrecisionFloat,
    ZeroPrecisionNano,
}

impl error::IntoError for FormatError {
    fn into_error(self) -> error::Error {
        self.into()
    }
}

impl From<FormatError> for error::Error {
    #[cold]
    #[inline(never)]
    fn from(err: FormatError) -> error::Error {
        error::ErrorKind::FmtStrtimeFormat(err).into()
    }
}

impl core::fmt::Display for FormatError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::FormatError::*;

        match *self {
            RequiresDate => f.write_str("requires date to format"),
            RequiresInstant => f.write_str(
                "requires instant (a timestamp or a date, time and offset)",
            ),
            RequiresTime => f.write_str("requires time to format"),
            RequiresOffset => f.write_str("requires time zone offset"),
            RequiresTimeZone => {
                f.write_str("requires IANA time zone identifier")
            }
            RequiresTimeZoneOrOffset => f.write_str(
                "requires IANA time zone identifier or \
                 time zone offset, but neither were present",
            ),
            InvalidUtf8 => {
                f.write_str("invalid format string, it must be valid UTF-8")
            }
            ZeroPrecisionFloat => {
                f.write_str("zero precision with %f is not allowed")
            }
            ZeroPrecisionNano => {
                f.write_str("zero precision with %N is not allowed")
            }
        }
    }
}
