#[cfg(not(test))]
pub(crate) use self::disabled::*;
#[cfg(test)]
pub(crate) use self::enabled::*;

#[cfg(not(test))]
mod disabled {
    #[derive(Clone, Debug)]
    pub(crate) enum Error {}

    impl core::fmt::Display for Error {
        fn fmt(&self, _: &mut core::fmt::Formatter) -> core::fmt::Result {
            unreachable!()
        }
    }
}

#[cfg(test)]
mod enabled {
    use alloc::boxed::Box;

    use crate::error;

    // `man zic` says that the max line length including the line
    // terminator is 2048. The `core::str::Lines` iterator doesn't include
    // the terminator, so we subtract 1 to account for that. Note that this
    // could potentially allow one extra byte in the case of a \r\n line
    // terminator, but this seems fine.
    pub(crate) const MAX_LINE_LEN: usize = 2047;

    #[derive(Clone, Debug)]
    pub(crate) enum Error {
        DuplicateLink { name: Box<str> },
        DuplicateLinkZone { name: Box<str> },
        DuplicateZone { name: Box<str> },
        DuplicateZoneLink { name: Box<str> },
        ExpectedCloseQuote,
        ExpectedColonAfterHour,
        ExpectedColonAfterMinute,
        ExpectedContinuationZoneThreeFields,
        ExpectedContinuationZoneLine { name: Box<str> },
        ExpectedDotAfterSeconds,
        ExpectedFirstZoneFourFields,
        ExpectedLinkTwoFields,
        ExpectedMinuteAfterHours,
        ExpectedNameBegin,
        ExpectedNanosecondDigits,
        ExpectedNonEmptyAbbreviation,
        ExpectedNonEmptyAt,
        ExpectedNonEmptyName,
        ExpectedNonEmptySave,
        ExpectedNonEmptyZoneName,
        ExpectedNothingAfterTime,
        ExpectedRuleNineFields { got: usize },
        ExpectedSecondAfterMinutes,
        ExpectedTimeOneHour,
        ExpectedUntilYear,
        ExpectedWhitespaceAfterQuotedField,
        ExpectedZoneNameComponentNoDots { component: Box<str> },
        FailedContinuationZone,
        FailedLinkLine,
        FailedParseDay,
        FailedParseFieldAt,
        FailedParseFieldFormat,
        FailedParseFieldFrom,
        FailedParseFieldIn,
        FailedParseFieldLetters,
        FailedParseFieldLinkName,
        FailedParseFieldLinkTarget,
        FailedParseFieldName,
        FailedParseFieldOn,
        FailedParseFieldRules,
        FailedParseFieldSave,
        FailedParseFieldStdOff,
        FailedParseFieldTo,
        FailedParseFieldUntil,
        FailedParseHour,
        FailedParseMinute,
        FailedParseMonth,
        FailedParseNanosecond,
        FailedParseSecond,
        FailedParseTimeDuration,
        FailedParseYear,
        FailedRule { name: Box<str> },
        FailedRuleLine,
        FailedZoneFirst,
        Line { number: usize },
        LineMaxLength,
        LineNul,
        LineOverflow,
        InvalidAbbreviation,
        InvalidRuleYear { start: i16, end: i16 },
        InvalidUtf8,
        UnrecognizedAtTimeSuffix,
        UnrecognizedDayOfMonthFormat,
        UnrecognizedDayOfWeek,
        UnrecognizedMonthName,
        UnrecognizedSaveTimeSuffix,
        UnrecognizedTrailingTimeDuration,
        UnrecognizedZicLine,
    }

    impl From<Error> for error::Error {
        #[cold]
        #[inline(never)]
        fn from(err: Error) -> error::Error {
            error::ErrorKind::TzZic(err).into()
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
                DuplicateLink { ref name } => {
                    write!(f, "found duplicate link with name `{name}`")
                }
                DuplicateLinkZone { ref name } => write!(
                    f,
                    "found link with name `{name}` that conflicts \
                     with a zone of the same name",
                ),
                DuplicateZone { ref name } => {
                    write!(f, "found duplicate zone with name `{name}`")
                }
                DuplicateZoneLink { ref name } => write!(
                    f,
                    "found zone with name `{name}` that conflicts \
                     with a link of the same name",
                ),
                ExpectedCloseQuote => {
                    f.write_str("found unclosed quote for field")
                }
                ExpectedColonAfterHour => {
                    f.write_str("expected `:` after hours")
                }
                ExpectedColonAfterMinute => {
                    f.write_str("expected `:` after minutes")
                }
                ExpectedContinuationZoneLine { ref name } => write!(
                    f,
                    "expected continuation zone line for `{name}`, \
                     but found end of data instead",
                ),
                ExpectedContinuationZoneThreeFields => f.write_str(
                    "expected continuation `ZONE` line \
                     to have at least 3 fields",
                ),
                ExpectedDotAfterSeconds => {
                    f.write_str("expected `.` after seconds")
                }
                ExpectedFirstZoneFourFields => f.write_str(
                    "expected first `ZONE` line to have at least 4 fields",
                ),
                ExpectedLinkTwoFields => {
                    f.write_str("expected exactly 2 fields after `LINK`")
                }
                ExpectedMinuteAfterHours => {
                    f.write_str("expected minute digits after `HH:`")
                }
                ExpectedNameBegin => f.write_str(
                    "`NAME` field cannot begin with a digit, `+` or `-`, \
                     but found `NAME` that begins with one of those",
                ),
                ExpectedNanosecondDigits => {
                    f.write_str("expected nanosecond digits after `HH:MM:SS.`")
                }
                ExpectedNonEmptyAbbreviation => f.write_str(
                    "empty time zone abbreviations are not allowed",
                ),
                ExpectedNonEmptyAt => {
                    f.write_str("`AT` field for rule cannot be empty")
                }
                ExpectedNonEmptyName => {
                    f.write_str("`NAME` field for rule cannot be empty")
                }
                ExpectedNonEmptySave => {
                    f.write_str("`SAVE` field for rule cannot be empty")
                }
                ExpectedNonEmptyZoneName => {
                    f.write_str("zone names cannot be empty")
                }
                ExpectedNothingAfterTime => f.write_str(
                    "expected no more fields after time of day, \
                     but found at least one",
                ),
                ExpectedRuleNineFields { got } => write!(
                    f,
                    "expected exactly 9 fields for rule, \
                     but found {got} fields",
                ),
                ExpectedSecondAfterMinutes => {
                    f.write_str("expected second digits after `HH:MM:`")
                }
                ExpectedTimeOneHour => f.write_str(
                    "expected time duration to contain \
                     at least one hour digit",
                ),
                ExpectedUntilYear => f.write_str("expected at least a year"),
                ExpectedWhitespaceAfterQuotedField => {
                    f.write_str("expected whitespace after quoted field")
                }
                ExpectedZoneNameComponentNoDots { ref component } => write!(
                    f,
                    "component `{component}` in zone name cannot \
                     be \".\" or \"..\"",
                ),
                FailedContinuationZone => {
                    f.write_str("failed to parse continuation `Zone` line")
                }
                FailedLinkLine => f.write_str("failed to parse `Link` line"),
                FailedParseDay => f.write_str("failed to parse day"),
                FailedParseFieldAt => {
                    f.write_str("failed to parse `NAME` field")
                }
                FailedParseFieldFormat => {
                    f.write_str("failed to parse `FORMAT` field")
                }
                FailedParseFieldFrom => {
                    f.write_str("failed to parse `FROM` field")
                }
                FailedParseFieldIn => {
                    f.write_str("failed to parse `IN` field")
                }
                FailedParseFieldLetters => {
                    f.write_str("failed to parse `LETTERS` field")
                }
                FailedParseFieldLinkName => {
                    f.write_str("failed to parse `LINK` name field")
                }
                FailedParseFieldLinkTarget => {
                    f.write_str("failed to parse `LINK` target field")
                }
                FailedParseFieldName => {
                    f.write_str("failed to parse `NAME` field")
                }
                FailedParseFieldOn => {
                    f.write_str("failed to parse `ON` field")
                }
                FailedParseFieldRules => {
                    f.write_str("failed to parse `RULES` field")
                }
                FailedParseFieldSave => {
                    f.write_str("failed to parse `SAVE` field")
                }
                FailedParseFieldStdOff => {
                    f.write_str("failed to parse `STDOFF` field")
                }
                FailedParseFieldTo => {
                    f.write_str("failed to parse `TO` field")
                }
                FailedParseFieldUntil => {
                    f.write_str("failed to parse `UNTIL` field")
                }
                FailedParseHour => f.write_str("failed to parse hour"),
                FailedParseMinute => f.write_str("failed to parse minute"),
                FailedParseMonth => f.write_str("failed to parse month"),
                FailedParseNanosecond => {
                    f.write_str("failed to parse nanosecond")
                }
                FailedParseSecond => f.write_str("failed to parse second"),
                FailedParseTimeDuration => {
                    f.write_str("failed to parse time duration")
                }
                FailedParseYear => f.write_str("failed to parse year"),
                FailedRule { name: ref rule } => {
                    write!(f, "failed to parse rule `{rule}`")
                }
                FailedRuleLine => f.write_str("failed to parse `Rule` line"),
                FailedZoneFirst => {
                    f.write_str("failed to parse first `Zone` line")
                }
                InvalidAbbreviation => f.write_str(
                    "time zone abbreviation \
                     contains invalid character; only \"+\", \"-\" and \
                     ASCII alpha-numeric characters are allowed",
                ),
                InvalidRuleYear { start, end } => write!(
                    f,
                    "found start year={start} \
                     to be greater than end year={end}"
                ),
                InvalidUtf8 => f.write_str("invalid UTF-8"),
                Line { number } => write!(f, "line {number}"),
                LineMaxLength => write!(
                    f,
                    "found line with length that exceeds \
                     max length of {MAX_LINE_LEN}",
                ),
                LineNul => f.write_str(
                    "found line with NUL byte, which isn't allowed",
                ),
                LineOverflow => f.write_str("line count overflowed"),
                UnrecognizedAtTimeSuffix => {
                    f.write_str("unrecognized `AT` time suffix")
                }
                UnrecognizedDayOfMonthFormat => {
                    f.write_str("unrecognized format for day-of-month")
                }
                UnrecognizedDayOfWeek => {
                    f.write_str("unrecognized day of the week")
                }
                UnrecognizedMonthName => {
                    f.write_str("unrecognized month name")
                }
                UnrecognizedSaveTimeSuffix => {
                    f.write_str("unrecognized `SAVE` time suffix")
                }
                UnrecognizedTrailingTimeDuration => {
                    f.write_str("found unrecognized suffix in time duration")
                }
                UnrecognizedZicLine => f.write_str("unrecognized zic line"),
            }
        }
    }
}
