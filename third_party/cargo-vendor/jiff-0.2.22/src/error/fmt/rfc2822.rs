use crate::{civil::Weekday, error, util::escape};

#[derive(Clone, Debug)]
pub(crate) enum Error {
    CommentClosingParenWithoutOpen,
    CommentOpeningParenWithoutClose,
    CommentTooManyNestedParens,
    EndOfInputDay,
    Empty,
    EmptyAfterWhitespace,
    EndOfInputComma,
    EndOfInputHour,
    EndOfInputMinute,
    EndOfInputMonth,
    EndOfInputOffset,
    EndOfInputSecond,
    EndOfInputTimeSeparator,
    FailedTimestamp,
    FailedZoned,
    InconsistentWeekday { parsed: Weekday, from_date: Weekday },
    InvalidDate,
    InvalidMonth,
    InvalidObsoleteOffset,
    InvalidWeekday { got_non_digit: u8 },
    NegativeYear,
    ParseDay,
    ParseHour,
    ParseMinute,
    ParseOffsetHour,
    ParseOffsetMinute,
    ParseSecond,
    ParseYear,
    TooShortMonth { len: u8 },
    TooShortOffset,
    TooShortWeekday { got_non_digit: u8, len: u8 },
    TooShortYear { len: u8 },
    UnexpectedByteComma { byte: u8 },
    UnexpectedByteTimeSeparator { byte: u8 },
    WhitespaceAfterDay,
    WhitespaceAfterMonth,
    WhitespaceAfterTime,
    WhitespaceAfterTimeForObsoleteOffset,
    WhitespaceAfterYear,
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
        error::ErrorKind::FmtRfc2822(err).into()
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::Error::*;

        match *self {
            CommentClosingParenWithoutOpen => f.write_str(
                "found closing parenthesis in comment with \
                 no matching opening parenthesis",
            ),
            CommentOpeningParenWithoutClose => f.write_str(
                "found opening parenthesis in comment with \
                 no matching closing parenthesis",
            ),
            CommentTooManyNestedParens => {
                f.write_str("found too many nested parenthesis in comment")
            }
            Empty => {
                f.write_str("expected RFC 2822 datetime, but got empty string")
            }
            EmptyAfterWhitespace => f.write_str(
                "expected RFC 2822 datetime, but got empty string \
                 after trimming leading whitespace",
            ),
            EndOfInputComma => f.write_str(
                "expected comma after parsed weekday in \
                 RFC 2822 datetime, but found end of input instead",
            ),
            EndOfInputDay => {
                f.write_str("expected numeric day, but found end of input")
            }
            EndOfInputHour => {
                f.write_str("expected two digit hour, but found end of input")
            }
            EndOfInputMinute => f.write_str(
                "expected two digit minute, but found end of input",
            ),
            EndOfInputMonth => f.write_str(
                "expected abbreviated month name, but found end of input",
            ),
            EndOfInputOffset => f.write_str(
                "expected sign for time zone offset, \
                 (or a legacy time zone name abbreviation), \
                 but found end of input",
            ),
            EndOfInputSecond => f.write_str(
                "expected two digit second, but found end of input",
            ),
            EndOfInputTimeSeparator => f.write_str(
                "expected time separator of `:`, but found end of input",
            ),
            FailedTimestamp => f.write_str(
                "failed to parse RFC 2822 datetime into Jiff timestamp",
            ),
            FailedZoned => f.write_str(
                "failed to parse RFC 2822 datetime into Jiff zoned datetime",
            ),
            InconsistentWeekday { parsed, from_date } => write!(
                f,
                "found parsed weekday of `{parsed:?}`, \
                 but parsed datetime has weekday `{from_date:?}`",
            ),
            InvalidDate => f.write_str("invalid date"),
            InvalidMonth => f.write_str(
                "expected abbreviated month name, \
                 but did not recognize a valid abbreviated month name",
            ),
            InvalidObsoleteOffset => f.write_str(
                "expected obsolete RFC 2822 time zone abbreviation, \
                 but did not recognize a valid abbreviation",
            ),
            InvalidWeekday { got_non_digit } => write!(
                f,
                "expected day at beginning of RFC 2822 datetime \
                 since first non-whitespace byte, `{first}`, \
                 is not a digit, but did not recognize a valid \
                 weekday abbreviation",
                first = escape::Byte(got_non_digit),
            ),
            NegativeYear => f.write_str(
                "datetime has negative year, \
                 which cannot be formatted with RFC 2822",
            ),
            ParseDay => f.write_str("failed to parse day"),
            ParseHour => f.write_str(
                "failed to parse hour (expects a two digit integer)",
            ),
            ParseMinute => f.write_str(
                "failed to parse minute (expects a two digit integer)",
            ),
            ParseOffsetHour => {
                f.write_str("failed to parse hours from time zone offset")
            }
            ParseOffsetMinute => {
                f.write_str("failed to parse minutes from time zone offset")
            }
            ParseSecond => f.write_str(
                "failed to parse second (expects a two digit integer)",
            ),
            ParseYear => f.write_str(
                "failed to parse year \
                 (expects a two, three or four digit integer)",
            ),
            TooShortMonth { len } => write!(
                f,
                "expected abbreviated month name, but remaining input \
                 is too short (remaining bytes is {len})",
            ),
            TooShortOffset => write!(
                f,
                "expected at least four digits for time zone offset \
                 after sign, but found fewer than four bytes remaining",
            ),
            TooShortWeekday { got_non_digit, len } => write!(
                f,
                "expected day at beginning of RFC 2822 datetime \
                 since first non-whitespace byte, `{first}`, \
                 is not a digit, but given string is too short \
                 (length is {len})",
                first = escape::Byte(got_non_digit),
            ),
            TooShortYear { len } => write!(
                f,
                "expected at least two ASCII digits for parsing \
                 a year, but only found {len}",
            ),
            UnexpectedByteComma { byte } => write!(
                f,
                "expected comma after parsed weekday in \
                 RFC 2822 datetime, but found `{got}` instead",
                got = escape::Byte(byte),
            ),
            UnexpectedByteTimeSeparator { byte } => write!(
                f,
                "expected time separator of `:`, but found `{got}`",
                got = escape::Byte(byte),
            ),
            WhitespaceAfterDay => {
                f.write_str("expected whitespace after parsing day")
            }
            WhitespaceAfterMonth => f.write_str(
                "expected whitespace after parsing abbreviated month name",
            ),
            WhitespaceAfterTime => f.write_str(
                "expected whitespace after parsing time: \
                 expected at least one whitespace character \
                 (space or tab), but found none",
            ),
            WhitespaceAfterTimeForObsoleteOffset => f.write_str(
                "expected obsolete RFC 2822 time zone abbreviation, \
                 but found no remaining non-whitespace characters \
                 after time",
            ),
            WhitespaceAfterYear => {
                f.write_str("expected whitespace after parsing year")
            }
        }
    }
}
