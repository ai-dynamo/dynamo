use crate::{error, util::escape};

#[derive(Clone, Debug)]
pub(crate) enum Error {
    Empty,
    ExpectedColonAfterMinute,
    ExpectedIntegerAfterSign,
    ExpectedMinuteAfterHour,
    ExpectedOneMoreUnitAfterComma,
    ExpectedOneSign,
    ExpectedSecondAfterMinute,
    ExpectedUnitSuffix,
    ExpectedWhitespaceAfterComma { byte: u8 },
    ExpectedWhitespaceAfterCommaEndOfInput,
    Failed,
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
        error::ErrorKind::FmtFriendly(err).into()
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::Error::*;

        match *self {
            Empty => f.write_str("an empty string is not valid"),
            ExpectedColonAfterMinute => f.write_str(
                "when parsing the `HH:MM:SS` format, \
                 expected to parse `:` following minute",
            ),
            ExpectedIntegerAfterSign => f.write_str(
                "expected duration to start \
                 with a unit value (a decimal integer) after an \
                 optional sign, but no integer was found",
            ),
            ExpectedMinuteAfterHour => f.write_str(
                "when parsing the `HH:MM:SS` format, \
                 expected to parse minute following hour",
            ),
            ExpectedOneMoreUnitAfterComma => f.write_str(
                "found comma at the end of duration, \
                 but a comma indicates at least one more \
                 unit follows",
            ),
            ExpectedOneSign => f.write_str(
                "expected to find either a prefix sign (+/-) or \
                 a suffix sign (`ago`), but found both",
            ),
            ExpectedSecondAfterMinute => f.write_str(
                "when parsing the `HH:MM:SS` format, \
                 expected to parse second following minute",
            ),
            ExpectedUnitSuffix => f.write_str(
                "expected to find unit designator suffix \
                 (e.g., `years` or `secs`) after parsing \
                 integer",
            ),
            ExpectedWhitespaceAfterComma { byte } => write!(
                f,
                "expected whitespace after comma, but found `{byte}`",
                byte = escape::Byte(byte),
            ),
            ExpectedWhitespaceAfterCommaEndOfInput => f.write_str(
                "expected whitespace after comma, but found end of input",
            ),
            Failed => f.write_str(
                "failed to parse input in the \"friendly\" duration format",
            ),
        }
    }
}
