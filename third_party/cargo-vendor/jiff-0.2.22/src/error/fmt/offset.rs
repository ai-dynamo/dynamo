use crate::{error, util::escape};

#[derive(Clone, Debug)]
pub(crate) enum Error {
    ColonAfterHours,
    EndOfInput,
    EndOfInputHour,
    EndOfInputMinute,
    EndOfInputNumeric,
    EndOfInputSecond,
    InvalidHours,
    InvalidMinutes,
    InvalidSeconds,
    InvalidSecondsFractional,
    InvalidSign,
    InvalidSignPlusOrMinus,
    MissingMinuteAfterHour,
    MissingSecondAfterMinute,
    NoColonAfterHours,
    ParseHours,
    ParseMinutes,
    ParseSeconds,
    PrecisionLoss,
    SeparatorAfterHours,
    SeparatorAfterMinutes,
    SubminutePrecisionNotEnabled,
    SubsecondPrecisionNotEnabled,
    UnexpectedLetterOffsetNoZulu(u8),
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
        error::ErrorKind::FmtOffset(err).into()
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::Error::*;

        match *self {
            ColonAfterHours => f.write_str(
                "parsed hour component of time zone offset, \
                 but found colon after hours which is not allowed",
            ),
            EndOfInput => {
                f.write_str("expected UTC offset, but found end of input")
            }
            EndOfInputHour => f.write_str(
                "expected two digit hour after sign, but found end of input",
            ),
            EndOfInputMinute => f.write_str(
                "expected two digit minute after hours, \
                 but found end of input",
            ),
            EndOfInputNumeric => f.write_str(
                "expected UTC numeric offset, but found end of input",
            ),
            EndOfInputSecond => f.write_str(
                "expected two digit second after minutes, \
                 but found end of input",
            ),
            InvalidHours => {
                f.write_str("failed to parse hours in UTC numeric offset")
            }
            InvalidMinutes => {
                f.write_str("failed to parse minutes in UTC numeric offset")
            }
            InvalidSeconds => {
                f.write_str("failed to parse seconds in UTC numeric offset")
            }
            InvalidSecondsFractional => f.write_str(
                "failed to parse fractional seconds in UTC numeric offset",
            ),
            InvalidSign => {
                f.write_str("failed to parse sign in UTC numeric offset")
            }
            InvalidSignPlusOrMinus => f.write_str(
                "expected `+` or `-` sign at start of UTC numeric offset",
            ),
            MissingMinuteAfterHour => f.write_str(
                "parsed hour component of time zone offset, \
                 but could not find required minute component",
            ),
            MissingSecondAfterMinute => f.write_str(
                "parsed hour and minute components of time zone offset, \
                 but could not find required second component",
            ),
            NoColonAfterHours => f.write_str(
                "parsed hour component of time zone offset, \
                 but could not find required colon separator",
            ),
            ParseHours => f.write_str(
                "failed to parse hours (requires a two digit integer)",
            ),
            ParseMinutes => f.write_str(
                "failed to parse minutes (requires a two digit integer)",
            ),
            ParseSeconds => f.write_str(
                "failed to parse seconds (requires a two digit integer)",
            ),
            PrecisionLoss => f.write_str(
                "due to precision loss from fractional seconds, \
                 time zone offset is rounded to a value that is out of bounds",
            ),
            SeparatorAfterHours => f.write_str(
                "failed to parse separator after hours in \
                 UTC numeric offset",
            ),
            SeparatorAfterMinutes => f.write_str(
                "failed to parse separator after minutes in \
                 UTC numeric offset",
            ),
            SubminutePrecisionNotEnabled => f.write_str(
                "subminute precision for UTC numeric offset \
                 is not enabled in this context (must provide only \
                 integral minutes)",
            ),
            SubsecondPrecisionNotEnabled => f.write_str(
                "subsecond precision for UTC numeric offset \
                 is not enabled in this context (must provide only \
                 integral minutes or seconds)",
            ),
            UnexpectedLetterOffsetNoZulu(byte) => write!(
                f,
                "found `{z}` where a numeric UTC offset \
                 was expected (this context does not permit \
                 the Zulu offset)",
                z = escape::Byte(byte),
            ),
        }
    }
}
