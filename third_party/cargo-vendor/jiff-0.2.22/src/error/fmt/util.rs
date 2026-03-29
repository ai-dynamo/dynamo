use crate::{error, Unit};

#[derive(Clone, Debug)]
pub(crate) enum Error {
    ConversionToSecondsFailed { unit: Unit },
    EmptyDuration,
    FailedValueSet { unit: Unit },
    InvalidFraction,
    InvalidFractionNanos,
    MissingFractionalDigits,
    NotAllowedCalendarUnit { unit: Unit },
    NotAllowedFractionalUnit { found: Unit },
    NotAllowedNegative,
    OutOfOrderHMS { found: Unit },
    OutOfOrderUnits { found: Unit, previous: Unit },
    OverflowForUnit { unit: Unit },
    OverflowForUnitFractional { unit: Unit },
    SignedOverflowForUnit { unit: Unit },
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
        error::ErrorKind::FmtUtil(err).into()
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::Error::*;

        match *self {
            ConversionToSecondsFailed { unit } => write!(
                f,
                "converting {unit} to seconds overflows \
                 a signed 64-bit integer",
                unit = unit.plural(),
            ),
            EmptyDuration => f.write_str("no parsed duration components"),
            FailedValueSet { unit } => write!(
                f,
                "failed to set value for {unit} unit on span",
                unit = unit.singular(),
            ),
            InvalidFraction => f.write_str(
                "failed to parse fractional component \
                 (up to 9 digits, nanosecond precision is allowed)",
            ),
            InvalidFractionNanos => f.write_str(
                "failed to set nanosecond value from fractional component",
            ),
            MissingFractionalDigits => f.write_str(
                "found decimal after seconds component, \
                 but did not find any digits after decimal",
            ),
            NotAllowedCalendarUnit { unit } => write!(
                f,
                "parsing calendar units ({unit} in this case) \
                 in this context is not supported \
                 (perhaps try parsing into a `jiff::Span` instead)",
                unit = unit.plural(),
            ),
            NotAllowedFractionalUnit { found } => write!(
                f,
                "fractional {found} are not supported",
                found = found.plural(),
            ),
            NotAllowedNegative => f.write_str(
                "cannot parse negative duration into unsigned \
                 `std::time::Duration`",
            ),
            OutOfOrderHMS { found } => write!(
                f,
                "found `HH:MM:SS` after unit {found}, \
                 but `HH:MM:SS` can only appear after \
                 years, months, weeks or days",
                found = found.singular(),
            ),
            OutOfOrderUnits { found, previous } => write!(
                f,
                "found value with unit {found} \
                 after unit {previous}, but units must be \
                 written from largest to smallest \
                 (and they can't be repeated)",
                found = found.singular(),
                previous = previous.singular(),
            ),
            OverflowForUnit { unit } => write!(
                f,
                "accumulated duration \
                 overflowed when adding value to unit {unit}",
                unit = unit.singular(),
            ),
            OverflowForUnitFractional { unit } => write!(
                f,
                "accumulated duration \
                 overflowed when adding fractional value to unit {unit}",
                unit = unit.singular(),
            ),
            SignedOverflowForUnit { unit } => write!(
                f,
                "value for {unit} is too big (or small) to fit into \
                 a signed 64-bit integer",
                unit = unit.plural(),
            ),
        }
    }
}
