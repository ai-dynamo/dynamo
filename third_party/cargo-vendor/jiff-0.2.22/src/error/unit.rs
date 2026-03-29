use crate::{error, util::b, Unit};

#[derive(Clone, Debug)]
pub(crate) enum UnitConfigError {
    CalendarUnitsNotAllowed { unit: Unit },
    CivilDate { given: Unit },
    CivilTime { given: Unit },
    IncrementDivide { unit: Unit, must_divide: MustDivide },
    LargestSmallerThanSmallest { smallest: Unit, largest: Unit },
    RelativeWeekOrDay { unit: Unit },
    RelativeYearOrMonth { unit: Unit },
    RelativeYearOrMonthGivenDaysAre24Hours { unit: Unit },
    RoundToUnitUnsupported { unit: Unit },
    SignedDurationCalendarUnit { smallest: Unit },
    TimeZoneOffset { given: Unit },
}

impl From<UnitConfigError> for error::Error {
    #[cold]
    #[inline(never)]
    fn from(err: UnitConfigError) -> error::Error {
        error::ErrorKind::UnitConfig(err).into()
    }
}

impl error::IntoError for UnitConfigError {
    fn into_error(self) -> error::Error {
        self.into()
    }
}

impl core::fmt::Display for UnitConfigError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::UnitConfigError::*;

        match *self {
            CalendarUnitsNotAllowed { unit } => write!(
                f,
                "operation can only be performed with units of hours \
                 or smaller, but found non-zero '{unit}' units \
                 (operations on `jiff::Timestamp`, `jiff::tz::Offset` \
                  and `jiff::civil::Time` don't support calendar \
                  units in a `jiff::Span`)",
                unit = unit.singular(),
            ),
            CivilDate { given } => write!(
                f,
                "'{unit}' not allowed as a date unit for rounding span",
                unit = given.singular()
            ),
            CivilTime { given } => write!(
                f,
                "'{unit}' not allowed as a time unit for rounding span",
                unit = given.singular()
            ),
            IncrementDivide { unit, must_divide: MustDivide::Days } => write!(
                f,
                "increment for rounding to '{unit}' \
                 must be equal to `1`",
                unit = unit.plural(),
            ),
            IncrementDivide { unit, must_divide } => write!(
                f,
                "increment for rounding to '{unit}' \
                 must divide into `{must_divide}` evenly",
                unit = unit.plural(),
                must_divide = must_divide.as_i64(),
            ),
            LargestSmallerThanSmallest { smallest, largest } => {
                write!(
                    f,
                    "largest unit ('{largest}') cannot be smaller than \
                     smallest unit ('{smallest}')",
                    largest = largest.singular(),
                    smallest = smallest.singular(),
                )
            }
            RelativeWeekOrDay { unit } => write!(
                f,
                "using unit '{unit}' in a span or configuration \
                 requires that either a relative reference time be given \
                 or `jiff::SpanRelativeTo::days_are_24_hours()` is used to \
                 indicate invariant 24-hour days, \
                 but neither were provided",
                unit = unit.singular(),
            ),
            RelativeYearOrMonth { unit } => write!(
                f,
                "using unit '{unit}' in a span or configuration \
                 requires that a relative reference time be given, \
                 but none was provided",
                unit = unit.singular(),
            ),
            RelativeYearOrMonthGivenDaysAre24Hours { unit } => write!(
                f,
                "using unit '{unit}' in span or configuration \
                 requires that a relative reference time be given \
                 (`jiff::SpanRelativeTo::days_are_24_hours()` was given \
                 but this only permits using days and weeks \
                 without a relative reference time)",
                unit = unit.singular(),
            ),
            RoundToUnitUnsupported { unit } => write!(
                f,
                "rounding to '{unit}' is not supported",
                unit = unit.plural(),
            ),
            SignedDurationCalendarUnit { smallest } => write!(
                f,
                "rounding `jiff::SignedDuration` failed because \
                 the smallest unit provided, '{plural}', is a calendar unit \
                 (to round by calendar units, you must use a `jiff::Span`)",
                plural = smallest.plural(),
            ),
            TimeZoneOffset { given } => write!(
                f,
                "'{unit}' not allowed when rounding time zone offset \
                 (must use hours, minutes or seconds)",
                unit = given.singular(),
            ),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum MustDivide {
    NanosPerCivilDay,
    MicrosPerCivilDay,
    MillisPerCivilDay,
    SecsPerCivilDay,
    MinsPerCivilDay,
    HoursPerCivilDay,
    NanosPerMicro,
    MicrosPerMilli,
    MillisPerSec,
    SecsPerMin,
    MinsPerHour,
    // A weird case because we require that the rounding increment, when
    // rounding to the nearest day for a date or datetime, is always `1`.
    Days,
}

impl MustDivide {
    pub(crate) fn as_i64(&self) -> i64 {
        use self::MustDivide::*;

        match *self {
            NanosPerCivilDay => b::NANOS_PER_CIVIL_DAY,
            MicrosPerCivilDay => b::MICROS_PER_CIVIL_DAY,
            MillisPerCivilDay => b::MILLIS_PER_CIVIL_DAY,
            SecsPerCivilDay => b::SECS_PER_CIVIL_DAY,
            MinsPerCivilDay => b::MINS_PER_CIVIL_DAY,
            HoursPerCivilDay => b::HOURS_PER_CIVIL_DAY,
            NanosPerMicro => b::NANOS_PER_MICRO,
            MicrosPerMilli => b::MICROS_PER_MILLI,
            MillisPerSec => b::MILLIS_PER_SEC,
            SecsPerMin => b::SECS_PER_MIN,
            MinsPerHour => b::MINS_PER_HOUR,
            Days => 2,
        }
    }
}
