/*!
A module for constants and various base utilities.

This module is a work-in-progress that may lead to helping us move off of
ranged integers. I'm not quite sure where this will go.
*/

#![allow(dead_code)]

use crate::{Error, SignedDuration};

pub(crate) const DAYS_PER_WEEK: i64 = 7;
pub(crate) const HOURS_PER_CIVIL_DAY: i64 = 24;
pub(crate) const MINS_PER_CIVIL_DAY: i64 = HOURS_PER_CIVIL_DAY * MINS_PER_HOUR;
pub(crate) const MINS_PER_HOUR: i64 = 60;
pub(crate) const SECS_PER_WEEK: i64 = DAYS_PER_WEEK * SECS_PER_CIVIL_DAY;
pub(crate) const SECS_PER_CIVIL_DAY: i64 = HOURS_PER_CIVIL_DAY * SECS_PER_HOUR;
pub(crate) const SECS_PER_HOUR: i64 = SECS_PER_MIN * MINS_PER_HOUR;
pub(crate) const SECS_PER_MIN: i64 = 60;
pub(crate) const MILLIS_PER_CIVIL_DAY: i64 =
    SECS_PER_CIVIL_DAY * MILLIS_PER_SEC;
pub(crate) const MILLIS_PER_SEC: i64 = 1_000;
pub(crate) const MICROS_PER_CIVIL_DAY: i64 =
    SECS_PER_CIVIL_DAY * MICROS_PER_SEC;
pub(crate) const MICROS_PER_SEC: i64 = MILLIS_PER_SEC * MICROS_PER_MILLI;
pub(crate) const MICROS_PER_MILLI: i64 = 1_000;
pub(crate) const NANOS_PER_WEEK: i64 = DAYS_PER_WEEK * NANOS_PER_CIVIL_DAY;
pub(crate) const NANOS_PER_CIVIL_DAY: i64 =
    HOURS_PER_CIVIL_DAY * NANOS_PER_HOUR;
pub(crate) const NANOS_PER_HOUR: i64 = MINS_PER_HOUR * NANOS_PER_MIN;
pub(crate) const NANOS_PER_MIN: i64 = SECS_PER_MIN * NANOS_PER_SEC;
pub(crate) const NANOS_PER_SEC: i64 = MILLIS_PER_SEC * NANOS_PER_MILLI;
pub(crate) const NANOS_PER_MILLI: i64 = MICROS_PER_MILLI * NANOS_PER_MICRO;
pub(crate) const NANOS_PER_MICRO: i64 = 1_000;

pub(crate) const DAYS_PER_WEEK_32: i32 = 7;
pub(crate) const HOURS_PER_CIVIL_DAY_32: i32 = 24;
pub(crate) const MINS_PER_HOUR_32: i32 = 60;
pub(crate) const SECS_PER_CIVIL_DAY_32: i32 =
    HOURS_PER_CIVIL_DAY_32 * SECS_PER_HOUR_32;
pub(crate) const SECS_PER_HOUR_32: i32 = SECS_PER_MIN_32 * MINS_PER_HOUR_32;
pub(crate) const SECS_PER_MIN_32: i32 = 60;
pub(crate) const MILLIS_PER_SEC_32: i32 = 1_000;
pub(crate) const MICROS_PER_SEC_32: i32 =
    MILLIS_PER_SEC_32 * MICROS_PER_MILLI_32;
pub(crate) const MICROS_PER_MILLI_32: i32 = 1_000;
pub(crate) const NANOS_PER_SEC_32: i32 =
    MILLIS_PER_SEC_32 * NANOS_PER_MILLI_32;
pub(crate) const NANOS_PER_MILLI_32: i32 =
    MICROS_PER_MILLI_32 * NANOS_PER_MICRO_32;
pub(crate) const NANOS_PER_MICRO_32: i32 = 1_000;

/// This macro writes out the boiler plate to define a boundary type.
///
/// Specifically, it implements the `Bounds` trait and provides a few
/// concrete methods. The concrete methods are mostly wrappers around
/// the generic trait methods. They are provided so that callers don't
/// have to import the `Bounds` trait to use them.
macro_rules! define_bounds {
    ($((
        // The name of the boundary type.
        $name:ident,
        // The underlying primitive type. This is usually, but not always,
        // the smallest signed primitive integer type that can represent both
        // the minimum and maximum boundary values.
        $ty:ident,
        // A short human readable description that appears in error messages
        // when the boundaries of this type are violated.
        $what:expr,
        // The minimum value.
        $min:expr,
        // The maximum value.
        $max:expr $(,)?
    )),* $(,)?) => {
        $(
            pub(crate) struct $name(());

            impl Bounds for $name {
                const WHAT: &'static str = $what;
                const MIN: Self::Primitive = $min;
                const MAX: Self::Primitive = $max;
                type Primitive = $ty;

                #[cold]
                #[inline(never)]
                fn error() -> BoundsError {
                    BoundsError::$name(RawBoundsError::new())
                }
            }

            impl $name {
                pub(crate) const MIN: $ty = <$name as Bounds>::MIN;
                pub(crate) const MAX: $ty = <$name as Bounds>::MAX;
                const LEN: i128 = Self::MAX as i128 - Self::MIN as i128 + 1;

                #[cold]
                pub(crate) const fn error() -> BoundsError {
                    BoundsError::$name(RawBoundsError::new())
                }

                #[cfg_attr(feature = "perf-inline", inline(always))]
                pub(crate) fn check(n: impl Into<i64>) -> Result<$ty, BoundsError> {
                    <$name as Bounds>::check(n)
                }

                #[cfg_attr(feature = "perf-inline", inline(always))]
                pub(crate) const fn checkc(n: i64) -> Result<$ty, BoundsError> {
                    match self::checkc::$ty(n) {
                        Ok(n) => Ok(n),
                        Err(err) => Err(BoundsError::$name(err)),
                    }
                }

                #[cfg_attr(feature = "perf-inline", inline(always))]
                pub(crate) fn check128(n: impl Into<i128>) -> Result<$ty, BoundsError> {
                    <$name as Bounds>::check128(n)
                }

                #[cfg_attr(feature = "perf-inline", inline(always))]
                pub(crate) fn parse(bytes: &[u8]) -> Result<$ty, Error> {
                    <$name as Bounds>::parse(bytes)
                }

                #[cfg_attr(feature = "perf-inline", inline(always))]
                pub(crate) fn checked_add(n1: $ty, n2: $ty) -> Result<$ty, BoundsError> {
                    <$name as Bounds>::checked_add(n1, n2)
                }

                #[cfg_attr(feature = "perf-inline", inline(always))]
                pub(crate) fn checked_sub(n1: $ty, n2: $ty) -> Result<$ty, BoundsError> {
                    <$name as Bounds>::checked_sub(n1, n2)
                }

                #[cfg_attr(feature = "perf-inline", inline(always))]
                pub(crate) fn checked_mul(n1: $ty, n2: $ty) -> Result<$ty, BoundsError> {
                    <$name as Bounds>::checked_mul(n1, n2)
                }

                #[cfg(test)]
                pub(crate) fn arbitrary(g: &mut quickcheck::Gen) -> $ty {
                    use quickcheck::Arbitrary;

                    let mut n: $ty = <$ty>::arbitrary(g);
                    n = n.wrapping_rem_euclid(Self::LEN as $ty);
                    n += Self::MIN;
                    n
                }
            }
        )*

        /// An error that indicates a value is out of its intended range.
        #[derive(Clone, Debug)]
        pub(crate) enum BoundsError {
            $($name(RawBoundsError<$name>),)*
        }

        impl core::fmt::Display for BoundsError {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                match *self {
                    $(BoundsError::$name(ref err) => err.fmt(f),)*
                }
            }
        }
    }
}

define_bounds! {
    (Century, i8, "century", 0, 99),
    (
        CivilDayNanosecond,
        i64,
        "nanoseconds (in one civil day)",
        0,
        NANOS_PER_CIVIL_DAY - 1,
    ),
    (
        CivilDaySecond,
        i32,
        "seconds (in one civil day)",
        0,
        SECS_PER_CIVIL_DAY_32 - 1,
    ),
    (Day, i8, "day", 1, 31),
    (DayOfYear, i16, "day-of-year", 1, 366),
    (Hour, i8, "hour", 0, 23),
    (Hour12, i8, "hour (12 hour clock)", 1, 12),
    (ISOWeek, i8, "iso-week", 1, 53),
    (ISOYear, i16, "iso-year", -9999, 9999),
    // This matches Temporal's range.
    // See: https://github.com/tc39/proposal-temporal/issues/2458#issuecomment-1380742911
    (Increment, i64, "rounding increment", 1, 1_000_000_000),
    (Increment32, i32, "rounding increment", 1, 1_000_000_000),
    // This is only used in parsing. A value of `60` gets clamped to `59`.
    (LeapSecond, i8, "second", 0, 60),
    (Microsecond, i16, "microsecond", 0, 999),
    (Millisecond, i16, "millisecond", 0, 999),
    (Minute, i8, "minute", 0, 59),
    (Month, i8, "month", 1, 12),
    (Nanosecond, i16, "nanosecond", 0, 999),
    (NthWeekday, i32, "nth weekday", SpanWeeks::MIN, SpanWeeks::MAX),
    // The number of hours allowed in a time zone offset.
    //
    // This number was somewhat arbitrarily chosen. In part because it's bigger
    // than any current offset in actual use by a wide margin, and in part
    // because POSIX `TZ` strings require the ability to store offsets in the
    // range `-24:59:59..=25:59:59`. Note though that we make the range a
    // little bigger with `-25:59:59..=25:59:59` so that negating an offset
    // always produces a valid offset.
    //
    // Note that RFC 8536 actually allows offsets to be much bigger, namely,
    // in the range `(-2^31, 2^31)`, where both ends are _exclusive_ (`-2^31`
    // is explicitly disallowed, and `2^31` overflows a signed 32-bit
    // integer). But RFC 8536 does say that it *should* be in the range
    // `[-89999, 93599]`, which matches POSIX. In order to keep our offset
    // small, we stick roughly to what POSIX requires.
    //
    // Note that we support a slightly bigger range of offsets than Temporal.
    // Temporal seems to support only up to 23 hours, but we go up to 25 hours.
    // This is done to support POSIX time zone strings, which also require 25
    // hours (plus the maximal minute/second components).
    (OffsetHours, i8, "time zone offset hours", -25, 25),
    (OffsetMinutes, i8, "time zone offset minutes", -59, 59),
    (OffsetSeconds, i8, "time zone offset seconds", -59, 59),
    (
        OffsetTotalSeconds,
        i32,
        "time zone offset total seconds",
        -Self::MAX,
        (OffsetHours::MAX as i32 * SECS_PER_HOUR_32)
        + (OffsetMinutes::MAX as i32 * MINS_PER_HOUR_32)
        + OffsetSeconds::MAX as i32,
    ),
    (Second, i8, "second", 0, 59),
    (SignedDurationSeconds, i64, "signed duration seconds", i64::MIN, i64::MAX),
    (SpanYears, i16, "years", -Self::MAX, (Year::LEN - 1) as i16),
    (SpanMonths, i32, "months", -Self::MAX, SpanYears::MAX as i32 * 12),
    (SpanWeeks, i32, "weeks", -Self::MAX, SpanDays::MAX / DAYS_PER_WEEK_32),
    (SpanDays, i32, "days", -Self::MAX, SpanHours::MAX / HOURS_PER_CIVIL_DAY_32),
    (SpanHours, i32, "hours", -Self::MAX, (SpanMinutes::MAX / MINS_PER_HOUR) as i32),
    (SpanMinutes, i64, "minutes", -Self::MAX, SpanSeconds::MAX / SECS_PER_MIN),
    // The maximum number of seconds that can be expressed with a span.
    //
    // All of our span types (except for years and months, since they have
    // variable length even in civil datetimes) are defined in terms of this
    // constant. The way it's defined is a little odd, so let's break it down.
    //
    // Firstly, a span of seconds should be able to represent at least the
    // complete span supported by `Timestamp`. Thus, it's based off of
    // `UnixSeconds::LEN`. That is, a span should be able to represent the
    // value `UnixSeconds::MAX - UnixSeconds::MIN`.
    //
    // Secondly, a span should also be able to account for any amount of
    // possible time that a time zone offset might add or subtract to an
    // `Timestamp`. This also means it can account for any difference between
    // two `civil::DateTime` values.
    //
    // Thirdly, we would like our span to be divisible by
    // `SECONDS_PER_CIVIL_DAY`. This isn't strictly required, but it makes
    // defining boundaries a little smoother. If it weren't divisible, then the
    // lower bounds on some types would need to be adjusted by one.
    //
    // Note that neither the existence of this constant nor defining our
    // spans based on it impacts the correctness of doing arithmetic on zoned
    // instants. Arithmetic on zoned instants still uses "civil" spans, but the
    // length of time for some units (like a day) might vary. The arithmetic
    // for zoned instants accounts for this explicitly. But it still must obey
    // the limits set here.
    (
        SpanSeconds,
        i64,
        "seconds",
        -Self::MAX,
        next_multiple_of(
            UnixSeconds::LEN as i64
                + OffsetTotalSeconds::MAX as i64
                + SECS_PER_CIVIL_DAY,
            SECS_PER_CIVIL_DAY,
        ),
    ),
    (
        SpanMilliseconds,
        i64,
        "milliseconds",
        -Self::MAX,
        SpanSeconds::MAX * MILLIS_PER_SEC,
    ),
    (
        SpanMicroseconds,
        i64,
        "microseconds",
        -Self::MAX,
        SpanMilliseconds::MAX * MICROS_PER_MILLI,
    ),
    (SpanMultiple, i64, "span multiple", i64::MIN  + 1, i64::MAX),
    // A range of the allowed number of nanoseconds.
    //
    // For this, we cannot cover the full span of supported time instants since
    // `UnixSeconds::MAX * NANOSECONDS_PER_SECOND` cannot fit into 64-bits. We
    // could use a `i128`, but it doesn't seem worth it.
    //
    // Also note that our min is equal to -max, so that the total number of
    // values in this range is one less than the number of distinct `i64`
    // values. We do that so that the absolute value is always defined.
    (SpanNanoseconds, i64, "nanoseconds", i64::MIN + 1, i64::MAX),
    (SubsecNanosecond, i32, "subsecond nanosecond", 0, NANOS_PER_SEC_32 - 1),
    (
        SignedSubsecNanosecond,
        i32,
        "subsecond nanosecond",
        -SubsecNanosecond::MAX,
        SubsecNanosecond::MAX,
    ),
    // The number of days from the Unix epoch for the Gregorian calendar.
    //
    // The range supported is based on the range of Unix timestamps that we
    // support.
    //
    // While I had originally used the "rate die" concept from Calendrical
    // Calculations, I found [Howard Hinnant's formulation][date-algorithms]
    // much more straight-forward.
    //
    // [date-algorithms]: http://howardhinnant.github.io/date_algorithms.html
    (
        UnixEpochDays,
        i32,
        "Unix epoch days",
        (UnixSeconds::MIN+ OffsetTotalSeconds::MIN as i64).div_euclid(SECS_PER_CIVIL_DAY) as i32,
        (UnixSeconds::MAX + OffsetTotalSeconds::MAX as i64).div_euclid(SECS_PER_CIVIL_DAY) as i32,
    ),
    (
        UnixMilliseconds,
        i64,
        "Unix timestamp milliseconds",
        UnixSeconds::MIN * MILLIS_PER_SEC,
        UnixSeconds::MAX * MILLIS_PER_SEC,
    ),
    (
        UnixMicroseconds,
        i64,
        "Unix timestamp microseconds",
        UnixMilliseconds::MIN * MICROS_PER_MILLI,
        UnixMilliseconds::MAX * MICROS_PER_MILLI,
    ),
    // The range of Unix seconds supported by Jiff.
    //
    // This range should correspond to the first second of `Year::MIN` up
    // through (and including) the last second of `Year::MAX`. Actually
    // computing that is non-trivial, however, it can be computed easily enough
    // using Unix programs like `date`:
    //
    // ```text
    // $ TZ=0 date -d 'Mon Jan  1 12:00:00 AM  -9999' +'%s'
    // date: invalid date ‘Mon Jan  1 12:00:00 AM  -9999’
    // $ TZ=0 date -d 'Fri Dec 31 23:59:59  9999' +'%s'
    // 253402300799
    // ```
    //
    // Well, almost easily enough. `date` apparently doesn't support negative
    // years. But it does support negative timestamps:
    //
    // ```text
    // $ TZ=0 date -d '@-377705116800'
    // Mon Jan  1 12:00:00 AM  -9999
    // $ TZ=0 date -d '@253402300799'
    // Fri Dec 31 11:59:59 PM  9999
    // ```
    //
    // With that said, we actually end up restricting the range a bit more
    // than what's above. Namely, what's above is what we support for civil
    // datetimes. Because of time zones, we need to choose whether all
    // `Timestamp` values can be infallibly converted to `civil::DateTime`
    // values, or whether all `civil::DateTime` values can be infallibly
    // converted to `Timestamp` values. I chose the former because getting
    // a civil datetime is important for formatting. If I didn't choose the
    // former, there would be some instants that could not be formatted. Thus,
    // we make room by shrinking the range of allowed instants by precisely the
    // maximum supported time zone offset.
    (
        UnixSeconds,
        i64,
        "Unix timestamp seconds",
        -377705116800 - OffsetTotalSeconds::MIN as i64,
        253402300799 - OffsetTotalSeconds::MAX as i64,
    ),
    (WeekNum, i8, "week-number", 0, 53),
    (WeekdayMondayZero, i8, "weekday (Monday 0-indexed)", 0, 6),
    (WeekdayMondayOne, i8, "weekday (Monday 1-indexed)", 1, 7),
    (WeekdaySundayZero, i8, "weekday (Sunday 0-indexed)", 0, 6),
    (WeekdaySundayOne, i8, "weekday (Sunday 1-indexed)", 1, 7),
    // The range of years supported by Jiff.
    //
    // This is ultimately where some of the other ranges (like `UnixSeconds`)
    // were determined from. That is, the range of years is the primary point
    // at which the space of supported time instants is derived from. If one
    // wanted to expand this range, you'd need to change it here and then
    // compute the corresponding min/max values for `UnixSeconds`. (Among other
    // things... Increasing the supported Jiff range is far more complicated
    // than just changing some ranges here.)
    (Year, i16, "year", -9999, 9999),
    (YearCE, i16, "CE year", 1, Year::MAX),
    (YearBCE, i16, "BCE year", 1, Year::MAX + 1),
    (YearTwoDigit, i16, "year (2 digits)", 0, 99),
    (
        ZonedDayNanoseconds,
        i64,
        "nanoseconds (in one zoned datetime day)",
        ZonedDaySeconds::MIN as i64 * NANOS_PER_SEC,
        ZonedDaySeconds::MAX as i64 * NANOS_PER_SEC,
    ),
    // The number of seconds permitted in a single day.
    //
    // This is mostly just a "sensible" cap on what is possible. We allow one
    // day to span up to 7 civil days.
    //
    // It must also be at least 1 second long.
    (
        ZonedDaySeconds,
        i32,
        "seconds (in one zoned datetime day)",
        1,
        7 * SECS_PER_CIVIL_DAY_32,
    ),
}

/// An interface for defining boundaries on integer values.
pub(crate) trait Bounds: Sized {
    /// A short human readable description of the values represented by these
    /// bounds.
    const WHAT: &'static str;

    /// The minimum boundary value.
    const MIN: Self::Primitive;

    /// The maximum boundary value.
    const MAX: Self::Primitive;

    /// The primitive integer representation for this boundary type.
    ///
    /// This is generally the smallest primitive integer type that fits the
    /// minimum and maximum allowed values.
    // MSRV: Ideally this would be a private trait. On newer versions
    // of Rust (not sure when it started exactly), it's allowed but
    // comes with a warn-by-default lint. I would like it to be private
    // to avoid accidentally using it elsewhere, since it makes casts
    // between integers very easy.
    type Primitive: Primitive;

    // We provide `check` and `check128` to avoid manifesting 128-bit integers
    // in the vast majority of cases. While in theory the compiler should be
    // able to see through it, this is such a primitive and common operation
    // used throughout Jiff, that we specialize the overwhelmingly common case
    // for 64-bit integers under the presumption that 64-bit integers (and
    // smaller) are either always fast enough or are slower in environments
    // where we care less about performance.

    /// Create an error when a value is outside the bounds for this type.
    fn error() -> BoundsError;

    /// Converts the 64-bit integer provided into the primitive representation
    /// of these bounds.
    ///
    /// # Errors
    ///
    /// This returns an error if the given integer does not fit in the bounds
    /// prescribed by this trait implementation.
    ///
    /// # Panics
    ///
    /// This panics when `debug_assertions` are enabled if the bounds of
    /// this implementation exceed what is representable in an `i64`. In
    /// this case, callers must use `check128`.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn check(n: impl Into<i64>) -> Result<Self::Primitive, BoundsError> {
        // These asserts confirm that we only call this routine when our
        // bounds fit into an i64. Otherwise, the `as_i64()` casts below
        // are incorrect.
        debug_assert!((i128::from(i64::MIN)..=i128::from(i64::MAX))
            .contains(&Self::MIN.as_i128()));
        debug_assert!((i128::from(i64::MIN)..=i128::from(i64::MAX))
            .contains(&Self::MAX.as_i128()));

        let n = n.into();
        if !(Self::MIN.as_i64() <= n && n <= Self::MAX.as_i64()) {
            return Err(Self::error());
        }
        Ok(Self::Primitive::from_i64(n))
    }

    /// Converts the 128-bit integer provided into the primitive representation
    /// of these bounds.
    ///
    /// # Errors
    ///
    /// This returns an error if the given integer does not fit in the bounds
    /// prescribed by this trait implementation.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn check128(n: impl Into<i128>) -> Result<Self::Primitive, BoundsError> {
        let n = n.into();
        if !(Self::MIN.as_i128() <= n && n <= Self::MAX.as_i128()) {
            return Err(Self::error());
        }
        Ok(Self::Primitive::from_i128(n))
    }

    /// Checks whether the given integer, in the same primitive representation
    /// as this boundary type, is in bounds.
    ///
    /// # Errors
    ///
    /// This returns an error if the given integer does not fit in the bounds
    /// prescribed by this trait implementation.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn check_self(n: Self::Primitive) -> Result<Self::Primitive, BoundsError> {
        if !(Self::MIN <= n && n <= Self::MAX) {
            return Err(Self::error());
        }
        Ok(n)
    }

    /// Parses a 64-bit integer from the beginning to the end of the given
    /// slice of bytes.
    ///
    /// Note that this can never parse a negative integer since it doesn't
    /// look for a sign. On success, the integer returned is always positive.
    ///
    /// # Errors
    ///
    /// If the given slice is not a valid integer (i.e., overflow or contains
    /// anything other than `[0-9]`) or is not in the bounds for this trait
    /// implementation, then an error is returned.
    ///
    /// Note that the error can either be a parsing error or it can be a
    /// boundary error.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse(bytes: &[u8]) -> Result<Self::Primitive, Error> {
        Ok(Self::check(crate::util::parse::i64(bytes)?)?)
    }

    /// Performs checked addition using this boundary type's primitive
    /// representation.
    ///
    /// # Errors
    ///
    /// If the result exceeds the boundaries of the primitive type or of the
    /// declared range for this type, then an error is returned.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn checked_add(
        n1: Self::Primitive,
        n2: Self::Primitive,
    ) -> Result<Self::Primitive, BoundsError> {
        Self::check_self(n1.checked_add(n2).ok_or_else(Self::error)?)
    }

    /// Performs checked subtraction using this boundary type's primitive
    /// representation.
    ///
    /// # Errors
    ///
    /// If the result exceeds the boundaries of the primitive type or of the
    /// declared range for this type, then an error is returned.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn checked_sub(
        n1: Self::Primitive,
        n2: Self::Primitive,
    ) -> Result<Self::Primitive, BoundsError> {
        Self::check_self(n1.checked_sub(n2).ok_or_else(Self::error)?)
    }

    /// Performs checked multiplication using this boundary type's primitive
    /// representation.
    ///
    /// # Errors
    ///
    /// If the result exceeds the boundaries of the primitive type or of the
    /// declared range for this type, then an error is returned.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn checked_mul(
        n1: Self::Primitive,
        n2: Self::Primitive,
    ) -> Result<Self::Primitive, BoundsError> {
        Self::check_self(n1.checked_mul(n2).ok_or_else(Self::error)?)
    }
}

/// A simple trait for making `int as int` usable in a generic context.
///
/// All of these methods require callers to ensure the cast is correct.
pub(crate) trait Primitive:
    Clone
    + Copy
    + Eq
    + PartialEq
    + PartialOrd
    + Ord
    + core::fmt::Debug
    + core::fmt::Display
{
    fn as_i8(self) -> i8;
    fn as_i16(self) -> i16;
    fn as_i32(self) -> i32;
    fn as_i64(self) -> i64;
    fn as_i128(self) -> i128;

    fn from_i8(n: i8) -> Self;
    fn from_i16(n: i16) -> Self;
    fn from_i32(n: i32) -> Self;
    fn from_i64(n: i64) -> Self;
    fn from_i128(n: i128) -> Self;

    fn checked_add(self, n: Self) -> Option<Self>;
    fn checked_sub(self, n: Self) -> Option<Self>;
    fn checked_mul(self, n: Self) -> Option<Self>;
}

macro_rules! impl_primitive {
    ($($intty:ty),*) => {
        $(
            impl Primitive for $intty {
                fn as_i8(self) -> i8 { self as i8 }
                fn as_i16(self) -> i16 { self as i16 }
                fn as_i32(self) -> i32 { self as i32 }
                fn as_i64(self) -> i64 { self as i64 }
                fn as_i128(self) -> i128 { self as i128 }

                fn from_i8(n: i8) -> Self { n as $intty }
                fn from_i16(n: i16) -> Self { n as $intty }
                fn from_i32(n: i32) -> Self { n as $intty }
                fn from_i64(n: i64) -> Self { n as $intty }
                fn from_i128(n: i128) -> Self { n as $intty }

                fn checked_add(self, n: $intty) -> Option<$intty> {
                    <$intty>::checked_add(self, n)
                }

                fn checked_sub(self, n: $intty) -> Option<$intty> {
                    <$intty>::checked_sub(self, n)
                }

                fn checked_mul(self, n: $intty) -> Option<$intty> {
                    <$intty>::checked_mul(self, n)
                }
            }
        )*
    }
}

impl_primitive!(i8, i16, i32, i64, i128);

impl From<BoundsError> for Error {
    fn from(err: BoundsError) -> Error {
        Error::bounds(err)
    }
}

impl crate::error::IntoError for BoundsError {
    fn into_error(self) -> Error {
        self.into()
    }
}

pub(crate) struct RawBoundsError<B>(core::marker::PhantomData<B>);

impl<B> RawBoundsError<B> {
    const fn new() -> RawBoundsError<B> {
        RawBoundsError(core::marker::PhantomData)
    }
}

impl<B> Clone for RawBoundsError<B> {
    fn clone(&self) -> RawBoundsError<B> {
        RawBoundsError::new()
    }
}

impl<B, P> core::fmt::Debug for RawBoundsError<B>
where
    B: Bounds<Primitive = P>,
    P: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("RawBoundsError")
            .field("what", &B::WHAT)
            .field("min", &B::MIN)
            .field("max", &B::MAX)
            .finish()
    }
}

impl<B, P> core::fmt::Display for RawBoundsError<B>
where
    B: Bounds<Primitive = P>,
    P: core::fmt::Display,
{
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(
            f,
            "parameter '{what}' is not in the required range of {min}..={max}",
            what = B::WHAT,
            min = B::MIN,
            max = B::MAX,
        )
    }
}

/// Like `BoundsError`, but maintained manually.
///
/// This is useful for range errors outside of the framework above.
#[derive(Clone, Debug)]
pub(crate) enum SpecialBoundsError {
    UnixNanoseconds,
    SignedDurationFloatOutOfRangeF32,
    SignedDurationFloatOutOfRangeF64,
    SignedToUnsignedDuration,
}

impl core::fmt::Display for SpecialBoundsError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::SpecialBoundsError::*;

        let (what, min, max) = match *self {
            UnixNanoseconds => (
                "Unix timestamp nanoseconds",
                UnixMicroseconds::MIN as i128 * (NANOS_PER_MICRO as i128),
                UnixMicroseconds::MAX as i128 * (NANOS_PER_MICRO as i128),
            ),
            SignedToUnsignedDuration => {
                return f.write_str(
                    "negative signed durations cannot be converted \
                     to an unsigned duration",
                );
            }
            SignedDurationFloatOutOfRangeF32 => {
                return write!(
                    f,
                    "parameter 'floating point seconds' is not in \
                     the required range of {min}..={max}",
                    min = i64::MIN as f32,
                    max = i64::MAX as f32,
                );
            }
            SignedDurationFloatOutOfRangeF64 => {
                return write!(
                    f,
                    "parameter 'floating point seconds' is not in \
                     the required range of {min}..={max}",
                    min = i64::MIN as f64,
                    max = i64::MAX as f64,
                );
            }
        };
        write!(
            f,
            "parameter '{what}' is not in the required range of {min}..={max}",
        )
    }
}

impl From<SpecialBoundsError> for Error {
    fn from(err: SpecialBoundsError) -> Error {
        Error::special_bounds(err)
    }
}

impl crate::error::IntoError for SpecialBoundsError {
    fn into_error(self) -> Error {
        self.into()
    }
}

/// A representation of a numeric sign.
///
/// Its `Display` impl emits the ASCII minus sign, `-` when this
/// is negative. It emits the empty string in all other cases.
#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, PartialEq, PartialOrd, Ord,
)]
#[repr(i8)]
pub(crate) enum Sign {
    #[default]
    Zero = 0,
    Positive = 1,
    Negative = -1,
}

impl Sign {
    pub(crate) fn is_zero(self) -> bool {
        matches!(self, Sign::Zero)
    }

    pub(crate) fn is_positive(self) -> bool {
        matches!(self, Sign::Positive)
    }

    pub(crate) fn is_negative(self) -> bool {
        matches!(self, Sign::Negative)
    }

    pub(crate) fn signum(self) -> i8 {
        self.as_i8()
    }

    pub(crate) fn as_i8(self) -> i8 {
        self as i8
    }

    pub(crate) fn as_i16(self) -> i16 {
        i16::from(self.as_i8())
    }

    pub(crate) fn as_i32(self) -> i32 {
        i32::from(self.as_i8())
    }

    pub(crate) fn as_i64(self) -> i64 {
        i64::from(self.as_i8())
    }

    pub(crate) fn as_i128(self) -> i128 {
        i128::from(self.as_i8())
    }

    pub(crate) fn from_ordinals<T: Ord>(t1: T, t2: T) -> Sign {
        use core::cmp::Ordering::*;
        match t1.cmp(&t2) {
            Less => Sign::Negative,
            Equal => Sign::Zero,
            Greater => Sign::Positive,
        }
    }
}

impl core::ops::Neg for Sign {
    type Output = Sign;

    fn neg(self) -> Sign {
        match self {
            Sign::Positive => Sign::Negative,
            Sign::Zero => Sign::Zero,
            Sign::Negative => Sign::Positive,
        }
    }
}

impl From<i8> for Sign {
    fn from(n: i8) -> Sign {
        Sign::from(i64::from(n))
    }
}

impl From<i16> for Sign {
    fn from(n: i16) -> Sign {
        Sign::from(i64::from(n))
    }
}

impl From<i32> for Sign {
    fn from(n: i32) -> Sign {
        Sign::from(i64::from(n))
    }
}

impl From<i64> for Sign {
    fn from(n: i64) -> Sign {
        if n == 0 {
            Sign::Zero
        } else if n > 0 {
            Sign::Positive
        } else {
            Sign::Negative
        }
    }
}

impl From<i128> for Sign {
    fn from(n: i128) -> Sign {
        if n == 0 {
            Sign::Zero
        } else if n > 0 {
            Sign::Positive
        } else {
            Sign::Negative
        }
    }
}

impl From<f64> for Sign {
    fn from(n: f64) -> Sign {
        use core::num::FpCategory::*;

        // This is a little odd, but we want +/- 0 to
        // always have a sign of zero, so as to be consistent
        // with how we deal with signed integers.
        //
        // As for NaN... It should generally be a bug if
        // Jiff ever materializes a NaN. Notably, I do not
        // believe there are any APIs in which a float is
        // given from the caller. Jiff only ever uses them
        // internally or returns them. So if we get a NaN,
        // it's on us. If we do, just assign it a zero sign?
        if matches!(n.classify(), Nan | Zero) {
            Sign::Zero
        } else if n.is_sign_positive() {
            Sign::Positive
        } else {
            Sign::Negative
        }
    }
}

impl From<SignedDuration> for Sign {
    fn from(n: SignedDuration) -> Sign {
        if n.is_zero() {
            Sign::Zero
        } else if n.is_positive() {
            Sign::Positive
        } else {
            Sign::Negative
        }
    }
}

impl core::ops::Mul<Sign> for Sign {
    type Output = Sign;
    fn mul(self, rhs: Sign) -> Sign {
        match (self, rhs) {
            (Sign::Zero, _) | (_, Sign::Zero) => Sign::Zero,
            (Sign::Positive, Sign::Positive) => Sign::Positive,
            (Sign::Negative, Sign::Negative) => Sign::Positive,
            (Sign::Positive, Sign::Negative) => Sign::Negative,
            (Sign::Negative, Sign::Positive) => Sign::Negative,
        }
    }
}

impl core::ops::Mul<i8> for Sign {
    type Output = i8;
    fn mul(self, n: i8) -> i8 {
        self.as_i8() * n
    }
}

impl core::ops::Mul<Sign> for i8 {
    type Output = i8;
    fn mul(self, n: Sign) -> i8 {
        self * n.as_i8()
    }
}

impl core::ops::Mul<i16> for Sign {
    type Output = i16;
    fn mul(self, n: i16) -> i16 {
        self.as_i16() * n
    }
}

impl core::ops::Mul<Sign> for i16 {
    type Output = i16;
    fn mul(self, n: Sign) -> i16 {
        self * n.as_i16()
    }
}

impl core::ops::Mul<i32> for Sign {
    type Output = i32;
    fn mul(self, n: i32) -> i32 {
        self.as_i32() * n
    }
}

impl core::ops::Mul<Sign> for i32 {
    type Output = i32;
    fn mul(self, n: Sign) -> i32 {
        self * n.as_i32()
    }
}

impl core::ops::Mul<i64> for Sign {
    type Output = i64;
    fn mul(self, n: i64) -> i64 {
        self.as_i64() * n
    }
}

impl core::ops::Mul<Sign> for i64 {
    type Output = i64;
    fn mul(self, n: Sign) -> i64 {
        self * n.as_i64()
    }
}

impl core::ops::Mul<i128> for Sign {
    type Output = i128;
    fn mul(self, n: i128) -> i128 {
        self.as_i128() * n
    }
}

impl core::ops::Mul<Sign> for i128 {
    type Output = i128;
    fn mul(self, n: Sign) -> i128 {
        self * n.as_i128()
    }
}

impl core::fmt::Display for Sign {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        if self.is_negative() {
            f.write_str("-")
        } else {
            Ok(())
        }
    }
}

mod checkc {
    use super::{Bounds, RawBoundsError};

    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(super) const fn i8<B>(n: i64) -> Result<i8, RawBoundsError<B>>
    where
        B: Bounds<Primitive = i8>,
    {
        // These asserts confirm that we only call this routine
        // when our bounds fit into an i64. Otherwise, the
        // `as` casts below are incorrect.
        debug_assert!(
            (i64::MIN as i128) <= (B::MIN as i128)
                && (B::MIN as i128) <= (i64::MAX as i128),
        );
        debug_assert!(
            (i64::MIN as i128) <= (B::MAX as i128)
                && (B::MAX as i128) <= (i64::MAX as i128),
        );

        if !((B::MIN as i64) <= n && n <= (B::MAX as i64)) {
            return Err(RawBoundsError::new());
        }
        Ok(n as i8)
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(super) const fn i16<B>(n: i64) -> Result<i16, RawBoundsError<B>>
    where
        B: Bounds<Primitive = i16>,
    {
        // These asserts confirm that we only call this routine
        // when our bounds fit into an i64. Otherwise, the
        // `as` casts below are incorrect.
        debug_assert!(
            (i64::MIN as i128) <= (B::MIN as i128)
                && (B::MIN as i128) <= (i64::MAX as i128),
        );
        debug_assert!(
            (i64::MIN as i128) <= (B::MAX as i128)
                && (B::MAX as i128) <= (i64::MAX as i128),
        );

        if !((B::MIN as i64) <= n && n <= (B::MAX as i64)) {
            return Err(RawBoundsError::new());
        }
        Ok(n as i16)
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(super) const fn i32<B>(n: i64) -> Result<i32, RawBoundsError<B>>
    where
        B: Bounds<Primitive = i32>,
    {
        // These asserts confirm that we only call this routine
        // when our bounds fit into an i64. Otherwise, the
        // `as` casts below are incorrect.
        debug_assert!(
            (i64::MIN as i128) <= (B::MIN as i128)
                && (B::MIN as i128) <= (i64::MAX as i128),
        );
        debug_assert!(
            (i64::MIN as i128) <= (B::MAX as i128)
                && (B::MAX as i128) <= (i64::MAX as i128),
        );

        if !((B::MIN as i64) <= n && n <= (B::MAX as i64)) {
            return Err(RawBoundsError::new());
        }
        Ok(n as i32)
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(super) const fn i64<B>(n: i64) -> Result<i64, RawBoundsError<B>>
    where
        B: Bounds<Primitive = i64>,
    {
        // These asserts confirm that we only call this routine
        // when our bounds fit into an i64. Otherwise, the
        // `as` casts below are incorrect.
        debug_assert!(
            (i64::MIN as i128) <= (B::MIN as i128)
                && (B::MIN as i128) <= (i64::MAX as i128),
        );
        debug_assert!(
            (i64::MIN as i128) <= (B::MAX as i128)
                && (B::MAX as i128) <= (i64::MAX as i128),
        );

        if !(B::MIN <= n && n <= B::MAX) {
            return Err(RawBoundsError::new());
        }
        Ok(n)
    }
}

/// Computes the next multiple of `rhs` that is greater than or equal to `lhs`.
///
/// Taken from:
/// https://github.com/rust-lang/rust/blob/eff958c59e8c07ba0515e164b825c9001b242294/library/core/src/num/int_macros.rs
const fn next_multiple_of(lhs: i64, rhs: i64) -> i64 {
    // This would otherwise fail when calculating `r` when self == T::MIN.
    if rhs == -1 {
        return lhs;
    }

    let r = lhs % rhs;
    let m = if (r > 0 && rhs < 0) || (r < 0 && rhs > 0) { r + rhs } else { r };
    if m == 0 {
        lhs
    } else {
        lhs + (rhs - m)
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;

    use super::*;

    #[test]
    fn size_of_bounds_error() {
        // It's okay if this grows, but I wrote this test initially
        // to confirm that a `BoundsError` is very small. (Since it's
        // just a bunch of variants of ZSTs.)
        assert_eq!(1, core::mem::size_of::<BoundsError>());
    }

    #[test]
    fn basic_error_functionality() {
        let err = Year::check(10_000).unwrap_err();
        assert_eq!(
            err.to_string(),
            "parameter 'year' is not in the required range of -9999..=9999",
        );
    }
}
