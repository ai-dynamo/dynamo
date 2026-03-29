use core::{cmp::Ordering, time::Duration as UnsignedDuration};

use crate::{
    civil::{Date, DateTime, Time},
    duration::{Duration, SDuration},
    error::{span::Error as E, unit::UnitConfigError, Error, ErrorContext},
    fmt::{friendly, temporal},
    tz::TimeZone,
    util::{b, borrow::DumbCow, round::Increment},
    RoundMode, SignedDuration, Timestamp, Zoned,
};

/// A macro helper, only used in tests, for comparing spans for equality.
#[cfg(test)]
macro_rules! span_eq {
    ($span1:expr, $span2:expr $(,)?) => {{
        assert_eq!($span1.fieldwise(), $span2.fieldwise());
    }};
    ($span1:expr, $span2:expr, $($tt:tt)*) => {{
        assert_eq!($span1.fieldwise(), $span2.fieldwise(), $($tt)*);
    }};
}

#[cfg(test)]
pub(crate) use span_eq;

/// A span of time represented via a mixture of calendar and clock units.
///
/// A span represents a duration of time in units of years, months, weeks,
/// days, hours, minutes, seconds, milliseconds, microseconds and nanoseconds.
/// Spans are used to as inputs to routines like
/// [`Zoned::checked_add`] and [`Date::saturating_sub`],
/// and are also outputs from routines like
/// [`Timestamp::since`] and [`DateTime::until`].
///
/// # Range of spans
///
/// Except for nanoseconds, each unit can represent the full span of time
/// expressible via any combination of datetime supported by Jiff. For example:
///
/// ```
/// use jiff::{civil::{DateTime, DateTimeDifference}, ToSpan, Unit};
///
/// let options = DateTimeDifference::new(DateTime::MAX).largest(Unit::Year);
/// assert_eq!(DateTime::MIN.until(options)?.get_years(), 19_998);
///
/// let options = options.largest(Unit::Day);
/// assert_eq!(DateTime::MIN.until(options)?.get_days(), 7_304_483);
///
/// let options = options.largest(Unit::Microsecond);
/// assert_eq!(
///     DateTime::MIN.until(options)?.get_microseconds(),
///     631_107_417_599_999_999i64,
/// );
///
/// let options = options.largest(Unit::Nanosecond);
/// // Span is too big, overflow!
/// assert!(DateTime::MIN.until(options).is_err());
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Building spans
///
/// A default or empty span corresponds to a duration of zero time:
///
/// ```
/// use jiff::Span;
///
/// assert!(Span::new().is_zero());
/// assert!(Span::default().is_zero());
/// ```
///
/// Spans are `Copy` types that have mutator methods on them for creating new
/// spans:
///
/// ```
/// use jiff::Span;
///
/// let span = Span::new().days(5).hours(8).minutes(1);
/// assert_eq!(span.to_string(), "P5DT8H1M");
/// ```
///
/// But Jiff provides a [`ToSpan`] trait that defines extension methods on
/// primitive signed integers to make span creation terser:
///
/// ```
/// use jiff::ToSpan;
///
/// let span = 5.days().hours(8).minutes(1);
/// assert_eq!(span.to_string(), "P5DT8H1M");
/// // singular units on integers can be used too:
/// let span = 1.day().hours(8).minutes(1);
/// assert_eq!(span.to_string(), "P1DT8H1M");
/// ```
///
/// # Negative spans
///
/// A span may be negative. All of these are equivalent:
///
/// ```
/// use jiff::{Span, ToSpan};
///
/// let span = -Span::new().days(5);
/// assert_eq!(span.to_string(), "-P5D");
///
/// let span = Span::new().days(5).negate();
/// assert_eq!(span.to_string(), "-P5D");
///
/// let span = Span::new().days(-5);
/// assert_eq!(span.to_string(), "-P5D");
///
/// let span = -Span::new().days(-5).negate();
/// assert_eq!(span.to_string(), "-P5D");
///
/// let span = -5.days();
/// assert_eq!(span.to_string(), "-P5D");
///
/// let span = (-5).days();
/// assert_eq!(span.to_string(), "-P5D");
///
/// let span = -(5.days());
/// assert_eq!(span.to_string(), "-P5D");
/// ```
///
/// The sign of a span applies to the entire span. When a span is negative,
/// then all of its units are negative:
///
/// ```
/// use jiff::ToSpan;
///
/// let span = -5.days().hours(10).minutes(1);
/// assert_eq!(span.get_days(), -5);
/// assert_eq!(span.get_hours(), -10);
/// assert_eq!(span.get_minutes(), -1);
/// ```
///
/// And if any of a span's units are negative, then the entire span is regarded
/// as negative:
///
/// ```
/// use jiff::ToSpan;
///
/// // It's the same thing.
/// let span = (-5).days().hours(-10).minutes(-1);
/// assert_eq!(span.get_days(), -5);
/// assert_eq!(span.get_hours(), -10);
/// assert_eq!(span.get_minutes(), -1);
///
/// // Still the same. All negative.
/// let span = 5.days().hours(-10).minutes(1);
/// assert_eq!(span.get_days(), -5);
/// assert_eq!(span.get_hours(), -10);
/// assert_eq!(span.get_minutes(), -1);
///
/// // But this is not! The negation in front applies
/// // to the entire span, which was already negative
/// // by virtue of at least one of its units being
/// // negative. So the negation operator in front turns
/// // the span positive.
/// let span = -5.days().hours(-10).minutes(-1);
/// assert_eq!(span.get_days(), 5);
/// assert_eq!(span.get_hours(), 10);
/// assert_eq!(span.get_minutes(), 1);
/// ```
///
/// You can also ask for the absolute value of a span:
///
/// ```
/// use jiff::Span;
///
/// let span = Span::new().days(5).hours(10).minutes(1).negate().abs();
/// assert_eq!(span.get_days(), 5);
/// assert_eq!(span.get_hours(), 10);
/// assert_eq!(span.get_minutes(), 1);
/// ```
///
/// # Parsing and printing
///
/// The `Span` type provides convenient trait implementations of
/// [`std::str::FromStr`] and [`std::fmt::Display`]:
///
/// ```
/// use jiff::{Span, ToSpan};
///
/// let span: Span = "P2m10dT2h30m".parse()?;
/// // By default, capital unit designator labels are used.
/// // This can be changed with `jiff::fmt::temporal::SpanPrinter::lowercase`.
/// assert_eq!(span.to_string(), "P2M10DT2H30M");
///
/// // Or use the "friendly" format by invoking the `Display` alternate:
/// assert_eq!(format!("{span:#}"), "2mo 10d 2h 30m");
///
/// // Parsing automatically supports both the ISO 8601 and "friendly"
/// // formats. Note that we use `Span::fieldwise` to create a `Span` that
/// // compares based on each field. To compare based on total duration, use
/// // `Span::compare` or `Span::total`.
/// let span: Span = "2mo 10d 2h 30m".parse()?;
/// assert_eq!(span, 2.months().days(10).hours(2).minutes(30).fieldwise());
/// let span: Span = "2 months, 10 days, 2 hours, 30 minutes".parse()?;
/// assert_eq!(span, 2.months().days(10).hours(2).minutes(30).fieldwise());
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// The format supported is a variation (nearly a subset) of the duration
/// format specified in [ISO 8601] _and_ a Jiff-specific "friendly" format.
/// Here are more examples:
///
/// ```
/// use jiff::{Span, ToSpan};
///
/// let spans = [
///     // ISO 8601
///     ("P40D", 40.days()),
///     ("P1y1d", 1.year().days(1)),
///     ("P3dT4h59m", 3.days().hours(4).minutes(59)),
///     ("PT2H30M", 2.hours().minutes(30)),
///     ("P1m", 1.month()),
///     ("P1w", 1.week()),
///     ("P1w4d", 1.week().days(4)),
///     ("PT1m", 1.minute()),
///     ("PT0.0021s", 2.milliseconds().microseconds(100)),
///     ("PT0s", 0.seconds()),
///     ("P0d", 0.seconds()),
///     (
///         "P1y1m1dT1h1m1.1s",
///         1.year().months(1).days(1).hours(1).minutes(1).seconds(1).milliseconds(100),
///     ),
///     // Jiff's "friendly" format
///     ("40d", 40.days()),
///     ("40 days", 40.days()),
///     ("1y1d", 1.year().days(1)),
///     ("1yr 1d", 1.year().days(1)),
///     ("3d4h59m", 3.days().hours(4).minutes(59)),
///     ("3 days, 4 hours, 59 minutes", 3.days().hours(4).minutes(59)),
///     ("3d 4h 59m", 3.days().hours(4).minutes(59)),
///     ("2h30m", 2.hours().minutes(30)),
///     ("2h 30m", 2.hours().minutes(30)),
///     ("1mo", 1.month()),
///     ("1w", 1.week()),
///     ("1 week", 1.week()),
///     ("1w4d", 1.week().days(4)),
///     ("1 wk 4 days", 1.week().days(4)),
///     ("1m", 1.minute()),
///     ("0.0021s", 2.milliseconds().microseconds(100)),
///     ("0s", 0.seconds()),
///     ("0d", 0.seconds()),
///     ("0 days", 0.seconds()),
///     (
///         "1y1mo1d1h1m1.1s",
///         1.year().months(1).days(1).hours(1).minutes(1).seconds(1).milliseconds(100),
///     ),
///     (
///         "1yr 1mo 1day 1hr 1min 1.1sec",
///         1.year().months(1).days(1).hours(1).minutes(1).seconds(1).milliseconds(100),
///     ),
///     (
///         "1 year, 1 month, 1 day, 1 hour, 1 minute 1.1 seconds",
///         1.year().months(1).days(1).hours(1).minutes(1).seconds(1).milliseconds(100),
///     ),
///     (
///         "1 year, 1 month, 1 day, 01:01:01.1",
///         1.year().months(1).days(1).hours(1).minutes(1).seconds(1).milliseconds(100),
///     ),
/// ];
/// for (string, span) in spans {
///     let parsed: Span = string.parse()?;
///     assert_eq!(
///         span.fieldwise(),
///         parsed.fieldwise(),
///         "result of parsing {string:?}",
///     );
/// }
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// For more details, see the [`fmt::temporal`](temporal) and
/// [`fmt::friendly`](friendly) modules.
///
/// [ISO 8601]: https://www.iso.org/iso-8601-date-and-time-format.html
///
/// # Comparisons
///
/// A `Span` does not implement the `PartialEq` or `Eq` traits. These traits
/// were implemented in an earlier version of Jiff, but they made it too
/// easy to introduce bugs. For example, `120.minutes()` and `2.hours()`
/// always correspond to the same total duration, but they have different
/// representations in memory and so didn't compare equivalent.
///
/// The reason why the `PartialEq` and `Eq` trait implementations do not do
/// comparisons with total duration is because it is fundamentally impossible
/// to do such comparisons without a reference date in all cases.
///
/// However, it is undeniably occasionally useful to do comparisons based
/// on the component fields, so long as such use cases can tolerate two
/// different spans comparing unequal even when their total durations are
/// equivalent. For example, many of the tests in Jiff (including the tests in
/// the documentation) work by comparing a `Span` to an expected result. This
/// is a good demonstration of when fieldwise comparisons are appropriate.
///
/// To do fieldwise comparisons with a span, use the [`Span::fieldwise`]
/// method. This method creates a [`SpanFieldwise`], which is just a `Span`
/// that implements `PartialEq` and `Eq` in a fieldwise manner. In other words,
/// it's a speed bump to ensure this is the kind of comparison you actually
/// want. For example:
///
/// ```
/// use jiff::ToSpan;
///
/// assert_ne!(1.hour().fieldwise(), 60.minutes().fieldwise());
/// // These also work since you only need one fieldwise span to do a compare:
/// assert_ne!(1.hour(), 60.minutes().fieldwise());
/// assert_ne!(1.hour().fieldwise(), 60.minutes());
/// ```
///
/// This is because doing true comparisons requires arithmetic and a relative
/// datetime in the general case, and which can fail due to overflow. This
/// operation is provided via [`Span::compare`]:
///
/// ```
/// use jiff::{civil::date, ToSpan};
///
/// // This doesn't need a reference date since it's only using time units.
/// assert_eq!(1.hour().compare(60.minutes())?, std::cmp::Ordering::Equal);
/// // But if you have calendar units, then you need a
/// // reference date at minimum:
/// assert!(1.month().compare(30.days()).is_err());
/// assert_eq!(
///     1.month().compare((30.days(), date(2025, 6, 1)))?,
///     std::cmp::Ordering::Equal,
/// );
/// // A month can be a differing number of days!
/// assert_eq!(
///     1.month().compare((30.days(), date(2025, 7, 1)))?,
///     std::cmp::Ordering::Greater,
/// );
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Arithmetic
///
/// Spans can be added or subtracted via [`Span::checked_add`] and
/// [`Span::checked_sub`]:
///
/// ```
/// use jiff::{Span, ToSpan};
///
/// let span1 = 2.hours().minutes(20);
/// let span2: Span = "PT89400s".parse()?;
/// assert_eq!(span1.checked_add(span2)?, 27.hours().minutes(10).fieldwise());
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// When your spans involve calendar units, a relative datetime must be
/// provided. (Because, for example, 1 month from March 1 is 31 days, but
/// 1 month from April 1 is 30 days.)
///
/// ```
/// use jiff::{civil::date, Span, ToSpan};
///
/// let span1 = 2.years().months(6).days(20);
/// let span2 = 400.days();
/// assert_eq!(
///     span1.checked_add((span2, date(2023, 1, 1)))?,
///     3.years().months(7).days(24).fieldwise(),
/// );
/// // The span changes when a leap year isn't included!
/// assert_eq!(
///     span1.checked_add((span2, date(2025, 1, 1)))?,
///     3.years().months(7).days(23).fieldwise(),
/// );
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Rounding and balancing
///
/// Unlike datetimes, multiple distinct `Span` values can actually correspond
/// to the same duration of time. For example, all of the following correspond
/// to the same duration:
///
/// * 2 hours, 30 minutes
/// * 150 minutes
/// * 1 hour, 90 minutes
///
/// The first is said to be balanced. That is, its biggest non-zero unit cannot
/// be expressed in an integer number of units bigger than hours. But the
/// second is unbalanced because 150 minutes can be split up into hours and
/// minutes. We call this sort of span a "top-heavy" unbalanced span. The third
/// span is also unbalanced, but it's "bottom-heavy" and rarely used. Jiff
/// will generally only produce spans of the first two types. In particular,
/// most `Span` producing APIs accept a "largest" [`Unit`] parameter, and the
/// result can be said to be a span "balanced up to the largest unit provided."
///
/// Balanced and unbalanced spans can be switched between as needed via
/// the [`Span::round`] API by providing a rounding configuration with
/// [`SpanRound::largest`]` set:
///
/// ```
/// use jiff::{SpanRound, ToSpan, Unit};
///
/// let span = 2.hours().minutes(30);
/// let unbalanced = span.round(SpanRound::new().largest(Unit::Minute))?;
/// assert_eq!(unbalanced, 150.minutes().fieldwise());
/// let balanced = unbalanced.round(SpanRound::new().largest(Unit::Hour))?;
/// assert_eq!(balanced, 2.hours().minutes(30).fieldwise());
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// Balancing can also be done as part of computing spans from two datetimes:
///
/// ```
/// use jiff::{civil::date, ToSpan, Unit};
///
/// let zdt1 = date(2024, 7, 7).at(15, 23, 0, 0).in_tz("America/New_York")?;
/// let zdt2 = date(2024, 11, 5).at(8, 0, 0, 0).in_tz("America/New_York")?;
///
/// // To make arithmetic reversible, the default largest unit for spans of
/// // time computed from zoned datetimes is hours:
/// assert_eq!(zdt1.until(&zdt2)?, 2_897.hour().minutes(37).fieldwise());
/// // But we can ask for the span to be balanced up to years:
/// assert_eq!(
///     zdt1.until((Unit::Year, &zdt2))?,
///     3.months().days(28).hours(16).minutes(37).fieldwise(),
/// );
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// While the [`Span::round`] API does balancing, it also, of course, does
/// rounding as well. Rounding occurs when the smallest unit is set to
/// something bigger than [`Unit::Nanosecond`]:
///
/// ```
/// use jiff::{ToSpan, Unit};
///
/// let span = 2.hours().minutes(30);
/// assert_eq!(span.round(Unit::Hour)?, 3.hours().fieldwise());
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// When rounding spans with calendar units (years, months or weeks), then a
/// relative datetime is required:
///
/// ```
/// use jiff::{civil::date, SpanRound, ToSpan, Unit};
///
/// let span = 10.years().months(11);
/// let options = SpanRound::new()
///     .smallest(Unit::Year)
///     .relative(date(2024, 1, 1));
/// assert_eq!(span.round(options)?, 11.years().fieldwise());
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Days are not always 24 hours!
///
/// That is, a `Span` is made up of uniform and non-uniform units.
///
/// A uniform unit is a unit whose elapsed duration is always the same.
/// A non-uniform unit is a unit whose elapsed duration is not always the same.
/// There are two things that can impact the length of a non-uniform unit:
/// the calendar date and the time zone.
///
/// Years and months are always considered non-uniform units. For example,
/// 1 month from `2024-04-01` is 30 days, while 1 month from `2024-05-01` is
/// 31 days. Similarly for years because of leap years.
///
/// Hours, minutes, seconds, milliseconds, microseconds and nanoseconds are
/// always considered uniform units.
///
/// Days are only considered non-uniform when in the presence of a zone aware
/// datetime. A day can be more or less than 24 hours, and it can be balanced
/// up and down, but only when a relative zoned datetime is given. This
/// typically happens because of DST (daylight saving time), but can also occur
/// because of other time zone transitions too.
///
/// ```
/// use jiff::{civil::date, SpanRound, ToSpan, Unit};
///
/// // 2024-03-10 in New York was 23 hours long,
/// // because of a jump to DST at 2am.
/// let zdt = date(2024, 3, 9).at(21, 0, 0, 0).in_tz("America/New_York")?;
/// // Goes from days to hours:
/// assert_eq!(
///     1.day().round(SpanRound::new().largest(Unit::Hour).relative(&zdt))?,
///     23.hours().fieldwise(),
/// );
/// // Goes from hours to days:
/// assert_eq!(
///     23.hours().round(SpanRound::new().largest(Unit::Day).relative(&zdt))?,
///     1.day().fieldwise(),
/// );
/// // 24 hours is more than 1 day starting at this time:
/// assert_eq!(
///     24.hours().round(SpanRound::new().largest(Unit::Day).relative(&zdt))?,
///     1.day().hours(1).fieldwise(),
/// );
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// And similarly, days can be longer than 24 hours:
///
/// ```
/// use jiff::{civil::date, SpanRound, ToSpan, Unit};
///
/// // 2024-11-03 in New York was 25 hours long,
/// // because of a repetition of the 1 o'clock AM hour.
/// let zdt = date(2024, 11, 2).at(21, 0, 0, 0).in_tz("America/New_York")?;
/// // Goes from days to hours:
/// assert_eq!(
///     1.day().round(SpanRound::new().largest(Unit::Hour).relative(&zdt))?,
///     25.hours().fieldwise(),
/// );
/// // Goes from hours to days:
/// assert_eq!(
///     25.hours().round(SpanRound::new().largest(Unit::Day).relative(&zdt))?,
///     1.day().fieldwise(),
/// );
/// // 24 hours is less than 1 day starting at this time,
/// // so it stays in units of hours even though we ask
/// // for days (because 24 isn't enough hours to make
/// // 1 day):
/// assert_eq!(
///     24.hours().round(SpanRound::new().largest(Unit::Day).relative(&zdt))?,
///     24.hours().fieldwise(),
/// );
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// The APIs on `Span` will otherwise treat days as non-uniform unless a
/// relative civil date is given, or there is an explicit opt-in to invariant
/// 24-hour days. For example:
///
/// ```
/// use jiff::{civil, SpanRelativeTo, ToSpan, Unit};
///
/// let span = 1.day();
///
/// // An error because days aren't always 24 hours:
/// assert_eq!(
///     span.total(Unit::Hour).unwrap_err().to_string(),
///     "using unit 'day' in a span or configuration requires that either \
///      a relative reference time be given or \
///      `jiff::SpanRelativeTo::days_are_24_hours()` is used to indicate \
///      invariant 24-hour days, but neither were provided",
/// );
/// // Opt into invariant 24 hour days without a relative date:
/// let marker = SpanRelativeTo::days_are_24_hours();
/// let hours = span.total((Unit::Hour, marker))?;
/// assert_eq!(hours, 24.0);
/// // Or use a relative civil date, and all days are 24 hours:
/// let date = civil::date(2020, 1, 1);
/// let hours = span.total((Unit::Hour, date))?;
/// assert_eq!(hours, 24.0);
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// In Jiff, all weeks are 7 days. And generally speaking, weeks only appear in
/// a `Span` if they were explicitly put there by the caller or if they were
/// explicitly requested by the caller in an API. For example:
///
/// ```
/// use jiff::{civil::date, ToSpan, Unit};
///
/// let dt1 = date(2024, 1, 1).at(0, 0, 0, 0);
/// let dt2 = date(2024, 7, 16).at(0, 0, 0, 0);
/// // Default units go up to days.
/// assert_eq!(dt1.until(dt2)?, 197.days().fieldwise());
/// // No weeks, even though we requested up to year.
/// assert_eq!(dt1.until((Unit::Year, dt2))?, 6.months().days(15).fieldwise());
/// // We get weeks only when we ask for them.
/// assert_eq!(dt1.until((Unit::Week, dt2))?, 28.weeks().days(1).fieldwise());
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Integration with [`std::time::Duration`] and [`SignedDuration`]
///
/// While Jiff primarily uses a `Span` for doing arithmetic on datetimes,
/// one can convert between a `Span` and a [`std::time::Duration`] or a
/// [`SignedDuration`]. The main difference between them is that a `Span`
/// always keeps tracks of its individual units, and a `Span` can represent
/// non-uniform units like months. In contrast, `Duration` and `SignedDuration`
/// are always an exact elapsed amount of time. They don't distinguish between
/// `120 seconds` and `2 minutes`. And they can't represent the concept of
/// "months" because a month doesn't have a single fixed amount of time.
///
/// However, an exact duration is still useful in certain contexts. Beyond
/// that, it serves as an interoperability point due to the presence of an
/// unsigned exact duration type in the standard library. Because of that,
/// Jiff provides `TryFrom` trait implementations for converting to and from a
/// `std::time::Duration` (and, of course, a `SignedDuration`). For example, to
/// convert from a `std::time::Duration` to a `Span`:
///
/// ```
/// use std::time::Duration;
///
/// use jiff::{Span, ToSpan};
///
/// let duration = Duration::new(86_400, 123_456_789);
/// let span = Span::try_from(duration)?;
/// // A duration-to-span conversion always results in a span with
/// // non-zero units no bigger than seconds.
/// assert_eq!(
///     span.fieldwise(),
///     86_400.seconds().milliseconds(123).microseconds(456).nanoseconds(789),
/// );
///
/// // Note that the conversion is fallible! For example:
/// assert!(Span::try_from(Duration::from_secs(u64::MAX)).is_err());
/// // At present, a Jiff `Span` can only represent a range of time equal to
/// // the range of time expressible via minimum and maximum Jiff timestamps.
/// // Which is roughly -9999-01-01 to 9999-12-31, or ~20,000 years.
/// assert!(Span::try_from(Duration::from_secs(999_999_999_999)).is_err());
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// And to convert from a `Span` to a `std::time::Duration`:
///
/// ```
/// use std::time::Duration;
///
/// use jiff::{Span, ToSpan};
///
/// let span = 86_400.seconds()
///     .milliseconds(123)
///     .microseconds(456)
///     .nanoseconds(789);
/// let duration = Duration::try_from(span)?;
/// assert_eq!(duration, Duration::new(86_400, 123_456_789));
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// Note that an error will occur when converting a `Span` to a
/// `std::time::Duration` using the `TryFrom` trait implementation with units
/// bigger than hours:
///
/// ```
/// use std::time::Duration;
///
/// use jiff::ToSpan;
///
/// let span = 2.days().hours(10);
/// assert_eq!(
///     Duration::try_from(span).unwrap_err().to_string(),
///     "failed to convert span to duration without relative datetime \
///      (must use `jiff::Span::to_duration` instead): using unit 'day' \
///      in a span or configuration requires that either a relative \
///      reference time be given or \
///      `jiff::SpanRelativeTo::days_are_24_hours()` is used to indicate \
///      invariant 24-hour days, but neither were provided",
/// );
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// Similar code can be written for `SignedDuration` as well.
///
/// If you need to convert such spans, then as the error suggests, you'll need
/// to use [`Span::to_duration`] with a relative date.
///
/// And note that since a `Span` is signed and a `std::time::Duration` is unsigned,
/// converting a negative `Span` to `std::time::Duration` will always fail. One can use
/// [`Span::signum`] to get the sign of the span and [`Span::abs`] to make the
/// span positive before converting it to a `Duration`:
///
/// ```
/// use std::time::Duration;
///
/// use jiff::{Span, ToSpan};
///
/// let span = -86_400.seconds().nanoseconds(1);
/// let (sign, duration) = (span.signum(), Duration::try_from(span.abs())?);
/// assert_eq!((sign, duration), (-1, Duration::new(86_400, 1)));
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// Or, consider using Jiff's own [`SignedDuration`] instead:
///
/// ```
/// # // See: https://github.com/rust-lang/rust/pull/121364
/// # #![allow(unknown_lints, ambiguous_negative_literals)]
/// use jiff::{SignedDuration, Span, ToSpan};
///
/// let span = -86_400.seconds().nanoseconds(1);
/// let duration = SignedDuration::try_from(span)?;
/// assert_eq!(duration, SignedDuration::new(-86_400, -1));
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Default)]
pub struct Span {
    sign: b::Sign,
    units: UnitSet,
    years: i16,
    months: i32,
    weeks: i32,
    days: i32,
    hours: i32,
    minutes: i64,
    seconds: i64,
    milliseconds: i64,
    microseconds: i64,
    nanoseconds: i64,
}

/// Infallible routines for setting units on a `Span`.
///
/// These are useful when the units are determined by the programmer or when
/// they have been validated elsewhere. In general, use these routines when
/// constructing an invalid `Span` should be considered a bug in the program.
impl Span {
    /// Creates a new span representing a zero duration. That is, a duration
    /// in which no time has passed.
    pub fn new() -> Span {
        Span::default()
    }

    /// Set the number of years on this span. The value may be negative.
    ///
    /// The fallible version of this method is [`Span::try_years`].
    ///
    /// # Panics
    ///
    /// This panics when the number of years is too small or too big.
    /// The minimum value is `-19,998`.
    /// The maximum value is `19,998`.
    #[inline]
    pub fn years<I: Into<i64>>(self, years: I) -> Span {
        self.try_years(years).expect("value for years is out of bounds")
    }

    /// Set the number of months on this span. The value may be negative.
    ///
    /// The fallible version of this method is [`Span::try_months`].
    ///
    /// # Panics
    ///
    /// This panics when the number of months is too small or too big.
    /// The minimum value is `-239,976`.
    /// The maximum value is `239,976`.
    #[inline]
    pub fn months<I: Into<i64>>(self, months: I) -> Span {
        self.try_months(months).expect("value for months is out of bounds")
    }

    /// Set the number of weeks on this span. The value may be negative.
    ///
    /// The fallible version of this method is [`Span::try_weeks`].
    ///
    /// # Panics
    ///
    /// This panics when the number of weeks is too small or too big.
    /// The minimum value is `-1,043,497`.
    /// The maximum value is `1_043_497`.
    #[inline]
    pub fn weeks<I: Into<i64>>(self, weeks: I) -> Span {
        self.try_weeks(weeks).expect("value for weeks is out of bounds")
    }

    /// Set the number of days on this span. The value may be negative.
    ///
    /// The fallible version of this method is [`Span::try_days`].
    ///
    /// # Panics
    ///
    /// This panics when the number of days is too small or too big.
    /// The minimum value is `-7,304,484`.
    /// The maximum value is `7,304,484`.
    #[inline]
    pub fn days<I: Into<i64>>(self, days: I) -> Span {
        self.try_days(days).expect("value for days is out of bounds")
    }

    /// Set the number of hours on this span. The value may be negative.
    ///
    /// The fallible version of this method is [`Span::try_hours`].
    ///
    /// # Panics
    ///
    /// This panics when the number of hours is too small or too big.
    /// The minimum value is `-175,307,616`.
    /// The maximum value is `175,307,616`.
    #[inline]
    pub fn hours<I: Into<i64>>(self, hours: I) -> Span {
        self.try_hours(hours).expect("value for hours is out of bounds")
    }

    /// Set the number of minutes on this span. The value may be negative.
    ///
    /// The fallible version of this method is [`Span::try_minutes`].
    ///
    /// # Panics
    ///
    /// This panics when the number of minutes is too small or too big.
    /// The minimum value is `-10,518,456,960`.
    /// The maximum value is `10,518,456,960`.
    #[inline]
    pub fn minutes<I: Into<i64>>(self, minutes: I) -> Span {
        self.try_minutes(minutes).expect("value for minutes is out of bounds")
    }

    /// Set the number of seconds on this span. The value may be negative.
    ///
    /// The fallible version of this method is [`Span::try_seconds`].
    ///
    /// # Panics
    ///
    /// This panics when the number of seconds is too small or too big.
    /// The minimum value is `-631,107,417,600`.
    /// The maximum value is `631,107,417,600`.
    #[inline]
    pub fn seconds<I: Into<i64>>(self, seconds: I) -> Span {
        self.try_seconds(seconds).expect("value for seconds is out of bounds")
    }

    /// Set the number of milliseconds on this span. The value may be negative.
    ///
    /// The fallible version of this method is [`Span::try_milliseconds`].
    ///
    /// # Panics
    ///
    /// This panics when the number of milliseconds is too small or too big.
    /// The minimum value is `-631,107,417,600,000`.
    /// The maximum value is `631,107,417,600,000`.
    #[inline]
    pub fn milliseconds<I: Into<i64>>(self, milliseconds: I) -> Span {
        self.try_milliseconds(milliseconds)
            .expect("value for milliseconds is out of bounds")
    }

    /// Set the number of microseconds on this span. The value may be negative.
    ///
    /// The fallible version of this method is [`Span::try_microseconds`].
    ///
    /// # Panics
    ///
    /// This panics when the number of microseconds is too small or too big.
    /// The minimum value is `-631,107,417,600,000,000`.
    /// The maximum value is `631,107,417,600,000,000`.
    #[inline]
    pub fn microseconds<I: Into<i64>>(self, microseconds: I) -> Span {
        self.try_microseconds(microseconds)
            .expect("value for microseconds is out of bounds")
    }

    /// Set the number of nanoseconds on this span. The value may be negative.
    ///
    /// Note that unlike all other units, a 64-bit integer number of
    /// nanoseconds is not big enough to represent all possible spans between
    /// all possible datetimes supported by Jiff. This means, for example, that
    /// computing a span between two datetimes that are far enough apart _and_
    /// requesting a largest unit of [`Unit::Nanosecond`], might return an
    /// error due to lack of precision.
    ///
    /// The fallible version of this method is [`Span::try_nanoseconds`].
    ///
    /// # Panics
    ///
    /// This panics when the number of nanoseconds is too small or too big.
    /// The minimum value is `-9,223,372,036,854,775,807`.
    /// The maximum value is `9,223,372,036,854,775,807`.
    #[inline]
    pub fn nanoseconds<I: Into<i64>>(self, nanoseconds: I) -> Span {
        self.try_nanoseconds(nanoseconds)
            .expect("value for nanoseconds is out of bounds")
    }
}

/// Fallible methods for setting units on a `Span`.
///
/// These methods are useful when the span is made up of user provided values
/// that may not be in range.
impl Span {
    /// Set the number of years on this span. The value may be negative.
    ///
    /// The panicking version of this method is [`Span::years`].
    ///
    /// # Errors
    ///
    /// This returns an error when the number of years is too small or too big.
    /// The minimum value is `-19,998`.
    /// The maximum value is `19,998`.
    #[inline]
    pub fn try_years<I: Into<i64>>(self, years: I) -> Result<Span, Error> {
        let years = b::SpanYears::check(years.into())?;
        let mut span = self.years_unchecked(years.abs());
        span.sign = self.resign(years, &span);
        Ok(span)
    }

    /// Set the number of months on this span. The value may be negative.
    ///
    /// The panicking version of this method is [`Span::months`].
    ///
    /// # Errors
    ///
    /// This returns an error when the number of months is too small or too big.
    /// The minimum value is `-239,976`.
    /// The maximum value is `239,976`.
    #[inline]
    pub fn try_months<I: Into<i64>>(self, months: I) -> Result<Span, Error> {
        let months = b::SpanMonths::check(months.into())?;
        let mut span = self.months_unchecked(months.abs());
        span.sign = self.resign(months, &span);
        Ok(span)
    }

    /// Set the number of weeks on this span. The value may be negative.
    ///
    /// The panicking version of this method is [`Span::weeks`].
    ///
    /// # Errors
    ///
    /// This returns an error when the number of weeks is too small or too big.
    /// The minimum value is `-1,043,497`.
    /// The maximum value is `1_043_497`.
    #[inline]
    pub fn try_weeks<I: Into<i64>>(self, weeks: I) -> Result<Span, Error> {
        let weeks = b::SpanWeeks::check(weeks.into())?;
        let mut span = self.weeks_unchecked(weeks.abs());
        span.sign = self.resign(weeks, &span);
        Ok(span)
    }

    /// Set the number of days on this span. The value may be negative.
    ///
    /// The panicking version of this method is [`Span::days`].
    ///
    /// # Errors
    ///
    /// This returns an error when the number of days is too small or too big.
    /// The minimum value is `-7,304,484`.
    /// The maximum value is `7,304,484`.
    #[inline]
    pub fn try_days<I: Into<i64>>(self, days: I) -> Result<Span, Error> {
        let days = b::SpanDays::check(days.into())?;
        let mut span = self.days_unchecked(days.abs());
        span.sign = self.resign(days, &span);
        Ok(span)
    }

    /// Set the number of hours on this span. The value may be negative.
    ///
    /// The panicking version of this method is [`Span::hours`].
    ///
    /// # Errors
    ///
    /// This returns an error when the number of hours is too small or too big.
    /// The minimum value is `-175,307,616`.
    /// The maximum value is `175,307,616`.
    #[inline]
    pub fn try_hours<I: Into<i64>>(self, hours: I) -> Result<Span, Error> {
        let hours = b::SpanHours::check(hours.into())?;
        let mut span = self.hours_unchecked(hours.abs());
        span.sign = self.resign(hours, &span);
        Ok(span)
    }

    /// Set the number of minutes on this span. The value may be negative.
    ///
    /// The panicking version of this method is [`Span::minutes`].
    ///
    /// # Errors
    ///
    /// This returns an error when the number of minutes is too small or too big.
    /// The minimum value is `-10,518,456,960`.
    /// The maximum value is `10,518,456,960`.
    #[inline]
    pub fn try_minutes<I: Into<i64>>(self, minutes: I) -> Result<Span, Error> {
        let minutes = b::SpanMinutes::check(minutes.into())?;
        let mut span = self.minutes_unchecked(minutes.abs());
        span.sign = self.resign(minutes, &span);
        Ok(span)
    }

    /// Set the number of seconds on this span. The value may be negative.
    ///
    /// The panicking version of this method is [`Span::seconds`].
    ///
    /// # Errors
    ///
    /// This returns an error when the number of seconds is too small or too big.
    /// The minimum value is `-631,107,417,600`.
    /// The maximum value is `631,107,417,600`.
    #[inline]
    pub fn try_seconds<I: Into<i64>>(self, seconds: I) -> Result<Span, Error> {
        let seconds = b::SpanSeconds::check(seconds.into())?;
        let mut span = self.seconds_unchecked(seconds.abs());
        span.sign = self.resign(seconds, &span);
        Ok(span)
    }

    /// Set the number of milliseconds on this span. The value may be negative.
    ///
    /// The panicking version of this method is [`Span::milliseconds`].
    ///
    /// # Errors
    ///
    /// This returns an error when the number of milliseconds is too small or
    /// too big.
    /// The minimum value is `-631,107,417,600,000`.
    /// The maximum value is `631,107,417,600,000`.
    #[inline]
    pub fn try_milliseconds<I: Into<i64>>(
        self,
        milliseconds: I,
    ) -> Result<Span, Error> {
        let milliseconds = b::SpanMilliseconds::check(milliseconds.into())?;
        let mut span = self.milliseconds_unchecked(milliseconds.abs());
        span.sign = self.resign(milliseconds, &span);
        Ok(span)
    }

    /// Set the number of microseconds on this span. The value may be negative.
    ///
    /// The panicking version of this method is [`Span::microseconds`].
    ///
    /// # Errors
    ///
    /// This returns an error when the number of microseconds is too small or
    /// too big.
    /// The minimum value is `-631,107,417,600,000,000`.
    /// The maximum value is `631,107,417,600,000,000`.
    #[inline]
    pub fn try_microseconds<I: Into<i64>>(
        self,
        microseconds: I,
    ) -> Result<Span, Error> {
        let microseconds = b::SpanMicroseconds::check(microseconds.into())?;
        let mut span = self.microseconds_unchecked(microseconds.abs());
        span.sign = self.resign(microseconds, &span);
        Ok(span)
    }

    /// Set the number of nanoseconds on this span. The value may be negative.
    ///
    /// Note that unlike all other units, a 64-bit integer number of
    /// nanoseconds is not big enough to represent all possible spans between
    /// all possible datetimes supported by Jiff. This means, for example, that
    /// computing a span between two datetimes that are far enough apart _and_
    /// requesting a largest unit of [`Unit::Nanosecond`], might return an
    /// error due to lack of precision.
    ///
    /// The panicking version of this method is [`Span::nanoseconds`].
    ///
    /// # Errors
    ///
    /// This returns an error when the number of nanoseconds is too small or
    /// too big.
    /// The minimum value is `-9,223,372,036,854,775,807`.
    /// The maximum value is `9,223,372,036,854,775,807`.
    #[inline]
    pub fn try_nanoseconds<I: Into<i64>>(
        self,
        nanoseconds: I,
    ) -> Result<Span, Error> {
        let nanoseconds = b::SpanNanoseconds::check(nanoseconds.into())?;
        let mut span = self.nanoseconds_unchecked(nanoseconds.abs());
        span.sign = self.resign(nanoseconds, &span);
        Ok(span)
    }
}

/// Routines for accessing the individual units in a `Span`.
impl Span {
    /// Returns the number of year units in this span.
    ///
    /// Note that this is not the same as the total number of years in the
    /// span. To get that, you'll need to use either [`Span::round`] or
    /// [`Span::total`].
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::{civil::date, ToSpan, Unit};
    ///
    /// let span = 3.years().months(24);
    /// assert_eq!(3, span.get_years());
    /// assert_eq!(5.0, span.total((Unit::Year, date(2024, 1, 1)))?);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn get_years(&self) -> i16 {
        self.sign * self.years
    }

    /// Returns the number of month units in this span.
    ///
    /// Note that this is not the same as the total number of months in the
    /// span. To get that, you'll need to use either [`Span::round`] or
    /// [`Span::total`].
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::{civil::date, ToSpan, Unit};
    ///
    /// let span = 7.months().days(59);
    /// assert_eq!(7, span.get_months());
    /// assert_eq!(9.0, span.total((Unit::Month, date(2022, 6, 1)))?);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn get_months(&self) -> i32 {
        self.sign * self.months
    }

    /// Returns the number of week units in this span.
    ///
    /// Note that this is not the same as the total number of weeks in the
    /// span. To get that, you'll need to use either [`Span::round`] or
    /// [`Span::total`].
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::{civil::date, ToSpan, Unit};
    ///
    /// let span = 3.weeks().days(14);
    /// assert_eq!(3, span.get_weeks());
    /// assert_eq!(5.0, span.total((Unit::Week, date(2024, 1, 1)))?);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn get_weeks(&self) -> i32 {
        self.sign * self.weeks
    }

    /// Returns the number of day units in this span.
    ///
    /// Note that this is not the same as the total number of days in the
    /// span. To get that, you'll need to use either [`Span::round`] or
    /// [`Span::total`].
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::{ToSpan, Unit, Zoned};
    ///
    /// let span = 3.days().hours(47);
    /// assert_eq!(3, span.get_days());
    ///
    /// let zdt: Zoned = "2024-03-07[America/New_York]".parse()?;
    /// assert_eq!(5.0, span.total((Unit::Day, &zdt))?);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn get_days(&self) -> i32 {
        self.sign * self.days
    }

    /// Returns the number of hour units in this span.
    ///
    /// Note that this is not the same as the total number of hours in the
    /// span. To get that, you'll need to use either [`Span::round`] or
    /// [`Span::total`].
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::{ToSpan, Unit};
    ///
    /// let span = 3.hours().minutes(120);
    /// assert_eq!(3, span.get_hours());
    /// assert_eq!(5.0, span.total(Unit::Hour)?);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn get_hours(&self) -> i32 {
        self.sign * self.hours
    }

    /// Returns the number of minute units in this span.
    ///
    /// Note that this is not the same as the total number of minutes in the
    /// span. To get that, you'll need to use either [`Span::round`] or
    /// [`Span::total`].
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::{ToSpan, Unit};
    ///
    /// let span = 3.minutes().seconds(120);
    /// assert_eq!(3, span.get_minutes());
    /// assert_eq!(5.0, span.total(Unit::Minute)?);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn get_minutes(&self) -> i64 {
        self.sign * self.minutes
    }

    /// Returns the number of second units in this span.
    ///
    /// Note that this is not the same as the total number of seconds in the
    /// span. To get that, you'll need to use either [`Span::round`] or
    /// [`Span::total`].
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::{ToSpan, Unit};
    ///
    /// let span = 3.seconds().milliseconds(2_000);
    /// assert_eq!(3, span.get_seconds());
    /// assert_eq!(5.0, span.total(Unit::Second)?);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn get_seconds(&self) -> i64 {
        self.sign * self.seconds
    }

    /// Returns the number of millisecond units in this span.
    ///
    /// Note that this is not the same as the total number of milliseconds in
    /// the span. To get that, you'll need to use either [`Span::round`] or
    /// [`Span::total`].
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::{ToSpan, Unit};
    ///
    /// let span = 3.milliseconds().microseconds(2_000);
    /// assert_eq!(3, span.get_milliseconds());
    /// assert_eq!(5.0, span.total(Unit::Millisecond)?);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn get_milliseconds(&self) -> i64 {
        self.sign * self.milliseconds
    }

    /// Returns the number of microsecond units in this span.
    ///
    /// Note that this is not the same as the total number of microseconds in
    /// the span. To get that, you'll need to use either [`Span::round`] or
    /// [`Span::total`].
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::{ToSpan, Unit};
    ///
    /// let span = 3.microseconds().nanoseconds(2_000);
    /// assert_eq!(3, span.get_microseconds());
    /// // Floating point precision may provide imprecise results.
    /// assert_eq!(5.0, span.total(Unit::Microsecond)?);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn get_microseconds(&self) -> i64 {
        self.sign * self.microseconds
    }

    /// Returns the number of nanosecond units in this span.
    ///
    /// Note that this is not the same as the total number of nanoseconds in
    /// the span. To get that, you'll need to use either [`Span::round`] or
    /// [`Span::total`].
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::{ToSpan, Unit};
    ///
    /// let span = 3.microseconds().nanoseconds(2_000);
    /// assert_eq!(2_000, span.get_nanoseconds());
    /// assert_eq!(5_000.0, span.total(Unit::Nanosecond)?);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn get_nanoseconds(&self) -> i64 {
        self.sign * self.nanoseconds
    }
}

/// Routines for manipulating, comparing and inspecting `Span` values.
impl Span {
    /// Returns a new span that is the absolute value of this span.
    ///
    /// If this span is zero or positive, then this is a no-op.
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::ToSpan;
    ///
    /// let span = -100.seconds();
    /// assert_eq!(span.to_string(), "-PT100S");
    /// let span = span.abs();
    /// assert_eq!(span.to_string(), "PT100S");
    /// ```
    #[inline]
    pub fn abs(self) -> Span {
        if self.is_zero() {
            return self;
        }
        Span { sign: b::Sign::Positive, ..self }
    }

    /// Returns a new span that negates this span.
    ///
    /// If this span is zero, then this is a no-op. If this span is negative,
    /// then the returned span is positive. If this span is positive, then
    /// the returned span is negative.
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::ToSpan;
    ///
    /// let span = 100.days();
    /// assert_eq!(span.to_string(), "P100D");
    /// let span = span.negate();
    /// assert_eq!(span.to_string(), "-P100D");
    /// ```
    ///
    /// # Example: available via the negation operator
    ///
    /// This routine can also be used via `-`:
    ///
    /// ```
    /// use jiff::ToSpan;
    ///
    /// let span = 100.days();
    /// assert_eq!(span.to_string(), "P100D");
    /// let span = -span;
    /// assert_eq!(span.to_string(), "-P100D");
    /// ```
    #[inline]
    pub fn negate(self) -> Span {
        Span { sign: -self.sign, ..self }
    }

    /// Returns the "sign number" or "signum" of this span.
    ///
    /// The number returned is `-1` when this span is negative,
    /// `0` when this span is zero and `1` when this span is positive.
    #[inline]
    pub fn signum(self) -> i8 {
        self.sign.signum()
    }

    /// Returns true if and only if this span is positive.
    ///
    /// This returns false when the span is zero or negative.
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::ToSpan;
    ///
    /// assert!(!2.months().is_negative());
    /// assert!((-2.months()).is_negative());
    /// ```
    #[inline]
    pub fn is_positive(self) -> bool {
        self.get_sign().is_positive()
    }

    /// Returns true if and only if this span is negative.
    ///
    /// This returns false when the span is zero or positive.
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::ToSpan;
    ///
    /// assert!(!2.months().is_negative());
    /// assert!((-2.months()).is_negative());
    /// ```
    #[inline]
    pub fn is_negative(self) -> bool {
        self.get_sign().is_negative()
    }

    /// Returns true if and only if every field in this span is set to `0`.
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::{Span, ToSpan};
    ///
    /// assert!(Span::new().is_zero());
    /// assert!(Span::default().is_zero());
    /// assert!(0.seconds().is_zero());
    /// assert!(!0.seconds().seconds(1).is_zero());
    /// assert!(0.seconds().seconds(1).seconds(0).is_zero());
    /// ```
    #[inline]
    pub fn is_zero(self) -> bool {
        self.sign.is_zero()
    }

    /// Returns this `Span` as a value with a type that implements the
    /// `Hash`, `Eq` and `PartialEq` traits in a fieldwise fashion.
    ///
    /// A `SpanFieldwise` is meant to make it easy to compare two spans in a
    /// "dumb" way based purely on its unit values. This is distinct from
    /// something like [`Span::compare`] that performs a comparison on the
    /// actual elapsed time of two spans.
    ///
    /// It is generally discouraged to use `SpanFieldwise` since spans that
    /// represent an equivalent elapsed amount of time may compare unequal.
    /// However, in some cases, it is useful to be able to assert precise
    /// field values. For example, Jiff itself makes heavy use of fieldwise
    /// comparisons for tests.
    ///
    /// # Example: the difference between `SpanFieldwise` and `Span::compare`
    ///
    /// In short, `SpanFieldwise` considers `2 hours` and `120 minutes` to be
    /// distinct values, but `Span::compare` considers them to be equivalent:
    ///
    /// ```
    /// use std::cmp::Ordering;
    /// use jiff::ToSpan;
    ///
    /// assert_ne!(120.minutes().fieldwise(), 2.hours().fieldwise());
    /// assert_eq!(120.minutes().compare(2.hours())?, Ordering::Equal);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn fieldwise(self) -> SpanFieldwise {
        SpanFieldwise(self)
    }

    /// Multiplies each field in this span by a given integer.
    ///
    /// If this would cause any individual field in this span to overflow, then
    /// this returns an error.
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::ToSpan;
    ///
    /// let span = 4.days().seconds(8);
    /// assert_eq!(span.checked_mul(2)?, 8.days().seconds(16).fieldwise());
    /// assert_eq!(span.checked_mul(-3)?, -12.days().seconds(24).fieldwise());
    /// // Notice that no re-balancing is done. It's "just" multiplication.
    /// assert_eq!(span.checked_mul(10)?, 40.days().seconds(80).fieldwise());
    ///
    /// let span = 10_000.years();
    /// // too big!
    /// assert!(span.checked_mul(3).is_err());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Example: available via the multiplication operator
    ///
    /// This method can be used via the `*` operator. Note though that a panic
    /// happens on overflow.
    ///
    /// ```
    /// use jiff::ToSpan;
    ///
    /// let span = 4.days().seconds(8);
    /// assert_eq!(span * 2, 8.days().seconds(16).fieldwise());
    /// assert_eq!(2 * span, 8.days().seconds(16).fieldwise());
    /// assert_eq!(span * -3, -12.days().seconds(24).fieldwise());
    /// assert_eq!(-3 * span, -12.days().seconds(24).fieldwise());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn checked_mul(mut self, rhs: i64) -> Result<Span, Error> {
        if rhs == 0 {
            return Ok(Span::default());
        } else if rhs == 1 {
            return Ok(self);
        }
        self.sign = self.sign * b::Sign::from(rhs);
        // This can only fail when `rhs == i64::MIN`, which is out of bounds
        // for all possible span units (including nanoseconds).
        let rhs = rhs.checked_abs().ok_or_else(b::SpanMultiple::error)?;
        // This is all somewhat odd, but since each of our span fields uses
        // a different primitive representation and range of allowed values,
        // we only seek to perform multiplications when they will actually
        // do something. Otherwise, we risk multiplying the mins/maxs of a
        // ranged integer and causing a spurious panic. Basically, the idea
        // here is the allowable values for our multiple depend on what we're
        // actually going to multiply with it. If our span has non-zero years,
        // then our multiple can't exceed the bounds of `SpanYears`, otherwise
        // it is guaranteed to overflow.
        if self.years != 0 {
            let rhs = b::SpanYears::check(rhs)?;
            self.years = b::SpanYears::checked_mul(self.years, rhs)?;
        }
        if self.months != 0 {
            let rhs = b::SpanMonths::check(rhs)?;
            self.months = b::SpanMonths::checked_mul(self.months, rhs)?;
        }
        if self.weeks != 0 {
            let rhs = b::SpanWeeks::check(rhs)?;
            self.weeks = b::SpanWeeks::checked_mul(self.weeks, rhs)?;
        }
        if self.days != 0 {
            let rhs = b::SpanDays::check(rhs)?;
            self.days = b::SpanDays::checked_mul(self.days, rhs)?;
        }
        if self.hours != 0 {
            let rhs = b::SpanHours::check(rhs)?;
            self.hours = b::SpanHours::checked_mul(self.hours, rhs)?;
        }
        if self.minutes != 0 {
            self.minutes = b::SpanMinutes::checked_mul(self.minutes, rhs)?;
        }
        if self.seconds != 0 {
            self.seconds = b::SpanSeconds::checked_mul(self.seconds, rhs)?;
        }
        if self.milliseconds != 0 {
            self.milliseconds =
                b::SpanMilliseconds::checked_mul(self.milliseconds, rhs)?;
        }
        if self.microseconds != 0 {
            self.microseconds =
                b::SpanMicroseconds::checked_mul(self.microseconds, rhs)?;
        }
        if self.nanoseconds != 0 {
            self.nanoseconds =
                b::SpanNanoseconds::checked_mul(self.nanoseconds, rhs)?;
        }
        // N.B. We don't need to update `self.units` here since it shouldn't
        // change. The only way it could is if a unit goes from zero to
        // non-zero (which can't happen, because multiplication by zero is
        // always zero), or if a unit goes from non-zero to zero. That also
        // can't happen because we handle the case of the factor being zero
        // specially above, and it returns a `Span` will all units zero
        // correctly.
        Ok(self)
    }

    /// Adds a span to this one and returns the sum as a new span.
    ///
    /// When adding a span with units greater than hours, callers must provide
    /// a relative datetime to anchor the spans.
    ///
    /// Arithmetic proceeds as specified in [RFC 5545]. Bigger units are
    /// added together before smaller units.
    ///
    /// This routine accepts anything that implements `Into<SpanArithmetic>`.
    /// There are some trait implementations that make using this routine
    /// ergonomic:
    ///
    /// * `From<Span> for SpanArithmetic` adds the given span to this one.
    /// * `From<(Span, civil::Date)> for SpanArithmetic` adds the given
    /// span to this one relative to the given date. There are also `From`
    /// implementations for `civil::DateTime` and `Zoned`.
    ///
    /// This also works with different duration types, such as
    /// [`SignedDuration`] and [`std::time::Duration`], via additional trait
    /// implementations:
    ///
    /// * `From<SignedDuration> for SpanArithmetic` adds the given duration to
    /// this one.
    /// * `From<(SignedDuration, civil::Date)> for SpanArithmetic` adds the
    /// given duration to this one relative to the given date. There are also
    /// `From` implementations for `civil::DateTime` and `Zoned`.
    ///
    /// And similarly for `std::time::Duration`.
    ///
    /// Adding a negative span is equivalent to subtracting its absolute value.
    ///
    /// The largest non-zero unit in the span returned is at most the largest
    /// non-zero unit among the two spans being added. For an absolute
    /// duration, its "largest" unit is considered to be nanoseconds.
    ///
    /// The sum returned is automatically re-balanced so that the span is not
    /// "bottom heavy."
    ///
    /// [RFC 5545]: https://datatracker.ietf.org/doc/html/rfc5545
    ///
    /// # Errors
    ///
    /// This returns an error when adding the two spans would overflow any
    /// individual field of a span. This will also return an error if either
    /// of the spans have non-zero units of days or greater and no relative
    /// reference time is provided.
    ///
    /// Callers may use [`SpanArithmetic::days_are_24_hours`] as a special
    /// marker instead of providing a relative civil date to indicate that
    /// all days should be 24 hours long. This also results in treating all
    /// weeks as seven 24 hour days (168 hours).
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::ToSpan;
    ///
    /// assert_eq!(
    ///     1.hour().checked_add(30.minutes())?,
    ///     1.hour().minutes(30).fieldwise(),
    /// );
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Example: re-balancing
    ///
    /// This example shows how units are automatically rebalanced into bigger
    /// units when appropriate.
    ///
    /// ```
    /// use jiff::ToSpan;
    ///
    /// let span1 = 2.hours().minutes(59);
    /// let span2 = 2.minutes();
    /// assert_eq!(span1.checked_add(span2)?, 3.hours().minutes(1).fieldwise());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Example: days are not assumed to be 24 hours by default
    ///
    /// When dealing with units involving days or weeks, one must either
    /// provide a relative datetime (shown in the following examples) or opt
    /// into invariant 24 hour days:
    ///
    /// ```
    /// use jiff::{SpanRelativeTo, ToSpan};
    ///
    /// let span1 = 2.days().hours(23);
    /// let span2 = 2.hours();
    /// assert_eq!(
    ///     span1.checked_add((span2, SpanRelativeTo::days_are_24_hours()))?,
    ///     3.days().hours(1).fieldwise(),
    /// );
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Example: adding spans with calendar units
    ///
    /// If you try to add two spans with calendar units without specifying a
    /// relative datetime, you'll get an error:
    ///
    /// ```
    /// use jiff::ToSpan;
    ///
    /// let span1 = 1.month().days(15);
    /// let span2 = 15.days();
    /// assert!(span1.checked_add(span2).is_err());
    /// ```
    ///
    /// A relative datetime is needed because calendar spans may correspond to
    /// different actual durations depending on where the span begins:
    ///
    /// ```
    /// use jiff::{civil::date, ToSpan};
    ///
    /// let span1 = 1.month().days(15);
    /// let span2 = 15.days();
    /// // 1 month from March 1 is 31 days...
    /// assert_eq!(
    ///     span1.checked_add((span2, date(2008, 3, 1)))?,
    ///     2.months().fieldwise(),
    /// );
    /// // ... but 1 month from April 1 is 30 days!
    /// assert_eq!(
    ///     span1.checked_add((span2, date(2008, 4, 1)))?,
    ///     1.month().days(30).fieldwise(),
    /// );
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Example: error on overflow
    ///
    /// Adding two spans can overflow, and this will result in an error:
    ///
    /// ```
    /// use jiff::ToSpan;
    ///
    /// assert!(19_998.years().checked_add(1.year()).is_err());
    /// ```
    ///
    /// # Example: adding an absolute duration to a span
    ///
    /// This shows how one isn't limited to just adding two spans together.
    /// One can also add absolute durations to a span.
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// use jiff::{SignedDuration, ToSpan};
    ///
    /// assert_eq!(
    ///     1.hour().checked_add(SignedDuration::from_mins(30))?,
    ///     1.hour().minutes(30).fieldwise(),
    /// );
    /// assert_eq!(
    ///     1.hour().checked_add(Duration::from_secs(30 * 60))?,
    ///     1.hour().minutes(30).fieldwise(),
    /// );
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// Note that even when adding an absolute duration, if the span contains
    /// non-uniform units, you still need to provide a relative datetime:
    ///
    /// ```
    /// use jiff::{civil::date, SignedDuration, ToSpan};
    ///
    /// // Might be 1 month or less than 1 month!
    /// let dur = SignedDuration::from_hours(30 * 24);
    /// // No relative datetime provided even when the span
    /// // contains non-uniform units results in an error.
    /// assert!(1.month().checked_add(dur).is_err());
    /// // In this case, 30 days is one month (April).
    /// assert_eq!(
    ///     1.month().checked_add((dur, date(2024, 3, 1)))?,
    ///     2.months().fieldwise(),
    /// );
    /// // In this case, 30 days is less than one month (May).
    /// assert_eq!(
    ///     1.month().checked_add((dur, date(2024, 4, 1)))?,
    ///     1.month().days(30).fieldwise(),
    /// );
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn checked_add<'a, A: Into<SpanArithmetic<'a>>>(
        &self,
        options: A,
    ) -> Result<Span, Error> {
        let options: SpanArithmetic<'_> = options.into();
        options.checked_add(*self)
    }

    #[inline]
    fn checked_add_span<'a>(
        &self,
        relative: Option<SpanRelativeTo<'a>>,
        span: &Span,
    ) -> Result<Span, Error> {
        let (span1, span2) = (*self, *span);
        let unit = span1.largest_unit().max(span2.largest_unit());
        let start = match relative {
            Some(r) => match r.to_relative(unit)? {
                None => return span1.checked_add_invariant(unit, &span2),
                Some(r) => r,
            },
            None => {
                requires_relative_date_err(unit)?;
                return span1.checked_add_invariant(unit, &span2);
            }
        };
        let mid = start.checked_add(span1)?;
        let end = mid.checked_add(span2)?;
        start.until(unit, &end)
    }

    #[inline]
    fn checked_add_duration<'a>(
        &self,
        relative: Option<SpanRelativeTo<'a>>,
        duration: SignedDuration,
    ) -> Result<Span, Error> {
        let (span1, dur2) = (*self, duration);
        let unit = span1.largest_unit();
        let start = match relative {
            Some(r) => match r.to_relative(unit)? {
                None => {
                    return span1.checked_add_invariant_duration(unit, dur2)
                }
                Some(r) => r,
            },
            None => {
                requires_relative_date_err(unit)?;
                return span1.checked_add_invariant_duration(unit, dur2);
            }
        };
        let mid = start.checked_add(span1)?;
        let end = mid.checked_add_duration(dur2)?;
        start.until(unit, &end)
    }

    /// Like `checked_add`, but only applies for invariant units. That is,
    /// when *both* spans whose non-zero units are all hours or smaller
    /// (or weeks or smaller with the "days are 24 hours" marker).
    #[inline]
    fn checked_add_invariant(
        &self,
        unit: Unit,
        span: &Span,
    ) -> Result<Span, Error> {
        assert!(unit <= Unit::Week);
        self.checked_add_invariant_duration(unit, span.to_invariant_duration())
    }

    /// Like `checked_add_invariant`, but adds an absolute duration.
    #[inline]
    fn checked_add_invariant_duration(
        &self,
        unit: Unit,
        rhs_duration: SignedDuration,
    ) -> Result<Span, Error> {
        assert!(unit <= Unit::Week);
        let self_duration = self.to_invariant_duration();
        // OK because maximal invariant `Span` is way below maximal
        // `SignedDuration`. Doubling it can never overflow.
        let sum = self_duration + rhs_duration;
        Span::from_invariant_duration(unit, sum)
    }

    /// This routine is identical to [`Span::checked_add`] with the given
    /// duration negated.
    ///
    /// # Errors
    ///
    /// This has the same error conditions as [`Span::checked_add`].
    ///
    /// # Example
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// use jiff::{SignedDuration, ToSpan};
    ///
    /// assert_eq!(
    ///     1.hour().checked_sub(30.minutes())?,
    ///     30.minutes().fieldwise(),
    /// );
    /// assert_eq!(
    ///     1.hour().checked_sub(SignedDuration::from_mins(30))?,
    ///     30.minutes().fieldwise(),
    /// );
    /// assert_eq!(
    ///     1.hour().checked_sub(Duration::from_secs(30 * 60))?,
    ///     30.minutes().fieldwise(),
    /// );
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn checked_sub<'a, A: Into<SpanArithmetic<'a>>>(
        &self,
        options: A,
    ) -> Result<Span, Error> {
        let mut options: SpanArithmetic<'_> = options.into();
        options.duration = options.duration.checked_neg()?;
        options.checked_add(*self)
    }

    /// Compares two spans in terms of how long they are. Negative spans are
    /// considered shorter than the zero span.
    ///
    /// Two spans compare equal when they correspond to the same duration
    /// of time, even if their individual fields are different. This is in
    /// contrast to the `Eq` trait implementation of `SpanFieldwise` (created
    /// by [`Span::fieldwise`]), which performs exact field-wise comparisons.
    /// This split exists because the comparison provided by this routine is
    /// "heavy" in that it may need to do datetime arithmetic to return an
    /// answer. In contrast, the `Eq` trait implementation is "cheap."
    ///
    /// This routine accepts anything that implements `Into<SpanCompare>`.
    /// There are some trait implementations that make using this routine
    /// ergonomic:
    ///
    /// * `From<Span> for SpanCompare` compares the given span to this one.
    /// * `From<(Span, civil::Date)> for SpanArithmetic` compares the given
    /// span to this one relative to the given date. There are also `From`
    /// implementations for `civil::DateTime` and `Zoned`.
    ///
    /// # Errors
    ///
    /// If either of the spans being compared have a non-zero calendar unit
    /// (units bigger than hours), then this routine requires a relative
    /// datetime. If one is not provided, then an error is returned.
    ///
    /// An error can also occur when adding either span to the relative
    /// datetime given results in overflow.
    ///
    /// Callers may use [`SpanArithmetic::days_are_24_hours`] as a special
    /// marker instead of providing a relative civil date to indicate that
    /// all days should be 24 hours long. This also results in treating all
    /// weeks as seven 24 hour days (168 hours).
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::ToSpan;
    ///
    /// let span1 = 3.hours();
    /// let span2 = 180.minutes();
    /// assert_eq!(span1.compare(span2)?, std::cmp::Ordering::Equal);
    /// // But notice that the two spans are not equal via `Eq`:
    /// assert_ne!(span1.fieldwise(), span2.fieldwise());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Example: negative spans are less than zero
    ///
    /// ```
    /// use jiff::ToSpan;
    ///
    /// let span1 = -1.second();
    /// let span2 = 0.seconds();
    /// assert_eq!(span1.compare(span2)?, std::cmp::Ordering::Less);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Example: comparisons take DST into account
    ///
    /// When a relative datetime is time zone aware, then DST is taken into
    /// account when comparing spans:
    ///
    /// ```
    /// use jiff::{civil, ToSpan, Zoned};
    ///
    /// let span1 = 79.hours().minutes(10);
    /// let span2 = 3.days().hours(7).seconds(630);
    /// let span3 = 3.days().hours(6).minutes(50);
    ///
    /// let relative: Zoned = "2020-11-01T00-07[America/Los_Angeles]".parse()?;
    /// let mut spans = [span1, span2, span3];
    /// spans.sort_by(|s1, s2| s1.compare((s2, &relative)).unwrap());
    /// assert_eq!(
    ///     spans.map(|sp| sp.fieldwise()),
    ///     [span1.fieldwise(), span3.fieldwise(), span2.fieldwise()],
    /// );
    ///
    /// // Compare with the result of sorting without taking DST into account.
    /// // We can that by providing a relative civil date:
    /// let relative = civil::date(2020, 11, 1);
    /// spans.sort_by(|s1, s2| s1.compare((s2, relative)).unwrap());
    /// assert_eq!(
    ///     spans.map(|sp| sp.fieldwise()),
    ///     [span3.fieldwise(), span1.fieldwise(), span2.fieldwise()],
    /// );
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// See the examples for [`Span::total`] if you want to sort spans without
    /// an `unwrap()` call.
    #[inline]
    pub fn compare<'a, C: Into<SpanCompare<'a>>>(
        &self,
        options: C,
    ) -> Result<Ordering, Error> {
        let options: SpanCompare<'_> = options.into();
        options.compare(*self)
    }

    /// Returns a floating point number representing the total number of a
    /// specific unit (as given) in this span. If the span is not evenly
    /// divisible by the requested units, then the number returned may have a
    /// fractional component.
    ///
    /// This routine accepts anything that implements `Into<SpanTotal>`. There
    /// are some trait implementations that make using this routine ergonomic:
    ///
    /// * `From<Unit> for SpanTotal` computes a total for the given unit in
    /// this span.
    /// * `From<(Unit, civil::Date)> for SpanTotal` computes a total for the
    /// given unit in this span, relative to the given date. There are also
    /// `From` implementations for `civil::DateTime` and `Zoned`.
    ///
    /// # Errors
    ///
    /// If this span has any non-zero calendar unit (units bigger than hours),
    /// then this routine requires a relative datetime. If one is not provided,
    /// then an error is returned.
    ///
    /// An error can also occur when adding the span to the relative
    /// datetime given results in overflow.
    ///
    /// Callers may use [`SpanArithmetic::days_are_24_hours`] as a special
    /// marker instead of providing a relative civil date to indicate that
    /// all days should be 24 hours long. This also results in treating all
    /// weeks as seven 24 hour days (168 hours).
    ///
    /// # Example
    ///
    /// This example shows how to find the number of seconds in a particular
    /// span:
    ///
    /// ```
    /// use jiff::{ToSpan, Unit};
    ///
    /// let span = 3.hours().minutes(10);
    /// assert_eq!(span.total(Unit::Second)?, 11_400.0);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Example: 24 hour days
    ///
    /// This shows how to find the total number of 24 hour days in
    /// `123,456,789` seconds.
    ///
    /// ```
    /// use jiff::{SpanTotal, ToSpan, Unit};
    ///
    /// let span = 123_456_789.seconds();
    /// assert_eq!(
    ///     span.total(SpanTotal::from(Unit::Day).days_are_24_hours())?,
    ///     1428.8980208333332,
    /// );
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Example: DST is taken into account
    ///
    /// The month of March 2024 in `America/New_York` had 31 days, but one of
    /// those days was 23 hours long due a transition into daylight saving
    /// time:
    ///
    /// ```
    /// use jiff::{civil::date, ToSpan, Unit};
    ///
    /// let span = 744.hours();
    /// let relative = date(2024, 3, 1).in_tz("America/New_York")?;
    /// // Because of the short day, 744 hours is actually a little *more* than
    /// // 1 month starting from 2024-03-01.
    /// assert_eq!(span.total((Unit::Month, &relative))?, 1.0013888888888889);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// Now compare what happens when the relative datetime is civil and not
    /// time zone aware:
    ///
    /// ```
    /// use jiff::{civil::date, ToSpan, Unit};
    ///
    /// let span = 744.hours();
    /// let relative = date(2024, 3, 1);
    /// assert_eq!(span.total((Unit::Month, relative))?, 1.0);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Example: infallible sorting
    ///
    /// The sorting example in [`Span::compare`] has to use `unwrap()` in
    /// its `sort_by(..)` call because `Span::compare` may fail and there
    /// is no "fallible" sorting routine in Rust's standard library (as of
    /// 2024-07-07). While the ways in which `Span::compare` can fail for
    /// a valid configuration are limited to overflow for "extreme" values, it
    /// is possible to sort spans infallibly by computing floating point
    /// representations for each span up-front:
    ///
    /// ```
    /// use jiff::{civil::Date, ToSpan, Unit, Zoned};
    ///
    /// let span1 = 79.hours().minutes(10);
    /// let span2 = 3.days().hours(7).seconds(630);
    /// let span3 = 3.days().hours(6).minutes(50);
    ///
    /// let relative: Zoned = "2020-11-01T00-07[America/Los_Angeles]".parse()?;
    /// let mut spans = [
    ///     (span1, span1.total((Unit::Day, &relative))?),
    ///     (span2, span2.total((Unit::Day, &relative))?),
    ///     (span3, span3.total((Unit::Day, &relative))?),
    /// ];
    /// spans.sort_by(|&(_, total1), &(_, total2)| total1.total_cmp(&total2));
    /// assert_eq!(
    ///     spans.map(|(sp, _)| sp.fieldwise()),
    ///     [span1.fieldwise(), span3.fieldwise(), span2.fieldwise()],
    /// );
    ///
    /// // Compare with the result of sorting without taking DST into account.
    /// // We do that here by providing a relative civil date.
    /// let relative: Date = "2020-11-01".parse()?;
    /// let mut spans = [
    ///     (span1, span1.total((Unit::Day, relative))?),
    ///     (span2, span2.total((Unit::Day, relative))?),
    ///     (span3, span3.total((Unit::Day, relative))?),
    /// ];
    /// spans.sort_by(|&(_, total1), &(_, total2)| total1.total_cmp(&total2));
    /// assert_eq!(
    ///     spans.map(|(sp, _)| sp.fieldwise()),
    ///     [span3.fieldwise(), span1.fieldwise(), span2.fieldwise()],
    /// );
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn total<'a, T: Into<SpanTotal<'a>>>(
        &self,
        options: T,
    ) -> Result<f64, Error> {
        let options: SpanTotal<'_> = options.into();
        options.total(*self)
    }

    /// Returns a new span that is balanced and rounded.
    ///
    /// Rounding a span has a number of parameters, all of which are optional.
    /// When no parameters are given, then no rounding or balancing is done,
    /// and the span as given is returned. That is, it's a no-op.
    ///
    /// The parameters are, in brief:
    ///
    /// * [`SpanRound::largest`] sets the largest [`Unit`] that is allowed to
    /// be non-zero in the span returned. When _only_ the largest unit is set,
    /// rounding itself doesn't occur and instead the span is merely balanced.
    /// * [`SpanRound::smallest`] sets the smallest [`Unit`] that is allowed to
    /// be non-zero in the span returned. By default, it is set to
    /// [`Unit::Nanosecond`], i.e., no rounding occurs. When the smallest unit
    /// is set to something bigger than nanoseconds, then the non-zero units
    /// in the span smaller than the smallest unit are used to determine how
    /// the span should be rounded. For example, rounding `1 hour 59 minutes`
    /// to the nearest hour using the default rounding mode would produce
    /// `2 hours`.
    /// * [`SpanRound::mode`] determines how to handle the remainder when
    /// rounding. The default is [`RoundMode::HalfExpand`], which corresponds
    /// to how you were taught to round in school. Alternative modes, like
    /// [`RoundMode::Trunc`], exist too. For example, a truncating rounding of
    /// `1 hour 59 minutes` to the nearest hour would produce `1 hour`.
    /// * [`SpanRound::increment`] sets the rounding granularity to use for
    /// the configured smallest unit. For example, if the smallest unit is
    /// minutes and the increment is 5, then the span returned will always have
    /// its minute units set to a multiple of `5`.
    /// * [`SpanRound::relative`] sets the datetime from which to interpret the
    /// span. This is required when rounding spans with calendar units (years,
    /// months or weeks). When a relative datetime is time zone aware, then
    /// rounding accounts for the fact that not all days are 24 hours long.
    /// When a relative datetime is omitted or is civil (not time zone aware),
    /// then days are always 24 hours long.
    ///
    /// # Constructing a [`SpanRound`]
    ///
    /// This routine accepts anything that implements `Into<SpanRound>`. There
    /// are a few key trait implementations that make this convenient:
    ///
    /// * `From<Unit> for SpanRound` will construct a rounding configuration
    /// where the smallest unit is set to the one given.
    /// * `From<(Unit, i64)> for SpanRound` will construct a rounding
    /// configuration where the smallest unit and the rounding increment are
    /// set to the ones given.
    ///
    /// To set other options (like the largest unit, the rounding mode and the
    /// relative datetime), one must explicitly create a `SpanRound` and pass
    /// it to this routine.
    ///
    /// # Errors
    ///
    /// In general, there are two main ways for rounding to fail: an improper
    /// configuration like trying to round a span with calendar units but
    /// without a relative datetime, or when overflow occurs. Overflow can
    /// occur when the span, added to the relative datetime if given, would
    /// exceed the minimum or maximum datetime values. Overflow can also occur
    /// if the span is too big to fit into the requested unit configuration.
    /// For example, a span like `19_998.years()` cannot be represented with a
    /// 64-bit integer number of nanoseconds.
    ///
    /// Callers may use [`SpanArithmetic::days_are_24_hours`] as a special
    /// marker instead of providing a relative civil date to indicate that
    /// all days should be 24 hours long. This also results in treating all
    /// weeks as seven 24 hour days (168 hours).
    ///
    /// # Example: balancing
    ///
    /// This example demonstrates balancing, not rounding. And in particular,
    /// this example shows how to balance a span as much as possible (i.e.,
    /// with units of hours or smaller) without needing to specify a relative
    /// datetime:
    ///
    /// ```
    /// use jiff::{SpanRound, ToSpan, Unit};
    ///
    /// let span = 123_456_789_123_456_789i64.nanoseconds();
    /// assert_eq!(
    ///     span.round(SpanRound::new().largest(Unit::Hour))?.fieldwise(),
    ///     34_293.hours().minutes(33).seconds(9)
    ///         .milliseconds(123).microseconds(456).nanoseconds(789),
    /// );
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// Or you can opt into invariant 24-hour days (and 7-day weeks) without a
    /// relative date with [`SpanRound::days_are_24_hours`]:
    ///
    /// ```
    /// use jiff::{SpanRound, ToSpan, Unit};
    ///
    /// let span = 123_456_789_123_456_789i64.nanoseconds();
    /// assert_eq!(
    ///     span.round(
    ///         SpanRound::new().largest(Unit::Day).days_are_24_hours(),
    ///     )?.fieldwise(),
    ///     1_428.days()
    ///         .hours(21).minutes(33).seconds(9)
    ///         .milliseconds(123).microseconds(456).nanoseconds(789),
    /// );
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Example: balancing and rounding
    ///
    /// This example is like the one before it, but where we round to the
    /// nearest second:
    ///
    /// ```
    /// use jiff::{SpanRound, ToSpan, Unit};
    ///
    /// let span = 123_456_789_123_456_789i64.nanoseconds();
    /// assert_eq!(
    ///     span.round(SpanRound::new().largest(Unit::Hour).smallest(Unit::Second))?,
    ///     34_293.hours().minutes(33).seconds(9).fieldwise(),
    /// );
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// Or, just rounding to the nearest hour can make use of the
    /// `From<Unit> for SpanRound` trait implementation:
    ///
    /// ```
    /// use jiff::{ToSpan, Unit};
    ///
    /// let span = 123_456_789_123_456_789i64.nanoseconds();
    /// assert_eq!(span.round(Unit::Hour)?, 34_294.hours().fieldwise());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Example: balancing with a relative datetime
    ///
    /// Even with calendar units, so long as a relative datetime is provided,
    /// it's easy to turn days into bigger units:
    ///
    /// ```
    /// use jiff::{civil::date, SpanRound, ToSpan, Unit};
    ///
    /// let span = 1_000.days();
    /// let relative = date(2000, 1, 1);
    /// let options = SpanRound::new().largest(Unit::Year).relative(relative);
    /// assert_eq!(span.round(options)?, 2.years().months(8).days(26).fieldwise());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Example: round to the nearest half-hour
    ///
    /// ```
    /// use jiff::{Span, ToSpan, Unit};
    ///
    /// let span: Span = "PT23h50m3.123s".parse()?;
    /// assert_eq!(span.round((Unit::Minute, 30))?, 24.hours().fieldwise());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Example: yearly quarters in a span
    ///
    /// This example shows how to find how many full 3 month quarters are in a
    /// particular span of time.
    ///
    /// ```
    /// use jiff::{civil::date, RoundMode, SpanRound, ToSpan, Unit};
    ///
    /// let span1 = 10.months().days(15);
    /// let round = SpanRound::new()
    ///     .smallest(Unit::Month)
    ///     .increment(3)
    ///     .mode(RoundMode::Trunc)
    ///     // A relative datetime must be provided when
    ///     // rounding involves calendar units.
    ///     .relative(date(2024, 1, 1));
    /// let span2 = span1.round(round)?;
    /// assert_eq!(span2.get_months() / 3, 3);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn round<'a, R: Into<SpanRound<'a>>>(
        self,
        options: R,
    ) -> Result<Span, Error> {
        let options: SpanRound<'a> = options.into();
        options.round(self)
    }

    /// Converts a `Span` to a [`SignedDuration`] relative to the date given.
    ///
    /// In most cases, it is unlikely that you'll need to use this routine to
    /// convert a `Span` to a `SignedDuration` and instead will be ably to
    /// use `SignedDuration::try_from(span)`. Namely, by default:
    ///
    /// * [`Zoned::until`] guarantees that the biggest non-zero unit is hours.
    /// * [`Timestamp::until`] guarantees that the biggest non-zero unit is
    /// seconds.
    /// * [`DateTime::until`] guarantees that the biggest non-zero unit is
    /// days.
    /// * [`Date::until`] guarantees that the biggest non-zero unit is days.
    /// * [`Time::until`] guarantees that the biggest non-zero unit is hours.
    ///
    /// In the above, only [`DateTime::until`] and [`Date::until`] return
    /// calendar units by default, and thus would require this routine. (In
    /// which case, one may pass [`SpanRelativeTo::days_are_24_hours`] or an
    /// actual relative date to resolve the length of a day.)
    ///
    /// Of course, one may change the defaults. For example, if one
    /// uses `Zoned::until` with the largest unit set to `Unit::Year`
    /// and the resulting `Span` includes non-zero calendar units, then
    /// `SignedDuration::try_from` will fail because there is no relative date.
    ///
    /// # Errors
    ///
    /// This returns an error if adding this span to the date given results in
    /// overflow. This can also return an error if one uses
    /// [`SpanRelativeTo::days_are_24_hours`] with a `Span` that has non-zero
    /// units greater than weeks.
    ///
    /// # Example: converting a span with calendar units to a `SignedDuration`
    ///
    /// This compares the number of seconds in a non-leap year with a leap
    /// year:
    ///
    /// ```
    /// use jiff::{civil::date, SignedDuration, ToSpan};
    ///
    /// let span = 1.year();
    ///
    /// let duration = span.to_duration(date(2024, 1, 1))?;
    /// assert_eq!(duration, SignedDuration::from_secs(31_622_400));
    /// let duration = span.to_duration(date(2023, 1, 1))?;
    /// assert_eq!(duration, SignedDuration::from_secs(31_536_000));
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Example: converting a span without a relative datetime
    ///
    /// If for some reason it doesn't make sense to include a
    /// relative datetime, you can use this routine to convert a
    /// `Span` with units up to weeks to a `SignedDuration` via the
    /// [`SpanRelativeTo::days_are_24_hours`] marker:
    ///
    /// ```
    /// use jiff::{civil::date, SignedDuration, SpanRelativeTo, ToSpan};
    ///
    /// let span = 1.week().days(1);
    ///
    /// let duration = span.to_duration(SpanRelativeTo::days_are_24_hours())?;
    /// assert_eq!(duration, SignedDuration::from_hours(192));
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn to_duration<'a, R: Into<SpanRelativeTo<'a>>>(
        &self,
        relative: R,
    ) -> Result<SignedDuration, Error> {
        let max_unit = self.largest_unit();
        let relative: SpanRelativeTo<'a> = relative.into();
        let Some(result) = relative.to_relative(max_unit).transpose() else {
            return Ok(self.to_invariant_duration());
        };
        let relspan = result
            .and_then(|r| r.into_relative_span(Unit::Second, *self))
            .with_context(|| match relative.kind {
                SpanRelativeToKind::Civil(_) => E::ToDurationCivil,
                SpanRelativeToKind::Zoned(_) => E::ToDurationZoned,
                SpanRelativeToKind::DaysAre24Hours => {
                    E::ToDurationDaysAre24Hours
                }
            })?;
        debug_assert!(relspan.span.largest_unit() <= Unit::Second);
        Ok(relspan.span.to_invariant_duration())
    }

    /// Converts the non-variable units of this `Span` to a `SignedDuration`.
    ///
    /// This includes days and weeks, even though they can be of varying
    /// length during time zone transitions. If this applies, then callers
    /// should set the days and weeks to `0` before calling this routine.
    ///
    /// All units above weeks are always ignored.
    #[inline]
    pub(crate) fn to_invariant_duration(&self) -> SignedDuration {
        // This guarantees, at compile time, that a maximal invariant Span
        // (that is, all units are weeks or lower and all units are set to
        // their maximum values) will still balance out to a number of seconds
        // that fits into a `i64`. This in turn implies that a `SignedDuration`
        // can represent all possible invariant positive spans.
        const _FITS_IN_U64: () = {
            debug_assert!(
                i64::MAX as i128
                    > ((b::SpanWeeks::MAX as i128 * b::SECS_PER_WEEK as i128)
                        + (b::SpanDays::MAX as i128
                            * b::SECS_PER_CIVIL_DAY as i128)
                        + (b::SpanHours::MAX as i128
                            * b::SECS_PER_HOUR as i128)
                        + (b::SpanMinutes::MAX as i128
                            * b::SECS_PER_MIN as i128)
                        + b::SpanSeconds::MAX as i128
                        + (b::SpanMilliseconds::MAX as i128
                            / b::MILLIS_PER_SEC as i128)
                        + (b::SpanMicroseconds::MAX as i128
                            / b::MICROS_PER_SEC as i128)
                        + (b::SpanNanoseconds::MAX as i128
                            / b::NANOS_PER_SEC as i128)),
            );
            ()
        };

        // OK because we have a compile time assert above that ensures our
        // nanoseconds are in the valid range of a `SignedDuration`.
        SignedDuration::from_civil_weeks32(self.get_weeks())
            + SignedDuration::from_civil_days32(self.get_days())
            + SignedDuration::from_hours32(self.get_hours())
            + SignedDuration::from_mins(self.get_minutes())
            + SignedDuration::from_secs(self.get_seconds())
            + SignedDuration::from_millis(self.get_milliseconds())
            + SignedDuration::from_micros(self.get_microseconds())
            + SignedDuration::from_nanos(self.get_nanoseconds())
    }

    /// Like `Span::to_invariant_duration`, except only considers units of
    /// hours and lower. All bigger units are ignored.
    #[inline]
    pub(crate) fn to_invariant_duration_time_only(&self) -> SignedDuration {
        // This guarantees, at compile time, that a maximal invariant Span
        // (that is, all units are weeks or lower and all units are set to
        // their maximum values) will still balance out to a number of seconds
        // that fits into a `i64`. This in turn implies that a `SignedDuration`
        // can represent all possible invariant positive spans.
        const _FITS_IN_U64: () = {
            debug_assert!(
                i64::MAX as i128
                    > ((b::SpanHours::MAX as i128 * b::SECS_PER_HOUR as i128)
                        + (b::SpanMinutes::MAX as i128
                            * b::SECS_PER_MIN as i128)
                        + b::SpanSeconds::MAX as i128
                        + (b::SpanMilliseconds::MAX as i128
                            / b::MILLIS_PER_SEC as i128)
                        + (b::SpanMicroseconds::MAX as i128
                            / b::MICROS_PER_SEC as i128)
                        + (b::SpanNanoseconds::MAX as i128
                            / b::NANOS_PER_SEC as i128)),
            );
            ()
        };

        // OK because we have a compile time assert above that ensures our
        // nanoseconds are in the valid range of a `SignedDuration`.
        SignedDuration::from_hours32(self.get_hours())
            + SignedDuration::from_mins(self.get_minutes())
            + SignedDuration::from_secs(self.get_seconds())
            + SignedDuration::from_millis(self.get_milliseconds())
            + SignedDuration::from_micros(self.get_microseconds())
            + SignedDuration::from_nanos(self.get_nanoseconds())
    }
}

/// Crate internal APIs that operate on ranged integer types.
impl Span {
    #[inline]
    fn try_unit(self, unit: Unit, value: i64) -> Result<Span, Error> {
        match unit {
            Unit::Year => self.try_years(value),
            Unit::Month => self.try_months(value),
            Unit::Week => self.try_weeks(value),
            Unit::Day => self.try_days(value),
            Unit::Hour => self.try_hours(value),
            Unit::Minute => self.try_minutes(value),
            Unit::Second => self.try_seconds(value),
            Unit::Millisecond => self.try_milliseconds(value),
            Unit::Microsecond => self.try_microseconds(value),
            Unit::Nanosecond => self.try_nanoseconds(value),
        }
    }

    #[inline]
    pub(crate) fn get_years_unsigned(&self) -> u16 {
        self.years as u16
    }

    #[inline]
    pub(crate) fn get_months_unsigned(&self) -> u32 {
        self.months as u32
    }

    #[inline]
    pub(crate) fn get_weeks_unsigned(&self) -> u32 {
        self.weeks as u32
    }

    #[inline]
    pub(crate) fn get_days_unsigned(&self) -> u32 {
        self.days as u32
    }

    #[inline]
    pub(crate) fn get_hours_unsigned(&self) -> u32 {
        self.hours as u32
    }

    #[inline]
    pub(crate) fn get_minutes_unsigned(&self) -> u64 {
        self.minutes as u64
    }

    #[inline]
    pub(crate) fn get_seconds_unsigned(&self) -> u64 {
        self.seconds as u64
    }

    #[inline]
    pub(crate) fn get_milliseconds_unsigned(&self) -> u64 {
        self.milliseconds as u64
    }

    #[inline]
    pub(crate) fn get_microseconds_unsigned(&self) -> u64 {
        self.microseconds as u64
    }

    #[inline]
    pub(crate) fn get_nanoseconds_unsigned(&self) -> u64 {
        self.nanoseconds as u64
    }

    #[inline]
    fn get_sign(&self) -> b::Sign {
        self.sign
    }

    #[inline]
    fn get_unit(&self, unit: Unit) -> i64 {
        match unit {
            Unit::Year => self.get_years().into(),
            Unit::Month => self.get_months().into(),
            Unit::Week => self.get_weeks().into(),
            Unit::Day => self.get_days().into(),
            Unit::Hour => self.get_hours().into(),
            Unit::Minute => self.get_minutes(),
            Unit::Second => self.get_seconds(),
            Unit::Millisecond => self.get_milliseconds(),
            Unit::Microsecond => self.get_microseconds(),
            Unit::Nanosecond => self.get_nanoseconds(),
        }
    }
}

/// Crate internal APIs that permit setting units without checks.
///
/// Callers should be very careful when using these. These notably also do
/// not handle updating the sign on the `Span` and require the precisely
/// correct integer primitive.
impl Span {
    #[inline]
    pub(crate) fn years_unchecked(mut self, years: i16) -> Span {
        self.years = years;
        self.units = self.units.set(Unit::Year, years == 0);
        self
    }

    #[inline]
    pub(crate) fn months_unchecked(mut self, months: i32) -> Span {
        self.months = months;
        self.units = self.units.set(Unit::Month, months == 0);
        self
    }

    #[inline]
    pub(crate) fn weeks_unchecked(mut self, weeks: i32) -> Span {
        self.weeks = weeks;
        self.units = self.units.set(Unit::Week, weeks == 0);
        self
    }

    #[inline]
    pub(crate) fn days_unchecked(mut self, days: i32) -> Span {
        self.days = days;
        self.units = self.units.set(Unit::Day, days == 0);
        self
    }

    #[inline]
    pub(crate) fn hours_unchecked(mut self, hours: i32) -> Span {
        self.hours = hours;
        self.units = self.units.set(Unit::Hour, hours == 0);
        self
    }

    #[inline]
    pub(crate) fn minutes_unchecked(mut self, minutes: i64) -> Span {
        self.minutes = minutes;
        self.units = self.units.set(Unit::Minute, minutes == 0);
        self
    }

    #[inline]
    pub(crate) fn seconds_unchecked(mut self, seconds: i64) -> Span {
        self.seconds = seconds;
        self.units = self.units.set(Unit::Second, seconds == 0);
        self
    }

    #[inline]
    pub(crate) fn milliseconds_unchecked(mut self, milliseconds: i64) -> Span {
        self.milliseconds = milliseconds;
        self.units = self.units.set(Unit::Millisecond, milliseconds == 0);
        self
    }

    #[inline]
    pub(crate) fn microseconds_unchecked(mut self, microseconds: i64) -> Span {
        self.microseconds = microseconds;
        self.units = self.units.set(Unit::Microsecond, microseconds == 0);
        self
    }

    #[inline]
    pub(crate) fn nanoseconds_unchecked(mut self, nanoseconds: i64) -> Span {
        self.nanoseconds = nanoseconds;
        self.units = self.units.set(Unit::Nanosecond, nanoseconds == 0);
        self
    }

    #[inline]
    pub(crate) fn sign_unchecked(self, sign: b::Sign) -> Span {
        Span { sign, ..self }
    }
}

/// Crate internal helper routines.
impl Span {
    /// Converts the given duration to a `Span` whose units do not
    /// exceed `largest`.
    ///
    /// Note that `largest` is capped at `Unit::Week`. Note though that if
    /// any unit greater than `Unit::Week` is given, then it is treated as
    /// `Unit::Day`. The only way to get weeks in the `Span` returned is to
    /// specifically request `Unit::Week`.
    ///
    /// And also note that days in this context are civil days. That is, they
    /// are always 24 hours long. Callers needing to deal with variable length
    /// days should do so outside of this routine and should not provide a
    /// `largest` unit bigger than `Unit::Hour`.
    pub(crate) fn from_invariant_duration(
        largest: Unit,
        mut dur: SignedDuration,
    ) -> Result<Span, Error> {
        let mut span = Span::new();

        if matches!(largest, Unit::Week) {
            let (weeks, rem) = dur.as_civil_weeks_with_remainder();
            span = span.try_weeks(weeks)?;
            dur = rem;
        }
        if largest >= Unit::Day {
            let (days, rem) = dur.as_civil_days_with_remainder();
            span = span.try_days(days)?;
            dur = rem;
        }
        if largest >= Unit::Hour {
            let (hours, rem) = dur.as_hours_with_remainder();
            span = span.try_hours(hours)?;
            dur = rem;
        }
        if largest >= Unit::Minute {
            let (mins, rem) = dur.as_mins_with_remainder();
            span = span.try_minutes(mins)?;
            dur = rem;
        }
        if largest >= Unit::Second {
            let (secs, rem) = dur.as_secs_with_remainder();
            span = span.try_seconds(secs)?;
            dur = rem;
        }
        if largest >= Unit::Millisecond {
            let (millis, rem) = dur.as_millis_with_remainder();
            let millis = i64::try_from(millis)
                .map_err(|_| b::SpanMilliseconds::error())?;
            span = span.try_milliseconds(millis)?;
            dur = rem;
        }
        if largest >= Unit::Microsecond {
            let (micros, rem) = dur.as_micros_with_remainder();
            let micros = i64::try_from(micros)
                .map_err(|_| b::SpanMicroseconds::error())?;
            span = span.try_microseconds(micros)?;
            dur = rem;
        }
        if largest >= Unit::Nanosecond {
            let nanos = i64::try_from(dur.as_nanos())
                .map_err(|_| b::SpanNanoseconds::error())?;
            span = span.try_nanoseconds(nanos)?;
        }

        Ok(span)
    }

    /// Converts the hour, minute and second units in this `Span` to seconds.
    ///
    /// This ignores all other units.
    #[inline]
    pub(crate) fn to_hms_seconds(&self) -> i64 {
        // This can never overflow because the maximal values for hours,
        // minutes and seconds (even when combined) can fit into an `i64`.
        let mut secs = self.seconds;
        secs += self.minutes * b::SECS_PER_MIN;
        secs += i64::from(self.hours) * b::SECS_PER_HOUR;
        self.sign * secs
    }

    /// Returns true if and only if this span has at least one non-zero
    /// fractional second unit.
    #[inline]
    pub(crate) fn has_fractional_seconds(&self) -> bool {
        static SUBSECOND: UnitSet = UnitSet::from_slice(&[
            Unit::Millisecond,
            Unit::Microsecond,
            Unit::Nanosecond,
        ]);
        !self.units().intersection(SUBSECOND).is_empty()
    }

    /// Returns an equivalent span, but with all non-calendar (units below
    /// days) set to zero.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn only_calendar(mut self) -> Span {
        self.hours = 0;
        self.minutes = 0;
        self.seconds = 0;
        self.milliseconds = 0;
        self.microseconds = 0;
        self.nanoseconds = 0;
        if !self.sign.is_zero()
            && self.years == 0
            && self.months == 0
            && self.weeks == 0
            && self.days == 0
        {
            self.sign = b::Sign::Zero;
        }
        self.units = self.units.only_calendar();
        self
    }

    /// Returns an equivalent span, but with all calendar (units above
    /// hours) set to zero.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn only_time(mut self) -> Span {
        self.years = 0;
        self.months = 0;
        self.weeks = 0;
        self.days = 0;
        if !self.sign.is_zero()
            && self.hours == 0
            && self.minutes == 0
            && self.seconds == 0
            && self.milliseconds == 0
            && self.microseconds == 0
            && self.nanoseconds == 0
        {
            self.sign = b::Sign::Zero;
        }
        self.units = self.units.only_time();
        self
    }

    /// Returns an equivalent span, but with all units greater than or equal to
    /// the one given set to zero.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn only_lower(self, unit: Unit) -> Span {
        let mut span = self;
        // Unit::Nanosecond is the minimum, so nothing can be smaller than it.
        if unit <= Unit::Microsecond {
            span = span.microseconds(0);
        }
        if unit <= Unit::Millisecond {
            span = span.milliseconds(0);
        }
        if unit <= Unit::Second {
            span = span.seconds(0);
        }
        if unit <= Unit::Minute {
            span = span.minutes(0);
        }
        if unit <= Unit::Hour {
            span = span.hours(0);
        }
        if unit <= Unit::Day {
            span = span.days(0);
        }
        if unit <= Unit::Week {
            span = span.weeks(0);
        }
        if unit <= Unit::Month {
            span = span.months(0);
        }
        if unit <= Unit::Year {
            span = span.years(0);
        }
        span
    }

    /// Returns an equivalent span, but with all units less than the one given
    /// set to zero.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn without_lower(self, unit: Unit) -> Span {
        let mut span = self;
        if unit > Unit::Nanosecond {
            span = span.nanoseconds(0);
        }
        if unit > Unit::Microsecond {
            span = span.microseconds(0);
        }
        if unit > Unit::Millisecond {
            span = span.milliseconds(0);
        }
        if unit > Unit::Second {
            span = span.seconds(0);
        }
        if unit > Unit::Minute {
            span = span.minutes(0);
        }
        if unit > Unit::Hour {
            span = span.hours(0);
        }
        if unit > Unit::Day {
            span = span.days(0);
        }
        if unit > Unit::Week {
            span = span.weeks(0);
        }
        if unit > Unit::Month {
            span = span.months(0);
        }
        // Unit::Year is the max, so nothing can be bigger than it.
        span
    }

    /// Returns an error corresponding to the smallest non-time non-zero unit.
    ///
    /// If all non-time units are zero, then this returns `None`.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn smallest_non_time_non_zero_unit_error(
        &self,
    ) -> Option<Error> {
        let non_time_unit = self.largest_calendar_unit()?;
        Some(Error::from(UnitConfigError::CalendarUnitsNotAllowed {
            unit: non_time_unit,
        }))
    }

    /// Returns the largest non-zero calendar unit, or `None` if there are no
    /// non-zero calendar units.
    #[inline]
    fn largest_calendar_unit(&self) -> Option<Unit> {
        self.units().only_calendar().largest_unit()
    }

    /// Returns the largest non-zero unit in this span.
    ///
    /// If all components of this span are zero, then `Unit::Nanosecond` is
    /// returned.
    #[inline]
    pub(crate) fn largest_unit(&self) -> Unit {
        self.units().largest_unit().unwrap_or(Unit::Nanosecond)
    }

    /// Returns the set of units on this `Span`.
    #[inline]
    pub(crate) fn units(&self) -> UnitSet {
        self.units
    }

    /// Returns a string containing the value of all non-zero fields.
    ///
    /// This is useful for debugging. Normally, this would be the "alternate"
    /// debug impl (perhaps), but that's what insta uses and I preferred having
    /// the friendly format used there since it is much more terse.
    #[cfg(feature = "alloc")]
    #[allow(dead_code)]
    pub(crate) fn debug(&self) -> alloc::string::String {
        use core::fmt::Write;

        let mut buf = alloc::string::String::new();
        write!(buf, "Span {{ sign: {:?}, units: {:?}", self.sign, self.units)
            .unwrap();
        if self.years != 0 {
            write!(buf, ", years: {:?}", self.years).unwrap();
        }
        if self.months != 0 {
            write!(buf, ", months: {:?}", self.months).unwrap();
        }
        if self.weeks != 0 {
            write!(buf, ", weeks: {:?}", self.weeks).unwrap();
        }
        if self.days != 0 {
            write!(buf, ", days: {:?}", self.days).unwrap();
        }
        if self.hours != 0 {
            write!(buf, ", hours: {:?}", self.hours).unwrap();
        }
        if self.minutes != 0 {
            write!(buf, ", minutes: {:?}", self.minutes).unwrap();
        }
        if self.seconds != 0 {
            write!(buf, ", seconds: {:?}", self.seconds).unwrap();
        }
        if self.milliseconds != 0 {
            write!(buf, ", milliseconds: {:?}", self.milliseconds).unwrap();
        }
        if self.microseconds != 0 {
            write!(buf, ", microseconds: {:?}", self.microseconds).unwrap();
        }
        if self.nanoseconds != 0 {
            write!(buf, ", nanoseconds: {:?}", self.nanoseconds).unwrap();
        }
        buf.push_str(" }}");
        buf
    }

    /// Given some new units to set on this span and the span updates with the
    /// new units, this determines the what the sign of `new` should be.
    #[inline]
    fn resign(&self, units: impl Into<i64>, new: &Span) -> b::Sign {
        fn imp(span: &Span, units: i64, new: &Span) -> b::Sign {
            // Negative units anywhere always makes the entire span negative.
            if units.is_negative() {
                return b::Sign::Negative;
            }
            let mut new_is_zero = new.sign.is_zero() && units == 0;
            // When `units == 0` and it was previously non-zero, then
            // `new.sign` won't be `0` and thus `new_is_zero` will be false
            // when it should be true. So in this case, we need to re-check all
            // the units to set the sign correctly.
            if units == 0 {
                new_is_zero = new.years == 0
                    && new.months == 0
                    && new.weeks == 0
                    && new.days == 0
                    && new.hours == 0
                    && new.minutes == 0
                    && new.seconds == 0
                    && new.milliseconds == 0
                    && new.microseconds == 0
                    && new.nanoseconds == 0;
            }
            match (span.is_zero(), new_is_zero) {
                (_, true) => b::Sign::Zero,
                (true, false) => b::Sign::from(units),
                // If the old and new span are both non-zero, and we know our
                // new units are not negative, then the sign remains unchanged.
                (false, false) => new.sign,
            }
        }
        imp(self, units.into(), new)
    }
}

impl core::fmt::Debug for Span {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use crate::fmt::StdFmtWrite;

        friendly::DEFAULT_SPAN_PRINTER
            .print_span(self, StdFmtWrite(f))
            .map_err(|_| core::fmt::Error)
    }
}

impl core::fmt::Display for Span {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use crate::fmt::StdFmtWrite;

        if f.alternate() {
            friendly::DEFAULT_SPAN_PRINTER
                .print_span(self, StdFmtWrite(f))
                .map_err(|_| core::fmt::Error)
        } else {
            temporal::DEFAULT_SPAN_PRINTER
                .print_span(self, StdFmtWrite(f))
                .map_err(|_| core::fmt::Error)
        }
    }
}

impl core::str::FromStr for Span {
    type Err = Error;

    #[inline]
    fn from_str(string: &str) -> Result<Span, Error> {
        parse_iso_or_friendly(string.as_bytes())
    }
}

impl core::ops::Neg for Span {
    type Output = Span;

    #[inline]
    fn neg(self) -> Span {
        self.negate()
    }
}

/// This multiplies each unit in a span by an integer.
///
/// This panics on overflow. For checked arithmetic, use [`Span::checked_mul`].
impl core::ops::Mul<i64> for Span {
    type Output = Span;

    #[inline]
    fn mul(self, rhs: i64) -> Span {
        self.checked_mul(rhs)
            .expect("multiplying `Span` by a scalar overflowed")
    }
}

/// This multiplies each unit in a span by an integer.
///
/// This panics on overflow. For checked arithmetic, use [`Span::checked_mul`].
impl core::ops::Mul<Span> for i64 {
    type Output = Span;

    #[inline]
    fn mul(self, rhs: Span) -> Span {
        rhs.checked_mul(self)
            .expect("multiplying `Span` by a scalar overflowed")
    }
}

/// Converts a `Span` to a [`std::time::Duration`].
///
/// # Errors
///
/// This can fail for only two reasons:
///
/// * The span is negative. This is an error because a `std::time::Duration` is
///   unsigned.)
/// * The span has any non-zero units greater than hours. This is an error
///   because it's impossible to determine the length of, e.g., a month without
///   a reference date.
///
/// This can never result in overflow because a `Duration` can represent a
/// bigger span of time than `Span` when limited to units of hours or lower.
///
/// If you need to convert a `Span` to a `Duration` that has non-zero
/// units bigger than hours, then please use [`Span::to_duration`] with a
/// corresponding relative date.
///
/// # Example: maximal span
///
/// This example shows the maximum possible span using units of hours or
/// smaller, and the corresponding `Duration` value:
///
/// ```
/// use std::time::Duration;
///
/// use jiff::Span;
///
/// let sp = Span::new()
///     .hours(175_307_616)
///     .minutes(10_518_456_960i64)
///     .seconds(631_107_417_600i64)
///     .milliseconds(631_107_417_600_000i64)
///     .microseconds(631_107_417_600_000_000i64)
///     .nanoseconds(9_223_372_036_854_775_807i64);
/// let duration = Duration::try_from(sp)?;
/// assert_eq!(duration, Duration::new(3_164_760_460_036, 854_775_807));
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Example: converting a negative span
///
/// Since a `Span` is signed and a `Duration` is unsigned, converting
/// a negative `Span` to `Duration` will always fail. One can use
/// [`Span::signum`] to get the sign of the span and [`Span::abs`] to make the
/// span positive before converting it to a `Duration`:
///
/// ```
/// use std::time::Duration;
///
/// use jiff::{Span, ToSpan};
///
/// let span = -86_400.seconds().nanoseconds(1);
/// let (sign, duration) = (span.signum(), Duration::try_from(span.abs())?);
/// assert_eq!((sign, duration), (-1, Duration::new(86_400, 1)));
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
impl TryFrom<Span> for UnsignedDuration {
    type Error = Error;

    #[inline]
    fn try_from(sp: Span) -> Result<UnsignedDuration, Error> {
        // This isn't needed, but improves error messages.
        if sp.is_negative() {
            return Err(Error::from(E::ConvertNegative));
        }
        SignedDuration::try_from(sp).and_then(UnsignedDuration::try_from)
    }
}

/// Converts a [`std::time::Duration`] to a `Span`.
///
/// The span returned from this conversion will only ever have non-zero units
/// of seconds or smaller.
///
/// # Errors
///
/// This only fails when the given `Duration` overflows the maximum number of
/// seconds representable by a `Span`.
///
/// # Example
///
/// This shows a basic conversion:
///
/// ```
/// use std::time::Duration;
///
/// use jiff::{Span, ToSpan};
///
/// let duration = Duration::new(86_400, 123_456_789);
/// let span = Span::try_from(duration)?;
/// // A duration-to-span conversion always results in a span with
/// // non-zero units no bigger than seconds.
/// assert_eq!(
///     span.fieldwise(),
///     86_400.seconds().milliseconds(123).microseconds(456).nanoseconds(789),
/// );
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Example: rounding
///
/// This example shows how to convert a `Duration` to a `Span`, and then round
/// it up to bigger units given a relative date:
///
/// ```
/// use std::time::Duration;
///
/// use jiff::{civil::date, Span, SpanRound, ToSpan, Unit};
///
/// let duration = Duration::new(450 * 86_401, 0);
/// let span = Span::try_from(duration)?;
/// // We get back a simple span of just seconds:
/// assert_eq!(span.fieldwise(), Span::new().seconds(450 * 86_401));
/// // But we can balance it up to bigger units:
/// let options = SpanRound::new()
///     .largest(Unit::Year)
///     .relative(date(2024, 1, 1));
/// assert_eq!(
///     span.round(options)?,
///     1.year().months(2).days(25).minutes(7).seconds(30).fieldwise(),
/// );
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
impl TryFrom<UnsignedDuration> for Span {
    type Error = Error;

    #[inline]
    fn try_from(d: UnsignedDuration) -> Result<Span, Error> {
        let sdur = SignedDuration::try_from(d)
            .map_err(|_| b::SpanSeconds::error())?;
        Span::try_from(sdur)
    }
}

/// Converts a `Span` to a [`SignedDuration`].
///
/// # Errors
///
/// This can fail for only when the span has any non-zero units greater than
/// hours. This is an error because it's impossible to determine the length of,
/// e.g., a month without a reference date.
///
/// This can never result in overflow because a `SignedDuration` can represent
/// a bigger span of time than `Span` when limited to units of hours or lower.
///
/// If you need to convert a `Span` to a `SignedDuration` that has non-zero
/// units bigger than hours, then please use [`Span::to_duration`] with a
/// corresponding relative date.
///
/// # Example: maximal span
///
/// This example shows the maximum possible span using units of hours or
/// smaller, and the corresponding `SignedDuration` value:
///
/// ```
/// use jiff::{SignedDuration, Span};
///
/// let sp = Span::new()
///     .hours(175_307_616)
///     .minutes(10_518_456_960i64)
///     .seconds(631_107_417_600i64)
///     .milliseconds(631_107_417_600_000i64)
///     .microseconds(631_107_417_600_000_000i64)
///     .nanoseconds(9_223_372_036_854_775_807i64);
/// let duration = SignedDuration::try_from(sp)?;
/// assert_eq!(duration, SignedDuration::new(3_164_760_460_036, 854_775_807));
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
impl TryFrom<Span> for SignedDuration {
    type Error = Error;

    #[inline]
    fn try_from(sp: Span) -> Result<SignedDuration, Error> {
        requires_relative_date_err(sp.largest_unit())
            .context(E::ConvertSpanToSignedDuration)?;
        Ok(sp.to_invariant_duration())
    }
}

/// Converts a [`SignedDuration`] to a `Span`.
///
/// The span returned from this conversion will only ever have non-zero units
/// of seconds or smaller.
///
/// # Errors
///
/// This only fails when the given `SignedDuration` overflows the maximum
/// number of seconds representable by a `Span`.
///
/// # Example
///
/// This shows a basic conversion:
///
/// ```
/// use jiff::{SignedDuration, Span, ToSpan};
///
/// let duration = SignedDuration::new(86_400, 123_456_789);
/// let span = Span::try_from(duration)?;
/// // A duration-to-span conversion always results in a span with
/// // non-zero units no bigger than seconds.
/// assert_eq!(
///     span.fieldwise(),
///     86_400.seconds().milliseconds(123).microseconds(456).nanoseconds(789),
/// );
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Example: rounding
///
/// This example shows how to convert a `SignedDuration` to a `Span`, and then
/// round it up to bigger units given a relative date:
///
/// ```
/// use jiff::{civil::date, SignedDuration, Span, SpanRound, ToSpan, Unit};
///
/// let duration = SignedDuration::new(450 * 86_401, 0);
/// let span = Span::try_from(duration)?;
/// // We get back a simple span of just seconds:
/// assert_eq!(span.fieldwise(), Span::new().seconds(450 * 86_401));
/// // But we can balance it up to bigger units:
/// let options = SpanRound::new()
///     .largest(Unit::Year)
///     .relative(date(2024, 1, 1));
/// assert_eq!(
///     span.round(options)?,
///     1.year().months(2).days(25).minutes(7).seconds(30).fieldwise(),
/// );
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
impl TryFrom<SignedDuration> for Span {
    type Error = Error;

    #[inline]
    fn try_from(d: SignedDuration) -> Result<Span, Error> {
        let seconds = d.as_secs();
        let nanoseconds = i64::from(d.subsec_nanos());
        let milliseconds = nanoseconds / b::NANOS_PER_MILLI;
        let microseconds =
            (nanoseconds % b::NANOS_PER_MILLI) / b::NANOS_PER_MICRO;
        let nanoseconds = nanoseconds % b::NANOS_PER_MICRO;

        let span = Span::new().try_seconds(seconds)?;
        // These are all OK because `|SignedDuration::subsec_nanos|` is
        // guaranteed to return less than 1_000_000_000 nanoseconds. And
        // splitting that up into millis, micros and nano components is
        // guaranteed to fit into the limits of a `Span`.
        Ok(span
            .milliseconds(milliseconds)
            .microseconds(microseconds)
            .nanoseconds(nanoseconds))
    }
}

#[cfg(feature = "serde")]
impl serde_core::Serialize for Span {
    #[inline]
    fn serialize<S: serde_core::Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde_core::Deserialize<'de> for Span {
    #[inline]
    fn deserialize<D: serde_core::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Span, D::Error> {
        use serde_core::de;

        struct SpanVisitor;

        impl<'de> de::Visitor<'de> for SpanVisitor {
            type Value = Span;

            fn expecting(
                &self,
                f: &mut core::fmt::Formatter,
            ) -> core::fmt::Result {
                f.write_str("a span duration string")
            }

            #[inline]
            fn visit_bytes<E: de::Error>(
                self,
                value: &[u8],
            ) -> Result<Span, E> {
                parse_iso_or_friendly(value).map_err(de::Error::custom)
            }

            #[inline]
            fn visit_str<E: de::Error>(self, value: &str) -> Result<Span, E> {
                self.visit_bytes(value.as_bytes())
            }
        }

        deserializer.deserialize_str(SpanVisitor)
    }
}

#[cfg(test)]
impl quickcheck::Arbitrary for Span {
    fn arbitrary(g: &mut quickcheck::Gen) -> Span {
        // In order to sample from the full space of possible spans, we need
        // to provide a relative datetime. But if we do that, then it's
        // possible the span plus the datetime overflows. So we pick one
        // datetime and shrink the size of the span we can produce.
        const MIN: i64 = -631_107_417_600_000_000;
        const MAX: i64 = 631_107_417_600_000_000;
        const LEN: i64 = MAX - MIN + 1;

        let mut nanos = i64::arbitrary(g).wrapping_rem_euclid(LEN);
        nanos += MIN;
        let relative =
            SpanRelativeTo::from(DateTime::constant(0, 1, 1, 0, 0, 0, 0));
        let round =
            SpanRound::new().largest(Unit::arbitrary(g)).relative(relative);
        Span::new().nanoseconds(nanos).round(round).unwrap()
    }

    fn shrink(&self) -> alloc::boxed::Box<dyn Iterator<Item = Self>> {
        alloc::boxed::Box::new(
            (
                (
                    self.get_years(),
                    self.get_months(),
                    self.get_weeks(),
                    self.get_days(),
                ),
                (
                    self.get_hours(),
                    self.get_minutes(),
                    self.get_seconds(),
                    self.get_milliseconds(),
                ),
                (self.get_microseconds(), self.get_nanoseconds()),
            )
                .shrink()
                .filter_map(
                    |(
                        (years, months, weeks, days),
                        (hours, minutes, seconds, milliseconds),
                        (microseconds, nanoseconds),
                    )| {
                        let span = Span::new()
                            .try_years(years)
                            .ok()?
                            .try_months(months)
                            .ok()?
                            .try_weeks(weeks)
                            .ok()?
                            .try_days(days)
                            .ok()?
                            .try_hours(hours)
                            .ok()?
                            .try_minutes(minutes)
                            .ok()?
                            .try_seconds(seconds)
                            .ok()?
                            .try_milliseconds(milliseconds)
                            .ok()?
                            .try_microseconds(microseconds)
                            .ok()?
                            .try_nanoseconds(nanoseconds)
                            .ok()?;
                        Some(span)
                    },
                ),
        )
    }
}

/// A wrapper for [`Span`] that implements the `Hash`, `Eq` and `PartialEq`
/// traits.
///
/// A `SpanFieldwise` is meant to make it easy to compare two spans in a "dumb"
/// way based purely on its unit values, while still providing a speed bump
/// to avoid accidentally doing this comparison on `Span` directly. This is
/// distinct from something like [`Span::compare`] that performs a comparison
/// on the actual elapsed time of two spans.
///
/// It is generally discouraged to use `SpanFieldwise` since spans that
/// represent an equivalent elapsed amount of time may compare unequal.
/// However, in some cases, it is useful to be able to assert precise field
/// values. For example, Jiff itself makes heavy use of fieldwise comparisons
/// for tests.
///
/// # Construction
///
/// While callers may use `SpanFieldwise(span)` (where `span` has type [`Span`])
/// to construct a value of this type, callers may find [`Span::fieldwise`]
/// more convenient. Namely, `Span::fieldwise` may avoid the need to explicitly
/// import `SpanFieldwise`.
///
/// # Trait implementations
///
/// In addition to implementing the `Hash`, `Eq` and `PartialEq` traits, this
/// type also provides `PartialEq` impls for comparing a `Span` with a
/// `SpanFieldwise`. This simplifies comparisons somewhat while still requiring
/// that at least one of the values has an explicit fieldwise comparison type.
///
/// # Safety
///
/// This type is guaranteed to have the same layout in memory as [`Span`].
///
/// # Example: the difference between `SpanFieldwise` and [`Span::compare`]
///
/// In short, `SpanFieldwise` considers `2 hours` and `120 minutes` to be
/// distinct values, but `Span::compare` considers them to be equivalent:
///
/// ```
/// use std::cmp::Ordering;
/// use jiff::ToSpan;
///
/// assert_ne!(120.minutes().fieldwise(), 2.hours().fieldwise());
/// assert_eq!(120.minutes().compare(2.hours())?, Ordering::Equal);
///
/// // These comparisons are allowed between a `Span` and a `SpanFieldwise`.
/// // Namely, as long as one value is "fieldwise," then the comparison is OK.
/// assert_ne!(120.minutes().fieldwise(), 2.hours());
/// assert_ne!(120.minutes(), 2.hours().fieldwise());
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug, Default)]
#[repr(transparent)]
pub struct SpanFieldwise(pub Span);

// Exists so that things like `-1.day().fieldwise()` works as expected.
impl core::ops::Neg for SpanFieldwise {
    type Output = SpanFieldwise;

    #[inline]
    fn neg(self) -> SpanFieldwise {
        SpanFieldwise(self.0.negate())
    }
}

impl Eq for SpanFieldwise {}

impl PartialEq for SpanFieldwise {
    fn eq(&self, rhs: &SpanFieldwise) -> bool {
        self.0.sign == rhs.0.sign
            && self.0.years == rhs.0.years
            && self.0.months == rhs.0.months
            && self.0.weeks == rhs.0.weeks
            && self.0.days == rhs.0.days
            && self.0.hours == rhs.0.hours
            && self.0.minutes == rhs.0.minutes
            && self.0.seconds == rhs.0.seconds
            && self.0.milliseconds == rhs.0.milliseconds
            && self.0.microseconds == rhs.0.microseconds
            && self.0.nanoseconds == rhs.0.nanoseconds
    }
}

impl<'a> PartialEq<SpanFieldwise> for &'a SpanFieldwise {
    fn eq(&self, rhs: &SpanFieldwise) -> bool {
        *self == rhs
    }
}

impl PartialEq<Span> for SpanFieldwise {
    fn eq(&self, rhs: &Span) -> bool {
        self == rhs.fieldwise()
    }
}

impl PartialEq<SpanFieldwise> for Span {
    fn eq(&self, rhs: &SpanFieldwise) -> bool {
        self.fieldwise() == *rhs
    }
}

impl<'a> PartialEq<SpanFieldwise> for &'a Span {
    fn eq(&self, rhs: &SpanFieldwise) -> bool {
        self.fieldwise() == *rhs
    }
}

impl core::hash::Hash for SpanFieldwise {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.0.sign.hash(state);
        self.0.years.hash(state);
        self.0.months.hash(state);
        self.0.weeks.hash(state);
        self.0.days.hash(state);
        self.0.hours.hash(state);
        self.0.minutes.hash(state);
        self.0.seconds.hash(state);
        self.0.milliseconds.hash(state);
        self.0.microseconds.hash(state);
        self.0.nanoseconds.hash(state);
    }
}

impl From<Span> for SpanFieldwise {
    fn from(span: Span) -> SpanFieldwise {
        SpanFieldwise(span)
    }
}

impl From<SpanFieldwise> for Span {
    fn from(span: SpanFieldwise) -> Span {
        span.0
    }
}

/// A trait for enabling concise literals for creating [`Span`] values.
///
/// In short, this trait lets you write something like `5.seconds()` or
/// `1.day()` to create a [`Span`]. Once a `Span` has been created, you can
/// use its mutator methods to add more fields. For example,
/// `1.day().hours(10)` is equivalent to `Span::new().days(1).hours(10)`.
///
/// This trait is implemented for the following integer types: `i8`, `i16`,
/// `i32` and `i64`.
///
/// Note that this trait is provided as a convenience and should generally
/// only be used for literals in your source code. You should not use this
/// trait on numbers provided by end users. Namely, if the number provided
/// is not within Jiff's span limits, then these trait methods will panic.
/// Instead, use fallible mutator constructors like [`Span::try_days`]
/// or [`Span::try_seconds`].
///
/// # Example
///
/// ```
/// use jiff::ToSpan;
///
/// assert_eq!(5.days().to_string(), "P5D");
/// assert_eq!(5.days().hours(10).to_string(), "P5DT10H");
///
/// // Negation works and it doesn't matter where the sign goes. It can be
/// // applied to the span itself or to the integer.
/// assert_eq!((-5.days()).to_string(), "-P5D");
/// assert_eq!((-5).days().to_string(), "-P5D");
/// ```
///
/// # Example: alternative via span parsing
///
/// Another way of tersely building a `Span` value is by parsing a ISO 8601
/// duration string:
///
/// ```
/// use jiff::Span;
///
/// let span = "P5y2m15dT23h30m10s".parse::<Span>()?;
/// assert_eq!(
///     span.fieldwise(),
///     Span::new().years(5).months(2).days(15).hours(23).minutes(30).seconds(10),
/// );
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait ToSpan: Sized {
    /// Create a new span from this integer in units of years.
    ///
    /// # Panics
    ///
    /// When `Span::new().years(self)` would panic.
    fn years(self) -> Span;

    /// Create a new span from this integer in units of months.
    ///
    /// # Panics
    ///
    /// When `Span::new().months(self)` would panic.
    fn months(self) -> Span;

    /// Create a new span from this integer in units of weeks.
    ///
    /// # Panics
    ///
    /// When `Span::new().weeks(self)` would panic.
    fn weeks(self) -> Span;

    /// Create a new span from this integer in units of days.
    ///
    /// # Panics
    ///
    /// When `Span::new().days(self)` would panic.
    fn days(self) -> Span;

    /// Create a new span from this integer in units of hours.
    ///
    /// # Panics
    ///
    /// When `Span::new().hours(self)` would panic.
    fn hours(self) -> Span;

    /// Create a new span from this integer in units of minutes.
    ///
    /// # Panics
    ///
    /// When `Span::new().minutes(self)` would panic.
    fn minutes(self) -> Span;

    /// Create a new span from this integer in units of seconds.
    ///
    /// # Panics
    ///
    /// When `Span::new().seconds(self)` would panic.
    fn seconds(self) -> Span;

    /// Create a new span from this integer in units of milliseconds.
    ///
    /// # Panics
    ///
    /// When `Span::new().milliseconds(self)` would panic.
    fn milliseconds(self) -> Span;

    /// Create a new span from this integer in units of microseconds.
    ///
    /// # Panics
    ///
    /// When `Span::new().microseconds(self)` would panic.
    fn microseconds(self) -> Span;

    /// Create a new span from this integer in units of nanoseconds.
    ///
    /// # Panics
    ///
    /// When `Span::new().nanoseconds(self)` would panic.
    fn nanoseconds(self) -> Span;

    /// Equivalent to `years()`, but reads better for singular units.
    #[inline]
    fn year(self) -> Span {
        self.years()
    }

    /// Equivalent to `months()`, but reads better for singular units.
    #[inline]
    fn month(self) -> Span {
        self.months()
    }

    /// Equivalent to `weeks()`, but reads better for singular units.
    #[inline]
    fn week(self) -> Span {
        self.weeks()
    }

    /// Equivalent to `days()`, but reads better for singular units.
    #[inline]
    fn day(self) -> Span {
        self.days()
    }

    /// Equivalent to `hours()`, but reads better for singular units.
    #[inline]
    fn hour(self) -> Span {
        self.hours()
    }

    /// Equivalent to `minutes()`, but reads better for singular units.
    #[inline]
    fn minute(self) -> Span {
        self.minutes()
    }

    /// Equivalent to `seconds()`, but reads better for singular units.
    #[inline]
    fn second(self) -> Span {
        self.seconds()
    }

    /// Equivalent to `milliseconds()`, but reads better for singular units.
    #[inline]
    fn millisecond(self) -> Span {
        self.milliseconds()
    }

    /// Equivalent to `microseconds()`, but reads better for singular units.
    #[inline]
    fn microsecond(self) -> Span {
        self.microseconds()
    }

    /// Equivalent to `nanoseconds()`, but reads better for singular units.
    #[inline]
    fn nanosecond(self) -> Span {
        self.nanoseconds()
    }
}

macro_rules! impl_to_span {
    ($ty:ty) => {
        impl ToSpan for $ty {
            #[inline]
            fn years(self) -> Span {
                Span::new().years(self)
            }
            #[inline]
            fn months(self) -> Span {
                Span::new().months(self)
            }
            #[inline]
            fn weeks(self) -> Span {
                Span::new().weeks(self)
            }
            #[inline]
            fn days(self) -> Span {
                Span::new().days(self)
            }
            #[inline]
            fn hours(self) -> Span {
                Span::new().hours(self)
            }
            #[inline]
            fn minutes(self) -> Span {
                Span::new().minutes(self)
            }
            #[inline]
            fn seconds(self) -> Span {
                Span::new().seconds(self)
            }
            #[inline]
            fn milliseconds(self) -> Span {
                Span::new().milliseconds(self)
            }
            #[inline]
            fn microseconds(self) -> Span {
                Span::new().microseconds(self)
            }
            #[inline]
            fn nanoseconds(self) -> Span {
                Span::new().nanoseconds(self)
            }
        }
    };
}

impl_to_span!(i8);
impl_to_span!(i16);
impl_to_span!(i32);
impl_to_span!(i64);

/// A way to refer to a single calendar or clock unit.
///
/// This type is principally used in APIs involving a [`Span`], which is a
/// duration of time. For example, routines like [`Zoned::until`] permit
/// specifying the largest unit of the span returned:
///
/// ```
/// use jiff::{Unit, Zoned};
///
/// let zdt1: Zoned = "2024-07-06 17:40-04[America/New_York]".parse()?;
/// let zdt2: Zoned = "2024-11-05 08:00-05[America/New_York]".parse()?;
/// let span = zdt1.until((Unit::Year, &zdt2))?;
/// assert_eq!(format!("{span:#}"), "3mo 29d 14h 20m");
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// But a `Unit` is also used in APIs for rounding datetimes themselves:
///
/// ```
/// use jiff::{Unit, Zoned};
///
/// let zdt: Zoned = "2024-07-06 17:44:22.158-04[America/New_York]".parse()?;
/// let nearest_minute = zdt.round(Unit::Minute)?;
/// assert_eq!(
///     nearest_minute.to_string(),
///     "2024-07-06T17:44:00-04:00[America/New_York]",
/// );
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Example: ordering
///
/// This example demonstrates that `Unit` has an ordering defined such that
/// bigger units compare greater than smaller units.
///
/// ```
/// use jiff::Unit;
///
/// assert!(Unit::Year > Unit::Nanosecond);
/// assert!(Unit::Day > Unit::Hour);
/// assert!(Unit::Hour > Unit::Minute);
/// assert!(Unit::Hour > Unit::Minute);
/// assert_eq!(Unit::Hour, Unit::Hour);
/// ```
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub enum Unit {
    /// A Gregorian calendar year. It usually has 365 days for non-leap years,
    /// and 366 days for leap years.
    Year = 9,
    /// A Gregorian calendar month. It usually has one of 28, 29, 30 or 31
    /// days.
    Month = 8,
    /// A week is 7 days that either begins on Sunday or Monday.
    Week = 7,
    /// A day is usually 24 hours, but some days may have different lengths
    /// due to time zone transitions.
    Day = 6,
    /// An hour is always 60 minutes.
    Hour = 5,
    /// A minute is always 60 seconds. (Jiff behaves as if leap seconds do not
    /// exist.)
    Minute = 4,
    /// A second is always 1,000 milliseconds.
    Second = 3,
    /// A millisecond is always 1,000 microseconds.
    Millisecond = 2,
    /// A microsecond is always 1,000 nanoseconds.
    Microsecond = 1,
    /// A nanosecond is the smallest granularity of time supported by Jiff.
    Nanosecond = 0,
}

impl Unit {
    /// Returns the next biggest unit, if one exists.
    pub(crate) fn next(&self) -> Option<Unit> {
        match *self {
            Unit::Year => None,
            Unit::Month => Some(Unit::Year),
            Unit::Week => Some(Unit::Month),
            Unit::Day => Some(Unit::Week),
            Unit::Hour => Some(Unit::Day),
            Unit::Minute => Some(Unit::Hour),
            Unit::Second => Some(Unit::Minute),
            Unit::Millisecond => Some(Unit::Second),
            Unit::Microsecond => Some(Unit::Millisecond),
            Unit::Nanosecond => Some(Unit::Microsecond),
        }
    }

    /// Returns the number of nanoseconds in this unit as a 96-bit integer.
    ///
    /// This will treat weeks and days as invariant. Callers must ensure this
    /// is appropriate to do.
    ///
    /// # Panics
    ///
    /// When this unit is always variable. That is, years or months.
    pub(crate) fn duration(self) -> SignedDuration {
        match self {
            Unit::Nanosecond => SignedDuration::from_nanos(1),
            Unit::Microsecond => SignedDuration::from_micros(1),
            Unit::Millisecond => SignedDuration::from_millis(1),
            Unit::Second => SignedDuration::from_secs(1),
            Unit::Minute => SignedDuration::from_mins32(1),
            Unit::Hour => SignedDuration::from_hours32(1),
            Unit::Day => SignedDuration::from_civil_days32(1),
            Unit::Week => SignedDuration::from_civil_weeks32(1),
            unit => unreachable!("{unit:?} has no definitive time interval"),
        }
    }

    /// Returns true when this unit is definitively variable.
    ///
    /// In effect, this is any unit bigger than 'day', because any such unit
    /// can vary in time depending on its reference point. A 'day' can as well,
    /// but we sorta special case 'day' to mean '24 hours' for cases where
    /// the user is dealing with civil time.
    fn is_variable(self) -> bool {
        matches!(self, Unit::Year | Unit::Month | Unit::Week | Unit::Day)
    }

    /// A human readable singular description of this unit of time.
    pub(crate) fn singular(&self) -> &'static str {
        match *self {
            Unit::Year => "year",
            Unit::Month => "month",
            Unit::Week => "week",
            Unit::Day => "day",
            Unit::Hour => "hour",
            Unit::Minute => "minute",
            Unit::Second => "second",
            Unit::Millisecond => "millisecond",
            Unit::Microsecond => "microsecond",
            Unit::Nanosecond => "nanosecond",
        }
    }

    /// A human readable plural description of this unit of time.
    pub(crate) fn plural(&self) -> &'static str {
        match *self {
            Unit::Year => "years",
            Unit::Month => "months",
            Unit::Week => "weeks",
            Unit::Day => "days",
            Unit::Hour => "hours",
            Unit::Minute => "minutes",
            Unit::Second => "seconds",
            Unit::Millisecond => "milliseconds",
            Unit::Microsecond => "microseconds",
            Unit::Nanosecond => "nanoseconds",
        }
    }

    /// A very succinct label corresponding to this unit.
    pub(crate) fn compact(&self) -> &'static str {
        match *self {
            Unit::Year => "y",
            Unit::Month => "mo",
            Unit::Week => "w",
            Unit::Day => "d",
            Unit::Hour => "h",
            Unit::Minute => "m",
            Unit::Second => "s",
            Unit::Millisecond => "ms",
            Unit::Microsecond => "µs",
            Unit::Nanosecond => "ns",
        }
    }

    /// Return this unit as a `usize`.
    ///
    /// This is use `unit as usize`.
    pub(crate) fn as_usize(&self) -> usize {
        *self as usize
    }

    /// The inverse of `unit as usize`.
    fn from_usize(n: usize) -> Option<Unit> {
        match n {
            0 => Some(Unit::Nanosecond),
            1 => Some(Unit::Microsecond),
            2 => Some(Unit::Millisecond),
            3 => Some(Unit::Second),
            4 => Some(Unit::Minute),
            5 => Some(Unit::Hour),
            6 => Some(Unit::Day),
            7 => Some(Unit::Week),
            8 => Some(Unit::Month),
            9 => Some(Unit::Year),
            _ => None,
        }
    }

    /// Returns an error corresponding the boundaries of this unit.
    ///
    /// This is useful in contexts where one is doing arithmetic on integers
    /// where the units of those integers aren't known statically.
    fn error(self) -> b::BoundsError {
        match self {
            Unit::Year => b::SpanYears::error(),
            Unit::Month => b::SpanMonths::error(),
            Unit::Week => b::SpanWeeks::error(),
            Unit::Day => b::SpanDays::error(),
            Unit::Hour => b::SpanHours::error(),
            Unit::Minute => b::SpanMinutes::error(),
            Unit::Second => b::SpanSeconds::error(),
            Unit::Millisecond => b::SpanMilliseconds::error(),
            Unit::Microsecond => b::SpanMicroseconds::error(),
            Unit::Nanosecond => b::SpanNanoseconds::error(),
        }
    }
}

#[cfg(test)]
impl quickcheck::Arbitrary for Unit {
    fn arbitrary(g: &mut quickcheck::Gen) -> Unit {
        Unit::from_usize(usize::arbitrary(g) % 10).unwrap()
    }

    fn shrink(&self) -> alloc::boxed::Box<dyn Iterator<Item = Self>> {
        alloc::boxed::Box::new(
            (*self as usize)
                .shrink()
                .map(|n| Unit::from_usize(n % 10).unwrap()),
        )
    }
}

/// Options for [`Span::checked_add`] and [`Span::checked_sub`].
///
/// This type provides a way to ergonomically add two spans with an optional
/// relative datetime. Namely, a relative datetime is only needed when at least
/// one of the two spans being added (or subtracted) has a non-zero calendar
/// unit (years, months, weeks or days). Otherwise, an error will be returned.
///
/// Callers may use [`SpanArithmetic::days_are_24_hours`] to opt into 24-hour
/// invariant days (and 7-day weeks) without providing a relative datetime.
///
/// The main way to construct values of this type is with its `From` trait
/// implementations:
///
/// * `From<Span> for SpanArithmetic` adds (or subtracts) the given span to the
/// receiver in [`Span::checked_add`] (or [`Span::checked_sub`]).
/// * `From<(Span, civil::Date)> for SpanArithmetic` adds (or subtracts)
/// the given span to the receiver in [`Span::checked_add`] (or
/// [`Span::checked_sub`]), relative to the given date. There are also `From`
/// implementations for `civil::DateTime`, `Zoned` and [`SpanRelativeTo`].
///
/// # Example
///
/// ```
/// use jiff::ToSpan;
///
/// assert_eq!(
///     1.hour().checked_add(30.minutes())?,
///     1.hour().minutes(30).fieldwise(),
/// );
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SpanArithmetic<'a> {
    duration: Duration,
    relative: Option<SpanRelativeTo<'a>>,
}

impl<'a> SpanArithmetic<'a> {
    /// This is a convenience function for setting the relative option on
    /// this configuration to [`SpanRelativeTo::days_are_24_hours`].
    ///
    /// # Example
    ///
    /// When doing arithmetic on spans involving days, either a relative
    /// datetime must be provided, or a special assertion opting into 24-hour
    /// days is required. Otherwise, you get an error.
    ///
    /// ```
    /// use jiff::{SpanArithmetic, ToSpan};
    ///
    /// let span1 = 2.days().hours(12);
    /// let span2 = 12.hours();
    /// // No relative date provided, which results in an error.
    /// assert_eq!(
    ///     span1.checked_add(span2).unwrap_err().to_string(),
    ///     "using unit 'day' in a span or configuration requires that \
    ///      either a relative reference time be given or \
    ///      `jiff::SpanRelativeTo::days_are_24_hours()` is used to indicate \
    ///      invariant 24-hour days, but neither were provided",
    /// );
    /// let sum = span1.checked_add(
    ///     SpanArithmetic::from(span2).days_are_24_hours(),
    /// )?;
    /// assert_eq!(sum, 3.days().fieldwise());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn days_are_24_hours(self) -> SpanArithmetic<'a> {
        self.relative(SpanRelativeTo::days_are_24_hours())
    }
}

impl<'a> SpanArithmetic<'a> {
    #[inline]
    fn relative<R: Into<SpanRelativeTo<'a>>>(
        self,
        relative: R,
    ) -> SpanArithmetic<'a> {
        SpanArithmetic { relative: Some(relative.into()), ..self }
    }

    #[inline]
    fn checked_add(self, span1: Span) -> Result<Span, Error> {
        match self.duration.to_signed()? {
            SDuration::Span(span2) => {
                span1.checked_add_span(self.relative, &span2)
            }
            SDuration::Absolute(dur2) => {
                span1.checked_add_duration(self.relative, dur2)
            }
        }
    }
}

impl From<Span> for SpanArithmetic<'static> {
    fn from(span: Span) -> SpanArithmetic<'static> {
        let duration = Duration::from(span);
        SpanArithmetic { duration, relative: None }
    }
}

impl<'a> From<&'a Span> for SpanArithmetic<'static> {
    fn from(span: &'a Span) -> SpanArithmetic<'static> {
        let duration = Duration::from(*span);
        SpanArithmetic { duration, relative: None }
    }
}

impl From<(Span, Date)> for SpanArithmetic<'static> {
    #[inline]
    fn from((span, date): (Span, Date)) -> SpanArithmetic<'static> {
        SpanArithmetic::from(span).relative(date)
    }
}

impl From<(Span, DateTime)> for SpanArithmetic<'static> {
    #[inline]
    fn from((span, datetime): (Span, DateTime)) -> SpanArithmetic<'static> {
        SpanArithmetic::from(span).relative(datetime)
    }
}

impl<'a> From<(Span, &'a Zoned)> for SpanArithmetic<'a> {
    #[inline]
    fn from((span, zoned): (Span, &'a Zoned)) -> SpanArithmetic<'a> {
        SpanArithmetic::from(span).relative(zoned)
    }
}

impl<'a> From<(Span, SpanRelativeTo<'a>)> for SpanArithmetic<'a> {
    #[inline]
    fn from(
        (span, relative): (Span, SpanRelativeTo<'a>),
    ) -> SpanArithmetic<'a> {
        SpanArithmetic::from(span).relative(relative)
    }
}

impl<'a> From<(&'a Span, Date)> for SpanArithmetic<'static> {
    #[inline]
    fn from((span, date): (&'a Span, Date)) -> SpanArithmetic<'static> {
        SpanArithmetic::from(span).relative(date)
    }
}

impl<'a> From<(&'a Span, DateTime)> for SpanArithmetic<'static> {
    #[inline]
    fn from(
        (span, datetime): (&'a Span, DateTime),
    ) -> SpanArithmetic<'static> {
        SpanArithmetic::from(span).relative(datetime)
    }
}

impl<'a, 'b> From<(&'a Span, &'b Zoned)> for SpanArithmetic<'b> {
    #[inline]
    fn from((span, zoned): (&'a Span, &'b Zoned)) -> SpanArithmetic<'b> {
        SpanArithmetic::from(span).relative(zoned)
    }
}

impl<'a, 'b> From<(&'a Span, SpanRelativeTo<'b>)> for SpanArithmetic<'b> {
    #[inline]
    fn from(
        (span, relative): (&'a Span, SpanRelativeTo<'b>),
    ) -> SpanArithmetic<'b> {
        SpanArithmetic::from(span).relative(relative)
    }
}

impl From<SignedDuration> for SpanArithmetic<'static> {
    fn from(duration: SignedDuration) -> SpanArithmetic<'static> {
        let duration = Duration::from(duration);
        SpanArithmetic { duration, relative: None }
    }
}

impl From<(SignedDuration, Date)> for SpanArithmetic<'static> {
    #[inline]
    fn from(
        (duration, date): (SignedDuration, Date),
    ) -> SpanArithmetic<'static> {
        SpanArithmetic::from(duration).relative(date)
    }
}

impl From<(SignedDuration, DateTime)> for SpanArithmetic<'static> {
    #[inline]
    fn from(
        (duration, datetime): (SignedDuration, DateTime),
    ) -> SpanArithmetic<'static> {
        SpanArithmetic::from(duration).relative(datetime)
    }
}

impl<'a> From<(SignedDuration, &'a Zoned)> for SpanArithmetic<'a> {
    #[inline]
    fn from(
        (duration, zoned): (SignedDuration, &'a Zoned),
    ) -> SpanArithmetic<'a> {
        SpanArithmetic::from(duration).relative(zoned)
    }
}

impl From<UnsignedDuration> for SpanArithmetic<'static> {
    fn from(duration: UnsignedDuration) -> SpanArithmetic<'static> {
        let duration = Duration::from(duration);
        SpanArithmetic { duration, relative: None }
    }
}

impl From<(UnsignedDuration, Date)> for SpanArithmetic<'static> {
    #[inline]
    fn from(
        (duration, date): (UnsignedDuration, Date),
    ) -> SpanArithmetic<'static> {
        SpanArithmetic::from(duration).relative(date)
    }
}

impl From<(UnsignedDuration, DateTime)> for SpanArithmetic<'static> {
    #[inline]
    fn from(
        (duration, datetime): (UnsignedDuration, DateTime),
    ) -> SpanArithmetic<'static> {
        SpanArithmetic::from(duration).relative(datetime)
    }
}

impl<'a> From<(UnsignedDuration, &'a Zoned)> for SpanArithmetic<'a> {
    #[inline]
    fn from(
        (duration, zoned): (UnsignedDuration, &'a Zoned),
    ) -> SpanArithmetic<'a> {
        SpanArithmetic::from(duration).relative(zoned)
    }
}

/// Options for [`Span::compare`].
///
/// This type provides a way to ergonomically compare two spans with an
/// optional relative datetime. Namely, a relative datetime is only needed when
/// at least one of the two spans being compared has a non-zero calendar unit
/// (years, months, weeks or days). Otherwise, an error will be returned.
///
/// Callers may use [`SpanCompare::days_are_24_hours`] to opt into 24-hour
/// invariant days (and 7-day weeks) without providing a relative datetime.
///
/// The main way to construct values of this type is with its `From` trait
/// implementations:
///
/// * `From<Span> for SpanCompare` compares the given span to the receiver
/// in [`Span::compare`].
/// * `From<(Span, civil::Date)> for SpanCompare` compares the given span
/// to the receiver in [`Span::compare`], relative to the given date. There
/// are also `From` implementations for `civil::DateTime`, `Zoned` and
/// [`SpanRelativeTo`].
///
/// # Example
///
/// ```
/// use jiff::ToSpan;
///
/// let span1 = 3.hours();
/// let span2 = 180.minutes();
/// assert_eq!(span1.compare(span2)?, std::cmp::Ordering::Equal);
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SpanCompare<'a> {
    span: Span,
    relative: Option<SpanRelativeTo<'a>>,
}

impl<'a> SpanCompare<'a> {
    /// This is a convenience function for setting the relative option on
    /// this configuration to [`SpanRelativeTo::days_are_24_hours`].
    ///
    /// # Example
    ///
    /// When comparing spans involving days, either a relative datetime must be
    /// provided, or a special assertion opting into 24-hour days is
    /// required. Otherwise, you get an error.
    ///
    /// ```
    /// use jiff::{SpanCompare, ToSpan};
    ///
    /// let span1 = 2.days().hours(12);
    /// let span2 = 60.hours();
    /// // No relative date provided, which results in an error.
    /// assert_eq!(
    ///     span1.compare(span2).unwrap_err().to_string(),
    ///     "using unit 'day' in a span or configuration requires that \
    ///      either a relative reference time be given or \
    ///      `jiff::SpanRelativeTo::days_are_24_hours()` is used to indicate \
    ///      invariant 24-hour days, but neither were provided",
    /// );
    /// let ordering = span1.compare(
    ///     SpanCompare::from(span2).days_are_24_hours(),
    /// )?;
    /// assert_eq!(ordering, std::cmp::Ordering::Equal);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn days_are_24_hours(self) -> SpanCompare<'a> {
        self.relative(SpanRelativeTo::days_are_24_hours())
    }
}

impl<'a> SpanCompare<'a> {
    #[inline]
    fn new(span: Span) -> SpanCompare<'static> {
        SpanCompare { span, relative: None }
    }

    #[inline]
    fn relative<R: Into<SpanRelativeTo<'a>>>(
        self,
        relative: R,
    ) -> SpanCompare<'a> {
        SpanCompare { relative: Some(relative.into()), ..self }
    }

    fn compare(self, span: Span) -> Result<Ordering, Error> {
        let (span1, span2) = (span, self.span);
        let unit = span1.largest_unit().max(span2.largest_unit());
        let start = match self.relative {
            Some(r) => match r.to_relative(unit)? {
                Some(r) => r,
                None => {
                    let dur1 = span1.to_invariant_duration();
                    let dur2 = span2.to_invariant_duration();
                    return Ok(dur1.cmp(&dur2));
                }
            },
            None => {
                requires_relative_date_err(unit)?;
                let dur1 = span1.to_invariant_duration();
                let dur2 = span2.to_invariant_duration();
                return Ok(dur1.cmp(&dur2));
            }
        };
        let end1 = start.checked_add(span1)?.to_duration();
        let end2 = start.checked_add(span2)?.to_duration();
        Ok(end1.cmp(&end2))
    }
}

impl From<Span> for SpanCompare<'static> {
    fn from(span: Span) -> SpanCompare<'static> {
        SpanCompare::new(span)
    }
}

impl<'a> From<&'a Span> for SpanCompare<'static> {
    fn from(span: &'a Span) -> SpanCompare<'static> {
        SpanCompare::new(*span)
    }
}

impl From<(Span, Date)> for SpanCompare<'static> {
    #[inline]
    fn from((span, date): (Span, Date)) -> SpanCompare<'static> {
        SpanCompare::from(span).relative(date)
    }
}

impl From<(Span, DateTime)> for SpanCompare<'static> {
    #[inline]
    fn from((span, datetime): (Span, DateTime)) -> SpanCompare<'static> {
        SpanCompare::from(span).relative(datetime)
    }
}

impl<'a> From<(Span, &'a Zoned)> for SpanCompare<'a> {
    #[inline]
    fn from((span, zoned): (Span, &'a Zoned)) -> SpanCompare<'a> {
        SpanCompare::from(span).relative(zoned)
    }
}

impl<'a> From<(Span, SpanRelativeTo<'a>)> for SpanCompare<'a> {
    #[inline]
    fn from((span, relative): (Span, SpanRelativeTo<'a>)) -> SpanCompare<'a> {
        SpanCompare::from(span).relative(relative)
    }
}

impl<'a> From<(&'a Span, Date)> for SpanCompare<'static> {
    #[inline]
    fn from((span, date): (&'a Span, Date)) -> SpanCompare<'static> {
        SpanCompare::from(span).relative(date)
    }
}

impl<'a> From<(&'a Span, DateTime)> for SpanCompare<'static> {
    #[inline]
    fn from((span, datetime): (&'a Span, DateTime)) -> SpanCompare<'static> {
        SpanCompare::from(span).relative(datetime)
    }
}

impl<'a, 'b> From<(&'a Span, &'b Zoned)> for SpanCompare<'b> {
    #[inline]
    fn from((span, zoned): (&'a Span, &'b Zoned)) -> SpanCompare<'b> {
        SpanCompare::from(span).relative(zoned)
    }
}

impl<'a, 'b> From<(&'a Span, SpanRelativeTo<'b>)> for SpanCompare<'b> {
    #[inline]
    fn from(
        (span, relative): (&'a Span, SpanRelativeTo<'b>),
    ) -> SpanCompare<'b> {
        SpanCompare::from(span).relative(relative)
    }
}

/// Options for [`Span::total`].
///
/// This type provides a way to ergonomically determine the number of a
/// particular unit in a span, with a potentially fractional component, with
/// an optional relative datetime. Namely, a relative datetime is only needed
/// when the span has a non-zero calendar unit (years, months, weeks or days).
/// Otherwise, an error will be returned.
///
/// Callers may use [`SpanTotal::days_are_24_hours`] to opt into 24-hour
/// invariant days (and 7-day weeks) without providing a relative datetime.
///
/// The main way to construct values of this type is with its `From` trait
/// implementations:
///
/// * `From<Unit> for SpanTotal` computes a total for the given unit in the
/// receiver span for [`Span::total`].
/// * `From<(Unit, civil::Date)> for SpanTotal` computes a total for the given
/// unit in the receiver span for [`Span::total`], relative to the given date.
/// There are also `From` implementations for `civil::DateTime`, `Zoned` and
/// [`SpanRelativeTo`].
///
/// # Example
///
/// This example shows how to find the number of seconds in a particular span:
///
/// ```
/// use jiff::{ToSpan, Unit};
///
/// let span = 3.hours().minutes(10);
/// assert_eq!(span.total(Unit::Second)?, 11_400.0);
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Example: 24 hour days
///
/// This shows how to find the total number of 24 hour days in `123,456,789`
/// seconds.
///
/// ```
/// use jiff::{SpanTotal, ToSpan, Unit};
///
/// let span = 123_456_789.seconds();
/// assert_eq!(
///     span.total(SpanTotal::from(Unit::Day).days_are_24_hours())?,
///     1428.8980208333332,
/// );
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Example: DST is taken into account
///
/// The month of March 2024 in `America/New_York` had 31 days, but one of those
/// days was 23 hours long due a transition into daylight saving time:
///
/// ```
/// use jiff::{civil::date, ToSpan, Unit};
///
/// let span = 744.hours();
/// let relative = date(2024, 3, 1).in_tz("America/New_York")?;
/// // Because of the short day, 744 hours is actually a little *more* than
/// // 1 month starting from 2024-03-01.
/// assert_eq!(span.total((Unit::Month, &relative))?, 1.0013888888888889);
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// Now compare what happens when the relative datetime is civil and not
/// time zone aware:
///
/// ```
/// use jiff::{civil::date, ToSpan, Unit};
///
/// let span = 744.hours();
/// let relative = date(2024, 3, 1);
/// assert_eq!(span.total((Unit::Month, relative))?, 1.0);
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SpanTotal<'a> {
    unit: Unit,
    relative: Option<SpanRelativeTo<'a>>,
}

impl<'a> SpanTotal<'a> {
    /// This is a convenience function for setting the relative option on
    /// this configuration to [`SpanRelativeTo::days_are_24_hours`].
    ///
    /// # Example
    ///
    /// When computing the total duration for spans involving days, either a
    /// relative datetime must be provided, or a special assertion opting into
    /// 24-hour days is required. Otherwise, you get an error.
    ///
    /// ```
    /// use jiff::{civil::date, SpanTotal, ToSpan, Unit};
    ///
    /// let span = 2.days().hours(12);
    ///
    /// // No relative date provided, which results in an error.
    /// assert_eq!(
    ///     span.total(Unit::Hour).unwrap_err().to_string(),
    ///     "using unit 'day' in a span or configuration requires that either \
    ///      a relative reference time be given or \
    ///      `jiff::SpanRelativeTo::days_are_24_hours()` is used to indicate \
    ///      invariant 24-hour days, but neither were provided",
    /// );
    ///
    /// // If we can assume all days are 24 hours, then we can assert it:
    /// let total = span.total(
    ///     SpanTotal::from(Unit::Hour).days_are_24_hours(),
    /// )?;
    /// assert_eq!(total, 60.0);
    ///
    /// // Or provide a relative datetime, which is preferred if possible:
    /// let total = span.total((Unit::Hour, date(2025, 1, 26)))?;
    /// assert_eq!(total, 60.0);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn days_are_24_hours(self) -> SpanTotal<'a> {
        self.relative(SpanRelativeTo::days_are_24_hours())
    }
}

impl<'a> SpanTotal<'a> {
    #[inline]
    fn new(unit: Unit) -> SpanTotal<'static> {
        SpanTotal { unit, relative: None }
    }

    #[inline]
    fn relative<R: Into<SpanRelativeTo<'a>>>(
        self,
        relative: R,
    ) -> SpanTotal<'a> {
        SpanTotal { relative: Some(relative.into()), ..self }
    }

    fn total(self, span: Span) -> Result<f64, Error> {
        let max_unit = self.unit.max(span.largest_unit());
        let relative = match self.relative {
            Some(r) => match r.to_relative(max_unit)? {
                Some(r) => r,
                None => {
                    return Ok(self.total_invariant(span));
                }
            },
            None => {
                requires_relative_date_err(max_unit)?;
                return Ok(self.total_invariant(span));
            }
        };
        let relspan = relative.into_relative_span(self.unit, span)?;
        if !self.unit.is_variable() {
            return Ok(self.total_invariant(relspan.span));
        }

        assert!(self.unit >= Unit::Day);
        let sign = relspan.span.get_sign();
        let (relative_start, relative_end) = match relspan.kind {
            RelativeSpanKind::Civil { start, end } => {
                let start = Relative::Civil(start);
                let end = Relative::Civil(end);
                (start, end)
            }
            RelativeSpanKind::Zoned { start, end } => {
                let start = Relative::Zoned(start);
                let end = Relative::Zoned(end);
                (start, end)
            }
        };
        let (relative0, relative1) = unit_start_and_end(
            &relative_start,
            relspan.span.without_lower(self.unit),
            self.unit,
            sign.as_i64(),
        )?;
        let denom = (relative1 - relative0).as_nanos() as f64;
        let numer = (relative_end.to_duration() - relative0).as_nanos() as f64;
        let unit_val = relspan.span.get_unit(self.unit) as f64;
        Ok(unit_val + (numer / denom) * (sign.as_i8() as f64))
    }

    #[inline]
    fn total_invariant(&self, span: Span) -> f64 {
        assert!(self.unit <= Unit::Week);
        let dur = span.to_invariant_duration().as_nanos();
        // We do this instead of using `SignedDuration::as_secs_f64()`
        // because of floating point precision. It seems that if we represent
        // our ratio as floats of seconds instead of nanoseconds, then there
        // is more loss of precision than using nanoseconds. Unfortunately,
        // this does mean manifesting `i128` values.
        (dur as f64) / (self.unit.duration().as_nanos() as f64)
    }
}

impl From<Unit> for SpanTotal<'static> {
    #[inline]
    fn from(unit: Unit) -> SpanTotal<'static> {
        SpanTotal::new(unit)
    }
}

impl From<(Unit, Date)> for SpanTotal<'static> {
    #[inline]
    fn from((unit, date): (Unit, Date)) -> SpanTotal<'static> {
        SpanTotal::from(unit).relative(date)
    }
}

impl From<(Unit, DateTime)> for SpanTotal<'static> {
    #[inline]
    fn from((unit, datetime): (Unit, DateTime)) -> SpanTotal<'static> {
        SpanTotal::from(unit).relative(datetime)
    }
}

impl<'a> From<(Unit, &'a Zoned)> for SpanTotal<'a> {
    #[inline]
    fn from((unit, zoned): (Unit, &'a Zoned)) -> SpanTotal<'a> {
        SpanTotal::from(unit).relative(zoned)
    }
}

impl<'a> From<(Unit, SpanRelativeTo<'a>)> for SpanTotal<'a> {
    #[inline]
    fn from((unit, relative): (Unit, SpanRelativeTo<'a>)) -> SpanTotal<'a> {
        SpanTotal::from(unit).relative(relative)
    }
}

/// Options for [`Span::round`].
///
/// This type provides a way to configure the rounding of a span. This
/// includes setting the smallest unit (i.e., the unit to round), the
/// largest unit, the rounding increment, the rounding mode (e.g., "ceil" or
/// "truncate") and the datetime that the span is relative to.
///
/// `Span::round` accepts anything that implements `Into<SpanRound>`. There are
/// a few key trait implementations that make this convenient:
///
/// * `From<Unit> for SpanRound` will construct a rounding configuration where
/// the smallest unit is set to the one given.
/// * `From<(Unit, i64)> for SpanRound` will construct a rounding configuration
/// where the smallest unit and the rounding increment are set to the ones
/// given.
///
/// In order to set other options (like the largest unit, the rounding mode
/// and the relative datetime), one must explicitly create a `SpanRound` and
/// pass it to `Span::round`.
///
/// # Example
///
/// This example shows how to find how many full 3 month quarters are in a
/// particular span of time.
///
/// ```
/// use jiff::{civil::date, RoundMode, SpanRound, ToSpan, Unit};
///
/// let span1 = 10.months().days(15);
/// let round = SpanRound::new()
///     .smallest(Unit::Month)
///     .increment(3)
///     .mode(RoundMode::Trunc)
///     // A relative datetime must be provided when
///     // rounding involves calendar units.
///     .relative(date(2024, 1, 1));
/// let span2 = span1.round(round)?;
/// assert_eq!(span2.get_months() / 3, 3);
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SpanRound<'a> {
    largest: Option<Unit>,
    smallest: Unit,
    mode: RoundMode,
    increment: i64,
    relative: Option<SpanRelativeTo<'a>>,
}

impl<'a> SpanRound<'a> {
    /// Create a new default configuration for rounding a span via
    /// [`Span::round`].
    ///
    /// The default configuration does no rounding.
    #[inline]
    pub fn new() -> SpanRound<'static> {
        SpanRound {
            largest: None,
            smallest: Unit::Nanosecond,
            mode: RoundMode::HalfExpand,
            increment: 1,
            relative: None,
        }
    }

    /// Set the smallest units allowed in the span returned. These are the
    /// units that the span is rounded to.
    ///
    /// # Errors
    ///
    /// The smallest units must be no greater than the largest units. If this
    /// is violated, then rounding a span with this configuration will result
    /// in an error.
    ///
    /// If a smallest unit bigger than days is selected without a relative
    /// datetime reference point, then an error is returned when using this
    /// configuration with [`Span::round`].
    ///
    /// # Example
    ///
    /// A basic example that rounds to the nearest minute:
    ///
    /// ```
    /// use jiff::{ToSpan, Unit};
    ///
    /// let span = 15.minutes().seconds(46);
    /// assert_eq!(span.round(Unit::Minute)?, 16.minutes().fieldwise());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn smallest(self, unit: Unit) -> SpanRound<'a> {
        SpanRound { smallest: unit, ..self }
    }

    /// Set the largest units allowed in the span returned.
    ///
    /// When a largest unit is not specified, then it defaults to the largest
    /// non-zero unit that is at least as big as the configured smallest
    /// unit. For example, given a span of `2 months 17 hours`, the default
    /// largest unit would be `Unit::Month`. The default implies that a span's
    /// units do not get "bigger" than what was given.
    ///
    /// Once a largest unit is set, there is no way to change this rounding
    /// configuration back to using the "automatic" default. Instead, callers
    /// must create a new configuration.
    ///
    /// If a largest unit is set and no other options are set, then the
    /// rounding operation can be said to be a "re-balancing." That is, the
    /// span won't lose precision, but the way in which it is expressed may
    /// change.
    ///
    /// # Errors
    ///
    /// The largest units, when set, must be at least as big as the smallest
    /// units (which defaults to [`Unit::Nanosecond`]). If this is violated,
    /// then rounding a span with this configuration will result in an error.
    ///
    /// If a largest unit bigger than days is selected without a relative
    /// datetime reference point, then an error is returned when using this
    /// configuration with [`Span::round`].
    ///
    /// # Example: re-balancing
    ///
    /// This shows how a span can be re-balanced without losing precision:
    ///
    /// ```
    /// use jiff::{SpanRound, ToSpan, Unit};
    ///
    /// let span = 86_401_123_456_789i64.nanoseconds();
    /// assert_eq!(
    ///     span.round(SpanRound::new().largest(Unit::Hour))?.fieldwise(),
    ///     24.hours().seconds(1).milliseconds(123).microseconds(456).nanoseconds(789),
    /// );
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// If you need to use a largest unit bigger than hours, then you must
    /// provide a relative datetime as a reference point (otherwise an error
    /// will occur):
    ///
    /// ```
    /// use jiff::{civil::date, SpanRound, ToSpan, Unit};
    ///
    /// let span = 3_968_000.seconds();
    /// let round = SpanRound::new()
    ///     .largest(Unit::Day)
    ///     .relative(date(2024, 7, 1));
    /// assert_eq!(
    ///     span.round(round)?,
    ///     45.days().hours(22).minutes(13).seconds(20).fieldwise(),
    /// );
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// As a special case for days, one can instead opt into invariant 24-hour
    /// days (and 7-day weeks) without providing an explicit relative date:
    ///
    /// ```
    /// use jiff::{SpanRound, ToSpan, Unit};
    ///
    /// let span = 86_401_123_456_789i64.nanoseconds();
    /// assert_eq!(
    ///     span.round(
    ///         SpanRound::new().largest(Unit::Day).days_are_24_hours(),
    ///     )?.fieldwise(),
    ///     1.day().seconds(1).milliseconds(123).microseconds(456).nanoseconds(789),
    /// );
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Example: re-balancing while taking DST into account
    ///
    /// When given a zone aware relative datetime, rounding will even take
    /// DST into account:
    ///
    /// ```
    /// use jiff::{SpanRound, ToSpan, Unit, Zoned};
    ///
    /// let span = 2756.hours();
    /// let zdt = "2020-01-01T00:00+01:00[Europe/Rome]".parse::<Zoned>()?;
    /// let round = SpanRound::new().largest(Unit::Year).relative(&zdt);
    /// assert_eq!(
    ///     span.round(round)?,
    ///     3.months().days(23).hours(21).fieldwise(),
    /// );
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// Now compare with the same operation, but on a civil datetime (which is
    /// not aware of time zone):
    ///
    /// ```
    /// use jiff::{civil::DateTime, SpanRound, ToSpan, Unit};
    ///
    /// let span = 2756.hours();
    /// let dt = "2020-01-01T00:00".parse::<DateTime>()?;
    /// let round = SpanRound::new().largest(Unit::Year).relative(dt);
    /// assert_eq!(
    ///     span.round(round)?,
    ///     3.months().days(23).hours(20).fieldwise(),
    /// );
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// The result is 1 hour shorter. This is because, in the zone
    /// aware re-balancing, it accounts for the transition into DST at
    /// `2020-03-29T01:00Z`, which skips an hour. This makes the span one hour
    /// longer because one of the days in the span is actually only 23 hours
    /// long instead of 24 hours.
    #[inline]
    pub fn largest(self, unit: Unit) -> SpanRound<'a> {
        SpanRound { largest: Some(unit), ..self }
    }

    /// Set the rounding mode.
    ///
    /// This defaults to [`RoundMode::HalfExpand`], which makes rounding work
    /// like how you were taught in school.
    ///
    /// # Example
    ///
    /// A basic example that rounds to the nearest minute, but changing its
    /// rounding mode to truncation:
    ///
    /// ```
    /// use jiff::{RoundMode, SpanRound, ToSpan, Unit};
    ///
    /// let span = 15.minutes().seconds(46);
    /// assert_eq!(
    ///     span.round(SpanRound::new()
    ///         .smallest(Unit::Minute)
    ///         .mode(RoundMode::Trunc),
    ///     )?,
    ///     // The default round mode does rounding like
    ///     // how you probably learned in school, and would
    ///     // result in rounding up to 16 minutes. But we
    ///     // change it to truncation here, which makes it
    ///     // round down.
    ///     15.minutes().fieldwise(),
    /// );
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn mode(self, mode: RoundMode) -> SpanRound<'a> {
        SpanRound { mode, ..self }
    }

    /// Set the rounding increment for the smallest unit.
    ///
    /// The default value is `1`. Other values permit rounding the smallest
    /// unit to the nearest integer increment specified. For example, if the
    /// smallest unit is set to [`Unit::Minute`], then a rounding increment of
    /// `30` would result in rounding in increments of a half hour. That is,
    /// the only minute value that could result would be `0` or `30`.
    ///
    /// # Errors
    ///
    /// When the smallest unit is less than days, the rounding increment must
    /// divide evenly into the next highest unit after the smallest unit
    /// configured (and must not be equivalent to it). For example, if the
    /// smallest unit is [`Unit::Nanosecond`], then *some* of the valid values
    /// for the rounding increment are `1`, `2`, `4`, `5`, `100` and `500`.
    /// Namely, any integer that divides evenly into `1,000` nanoseconds since
    /// there are `1,000` nanoseconds in the next highest unit (microseconds).
    ///
    /// In all cases, the increment must be greater than zero and less than
    /// or equal to `1_000_000_000`.
    ///
    /// The error will occur when computing the span, and not when setting
    /// the increment here.
    ///
    /// # Example
    ///
    /// This shows how to round a span to the nearest 5 minute increment:
    ///
    /// ```
    /// use jiff::{ToSpan, Unit};
    ///
    /// let span = 4.hours().minutes(2).seconds(30);
    /// assert_eq!(
    ///     span.round((Unit::Minute, 5))?,
    ///     4.hours().minutes(5).fieldwise(),
    /// );
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn increment(self, increment: i64) -> SpanRound<'a> {
        SpanRound { increment, ..self }
    }

    /// Set the relative datetime to use when rounding a span.
    ///
    /// A relative datetime is only required when calendar units (units greater
    /// than days) are involved. This includes having calendar units in the
    /// original span, or calendar units in the configured smallest or largest
    /// unit. A relative datetime is required when calendar units are used
    /// because the duration of a particular calendar unit (like 1 month or 1
    /// year) is variable and depends on the date. For example, 1 month from
    /// 2024-01-01 is 31 days, but 1 month from 2024-02-01 is 29 days.
    ///
    /// A relative datetime is provided by anything that implements
    /// `Into<SpanRelativeTo>`. There are a few convenience trait
    /// implementations provided:
    ///
    /// * `From<&Zoned> for SpanRelativeTo` uses a zone aware datetime to do
    /// rounding. In this case, rounding will take time zone transitions into
    /// account. In particular, when using a zoned relative datetime, not all
    /// days are necessarily 24 hours.
    /// * `From<civil::DateTime> for SpanRelativeTo` uses a civil datetime. In
    /// this case, all days will be considered 24 hours long.
    /// * `From<civil::Date> for SpanRelativeTo` uses a civil date. In this
    /// case, all days will be considered 24 hours long.
    ///
    /// Note that one can impose 24-hour days without providing a reference
    /// date via [`SpanRelativeTo::days_are_24_hours`].
    ///
    /// # Errors
    ///
    /// If rounding involves a calendar unit (units bigger than hours) and no
    /// relative datetime is provided, then this configuration will lead to
    /// an error when used with [`Span::round`].
    ///
    /// # Example
    ///
    /// This example shows very precisely how a DST transition can impact
    /// rounding and re-balancing. For example, consider the day `2024-11-03`
    /// in `America/New_York`. On this day, the 1 o'clock hour was repeated,
    /// making the day 24 hours long. This will be taken into account when
    /// rounding if a zoned datetime is provided as a reference point:
    ///
    /// ```
    /// use jiff::{SpanRound, ToSpan, Unit, Zoned};
    ///
    /// let zdt = "2024-11-03T00-04[America/New_York]".parse::<Zoned>()?;
    /// let round = SpanRound::new().largest(Unit::Hour).relative(&zdt);
    /// assert_eq!(1.day().round(round)?, 25.hours().fieldwise());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// And similarly for `2024-03-10`, where the 2 o'clock hour was skipped
    /// entirely:
    ///
    /// ```
    /// use jiff::{SpanRound, ToSpan, Unit, Zoned};
    ///
    /// let zdt = "2024-03-10T00-05[America/New_York]".parse::<Zoned>()?;
    /// let round = SpanRound::new().largest(Unit::Hour).relative(&zdt);
    /// assert_eq!(1.day().round(round)?, 23.hours().fieldwise());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn relative<R: Into<SpanRelativeTo<'a>>>(
        self,
        relative: R,
    ) -> SpanRound<'a> {
        SpanRound { relative: Some(relative.into()), ..self }
    }

    /// This is a convenience function for setting the relative option on
    /// this configuration to [`SpanRelativeTo::days_are_24_hours`].
    ///
    /// # Example
    ///
    /// When rounding spans involving days, either a relative datetime must be
    /// provided, or a special assertion opting into 24-hour days is
    /// required. Otherwise, you get an error.
    ///
    /// ```
    /// use jiff::{SpanRound, ToSpan, Unit};
    ///
    /// let span = 2.days().hours(12);
    /// // No relative date provided, which results in an error.
    /// assert_eq!(
    ///     span.round(Unit::Day).unwrap_err().to_string(),
    ///     "error with `smallest` rounding option: using unit 'day' in a \
    ///      span or configuration requires that either a relative reference \
    ///      time be given or `jiff::SpanRelativeTo::days_are_24_hours()` is \
    ///      used to indicate invariant 24-hour days, but neither were \
    ///      provided",
    /// );
    /// let rounded = span.round(
    ///     SpanRound::new().smallest(Unit::Day).days_are_24_hours(),
    /// )?;
    /// assert_eq!(rounded, 3.days().fieldwise());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn days_are_24_hours(self) -> SpanRound<'a> {
        self.relative(SpanRelativeTo::days_are_24_hours())
    }

    /// Returns the configured smallest unit on this round configuration.
    #[inline]
    pub(crate) fn get_smallest(&self) -> Unit {
        self.smallest
    }

    /// Returns the configured largest unit on this round configuration.
    #[inline]
    pub(crate) fn get_largest(&self) -> Option<Unit> {
        self.largest
    }

    /// Returns true only when rounding a span *may* change it. When it
    /// returns false, and if the span is already balanced according to
    /// the largest unit in this round configuration, then it is guaranteed
    /// that rounding is a no-op.
    ///
    /// This is useful to avoid rounding calls after doing span arithmetic
    /// on datetime types. This works because the "largest" unit is used to
    /// construct a balanced span for the difference between two datetimes.
    /// So we already know the span has been balanced. If this weren't the
    /// case, then the largest unit being different from the one in the span
    /// could result in rounding making a change. (And indeed, in the general
    /// case of span rounding below, we do a more involved check for this.)
    #[inline]
    pub(crate) fn rounding_may_change_span(&self) -> bool {
        self.smallest > Unit::Nanosecond || self.increment != 1
    }

    /// Like `SpanRound::rounding_may_change_span`, but applies to contexts
    /// where only calendar units are applicable.
    ///
    /// At time of writing (2026-02-05), this is only used for `civil::Date`.
    #[inline]
    pub(crate) fn rounding_calendar_only_may_change_span(&self) -> bool {
        self.smallest > Unit::Day || self.increment != 1
    }

    /// Does the actual span rounding.
    fn round(&self, span: Span) -> Result<Span, Error> {
        let existing_largest = span.largest_unit();
        let largest = self
            .largest
            .unwrap_or_else(|| self.smallest.max(existing_largest));
        let max = existing_largest.max(largest);
        let increment = Increment::for_span(self.smallest, self.increment)?;
        if largest < self.smallest {
            return Err(Error::from(
                UnitConfigError::LargestSmallerThanSmallest {
                    smallest: self.smallest,
                    largest,
                },
            ));
        }

        let relative = match self.relative {
            Some(ref r) => {
                match r.to_relative(max)? {
                    Some(r) => r,
                    None => {
                        // If our reference point is civil time, then its units
                        // are invariant as long as we are using day-or-lower
                        // everywhere. That is, the length of the duration is
                        // independent of the reference point. In which case,
                        // rounding is a simple matter of converting the span
                        // to a number of nanoseconds and then rounding that.
                        return Ok(round_span_invariant(
                            span, largest, &increment, self.mode,
                        )?);
                    }
                }
            }
            None => {
                // This is only okay if none of our units are above 'hour'.
                // A `Span` can still be rounded without a relative datetime
                // when it has weeks/days units, but that requires explicitly
                // specifying a special relative date marker, which is handled
                // by the `Some` case above.
                requires_relative_date_err(self.smallest)
                    .context(E::OptionSmallest)?;
                if let Some(largest) = self.largest {
                    requires_relative_date_err(largest)
                        .context(E::OptionLargest)?;
                }
                requires_relative_date_err(existing_largest)
                    .context(E::OptionLargestInSpan)?;
                assert!(max <= Unit::Week);
                return Ok(round_span_invariant(
                    span, largest, &increment, self.mode,
                )?);
            }
        };
        relative.round(span, largest, &increment, self.mode)
    }
}

impl Default for SpanRound<'static> {
    fn default() -> SpanRound<'static> {
        SpanRound::new()
    }
}

impl From<Unit> for SpanRound<'static> {
    fn from(unit: Unit) -> SpanRound<'static> {
        SpanRound::default().smallest(unit)
    }
}

impl From<(Unit, i64)> for SpanRound<'static> {
    fn from((unit, increment): (Unit, i64)) -> SpanRound<'static> {
        SpanRound::default().smallest(unit).increment(increment)
    }
}

/// A relative datetime for use with [`Span`] APIs.
///
/// A relative datetime can be one of the following: [`civil::Date`](Date),
/// [`civil::DateTime`](DateTime) or [`Zoned`]. It can be constructed from any
/// of the preceding types via `From` trait implementations.
///
/// A relative datetime is used to indicate how the calendar units of a `Span`
/// should be interpreted. For example, the span "1 month" does not have a
/// fixed meaning. One month from `2024-03-01` is 31 days, but one month from
/// `2024-04-01` is 30 days. Similar for years.
///
/// When a relative datetime in time zone aware (i.e., it is a `Zoned`), then
/// operations on a `Span` will also consider its day units to be variable in
/// length. For example, `2024-03-10` in `America/New_York` was only 23 hours
/// long, where as `2024-11-03` in `America/New_York` was 25 hours long. When
/// a relative datetime is civil, then days are considered to always be of a
/// fixed 24 hour length.
///
/// This type is principally used as an input to one of several different
/// [`Span`] APIs:
///
/// * [`Span::round`] rounds spans. A relative datetime is necessary when
/// dealing with calendar units. (But spans without calendar units can be
/// rounded without providing a relative datetime.)
/// * Span arithmetic via [`Span::checked_add`] and [`Span::checked_sub`].
/// A relative datetime is needed when adding or subtracting spans with
/// calendar units.
/// * Span comparisons via [`Span::compare`] require a relative datetime when
/// comparing spans with calendar units.
/// * Computing the "total" duration as a single floating point number via
/// [`Span::total`] also requires a relative datetime when dealing with
/// calendar units.
///
/// # Example
///
/// This example shows how to round a span with larger calendar units to
/// smaller units:
///
/// ```
/// use jiff::{SpanRound, ToSpan, Unit, Zoned};
///
/// let zdt: Zoned = "2012-01-01[Antarctica/Troll]".parse()?;
/// let round = SpanRound::new().largest(Unit::Day).relative(&zdt);
/// assert_eq!(1.year().round(round)?, 366.days().fieldwise());
///
/// // If you tried this without a relative datetime, it would fail:
/// let round = SpanRound::new().largest(Unit::Day);
/// assert!(1.year().round(round).is_err());
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SpanRelativeTo<'a> {
    kind: SpanRelativeToKind<'a>,
}

impl<'a> SpanRelativeTo<'a> {
    /// Creates a special marker that indicates all days ought to be assumed
    /// to be 24 hours without providing a relative reference time.
    ///
    /// This is relevant to the following APIs:
    ///
    /// * [`Span::checked_add`]
    /// * [`Span::checked_sub`]
    /// * [`Span::compare`]
    /// * [`Span::total`]
    /// * [`Span::round`]
    /// * [`Span::to_duration`]
    ///
    /// Specifically, in a previous version of Jiff, the above APIs permitted
    /// _silently_ assuming that days are always 24 hours when a relative
    /// reference date wasn't provided. In the current version of Jiff, this
    /// silent interpretation no longer happens and instead an error will
    /// occur.
    ///
    /// If you need to use these APIs with spans that contain non-zero units
    /// of days or weeks but without a relative reference date, then you may
    /// use this routine to create a special marker for `SpanRelativeTo` that
    /// permits the APIs above to assume days are always 24 hours.
    ///
    /// # Motivation
    ///
    /// The purpose of the marker is two-fold:
    ///
    /// * Requiring the marker is important for improving the consistency of
    /// `Span` APIs. Previously, some APIs (like [`Timestamp::checked_add`])
    /// would always return an error if the `Span` given had non-zero
    /// units of days or greater. On the other hand, other APIs (like
    /// [`Span::checked_add`]) would automatically assume days were always
    /// 24 hours if no relative reference time was given and either span had
    /// non-zero units of days. With this marker, APIs _never_ assume days are
    /// always 24 hours automatically.
    /// * When it _is_ appropriate to assume all days are 24 hours
    /// (for example, when only dealing with spans derived from
    /// [`civil`](crate::civil) datetimes) and where providing a relative
    /// reference datetime doesn't make sense. In this case, one _could_
    /// provide a "dummy" reference date since the precise date in civil time
    /// doesn't impact the length of a day. But a marker like the one returned
    /// here is more explicit for the purpose of assuming days are always 24
    /// hours.
    ///
    /// With that said, ideally, callers should provide a relative reference
    /// datetime if possible.
    ///
    /// See [Issue #48] for more discussion on this topic.
    ///
    /// # Example: different interpretations of "1 day"
    ///
    /// This example shows how "1 day" can be interpreted differently via the
    /// [`Span::total`] API:
    ///
    /// ```
    /// use jiff::{SpanRelativeTo, ToSpan, Unit, Zoned};
    ///
    /// let span = 1.day();
    ///
    /// // An error because days aren't always 24 hours:
    /// assert_eq!(
    ///     span.total(Unit::Hour).unwrap_err().to_string(),
    ///     "using unit 'day' in a span or configuration requires that either \
    ///      a relative reference time be given or \
    ///      `jiff::SpanRelativeTo::days_are_24_hours()` is used to indicate \
    ///      invariant 24-hour days, but neither were provided",
    /// );
    /// // Opt into invariant 24 hour days without a relative date:
    /// let marker = SpanRelativeTo::days_are_24_hours();
    /// let hours = span.total((Unit::Hour, marker))?;
    /// assert_eq!(hours, 24.0);
    /// // Days can be shorter than 24 hours:
    /// let zdt: Zoned = "2024-03-10[America/New_York]".parse()?;
    /// let hours = span.total((Unit::Hour, &zdt))?;
    /// assert_eq!(hours, 23.0);
    /// // Days can be longer than 24 hours:
    /// let zdt: Zoned = "2024-11-03[America/New_York]".parse()?;
    /// let hours = span.total((Unit::Hour, &zdt))?;
    /// assert_eq!(hours, 25.0);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// Similar behavior applies to the other APIs listed above.
    ///
    /// # Example: different interpretations of "1 week"
    ///
    /// This example shows how "1 week" can be interpreted differently via the
    /// [`Span::total`] API:
    ///
    /// ```
    /// use jiff::{SpanRelativeTo, ToSpan, Unit, Zoned};
    ///
    /// let span = 1.week();
    ///
    /// // An error because days aren't always 24 hours:
    /// assert_eq!(
    ///     span.total(Unit::Hour).unwrap_err().to_string(),
    ///     "using unit 'week' in a span or configuration requires that either \
    ///      a relative reference time be given or \
    ///      `jiff::SpanRelativeTo::days_are_24_hours()` is used to indicate \
    ///      invariant 24-hour days, but neither were provided",
    /// );
    /// // Opt into invariant 24 hour days without a relative date:
    /// let marker = SpanRelativeTo::days_are_24_hours();
    /// let hours = span.total((Unit::Hour, marker))?;
    /// assert_eq!(hours, 168.0);
    /// // Weeks can be shorter than 24*7 hours:
    /// let zdt: Zoned = "2024-03-10[America/New_York]".parse()?;
    /// let hours = span.total((Unit::Hour, &zdt))?;
    /// assert_eq!(hours, 167.0);
    /// // Weeks can be longer than 24*7 hours:
    /// let zdt: Zoned = "2024-11-03[America/New_York]".parse()?;
    /// let hours = span.total((Unit::Hour, &zdt))?;
    /// assert_eq!(hours, 169.0);
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Example: working with [`civil::Date`](crate::civil::Date)
    ///
    /// A `Span` returned by computing the difference in time between two
    /// [`civil::Date`](crate::civil::Date)s will have a non-zero number of
    /// days. In older versions of Jiff, if one wanted to add spans returned by
    /// these APIs, you could do so without futzing with relative dates. But
    /// now you either need to provide a relative date:
    ///
    /// ```
    /// use jiff::{civil::date, ToSpan};
    ///
    /// let d1 = date(2025, 1, 18);
    /// let d2 = date(2025, 1, 26);
    /// let d3 = date(2025, 2, 14);
    ///
    /// let span1 = d2 - d1;
    /// let span2 = d3 - d2;
    /// let total = span1.checked_add((span2, d1))?;
    /// assert_eq!(total, 27.days().fieldwise());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// Or you can provide a marker indicating that days are always 24 hours.
    /// This is fine for this use case since one is only doing civil calendar
    /// arithmetic and not working with time zones:
    ///
    /// ```
    /// use jiff::{civil::date, SpanRelativeTo, ToSpan};
    ///
    /// let d1 = date(2025, 1, 18);
    /// let d2 = date(2025, 1, 26);
    /// let d3 = date(2025, 2, 14);
    ///
    /// let span1 = d2 - d1;
    /// let span2 = d3 - d2;
    /// let total = span1.checked_add(
    ///     (span2, SpanRelativeTo::days_are_24_hours()),
    /// )?;
    /// assert_eq!(total, 27.days().fieldwise());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// [Issue #48]: https://github.com/BurntSushi/jiff/issues/48
    #[inline]
    pub const fn days_are_24_hours() -> SpanRelativeTo<'static> {
        let kind = SpanRelativeToKind::DaysAre24Hours;
        SpanRelativeTo { kind }
    }

    /// Converts this public API relative datetime into a more versatile
    /// internal representation of the same concept.
    ///
    /// The unit given should be the maximal non-zero unit present in the
    /// operation. (Which might involve two spans, in which case, it is the
    /// maximal non-zero unit across both spans.)
    ///
    /// Basically, the internal `Relative` type is `Cow` which means it isn't
    /// `Copy`. But it can present a more uniform API. The public API type
    /// doesn't have `Cow` so that it can be `Copy`.
    ///
    /// We also take this opportunity to attach some convenient data, such
    /// as a timestamp when the relative datetime type is civil.
    ///
    /// This can return `None` if this `SpanRelativeTo` isn't actually a
    /// datetime but a "marker" indicating some unit (like days) should be
    /// treated as invariant. Or `None` is returned when the given unit is
    /// always invariant (hours or smaller).
    ///
    /// In effect, given that `unit` is the maximal unit involved, `None` is
    /// returned when it's safe to assume that all units in the spans can be
    /// interpreted as invariant (even if they can sometimes be varying).
    ///
    /// # Errors
    ///
    /// If there was a problem doing this conversion, then an error is
    /// returned. In practice, this only occurs for a civil datetime near the
    /// civil datetime minimum and maximum values.
    fn to_relative(&self, unit: Unit) -> Result<Option<Relative<'a>>, Error> {
        if !unit.is_variable() {
            return Ok(None);
        }
        match self.kind {
            SpanRelativeToKind::Civil(dt) => {
                Ok(Some(Relative::Civil(RelativeCivil::new(dt)?)))
            }
            SpanRelativeToKind::Zoned(zdt) => {
                Ok(Some(Relative::Zoned(RelativeZoned {
                    zoned: DumbCow::Borrowed(zdt),
                })))
            }
            SpanRelativeToKind::DaysAre24Hours => {
                if matches!(unit, Unit::Year | Unit::Month) {
                    return Err(Error::from(
                        UnitConfigError::RelativeYearOrMonthGivenDaysAre24Hours {
                            unit,
                        },
                    ));
                }
                Ok(None)
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum SpanRelativeToKind<'a> {
    Civil(DateTime),
    Zoned(&'a Zoned),
    DaysAre24Hours,
}

impl<'a> From<&'a Zoned> for SpanRelativeTo<'a> {
    fn from(zdt: &'a Zoned) -> SpanRelativeTo<'a> {
        SpanRelativeTo { kind: SpanRelativeToKind::Zoned(zdt) }
    }
}

impl From<DateTime> for SpanRelativeTo<'static> {
    fn from(dt: DateTime) -> SpanRelativeTo<'static> {
        SpanRelativeTo { kind: SpanRelativeToKind::Civil(dt) }
    }
}

impl From<Date> for SpanRelativeTo<'static> {
    fn from(date: Date) -> SpanRelativeTo<'static> {
        let dt = DateTime::from_parts(date, Time::midnight());
        SpanRelativeTo { kind: SpanRelativeToKind::Civil(dt) }
    }
}

/// A bit set that keeps track of all non-zero units on a `Span`.
///
/// Because of alignment, adding this to a `Span` does not make it any bigger.
///
/// The benefit of this bit set is to make it extremely cheap to enable fast
/// paths in various places. For example, doing arithmetic on a `Date` with an
/// arbitrary `Span` is pretty involved. But if you know the `Span` only
/// consists of non-zero units of days (and zero for all other units), then you
/// can take a much cheaper path.
#[derive(Clone, Copy, Default)]
pub(crate) struct UnitSet(u16);

impl UnitSet {
    /// Return a bit set representing all units as zero.
    #[inline]
    const fn empty() -> UnitSet {
        UnitSet(0)
    }

    /// Set the given `unit` to `is_zero` status in this set.
    ///
    /// When `is_zero` is false, the unit is added to this set. Otherwise,
    /// the unit is removed from this set.
    #[inline]
    const fn set(self, unit: Unit, is_zero: bool) -> UnitSet {
        let bit = 1 << unit as usize;
        if is_zero {
            UnitSet(self.0 & !bit)
        } else {
            UnitSet(self.0 | bit)
        }
    }

    /// Returns the set constructed from the given slice of units.
    #[inline]
    pub(crate) const fn from_slice(units: &[Unit]) -> UnitSet {
        let mut set = UnitSet::empty();
        let mut i = 0;
        while i < units.len() {
            set = set.set(units[i], false);
            i += 1;
        }
        set
    }

    /// Returns true if and only if no units are in this set.
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.0 == 0
    }

    /// Returns true when this `Span` contains a non-zero value for the given
    /// unit.
    #[inline]
    pub(crate) fn contains(self, unit: Unit) -> bool {
        (self.0 & (1 << unit as usize)) != 0
    }

    /// Returns true if and only if this `Span` contains precisely one
    /// non-zero unit corresponding to the unit given.
    #[inline]
    pub(crate) fn contains_only(self, unit: Unit) -> bool {
        self.0 == (1 << unit as usize)
    }

    /// Returns this set, but with only calendar units.
    #[inline]
    pub(crate) fn only_calendar(self) -> UnitSet {
        UnitSet(self.0 & 0b0000_0011_1100_0000)
    }

    /// Returns this set, but with only time units.
    #[inline]
    pub(crate) fn only_time(self) -> UnitSet {
        UnitSet(self.0 & 0b0000_0000_0011_1111)
    }

    /// Returns the intersection of this set and the one given.
    #[inline]
    pub(crate) fn intersection(self, other: UnitSet) -> UnitSet {
        UnitSet(self.0 & other.0)
    }

    /// Returns the largest unit in this set, or `None` if none are present.
    #[inline]
    pub(crate) fn largest_unit(self) -> Option<Unit> {
        let zeros = usize::try_from(self.0.leading_zeros()).ok()?;
        15usize.checked_sub(zeros).and_then(Unit::from_usize)
    }
}

// N.B. This `Debug` impl isn't typically used.
//
// This is because the `Debug` impl for `Span` just emits itself in the
// friendly duration format, which doesn't include internal representation
// details like this set. It is included in `Span::debug`, but this isn't
// part of the public crate API.
impl core::fmt::Debug for UnitSet {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "{{")?;
        let mut units = *self;
        let mut i = 0;
        while let Some(unit) = units.largest_unit() {
            if i > 0 {
                write!(f, ", ")?;
            }
            i += 1;
            write!(f, "{}", unit.compact())?;
            units = units.set(unit, false);
        }
        if i == 0 {
            write!(f, "∅")?;
        }
        write!(f, "}}")
    }
}

/// An internal abstraction for managing a relative datetime for use in some
/// `Span` APIs.
///
/// This is effectively the same as a `SpanRelativeTo`, but uses a `Cow<Zoned>`
/// instead of a `&Zoned`. This makes it non-`Copy`, but allows us to craft a
/// more uniform API. (i.e., `relative + span = relative` instead of `relative
/// + span = owned_relative` or whatever.) Note that the `Copy` impl on
/// `SpanRelativeTo` means it has to accept a `&Zoned`. It can't ever take a
/// `Zoned` since it is non-Copy.
///
/// NOTE: Separately from above, I think it's plausible that this type could be
/// designed a bit differently. Namely, something like this:
///
/// ```text
/// struct Relative<'a> {
///     tz: Option<&'a TimeZone>,
///     dt: DateTime,
///     ts: Timestamp,
/// }
/// ```
///
/// That is, we do zone aware stuff but without an actual `Zoned` type. But I
/// think in order to make that work, we would need to expose most of the
/// `Zoned` API as functions on its component types (DateTime, Timestamp and
/// TimeZone). I think we are likely to want to do that for public API reasons,
/// but I'd like to resist it since I think it will add a lot of complexity.
/// Or maybe we need a `Unzoned` type that is `DateTime` and `Timestamp`, but
/// requires passing the time zone in to each of its methods. That might work
/// quite well, even if it was just an internal type.
///
/// Anyway, I'm not 100% sure the above would work, but I think it would. It
/// would be nicer because everything would be `Copy` all the time. We'd never
/// need a `Cow<TimeZone>` for example, because we never need to change or
/// create a new time zone.
#[derive(Clone, Debug)]
enum Relative<'a> {
    Civil(RelativeCivil),
    Zoned(RelativeZoned<'a>),
}

impl<'a> Relative<'a> {
    /// Adds the given span to this relative datetime.
    ///
    /// This defers to either [`DateTime::checked_add`] or
    /// [`Zoned::checked_add`], depending on the type of relative datetime.
    ///
    /// The `Relative` datetime returned is guaranteed to have the same
    /// internal datetie type as `self`.
    ///
    /// # Errors
    ///
    /// This returns an error in the same cases as the underlying checked
    /// arithmetic APIs. In general, this occurs when adding the given `span`
    /// would result in overflow.
    fn checked_add(&'a self, span: Span) -> Result<Relative<'a>, Error> {
        match *self {
            Relative::Civil(dt) => Ok(Relative::Civil(dt.checked_add(span)?)),
            Relative::Zoned(ref zdt) => {
                Ok(Relative::Zoned(zdt.checked_add(span)?))
            }
        }
    }

    fn checked_add_duration(
        &'a self,
        duration: SignedDuration,
    ) -> Result<Relative<'a>, Error> {
        match *self {
            Relative::Civil(dt) => {
                Ok(Relative::Civil(dt.checked_add_duration(duration)?))
            }
            Relative::Zoned(ref zdt) => {
                Ok(Relative::Zoned(zdt.checked_add_duration(duration)?))
            }
        }
    }

    /// Returns the span of time from this relative datetime to the one given,
    /// with units as large as `largest`.
    ///
    /// # Errors
    ///
    /// This returns an error in the same cases as when the underlying
    /// [`DateTime::until`] or [`Zoned::until`] fail. Because this doesn't
    /// set or expose any rounding configuration, this can generally only
    /// occur when `largest` is `Unit::Nanosecond` and the span of time
    /// between `self` and `other` is too big to represent as a 64-bit integer
    /// nanosecond count.
    ///
    /// # Panics
    ///
    /// This panics if `self` and `other` are different internal datetime
    /// types. For example, if `self` was a civil datetime and `other` were
    /// a zoned datetime.
    fn until(&self, largest: Unit, other: &Relative) -> Result<Span, Error> {
        match (self, other) {
            (&Relative::Civil(ref dt1), &Relative::Civil(ref dt2)) => {
                dt1.until(largest, dt2)
            }
            (&Relative::Zoned(ref zdt1), &Relative::Zoned(ref zdt2)) => {
                zdt1.until(largest, zdt2)
            }
            // This would be bad if `Relative` were a public API, but in
            // practice, this case never occurs because we don't mixup our
            // `Relative` datetime types.
            _ => unreachable!(),
        }
    }

    /// Converts this relative datetime to a nanosecond in UTC time.
    ///
    /// # Errors
    ///
    /// If there was a problem doing this conversion, then an error is
    /// returned. In practice, this only occurs for a civil datetime near the
    /// civil datetime minimum and maximum values.
    fn to_duration(&self) -> SignedDuration {
        match *self {
            Relative::Civil(dt) => dt.timestamp.as_duration(),
            Relative::Zoned(ref zdt) => zdt.zoned.timestamp().as_duration(),
        }
    }

    /// Create a balanced span of time relative to this datetime.
    ///
    /// The relative span returned has the same internal datetime type
    /// (civil or zoned) as this relative datetime.
    ///
    /// # Errors
    ///
    /// This returns an error when the span in this range cannot be
    /// represented. In general, this only occurs when asking for largest units
    /// of `Unit::Nanosecond` *and* when the span is too big to fit into a
    /// 64-bit nanosecond count.
    ///
    /// This can also return an error in other extreme cases, such as when
    /// adding the given span to this relative datetime results in overflow,
    /// or if this relative datetime is a civil datetime and it couldn't be
    /// converted to a timestamp in UTC.
    fn into_relative_span(
        self,
        largest: Unit,
        span: Span,
    ) -> Result<RelativeSpan<'a>, Error> {
        let kind = match self {
            Relative::Civil(start) => {
                let end = start.checked_add(span)?;
                RelativeSpanKind::Civil { start, end }
            }
            Relative::Zoned(start) => {
                let end = start.checked_add(span)?;
                RelativeSpanKind::Zoned { start, end }
            }
        };
        let relspan = kind.into_relative_span(largest)?;
        if !span.get_sign().is_zero()
            && !relspan.span.get_sign().is_zero()
            && span.get_sign() != relspan.span.get_sign()
        {
            // I haven't quite figured out when this case is hit. I think it's
            // actually impossible right? Balancing a duration should not flip
            // the sign.
            //
            // ref: https://github.com/fullcalendar/temporal-polyfill/blob/9e001042864394247181d1a5d591c18057ce32d2/packages/temporal-polyfill/src/internal/durationMath.ts#L236-L238
            unreachable!(
                "balanced span should have same sign as original span"
            )
        }
        Ok(relspan)
    }

    /// Rounds the given span using the given rounding configuration.
    fn round(
        self,
        span: Span,
        largest: Unit,
        increment: &Increment,
        mode: RoundMode,
    ) -> Result<Span, Error> {
        let relspan = self.into_relative_span(largest, span)?;
        if relspan.span.get_sign().is_zero() {
            return Ok(relspan.span);
        }
        let nudge = match relspan.kind {
            RelativeSpanKind::Civil { start, end } => {
                if increment.unit() > Unit::Day {
                    Nudge::relative_calendar(
                        relspan.span,
                        &Relative::Civil(start),
                        &Relative::Civil(end),
                        increment,
                        mode,
                    )?
                } else {
                    Nudge::relative_invariant(
                        relspan.span,
                        end.timestamp.as_duration(),
                        largest,
                        increment,
                        mode,
                    )?
                }
            }
            RelativeSpanKind::Zoned { ref start, ref end } => {
                if increment.unit() >= Unit::Day {
                    Nudge::relative_calendar(
                        relspan.span,
                        &Relative::Zoned(start.borrowed()),
                        &Relative::Zoned(end.borrowed()),
                        increment,
                        mode,
                    )?
                } else if largest >= Unit::Day {
                    // This is a special case for zoned datetimes when rounding
                    // could bleed into variable units.
                    Nudge::relative_zoned_time(
                        relspan.span,
                        start,
                        increment,
                        mode,
                    )?
                } else {
                    // Otherwise, rounding is the same as civil datetime.
                    Nudge::relative_invariant(
                        relspan.span,
                        end.zoned.timestamp().as_duration(),
                        largest,
                        increment,
                        mode,
                    )?
                }
            }
        };
        nudge.bubble(&relspan, increment.unit(), largest)
    }
}

/// A balanced span between a range of civil or zoned datetimes.
///
/// The span is always balanced up to a certain unit as given to
/// `RelativeSpanKind::into_relative_span`.
#[derive(Clone, Debug)]
struct RelativeSpan<'a> {
    span: Span,
    kind: RelativeSpanKind<'a>,
}

/// A civil or zoned datetime range of time.
#[derive(Clone, Debug)]
enum RelativeSpanKind<'a> {
    Civil { start: RelativeCivil, end: RelativeCivil },
    Zoned { start: RelativeZoned<'a>, end: RelativeZoned<'a> },
}

impl<'a> RelativeSpanKind<'a> {
    /// Create a balanced `RelativeSpan` from this range of time.
    ///
    /// # Errors
    ///
    /// This returns an error when the span in this range cannot be
    /// represented. In general, this only occurs when asking for largest units
    /// of `Unit::Nanosecond` *and* when the span is too big to fit into a
    /// 64-bit nanosecond count.
    fn into_relative_span(
        self,
        largest: Unit,
    ) -> Result<RelativeSpan<'a>, Error> {
        let span = match self {
            RelativeSpanKind::Civil { ref start, ref end } => start
                .datetime
                .until((largest, end.datetime))
                .context(E::FailedSpanBetweenDateTimes { unit: largest })?,
            RelativeSpanKind::Zoned { ref start, ref end } => {
                start.zoned.until((largest, &*end.zoned)).context(
                    E::FailedSpanBetweenZonedDateTimes { unit: largest },
                )?
            }
        };
        Ok(RelativeSpan { span, kind: self })
    }
}

/// A wrapper around a civil datetime and a timestamp corresponding to that
/// civil datetime in UTC.
///
/// Haphazardly interpreting a civil datetime in UTC is an odd and *usually*
/// incorrect thing to do. But the way we use it here is basically just to give
/// it an "anchoring" point such that we can represent it using a single
/// integer for rounding purposes. It is only used in a context *relative* to
/// another civil datetime interpreted in UTC. In this fashion, the selection
/// of UTC specifically doesn't really matter. We could use any time zone.
/// (Although, it must be a time zone without any transitions, otherwise we
/// could wind up with time zone aware results in a context where that would
/// be unexpected since this is civil time.)
#[derive(Clone, Copy, Debug)]
struct RelativeCivil {
    datetime: DateTime,
    timestamp: Timestamp,
}

impl RelativeCivil {
    /// Creates a new relative wrapper around the given civil datetime.
    ///
    /// This wrapper bundles a timestamp for the given datetime by interpreting
    /// it as being in UTC. This is an "odd" thing to do, but it's only used
    /// in the context of determining the length of time between two civil
    /// datetimes. So technically, any time zone without transitions could be
    /// used.
    ///
    /// # Errors
    ///
    /// This returns an error if the datetime could not be converted to a
    /// timestamp. This only occurs near the minimum and maximum civil datetime
    /// values.
    fn new(datetime: DateTime) -> Result<RelativeCivil, Error> {
        let timestamp = datetime
            .to_zoned(TimeZone::UTC)
            .context(E::ConvertDateTimeToTimestamp)?
            .timestamp();
        Ok(RelativeCivil { datetime, timestamp })
    }

    /// Returns the result of [`DateTime::checked_add`].
    ///
    /// # Errors
    ///
    /// Returns an error in the same cases as `DateTime::checked_add`. That is,
    /// when adding the span to this zoned datetime would overflow.
    ///
    /// This also returns an error if the resulting datetime could not be
    /// converted to a timestamp in UTC. This only occurs near the minimum and
    /// maximum datetime values.
    fn checked_add(&self, span: Span) -> Result<RelativeCivil, Error> {
        let datetime = self.datetime.checked_add(span)?;
        let timestamp = datetime
            .to_zoned(TimeZone::UTC)
            .context(E::ConvertDateTimeToTimestamp)?
            .timestamp();
        Ok(RelativeCivil { datetime, timestamp })
    }

    /// Returns the result of [`DateTime::checked_add`] with an absolute
    /// duration.
    ///
    /// # Errors
    ///
    /// Returns an error in the same cases as `DateTime::checked_add`. That is,
    /// when adding the span to this zoned datetime would overflow.
    ///
    /// This also returns an error if the resulting datetime could not be
    /// converted to a timestamp in UTC. This only occurs near the minimum and
    /// maximum datetime values.
    fn checked_add_duration(
        &self,
        duration: SignedDuration,
    ) -> Result<RelativeCivil, Error> {
        let datetime = self.datetime.checked_add(duration)?;
        let timestamp = datetime
            .to_zoned(TimeZone::UTC)
            .context(E::ConvertDateTimeToTimestamp)?
            .timestamp();
        Ok(RelativeCivil { datetime, timestamp })
    }

    /// Returns the result of [`DateTime::until`].
    ///
    /// # Errors
    ///
    /// Returns an error in the same cases as `DateTime::until`. That is, when
    /// the span for the given largest unit cannot be represented. This can
    /// generally only happen when `largest` is `Unit::Nanosecond` and the span
    /// cannot be represented as a 64-bit integer of nanoseconds.
    fn until(
        &self,
        largest: Unit,
        other: &RelativeCivil,
    ) -> Result<Span, Error> {
        self.datetime
            .until((largest, other.datetime))
            .context(E::FailedSpanBetweenDateTimes { unit: largest })
    }
}

/// A simple wrapper around a possibly borrowed `Zoned`.
#[derive(Clone, Debug)]
struct RelativeZoned<'a> {
    zoned: DumbCow<'a, Zoned>,
}

impl<'a> RelativeZoned<'a> {
    /// Returns the result of [`Zoned::checked_add`].
    ///
    /// # Errors
    ///
    /// Returns an error in the same cases as `Zoned::checked_add`. That is,
    /// when adding the span to this zoned datetime would overflow.
    fn checked_add(
        &self,
        span: Span,
    ) -> Result<RelativeZoned<'static>, Error> {
        let zoned = self.zoned.checked_add(span)?;
        Ok(RelativeZoned { zoned: DumbCow::Owned(zoned) })
    }

    /// Returns the result of [`Zoned::checked_add`] with an absolute duration.
    ///
    /// # Errors
    ///
    /// Returns an error in the same cases as `Zoned::checked_add`. That is,
    /// when adding the span to this zoned datetime would overflow.
    fn checked_add_duration(
        &self,
        duration: SignedDuration,
    ) -> Result<RelativeZoned<'static>, Error> {
        let zoned = self.zoned.checked_add(duration)?;
        Ok(RelativeZoned { zoned: DumbCow::Owned(zoned) })
    }

    /// Returns the result of [`Zoned::until`].
    ///
    /// # Errors
    ///
    /// Returns an error in the same cases as `Zoned::until`. That is, when
    /// the span for the given largest unit cannot be represented. This can
    /// generally only happen when `largest` is `Unit::Nanosecond` and the span
    /// cannot be represented as a 64-bit integer of nanoseconds.
    fn until(
        &self,
        largest: Unit,
        other: &RelativeZoned<'a>,
    ) -> Result<Span, Error> {
        self.zoned
            .until((largest, &*other.zoned))
            .context(E::FailedSpanBetweenZonedDateTimes { unit: largest })
    }

    /// Returns the borrowed version of self; useful when you need to convert
    /// `&RelativeZoned` into `RelativeZoned` without cloning anything.
    fn borrowed(&'a self) -> RelativeZoned<'a> {
        RelativeZoned { zoned: self.zoned.borrowed() }
    }
}

// The code below is the "core" rounding logic for spans. It was greatly
// inspired by this gist[1] and the fullcalendar Temporal polyfill[2]. In
// particular, the algorithm implemented below is a major simplification from
// how Temporal used to work[3]. Parts of it are still in rough and unclear
// shape IMO.
//
// [1]: https://gist.github.com/arshaw/36d3152c21482bcb78ea2c69591b20e0
// [2]: https://github.com/fullcalendar/temporal-polyfill
// [3]: https://github.com/tc39/proposal-temporal/issues/2792

/// The result of a span rounding strategy. There are three:
///
/// * Rounding spans relative to civil datetimes using only invariant
/// units (days or less). This is achieved by converting the span to a simple
/// integer number of nanoseconds and then rounding that.
/// * Rounding spans relative to either a civil datetime or a zoned datetime
/// where rounding might involve changing non-uniform units. That is, when
/// the smallest unit is greater than days for civil datetimes and greater
/// than hours for zoned datetimes.
/// * Rounding spans relative to a zoned datetime whose smallest unit is
/// less than days.
///
/// Each of these might produce a bottom heavy span that needs to be
/// re-balanced. This type represents that result via one of three constructors
/// corresponding to each of the above strategies, and then provides a routine
/// for rebalancing via "bubbling."
#[derive(Debug)]
struct Nudge {
    /// A possibly bottom heavy rounded span.
    span: Span,
    /// The nanosecond timestamp corresponding to `relative + span`, where
    /// `span` is the (possibly bottom heavy) rounded span.
    rounded_relative_end: SignedDuration,
    /// Whether rounding may have created a bottom heavy span such that a
    /// calendar unit might need to be incremented after re-balancing smaller
    /// units.
    grew_big_unit: bool,
}

impl Nudge {
    /// Performs rounding on the given span limited to invariant units.
    ///
    /// For civil datetimes, this means the smallest unit must be days or less,
    /// but the largest unit can be bigger. For zoned datetimes, this means
    /// that *both* the largest and smallest unit must be hours or less. This
    /// is because zoned datetimes with rounding that can spill up to days
    /// requires special handling.
    ///
    /// It works by converting the span to a single integer number of
    /// nanoseconds, rounding it and then converting back to a span.
    fn relative_invariant(
        balanced: Span,
        relative_end: SignedDuration,
        largest: Unit,
        increment: &Increment,
        mode: RoundMode,
    ) -> Result<Nudge, Error> {
        // Ensures this is only called when rounding invariant units.
        // Technically, it would be fine to allow weeks here, but this
        // code doesn't handle the special case of smallest==Week because
        // it just didn't account for it originally. But it probably should.
        // Then we could use this routine for rounding civil datetimes when
        // smallest==Week.
        assert!(increment.unit() <= Unit::Day);

        let sign = balanced.get_sign();
        let balanced_nanos = balanced.to_invariant_duration();
        let rounded_nanos = increment.round(mode, balanced_nanos)?;
        let span = Span::from_invariant_duration(largest, rounded_nanos)
            .context(E::ConvertNanoseconds { unit: largest })?
            .years(balanced.get_years())
            .months(balanced.get_months())
            .weeks(balanced.get_weeks());

        let diff_nanos = rounded_nanos - balanced_nanos;
        let diff_days =
            rounded_nanos.as_civil_days() - balanced_nanos.as_civil_days();
        let grew_big_unit = b::Sign::from(diff_days) == sign;
        let rounded_relative_end = relative_end + diff_nanos;
        Ok(Nudge { span, rounded_relative_end, grew_big_unit })
    }

    /// Performs rounding on the given span where the smallest unit configured
    /// implies that rounding will cover calendar or "non-uniform" units. (That
    /// is, units whose length can change based on the relative datetime.)
    fn relative_calendar(
        balanced: Span,
        relative_start: &Relative<'_>,
        relative_end: &Relative<'_>,
        increment: &Increment,
        mode: RoundMode,
    ) -> Result<Nudge, Error> {
        // This implementation is quite tricky and subtle. It is loosely based
        // on the fullcalendar polyfill for Temporal:
        // repo: https://github.com/fullcalendar/temporal-polyfill/
        // commit: bc1baf3875392ecd8522d40e7eecb55fa582808c
        // file: packages/temporal-polyfill/src/internal/round.ts
        // lines: L647-L711
        //
        // This diverges from the fullcalendar polyfill, however, by avoiding
        // the use of floating point. In part because almost all of Jiff avoids
        // floats (two exceptions being `Span::total` and
        // `SignedDuration::try_from_secs_{f32,f64}`). We used to use floats
        // here, and it drove be nuts because it was impossible to avoid in
        // this code path. And it also required duplicating our rounding code
        // just to handle this case.
        //
        // In any case, I stared at this code and fullcalendar's implementation
        // for quite some time, and I believe converted it over to sticking
        // with just integers. The key insight is moving everything to
        // nanoseconds and rounding there. I'm not quite sure why fullcalendar
        // doesn't do it this way. I think they are more tightly coupled with
        // Javascript's `number`, and so avoiding floats probably isn't a
        // priority?
        //
        // OK, so how does this work? The basic idea here is that, e.g., we
        // don't know long "1 month" or "1 year" is. (Or "1 day" in the case of
        // a zoned datetime.) So how do we know whether to e.g., round "1 month
        // 15 days" up to "2 month" or down to "1 month"? (For the `HalfExpand`
        // rounding mode.) So for this case, we need to compute what "1 month"
        // actually means in the context of our relative datetime (which we
        // must have if we're calling this routine). This is done by doing
        // some span arithemtic to calculate the length of time for one
        // increment's worth of the units we are rounding to.
        //
        // This approach is overall very similar to the approach we used for
        // rounding zoned datetimes. (Because the length of a day may vary.)

        // This bit is actually pretty important context: this code *only*
        // covers the case when the user asks for a smallest unit that is a
        // calendar unit. That means that the rounding we're doing is for a
        // unit that is potentially of varying length. If our relative datetime
        // is civil, then weeks/days won't be varying length, but the code
        // below still handles it. (Although this code won't be used when we
        // have a relative civil datetime and request a smallest unit of days.
        // That's because we can fall back to code assuming that days are an
        // invariant unit.)
        assert!(increment.unit() >= Unit::Day);

        let increment_units = i64::from(increment.value());
        let smallest = increment.unit();
        let sign = balanced.get_sign();
        // We want to measure the length of `increment`, which is done by
        // adding spans to `relative_start` with just one unit adjusted. The
        // `truncated` value is our starting point. The next will be
        // `truncated + (sign * increment)`.
        let truncated =
            increment_units * (balanced.get_unit(smallest) / increment_units);
        // Drop all units below smallest. We specifically don't want them and
        // here is where we no longer need them. `relative_end` still captures
        // how "close" we are between increments of `smallest`.
        let span =
            balanced.without_lower(smallest).try_unit(smallest, truncated)?;
        // This is OK because the increment value is guaranteed to be in the
        // range `1..=1_000_000_000`. Therefore, multiplying by {-1,0,1} is
        // always valid.
        //
        // The "amount" refers to the length of time (in units of `smallest`)
        // we want to measure. We don't actually know how long an `increment`
        // is. So we "measure" is by adding (or subtracting, depending on the
        // sign of the original span) to our relative datetime.
        let amount = sign * increment_units;
        // This is the "measurement" we mentioned above. We get back
        // `relative_start + span` and `relative_start + span + extra`,
        // where `extra` is `span` but with its `smallest` units set to
        // `amount`. Thus, `relative1 - relative0` corresponds to the length
        // in time, in nanoseconds, of `increment` units of `smallest`.
        let (relative0, relative1) =
            unit_start_and_end(relative_start, span, smallest, amount)?;

        // This corresponds to how far our original span gets us to the next
        // increment. That is, `relative_end = relative_start + original_span`.
        // (We actually don't have `original_span` here, but that's how
        // `relative_end` is computed. `balanced` is then computed from
        // `relative_start.until((largest_unit, relative_end))`.
        let progress_nanos = relative_end.to_duration() - relative0;
        // This is the length of `increment` units of `smallest`, but in units
        // of nanoseconds. The `.abs()` is OK because the difference in time,
        // even in nanoseconds, between -9999-01-01 and 9999-12-31 will never
        // be `SignedDuration::MIN`.
        let increment_nanos = (relative1 - relative0).abs();
        // Now we finally do the actual rounding: we round how much "progress"
        // we've made toward `relative1` by rounding `progress_nanos` to the
        // nearest increment value.
        //
        // The rounded nanoseconds returned can be greater than
        // `increment_nanos` when `smallest==Unit::Week` and the *original*
        // span had non-zero week units. This is because computing a balanced
        // span eliminates the week units, and so it is expected that
        // `relative_end` might be much bigger (or smaller, for negative spans)
        // than `relative1`.
        let rounded_nanos =
            mode.round_by_duration(progress_nanos, increment_nanos)?;
        // If we rounded up, then it's possible we might need to to re-balance
        // our span. (This happens in `bubble`.)
        let grew_big_unit =
            sign == b::Sign::from(rounded_nanos - progress_nanos);
        // These asserts check an assumption that, since we're dealing with
        // calendar units, and because time zone transitions never have
        // precision less than 1 second, it follows that the *length* of
        // the increment at nanosecond precision will never have non-zero
        // sub-seconds. This also guarantees that the result of rounding will
        // also never have non-zero sub-seconds (since the result of rounding
        // has to be a multiple of the increment).
        //
        // This is an important assumption to check, because we drop the
        // sub-second components on these durations in order to do division on
        // them below via 64-bit integers. (Otherwise we'd have to use 128-bit
        // integers.)
        debug_assert_eq!(rounded_nanos.subsec_nanos(), 0);
        debug_assert_eq!(increment_nanos.subsec_nanos(), 0);
        // Now we need to get back to our original units. We started with
        // `truncated`, so just add the number of units we covered via
        // rounding. We must multiply by `increment` because `rounded /
        // increment` gets us back the number of *increments* we rounded over.
        // But the actual number of units may be bigger.
        let span = span.try_unit(
            smallest,
            truncated
                + (increment_units
                    * (rounded_nanos.as_secs() / increment_nanos.as_secs())),
        )?;
        // If we rounded up, then the time we don't want to exceed is
        // `relative1`. Otherwise, we don't want to exceed `relative0`.
        // (This is used later in `bubble`.)
        let rounded_relative_end =
            if grew_big_unit { relative1 } else { relative0 };
        Ok(Nudge { span, rounded_relative_end, grew_big_unit })
    }

    /// Performs rounding on the given span where the smallest unit is hours
    /// or less *and* the relative datetime is time zone aware.
    fn relative_zoned_time(
        balanced: Span,
        relative_start: &RelativeZoned<'_>,
        increment: &Increment,
        mode: RoundMode,
    ) -> Result<Nudge, Error> {
        let sign = balanced.get_sign();
        let time_dur = balanced.only_lower(Unit::Day).to_invariant_duration();
        let mut rounded_time_nanos = increment.round(mode, time_dur)?;
        let (relative0, relative1) = unit_start_and_end(
            &Relative::Zoned(relative_start.borrowed()),
            balanced.without_lower(Unit::Day),
            Unit::Day,
            sign.as_i64(),
        )?;
        let day_nanos = relative1 - relative0;
        let beyond_day_nanos = rounded_time_nanos - day_nanos;

        let mut day_delta = 0;
        let rounded_relative_end = if beyond_day_nanos.is_zero()
            || b::Sign::from(beyond_day_nanos) == sign
        {
            day_delta += 1;
            rounded_time_nanos = increment.round(mode, beyond_day_nanos)?;
            relative1 + rounded_time_nanos
        } else {
            relative0 + rounded_time_nanos
        };

        let span =
            Span::from_invariant_duration(Unit::Hour, rounded_time_nanos)
                .context(E::ConvertNanoseconds { unit: Unit::Hour })?
                .years(balanced.get_years())
                .months(balanced.get_months())
                .weeks(balanced.get_weeks())
                .days(balanced.get_days() + day_delta);
        let grew_big_unit = day_delta != 0;
        Ok(Nudge { span, rounded_relative_end, grew_big_unit })
    }

    /// This "bubbles" up the units in a potentially "bottom heavy" span to
    /// larger units. For example, P1m50d relative to March 1 is bottom heavy.
    /// This routine will bubble the days up to months to get P2m19d.
    ///
    /// # Errors
    ///
    /// This routine fails if any arithmetic on the individual units fails, or
    /// when span arithmetic on the relative datetime given fails.
    fn bubble(
        &self,
        relative: &RelativeSpan,
        smallest: Unit,
        largest: Unit,
    ) -> Result<Span, Error> {
        if !self.grew_big_unit || smallest == Unit::Week {
            return Ok(self.span);
        }

        let smallest = smallest.max(Unit::Day);
        let mut balanced = self.span;
        let sign = balanced.get_sign();
        let mut unit = smallest;
        while let Some(u) = unit.next() {
            unit = u;
            if unit > largest {
                break;
            }
            // We only bubble smaller units up into weeks when the largest unit
            // is explicitly set to weeks. Otherwise, we leave it as-is.
            if unit == Unit::Week && largest != Unit::Week {
                continue;
            }

            let span_start = balanced.without_lower(unit);
            let new_units = span_start
                .get_unit(unit)
                .checked_add(sign.as_i64())
                .ok_or_else(|| unit.error())?;
            let span_end = span_start.try_unit(unit, new_units)?;
            let threshold = match relative.kind {
                RelativeSpanKind::Civil { ref start, .. } => {
                    start.checked_add(span_end)?.timestamp
                }
                RelativeSpanKind::Zoned { ref start, .. } => {
                    start.checked_add(span_end)?.zoned.timestamp()
                }
            };
            // If we overshoot our expected endpoint, then bail.
            let beyond = self.rounded_relative_end - threshold.as_duration();
            if beyond.is_zero() || b::Sign::from(beyond) == sign {
                balanced = span_end;
            } else {
                break;
            }
        }
        Ok(balanced)
    }
}

/// Rounds a span consisting of only invariant units.
///
/// This only applies when the max of the units in the span being rounded,
/// the largest configured unit and the smallest configured unit are all
/// invariant. That is, hours or lower for spans without a relative datetime,
/// or weeks or lower for spans with a `SpanRelativeTo::days_are_24_hours()`
/// marker.
///
/// All we do here is convert the span to an integer number of nanoseconds,
/// round that and then convert back. There aren't any tricky corner cases to
/// consider here.
fn round_span_invariant(
    span: Span,
    largest: Unit,
    increment: &Increment,
    mode: RoundMode,
) -> Result<Span, Error> {
    debug_assert!(increment.unit() <= Unit::Week);
    debug_assert!(largest <= Unit::Week);
    let dur = span.to_invariant_duration();
    let rounded = increment.round(mode, dur)?;
    Span::from_invariant_duration(largest, rounded)
        .context(E::ConvertNanoseconds { unit: largest })
}

/// Returns the nanosecond timestamps of `relative + span` and `relative +
/// {amount of unit} + span`.
///
/// This is useful for determining the actual length, in nanoseconds, of some
/// unit amount (usually a single unit). Usually, this is called with a span
/// whose units lower than `unit` are zeroed out and with an `amount` that
/// is `-1` or `1` or `0`. So for example, if `unit` were `Unit::Day`, then
/// you'd get back two nanosecond timestamps relative to the relative datetime
/// given that start exactly "one day" apart. (Which might be different than 24
/// hours, depending on the time zone.)
///
/// # Errors
///
/// This returns an error if adding the units overflows, or if doing the span
/// arithmetic on `relative` overflows.
fn unit_start_and_end(
    relative: &Relative<'_>,
    span: Span,
    unit: Unit,
    amount: i64,
) -> Result<(SignedDuration, SignedDuration), Error> {
    let amount =
        span.get_unit(unit).checked_add(amount).ok_or_else(|| unit.error())?;
    let span_amount = span.try_unit(unit, amount)?;
    let relative0 = relative.checked_add(span)?.to_duration();
    let relative1 = relative.checked_add(span_amount)?.to_duration();
    // This assertion gives better failure modes to what would otherwise be
    // subtle errors downstream if the durations returned here were equivalent.
    // It would imply that the physical time duration between them is zero,
    // and thus adding two spans---where one is strictly bigger/smaller than
    // the other---would produce identical timestamps.
    assert_ne!(
        relative0, relative1,
        "adding different spans should produce different timestamps"
    );
    Ok((relative0, relative1))
}

/// A common parsing function that works in bytes.
///
/// Specifically, this parses either an ISO 8601 duration into a `Span` or
/// a "friendly" duration into a `Span`. It also tries to give decent error
/// messages.
///
/// This works because the friendly and ISO 8601 formats have non-overlapping
/// prefixes. Both can start with a `+` or `-`, but aside from that, an ISO
/// 8601 duration _always_ has to start with a `P` or `p`. We can utilize this
/// property to very quickly determine how to parse the input. We just need to
/// handle the possibly ambiguous case with a leading sign a little carefully
/// in order to ensure good error messages.
///
/// (We do the same thing for `SignedDuration`.)
#[cfg_attr(feature = "perf-inline", inline(always))]
fn parse_iso_or_friendly(bytes: &[u8]) -> Result<Span, Error> {
    let Some((&byte, tail)) = bytes.split_first() else {
        return Err(crate::Error::from(
            crate::error::fmt::Error::HybridDurationEmpty,
        ));
    };
    let mut first = byte;
    // N.B. Unsigned durations don't support negative durations (of
    // course), but we still check for it here so that we can defer to
    // the dedicated parsers. They will provide their own error messages.
    if first == b'+' || first == b'-' {
        let Some(&byte) = tail.first() else {
            return Err(crate::Error::from(
                crate::error::fmt::Error::HybridDurationPrefix { sign: first },
            ));
        };
        first = byte;
    }
    if first == b'P' || first == b'p' {
        temporal::DEFAULT_SPAN_PARSER.parse_span(bytes)
    } else {
        friendly::DEFAULT_SPAN_PARSER.parse_span(bytes)
    }
}

fn requires_relative_date_err(unit: Unit) -> Result<(), Error> {
    if unit.is_variable() {
        return Err(Error::from(if matches!(unit, Unit::Week | Unit::Day) {
            UnitConfigError::RelativeWeekOrDay { unit }
        } else {
            UnitConfigError::RelativeYearOrMonth { unit }
        }));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use alloc::string::ToString;

    use crate::{civil::date, RoundMode};

    use super::*;

    #[test]
    fn test_total() {
        if crate::tz::db().is_definitively_empty() {
            return;
        }

        let span = 130.hours().minutes(20);
        let total = span.total(Unit::Second).unwrap();
        assert_eq!(total, 469200.0);

        let span = 123456789.seconds();
        let total = span
            .total(SpanTotal::from(Unit::Day).days_are_24_hours())
            .unwrap();
        assert_eq!(total, 1428.8980208333332);

        let span = 2756.hours();
        let dt = date(2020, 1, 1).at(0, 0, 0, 0);
        let zdt = dt.in_tz("Europe/Rome").unwrap();
        let total = span.total((Unit::Month, &zdt)).unwrap();
        assert_eq!(total, 3.7958333333333334);
        let total = span.total((Unit::Month, dt)).unwrap();
        assert_eq!(total, 3.7944444444444443);
    }

    #[test]
    fn test_compare() {
        if crate::tz::db().is_definitively_empty() {
            return;
        }

        let span1 = 79.hours().minutes(10);
        let span2 = 79.hours().seconds(630);
        let span3 = 78.hours().minutes(50);
        let mut array = [span1, span2, span3];
        array.sort_by(|sp1, sp2| sp1.compare(sp2).unwrap());
        assert_eq!(array, [span3, span1, span2].map(SpanFieldwise));

        let day24 = SpanRelativeTo::days_are_24_hours();
        let span1 = 79.hours().minutes(10);
        let span2 = 3.days().hours(7).seconds(630);
        let span3 = 3.days().hours(6).minutes(50);
        let mut array = [span1, span2, span3];
        array.sort_by(|sp1, sp2| sp1.compare((sp2, day24)).unwrap());
        assert_eq!(array, [span3, span1, span2].map(SpanFieldwise));

        let dt = date(2020, 11, 1).at(0, 0, 0, 0);
        let zdt = dt.in_tz("America/Los_Angeles").unwrap();
        array.sort_by(|sp1, sp2| sp1.compare((sp2, &zdt)).unwrap());
        assert_eq!(array, [span1, span3, span2].map(SpanFieldwise));
    }

    #[test]
    fn test_checked_add() {
        let span1 = 1.hour();
        let span2 = 30.minutes();
        let sum = span1.checked_add(span2).unwrap();
        span_eq!(sum, 1.hour().minutes(30));

        let span1 = 1.hour().minutes(30);
        let span2 = 2.hours().minutes(45);
        let sum = span1.checked_add(span2).unwrap();
        span_eq!(sum, 4.hours().minutes(15));

        let span = 50
            .years()
            .months(50)
            .days(50)
            .hours(50)
            .minutes(50)
            .seconds(50)
            .milliseconds(500)
            .microseconds(500)
            .nanoseconds(500);
        let relative = date(1900, 1, 1).at(0, 0, 0, 0);
        let sum = span.checked_add((span, relative)).unwrap();
        let expected = 108
            .years()
            .months(7)
            .days(12)
            .hours(5)
            .minutes(41)
            .seconds(41)
            .milliseconds(1)
            .microseconds(1)
            .nanoseconds(0);
        span_eq!(sum, expected);

        let span = 1.month().days(15);
        let relative = date(2000, 2, 1).at(0, 0, 0, 0);
        let sum = span.checked_add((span, relative)).unwrap();
        span_eq!(sum, 3.months());
        let relative = date(2000, 3, 1).at(0, 0, 0, 0);
        let sum = span.checked_add((span, relative)).unwrap();
        span_eq!(sum, 2.months().days(30));
    }

    #[test]
    fn test_round_day_time() {
        let span = 29.seconds();
        let rounded = span.round(Unit::Minute).unwrap();
        span_eq!(rounded, 0.minute());

        let span = 30.seconds();
        let rounded = span.round(Unit::Minute).unwrap();
        span_eq!(rounded, 1.minute());

        let span = 8.seconds();
        let rounded = span
            .round(
                SpanRound::new()
                    .smallest(Unit::Nanosecond)
                    .largest(Unit::Microsecond),
            )
            .unwrap();
        span_eq!(rounded, 8_000_000.microseconds());

        let span = 130.minutes();
        let rounded = span
            .round(SpanRound::new().largest(Unit::Day).days_are_24_hours())
            .unwrap();
        span_eq!(rounded, 2.hours().minutes(10));

        let span = 10.minutes().seconds(52);
        let rounded = span.round(Unit::Minute).unwrap();
        span_eq!(rounded, 11.minutes());

        let span = 10.minutes().seconds(52);
        let rounded = span
            .round(
                SpanRound::new().smallest(Unit::Minute).mode(RoundMode::Trunc),
            )
            .unwrap();
        span_eq!(rounded, 10.minutes());

        let span = 2.hours().minutes(34).seconds(18);
        let rounded =
            span.round(SpanRound::new().largest(Unit::Second)).unwrap();
        span_eq!(rounded, 9258.seconds());

        let span = 6.minutes();
        let rounded = span
            .round(
                SpanRound::new()
                    .smallest(Unit::Minute)
                    .increment(5)
                    .mode(RoundMode::Ceil),
            )
            .unwrap();
        span_eq!(rounded, 10.minutes());
    }

    #[test]
    fn test_round_relative_zoned_calendar() {
        if crate::tz::db().is_definitively_empty() {
            return;
        }

        let span = 2756.hours();
        let relative =
            date(2020, 1, 1).at(0, 0, 0, 0).in_tz("America/New_York").unwrap();
        let options = SpanRound::new()
            .largest(Unit::Year)
            .smallest(Unit::Day)
            .relative(&relative);
        let rounded = span.round(options).unwrap();
        span_eq!(rounded, 3.months().days(24));

        let span = 24.hours().nanoseconds(5);
        let relative = date(2000, 10, 29)
            .at(0, 0, 0, 0)
            .in_tz("America/Vancouver")
            .unwrap();
        let options = SpanRound::new()
            .largest(Unit::Day)
            .smallest(Unit::Minute)
            .relative(&relative)
            .mode(RoundMode::Expand)
            .increment(30);
        let rounded = span.round(options).unwrap();
        // It seems like this is the correct answer, although it apparently
        // differs from Temporal and the FullCalendar polyfill. I'm not sure
        // what accounts for the difference in the implementation.
        //
        // See: https://github.com/tc39/proposal-temporal/pull/2758#discussion_r1597255245
        span_eq!(rounded, 24.hours().minutes(30));

        // Ref: https://github.com/tc39/proposal-temporal/issues/2816#issuecomment-2115608460
        let span = -1.month().hours(24);
        let relative: crate::Zoned = date(2024, 4, 11)
            .at(2, 0, 0, 0)
            .in_tz("America/New_York")
            .unwrap();
        let options =
            SpanRound::new().smallest(Unit::Millisecond).relative(&relative);
        let rounded = span.round(options).unwrap();
        span_eq!(rounded, -1.month().days(1).hours(1));
        let dt = relative.checked_add(span).unwrap();
        let diff = relative.until((Unit::Month, &dt)).unwrap();
        span_eq!(diff, -1.month().days(1).hours(1));

        // Like the above, but don't use a datetime near a DST transition. In
        // this case, a day is a normal 24 hours. (Unlike above, where the
        // duration includes a 23 hour day, and so an additional hour has to be
        // added to the span to account for that.)
        let span = -1.month().hours(24);
        let relative = date(2024, 6, 11)
            .at(2, 0, 0, 0)
            .in_tz("America/New_York")
            .unwrap();
        let options =
            SpanRound::new().smallest(Unit::Millisecond).relative(&relative);
        let rounded = span.round(options).unwrap();
        span_eq!(rounded, -1.month().days(1));
    }

    #[test]
    fn test_round_relative_zoned_time() {
        if crate::tz::db().is_definitively_empty() {
            return;
        }

        let span = 2756.hours();
        let relative =
            date(2020, 1, 1).at(0, 0, 0, 0).in_tz("America/New_York").unwrap();
        let options = SpanRound::new().largest(Unit::Year).relative(&relative);
        let rounded = span.round(options).unwrap();
        span_eq!(rounded, 3.months().days(23).hours(21));

        let span = 2756.hours();
        let relative =
            date(2020, 9, 1).at(0, 0, 0, 0).in_tz("America/New_York").unwrap();
        let options = SpanRound::new().largest(Unit::Year).relative(&relative);
        let rounded = span.round(options).unwrap();
        span_eq!(rounded, 3.months().days(23).hours(19));

        let span = 3.hours();
        let relative =
            date(2020, 3, 8).at(0, 0, 0, 0).in_tz("America/New_York").unwrap();
        let options = SpanRound::new().largest(Unit::Year).relative(&relative);
        let rounded = span.round(options).unwrap();
        span_eq!(rounded, 3.hours());
    }

    #[test]
    fn test_round_relative_day_time() {
        let span = 2756.hours();
        let options =
            SpanRound::new().largest(Unit::Year).relative(date(2020, 1, 1));
        let rounded = span.round(options).unwrap();
        span_eq!(rounded, 3.months().days(23).hours(20));

        let span = 2756.hours();
        let options =
            SpanRound::new().largest(Unit::Year).relative(date(2020, 9, 1));
        let rounded = span.round(options).unwrap();
        span_eq!(rounded, 3.months().days(23).hours(20));

        let span = 190.days();
        let options =
            SpanRound::new().largest(Unit::Year).relative(date(2020, 1, 1));
        let rounded = span.round(options).unwrap();
        span_eq!(rounded, 6.months().days(8));

        let span = 30
            .days()
            .hours(23)
            .minutes(59)
            .seconds(59)
            .milliseconds(999)
            .microseconds(999)
            .nanoseconds(999);
        let options = SpanRound::new()
            .smallest(Unit::Microsecond)
            .largest(Unit::Year)
            .relative(date(2024, 5, 1));
        let rounded = span.round(options).unwrap();
        span_eq!(rounded, 1.month());

        let span = 364
            .days()
            .hours(23)
            .minutes(59)
            .seconds(59)
            .milliseconds(999)
            .microseconds(999)
            .nanoseconds(999);
        let options = SpanRound::new()
            .smallest(Unit::Microsecond)
            .largest(Unit::Year)
            .relative(date(2023, 1, 1));
        let rounded = span.round(options).unwrap();
        span_eq!(rounded, 1.year());

        let span = 365
            .days()
            .hours(23)
            .minutes(59)
            .seconds(59)
            .milliseconds(999)
            .microseconds(999)
            .nanoseconds(999);
        let options = SpanRound::new()
            .smallest(Unit::Microsecond)
            .largest(Unit::Year)
            .relative(date(2023, 1, 1));
        let rounded = span.round(options).unwrap();
        span_eq!(rounded, 1.year().days(1));

        let span = 365
            .days()
            .hours(23)
            .minutes(59)
            .seconds(59)
            .milliseconds(999)
            .microseconds(999)
            .nanoseconds(999);
        let options = SpanRound::new()
            .smallest(Unit::Microsecond)
            .largest(Unit::Year)
            .relative(date(2024, 1, 1));
        let rounded = span.round(options).unwrap();
        span_eq!(rounded, 1.year());

        let span = 3.hours();
        let options =
            SpanRound::new().largest(Unit::Year).relative(date(2020, 3, 8));
        let rounded = span.round(options).unwrap();
        span_eq!(rounded, 3.hours());
    }

    #[test]
    fn span_sign() {
        assert_eq!(Span::new().get_sign(), b::Sign::Zero);
        assert_eq!(Span::new().days(1).get_sign(), b::Sign::Positive);
        assert_eq!(Span::new().days(-1).get_sign(), b::Sign::Negative);
        assert_eq!(Span::new().days(1).days(0).get_sign(), b::Sign::Zero);
        assert_eq!(Span::new().days(-1).days(0).get_sign(), b::Sign::Zero);
        assert_eq!(
            Span::new().years(1).days(1).days(0).get_sign(),
            b::Sign::Positive,
        );
        assert_eq!(
            Span::new().years(-1).days(-1).days(0).get_sign(),
            b::Sign::Negative,
        );
    }

    #[test]
    fn span_size() {
        #[cfg(target_pointer_width = "64")]
        {
            #[cfg(debug_assertions)]
            {
                assert_eq!(core::mem::align_of::<Span>(), 8);
                assert_eq!(core::mem::size_of::<Span>(), 64);
            }
            #[cfg(not(debug_assertions))]
            {
                assert_eq!(core::mem::align_of::<Span>(), 8);
                assert_eq!(core::mem::size_of::<Span>(), 64);
            }
        }
    }

    quickcheck::quickcheck! {
        fn prop_roundtrip_span_nanoseconds(span: Span) -> quickcheck::TestResult {
            let largest = span.largest_unit();
            if largest > Unit::Day {
                return quickcheck::TestResult::discard();
            }
            let dur = span.to_invariant_duration();
            let got = Span::from_invariant_duration(largest, dur).unwrap();
            quickcheck::TestResult::from_bool(dur == got.to_invariant_duration())
        }
    }

    /// # `serde` deserializer compatibility test
    ///
    /// Serde YAML used to be unable to deserialize `jiff` types,
    /// as deserializing from bytes is not supported by the deserializer.
    ///
    /// - <https://github.com/BurntSushi/jiff/issues/138>
    /// - <https://github.com/BurntSushi/jiff/discussions/148>
    #[test]
    fn span_deserialize_yaml() {
        let expected = Span::new()
            .years(1)
            .months(2)
            .weeks(3)
            .days(4)
            .hours(5)
            .minutes(6)
            .seconds(7);

        let deserialized: Span =
            serde_yaml::from_str("P1y2m3w4dT5h6m7s").unwrap();

        span_eq!(deserialized, expected);

        let deserialized: Span =
            serde_yaml::from_slice("P1y2m3w4dT5h6m7s".as_bytes()).unwrap();

        span_eq!(deserialized, expected);

        let cursor = Cursor::new(b"P1y2m3w4dT5h6m7s");
        let deserialized: Span = serde_yaml::from_reader(cursor).unwrap();

        span_eq!(deserialized, expected);
    }

    #[test]
    fn display() {
        let span = Span::new()
            .years(1)
            .months(2)
            .weeks(3)
            .days(4)
            .hours(5)
            .minutes(6)
            .seconds(7)
            .milliseconds(8)
            .microseconds(9)
            .nanoseconds(10);
        insta::assert_snapshot!(
            span,
            @"P1Y2M3W4DT5H6M7.00800901S",
        );
        insta::assert_snapshot!(
            alloc::format!("{span:#}"),
            @"1y 2mo 3w 4d 5h 6m 7s 8ms 9µs 10ns",
        );
    }

    /// This test ensures that we can parse `humantime` formatted durations.
    #[test]
    fn humantime_compatibility_parse() {
        let dur = std::time::Duration::new(60 * 60 * 24 * 411, 123_456_789);
        let formatted = humantime::format_duration(dur).to_string();
        assert_eq!(
            formatted,
            "1year 1month 15days 7h 26m 24s 123ms 456us 789ns"
        );
        let expected = 1
            .year()
            .months(1)
            .days(15)
            .hours(7)
            .minutes(26)
            .seconds(24)
            .milliseconds(123)
            .microseconds(456)
            .nanoseconds(789);
        span_eq!(formatted.parse::<Span>().unwrap(), expected);
    }

    /// This test ensures that we can print a `Span` that `humantime` can
    /// parse.
    ///
    /// Note that this isn't the default since `humantime`'s parser is
    /// pretty limited. e.g., It doesn't support things like `nsecs`
    /// despite supporting `secs`. And other reasons. See the docs on
    /// `Designator::HumanTime` for why we sadly provide a custom variant for
    /// it.
    #[test]
    fn humantime_compatibility_print() {
        static PRINTER: friendly::SpanPrinter = friendly::SpanPrinter::new()
            .designator(friendly::Designator::HumanTime);

        let span = 1
            .year()
            .months(1)
            .days(15)
            .hours(7)
            .minutes(26)
            .seconds(24)
            .milliseconds(123)
            .microseconds(456)
            .nanoseconds(789);
        let formatted = PRINTER.span_to_string(&span);
        assert_eq!(formatted, "1y 1month 15d 7h 26m 24s 123ms 456us 789ns");

        let dur = humantime::parse_duration(&formatted).unwrap();
        let expected =
            std::time::Duration::new(60 * 60 * 24 * 411, 123_456_789);
        assert_eq!(dur, expected);
    }

    #[test]
    fn from_str() {
        let p = |s: &str| -> Result<Span, Error> { s.parse() };

        insta::assert_snapshot!(
            p("1 day").unwrap(),
            @"P1D",
        );
        insta::assert_snapshot!(
            p("+1 day").unwrap(),
            @"P1D",
        );
        insta::assert_snapshot!(
            p("-1 day").unwrap(),
            @"-P1D",
        );
        insta::assert_snapshot!(
            p("P1d").unwrap(),
            @"P1D",
        );
        insta::assert_snapshot!(
            p("+P1d").unwrap(),
            @"P1D",
        );
        insta::assert_snapshot!(
            p("-P1d").unwrap(),
            @"-P1D",
        );

        insta::assert_snapshot!(
            p("").unwrap_err(),
            @r#"an empty string is not a valid duration in either the ISO 8601 format or Jiff's "friendly" format"#,
        );
        insta::assert_snapshot!(
            p("+").unwrap_err(),
            @r#"found nothing after sign `+`, which is not a valid duration in either the ISO 8601 format or Jiff's "friendly" format"#,
        );
        insta::assert_snapshot!(
            p("-").unwrap_err(),
            @r#"found nothing after sign `-`, which is not a valid duration in either the ISO 8601 format or Jiff's "friendly" format"#,
        );
    }

    #[test]
    fn serde_deserialize() {
        let p = |s: &str| -> Result<Span, serde_json::Error> {
            serde_json::from_str(&alloc::format!("\"{s}\""))
        };

        insta::assert_snapshot!(
            p("1 day").unwrap(),
            @"P1D",
        );
        insta::assert_snapshot!(
            p("+1 day").unwrap(),
            @"P1D",
        );
        insta::assert_snapshot!(
            p("-1 day").unwrap(),
            @"-P1D",
        );
        insta::assert_snapshot!(
            p("P1d").unwrap(),
            @"P1D",
        );
        insta::assert_snapshot!(
            p("+P1d").unwrap(),
            @"P1D",
        );
        insta::assert_snapshot!(
            p("-P1d").unwrap(),
            @"-P1D",
        );

        insta::assert_snapshot!(
            p("").unwrap_err(),
            @r#"an empty string is not a valid duration in either the ISO 8601 format or Jiff's "friendly" format at line 1 column 2"#,
        );
        insta::assert_snapshot!(
            p("+").unwrap_err(),
            @r#"found nothing after sign `+`, which is not a valid duration in either the ISO 8601 format or Jiff's "friendly" format at line 1 column 3"#,
        );
        insta::assert_snapshot!(
            p("-").unwrap_err(),
            @r#"found nothing after sign `-`, which is not a valid duration in either the ISO 8601 format or Jiff's "friendly" format at line 1 column 3"#,
        );
    }

    // This ensures that adding maximum invariant durations doesn't overflow.
    #[test]
    fn maximum_invariant_duration() {
        let span = Span::new()
            .weeks(b::SpanWeeks::MAX)
            .days(b::SpanDays::MAX)
            .hours(b::SpanHours::MAX)
            .minutes(b::SpanMinutes::MAX)
            .seconds(b::SpanSeconds::MAX)
            .milliseconds(b::SpanMilliseconds::MAX)
            .microseconds(b::SpanMicroseconds::MAX)
            .nanoseconds(b::SpanNanoseconds::MAX);

        let dur = span.to_invariant_duration();
        assert_eq!(dur.as_secs(), 4_426_974_863_236);
        assert_eq!(
            dur,
            // 1229715239h 47m 16s 854ms 775µs 807ns
            SignedDuration::new(
                1_229_715_239 * 60 * 60 + 47 * 60 + 16,
                854_775_807
            ),
        );

        let sum = dur + dur;
        assert_eq!(sum.as_secs(), 8_853_949_726_473);
        assert_eq!(
            sum,
            // 2459430479h 34m 33s 709ms 551µs 614ns
            SignedDuration::new(
                2_459_430_479 * 60 * 60 + 34 * 60 + 33,
                709_551_614,
            ),
        );
    }

    // This ensures that adding minimum invariant durations doesn't overflow.
    #[test]
    fn minimum_invariant_duration() {
        let span = Span::new()
            .weeks(b::SpanWeeks::MIN)
            .days(b::SpanDays::MIN)
            .hours(b::SpanHours::MIN)
            .minutes(b::SpanMinutes::MIN)
            .seconds(b::SpanSeconds::MIN)
            .milliseconds(b::SpanMilliseconds::MIN)
            .microseconds(b::SpanMicroseconds::MIN)
            .nanoseconds(b::SpanNanoseconds::MIN);

        let dur = span.to_invariant_duration();
        assert_eq!(dur.as_secs(), -4_426_974_863_236);
        assert_eq!(
            dur,
            // -1229715239h 47m 16s 854ms 775µs 807ns
            -SignedDuration::new(
                1_229_715_239 * 60 * 60 + 47 * 60 + 16,
                854_775_807
            ),
        );

        let sum = dur + dur;
        assert_eq!(sum.as_secs(), -8_853_949_726_473);
        assert_eq!(
            sum,
            // -2459430479h 34m 33s 709ms 551µs 614ns
            -SignedDuration::new(
                2_459_430_479 * 60 * 60 + 34 * 60 + 33,
                709_551_614,
            ),
        );
    }
}
