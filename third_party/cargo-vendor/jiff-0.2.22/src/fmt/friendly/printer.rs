use crate::{
    fmt::{
        buffer::{ArrayBuffer, BorrowedBuffer},
        Write,
    },
    Error, SignedDuration, Span, Unit,
};

const SECS_PER_HOUR: u64 = MINS_PER_HOUR * SECS_PER_MIN;
const SECS_PER_MIN: u64 = 60;
const MINS_PER_HOUR: u64 = 60;
const NANOS_PER_HOUR: u128 =
    (SECS_PER_MIN * MINS_PER_HOUR * NANOS_PER_SEC) as u128;
const NANOS_PER_MIN: u128 = (SECS_PER_MIN * NANOS_PER_SEC) as u128;
const NANOS_PER_SEC: u64 = 1_000_000_000;
const NANOS_PER_MILLI: u32 = 1_000_000;
const NANOS_PER_MICRO: u32 = 1_000;

/// Defines the maximum possible length (in bytes) of a `Span` printed in the
/// friendly format.
///
/// The way I computed this length was by using settings that would produce the
/// maximal amount of text and using a negative `Span` with each unit set to
/// its minimum value:
///
/// ```ignore
/// SpanPrinter::new()
///     .padding(u8::MAX)
///     .designator(friendly::Designator::Verbose)
///     .spacing(friendly::Spacing::BetweenUnitsAndDesignators)
///     .comma_after_designator(true);
/// ```
const MAX_SPAN_LEN: usize = 306;

/// Defines the maximum possible length (in bytes) of a `SignedDuration`
/// printed in the friendly format.
///
/// See `MAX_SPAN_LEN` for how this was computed. In this case,
/// `SignedDuration::MIN` was used.
const MAX_SIGNED_DURATION_LEN: usize = 194;

/// Defines the maximum possible length (in bytes) of a `std::time::Duration`
/// printed in the friendly format.
///
/// See `MAX_SPAN_LEN` for how this was computed. In this case,
/// `std::time::Duration::MAX - std::time::Duration::from_secs(16)` was used
/// to find the maximal length. This was a little tricky because the maximal
/// unsigned duration has minute units equivalent to `0`, and is thus quite a
/// bit shorter than the maximum possible length.
///
/// Note that this is shorter than `MAX_SIGNED_DURATION` because one cannot
/// get a negative formatted duration from a `std::time::Duration`. So there's
/// no way to get an ` ago` suffix (or a `-` prefix).
const MAX_UNSIGNED_DURATION_LEN: usize = 190;

/// Configuration for [`SpanPrinter::designator`].
///
/// This controls which kinds of designators to use when formatting a
/// "friendly" duration. Generally, this only provides one axis of control:
/// the length of each designator.
///
/// # Example
///
/// ```
/// use jiff::{fmt::friendly::{Designator, SpanPrinter}, ToSpan};
///
/// let span = 1.year().months(2);
///
/// let printer = SpanPrinter::new();
/// assert_eq!(printer.span_to_string(&span), "1y 2mo");
///
/// let printer = SpanPrinter::new().designator(Designator::Short);
/// assert_eq!(printer.span_to_string(&span), "1yr 2mos");
///
/// let printer = SpanPrinter::new().designator(Designator::Verbose);
/// assert_eq!(printer.span_to_string(&span), "1year 2months");
///
/// let printer = SpanPrinter::new().designator(Designator::HumanTime);
/// assert_eq!(printer.span_to_string(&span), "1y 2months");
/// ```
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Designator {
    /// This writes out the full word of each unit designation. For example,
    /// `year`.
    Verbose,
    /// This writes out a short but not minimal label for each unit. For
    /// example, `yr` for `year` and `yrs` for `years`.
    Short,
    /// This writes out the shortest possible label for each unit that is still
    /// generally recognizable. For example, `y`. Note that in the compact
    /// representation, and unlike the verbose and short representations, there
    /// is no distinction between singular or plural.
    Compact,
    /// A special mode that uses designator labels that are known to be
    /// compatible with the `humantime` crate.
    ///
    /// None of `Verbose`, `Short` or `Compact` are compatible with
    /// `humantime`.
    ///
    /// `Compact` is, on its own, nearly compatible. When using `Compact`, all
    /// designator labels are parsable by `humantime` except for months and
    /// microseconds. For months, Jiff uses `mo` and `mos`, but `humantime`
    /// only parses `months`, `month` and `M`. Jiff specifically doesn't
    /// support `M` for months because of the confusability with minutes.
    /// For microseconds, Jiff uses `µs` which `humantime` does not support
    /// parsing.
    ///
    /// Most of the designator labels Jiff uses for `Short` aren't supported
    /// by `humantime`. And even when they are, `humantime` is inconsistent.
    /// For example, `humantime` supports `sec` and `secs`, but only `nsec`
    /// and not `nsecs`.
    ///
    /// Finally, for `Verbose`, humantime supports spelling out some units
    /// in their entirety (e.g., `seconds`) but not others (e.g., `nanoseconds`
    /// is not supported by `humantime`).
    ///
    /// Therefore, this custom variant is provided so that designator labels
    /// that are compatible with both Jiff and `humantime`, even when there
    /// isn't a coherent concept otherwise connecting their style.
    HumanTime,
}

/// Configuration for [`SpanPrinter::spacing`].
///
/// This controls how much or how little whitespace is inserted into a
/// "friendly" formatted duration. Typically, one wants less whitespace when
/// using short unit designators (i.e., `y` instead of `years`), and more
/// whitespace when using longer unit designators.
///
/// # Example
///
/// ```
/// use jiff::{
///     fmt::friendly::{Designator, Spacing, SpanPrinter},
///     ToSpan,
/// };
///
/// let span = 1.year().months(2);
///
/// // The default tries to balance spacing with compact
/// // unit designators.
/// let printer = SpanPrinter::new();
/// assert_eq!(printer.span_to_string(&span), "1y 2mo");
///
/// // But you can use slightly more descriptive
/// // designators without being too verbose.
/// let printer = SpanPrinter::new()
///     .designator(Designator::Short);
/// assert_eq!(printer.span_to_string(&span), "1yr 2mos");
///
/// // When spacing is removed, it usually looks nicer
/// // to use compact unit designators.
/// let printer = SpanPrinter::new()
///     .spacing(Spacing::None)
///     .designator(Designator::Compact);
/// assert_eq!(printer.span_to_string(&span), "1y2mo");
///
/// // Conversely, when using more spacing, it usually
/// // looks nicer to use verbose unit designators.
/// let printer = SpanPrinter::new()
///     .spacing(Spacing::BetweenUnitsAndDesignators)
///     .designator(Designator::Verbose);
/// assert_eq!(printer.span_to_string(&span), "1 year 2 months");
/// ```
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Spacing {
    /// Does not insert any ASCII whitespace.
    ///
    /// Except in the case that [`SpanPrinter::hours_minutes_seconds`] is
    /// enabled and one is formatting a span with non-zero calendar units, then
    /// an ASCII whitespace is inserted between the calendar and non-calendar
    /// units even when `Spacing::None` is used.
    None,
    /// Inserts one ASCII whitespace between the unit designator and the next
    /// unit value.
    ///
    /// For example, `1year 2months`.
    BetweenUnits,
    /// Inserts one ASCII whitespace between the unit value and the unit
    /// designator, in addition to inserting one ASCII whitespace between the
    /// unit designator and the next unit value.
    ///
    /// For example, `1 year 2 months`.
    BetweenUnitsAndDesignators,
}

impl Spacing {
    fn between_units(self) -> Option<u8> {
        match self {
            Spacing::None => None,
            Spacing::BetweenUnits => Some(b' '),
            Spacing::BetweenUnitsAndDesignators => Some(b' '),
        }
    }

    fn between_units_and_designators(self) -> Option<u8> {
        match self {
            Spacing::None => None,
            Spacing::BetweenUnits => None,
            Spacing::BetweenUnitsAndDesignators => Some(b' '),
        }
    }
}

/// Configuration for [`SpanPrinter::direction`].
///
/// This controls how the sign, if at all, is included in the formatted
/// duration.
///
/// When using the "hours-minutes-seconds" format, `Auto` and `Suffix` are
/// both treated as equivalent to `Sign` when all calendar units (days and
/// greater) are zero.
///
/// # Example
///
/// ```
/// use jiff::{fmt::friendly::{Direction, SpanPrinter}, SignedDuration};
///
/// let duration = SignedDuration::from_secs(-1);
///
/// let printer = SpanPrinter::new();
/// assert_eq!(printer.duration_to_string(&duration), "1s ago");
///
/// let printer = SpanPrinter::new().direction(Direction::Sign);
/// assert_eq!(printer.duration_to_string(&duration), "-1s");
/// ```
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Direction {
    /// Sets the sign format based on other configuration options.
    ///
    /// When [`SpanPrinter::spacing`] is set to [`Spacing::None`], then
    /// `Auto` is equivalent to `Sign`.
    ///
    /// When using the "hours-minutes-seconds" format, `Auto` is equivalent to
    /// `Sign` when all calendar units (days and greater) are zero.
    ///
    /// Otherwise, `Auto` is equivalent to `Suffix`.
    ///
    /// This is the default used by [`SpanPrinter`].
    Auto,
    /// When set, a sign is only written when the span or duration is negative.
    /// And when it is written, it is written as a prefix of the formatted
    /// duration.
    Sign,
    /// A sign is always written, with `-` for negative spans and `+` for all
    /// non-negative spans. The sign is always written as a prefix of the
    /// formatted duration.
    ForceSign,
    /// When set, a sign is only written when the span or duration is negative.
    /// And when it is written, it is written as a suffix via a trailing ` ago`
    /// string.
    Suffix,
}

impl Direction {
    /// Returns the sign string to use (as either a prefix or a suffix) based
    /// on the given parameters.
    ///
    /// This lets us do the case analysis for how to write the sign exactly
    /// once.
    fn sign(
        self,
        printer: &SpanPrinter,
        has_calendar: bool,
        signum: i8,
    ) -> Option<DirectionSign> {
        match self {
            Direction::Auto => match printer.spacing {
                Spacing::None => {
                    if signum < 0 {
                        Some(DirectionSign::Prefix(b'-'))
                    } else {
                        None
                    }
                }
                Spacing::BetweenUnits
                | Spacing::BetweenUnitsAndDesignators => {
                    if signum < 0 {
                        if printer.hms && !has_calendar {
                            Some(DirectionSign::Prefix(b'-'))
                        } else {
                            Some(DirectionSign::Suffix)
                        }
                    } else {
                        None
                    }
                }
            },
            Direction::Sign => {
                if signum < 0 {
                    Some(DirectionSign::Prefix(b'-'))
                } else {
                    None
                }
            }
            Direction::ForceSign => {
                Some(DirectionSign::Prefix(if signum < 0 {
                    b'-'
                } else {
                    b'+'
                }))
            }
            Direction::Suffix => {
                if signum < 0 {
                    Some(DirectionSign::Suffix)
                } else {
                    None
                }
            }
        }
    }
}

/// The sign to write and whether it should be a prefix or a suffix.
#[derive(Clone, Copy, Debug)]
enum DirectionSign {
    Prefix(u8),
    Suffix,
}

/// Configuration for [`SpanPrinter::fractional`].
///
/// This controls what kind of fractional unit to use when printing a duration.
/// The default, unless [`SpanPrinter::hours_minutes_seconds`] is enabled, is
/// to not write any fractional numbers at all.
///
/// The fractional unit set refers to the smallest whole integer that can occur
/// in a "friendly" formatted duration. If there are any non-zero units less
/// than the fractional unit in the duration, then they are formatted as a
/// fraction.
///
/// # Example
///
/// This example shows how to write the same duration with different
/// fractional settings:
///
/// ```
/// use jiff::{fmt::friendly::{FractionalUnit, SpanPrinter}, SignedDuration};
///
/// let duration = SignedDuration::from_secs(3663);
///
/// let printer = SpanPrinter::new()
///     .fractional(Some(FractionalUnit::Hour));
/// assert_eq!(printer.duration_to_string(&duration), "1.0175h");
///
/// let printer = SpanPrinter::new()
///     .fractional(Some(FractionalUnit::Minute));
/// assert_eq!(printer.duration_to_string(&duration), "1h 1.05m");
///
/// let printer = SpanPrinter::new()
///     .fractional(Some(FractionalUnit::Second));
/// assert_eq!(printer.duration_to_string(&duration), "1h 1m 3s");
/// ```
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum FractionalUnit {
    /// The smallest whole integer unit allowed is hours.
    ///
    /// **WARNING**: Since fractional units are limited to 9 decimal places,
    /// using this setting could result in precision loss.
    Hour,
    /// The smallest whole integer unit allowed is minutes.
    ///
    /// **WARNING**: Since fractional units are limited to 9 decimal places,
    /// using this setting could result in precision loss.
    Minute,
    /// The smallest whole integer unit allowed is seconds.
    Second,
    /// The smallest whole integer unit allowed is milliseconds.
    Millisecond,
    /// The smallest whole integer unit allowed is microseconds.
    Microsecond,
}

impl From<FractionalUnit> for Unit {
    fn from(u: FractionalUnit) -> Unit {
        match u {
            FractionalUnit::Hour => Unit::Hour,
            FractionalUnit::Minute => Unit::Minute,
            FractionalUnit::Second => Unit::Second,
            FractionalUnit::Millisecond => Unit::Millisecond,
            FractionalUnit::Microsecond => Unit::Microsecond,
        }
    }
}

/// A printer for Jiff's "friendly" duration format.
///
/// This printer provides a lot of different knobs for controlling how
/// durations are formatted. It supports formatting both [`SignedDuration`]
/// and [`Span`].
///
/// # Example: automatic use through `Display`
///
/// The default configuration of this printer is used for "alternate" display
/// formatting for both [`SignedDuration`] and [`Span`]:
///
/// ```
/// use jiff::{SignedDuration, ToSpan};
///
/// let span = 1.year().months(2).hours(15).seconds(30).nanoseconds(1);
/// assert_eq!(format!("{span:#}"), "1y 2mo 15h 30s 1ns");
///
/// let sdur = SignedDuration::new(15 * 60 * 60 + 30, 1);
/// assert_eq!(format!("{sdur:#}"), "15h 30s 1ns");
/// ```
///
/// # Example: variety of formatting configurations
///
/// This example shows a few different ways of formatting the same `Span`:
///
/// ```
/// use jiff::{
///     fmt::friendly::{Designator, Spacing, SpanPrinter},
///     ToSpan,
/// };
///
/// let span = 1.year().months(2).hours(15).seconds(30).nanoseconds(1);
///
/// let printer = SpanPrinter::new();
/// assert_eq!(
///     printer.span_to_string(&span),
///     "1y 2mo 15h 30s 1ns",
/// );
///
/// let printer = SpanPrinter::new()
///     .designator(Designator::Short);
/// assert_eq!(
///     printer.span_to_string(&span),
///     "1yr 2mos 15hrs 30secs 1nsec",
/// );
///
/// let printer = SpanPrinter::new()
///     .spacing(Spacing::None)
///     .designator(Designator::Compact);
/// assert_eq!(
///     printer.span_to_string(&span),
///     "1y2mo15h30s1ns",
/// );
///
/// let printer = SpanPrinter::new()
///     .spacing(Spacing::BetweenUnitsAndDesignators)
///     .comma_after_designator(true)
///     .designator(Designator::Verbose);
/// assert_eq!(
///     printer.span_to_string(&span),
///     "1 year, 2 months, 15 hours, 30 seconds, 1 nanosecond",
/// );
///
/// let printer = SpanPrinter::new()
///     .hours_minutes_seconds(true)
///     .spacing(Spacing::BetweenUnitsAndDesignators)
///     .comma_after_designator(true)
///     .designator(Designator::Verbose);
/// assert_eq!(
///     printer.span_to_string(&span),
///     "1 year, 2 months, 15:00:30.000000001",
/// );
/// ```
///
/// # Example: negative durations
///
/// By default, a negative duration will be represented with an ` ago` suffix:
///
/// ```
/// use jiff::{fmt::friendly::SpanPrinter, ToSpan};
///
/// let span = -1.year().months(2).hours(15).seconds(30).nanoseconds(1);
///
/// let printer = SpanPrinter::new();
/// assert_eq!(
///     printer.span_to_string(&span),
///     "1y 2mo 15h 30s 1ns ago",
/// );
/// ```
///
/// But one can also use a prefix `-` sign instead. Usually this works better
/// without any spacing and compact designators:
///
/// ```
/// use jiff::{fmt::friendly::{Designator, Spacing, SpanPrinter}, ToSpan};
///
/// let span = -1.year().months(2).hours(15).seconds(30).nanoseconds(1);
///
/// let printer = SpanPrinter::new()
///     .spacing(Spacing::None)
///     .designator(Designator::Compact);
/// assert_eq!(
///     printer.span_to_string(&span),
///     "-1y2mo15h30s1ns",
/// );
/// ```
#[derive(Clone, Debug)]
pub struct SpanPrinter {
    designators: &'static Designators,
    spacing: Spacing,
    direction: Direction,
    fractional: Option<FractionalUnit>,
    comma_after_designator: bool,
    hms: bool,
    padding: Option<u8>,
    precision: Option<u8>,
    zero_unit: Unit,
}

impl SpanPrinter {
    /// Creates a new printer for the "friendly" duration format.
    ///
    /// The printer returned uses the default configuration. This is
    /// identical to `SpanPrinter::default`, but it can be used in a `const`
    /// context.
    ///
    /// # Example
    ///
    /// This example shows how to format a duration directly to a `Vec<u8>`.
    ///
    /// ```
    /// use jiff::{fmt::friendly::SpanPrinter, ToSpan};
    ///
    /// static PRINTER: SpanPrinter = SpanPrinter::new();
    ///
    /// let span = 1.year().months(2);
    /// let mut buf = vec![];
    /// // Writing to a `Vec<u8>` never fails (aside from OOM).
    /// PRINTER.print_span(&span, &mut buf).unwrap();
    /// assert_eq!(buf, b"1y 2mo");
    /// ```
    #[inline]
    pub const fn new() -> SpanPrinter {
        SpanPrinter {
            designators: Designators::new(Designator::Compact),
            spacing: Spacing::BetweenUnits,
            direction: Direction::Auto,
            fractional: None,
            comma_after_designator: false,
            hms: false,
            padding: None,
            precision: None,
            zero_unit: Unit::Second,
        }
    }

    /// Configures the kind of unit designators to use.
    ///
    /// There are no specific advantages or disadvantages to the kind
    /// of designator you pick other than aesthetic preference. Shorter
    /// designators are also likely faster to parse and print.
    ///
    /// The default is [`Designator::Compact`], which uses things like `yr`
    /// instead of `year` (verbose) or `y` (compact).
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::{
    ///     fmt::friendly::{Designator, SpanPrinter},
    ///     ToSpan,
    /// };
    ///
    /// let span = 1.year().months(2);
    ///
    /// let printer = SpanPrinter::new();
    /// assert_eq!(printer.span_to_string(&span), "1y 2mo");
    ///
    /// let printer = SpanPrinter::new().designator(Designator::Short);
    /// assert_eq!(printer.span_to_string(&span), "1yr 2mos");
    ///
    /// let printer = SpanPrinter::new().designator(Designator::Verbose);
    /// assert_eq!(printer.span_to_string(&span), "1year 2months");
    /// ```
    #[inline]
    pub const fn designator(self, designator: Designator) -> SpanPrinter {
        SpanPrinter { designators: Designators::new(designator), ..self }
    }

    /// Configures the spacing between the units and the designator labels.
    ///
    /// The default is [`Spacing::BetweenUnits`], which results in durations
    /// like `1y 2mo`. `Spacing::None` would result in `1y2mo` and
    /// `Spacing::BetweenUnitsAndDesignators` would result in `1 y 2 mo`.
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::{
    ///     fmt::friendly::{Designator, Spacing, SpanPrinter},
    ///     ToSpan,
    /// };
    ///
    /// let span = 1.year().months(2);
    ///
    /// // The default tries to balance spacing with compact
    /// // unit designators.
    /// let printer = SpanPrinter::new();
    /// assert_eq!(printer.span_to_string(&span), "1y 2mo");
    ///
    /// // But you can use slightly more descriptive
    /// // designators without being too verbose.
    /// let printer = SpanPrinter::new()
    ///     .designator(Designator::Short);
    /// assert_eq!(printer.span_to_string(&span), "1yr 2mos");
    ///
    /// // When spacing is removed, it usually looks nicer
    /// // to use compact unit designators.
    /// let printer = SpanPrinter::new()
    ///     .spacing(Spacing::None)
    ///     .designator(Designator::Compact);
    /// assert_eq!(printer.span_to_string(&span), "1y2mo");
    ///
    /// // Conversely, when using more spacing, it usually
    /// // looks nicer to use verbose unit designators.
    /// let printer = SpanPrinter::new()
    ///     .spacing(Spacing::BetweenUnitsAndDesignators)
    ///     .designator(Designator::Verbose);
    /// assert_eq!(printer.span_to_string(&span), "1 year 2 months");
    /// ```
    ///
    /// # Example: `Spacing::None` can still result in whitespace
    ///
    /// In the case that [`SpanPrinter::hours_minutes_seconds`] is enabled
    /// and one is formatting a span with non-zero calendar units, then an
    /// ASCII whitespace is inserted between the calendar and non-calendar
    /// units even when `Spacing::None` is used:
    ///
    /// ```
    /// use jiff::{fmt::friendly::{Spacing, SpanPrinter}, ToSpan};
    ///
    /// let span = 1.year().months(2).hours(15);
    ///
    /// let printer = SpanPrinter::new()
    ///     .spacing(Spacing::None)
    ///     .hours_minutes_seconds(true);
    /// assert_eq!(printer.span_to_string(&span), "1y2mo 15:00:00");
    /// ```
    #[inline]
    pub const fn spacing(self, spacing: Spacing) -> SpanPrinter {
        SpanPrinter { spacing, ..self }
    }

    /// Configures how and when the sign for the duration is written.
    ///
    /// The default is [`Direction::Auto`]. In most cases, this results in
    /// writing the suffix ` ago` after printing the duration units when the
    /// sign of the duration is negative. And when the sign is positive, there
    /// is no suffix. However, this can vary based on other settings. For
    /// example, when [`SpanPrinter::spacing`] is set to [`Spacing::None`],
    /// then `Direction::Auto` is treated as if it were [`Direction::Sign`].
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::{fmt::friendly::{Direction, SpanPrinter}, SignedDuration};
    ///
    /// let duration = SignedDuration::from_secs(-1);
    ///
    /// let printer = SpanPrinter::new();
    /// assert_eq!(printer.duration_to_string(&duration), "1s ago");
    ///
    /// let printer = SpanPrinter::new().direction(Direction::Sign);
    /// assert_eq!(printer.duration_to_string(&duration), "-1s");
    /// ```
    #[inline]
    pub const fn direction(self, direction: Direction) -> SpanPrinter {
        SpanPrinter { direction, ..self }
    }

    /// Enable fractional formatting for the given unit.
    ///
    /// When [`SpanPrinter::hours_minutes_seconds`] is enabled, then this
    /// setting is automatically set to [`FractionalUnit::Second`]. Otherwise,
    /// it defaults to `None`, which means no fractions are ever written.
    ///
    /// # Example
    ///
    /// This example shows how to write the same duration with different
    /// fractional settings:
    ///
    /// ```
    /// use jiff::{fmt::friendly::{FractionalUnit, SpanPrinter}, SignedDuration};
    ///
    /// let duration = SignedDuration::from_secs(3663);
    ///
    /// let printer = SpanPrinter::new()
    ///     .fractional(Some(FractionalUnit::Hour));
    /// assert_eq!(printer.duration_to_string(&duration), "1.0175h");
    ///
    /// let printer = SpanPrinter::new()
    ///     .fractional(Some(FractionalUnit::Minute));
    /// assert_eq!(printer.duration_to_string(&duration), "1h 1.05m");
    ///
    /// let printer = SpanPrinter::new()
    ///     .fractional(Some(FractionalUnit::Second));
    /// assert_eq!(printer.duration_to_string(&duration), "1h 1m 3s");
    /// ```
    ///
    /// # Example: precision loss
    ///
    /// Because the "friendly" format is limited to 9 decimal places, when
    /// using `FractionalUnit::Hour` or `FractionalUnit::Minute`, it is
    /// possible for precision loss to occur.
    ///
    /// ```
    /// use jiff::{fmt::friendly::{FractionalUnit, SpanPrinter}, SignedDuration};
    ///
    /// // one nanosecond
    /// let duration = SignedDuration::new(0, 1);
    ///
    /// let printer = SpanPrinter::new()
    ///     .fractional(Some(FractionalUnit::Hour));
    /// assert_eq!(printer.duration_to_string(&duration), "0h");
    ///
    /// let printer = SpanPrinter::new()
    ///     .fractional(Some(FractionalUnit::Minute));
    /// assert_eq!(printer.duration_to_string(&duration), "0m");
    /// ```
    #[inline]
    pub const fn fractional(
        self,
        unit: Option<FractionalUnit>,
    ) -> SpanPrinter {
        SpanPrinter { fractional: unit, ..self }
    }

    /// When enabled, commas are written after unit designators.
    ///
    /// This is disabled by default.
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::{fmt::friendly::{Designator, Spacing, SpanPrinter}, ToSpan};
    ///
    /// static PRINTER: SpanPrinter = SpanPrinter::new()
    ///     .designator(Designator::Verbose)
    ///     .spacing(Spacing::BetweenUnitsAndDesignators)
    ///     .comma_after_designator(true);
    ///
    /// let span = 5.years().months(3).milliseconds(123);
    /// assert_eq!(
    ///     PRINTER.span_to_string(&span),
    ///     "5 years, 3 months, 123 milliseconds",
    /// );
    /// ```
    #[inline]
    pub const fn comma_after_designator(self, yes: bool) -> SpanPrinter {
        SpanPrinter { comma_after_designator: yes, ..self }
    }

    /// Formats the span or duration into a `HH:MM:SS[.fffffffff]` format.
    ///
    /// When formatting a `Span` with non-zero calendar units (units of days
    /// or greater), then the calendar units are formatted as typical with
    /// their corresponding designators. For example, `1d 01:00:00`. Note
    /// that when formatting a `SignedDuration`, calendar units are never used.
    ///
    /// When this is enabled, many of the other options are either ignored or
    /// fixed to a specific setting:
    ///
    /// * Since this format does not use any unit designators for units of
    /// hours or smaller, the [`SpanPrinter::designator`] setting is ignored
    /// for hours or smaller. It is still used when formatting a `Span` with
    /// non-zero calendar units.
    /// * [`SpanPrinter::spacing`] setting is ignored for units of hours or
    /// smaller.
    /// * The [`SpanPrinter::fractional`] setting is forcefully set to
    /// [`FractionalUnit::Second`]. It cannot be changed.
    /// * The [`SpanPrinter::comma_after_designator`] setting is ignored for
    /// units of hours or smaller.
    /// * When the padding is not specified, it defaults to `2` for hours,
    /// minutes and seconds and `0` for any calendar units present.
    /// * The precision setting is respected as documented.
    ///
    /// This format is useful in contexts for interfacing with existing systems
    /// that require this style of format, or if the `HH:MM:SS` is just in
    /// general preferred.
    ///
    /// # Loss of fidelity
    ///
    /// When using this format with a `Span`, sub-second units are formatted
    /// as a fractional second. This means that `1000 milliseconds` and
    /// `1 second` format to precisely the same string. This is similar to the
    /// loss of fidelity when using [`fmt::temporal`](crate::fmt::temporal)
    /// to format spans in the ISO 8601 duration format.
    ///
    /// # Example
    ///
    /// This shows how to format a `Span` in `HH:MM:SS` format:
    ///
    /// ```
    /// use jiff::{fmt::friendly::SpanPrinter, ToSpan};
    ///
    /// static PRINTER: SpanPrinter =
    ///     SpanPrinter::new().hours_minutes_seconds(true);
    ///
    /// let span = 2.hours().minutes(59).seconds(15).milliseconds(123);
    /// assert_eq!(PRINTER.span_to_string(&span), "02:59:15.123");
    /// assert_eq!(PRINTER.span_to_string(&-span), "-02:59:15.123");
    ///
    /// // This shows what happens with calendar units.
    /// let span = 15.days().hours(2).minutes(59).seconds(15).milliseconds(123);
    /// assert_eq!(PRINTER.span_to_string(&span), "15d 02:59:15.123");
    /// // Notice that because calendar units are specified and the sign
    /// // setting is set to "auto" by default, it has switched to a suffix.
    /// assert_eq!(PRINTER.span_to_string(&-span), "15d 02:59:15.123 ago");
    /// ```
    ///
    /// And this shows the same, but with a [`SignedDuration`]:
    ///
    /// ```
    /// use jiff::{fmt::friendly::SpanPrinter, SignedDuration};
    ///
    /// static PRINTER: SpanPrinter =
    ///     SpanPrinter::new().hours_minutes_seconds(true);
    ///
    /// let duration = SignedDuration::new(
    ///     2 * 60 * 60 + 59 * 60 + 15,
    ///     123_000_000,
    /// );
    /// assert_eq!(PRINTER.duration_to_string(&duration), "02:59:15.123");
    /// assert_eq!(PRINTER.duration_to_string(&-duration), "-02:59:15.123");
    /// ```
    ///
    /// # Example: `Span` versus `SignedDuration`
    ///
    /// The main advantage of a `Span` is that, except for fractional
    /// components, the unit values emitted correspond precisely to the values
    /// in the `Span`. Where as for a `SignedDuration`, the units are always
    /// computed from a single absolute duration in a way that is always
    /// balanced:
    ///
    /// ```
    /// use jiff::{fmt::friendly::SpanPrinter, SignedDuration, ToSpan};
    ///
    /// static PRINTER: SpanPrinter =
    ///     SpanPrinter::new().hours_minutes_seconds(true);
    ///
    /// let span = 120.minutes();
    /// assert_eq!(PRINTER.span_to_string(&span), "00:120:00");
    ///
    /// let duration = SignedDuration::from_mins(120);
    /// assert_eq!(PRINTER.duration_to_string(&duration), "02:00:00");
    /// ```
    ///
    /// Of course, a balanced duration is sometimes what you want. But `Span`
    /// affords the flexibility of controlling precisely what the unit values
    /// are.
    #[inline]
    pub const fn hours_minutes_seconds(self, yes: bool) -> SpanPrinter {
        SpanPrinter { hms: yes, ..self }
    }

    /// The padding to use when writing unit values.
    ///
    /// If a unit value has fewer digits than specified here, it is padded to
    /// the left with zeroes. (To control precision, i.e., padding to the right
    /// when writing fractional values, use [`SpanPrinter::precision`].)
    ///
    /// By default, when writing in the hours-minutes-seconds format, a padding
    /// of `2` is used for units of hours, minutes and seconds. Otherwise, a
    /// padding of `0` is used.
    ///
    /// Padding is clamped to a maximum value of `20` (corresponding to the
    /// number of digits in `u64::MAX`).
    ///
    /// # Example
    ///
    /// This shows some examples of configuring padding when writing in default
    /// format with unit designators:
    ///
    /// ```
    /// use jiff::{fmt::friendly::SpanPrinter, ToSpan};
    ///
    /// let printer = SpanPrinter::new();
    /// assert_eq!(printer.span_to_string(&1.hour()), "1h");
    /// let printer = SpanPrinter::new().padding(3);
    /// assert_eq!(printer.span_to_string(&1.hour()), "001h");
    /// ```
    ///
    /// And this shows some examples with the hours-minutes-seconds format.
    /// Notice how padding is enabled by default.
    ///
    /// ```
    /// use jiff::{fmt::friendly::SpanPrinter, ToSpan};
    ///
    /// let printer = SpanPrinter::new().hours_minutes_seconds(true);
    /// assert_eq!(printer.span_to_string(&1.hour()), "01:00:00");
    /// let printer = SpanPrinter::new().hours_minutes_seconds(true).padding(0);
    /// assert_eq!(printer.span_to_string(&1.hour()), "1:0:0");
    ///
    /// // In this case, under the default configuration, the padding
    /// // for calendar units is 0 but the padding for time units is 2.
    /// let printer = SpanPrinter::new().hours_minutes_seconds(true);
    /// assert_eq!(printer.span_to_string(&1.day().hours(1)), "1d 01:00:00");
    /// ```
    #[inline]
    pub const fn padding(self, digits: u8) -> SpanPrinter {
        SpanPrinter { padding: Some(digits), ..self }
    }

    /// The precision to use when writing fractional unit values.
    ///
    /// This setting has no effect if fractional formatting isn't enabled.
    /// Fractional formatting is only enabled when [`SpanPrinter::fractional`]
    /// is set or if [`SpanPrinter::hours_minutes_seconds`] are enabled.
    /// Neither are enabled by default.
    ///
    /// A precision of `Some(0)` implies that truncation of any fractional
    /// component always occurs.
    ///
    /// The default value is `None`, which means the precision is automatically
    /// determined from the value. If no fractional component is needed, then
    /// none will be printed.
    ///
    /// Precision is capped to a maximum value of `9` (corresponding to the
    /// maximum precision supported by Jiff).
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::{fmt::friendly::{FractionalUnit, SpanPrinter}, ToSpan};
    ///
    /// // No effect, because fractions aren't enabled.
    /// let printer = SpanPrinter::new().precision(Some(2));
    /// assert_eq!(printer.span_to_string(&1.hour()), "1h");
    ///
    /// // Precision setting takes effect!
    /// let printer = SpanPrinter::new()
    ///     .precision(Some(2))
    ///     .fractional(Some(FractionalUnit::Hour));
    /// assert_eq!(printer.span_to_string(&1.hour()), "1.00h");
    ///
    /// // The HH:MM:SS format automatically enables fractional
    /// // second values.
    /// let printer = SpanPrinter::new()
    ///     // Truncate to millisecond precision.
    ///     .precision(Some(3))
    ///     .hours_minutes_seconds(true);
    /// let span = 1.second().milliseconds(1).microseconds(1).nanoseconds(1);
    /// assert_eq!(printer.span_to_string(&span), "00:00:01.001");
    ///
    /// // Same as above, but with the designator or "expanded"
    /// // format. This requires explicitly enabling fractional
    /// // units.
    /// let printer = SpanPrinter::new()
    ///     // Truncate to millisecond precision.
    ///     .precision(Some(3))
    ///     .fractional(Some(FractionalUnit::Second));
    /// let span = 1.second().milliseconds(1).microseconds(1).nanoseconds(1);
    /// assert_eq!(printer.span_to_string(&span), "1.001s");
    /// ```
    #[inline]
    pub const fn precision(self, precision: Option<u8>) -> SpanPrinter {
        SpanPrinter { precision, ..self }
    }

    /// Sets the unit to use when printing a duration that is zero.
    ///
    /// When [`SpanPrinter::fractional`] is set, then this setting is ignored
    /// and the zero unit corresponds to the fractional unit specified.
    ///
    /// This defaults to [`Unit::Second`].
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::{fmt::friendly::{FractionalUnit, SpanPrinter}, ToSpan, Unit};
    ///
    /// // The default just always uses seconds.
    /// let printer = SpanPrinter::new();
    /// assert_eq!(printer.span_to_string(&0.years()), "0s");
    ///
    /// // We can set our own unit.
    /// let printer = SpanPrinter::new().zero_unit(Unit::Year);
    /// assert_eq!(printer.span_to_string(&0.years()), "0y");
    ///
    /// // But it's overridden if fractional units are set.
    /// let printer = SpanPrinter::new()
    ///     .zero_unit(Unit::Year)
    ///     .fractional(Some(FractionalUnit::Minute));
    /// assert_eq!(printer.span_to_string(&0.years()), "0m");
    ///
    /// // One use case for this option is if you're rounding
    /// // spans and want the zero unit to reflect the smallest
    /// // unit you're using.
    /// let printer = SpanPrinter::new().zero_unit(Unit::Minute);
    /// let span = 5.hours().minutes(30).seconds(59);
    /// let rounded = span.round(Unit::Minute)?;
    /// assert_eq!(printer.span_to_string(&rounded), "5h 31m");
    ///
    /// let span = 5.seconds();
    /// let rounded = span.round(Unit::Minute)?;
    /// assert_eq!(printer.span_to_string(&rounded), "0m");
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// The same applies for `SignedDuration`:
    ///
    /// ```
    /// use jiff::{fmt::friendly::SpanPrinter, SignedDuration, Unit};
    ///
    /// // The default just always uses seconds.
    /// let printer = SpanPrinter::new();
    /// assert_eq!(printer.duration_to_string(&SignedDuration::ZERO), "0s");
    ///
    /// // We can set our own unit.
    /// let printer = SpanPrinter::new().zero_unit(Unit::Minute);
    /// assert_eq!(printer.duration_to_string(&SignedDuration::ZERO), "0m");
    /// ```
    #[inline]
    pub const fn zero_unit(self, unit: Unit) -> SpanPrinter {
        SpanPrinter { zero_unit: unit, ..self }
    }

    /// Format a `Span` into a string using the "friendly" format.
    ///
    /// This is a convenience routine for [`SpanPrinter::print_span`] with a
    /// `String`.
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::{fmt::friendly::SpanPrinter, ToSpan};
    ///
    /// static PRINTER: SpanPrinter = SpanPrinter::new();
    ///
    /// let span = 3.years().months(5);
    /// assert_eq!(PRINTER.span_to_string(&span), "3y 5mo");
    /// ```
    #[cfg(any(test, feature = "alloc"))]
    pub fn span_to_string(&self, span: &Span) -> alloc::string::String {
        let mut buf = alloc::string::String::with_capacity(4);
        // OK because writing to `String` never fails.
        self.print_span(span, &mut buf).unwrap();
        buf
    }

    /// Format a `SignedDuration` into a string using the "friendly" format.
    ///
    /// This balances the units of the duration up to at most hours
    /// automatically.
    ///
    /// This is a convenience routine for [`SpanPrinter::print_duration`] with
    /// a `String`.
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::{fmt::friendly::{FractionalUnit, SpanPrinter}, SignedDuration};
    ///
    /// static PRINTER: SpanPrinter = SpanPrinter::new();
    ///
    /// let dur = SignedDuration::new(86_525, 123_000_789);
    /// assert_eq!(
    ///     PRINTER.duration_to_string(&dur),
    ///     "24h 2m 5s 123ms 789ns",
    /// );
    /// assert_eq!(
    ///     PRINTER.duration_to_string(&-dur),
    ///     "24h 2m 5s 123ms 789ns ago",
    /// );
    ///
    /// // Or, if you prefer fractional seconds:
    /// static PRINTER_FRACTIONAL: SpanPrinter = SpanPrinter::new()
    ///     .fractional(Some(FractionalUnit::Second));
    /// assert_eq!(
    ///     PRINTER_FRACTIONAL.duration_to_string(&-dur),
    ///     "24h 2m 5.123000789s ago",
    /// );
    /// ```
    #[cfg(any(test, feature = "alloc"))]
    pub fn duration_to_string(
        &self,
        duration: &SignedDuration,
    ) -> alloc::string::String {
        let mut buf = alloc::string::String::with_capacity(4);
        // OK because writing to `String` never fails.
        self.print_duration(duration, &mut buf).unwrap();
        buf
    }

    /// Format a `std::time::Duration` into a string using the "friendly"
    /// format.
    ///
    /// This balances the units of the duration up to at most hours
    /// automatically.
    ///
    /// This is a convenience routine for
    /// [`SpanPrinter::print_unsigned_duration`] with a `String`.
    ///
    /// # Example
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// use jiff::fmt::friendly::{FractionalUnit, SpanPrinter};
    ///
    /// static PRINTER: SpanPrinter = SpanPrinter::new();
    ///
    /// let dur = Duration::new(86_525, 123_000_789);
    /// assert_eq!(
    ///     PRINTER.unsigned_duration_to_string(&dur),
    ///     "24h 2m 5s 123ms 789ns",
    /// );
    ///
    /// // Or, if you prefer fractional seconds:
    /// static PRINTER_FRACTIONAL: SpanPrinter = SpanPrinter::new()
    ///     .fractional(Some(FractionalUnit::Second));
    /// assert_eq!(
    ///     PRINTER_FRACTIONAL.unsigned_duration_to_string(&dur),
    ///     "24h 2m 5.123000789s",
    /// );
    /// ```
    #[cfg(any(test, feature = "alloc"))]
    pub fn unsigned_duration_to_string(
        &self,
        duration: &core::time::Duration,
    ) -> alloc::string::String {
        let mut buf = alloc::string::String::with_capacity(4);
        // OK because writing to `String` never fails.
        self.print_unsigned_duration(duration, &mut buf).unwrap();
        buf
    }

    /// Print a `Span` to the given writer using the "friendly" format.
    ///
    /// # Errors
    ///
    /// This only returns an error when writing to the given [`Write`]
    /// implementation would fail. Some such implementations, like for `String`
    /// and `Vec<u8>`, never fail (unless memory allocation fails). In such
    /// cases, it would be appropriate to call `unwrap()` on the result.
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::{fmt::friendly::SpanPrinter, ToSpan};
    ///
    /// static PRINTER: SpanPrinter = SpanPrinter::new();
    ///
    /// let span = 3.years().months(5);
    ///
    /// let mut buf = String::new();
    /// // Printing to a `String` can never fail.
    /// PRINTER.print_span(&span, &mut buf).unwrap();
    /// assert_eq!(buf, "3y 5mo");
    /// ```
    pub fn print_span<W: Write>(
        &self,
        span: &Span,
        mut wtr: W,
    ) -> Result<(), Error> {
        let mut buf = ArrayBuffer::<MAX_SPAN_LEN>::default();
        let mut bbuf = buf.as_borrowed();
        if self.hms {
            self.print_span_hms(span, &mut bbuf);
        } else {
            self.print_span_designators(span, &mut bbuf);
        }
        wtr.write_str(bbuf.filled())
    }

    /// Print a `SignedDuration` to the given writer using the "friendly"
    /// format.
    ///
    /// This balances the units of the duration up to at most hours
    /// automatically.
    ///
    /// # Errors
    ///
    /// This only returns an error when writing to the given [`Write`]
    /// implementation would fail. Some such implementations, like for `String`
    /// and `Vec<u8>`, never fail (unless memory allocation fails). In such
    /// cases, it would be appropriate to call `unwrap()` on the result.
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::{fmt::friendly::SpanPrinter, SignedDuration};
    ///
    /// static PRINTER: SpanPrinter = SpanPrinter::new();
    ///
    /// let dur = SignedDuration::new(86_525, 123_000_789);
    ///
    /// let mut buf = String::new();
    /// // Printing to a `String` can never fail.
    /// PRINTER.print_duration(&dur, &mut buf).unwrap();
    /// assert_eq!(buf, "24h 2m 5s 123ms 789ns");
    ///
    /// // Negative durations are supported.
    /// buf.clear();
    /// PRINTER.print_duration(&-dur, &mut buf).unwrap();
    /// assert_eq!(buf, "24h 2m 5s 123ms 789ns ago");
    /// ```
    pub fn print_duration<W: Write>(
        &self,
        duration: &SignedDuration,
        mut wtr: W,
    ) -> Result<(), Error> {
        let mut buf = ArrayBuffer::<MAX_SIGNED_DURATION_LEN>::default();
        let mut bbuf = buf.as_borrowed();
        if self.hms {
            self.print_signed_duration_hms(duration, &mut bbuf);
        } else {
            self.print_signed_duration_designators(duration, &mut bbuf);
        }
        wtr.write_str(bbuf.filled())
    }

    /// Print a `std::time::Duration` to the given writer using the "friendly"
    /// format.
    ///
    /// This balances the units of the duration up to at most hours
    /// automatically.
    ///
    /// # Errors
    ///
    /// This only returns an error when writing to the given [`Write`]
    /// implementation would fail. Some such implementations, like for `String`
    /// and `Vec<u8>`, never fail (unless memory allocation fails). In such
    /// cases, it would be appropriate to call `unwrap()` on the result.
    ///
    /// # Example
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// use jiff::fmt::friendly::SpanPrinter;
    ///
    /// static PRINTER: SpanPrinter = SpanPrinter::new();
    ///
    /// let dur = Duration::new(86_525, 123_000_789);
    ///
    /// let mut buf = String::new();
    /// // Printing to a `String` can never fail.
    /// PRINTER.print_unsigned_duration(&dur, &mut buf).unwrap();
    /// assert_eq!(buf, "24h 2m 5s 123ms 789ns");
    /// ```
    pub fn print_unsigned_duration<W: Write>(
        &self,
        duration: &core::time::Duration,
        mut wtr: W,
    ) -> Result<(), Error> {
        let mut buf = ArrayBuffer::<MAX_UNSIGNED_DURATION_LEN>::default();
        let mut bbuf = buf.as_borrowed();
        if self.hms {
            self.print_unsigned_duration_hms(duration, &mut bbuf);
        } else {
            self.print_unsigned_duration_designators(duration, &mut bbuf);
        }
        wtr.write_str(bbuf.filled())
    }

    fn print_span_designators(
        &self,
        span: &Span,
        bbuf: &mut BorrowedBuffer<'_>,
    ) {
        let mut wtr = DesignatorWriter::new(self, bbuf, false, span.signum());
        wtr.maybe_write_prefix_sign();
        match self.fractional {
            None => {
                self.print_span_designators_non_fraction(span, &mut wtr);
            }
            Some(unit) => {
                self.print_span_designators_fractional(span, unit, &mut wtr);
            }
        }
        wtr.maybe_write_zero();
        wtr.maybe_write_suffix_sign();
    }

    fn print_span_designators_non_fraction<'p, 'w, 'd>(
        &self,
        span: &Span,
        wtr: &mut DesignatorWriter<'p, 'w, 'd>,
    ) {
        let units = span.units();

        if units.contains(Unit::Year) {
            wtr.write(Unit::Year, span.get_years_unsigned().into());
        }
        if units.contains(Unit::Month) {
            wtr.write(Unit::Month, span.get_months_unsigned().into());
        }
        if units.contains(Unit::Week) {
            wtr.write(Unit::Week, span.get_weeks_unsigned().into());
        }
        if units.contains(Unit::Day) {
            wtr.write(Unit::Day, span.get_days_unsigned().into());
        }
        if units.contains(Unit::Hour) {
            wtr.write(Unit::Hour, span.get_hours_unsigned().into());
        }
        if units.contains(Unit::Minute) {
            wtr.write(Unit::Minute, span.get_minutes_unsigned());
        }
        if units.contains(Unit::Second) {
            wtr.write(Unit::Second, span.get_seconds_unsigned());
        }
        if units.contains(Unit::Millisecond) {
            wtr.write(Unit::Millisecond, span.get_milliseconds_unsigned());
        }
        if units.contains(Unit::Microsecond) {
            wtr.write(Unit::Microsecond, span.get_microseconds_unsigned());
        }
        if units.contains(Unit::Nanosecond) {
            wtr.write(Unit::Nanosecond, span.get_nanoseconds_unsigned());
        }
    }

    fn print_span_calendar_designators_non_fraction<'p, 'w, 'd>(
        &self,
        span: &Span,
        wtr: &mut DesignatorWriter<'p, 'w, 'd>,
    ) {
        let units = span.units();

        if units.contains(Unit::Year) {
            wtr.write(Unit::Year, span.get_years_unsigned().into());
        }
        if units.contains(Unit::Month) {
            wtr.write(Unit::Month, span.get_months_unsigned().into());
        }
        if units.contains(Unit::Week) {
            wtr.write(Unit::Week, span.get_weeks_unsigned().into());
        }
        if units.contains(Unit::Day) {
            wtr.write(Unit::Day, span.get_days_unsigned().into());
        }
    }

    #[inline(never)]
    fn print_span_designators_fractional<'p, 'w, 'd>(
        &self,
        span: &Span,
        unit: FractionalUnit,
        wtr: &mut DesignatorWriter<'p, 'w, 'd>,
    ) {
        // OK because the biggest FractionalUnit is Hour, and there is always
        // a Unit bigger than hour.
        let split_at = Unit::from(unit).next().unwrap();
        let non_fractional = span.without_lower(split_at);
        let fractional = span.only_lower(split_at);
        self.print_span_designators_non_fraction(&non_fractional, wtr);
        wtr.write_fractional_duration(
            unit,
            &fractional.to_invariant_duration().unsigned_abs(),
        );
    }

    fn print_span_hms(&self, span: &Span, bbuf: &mut BorrowedBuffer<'_>) {
        let has_cal = !span.units().only_calendar().is_empty();
        let mut wtr =
            DesignatorWriter::new(self, bbuf, has_cal, span.signum());
        let span = span.abs();

        wtr.maybe_write_prefix_sign();
        if has_cal {
            self.print_span_calendar_designators_non_fraction(&span, &mut wtr);
            wtr.finish_preceding();
            // When spacing is disabled, then `finish_preceding` won't write
            // any spaces. But this would result in, e.g., `1yr15:00:00`, which
            // is just totally wrong. So detect that case here and insert a
            // space forcefully.
            if matches!(self.spacing, Spacing::None) {
                wtr.bbuf.write_ascii_char(b' ');
            }
        }

        let padding = self.padding.unwrap_or(2);
        wtr.bbuf.write_int_pad(span.get_hours_unsigned(), b'0', padding);
        wtr.bbuf.write_ascii_char(b':');
        wtr.bbuf.write_int_pad(span.get_minutes_unsigned(), b'0', padding);
        wtr.bbuf.write_ascii_char(b':');
        // You'd think we could do better here from a code size
        // perspective. But when I tried to inline the logic to
        // get a `SignedDuration` from just sub-minute units,
        // code size actually increased. ¯\_(ツ)_/¯
        let fp = FractionalPrinter::from_span_seconds(
            &span.only_lower(Unit::Minute),
            padding,
            self.precision,
        );
        fp.print(wtr.bbuf);
        wtr.maybe_write_suffix_sign();
    }

    fn print_signed_duration_designators(
        &self,
        dur: &SignedDuration,
        bbuf: &mut BorrowedBuffer<'_>,
    ) {
        let mut wtr = DesignatorWriter::new(self, bbuf, false, dur.signum());
        wtr.maybe_write_prefix_sign();
        self.print_duration_designators(&dur.unsigned_abs(), &mut wtr);
        wtr.maybe_write_zero();
        wtr.maybe_write_suffix_sign();
    }

    fn print_unsigned_duration_designators(
        &self,
        dur: &core::time::Duration,
        bbuf: &mut BorrowedBuffer<'_>,
    ) {
        let mut wtr = DesignatorWriter::new(self, bbuf, false, 1);
        wtr.maybe_write_prefix_sign();
        self.print_duration_designators(dur, &mut wtr);
        wtr.maybe_write_zero();
    }

    fn print_duration_designators(
        &self,
        dur: &core::time::Duration,
        wtr: &mut DesignatorWriter<'_, '_, '_>,
    ) {
        match self.fractional {
            None => {
                let mut secs = dur.as_secs();
                wtr.write(Unit::Hour, secs / SECS_PER_HOUR);
                secs %= MINS_PER_HOUR * SECS_PER_MIN;
                wtr.write(Unit::Minute, secs / SECS_PER_MIN);
                wtr.write(Unit::Second, secs % SECS_PER_MIN);
                let mut nanos = dur.subsec_nanos();
                wtr.write(Unit::Millisecond, (nanos / NANOS_PER_MILLI).into());
                nanos %= NANOS_PER_MILLI;
                wtr.write(Unit::Microsecond, (nanos / NANOS_PER_MICRO).into());
                wtr.write(Unit::Nanosecond, (nanos % NANOS_PER_MICRO).into());
            }
            Some(FractionalUnit::Hour) => {
                wtr.write_fractional_duration(FractionalUnit::Hour, &dur);
            }
            Some(FractionalUnit::Minute) => {
                let mut secs = dur.as_secs();
                wtr.write(Unit::Hour, secs / SECS_PER_HOUR);
                secs %= MINS_PER_HOUR * SECS_PER_MIN;

                let leftovers =
                    core::time::Duration::new(secs, dur.subsec_nanos());
                wtr.write_fractional_duration(
                    FractionalUnit::Minute,
                    &leftovers,
                );
            }
            Some(FractionalUnit::Second) => {
                let mut secs = dur.as_secs();
                wtr.write(Unit::Hour, secs / SECS_PER_HOUR);
                secs %= MINS_PER_HOUR * SECS_PER_MIN;
                wtr.write(Unit::Minute, secs / SECS_PER_MIN);
                secs %= SECS_PER_MIN;

                let leftovers =
                    core::time::Duration::new(secs, dur.subsec_nanos());
                wtr.write_fractional_duration(
                    FractionalUnit::Second,
                    &leftovers,
                );
            }
            Some(FractionalUnit::Millisecond) => {
                let mut secs = dur.as_secs();
                wtr.write(Unit::Hour, secs / SECS_PER_HOUR);
                secs %= MINS_PER_HOUR * SECS_PER_MIN;
                wtr.write(Unit::Minute, secs / SECS_PER_MIN);
                wtr.write(Unit::Second, secs % SECS_PER_MIN);

                let leftovers =
                    core::time::Duration::new(0, dur.subsec_nanos());
                wtr.write_fractional_duration(
                    FractionalUnit::Millisecond,
                    &leftovers,
                );
            }
            Some(FractionalUnit::Microsecond) => {
                let mut secs = dur.as_secs();
                wtr.write(Unit::Hour, secs / SECS_PER_HOUR);
                secs %= MINS_PER_HOUR * SECS_PER_MIN;
                wtr.write(Unit::Minute, secs / SECS_PER_MIN);
                wtr.write(Unit::Second, secs % SECS_PER_MIN);
                let mut nanos = dur.subsec_nanos();
                wtr.write(Unit::Millisecond, (nanos / NANOS_PER_MILLI).into());
                nanos %= NANOS_PER_MILLI;

                let leftovers = core::time::Duration::new(0, nanos);
                wtr.write_fractional_duration(
                    FractionalUnit::Microsecond,
                    &leftovers,
                );
            }
        }
    }

    fn print_signed_duration_hms(
        &self,
        dur: &SignedDuration,
        bbuf: &mut BorrowedBuffer<'_>,
    ) {
        if dur.is_negative() {
            if !matches!(self.direction, Direction::Suffix) {
                bbuf.write_ascii_char(b'-');
            }
        } else if let Direction::ForceSign = self.direction {
            bbuf.write_ascii_char(b'+');
        }
        self.print_duration_hms(&dur.unsigned_abs(), bbuf);
        if dur.is_negative() {
            if matches!(self.direction, Direction::Suffix) {
                bbuf.write_str(" ago");
            }
        }
    }

    fn print_unsigned_duration_hms(
        &self,
        dur: &core::time::Duration,
        bbuf: &mut BorrowedBuffer<'_>,
    ) {
        if let Direction::ForceSign = self.direction {
            bbuf.write_ascii_char(b'+');
        }
        self.print_duration_hms(dur, bbuf);
    }

    fn print_duration_hms(
        &self,
        udur: &core::time::Duration,
        bbuf: &mut BorrowedBuffer<'_>,
    ) {
        // N.B. It should be technically correct to convert a `SignedDuration`
        // (or `core::time::Duration`) to `Span` (since this process balances)
        // and then format the `Span` as-is. But this doesn't work because the
        // range of a `SignedDuration` (and `core::time::Duration`) is much
        // bigger.

        let padding = self.padding.unwrap_or(2);

        let mut secs = udur.as_secs();
        // OK because guaranteed to be bigger than i64::MIN.
        let hours = secs / (MINS_PER_HOUR * SECS_PER_MIN);
        secs %= MINS_PER_HOUR * SECS_PER_MIN;
        // OK because guaranteed to be bigger than i64::MIN.
        let minutes = secs / SECS_PER_MIN;
        // OK because guaranteed to be bigger than i64::MIN.
        secs = secs % SECS_PER_MIN;

        bbuf.write_int_pad(hours, b'0', padding);
        bbuf.write_ascii_char(b':');
        bbuf.write_int_pad(minutes, b'0', padding);
        bbuf.write_ascii_char(b':');
        let fp = FractionalPrinter::from_duration_seconds(
            // OK because -999_999_999 <= nanos <= 999_999_999 and secs < 60.
            &core::time::Duration::new(secs, udur.subsec_nanos()),
            padding,
            self.precision,
        );
        fp.print(bbuf);
    }
}

impl Default for SpanPrinter {
    fn default() -> SpanPrinter {
        SpanPrinter::new()
    }
}

/// A type that represents the designator choice.
///
/// Basically, whether we want verbose, short or compact designators. This in
/// turn permits lookups based on `Unit`, which makes writing generic code for
/// writing designators a bit nicer and still fast.
#[derive(Clone, Debug)]
struct Designators {
    singular: [&'static str; 10],
    plural: [&'static str; 10],
}

impl Designators {
    const VERBOSE_SINGULAR: [&'static str; 10] = [
        "nanosecond",
        "microsecond",
        "millisecond",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
    ];
    const VERBOSE_PLURAL: [&'static str; 10] = [
        "nanoseconds",
        "microseconds",
        "milliseconds",
        "seconds",
        "minutes",
        "hours",
        "days",
        "weeks",
        "months",
        "years",
    ];

    const SHORT_SINGULAR: [&'static str; 10] =
        ["nsec", "µsec", "msec", "sec", "min", "hr", "day", "wk", "mo", "yr"];
    const SHORT_PLURAL: [&'static str; 10] = [
        "nsecs", "µsecs", "msecs", "secs", "mins", "hrs", "days", "wks",
        "mos", "yrs",
    ];

    const COMPACT: [&'static str; 10] =
        ["ns", "µs", "ms", "s", "m", "h", "d", "w", "mo", "y"];

    const HUMAN_TIME_SINGULAR: [&'static str; 10] =
        ["ns", "us", "ms", "s", "m", "h", "d", "w", "month", "y"];
    const HUMAN_TIME_PLURAL: [&'static str; 10] =
        ["ns", "us", "ms", "s", "m", "h", "d", "w", "months", "y"];

    const fn new(config: Designator) -> &'static Designators {
        match config {
            Designator::Verbose => &Designators {
                singular: Designators::VERBOSE_SINGULAR,
                plural: Designators::VERBOSE_PLURAL,
            },
            Designator::Short => &Designators {
                singular: Designators::SHORT_SINGULAR,
                plural: Designators::SHORT_PLURAL,
            },
            Designator::Compact => &Designators {
                singular: Designators::COMPACT,
                plural: Designators::COMPACT,
            },
            Designator::HumanTime => &Designators {
                singular: Designators::HUMAN_TIME_SINGULAR,
                plural: Designators::HUMAN_TIME_PLURAL,
            },
        }
    }

    fn designator(&self, unit: impl Into<Unit>, plural: bool) -> &'static str {
        let unit = unit.into();
        let index = unit as usize;
        if plural {
            self.plural[index]
        } else {
            self.singular[index]
        }
    }
}

/// An abstraction for writing the "designator" variant of the friendly format.
///
/// This takes care of computing some initial state and keeping track of some
/// mutable state that influences printing. For example, whether to write a
/// delimiter or not (one should only come after a unit that has been written).
#[derive(Debug)]
struct DesignatorWriter<'p, 'w, 'd> {
    printer: &'p SpanPrinter,
    bbuf: &'w mut BorrowedBuffer<'d>,
    sign: Option<DirectionSign>,
    padding: u8,
    precision: Option<u8>,
    written_non_zero_unit: bool,
}

impl<'p, 'w, 'd> DesignatorWriter<'p, 'w, 'd> {
    fn new(
        printer: &'p SpanPrinter,
        bbuf: &'w mut BorrowedBuffer<'d>,
        has_calendar: bool,
        signum: i8,
    ) -> DesignatorWriter<'p, 'w, 'd> {
        let sign = printer.direction.sign(printer, has_calendar, signum);
        DesignatorWriter {
            printer,
            bbuf,
            sign,
            padding: printer.padding.unwrap_or(0),
            precision: printer.precision,
            written_non_zero_unit: false,
        }
    }

    fn maybe_write_prefix_sign(&mut self) {
        if let Some(DirectionSign::Prefix(sign)) = self.sign {
            self.bbuf.write_ascii_char(sign);
        }
    }

    fn maybe_write_suffix_sign(&mut self) {
        if let Some(DirectionSign::Suffix) = self.sign {
            self.bbuf.write_str(" ago");
        }
    }

    fn maybe_write_zero(&mut self) {
        #[cold]
        #[inline(never)]
        fn imp(wtr: &mut DesignatorWriter<'_, '_, '_>) {
            wtr.bbuf.write_int_pad(0u64, b'0', wtr.padding);
            if let Some(byte) =
                wtr.printer.spacing.between_units_and_designators()
            {
                wtr.bbuf.write_ascii_char(byte);
            }
            // If a fractional unit is set, then we should use that unit
            // specifically to express "zero."
            let unit = wtr
                .printer
                .fractional
                .map(Unit::from)
                .unwrap_or(wtr.printer.zero_unit);
            wtr.bbuf.write_str(wtr.printer.designators.designator(unit, true));
        }

        if !self.written_non_zero_unit {
            imp(self);
        }
    }

    #[inline(never)]
    fn write(&mut self, unit: Unit, value: u64) {
        if value == 0 {
            return;
        }
        self.finish_preceding();
        self.written_non_zero_unit = true;
        self.bbuf.write_int_pad0(value, self.padding);
        if let Some(byte) =
            self.printer.spacing.between_units_and_designators()
        {
            self.bbuf.write_ascii_char(byte);
        }
        self.bbuf
            .write_str(self.printer.designators.designator(unit, value != 1));
    }

    fn write_fractional_duration(
        &mut self,
        unit: FractionalUnit,
        duration: &core::time::Duration,
    ) {
        let fp = FractionalPrinter::from_duration(
            duration,
            unit,
            self.padding,
            self.precision,
        );
        if !fp.must_write_digits() {
            return;
        }
        self.finish_preceding();
        self.written_non_zero_unit = true;
        fp.print(&mut *self.bbuf);
        if let Some(byte) =
            self.printer.spacing.between_units_and_designators()
        {
            self.bbuf.write_ascii_char(byte);
        }
        self.bbuf.write_str(
            self.printer.designators.designator(unit, fp.is_plural()),
        );
    }

    fn finish_preceding(&mut self) {
        if self.written_non_zero_unit {
            if self.printer.comma_after_designator {
                self.bbuf.write_ascii_char(b',');
            }
            if let Some(byte) = self.printer.spacing.between_units() {
                self.bbuf.write_ascii_char(byte);
            }
        }
    }
}

/// A printer for a fraction with an integer and fraction component.
///
/// This also includes the formatter for the integer component and the
/// formatter for the fractional component.
struct FractionalPrinter {
    integer: u64,
    fraction: u32,
    padding: u8,
    precision: Option<u8>,
}

impl FractionalPrinter {
    /// Build a fractional printer for the `Span` given. This includes the `.`.
    ///
    /// Callers must ensure that all units greater than `FractionalUnit` are
    /// zero in the span given.
    ///
    /// Note that the printer returned only prints a fractional component
    /// if necessary. For example, if the fractional component is zero and
    /// precision is `None`, or if `precision` is `Some(0)`, then no fractional
    /// component will be emitted.
    fn from_span_seconds(
        span: &Span,
        padding: u8,
        precision: Option<u8>,
    ) -> FractionalPrinter {
        FractionalPrinter::from_duration_seconds(
            &span.to_invariant_duration().unsigned_abs(),
            padding,
            precision,
        )
    }

    /// Like `from_span_seconds`, but for `SignedDuration`.
    fn from_duration_seconds(
        dur: &core::time::Duration,
        padding: u8,
        precision: Option<u8>,
    ) -> FractionalPrinter {
        let integer = dur.as_secs();
        let fraction = u32::from(dur.subsec_nanos());
        FractionalPrinter { integer, fraction, padding, precision }
    }

    /// Like `from_duration_seconds`, but for any fractional unit.
    fn from_duration(
        dur: &core::time::Duration,
        unit: FractionalUnit,
        padding: u8,
        precision: Option<u8>,
    ) -> FractionalPrinter {
        match unit {
            FractionalUnit::Hour => {
                let integer = dur.as_secs() / SECS_PER_HOUR;
                let mut fraction = dur.as_nanos() % NANOS_PER_HOUR;
                // Drop precision since we're only allowed 9 decimal places.
                fraction /= u128::from(SECS_PER_HOUR);
                // OK because NANOS_PER_HOUR / SECS_PER_HOUR fits in a u32.
                let fraction = u32::try_from(fraction).unwrap();
                FractionalPrinter { integer, fraction, padding, precision }
            }
            FractionalUnit::Minute => {
                let integer = dur.as_secs() / SECS_PER_MIN;
                let mut fraction = dur.as_nanos() % NANOS_PER_MIN;
                // Drop precision since we're only allowed 9 decimal places.
                fraction /= u128::from(SECS_PER_MIN);
                // OK because NANOS_PER_MIN fits in an u32.
                let fraction = u32::try_from(fraction).unwrap();
                FractionalPrinter { integer, fraction, padding, precision }
            }
            FractionalUnit::Second => {
                let integer = dur.as_secs();
                let fraction = u32::from(dur.subsec_nanos());
                FractionalPrinter { integer, fraction, padding, precision }
            }
            FractionalUnit::Millisecond => {
                // Unwrap is OK, but this is subtle. For printing a
                // SignedDuration, as_millis() can never return anything
                // bigger than 1 second, because the duration given is reduced
                // in a balanced fashion before hitting this routine. But
                // for printing a Span, it can, since spans can be totally
                // unbalanced. But Spans have limits on their units such that
                // each will fit into an i64. So this is also okay in that case
                // too.
                let integer = u64::try_from(dur.as_millis()).unwrap();
                let fraction =
                    u32::from((dur.subsec_nanos() % NANOS_PER_MILLI) * 1_000);
                FractionalPrinter { integer, fraction, padding, precision }
            }
            FractionalUnit::Microsecond => {
                // Unwrap is OK, but this is subtle. For printing a
                // SignedDuration, as_millis() can never return anything
                // bigger than 1 second, because the duration given is reduced
                // in a balanced fashion before hitting this routine. But
                // for printing a Span, it can, since spans can be totally
                // unbalanced. But Spans have limits on their units such that
                // each will fit into an i64. So this is also okay in that case
                // too.
                let integer = u64::try_from(dur.as_micros()).unwrap();
                let fraction = u32::from(
                    (dur.subsec_nanos() % NANOS_PER_MICRO) * 1_000_000,
                );
                FractionalPrinter { integer, fraction, padding, precision }
            }
        }
    }

    /// Returns true if both the integer and fractional component are zero.
    fn is_zero(&self) -> bool {
        self.integer == 0 && self.fraction == 0
    }

    /// Returns true if this integer/fraction should be considered plural
    /// when choosing what designator to use.
    fn is_plural(&self) -> bool {
        self.integer != 1
            || (self.fraction != 0 && !self.has_zero_fixed_precision())
    }

    /// Returns true if and only if this printer must write some kind of number
    /// when `print` is called.
    ///
    /// The only case where this returns `false` is when both the integer and
    /// fractional component are zero *and* the precision is fixed to a number
    /// greater than zero.
    fn must_write_digits(&self) -> bool {
        !self.is_zero() || self.has_non_zero_fixed_precision()
    }

    /// Returns true if and only if at least one digit will be written for the
    /// given value.
    ///
    /// This is useful for callers that need to know whether to write
    /// a decimal separator, e.g., `.`, before the digits.
    fn will_write_digits(&self) -> bool {
        self.precision.map_or_else(|| self.fraction != 0, |p| p > 0)
    }

    /// Returns true if and only if this formatter has an explicit non-zero
    /// precision setting.
    ///
    /// This is useful for determining whether something like `0.000` needs to
    /// be written in the case of a `precision=Some(3)` setting and a zero
    /// value.
    fn has_non_zero_fixed_precision(&self) -> bool {
        self.precision.map_or(false, |p| p > 0)
    }

    /// Returns true if and only if this formatter has fixed zero precision.
    /// That is, no matter what is given as input, a fraction is never written.
    fn has_zero_fixed_precision(&self) -> bool {
        self.precision.map_or(false, |p| p == 0)
    }

    /// Prints the integer and optional fractional component.
    ///
    /// This will always print the integer, even if it's zero. Therefore, if
    /// the caller wants to omit printing zero, the caller should do their own
    /// conditional logic.
    fn print(&self, bbuf: &mut BorrowedBuffer<'_>) {
        bbuf.write_int_pad(self.integer, b'0', self.padding);
        if self.will_write_digits() {
            bbuf.write_ascii_char(b'.');
            bbuf.write_fraction(self.precision, self.fraction);
        }
    }
}

#[cfg(feature = "alloc")]
#[cfg(test)]
mod tests {
    use crate::ToSpan;

    use super::*;

    #[test]
    fn print_span_designator_default() {
        let printer = || SpanPrinter::new();
        let p = |span| printer().span_to_string(&span);

        insta::assert_snapshot!(p(1.second()), @"1s");
        insta::assert_snapshot!(p(2.seconds()), @"2s");
        insta::assert_snapshot!(p(10.seconds()), @"10s");
        insta::assert_snapshot!(p(1.minute().seconds(40)), @"1m 40s");

        insta::assert_snapshot!(p(1.minute()), @"1m");
        insta::assert_snapshot!(p(2.minutes()), @"2m");
        insta::assert_snapshot!(p(10.minutes()), @"10m");
        insta::assert_snapshot!(p(1.hour().minutes(40)), @"1h 40m");

        insta::assert_snapshot!(p(1.hour()), @"1h");
        insta::assert_snapshot!(p(2.hours()), @"2h");
        insta::assert_snapshot!(p(10.hours()), @"10h");
        insta::assert_snapshot!(p(100.hours()), @"100h");

        insta::assert_snapshot!(
            p(1.hour().minutes(1).seconds(1)),
            @"1h 1m 1s",
        );
        insta::assert_snapshot!(
            p(2.hours().minutes(2).seconds(2)),
            @"2h 2m 2s",
        );
        insta::assert_snapshot!(
            p(10.hours().minutes(10).seconds(10)),
            @"10h 10m 10s",
        );
        insta::assert_snapshot!(
            p(100.hours().minutes(100).seconds(100)),
            @"100h 100m 100s",
        );

        insta::assert_snapshot!(p(-1.hour()), @"1h ago");
        insta::assert_snapshot!(p(-1.hour().seconds(30)), @"1h 30s ago");

        insta::assert_snapshot!(
            p(1.second().milliseconds(2000)),
            @"1s 2000ms",
        );
    }

    #[test]
    fn print_span_designator_verbose() {
        let printer = || SpanPrinter::new().designator(Designator::Verbose);
        let p = |span| printer().span_to_string(&span);

        insta::assert_snapshot!(p(1.second()), @"1second");
        insta::assert_snapshot!(p(2.seconds()), @"2seconds");
        insta::assert_snapshot!(p(10.seconds()), @"10seconds");
        insta::assert_snapshot!(p(1.minute().seconds(40)), @"1minute 40seconds");

        insta::assert_snapshot!(p(1.minute()), @"1minute");
        insta::assert_snapshot!(p(2.minutes()), @"2minutes");
        insta::assert_snapshot!(p(10.minutes()), @"10minutes");
        insta::assert_snapshot!(p(1.hour().minutes(40)), @"1hour 40minutes");

        insta::assert_snapshot!(p(1.hour()), @"1hour");
        insta::assert_snapshot!(p(2.hours()), @"2hours");
        insta::assert_snapshot!(p(10.hours()), @"10hours");
        insta::assert_snapshot!(p(100.hours()), @"100hours");

        insta::assert_snapshot!(
            p(1.hour().minutes(1).seconds(1)),
            @"1hour 1minute 1second",
        );
        insta::assert_snapshot!(
            p(2.hours().minutes(2).seconds(2)),
            @"2hours 2minutes 2seconds",
        );
        insta::assert_snapshot!(
            p(10.hours().minutes(10).seconds(10)),
            @"10hours 10minutes 10seconds",
        );
        insta::assert_snapshot!(
            p(100.hours().minutes(100).seconds(100)),
            @"100hours 100minutes 100seconds",
        );

        insta::assert_snapshot!(p(-1.hour()), @"1hour ago");
        insta::assert_snapshot!(p(-1.hour().seconds(30)), @"1hour 30seconds ago");
    }

    #[test]
    fn print_span_designator_short() {
        let printer = || SpanPrinter::new().designator(Designator::Short);
        let p = |span| printer().span_to_string(&span);

        insta::assert_snapshot!(p(1.second()), @"1sec");
        insta::assert_snapshot!(p(2.seconds()), @"2secs");
        insta::assert_snapshot!(p(10.seconds()), @"10secs");
        insta::assert_snapshot!(p(1.minute().seconds(40)), @"1min 40secs");

        insta::assert_snapshot!(p(1.minute()), @"1min");
        insta::assert_snapshot!(p(2.minutes()), @"2mins");
        insta::assert_snapshot!(p(10.minutes()), @"10mins");
        insta::assert_snapshot!(p(1.hour().minutes(40)), @"1hr 40mins");

        insta::assert_snapshot!(p(1.hour()), @"1hr");
        insta::assert_snapshot!(p(2.hours()), @"2hrs");
        insta::assert_snapshot!(p(10.hours()), @"10hrs");
        insta::assert_snapshot!(p(100.hours()), @"100hrs");

        insta::assert_snapshot!(
            p(1.hour().minutes(1).seconds(1)),
            @"1hr 1min 1sec",
        );
        insta::assert_snapshot!(
            p(2.hours().minutes(2).seconds(2)),
            @"2hrs 2mins 2secs",
        );
        insta::assert_snapshot!(
            p(10.hours().minutes(10).seconds(10)),
            @"10hrs 10mins 10secs",
        );
        insta::assert_snapshot!(
            p(100.hours().minutes(100).seconds(100)),
            @"100hrs 100mins 100secs",
        );

        insta::assert_snapshot!(p(-1.hour()), @"1hr ago");
        insta::assert_snapshot!(p(-1.hour().seconds(30)), @"1hr 30secs ago");
    }

    #[test]
    fn print_span_designator_compact() {
        let printer = || SpanPrinter::new().designator(Designator::Compact);
        let p = |span| printer().span_to_string(&span);

        insta::assert_snapshot!(p(1.second()), @"1s");
        insta::assert_snapshot!(p(2.seconds()), @"2s");
        insta::assert_snapshot!(p(10.seconds()), @"10s");
        insta::assert_snapshot!(p(1.minute().seconds(40)), @"1m 40s");

        insta::assert_snapshot!(p(1.minute()), @"1m");
        insta::assert_snapshot!(p(2.minutes()), @"2m");
        insta::assert_snapshot!(p(10.minutes()), @"10m");
        insta::assert_snapshot!(p(1.hour().minutes(40)), @"1h 40m");

        insta::assert_snapshot!(p(1.hour()), @"1h");
        insta::assert_snapshot!(p(2.hours()), @"2h");
        insta::assert_snapshot!(p(10.hours()), @"10h");
        insta::assert_snapshot!(p(100.hours()), @"100h");

        insta::assert_snapshot!(
            p(1.hour().minutes(1).seconds(1)),
            @"1h 1m 1s",
        );
        insta::assert_snapshot!(
            p(2.hours().minutes(2).seconds(2)),
            @"2h 2m 2s",
        );
        insta::assert_snapshot!(
            p(10.hours().minutes(10).seconds(10)),
            @"10h 10m 10s",
        );
        insta::assert_snapshot!(
            p(100.hours().minutes(100).seconds(100)),
            @"100h 100m 100s",
        );

        insta::assert_snapshot!(p(-1.hour()), @"1h ago");
        insta::assert_snapshot!(p(-1.hour().seconds(30)), @"1h 30s ago");
    }

    #[test]
    fn print_span_designator_direction_force() {
        let printer = || SpanPrinter::new().direction(Direction::ForceSign);
        let p = |span| printer().span_to_string(&span);

        insta::assert_snapshot!(p(1.second()), @"+1s");
        insta::assert_snapshot!(p(2.seconds()), @"+2s");
        insta::assert_snapshot!(p(10.seconds()), @"+10s");
        insta::assert_snapshot!(p(1.minute().seconds(40)), @"+1m 40s");

        insta::assert_snapshot!(p(1.minute()), @"+1m");
        insta::assert_snapshot!(p(2.minutes()), @"+2m");
        insta::assert_snapshot!(p(10.minutes()), @"+10m");
        insta::assert_snapshot!(p(1.hour().minutes(40)), @"+1h 40m");

        insta::assert_snapshot!(p(1.hour()), @"+1h");
        insta::assert_snapshot!(p(2.hours()), @"+2h");
        insta::assert_snapshot!(p(10.hours()), @"+10h");
        insta::assert_snapshot!(p(100.hours()), @"+100h");

        insta::assert_snapshot!(
            p(1.hour().minutes(1).seconds(1)),
            @"+1h 1m 1s",
        );
        insta::assert_snapshot!(
            p(2.hours().minutes(2).seconds(2)),
            @"+2h 2m 2s",
        );
        insta::assert_snapshot!(
            p(10.hours().minutes(10).seconds(10)),
            @"+10h 10m 10s",
        );
        insta::assert_snapshot!(
            p(100.hours().minutes(100).seconds(100)),
            @"+100h 100m 100s",
        );

        insta::assert_snapshot!(p(-1.hour()), @"-1h");
        insta::assert_snapshot!(p(-1.hour().seconds(30)), @"-1h 30s");
    }

    #[test]
    fn print_span_designator_padding() {
        let printer = || SpanPrinter::new().padding(2);
        let p = |span| printer().span_to_string(&span);

        insta::assert_snapshot!(p(1.second()), @"01s");
        insta::assert_snapshot!(p(2.seconds()), @"02s");
        insta::assert_snapshot!(p(10.seconds()), @"10s");
        insta::assert_snapshot!(p(1.minute().seconds(40)), @"01m 40s");

        insta::assert_snapshot!(p(1.minute()), @"01m");
        insta::assert_snapshot!(p(2.minutes()), @"02m");
        insta::assert_snapshot!(p(10.minutes()), @"10m");
        insta::assert_snapshot!(p(1.hour().minutes(40)), @"01h 40m");

        insta::assert_snapshot!(p(1.hour()), @"01h");
        insta::assert_snapshot!(p(2.hours()), @"02h");
        insta::assert_snapshot!(p(10.hours()), @"10h");
        insta::assert_snapshot!(p(100.hours()), @"100h");

        insta::assert_snapshot!(
            p(1.hour().minutes(1).seconds(1)),
            @"01h 01m 01s",
        );
        insta::assert_snapshot!(
            p(2.hours().minutes(2).seconds(2)),
            @"02h 02m 02s",
        );
        insta::assert_snapshot!(
            p(10.hours().minutes(10).seconds(10)),
            @"10h 10m 10s",
        );
        insta::assert_snapshot!(
            p(100.hours().minutes(100).seconds(100)),
            @"100h 100m 100s",
        );

        insta::assert_snapshot!(p(-1.hour()), @"01h ago");
        insta::assert_snapshot!(p(-1.hour().seconds(30)), @"01h 30s ago");
    }

    #[test]
    fn print_span_designator_spacing_none() {
        let printer = || SpanPrinter::new().spacing(Spacing::None);
        let p = |span| printer().span_to_string(&span);

        insta::assert_snapshot!(p(1.second()), @"1s");
        insta::assert_snapshot!(p(2.seconds()), @"2s");
        insta::assert_snapshot!(p(10.seconds()), @"10s");
        insta::assert_snapshot!(p(1.minute().seconds(40)), @"1m40s");

        insta::assert_snapshot!(p(1.minute()), @"1m");
        insta::assert_snapshot!(p(2.minutes()), @"2m");
        insta::assert_snapshot!(p(10.minutes()), @"10m");
        insta::assert_snapshot!(p(1.hour().minutes(40)), @"1h40m");

        insta::assert_snapshot!(p(1.hour()), @"1h");
        insta::assert_snapshot!(p(2.hours()), @"2h");
        insta::assert_snapshot!(p(10.hours()), @"10h");
        insta::assert_snapshot!(p(100.hours()), @"100h");

        insta::assert_snapshot!(
            p(1.hour().minutes(1).seconds(1)),
            @"1h1m1s",
        );
        insta::assert_snapshot!(
            p(2.hours().minutes(2).seconds(2)),
            @"2h2m2s",
        );
        insta::assert_snapshot!(
            p(10.hours().minutes(10).seconds(10)),
            @"10h10m10s",
        );
        insta::assert_snapshot!(
            p(100.hours().minutes(100).seconds(100)),
            @"100h100m100s",
        );

        insta::assert_snapshot!(p(-1.hour()), @"-1h");
        insta::assert_snapshot!(p(-1.hour().seconds(30)), @"-1h30s");
    }

    #[test]
    fn print_span_designator_spacing_more() {
        let printer =
            || SpanPrinter::new().spacing(Spacing::BetweenUnitsAndDesignators);
        let p = |span| printer().span_to_string(&span);

        insta::assert_snapshot!(p(1.second()), @"1 s");
        insta::assert_snapshot!(p(2.seconds()), @"2 s");
        insta::assert_snapshot!(p(10.seconds()), @"10 s");
        insta::assert_snapshot!(p(1.minute().seconds(40)), @"1 m 40 s");

        insta::assert_snapshot!(p(1.minute()), @"1 m");
        insta::assert_snapshot!(p(2.minutes()), @"2 m");
        insta::assert_snapshot!(p(10.minutes()), @"10 m");
        insta::assert_snapshot!(p(1.hour().minutes(40)), @"1 h 40 m");

        insta::assert_snapshot!(p(1.hour()), @"1 h");
        insta::assert_snapshot!(p(2.hours()), @"2 h");
        insta::assert_snapshot!(p(10.hours()), @"10 h");
        insta::assert_snapshot!(p(100.hours()), @"100 h");

        insta::assert_snapshot!(
            p(1.hour().minutes(1).seconds(1)),
            @"1 h 1 m 1 s",
        );
        insta::assert_snapshot!(
            p(2.hours().minutes(2).seconds(2)),
            @"2 h 2 m 2 s",
        );
        insta::assert_snapshot!(
            p(10.hours().minutes(10).seconds(10)),
            @"10 h 10 m 10 s",
        );
        insta::assert_snapshot!(
            p(100.hours().minutes(100).seconds(100)),
            @"100 h 100 m 100 s",
        );

        insta::assert_snapshot!(p(-1.hour()), @"1 h ago");
        insta::assert_snapshot!(p(-1.hour().seconds(30)), @"1 h 30 s ago");
    }

    #[test]
    fn print_span_designator_spacing_comma() {
        let printer = || {
            SpanPrinter::new()
                .comma_after_designator(true)
                .spacing(Spacing::BetweenUnitsAndDesignators)
        };
        let p = |span| printer().span_to_string(&span);

        insta::assert_snapshot!(p(1.second()), @"1 s");
        insta::assert_snapshot!(p(2.seconds()), @"2 s");
        insta::assert_snapshot!(p(10.seconds()), @"10 s");
        insta::assert_snapshot!(p(1.minute().seconds(40)), @"1 m, 40 s");

        insta::assert_snapshot!(p(1.minute()), @"1 m");
        insta::assert_snapshot!(p(2.minutes()), @"2 m");
        insta::assert_snapshot!(p(10.minutes()), @"10 m");
        insta::assert_snapshot!(p(1.hour().minutes(40)), @"1 h, 40 m");

        insta::assert_snapshot!(p(1.hour()), @"1 h");
        insta::assert_snapshot!(p(2.hours()), @"2 h");
        insta::assert_snapshot!(p(10.hours()), @"10 h");
        insta::assert_snapshot!(p(100.hours()), @"100 h");

        insta::assert_snapshot!(
            p(1.hour().minutes(1).seconds(1)),
            @"1 h, 1 m, 1 s",
        );
        insta::assert_snapshot!(
            p(2.hours().minutes(2).seconds(2)),
            @"2 h, 2 m, 2 s",
        );
        insta::assert_snapshot!(
            p(10.hours().minutes(10).seconds(10)),
            @"10 h, 10 m, 10 s",
        );
        insta::assert_snapshot!(
            p(100.hours().minutes(100).seconds(100)),
            @"100 h, 100 m, 100 s",
        );

        insta::assert_snapshot!(p(-1.hour()), @"1 h ago");
        insta::assert_snapshot!(p(-1.hour().seconds(30)), @"1 h, 30 s ago");
    }

    #[test]
    fn print_span_designator_fractional_hour() {
        let printer =
            || SpanPrinter::new().fractional(Some(FractionalUnit::Hour));
        let p = |span| printer().span_to_string(&span);
        let pp = |precision, span| {
            printer().precision(Some(precision)).span_to_string(&span)
        };

        insta::assert_snapshot!(p(1.hour()), @"1h");
        insta::assert_snapshot!(pp(0, 1.hour()), @"1h");
        insta::assert_snapshot!(pp(1, 1.hour()), @"1.0h");
        insta::assert_snapshot!(pp(2, 1.hour()), @"1.00h");

        insta::assert_snapshot!(p(1.hour().minutes(30)), @"1.5h");
        insta::assert_snapshot!(pp(0, 1.hour().minutes(30)), @"1h");
        insta::assert_snapshot!(pp(1, 1.hour().minutes(30)), @"1.5h");
        insta::assert_snapshot!(pp(2, 1.hour().minutes(30)), @"1.50h");

        insta::assert_snapshot!(p(1.hour().minutes(3)), @"1.05h");
        insta::assert_snapshot!(p(1.hour().minutes(3).nanoseconds(1)), @"1.05h");
        insta::assert_snapshot!(p(1.second()), @"0.000277777h");
        // precision loss!
        insta::assert_snapshot!(p(1.second().nanoseconds(1)), @"0.000277777h");
        insta::assert_snapshot!(p(0.seconds()), @"0h");
        // precision loss!
        insta::assert_snapshot!(p(1.nanosecond()), @"0h");
    }

    #[test]
    fn print_span_designator_fractional_minute() {
        let printer =
            || SpanPrinter::new().fractional(Some(FractionalUnit::Minute));
        let p = |span| printer().span_to_string(&span);
        let pp = |precision, span| {
            printer().precision(Some(precision)).span_to_string(&span)
        };

        insta::assert_snapshot!(p(1.hour()), @"1h");
        insta::assert_snapshot!(p(1.hour().minutes(30)), @"1h 30m");

        insta::assert_snapshot!(p(1.minute()), @"1m");
        insta::assert_snapshot!(pp(0, 1.minute()), @"1m");
        insta::assert_snapshot!(pp(1, 1.minute()), @"1.0m");
        insta::assert_snapshot!(pp(2, 1.minute()), @"1.00m");

        insta::assert_snapshot!(p(1.minute().seconds(30)), @"1.5m");
        insta::assert_snapshot!(pp(0, 1.minute().seconds(30)), @"1m");
        insta::assert_snapshot!(pp(1, 1.minute().seconds(30)), @"1.5m");
        insta::assert_snapshot!(pp(2, 1.minute().seconds(30)), @"1.50m");

        insta::assert_snapshot!(p(1.hour().nanoseconds(1)), @"1h");
        insta::assert_snapshot!(p(1.minute().seconds(3)), @"1.05m");
        insta::assert_snapshot!(p(1.minute().seconds(3).nanoseconds(1)), @"1.05m");
        insta::assert_snapshot!(p(1.second()), @"0.016666666m");
        // precision loss!
        insta::assert_snapshot!(p(1.second().nanoseconds(1)), @"0.016666666m");
        insta::assert_snapshot!(p(0.seconds()), @"0m");
        // precision loss!
        insta::assert_snapshot!(p(1.nanosecond()), @"0m");
    }

    #[test]
    fn print_span_designator_fractional_second() {
        let printer =
            || SpanPrinter::new().fractional(Some(FractionalUnit::Second));
        let p = |span| printer().span_to_string(&span);
        let pp = |precision, span| {
            printer().precision(Some(precision)).span_to_string(&span)
        };

        insta::assert_snapshot!(p(1.hour()), @"1h");
        insta::assert_snapshot!(p(1.hour().minutes(30)), @"1h 30m");

        insta::assert_snapshot!(p(1.second()), @"1s");
        insta::assert_snapshot!(pp(0, 1.second()), @"1s");
        insta::assert_snapshot!(pp(1, 1.second()), @"1.0s");
        insta::assert_snapshot!(pp(2, 1.second()), @"1.00s");

        insta::assert_snapshot!(p(1.second().milliseconds(500)), @"1.5s");
        insta::assert_snapshot!(pp(0, 1.second().milliseconds(500)), @"1s");
        insta::assert_snapshot!(pp(1, 1.second().milliseconds(500)), @"1.5s");
        insta::assert_snapshot!(pp(2, 1.second().milliseconds(500)), @"1.50s");

        insta::assert_snapshot!(p(1.second().nanoseconds(1)), @"1.000000001s");
        insta::assert_snapshot!(p(1.nanosecond()), @"0.000000001s");
        insta::assert_snapshot!(p(0.seconds()), @"0s");

        insta::assert_snapshot!(p(1.second().milliseconds(2000)), @"3s");
    }

    #[test]
    fn print_span_designator_fractional_millisecond() {
        let printer = || {
            SpanPrinter::new().fractional(Some(FractionalUnit::Millisecond))
        };
        let p = |span| printer().span_to_string(&span);
        let pp = |precision, span| {
            printer().precision(Some(precision)).span_to_string(&span)
        };

        insta::assert_snapshot!(p(1.hour()), @"1h");
        insta::assert_snapshot!(p(1.hour().minutes(30)), @"1h 30m");
        insta::assert_snapshot!(
            p(1.hour().minutes(30).seconds(10)),
            @"1h 30m 10s",
        );

        insta::assert_snapshot!(p(1.second()), @"1s");
        insta::assert_snapshot!(pp(0, 1.second()), @"1s");
        insta::assert_snapshot!(pp(1, 1.second()), @"1s 0.0ms");
        insta::assert_snapshot!(pp(2, 1.second()), @"1s 0.00ms");

        insta::assert_snapshot!(p(1.second().milliseconds(500)), @"1s 500ms");
        insta::assert_snapshot!(
            pp(0, 1.second().milliseconds(1).microseconds(500)),
            @"1s 1ms",
        );
        insta::assert_snapshot!(
            pp(1, 1.second().milliseconds(1).microseconds(500)),
            @"1s 1.5ms",
        );
        insta::assert_snapshot!(
            pp(2, 1.second().milliseconds(1).microseconds(500)),
            @"1s 1.50ms",
        );

        insta::assert_snapshot!(p(1.millisecond().nanoseconds(1)), @"1.000001ms");
        insta::assert_snapshot!(p(1.nanosecond()), @"0.000001ms");
        insta::assert_snapshot!(p(0.seconds()), @"0ms");
    }

    #[test]
    fn print_span_designator_fractional_microsecond() {
        let printer = || {
            SpanPrinter::new().fractional(Some(FractionalUnit::Microsecond))
        };
        let p = |span| printer().span_to_string(&span);
        let pp = |precision, span| {
            printer().precision(Some(precision)).span_to_string(&span)
        };

        insta::assert_snapshot!(p(1.hour()), @"1h");
        insta::assert_snapshot!(p(1.hour().minutes(30)), @"1h 30m");
        insta::assert_snapshot!(
            p(1.hour().minutes(30).seconds(10)),
            @"1h 30m 10s",
        );

        insta::assert_snapshot!(p(1.second()), @"1s");
        insta::assert_snapshot!(pp(0, 1.second()), @"1s");
        insta::assert_snapshot!(pp(1, 1.second()), @"1s 0.0µs");
        insta::assert_snapshot!(pp(2, 1.second()), @"1s 0.00µs");

        insta::assert_snapshot!(p(1.second().milliseconds(500)), @"1s 500ms");
        insta::assert_snapshot!(
            pp(0, 1.second().milliseconds(1).microseconds(500)),
            @"1s 1ms 500µs",
        );
        insta::assert_snapshot!(
            pp(1, 1.second().milliseconds(1).microseconds(500)),
            @"1s 1ms 500.0µs",
        );
        insta::assert_snapshot!(
            pp(2, 1.second().milliseconds(1).microseconds(500)),
            @"1s 1ms 500.00µs",
        );

        insta::assert_snapshot!(
            p(1.millisecond().nanoseconds(1)),
            @"1ms 0.001µs",
        );
        insta::assert_snapshot!(p(1.nanosecond()), @"0.001µs");
        insta::assert_snapshot!(p(0.second()), @"0µs");
    }

    #[test]
    fn print_signed_duration_designator_default() {
        let printer = || SpanPrinter::new();
        let p = |secs| {
            printer().duration_to_string(&SignedDuration::from_secs(secs))
        };

        insta::assert_snapshot!(p(1), @"1s");
        insta::assert_snapshot!(p(2), @"2s");
        insta::assert_snapshot!(p(10), @"10s");
        insta::assert_snapshot!(p(100), @"1m 40s");

        insta::assert_snapshot!(p(1 * 60), @"1m");
        insta::assert_snapshot!(p(2 * 60), @"2m");
        insta::assert_snapshot!(p(10 * 60), @"10m");
        insta::assert_snapshot!(p(100 * 60), @"1h 40m");

        insta::assert_snapshot!(p(1 * 60 * 60), @"1h");
        insta::assert_snapshot!(p(2 * 60 * 60), @"2h");
        insta::assert_snapshot!(p(10 * 60 * 60), @"10h");
        insta::assert_snapshot!(p(100 * 60 * 60), @"100h");

        insta::assert_snapshot!(
            p(60 * 60 + 60 + 1),
            @"1h 1m 1s",
        );
        insta::assert_snapshot!(
            p(2 * 60 * 60 + 2 * 60 + 2),
            @"2h 2m 2s",
        );
        insta::assert_snapshot!(
            p(10 * 60 * 60 + 10 * 60 + 10),
            @"10h 10m 10s",
        );
        insta::assert_snapshot!(
            p(100 * 60 * 60 + 100 * 60 + 100),
            @"101h 41m 40s",
        );

        insta::assert_snapshot!(p(-1 * 60 * 60), @"1h ago");
        insta::assert_snapshot!(p(-1 * 60 * 60 - 30), @"1h 30s ago");
    }

    #[test]
    fn print_signed_duration_designator_verbose() {
        let printer = || SpanPrinter::new().designator(Designator::Verbose);
        let p = |secs| {
            printer().duration_to_string(&SignedDuration::from_secs(secs))
        };

        insta::assert_snapshot!(p(1), @"1second");
        insta::assert_snapshot!(p(2), @"2seconds");
        insta::assert_snapshot!(p(10), @"10seconds");
        insta::assert_snapshot!(p(100), @"1minute 40seconds");

        insta::assert_snapshot!(p(1 * 60), @"1minute");
        insta::assert_snapshot!(p(2 * 60), @"2minutes");
        insta::assert_snapshot!(p(10 * 60), @"10minutes");
        insta::assert_snapshot!(p(100 * 60), @"1hour 40minutes");

        insta::assert_snapshot!(p(1 * 60 * 60), @"1hour");
        insta::assert_snapshot!(p(2 * 60 * 60), @"2hours");
        insta::assert_snapshot!(p(10 * 60 * 60), @"10hours");
        insta::assert_snapshot!(p(100 * 60 * 60), @"100hours");

        insta::assert_snapshot!(
            p(60 * 60 + 60 + 1),
            @"1hour 1minute 1second",
        );
        insta::assert_snapshot!(
            p(2 * 60 * 60 + 2 * 60 + 2),
            @"2hours 2minutes 2seconds",
        );
        insta::assert_snapshot!(
            p(10 * 60 * 60 + 10 * 60 + 10),
            @"10hours 10minutes 10seconds",
        );
        insta::assert_snapshot!(
            p(100 * 60 * 60 + 100 * 60 + 100),
            @"101hours 41minutes 40seconds",
        );

        insta::assert_snapshot!(p(-1 * 60 * 60), @"1hour ago");
        insta::assert_snapshot!(p(-1 * 60 * 60 - 30), @"1hour 30seconds ago");
    }

    #[test]
    fn print_signed_duration_designator_short() {
        let printer = || SpanPrinter::new().designator(Designator::Short);
        let p = |secs| {
            printer().duration_to_string(&SignedDuration::from_secs(secs))
        };

        insta::assert_snapshot!(p(1), @"1sec");
        insta::assert_snapshot!(p(2), @"2secs");
        insta::assert_snapshot!(p(10), @"10secs");
        insta::assert_snapshot!(p(100), @"1min 40secs");

        insta::assert_snapshot!(p(1 * 60), @"1min");
        insta::assert_snapshot!(p(2 * 60), @"2mins");
        insta::assert_snapshot!(p(10 * 60), @"10mins");
        insta::assert_snapshot!(p(100 * 60), @"1hr 40mins");

        insta::assert_snapshot!(p(1 * 60 * 60), @"1hr");
        insta::assert_snapshot!(p(2 * 60 * 60), @"2hrs");
        insta::assert_snapshot!(p(10 * 60 * 60), @"10hrs");
        insta::assert_snapshot!(p(100 * 60 * 60), @"100hrs");

        insta::assert_snapshot!(
            p(60 * 60 + 60 + 1),
            @"1hr 1min 1sec",
        );
        insta::assert_snapshot!(
            p(2 * 60 * 60 + 2 * 60 + 2),
            @"2hrs 2mins 2secs",
        );
        insta::assert_snapshot!(
            p(10 * 60 * 60 + 10 * 60 + 10),
            @"10hrs 10mins 10secs",
        );
        insta::assert_snapshot!(
            p(100 * 60 * 60 + 100 * 60 + 100),
            @"101hrs 41mins 40secs",
        );

        insta::assert_snapshot!(p(-1 * 60 * 60), @"1hr ago");
        insta::assert_snapshot!(p(-1 * 60 * 60 - 30), @"1hr 30secs ago");
    }

    #[test]
    fn print_signed_duration_designator_compact() {
        let printer = || SpanPrinter::new().designator(Designator::Compact);
        let p = |secs| {
            printer().duration_to_string(&SignedDuration::from_secs(secs))
        };

        insta::assert_snapshot!(p(1), @"1s");
        insta::assert_snapshot!(p(2), @"2s");
        insta::assert_snapshot!(p(10), @"10s");
        insta::assert_snapshot!(p(100), @"1m 40s");

        insta::assert_snapshot!(p(1 * 60), @"1m");
        insta::assert_snapshot!(p(2 * 60), @"2m");
        insta::assert_snapshot!(p(10 * 60), @"10m");
        insta::assert_snapshot!(p(100 * 60), @"1h 40m");

        insta::assert_snapshot!(p(1 * 60 * 60), @"1h");
        insta::assert_snapshot!(p(2 * 60 * 60), @"2h");
        insta::assert_snapshot!(p(10 * 60 * 60), @"10h");
        insta::assert_snapshot!(p(100 * 60 * 60), @"100h");

        insta::assert_snapshot!(
            p(60 * 60 + 60 + 1),
            @"1h 1m 1s",
        );
        insta::assert_snapshot!(
            p(2 * 60 * 60 + 2 * 60 + 2),
            @"2h 2m 2s",
        );
        insta::assert_snapshot!(
            p(10 * 60 * 60 + 10 * 60 + 10),
            @"10h 10m 10s",
        );
        insta::assert_snapshot!(
            p(100 * 60 * 60 + 100 * 60 + 100),
            @"101h 41m 40s",
        );

        insta::assert_snapshot!(p(-1 * 60 * 60), @"1h ago");
        insta::assert_snapshot!(p(-1 * 60 * 60 - 30), @"1h 30s ago");
    }

    #[test]
    fn print_signed_duration_designator_direction_force() {
        let printer = || SpanPrinter::new().direction(Direction::ForceSign);
        let p = |secs| {
            printer().duration_to_string(&SignedDuration::from_secs(secs))
        };

        insta::assert_snapshot!(p(1), @"+1s");
        insta::assert_snapshot!(p(2), @"+2s");
        insta::assert_snapshot!(p(10), @"+10s");
        insta::assert_snapshot!(p(100), @"+1m 40s");

        insta::assert_snapshot!(p(1 * 60), @"+1m");
        insta::assert_snapshot!(p(2 * 60), @"+2m");
        insta::assert_snapshot!(p(10 * 60), @"+10m");
        insta::assert_snapshot!(p(100 * 60), @"+1h 40m");

        insta::assert_snapshot!(p(1 * 60 * 60), @"+1h");
        insta::assert_snapshot!(p(2 * 60 * 60), @"+2h");
        insta::assert_snapshot!(p(10 * 60 * 60), @"+10h");
        insta::assert_snapshot!(p(100 * 60 * 60), @"+100h");

        insta::assert_snapshot!(
            p(60 * 60 + 60 + 1),
            @"+1h 1m 1s",
        );
        insta::assert_snapshot!(
            p(2 * 60 * 60 + 2 * 60 + 2),
            @"+2h 2m 2s",
        );
        insta::assert_snapshot!(
            p(10 * 60 * 60 + 10 * 60 + 10),
            @"+10h 10m 10s",
        );
        insta::assert_snapshot!(
            p(100 * 60 * 60 + 100 * 60 + 100),
            @"+101h 41m 40s",
        );

        insta::assert_snapshot!(p(-1 * 60 * 60), @"-1h");
        insta::assert_snapshot!(p(-1 * 60 * 60 - 30), @"-1h 30s");
    }

    #[test]
    fn print_signed_duration_designator_padding() {
        let printer = || SpanPrinter::new().padding(2);
        let p = |secs| {
            printer().duration_to_string(&SignedDuration::from_secs(secs))
        };

        insta::assert_snapshot!(p(1), @"01s");
        insta::assert_snapshot!(p(2), @"02s");
        insta::assert_snapshot!(p(10), @"10s");
        insta::assert_snapshot!(p(100), @"01m 40s");

        insta::assert_snapshot!(p(1 * 60), @"01m");
        insta::assert_snapshot!(p(2 * 60), @"02m");
        insta::assert_snapshot!(p(10 * 60), @"10m");
        insta::assert_snapshot!(p(100 * 60), @"01h 40m");

        insta::assert_snapshot!(p(1 * 60 * 60), @"01h");
        insta::assert_snapshot!(p(2 * 60 * 60), @"02h");
        insta::assert_snapshot!(p(10 * 60 * 60), @"10h");
        insta::assert_snapshot!(p(100 * 60 * 60), @"100h");

        insta::assert_snapshot!(
            p(60 * 60 + 60 + 1),
            @"01h 01m 01s",
        );
        insta::assert_snapshot!(
            p(2 * 60 * 60 + 2 * 60 + 2),
            @"02h 02m 02s",
        );
        insta::assert_snapshot!(
            p(10 * 60 * 60 + 10 * 60 + 10),
            @"10h 10m 10s",
        );
        insta::assert_snapshot!(
            p(100 * 60 * 60 + 100 * 60 + 100),
            @"101h 41m 40s",
        );

        insta::assert_snapshot!(p(-1 * 60 * 60), @"01h ago");
        insta::assert_snapshot!(p(-1 * 60 * 60 - 30), @"01h 30s ago");
    }

    #[test]
    fn print_signed_duration_designator_spacing_none() {
        let printer = || SpanPrinter::new().spacing(Spacing::None);
        let p = |secs| {
            printer().duration_to_string(&SignedDuration::from_secs(secs))
        };

        insta::assert_snapshot!(p(1), @"1s");
        insta::assert_snapshot!(p(2), @"2s");
        insta::assert_snapshot!(p(10), @"10s");
        insta::assert_snapshot!(p(100), @"1m40s");

        insta::assert_snapshot!(p(1 * 60), @"1m");
        insta::assert_snapshot!(p(2 * 60), @"2m");
        insta::assert_snapshot!(p(10 * 60), @"10m");
        insta::assert_snapshot!(p(100 * 60), @"1h40m");

        insta::assert_snapshot!(p(1 * 60 * 60), @"1h");
        insta::assert_snapshot!(p(2 * 60 * 60), @"2h");
        insta::assert_snapshot!(p(10 * 60 * 60), @"10h");
        insta::assert_snapshot!(p(100 * 60 * 60), @"100h");

        insta::assert_snapshot!(
            p(60 * 60 + 60 + 1),
            @"1h1m1s",
        );
        insta::assert_snapshot!(
            p(2 * 60 * 60 + 2 * 60 + 2),
            @"2h2m2s",
        );
        insta::assert_snapshot!(
            p(10 * 60 * 60 + 10 * 60 + 10),
            @"10h10m10s",
        );
        insta::assert_snapshot!(
            p(100 * 60 * 60 + 100 * 60 + 100),
            @"101h41m40s",
        );

        insta::assert_snapshot!(p(-1 * 60 * 60), @"-1h");
        insta::assert_snapshot!(p(-1 * 60 * 60 - 30), @"-1h30s");
    }

    #[test]
    fn print_signed_duration_designator_spacing_more() {
        let printer =
            || SpanPrinter::new().spacing(Spacing::BetweenUnitsAndDesignators);
        let p = |secs| {
            printer().duration_to_string(&SignedDuration::from_secs(secs))
        };

        insta::assert_snapshot!(p(1), @"1 s");
        insta::assert_snapshot!(p(2), @"2 s");
        insta::assert_snapshot!(p(10), @"10 s");
        insta::assert_snapshot!(p(100), @"1 m 40 s");

        insta::assert_snapshot!(p(1 * 60), @"1 m");
        insta::assert_snapshot!(p(2 * 60), @"2 m");
        insta::assert_snapshot!(p(10 * 60), @"10 m");
        insta::assert_snapshot!(p(100 * 60), @"1 h 40 m");

        insta::assert_snapshot!(p(1 * 60 * 60), @"1 h");
        insta::assert_snapshot!(p(2 * 60 * 60), @"2 h");
        insta::assert_snapshot!(p(10 * 60 * 60), @"10 h");
        insta::assert_snapshot!(p(100 * 60 * 60), @"100 h");

        insta::assert_snapshot!(
            p(60 * 60 + 60 + 1),
            @"1 h 1 m 1 s",
        );
        insta::assert_snapshot!(
            p(2 * 60 * 60 + 2 * 60 + 2),
            @"2 h 2 m 2 s",
        );
        insta::assert_snapshot!(
            p(10 * 60 * 60 + 10 * 60 + 10),
            @"10 h 10 m 10 s",
        );
        insta::assert_snapshot!(
            p(100 * 60 * 60 + 100 * 60 + 100),
            @"101 h 41 m 40 s",
        );

        insta::assert_snapshot!(p(-1 * 60 * 60), @"1 h ago");
        insta::assert_snapshot!(p(-1 * 60 * 60 - 30), @"1 h 30 s ago");
    }

    #[test]
    fn print_signed_duration_designator_spacing_comma() {
        let printer = || {
            SpanPrinter::new()
                .comma_after_designator(true)
                .spacing(Spacing::BetweenUnitsAndDesignators)
        };
        let p = |secs| {
            printer().duration_to_string(&SignedDuration::from_secs(secs))
        };

        insta::assert_snapshot!(p(1), @"1 s");
        insta::assert_snapshot!(p(2), @"2 s");
        insta::assert_snapshot!(p(10), @"10 s");
        insta::assert_snapshot!(p(100), @"1 m, 40 s");

        insta::assert_snapshot!(p(1 * 60), @"1 m");
        insta::assert_snapshot!(p(2 * 60), @"2 m");
        insta::assert_snapshot!(p(10 * 60), @"10 m");
        insta::assert_snapshot!(p(100 * 60), @"1 h, 40 m");

        insta::assert_snapshot!(p(1 * 60 * 60), @"1 h");
        insta::assert_snapshot!(p(2 * 60 * 60), @"2 h");
        insta::assert_snapshot!(p(10 * 60 * 60), @"10 h");
        insta::assert_snapshot!(p(100 * 60 * 60), @"100 h");

        insta::assert_snapshot!(
            p(60 * 60 + 60 + 1),
            @"1 h, 1 m, 1 s",
        );
        insta::assert_snapshot!(
            p(2 * 60 * 60 + 2 * 60 + 2),
            @"2 h, 2 m, 2 s",
        );
        insta::assert_snapshot!(
            p(10 * 60 * 60 + 10 * 60 + 10),
            @"10 h, 10 m, 10 s",
        );
        insta::assert_snapshot!(
            p(100 * 60 * 60 + 100 * 60 + 100),
            @"101 h, 41 m, 40 s",
        );

        insta::assert_snapshot!(p(-1 * 60 * 60), @"1 h ago");
        insta::assert_snapshot!(p(-1 * 60 * 60 - 30), @"1 h, 30 s ago");
    }

    #[test]
    fn print_signed_duration_designator_fractional_hour() {
        let printer =
            || SpanPrinter::new().fractional(Some(FractionalUnit::Hour));
        let p = |secs, nanos| {
            printer().duration_to_string(&SignedDuration::new(secs, nanos))
        };
        let pp = |precision, secs, nanos| {
            printer()
                .precision(Some(precision))
                .duration_to_string(&SignedDuration::new(secs, nanos))
        };

        insta::assert_snapshot!(p(1 * 60 * 60, 0), @"1h");
        insta::assert_snapshot!(pp(0, 1 * 60 * 60, 0), @"1h");
        insta::assert_snapshot!(pp(1, 1 * 60 * 60, 0), @"1.0h");
        insta::assert_snapshot!(pp(2, 1 * 60 * 60, 0), @"1.00h");

        insta::assert_snapshot!(p(1 * 60 * 60 + 30 * 60, 0), @"1.5h");
        insta::assert_snapshot!(pp(0, 1 * 60 * 60 + 30 * 60, 0), @"1h");
        insta::assert_snapshot!(pp(1, 1 * 60 * 60 + 30 * 60, 0), @"1.5h");
        insta::assert_snapshot!(pp(2, 1 * 60 * 60 + 30 * 60, 0), @"1.50h");

        insta::assert_snapshot!(p(1 * 60 * 60 + 3 * 60, 0), @"1.05h");
        insta::assert_snapshot!(p(1 * 60 * 60 + 3 * 60, 1), @"1.05h");
        insta::assert_snapshot!(p(1, 0), @"0.000277777h");
        // precision loss!
        insta::assert_snapshot!(p(1, 1), @"0.000277777h");
        insta::assert_snapshot!(p(0, 0), @"0h");
        // precision loss!
        insta::assert_snapshot!(p(0, 1), @"0h");

        insta::assert_snapshot!(
            printer().duration_to_string(&SignedDuration::MIN),
            @"2562047788015215.502499999h ago",
        );
    }

    #[test]
    fn print_signed_duration_designator_fractional_minute() {
        let printer =
            || SpanPrinter::new().fractional(Some(FractionalUnit::Minute));
        let p = |secs, nanos| {
            printer().duration_to_string(&SignedDuration::new(secs, nanos))
        };
        let pp = |precision, secs, nanos| {
            printer()
                .precision(Some(precision))
                .duration_to_string(&SignedDuration::new(secs, nanos))
        };

        insta::assert_snapshot!(p(1 * 60 * 60, 0), @"1h");
        insta::assert_snapshot!(p(1 * 60 * 60 + 30 * 60, 0), @"1h 30m");

        insta::assert_snapshot!(p(60, 0), @"1m");
        insta::assert_snapshot!(pp(0, 60, 0), @"1m");
        insta::assert_snapshot!(pp(1, 60, 0), @"1.0m");
        insta::assert_snapshot!(pp(2, 60, 0), @"1.00m");

        insta::assert_snapshot!(p(90, 0), @"1.5m");
        insta::assert_snapshot!(pp(0, 90, 0), @"1m");
        insta::assert_snapshot!(pp(1, 90, 0), @"1.5m");
        insta::assert_snapshot!(pp(2, 90, 0), @"1.50m");

        insta::assert_snapshot!(p(1 * 60 * 60, 1), @"1h");
        insta::assert_snapshot!(p(63, 0), @"1.05m");
        insta::assert_snapshot!(p(63, 1), @"1.05m");
        insta::assert_snapshot!(p(1, 0), @"0.016666666m");
        // precision loss!
        insta::assert_snapshot!(p(1, 1), @"0.016666666m");
        insta::assert_snapshot!(p(0, 0), @"0m");
        // precision loss!
        insta::assert_snapshot!(p(0, 1), @"0m");

        insta::assert_snapshot!(
            printer().duration_to_string(&SignedDuration::MIN),
            @"2562047788015215h 30.149999999m ago",
        );
    }

    #[test]
    fn print_signed_duration_designator_fractional_second() {
        let printer =
            || SpanPrinter::new().fractional(Some(FractionalUnit::Second));
        let p = |secs, nanos| {
            printer().duration_to_string(&SignedDuration::new(secs, nanos))
        };
        let pp = |precision, secs, nanos| {
            printer()
                .precision(Some(precision))
                .duration_to_string(&SignedDuration::new(secs, nanos))
        };

        insta::assert_snapshot!(p(1 * 60 * 60, 0), @"1h");
        insta::assert_snapshot!(p(1 * 60 * 60 + 30 * 60, 0), @"1h 30m");

        insta::assert_snapshot!(p(1, 0), @"1s");
        insta::assert_snapshot!(pp(0, 1, 0), @"1s");
        insta::assert_snapshot!(pp(1, 1, 0), @"1.0s");
        insta::assert_snapshot!(pp(2, 1, 0), @"1.00s");

        insta::assert_snapshot!(p(1, 500_000_000), @"1.5s");
        insta::assert_snapshot!(pp(0, 1, 500_000_000), @"1s");
        insta::assert_snapshot!(pp(1, 1, 500_000_000), @"1.5s");
        insta::assert_snapshot!(pp(2, 1, 500_000_000), @"1.50s");

        insta::assert_snapshot!(p(1, 1), @"1.000000001s");
        insta::assert_snapshot!(p(0, 1), @"0.000000001s");
        insta::assert_snapshot!(p(0, 0), @"0s");

        insta::assert_snapshot!(
            printer().duration_to_string(&SignedDuration::MIN),
            @"2562047788015215h 30m 8.999999999s ago",
        );
    }

    #[test]
    fn print_signed_duration_designator_fractional_millisecond() {
        let printer = || {
            SpanPrinter::new().fractional(Some(FractionalUnit::Millisecond))
        };
        let p = |secs, nanos| {
            printer().duration_to_string(&SignedDuration::new(secs, nanos))
        };
        let pp = |precision, secs, nanos| {
            printer()
                .precision(Some(precision))
                .duration_to_string(&SignedDuration::new(secs, nanos))
        };

        insta::assert_snapshot!(p(1 * 60 * 60, 0), @"1h");
        insta::assert_snapshot!(p(1 * 60 * 60 + 30 * 60, 0), @"1h 30m");
        insta::assert_snapshot!(
            p(1 * 60 * 60 + 30 * 60 + 10, 0),
            @"1h 30m 10s",
        );

        insta::assert_snapshot!(p(1, 0), @"1s");
        insta::assert_snapshot!(pp(0, 1, 0), @"1s");
        insta::assert_snapshot!(pp(1, 1, 0), @"1s 0.0ms");
        insta::assert_snapshot!(pp(2, 1, 0), @"1s 0.00ms");

        insta::assert_snapshot!(p(1, 500_000_000), @"1s 500ms");
        insta::assert_snapshot!(pp(0, 1, 1_500_000), @"1s 1ms");
        insta::assert_snapshot!(pp(1, 1, 1_500_000), @"1s 1.5ms");
        insta::assert_snapshot!(pp(2, 1, 1_500_000), @"1s 1.50ms");

        insta::assert_snapshot!(p(0, 1_000_001), @"1.000001ms");
        insta::assert_snapshot!(p(0, 0_000_001), @"0.000001ms");
        insta::assert_snapshot!(p(0, 0), @"0ms");

        insta::assert_snapshot!(
            printer().duration_to_string(&SignedDuration::MIN),
            @"2562047788015215h 30m 8s 999.999999ms ago",
        );
    }

    #[test]
    fn print_signed_duration_designator_fractional_microsecond() {
        let printer = || {
            SpanPrinter::new().fractional(Some(FractionalUnit::Microsecond))
        };
        let p = |secs, nanos| {
            printer().duration_to_string(&SignedDuration::new(secs, nanos))
        };
        let pp = |precision, secs, nanos| {
            printer()
                .precision(Some(precision))
                .duration_to_string(&SignedDuration::new(secs, nanos))
        };

        insta::assert_snapshot!(p(1 * 60 * 60, 0), @"1h");
        insta::assert_snapshot!(p(1 * 60 * 60 + 30 * 60, 0), @"1h 30m");
        insta::assert_snapshot!(
            p(1 * 60 * 60 + 30 * 60 + 10, 0),
            @"1h 30m 10s",
        );

        insta::assert_snapshot!(p(1, 0), @"1s");
        insta::assert_snapshot!(pp(0, 1, 0), @"1s");
        insta::assert_snapshot!(pp(1, 1, 0), @"1s 0.0µs");
        insta::assert_snapshot!(pp(2, 1, 0), @"1s 0.00µs");

        insta::assert_snapshot!(p(1, 500_000_000), @"1s 500ms");
        insta::assert_snapshot!(pp(0, 1, 1_500_000), @"1s 1ms 500µs");
        insta::assert_snapshot!(pp(1, 1, 1_500_000), @"1s 1ms 500.0µs");
        insta::assert_snapshot!(pp(2, 1, 1_500_000), @"1s 1ms 500.00µs");

        insta::assert_snapshot!(p(0, 1_000_001), @"1ms 0.001µs");
        insta::assert_snapshot!(p(0, 0_000_001), @"0.001µs");
        insta::assert_snapshot!(p(0, 0), @"0µs");

        insta::assert_snapshot!(
            printer().duration_to_string(&SignedDuration::MIN),
            @"2562047788015215h 30m 8s 999ms 999.999µs ago",
        );
    }

    #[test]
    fn print_unsigned_duration_designator_default() {
        let printer = || SpanPrinter::new();
        let p = |secs| {
            printer().unsigned_duration_to_string(
                &core::time::Duration::from_secs(secs),
            )
        };

        insta::assert_snapshot!(p(1), @"1s");
        insta::assert_snapshot!(p(2), @"2s");
        insta::assert_snapshot!(p(10), @"10s");
        insta::assert_snapshot!(p(100), @"1m 40s");

        insta::assert_snapshot!(p(1 * 60), @"1m");
        insta::assert_snapshot!(p(2 * 60), @"2m");
        insta::assert_snapshot!(p(10 * 60), @"10m");
        insta::assert_snapshot!(p(100 * 60), @"1h 40m");

        insta::assert_snapshot!(p(1 * 60 * 60), @"1h");
        insta::assert_snapshot!(p(2 * 60 * 60), @"2h");
        insta::assert_snapshot!(p(10 * 60 * 60), @"10h");
        insta::assert_snapshot!(p(100 * 60 * 60), @"100h");

        insta::assert_snapshot!(
            p(60 * 60 + 60 + 1),
            @"1h 1m 1s",
        );
        insta::assert_snapshot!(
            p(2 * 60 * 60 + 2 * 60 + 2),
            @"2h 2m 2s",
        );
        insta::assert_snapshot!(
            p(10 * 60 * 60 + 10 * 60 + 10),
            @"10h 10m 10s",
        );
        insta::assert_snapshot!(
            p(100 * 60 * 60 + 100 * 60 + 100),
            @"101h 41m 40s",
        );
    }

    #[test]
    fn print_unsigned_duration_designator_verbose() {
        let printer = || SpanPrinter::new().designator(Designator::Verbose);
        let p = |secs| {
            printer().unsigned_duration_to_string(
                &core::time::Duration::from_secs(secs),
            )
        };

        insta::assert_snapshot!(p(1), @"1second");
        insta::assert_snapshot!(p(2), @"2seconds");
        insta::assert_snapshot!(p(10), @"10seconds");
        insta::assert_snapshot!(p(100), @"1minute 40seconds");

        insta::assert_snapshot!(p(1 * 60), @"1minute");
        insta::assert_snapshot!(p(2 * 60), @"2minutes");
        insta::assert_snapshot!(p(10 * 60), @"10minutes");
        insta::assert_snapshot!(p(100 * 60), @"1hour 40minutes");

        insta::assert_snapshot!(p(1 * 60 * 60), @"1hour");
        insta::assert_snapshot!(p(2 * 60 * 60), @"2hours");
        insta::assert_snapshot!(p(10 * 60 * 60), @"10hours");
        insta::assert_snapshot!(p(100 * 60 * 60), @"100hours");

        insta::assert_snapshot!(
            p(60 * 60 + 60 + 1),
            @"1hour 1minute 1second",
        );
        insta::assert_snapshot!(
            p(2 * 60 * 60 + 2 * 60 + 2),
            @"2hours 2minutes 2seconds",
        );
        insta::assert_snapshot!(
            p(10 * 60 * 60 + 10 * 60 + 10),
            @"10hours 10minutes 10seconds",
        );
        insta::assert_snapshot!(
            p(100 * 60 * 60 + 100 * 60 + 100),
            @"101hours 41minutes 40seconds",
        );
    }

    #[test]
    fn print_unsigned_duration_designator_short() {
        let printer = || SpanPrinter::new().designator(Designator::Short);
        let p = |secs| {
            printer().unsigned_duration_to_string(
                &core::time::Duration::from_secs(secs),
            )
        };

        insta::assert_snapshot!(p(1), @"1sec");
        insta::assert_snapshot!(p(2), @"2secs");
        insta::assert_snapshot!(p(10), @"10secs");
        insta::assert_snapshot!(p(100), @"1min 40secs");

        insta::assert_snapshot!(p(1 * 60), @"1min");
        insta::assert_snapshot!(p(2 * 60), @"2mins");
        insta::assert_snapshot!(p(10 * 60), @"10mins");
        insta::assert_snapshot!(p(100 * 60), @"1hr 40mins");

        insta::assert_snapshot!(p(1 * 60 * 60), @"1hr");
        insta::assert_snapshot!(p(2 * 60 * 60), @"2hrs");
        insta::assert_snapshot!(p(10 * 60 * 60), @"10hrs");
        insta::assert_snapshot!(p(100 * 60 * 60), @"100hrs");

        insta::assert_snapshot!(
            p(60 * 60 + 60 + 1),
            @"1hr 1min 1sec",
        );
        insta::assert_snapshot!(
            p(2 * 60 * 60 + 2 * 60 + 2),
            @"2hrs 2mins 2secs",
        );
        insta::assert_snapshot!(
            p(10 * 60 * 60 + 10 * 60 + 10),
            @"10hrs 10mins 10secs",
        );
        insta::assert_snapshot!(
            p(100 * 60 * 60 + 100 * 60 + 100),
            @"101hrs 41mins 40secs",
        );
    }

    #[test]
    fn print_unsigned_duration_designator_compact() {
        let printer = || SpanPrinter::new().designator(Designator::Compact);
        let p = |secs| {
            printer().unsigned_duration_to_string(
                &core::time::Duration::from_secs(secs),
            )
        };

        insta::assert_snapshot!(p(1), @"1s");
        insta::assert_snapshot!(p(2), @"2s");
        insta::assert_snapshot!(p(10), @"10s");
        insta::assert_snapshot!(p(100), @"1m 40s");

        insta::assert_snapshot!(p(1 * 60), @"1m");
        insta::assert_snapshot!(p(2 * 60), @"2m");
        insta::assert_snapshot!(p(10 * 60), @"10m");
        insta::assert_snapshot!(p(100 * 60), @"1h 40m");

        insta::assert_snapshot!(p(1 * 60 * 60), @"1h");
        insta::assert_snapshot!(p(2 * 60 * 60), @"2h");
        insta::assert_snapshot!(p(10 * 60 * 60), @"10h");
        insta::assert_snapshot!(p(100 * 60 * 60), @"100h");

        insta::assert_snapshot!(
            p(60 * 60 + 60 + 1),
            @"1h 1m 1s",
        );
        insta::assert_snapshot!(
            p(2 * 60 * 60 + 2 * 60 + 2),
            @"2h 2m 2s",
        );
        insta::assert_snapshot!(
            p(10 * 60 * 60 + 10 * 60 + 10),
            @"10h 10m 10s",
        );
        insta::assert_snapshot!(
            p(100 * 60 * 60 + 100 * 60 + 100),
            @"101h 41m 40s",
        );
    }

    #[test]
    fn print_unsigned_duration_designator_direction_force() {
        let printer = || SpanPrinter::new().direction(Direction::ForceSign);
        let p = |secs| {
            printer().unsigned_duration_to_string(
                &core::time::Duration::from_secs(secs),
            )
        };

        insta::assert_snapshot!(p(1), @"+1s");
        insta::assert_snapshot!(p(2), @"+2s");
        insta::assert_snapshot!(p(10), @"+10s");
        insta::assert_snapshot!(p(100), @"+1m 40s");

        insta::assert_snapshot!(p(1 * 60), @"+1m");
        insta::assert_snapshot!(p(2 * 60), @"+2m");
        insta::assert_snapshot!(p(10 * 60), @"+10m");
        insta::assert_snapshot!(p(100 * 60), @"+1h 40m");

        insta::assert_snapshot!(p(1 * 60 * 60), @"+1h");
        insta::assert_snapshot!(p(2 * 60 * 60), @"+2h");
        insta::assert_snapshot!(p(10 * 60 * 60), @"+10h");
        insta::assert_snapshot!(p(100 * 60 * 60), @"+100h");

        insta::assert_snapshot!(
            p(60 * 60 + 60 + 1),
            @"+1h 1m 1s",
        );
        insta::assert_snapshot!(
            p(2 * 60 * 60 + 2 * 60 + 2),
            @"+2h 2m 2s",
        );
        insta::assert_snapshot!(
            p(10 * 60 * 60 + 10 * 60 + 10),
            @"+10h 10m 10s",
        );
        insta::assert_snapshot!(
            p(100 * 60 * 60 + 100 * 60 + 100),
            @"+101h 41m 40s",
        );
    }

    #[test]
    fn print_unsigned_duration_designator_padding() {
        let printer = || SpanPrinter::new().padding(2);
        let p = |secs| {
            printer().unsigned_duration_to_string(
                &core::time::Duration::from_secs(secs),
            )
        };

        insta::assert_snapshot!(p(1), @"01s");
        insta::assert_snapshot!(p(2), @"02s");
        insta::assert_snapshot!(p(10), @"10s");
        insta::assert_snapshot!(p(100), @"01m 40s");

        insta::assert_snapshot!(p(1 * 60), @"01m");
        insta::assert_snapshot!(p(2 * 60), @"02m");
        insta::assert_snapshot!(p(10 * 60), @"10m");
        insta::assert_snapshot!(p(100 * 60), @"01h 40m");

        insta::assert_snapshot!(p(1 * 60 * 60), @"01h");
        insta::assert_snapshot!(p(2 * 60 * 60), @"02h");
        insta::assert_snapshot!(p(10 * 60 * 60), @"10h");
        insta::assert_snapshot!(p(100 * 60 * 60), @"100h");

        insta::assert_snapshot!(
            p(60 * 60 + 60 + 1),
            @"01h 01m 01s",
        );
        insta::assert_snapshot!(
            p(2 * 60 * 60 + 2 * 60 + 2),
            @"02h 02m 02s",
        );
        insta::assert_snapshot!(
            p(10 * 60 * 60 + 10 * 60 + 10),
            @"10h 10m 10s",
        );
        insta::assert_snapshot!(
            p(100 * 60 * 60 + 100 * 60 + 100),
            @"101h 41m 40s",
        );
    }

    #[test]
    fn print_unsigned_duration_designator_spacing_none() {
        let printer = || SpanPrinter::new().spacing(Spacing::None);
        let p = |secs| {
            printer().unsigned_duration_to_string(
                &core::time::Duration::from_secs(secs),
            )
        };

        insta::assert_snapshot!(p(1), @"1s");
        insta::assert_snapshot!(p(2), @"2s");
        insta::assert_snapshot!(p(10), @"10s");
        insta::assert_snapshot!(p(100), @"1m40s");

        insta::assert_snapshot!(p(1 * 60), @"1m");
        insta::assert_snapshot!(p(2 * 60), @"2m");
        insta::assert_snapshot!(p(10 * 60), @"10m");
        insta::assert_snapshot!(p(100 * 60), @"1h40m");

        insta::assert_snapshot!(p(1 * 60 * 60), @"1h");
        insta::assert_snapshot!(p(2 * 60 * 60), @"2h");
        insta::assert_snapshot!(p(10 * 60 * 60), @"10h");
        insta::assert_snapshot!(p(100 * 60 * 60), @"100h");

        insta::assert_snapshot!(
            p(60 * 60 + 60 + 1),
            @"1h1m1s",
        );
        insta::assert_snapshot!(
            p(2 * 60 * 60 + 2 * 60 + 2),
            @"2h2m2s",
        );
        insta::assert_snapshot!(
            p(10 * 60 * 60 + 10 * 60 + 10),
            @"10h10m10s",
        );
        insta::assert_snapshot!(
            p(100 * 60 * 60 + 100 * 60 + 100),
            @"101h41m40s",
        );
    }

    #[test]
    fn print_unsigned_duration_designator_spacing_more() {
        let printer =
            || SpanPrinter::new().spacing(Spacing::BetweenUnitsAndDesignators);
        let p = |secs| {
            printer().unsigned_duration_to_string(
                &core::time::Duration::from_secs(secs),
            )
        };

        insta::assert_snapshot!(p(1), @"1 s");
        insta::assert_snapshot!(p(2), @"2 s");
        insta::assert_snapshot!(p(10), @"10 s");
        insta::assert_snapshot!(p(100), @"1 m 40 s");

        insta::assert_snapshot!(p(1 * 60), @"1 m");
        insta::assert_snapshot!(p(2 * 60), @"2 m");
        insta::assert_snapshot!(p(10 * 60), @"10 m");
        insta::assert_snapshot!(p(100 * 60), @"1 h 40 m");

        insta::assert_snapshot!(p(1 * 60 * 60), @"1 h");
        insta::assert_snapshot!(p(2 * 60 * 60), @"2 h");
        insta::assert_snapshot!(p(10 * 60 * 60), @"10 h");
        insta::assert_snapshot!(p(100 * 60 * 60), @"100 h");

        insta::assert_snapshot!(
            p(60 * 60 + 60 + 1),
            @"1 h 1 m 1 s",
        );
        insta::assert_snapshot!(
            p(2 * 60 * 60 + 2 * 60 + 2),
            @"2 h 2 m 2 s",
        );
        insta::assert_snapshot!(
            p(10 * 60 * 60 + 10 * 60 + 10),
            @"10 h 10 m 10 s",
        );
        insta::assert_snapshot!(
            p(100 * 60 * 60 + 100 * 60 + 100),
            @"101 h 41 m 40 s",
        );
    }

    #[test]
    fn print_unsigned_duration_designator_spacing_comma() {
        let printer = || {
            SpanPrinter::new()
                .comma_after_designator(true)
                .spacing(Spacing::BetweenUnitsAndDesignators)
        };
        let p = |secs| {
            printer().unsigned_duration_to_string(
                &core::time::Duration::from_secs(secs),
            )
        };

        insta::assert_snapshot!(p(1), @"1 s");
        insta::assert_snapshot!(p(2), @"2 s");
        insta::assert_snapshot!(p(10), @"10 s");
        insta::assert_snapshot!(p(100), @"1 m, 40 s");

        insta::assert_snapshot!(p(1 * 60), @"1 m");
        insta::assert_snapshot!(p(2 * 60), @"2 m");
        insta::assert_snapshot!(p(10 * 60), @"10 m");
        insta::assert_snapshot!(p(100 * 60), @"1 h, 40 m");

        insta::assert_snapshot!(p(1 * 60 * 60), @"1 h");
        insta::assert_snapshot!(p(2 * 60 * 60), @"2 h");
        insta::assert_snapshot!(p(10 * 60 * 60), @"10 h");
        insta::assert_snapshot!(p(100 * 60 * 60), @"100 h");

        insta::assert_snapshot!(
            p(60 * 60 + 60 + 1),
            @"1 h, 1 m, 1 s",
        );
        insta::assert_snapshot!(
            p(2 * 60 * 60 + 2 * 60 + 2),
            @"2 h, 2 m, 2 s",
        );
        insta::assert_snapshot!(
            p(10 * 60 * 60 + 10 * 60 + 10),
            @"10 h, 10 m, 10 s",
        );
        insta::assert_snapshot!(
            p(100 * 60 * 60 + 100 * 60 + 100),
            @"101 h, 41 m, 40 s",
        );
    }

    #[test]
    fn print_unsigned_duration_designator_fractional_hour() {
        let printer =
            || SpanPrinter::new().fractional(Some(FractionalUnit::Hour));
        let p = |secs, nanos| {
            printer().unsigned_duration_to_string(&core::time::Duration::new(
                secs, nanos,
            ))
        };
        let pp = |precision, secs, nanos| {
            printer()
                .precision(Some(precision))
                .duration_to_string(&SignedDuration::new(secs, nanos))
        };

        insta::assert_snapshot!(p(1 * 60 * 60, 0), @"1h");
        insta::assert_snapshot!(pp(0, 1 * 60 * 60, 0), @"1h");
        insta::assert_snapshot!(pp(1, 1 * 60 * 60, 0), @"1.0h");
        insta::assert_snapshot!(pp(2, 1 * 60 * 60, 0), @"1.00h");

        insta::assert_snapshot!(p(1 * 60 * 60 + 30 * 60, 0), @"1.5h");
        insta::assert_snapshot!(pp(0, 1 * 60 * 60 + 30 * 60, 0), @"1h");
        insta::assert_snapshot!(pp(1, 1 * 60 * 60 + 30 * 60, 0), @"1.5h");
        insta::assert_snapshot!(pp(2, 1 * 60 * 60 + 30 * 60, 0), @"1.50h");

        insta::assert_snapshot!(p(1 * 60 * 60 + 3 * 60, 0), @"1.05h");
        insta::assert_snapshot!(p(1 * 60 * 60 + 3 * 60, 1), @"1.05h");
        insta::assert_snapshot!(p(1, 0), @"0.000277777h");
        // precision loss!
        insta::assert_snapshot!(p(1, 1), @"0.000277777h");
        insta::assert_snapshot!(p(0, 0), @"0h");
        // precision loss!
        insta::assert_snapshot!(p(0, 1), @"0h");
    }

    #[test]
    fn print_unsigned_duration_designator_fractional_minute() {
        let printer =
            || SpanPrinter::new().fractional(Some(FractionalUnit::Minute));
        let p = |secs, nanos| {
            printer().unsigned_duration_to_string(&core::time::Duration::new(
                secs, nanos,
            ))
        };
        let pp = |precision, secs, nanos| {
            printer()
                .precision(Some(precision))
                .duration_to_string(&SignedDuration::new(secs, nanos))
        };

        insta::assert_snapshot!(p(1 * 60 * 60, 0), @"1h");
        insta::assert_snapshot!(p(1 * 60 * 60 + 30 * 60, 0), @"1h 30m");

        insta::assert_snapshot!(p(60, 0), @"1m");
        insta::assert_snapshot!(pp(0, 60, 0), @"1m");
        insta::assert_snapshot!(pp(1, 60, 0), @"1.0m");
        insta::assert_snapshot!(pp(2, 60, 0), @"1.00m");

        insta::assert_snapshot!(p(90, 0), @"1.5m");
        insta::assert_snapshot!(pp(0, 90, 0), @"1m");
        insta::assert_snapshot!(pp(1, 90, 0), @"1.5m");
        insta::assert_snapshot!(pp(2, 90, 0), @"1.50m");

        insta::assert_snapshot!(p(1 * 60 * 60, 1), @"1h");
        insta::assert_snapshot!(p(63, 0), @"1.05m");
        insta::assert_snapshot!(p(63, 1), @"1.05m");
        insta::assert_snapshot!(p(1, 0), @"0.016666666m");
        // precision loss!
        insta::assert_snapshot!(p(1, 1), @"0.016666666m");
        insta::assert_snapshot!(p(0, 0), @"0m");
        // precision loss!
        insta::assert_snapshot!(p(0, 1), @"0m");
    }

    #[test]
    fn print_unsigned_duration_designator_fractional_second() {
        let printer =
            || SpanPrinter::new().fractional(Some(FractionalUnit::Second));
        let p = |secs, nanos| {
            printer().unsigned_duration_to_string(&core::time::Duration::new(
                secs, nanos,
            ))
        };
        let pp = |precision, secs, nanos| {
            printer()
                .precision(Some(precision))
                .duration_to_string(&SignedDuration::new(secs, nanos))
        };

        insta::assert_snapshot!(p(1 * 60 * 60, 0), @"1h");
        insta::assert_snapshot!(p(1 * 60 * 60 + 30 * 60, 0), @"1h 30m");

        insta::assert_snapshot!(p(1, 0), @"1s");
        insta::assert_snapshot!(pp(0, 1, 0), @"1s");
        insta::assert_snapshot!(pp(1, 1, 0), @"1.0s");
        insta::assert_snapshot!(pp(2, 1, 0), @"1.00s");

        insta::assert_snapshot!(p(1, 500_000_000), @"1.5s");
        insta::assert_snapshot!(pp(0, 1, 500_000_000), @"1s");
        insta::assert_snapshot!(pp(1, 1, 500_000_000), @"1.5s");
        insta::assert_snapshot!(pp(2, 1, 500_000_000), @"1.50s");

        insta::assert_snapshot!(p(1, 1), @"1.000000001s");
        insta::assert_snapshot!(p(0, 1), @"0.000000001s");
        insta::assert_snapshot!(p(0, 0), @"0s");
    }

    #[test]
    fn print_unsigned_duration_designator_fractional_millisecond() {
        let printer = || {
            SpanPrinter::new().fractional(Some(FractionalUnit::Millisecond))
        };
        let p = |secs, nanos| {
            printer().unsigned_duration_to_string(&core::time::Duration::new(
                secs, nanos,
            ))
        };
        let pp = |precision, secs, nanos| {
            printer()
                .precision(Some(precision))
                .duration_to_string(&SignedDuration::new(secs, nanos))
        };

        insta::assert_snapshot!(p(1 * 60 * 60, 0), @"1h");
        insta::assert_snapshot!(p(1 * 60 * 60 + 30 * 60, 0), @"1h 30m");
        insta::assert_snapshot!(
            p(1 * 60 * 60 + 30 * 60 + 10, 0),
            @"1h 30m 10s",
        );

        insta::assert_snapshot!(p(1, 0), @"1s");
        insta::assert_snapshot!(pp(0, 1, 0), @"1s");
        insta::assert_snapshot!(pp(1, 1, 0), @"1s 0.0ms");
        insta::assert_snapshot!(pp(2, 1, 0), @"1s 0.00ms");

        insta::assert_snapshot!(p(1, 500_000_000), @"1s 500ms");
        insta::assert_snapshot!(pp(0, 1, 1_500_000), @"1s 1ms");
        insta::assert_snapshot!(pp(1, 1, 1_500_000), @"1s 1.5ms");
        insta::assert_snapshot!(pp(2, 1, 1_500_000), @"1s 1.50ms");

        insta::assert_snapshot!(p(0, 1_000_001), @"1.000001ms");
        insta::assert_snapshot!(p(0, 0_000_001), @"0.000001ms");
        insta::assert_snapshot!(p(0, 0), @"0ms");
    }

    #[test]
    fn print_unsigned_duration_designator_fractional_microsecond() {
        let printer = || {
            SpanPrinter::new().fractional(Some(FractionalUnit::Microsecond))
        };
        let p = |secs, nanos| {
            printer().unsigned_duration_to_string(&core::time::Duration::new(
                secs, nanos,
            ))
        };
        let pp = |precision, secs, nanos| {
            printer().precision(Some(precision)).unsigned_duration_to_string(
                &core::time::Duration::new(secs, nanos),
            )
        };

        insta::assert_snapshot!(p(1 * 60 * 60, 0), @"1h");
        insta::assert_snapshot!(p(1 * 60 * 60 + 30 * 60, 0), @"1h 30m");
        insta::assert_snapshot!(
            p(1 * 60 * 60 + 30 * 60 + 10, 0),
            @"1h 30m 10s",
        );

        insta::assert_snapshot!(p(1, 0), @"1s");
        insta::assert_snapshot!(pp(0, 1, 0), @"1s");
        insta::assert_snapshot!(pp(1, 1, 0), @"1s 0.0µs");
        insta::assert_snapshot!(pp(2, 1, 0), @"1s 0.00µs");

        insta::assert_snapshot!(p(1, 500_000_000), @"1s 500ms");
        insta::assert_snapshot!(pp(0, 1, 1_500_000), @"1s 1ms 500µs");
        insta::assert_snapshot!(pp(1, 1, 1_500_000), @"1s 1ms 500.0µs");
        insta::assert_snapshot!(pp(2, 1, 1_500_000), @"1s 1ms 500.00µs");

        insta::assert_snapshot!(p(0, 1_000_001), @"1ms 0.001µs");
        insta::assert_snapshot!(p(0, 0_000_001), @"0.001µs");
        insta::assert_snapshot!(p(0, 0), @"0µs");
    }

    #[test]
    fn print_span_hms() {
        let printer = || SpanPrinter::new().hours_minutes_seconds(true);
        let p = |span| printer().span_to_string(&span);

        insta::assert_snapshot!(p(1.second()), @"00:00:01");
        insta::assert_snapshot!(p(2.seconds()), @"00:00:02");
        insta::assert_snapshot!(p(10.seconds()), @"00:00:10");
        insta::assert_snapshot!(p(100.seconds()), @"00:00:100");

        insta::assert_snapshot!(p(1.minute()), @"00:01:00");
        insta::assert_snapshot!(p(2.minutes()), @"00:02:00");
        insta::assert_snapshot!(p(10.minutes()), @"00:10:00");
        insta::assert_snapshot!(p(100.minutes()), @"00:100:00");

        insta::assert_snapshot!(p(1.hour()), @"01:00:00");
        insta::assert_snapshot!(p(2.hours()), @"02:00:00");
        insta::assert_snapshot!(p(10.hours()), @"10:00:00");
        insta::assert_snapshot!(p(100.hours()), @"100:00:00");

        insta::assert_snapshot!(
            p(1.hour().minutes(1).seconds(1)),
            @"01:01:01",
        );
        insta::assert_snapshot!(
            p(2.hours().minutes(2).seconds(2)),
            @"02:02:02",
        );
        insta::assert_snapshot!(
            p(10.hours().minutes(10).seconds(10)),
            @"10:10:10",
        );
        insta::assert_snapshot!(
            p(100.hours().minutes(100).seconds(100)),
            @"100:100:100",
        );

        insta::assert_snapshot!(
            p(1.day().hours(1).minutes(1).seconds(1)),
            @"1d 01:01:01",
        );
        insta::assert_snapshot!(
            p(1.day()),
            @"1d 00:00:00",
        );
        insta::assert_snapshot!(
            p(1.day().seconds(2)),
            @"1d 00:00:02",
        );
    }

    #[test]
    fn print_span_hms_fmt() {
        let printer = || {
            SpanPrinter::new()
                .hours_minutes_seconds(true)
                .comma_after_designator(true)
                .spacing(Spacing::BetweenUnitsAndDesignators)
        };
        let p = |span| printer().span_to_string(&span);

        insta::assert_snapshot!(
            p(1.day().hours(1).minutes(1).seconds(1)),
            @"1 d, 01:01:01",
        );
        insta::assert_snapshot!(
            p(1.year().months(1).weeks(1).days(1).hours(1).minutes(1).seconds(1)),
            @"1 y, 1 mo, 1 w, 1 d, 01:01:01",
        );
        insta::assert_snapshot!(
            p(1.day().hours(1).minutes(1).seconds(1).nanoseconds(1)),
            @"1 d, 01:01:01.000000001",
        );
    }

    #[test]
    fn print_span_hms_sign() {
        let printer = |direction| {
            SpanPrinter::new().hours_minutes_seconds(true).direction(direction)
        };
        let p = |direction, span| printer(direction).span_to_string(&span);

        insta::assert_snapshot!(
            p(Direction::Auto, 1.hour()),
            @"01:00:00",
        );
        insta::assert_snapshot!(
            p(Direction::Sign, 1.hour()),
            @"01:00:00",
        );
        insta::assert_snapshot!(
            p(Direction::ForceSign, 1.hour()),
            @"+01:00:00",
        );
        insta::assert_snapshot!(
            p(Direction::Suffix, 1.hour()),
            @"01:00:00",
        );
        insta::assert_snapshot!(
            p(Direction::Auto, -1.hour()),
            @"-01:00:00",
        );
        insta::assert_snapshot!(
            p(Direction::Sign, -1.hour()),
            @"-01:00:00",
        );
        insta::assert_snapshot!(
            p(Direction::ForceSign, -1.hour()),
            @"-01:00:00",
        );
        insta::assert_snapshot!(
            p(Direction::Suffix, -1.hour()),
            @"01:00:00 ago",
        );

        insta::assert_snapshot!(
            p(Direction::Auto, 1.day().hours(1)),
            @"1d 01:00:00",
        );
        insta::assert_snapshot!(
            p(Direction::Sign, 1.day().hours(1)),
            @"1d 01:00:00",
        );
        insta::assert_snapshot!(
            p(Direction::ForceSign, 1.day().hours(1)),
            @"+1d 01:00:00",
        );
        insta::assert_snapshot!(
            p(Direction::Suffix, 1.day().hours(1)),
            @"1d 01:00:00",
        );
        // This is the main change from above. With non-zero
        // calendar units, the default for expressing a negative
        // sign switches to a suffix in the HH:MM:SS format.
        insta::assert_snapshot!(
            p(Direction::Auto, -1.day().hours(1)),
            @"1d 01:00:00 ago",
        );
        insta::assert_snapshot!(
            p(Direction::Sign, -1.day().hours(1)),
            @"-1d 01:00:00",
        );
        insta::assert_snapshot!(
            p(Direction::ForceSign, -1.day().hours(1)),
            @"-1d 01:00:00",
        );
        insta::assert_snapshot!(
            p(Direction::Suffix, -1.day().hours(1)),
            @"1d 01:00:00 ago",
        );
    }

    #[test]
    fn print_span_hms_fraction_auto() {
        let printer = || SpanPrinter::new().hours_minutes_seconds(true);
        let p = |span| printer().span_to_string(&span);

        insta::assert_snapshot!(p(1.nanosecond()), @"00:00:00.000000001");
        insta::assert_snapshot!(p(-1.nanosecond()), @"-00:00:00.000000001");
        insta::assert_snapshot!(
            printer().direction(Direction::ForceSign).span_to_string(&1.nanosecond()),
            @"+00:00:00.000000001",
        );

        insta::assert_snapshot!(
            p(1.second().nanoseconds(123)),
            @"00:00:01.000000123",
        );
        insta::assert_snapshot!(
            p(1.second().milliseconds(123)),
            @"00:00:01.123",
        );
        insta::assert_snapshot!(
            p(1.second().milliseconds(1_123)),
            @"00:00:02.123",
        );
        insta::assert_snapshot!(
            p(1.second().milliseconds(61_123)),
            @"00:00:62.123",
        );
    }

    #[test]
    fn print_span_hms_fraction_fixed_precision() {
        let printer = || SpanPrinter::new().hours_minutes_seconds(true);
        let p = |precision, span| {
            printer().precision(Some(precision)).span_to_string(&span)
        };

        insta::assert_snapshot!(p(3, 1.second()), @"00:00:01.000");
        insta::assert_snapshot!(
            p(3, 1.second().milliseconds(1)),
            @"00:00:01.001",
        );
        insta::assert_snapshot!(
            p(3, 1.second().milliseconds(123)),
            @"00:00:01.123",
        );
        insta::assert_snapshot!(
            p(3, 1.second().milliseconds(100)),
            @"00:00:01.100",
        );

        insta::assert_snapshot!(p(0, 1.second()), @"00:00:01");
        insta::assert_snapshot!(p(0, 1.second().milliseconds(1)), @"00:00:01");
        insta::assert_snapshot!(
            p(1, 1.second().milliseconds(999)),
            @"00:00:01.9",
        );
    }

    #[test]
    fn print_signed_duration_hms() {
        let printer = || SpanPrinter::new().hours_minutes_seconds(true);
        let p = |secs| {
            printer().duration_to_string(&SignedDuration::from_secs(secs))
        };

        // Note the differences with `Span`, since with a `SignedDuration`,
        // all units are balanced.

        insta::assert_snapshot!(p(1), @"00:00:01");
        insta::assert_snapshot!(p(2), @"00:00:02");
        insta::assert_snapshot!(p(10), @"00:00:10");
        insta::assert_snapshot!(p(100), @"00:01:40");

        insta::assert_snapshot!(p(1 * 60), @"00:01:00");
        insta::assert_snapshot!(p(2 * 60), @"00:02:00");
        insta::assert_snapshot!(p(10 * 60), @"00:10:00");
        insta::assert_snapshot!(p(100 * 60), @"01:40:00");

        insta::assert_snapshot!(p(1 * 60 * 60), @"01:00:00");
        insta::assert_snapshot!(p(2 * 60 * 60), @"02:00:00");
        insta::assert_snapshot!(p(10 * 60 * 60), @"10:00:00");
        insta::assert_snapshot!(p(100 * 60 * 60), @"100:00:00");

        insta::assert_snapshot!(
            p(60 * 60 + 60 + 1),
            @"01:01:01",
        );
        insta::assert_snapshot!(
            p(2 * 60 * 60 + 2 * 60 + 2),
            @"02:02:02",
        );
        insta::assert_snapshot!(
            p(10 * 60 * 60 + 10 * 60 + 10),
            @"10:10:10",
        );
        insta::assert_snapshot!(
            p(100 * 60 * 60 + 100 * 60 + 100),
            @"101:41:40",
        );
    }

    #[test]
    fn print_signed_duration_hms_sign() {
        let printer = |direction| {
            SpanPrinter::new().hours_minutes_seconds(true).direction(direction)
        };
        let p = |direction, secs| {
            printer(direction)
                .duration_to_string(&SignedDuration::from_secs(secs))
        };

        insta::assert_snapshot!(p(Direction::Auto, 1), @"00:00:01");
        insta::assert_snapshot!(p(Direction::Sign, 1), @"00:00:01");
        insta::assert_snapshot!(p(Direction::ForceSign, 1), @"+00:00:01");
        insta::assert_snapshot!(p(Direction::Suffix, 1), @"00:00:01");

        insta::assert_snapshot!(p(Direction::Auto, -1), @"-00:00:01");
        insta::assert_snapshot!(p(Direction::Sign, -1), @"-00:00:01");
        insta::assert_snapshot!(p(Direction::ForceSign, -1), @"-00:00:01");
        insta::assert_snapshot!(p(Direction::Suffix, -1), @"00:00:01 ago");
    }

    #[test]
    fn print_signed_duration_hms_fraction_auto() {
        let printer = || SpanPrinter::new().hours_minutes_seconds(true);
        let p = |secs, nanos| {
            printer().duration_to_string(&SignedDuration::new(secs, nanos))
        };

        insta::assert_snapshot!(p(0, 1), @"00:00:00.000000001");
        insta::assert_snapshot!(p(0, -1), @"-00:00:00.000000001");
        insta::assert_snapshot!(
            printer().direction(Direction::ForceSign).duration_to_string(
                &SignedDuration::new(0, 1),
            ),
            @"+00:00:00.000000001",
        );

        insta::assert_snapshot!(
            p(1, 123),
            @"00:00:01.000000123",
        );
        insta::assert_snapshot!(
            p(1, 123_000_000),
            @"00:00:01.123",
        );
        insta::assert_snapshot!(
            p(1, 1_123_000_000),
            @"00:00:02.123",
        );
        insta::assert_snapshot!(
            p(61, 1_123_000_000),
            @"00:01:02.123",
        );
    }

    #[test]
    fn print_signed_duration_hms_fraction_fixed_precision() {
        let printer = || SpanPrinter::new().hours_minutes_seconds(true);
        let p = |precision, secs, nanos| {
            printer()
                .precision(Some(precision))
                .duration_to_string(&SignedDuration::new(secs, nanos))
        };

        insta::assert_snapshot!(p(3, 1, 0), @"00:00:01.000");
        insta::assert_snapshot!(
            p(3, 1, 1_000_000),
            @"00:00:01.001",
        );
        insta::assert_snapshot!(
            p(3, 1, 123_000_000),
            @"00:00:01.123",
        );
        insta::assert_snapshot!(
            p(3, 1, 100_000_000),
            @"00:00:01.100",
        );

        insta::assert_snapshot!(p(0, 1, 0), @"00:00:01");
        insta::assert_snapshot!(p(0, 1, 1_000_000), @"00:00:01");
        insta::assert_snapshot!(
            p(1, 1, 999_000_000),
            @"00:00:01.9",
        );
    }

    #[test]
    fn print_unsigned_duration_hms() {
        let printer = || SpanPrinter::new().hours_minutes_seconds(true);
        let p = |secs| {
            printer().unsigned_duration_to_string(
                &core::time::Duration::from_secs(secs),
            )
        };

        // Note the differences with `Span`, since with a `Duration`,
        // all units are balanced.

        insta::assert_snapshot!(p(1), @"00:00:01");
        insta::assert_snapshot!(p(2), @"00:00:02");
        insta::assert_snapshot!(p(10), @"00:00:10");
        insta::assert_snapshot!(p(100), @"00:01:40");

        insta::assert_snapshot!(p(1 * 60), @"00:01:00");
        insta::assert_snapshot!(p(2 * 60), @"00:02:00");
        insta::assert_snapshot!(p(10 * 60), @"00:10:00");
        insta::assert_snapshot!(p(100 * 60), @"01:40:00");

        insta::assert_snapshot!(p(1 * 60 * 60), @"01:00:00");
        insta::assert_snapshot!(p(2 * 60 * 60), @"02:00:00");
        insta::assert_snapshot!(p(10 * 60 * 60), @"10:00:00");
        insta::assert_snapshot!(p(100 * 60 * 60), @"100:00:00");

        insta::assert_snapshot!(
            p(60 * 60 + 60 + 1),
            @"01:01:01",
        );
        insta::assert_snapshot!(
            p(2 * 60 * 60 + 2 * 60 + 2),
            @"02:02:02",
        );
        insta::assert_snapshot!(
            p(10 * 60 * 60 + 10 * 60 + 10),
            @"10:10:10",
        );
        insta::assert_snapshot!(
            p(100 * 60 * 60 + 100 * 60 + 100),
            @"101:41:40",
        );
    }

    #[test]
    fn print_unsigned_duration_hms_sign() {
        let printer = |direction| {
            SpanPrinter::new().hours_minutes_seconds(true).direction(direction)
        };
        let p = |direction, secs| {
            printer(direction).unsigned_duration_to_string(
                &core::time::Duration::from_secs(secs),
            )
        };

        insta::assert_snapshot!(p(Direction::Auto, 1), @"00:00:01");
        insta::assert_snapshot!(p(Direction::Sign, 1), @"00:00:01");
        insta::assert_snapshot!(p(Direction::ForceSign, 1), @"+00:00:01");
        insta::assert_snapshot!(p(Direction::Suffix, 1), @"00:00:01");
    }

    #[test]
    fn print_unsigned_duration_hms_fraction_auto() {
        let printer = || SpanPrinter::new().hours_minutes_seconds(true);
        let p = |secs, nanos| {
            printer().unsigned_duration_to_string(&core::time::Duration::new(
                secs, nanos,
            ))
        };

        insta::assert_snapshot!(p(0, 1), @"00:00:00.000000001");
        insta::assert_snapshot!(
            printer().direction(Direction::ForceSign).duration_to_string(
                &SignedDuration::new(0, 1),
            ),
            @"+00:00:00.000000001",
        );

        insta::assert_snapshot!(
            p(1, 123),
            @"00:00:01.000000123",
        );
        insta::assert_snapshot!(
            p(1, 123_000_000),
            @"00:00:01.123",
        );
        insta::assert_snapshot!(
            p(1, 1_123_000_000),
            @"00:00:02.123",
        );
        insta::assert_snapshot!(
            p(61, 1_123_000_000),
            @"00:01:02.123",
        );
    }

    #[test]
    fn print_unsigned_duration_hms_fraction_fixed_precision() {
        let printer = || SpanPrinter::new().hours_minutes_seconds(true);
        let p = |precision, secs, nanos| {
            printer().precision(Some(precision)).unsigned_duration_to_string(
                &core::time::Duration::new(secs, nanos),
            )
        };

        insta::assert_snapshot!(p(3, 1, 0), @"00:00:01.000");
        insta::assert_snapshot!(
            p(3, 1, 1_000_000),
            @"00:00:01.001",
        );
        insta::assert_snapshot!(
            p(3, 1, 123_000_000),
            @"00:00:01.123",
        );
        insta::assert_snapshot!(
            p(3, 1, 100_000_000),
            @"00:00:01.100",
        );

        insta::assert_snapshot!(p(0, 1, 0), @"00:00:01");
        insta::assert_snapshot!(p(0, 1, 1_000_000), @"00:00:01");
        insta::assert_snapshot!(
            p(1, 1, 999_000_000),
            @"00:00:01.9",
        );
    }

    /// Tests that we can write the maximal string successfully.
    ///
    /// This test doesn't guarantee that we'll always attempt the true
    /// maximum. The maximums here were determined by human inspection.
    #[test]
    fn maximums() {
        let p = SpanPrinter::new()
            .padding(u8::MAX)
            .designator(Designator::Verbose)
            .spacing(Spacing::BetweenUnitsAndDesignators)
            .comma_after_designator(true);

        let span = -19_998
            .year()
            .months(239_976)
            .weeks(1_043_497)
            .days(7_304_484)
            .hours(175_307_616)
            .minutes(10_518_456_960i64)
            .seconds(631_107_417_600i64)
            .milliseconds(631_107_417_600_000i64)
            .microseconds(631_107_416_600_000_000i64)
            .nanoseconds(9_223_372_036_854_775_807i64);
        insta::assert_snapshot!(
            p.span_to_string(&span),
            @"00000000000000019998 years, 00000000000000239976 months, 00000000000001043497 weeks, 00000000000007304484 days, 00000000000175307616 hours, 00000000010518456960 minutes, 00000000631107417600 seconds, 00000631107417600000 milliseconds, 00631107416600000000 microseconds, 09223372036854775807 nanoseconds ago",
        );

        let sdur = SignedDuration::MAX;
        insta::assert_snapshot!(
            p.duration_to_string(&sdur),
            @"00002562047788015215 hours, 00000000000000000030 minutes, 00000000000000000007 seconds, 00000000000000000999 milliseconds, 00000000000000000999 microseconds, 00000000000000000999 nanoseconds",
        );

        let udur =
            core::time::Duration::MAX - core::time::Duration::from_secs(16);
        insta::assert_snapshot!(
            p.unsigned_duration_to_string(&udur),
            @"00005124095576030430 hours, 00000000000000000059 minutes, 00000000000000000059 seconds, 00000000000000000999 milliseconds, 00000000000000000999 microseconds, 00000000000000000999 nanoseconds",
        );
    }
}
