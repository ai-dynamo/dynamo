use crate::{
    error::{
        unit::{MustDivide, UnitConfigError},
        util::RoundingIncrementError,
        Error, ErrorContext,
    },
    util::b,
    SignedDuration, Unit,
};

/// A representation of a rounding increment in a particular unit.
///
/// This implements the core rounding interface of Jiff, shared across all
/// types. The constructors on `Increment` know which units *and* increments
/// are valid for a particular rounding configuration. For example, only
/// `Increment::for_span` supports any `unit` value. All other constructors
/// have some kind of limitation (e.g., you can't round a date to the nearest
/// month).
///
/// For most cases, rounding is done via `Increment::round`. This takes a
/// quantity in a number of nanoseconds and rounds it to the nearest increment
/// (after converting the increment to nanoseconds as well, based on the unit
/// provided). This API makes it difficult for callers to get rounding wrong.
/// They just need to use the right constructor and provide a quantity in units
/// of nanoseconds.
///
/// For lower level needs, one can use `RoundMode::round_by_duration` directly.
/// This doesn't know about any units other than nanoseconds. It's useful in
/// contexts when dealing with variable length units and there are no rules
/// for what is a valid increment.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Increment {
    unit: Unit,
    value: i32,
}

impl Increment {
    /// Return a rounding increment suitable for use when rounding a `Span`.
    pub(crate) fn for_span(
        unit: Unit,
        value: i64,
    ) -> Result<Increment, Error> {
        // Indexed by `Unit`.
        static LIMITS: &[MustDivide] = &[
            MustDivide::NanosPerMicro,
            MustDivide::MicrosPerMilli,
            MustDivide::MillisPerSec,
            MustDivide::SecsPerMin,
            MustDivide::MinsPerHour,
            MustDivide::HoursPerCivilDay,
        ];

        // We allow any kind of increment for calendar units, but for time
        // units, they have to divide evenly into the next highest unit (and
        // also be less than that). The reason for this is that calendar
        // units vary, where as for time units, given a balanced span, you
        // know that time units will always spill over into days so that
        // hours/minutes/... will never exceed 24/60/...
        if unit < Unit::Day {
            Increment::for_limits(unit, value, LIMITS)
                .context(RoundingIncrementError::ForSpan)
        } else {
            // For calendar units, just verify that the increment is in the
            // correct range.
            let value = b::Increment32::check(value)
                .context(RoundingIncrementError::ForSpan)?;
            Ok(Increment { unit, value })
        }
    }

    /// Return a rounding increment suitable for use when rounding a
    /// `SignedDuration`.
    pub(crate) fn for_signed_duration(
        unit: Unit,
        value: i64,
    ) -> Result<Increment, Error> {
        if unit > Unit::Hour {
            return Err(Error::from(
                UnitConfigError::SignedDurationCalendarUnit { smallest: unit },
            ));
        }

        // For signed durations, we allow any increment that is within range.
        let value = b::Increment32::check(value)
            .context(RoundingIncrementError::ForSignedDuration)?;
        Ok(Increment { unit, value })
    }

    /// Return a rounding increment suitable for use when rounding a
    /// `tz::Offset`.
    pub(crate) fn for_offset(
        unit: Unit,
        value: i64,
    ) -> Result<Increment, Error> {
        if !(Unit::Second <= unit && unit <= Unit::Hour) {
            return Err(Error::from(UnitConfigError::TimeZoneOffset {
                given: unit,
            }));
        }

        // For time zone offsets, we allow any increment that is within range.
        let value = b::Increment32::check(value)
            .context(RoundingIncrementError::ForOffset)?;
        Ok(Increment { unit, value })
    }

    /// Return a rounding increment suitable for use when rounding a
    /// `civil::DateTime` or a `Zoned`.
    pub(crate) fn for_datetime(
        unit: Unit,
        value: i64,
    ) -> Result<Increment, Error> {
        // Indexed by `Unit`.
        static LIMITS: &[MustDivide] = &[
            MustDivide::NanosPerMicro,
            MustDivide::MicrosPerMilli,
            MustDivide::MillisPerSec,
            MustDivide::SecsPerMin,
            MustDivide::MinsPerHour,
            MustDivide::HoursPerCivilDay,
            MustDivide::Days,
        ];
        Increment::for_limits(unit, value, LIMITS)
            .context(RoundingIncrementError::ForDateTime)
    }

    /// Return a rounding increment suitable for use when rounding a
    /// `civil::Time`.
    pub(crate) fn for_time(
        unit: Unit,
        value: i64,
    ) -> Result<Increment, Error> {
        // Indexed by `Unit`.
        static LIMITS: &[MustDivide] = &[
            MustDivide::NanosPerMicro,
            MustDivide::MicrosPerMilli,
            MustDivide::MillisPerSec,
            MustDivide::SecsPerMin,
            MustDivide::MinsPerHour,
            MustDivide::HoursPerCivilDay,
        ];
        Increment::for_limits(unit, value, LIMITS)
            .context(RoundingIncrementError::ForTime)
    }

    /// Return a rounding increment suitable for use when rounding a
    /// `civil::Timestamp`.
    pub(crate) fn for_timestamp(
        unit: Unit,
        value: i64,
    ) -> Result<Increment, Error> {
        // Indexed by `Unit`.
        static MAXIMUMS: &[MustDivide] = &[
            MustDivide::NanosPerCivilDay,
            MustDivide::MicrosPerCivilDay,
            MustDivide::MillisPerCivilDay,
            MustDivide::SecsPerCivilDay,
            MustDivide::MinsPerCivilDay,
            MustDivide::HoursPerCivilDay,
        ];
        Increment::for_maximums(unit, value, MAXIMUMS)
            .context(RoundingIncrementError::ForTimestamp)
    }

    /// Rounds the given quantity to the nearest increment value (in terms of
    /// the units given to this increment's constructor). Rounding down or up
    /// is determined by the mode given.
    pub(crate) fn round(
        &self,
        mode: RoundMode,
        quantity: SignedDuration,
    ) -> Result<SignedDuration, Error> {
        // OK because the max for the number of nanoseconds in a unit
        // is weeks at `604_800_000_000_000` and `increment` cannot be
        // any larger than `1_000_000_000`. The multiplication of these two
        // values does not exceed `SignedDuration::MAX`.
        let increment = self.value() * self.unit().duration();
        mode.round_by_duration(quantity, increment)
    }

    /// Returns this increment's unit.
    ///
    /// The increment value is always in terms of this unit.
    pub(crate) fn unit(&self) -> Unit {
        self.unit
    }

    /// Returns this increment's value.
    ///
    /// e.g., "round to the nearest 15th minute" is expressed with an increment
    /// value of `15` and a unit of `Unit::Minute`.
    ///
    /// The value returned is guaranteed to be in the range
    /// `1..=1_000_000_000`.
    pub(crate) fn value(&self) -> i32 {
        self.value
    }
}

/// Lower level constructors that require the caller to know the precise
/// limits or maximums. These should generally stay unexported.
impl Increment {
    /// Validate the `increment` value for the given `unit` and limits.
    ///
    /// Specifically, this ensures that rounding to the given `unit` is
    /// supported and that the specific value is *less than* to the
    /// corresponding limit.
    fn for_limits(
        unit: Unit,
        value: i64,
        limits: &[MustDivide],
    ) -> Result<Increment, Error> {
        let Some(&must_divide) = limits.get(unit as usize) else {
            return Err(Error::from(
                UnitConfigError::RoundToUnitUnsupported { unit },
            ));
        };

        let value32 = b::Increment32::check(value)?;
        let must_divide_i64 = must_divide.as_i64();
        if value < must_divide_i64 && must_divide_i64 % value == 0 {
            return Ok(Increment { unit, value: value32 });
        }
        Err(Error::from(UnitConfigError::IncrementDivide {
            unit,
            must_divide,
        }))
    }

    /// Validate the `increment` value for the given `unit` and maximums.
    ///
    /// Specifically, this ensures that rounding to the given `unit` is
    /// supported and that the specific value is *less than or equal* to the
    /// corresponding maximum.
    fn for_maximums(
        unit: Unit,
        value: i64,
        maximums: &[MustDivide],
    ) -> Result<Increment, Error> {
        let Some(&must_divide) = maximums.get(unit as usize) else {
            return Err(Error::from(
                UnitConfigError::RoundToUnitUnsupported { unit },
            ));
        };

        let value32 = b::Increment32::check(value)?;
        let must_divide_i64 = must_divide.as_i64();
        if value <= must_divide_i64 && must_divide_i64 % value == 0 {
            return Ok(Increment { unit, value: value32 });
        }
        Err(Error::from(UnitConfigError::IncrementDivide {
            unit,
            must_divide,
        }))
    }
}

/// The mode for dealing with the remainder when rounding datetimes or spans.
///
/// This is used in APIs like [`Span::round`](crate::Span::round) for rounding
/// spans, and APIs like [`Zoned::round`](crate::Zoned::round) for rounding
/// datetimes.
///
/// In the documentation for each variant, we refer to concepts like the
/// "smallest" unit and the "rounding increment." These are best described
/// in the documentation for what you're rounding. For example,
/// [`SpanRound::smallest`](crate::SpanRound::smallest)
/// and [`SpanRound::increment`](crate::SpanRound::increment).
///
/// # Example
///
/// This shows how to round a span with a different rounding mode than the
/// default:
///
/// ```
/// use jiff::{RoundMode, SpanRound, ToSpan, Unit};
///
/// // The default rounds like how you were taught in school:
/// assert_eq!(
///     1.hour().minutes(59).round(Unit::Hour)?,
///     2.hours().fieldwise(),
/// );
/// // But we can change the mode, e.g., truncation:
/// let options = SpanRound::new().smallest(Unit::Hour).mode(RoundMode::Trunc);
/// assert_eq!(
///     1.hour().minutes(59).round(options)?,
///     1.hour().fieldwise(),
/// );
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RoundMode {
    /// Rounds toward positive infinity.
    ///
    /// For negative spans and datetimes, this option will make the value
    /// smaller, which could be unexpected. To round away from zero, use
    /// `Expand`.
    Ceil,
    /// Rounds toward negative infinity.
    ///
    /// This mode acts like `Trunc` for positive spans and datetimes, but
    /// for negative values it will make the value larger, which could be
    /// unexpected. To round towards zero, use `Trunc`.
    Floor,
    /// Rounds away from zero like `Ceil` for positive spans and datetimes,
    /// and like `Floor` for negative spans and datetimes.
    Expand,
    /// Rounds toward zero, chopping off any fractional part of a unit.
    ///
    /// This is the default when rounding spans returned from
    /// datetime arithmetic. (But it is not the default for
    /// [`Span::round`](crate::Span::round).)
    Trunc,
    /// Rounds to the nearest allowed value like `HalfExpand`, but when there
    /// is a tie, round towards positive infinity like `Ceil`.
    HalfCeil,
    /// Rounds to the nearest allowed value like `HalfExpand`, but when there
    /// is a tie, round towards negative infinity like `Floor`.
    HalfFloor,
    /// Rounds to the nearest value allowed by the rounding increment and the
    /// smallest unit. When there is a tie, round away from zero like `Ceil`
    /// for positive spans and datetimes and like `Floor` for negative spans
    /// and datetimes.
    ///
    /// This corresponds to how rounding is often taught in school.
    ///
    /// This is the default for rounding spans and datetimes.
    HalfExpand,
    /// Rounds to the nearest allowed value like `HalfExpand`, but when there
    /// is a tie, round towards zero like `Trunc`.
    HalfTrunc,
    /// Rounds to the nearest allowed value like `HalfExpand`, but when there
    /// is a tie, round towards the value that is an even multiple of the
    /// rounding increment. For example, with a rounding increment of `3`,
    /// the number `10` would round up to `12` instead of down to `9`, because
    /// `12` is an even multiple of `3`, where as `9` is is an odd multiple.
    HalfEven,
}

impl RoundMode {
    /// Round `quantity` to the nearest `increment` using this rounding mode.
    ///
    /// If this the rounding result would overflow `SignedDuration`, then an
    /// error is returned.
    ///
    /// Callers should generally prefer higher level APIs. But this one is
    /// unavoidable when the increment isn't tied to an invariant length and
    /// can vary.
    ///
    /// # Panics
    ///
    /// Callers must ensure that `increment` has a number of nanoseconds
    /// greater than or equal to `1`.
    pub(crate) fn round_by_duration(
        self,
        quantity: SignedDuration,
        increment: SignedDuration,
    ) -> Result<SignedDuration, Error> {
        let quantity = quantity.as_nanos();
        let increment = increment.as_nanos();
        assert!(increment >= 1);
        let rounded = self.round_by_i128(quantity, increment);
        // This can fail in the case where `quantity` is somehow rounded to a
        // value above/below a signed duration's max/min. This cannot *usually*
        // happen in practice because our `quantity` is typically limited by
        // the maximum Jiff datetime span. Even spanning from -9999 to 9999, a
        // 96-bit integer number of nanoseconds still can't cause an overflow.
        // Moreover, `increment` is usually capped to `1_000_000_000`. The only
        // case where it isn't is when we're rounding based on variable length
        // units (in which case, this lower level rounding routine is called
        // directly). For example, when rounding to years with an increment
        // (set by the user) to `15_000`. That will get translated to 15,000
        // years *in nanoseconds*.
        //
        // Even still, the complete span of time supported by Jiff can
        // comfortably fit into a `SignedDuration`. And because of the limit
        // placed on `increment`, I don't believe this error can ever actually
        // occur.
        SignedDuration::try_from_nanos_i128(rounded)
            .ok_or_else(|| Error::from(b::SignedDurationSeconds::error()))
    }

    /// The internal API that does the actual rounding.
    ///
    /// This does not error on overflow because callers can never use values
    /// close to the `i128` limits. If an overflow would otherwise occur due to
    /// rounding, then it saturates instead of panicking or returning an error.
    ///
    /// I wish we could do modulus on `SignedDuration` directly. Then we
    /// wouldn't need 128-bit arithmetic at all. But I looked into making
    /// `SignedDuration % SignedDuration` work, and it was quite involved? So I
    /// guess we should just let the compiler handle it for `i128`. We'd also
    /// need division.
    ///
    /// # Panics
    ///
    /// When `increment < 1`.
    fn round_by_i128(self, quantity: i128, increment: i128) -> i128 {
        assert!(increment >= 1);
        // ref: https://tc39.es/proposal-temporal/#sec-temporal-roundnumbertoincrement
        let mode = self;
        let mut quotient = quantity / increment;
        let remainder = quantity % increment;
        if remainder == 0 {
            return quantity;
        }
        let sign = if remainder < 0 { -1 } else { 1 };
        let tiebreaker = (remainder * 2).abs();
        let tie = tiebreaker == increment;
        let expand_is_nearer = tiebreaker > increment;
        // ref: https://tc39.es/proposal-temporal/#sec-temporal-roundnumbertoincrement
        match mode {
            RoundMode::Ceil => {
                if sign > 0 {
                    quotient += sign;
                }
            }
            RoundMode::Floor => {
                if sign < 0 {
                    quotient += sign;
                }
            }
            RoundMode::Expand => {
                quotient += sign;
            }
            RoundMode::Trunc => {}
            RoundMode::HalfCeil => {
                if expand_is_nearer || (tie && sign > 0) {
                    quotient += sign;
                }
            }
            RoundMode::HalfFloor => {
                if expand_is_nearer || (tie && sign < 0) {
                    quotient += sign;
                }
            }
            RoundMode::HalfExpand => {
                if expand_is_nearer || tie {
                    quotient += sign;
                }
            }
            RoundMode::HalfTrunc => {
                if expand_is_nearer {
                    quotient += sign;
                }
            }
            RoundMode::HalfEven => {
                if expand_is_nearer || (tie && quotient.rem_euclid(2) == 1) {
                    quotient += sign;
                }
            }
        }
        // We use saturating arithmetic here because this can overflow when
        // `quantity` is the max value. Since we're rounding, we just refuse to
        // go over the maximum. This can't happen in practice because this is
        // a private API and all paths to this API enforce that we are never
        // going to round over the `i128` maximum.
        quotient.saturating_mul(increment)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn for_span() {
        let get = |unit, value| Increment::for_span(unit, value);

        assert!(get(Unit::Nanosecond, 1).is_ok());
        assert!(get(Unit::Nanosecond, 500).is_ok());
        assert!(get(Unit::Nanosecond, -1).is_err());
        assert!(get(Unit::Nanosecond, 0).is_err());
        assert!(get(Unit::Nanosecond, 7).is_err());
        assert!(get(Unit::Nanosecond, 1_000).is_err());
        assert!(get(Unit::Nanosecond, i64::MIN).is_err());
        assert!(get(Unit::Nanosecond, i64::MAX).is_err());

        assert!(get(Unit::Microsecond, 1).is_ok());
        assert!(get(Unit::Microsecond, 500).is_ok());
        assert!(get(Unit::Microsecond, -1).is_err());
        assert!(get(Unit::Microsecond, 0).is_err());
        assert!(get(Unit::Microsecond, 7).is_err());
        assert!(get(Unit::Microsecond, 1_000).is_err());
        assert!(get(Unit::Microsecond, i64::MIN).is_err());
        assert!(get(Unit::Microsecond, i64::MAX).is_err());

        assert!(get(Unit::Millisecond, 1).is_ok());
        assert!(get(Unit::Millisecond, 500).is_ok());
        assert!(get(Unit::Millisecond, -1).is_err());
        assert!(get(Unit::Millisecond, 0).is_err());
        assert!(get(Unit::Millisecond, 7).is_err());
        assert!(get(Unit::Millisecond, 1_000).is_err());
        assert!(get(Unit::Millisecond, i64::MIN).is_err());
        assert!(get(Unit::Millisecond, i64::MAX).is_err());

        assert!(get(Unit::Second, 1).is_ok());
        assert!(get(Unit::Second, 15).is_ok());
        assert!(get(Unit::Second, -1).is_err());
        assert!(get(Unit::Second, 0).is_err());
        assert!(get(Unit::Second, 7).is_err());
        assert!(get(Unit::Second, 1_000).is_err());
        assert!(get(Unit::Second, i64::MIN).is_err());
        assert!(get(Unit::Second, i64::MAX).is_err());

        assert!(get(Unit::Minute, 1).is_ok());
        assert!(get(Unit::Minute, 15).is_ok());
        assert!(get(Unit::Minute, -1).is_err());
        assert!(get(Unit::Minute, 0).is_err());
        assert!(get(Unit::Minute, 7).is_err());
        assert!(get(Unit::Minute, 1_000).is_err());
        assert!(get(Unit::Minute, i64::MIN).is_err());
        assert!(get(Unit::Minute, i64::MAX).is_err());

        assert!(get(Unit::Hour, 1).is_ok());
        assert!(get(Unit::Hour, 6).is_ok());
        assert!(get(Unit::Hour, 12).is_ok());
        assert!(get(Unit::Hour, -1).is_err());
        assert!(get(Unit::Hour, 0).is_err());
        assert!(get(Unit::Hour, 15).is_err());
        assert!(get(Unit::Hour, 18).is_err());
        assert!(get(Unit::Hour, 23).is_err());
        assert!(get(Unit::Hour, 24).is_err());
        assert!(get(Unit::Hour, i64::MIN).is_err());
        assert!(get(Unit::Hour, i64::MAX).is_err());

        assert!(get(Unit::Day, 1).is_ok());
        assert!(get(Unit::Day, 6).is_ok());
        assert!(get(Unit::Day, 7).is_ok());
        assert!(get(Unit::Day, 12).is_ok());
        assert!(get(Unit::Day, 30).is_ok());
        assert!(get(Unit::Day, 31).is_ok());
        assert!(get(Unit::Day, 100).is_ok());
        assert!(get(Unit::Day, 10_000_000).is_ok());
        assert!(get(Unit::Day, 1_000_000_000).is_ok());
        assert!(get(Unit::Day, -1).is_err());
        assert!(get(Unit::Day, 0).is_err());
        assert!(get(Unit::Day, 1_000_000_001).is_err());
        assert!(get(Unit::Day, i64::MIN).is_err());
        assert!(get(Unit::Day, i64::MAX).is_err());

        assert!(get(Unit::Week, 1).is_ok());
        assert!(get(Unit::Week, 6).is_ok());
        assert!(get(Unit::Week, 7).is_ok());
        assert!(get(Unit::Week, 12).is_ok());
        assert!(get(Unit::Week, 30).is_ok());
        assert!(get(Unit::Week, 31).is_ok());
        assert!(get(Unit::Week, 100).is_ok());
        assert!(get(Unit::Week, 2_000_000).is_ok());
        assert!(get(Unit::Week, 1_000_000_000).is_ok());
        assert!(get(Unit::Week, -1).is_err());
        assert!(get(Unit::Week, 0).is_err());
        assert!(get(Unit::Week, 1_000_000_001).is_err());
        assert!(get(Unit::Week, i64::MIN).is_err());
        assert!(get(Unit::Week, i64::MAX).is_err());

        assert!(get(Unit::Month, 1).is_ok());
        assert!(get(Unit::Month, 6).is_ok());
        assert!(get(Unit::Month, 7).is_ok());
        assert!(get(Unit::Month, 12).is_ok());
        assert!(get(Unit::Month, 30).is_ok());
        assert!(get(Unit::Month, 31).is_ok());
        assert!(get(Unit::Month, 100).is_ok());
        assert!(get(Unit::Month, 300_000).is_ok());
        assert!(get(Unit::Month, 1_000_000_000).is_ok());
        assert!(get(Unit::Month, -1).is_err());
        assert!(get(Unit::Month, 0).is_err());
        assert!(get(Unit::Month, 1_000_000_001).is_err());
        assert!(get(Unit::Month, i64::MIN).is_err());
        assert!(get(Unit::Month, i64::MAX).is_err());

        assert!(get(Unit::Year, 1).is_ok());
        assert!(get(Unit::Year, 6).is_ok());
        assert!(get(Unit::Year, 7).is_ok());
        assert!(get(Unit::Year, 12).is_ok());
        assert!(get(Unit::Year, 30).is_ok());
        assert!(get(Unit::Year, 31).is_ok());
        assert!(get(Unit::Year, 100).is_ok());
        assert!(get(Unit::Year, 50_000).is_ok());
        assert!(get(Unit::Year, 1_000_000_000).is_ok());
        assert!(get(Unit::Year, -1).is_err());
        assert!(get(Unit::Year, 0).is_err());
        assert!(get(Unit::Year, 1_000_000_001).is_err());
        assert!(get(Unit::Year, i64::MIN).is_err());
        assert!(get(Unit::Year, i64::MAX).is_err());
    }

    #[test]
    fn for_datetime() {
        let get = |unit, value| Increment::for_datetime(unit, value);

        assert!(get(Unit::Nanosecond, 1).is_ok());
        assert!(get(Unit::Nanosecond, 500).is_ok());
        assert!(get(Unit::Nanosecond, -1).is_err());
        assert!(get(Unit::Nanosecond, 0).is_err());
        assert!(get(Unit::Nanosecond, 7).is_err());
        assert!(get(Unit::Nanosecond, 1_000).is_err());
        assert!(get(Unit::Nanosecond, i64::MIN).is_err());
        assert!(get(Unit::Nanosecond, i64::MAX).is_err());

        assert!(get(Unit::Microsecond, 1).is_ok());
        assert!(get(Unit::Microsecond, 500).is_ok());
        assert!(get(Unit::Microsecond, -1).is_err());
        assert!(get(Unit::Microsecond, 0).is_err());
        assert!(get(Unit::Microsecond, 7).is_err());
        assert!(get(Unit::Microsecond, 1_000).is_err());
        assert!(get(Unit::Microsecond, i64::MIN).is_err());
        assert!(get(Unit::Microsecond, i64::MAX).is_err());

        assert!(get(Unit::Millisecond, 1).is_ok());
        assert!(get(Unit::Millisecond, 500).is_ok());
        assert!(get(Unit::Millisecond, -1).is_err());
        assert!(get(Unit::Millisecond, 0).is_err());
        assert!(get(Unit::Millisecond, 7).is_err());
        assert!(get(Unit::Millisecond, 1_000).is_err());
        assert!(get(Unit::Millisecond, i64::MIN).is_err());
        assert!(get(Unit::Millisecond, i64::MAX).is_err());

        assert!(get(Unit::Second, 1).is_ok());
        assert!(get(Unit::Second, 15).is_ok());
        assert!(get(Unit::Second, -1).is_err());
        assert!(get(Unit::Second, 0).is_err());
        assert!(get(Unit::Second, 7).is_err());
        assert!(get(Unit::Second, 1_000).is_err());
        assert!(get(Unit::Second, i64::MIN).is_err());
        assert!(get(Unit::Second, i64::MAX).is_err());

        assert!(get(Unit::Minute, 1).is_ok());
        assert!(get(Unit::Minute, 15).is_ok());
        assert!(get(Unit::Minute, -1).is_err());
        assert!(get(Unit::Minute, 0).is_err());
        assert!(get(Unit::Minute, 7).is_err());
        assert!(get(Unit::Minute, 1_000).is_err());
        assert!(get(Unit::Minute, i64::MIN).is_err());
        assert!(get(Unit::Minute, i64::MAX).is_err());

        assert!(get(Unit::Hour, 1).is_ok());
        assert!(get(Unit::Hour, 6).is_ok());
        assert!(get(Unit::Hour, 12).is_ok());
        assert!(get(Unit::Hour, -1).is_err());
        assert!(get(Unit::Hour, 0).is_err());
        assert!(get(Unit::Hour, 15).is_err());
        assert!(get(Unit::Hour, 18).is_err());
        assert!(get(Unit::Hour, 23).is_err());
        assert!(get(Unit::Hour, 24).is_err());
        assert!(get(Unit::Hour, i64::MIN).is_err());
        assert!(get(Unit::Hour, i64::MAX).is_err());

        assert!(get(Unit::Day, 1).is_ok());
        assert!(get(Unit::Day, -1).is_err());
        assert!(get(Unit::Day, 0).is_err());
        assert!(get(Unit::Day, 2).is_err());
        assert!(get(Unit::Day, 4).is_err());
        assert!(get(Unit::Day, 7).is_err());
        assert!(get(Unit::Day, i64::MIN).is_err());
        assert!(get(Unit::Day, i64::MAX).is_err());
    }

    #[test]
    fn for_time() {
        let get = |unit, value| Increment::for_time(unit, value);

        assert!(get(Unit::Nanosecond, 1).is_ok());
        assert!(get(Unit::Nanosecond, 500).is_ok());
        assert!(get(Unit::Nanosecond, -1).is_err());
        assert!(get(Unit::Nanosecond, 0).is_err());
        assert!(get(Unit::Nanosecond, 7).is_err());
        assert!(get(Unit::Nanosecond, 1_000).is_err());
        assert!(get(Unit::Nanosecond, i64::MIN).is_err());
        assert!(get(Unit::Nanosecond, i64::MAX).is_err());

        assert!(get(Unit::Microsecond, 1).is_ok());
        assert!(get(Unit::Microsecond, 500).is_ok());
        assert!(get(Unit::Microsecond, -1).is_err());
        assert!(get(Unit::Microsecond, 0).is_err());
        assert!(get(Unit::Microsecond, 7).is_err());
        assert!(get(Unit::Microsecond, 1_000).is_err());
        assert!(get(Unit::Microsecond, i64::MIN).is_err());
        assert!(get(Unit::Microsecond, i64::MAX).is_err());

        assert!(get(Unit::Millisecond, 1).is_ok());
        assert!(get(Unit::Millisecond, 500).is_ok());
        assert!(get(Unit::Millisecond, -1).is_err());
        assert!(get(Unit::Millisecond, 0).is_err());
        assert!(get(Unit::Millisecond, 7).is_err());
        assert!(get(Unit::Millisecond, 1_000).is_err());
        assert!(get(Unit::Millisecond, i64::MIN).is_err());
        assert!(get(Unit::Millisecond, i64::MAX).is_err());

        assert!(get(Unit::Second, 1).is_ok());
        assert!(get(Unit::Second, 15).is_ok());
        assert!(get(Unit::Second, -1).is_err());
        assert!(get(Unit::Second, 0).is_err());
        assert!(get(Unit::Second, 7).is_err());
        assert!(get(Unit::Second, 1_000).is_err());
        assert!(get(Unit::Second, i64::MIN).is_err());
        assert!(get(Unit::Second, i64::MAX).is_err());

        assert!(get(Unit::Minute, 1).is_ok());
        assert!(get(Unit::Minute, 15).is_ok());
        assert!(get(Unit::Minute, -1).is_err());
        assert!(get(Unit::Minute, 0).is_err());
        assert!(get(Unit::Minute, 7).is_err());
        assert!(get(Unit::Minute, 1_000).is_err());
        assert!(get(Unit::Minute, i64::MIN).is_err());
        assert!(get(Unit::Minute, i64::MAX).is_err());

        assert!(get(Unit::Hour, 1).is_ok());
        assert!(get(Unit::Hour, 6).is_ok());
        assert!(get(Unit::Hour, 12).is_ok());
        assert!(get(Unit::Hour, -1).is_err());
        assert!(get(Unit::Hour, 0).is_err());
        assert!(get(Unit::Hour, 15).is_err());
        assert!(get(Unit::Hour, 18).is_err());
        assert!(get(Unit::Hour, 23).is_err());
        assert!(get(Unit::Hour, 24).is_err());
        assert!(get(Unit::Hour, i64::MIN).is_err());
        assert!(get(Unit::Hour, i64::MAX).is_err());

        assert!(get(Unit::Day, 1).is_err());
        assert!(get(Unit::Day, -1).is_err());
        assert!(get(Unit::Day, 0).is_err());
        assert!(get(Unit::Day, 2).is_err());
        assert!(get(Unit::Day, 4).is_err());
        assert!(get(Unit::Day, 7).is_err());
        assert!(get(Unit::Day, i64::MIN).is_err());
        assert!(get(Unit::Day, i64::MAX).is_err());
    }

    #[test]
    fn for_timestamp() {
        let get = |unit, value| Increment::for_timestamp(unit, value);

        assert!(get(Unit::Nanosecond, 1).is_ok());
        assert!(get(Unit::Nanosecond, 500).is_ok());
        assert!(get(Unit::Nanosecond, 1_000).is_ok());
        assert!(get(Unit::Nanosecond, 1_000_000_000).is_ok());
        assert!(get(Unit::Nanosecond, -1).is_err());
        assert!(get(Unit::Nanosecond, 0).is_err());
        assert!(get(Unit::Nanosecond, 7).is_err());
        assert!(get(Unit::Nanosecond, 1_000_000_001).is_err());
        assert!(get(Unit::Nanosecond, i64::MIN).is_err());
        assert!(get(Unit::Nanosecond, i64::MAX).is_err());

        assert!(get(Unit::Microsecond, 1).is_ok());
        assert!(get(Unit::Microsecond, 500).is_ok());
        assert!(get(Unit::Microsecond, 1_000).is_ok());
        assert!(get(Unit::Microsecond, 2_000).is_ok());
        assert!(get(Unit::Microsecond, -1).is_err());
        assert!(get(Unit::Microsecond, 0).is_err());
        assert!(get(Unit::Microsecond, 7).is_err());
        assert!(get(Unit::Microsecond, 1_000_000_000).is_err());
        assert!(get(Unit::Microsecond, 1_000_000_001).is_err());
        assert!(get(Unit::Microsecond, i64::MIN).is_err());
        assert!(get(Unit::Microsecond, i64::MAX).is_err());

        assert!(get(Unit::Millisecond, 1).is_ok());
        assert!(get(Unit::Millisecond, 500).is_ok());
        assert!(get(Unit::Millisecond, 1_000).is_ok());
        assert!(get(Unit::Millisecond, 2_000).is_ok());
        assert!(get(Unit::Millisecond, 86_400_000).is_ok());
        assert!(get(Unit::Millisecond, -1).is_err());
        assert!(get(Unit::Millisecond, 0).is_err());
        assert!(get(Unit::Millisecond, 7).is_err());
        assert!(get(Unit::Millisecond, 1_000_000_000).is_err());
        assert!(get(Unit::Millisecond, 1_000_000_001).is_err());
        assert!(get(Unit::Millisecond, i64::MIN).is_err());
        assert!(get(Unit::Millisecond, i64::MAX).is_err());

        assert!(get(Unit::Second, 1).is_ok());
        assert!(get(Unit::Second, 15).is_ok());
        assert!(get(Unit::Second, 3_600).is_ok());
        assert!(get(Unit::Second, 86_400).is_ok());
        assert!(get(Unit::Second, -1).is_err());
        assert!(get(Unit::Second, 0).is_err());
        assert!(get(Unit::Second, 7).is_err());
        assert!(get(Unit::Second, 1_000).is_err());
        assert!(get(Unit::Second, 86_401).is_err());
        assert!(get(Unit::Second, 172_800).is_err());
        assert!(get(Unit::Second, 1_000_000_000).is_err());
        assert!(get(Unit::Second, 1_000_000_001).is_err());
        assert!(get(Unit::Second, i64::MIN).is_err());
        assert!(get(Unit::Second, i64::MAX).is_err());

        assert!(get(Unit::Minute, 1).is_ok());
        assert!(get(Unit::Minute, 15).is_ok());
        assert!(get(Unit::Minute, 1_440).is_ok());
        assert!(get(Unit::Minute, -1).is_err());
        assert!(get(Unit::Minute, 0).is_err());
        assert!(get(Unit::Minute, 7).is_err());
        assert!(get(Unit::Minute, 1_000).is_err());
        assert!(get(Unit::Minute, 1_441).is_err());
        assert!(get(Unit::Minute, 2_880).is_err());
        assert!(get(Unit::Minute, 1_000_000_000).is_err());
        assert!(get(Unit::Minute, 1_000_000_001).is_err());
        assert!(get(Unit::Minute, i64::MIN).is_err());
        assert!(get(Unit::Minute, i64::MAX).is_err());

        assert!(get(Unit::Hour, 1).is_ok());
        assert!(get(Unit::Hour, 6).is_ok());
        assert!(get(Unit::Hour, 12).is_ok());
        assert!(get(Unit::Hour, 24).is_ok());
        assert!(get(Unit::Hour, -1).is_err());
        assert!(get(Unit::Hour, 0).is_err());
        assert!(get(Unit::Hour, 15).is_err());
        assert!(get(Unit::Hour, 18).is_err());
        assert!(get(Unit::Hour, 23).is_err());
        assert!(get(Unit::Hour, 25).is_err());
        assert!(get(Unit::Hour, 1_000_000_000).is_err());
        assert!(get(Unit::Hour, 1_000_000_001).is_err());
        assert!(get(Unit::Hour, i64::MIN).is_err());
        assert!(get(Unit::Hour, i64::MAX).is_err());

        assert!(get(Unit::Day, 1).is_err());
        assert!(get(Unit::Day, -1).is_err());
        assert!(get(Unit::Day, 0).is_err());
        assert!(get(Unit::Day, 2).is_err());
        assert!(get(Unit::Day, 4).is_err());
        assert!(get(Unit::Day, 7).is_err());
        assert!(get(Unit::Day, i64::MIN).is_err());
        assert!(get(Unit::Day, i64::MAX).is_err());

        assert!(get(Unit::Week, 1).is_err());
        assert!(get(Unit::Week, -1).is_err());
        assert!(get(Unit::Week, 0).is_err());
        assert!(get(Unit::Week, 2).is_err());
        assert!(get(Unit::Week, 4).is_err());
        assert!(get(Unit::Week, 7).is_err());
        assert!(get(Unit::Week, i64::MIN).is_err());
        assert!(get(Unit::Week, i64::MAX).is_err());

        assert!(get(Unit::Month, 1).is_err());
        assert!(get(Unit::Month, -1).is_err());
        assert!(get(Unit::Month, 0).is_err());
        assert!(get(Unit::Month, 2).is_err());
        assert!(get(Unit::Month, 4).is_err());
        assert!(get(Unit::Month, 7).is_err());
        assert!(get(Unit::Month, i64::MIN).is_err());
        assert!(get(Unit::Month, i64::MAX).is_err());

        assert!(get(Unit::Year, 1).is_err());
        assert!(get(Unit::Year, -1).is_err());
        assert!(get(Unit::Year, 0).is_err());
        assert!(get(Unit::Year, 2).is_err());
        assert!(get(Unit::Year, 4).is_err());
        assert!(get(Unit::Year, 7).is_err());
        assert!(get(Unit::Year, i64::MIN).is_err());
        assert!(get(Unit::Year, i64::MAX).is_err());
    }

    // Some ad hoc tests I wrote while writing the rounding increment code.
    #[test]
    fn round_to_increment_half_expand_ad_hoc() {
        let round = |quantity: i128, increment: i128| -> i128 {
            RoundMode::HalfExpand.round_by_i128(quantity, increment)
        };
        assert_eq!(26, round(20, 13));

        assert_eq!(0, round(29, 60));
        assert_eq!(60, round(30, 60));
        assert_eq!(60, round(31, 60));

        assert_eq!(0, round(3, 7));
        assert_eq!(7, round(4, 7));
    }

    // The Temporal tests are inspired by the table from here:
    // https://tc39.es/proposal-temporal/#sec-temporal-roundnumbertoincrement
    //
    // The main difference is that our rounding function specifically does not
    // use floating point, so we tweak the values a bit.

    #[test]
    fn round_to_increment_temporal_table_ceil() {
        let round = |quantity: i128, increment: i128| -> i128 {
            RoundMode::Ceil.round_by_i128(quantity, increment)
        };
        assert_eq!(-10, round(-15, 10));
        assert_eq!(0, round(-5, 10));
        assert_eq!(10, round(4, 10));
        assert_eq!(10, round(5, 10));
        assert_eq!(10, round(6, 10));
        assert_eq!(20, round(15, 10));

        assert_eq!(i128::MAX, round(i128::MAX, 1));
        assert_eq!(i128::MAX, round(i128::MAX, i128::MAX));

        // TODO: This test currently panics on overflow.
        // We should overall scrutinize the input constraints
        // for our rounding routines and align with their call
        // sites. We were previously relying on ranged integers
        // to holistically verify we weren't doing anything that
        // could result in overflow. I suspect that we can't
        // actually support the i128 quantity range (and I don't
        // think we need to). The question is whether we should
        // make the rounding interface fallible or if this
        // should just document panicking conditions.
        //
        // Oh, we should introduce a new internal `Increment`
        // wrapper that provides guarantees about its range.
        // assert_eq!(i128::MIN, round(i128::MIN, -1));
        // assert_eq!(i128::MIN, round(i128::MIN, i128::MIN));
    }

    #[test]
    fn round_to_increment_temporal_table_floor() {
        let round = |quantity: i128, increment: i128| -> i128 {
            RoundMode::Floor.round_by_i128(quantity, increment)
        };
        assert_eq!(-20, round(-15, 10));
        assert_eq!(-10, round(-5, 10));
        assert_eq!(0, round(4, 10));
        assert_eq!(0, round(5, 10));
        assert_eq!(0, round(6, 10));
        assert_eq!(10, round(15, 10));
    }

    #[test]
    fn round_to_increment_temporal_table_expand() {
        let round = |quantity: i128, increment: i128| -> i128 {
            RoundMode::Expand.round_by_i128(quantity, increment)
        };
        assert_eq!(-20, round(-15, 10));
        assert_eq!(-10, round(-5, 10));
        assert_eq!(10, round(4, 10));
        assert_eq!(10, round(5, 10));
        assert_eq!(10, round(6, 10));
        assert_eq!(20, round(15, 10));
    }

    #[test]
    fn round_to_increment_temporal_table_trunc() {
        let round = |quantity: i128, increment: i128| -> i128 {
            RoundMode::Trunc.round_by_i128(quantity, increment)
        };
        assert_eq!(-10, round(-15, 10));
        assert_eq!(0, round(-5, 10));
        assert_eq!(0, round(4, 10));
        assert_eq!(0, round(5, 10));
        assert_eq!(0, round(6, 10));
        assert_eq!(10, round(15, 10));
    }

    #[test]
    fn round_to_increment_temporal_table_half_ceil() {
        let round = |quantity: i128, increment: i128| -> i128 {
            RoundMode::HalfCeil.round_by_i128(quantity, increment)
        };
        assert_eq!(-10, round(-15, 10));
        assert_eq!(0, round(-5, 10));
        assert_eq!(0, round(4, 10));
        assert_eq!(10, round(5, 10));
        assert_eq!(10, round(6, 10));
        assert_eq!(20, round(15, 10));
    }

    #[test]
    fn round_to_increment_temporal_table_half_floor() {
        let round = |quantity: i128, increment: i128| -> i128 {
            RoundMode::HalfFloor.round_by_i128(quantity, increment)
        };
        assert_eq!(-20, round(-15, 10));
        assert_eq!(-10, round(-5, 10));
        assert_eq!(0, round(4, 10));
        assert_eq!(0, round(5, 10));
        assert_eq!(10, round(6, 10));
        assert_eq!(10, round(15, 10));
    }

    #[test]
    fn round_to_increment_temporal_table_half_expand() {
        let round = |quantity: i128, increment: i128| -> i128 {
            RoundMode::HalfExpand.round_by_i128(quantity, increment)
        };
        assert_eq!(-20, round(-15, 10));
        assert_eq!(-10, round(-5, 10));
        assert_eq!(0, round(4, 10));
        assert_eq!(10, round(5, 10));
        assert_eq!(10, round(6, 10));
        assert_eq!(20, round(15, 10));
    }

    #[test]
    fn round_to_increment_temporal_table_half_trunc() {
        let round = |quantity: i128, increment: i128| -> i128 {
            RoundMode::HalfTrunc.round_by_i128(quantity, increment)
        };
        assert_eq!(-10, round(-15, 10));
        assert_eq!(0, round(-5, 10));
        assert_eq!(0, round(4, 10));
        assert_eq!(0, round(5, 10));
        assert_eq!(10, round(6, 10));
        assert_eq!(10, round(15, 10));
    }

    #[test]
    fn round_to_increment_temporal_table_half_even() {
        let round = |quantity: i128, increment: i128| -> i128 {
            RoundMode::HalfEven.round_by_i128(quantity, increment)
        };
        assert_eq!(-20, round(-15, 10));
        assert_eq!(0, round(-5, 10));
        assert_eq!(0, round(4, 10));
        assert_eq!(0, round(5, 10));
        assert_eq!(10, round(6, 10));
        assert_eq!(20, round(15, 10));
    }

    // Tests that the maximum possible increment, in units of nanoseconds,
    // fits into a `SignedDuration`.
    #[test]
    fn maximum_increment_nanos() {
        assert!(Increment::for_span(Unit::Week, 1_000_000_000).is_ok());
        assert!(Increment::for_span(Unit::Week, 1_000_000_001).is_err());
        assert_eq!(
            Unit::Week.duration() * 1_000_000_000,
            SignedDuration::from_secs(168_000_000_000 * 60 * 60)
        );
    }

    // Tests some edge cases where rounding can actually overflow. We ensure
    // that an error is returned instead of panicking.
    //
    // I do not believe it's possible for these overflows to be observed using
    // Jiff's public API however.
    #[test]
    fn round_by_duration_overflow() {
        let mode = RoundMode::Expand;
        let quantity = SignedDuration::MAX;
        let increment = SignedDuration::MAX - SignedDuration::from_secs(1);
        assert!(mode.round_by_duration(quantity, increment).is_err());

        let mode = RoundMode::Expand;
        let quantity = SignedDuration::MIN;
        let increment = SignedDuration::MAX;
        assert!(mode.round_by_duration(quantity, increment).is_err());
    }
}
