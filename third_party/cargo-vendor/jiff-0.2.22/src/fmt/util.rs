use crate::{
    error::{fmt::util::Error as E, ErrorContext},
    fmt::Parsed,
    util::{
        b::{self, Sign},
        parse,
    },
    Error, SignedDuration, Span, Unit,
};

/// A container for holding a partially parsed duration.
///
/// This is used for parsing into `Span`, `SignedDuration` and (hopefully
/// soon) `std::time::Duration`. It's _also_ used for both the ISO 8601
/// duration and "friendly" format.
///
/// This replaced a significant chunk of code that was bespoke to each
/// combination of duration type _and_ format.
///
/// The idea behind it is that we parse each duration component as an unsigned
/// 64-bit integer and keep track of the sign separately. This is a critical
/// aspect that was motivated by being able to roundtrip all legal values of
/// a 96-bit signed integer number of nanoseconds (i.e., `SignedDuration`).
/// In particular, if we used `i64` to represent each component, then it
/// makes it much more difficult to parse, e.g., `9223372036854775808
/// seconds ago`. Namely, `9223372036854775808` is not a valid `i64` but
/// `-9223372036854775808` is. Notably, the sign is indicated by a suffix,
/// so we don't know it's negative when parsing the integer itself. So we
/// represent all components as their unsigned absolute value and apply the
/// sign at the end.
///
/// This also centralizes a lot of thorny duration math and opens up the
/// opportunity for tighter optimization.
#[derive(Debug, Default)]
pub(crate) struct DurationUnits {
    /// The parsed unit values in descending order. That is, nanoseconds are
    /// at index 0 while years are at index 9.
    values: [u64; 10],
    /// Any fractional component parsed. The fraction is necessarily a fraction
    /// of the minimum unit if present.
    fraction: Option<u32>,
    /// The sign of the duration. This may be set at any time.
    ///
    /// Note that this defaults to zero! So callers will always want to set
    /// this.
    sign: Sign,
    /// The smallest unit value that was explicitly set.
    min: Option<Unit>,
    /// The largest unit value that was explicitly set.
    max: Option<Unit>,
    /// Whether there are any non-zero units.
    any_non_zero_units: bool,
}

impl DurationUnits {
    /// Set the duration component value for the given unit.
    ///
    /// The value here is always unsigned. To deal with negative values, set
    /// the sign independently. It will be accounted for when using one of this
    /// type's methods for converting to a concrete duration type.
    ///
    /// # Panics
    ///
    /// When this is called after `set_fraction`.
    ///
    /// # Errors
    ///
    /// Since this is meant to be used in service of duration parsing and all
    /// duration parsing proceeds from largest to smallest units, this will
    /// return an error if the given unit is bigger than or equal to any
    /// previously set unit. This also implies that this can only be called
    /// at most once for each unit value.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn set_unit_value(
        &mut self,
        unit: Unit,
        value: u64,
    ) -> Result<(), Error> {
        assert!(self.fraction.is_none());

        if let Some(min) = self.min {
            if min <= unit {
                return Err(Error::from(E::OutOfOrderUnits {
                    found: unit,
                    previous: min,
                }));
            }
        }
        // Given the above check, the given unit must be smaller than any we
        // have seen so far.
        self.min = Some(unit);
        // The maximum unit is always the first unit set, since we can never
        // see a unit bigger than it without an error occurring.
        if self.max.is_none() {
            self.max = Some(unit);
        }
        self.values[unit.as_usize()] = value;
        self.any_non_zero_units = self.any_non_zero_units || value != 0;
        Ok(())
    }

    /// A convenience routine for setting values parsed from an `HH:MM:SS`
    /// format (including the fraction).
    ///
    /// # Errors
    ///
    /// This forwards errors from `DurationUnits::set_unit_value`. It will also
    /// return an error is the minimum parsed unit (so far) is smaller than
    /// days. (Since `HH:MM:SS` can only appear after units of years, months,
    /// weeks or days.)
    pub(crate) fn set_hms(
        &mut self,
        hours: u64,
        minutes: u64,
        seconds: u64,
        fraction: Option<u32>,
    ) -> Result<(), Error> {
        if let Some(min) = self.min {
            if min <= Unit::Hour {
                return Err(Error::from(E::OutOfOrderHMS { found: min }));
            }
        }
        self.set_unit_value(Unit::Hour, hours)?;
        self.set_unit_value(Unit::Minute, minutes)?;
        self.set_unit_value(Unit::Second, seconds)?;
        if let Some(fraction) = fraction {
            self.set_fraction(fraction)?;
        }
        Ok(())
    }

    /// Set the fractional value.
    ///
    /// This is always interpreted as a fraction of the minimal unit.
    ///
    /// Callers must ensure this is called after the last call to
    /// `DurationUnits::set_unit_value`.
    ///
    /// # Panics
    ///
    /// When `fraction` is not in the range `0..=999_999_999`. Callers are
    /// expected to uphold this invariant.
    ///
    /// # Errors
    ///
    /// This will return an error if the minimum unit is `Unit::Nanosecond`.
    /// (Because fractional nanoseconds are not supported.) This will also
    /// return an error if the minimum unit is bigger than `Unit::Hour`.
    pub(crate) fn set_fraction(&mut self, fraction: u32) -> Result<(), Error> {
        assert!(fraction <= 999_999_999);
        if let Some(min) = self.min {
            if min > Unit::Hour || min == Unit::Nanosecond {
                return Err(Error::from(E::NotAllowedFractionalUnit {
                    found: min,
                }));
            }
        }
        self.fraction = Some(fraction);
        Ok(())
    }

    /// Set the sign associated with the components.
    ///
    /// The sign applies to the entire duration. There is no support for
    /// having some components signed and some unsigned.
    ///
    /// If no sign is set, then it is assumed to be zero. Note also that
    /// even if a sign is explicitly set *and* all unit values are zero,
    /// then the sign will be set to zero.
    pub(crate) fn set_sign(&mut self, sign: Sign) {
        self.sign = sign;
    }

    /// Convert these duration components to a `Span`.
    ///
    /// # Errors
    ///
    /// If any individual unit exceeds the limits of a `Span`, or if the units
    /// combine to exceed what can be represented by a `Span`, then this
    /// returns an error.
    ///
    /// This also returns an error if no units were set.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn to_span(&self) -> Result<Span, Error> {
        // When every unit value is less than this, *and* there is
        // no fractional component, then we trigger a fast path that
        // doesn't need to bother with error handling and careful
        // handling of the sign.
        //
        // Why do we use the maximum year value? Because years are
        // the "biggest" unit, it follows that there can't be any
        // other unit whose limit is smaller than years as a
        // dimenionless quantity. That is, if all parsed unit values
        // are no bigger than the maximum year, then we know all
        // parsed unit values are necessarily within their
        // appropriate limits.
        const LIMIT: u64 = b::SpanYears::MAX as u64;

        // If we have a fraction or a particularly large unit,
        // bail out to the general case.
        if self.fraction.is_some()
            || self.values.iter().any(|&value| value > LIMIT)
            // If no unit was set, it's an error case.
            || self.max.is_none()
        {
            return self.to_span_general();
        }

        let mut span = Span::new();

        let years = self.values[Unit::Year.as_usize()] as i16;
        let months = self.values[Unit::Month.as_usize()] as i32;
        let weeks = self.values[Unit::Week.as_usize()] as i32;
        let days = self.values[Unit::Day.as_usize()] as i32;
        let hours = self.values[Unit::Hour.as_usize()] as i32;
        let mins = self.values[Unit::Minute.as_usize()] as i64;
        let secs = self.values[Unit::Second.as_usize()] as i64;
        let millis = self.values[Unit::Millisecond.as_usize()] as i64;
        let micros = self.values[Unit::Microsecond.as_usize()] as i64;
        let nanos = self.values[Unit::Nanosecond.as_usize()] as i64;

        span = span.years_unchecked(years);
        span = span.months_unchecked(months);
        span = span.weeks_unchecked(weeks);
        span = span.days_unchecked(days);
        span = span.hours_unchecked(hours);
        span = span.minutes_unchecked(mins);
        span = span.seconds_unchecked(secs);
        span = span.milliseconds_unchecked(millis);
        span = span.microseconds_unchecked(micros);
        span = span.nanoseconds_unchecked(nanos);

        // The unchecked setters above don't manipulate
        // the sign, which defaults to zero. So we need to
        // set it even when it's positive.
        span = span.sign_unchecked(self.get_sign());

        Ok(span)
    }

    /// The "general" implementation of `DurationUnits::to_span`.
    ///
    /// This handles all possible cases, including fractional units, with good
    /// error handling. Basically, we take this path when we think an error
    /// _could_ occur. But this function is more bloaty and does more work, so
    /// the more it can be avoided, the better.
    #[cold]
    #[inline(never)]
    fn to_span_general(&self) -> Result<Span, Error> {
        #[cfg_attr(feature = "perf-inline", inline(always))]
        fn set_time_unit(
            unit: Unit,
            value: i64,
            span: Span,
            set: impl FnOnce(Span) -> Result<Span, Error>,
        ) -> Result<Span, Error> {
            #[cold]
            #[inline(never)]
            fn fractional_fallback(
                err: Error,
                unit: Unit,
                value: i64,
                span: Span,
            ) -> Result<Span, Error> {
                // Fractional calendar units aren't supported. Neither are
                // fractional nanoseconds. So there's nothing we can do in
                // this case.
                if unit > Unit::Hour || unit == Unit::Nanosecond {
                    Err(err)
                } else {
                    // This is annoying, but because we can write out a larger
                    // number of hours/minutes/seconds than what we actually
                    // support, we need to be prepared to parse an unbalanced
                    // span if our time units are too big here. In essence,
                    // this lets a single time unit "overflow" into smaller
                    // units if it exceeds the limits.
                    fractional_time_to_span(unit, value, 0, span)
                }
            }

            set(span)
                .or_else(|err| fractional_fallback(err, unit, value, span))
                .context(E::FailedValueSet { unit })
        }

        let (min, _) = self.get_min_max_units()?;
        let mut span = Span::new();

        if self.values[Unit::Year.as_usize()] != 0 {
            let value = self.get_unit_value(Unit::Year)?;
            span = span
                .try_years(value)
                .context(E::FailedValueSet { unit: Unit::Year })?;
        }
        if self.values[Unit::Month.as_usize()] != 0 {
            let value = self.get_unit_value(Unit::Month)?;
            span = span
                .try_months(value)
                .context(E::FailedValueSet { unit: Unit::Month })?;
        }
        if self.values[Unit::Week.as_usize()] != 0 {
            let value = self.get_unit_value(Unit::Week)?;
            span = span
                .try_weeks(value)
                .context(E::FailedValueSet { unit: Unit::Week })?;
        }
        if self.values[Unit::Day.as_usize()] != 0 {
            let value = self.get_unit_value(Unit::Day)?;
            span = span
                .try_days(value)
                .context(E::FailedValueSet { unit: Unit::Day })?;
        }
        if self.values[Unit::Hour.as_usize()] != 0 {
            let value = self.get_unit_value(Unit::Hour)?;
            span = set_time_unit(Unit::Hour, value, span, |span| {
                span.try_hours(value)
            })?;
        }
        if self.values[Unit::Minute.as_usize()] != 0 {
            let value = self.get_unit_value(Unit::Minute)?;
            span = set_time_unit(Unit::Minute, value, span, |span| {
                span.try_minutes(value)
            })?;
        }
        if self.values[Unit::Second.as_usize()] != 0 {
            let value = self.get_unit_value(Unit::Second)?;
            span = set_time_unit(Unit::Second, value, span, |span| {
                span.try_seconds(value)
            })?;
        }
        if self.values[Unit::Millisecond.as_usize()] != 0 {
            let value = self.get_unit_value(Unit::Millisecond)?;
            span = set_time_unit(Unit::Millisecond, value, span, |span| {
                span.try_milliseconds(value)
            })?;
        }
        if self.values[Unit::Microsecond.as_usize()] != 0 {
            let value = self.get_unit_value(Unit::Microsecond)?;
            span = set_time_unit(Unit::Microsecond, value, span, |span| {
                span.try_microseconds(value)
            })?;
        }
        if self.values[Unit::Nanosecond.as_usize()] != 0 {
            let value = self.get_unit_value(Unit::Nanosecond)?;
            span = set_time_unit(Unit::Nanosecond, value, span, |span| {
                span.try_nanoseconds(value)
            })?;
        }

        if let Some(fraction) = self.get_fraction()? {
            let value = self.get_unit_value(min)?;
            span = fractional_time_to_span(min, value, fraction, span)?;
        }

        Ok(span)
    }

    /// Convert these duration components to a `SignedDuration`.
    ///
    /// # Errors
    ///
    /// If the total number of nanoseconds represented by all units combined
    /// exceeds what can bit in a 96-bit signed integer, then an error is
    /// returned.
    ///
    /// An error is also returned if any calendar units (days or greater) were
    /// set or if no units were set.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn to_signed_duration(&self) -> Result<SignedDuration, Error> {
        // When every unit value is less than this, *and* there is
        // no fractional component, then we trigger a fast path that
        // doesn't need to bother with error handling and careful
        // handling of the sign.
        //
        // Why `999`? Well, I think it's nice to use one limit for all
        // units to make the comparisons simpler (although we could
        // use more targeted values to admit more cases, I didn't try
        // that). But specifically, this means we can have `999ms 999us
        // 999ns` as a maximal subsecond value without overflowing
        // the nanosecond component of a `SignedDuration`. This lets
        // us "just do math" without needing to check each result and
        // handle errors.
        const LIMIT: u64 = 999;

        if self.fraction.is_some()
            || self.values[..Unit::Day.as_usize()]
                .iter()
                .any(|&value| value > LIMIT)
            || self.max.map_or(true, |max| max > Unit::Hour)
        {
            return self.to_signed_duration_general();
        }

        let hours = self.values[Unit::Hour.as_usize()] as i64;
        let mins = self.values[Unit::Minute.as_usize()] as i64;
        let secs = self.values[Unit::Second.as_usize()] as i64;
        let millis = self.values[Unit::Millisecond.as_usize()] as i32;
        let micros = self.values[Unit::Microsecond.as_usize()] as i32;
        let nanos = self.values[Unit::Nanosecond.as_usize()] as i32;

        let total_secs = (hours * 3600) + (mins * 60) + secs;
        let total_nanos = (millis * 1_000_000) + (micros * 1_000) + nanos;
        let mut sdur =
            SignedDuration::new_without_nano_overflow(total_secs, total_nanos);
        if self.get_sign().is_negative() {
            sdur = -sdur;
        }

        Ok(sdur)
    }

    /// The "general" implementation of `DurationUnits::to_signed_duration`.
    ///
    /// This handles all possible cases, including fractional units, with good
    /// error handling. Basically, we take this path when we think an error
    /// _could_ occur. But this function is more bloaty and does more work, so
    /// the more it can be avoided, the better.
    #[cold]
    #[inline(never)]
    fn to_signed_duration_general(&self) -> Result<SignedDuration, Error> {
        let (min, max) = self.get_min_max_units()?;
        if max > Unit::Hour {
            return Err(Error::from(E::NotAllowedCalendarUnit { unit: max }));
        }

        let mut sdur = SignedDuration::ZERO;
        if self.values[Unit::Hour.as_usize()] != 0 {
            let value = self.get_unit_value(Unit::Hour)?;
            sdur = SignedDuration::try_from_hours(value)
                .and_then(|nanos| sdur.checked_add(nanos))
                .ok_or(E::OverflowForUnit { unit: Unit::Hour })?;
        }
        if self.values[Unit::Minute.as_usize()] != 0 {
            let value = self.get_unit_value(Unit::Minute)?;
            sdur = SignedDuration::try_from_mins(value)
                .and_then(|nanos| sdur.checked_add(nanos))
                .ok_or(E::OverflowForUnit { unit: Unit::Minute })?;
        }
        if self.values[Unit::Second.as_usize()] != 0 {
            let value = self.get_unit_value(Unit::Second)?;
            sdur = SignedDuration::from_secs(value)
                .checked_add(sdur)
                .ok_or(E::OverflowForUnit { unit: Unit::Second })?;
        }
        if self.values[Unit::Millisecond.as_usize()] != 0 {
            let value = self.get_unit_value(Unit::Millisecond)?;
            sdur = SignedDuration::from_millis(value)
                .checked_add(sdur)
                .ok_or(E::OverflowForUnit { unit: Unit::Millisecond })?;
        }
        if self.values[Unit::Microsecond.as_usize()] != 0 {
            let value = self.get_unit_value(Unit::Microsecond)?;
            sdur = SignedDuration::from_micros(value)
                .checked_add(sdur)
                .ok_or(E::OverflowForUnit { unit: Unit::Microsecond })?;
        }
        if self.values[Unit::Nanosecond.as_usize()] != 0 {
            let value = self.get_unit_value(Unit::Nanosecond)?;
            sdur = SignedDuration::from_nanos(value)
                .checked_add(sdur)
                .ok_or(E::OverflowForUnit { unit: Unit::Nanosecond })?;
        }

        if let Some(fraction) = self.get_fraction()? {
            sdur = sdur
                .checked_add(fractional_duration(min, fraction)?)
                .ok_or(E::OverflowForUnitFractional { unit: min })?;
        }

        Ok(sdur)
    }

    /// Convert these duration components to a `core::time::Duration`.
    ///
    /// # Errors
    ///
    /// If the total number of nanoseconds represented by all units combined
    /// exceeds what can bit in a 96-bit signed integer, then an error is
    /// returned.
    ///
    /// An error is also returned if any calendar units (days or greater) were
    /// set or if no units were set.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn to_unsigned_duration(
        &self,
    ) -> Result<core::time::Duration, Error> {
        // When every unit value is less than this, *and* there is
        // no fractional component, then we trigger a fast path that
        // doesn't need to bother with error handling and careful
        // handling of the sign.
        //
        // Why `999`? Well, I think it's nice to use one limit for all
        // units to make the comparisons simpler (although we could
        // use more targeted values to admit more cases, I didn't try
        // that). But specifically, this means we can have `999ms 999us
        // 999ns` as a maximal subsecond value without overflowing
        // the nanosecond component of a `core::time::Duration`. This lets
        // us "just do math" without needing to check each result and
        // handle errors.
        const LIMIT: u64 = 999;

        if self.fraction.is_some()
            || self.values[..Unit::Day.as_usize()]
                .iter()
                .any(|&value| value > LIMIT)
            || self.max.map_or(true, |max| max > Unit::Hour)
            || self.sign.is_negative()
        {
            return self.to_unsigned_duration_general();
        }

        let hours = self.values[Unit::Hour.as_usize()];
        let mins = self.values[Unit::Minute.as_usize()];
        let secs = self.values[Unit::Second.as_usize()];
        let millis = self.values[Unit::Millisecond.as_usize()] as u32;
        let micros = self.values[Unit::Microsecond.as_usize()] as u32;
        let nanos = self.values[Unit::Nanosecond.as_usize()] as u32;

        let total_secs = (hours * 3600) + (mins * 60) + secs;
        let total_nanos = (millis * 1_000_000) + (micros * 1_000) + nanos;
        let sdur = core::time::Duration::new(total_secs, total_nanos);

        Ok(sdur)
    }

    /// The "general" implementation of `DurationUnits::to_unsigned_duration`.
    ///
    /// This handles all possible cases, including fractional units, with good
    /// error handling. Basically, we take this path when we think an error
    /// _could_ occur. But this function is more bloaty and does more work, so
    /// the more it can be avoided, the better.
    #[cold]
    #[inline(never)]
    fn to_unsigned_duration_general(
        &self,
    ) -> Result<core::time::Duration, Error> {
        #[inline]
        const fn try_from_hours(hours: u64) -> Option<core::time::Duration> {
            // OK because (SECS_PER_MINUTE*MINS_PER_HOUR)!={-1,0}.
            const MAX_HOUR: u64 = u64::MAX / (60 * 60);
            if hours > MAX_HOUR {
                return None;
            }
            Some(core::time::Duration::from_secs(hours * 60 * 60))
        }

        #[inline]
        const fn try_from_mins(mins: u64) -> Option<core::time::Duration> {
            // OK because SECS_PER_MINUTE!={-1,0}.
            const MAX_MINUTE: u64 = u64::MAX / 60;
            if mins > MAX_MINUTE {
                return None;
            }
            Some(core::time::Duration::from_secs(mins * 60))
        }

        if self.sign.is_negative() {
            return Err(Error::from(E::NotAllowedNegative));
        }

        let (min, max) = self.get_min_max_units()?;
        if max > Unit::Hour {
            return Err(Error::from(E::NotAllowedCalendarUnit { unit: max }));
        }

        let mut sdur = core::time::Duration::ZERO;
        if self.values[Unit::Hour.as_usize()] != 0 {
            let value = self.values[Unit::Hour.as_usize()];
            sdur = try_from_hours(value)
                .and_then(|nanos| sdur.checked_add(nanos))
                .ok_or(E::OverflowForUnit { unit: Unit::Hour })?;
        }
        if self.values[Unit::Minute.as_usize()] != 0 {
            let value = self.values[Unit::Minute.as_usize()];
            sdur = try_from_mins(value)
                .and_then(|nanos| sdur.checked_add(nanos))
                .ok_or(E::OverflowForUnit { unit: Unit::Minute })?;
        }
        if self.values[Unit::Second.as_usize()] != 0 {
            let value = self.values[Unit::Second.as_usize()];
            sdur = core::time::Duration::from_secs(value)
                .checked_add(sdur)
                .ok_or(E::OverflowForUnit { unit: Unit::Second })?;
        }
        if self.values[Unit::Millisecond.as_usize()] != 0 {
            let value = self.values[Unit::Millisecond.as_usize()];
            sdur = core::time::Duration::from_millis(value)
                .checked_add(sdur)
                .ok_or(E::OverflowForUnit { unit: Unit::Millisecond })?;
        }
        if self.values[Unit::Microsecond.as_usize()] != 0 {
            let value = self.values[Unit::Microsecond.as_usize()];
            sdur = core::time::Duration::from_micros(value)
                .checked_add(sdur)
                .ok_or(E::OverflowForUnit { unit: Unit::Microsecond })?;
        }
        if self.values[Unit::Nanosecond.as_usize()] != 0 {
            let value = self.values[Unit::Nanosecond.as_usize()];
            sdur = core::time::Duration::from_nanos(value)
                .checked_add(sdur)
                .ok_or(E::OverflowForUnit { unit: Unit::Nanosecond })?;
        }

        if let Some(fraction) = self.get_fraction()? {
            sdur = sdur
                .checked_add(
                    fractional_duration(min, fraction)?.unsigned_abs(),
                )
                .ok_or(E::OverflowForUnitFractional { unit: Unit::Hour })?;
        }

        Ok(sdur)
    }

    /// Returns the minimum unit set.
    ///
    /// This only returns `None` when no units have been set.
    pub(crate) fn get_min(&self) -> Option<Unit> {
        self.min
    }

    /// Returns the minimum and maximum units set.
    ///
    /// This returns an error if no units were set. (Since this means there
    /// were no parsed duration components.)
    fn get_min_max_units(&self) -> Result<(Unit, Unit), Error> {
        let (Some(min), Some(max)) = (self.min, self.max) else {
            return Err(Error::from(E::EmptyDuration));
        };
        Ok((min, max))
    }

    /// Returns the corresponding unit value using the set signed-ness.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn get_unit_value(&self, unit: Unit) -> Result<i64, Error> {
        const I64_MIN_ABS: u64 = i64::MIN.unsigned_abs();

        #[cold]
        #[inline(never)]
        fn general(unit: Unit, value: u64, sign: Sign) -> Result<i64, Error> {
            // As a weird special case, when we need to represent i64::MIN,
            // we'll have a unit value of `|i64::MIN|` as a `u64`. We can't
            // convert that to a positive `i64` first, since it will overflow.
            if sign.is_negative() && value == I64_MIN_ABS {
                return Ok(i64::MIN);
            }
            // Otherwise, if a conversion to `i64` fails, then that failure
            // is correct.
            let mut value = i64::try_from(value)
                .map_err(|_| E::SignedOverflowForUnit { unit })?;
            if sign.is_negative() {
                value = value
                    .checked_neg()
                    .ok_or(E::SignedOverflowForUnit { unit })?;
            }
            Ok(value)
        }

        let sign = self.get_sign();
        let value = self.values[unit.as_usize()];
        if value >= I64_MIN_ABS {
            return general(unit, value, sign);
        }
        let mut value = value as i64;
        if sign.is_negative() {
            value = -value;
        }
        Ok(value)
    }

    /// Returns the fraction using the set signed-ness.
    ///
    /// This returns `None` when no fraction has been set.
    fn get_fraction(&self) -> Result<Option<i32>, Error> {
        let Some(fraction) = self.fraction else {
            return Ok(None);
        };
        // OK because `set_fraction` guarantees `0..=999_999_999`.
        let mut fraction = fraction as i32;
        if self.get_sign().is_negative() {
            // OK because `set_fraction` guarantees `0..=999_999_999`.
            fraction = -fraction;
        }
        Ok(Some(fraction))
    }

    /// Returns the sign that should be applied to each individual unit.
    fn get_sign(&self) -> Sign {
        if self.any_non_zero_units {
            self.sign
        } else {
            Sign::Zero
        }
    }
}

/// Parses an optional fractional number from the start of `input`.
///
/// If `input` does not begin with a `.` (or a `,`), then this returns `None`
/// and no input is consumed. Otherwise, up to 9 ASCII digits are parsed after
/// the decimal separator.
///
/// While this is most typically used to parse the fractional component of
/// second units, it is also used to parse the fractional component of hours or
/// minutes in ISO 8601 duration parsing, and milliseconds and microseconds in
/// the "friendly" duration format. The return type in that case is obviously a
/// misnomer, but the range of possible values is still correct. (That is, the
/// fractional component of an hour is still limited to 9 decimal places per
/// the Temporal spec.)
///
/// The number returned is guaranteed to be in the range `0..=999_999_999`.
#[cfg_attr(feature = "perf-inline", inline(always))]
pub(crate) fn parse_temporal_fraction<'i>(
    input: &'i [u8],
) -> Result<Parsed<'i, Option<u32>>, Error> {
    // TimeFraction :::
    //   TemporalDecimalFraction
    //
    // TemporalDecimalFraction :::
    //   TemporalDecimalSeparator DecimalDigit
    //   TemporalDecimalSeparator DecimalDigit DecimalDigit
    //   TemporalDecimalSeparator DecimalDigit DecimalDigit DecimalDigit
    //   TemporalDecimalSeparator DecimalDigit DecimalDigit DecimalDigit
    //                            DecimalDigit
    //   TemporalDecimalSeparator DecimalDigit DecimalDigit DecimalDigit
    //                            DecimalDigit DecimalDigit
    //   TemporalDecimalSeparator DecimalDigit DecimalDigit DecimalDigit
    //                            DecimalDigit DecimalDigit DecimalDigit
    //   TemporalDecimalSeparator DecimalDigit DecimalDigit DecimalDigit
    //                            DecimalDigit DecimalDigit DecimalDigit
    //                            DecimalDigit
    //   TemporalDecimalSeparator DecimalDigit DecimalDigit DecimalDigit
    //                            DecimalDigit DecimalDigit DecimalDigit
    //                            DecimalDigit DecimalDigit
    //   TemporalDecimalSeparator DecimalDigit DecimalDigit DecimalDigit
    //                            DecimalDigit DecimalDigit DecimalDigit
    //                            DecimalDigit DecimalDigit DecimalDigit
    //
    // TemporalDecimalSeparator ::: one of
    //   . ,
    //
    // DecimalDigit :: one of
    //   0 1 2 3 4 5 6 7 8 9

    #[inline(never)]
    fn imp<'i>(mut input: &'i [u8]) -> Result<Parsed<'i, Option<u32>>, Error> {
        let mkdigits = parse::slicer(input);
        while mkdigits(input).len() <= 8
            && input.first().map_or(false, u8::is_ascii_digit)
        {
            input = &input[1..];
        }
        let digits = mkdigits(input);
        if digits.is_empty() {
            return Err(Error::from(E::MissingFractionalDigits));
        }
        // I believe this error can never happen, since we know we have no more
        // than 9 ASCII digits. Any sequence of 9 ASCII digits can be parsed
        // into an `i64`.
        let nanoseconds =
            parse::fraction(digits).context(E::InvalidFraction)?;
        // OK because parsing is forcefully limited to 9 digits,
        // which can never be greater than `999_999_99`,
        // which is less than `u32::MAX`.
        let nanoseconds = nanoseconds as u32;
        Ok(Parsed { value: Some(nanoseconds), input })
    }

    if input.is_empty() || (input[0] != b'.' && input[0] != b',') {
        return Ok(Parsed { value: None, input });
    }
    imp(&input[1..])
}

/// This routine returns a span based on the given unit and value with
/// fractional time applied to it.
///
/// For example, given a span like `P1dT1.5h`, the `unit` would be
/// `Unit::Hour`, the `value` would be `1` and the `fraction` would be
/// `500_000_000`. The span given would just be `1d`. The span returned would
/// be `P1dT1h30m`.
///
/// Note that `fraction` can be a fractional hour, minute, second, millisecond
/// or microsecond (even though its type suggests its only a fraction of a
/// second). When milliseconds or microseconds, the given fraction has any
/// sub-nanosecond precision truncated.
///
/// # Errors
///
/// This can error if the resulting units would be too large for the limits on
/// a `span`. This also errors if `unit` is not `Hour`, `Minute`, `Second`,
/// `Millisecond` or `Microsecond`.
#[inline(never)]
fn fractional_time_to_span(
    unit: Unit,
    value: i64,
    fraction: i32,
    mut span: Span,
) -> Result<Span, Error> {
    const MAX_HOURS: i64 = b::SpanHours::MAX as i64;
    const MAX_MINS: i64 = b::SpanMinutes::MAX;
    const MAX_SECS: i64 = b::SpanSeconds::MAX;
    const MAX_MILLIS: i128 = b::SpanMilliseconds::MAX as i128;
    const MAX_MICROS: i128 = b::SpanMicroseconds::MAX as i128;
    const MIN_HOURS: i64 = b::SpanHours::MIN as i64;
    const MIN_MINS: i64 = b::SpanMinutes::MIN;
    const MIN_SECS: i64 = b::SpanSeconds::MIN;
    const MIN_MILLIS: i128 = b::SpanMilliseconds::MIN as i128;
    const MIN_MICROS: i128 = b::SpanMicroseconds::MIN as i128;

    // We switch everything over to nanoseconds and then divy that up as
    // appropriate. In general, we always create a balanced span, but there
    // are some cases where we can't. For example, if one serializes a span
    // with both the maximum number of seconds and the maximum number of
    // milliseconds, then this just can't be balanced due to the limits on
    // each of the units. When this kind of span is serialized to a string,
    // it results in a second value that is actually bigger than the maximum
    // allowed number of seconds in a span. So here, we have to reverse that
    // operation and spread the seconds over smaller units. This in turn
    // creates an unbalanced span. Annoying.
    //
    // The above is why we have `if unit_value > MAX { <do adjustments> }` in
    // the balancing code below. Basically, if we overshoot our limit, we back
    // out anything over the limit and carry it over to the lesser units. If
    // our value is truly too big, then the final call to set nanoseconds will
    // fail.
    let mut sdur = fractional_time_to_duration(unit, value, fraction)?;

    if unit >= Unit::Hour && !sdur.is_zero() {
        let (mut hours, rem) = sdur.as_hours_with_remainder();
        sdur = rem;
        if hours > MAX_HOURS {
            sdur += SignedDuration::from_hours(hours - MAX_HOURS);
            hours = MAX_HOURS;
        } else if hours < MIN_HOURS {
            sdur += SignedDuration::from_hours(hours - MIN_HOURS);
            hours = MIN_HOURS;
        }
        // OK because we just checked that our units are in range.
        span = span.hours(hours);
    }
    if unit >= Unit::Minute && !sdur.is_zero() {
        let (mut mins, rem) = sdur.as_mins_with_remainder();
        sdur = rem;
        if mins > MAX_MINS {
            sdur += SignedDuration::from_mins(mins - MAX_MINS);
            mins = MAX_MINS;
        } else if mins < MIN_MINS {
            sdur += SignedDuration::from_mins(mins - MIN_MINS);
            mins = MIN_MINS;
        }
        // OK because we just checked that our units are in range.
        span = span.minutes(mins);
    }
    if unit >= Unit::Second && !sdur.is_zero() {
        let (mut secs, rem) = sdur.as_secs_with_remainder();
        sdur = rem;
        if secs > MAX_SECS {
            sdur += SignedDuration::from_secs(secs - MAX_SECS);
            secs = MAX_SECS;
        } else if secs < MIN_SECS {
            sdur += SignedDuration::from_secs(secs - MIN_SECS);
            secs = MIN_SECS;
        }
        // OK because we just checked that our units are in range.
        span = span.seconds(secs);
    }
    if unit >= Unit::Millisecond && !sdur.is_zero() {
        let (mut millis, rem) = sdur.as_millis_with_remainder();
        sdur = rem;
        if millis > MAX_MILLIS {
            sdur += SignedDuration::from_millis_i128(millis - MAX_MILLIS);
            millis = MAX_MILLIS;
        } else if millis < MIN_MILLIS {
            sdur += SignedDuration::from_millis_i128(millis - MIN_MILLIS);
            millis = MIN_MILLIS;
        }
        // OK because we just checked that our units are in range.
        span = span.milliseconds(i64::try_from(millis).unwrap());
    }
    if unit >= Unit::Microsecond && !sdur.is_zero() {
        let (mut micros, rem) = sdur.as_micros_with_remainder();
        sdur = rem;
        if micros > MAX_MICROS {
            sdur += SignedDuration::from_micros_i128(micros - MAX_MICROS);
            micros = MAX_MICROS;
        } else if micros < MIN_MICROS {
            sdur += SignedDuration::from_micros_i128(micros - MIN_MICROS);
            micros = MIN_MICROS;
        }
        // OK because we just checked that our units are in range.
        span = span.microseconds(i64::try_from(micros).unwrap());
    }
    if !sdur.is_zero() {
        let nanos = sdur.as_nanos();
        let nanos64 =
            i64::try_from(nanos).map_err(|_| E::InvalidFractionNanos)?;
        span =
            span.try_nanoseconds(nanos64).context(E::InvalidFractionNanos)?;
    }

    Ok(span)
}

/// Like `fractional_time_to_span`, but just converts the fraction of the given
/// unit to a signed duration.
///
/// Since a signed duration doesn't keep track of individual units, there is
/// no loss of fidelity between it and ISO 8601 durations like there is for
/// `Span`.
///
/// Note that `fraction` can be a fractional hour, minute, second, millisecond
/// or microsecond (even though its type suggests it's only a fraction of a
/// second). When milliseconds or microseconds, the given fraction has any
/// sub-nanosecond precision truncated.
///
/// # Errors
///
/// This returns an error if `unit` is not `Hour`, `Minute`, `Second`,
/// `Millisecond` or `Microsecond`.
#[inline(never)]
fn fractional_time_to_duration(
    unit: Unit,
    value: i64,
    fraction: i32,
) -> Result<SignedDuration, Error> {
    let sdur = duration_unit_value(unit, value)?;
    let fraction_dur = fractional_duration(unit, fraction)?;
    Ok(sdur
        .checked_add(fraction_dur)
        .ok_or(E::OverflowForUnitFractional { unit })?)
}

/// Converts the fraction of the given unit to a signed duration.
///
/// Since a signed duration doesn't keep track of individual units, there is
/// no loss of fidelity between it and ISO 8601 durations like there is for
/// `Span`. Thus, we can do something far less complicated.
///
/// # Panics
///
/// When `fraction` isn't in the range `-999_999_999..=999_999_999`.
///
/// # Errors
///
/// This returns an error if `unit` is not `Hour`, `Minute`, `Second`,
/// `Millisecond` or `Microsecond`.
#[inline(never)]
fn fractional_duration(
    unit: Unit,
    fraction: i32,
) -> Result<SignedDuration, Error> {
    let fraction = i64::from(fraction);
    let nanos = match unit {
        Unit::Hour => fraction * b::SECS_PER_HOUR,
        Unit::Minute => fraction * b::SECS_PER_MIN,
        Unit::Second => fraction,
        Unit::Millisecond => fraction / b::NANOS_PER_MICRO,
        Unit::Microsecond => fraction / b::NANOS_PER_MILLI,
        unit => {
            return Err(Error::from(E::NotAllowedFractionalUnit {
                found: unit,
            }));
        }
    };
    Ok(SignedDuration::from_nanos(nanos))
}

/// Returns the given parsed value, interpreted as the given unit, as a
/// `SignedDuration`.
///
/// If the given unit is not supported for signed durations (i.e., calendar
/// units), or if converting the given value to a `SignedDuration` for the
/// given units overflows, then an error is returned.
#[cfg_attr(feature = "perf-inline", inline(always))]
fn duration_unit_value(
    unit: Unit,
    value: i64,
) -> Result<SignedDuration, Error> {
    // Convert our parsed unit into a number of nanoseconds.
    //
    // Note also that overflow isn't possible here for units less than minutes,
    // since a `SignedDuration` supports all `i64` second values.
    let sdur = match unit {
        Unit::Hour => {
            let seconds = value
                .checked_mul(b::SECS_PER_HOUR)
                .ok_or(E::ConversionToSecondsFailed { unit: Unit::Hour })?;
            SignedDuration::from_secs(seconds)
        }
        Unit::Minute => {
            let seconds = value
                .checked_mul(b::SECS_PER_MIN)
                .ok_or(E::ConversionToSecondsFailed { unit: Unit::Minute })?;
            SignedDuration::from_secs(seconds)
        }
        Unit::Second => SignedDuration::from_secs(value),
        Unit::Millisecond => SignedDuration::from_millis(value),
        Unit::Microsecond => SignedDuration::from_micros(value),
        Unit::Nanosecond => SignedDuration::from_nanos(value),
        unsupported => {
            return Err(Error::from(E::NotAllowedCalendarUnit {
                unit: unsupported,
            }))
        }
    };
    Ok(sdur)
}
