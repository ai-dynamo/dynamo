use crate::{
    error::{fmt::friendly::Error as E, ErrorContext},
    fmt::{
        friendly::parser_label,
        util::{parse_temporal_fraction, DurationUnits},
        Parsed,
    },
    util::{b::Sign, parse},
    Error, SignedDuration, Span, Unit,
};

/// A parser for Jiff's "friendly" duration format.
///
/// See the [module documentation](super) for more details on the precise
/// format supported by this parser.
///
/// Unlike [`SpanPrinter`](super::SpanPrinter), this parser doesn't have any
/// configuration knobs. While it may grow some in the future, the approach
/// taken here is for the parser to support the entire grammar. That is, the
/// parser can parse anything emitted by `SpanPrinter`. (And indeed, the
/// parser can even handle things that the printer can't emit due to lack of
/// configurability. For example, `1hour1m` is a valid friendly duration,
/// but `SpanPrinter` cannot emit it due to a mixing of verbose and compact
/// designator labels.)
///
/// # Advice
///
/// Since this parser has no configuration, there are generally only two reasons
/// why you might want to use this type specifically:
///
/// 1. You need to parse from `&[u8]`.
/// 2. You need to parse _only_ the "friendly" format.
///
/// Otherwise, you can use the `FromStr` implementations on both `Span` and
/// `SignedDuration`, which automatically support the friendly format in
/// addition to the ISO 8601 format simultaneously:
///
/// ```
/// use jiff::{SignedDuration, Span, ToSpan};
///
/// let span: Span = "5 years, 2 months".parse()?;
/// assert_eq!(span, 5.years().months(2).fieldwise());
///
/// let sdur: SignedDuration = "5 hours, 2 minutes".parse()?;
/// assert_eq!(sdur, SignedDuration::new(5 * 60 * 60 + 2 * 60, 0));
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Example
///
/// This example shows how to parse a `Span` directly from `&str`:
///
/// ```
/// use jiff::{fmt::friendly::SpanParser, ToSpan};
///
/// static PARSER: SpanParser = SpanParser::new();
///
/// let string = "1 year, 3 months, 15:00:01.3";
/// let span = PARSER.parse_span(string)?;
/// assert_eq!(
///     span,
///     1.year().months(3).hours(15).seconds(1).milliseconds(300).fieldwise(),
/// );
///
/// // Negative durations are supported too!
/// let string = "1 year, 3 months, 15:00:01.3 ago";
/// let span = PARSER.parse_span(string)?;
/// assert_eq!(
///     span,
///     -1.year().months(3).hours(15).seconds(1).milliseconds(300).fieldwise(),
/// );
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug, Default)]
pub struct SpanParser {
    _private: (),
}

impl SpanParser {
    /// Creates a new parser for the "friendly" duration format.
    ///
    /// The parser returned uses the default configuration. (Although, at time
    /// of writing, there are no available configuration options for this
    /// parser.) This is identical to `SpanParser::default`, but it can be used
    /// in a `const` context.
    ///
    /// # Example
    ///
    /// This example shows how to parse a `Span` directly from `&[u8]`:
    ///
    /// ```
    /// use jiff::{fmt::friendly::SpanParser, ToSpan};
    ///
    /// static PARSER: SpanParser = SpanParser::new();
    ///
    /// let bytes = b"1 year 3 months 15 hours 1300ms";
    /// let span = PARSER.parse_span(bytes)?;
    /// assert_eq!(
    ///     span,
    ///     1.year().months(3).hours(15).milliseconds(1300).fieldwise(),
    /// );
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub const fn new() -> SpanParser {
        SpanParser { _private: () }
    }

    /// Run the parser on the given string (which may be plain bytes) and,
    /// if successful, return the parsed `Span`.
    ///
    /// See the [module documentation](super) for more details on the specific
    /// grammar supported by this parser.
    ///
    /// # Example
    ///
    /// This shows a number of different duration formats that can be parsed
    /// into a `Span`:
    ///
    /// ```
    /// use jiff::{fmt::friendly::SpanParser, ToSpan};
    ///
    /// let spans = [
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
    ///     (
    ///         "1 yr, 1 month, 1 d, 1 h, 1 min 1.1 second",
    ///         1.year().months(1).days(1).hours(1).minutes(1).seconds(1).milliseconds(100),
    ///     ),
    /// ];
    ///
    /// static PARSER: SpanParser = SpanParser::new();
    /// for (string, span) in spans {
    ///     let parsed = PARSER.parse_span(string)?;
    ///     assert_eq!(
    ///         span.fieldwise(),
    ///         parsed.fieldwise(),
    ///         "result of parsing {string:?}",
    ///     );
    /// }
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn parse_span<I: AsRef<[u8]>>(&self, input: I) -> Result<Span, Error> {
        #[inline(never)]
        fn imp(span_parser: &SpanParser, input: &[u8]) -> Result<Span, Error> {
            let mut builder = DurationUnits::default();
            let parsed = span_parser.parse(input, &mut builder)?;
            let parsed = parsed.and_then(|_| builder.to_span())?;
            parsed.into_full()
        }

        let input = input.as_ref();
        imp(self, input).context(E::Failed)
    }

    /// Run the parser on the given string (which may be plain bytes) and,
    /// if successful, return the parsed `SignedDuration`.
    ///
    /// See the [module documentation](super) for more details on the specific
    /// grammar supported by this parser.
    ///
    /// # Example
    ///
    /// This shows a number of different duration formats that can be parsed
    /// into a `SignedDuration`:
    ///
    /// ```
    /// use jiff::{fmt::friendly::SpanParser, SignedDuration};
    ///
    /// let durations = [
    ///     ("2h30m", SignedDuration::from_secs(2 * 60 * 60 + 30 * 60)),
    ///     ("2 hrs 30 mins", SignedDuration::from_secs(2 * 60 * 60 + 30 * 60)),
    ///     ("2 hours 30 minutes", SignedDuration::from_secs(2 * 60 * 60 + 30 * 60)),
    ///     ("2 hrs 30 minutes", SignedDuration::from_secs(2 * 60 * 60 + 30 * 60)),
    ///     ("2.5h", SignedDuration::from_secs(2 * 60 * 60 + 30 * 60)),
    ///     ("1m", SignedDuration::from_mins(1)),
    ///     ("1.5m", SignedDuration::from_secs(90)),
    ///     ("0.0021s", SignedDuration::new(0, 2_100_000)),
    ///     ("0s", SignedDuration::ZERO),
    ///     ("0.000000001s", SignedDuration::from_nanos(1)),
    /// ];
    ///
    /// static PARSER: SpanParser = SpanParser::new();
    /// for (string, duration) in durations {
    ///     let parsed = PARSER.parse_duration(string)?;
    ///     assert_eq!(duration, parsed, "result of parsing {string:?}");
    /// }
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn parse_duration<I: AsRef<[u8]>>(
        &self,
        input: I,
    ) -> Result<SignedDuration, Error> {
        #[inline(never)]
        fn imp(
            span_parser: &SpanParser,
            input: &[u8],
        ) -> Result<SignedDuration, Error> {
            let mut builder = DurationUnits::default();
            let parsed = span_parser.parse(input, &mut builder)?;
            let parsed = parsed.and_then(|_| builder.to_signed_duration())?;
            parsed.into_full()
        }

        let input = input.as_ref();
        imp(self, input).context(E::Failed)
    }

    /// Run the parser on the given string (which may be plain bytes) and,
    /// if successful, return the parsed `std::time::Duration`.
    ///
    /// See the [module documentation](super) for more details on the specific
    /// grammar supported by this parser.
    ///
    /// # Example
    ///
    /// This shows a number of different duration formats that can be parsed
    /// into a `std::time::Duration`:
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// use jiff::fmt::friendly::SpanParser;
    ///
    /// let durations = [
    ///     ("2h30m", Duration::from_secs(2 * 60 * 60 + 30 * 60)),
    ///     ("2 hrs 30 mins", Duration::from_secs(2 * 60 * 60 + 30 * 60)),
    ///     ("2 hours 30 minutes", Duration::from_secs(2 * 60 * 60 + 30 * 60)),
    ///     ("2 hrs 30 minutes", Duration::from_secs(2 * 60 * 60 + 30 * 60)),
    ///     ("2.5h", Duration::from_secs(2 * 60 * 60 + 30 * 60)),
    ///     ("1m", Duration::from_secs(1 * 60)),
    ///     ("1.5m", Duration::from_secs(90)),
    ///     ("0.0021s", Duration::new(0, 2_100_000)),
    ///     ("0s", Duration::ZERO),
    ///     ("0.000000001s", Duration::from_nanos(1)),
    /// ];
    ///
    /// static PARSER: SpanParser = SpanParser::new();
    /// for (string, duration) in durations {
    ///     let parsed = PARSER.parse_unsigned_duration(string)?;
    ///     assert_eq!(duration, parsed, "result of parsing {string:?}");
    /// }
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn parse_unsigned_duration<I: AsRef<[u8]>>(
        &self,
        input: I,
    ) -> Result<core::time::Duration, Error> {
        #[inline(never)]
        fn imp(
            span_parser: &SpanParser,
            input: &[u8],
        ) -> Result<core::time::Duration, Error> {
            let mut builder = DurationUnits::default();
            let parsed = span_parser.parse(input, &mut builder)?;
            let parsed =
                parsed.and_then(|_| builder.to_unsigned_duration())?;
            let d = parsed.value;
            parsed.into_full_with(format_args!("{d:?}"))
        }

        let input = input.as_ref();
        imp(self, input).context(E::Failed)
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse<'i>(
        &self,
        input: &'i [u8],
        builder: &mut DurationUnits,
    ) -> Result<Parsed<'i, ()>, Error> {
        if input.is_empty() {
            return Err(Error::from(E::Empty));
        }
        // Guard prefix sign parsing to avoid the function call, which is
        // marked unlineable to keep the fast path tighter.
        let (sign, input) =
            if !input.first().map_or(false, |&b| matches!(b, b'+' | b'-')) {
                (None, input)
            } else {
                let Parsed { value: sign, input } =
                    self.parse_prefix_sign(input);
                (sign, input)
            };

        let Parsed { value, input } = self.parse_unit_value(input)?;
        let Some(first_unit_value) = value else {
            return Err(Error::from(E::ExpectedIntegerAfterSign));
        };

        let Parsed { input, .. } =
            self.parse_duration_units(input, first_unit_value, builder)?;

        // As with the prefix sign parsing, guard it to avoid calling the
        // function.
        let (sign, input) = if !input.first().map_or(false, is_whitespace) {
            (sign.unwrap_or(Sign::Positive), input)
        } else {
            let parsed = self.parse_suffix_sign(sign, input)?;
            (parsed.value, parsed.input)
        };
        builder.set_sign(sign);
        Ok(Parsed { value: (), input })
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_duration_units<'i>(
        &self,
        mut input: &'i [u8],
        first_unit_value: u64,
        builder: &mut DurationUnits,
    ) -> Result<Parsed<'i, ()>, Error> {
        let mut parsed_any_after_comma = true;
        let mut value = first_unit_value;
        loop {
            let parsed = self.parse_hms_maybe(input, value)?;
            input = parsed.input;
            if let Some(hms) = parsed.value {
                builder.set_hms(
                    hms.hour,
                    hms.minute,
                    hms.second,
                    hms.fraction,
                )?;
                break;
            }

            let fraction =
                if input.first().map_or(false, |&b| b == b'.' || b == b',') {
                    let parsed = parse_temporal_fraction(input)?;
                    input = parsed.input;
                    parsed.value
                } else {
                    None
                };

            // Eat any optional whitespace between the unit value and label.
            input = self.parse_optional_whitespace(input).input;

            // Parse the actual unit label/designator.
            let parsed = self.parse_unit_designator(input)?;
            input = parsed.input;
            let unit = parsed.value;

            // A comma is allowed to immediately follow the designator.
            // Since this is a rarer case, we guard it with a check to see
            // if the comma is there and only then call the function (which is
            // marked unlineable to try and keep the hot path tighter).
            if input.first().map_or(false, |&b| b == b',') {
                input = self.parse_optional_comma(input)?.input;
                parsed_any_after_comma = false;
            }

            builder.set_unit_value(unit, value)?;
            if let Some(fraction) = fraction {
                builder.set_fraction(fraction)?;
                // Once we see a fraction, we are done. We don't permit parsing
                // any more units. That is, a fraction can only occur on the
                // lowest unit of time.
                break;
            }

            // Eat any optional whitespace after the designator (or comma) and
            // before the next unit value. But if we don't see a unit value,
            // we don't eat the whitespace.
            let after_whitespace = self.parse_optional_whitespace(input).input;
            let parsed = self.parse_unit_value(after_whitespace)?;
            value = match parsed.value {
                None => break,
                Some(value) => value,
            };
            input = parsed.input;
            parsed_any_after_comma = true;
        }
        if !parsed_any_after_comma {
            return Err(Error::from(E::ExpectedOneMoreUnitAfterComma));
        }
        Ok(Parsed { value: (), input })
    }

    /// This possibly parses a `HH:MM:SS[.fraction]`.
    ///
    /// This expects that a unit value has been parsed and looks for a `:`
    /// at `input[0]`. If `:` is found, then this proceeds to parse HMS.
    /// Otherwise, a `None` value is returned.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_hms_maybe<'i>(
        &self,
        input: &'i [u8],
        hour: u64,
    ) -> Result<Parsed<'i, Option<HMS>>, Error> {
        let Some((&first, tail)) = input.split_first() else {
            return Ok(Parsed { input, value: None });
        };
        if first != b':' {
            return Ok(Parsed { input, value: None });
        }
        let Parsed { input, value } = self.parse_hms(tail, hour)?;
        Ok(Parsed { input, value: Some(value) })
    }

    /// This parses a `HH:MM:SS[.fraction]` when it is known/expected to be
    /// present.
    ///
    /// This is also marked as non-inlined since we expect this to be a
    /// less common case. Where as `parse_hms_maybe` is called unconditionally
    /// to check to see if the HMS should be parsed.
    ///
    /// This assumes that the beginning of `input` immediately follows the
    /// first `:` in `HH:MM:SS[.fraction]`.
    #[inline(never)]
    fn parse_hms<'i>(
        &self,
        input: &'i [u8],
        hour: u64,
    ) -> Result<Parsed<'i, HMS>, Error> {
        let Parsed { input, value } = self.parse_unit_value(input)?;
        let minute = value.ok_or(E::ExpectedMinuteAfterHour)?;

        let (&first, input) =
            input.split_first().ok_or(E::ExpectedColonAfterMinute)?;
        if first != b':' {
            return Err(Error::from(E::ExpectedColonAfterMinute));
        }

        let Parsed { input, value } = self.parse_unit_value(input)?;
        let second = value.ok_or(E::ExpectedSecondAfterMinute)?;
        let (fraction, input) =
            if input.first().map_or(false, |&b| b == b'.' || b == b',') {
                let parsed = parse_temporal_fraction(input)?;
                (parsed.value, parsed.input)
            } else {
                (None, input)
            };
        let hms = HMS { hour, minute, second, fraction };
        Ok(Parsed { input, value: hms })
    }

    /// Parsed a unit value, i.e., an integer.
    ///
    /// If no digits (`[0-9]`) were found at the current position of the parser
    /// then `None` is returned. This means, for example, that parsing a
    /// duration should stop.
    ///
    /// Note that this is safe to call on untrusted input. It will not attempt
    /// to consume more input than could possibly fit into a parsed integer.
    ///
    /// Since this returns a `u64`, it is possible that an integer that cannot
    /// fit into an `i64` is returned. Callers should handle this. (Indeed,
    /// `DurationUnits` handles this case.)
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_unit_value<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, Option<u64>>, Error> {
        let (value, input) = parse::u64_prefix(input)?;
        Ok(Parsed { value, input })
    }

    /// Parse a unit designator, e.g., `years` or `nano`.
    ///
    /// If no designator could be found, including if the given `input` is
    /// empty, then this return an error.
    ///
    /// This does not attempt to handle leading or trailing whitespace.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_unit_designator<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, Unit>, Error> {
        let (unit, len) =
            parser_label::find(input).ok_or(E::ExpectedUnitSuffix)?;
        Ok(Parsed { value: unit, input: &input[len..] })
    }

    /// Parses an optional prefix sign from the given input.
    ///
    /// A prefix sign is either a `+` or a `-`. If neither are found, then
    /// `None` is returned.
    #[inline(never)]
    fn parse_prefix_sign<'i>(
        &self,
        input: &'i [u8],
    ) -> Parsed<'i, Option<Sign>> {
        let Some(sign) = input.first().copied() else {
            return Parsed { value: None, input };
        };
        let sign = if sign == b'+' {
            Sign::Positive
        } else if sign == b'-' {
            Sign::Negative
        } else {
            return Parsed { value: None, input };
        };
        Parsed { value: Some(sign), input: &input[1..] }
    }

    /// Parses an optional suffix sign from the given input.
    ///
    /// This requires, as input, the result of parsing a prefix sign since this
    /// will return an error if both a prefix and a suffix sign were found.
    ///
    /// A suffix sign is the string `ago`. Any other string means that there is
    /// no suffix sign. This will also look for mandatory whitespace and eat
    /// any additional optional whitespace. i.e., This should be called
    /// immediately after parsing the last unit designator/label.
    ///
    /// Regardless of whether a prefix or suffix sign was found, a definitive
    /// sign is returned. (When there's no prefix or suffix sign, then the sign
    /// returned is positive.)
    #[inline(never)]
    fn parse_suffix_sign<'i>(
        &self,
        prefix_sign: Option<Sign>,
        mut input: &'i [u8],
    ) -> Result<Parsed<'i, Sign>, Error> {
        if !input.first().map_or(false, is_whitespace) {
            let sign = prefix_sign.unwrap_or(Sign::Positive);
            return Ok(Parsed { value: sign, input });
        }
        // Eat any additional whitespace we find before looking for 'ago'.
        input = self.parse_optional_whitespace(&input[1..]).input;
        let (suffix_sign, input) =
            if let Some(tail) = input.strip_prefix(b"ago") {
                (Some(Sign::Negative), tail)
            } else {
                (None, input)
            };
        let sign = match (prefix_sign, suffix_sign) {
            (Some(_), Some(_)) => {
                return Err(Error::from(E::ExpectedOneSign));
            }
            (Some(sign), None) => sign,
            (None, Some(sign)) => sign,
            (None, None) => Sign::Positive,
        };
        Ok(Parsed { value: sign, input })
    }

    /// Parses an optional comma following a unit designator.
    ///
    /// If a comma is seen, then it is mandatory that it be followed by
    /// whitespace.
    ///
    /// This also takes care to provide a custom error message if the end of
    /// input is seen after a comma.
    ///
    /// If `input` doesn't start with a comma, then this is a no-op.
    #[inline(never)]
    fn parse_optional_comma<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, ()>, Error> {
        let Some((&first, tail)) = input.split_first() else {
            return Ok(Parsed { value: (), input });
        };
        if first != b',' {
            return Ok(Parsed { value: (), input });
        }

        let (second, input) = tail
            .split_first()
            .ok_or(E::ExpectedWhitespaceAfterCommaEndOfInput)?;
        if !is_whitespace(second) {
            return Err(Error::from(E::ExpectedWhitespaceAfterComma {
                byte: *second,
            }));
        }
        Ok(Parsed { value: (), input })
    }

    /// Parses zero or more bytes of ASCII whitespace.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_optional_whitespace<'i>(
        &self,
        mut input: &'i [u8],
    ) -> Parsed<'i, ()> {
        while input.first().map_or(false, is_whitespace) {
            input = &input[1..];
        }
        Parsed { value: (), input }
    }
}

/// A type that represents the parsed components of `HH:MM:SS[.fraction]`.
#[derive(Debug)]
struct HMS {
    hour: u64,
    minute: u64,
    second: u64,
    fraction: Option<u32>,
}

/// Returns true if the byte is ASCII whitespace.
#[cfg_attr(feature = "perf-inline", inline(always))]
fn is_whitespace(byte: &u8) -> bool {
    matches!(*byte, b' ' | b'\t' | b'\n' | b'\r' | b'\x0C')
}

#[cfg(feature = "alloc")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_span_basic() {
        let p = |s: &str| SpanParser::new().parse_span(s).unwrap();

        insta::assert_snapshot!(p("5 years"), @"P5Y");
        insta::assert_snapshot!(p("5 years 4 months"), @"P5Y4M");
        insta::assert_snapshot!(p("5 years 4 months 3 hours"), @"P5Y4MT3H");
        insta::assert_snapshot!(p("5 years, 4 months, 3 hours"), @"P5Y4MT3H");

        insta::assert_snapshot!(p("01:02:03"), @"PT1H2M3S");
        insta::assert_snapshot!(p("5 days 01:02:03"), @"P5DT1H2M3S");
        // This is Python's `str(timedelta)` format!
        insta::assert_snapshot!(p("5 days, 01:02:03"), @"P5DT1H2M3S");
        insta::assert_snapshot!(p("3yrs 5 days 01:02:03"), @"P3Y5DT1H2M3S");
        insta::assert_snapshot!(p("3yrs 5 days, 01:02:03"), @"P3Y5DT1H2M3S");
        insta::assert_snapshot!(
            p("3yrs 5 days, 01:02:03.123456789"),
            @"P3Y5DT1H2M3.123456789S",
        );
        insta::assert_snapshot!(p("999:999:999"), @"PT999H999M999S");
    }

    #[test]
    fn parse_span_fractional() {
        let p = |s: &str| SpanParser::new().parse_span(s).unwrap();

        insta::assert_snapshot!(p("1.5hrs"), @"PT1H30M");
        insta::assert_snapshot!(p("1.5mins"), @"PT1M30S");
        insta::assert_snapshot!(p("1.5secs"), @"PT1.5S");
        insta::assert_snapshot!(p("1.5msecs"), @"PT0.0015S");
        insta::assert_snapshot!(p("1.5µsecs"), @"PT0.0000015S");

        insta::assert_snapshot!(p("1d 1.5hrs"), @"P1DT1H30M");
        insta::assert_snapshot!(p("1h 1.5mins"), @"PT1H1M30S");
        insta::assert_snapshot!(p("1m 1.5secs"), @"PT1M1.5S");
        insta::assert_snapshot!(p("1s 1.5msecs"), @"PT1.0015S");
        insta::assert_snapshot!(p("1ms 1.5µsecs"), @"PT0.0010015S");

        insta::assert_snapshot!(p("1s2000ms"), @"PT3S");
    }

    #[test]
    fn parse_span_boundaries() {
        let p = |s: &str| SpanParser::new().parse_span(s).unwrap();

        insta::assert_snapshot!(p("19998 years"), @"P19998Y");
        insta::assert_snapshot!(p("19998 years ago"), @"-P19998Y");
        insta::assert_snapshot!(p("239976 months"), @"P239976M");
        insta::assert_snapshot!(p("239976 months ago"), @"-P239976M");
        insta::assert_snapshot!(p("1043497 weeks"), @"P1043497W");
        insta::assert_snapshot!(p("1043497 weeks ago"), @"-P1043497W");
        insta::assert_snapshot!(p("7304484 days"), @"P7304484D");
        insta::assert_snapshot!(p("7304484 days ago"), @"-P7304484D");
        insta::assert_snapshot!(p("175307616 hours"), @"PT175307616H");
        insta::assert_snapshot!(p("175307616 hours ago"), @"-PT175307616H");
        insta::assert_snapshot!(p("10518456960 minutes"), @"PT10518456960M");
        insta::assert_snapshot!(p("10518456960 minutes ago"), @"-PT10518456960M");
        insta::assert_snapshot!(p("631107417600 seconds"), @"PT631107417600S");
        insta::assert_snapshot!(p("631107417600 seconds ago"), @"-PT631107417600S");
        insta::assert_snapshot!(p("631107417600000 milliseconds"), @"PT631107417600S");
        insta::assert_snapshot!(p("631107417600000 milliseconds ago"), @"-PT631107417600S");
        insta::assert_snapshot!(p("631107417600000000 microseconds"), @"PT631107417600S");
        insta::assert_snapshot!(p("631107417600000000 microseconds ago"), @"-PT631107417600S");
        insta::assert_snapshot!(p("9223372036854775807 nanoseconds"), @"PT9223372036.854775807S");
        insta::assert_snapshot!(p("9223372036854775807 nanoseconds ago"), @"-PT9223372036.854775807S");

        insta::assert_snapshot!(p("175307617 hours"), @"PT175307616H60M");
        insta::assert_snapshot!(p("175307617 hours ago"), @"-PT175307616H60M");
        insta::assert_snapshot!(p("10518456961 minutes"), @"PT10518456960M60S");
        insta::assert_snapshot!(p("10518456961 minutes ago"), @"-PT10518456960M60S");
        insta::assert_snapshot!(p("631107417601 seconds"), @"PT631107417601S");
        insta::assert_snapshot!(p("631107417601 seconds ago"), @"-PT631107417601S");
        insta::assert_snapshot!(p("631107417600001 milliseconds"), @"PT631107417600.001S");
        insta::assert_snapshot!(p("631107417600001 milliseconds ago"), @"-PT631107417600.001S");
        insta::assert_snapshot!(p("631107417600000001 microseconds"), @"PT631107417600.000001S");
        insta::assert_snapshot!(p("631107417600000001 microseconds ago"), @"-PT631107417600.000001S");
        // We don't include nanoseconds here, because that will fail to
        // parse due to overflowing i64.
    }

    #[test]
    fn err_span_basic() {
        let p = |s: &str| SpanParser::new().parse_span(s).unwrap_err();

        insta::assert_snapshot!(
            p(""),
            @r#"failed to parse input in the "friendly" duration format: an empty string is not valid"#,
        );
        insta::assert_snapshot!(
            p(" "),
            @r#"failed to parse input in the "friendly" duration format: expected duration to start with a unit value (a decimal integer) after an optional sign, but no integer was found"#,
        );
        insta::assert_snapshot!(
            p("a"),
            @r#"failed to parse input in the "friendly" duration format: expected duration to start with a unit value (a decimal integer) after an optional sign, but no integer was found"#,
        );
        insta::assert_snapshot!(
            p("2 months 1 year"),
            @r#"failed to parse input in the "friendly" duration format: found value with unit year after unit month, but units must be written from largest to smallest (and they can't be repeated)"#,
        );
        insta::assert_snapshot!(
            p("1 year 1 mont"),
            @r#"failed to parse input in the "friendly" duration format: parsed value 'P1Y1M', but unparsed input "nt" remains (expected no unparsed input)"#,
        );
        insta::assert_snapshot!(
            p("2 months,"),
            @r#"failed to parse input in the "friendly" duration format: expected whitespace after comma, but found end of input"#,
        );
        insta::assert_snapshot!(
            p("2 months, "),
            @r#"failed to parse input in the "friendly" duration format: found comma at the end of duration, but a comma indicates at least one more unit follows"#,
        );
        insta::assert_snapshot!(
            p("2 months ,"),
            @r#"failed to parse input in the "friendly" duration format: parsed value 'P2M', but unparsed input "," remains (expected no unparsed input)"#,
        );
    }

    #[test]
    fn err_span_sign() {
        let p = |s: &str| SpanParser::new().parse_span(s).unwrap_err();

        insta::assert_snapshot!(
            p("1yago"),
            @r#"failed to parse input in the "friendly" duration format: parsed value 'P1Y', but unparsed input "ago" remains (expected no unparsed input)"#,
        );
        insta::assert_snapshot!(
            p("1 year 1 monthago"),
            @r#"failed to parse input in the "friendly" duration format: parsed value 'P1Y1M', but unparsed input "ago" remains (expected no unparsed input)"#,
        );
        insta::assert_snapshot!(
            p("+1 year 1 month ago"),
            @r#"failed to parse input in the "friendly" duration format: expected to find either a prefix sign (+/-) or a suffix sign (`ago`), but found both"#,
        );
        insta::assert_snapshot!(
            p("-1 year 1 month ago"),
            @r#"failed to parse input in the "friendly" duration format: expected to find either a prefix sign (+/-) or a suffix sign (`ago`), but found both"#,
        );
    }

    #[test]
    fn err_span_overflow_fraction() {
        let p = |s: &str| SpanParser::new().parse_span(s).unwrap();
        let pe = |s: &str| SpanParser::new().parse_span(s).unwrap_err();

        insta::assert_snapshot!(
            // One fewer micro, and this parses okay. The error occurs because
            // the maximum number of microseconds is subtracted off, and we're
            // left over with a value that overflows an i64.
            pe("640330789636854776 micros"),
            @r#"failed to parse input in the "friendly" duration format: failed to set value for microsecond unit on span: failed to set nanosecond value from fractional component"#,
        );
        // one fewer is okay
        insta::assert_snapshot!(
            p("640330789636854775 micros"),
            @"PT640330789636.854775S"
        );

        insta::assert_snapshot!(
            // This is like the test above, but actually exercises a slightly
            // different error path by using an explicit fraction. Here, if
            // we had x.807 micros, it would parse successfully.
            pe("640330789636854775.808 micros"),
            @r#"failed to parse input in the "friendly" duration format: failed to set nanosecond value from fractional component"#,
        );
        // one fewer is okay
        insta::assert_snapshot!(
            p("640330789636854775.807 micros"),
            @"PT640330789636.854775807S"
        );
    }

    #[test]
    fn err_span_overflow_units() {
        let p = |s: &str| SpanParser::new().parse_span(s).unwrap_err();

        insta::assert_snapshot!(
            p("19999 years"),
            @r#"failed to parse input in the "friendly" duration format: failed to set value for year unit on span: parameter 'years' is not in the required range of -19998..=19998"#,
        );
        insta::assert_snapshot!(
            p("19999 years ago"),
            @r#"failed to parse input in the "friendly" duration format: failed to set value for year unit on span: parameter 'years' is not in the required range of -19998..=19998"#,
        );

        insta::assert_snapshot!(
            p("239977 months"),
            @r#"failed to parse input in the "friendly" duration format: failed to set value for month unit on span: parameter 'months' is not in the required range of -239976..=239976"#,
        );
        insta::assert_snapshot!(
            p("239977 months ago"),
            @r#"failed to parse input in the "friendly" duration format: failed to set value for month unit on span: parameter 'months' is not in the required range of -239976..=239976"#,
        );

        insta::assert_snapshot!(
            p("1043498 weeks"),
            @r#"failed to parse input in the "friendly" duration format: failed to set value for week unit on span: parameter 'weeks' is not in the required range of -1043497..=1043497"#,
        );
        insta::assert_snapshot!(
            p("1043498 weeks ago"),
            @r#"failed to parse input in the "friendly" duration format: failed to set value for week unit on span: parameter 'weeks' is not in the required range of -1043497..=1043497"#,
        );

        insta::assert_snapshot!(
            p("7304485 days"),
            @r#"failed to parse input in the "friendly" duration format: failed to set value for day unit on span: parameter 'days' is not in the required range of -7304484..=7304484"#,
        );
        insta::assert_snapshot!(
            p("7304485 days ago"),
            @r#"failed to parse input in the "friendly" duration format: failed to set value for day unit on span: parameter 'days' is not in the required range of -7304484..=7304484"#,
        );

        insta::assert_snapshot!(
            p("9223372036854775808 nanoseconds"),
            @r#"failed to parse input in the "friendly" duration format: value for nanoseconds is too big (or small) to fit into a signed 64-bit integer"#,
        );
        insta::assert_snapshot!(
            p("9223372036854775808 nanoseconds ago"),
            @r#"failed to parse input in the "friendly" duration format: failed to set value for nanosecond unit on span: parameter 'nanoseconds' is not in the required range of -9223372036854775807..=9223372036854775807"#,
        );
    }

    #[test]
    fn err_span_fraction() {
        let p = |s: &str| SpanParser::new().parse_span(s).unwrap_err();

        insta::assert_snapshot!(
            p("1.5 years"),
            @r#"failed to parse input in the "friendly" duration format: fractional years are not supported"#,
        );
        insta::assert_snapshot!(
            p("1.5 nanos"),
            @r#"failed to parse input in the "friendly" duration format: fractional nanoseconds are not supported"#,
        );
    }

    #[test]
    fn err_span_hms() {
        let p = |s: &str| SpanParser::new().parse_span(s).unwrap_err();

        insta::assert_snapshot!(
            p("05:"),
            @r#"failed to parse input in the "friendly" duration format: when parsing the `HH:MM:SS` format, expected to parse minute following hour"#,
        );
        insta::assert_snapshot!(
            p("05:06"),
            @r#"failed to parse input in the "friendly" duration format: when parsing the `HH:MM:SS` format, expected to parse `:` following minute"#,
        );
        insta::assert_snapshot!(
            p("05:06:"),
            @r#"failed to parse input in the "friendly" duration format: when parsing the `HH:MM:SS` format, expected to parse second following minute"#,
        );
        insta::assert_snapshot!(
            p("2 hours, 05:06:07"),
            @r#"failed to parse input in the "friendly" duration format: found `HH:MM:SS` after unit hour, but `HH:MM:SS` can only appear after years, months, weeks or days"#,
        );
    }

    #[test]
    fn parse_signed_duration_basic() {
        let p = |s: &str| SpanParser::new().parse_duration(s).unwrap();

        insta::assert_snapshot!(p("1 hour, 2 minutes, 3 seconds"), @"PT1H2M3S");
        insta::assert_snapshot!(p("01:02:03"), @"PT1H2M3S");
        insta::assert_snapshot!(p("999:999:999"), @"PT1015H55M39S");
    }

    #[test]
    fn parse_signed_duration_negate() {
        let p = |s: &str| SpanParser::new().parse_duration(s).unwrap();
        let perr = |s: &str| SpanParser::new().parse_duration(s).unwrap_err();

        insta::assert_snapshot!(
            p("9223372036854775807s"),
            @"PT2562047788015215H30M7S",
        );
        insta::assert_snapshot!(
            perr("9223372036854775808s"),
            @r#"failed to parse input in the "friendly" duration format: value for seconds is too big (or small) to fit into a signed 64-bit integer"#,
        );
        insta::assert_snapshot!(
            p("-9223372036854775808s"),
            @"-PT2562047788015215H30M8S",
        );
    }

    #[test]
    fn parse_signed_duration_fractional() {
        let p = |s: &str| SpanParser::new().parse_duration(s).unwrap();

        insta::assert_snapshot!(p("1.5hrs"), @"PT1H30M");
        insta::assert_snapshot!(p("1.5mins"), @"PT1M30S");
        insta::assert_snapshot!(p("1.5secs"), @"PT1.5S");
        insta::assert_snapshot!(p("1.5msecs"), @"PT0.0015S");
        insta::assert_snapshot!(p("1.5µsecs"), @"PT0.0000015S");

        insta::assert_snapshot!(p("1h 1.5mins"), @"PT1H1M30S");
        insta::assert_snapshot!(p("1m 1.5secs"), @"PT1M1.5S");
        insta::assert_snapshot!(p("1s 1.5msecs"), @"PT1.0015S");
        insta::assert_snapshot!(p("1ms 1.5µsecs"), @"PT0.0010015S");

        insta::assert_snapshot!(p("1s2000ms"), @"PT3S");
    }

    #[test]
    fn parse_signed_duration_boundaries() {
        let p = |s: &str| SpanParser::new().parse_duration(s).unwrap();
        let pe = |s: &str| SpanParser::new().parse_duration(s).unwrap_err();

        insta::assert_snapshot!(p("175307616 hours"), @"PT175307616H");
        insta::assert_snapshot!(p("175307616 hours ago"), @"-PT175307616H");
        insta::assert_snapshot!(p("10518456960 minutes"), @"PT175307616H");
        insta::assert_snapshot!(p("10518456960 minutes ago"), @"-PT175307616H");
        insta::assert_snapshot!(p("631107417600 seconds"), @"PT175307616H");
        insta::assert_snapshot!(p("631107417600 seconds ago"), @"-PT175307616H");
        insta::assert_snapshot!(p("631107417600000 milliseconds"), @"PT175307616H");
        insta::assert_snapshot!(p("631107417600000 milliseconds ago"), @"-PT175307616H");
        insta::assert_snapshot!(p("631107417600000000 microseconds"), @"PT175307616H");
        insta::assert_snapshot!(p("631107417600000000 microseconds ago"), @"-PT175307616H");
        insta::assert_snapshot!(p("9223372036854775807 nanoseconds"), @"PT2562047H47M16.854775807S");
        insta::assert_snapshot!(p("9223372036854775807 nanoseconds ago"), @"-PT2562047H47M16.854775807S");

        insta::assert_snapshot!(p("175307617 hours"), @"PT175307617H");
        insta::assert_snapshot!(p("175307617 hours ago"), @"-PT175307617H");
        insta::assert_snapshot!(p("10518456961 minutes"), @"PT175307616H1M");
        insta::assert_snapshot!(p("10518456961 minutes ago"), @"-PT175307616H1M");
        insta::assert_snapshot!(p("631107417601 seconds"), @"PT175307616H1S");
        insta::assert_snapshot!(p("631107417601 seconds ago"), @"-PT175307616H1S");
        insta::assert_snapshot!(p("631107417600001 milliseconds"), @"PT175307616H0.001S");
        insta::assert_snapshot!(p("631107417600001 milliseconds ago"), @"-PT175307616H0.001S");
        insta::assert_snapshot!(p("631107417600000001 microseconds"), @"PT175307616H0.000001S");
        insta::assert_snapshot!(p("631107417600000001 microseconds ago"), @"-PT175307616H0.000001S");
        // We don't include nanoseconds here, because that will fail to
        // parse due to overflowing i64.

        // The above were copied from the corresponding `Span` test, which has
        // tighter limits on components. But a `SignedDuration` supports the
        // full range of `i64` seconds.
        insta::assert_snapshot!(p("2562047788015215hours"), @"PT2562047788015215H");
        insta::assert_snapshot!(p("-2562047788015215hours"), @"-PT2562047788015215H");
        insta::assert_snapshot!(
            pe("2562047788015216hrs"),
            @r#"failed to parse input in the "friendly" duration format: accumulated duration overflowed when adding value to unit hour"#,
        );

        insta::assert_snapshot!(p("153722867280912930minutes"), @"PT2562047788015215H30M");
        insta::assert_snapshot!(p("153722867280912930minutes ago"), @"-PT2562047788015215H30M");
        insta::assert_snapshot!(
            pe("153722867280912931mins"),
            @r#"failed to parse input in the "friendly" duration format: accumulated duration overflowed when adding value to unit minute"#,
        );

        insta::assert_snapshot!(p("9223372036854775807seconds"), @"PT2562047788015215H30M7S");
        insta::assert_snapshot!(p("-9223372036854775807seconds"), @"-PT2562047788015215H30M7S");
        insta::assert_snapshot!(
            pe("9223372036854775808s"),
            @r#"failed to parse input in the "friendly" duration format: value for seconds is too big (or small) to fit into a signed 64-bit integer"#,
        );
        insta::assert_snapshot!(
            p("-9223372036854775808s"),
            @"-PT2562047788015215H30M8S",
        );
    }

    #[test]
    fn err_signed_duration_basic() {
        let p = |s: &str| SpanParser::new().parse_duration(s).unwrap_err();

        insta::assert_snapshot!(
            p(""),
            @r#"failed to parse input in the "friendly" duration format: an empty string is not valid"#,
        );
        insta::assert_snapshot!(
            p(" "),
            @r#"failed to parse input in the "friendly" duration format: expected duration to start with a unit value (a decimal integer) after an optional sign, but no integer was found"#,
        );
        insta::assert_snapshot!(
            p("5"),
            @r#"failed to parse input in the "friendly" duration format: expected to find unit designator suffix (e.g., `years` or `secs`) after parsing integer"#,
        );
        insta::assert_snapshot!(
            p("a"),
            @r#"failed to parse input in the "friendly" duration format: expected duration to start with a unit value (a decimal integer) after an optional sign, but no integer was found"#,
        );
        insta::assert_snapshot!(
            p("2 minutes 1 hour"),
            @r#"failed to parse input in the "friendly" duration format: found value with unit hour after unit minute, but units must be written from largest to smallest (and they can't be repeated)"#,
        );
        insta::assert_snapshot!(
            p("1 hour 1 minut"),
            @r#"failed to parse input in the "friendly" duration format: parsed value 'PT1H1M', but unparsed input "ut" remains (expected no unparsed input)"#,
        );
        insta::assert_snapshot!(
            p("2 minutes,"),
            @r#"failed to parse input in the "friendly" duration format: expected whitespace after comma, but found end of input"#,
        );
        insta::assert_snapshot!(
            p("2 minutes, "),
            @r#"failed to parse input in the "friendly" duration format: found comma at the end of duration, but a comma indicates at least one more unit follows"#,
        );
        insta::assert_snapshot!(
            p("2 minutes ,"),
            @r#"failed to parse input in the "friendly" duration format: parsed value 'PT2M', but unparsed input "," remains (expected no unparsed input)"#,
        );
    }

    #[test]
    fn err_signed_duration_sign() {
        let p = |s: &str| SpanParser::new().parse_duration(s).unwrap_err();

        insta::assert_snapshot!(
            p("1hago"),
            @r#"failed to parse input in the "friendly" duration format: parsed value 'PT1H', but unparsed input "ago" remains (expected no unparsed input)"#,
        );
        insta::assert_snapshot!(
            p("1 hour 1 minuteago"),
            @r#"failed to parse input in the "friendly" duration format: parsed value 'PT1H1M', but unparsed input "ago" remains (expected no unparsed input)"#,
        );
        insta::assert_snapshot!(
            p("+1 hour 1 minute ago"),
            @r#"failed to parse input in the "friendly" duration format: expected to find either a prefix sign (+/-) or a suffix sign (`ago`), but found both"#,
        );
        insta::assert_snapshot!(
            p("-1 hour 1 minute ago"),
            @r#"failed to parse input in the "friendly" duration format: expected to find either a prefix sign (+/-) or a suffix sign (`ago`), but found both"#,
        );
    }

    #[test]
    fn err_signed_duration_overflow_fraction() {
        let p = |s: &str| SpanParser::new().parse_duration(s).unwrap();
        let pe = |s: &str| SpanParser::new().parse_duration(s).unwrap_err();

        insta::assert_snapshot!(
            // Unlike `Span`, this just overflows because it can't be parsed
            // as a 64-bit integer.
            pe("9223372036854775808 micros"),
            @r#"failed to parse input in the "friendly" duration format: value for microseconds is too big (or small) to fit into a signed 64-bit integer"#,
        );
        // one fewer is okay
        insta::assert_snapshot!(
            p("9223372036854775807 micros"),
            @"PT2562047788H54.775807S"
        );
    }

    #[test]
    fn err_signed_duration_fraction() {
        let p = |s: &str| SpanParser::new().parse_duration(s).unwrap_err();

        insta::assert_snapshot!(
            p("1.5 nanos"),
            @r#"failed to parse input in the "friendly" duration format: fractional nanoseconds are not supported"#,
        );
    }

    #[test]
    fn err_signed_duration_hms() {
        let p = |s: &str| SpanParser::new().parse_duration(s).unwrap_err();

        insta::assert_snapshot!(
            p("05:"),
            @r#"failed to parse input in the "friendly" duration format: when parsing the `HH:MM:SS` format, expected to parse minute following hour"#,
        );
        insta::assert_snapshot!(
            p("05:06"),
            @r#"failed to parse input in the "friendly" duration format: when parsing the `HH:MM:SS` format, expected to parse `:` following minute"#,
        );
        insta::assert_snapshot!(
            p("05:06:"),
            @r#"failed to parse input in the "friendly" duration format: when parsing the `HH:MM:SS` format, expected to parse second following minute"#,
        );
        insta::assert_snapshot!(
            p("2 hours, 05:06:07"),
            @r#"failed to parse input in the "friendly" duration format: found `HH:MM:SS` after unit hour, but `HH:MM:SS` can only appear after years, months, weeks or days"#,
        );
    }

    #[test]
    fn parse_unsigned_duration_basic() {
        let p = |s: &str| {
            let dur = SpanParser::new().parse_unsigned_duration(s).unwrap();
            crate::fmt::temporal::SpanPrinter::new()
                .unsigned_duration_to_string(&dur)
        };

        insta::assert_snapshot!(
            p("1 hour, 2 minutes, 3 seconds"),
            @"PT1H2M3S",
        );
        insta::assert_snapshot!(p("01:02:03"), @"PT1H2M3S");
        insta::assert_snapshot!(p("999:999:999"), @"PT1015H55M39S");
        insta::assert_snapshot!(
            p("+1hr"),
            @"PT1H",
        );
    }

    #[test]
    fn parse_unsigned_duration_negate() {
        let p = |s: &str| {
            let dur = SpanParser::new().parse_unsigned_duration(s).unwrap();
            crate::fmt::temporal::SpanPrinter::new()
                .unsigned_duration_to_string(&dur)
        };
        let perr = |s: &str| {
            SpanParser::new().parse_unsigned_duration(s).unwrap_err()
        };

        insta::assert_snapshot!(
            p("18446744073709551615s"),
            @"PT5124095576030431H15S",
        );
        insta::assert_snapshot!(
            perr("18446744073709551616s"),
            @r#"failed to parse input in the "friendly" duration format: number too big to parse into 64-bit integer"#,
        );
        insta::assert_snapshot!(
            perr("-1s"),
            @r#"failed to parse input in the "friendly" duration format: cannot parse negative duration into unsigned `std::time::Duration`"#,
        );
    }

    #[test]
    fn parse_unsigned_duration_fractional() {
        let p = |s: &str| {
            let dur = SpanParser::new().parse_unsigned_duration(s).unwrap();
            crate::fmt::temporal::SpanPrinter::new()
                .unsigned_duration_to_string(&dur)
        };

        insta::assert_snapshot!(p("1.5hrs"), @"PT1H30M");
        insta::assert_snapshot!(p("1.5mins"), @"PT1M30S");
        insta::assert_snapshot!(p("1.5secs"), @"PT1.5S");
        insta::assert_snapshot!(p("1.5msecs"), @"PT0.0015S");
        insta::assert_snapshot!(p("1.5µsecs"), @"PT0.0000015S");

        insta::assert_snapshot!(p("1h 1.5mins"), @"PT1H1M30S");
        insta::assert_snapshot!(p("1m 1.5secs"), @"PT1M1.5S");
        insta::assert_snapshot!(p("1s 1.5msecs"), @"PT1.0015S");
        insta::assert_snapshot!(p("1ms 1.5µsecs"), @"PT0.0010015S");

        insta::assert_snapshot!(p("1s2000ms"), @"PT3S");
    }

    #[test]
    fn parse_unsigned_duration_boundaries() {
        let p = |s: &str| {
            let dur = SpanParser::new().parse_unsigned_duration(s).unwrap();
            crate::fmt::temporal::SpanPrinter::new()
                .unsigned_duration_to_string(&dur)
        };
        let pe = |s: &str| SpanParser::new().parse_duration(s).unwrap_err();

        insta::assert_snapshot!(p("175307616 hours"), @"PT175307616H");
        insta::assert_snapshot!(p("10518456960 minutes"), @"PT175307616H");
        insta::assert_snapshot!(p("631107417600 seconds"), @"PT175307616H");
        insta::assert_snapshot!(p("631107417600000 milliseconds"), @"PT175307616H");
        insta::assert_snapshot!(p("631107417600000000 microseconds"), @"PT175307616H");
        insta::assert_snapshot!(p("9223372036854775807 nanoseconds"), @"PT2562047H47M16.854775807S");

        insta::assert_snapshot!(p("175307617 hours"), @"PT175307617H");
        insta::assert_snapshot!(p("10518456961 minutes"), @"PT175307616H1M");
        insta::assert_snapshot!(p("631107417601 seconds"), @"PT175307616H1S");
        insta::assert_snapshot!(p("631107417600001 milliseconds"), @"PT175307616H0.001S");
        insta::assert_snapshot!(p("631107417600000001 microseconds"), @"PT175307616H0.000001S");

        // The above were copied from the corresponding `Span` test, which has
        // tighter limits on components. But a `std::time::Duration` supports
        // the full range of `u64` seconds.
        insta::assert_snapshot!(p("5124095576030431hours"), @"PT5124095576030431H");
        insta::assert_snapshot!(
            pe("5124095576030432hrs"),
            @r#"failed to parse input in the "friendly" duration format: accumulated duration overflowed when adding value to unit hour"#,
        );

        insta::assert_snapshot!(p("307445734561825860minutes"), @"PT5124095576030431H");
        insta::assert_snapshot!(
            pe("307445734561825861mins"),
            @r#"failed to parse input in the "friendly" duration format: accumulated duration overflowed when adding value to unit minute"#,
        );

        insta::assert_snapshot!(p("18446744073709551615seconds"), @"PT5124095576030431H15S");
        insta::assert_snapshot!(
            pe("18446744073709551616s"),
            @r#"failed to parse input in the "friendly" duration format: number too big to parse into 64-bit integer"#,
        );
    }

    #[test]
    fn err_unsigned_duration_basic() {
        let p = |s: &str| {
            SpanParser::new().parse_unsigned_duration(s).unwrap_err()
        };

        insta::assert_snapshot!(
            p(""),
            @r#"failed to parse input in the "friendly" duration format: an empty string is not valid"#,
        );
        insta::assert_snapshot!(
            p(" "),
            @r#"failed to parse input in the "friendly" duration format: expected duration to start with a unit value (a decimal integer) after an optional sign, but no integer was found"#,
        );
        insta::assert_snapshot!(
            p("5"),
            @r#"failed to parse input in the "friendly" duration format: expected to find unit designator suffix (e.g., `years` or `secs`) after parsing integer"#,
        );
        insta::assert_snapshot!(
            p("a"),
            @r#"failed to parse input in the "friendly" duration format: expected duration to start with a unit value (a decimal integer) after an optional sign, but no integer was found"#,
        );
        insta::assert_snapshot!(
            p("2 minutes 1 hour"),
            @r#"failed to parse input in the "friendly" duration format: found value with unit hour after unit minute, but units must be written from largest to smallest (and they can't be repeated)"#,
        );
        insta::assert_snapshot!(
            p("1 hour 1 minut"),
            @r#"failed to parse input in the "friendly" duration format: parsed value '3660s', but unparsed input "ut" remains (expected no unparsed input)"#,
        );
        insta::assert_snapshot!(
            p("2 minutes,"),
            @r#"failed to parse input in the "friendly" duration format: expected whitespace after comma, but found end of input"#,
        );
        insta::assert_snapshot!(
            p("2 minutes, "),
            @r#"failed to parse input in the "friendly" duration format: found comma at the end of duration, but a comma indicates at least one more unit follows"#,
        );
        insta::assert_snapshot!(
            p("2 minutes ,"),
            @r#"failed to parse input in the "friendly" duration format: parsed value '120s', but unparsed input "," remains (expected no unparsed input)"#,
        );
    }

    #[test]
    fn err_unsigned_duration_sign() {
        let p = |s: &str| {
            SpanParser::new().parse_unsigned_duration(s).unwrap_err()
        };

        insta::assert_snapshot!(
            p("1hago"),
            @r#"failed to parse input in the "friendly" duration format: parsed value '3600s', but unparsed input "ago" remains (expected no unparsed input)"#,
        );
        insta::assert_snapshot!(
            p("1 hour 1 minuteago"),
            @r#"failed to parse input in the "friendly" duration format: parsed value '3660s', but unparsed input "ago" remains (expected no unparsed input)"#,
        );
        insta::assert_snapshot!(
            p("+1 hour 1 minute ago"),
            @r#"failed to parse input in the "friendly" duration format: expected to find either a prefix sign (+/-) or a suffix sign (`ago`), but found both"#,
        );
        insta::assert_snapshot!(
            p("-1 hour 1 minute ago"),
            @r#"failed to parse input in the "friendly" duration format: expected to find either a prefix sign (+/-) or a suffix sign (`ago`), but found both"#,
        );
    }

    #[test]
    fn err_unsigned_duration_overflow_fraction() {
        let p = |s: &str| {
            let dur = SpanParser::new().parse_unsigned_duration(s).unwrap();
            crate::fmt::temporal::SpanPrinter::new()
                .unsigned_duration_to_string(&dur)
        };
        let pe = |s: &str| {
            SpanParser::new().parse_unsigned_duration(s).unwrap_err()
        };

        insta::assert_snapshot!(
            // Unlike `Span`, this just overflows because it can't be parsed
            // as a 64-bit integer.
            pe("18446744073709551616 micros"),
            @r#"failed to parse input in the "friendly" duration format: number too big to parse into 64-bit integer"#,
        );
        // one fewer is okay
        insta::assert_snapshot!(
            p("18446744073709551615 micros"),
            @"PT5124095576H1M49.551615S"
        );
    }

    #[test]
    fn err_unsigned_duration_fraction() {
        let p = |s: &str| {
            SpanParser::new().parse_unsigned_duration(s).unwrap_err()
        };

        insta::assert_snapshot!(
            p("1.5 nanos"),
            @r#"failed to parse input in the "friendly" duration format: fractional nanoseconds are not supported"#,
        );
    }

    #[test]
    fn err_unsigned_duration_hms() {
        let p = |s: &str| {
            SpanParser::new().parse_unsigned_duration(s).unwrap_err()
        };

        insta::assert_snapshot!(
            p("05:"),
            @r#"failed to parse input in the "friendly" duration format: when parsing the `HH:MM:SS` format, expected to parse minute following hour"#,
        );
        insta::assert_snapshot!(
            p("05:06"),
            @r#"failed to parse input in the "friendly" duration format: when parsing the `HH:MM:SS` format, expected to parse `:` following minute"#,
        );
        insta::assert_snapshot!(
            p("05:06:"),
            @r#"failed to parse input in the "friendly" duration format: when parsing the `HH:MM:SS` format, expected to parse second following minute"#,
        );
        insta::assert_snapshot!(
            p("2 hours, 05:06:07"),
            @r#"failed to parse input in the "friendly" duration format: found `HH:MM:SS` after unit hour, but `HH:MM:SS` can only appear after years, months, weeks or days"#,
        );
    }
}
