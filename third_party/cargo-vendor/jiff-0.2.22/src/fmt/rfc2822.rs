/*!
Support for printing and parsing instants using the [RFC 2822] datetime format.

RFC 2822 is most commonly found when dealing with email messages.

Since RFC 2822 only supports specifying a complete instant in time, the parser
and printer in this module only use [`Zoned`] and [`Timestamp`]. If you need
inexact time, you can get it from [`Zoned`] via [`Zoned::datetime`].

[RFC 2822]: https://datatracker.ietf.org/doc/html/rfc2822

# Incomplete support

The RFC 2822 support in this crate is technically incomplete. Specifically,
it does not support parsing comments within folding whitespace. It will parse
comments after the datetime itself (including nested comments). See [Issue
#39][issue39] for an example. If you find a real world use case for parsing
comments within whitespace at any point in the datetime string, please file
an issue. That is, the main reason it isn't currently supported is because
it didn't seem worth the implementation complexity to account for it. But if
there are real world use cases that need it, then that would be sufficient
justification for adding it.

RFC 2822 support should otherwise be complete, including support for parsing
obsolete offsets.

[issue39]: https://github.com/BurntSushi/jiff/issues/39

# Warning

The RFC 2822 format only supports writing a precise instant in time
expressed via a time zone offset. It does *not* support serializing
the time zone itself. This means that if you format a zoned datetime
in a time zone like `America/New_York` and then deserialize it, the
zoned datetime you get back will be a "fixed offset" zoned datetime.
This in turn means it will not perform daylight saving time safe
arithmetic.

Basically, you should use the RFC 2822 format if it's required (for
example, when dealing with email). But you should not choose it as a
general interchange format for new applications.
*/

use crate::{
    civil::{Date, DateTime, Time, Weekday},
    error::{fmt::rfc2822::Error as E, ErrorContext},
    fmt::{buffer::BorrowedBuffer, Parsed, Write},
    tz::{Offset, TimeZone},
    util::{b, parse},
    Error, Timestamp, Zoned,
};

/// The default date time parser that we use throughout Jiff.
pub(crate) static DEFAULT_DATETIME_PARSER: DateTimeParser =
    DateTimeParser::new();

/// The default date time printer that we use throughout Jiff.
pub(crate) static DEFAULT_DATETIME_PRINTER: DateTimePrinter =
    DateTimePrinter::new();

/// The maximum number bytes that can be written by the RFC 2822 printer.
///
/// We reserve a heap or stack buffer up front before printing, and we want to
/// ensure we have enough space to write the longest possible RFC 2822 string.
const PRINTER_MAX_BYTES_RFC2822: usize = 31;

/// Same idea, but for RFC 9110.
///
/// The difference comes from always using `GMT` instead of, e.g., `-0400`.
const PRINTER_MAX_BYTES_RFC9110: usize = 29;

/// Convert a [`Zoned`] to an [RFC 2822] datetime string.
///
/// This is a convenience function for using [`DateTimePrinter`]. In
/// particular, this always creates and allocates a new `String`. For writing
/// to an existing string, or converting a [`Timestamp`] to an RFC 2822
/// datetime string, you'll need to use `DateTimePrinter`.
///
/// [RFC 2822]: https://datatracker.ietf.org/doc/html/rfc2822
///
/// # Warning
///
/// The RFC 2822 format only supports writing a precise instant in time
/// expressed via a time zone offset. It does *not* support serializing
/// the time zone itself. This means that if you format a zoned datetime
/// in a time zone like `America/New_York` and then deserialize it, the
/// zoned datetime you get back will be a "fixed offset" zoned datetime.
/// This in turn means it will not perform daylight saving time safe
/// arithmetic.
///
/// Basically, you should use the RFC 2822 format if it's required (for
/// example, when dealing with email). But you should not choose it as a
/// general interchange format for new applications.
///
/// # Errors
///
/// This returns an error if the year corresponding to this timestamp cannot be
/// represented in the RFC 2822 format. For example, a negative year.
///
/// # Example
///
/// This example shows how to convert a zoned datetime to the RFC 2822 format:
///
/// ```
/// use jiff::{civil::date, fmt::rfc2822};
///
/// let zdt = date(2024, 6, 15).at(7, 0, 0, 0).in_tz("Australia/Tasmania")?;
/// assert_eq!(rfc2822::to_string(&zdt)?, "Sat, 15 Jun 2024 07:00:00 +1000");
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[cfg(feature = "alloc")]
#[inline]
pub fn to_string(zdt: &Zoned) -> Result<alloc::string::String, Error> {
    let mut buf = alloc::string::String::new();
    DEFAULT_DATETIME_PRINTER.print_zoned(zdt, &mut buf)?;
    Ok(buf)
}

/// Parse an [RFC 2822] datetime string into a [`Zoned`].
///
/// This is a convenience function for using [`DateTimeParser`]. In particular,
/// this takes a `&str` while the `DateTimeParser` accepts a `&[u8]`.
/// Moreover, if any configuration options are added to RFC 2822 parsing (none
/// currently exist at time of writing), then it will be necessary to use a
/// `DateTimeParser` to toggle them. Additionally, a `DateTimeParser` is needed
/// for parsing into a [`Timestamp`].
///
/// [RFC 2822]: https://datatracker.ietf.org/doc/html/rfc2822
///
/// # Warning
///
/// The RFC 2822 format only supports writing a precise instant in time
/// expressed via a time zone offset. It does *not* support serializing
/// the time zone itself. This means that if you format a zoned datetime
/// in a time zone like `America/New_York` and then deserialize it, the
/// zoned datetime you get back will be a "fixed offset" zoned datetime.
/// This in turn means it will not perform daylight saving time safe
/// arithmetic.
///
/// Basically, you should use the RFC 2822 format if it's required (for
/// example, when dealing with email). But you should not choose it as a
/// general interchange format for new applications.
///
/// # Errors
///
/// This returns an error if the datetime string given is invalid or if it
/// is valid but doesn't fit in the datetime range supported by Jiff. For
/// example, RFC 2822 supports offsets up to 99 hours and 59 minutes,
/// but Jiff's maximum offset is 25 hours, 59 minutes and 59 seconds.
///
/// # Example
///
/// This example shows how serializing a zoned datetime to RFC 2822 format
/// and then deserializing will drop information:
///
/// ```
/// use jiff::{civil::date, fmt::rfc2822};
///
/// let zdt = date(2024, 7, 13)
///     .at(15, 9, 59, 789_000_000)
///     .in_tz("America/New_York")?;
/// // The default format (i.e., Temporal) guarantees lossless
/// // serialization.
/// assert_eq!(zdt.to_string(), "2024-07-13T15:09:59.789-04:00[America/New_York]");
///
/// let rfc2822 = rfc2822::to_string(&zdt)?;
/// // Notice that the time zone name and fractional seconds have been dropped!
/// assert_eq!(rfc2822, "Sat, 13 Jul 2024 15:09:59 -0400");
/// // And of course, if we parse it back, all that info is still lost.
/// // Which means this `zdt` cannot do DST safe arithmetic!
/// let zdt = rfc2822::parse(&rfc2822)?;
/// assert_eq!(zdt.to_string(), "2024-07-13T15:09:59-04:00[-04:00]");
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[inline]
pub fn parse(string: &str) -> Result<Zoned, Error> {
    DEFAULT_DATETIME_PARSER.parse_zoned(string)
}

/// A parser for [RFC 2822] datetimes.
///
/// [RFC 2822]: https://datatracker.ietf.org/doc/html/rfc2822
///
/// # Warning
///
/// The RFC 2822 format only supports writing a precise instant in time
/// expressed via a time zone offset. It does *not* support serializing
/// the time zone itself. This means that if you format a zoned datetime
/// in a time zone like `America/New_York` and then deserialize it, the
/// zoned datetime you get back will be a "fixed offset" zoned datetime.
/// This in turn means it will not perform daylight saving time safe
/// arithmetic.
///
/// Basically, you should use the RFC 2822 format if it's required (for
/// example, when dealing with email). But you should not choose it as a
/// general interchange format for new applications.
///
/// # Example
///
/// This example shows how serializing a zoned datetime to RFC 2822 format
/// and then deserializing will drop information:
///
/// ```
/// use jiff::{civil::date, fmt::rfc2822};
///
/// let zdt = date(2024, 7, 13)
///     .at(15, 9, 59, 789_000_000)
///     .in_tz("America/New_York")?;
/// // The default format (i.e., Temporal) guarantees lossless
/// // serialization.
/// assert_eq!(zdt.to_string(), "2024-07-13T15:09:59.789-04:00[America/New_York]");
///
/// let rfc2822 = rfc2822::to_string(&zdt)?;
/// // Notice that the time zone name and fractional seconds have been dropped!
/// assert_eq!(rfc2822, "Sat, 13 Jul 2024 15:09:59 -0400");
/// // And of course, if we parse it back, all that info is still lost.
/// // Which means this `zdt` cannot do DST safe arithmetic!
/// let zdt = rfc2822::parse(&rfc2822)?;
/// assert_eq!(zdt.to_string(), "2024-07-13T15:09:59-04:00[-04:00]");
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct DateTimeParser {
    relaxed_weekday: bool,
}

impl DateTimeParser {
    /// Create a new RFC 2822 datetime parser with the default configuration.
    #[inline]
    pub const fn new() -> DateTimeParser {
        DateTimeParser { relaxed_weekday: false }
    }

    /// When enabled, parsing will permit the weekday to be inconsistent with
    /// the date. When enabled, the weekday is still parsed and can result in
    /// an error if it isn't _a_ valid weekday. Only the error checking for
    /// whether it is _the_ correct weekday for the parsed date is disabled.
    ///
    /// This is sometimes useful for interaction with systems that don't do
    /// strict error checking.
    ///
    /// This is disabled by default. And note that RFC 2822 compliance requires
    /// that the weekday is consistent with the date.
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::{civil::date, fmt::rfc2822};
    ///
    /// let string = "Sun, 13 Jul 2024 15:09:59 -0400";
    /// // The above normally results in an error, since 2024-07-13 is a
    /// // Saturday:
    /// assert!(rfc2822::parse(string).is_err());
    /// // But we can relax the error checking:
    /// static P: rfc2822::DateTimeParser = rfc2822::DateTimeParser::new()
    ///     .relaxed_weekday(true);
    /// assert_eq!(
    ///     P.parse_zoned(string)?,
    ///     date(2024, 7, 13).at(15, 9, 59, 0).in_tz("America/New_York")?,
    /// );
    /// // But note that something that isn't recognized as a valid weekday
    /// // will still result in an error:
    /// assert!(P.parse_zoned("Wat, 13 Jul 2024 15:09:59 -0400").is_err());
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub const fn relaxed_weekday(self, yes: bool) -> DateTimeParser {
        DateTimeParser { relaxed_weekday: yes, ..self }
    }

    /// Parse a datetime string into a [`Zoned`] value.
    ///
    /// Note that RFC 2822 does not support time zone annotations. The zoned
    /// datetime returned will therefore always have a fixed offset time zone.
    ///
    /// # Warning
    ///
    /// The RFC 2822 format only supports writing a precise instant in time
    /// expressed via a time zone offset. It does *not* support serializing
    /// the time zone itself. This means that if you format a zoned datetime
    /// in a time zone like `America/New_York` and then deserialize it, the
    /// zoned datetime you get back will be a "fixed offset" zoned datetime.
    /// This in turn means it will not perform daylight saving time safe
    /// arithmetic.
    ///
    /// Basically, you should use the RFC 2822 format if it's required (for
    /// example, when dealing with email). But you should not choose it as a
    /// general interchange format for new applications.
    ///
    /// # Errors
    ///
    /// This returns an error if the datetime string given is invalid or if it
    /// is valid but doesn't fit in the datetime range supported by Jiff. For
    /// example, RFC 2822 supports offsets up to 99 hours and 59 minutes,
    /// but Jiff's maximum offset is 25 hours, 59 minutes and 59 seconds.
    ///
    /// # Example
    ///
    /// This shows a basic example of parsing a `Timestamp` from an RFC 2822
    /// datetime string.
    ///
    /// ```
    /// use jiff::fmt::rfc2822::DateTimeParser;
    ///
    /// static PARSER: DateTimeParser = DateTimeParser::new();
    ///
    /// let zdt = PARSER.parse_zoned("Thu, 29 Feb 2024 05:34 -0500")?;
    /// assert_eq!(zdt.to_string(), "2024-02-29T05:34:00-05:00[-05:00]");
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn parse_zoned<I: AsRef<[u8]>>(
        &self,
        input: I,
    ) -> Result<Zoned, Error> {
        let input = input.as_ref();
        let zdt = self
            .parse_zoned_internal(input)
            .context(E::FailedZoned)?
            .into_full()?;
        Ok(zdt)
    }

    /// Parse an RFC 2822 datetime string into a [`Timestamp`].
    ///
    /// # Errors
    ///
    /// This returns an error if the datetime string given is invalid or if it
    /// is valid but doesn't fit in the datetime range supported by Jiff. For
    /// example, RFC 2822 supports offsets up to 99 hours and 59 minutes,
    /// but Jiff's maximum offset is 25 hours, 59 minutes and 59 seconds.
    ///
    /// # Example
    ///
    /// This shows a basic example of parsing a `Timestamp` from an RFC 2822
    /// datetime string.
    ///
    /// ```
    /// use jiff::fmt::rfc2822::DateTimeParser;
    ///
    /// static PARSER: DateTimeParser = DateTimeParser::new();
    ///
    /// let timestamp = PARSER.parse_timestamp("Thu, 29 Feb 2024 05:34 -0500")?;
    /// assert_eq!(timestamp.to_string(), "2024-02-29T10:34:00Z");
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn parse_timestamp<I: AsRef<[u8]>>(
        &self,
        input: I,
    ) -> Result<Timestamp, Error> {
        let input = input.as_ref();
        let ts = self
            .parse_timestamp_internal(input)
            .context(E::FailedTimestamp)?
            .into_full()?;
        Ok(ts)
    }

    /// Parses an RFC 2822 datetime as a zoned datetime.
    ///
    /// Note that this doesn't check that the input has been completely
    /// consumed.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_zoned_internal<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, Zoned>, Error> {
        let Parsed { value: (dt, offset), input } =
            self.parse_datetime_offset(input)?;
        let ts = offset.to_timestamp(dt)?;
        let zdt = ts.to_zoned(TimeZone::fixed(offset));
        Ok(Parsed { value: zdt, input })
    }

    /// Parses an RFC 2822 datetime as a timestamp.
    ///
    /// Note that this doesn't check that the input has been completely
    /// consumed.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_timestamp_internal<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, Timestamp>, Error> {
        let Parsed { value: (dt, offset), input } =
            self.parse_datetime_offset(input)?;
        let ts = offset.to_timestamp(dt)?;
        Ok(Parsed { value: ts, input })
    }

    /// Parse the entirety of the given input into RFC 2822 components: a civil
    /// datetime and its offset.
    ///
    /// This also consumes any trailing (superfluous) whitespace.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_datetime_offset<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, (DateTime, Offset)>, Error> {
        let input = input.as_ref();
        let Parsed { value: dt, input } = self.parse_datetime(input)?;
        let Parsed { value: offset, input } = self.parse_offset(input)?;
        let Parsed { input, .. } = self.skip_whitespace(input);
        let input = if input.is_empty() {
            input
        } else {
            self.skip_comment(input)?.input
        };
        Ok(Parsed { value: (dt, offset), input })
    }

    /// Parses a civil datetime from an RFC 2822 string. The input may have
    /// leading whitespace.
    ///
    /// This also parses and trailing whitespace, including requiring at least
    /// one whitespace character.
    ///
    /// This basically parses everything except for the zone.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_datetime<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, DateTime>, Error> {
        if input.is_empty() {
            return Err(Error::from(E::Empty));
        }
        let Parsed { input, .. } = self.skip_whitespace(input);
        if input.is_empty() {
            return Err(Error::from(E::EmptyAfterWhitespace));
        }
        let Parsed { value: wd, input } = self.parse_weekday(input)?;
        let Parsed { value: day, input } = self.parse_day(input)?;
        let Parsed { value: month, input } = self.parse_month(input)?;
        let Parsed { value: year, input } = self.parse_year(input)?;

        let Parsed { value: hour, input } = self.parse_hour(input)?;
        let Parsed { input, .. } = self.skip_whitespace(input);
        let Parsed { input, .. } = self.parse_time_separator(input)?;
        let Parsed { input, .. } = self.skip_whitespace(input);
        let Parsed { value: minute, input } = self.parse_minute(input)?;

        let Parsed { value: whitespace_after_minute, input } =
            self.skip_whitespace(input);
        let (second, input) = if !input.starts_with(b":") {
            if !whitespace_after_minute {
                return Err(Error::from(E::WhitespaceAfterTime));
            }
            (0, input)
        } else {
            let Parsed { input, .. } = self.parse_time_separator(input)?;
            let Parsed { input, .. } = self.skip_whitespace(input);
            let Parsed { value: second, input } = self.parse_second(input)?;
            let Parsed { input, .. } = self.parse_whitespace(input)?;
            (second, input)
        };

        let date = Date::new(year, month, day).context(E::InvalidDate)?;
        // OK because hour, minute and second have been verified as being
        // in bounds. And all combinations of such in-bound values are also
        // valid `Time` values.
        let time = Time::new(hour, minute, second, 0).unwrap();
        let dt = DateTime::from_parts(date, time);
        if let Some(wd) = wd {
            if !self.relaxed_weekday && wd != dt.weekday() {
                return Err(Error::from(E::InconsistentWeekday {
                    parsed: wd,
                    from_date: dt.weekday(),
                }));
            }
        }
        Ok(Parsed { value: dt, input })
    }

    /// Parses an optional weekday at the beginning of an RFC 2822 datetime.
    ///
    /// This expects that any optional whitespace preceding the start of an
    /// optional day has been stripped and that the input has at least one
    /// byte.
    ///
    /// When the first byte of the given input is a digit (or is empty), then
    /// this returns `None`, as it implies a day is not present. But if it
    /// isn't a digit, then we assume that it must be a weekday and return an
    /// error based on that assumption if we couldn't recognize a weekday.
    ///
    /// If a weekday is parsed, then this also skips any trailing whitespace
    /// (and requires at least one whitespace character).
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_weekday<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, Option<Weekday>>, Error> {
        // An empty input is invalid, but we let that case be
        // handled by the caller. Otherwise, we know there MUST
        // be a present day if the first character isn't an ASCII
        // digit.
        if matches!(input[0], b'0'..=b'9') {
            return Ok(Parsed { value: None, input });
        }
        if let Ok(len) = u8::try_from(input.len()) {
            if len < 4 {
                return Err(Error::from(E::TooShortWeekday {
                    got_non_digit: input[0],
                    len,
                }));
            }
        }
        let b1 = input[0];
        let b2 = input[1];
        let b3 = input[2];
        let wd = match &[
            b1.to_ascii_lowercase(),
            b2.to_ascii_lowercase(),
            b3.to_ascii_lowercase(),
        ] {
            b"sun" => Weekday::Sunday,
            b"mon" => Weekday::Monday,
            b"tue" => Weekday::Tuesday,
            b"wed" => Weekday::Wednesday,
            b"thu" => Weekday::Thursday,
            b"fri" => Weekday::Friday,
            b"sat" => Weekday::Saturday,
            _ => {
                return Err(Error::from(E::InvalidWeekday {
                    got_non_digit: input[0],
                }));
            }
        };
        let Parsed { input, .. } = self.skip_whitespace(&input[3..]);
        let Some(should_be_comma) = input.get(0).copied() else {
            return Err(Error::from(E::EndOfInputComma));
        };
        if should_be_comma != b',' {
            return Err(Error::from(E::UnexpectedByteComma {
                byte: should_be_comma,
            }));
        }
        let Parsed { input, .. } = self.skip_whitespace(&input[1..]);
        Ok(Parsed { value: Some(wd), input })
    }

    /// Parses a 1 or 2 digit day.
    ///
    /// This assumes the input starts with what must be an ASCII digit (or it
    /// may be empty).
    ///
    /// This also parses at least one mandatory whitespace character after the
    /// day.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_day<'i>(&self, input: &'i [u8]) -> Result<Parsed<'i, i8>, Error> {
        if input.is_empty() {
            return Err(Error::from(E::EndOfInputDay));
        }
        let mut digits = 1;
        if input.len() >= 2 && matches!(input[1], b'0'..=b'9') {
            digits = 2;
        }
        let (day, input) = input.split_at(digits);
        let day = b::Day::parse(day).context(E::ParseDay)?;
        let Parsed { input, .. } =
            self.parse_whitespace(input).context(E::WhitespaceAfterDay)?;
        Ok(Parsed { value: day, input })
    }

    /// Parses an abbreviated month name.
    ///
    /// This assumes the input starts with what must be the beginning of a
    /// month name (or the input may be empty).
    ///
    /// This also parses at least one mandatory whitespace character after the
    /// month name.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_month<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, i8>, Error> {
        if input.is_empty() {
            return Err(Error::from(E::EndOfInputMonth));
        }
        if let Ok(len) = u8::try_from(input.len()) {
            if len < 3 {
                return Err(Error::from(E::TooShortMonth { len }));
            }
        }
        let b1 = input[0].to_ascii_lowercase();
        let b2 = input[1].to_ascii_lowercase();
        let b3 = input[2].to_ascii_lowercase();
        let month = match &[b1, b2, b3] {
            b"jan" => 1,
            b"feb" => 2,
            b"mar" => 3,
            b"apr" => 4,
            b"may" => 5,
            b"jun" => 6,
            b"jul" => 7,
            b"aug" => 8,
            b"sep" => 9,
            b"oct" => 10,
            b"nov" => 11,
            b"dec" => 12,
            _ => return Err(Error::from(E::InvalidMonth)),
        };
        let Parsed { input, .. } = self
            .parse_whitespace(&input[3..])
            .context(E::WhitespaceAfterMonth)?;
        Ok(Parsed { value: month, input })
    }

    /// Parses a 2, 3 or 4 digit year.
    ///
    /// This assumes the input starts with what must be an ASCII digit (or it
    /// may be empty).
    ///
    /// This also parses at least one mandatory whitespace character after the
    /// day.
    ///
    /// The 2 or 3 digit years are "obsolete," which we support by following
    /// the rules in RFC 2822:
    ///
    /// > Where a two or three digit year occurs in a date, the year is to be
    /// > interpreted as follows: If a two digit year is encountered whose
    /// > value is between 00 and 49, the year is interpreted by adding 2000,
    /// > ending up with a value between 2000 and 2049. If a two digit year is
    /// > encountered with a value between 50 and 99, or any three digit year
    /// > is encountered, the year is interpreted by adding 1900.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_year<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, i16>, Error> {
        let mut digits = 0;
        while digits <= 3
            && !input[digits..].is_empty()
            && matches!(input[digits], b'0'..=b'9')
        {
            digits += 1;
        }
        if let Ok(len) = u8::try_from(digits) {
            if len <= 1 {
                return Err(Error::from(E::TooShortYear { len }));
            }
        }
        let (year, input) = input.split_at(digits);
        let year = b::Year::parse(year).context(E::ParseYear)?;
        let year = match digits {
            2 if year <= 49 => year + 2000,
            2 | 3 => year + 1900,
            4 => year,
            _ => unreachable!("digits={digits} must be 2, 3 or 4"),
        };
        let Parsed { input, .. } =
            self.parse_whitespace(input).context(E::WhitespaceAfterYear)?;
        Ok(Parsed { value: year, input })
    }

    /// Parses a 2-digit hour. This assumes the input begins with what should
    /// be an ASCII digit. (i.e., It doesn't trim leading whitespace.)
    ///
    /// This parses a mandatory trailing `:`, advancing the input to
    /// immediately after it.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_hour<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, i8>, Error> {
        let (hour, input) = parse::split(input, 2).ok_or(E::EndOfInputHour)?;
        let hour = b::Hour::parse(hour).context(E::ParseHour)?;
        Ok(Parsed { value: hour, input })
    }

    /// Parses a 2-digit minute. This assumes the input begins with what should
    /// be an ASCII digit. (i.e., It doesn't trim leading whitespace.)
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_minute<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, i8>, Error> {
        let (minute, input) =
            parse::split(input, 2).ok_or(E::EndOfInputMinute)?;
        let minute = b::Minute::parse(minute).context(E::ParseMinute)?;
        Ok(Parsed { value: minute, input })
    }

    /// Parses a 2-digit second. This assumes the input begins with what should
    /// be an ASCII digit. (i.e., It doesn't trim leading whitespace.)
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_second<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, i8>, Error> {
        let (second, input) =
            parse::split(input, 2).ok_or(E::EndOfInputSecond)?;
        let mut second =
            b::LeapSecond::parse(second).context(E::ParseSecond)?;
        if second == 60 {
            second = 59;
        }
        Ok(Parsed { value: second, input })
    }

    /// Parses a time zone offset (including obsolete offsets like EDT).
    ///
    /// This assumes the offset must begin at the beginning of `input`. That
    /// is, any leading whitespace should already have been trimmed.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_offset<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, Offset>, Error> {
        let sign = input.get(0).copied().ok_or(E::EndOfInputOffset)?;
        let sign = if sign == b'+' {
            b::Sign::Positive
        } else if sign == b'-' {
            b::Sign::Negative
        } else {
            return self.parse_offset_obsolete(input);
        };
        let input = &input[1..];
        let (hhmm, input) = parse::split(input, 4).ok_or(E::TooShortOffset)?;

        let hh =
            b::OffsetHours::parse(&hhmm[0..2]).context(E::ParseOffsetHour)?;
        let mm = b::OffsetMinutes::parse(&hhmm[2..4])
            .context(E::ParseOffsetMinute)?;

        let seconds = sign * (i32::from(hh) * 3_600 + i32::from(mm) * 60);
        // OK because we check the bounds of both hours and minutes.
        let offset = Offset::from_seconds(seconds).unwrap();
        Ok(Parsed { value: offset, input })
    }

    /// Parses an obsolete time zone offset.
    #[inline(never)]
    fn parse_offset_obsolete<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, Offset>, Error> {
        let mut letters = [0; 5];
        let mut len = 0;
        while len <= 4
            && !input[len..].is_empty()
            && !is_whitespace(input[len])
        {
            letters[len] = input[len].to_ascii_lowercase();
            len += 1;
        }
        if len == 0 {
            return Err(Error::from(E::WhitespaceAfterTimeForObsoleteOffset));
        }
        let offset = match &letters[..len] {
            b"ut" | b"gmt" | b"z" => Offset::UTC,
            b"est" => Offset::constant(-5),
            b"edt" => Offset::constant(-4),
            b"cst" => Offset::constant(-6),
            b"cdt" => Offset::constant(-5),
            b"mst" => Offset::constant(-7),
            b"mdt" => Offset::constant(-6),
            b"pst" => Offset::constant(-8),
            b"pdt" => Offset::constant(-7),
            name => {
                if name.len() == 1
                    && matches!(name[0], b'a'..=b'i' | b'k'..=b'z')
                {
                    // Section 4.3 indicates these as military time:
                    //
                    // > The 1 character military time zones were defined in
                    // > a non-standard way in [RFC822] and are therefore
                    // > unpredictable in their meaning. The original
                    // > definitions of the military zones "A" through "I" are
                    // > equivalent to "+0100" through "+0900" respectively;
                    // > "K", "L", and "M" are equivalent to "+1000", "+1100",
                    // > and "+1200" respectively; "N" through "Y" are
                    // > equivalent to "-0100" through "-1200" respectively;
                    // > and "Z" is equivalent to "+0000". However, because of
                    // > the error in [RFC822], they SHOULD all be considered
                    // > equivalent to "-0000" unless there is out-of-band
                    // > information confirming their meaning.
                    //
                    // So just treat them as UTC.
                    Offset::UTC
                } else if name.len() >= 3
                    && name.iter().all(|&b| matches!(b, b'a'..=b'z'))
                {
                    // Section 4.3 also says that anything that _looks_ like a
                    // zone name should just be -0000 too:
                    //
                    // > Other multi-character (usually between 3 and 5)
                    // > alphabetic time zones have been used in Internet
                    // > messages. Any such time zone whose meaning is not
                    // > known SHOULD be considered equivalent to "-0000"
                    // > unless there is out-of-band information confirming
                    // > their meaning.
                    Offset::UTC
                } else {
                    // But anything else we throw our hands up I guess.
                    return Err(Error::from(E::InvalidObsoleteOffset));
                }
            }
        };
        Ok(Parsed { value: offset, input: &input[len..] })
    }

    /// Parses a time separator. This returns an error if one couldn't be
    /// found.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_time_separator<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, ()>, Error> {
        if input.is_empty() {
            return Err(Error::from(E::EndOfInputTimeSeparator));
        }
        if input[0] != b':' {
            return Err(Error::from(E::UnexpectedByteTimeSeparator {
                byte: input[0],
            }));
        }
        Ok(Parsed { value: (), input: &input[1..] })
    }

    /// Parses at least one whitespace character. If no whitespace was found,
    /// then this returns an error.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_whitespace<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, ()>, Error> {
        let Parsed { input, value: had_whitespace } =
            self.skip_whitespace(input);
        if !had_whitespace {
            return Err(Error::from(E::WhitespaceAfterTime));
        }
        Ok(Parsed { value: (), input })
    }

    /// Skips over any ASCII whitespace at the beginning of `input`.
    ///
    /// This returns the input unchanged if it does not begin with whitespace.
    /// The resulting value is `true` if any whitespace was consumed,
    /// and `false` if none was.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn skip_whitespace<'i>(&self, mut input: &'i [u8]) -> Parsed<'i, bool> {
        let mut found_whitespace = false;
        while input.first().map_or(false, |&b| is_whitespace(b)) {
            input = &input[1..];
            found_whitespace = true;
        }
        Parsed { value: found_whitespace, input }
    }

    /// This attempts to parse and skip any trailing "comment" in an RFC 2822
    /// datetime.
    ///
    /// This is a bit more relaxed than what RFC 2822 specifies. We basically
    /// just try to balance parenthesis and skip over escapes.
    ///
    /// This assumes that if a comment exists, its opening parenthesis is at
    /// the beginning of `input`. That is, any leading whitespace has been
    /// stripped.
    #[inline(never)]
    fn skip_comment<'i>(
        &self,
        mut input: &'i [u8],
    ) -> Result<Parsed<'i, ()>, Error> {
        if !input.starts_with(b"(") {
            return Ok(Parsed { value: (), input });
        }
        input = &input[1..];
        let mut depth: u8 = 1;
        let mut escape = false;
        for byte in input.iter().copied() {
            input = &input[1..];
            if escape {
                escape = false;
            } else if byte == b'\\' {
                escape = true;
            } else if byte == b')' {
                // I believe this error case is actually impossible, since as
                // soon as we hit 0, we break out. If there is more "comment,"
                // then it will flag an error as unparsed input.
                depth = depth
                    .checked_sub(1)
                    .ok_or(E::CommentClosingParenWithoutOpen)?;
                if depth == 0 {
                    break;
                }
            } else if byte == b'(' {
                depth = depth
                    .checked_add(1)
                    .ok_or(E::CommentTooManyNestedParens)?;
            }
        }
        if depth > 0 {
            return Err(Error::from(E::CommentOpeningParenWithoutClose));
        }
        let Parsed { input, .. } = self.skip_whitespace(input);
        Ok(Parsed { value: (), input })
    }
}

/// A printer for [RFC 2822] datetimes.
///
/// This printer converts an in memory representation of a precise instant in
/// time to an RFC 2822 formatted string. That is, [`Zoned`] or [`Timestamp`],
/// since all other datetime types in Jiff are inexact.
///
/// [RFC 2822]: https://datatracker.ietf.org/doc/html/rfc2822
///
/// # Warning
///
/// The RFC 2822 format only supports writing a precise instant in time
/// expressed via a time zone offset. It does *not* support serializing
/// the time zone itself. This means that if you format a zoned datetime
/// in a time zone like `America/New_York` and then deserialize it, the
/// zoned datetime you get back will be a "fixed offset" zoned datetime.
/// This in turn means it will not perform daylight saving time safe
/// arithmetic.
///
/// Basically, you should use the RFC 2822 format if it's required (for
/// example, when dealing with email). But you should not choose it as a
/// general interchange format for new applications.
///
/// # Example
///
/// This example shows how to convert a zoned datetime to the RFC 2822 format:
///
/// ```
/// use jiff::{civil::date, fmt::rfc2822::DateTimePrinter};
///
/// const PRINTER: DateTimePrinter = DateTimePrinter::new();
///
/// let zdt = date(2024, 6, 15).at(7, 0, 0, 0).in_tz("Australia/Tasmania")?;
///
/// let mut buf = String::new();
/// PRINTER.print_zoned(&zdt, &mut buf)?;
/// assert_eq!(buf, "Sat, 15 Jun 2024 07:00:00 +1000");
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Example: using adapters with `std::io::Write` and `std::fmt::Write`
///
/// By using the [`StdIoWrite`](super::StdIoWrite) and
/// [`StdFmtWrite`](super::StdFmtWrite) adapters, one can print datetimes
/// directly to implementations of `std::io::Write` and `std::fmt::Write`,
/// respectively. The example below demonstrates writing to anything
/// that implements `std::io::Write`. Similar code can be written for
/// `std::fmt::Write`.
///
/// ```no_run
/// use std::{fs::File, io::{BufWriter, Write}, path::Path};
///
/// use jiff::{civil::date, fmt::{StdIoWrite, rfc2822::DateTimePrinter}};
///
/// let zdt = date(2024, 6, 15).at(7, 0, 0, 0).in_tz("Asia/Kolkata")?;
///
/// let path = Path::new("/tmp/output");
/// let mut file = BufWriter::new(File::create(path)?);
/// DateTimePrinter::new().print_zoned(&zdt, StdIoWrite(&mut file)).unwrap();
/// file.flush()?;
/// assert_eq!(
///     std::fs::read_to_string(path)?,
///     "Sat, 15 Jun 2024 07:00:00 +0530",
/// );
///
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct DateTimePrinter {
    // The RFC 2822 printer has no configuration at present.
    _private: (),
}

impl DateTimePrinter {
    /// Create a new RFC 2822 datetime printer with the default configuration.
    #[inline]
    pub const fn new() -> DateTimePrinter {
        DateTimePrinter { _private: () }
    }

    /// Format a `Zoned` datetime into a string.
    ///
    /// This never emits `-0000` as the offset in the RFC 2822 format. If you
    /// desire a `-0000` offset, use [`DateTimePrinter::print_timestamp`] via
    /// [`Zoned::timestamp`].
    ///
    /// Moreover, since RFC 2822 does not support fractional seconds, this
    /// routine prints the zoned datetime as if truncating any fractional
    /// seconds.
    ///
    /// This is a convenience routine for [`DateTimePrinter::print_zoned`]
    /// with a `String`.
    ///
    /// # Warning
    ///
    /// The RFC 2822 format only supports writing a precise instant in time
    /// expressed via a time zone offset. It does *not* support serializing
    /// the time zone itself. This means that if you format a zoned datetime
    /// in a time zone like `America/New_York` and then deserialize it, the
    /// zoned datetime you get back will be a "fixed offset" zoned datetime.
    /// This in turn means it will not perform daylight saving time safe
    /// arithmetic.
    ///
    /// Basically, you should use the RFC 2822 format if it's required (for
    /// example, when dealing with email). But you should not choose it as a
    /// general interchange format for new applications.
    ///
    /// # Errors
    ///
    /// This can return an error if the year corresponding to this timestamp
    /// cannot be represented in the RFC 2822 format. For example, a negative
    /// year.
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::{civil::date, fmt::rfc2822::DateTimePrinter};
    ///
    /// const PRINTER: DateTimePrinter = DateTimePrinter::new();
    ///
    /// let zdt = date(2024, 6, 15).at(7, 0, 0, 0).in_tz("America/New_York")?;
    /// assert_eq!(
    ///     PRINTER.zoned_to_string(&zdt)?,
    ///     "Sat, 15 Jun 2024 07:00:00 -0400",
    /// );
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[cfg(feature = "alloc")]
    pub fn zoned_to_string(
        &self,
        zdt: &Zoned,
    ) -> Result<alloc::string::String, Error> {
        // Writing directly into the unused capacity of a `String` saves about
        // 40% on a micro-benchmark compared to just passing a `&mut String`
        // to `print_zoned`.
        let mut buf =
            alloc::string::String::with_capacity(PRINTER_MAX_BYTES_RFC2822);
        self.print_zoned(zdt, &mut buf)?;
        Ok(buf)
    }

    /// Format a `Timestamp` datetime into a string.
    ///
    /// This always emits `-0000` as the offset in the RFC 2822 format. If you
    /// desire a `+0000` offset, use [`DateTimePrinter::print_zoned`] with a
    /// zoned datetime with [`TimeZone::UTC`].
    ///
    /// Moreover, since RFC 2822 does not support fractional seconds, this
    /// routine prints the timestamp as if truncating any fractional seconds.
    ///
    /// This is a convenience routine for [`DateTimePrinter::print_timestamp`]
    /// with a `String`.
    ///
    /// # Errors
    ///
    /// This returns an error if the year corresponding to this
    /// timestamp cannot be represented in the RFC 2822 format. For example, a
    /// negative year.
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::{fmt::rfc2822::DateTimePrinter, Timestamp};
    ///
    /// let timestamp = Timestamp::from_second(1)
    ///     .expect("one second after Unix epoch is always valid");
    /// assert_eq!(
    ///     DateTimePrinter::new().timestamp_to_string(&timestamp)?,
    ///     "Thu, 1 Jan 1970 00:00:01 -0000",
    /// );
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[cfg(feature = "alloc")]
    pub fn timestamp_to_string(
        &self,
        timestamp: &Timestamp,
    ) -> Result<alloc::string::String, Error> {
        let mut buf =
            alloc::string::String::with_capacity(PRINTER_MAX_BYTES_RFC2822);
        self.print_timestamp(timestamp, &mut buf)?;
        Ok(buf)
    }

    /// Format a `Timestamp` datetime into a string in a way that is explicitly
    /// compatible with [RFC 9110]. This is typically useful in contexts where
    /// strict compatibility with HTTP is desired.
    ///
    /// This always emits `GMT` as the offset and always uses two digits for
    /// the day. This results in a fixed length format that always uses 29
    /// characters.
    ///
    /// Since neither RFC 2822 nor RFC 9110 supports fractional seconds, this
    /// routine prints the timestamp as if truncating any fractional seconds.
    ///
    /// This is a convenience routine for
    /// [`DateTimePrinter::print_timestamp_rfc9110`] with a `String`.
    ///
    /// # Errors
    ///
    /// This returns an error if the year corresponding to this timestamp
    /// cannot be represented in the RFC 2822 or RFC 9110 format. For example,
    /// a negative year.
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::{fmt::rfc2822::DateTimePrinter, Timestamp};
    ///
    /// let timestamp = Timestamp::from_second(1)
    ///     .expect("one second after Unix epoch is always valid");
    /// assert_eq!(
    ///     DateTimePrinter::new().timestamp_to_rfc9110_string(&timestamp)?,
    ///     "Thu, 01 Jan 1970 00:00:01 GMT",
    /// );
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// [RFC 9110]: https://datatracker.ietf.org/doc/html/rfc9110#section-5.6.7-15
    #[cfg(feature = "alloc")]
    pub fn timestamp_to_rfc9110_string(
        &self,
        timestamp: &Timestamp,
    ) -> Result<alloc::string::String, Error> {
        let mut buf =
            alloc::string::String::with_capacity(PRINTER_MAX_BYTES_RFC9110);
        self.print_timestamp_rfc9110(timestamp, &mut buf)?;
        Ok(buf)
    }

    /// Print a `Zoned` datetime to the given writer.
    ///
    /// This never emits `-0000` as the offset in the RFC 2822 format. If you
    /// desire a `-0000` offset, use [`DateTimePrinter::print_timestamp`] via
    /// [`Zoned::timestamp`].
    ///
    /// Moreover, since RFC 2822 does not support fractional seconds, this
    /// routine prints the zoned datetime as if truncating any fractional
    /// seconds.
    ///
    /// # Warning
    ///
    /// The RFC 2822 format only supports writing a precise instant in time
    /// expressed via a time zone offset. It does *not* support serializing
    /// the time zone itself. This means that if you format a zoned datetime
    /// in a time zone like `America/New_York` and then deserialize it, the
    /// zoned datetime you get back will be a "fixed offset" zoned datetime.
    /// This in turn means it will not perform daylight saving time safe
    /// arithmetic.
    ///
    /// Basically, you should use the RFC 2822 format if it's required (for
    /// example, when dealing with email). But you should not choose it as a
    /// general interchange format for new applications.
    ///
    /// # Errors
    ///
    /// This returns an error when writing to the given [`Write`]
    /// implementation would fail. Some such implementations, like for `String`
    /// and `Vec<u8>`, never fail (unless memory allocation fails).
    ///
    /// This can also return an error if the year corresponding to this
    /// timestamp cannot be represented in the RFC 2822 format. For example, a
    /// negative year.
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::{civil::date, fmt::rfc2822::DateTimePrinter};
    ///
    /// const PRINTER: DateTimePrinter = DateTimePrinter::new();
    ///
    /// let zdt = date(2024, 6, 15).at(7, 0, 0, 0).in_tz("America/New_York")?;
    ///
    /// let mut buf = String::new();
    /// PRINTER.print_zoned(&zdt, &mut buf)?;
    /// assert_eq!(buf, "Sat, 15 Jun 2024 07:00:00 -0400");
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn print_zoned<W: Write>(
        &self,
        zdt: &Zoned,
        mut wtr: W,
    ) -> Result<(), Error> {
        BorrowedBuffer::with_writer::<PRINTER_MAX_BYTES_RFC2822>(
            &mut wtr,
            PRINTER_MAX_BYTES_RFC2822,
            |bbuf| {
                self.print_civil_with_offset(
                    zdt.datetime(),
                    Some(zdt.offset()),
                    bbuf,
                )
            },
        )
    }

    /// Print a `Timestamp` datetime to the given writer.
    ///
    /// This always emits `-0000` as the offset in the RFC 2822 format. If you
    /// desire a `+0000` offset, use [`DateTimePrinter::print_zoned`] with a
    /// zoned datetime with [`TimeZone::UTC`].
    ///
    /// Moreover, since RFC 2822 does not support fractional seconds, this
    /// routine prints the timestamp as if truncating any fractional seconds.
    ///
    /// # Errors
    ///
    /// This returns an error when writing to the given [`Write`]
    /// implementation would fail. Some such implementations, like for `String`
    /// and `Vec<u8>`, never fail (unless memory allocation fails).
    ///
    /// This can also return an error if the year corresponding to this
    /// timestamp cannot be represented in the RFC 2822 format. For example, a
    /// negative year.
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::{fmt::rfc2822::DateTimePrinter, Timestamp};
    ///
    /// let timestamp = Timestamp::from_second(1)
    ///     .expect("one second after Unix epoch is always valid");
    ///
    /// let mut buf = String::new();
    /// DateTimePrinter::new().print_timestamp(&timestamp, &mut buf)?;
    /// assert_eq!(buf, "Thu, 1 Jan 1970 00:00:01 -0000");
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn print_timestamp<W: Write>(
        &self,
        timestamp: &Timestamp,
        mut wtr: W,
    ) -> Result<(), Error> {
        let dt = TimeZone::UTC.to_datetime(*timestamp);
        BorrowedBuffer::with_writer::<PRINTER_MAX_BYTES_RFC2822>(
            &mut wtr,
            PRINTER_MAX_BYTES_RFC2822,
            |bbuf| self.print_civil_with_offset(dt, None, bbuf),
        )
    }

    /// Print a `Timestamp` datetime to the given writer in a way that is
    /// explicitly compatible with [RFC 9110]. This is typically useful in
    /// contexts where strict compatibility with HTTP is desired.
    ///
    /// This always emits `GMT` as the offset and always uses two digits for
    /// the day. This results in a fixed length format that always uses 29
    /// characters.
    ///
    /// Since neither RFC 2822 nor RFC 9110 supports fractional seconds, this
    /// routine prints the timestamp as if truncating any fractional seconds.
    ///
    /// # Errors
    ///
    /// This returns an error when writing to the given [`Write`]
    /// implementation would fail. Some such implementations, like for `String`
    /// and `Vec<u8>`, never fail (unless memory allocation fails).
    ///
    /// This can also return an error if the year corresponding to this
    /// timestamp cannot be represented in the RFC 2822 or RFC 9110 format. For
    /// example, a negative year.
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::{fmt::rfc2822::DateTimePrinter, Timestamp};
    ///
    /// let timestamp = Timestamp::from_second(1)
    ///     .expect("one second after Unix epoch is always valid");
    ///
    /// let mut buf = String::new();
    /// DateTimePrinter::new().print_timestamp_rfc9110(&timestamp, &mut buf)?;
    /// assert_eq!(buf, "Thu, 01 Jan 1970 00:00:01 GMT");
    ///
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// [RFC 9110]: https://datatracker.ietf.org/doc/html/rfc9110#section-5.6.7-15
    pub fn print_timestamp_rfc9110<W: Write>(
        &self,
        timestamp: &Timestamp,
        mut wtr: W,
    ) -> Result<(), Error> {
        let dt = TimeZone::UTC.to_datetime(*timestamp);
        BorrowedBuffer::with_writer::<PRINTER_MAX_BYTES_RFC9110>(
            &mut wtr,
            PRINTER_MAX_BYTES_RFC9110,
            |bbuf| self.print_civil_always_utc(dt, bbuf),
        )
    }

    #[inline(never)]
    fn print_civil_with_offset(
        &self,
        dt: DateTime,
        offset: Option<Offset>,
        buf: &mut BorrowedBuffer<'_>,
    ) -> Result<(), Error> {
        if dt.year() < 0 {
            // RFC 2822 actually says the year must be at least 1900, but
            // other implementations (like Chrono) allow any positive 4-digit
            // year.
            return Err(Error::from(E::NegativeYear));
        }

        buf.write_str(weekday_abbrev(dt.weekday()));
        buf.write_str(", ");
        buf.write_int(dt.day().unsigned_abs());
        buf.write_ascii_char(b' ');
        buf.write_str(month_name(dt.month()));
        buf.write_ascii_char(b' ');
        buf.write_int_pad4(dt.year().unsigned_abs());
        buf.write_ascii_char(b' ');
        buf.write_int_pad2(dt.hour().unsigned_abs());
        buf.write_ascii_char(b':');
        buf.write_int_pad2(dt.minute().unsigned_abs());
        buf.write_ascii_char(b':');
        buf.write_int_pad2(dt.second().unsigned_abs());
        buf.write_ascii_char(b' ');

        let Some(offset) = offset else {
            buf.write_str("-0000");
            return Ok(());
        };
        buf.write_ascii_char(if offset.is_negative() { b'-' } else { b'+' });
        let (offset_hours, offset_minutes) = offset.round_to_nearest_minute();
        buf.write_int_pad2(offset_hours);
        buf.write_int_pad2(offset_minutes);

        Ok(())
    }

    #[inline(never)]
    fn print_civil_always_utc(
        &self,
        dt: DateTime,
        buf: &mut BorrowedBuffer<'_>,
    ) -> Result<(), Error> {
        if dt.year() < 0 {
            // RFC 2822 actually says the year must be at least 1900, but
            // other implementations (like Chrono) allow any positive 4-digit
            // year.
            return Err(Error::from(E::NegativeYear));
        }

        buf.write_str(weekday_abbrev(dt.weekday()));
        buf.write_str(", ");
        buf.write_int_pad2(dt.day().unsigned_abs());
        buf.write_str(" ");
        buf.write_str(month_name(dt.month()));
        buf.write_str(" ");
        buf.write_int_pad4(dt.year().unsigned_abs());
        buf.write_str(" ");
        buf.write_int_pad2(dt.hour().unsigned_abs());
        buf.write_str(":");
        buf.write_int_pad2(dt.minute().unsigned_abs());
        buf.write_str(":");
        buf.write_int_pad2(dt.second().unsigned_abs());
        buf.write_str(" ");
        buf.write_str("GMT");
        Ok(())
    }
}

fn weekday_abbrev(wd: Weekday) -> &'static str {
    match wd {
        Weekday::Sunday => "Sun",
        Weekday::Monday => "Mon",
        Weekday::Tuesday => "Tue",
        Weekday::Wednesday => "Wed",
        Weekday::Thursday => "Thu",
        Weekday::Friday => "Fri",
        Weekday::Saturday => "Sat",
    }
}

fn month_name(month: i8) -> &'static str {
    match month {
        1 => "Jan",
        2 => "Feb",
        3 => "Mar",
        4 => "Apr",
        5 => "May",
        6 => "Jun",
        7 => "Jul",
        8 => "Aug",
        9 => "Sep",
        10 => "Oct",
        11 => "Nov",
        12 => "Dec",
        _ => unreachable!("invalid month value {month}"),
    }
}

/// Returns true if the given byte is "whitespace" as defined by RFC 2822.
///
/// From S2.2.2:
///
/// > Many of these tokens are allowed (according to their syntax) to be
/// > introduced or end with comments (as described in section 3.2.3) as well
/// > as the space (SP, ASCII value 32) and horizontal tab (HTAB, ASCII value
/// > 9) characters (together known as the white space characters, WSP), and
/// > those WSP characters are subject to header "folding" and "unfolding" as
/// > described in section 2.2.3.
///
/// In other words, ASCII space or tab.
///
/// With all that said, it seems odd to limit this to just spaces or tabs, so
/// we relax this and let it absorb any kind of ASCII whitespace. This also
/// handles, I believe, most cases of "folding" whitespace. (By treating `\r`
/// and `\n` as whitespace.)
fn is_whitespace(byte: u8) -> bool {
    byte.is_ascii_whitespace()
}

#[cfg(feature = "alloc")]
#[cfg(test)]
mod tests {
    use alloc::string::{String, ToString};

    use crate::civil::date;

    use super::*;

    #[test]
    fn ok_parse_basic() {
        let p = |input| DateTimeParser::new().parse_zoned(input).unwrap();

        insta::assert_debug_snapshot!(
            p("Wed, 10 Jan 2024 05:34:45 -0500"),
            @"2024-01-10T05:34:45-05:00[-05:00]",
        );
        insta::assert_debug_snapshot!(
            p("Tue, 9 Jan 2024 05:34:45 -0500"),
            @"2024-01-09T05:34:45-05:00[-05:00]",
        );
        insta::assert_debug_snapshot!(
            p("Tue, 09 Jan 2024 05:34:45 -0500"),
            @"2024-01-09T05:34:45-05:00[-05:00]",
        );
        insta::assert_debug_snapshot!(
            p("10 Jan 2024 05:34:45 -0500"),
            @"2024-01-10T05:34:45-05:00[-05:00]",
        );
        insta::assert_debug_snapshot!(
            p("10 Jan 2024 05:34 -0500"),
            @"2024-01-10T05:34:00-05:00[-05:00]",
        );
        insta::assert_debug_snapshot!(
            p("10 Jan 2024 05:34:45 +0500"),
            @"2024-01-10T05:34:45+05:00[+05:00]",
        );
        insta::assert_debug_snapshot!(
            p("Thu, 29 Feb 2024 05:34 -0500"),
            @"2024-02-29T05:34:00-05:00[-05:00]",
        );

        // leap second constraining
        insta::assert_debug_snapshot!(
            p("10 Jan 2024 05:34:60 -0500"),
            @"2024-01-10T05:34:59-05:00[-05:00]",
        );
    }

    #[test]
    fn ok_parse_obsolete_zone() {
        let p = |input| DateTimeParser::new().parse_zoned(input).unwrap();

        insta::assert_debug_snapshot!(
            p("Wed, 10 Jan 2024 05:34:45 EST"),
            @"2024-01-10T05:34:45-05:00[-05:00]",
        );
        insta::assert_debug_snapshot!(
            p("Wed, 10 Jan 2024 05:34:45 EDT"),
            @"2024-01-10T05:34:45-04:00[-04:00]",
        );
        insta::assert_debug_snapshot!(
            p("Wed, 10 Jan 2024 05:34:45 CST"),
            @"2024-01-10T05:34:45-06:00[-06:00]",
        );
        insta::assert_debug_snapshot!(
            p("Wed, 10 Jan 2024 05:34:45 CDT"),
            @"2024-01-10T05:34:45-05:00[-05:00]",
        );
        insta::assert_debug_snapshot!(
            p("Wed, 10 Jan 2024 05:34:45 mst"),
            @"2024-01-10T05:34:45-07:00[-07:00]",
        );
        insta::assert_debug_snapshot!(
            p("Wed, 10 Jan 2024 05:34:45 mdt"),
            @"2024-01-10T05:34:45-06:00[-06:00]",
        );
        insta::assert_debug_snapshot!(
            p("Wed, 10 Jan 2024 05:34:45 pst"),
            @"2024-01-10T05:34:45-08:00[-08:00]",
        );
        insta::assert_debug_snapshot!(
            p("Wed, 10 Jan 2024 05:34:45 pdt"),
            @"2024-01-10T05:34:45-07:00[-07:00]",
        );

        // Various things that mean UTC.
        insta::assert_debug_snapshot!(
            p("Wed, 10 Jan 2024 05:34:45 UT"),
            @"2024-01-10T05:34:45+00:00[UTC]",
        );
        insta::assert_debug_snapshot!(
            p("Wed, 10 Jan 2024 05:34:45 Z"),
            @"2024-01-10T05:34:45+00:00[UTC]",
        );
        insta::assert_debug_snapshot!(
            p("Wed, 10 Jan 2024 05:34:45 gmt"),
            @"2024-01-10T05:34:45+00:00[UTC]",
        );

        // Even things that are unrecognized just get treated as having
        // an offset of 0.
        insta::assert_debug_snapshot!(
            p("Wed, 10 Jan 2024 05:34:45 XXX"),
            @"2024-01-10T05:34:45+00:00[UTC]",
        );
        insta::assert_debug_snapshot!(
            p("Wed, 10 Jan 2024 05:34:45 ABCDE"),
            @"2024-01-10T05:34:45+00:00[UTC]",
        );
        insta::assert_debug_snapshot!(
            p("Wed, 10 Jan 2024 05:34:45 FUCK"),
            @"2024-01-10T05:34:45+00:00[UTC]",
        );
    }

    // whyyyyyyyyyyyyy
    #[test]
    fn ok_parse_comment() {
        let p = |input| DateTimeParser::new().parse_zoned(input).unwrap();

        insta::assert_debug_snapshot!(
            p("Wed, 10 Jan 2024 05:34:45 -0500 (wat)"),
            @"2024-01-10T05:34:45-05:00[-05:00]",
        );
        insta::assert_debug_snapshot!(
            p("Wed, 10 Jan 2024 05:34:45 -0500 (w(a)t)"),
            @"2024-01-10T05:34:45-05:00[-05:00]",
        );
        insta::assert_debug_snapshot!(
            p(r"Wed, 10 Jan 2024 05:34:45 -0500 (w\(a\)t)"),
            @"2024-01-10T05:34:45-05:00[-05:00]",
        );
    }

    #[test]
    fn ok_parse_whitespace() {
        let p = |input| DateTimeParser::new().parse_zoned(input).unwrap();

        insta::assert_debug_snapshot!(
            p("Wed, 10 \t   Jan \n\r\n\n 2024       05:34:45    -0500"),
            @"2024-01-10T05:34:45-05:00[-05:00]",
        );
        insta::assert_debug_snapshot!(
            p("Wed, 10 Jan 2024 05:34:45 -0500 "),
            @"2024-01-10T05:34:45-05:00[-05:00]",
        );
        // Whitespace around the comma is optional
        insta::assert_debug_snapshot!(
            p("Wed,10 Jan 2024 05:34:45 -0500"),
            @"2024-01-10T05:34:45-05:00[-05:00]",
        );
        insta::assert_debug_snapshot!(
            p("Wed    ,     10 Jan 2024 05:34:45 -0500"),
            @"2024-01-10T05:34:45-05:00[-05:00]",
        );
        insta::assert_debug_snapshot!(
            p("Wed    ,10 Jan 2024 05:34:45 -0500"),
            @"2024-01-10T05:34:45-05:00[-05:00]",
        );
        // Whitespace is allowed around the time components
        insta::assert_debug_snapshot!(
            p("Wed, 10 Jan 2024 05   :34:  45 -0500"),
            @"2024-01-10T05:34:45-05:00[-05:00]",
        );
        insta::assert_debug_snapshot!(
            p("Wed, 10 Jan 2024 05:  34 :45 -0500"),
            @"2024-01-10T05:34:45-05:00[-05:00]",
        );
        insta::assert_debug_snapshot!(
            p("Wed, 10 Jan 2024 05 :  34 :   45 -0500"),
            @"2024-01-10T05:34:45-05:00[-05:00]",
        );
    }

    #[test]
    fn err_parse_invalid() {
        let p = |input| {
            DateTimeParser::new().parse_zoned(input).unwrap_err().to_string()
        };

        insta::assert_snapshot!(
            p("Thu, 10 Jan 2024 05:34:45 -0500"),
            @"failed to parse RFC 2822 datetime into Jiff zoned datetime: found parsed weekday of `Thursday`, but parsed datetime has weekday `Wednesday`",
        );
        insta::assert_snapshot!(
            p("Wed, 29 Feb 2023 05:34:45 -0500"),
            @"failed to parse RFC 2822 datetime into Jiff zoned datetime: invalid date: parameter 'day' for `2023-02` is invalid, must be in range `1..=28`",
        );
        insta::assert_snapshot!(
            p("Mon, 31 Jun 2024 05:34:45 -0500"),
            @"failed to parse RFC 2822 datetime into Jiff zoned datetime: invalid date: parameter 'day' for `2024-06` is invalid, must be in range `1..=30`",
        );
        insta::assert_snapshot!(
            p("Tue, 32 Jun 2024 05:34:45 -0500"),
            @"failed to parse RFC 2822 datetime into Jiff zoned datetime: failed to parse day: parameter 'day' is not in the required range of 1..=31",
        );
        insta::assert_snapshot!(
            p("Sun, 30 Jun 2024 24:00:00 -0500"),
            @"failed to parse RFC 2822 datetime into Jiff zoned datetime: failed to parse hour (expects a two digit integer): parameter 'hour' is not in the required range of 0..=23",
        );
        // No whitespace after time
        insta::assert_snapshot!(
            p("Wed, 10 Jan 2024 05:34MST"),
            @r###"failed to parse RFC 2822 datetime into Jiff zoned datetime: expected whitespace after parsing time: expected at least one whitespace character (space or tab), but found none"###,
        );
    }

    #[test]
    fn err_parse_incomplete() {
        let p = |input| {
            DateTimeParser::new().parse_zoned(input).unwrap_err().to_string()
        };

        insta::assert_snapshot!(
            p(""),
            @"failed to parse RFC 2822 datetime into Jiff zoned datetime: expected RFC 2822 datetime, but got empty string",
        );
        insta::assert_snapshot!(
            p(" "),
            @"failed to parse RFC 2822 datetime into Jiff zoned datetime: expected RFC 2822 datetime, but got empty string after trimming leading whitespace",
        );
        insta::assert_snapshot!(
            p("Wat"),
            @"failed to parse RFC 2822 datetime into Jiff zoned datetime: expected day at beginning of RFC 2822 datetime since first non-whitespace byte, `W`, is not a digit, but given string is too short (length is 3)",
        );
        insta::assert_snapshot!(
            p("Wed"),
            @"failed to parse RFC 2822 datetime into Jiff zoned datetime: expected day at beginning of RFC 2822 datetime since first non-whitespace byte, `W`, is not a digit, but given string is too short (length is 3)",
        );
        insta::assert_snapshot!(
            p("Wed "),
            @"failed to parse RFC 2822 datetime into Jiff zoned datetime: expected comma after parsed weekday in RFC 2822 datetime, but found end of input instead",
        );
        insta::assert_snapshot!(
            p("Wed   ,"),
            @"failed to parse RFC 2822 datetime into Jiff zoned datetime: expected numeric day, but found end of input",
        );
        insta::assert_snapshot!(
            p("Wed   ,   "),
            @"failed to parse RFC 2822 datetime into Jiff zoned datetime: expected numeric day, but found end of input",
        );
        insta::assert_snapshot!(
            p("Wat, "),
            @"failed to parse RFC 2822 datetime into Jiff zoned datetime: expected day at beginning of RFC 2822 datetime since first non-whitespace byte, `W`, is not a digit, but did not recognize a valid weekday abbreviation",
        );
        insta::assert_snapshot!(
            p("Wed, "),
            @"failed to parse RFC 2822 datetime into Jiff zoned datetime: expected numeric day, but found end of input",
        );
        insta::assert_snapshot!(
            p("Wed, 1"),
            @"failed to parse RFC 2822 datetime into Jiff zoned datetime: expected whitespace after parsing day: expected whitespace after parsing time: expected at least one whitespace character (space or tab), but found none",
        );
        insta::assert_snapshot!(
            p("Wed, 10"),
            @"failed to parse RFC 2822 datetime into Jiff zoned datetime: expected whitespace after parsing day: expected whitespace after parsing time: expected at least one whitespace character (space or tab), but found none",
        );
        insta::assert_snapshot!(
            p("Wed, 10 J"),
            @"failed to parse RFC 2822 datetime into Jiff zoned datetime: expected abbreviated month name, but remaining input is too short (remaining bytes is 1)",
        );
        insta::assert_snapshot!(
            p("Wed, 10 Wat"),
            @"failed to parse RFC 2822 datetime into Jiff zoned datetime: expected abbreviated month name, but did not recognize a valid abbreviated month name",
        );
        insta::assert_snapshot!(
            p("Wed, 10 Jan"),
            @"failed to parse RFC 2822 datetime into Jiff zoned datetime: expected whitespace after parsing abbreviated month name: expected whitespace after parsing time: expected at least one whitespace character (space or tab), but found none",
        );
        insta::assert_snapshot!(
            p("Wed, 10 Jan 2"),
            @"failed to parse RFC 2822 datetime into Jiff zoned datetime: expected at least two ASCII digits for parsing a year, but only found 1",
        );
        insta::assert_snapshot!(
            p("Wed, 10 Jan 2024"),
            @"failed to parse RFC 2822 datetime into Jiff zoned datetime: expected whitespace after parsing year: expected whitespace after parsing time: expected at least one whitespace character (space or tab), but found none",
        );
        insta::assert_snapshot!(
            p("Wed, 10 Jan 2024 05"),
            @"failed to parse RFC 2822 datetime into Jiff zoned datetime: expected time separator of `:`, but found end of input",
        );
        insta::assert_snapshot!(
            p("Wed, 10 Jan 2024 053"),
            @"failed to parse RFC 2822 datetime into Jiff zoned datetime: expected time separator of `:`, but found `3`",
        );
        insta::assert_snapshot!(
            p("Wed, 10 Jan 2024 05:34"),
            @"failed to parse RFC 2822 datetime into Jiff zoned datetime: expected whitespace after parsing time: expected at least one whitespace character (space or tab), but found none",
        );
        insta::assert_snapshot!(
            p("Wed, 10 Jan 2024 05:34:"),
            @"failed to parse RFC 2822 datetime into Jiff zoned datetime: expected two digit second, but found end of input",
        );
        insta::assert_snapshot!(
            p("Wed, 10 Jan 2024 05:34:45"),
            @"failed to parse RFC 2822 datetime into Jiff zoned datetime: expected whitespace after parsing time: expected at least one whitespace character (space or tab), but found none",
        );
        insta::assert_snapshot!(
            p("Wed, 10 Jan 2024 05:34:45 J"),
            @"failed to parse RFC 2822 datetime into Jiff zoned datetime: expected obsolete RFC 2822 time zone abbreviation, but did not recognize a valid abbreviation",
        );
    }

    #[test]
    fn err_parse_comment() {
        let p = |input| {
            DateTimeParser::new().parse_zoned(input).unwrap_err().to_string()
        };

        insta::assert_snapshot!(
            p(r"Wed, 10 Jan 2024 05:34:45 -0500 (wa)t)"),
            @r###"parsed value '2024-01-10T05:34:45-05:00[-05:00]', but unparsed input "t)" remains (expected no unparsed input)"###,
        );
        insta::assert_snapshot!(
            p(r"Wed, 10 Jan 2024 05:34:45 -0500 (wa(t)"),
            @"failed to parse RFC 2822 datetime into Jiff zoned datetime: found opening parenthesis in comment with no matching closing parenthesis",
        );
        insta::assert_snapshot!(
            p(r"Wed, 10 Jan 2024 05:34:45 -0500 (w"),
            @"failed to parse RFC 2822 datetime into Jiff zoned datetime: found opening parenthesis in comment with no matching closing parenthesis",
        );
        insta::assert_snapshot!(
            p(r"Wed, 10 Jan 2024 05:34:45 -0500 ("),
            @"failed to parse RFC 2822 datetime into Jiff zoned datetime: found opening parenthesis in comment with no matching closing parenthesis",
        );
        insta::assert_snapshot!(
            p(r"Wed, 10 Jan 2024 05:34:45 -0500 (  "),
            @"failed to parse RFC 2822 datetime into Jiff zoned datetime: found opening parenthesis in comment with no matching closing parenthesis",
        );
    }

    #[test]
    fn ok_print_zoned() {
        if crate::tz::db().is_definitively_empty() {
            return;
        }

        let p = |zdt: &Zoned| -> String {
            let mut buf = String::new();
            DateTimePrinter::new().print_zoned(&zdt, &mut buf).unwrap();
            buf
        };

        let zdt = date(2024, 1, 10)
            .at(5, 34, 45, 0)
            .in_tz("America/New_York")
            .unwrap();
        insta::assert_snapshot!(p(&zdt), @"Wed, 10 Jan 2024 05:34:45 -0500");

        let zdt = date(2024, 2, 5)
            .at(5, 34, 45, 0)
            .in_tz("America/New_York")
            .unwrap();
        insta::assert_snapshot!(p(&zdt), @"Mon, 5 Feb 2024 05:34:45 -0500");

        let zdt = date(2024, 7, 31)
            .at(5, 34, 45, 0)
            .in_tz("America/New_York")
            .unwrap();
        insta::assert_snapshot!(p(&zdt), @"Wed, 31 Jul 2024 05:34:45 -0400");

        let zdt = date(2024, 3, 5).at(5, 34, 45, 0).in_tz("UTC").unwrap();
        // Notice that this prints a +0000 offset.
        // But when printing a Timestamp, a -0000 offset is used.
        // This is because in the case of Timestamp, the "true"
        // offset is not known.
        insta::assert_snapshot!(p(&zdt), @"Tue, 5 Mar 2024 05:34:45 +0000");
    }

    #[test]
    fn ok_print_timestamp() {
        if crate::tz::db().is_definitively_empty() {
            return;
        }

        let p = |ts: Timestamp| -> String {
            let mut buf = String::new();
            DateTimePrinter::new().print_timestamp(&ts, &mut buf).unwrap();
            buf
        };

        let ts = date(2024, 1, 10)
            .at(5, 34, 45, 0)
            .in_tz("America/New_York")
            .unwrap()
            .timestamp();
        insta::assert_snapshot!(p(ts), @"Wed, 10 Jan 2024 10:34:45 -0000");

        let ts = date(2024, 2, 5)
            .at(5, 34, 45, 0)
            .in_tz("America/New_York")
            .unwrap()
            .timestamp();
        insta::assert_snapshot!(p(ts), @"Mon, 5 Feb 2024 10:34:45 -0000");

        let ts = date(2024, 7, 31)
            .at(5, 34, 45, 0)
            .in_tz("America/New_York")
            .unwrap()
            .timestamp();
        insta::assert_snapshot!(p(ts), @"Wed, 31 Jul 2024 09:34:45 -0000");

        let ts = date(2024, 3, 5)
            .at(5, 34, 45, 0)
            .in_tz("UTC")
            .unwrap()
            .timestamp();
        // Notice that this prints a +0000 offset.
        // But when printing a Timestamp, a -0000 offset is used.
        // This is because in the case of Timestamp, the "true"
        // offset is not known.
        insta::assert_snapshot!(p(ts), @"Tue, 5 Mar 2024 05:34:45 -0000");
    }

    #[test]
    fn ok_minimum_offset_roundtrip() {
        let zdt = date(2025, 12, 25)
            .at(17, 0, 0, 0)
            .to_zoned(TimeZone::fixed(Offset::MIN))
            .unwrap();
        let string = DateTimePrinter::new().zoned_to_string(&zdt).unwrap();
        assert_eq!(string, "Thu, 25 Dec 2025 17:00:00 -2559");

        let got: Zoned = DateTimeParser::new().parse_zoned(&string).unwrap();
        // Since we started with a zoned datetime with a minimal offset
        // (to second precision) and RFC 2822 only supports minute precision
        // in time zone offsets, printing the zoned datetime rounds the offset.
        // But this would normally result in an offset beyond Jiff's limits,
        // so in this case, the offset truncates to the minimum supported
        // value by both Jiff and RFC 2822. That's what we test for here.
        let expected = date(2025, 12, 25)
            .at(17, 0, 0, 0)
            .to_zoned(TimeZone::fixed(-Offset::hms(25, 59, 0)))
            .unwrap();
        assert_eq!(expected, got);
    }

    #[test]
    fn ok_maximum_offset_roundtrip() {
        let zdt = date(2025, 12, 25)
            .at(17, 0, 0, 0)
            .to_zoned(TimeZone::fixed(Offset::MAX))
            .unwrap();
        let string = DateTimePrinter::new().zoned_to_string(&zdt).unwrap();
        assert_eq!(string, "Thu, 25 Dec 2025 17:00:00 +2559");

        let got: Zoned = DateTimeParser::new().parse_zoned(&string).unwrap();
        // Since we started with a zoned datetime with a maximal offset
        // (to second precision) and RFC 2822 only supports minute precision
        // in time zone offsets, printing the zoned datetime rounds the offset.
        // But this would normally result in an offset beyond Jiff's limits,
        // so in this case, the offset truncates to the maximum supported
        // value by both Jiff and RFC 2822. That's what we test for here.
        let expected = date(2025, 12, 25)
            .at(17, 0, 0, 0)
            .to_zoned(TimeZone::fixed(Offset::hms(25, 59, 0)))
            .unwrap();
        assert_eq!(expected, got);
    }

    #[test]
    fn ok_print_rfc9110_timestamp() {
        if crate::tz::db().is_definitively_empty() {
            return;
        }

        let p = |ts: Timestamp| -> String {
            let mut buf = String::new();
            DateTimePrinter::new()
                .print_timestamp_rfc9110(&ts, &mut buf)
                .unwrap();
            buf
        };

        let ts = date(2024, 1, 10)
            .at(5, 34, 45, 0)
            .in_tz("America/New_York")
            .unwrap()
            .timestamp();
        insta::assert_snapshot!(p(ts), @"Wed, 10 Jan 2024 10:34:45 GMT");

        let ts = date(2024, 2, 5)
            .at(5, 34, 45, 0)
            .in_tz("America/New_York")
            .unwrap()
            .timestamp();
        insta::assert_snapshot!(p(ts), @"Mon, 05 Feb 2024 10:34:45 GMT");

        let ts = date(2024, 7, 31)
            .at(5, 34, 45, 0)
            .in_tz("America/New_York")
            .unwrap()
            .timestamp();
        insta::assert_snapshot!(p(ts), @"Wed, 31 Jul 2024 09:34:45 GMT");

        let ts = date(2024, 3, 5)
            .at(5, 34, 45, 0)
            .in_tz("UTC")
            .unwrap()
            .timestamp();
        // Notice that this prints a +0000 offset.
        // But when printing a Timestamp, a -0000 offset is used.
        // This is because in the case of Timestamp, the "true"
        // offset is not known.
        insta::assert_snapshot!(p(ts), @"Tue, 05 Mar 2024 05:34:45 GMT");
    }

    #[test]
    fn err_print_zoned() {
        if crate::tz::db().is_definitively_empty() {
            return;
        }

        let p = |zdt: &Zoned| -> String {
            let mut buf = String::new();
            DateTimePrinter::new()
                .print_zoned(&zdt, &mut buf)
                .unwrap_err()
                .to_string()
        };

        let zdt = date(-1, 1, 10)
            .at(5, 34, 45, 0)
            .in_tz("America/New_York")
            .unwrap();
        insta::assert_snapshot!(p(&zdt), @"datetime has negative year, which cannot be formatted with RFC 2822");
    }

    #[test]
    fn err_print_timestamp() {
        if crate::tz::db().is_definitively_empty() {
            return;
        }

        let p = |ts: Timestamp| -> String {
            let mut buf = String::new();
            DateTimePrinter::new()
                .print_timestamp(&ts, &mut buf)
                .unwrap_err()
                .to_string()
        };

        let ts = date(-1, 1, 10)
            .at(5, 34, 45, 0)
            .in_tz("America/New_York")
            .unwrap()
            .timestamp();
        insta::assert_snapshot!(p(ts), @"datetime has negative year, which cannot be formatted with RFC 2822");
    }
}
