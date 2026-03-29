use crate::{
    civil::{Date, DateTime, ISOWeekDate, Time, Weekday},
    error::{fmt::temporal::Error as E, Error, ErrorContext},
    fmt::{
        offset::{self, ParsedOffset},
        rfc9557::{self, ParsedAnnotations},
        temporal::Pieces,
        util::{parse_temporal_fraction, DurationUnits},
        Parsed,
    },
    span::Span,
    tz::{
        AmbiguousZoned, Disambiguation, Offset, OffsetConflict, TimeZone,
        TimeZoneDatabase,
    },
    util::{
        b::{self, Sign},
        escape, parse,
    },
    SignedDuration, Timestamp, Unit, Zoned,
};

/// The datetime components parsed from a string.
#[derive(Debug)]
pub(super) struct ParsedDateTime<'i> {
    /// A required civil date.
    date: ParsedDate,
    /// An optional civil time.
    time: Option<ParsedTime>,
    /// An optional UTC offset.
    offset: Option<ParsedOffset>,
    /// An optional RFC 9557 annotations parsed.
    ///
    /// An empty `ParsedAnnotations` is valid and possible, so this bakes
    /// optionality into the type and doesn't need to be an `Option` itself.
    annotations: ParsedAnnotations<'i>,
}

impl<'i> ParsedDateTime<'i> {
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(super) fn to_pieces(&self) -> Result<Pieces<'i>, Error> {
        let mut pieces = Pieces::from(self.date.date);
        if let Some(ref time) = self.time {
            pieces = pieces.with_time(time.time);
        }
        if let Some(ref offset) = self.offset {
            pieces = pieces.with_offset(offset.to_pieces_offset()?);
        }
        if let Some(ann) = self.annotations.to_time_zone_annotation()? {
            pieces = pieces.with_time_zone_annotation(ann);
        }
        Ok(pieces)
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(super) fn to_zoned(
        &self,
        db: &TimeZoneDatabase,
        offset_conflict: OffsetConflict,
        disambiguation: Disambiguation,
    ) -> Result<Zoned, Error> {
        self.to_ambiguous_zoned(db, offset_conflict)?
            .disambiguate(disambiguation)
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn to_ambiguous_zoned(
        &self,
        db: &TimeZoneDatabase,
        offset_conflict: OffsetConflict,
    ) -> Result<AmbiguousZoned, Error> {
        let time = self.time.as_ref().map_or(Time::midnight(), |p| p.time);
        let dt = DateTime::from_parts(self.date.date, time);

        // We always require a time zone when parsing a zoned instant.
        let tz_annotation = self
            .annotations
            .to_time_zone_annotation()?
            .ok_or(E::MissingTimeZoneAnnotation)?;
        let tz = tz_annotation.to_time_zone_with(db)?;

        // If there's no offset, then our only choice, regardless of conflict
        // resolution preference, is to use the time zone. That is, there is no
        // possible conflict.
        let Some(ref parsed_offset) = self.offset else {
            return Ok(tz.into_ambiguous_zoned(dt));
        };
        if parsed_offset.is_zulu() {
            // When `Z` is used, that means the offset to local time is not
            // known. In this case, there really can't be a conflict because
            // there is an explicit acknowledgment that the offset could be
            // anything. So we just always accept `Z` as if it were `UTC` and
            // respect that. If we didn't have this special check, we'd fall
            // below and the `Z` would just be treated as `+00:00`, which would
            // likely result in `OffsetConflict::Reject` raising an error.
            // (Unless the actual correct offset at the time is `+00:00` for
            // the time zone parsed.)
            return OffsetConflict::AlwaysOffset.resolve(dt, Offset::UTC, tz);
        }
        let offset = parsed_offset.to_offset()?;
        let is_equal = |parsed: Offset, candidate: Offset| {
            // If they're equal down to the second, then no amount of rounding
            // or whatever should change that.
            if parsed == candidate {
                return true;
            }
            // If the candidate offset we're considering is a whole minute,
            // then we never need rounding.
            //
            // Alternatively, if the parsed offset has an explicit sub-minute
            // component (even if it's zero), we should use exact equality.
            // (The error message for this case when "reject" offset
            // conflict resolution is used is not the best. But this case
            // is stupidly rare, so I'm not sure it's worth the effort to
            // improve the error message. I'd be open to a simple patch
            // though.)
            if candidate.seconds() % b::SECS_PER_MIN_32 == 0
                || parsed_offset.has_subminute()
            {
                return parsed == candidate;
            }
            let Ok(candidate) = candidate.round(Unit::Minute) else {
                // This is a degenerate case and this is the only sensible
                // thing to do.
                return parsed == candidate;
            };
            parsed == candidate
        };
        offset_conflict.resolve_with(dt, offset, tz, is_equal)
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(super) fn to_timestamp(&self) -> Result<Timestamp, Error> {
        let time = self
            .time
            .as_ref()
            .map(|p| p.time)
            .ok_or(E::MissingTimeInTimestamp)?;
        let parsed_offset =
            self.offset.as_ref().ok_or(E::MissingOffsetInTimestamp)?;
        let offset = parsed_offset.to_offset()?;
        let dt = DateTime::from_parts(self.date.date, time);
        let timestamp = offset
            .to_timestamp(dt)
            .context(E::ConvertDateTimeToTimestamp { offset })?;
        Ok(timestamp)
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(super) fn to_datetime(&self) -> Result<DateTime, Error> {
        if self.offset.as_ref().map_or(false, |o| o.is_zulu()) {
            return Err(Error::from(E::CivilDateTimeZulu));
        }
        Ok(DateTime::from_parts(self.date.date, self.time()))
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(super) fn to_date(&self) -> Result<Date, Error> {
        if self.offset.as_ref().map_or(false, |o| o.is_zulu()) {
            return Err(Error::from(E::CivilDateTimeZulu));
        }
        Ok(self.date.date)
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn time(&self) -> Time {
        self.time.as_ref().map(|p| p.time).unwrap_or(Time::midnight())
    }
}

impl<'i> core::fmt::Display for ParsedDateTime<'i> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        core::fmt::Display::fmt(&self.date, f)?;
        if let Some(ref time) = self.time {
            core::fmt::Display::fmt(&time, f)?;
        }
        if let Some(ref offset) = self.offset {
            core::fmt::Display::fmt(&offset, f)?;
        }
        core::fmt::Display::fmt(&self.annotations, f)
    }
}

/// The result of parsing a Gregorian calendar civil date.
#[derive(Debug)]
pub(super) struct ParsedDate {
    /// The actual parsed date.
    date: Date,
}

impl core::fmt::Display for ParsedDate {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        core::fmt::Display::fmt(&self.date, f)
    }
}

/// The result of parsing a 24-hour civil time.
#[derive(Debug)]
pub(super) struct ParsedTime {
    /// The actual parsed time.
    time: Time,
    /// Whether the time was parsed in extended format or not.
    extended: bool,
}

impl ParsedTime {
    pub(super) fn to_time(&self) -> Time {
        self.time
    }
}

impl core::fmt::Display for ParsedTime {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        core::fmt::Display::fmt(&self.time, f)
    }
}

#[derive(Debug)]
pub(super) struct ParsedTimeZone<'i> {
    /// The original input that the time zone was parsed from.
    input: escape::Bytes<'i>,
    /// The kind of time zone parsed.
    kind: ParsedTimeZoneKind<'i>,
}

impl<'i> core::fmt::Display for ParsedTimeZone<'i> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        core::fmt::Display::fmt(&self.input, f)
    }
}

#[derive(Debug)]
pub(super) enum ParsedTimeZoneKind<'i> {
    Named(&'i str),
    Offset(ParsedOffset),
    #[cfg(feature = "alloc")]
    Posix(crate::tz::posix::PosixTimeZoneOwned),
}

impl<'i> ParsedTimeZone<'i> {
    pub(super) fn into_time_zone(
        self,
        db: &TimeZoneDatabase,
    ) -> Result<TimeZone, Error> {
        match self.kind {
            ParsedTimeZoneKind::Named(iana_name) => {
                db.get(iana_name).context(E::FailedTzdbLookup)
            }
            ParsedTimeZoneKind::Offset(poff) => {
                let offset =
                    poff.to_offset().context(E::FailedOffsetNumeric)?;
                Ok(TimeZone::fixed(offset))
            }
            #[cfg(feature = "alloc")]
            ParsedTimeZoneKind::Posix(posix_tz) => {
                Ok(TimeZone::from_posix_tz(posix_tz))
            }
        }
    }
}

/// A parser for Temporal datetimes.
#[derive(Debug)]
pub(super) struct DateTimeParser {
    /// There are currently no configuration options for this parser.
    _priv: (),
}

impl DateTimeParser {
    /// Create a new Temporal datetime parser with the default configuration.
    pub(super) const fn new() -> DateTimeParser {
        DateTimeParser { _priv: () }
    }

    // TemporalDateTimeString[Zoned] :::
    //   AnnotatedDateTime[?Zoned]
    //
    // AnnotatedDateTime[Zoned] :::
    //   [~Zoned] DateTime TimeZoneAnnotation[opt] Annotations[opt]
    //   [+Zoned] DateTime TimeZoneAnnotation Annotations[opt]
    //
    // DateTime :::
    //   Date
    //   Date DateTimeSeparator TimeSpec DateTimeUTCOffset[opt]
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(super) fn parse_temporal_datetime<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, ParsedDateTime<'i>>, Error> {
        let Parsed { value: date, input } = self.parse_date_spec(input)?;
        let Some((&first, tail)) = input.split_first() else {
            let value = ParsedDateTime {
                date,
                time: None,
                offset: None,
                annotations: ParsedAnnotations::none(),
            };
            return Ok(Parsed { value, input });
        };
        let (time, offset, input) = if !matches!(first, b' ' | b'T' | b't') {
            (None, None, input)
        } else {
            let input = tail;
            // If there's a separator, then we must parse a time and we are
            // *allowed* to parse an offset. But without a separator, we don't
            // support offsets. Just annotations (which are parsed below).
            let Parsed { value: time, input } = self.parse_time_spec(input)?;
            let Parsed { value: offset, input } = self.parse_offset(input)?;
            (Some(time), offset, input)
        };
        let Parsed { value: annotations, input } =
            self.parse_annotations(input)?;
        let value = ParsedDateTime { date, time, offset, annotations };
        Ok(Parsed { value, input })
    }

    // TemporalTimeString :::
    //   AnnotatedTime
    //   AnnotatedDateTimeTimeRequired
    //
    // AnnotatedTime :::
    //   TimeDesignator TimeSpec
    //                  DateTimeUTCOffset[opt]
    //                  TimeZoneAnnotation[opt]
    //                  Annotations[opt]
    //   TimeSpecWithOptionalOffsetNotAmbiguous TimeZoneAnnotation[opt]
    //                                          Annotations[opt]
    //
    // TimeSpecWithOptionalOffsetNotAmbiguous :::
    //   TimeSpec DateTimeUTCOffsetopt (but not one of ValidMonthDay or DateSpecYearMonth)
    //
    // TimeDesignator ::: one of
    //   T t
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(super) fn parse_temporal_time<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, ParsedTime>, Error> {
        let mkslice = parse::slicer(input);

        if let Some(input) =
            input.strip_prefix(b"T").or_else(|| input.strip_prefix(b"t"))
        {
            let Parsed { value: time, input } = self.parse_time_spec(input)?;
            let Parsed { value: offset, input } = self.parse_offset(input)?;
            if offset.map_or(false, |o| o.is_zulu()) {
                return Err(Error::from(E::CivilDateTimeZulu));
            }
            let Parsed { input, .. } = self.parse_annotations(input)?;
            return Ok(Parsed { value: time, input });
        }
        // We now look for a full datetime and extract the time from that.
        // We do this before looking for a non-T time-only component because
        // otherwise things like `2024-06-01T01:02:03` end up having `2024-06`
        // parsed as a `HHMM-OFFSET` time, and then result in an "ambiguous"
        // error.
        //
        // This is largely a result of us trying to parse a time off of the
        // beginning of the input without assuming that the time must consume
        // the entire input.
        if let Ok(parsed) = self.parse_temporal_datetime(input) {
            let Parsed { value: dt, input } = parsed;
            if dt.offset.map_or(false, |o| o.is_zulu()) {
                return Err(Error::from(E::CivilDateTimeZulu));
            }
            let Some(time) = dt.time else {
                return Err(Error::from(E::MissingTimeInDate));
            };
            return Ok(Parsed { value: time, input });
        }

        // At this point, we look for something that is a time that doesn't
        // start with a `T`. We need to check that it isn't ambiguous with a
        // possible date.
        let Parsed { value: time, input } = self.parse_time_spec(input)?;
        let Parsed { value: offset, input } = self.parse_offset(input)?;
        if offset.map_or(false, |o| o.is_zulu()) {
            return Err(Error::from(E::CivilDateTimeZulu));
        }
        // The possible ambiguities occur with the time AND the
        // optional offset, so try to parse what we have so far as
        // either a "month-day" or a "year-month." If either succeeds,
        // then the time is ambiguous and we can report an error.
        //
        // ... but this can only happen when the time was parsed in
        // "basic" mode. i.e., without the `:` separators.
        if !time.extended {
            let possibly_ambiguous = mkslice(input);
            if self.parse_month_day(possibly_ambiguous).is_ok() {
                return Err(Error::from(E::AmbiguousTimeMonthDay));
            }
            if self.parse_year_month(possibly_ambiguous).is_ok() {
                return Err(Error::from(E::AmbiguousTimeYearMonth));
            }
        }
        // OK... carry on.
        let Parsed { input, .. } = self.parse_annotations(input)?;
        Ok(Parsed { value: time, input })
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(super) fn parse_time_zone<'i>(
        &self,
        mut input: &'i [u8],
    ) -> Result<Parsed<'i, ParsedTimeZone<'i>>, Error> {
        let &first = input.first().ok_or(E::EmptyTimeZone)?;
        let original = escape::Bytes(input);
        if matches!(first, b'+' | b'-') {
            static P: offset::Parser = offset::Parser::new()
                .zulu(false)
                .subminute(true)
                .subsecond(false);
            let Parsed { value: offset, input } = P.parse(input)?;
            let kind = ParsedTimeZoneKind::Offset(offset);
            let value = ParsedTimeZone { input: original, kind };
            return Ok(Parsed { value, input });
        }

        // Creates a "named" parsed time zone, generally meant to
        // be an IANA time zone identifier. We do this in a couple
        // different cases below, hence the helper function.
        let mknamed = |consumed, remaining| {
            let tzid = core::str::from_utf8(consumed)
                .map_err(|_| E::InvalidTimeZoneUtf8)?;
            let kind = ParsedTimeZoneKind::Named(tzid);
            let value = ParsedTimeZone { input: original, kind };
            Ok(Parsed { value, input: remaining })
        };
        // This part get tricky. The common case is absolutely an IANA time
        // zone identifier. So we try to parse something that looks like an IANA
        // tz id.
        //
        // In theory, IANA tz ids can never be valid POSIX TZ strings, since
        // POSIX TZ strings minimally require an offset in them (e.g., `EST5`)
        // and IANA tz ids aren't supposed to contain numbers. But there are
        // some legacy IANA tz ids (`EST5EDT`) that do contain numbers.
        //
        // However, the legacy IANA tz ids, like `EST5EDT`, are pretty much
        // nonsense as POSIX TZ strings since there is no DST transition rule.
        // So in cases of nonsense tz ids, we assume they are IANA tz ids.
        let mkconsumed = parse::slicer(input);
        let mut saw_number = false;
        loop {
            let Some((&byte, tail)) = input.split_first() else { break };
            if byte.is_ascii_whitespace() {
                break;
            }
            saw_number = saw_number || byte.is_ascii_digit();
            input = tail;
        }
        let consumed = mkconsumed(input);
        if !saw_number {
            return mknamed(consumed, input);
        }
        #[cfg(not(feature = "alloc"))]
        {
            Err(Error::from(E::AllocPosixTimeZone))
        }
        #[cfg(feature = "alloc")]
        {
            use crate::tz::posix::PosixTimeZone;

            match PosixTimeZone::parse_prefix(consumed) {
                Ok((posix_tz, input)) => {
                    let kind = ParsedTimeZoneKind::Posix(posix_tz);
                    let value = ParsedTimeZone { input: original, kind };
                    Ok(Parsed { value, input })
                }
                // We get here for invalid POSIX tz strings, or even if
                // they are technically valid according to POSIX but not
                // "reasonable", i.e., `EST5EDT`. Which in that case would
                // end up doing an IANA tz lookup. (And it might hit because
                // `EST5EDT` is a legacy IANA tz id. Lol.)
                Err(_) => mknamed(consumed, input),
            }
        }
    }

    /// Parses an ISO 8601 week date.
    ///
    /// Note that this isn't part of the Temporal ISO 8601 spec. We put it here
    /// because it shares a fair bit of code with parsing regular ISO 8601
    /// dates.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(super) fn parse_iso_week_date<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, ISOWeekDate>, Error> {
        // Parse year component.
        let Parsed { value: year, input } =
            self.parse_year(input).context(E::FailedYearInDate)?;
        let extended = input.starts_with(b"-");

        // Parse optional separator.
        let Parsed { input, .. } = self
            .parse_date_separator(input, extended)
            .context(E::FailedSeparatorAfterYear)?;

        // Parse 'W' prefix before week num.
        let Parsed { input, .. } = self
            .parse_week_prefix(input)
            .context(E::FailedWeekNumberPrefixInDate)?;

        // Parse week num component.
        let Parsed { value: week, input } =
            self.parse_week_num(input).context(E::FailedWeekNumberInDate)?;

        // Parse optional separator.
        let Parsed { input, .. } = self
            .parse_date_separator(input, extended)
            .context(E::FailedSeparatorAfterWeekNumber)?;

        // Parse day component.
        let Parsed { value: weekday, input } =
            self.parse_weekday(input).context(E::FailedWeekdayInDate)?;

        let iso_week_date = ISOWeekDate::new(year, week, weekday)
            .context(E::InvalidWeekDate)?;

        Ok(Parsed { value: iso_week_date, input: input })
    }

    // Date :::
    //   DateYear - DateMonth - DateDay
    //   DateYear DateMonth DateDay
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_date_spec<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, ParsedDate>, Error> {
        // Parse year component.
        let Parsed { value: year, input } =
            self.parse_year(input).context(E::FailedYearInDate)?;
        let extended = input.starts_with(b"-");

        // Parse optional separator.
        let Parsed { input, .. } = self
            .parse_date_separator(input, extended)
            .context(E::FailedSeparatorAfterYear)?;

        // Parse month component.
        let Parsed { value: month, input } =
            self.parse_month(input).context(E::FailedMonthInDate)?;

        // Parse optional separator.
        let Parsed { input, .. } = self
            .parse_date_separator(input, extended)
            .context(E::FailedSeparatorAfterMonth)?;

        // Parse day component.
        let Parsed { value: day, input } =
            self.parse_day(input).context(E::FailedDayInDate)?;

        let date = Date::new(year, month, day).context(E::InvalidDate)?;
        let value = ParsedDate { date };
        Ok(Parsed { value, input })
    }

    // TimeSpec :::
    //   TimeHour
    //   TimeHour : TimeMinute
    //   TimeHour TimeMinute
    //   TimeHour : TimeMinute : TimeSecond TimeFraction[opt]
    //   TimeHour TimeMinute TimeSecond TimeFraction[opt]
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_time_spec<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, ParsedTime>, Error> {
        // Parse hour component.
        let Parsed { value: hour, input } =
            self.parse_hour(input).context(E::FailedHourInTime)?;
        let extended = input.starts_with(b":");

        // Parse optional minute component.
        let Parsed { value: has_minute, input } =
            self.parse_time_separator(input, extended);
        if !has_minute {
            // OK because we know `hour` is in bounds and all combinations of
            // `hour` with zeros are valid `Time` values.
            let time = Time::new(hour, 0, 0, 0).unwrap();
            let value = ParsedTime { time, extended };
            return Ok(Parsed { value, input });
        }
        let Parsed { value: minute, input } =
            self.parse_minute(input).context(E::FailedMinuteInTime)?;

        // Parse optional second component.
        let Parsed { value: has_second, input } =
            self.parse_time_separator(input, extended);
        if !has_second {
            // OK because we know `hour` and `minute` are in bounds and all
            // combinations of `hour` and `minute` with zero seconds/nanos are
            // valid `Time` values.
            let time = Time::new(hour, minute, 0, 0).unwrap();
            let value = ParsedTime { time, extended };
            return Ok(Parsed { value, input });
        }
        let Parsed { value: second, input } =
            self.parse_second(input).context(E::FailedSecondInTime)?;

        // Parse an optional fractional component.
        let Parsed { value: nanosecond, input } =
            parse_temporal_fraction(input)
                .context(E::FailedFractionalSecondInTime)?;

        // OK because we know that all our components are in bounds and all
        // combinations of in-bounds components are valid `Time` values.
        let time = Time::new(
            hour,
            minute,
            second,
            // OK because `parse_temporal_fraction` guarantees
            // `0..=999_999_999`.
            nanosecond.map(|n| i32::try_from(n).unwrap()).unwrap_or(0),
        )
        .unwrap();
        let value = ParsedTime { time, extended };
        Ok(Parsed { value, input })
    }

    // ValidMonthDay :::
    //   DateMonth -[opt] 0 NonZeroDigit
    //   DateMonth -[opt] 1 DecimalDigit
    //   DateMonth -[opt] 2 DecimalDigit
    //   DateMonth -[opt] 30 but not one of 0230 or 02-30
    //   DateMonthWithThirtyOneDays -opt 31
    //
    // DateMonthWithThirtyOneDays ::: one of
    //   01 03 05 07 08 10 12
    //
    // NOTE: Jiff doesn't have a "month-day" type, but we still have a parsing
    // function for it so that we can detect ambiguous time strings.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_month_day<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, ()>, Error> {
        // Parse month component.
        let Parsed { value: month, mut input } =
            self.parse_month(input).context(E::FailedMonthInMonthDay)?;

        // Skip over optional separator.
        if let Some(tail) = input.strip_prefix(b"-") {
            input = tail;
        }

        // Parse day component.
        let Parsed { value: day, input } =
            self.parse_day(input).context(E::FailedDayInMonthDay)?;

        // Check that the month-day is valid. Since Temporal's month-day
        // permits 02-29, we use a leap year. The error message here is
        // probably confusing, but these errors should never be exposed to the
        // user.
        let _ = Date::new(2024, month, day).context(E::InvalidMonthDay)?;

        // We have a valid year-month. But we don't return it because we just
        // need to check validity.
        Ok(Parsed { value: (), input })
    }

    // DateSpecYearMonth :::
    //   DateYear -[opt] DateMonth
    //
    // NOTE: Jiff doesn't have a "year-month" type, but we still have a parsing
    // function for it so that we can detect ambiguous time strings.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_year_month<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, ()>, Error> {
        // Parse year component.
        let Parsed { value: year, mut input } =
            self.parse_year(input).context(E::FailedYearInYearMonth)?;

        // Skip over optional separator.
        if let Some(tail) = input.strip_prefix(b"-") {
            input = tail;
        }

        // Parse month component.
        let Parsed { value: month, input } =
            self.parse_month(input).context(E::FailedMonthInYearMonth)?;

        // Check that the year-month is valid. We just use a day of 1, since
        // every month in every year must have a day 1.
        let _ = Date::new(year, month, 1).context(E::InvalidYearMonth)?;

        // We have a valid year-month. But we don't return it because we just
        // need to check validity.
        Ok(Parsed { value: (), input })
    }

    // DateYear :::
    //   DecimalDigit DecimalDigit DecimalDigit DecimalDigit
    //   TemporalSign DecimalDigit DecimalDigit DecimalDigit DecimalDigit DecimalDigit DecimalDigit
    //
    // NOTE: I don't really like the fact that in order to write a negative
    // year, you need to use the six digit variant. Like, why not allow
    // `-0001`? I'm not sure why, so for Chesterton's fence reasons, I'm
    // sticking with the Temporal spec. But I may loosen this in the future. We
    // should be careful not to introduce any possible ambiguities, though, I
    // don't think there are any?
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_year<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, i16>, Error> {
        let Parsed { value: sign, input } = self.parse_year_sign(input);
        if let Some(sign) = sign {
            return self.parse_signed_year(input, sign);
        }

        let (year, input) =
            parse::split(input, 4).ok_or(E::ExpectedFourDigitYear)?;
        let year = b::Year::parse(year).context(E::ParseYearFourDigit)?;
        Ok(Parsed { value: year, input })
    }

    #[cold]
    #[inline(never)]
    fn parse_signed_year<'i>(
        &self,
        input: &'i [u8],
        sign: Sign,
    ) -> Result<Parsed<'i, i16>, Error> {
        let (year, input) =
            parse::split(input, 6).ok_or(E::ExpectedSixDigitYear)?;
        let year = b::Year::parse(year).context(E::ParseYearSixDigit)?;
        if year == 0 && sign.is_negative() {
            return Err(Error::from(E::InvalidYearZero));
        }
        Ok(Parsed { value: sign * year, input })
    }

    // DateMonth :::
    //   0 NonZeroDigit
    //   10
    //   11
    //   12
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_month<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, i8>, Error> {
        let (month, input) =
            parse::split(input, 2).ok_or(E::ExpectedTwoDigitMonth)?;
        let month = b::Month::parse(month).context(E::ParseMonthTwoDigit)?;
        Ok(Parsed { value: month, input })
    }

    // DateDay :::
    //   0 NonZeroDigit
    //   1 DecimalDigit
    //   2 DecimalDigit
    //   30
    //   31
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_day<'i>(&self, input: &'i [u8]) -> Result<Parsed<'i, i8>, Error> {
        let (day, input) =
            parse::split(input, 2).ok_or(E::ExpectedTwoDigitDay)?;
        let day = b::Day::parse(day).context(E::ParseDayTwoDigit)?;
        Ok(Parsed { value: day, input })
    }

    // TimeHour :::
    //   Hour
    //
    // Hour :::
    //   0 DecimalDigit
    //   1 DecimalDigit
    //   20
    //   21
    //   22
    //   23
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_hour<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, i8>, Error> {
        let (hour, input) =
            parse::split(input, 2).ok_or(E::ExpectedTwoDigitHour)?;
        let hour = b::Hour::parse(hour).context(E::ParseHourTwoDigit)?;
        Ok(Parsed { value: hour, input })
    }

    // TimeMinute :::
    //   MinuteSecond
    //
    // MinuteSecond :::
    //   0 DecimalDigit
    //   1 DecimalDigit
    //   2 DecimalDigit
    //   3 DecimalDigit
    //   4 DecimalDigit
    //   5 DecimalDigit
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_minute<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, i8>, Error> {
        let (minute, input) =
            parse::split(input, 2).ok_or(E::ExpectedTwoDigitMinute)?;
        let minute =
            b::Minute::parse(minute).context(E::ParseMinuteTwoDigit)?;
        Ok(Parsed { value: minute, input })
    }

    // TimeSecond :::
    //   MinuteSecond
    //   60
    //
    // MinuteSecond :::
    //   0 DecimalDigit
    //   1 DecimalDigit
    //   2 DecimalDigit
    //   3 DecimalDigit
    //   4 DecimalDigit
    //   5 DecimalDigit
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_second<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, i8>, Error> {
        let (second, input) =
            parse::split(input, 2).ok_or(E::ExpectedTwoDigitSecond)?;
        let mut second =
            b::LeapSecond::parse(second).context(E::ParseSecondTwoDigit)?;
        // NOTE: I believe Temporal allows one to make this configurable. That
        // is, to reject it. But for now, we just always clamp a leap second.
        if second == 60 {
            second = 59;
        }
        Ok(Parsed { value: second, input })
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_offset<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, Option<ParsedOffset>>, Error> {
        const P: offset::Parser =
            offset::Parser::new().zulu(true).subminute(true);
        P.parse_optional(input)
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_annotations<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, ParsedAnnotations<'i>>, Error> {
        const P: rfc9557::Parser = rfc9557::Parser::new();
        if input.first().map_or(true, |&b| b != b'[') {
            let value = ParsedAnnotations::none();
            return Ok(Parsed { input, value });
        }
        P.parse(input)
    }

    /// Parses the separator that is expected to appear between
    /// date components.
    ///
    /// When in extended mode, a `-` is expected. When not in extended mode,
    /// no input is consumed and this routine never fails.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_date_separator<'i>(
        &self,
        input: &'i [u8],
        extended: bool,
    ) -> Result<Parsed<'i, ()>, Error> {
        if !extended {
            // If we see a '-' when not in extended mode, then we can report
            // a better error message than, e.g., "-3 isn't a valid day."
            if input.starts_with(b"-") {
                return Err(Error::from(E::ExpectedNoSeparator));
            }
            return Ok(Parsed { value: (), input });
        }
        let (&first, input) =
            input.split_first().ok_or(E::ExpectedSeparatorFoundEndOfInput)?;
        if first != b'-' {
            return Err(Error::from(E::ExpectedSeparatorFoundByte {
                byte: first,
            }));
        }
        Ok(Parsed { value: (), input })
    }

    /// Parses the separator that is expected to appear between time
    /// components. When `true` is returned, we expect to parse the next
    /// component. When `false` is returned, then no separator was found and
    /// there is no expectation of finding another component.
    ///
    /// When in extended mode, true is returned if and only if a separator is
    /// found.
    ///
    /// When in basic mode (not extended), then a subsequent component is only
    /// expected when `input` begins with two ASCII digits.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_time_separator<'i>(
        &self,
        mut input: &'i [u8],
        extended: bool,
    ) -> Parsed<'i, bool> {
        if !extended {
            let expected = parse::split(input, 2)
                .map_or(false, |(prefix, _)| {
                    prefix.iter().all(u8::is_ascii_digit)
                });
            return Parsed { value: expected, input };
        }
        let mut is_separator = false;
        if let Some(tail) = input.strip_prefix(b":") {
            is_separator = true;
            input = tail;
        }
        Parsed { value: is_separator, input }
    }

    // TemporalSign :::
    //   ASCIISign
    //   <MINUS>
    //
    // ASCIISign ::: one of
    //   + -
    //
    // NOTE: We specifically only support ASCII signs. I think Temporal needs
    // to support `<MINUS>` because of other things in ECMA script that
    // require it?[1]
    //
    // [1]: https://github.com/tc39/proposal-temporal/issues/2843
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_year_sign<'i>(
        &self,
        input: &'i [u8],
    ) -> Parsed<'i, Option<Sign>> {
        let Some((&sign, tail)) = input.split_first() else {
            return Parsed { value: None, input };
        };
        let sign = if sign == b'+' {
            Sign::Positive
        } else if sign == b'-' {
            Sign::Negative
        } else {
            return Parsed { value: None, input };
        };
        Parsed { value: Some(sign), input: tail }
    }

    /// Parses the `W` that is expected to appear before the week component in
    /// an ISO 8601 week date.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_week_prefix<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, ()>, Error> {
        let (&first, input) =
            input.split_first().ok_or(E::ExpectedWeekPrefixFoundEndOfInput)?;
        if !matches!(first, b'W' | b'w') {
            return Err(Error::from(E::ExpectedWeekPrefixFoundByte {
                byte: first,
            }));
        }
        Ok(Parsed { value: (), input })
    }

    /// Parses the week number that follows a `W` in an ISO week date.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_week_num<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, i8>, Error> {
        let (week_num, input) =
            parse::split(input, 2).ok_or(E::ExpectedTwoDigitWeekNumber)?;
        let week_num =
            b::ISOWeek::parse(week_num).context(E::ParseWeekNumberTwoDigit)?;
        Ok(Parsed { value: week_num, input })
    }

    /// Parses the weekday (1-indexed, starting with Monday) in an ISO 8601
    /// week date.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_weekday<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, Weekday>, Error> {
        let (weekday, input) =
            parse::split(input, 1).ok_or(E::ExpectedOneDigitWeekday)?;
        let weekday = b::WeekdayMondayOne::parse(weekday)
            .context(E::ParseWeekdayOneDigit)?;
        // OK because we know `weekday` is in bounds from above.
        let weekday = Weekday::from_monday_one_offset(weekday).unwrap();
        Ok(Parsed { value: weekday, input })
    }
}

/// A parser for Temporal spans.
///
/// Note that in Temporal, a "span" is called a "duration."
#[derive(Debug)]
pub(super) struct SpanParser {
    /// There are currently no configuration options for this parser.
    _priv: (),
}

impl SpanParser {
    /// Create a new Temporal span parser with the default configuration.
    pub(super) const fn new() -> SpanParser {
        SpanParser { _priv: () }
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(super) fn parse_span<I: AsRef<[u8]>>(
        &self,
        input: I,
    ) -> Result<Span, Error> {
        #[inline(never)]
        fn imp(p: &SpanParser, input: &[u8]) -> Result<Span, Error> {
            let mut builder = DurationUnits::default();
            let parsed = p.parse_calendar_and_time(input, &mut builder)?;
            let parsed = parsed.and_then(|_| builder.to_span())?;
            parsed.into_full()
        }
        imp(self, input.as_ref())
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(super) fn parse_signed_duration<I: AsRef<[u8]>>(
        &self,
        input: I,
    ) -> Result<SignedDuration, Error> {
        #[inline(never)]
        fn imp(p: &SpanParser, input: &[u8]) -> Result<SignedDuration, Error> {
            let mut builder = DurationUnits::default();
            let parsed = p.parse_time_only(input, &mut builder)?;
            let parsed = parsed.and_then(|_| builder.to_signed_duration())?;
            parsed.into_full()
        }
        imp(self, input.as_ref())
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(super) fn parse_unsigned_duration<I: AsRef<[u8]>>(
        &self,
        input: I,
    ) -> Result<core::time::Duration, Error> {
        #[inline(never)]
        fn imp(
            p: &SpanParser,
            input: &[u8],
        ) -> Result<core::time::Duration, Error> {
            let mut builder = DurationUnits::default();
            let parsed = p.parse_time_only(input, &mut builder)?;
            let parsed =
                parsed.and_then(|_| builder.to_unsigned_duration())?;
            let d = parsed.value;
            parsed.into_full_with(format_args!("{d:?}"))
        }
        imp(self, input.as_ref())
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_calendar_and_time<'i>(
        &self,
        input: &'i [u8],
        builder: &mut DurationUnits,
    ) -> Result<Parsed<'i, ()>, Error> {
        let (sign, input) =
            if !input.first().map_or(false, |&b| matches!(b, b'+' | b'-')) {
                (Sign::Positive, input)
            } else {
                let Parsed { value: sign, input } = self.parse_sign(input);
                (sign, input)
            };

        let Parsed { input, .. } = self.parse_duration_designator(input)?;
        let Parsed { input, .. } = self.parse_date_units(input, builder)?;
        let Parsed { value: has_time, mut input } =
            self.parse_time_designator(input);
        if has_time {
            let parsed = self.parse_time_units(input, builder)?;
            input = parsed.input;

            if builder.get_min().map_or(true, |min| min > Unit::Hour) {
                return Err(Error::from(E::ExpectedTimeUnits));
            }
        }
        builder.set_sign(sign);
        Ok(Parsed { value: (), input })
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_time_only<'i>(
        &self,
        input: &'i [u8],
        builder: &mut DurationUnits,
    ) -> Result<Parsed<'i, ()>, Error> {
        let (sign, input) =
            if !input.first().map_or(false, |&b| matches!(b, b'+' | b'-')) {
                (Sign::Positive, input)
            } else {
                let Parsed { value: sign, input } = self.parse_sign(input);
                (sign, input)
            };

        let Parsed { input, .. } = self.parse_duration_designator(input)?;
        let Parsed { value: has_time, input } =
            self.parse_time_designator(input);
        if !has_time {
            return Err(Error::from(E::ExpectedTimeDesignator));
        }

        let Parsed { input, .. } = self.parse_time_units(input, builder)?;
        if builder.get_min().map_or(true, |min| min > Unit::Hour) {
            return Err(Error::from(E::ExpectedTimeUnits));
        }
        builder.set_sign(sign);
        Ok(Parsed { value: (), input })
    }

    /// Parses consecutive units from an ISO 8601 duration string into the
    /// `DurationUnits` given.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_date_units<'i>(
        &self,
        mut input: &'i [u8],
        builder: &mut DurationUnits,
    ) -> Result<Parsed<'i, ()>, Error> {
        loop {
            let parsed = self.parse_unit_value(input)?;
            input = parsed.input;
            let Some(value) = parsed.value else { break };

            let parsed = self.parse_unit_date_designator(input)?;
            input = parsed.input;
            let unit = parsed.value;

            builder.set_unit_value(unit, value)?;
        }
        Ok(Parsed { value: (), input })
    }

    /// Parses consecutive time units from an ISO 8601 duration string into the
    /// `DurationUnits` given.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_time_units<'i>(
        &self,
        mut input: &'i [u8],
        builder: &mut DurationUnits,
    ) -> Result<Parsed<'i, ()>, Error> {
        loop {
            let parsed = self.parse_unit_value(input)?;
            input = parsed.input;
            let Some(value) = parsed.value else { break };

            let parsed = parse_temporal_fraction(input)?;
            input = parsed.input;
            let fraction = parsed.value;

            let parsed = self.parse_unit_time_designator(input)?;
            input = parsed.input;
            let unit = parsed.value;

            builder.set_unit_value(unit, value)?;
            if let Some(fraction) = fraction {
                builder.set_fraction(fraction)?;
                // Once we see a fraction, we are done. We don't permit parsing
                // any more units. That is, a fraction can only occur on the
                // lowest unit of time.
                break;
            }
        }
        Ok(Parsed { value: (), input })
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_unit_value<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, Option<u64>>, Error> {
        let (value, input) = parse::u64_prefix(input)?;
        Ok(Parsed { value, input })
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_unit_date_designator<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, Unit>, Error> {
        let (&first, input) = input
            .split_first()
            .ok_or(E::ExpectedDateDesignatorFoundEndOfInput)?;
        let unit = match first {
            b'Y' | b'y' => Unit::Year,
            b'M' | b'm' => Unit::Month,
            b'W' | b'w' => Unit::Week,
            b'D' | b'd' => Unit::Day,
            _ => {
                return Err(Error::from(E::ExpectedDateDesignatorFoundByte {
                    byte: first,
                }));
            }
        };
        Ok(Parsed { value: unit, input })
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_unit_time_designator<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, Unit>, Error> {
        let (&first, input) = input
            .split_first()
            .ok_or(E::ExpectedTimeDesignatorFoundEndOfInput)?;
        let unit = match first {
            b'H' | b'h' => Unit::Hour,
            b'M' | b'm' => Unit::Minute,
            b'S' | b's' => Unit::Second,
            _ => {
                return Err(Error::from(E::ExpectedTimeDesignatorFoundByte {
                    byte: first,
                }));
            }
        };
        Ok(Parsed { value: unit, input })
    }

    // DurationDesignator ::: one of
    //   P p
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_duration_designator<'i>(
        &self,
        input: &'i [u8],
    ) -> Result<Parsed<'i, ()>, Error> {
        let (&first, input) = input
            .split_first()
            .ok_or(E::ExpectedDurationDesignatorFoundEndOfInput)?;
        if !matches!(first, b'P' | b'p') {
            return Err(Error::from(E::ExpectedDurationDesignatorFoundByte {
                byte: first,
            }));
        }
        Ok(Parsed { value: (), input })
    }

    // TimeDesignator ::: one of
    //   T t
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_time_designator<'i>(&self, input: &'i [u8]) -> Parsed<'i, bool> {
        let Some((&first, tail)) = input.split_first() else {
            return Parsed { value: false, input };
        };
        if !matches!(first, b'T' | b't') {
            return Parsed { value: false, input };
        }
        Parsed { value: true, input: tail }
    }

    // TemporalSign :::
    //   ASCIISign
    //   <MINUS>
    //
    // NOTE: Like with other things with signs, we don't support the Unicode
    // <MINUS> sign. Just ASCII.
    #[cold]
    #[inline(never)]
    fn parse_sign<'i>(&self, input: &'i [u8]) -> Parsed<'i, Sign> {
        if let Some(tail) = input.strip_prefix(b"+") {
            Parsed { value: Sign::Positive, input: tail }
        } else if let Some(tail) = input.strip_prefix(b"-") {
            Parsed { value: Sign::Negative, input: tail }
        } else {
            Parsed { value: Sign::Positive, input }
        }
    }
}

#[cfg(feature = "alloc")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ok_signed_duration() {
        let p = |input: &[u8]| {
            SpanParser::new().parse_signed_duration(input).unwrap()
        };

        insta::assert_debug_snapshot!(p(b"PT0s"), @"0s");
        insta::assert_debug_snapshot!(p(b"PT0.000000001s"), @"1ns");
        insta::assert_debug_snapshot!(p(b"PT1s"), @"1s");
        insta::assert_debug_snapshot!(p(b"PT59s"), @"59s");
        insta::assert_debug_snapshot!(p(b"PT60s"), @"60s");
        insta::assert_debug_snapshot!(p(b"PT1m"), @"60s");
        insta::assert_debug_snapshot!(p(b"PT1m0.000000001s"), @"60s 1ns");
        insta::assert_debug_snapshot!(p(b"PT1.25m"), @"75s");
        insta::assert_debug_snapshot!(p(b"PT1h"), @"3600s");
        insta::assert_debug_snapshot!(p(b"PT1h0.000000001s"), @"3600s 1ns");
        insta::assert_debug_snapshot!(p(b"PT1.25h"), @"4500s");

        insta::assert_debug_snapshot!(p(b"-PT2562047788015215h30m8.999999999s"), @"-9223372036854775808s 999999999ns");
        insta::assert_debug_snapshot!(p(b"PT2562047788015215h30m7.999999999s"), @"9223372036854775807s 999999999ns");

        insta::assert_debug_snapshot!(p(b"PT9223372036854775807S"), @"9223372036854775807s");
        insta::assert_debug_snapshot!(p(b"-PT9223372036854775808S"), @"-9223372036854775808s");
    }

    #[test]
    fn err_signed_duration() {
        let p = |input: &[u8]| {
            SpanParser::new().parse_signed_duration(input).unwrap_err()
        };

        insta::assert_snapshot!(
            p(b"P0d"),
            @"parsing ISO 8601 duration in this context requires that the duration contain a time component and no components of days or greater",
        );
        insta::assert_snapshot!(
            p(b"PT0d"),
            @"expected to find time unit designator suffix (`H`, `M` or `S`), but found `d` instead",
        );
        insta::assert_snapshot!(
            p(b"P0dT1s"),
            @"parsing ISO 8601 duration in this context requires that the duration contain a time component and no components of days or greater",
        );

        insta::assert_snapshot!(
            p(b""),
            @"expected to find duration beginning with `P` or `p`, but found end of input",
        );
        insta::assert_snapshot!(
            p(b"P"),
            @"parsing ISO 8601 duration in this context requires that the duration contain a time component and no components of days or greater",
        );
        insta::assert_snapshot!(
            p(b"PT"),
            @"found a time designator (`T` or `t`) in an ISO 8601 duration string, but did not find any time units",
        );
        insta::assert_snapshot!(
            p(b"PTs"),
            @"found a time designator (`T` or `t`) in an ISO 8601 duration string, but did not find any time units",
        );

        insta::assert_snapshot!(
            p(b"PT1s1m"),
            @"found value with unit minute after unit second, but units must be written from largest to smallest (and they can't be repeated)",
        );
        insta::assert_snapshot!(
            p(b"PT1s1h"),
            @"found value with unit hour after unit second, but units must be written from largest to smallest (and they can't be repeated)",
        );
        insta::assert_snapshot!(
            p(b"PT1m1h"),
            @"found value with unit hour after unit minute, but units must be written from largest to smallest (and they can't be repeated)",
        );

        insta::assert_snapshot!(
            p(b"-PT9223372036854775809s"),
            @"value for seconds is too big (or small) to fit into a signed 64-bit integer",
        );
        insta::assert_snapshot!(
            p(b"PT9223372036854775808s"),
            @"value for seconds is too big (or small) to fit into a signed 64-bit integer",
        );

        insta::assert_snapshot!(
            p(b"PT1m9223372036854775807s"),
            @"accumulated duration overflowed when adding value to unit second",
        );
        insta::assert_snapshot!(
            p(b"PT2562047788015215.6h"),
            @"accumulated duration overflowed when adding fractional value to unit hour",
        );
    }

    #[test]
    fn ok_unsigned_duration() {
        let p = |input: &[u8]| {
            SpanParser::new().parse_unsigned_duration(input).unwrap()
        };

        insta::assert_debug_snapshot!(p(b"PT0s"), @"0ns");
        insta::assert_debug_snapshot!(p(b"PT0.000000001s"), @"1ns");
        insta::assert_debug_snapshot!(p(b"PT1s"), @"1s");
        insta::assert_debug_snapshot!(p(b"+PT1s"), @"1s");
        insta::assert_debug_snapshot!(p(b"PT59s"), @"59s");
        insta::assert_debug_snapshot!(p(b"PT60s"), @"60s");
        insta::assert_debug_snapshot!(p(b"PT1m"), @"60s");
        insta::assert_debug_snapshot!(p(b"PT1m0.000000001s"), @"60.000000001s");
        insta::assert_debug_snapshot!(p(b"PT1.25m"), @"75s");
        insta::assert_debug_snapshot!(p(b"PT1h"), @"3600s");
        insta::assert_debug_snapshot!(p(b"PT1h0.000000001s"), @"3600.000000001s");
        insta::assert_debug_snapshot!(p(b"PT1.25h"), @"4500s");

        insta::assert_debug_snapshot!(p(b"PT2562047788015215h30m7.999999999s"), @"9223372036854775807.999999999s");
        insta::assert_debug_snapshot!(p(b"PT5124095576030431H15.999999999S"), @"18446744073709551615.999999999s");

        insta::assert_debug_snapshot!(p(b"PT9223372036854775807S"), @"9223372036854775807s");
        insta::assert_debug_snapshot!(p(b"PT9223372036854775808S"), @"9223372036854775808s");
        insta::assert_debug_snapshot!(p(b"PT18446744073709551615S"), @"18446744073709551615s");
        insta::assert_debug_snapshot!(p(b"PT1M18446744073709551555S"), @"18446744073709551615s");
    }

    #[test]
    fn err_unsigned_duration() {
        #[track_caller]
        fn p(input: &[u8]) -> crate::Error {
            SpanParser::new().parse_unsigned_duration(input).unwrap_err()
        }

        insta::assert_snapshot!(
            p(b"-PT1S"),
            @"cannot parse negative duration into unsigned `std::time::Duration`",
        );
        insta::assert_snapshot!(
            p(b"-PT0S"),
            @"cannot parse negative duration into unsigned `std::time::Duration`",
        );

        insta::assert_snapshot!(
            p(b"P0d"),
            @"parsing ISO 8601 duration in this context requires that the duration contain a time component and no components of days or greater",
        );
        insta::assert_snapshot!(
            p(b"PT0d"),
            @"expected to find time unit designator suffix (`H`, `M` or `S`), but found `d` instead",
        );
        insta::assert_snapshot!(
            p(b"P0dT1s"),
            @"parsing ISO 8601 duration in this context requires that the duration contain a time component and no components of days or greater",
        );

        insta::assert_snapshot!(
            p(b""),
            @"expected to find duration beginning with `P` or `p`, but found end of input",
        );
        insta::assert_snapshot!(
            p(b"P"),
            @"parsing ISO 8601 duration in this context requires that the duration contain a time component and no components of days or greater",
        );
        insta::assert_snapshot!(
            p(b"PT"),
            @"found a time designator (`T` or `t`) in an ISO 8601 duration string, but did not find any time units",
        );
        insta::assert_snapshot!(
            p(b"PTs"),
            @"found a time designator (`T` or `t`) in an ISO 8601 duration string, but did not find any time units",
        );

        insta::assert_snapshot!(
            p(b"PT1s1m"),
            @"found value with unit minute after unit second, but units must be written from largest to smallest (and they can't be repeated)",
        );
        insta::assert_snapshot!(
            p(b"PT1s1h"),
            @"found value with unit hour after unit second, but units must be written from largest to smallest (and they can't be repeated)",
        );
        insta::assert_snapshot!(
            p(b"PT1m1h"),
            @"found value with unit hour after unit minute, but units must be written from largest to smallest (and they can't be repeated)",
        );

        insta::assert_snapshot!(
            p(b"-PT9223372036854775809S"),
            @"cannot parse negative duration into unsigned `std::time::Duration`",
        );
        insta::assert_snapshot!(
            p(b"PT18446744073709551616S"),
            @"number too big to parse into 64-bit integer",
        );

        insta::assert_snapshot!(
            p(b"PT5124095576030431H16.999999999S"),
            @"accumulated duration overflowed when adding value to unit second",
        );
        insta::assert_snapshot!(
            p(b"PT1M18446744073709551556S"),
            @"accumulated duration overflowed when adding value to unit second",
        );
        insta::assert_snapshot!(
            p(b"PT5124095576030431.5H"),
            @"accumulated duration overflowed when adding fractional value to unit hour",
        );
    }

    #[test]
    fn ok_temporal_duration_basic() {
        let p = |input: &[u8]| SpanParser::new().parse_span(input).unwrap();

        insta::assert_debug_snapshot!(p(b"P5d"), @"5d");
        insta::assert_debug_snapshot!(p(b"-P5d"), @"5d ago");
        insta::assert_debug_snapshot!(p(b"+P5d"), @"5d");
        insta::assert_debug_snapshot!(p(b"P5DT1s"), @"5d 1s");
        insta::assert_debug_snapshot!(p(b"PT1S"), @"1s");
        insta::assert_debug_snapshot!(p(b"PT0S"), @"0s");
        insta::assert_debug_snapshot!(p(b"P0Y"), @"0s");
        insta::assert_debug_snapshot!(p(b"P1Y1M1W1DT1H1M1S"), @"1y 1mo 1w 1d 1h 1m 1s");
        insta::assert_debug_snapshot!(p(b"P1y1m1w1dT1h1m1s"), @"1y 1mo 1w 1d 1h 1m 1s");
    }

    #[test]
    fn ok_temporal_duration_fractional() {
        let p = |input: &[u8]| SpanParser::new().parse_span(input).unwrap();

        insta::assert_debug_snapshot!(p(b"PT0.5h"), @"30m");
        insta::assert_debug_snapshot!(p(b"PT0.123456789h"), @"7m 24s 444ms 440µs 400ns");
        insta::assert_debug_snapshot!(p(b"PT1.123456789h"), @"1h 7m 24s 444ms 440µs 400ns");

        insta::assert_debug_snapshot!(p(b"PT0.5m"), @"30s");
        insta::assert_debug_snapshot!(p(b"PT0.123456789m"), @"7s 407ms 407µs 340ns");
        insta::assert_debug_snapshot!(p(b"PT1.123456789m"), @"1m 7s 407ms 407µs 340ns");

        insta::assert_debug_snapshot!(p(b"PT0.5s"), @"500ms");
        insta::assert_debug_snapshot!(p(b"PT0.123456789s"), @"123ms 456µs 789ns");
        insta::assert_debug_snapshot!(p(b"PT1.123456789s"), @"1s 123ms 456µs 789ns");

        // The tests below all have a whole second value that exceeds the
        // maximum allowed seconds in a span. But they should still parse
        // correctly by spilling over into milliseconds, microseconds and
        // nanoseconds.
        insta::assert_debug_snapshot!(p(b"PT1902545624836.854775807s"), @"631107417600s 631107417600000ms 631107417600000000µs 9223372036854775807ns");
        insta::assert_debug_snapshot!(p(b"PT175307616h10518456960m640330789636.854775807s"), @"175307616h 10518456960m 631107417600s 9223372036854ms 775µs 807ns");
        insta::assert_debug_snapshot!(p(b"-PT1902545624836.854775807s"), @"631107417600s 631107417600000ms 631107417600000000µs 9223372036854775807ns ago");
        insta::assert_debug_snapshot!(p(b"-PT175307616h10518456960m640330789636.854775807s"), @"175307616h 10518456960m 631107417600s 9223372036854ms 775µs 807ns ago");
    }

    #[test]
    fn ok_temporal_duration_unbalanced() {
        let p = |input: &[u8]| SpanParser::new().parse_span(input).unwrap();

        insta::assert_debug_snapshot!(
            p(b"PT175307616h10518456960m1774446656760s"), @"175307616h 10518456960m 631107417600s 631107417600000ms 512231821560000000µs");
        insta::assert_debug_snapshot!(
            p(b"Pt843517082H"), @"175307616h 10518456960m 631107417600s 631107417600000ms 512231824800000000µs");
        insta::assert_debug_snapshot!(
            p(b"Pt843517081H"), @"175307616h 10518456960m 631107417600s 631107417600000ms 512231821200000000µs");
    }

    #[test]
    fn ok_temporal_datetime_basic() {
        let p = |input| {
            DateTimeParser::new().parse_temporal_datetime(input).unwrap()
        };

        insta::assert_debug_snapshot!(p(b"2024-06-01"), @r#"
        Parsed {
            value: ParsedDateTime {
                date: ParsedDate {
                    date: 2024-06-01,
                },
                time: None,
                offset: None,
                annotations: ParsedAnnotations {
                    time_zone: None,
                },
            },
            input: "",
        }
        "#);
        insta::assert_debug_snapshot!(p(b"2024-06-01[America/New_York]"), @r#"
        Parsed {
            value: ParsedDateTime {
                date: ParsedDate {
                    date: 2024-06-01,
                },
                time: None,
                offset: None,
                annotations: ParsedAnnotations {
                    time_zone: Some(
                        Named {
                            critical: false,
                            name: "America/New_York",
                        },
                    ),
                },
            },
            input: "",
        }
        "#);
        insta::assert_debug_snapshot!(p(b"2024-06-01T01:02:03"), @r#"
        Parsed {
            value: ParsedDateTime {
                date: ParsedDate {
                    date: 2024-06-01,
                },
                time: Some(
                    ParsedTime {
                        time: 01:02:03,
                        extended: true,
                    },
                ),
                offset: None,
                annotations: ParsedAnnotations {
                    time_zone: None,
                },
            },
            input: "",
        }
        "#);
        insta::assert_debug_snapshot!(p(b"2024-06-01T01:02:03-05"), @r#"
        Parsed {
            value: ParsedDateTime {
                date: ParsedDate {
                    date: 2024-06-01,
                },
                time: Some(
                    ParsedTime {
                        time: 01:02:03,
                        extended: true,
                    },
                ),
                offset: Some(
                    ParsedOffset {
                        kind: Numeric(
                            -05,
                        ),
                    },
                ),
                annotations: ParsedAnnotations {
                    time_zone: None,
                },
            },
            input: "",
        }
        "#);
        insta::assert_debug_snapshot!(p(b"2024-06-01T01:02:03-05[America/New_York]"), @r#"
        Parsed {
            value: ParsedDateTime {
                date: ParsedDate {
                    date: 2024-06-01,
                },
                time: Some(
                    ParsedTime {
                        time: 01:02:03,
                        extended: true,
                    },
                ),
                offset: Some(
                    ParsedOffset {
                        kind: Numeric(
                            -05,
                        ),
                    },
                ),
                annotations: ParsedAnnotations {
                    time_zone: Some(
                        Named {
                            critical: false,
                            name: "America/New_York",
                        },
                    ),
                },
            },
            input: "",
        }
        "#);
        insta::assert_debug_snapshot!(p(b"2024-06-01T01:02:03Z[America/New_York]"), @r#"
        Parsed {
            value: ParsedDateTime {
                date: ParsedDate {
                    date: 2024-06-01,
                },
                time: Some(
                    ParsedTime {
                        time: 01:02:03,
                        extended: true,
                    },
                ),
                offset: Some(
                    ParsedOffset {
                        kind: Zulu,
                    },
                ),
                annotations: ParsedAnnotations {
                    time_zone: Some(
                        Named {
                            critical: false,
                            name: "America/New_York",
                        },
                    ),
                },
            },
            input: "",
        }
        "#);
        insta::assert_debug_snapshot!(p(b"2024-06-01T01:02:03-01[America/New_York]"), @r#"
        Parsed {
            value: ParsedDateTime {
                date: ParsedDate {
                    date: 2024-06-01,
                },
                time: Some(
                    ParsedTime {
                        time: 01:02:03,
                        extended: true,
                    },
                ),
                offset: Some(
                    ParsedOffset {
                        kind: Numeric(
                            -01,
                        ),
                    },
                ),
                annotations: ParsedAnnotations {
                    time_zone: Some(
                        Named {
                            critical: false,
                            name: "America/New_York",
                        },
                    ),
                },
            },
            input: "",
        }
        "#);
    }

    #[test]
    fn ok_temporal_datetime_incomplete() {
        let p = |input| {
            DateTimeParser::new().parse_temporal_datetime(input).unwrap()
        };

        insta::assert_debug_snapshot!(p(b"2024-06-01T01"), @r#"
        Parsed {
            value: ParsedDateTime {
                date: ParsedDate {
                    date: 2024-06-01,
                },
                time: Some(
                    ParsedTime {
                        time: 01:00:00,
                        extended: false,
                    },
                ),
                offset: None,
                annotations: ParsedAnnotations {
                    time_zone: None,
                },
            },
            input: "",
        }
        "#);
        insta::assert_debug_snapshot!(p(b"2024-06-01T0102"), @r#"
        Parsed {
            value: ParsedDateTime {
                date: ParsedDate {
                    date: 2024-06-01,
                },
                time: Some(
                    ParsedTime {
                        time: 01:02:00,
                        extended: false,
                    },
                ),
                offset: None,
                annotations: ParsedAnnotations {
                    time_zone: None,
                },
            },
            input: "",
        }
        "#);
        insta::assert_debug_snapshot!(p(b"2024-06-01T01:02"), @r#"
        Parsed {
            value: ParsedDateTime {
                date: ParsedDate {
                    date: 2024-06-01,
                },
                time: Some(
                    ParsedTime {
                        time: 01:02:00,
                        extended: true,
                    },
                ),
                offset: None,
                annotations: ParsedAnnotations {
                    time_zone: None,
                },
            },
            input: "",
        }
        "#);
    }

    #[test]
    fn ok_temporal_datetime_separator() {
        let p = |input| {
            DateTimeParser::new().parse_temporal_datetime(input).unwrap()
        };

        insta::assert_debug_snapshot!(p(b"2024-06-01t01:02:03"), @r#"
        Parsed {
            value: ParsedDateTime {
                date: ParsedDate {
                    date: 2024-06-01,
                },
                time: Some(
                    ParsedTime {
                        time: 01:02:03,
                        extended: true,
                    },
                ),
                offset: None,
                annotations: ParsedAnnotations {
                    time_zone: None,
                },
            },
            input: "",
        }
        "#);
        insta::assert_debug_snapshot!(p(b"2024-06-01 01:02:03"), @r#"
        Parsed {
            value: ParsedDateTime {
                date: ParsedDate {
                    date: 2024-06-01,
                },
                time: Some(
                    ParsedTime {
                        time: 01:02:03,
                        extended: true,
                    },
                ),
                offset: None,
                annotations: ParsedAnnotations {
                    time_zone: None,
                },
            },
            input: "",
        }
        "#);
    }

    #[test]
    fn ok_temporal_time_basic() {
        let p =
            |input| DateTimeParser::new().parse_temporal_time(input).unwrap();

        insta::assert_debug_snapshot!(p(b"01:02:03"), @r#"
        Parsed {
            value: ParsedTime {
                time: 01:02:03,
                extended: true,
            },
            input: "",
        }
        "#);
        insta::assert_debug_snapshot!(p(b"130113"), @r#"
        Parsed {
            value: ParsedTime {
                time: 13:01:13,
                extended: false,
            },
            input: "",
        }
        "#);
        insta::assert_debug_snapshot!(p(b"T01:02:03"), @r#"
        Parsed {
            value: ParsedTime {
                time: 01:02:03,
                extended: true,
            },
            input: "",
        }
        "#);
        insta::assert_debug_snapshot!(p(b"T010203"), @r#"
        Parsed {
            value: ParsedTime {
                time: 01:02:03,
                extended: false,
            },
            input: "",
        }
        "#);
    }

    #[test]
    fn ok_temporal_time_from_full_datetime() {
        let p =
            |input| DateTimeParser::new().parse_temporal_time(input).unwrap();

        insta::assert_debug_snapshot!(p(b"2024-06-01T01:02:03"), @r#"
        Parsed {
            value: ParsedTime {
                time: 01:02:03,
                extended: true,
            },
            input: "",
        }
        "#);
        insta::assert_debug_snapshot!(p(b"2024-06-01T01:02:03.123"), @r#"
        Parsed {
            value: ParsedTime {
                time: 01:02:03.123,
                extended: true,
            },
            input: "",
        }
        "#);
        insta::assert_debug_snapshot!(p(b"2024-06-01T01"), @r#"
        Parsed {
            value: ParsedTime {
                time: 01:00:00,
                extended: false,
            },
            input: "",
        }
        "#);
        insta::assert_debug_snapshot!(p(b"2024-06-01T0102"), @r#"
        Parsed {
            value: ParsedTime {
                time: 01:02:00,
                extended: false,
            },
            input: "",
        }
        "#);
        insta::assert_debug_snapshot!(p(b"2024-06-01T010203"), @r#"
        Parsed {
            value: ParsedTime {
                time: 01:02:03,
                extended: false,
            },
            input: "",
        }
        "#);
        insta::assert_debug_snapshot!(p(b"2024-06-01T010203-05"), @r#"
        Parsed {
            value: ParsedTime {
                time: 01:02:03,
                extended: false,
            },
            input: "",
        }
        "#);
        insta::assert_debug_snapshot!(
            p(b"2024-06-01T010203-05[America/New_York]"), @r#"
        Parsed {
            value: ParsedTime {
                time: 01:02:03,
                extended: false,
            },
            input: "",
        }
        "#);
        insta::assert_debug_snapshot!(
            p(b"2024-06-01T010203[America/New_York]"), @r#"
        Parsed {
            value: ParsedTime {
                time: 01:02:03,
                extended: false,
            },
            input: "",
        }
        "#);
    }

    #[test]
    fn err_temporal_time_ambiguous() {
        let p = |input| {
            DateTimeParser::new().parse_temporal_time(input).unwrap_err()
        };

        insta::assert_snapshot!(
            p(b"010203"),
            @"parsed time is ambiguous with a month-day date",
        );
        insta::assert_snapshot!(
            p(b"130112"),
            @"parsed time is ambiguous with a year-month date",
        );
    }

    #[test]
    fn err_temporal_time_missing_time() {
        let p = |input| {
            DateTimeParser::new().parse_temporal_time(input).unwrap_err()
        };

        insta::assert_snapshot!(
            p(b"2024-06-01[America/New_York]"),
            @"successfully parsed date, but no time component was found",
        );
        // 2099 is not a valid time, but 2099-12-01 is a valid date, so this
        // carves a path where a full datetime parse is OK, but a basic
        // time-only parse is not.
        insta::assert_snapshot!(
            p(b"2099-12-01[America/New_York]"),
            @"successfully parsed date, but no time component was found",
        );
        // Like above, but this time we use an invalid date. As a result, we
        // get an error reported not on the invalid date, but on how it is an
        // invalid time. (Because we're asking for a time here.)
        insta::assert_snapshot!(
            p(b"2099-13-01[America/New_York]"),
            @"failed to parse minute in time: failed to parse two digit integer as minute: parameter 'minute' is not in the required range of 0..=59",
        );
    }

    #[test]
    fn err_temporal_time_zulu() {
        let p = |input| {
            DateTimeParser::new().parse_temporal_time(input).unwrap_err()
        };

        insta::assert_snapshot!(
            p(b"T00:00:00Z"),
            @"cannot parse civil date/time from string with a Zulu offset, parse as a `jiff::Timestamp` first and convert to a civil date/time instead",
        );
        insta::assert_snapshot!(
            p(b"00:00:00Z"),
            @"cannot parse civil date/time from string with a Zulu offset, parse as a `jiff::Timestamp` first and convert to a civil date/time instead",
        );
        insta::assert_snapshot!(
            p(b"000000Z"),
            @"cannot parse civil date/time from string with a Zulu offset, parse as a `jiff::Timestamp` first and convert to a civil date/time instead",
        );
        insta::assert_snapshot!(
            p(b"2099-12-01T00:00:00Z"),
            @"cannot parse civil date/time from string with a Zulu offset, parse as a `jiff::Timestamp` first and convert to a civil date/time instead",
        );
    }

    #[test]
    fn ok_date_basic() {
        let p = |input| DateTimeParser::new().parse_date_spec(input).unwrap();

        insta::assert_debug_snapshot!(p(b"2010-03-14"), @r#"
        Parsed {
            value: ParsedDate {
                date: 2010-03-14,
            },
            input: "",
        }
        "#);
        insta::assert_debug_snapshot!(p(b"20100314"), @r#"
        Parsed {
            value: ParsedDate {
                date: 2010-03-14,
            },
            input: "",
        }
        "#);
        insta::assert_debug_snapshot!(p(b"2010-03-14T01:02:03"), @r#"
        Parsed {
            value: ParsedDate {
                date: 2010-03-14,
            },
            input: "T01:02:03",
        }
        "#);
        insta::assert_debug_snapshot!(p(b"-009999-03-14"), @r#"
        Parsed {
            value: ParsedDate {
                date: -009999-03-14,
            },
            input: "",
        }
        "#);
        insta::assert_debug_snapshot!(p(b"+009999-03-14"), @r#"
        Parsed {
            value: ParsedDate {
                date: 9999-03-14,
            },
            input: "",
        }
        "#);
    }

    #[test]
    fn err_date_empty() {
        insta::assert_snapshot!(
            DateTimeParser::new().parse_date_spec(b"").unwrap_err(),
            @"failed to parse year in date: expected four digit year (or leading sign for six digit year), but found end of input",
        );
    }

    #[test]
    fn err_date_year() {
        insta::assert_snapshot!(
            DateTimeParser::new().parse_date_spec(b"123").unwrap_err(),
            @"failed to parse year in date: expected four digit year (or leading sign for six digit year), but found end of input",
        );
        insta::assert_snapshot!(
            DateTimeParser::new().parse_date_spec(b"123a").unwrap_err(),
            @"failed to parse year in date: failed to parse four digit integer as year: invalid digit, expected 0-9 but got a",
        );

        insta::assert_snapshot!(
            DateTimeParser::new().parse_date_spec(b"-9999").unwrap_err(),
            @"failed to parse year in date: expected six digit year (because of a leading sign), but found end of input",
        );
        insta::assert_snapshot!(
            DateTimeParser::new().parse_date_spec(b"+9999").unwrap_err(),
            @"failed to parse year in date: expected six digit year (because of a leading sign), but found end of input",
        );
        insta::assert_snapshot!(
            DateTimeParser::new().parse_date_spec(b"-99999").unwrap_err(),
            @"failed to parse year in date: expected six digit year (because of a leading sign), but found end of input",
        );
        insta::assert_snapshot!(
            DateTimeParser::new().parse_date_spec(b"+99999").unwrap_err(),
            @"failed to parse year in date: expected six digit year (because of a leading sign), but found end of input",
        );
        insta::assert_snapshot!(
            DateTimeParser::new().parse_date_spec(b"-99999a").unwrap_err(),
            @"failed to parse year in date: failed to parse six digit integer as year: invalid digit, expected 0-9 but got a",
        );
        insta::assert_snapshot!(
            DateTimeParser::new().parse_date_spec(b"+999999").unwrap_err(),
            @"failed to parse year in date: failed to parse six digit integer as year: parameter 'year' is not in the required range of -9999..=9999",
        );
        insta::assert_snapshot!(
            DateTimeParser::new().parse_date_spec(b"-010000").unwrap_err(),
            @"failed to parse year in date: failed to parse six digit integer as year: parameter 'year' is not in the required range of -9999..=9999",
        );
    }

    #[test]
    fn err_date_month() {
        insta::assert_snapshot!(
            DateTimeParser::new().parse_date_spec(b"2024-").unwrap_err(),
            @"failed to parse month in date: expected two digit month, but found end of input",
        );
        insta::assert_snapshot!(
            DateTimeParser::new().parse_date_spec(b"2024").unwrap_err(),
            @"failed to parse month in date: expected two digit month, but found end of input",
        );
        insta::assert_snapshot!(
            DateTimeParser::new().parse_date_spec(b"2024-13-01").unwrap_err(),
            @"failed to parse month in date: failed to parse two digit integer as month: parameter 'month' is not in the required range of 1..=12",
        );
        insta::assert_snapshot!(
            DateTimeParser::new().parse_date_spec(b"20241301").unwrap_err(),
            @"failed to parse month in date: failed to parse two digit integer as month: parameter 'month' is not in the required range of 1..=12",
        );
    }

    #[test]
    fn err_date_day() {
        insta::assert_snapshot!(
            DateTimeParser::new().parse_date_spec(b"2024-12-").unwrap_err(),
            @"failed to parse day in date: expected two digit day, but found end of input",
        );
        insta::assert_snapshot!(
            DateTimeParser::new().parse_date_spec(b"202412").unwrap_err(),
            @"failed to parse day in date: expected two digit day, but found end of input",
        );
        insta::assert_snapshot!(
            DateTimeParser::new().parse_date_spec(b"2024-12-40").unwrap_err(),
            @"failed to parse day in date: failed to parse two digit integer as day: parameter 'day' is not in the required range of 1..=31",
        );
        insta::assert_snapshot!(
            DateTimeParser::new().parse_date_spec(b"2024-11-31").unwrap_err(),
            @"parsed date is not valid: parameter 'day' for `2024-11` is invalid, must be in range `1..=30`",
        );
        insta::assert_snapshot!(
            DateTimeParser::new().parse_date_spec(b"2024-02-30").unwrap_err(),
            @"parsed date is not valid: parameter 'day' for `2024-02` is invalid, must be in range `1..=29`",
        );
        insta::assert_snapshot!(
            DateTimeParser::new().parse_date_spec(b"2023-02-29").unwrap_err(),
            @"parsed date is not valid: parameter 'day' for `2023-02` is invalid, must be in range `1..=28`",
        );
    }

    #[test]
    fn err_date_separator() {
        insta::assert_snapshot!(
            DateTimeParser::new().parse_date_spec(b"2024-1231").unwrap_err(),
            @"failed to parse separator after month: expected `-` separator, but found `3`",
        );
        insta::assert_snapshot!(
            DateTimeParser::new().parse_date_spec(b"202412-31").unwrap_err(),
            @"failed to parse separator after month: expected no separator since none was found after the year, but found a `-` separator",
        );
    }

    #[test]
    fn ok_time_basic() {
        let p = |input| DateTimeParser::new().parse_time_spec(input).unwrap();

        insta::assert_debug_snapshot!(p(b"01:02:03"), @r#"
        Parsed {
            value: ParsedTime {
                time: 01:02:03,
                extended: true,
            },
            input: "",
        }
        "#);
        insta::assert_debug_snapshot!(p(b"010203"), @r#"
        Parsed {
            value: ParsedTime {
                time: 01:02:03,
                extended: false,
            },
            input: "",
        }
        "#);
    }

    #[test]
    fn ok_time_fractional() {
        let p = |input| DateTimeParser::new().parse_time_spec(input).unwrap();

        insta::assert_debug_snapshot!(p(b"01:02:03.123456789"), @r#"
        Parsed {
            value: ParsedTime {
                time: 01:02:03.123456789,
                extended: true,
            },
            input: "",
        }
        "#);
        insta::assert_debug_snapshot!(p(b"010203.123456789"), @r#"
        Parsed {
            value: ParsedTime {
                time: 01:02:03.123456789,
                extended: false,
            },
            input: "",
        }
        "#);

        insta::assert_debug_snapshot!(p(b"01:02:03.9"), @r#"
        Parsed {
            value: ParsedTime {
                time: 01:02:03.9,
                extended: true,
            },
            input: "",
        }
        "#);
    }

    #[test]
    fn ok_time_no_fractional() {
        let p = |input| DateTimeParser::new().parse_time_spec(input).unwrap();

        insta::assert_debug_snapshot!(p(b"01:02.123456789"), @r#"
        Parsed {
            value: ParsedTime {
                time: 01:02:00,
                extended: true,
            },
            input: ".123456789",
        }
        "#);
    }

    #[test]
    fn ok_time_leap() {
        let p = |input| DateTimeParser::new().parse_time_spec(input).unwrap();

        insta::assert_debug_snapshot!(p(b"01:02:60"), @r#"
        Parsed {
            value: ParsedTime {
                time: 01:02:59,
                extended: true,
            },
            input: "",
        }
        "#);
    }

    #[test]
    fn ok_time_mixed_format() {
        let p = |input| DateTimeParser::new().parse_time_spec(input).unwrap();

        insta::assert_debug_snapshot!(p(b"01:0203"), @r#"
        Parsed {
            value: ParsedTime {
                time: 01:02:00,
                extended: true,
            },
            input: "03",
        }
        "#);
        insta::assert_debug_snapshot!(p(b"0102:03"), @r#"
        Parsed {
            value: ParsedTime {
                time: 01:02:00,
                extended: false,
            },
            input: ":03",
        }
        "#);
    }

    #[test]
    fn err_time_empty() {
        insta::assert_snapshot!(
            DateTimeParser::new().parse_time_spec(b"").unwrap_err(),
            @"failed to parse hour in time: expected two digit hour, but found end of input",
        );
    }

    #[test]
    fn err_time_hour() {
        insta::assert_snapshot!(
            DateTimeParser::new().parse_time_spec(b"a").unwrap_err(),
            @"failed to parse hour in time: expected two digit hour, but found end of input",
        );
        insta::assert_snapshot!(
            DateTimeParser::new().parse_time_spec(b"1a").unwrap_err(),
            @"failed to parse hour in time: failed to parse two digit integer as hour: invalid digit, expected 0-9 but got a",
        );
        insta::assert_snapshot!(
            DateTimeParser::new().parse_time_spec(b"24").unwrap_err(),
            @"failed to parse hour in time: failed to parse two digit integer as hour: parameter 'hour' is not in the required range of 0..=23",
        );
    }

    #[test]
    fn err_time_minute() {
        insta::assert_snapshot!(
            DateTimeParser::new().parse_time_spec(b"01:").unwrap_err(),
            @"failed to parse minute in time: expected two digit minute, but found end of input",
        );
        insta::assert_snapshot!(
            DateTimeParser::new().parse_time_spec(b"01:a").unwrap_err(),
            @"failed to parse minute in time: expected two digit minute, but found end of input",
        );
        insta::assert_snapshot!(
            DateTimeParser::new().parse_time_spec(b"01:1a").unwrap_err(),
            @"failed to parse minute in time: failed to parse two digit integer as minute: invalid digit, expected 0-9 but got a",
        );
        insta::assert_snapshot!(
            DateTimeParser::new().parse_time_spec(b"01:60").unwrap_err(),
            @"failed to parse minute in time: failed to parse two digit integer as minute: parameter 'minute' is not in the required range of 0..=59",
        );
    }

    #[test]
    fn err_time_second() {
        insta::assert_snapshot!(
            DateTimeParser::new().parse_time_spec(b"01:02:").unwrap_err(),
            @"failed to parse second in time: expected two digit second, but found end of input",
        );
        insta::assert_snapshot!(
            DateTimeParser::new().parse_time_spec(b"01:02:a").unwrap_err(),
            @"failed to parse second in time: expected two digit second, but found end of input",
        );
        insta::assert_snapshot!(
            DateTimeParser::new().parse_time_spec(b"01:02:1a").unwrap_err(),
            @"failed to parse second in time: failed to parse two digit integer as second: invalid digit, expected 0-9 but got a",
        );
        insta::assert_snapshot!(
            DateTimeParser::new().parse_time_spec(b"01:02:61").unwrap_err(),
            @"failed to parse second in time: failed to parse two digit integer as second: parameter 'second' is not in the required range of 0..=60",
        );
    }

    #[test]
    fn err_time_fractional() {
        insta::assert_snapshot!(
            DateTimeParser::new().parse_time_spec(b"01:02:03.").unwrap_err(),
            @"failed to parse fractional seconds in time: found decimal after seconds component, but did not find any digits after decimal",
        );
        insta::assert_snapshot!(
            DateTimeParser::new().parse_time_spec(b"01:02:03.a").unwrap_err(),
            @"failed to parse fractional seconds in time: found decimal after seconds component, but did not find any digits after decimal",
        );
    }

    #[test]
    fn ok_iso_week_date_parse_basic() {
        fn p(input: &str) -> Parsed<'_, ISOWeekDate> {
            DateTimeParser::new()
                .parse_iso_week_date(input.as_bytes())
                .unwrap()
        }

        insta::assert_debug_snapshot!( p("2024-W01-5"), @r#"
        Parsed {
            value: ISOWeekDate {
                year: 2024,
                week: 1,
                weekday: Friday,
            },
            input: "",
        }
        "#);
        insta::assert_debug_snapshot!( p("2024-W52-7"), @r#"
        Parsed {
            value: ISOWeekDate {
                year: 2024,
                week: 52,
                weekday: Sunday,
            },
            input: "",
        }
        "#);
        insta::assert_debug_snapshot!( p("2004-W53-6"), @r#"
        Parsed {
            value: ISOWeekDate {
                year: 2004,
                week: 53,
                weekday: Saturday,
            },
            input: "",
        }
        "#);
        insta::assert_debug_snapshot!( p("2009-W01-1"), @r#"
        Parsed {
            value: ISOWeekDate {
                year: 2009,
                week: 1,
                weekday: Monday,
            },
            input: "",
        }
        "#);

        insta::assert_debug_snapshot!( p("2024W015"), @r#"
        Parsed {
            value: ISOWeekDate {
                year: 2024,
                week: 1,
                weekday: Friday,
            },
            input: "",
        }
        "#);
        insta::assert_debug_snapshot!( p("2024W527"), @r#"
        Parsed {
            value: ISOWeekDate {
                year: 2024,
                week: 52,
                weekday: Sunday,
            },
            input: "",
        }
        "#);
        insta::assert_debug_snapshot!( p("2004W536"), @r#"
        Parsed {
            value: ISOWeekDate {
                year: 2004,
                week: 53,
                weekday: Saturday,
            },
            input: "",
        }
        "#);
        insta::assert_debug_snapshot!( p("2009W011"), @r#"
        Parsed {
            value: ISOWeekDate {
                year: 2009,
                week: 1,
                weekday: Monday,
            },
            input: "",
        }
        "#);

        // Lowercase should be okay. This matches how
        // we support `T` or `t`.
        insta::assert_debug_snapshot!( p("2009w011"), @r#"
        Parsed {
            value: ISOWeekDate {
                year: 2009,
                week: 1,
                weekday: Monday,
            },
            input: "",
        }
        "#);
    }

    #[test]
    fn err_iso_week_date_year() {
        let p = |input: &str| {
            DateTimeParser::new()
                .parse_iso_week_date(input.as_bytes())
                .unwrap_err()
        };

        insta::assert_snapshot!(
            p("123"),
            @"failed to parse year in date: expected four digit year (or leading sign for six digit year), but found end of input",
        );
        insta::assert_snapshot!(
            p("123a"),
            @"failed to parse year in date: failed to parse four digit integer as year: invalid digit, expected 0-9 but got a",
        );

        insta::assert_snapshot!(
            p("-9999"),
            @"failed to parse year in date: expected six digit year (because of a leading sign), but found end of input",
        );
        insta::assert_snapshot!(
            p("+9999"),
            @"failed to parse year in date: expected six digit year (because of a leading sign), but found end of input",
        );
        insta::assert_snapshot!(
            p("-99999"),
            @"failed to parse year in date: expected six digit year (because of a leading sign), but found end of input",
        );
        insta::assert_snapshot!(
            p("+99999"),
            @"failed to parse year in date: expected six digit year (because of a leading sign), but found end of input",
        );
        insta::assert_snapshot!(
            p("-99999a"),
            @"failed to parse year in date: failed to parse six digit integer as year: invalid digit, expected 0-9 but got a",
        );
        insta::assert_snapshot!(
            p("+999999"),
            @"failed to parse year in date: failed to parse six digit integer as year: parameter 'year' is not in the required range of -9999..=9999",
        );
        insta::assert_snapshot!(
            p("-010000"),
            @"failed to parse year in date: failed to parse six digit integer as year: parameter 'year' is not in the required range of -9999..=9999",
        );
    }

    #[test]
    fn err_iso_week_date_week_prefix() {
        let p = |input: &str| {
            DateTimeParser::new()
                .parse_iso_week_date(input.as_bytes())
                .unwrap_err()
        };

        insta::assert_snapshot!(
            p("2024-"),
            @"failed to parse week number prefix in date: expected `W` or `w`, but found end of input",
        );
        insta::assert_snapshot!(
            p("2024"),
            @"failed to parse week number prefix in date: expected `W` or `w`, but found end of input",
        );
    }

    #[test]
    fn err_iso_week_date_week_number() {
        let p = |input: &str| {
            DateTimeParser::new()
                .parse_iso_week_date(input.as_bytes())
                .unwrap_err()
        };

        insta::assert_snapshot!(
            p("2024-W"),
            @"failed to parse week number in date: expected two digit week number, but found end of input",
        );
        insta::assert_snapshot!(
            p("2024-W1"),
            @"failed to parse week number in date: expected two digit week number, but found end of input",
        );
        insta::assert_snapshot!(
            p("2024-W53-1"),
            @"parsed week date is not valid: parameter 'iso-week' is not in the required range of 1..=53",
        );
        insta::assert_snapshot!(
            p("2030W531"),
            @"parsed week date is not valid: parameter 'iso-week' is not in the required range of 1..=53",
        );
    }

    #[test]
    fn err_iso_week_date_parse_incomplete() {
        let p = |input: &str| {
            DateTimeParser::new()
                .parse_iso_week_date(input.as_bytes())
                .unwrap_err()
        };

        insta::assert_snapshot!(
            p("2024-W53-1"),
            @"parsed week date is not valid: parameter 'iso-week' is not in the required range of 1..=53",
        );
        insta::assert_snapshot!(
            p("2025-W53-1"),
            @"parsed week date is not valid: parameter 'iso-week' is not in the required range of 1..=53",
        );
    }

    #[test]
    fn err_iso_week_date_date_day() {
        let p = |input: &str| {
            DateTimeParser::new()
                .parse_iso_week_date(input.as_bytes())
                .unwrap_err()
        };
        insta::assert_snapshot!(
            p("2024-W12-"),
            @"failed to parse weekday in date: expected one digit weekday, but found end of input",
        );
        insta::assert_snapshot!(
            p("2024W12"),
            @"failed to parse weekday in date: expected one digit weekday, but found end of input",
        );
        insta::assert_snapshot!(
            p("2024-W11-8"),
            @"failed to parse weekday in date: failed to parse one digit integer as weekday: parameter 'weekday (Monday 1-indexed)' is not in the required range of 1..=7",
        );
    }

    #[test]
    fn err_iso_week_date_date_separator() {
        let p = |input: &str| {
            DateTimeParser::new()
                .parse_iso_week_date(input.as_bytes())
                .unwrap_err()
        };
        insta::assert_snapshot!(
            p("2024-W521"),
            @"failed to parse separator after week number: expected `-` separator, but found `1`",
        );
        insta::assert_snapshot!(
            p("2024W01-5"),
            @"failed to parse separator after week number: expected no separator since none was found after the year, but found a `-` separator",
        );
    }
}
