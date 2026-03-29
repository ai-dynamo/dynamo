use core::fmt::Debug;

use super::{
    util::{
        array_str::Abbreviation,
        itime::{
            IAmbiguousOffset, IDate, IDateTime, IOffset, ITime, ITimeSecond,
            ITimestamp, IWeekday,
        },
    },
    PosixDay, PosixDayTime, PosixDst, PosixOffset, PosixRule, PosixTime,
    PosixTimeZone,
};

impl PosixTimeZone<Abbreviation> {
    /// Parse a POSIX `TZ` environment variable, assuming it's a rule and not
    /// an implementation defined value, from the given bytes.
    pub fn parse(
        bytes: &[u8],
    ) -> Result<PosixTimeZone<Abbreviation>, PosixTimeZoneError> {
        // We enable the IANA v3+ extensions here. (Namely, that the time
        // specification hour value has the range `-167..=167` instead of
        // `0..=24`.) Requiring strict POSIX rules doesn't seem necessary
        // since the extension is a strict superset. Plus, GNU tooling
        // seems to accept the extension.
        let parser = Parser { ianav3plus: true, ..Parser::new(bytes) };
        parser.parse()
    }

    // only-jiff-start
    /// Like parse, but parses a prefix of the input given and returns whatever
    /// is remaining.
    pub fn parse_prefix<'b>(
        bytes: &'b [u8],
    ) -> Result<(PosixTimeZone<Abbreviation>, &'b [u8]), PosixTimeZoneError>
    {
        let parser = Parser { ianav3plus: true, ..Parser::new(bytes) };
        parser.parse_prefix()
    }
    // only-jiff-end
}

impl<ABBREV: AsRef<str> + Debug> PosixTimeZone<ABBREV> {
    /// Returns the appropriate time zone offset to use for the given
    /// timestamp.
    ///
    /// If you need information like whether the offset is in DST or not, or
    /// the time zone abbreviation, then use `PosixTimeZone::to_offset_info`.
    /// But that API may be more expensive to use, so only use it if you need
    /// the additional data.
    pub(crate) fn to_offset(&self, timestamp: ITimestamp) -> IOffset {
        let std_offset = self.std_offset.to_ioffset();
        if self.dst.is_none() {
            return std_offset;
        }

        let dt = timestamp.to_datetime(IOffset::UTC);
        self.dst_info_utc(dt.date.year)
            .filter(|dst_info| dst_info.in_dst(dt))
            .map(|dst_info| dst_info.offset().to_ioffset())
            .unwrap_or_else(|| std_offset)
    }

    /// Returns the appropriate time zone offset to use for the given
    /// timestamp.
    ///
    /// This also includes whether the offset returned should be considered
    /// to be "DST" or not, along with the time zone abbreviation (e.g., EST
    /// for standard time in New York, and EDT for DST in New York).
    pub(crate) fn to_offset_info(
        &self,
        timestamp: ITimestamp,
    ) -> (IOffset, &'_ str, bool) {
        let std_offset = self.std_offset.to_ioffset();
        if self.dst.is_none() {
            return (std_offset, self.std_abbrev.as_ref(), false);
        }

        let dt = timestamp.to_datetime(IOffset::UTC);
        self.dst_info_utc(dt.date.year)
            .filter(|dst_info| dst_info.in_dst(dt))
            .map(|dst_info| {
                (
                    dst_info.offset().to_ioffset(),
                    dst_info.dst.abbrev.as_ref(),
                    true,
                )
            })
            .unwrap_or_else(|| (std_offset, self.std_abbrev.as_ref(), false))
    }

    /// Returns a possibly ambiguous timestamp for the given civil datetime.
    ///
    /// The given datetime should correspond to the "wall" clock time of what
    /// humans use to tell time for this time zone.
    ///
    /// Note that "ambiguous timestamp" is represented by the possible
    /// selection of offsets that could be applied to the given datetime. In
    /// general, it is only ambiguous around transitions to-and-from DST. The
    /// ambiguity can arise as a "fold" (when a particular wall clock time is
    /// repeated) or as a "gap" (when a particular wall clock time is skipped
    /// entirely).
    pub(crate) fn to_ambiguous_kind(&self, dt: IDateTime) -> IAmbiguousOffset {
        let year = dt.date.year;
        let std_offset = self.std_offset.to_ioffset();
        let Some(dst_info) = self.dst_info_wall(year) else {
            return IAmbiguousOffset::Unambiguous { offset: std_offset };
        };
        let dst_offset = dst_info.offset().to_ioffset();
        let diff = dst_offset.second - std_offset.second;
        // When the difference between DST and standard is positive, that means
        // STD->DST results in a gap while DST->STD results in a fold. However,
        // when the difference is negative, that means STD->DST results in a
        // fold while DST->STD results in a gap. The former is by far the most
        // common. The latter is a bit weird, but real cases do exist. For
        // example, Dublin has DST in winter (UTC+01) and STD in the summer
        // (UTC+00).
        //
        // When the difference is zero, then we have a weird POSIX time zone
        // where a DST transition rule was specified, but was set to explicitly
        // be the same as STD. In this case, there can be no ambiguity. (The
        // zero case is strictly redundant. Both the diff < 0 and diff > 0
        // cases handle the zero case correctly. But we write it out for
        // clarity.)
        if diff == 0 {
            debug_assert_eq!(std_offset, dst_offset);
            IAmbiguousOffset::Unambiguous { offset: std_offset }
        } else if diff.is_negative() {
            // For DST transitions that always move behind one hour, ambiguous
            // timestamps only occur when the given civil datetime falls in the
            // standard time range.
            if dst_info.in_dst(dt) {
                IAmbiguousOffset::Unambiguous { offset: dst_offset }
            } else {
                let fold_start = dst_info.start.saturating_add_seconds(diff);
                let gap_end =
                    dst_info.end.saturating_add_seconds(diff.saturating_neg());
                if fold_start <= dt && dt < dst_info.start {
                    IAmbiguousOffset::Fold {
                        before: std_offset,
                        after: dst_offset,
                    }
                } else if dst_info.end <= dt && dt < gap_end {
                    IAmbiguousOffset::Gap {
                        before: dst_offset,
                        after: std_offset,
                    }
                } else {
                    IAmbiguousOffset::Unambiguous { offset: std_offset }
                }
            }
        } else {
            // For DST transitions that always move ahead one hour, ambiguous
            // timestamps only occur when the given civil datetime falls in the
            // DST range.
            if !dst_info.in_dst(dt) {
                IAmbiguousOffset::Unambiguous { offset: std_offset }
            } else {
                // PERF: I wonder if it makes sense to pre-compute these?
                // Probably not, because we have to do it based on year of
                // datetime given. But if we ever add a "caching" layer for
                // POSIX time zones, then it might be worth adding these to it.
                let gap_end = dst_info.start.saturating_add_seconds(diff);
                let fold_start =
                    dst_info.end.saturating_add_seconds(diff.saturating_neg());
                if dst_info.start <= dt && dt < gap_end {
                    IAmbiguousOffset::Gap {
                        before: std_offset,
                        after: dst_offset,
                    }
                } else if fold_start <= dt && dt < dst_info.end {
                    IAmbiguousOffset::Fold {
                        before: dst_offset,
                        after: std_offset,
                    }
                } else {
                    IAmbiguousOffset::Unambiguous { offset: dst_offset }
                }
            }
        }
    }

    /// Returns the timestamp of the most recent time zone transition prior
    /// to the timestamp given. If one doesn't exist, `None` is returned.
    pub(crate) fn previous_transition(
        &self,
        timestamp: ITimestamp,
    ) -> Option<(ITimestamp, IOffset, &'_ str, bool)> {
        let dt = timestamp.to_datetime(IOffset::UTC);
        let dst_info = self.dst_info_utc(dt.date.year)?;
        let (earlier, later) = dst_info.ordered();
        let (prev, dst_info) = if dt > later {
            (later, dst_info)
        } else if dt > earlier {
            (earlier, dst_info)
        } else {
            let prev_year = dt.date.prev_year().ok()?;
            let dst_info = self.dst_info_utc(prev_year)?;
            let (_, later) = dst_info.ordered();
            (later, dst_info)
        };

        let timestamp = prev.to_timestamp_checked(IOffset::UTC)?;
        let dt = timestamp.to_datetime(IOffset::UTC);
        let (offset, abbrev, dst) = if dst_info.in_dst(dt) {
            (dst_info.offset(), dst_info.dst.abbrev.as_ref(), true)
        } else {
            (&self.std_offset, self.std_abbrev.as_ref(), false)
        };
        Some((timestamp, offset.to_ioffset(), abbrev, dst))
    }

    /// Returns the timestamp of the soonest time zone transition after the
    /// timestamp given. If one doesn't exist, `None` is returned.
    pub(crate) fn next_transition(
        &self,
        timestamp: ITimestamp,
    ) -> Option<(ITimestamp, IOffset, &'_ str, bool)> {
        let dt = timestamp.to_datetime(IOffset::UTC);
        let dst_info = self.dst_info_utc(dt.date.year)?;
        let (earlier, later) = dst_info.ordered();
        let (next, dst_info) = if dt < earlier {
            (earlier, dst_info)
        } else if dt < later {
            (later, dst_info)
        } else {
            let next_year = dt.date.next_year().ok()?;
            let dst_info = self.dst_info_utc(next_year)?;
            let (earlier, _) = dst_info.ordered();
            (earlier, dst_info)
        };

        let timestamp = next.to_timestamp_checked(IOffset::UTC)?;
        let dt = timestamp.to_datetime(IOffset::UTC);
        let (offset, abbrev, dst) = if dst_info.in_dst(dt) {
            (dst_info.offset(), dst_info.dst.abbrev.as_ref(), true)
        } else {
            (&self.std_offset, self.std_abbrev.as_ref(), false)
        };
        Some((timestamp, offset.to_ioffset(), abbrev, dst))
    }

    /// Returns the range in which DST occurs.
    ///
    /// The civil datetimes returned are in UTC. This is useful for determining
    /// whether a timestamp is in DST or not.
    fn dst_info_utc(&self, year: i16) -> Option<DstInfo<'_, ABBREV>> {
        let dst = self.dst.as_ref()?;
        // DST time starts with respect to standard time, so offset it by the
        // standard offset.
        let start =
            dst.rule.start.to_datetime(year, self.std_offset.to_ioffset());
        // DST time ends with respect to DST time, so offset it by the DST
        // offset.
        let mut end = dst.rule.end.to_datetime(year, dst.offset.to_ioffset());
        // This is a whacky special case when DST is permanent, but the math
        // using to calculate the start/end datetimes ends up leaving a gap
        // for standard time to appear. In which case, it's possible for a
        // timestamp at the end of a calendar year to get standard time when
        // it really should be DST.
        //
        // We detect this case by re-interpreting the end of the boundary using
        // the standard offset. If we get a datetime that is in a different
        // year, then it follows that standard time is actually impossible to
        // occur.
        //
        // These weird POSIX time zones can occur as the TZ strings in
        // a TZif file compiled using rearguard semantics. For example,
        // `Africa/Casablanca` has:
        //
        //     XXX-2<+01>-1,0/0,J365/23
        //
        // Notice here that DST is actually one hour *behind* (it is usually
        // one hour *ahead*) _and_ it ends at 23:00:00 on the last day of the
        // year. But if it ends at 23:00, then jumping to standard time moves
        // the clocks *forward*. Which would bring us to 00:00:00 on the first
        // of the next year... but that is when DST begins! Hence, DST is
        // permanent.
        //
        // Ideally, this could just be handled by our math automatically. But
        // I couldn't figure out how to make it work. In particular, in the
        // above example for year 2087, we get
        //
        //     start == 2087-01-01T00:00:00Z
        //     end == 2087-12-31T22:00:00Z
        //
        // Which leaves a two hour gap for a timestamp to get erroneously
        // categorized as standard time.
        //
        // ... so we special case this. We could pre-compute whether a POSIX
        // time zone is in permanent DST at construction time, but it's not
        // obvious to me that it's worth it. Especially since this is an
        // exceptionally rare case.
        //
        // Note that I did try to consult tzcode's (incredibly inscrutable)
        // `localtime` implementation to figure out how they deal with it. At
        // first, it looks like they don't have any special handling for this
        // case. But looking more closely, they skip any time zone transitions
        // generated by POSIX time zones whose rule spans more than 1 year:
        //
        //     https://github.com/eggert/tz/blob/8d65db9786753f3b263087e31c59d191561d63e3/localtime.c#L1717-L1735
        //
        // By just ignoring them, I think it achieves the desired effect of
        // permanent DST. But I'm not 100% confident in my understanding of
        // the code.
        if start.date.month == 1
            && start.date.day == 1
            && start.time == ITime::MIN
            // NOTE: This should come last because it is potentially expensive.
            && year
                != end.saturating_add_seconds(self.std_offset.second).date.year
        {
            end = IDateTime {
                date: IDate { year, month: 12, day: 31 },
                time: ITime::MAX,
            };
        }
        Some(DstInfo { dst, start, end })
    }

    /// Returns the range in which DST occurs.
    ///
    /// The civil datetimes returned are in "wall clock time." That is, they
    /// represent the transitions as they are seen from humans reading a clock
    /// within the geographic location of that time zone.
    fn dst_info_wall(&self, year: i16) -> Option<DstInfo<'_, ABBREV>> {
        let dst = self.dst.as_ref()?;
        // POSIX time zones express their DST transitions in terms of wall
        // clock time. Since this method specifically is returning wall
        // clock times, we don't want to offset our datetimes at all.
        let start = dst.rule.start.to_datetime(year, IOffset::UTC);
        let end = dst.rule.end.to_datetime(year, IOffset::UTC);
        Some(DstInfo { dst, start, end })
    }

    /// Returns the DST transition rule. This panics if this time zone doesn't
    /// have DST.
    #[cfg(test)]
    fn rule(&self) -> &PosixRule {
        &self.dst.as_ref().unwrap().rule
    }
}

impl<ABBREV: AsRef<str>> core::fmt::Display for PosixTimeZone<ABBREV> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        core::fmt::Display::fmt(
            &AbbreviationDisplay(self.std_abbrev.as_ref()),
            f,
        )?;
        core::fmt::Display::fmt(&self.std_offset, f)?;
        if let Some(ref dst) = self.dst {
            dst.display(&self.std_offset, f)?;
        }
        Ok(())
    }
}

impl<ABBREV: AsRef<str>> PosixDst<ABBREV> {
    fn display(
        &self,
        std_offset: &PosixOffset,
        f: &mut core::fmt::Formatter,
    ) -> core::fmt::Result {
        core::fmt::Display::fmt(
            &AbbreviationDisplay(self.abbrev.as_ref()),
            f,
        )?;
        // The overwhelming common case is that DST is exactly one hour ahead
        // of standard time. So common that this is the default. So don't write
        // the offset if we don't need to.
        let default = PosixOffset { second: std_offset.second + 3600 };
        if self.offset != default {
            core::fmt::Display::fmt(&self.offset, f)?;
        }
        f.write_str(",")?;
        core::fmt::Display::fmt(&self.rule, f)?;
        Ok(())
    }
}

impl core::fmt::Display for PosixRule {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        core::fmt::Display::fmt(&self.start, f)?;
        f.write_str(",")?;
        core::fmt::Display::fmt(&self.end, f)?;
        Ok(())
    }
}

impl PosixDayTime {
    /// Turns this POSIX datetime spec into a civil datetime in the year given
    /// with the given offset. The datetimes returned are offset by the given
    /// offset. For wall clock time, an offset of `0` should be given. For
    /// UTC time, the offset (standard or DST) corresponding to this time
    /// spec should be given.
    ///
    /// The datetime returned is guaranteed to have a year component equal
    /// to the year given. This guarantee is upheld even when the datetime
    /// specification (combined with the offset) would extend past the end of
    /// the year (or before the start of the year). In this case, the maximal
    /// (or minimal) datetime for the given year is returned.
    pub(crate) fn to_datetime(&self, year: i16, offset: IOffset) -> IDateTime {
        let mkmin = || IDateTime {
            date: IDate { year, month: 1, day: 1 },
            time: ITime::MIN,
        };
        let mkmax = || IDateTime {
            date: IDate { year, month: 12, day: 31 },
            time: ITime::MAX,
        };
        let Some(date) = self.date.to_date(year) else { return mkmax() };
        // The range on `self.time` is `-604799..=604799`, and the range
        // on `offset.second` is `-93599..=93599`. Therefore, subtracting
        // them can never overflow an `i32`.
        let offset = self.time.second - offset.second;
        // If the time goes negative or above 86400, then we might have
        // to adjust our date.
        let days = offset.div_euclid(86400);
        let second = offset.rem_euclid(86400);

        let Ok(date) = date.checked_add_days(days) else {
            return if offset < 0 { mkmin() } else { mkmax() };
        };
        if date.year < year {
            mkmin()
        } else if date.year > year {
            mkmax()
        } else {
            let time = ITimeSecond { second }.to_time();
            IDateTime { date, time }
        }
    }
}

impl core::fmt::Display for PosixDayTime {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        core::fmt::Display::fmt(&self.date, f)?;
        // This is the default time, so don't write it if we
        // don't need to.
        if self.time != PosixTime::DEFAULT {
            f.write_str("/")?;
            core::fmt::Display::fmt(&self.time, f)?;
        }
        Ok(())
    }
}

impl PosixDay {
    /// Convert this date specification to a civil date in the year given.
    ///
    /// If this date specification couldn't be turned into a date in the year
    /// given, then `None` is returned. This happens when `366` is given as
    /// a day, but the year given is not a leap year. In this case, callers may
    /// want to assume a datetime that is maximal for the year given.
    fn to_date(&self, year: i16) -> Option<IDate> {
        match *self {
            PosixDay::JulianOne(day) => {
                // Parsing validates that our day is 1-365 which will always
                // succeed for all possible year values. That is, every valid
                // year has a December 31.
                Some(
                    IDate::from_day_of_year_no_leap(year, day)
                        .expect("Julian `J day` should be in bounds"),
                )
            }
            PosixDay::JulianZero(day) => {
                // OK because our value for `day` is validated to be `0..=365`,
                // and since it is an `i16`, it is always valid to add 1.
                //
                // Also, while `day+1` is guaranteed to be in `1..=366`, it is
                // possible that `366` is invalid, for when `year` is not a
                // leap year. In this case, we throw our hands up, and ask the
                // caller to make a decision for how to deal with it. Why does
                // POSIX go out of its way to specifically not specify behavior
                // in error cases?
                IDate::from_day_of_year(year, day + 1).ok()
            }
            PosixDay::WeekdayOfMonth { month, week, weekday } => {
                let weekday = IWeekday::from_sunday_zero_offset(weekday);
                let first = IDate { year, month, day: 1 };
                let week = if week == 5 { -1 } else { week };
                debug_assert!(week == -1 || (1..=4).contains(&week));
                // This is maybe non-obvious, but this will always succeed
                // because it can only fail when the week number is one of
                // {-5, 0, 5}. Since we've validated that 'week' is in 1..=5,
                // we know it can't be 0. Moreover, because of the conditional
                // above and since `5` actually means "last weekday of month,"
                // that case will always translate to `-1`.
                //
                // Also, I looked at how other libraries deal with this case,
                // and almost all of them just do a bunch of inline hairy
                // arithmetic here. I suppose I could be reduced to such
                // things if perf called for it, but we have a nice civil date
                // abstraction. So use it, god damn it. (Well, we did, and now
                // we have a lower level IDate abstraction. But it's still
                // an abstraction!)
                Some(
                    first
                        .nth_weekday_of_month(week, weekday)
                        .expect("nth weekday always exists"),
                )
            }
        }
    }
}

impl core::fmt::Display for PosixDay {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        match *self {
            PosixDay::JulianOne(n) => {
                f.write_str("J")?;
                core::fmt::Display::fmt(&n, f)
            }
            PosixDay::JulianZero(n) => core::fmt::Display::fmt(&n, f),
            PosixDay::WeekdayOfMonth { month, week, weekday } => {
                f.write_str("M")?;
                core::fmt::Display::fmt(&month, f)?;
                f.write_str(".")?;
                core::fmt::Display::fmt(&week, f)?;
                f.write_str(".")?;
                core::fmt::Display::fmt(&weekday, f)?;
                Ok(())
            }
        }
    }
}

impl PosixTime {
    const DEFAULT: PosixTime = PosixTime { second: 2 * 60 * 60 };
}

impl core::fmt::Display for PosixTime {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        if self.second.is_negative() {
            f.write_str("-")?;
            // The default is positive, so when
            // positive, we write nothing.
        }
        let second = self.second.unsigned_abs();
        let h = second / 3600;
        let m = (second / 60) % 60;
        let s = second % 60;
        core::fmt::Display::fmt(&h, f)?;
        if m != 0 || s != 0 {
            write!(f, ":{m:02}")?;
            if s != 0 {
                write!(f, ":{s:02}")?;
            }
        }
        Ok(())
    }
}

impl PosixOffset {
    fn to_ioffset(&self) -> IOffset {
        IOffset { second: self.second }
    }
}

impl core::fmt::Display for PosixOffset {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        // Yes, this is backwards. Blame POSIX.
        // N.B. `+` is the default, so we don't
        // need to write that out.
        if self.second > 0 {
            f.write_str("-")?;
        }
        let second = self.second.unsigned_abs();
        let h = second / 3600;
        let m = (second / 60) % 60;
        let s = second % 60;
        core::fmt::Display::fmt(&h, f)?;
        if m != 0 || s != 0 {
            write!(f, ":{m:02}")?;
            if s != 0 {
                write!(f, ":{s:02}")?;
            }
        }
        Ok(())
    }
}

/// A helper type for formatting a time zone abbreviation.
///
/// Basically, this will write the `<` and `>` quotes if necessary, and
/// otherwise write out the abbreviation in its unquoted form.
#[derive(Debug)]
struct AbbreviationDisplay<S>(S);

impl<S: AsRef<str>> core::fmt::Display for AbbreviationDisplay<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        let s = self.0.as_ref();
        if s.chars().any(|ch| ch == '+' || ch == '-') {
            f.write_str("<")?;
            core::fmt::Display::fmt(&s, f)?;
            f.write_str(">")
        } else {
            core::fmt::Display::fmt(&s, f)
        }
    }
}

/// The daylight saving time (DST) info for a POSIX time zone in a particular
/// year.
#[derive(Debug, Eq, PartialEq)]
struct DstInfo<'a, ABBREV> {
    /// The DST transition rule that generated this info.
    dst: &'a PosixDst<ABBREV>,
    /// The start time (inclusive) that DST begins.
    ///
    /// Note that this may be greater than `end`. This tends to happen in the
    /// southern hemisphere.
    ///
    /// Note also that this may be in UTC or in wall clock civil
    /// time. It depends on whether `PosixTimeZone::dst_info_utc` or
    /// `PosixTimeZone::dst_info_wall` was used.
    start: IDateTime,
    /// The end time (exclusive) that DST ends.
    ///
    /// Note that this may be less than `start`. This tends to happen in the
    /// southern hemisphere.
    ///
    /// Note also that this may be in UTC or in wall clock civil
    /// time. It depends on whether `PosixTimeZone::dst_info_utc` or
    /// `PosixTimeZone::dst_info_wall` was used.
    end: IDateTime,
}

impl<'a, ABBREV> DstInfo<'a, ABBREV> {
    /// Returns true if and only if the given civil datetime ought to be
    /// considered in DST.
    fn in_dst(&self, utc_dt: IDateTime) -> bool {
        if self.start <= self.end {
            self.start <= utc_dt && utc_dt < self.end
        } else {
            !(self.end <= utc_dt && utc_dt < self.start)
        }
    }

    /// Returns the earlier and later times for this DST info.
    fn ordered(&self) -> (IDateTime, IDateTime) {
        if self.start <= self.end {
            (self.start, self.end)
        } else {
            (self.end, self.start)
        }
    }

    /// Returns the DST offset.
    fn offset(&self) -> &PosixOffset {
        &self.dst.offset
    }
}

/// A parser for POSIX time zones.
#[derive(Debug)]
struct Parser<'s> {
    /// The `TZ` string that we're parsing.
    tz: &'s [u8],
    /// The parser's current position in `tz`.
    pos: core::cell::Cell<usize>,
    /// Whether to use IANA rules, i.e., when parsing a TZ string in a TZif
    /// file of version 3 or greater. From `tzfile(5)`:
    ///
    /// > First, the hours part of its transition times may be signed and range
    /// > from `-167` through `167` instead of the POSIX-required unsigned
    /// > values from `0` through `24`. Second, DST is in effect all year if
    /// > it starts January 1 at 00:00 and ends December 31 at 24:00 plus the
    /// > difference between daylight saving and standard time.
    ///
    /// At time of writing, I don't think I understand the significance of
    /// the second part above. (RFC 8536 elaborates that it is meant to be an
    /// explicit clarification of something that POSIX itself implies.) But the
    /// first part is clear: it permits the hours to be a bigger range.
    ianav3plus: bool,
}

impl<'s> Parser<'s> {
    /// Create a new parser for extracting a POSIX time zone from the given
    /// bytes.
    fn new<B: ?Sized + AsRef<[u8]>>(tz: &'s B) -> Parser<'s> {
        Parser {
            tz: tz.as_ref(),
            pos: core::cell::Cell::new(0),
            ianav3plus: false,
        }
    }

    /// Parses a POSIX time zone from the current position of the parser and
    /// ensures that the entire TZ string corresponds to a single valid POSIX
    /// time zone.
    fn parse(
        &self,
    ) -> Result<PosixTimeZone<Abbreviation>, PosixTimeZoneError> {
        let (time_zone, remaining) = self.parse_prefix()?;
        if !remaining.is_empty() {
            return Err(ErrorKind::FoundRemaining.into());
        }
        Ok(time_zone)
    }

    /// Parses a POSIX time zone from the current position of the parser and
    /// returns the remaining input.
    fn parse_prefix(
        &self,
    ) -> Result<(PosixTimeZone<Abbreviation>, &'s [u8]), PosixTimeZoneError>
    {
        let time_zone = self.parse_posix_time_zone()?;
        Ok((time_zone, self.remaining()))
    }

    /// Parse a POSIX time zone from the current position of the parser.
    ///
    /// Upon success, the parser will be positioned immediately following the
    /// TZ string.
    fn parse_posix_time_zone(
        &self,
    ) -> Result<PosixTimeZone<Abbreviation>, PosixTimeZoneError> {
        if self.is_done() {
            return Err(ErrorKind::Empty.into());
        }
        let std_abbrev =
            self.parse_abbreviation().map_err(ErrorKind::AbbreviationStd)?;
        let std_offset =
            self.parse_posix_offset().map_err(ErrorKind::OffsetStd)?;
        let mut dst = None;
        if !self.is_done()
            && (self.byte().is_ascii_alphabetic() || self.byte() == b'<')
        {
            dst = Some(self.parse_posix_dst(&std_offset)?);
        }
        Ok(PosixTimeZone { std_abbrev, std_offset, dst })
    }

    /// Parse a DST zone with an optional explicit transition rule.
    ///
    /// This assumes the parser is positioned at the first byte of the DST
    /// abbreviation.
    ///
    /// Upon success, the parser will be positioned immediately after the end
    /// of the DST transition rule (which might just be the abbreviation, but
    /// might also include explicit start/end datetime specifications).
    fn parse_posix_dst(
        &self,
        std_offset: &PosixOffset,
    ) -> Result<PosixDst<Abbreviation>, PosixTimeZoneError> {
        let abbrev =
            self.parse_abbreviation().map_err(ErrorKind::AbbreviationDst)?;
        if self.is_done() {
            return Err(ErrorKind::FoundDstNoRule.into());
        }
        // This is the default: one hour ahead of standard time. We may
        // override this if the DST portion specifies an offset. (But it
        // usually doesn't.)
        let mut offset = PosixOffset { second: std_offset.second + 3600 };
        if self.byte() != b',' {
            offset =
                self.parse_posix_offset().map_err(ErrorKind::OffsetDst)?;
            if self.is_done() {
                return Err(ErrorKind::FoundDstNoRuleWithOffset.into());
            }
        }
        if self.byte() != b',' {
            return Err(ErrorKind::ExpectedCommaAfterDst.into());
        }
        if !self.bump() {
            return Err(ErrorKind::FoundEndAfterComma.into());
        }
        let rule = self.parse_rule().map_err(ErrorKind::Rule)?;
        Ok(PosixDst { abbrev, offset, rule })
    }

    /// Parse a time zone abbreviation.
    ///
    /// This assumes the parser is positioned at the first byte of
    /// the abbreviation. This is either the first character in the
    /// abbreviation, or the opening quote of a quoted abbreviation.
    ///
    /// Upon success, the parser will be positioned immediately following
    /// the abbreviation name.
    ///
    /// The string returned is guaranteed to be no more than 30 bytes.
    /// (This restriction is somewhat arbitrary, but it's so we can put
    /// the abbreviation in a fixed capacity array.)
    fn parse_abbreviation(&self) -> Result<Abbreviation, AbbreviationError> {
        if self.byte() == b'<' {
            if !self.bump() {
                return Err(AbbreviationError::Quoted(
                    QuotedAbbreviationError::UnexpectedEndAfterOpening,
                ));
            }
            self.parse_quoted_abbreviation().map_err(AbbreviationError::Quoted)
        } else {
            self.parse_unquoted_abbreviation()
                .map_err(AbbreviationError::Unquoted)
        }
    }

    /// Parses an unquoted time zone abbreviation.
    ///
    /// This assumes the parser is position at the first byte in the
    /// abbreviation.
    ///
    /// Upon success, the parser will be positioned immediately after the
    /// last byte in the abbreviation.
    ///
    /// The string returned is guaranteed to be no more than 30 bytes.
    /// (This restriction is somewhat arbitrary, but it's so we can put
    /// the abbreviation in a fixed capacity array.)
    fn parse_unquoted_abbreviation(
        &self,
    ) -> Result<Abbreviation, UnquotedAbbreviationError> {
        let start = self.pos();
        for i in 0.. {
            if !self.byte().is_ascii_alphabetic() {
                break;
            }
            if i >= Abbreviation::capacity() {
                return Err(UnquotedAbbreviationError::TooLong);
            }
            if !self.bump() {
                break;
            }
        }
        let end = self.pos();
        let abbrev =
            core::str::from_utf8(&self.tz[start..end]).map_err(|_| {
                // NOTE: I believe this error is technically impossible
                // since the loop above restricts letters in an
                // abbreviation to ASCII. So everything from `start` to
                // `end` is ASCII and thus should be UTF-8. But it doesn't
                // cost us anything to report an error here in case the
                // code above evolves somehow.
                UnquotedAbbreviationError::InvalidUtf8
            })?;
        if abbrev.len() < 3 {
            return Err(UnquotedAbbreviationError::TooShort);
        }
        // OK because we verified above that the abbreviation
        // does not exceed `Abbreviation::capacity`.
        Ok(Abbreviation::new(abbrev).unwrap())
    }

    /// Parses a quoted time zone abbreviation.
    ///
    /// This assumes the parser is positioned immediately after the opening
    /// `<` quote. That is, at the first byte in the abbreviation.
    ///
    /// Upon success, the parser will be positioned immediately after the
    /// closing `>` quote.
    ///
    /// The string returned is guaranteed to be no more than 30 bytes.
    /// (This restriction is somewhat arbitrary, but it's so we can put
    /// the abbreviation in a fixed capacity array.)
    fn parse_quoted_abbreviation(
        &self,
    ) -> Result<Abbreviation, QuotedAbbreviationError> {
        let start = self.pos();
        for i in 0.. {
            if !self.byte().is_ascii_alphanumeric()
                && self.byte() != b'+'
                && self.byte() != b'-'
            {
                break;
            }
            if i >= Abbreviation::capacity() {
                return Err(QuotedAbbreviationError::TooLong);
            }
            if !self.bump() {
                break;
            }
        }
        let end = self.pos();
        let abbrev =
            core::str::from_utf8(&self.tz[start..end]).map_err(|_| {
                // NOTE: I believe this error is technically impossible
                // since the loop above restricts letters in an
                // abbreviation to ASCII. So everything from `start` to
                // `end` is ASCII and thus should be UTF-8. But it doesn't
                // cost us anything to report an error here in case the
                // code above evolves somehow.
                QuotedAbbreviationError::InvalidUtf8
            })?;
        if self.is_done() {
            return Err(QuotedAbbreviationError::UnexpectedEnd);
        }
        if self.byte() != b'>' {
            return Err(QuotedAbbreviationError::UnexpectedLastByte);
        }
        self.bump();
        if abbrev.len() < 3 {
            return Err(QuotedAbbreviationError::TooShort);
        }
        // OK because we verified above that the abbreviation
        // does not exceed `Abbreviation::capacity`.
        Ok(Abbreviation::new(abbrev).unwrap())
    }

    /// Parse a POSIX time offset.
    ///
    /// This assumes the parser is positioned at the first byte of the
    /// offset. This can either be a digit (for a positive offset) or the
    /// sign of the offset (which must be either `-` or `+`).
    ///
    /// Upon success, the parser will be positioned immediately after the
    /// end of the offset.
    fn parse_posix_offset(&self) -> Result<PosixOffset, PosixOffsetError> {
        let sign = self.parse_optional_sign()?.unwrap_or(1);
        let hour = self.parse_hour_posix()?;
        let (mut minute, mut second) = (0, 0);
        if self.maybe_byte() == Some(b':') {
            if !self.bump() {
                return Err(PosixOffsetError::IncompleteMinutes);
            }
            minute = self.parse_minute()?;
            if self.maybe_byte() == Some(b':') {
                if !self.bump() {
                    return Err(PosixOffsetError::IncompleteSeconds);
                }
                second = self.parse_second()?;
            }
        }
        let mut offset = PosixOffset { second: i32::from(hour) * 3600 };
        offset.second += i32::from(minute) * 60;
        offset.second += i32::from(second);
        // Yes, we flip the sign, because POSIX is backwards.
        // For example, `EST5` corresponds to `-05:00`.
        offset.second *= i32::from(-sign);
        // Must be true because the parsing routines for hours, minutes
        // and seconds enforce they are in the ranges -24..=24, 0..=59
        // and 0..=59, respectively.
        assert!(
            -89999 <= offset.second && offset.second <= 89999,
            "POSIX offset seconds {} is out of range",
            offset.second
        );
        Ok(offset)
    }

    /// Parses a POSIX DST transition rule.
    ///
    /// This assumes the parser is positioned at the first byte in the
    /// rule. That is, it comes immediately after the DST abbreviation or
    /// its optional offset.
    ///
    /// Upon success, the parser will be positioned immediately after the
    /// DST transition rule. In typical cases, this corresponds to the end
    /// of the TZ string.
    fn parse_rule(&self) -> Result<PosixRule, PosixRuleError> {
        let start = self
            .parse_posix_datetime()
            .map_err(PosixRuleError::DateTimeStart)?;
        if self.maybe_byte() != Some(b',') || !self.bump() {
            return Err(PosixRuleError::ExpectedEnd);
        }
        let end = self
            .parse_posix_datetime()
            .map_err(PosixRuleError::DateTimeEnd)?;
        Ok(PosixRule { start, end })
    }

    /// Parses a POSIX datetime specification.
    ///
    /// This assumes the parser is position at the first byte where a
    /// datetime specification is expected to occur.
    ///
    /// Upon success, the parser will be positioned after the datetime
    /// specification. This will either be immediately after the date, or
    /// if it's present, the time part of the specification.
    fn parse_posix_datetime(
        &self,
    ) -> Result<PosixDayTime, PosixDateTimeError> {
        let mut daytime = PosixDayTime {
            date: self.parse_posix_date()?,
            time: PosixTime::DEFAULT,
        };
        if self.maybe_byte() != Some(b'/') {
            return Ok(daytime);
        }
        if !self.bump() {
            return Err(PosixDateTimeError::ExpectedTime);
        }
        daytime.time = self.parse_posix_time()?;
        Ok(daytime)
    }

    /// Parses a POSIX date specification.
    ///
    /// This assumes the parser is positioned at the first byte of the date
    /// specification. This can be `J` (for one based Julian day without
    /// leap days), `M` (for "weekday of month") or a digit starting the
    /// zero based Julian day with leap days. This routine will validate
    /// that the position points to one of these possible values. That is,
    /// the caller doesn't need to parse the `M` or the `J` or the leading
    /// digit. The caller should just call this routine when it *expect* a
    /// date specification to follow.
    ///
    /// Upon success, the parser will be positioned immediately after the
    /// date specification.
    fn parse_posix_date(&self) -> Result<PosixDay, PosixDateError> {
        match self.byte() {
            b'J' => {
                if !self.bump() {
                    return Err(PosixDateError::ExpectedJulianNoLeap);
                }
                Ok(PosixDay::JulianOne(self.parse_posix_julian_day_no_leap()?))
            }
            b'0'..=b'9' => Ok(PosixDay::JulianZero(
                self.parse_posix_julian_day_with_leap()?,
            )),
            b'M' => {
                if !self.bump() {
                    return Err(PosixDateError::ExpectedMonthWeekWeekday);
                }
                let (month, week, weekday) = self.parse_weekday_of_month()?;
                Ok(PosixDay::WeekdayOfMonth { month, week, weekday })
            }
            _ => Err(PosixDateError::UnexpectedByte),
        }
    }

    /// Parses a POSIX Julian day that does not include leap days
    /// (`1 <= n <= 365`).
    ///
    /// This assumes the parser is positioned just after the `J` and at the
    /// first digit of the Julian day. Upon success, the parser will be
    /// positioned immediately following the day number.
    fn parse_posix_julian_day_no_leap(
        &self,
    ) -> Result<i16, PosixJulianNoLeapError> {
        let number = self
            .parse_number_with_upto_n_digits(3)
            .map_err(PosixJulianNoLeapError::Parse)?;
        let number = i16::try_from(number)
            .map_err(|_| PosixJulianNoLeapError::Range)?;
        if !(1 <= number && number <= 365) {
            return Err(PosixJulianNoLeapError::Range);
        }
        Ok(number)
    }

    /// Parses a POSIX Julian day that includes leap days (`0 <= n <=
    /// 365`).
    ///
    /// This assumes the parser is positioned at the first digit of the
    /// Julian day. Upon success, the parser will be positioned immediately
    /// following the day number.
    fn parse_posix_julian_day_with_leap(
        &self,
    ) -> Result<i16, PosixJulianLeapError> {
        let number = self
            .parse_number_with_upto_n_digits(3)
            .map_err(PosixJulianLeapError::Parse)?;
        let number =
            i16::try_from(number).map_err(|_| PosixJulianLeapError::Range)?;
        if !(0 <= number && number <= 365) {
            return Err(PosixJulianLeapError::Range);
        }
        Ok(number)
    }

    /// Parses a POSIX "weekday of month" specification.
    ///
    /// This assumes the parser is positioned just after the `M` byte and
    /// at the first digit of the month. Upon success, the parser will be
    /// positioned immediately following the "weekday of the month" that
    /// was parsed.
    ///
    /// The tuple returned is month (1..=12), week (1..=5) and weekday
    /// (0..=6 with 0=Sunday).
    fn parse_weekday_of_month(
        &self,
    ) -> Result<(i8, i8, i8), WeekdayOfMonthError> {
        let month = self.parse_month()?;
        if self.maybe_byte() != Some(b'.') {
            return Err(WeekdayOfMonthError::ExpectedDotAfterMonth);
        }
        if !self.bump() {
            return Err(WeekdayOfMonthError::ExpectedWeekAfterMonth);
        }
        let week = self.parse_week()?;
        if self.maybe_byte() != Some(b'.') {
            return Err(WeekdayOfMonthError::ExpectedDotAfterWeek);
        }
        if !self.bump() {
            return Err(WeekdayOfMonthError::ExpectedDayOfWeekAfterWeek);
        }
        let weekday = self.parse_weekday()?;
        Ok((month, week, weekday))
    }

    /// This parses a POSIX time specification in the format
    /// `[+/-]hh?[:mm[:ss]]`.
    ///
    /// This assumes the parser is positioned at the first `h` (or the
    /// sign, if present). Upon success, the parser will be positioned
    /// immediately following the end of the time specification.
    fn parse_posix_time(&self) -> Result<PosixTime, PosixTimeError> {
        let (sign, hour) = if self.ianav3plus {
            let sign = self.parse_optional_sign()?.unwrap_or(1);
            let hour = self.parse_hour_ianav3plus()?;
            (sign, hour)
        } else {
            (1, i16::from(self.parse_hour_posix()?))
        };
        let (mut minute, mut second) = (0, 0);
        if self.maybe_byte() == Some(b':') {
            if !self.bump() {
                return Err(PosixTimeError::IncompleteMinutes);
            }
            minute = self.parse_minute()?;
            if self.maybe_byte() == Some(b':') {
                if !self.bump() {
                    return Err(PosixTimeError::IncompleteSeconds);
                }
                second = self.parse_second()?;
            }
        }
        let mut time = PosixTime { second: i32::from(hour) * 3600 };
        time.second += i32::from(minute) * 60;
        time.second += i32::from(second);
        time.second *= i32::from(sign);
        // Must be true because the parsing routines for hours, minutes
        // and seconds enforce they are in the ranges -167..=167, 0..=59
        // and 0..=59, respectively.
        assert!(
            -604799 <= time.second && time.second <= 604799,
            "POSIX time seconds {} is out of range",
            time.second
        );
        Ok(time)
    }

    /// Parses a month.
    ///
    /// This is expected to be positioned at the first digit. Upon success,
    /// the parser will be positioned after the month (which may contain
    /// two digits).
    fn parse_month(&self) -> Result<i8, MonthError> {
        let number = self
            .parse_number_with_upto_n_digits(2)
            .map_err(MonthError::Parse)?;
        let number = i8::try_from(number).map_err(|_| MonthError::Range)?;
        if !(1 <= number && number <= 12) {
            return Err(MonthError::Range);
        }
        Ok(number)
    }

    /// Parses a week-of-month number.
    ///
    /// This is expected to be positioned at the first digit. Upon success,
    /// the parser will be positioned after the week digit.
    fn parse_week(&self) -> Result<i8, WeekOfMonthError> {
        let number = self
            .parse_number_with_exactly_n_digits(1)
            .map_err(WeekOfMonthError::Parse)?;
        let number =
            i8::try_from(number).map_err(|_| WeekOfMonthError::Range)?;
        if !(1 <= number && number <= 5) {
            return Err(WeekOfMonthError::Range);
        }
        Ok(number)
    }

    /// Parses a weekday number.
    ///
    /// This is expected to be positioned at the first digit. Upon success,
    /// the parser will be positioned after the week digit.
    ///
    /// The weekday returned is guaranteed to be in the range `0..=6`, with
    /// `0` corresponding to Sunday.
    fn parse_weekday(&self) -> Result<i8, WeekdayError> {
        let number = self
            .parse_number_with_exactly_n_digits(1)
            .map_err(WeekdayError::Parse)?;
        let number = i8::try_from(number).map_err(|_| WeekdayError::Range)?;
        if !(0 <= number && number <= 6) {
            return Err(WeekdayError::Range);
        }
        Ok(number)
    }

    /// Parses an hour from a POSIX time specification with the IANA
    /// v3+ extension. That is, the hour may be in the range `0..=167`.
    /// (Callers should parse an optional sign preceding the hour digits
    /// when IANA V3+ parsing is enabled.)
    ///
    /// The hour is allowed to be a single digit (unlike minutes or
    /// seconds).
    ///
    /// This assumes the parser is positioned at the position where the
    /// first hour digit should occur. Upon success, the parser will be
    /// positioned immediately after the last hour digit.
    fn parse_hour_ianav3plus(&self) -> Result<i16, HourIanaError> {
        // Callers should only be using this method when IANA v3+ parsing
        // is enabled.
        assert!(self.ianav3plus);
        let number = self
            .parse_number_with_upto_n_digits(3)
            .map_err(HourIanaError::Parse)?;
        let number =
            i16::try_from(number).map_err(|_| HourIanaError::Range)?;
        if !(0 <= number && number <= 167) {
            // The error message says -167 but the check above uses 0.
            // This is because the caller is responsible for parsing
            // the sign.
            return Err(HourIanaError::Range);
        }
        Ok(number)
    }

    /// Parses an hour from a POSIX time specification, with the allowed
    /// range being `0..=24`.
    ///
    /// The hour is allowed to be a single digit (unlike minutes or
    /// seconds).
    ///
    /// This assumes the parser is positioned at the position where the
    /// first hour digit should occur. Upon success, the parser will be
    /// positioned immediately after the last hour digit.
    fn parse_hour_posix(&self) -> Result<i8, HourPosixError> {
        let number = self
            .parse_number_with_upto_n_digits(2)
            .map_err(HourPosixError::Parse)?;
        let number =
            i8::try_from(number).map_err(|_| HourPosixError::Range)?;
        if !(0 <= number && number <= 24) {
            return Err(HourPosixError::Range);
        }
        Ok(number)
    }

    /// Parses a minute from a POSIX time specification.
    ///
    /// The minute must be exactly two digits.
    ///
    /// This assumes the parser is positioned at the position where the
    /// first minute digit should occur. Upon success, the parser will be
    /// positioned immediately after the second minute digit.
    fn parse_minute(&self) -> Result<i8, MinuteError> {
        let number = self
            .parse_number_with_exactly_n_digits(2)
            .map_err(MinuteError::Parse)?;
        let number = i8::try_from(number).map_err(|_| MinuteError::Range)?;
        if !(0 <= number && number <= 59) {
            return Err(MinuteError::Range);
        }
        Ok(number)
    }

    /// Parses a second from a POSIX time specification.
    ///
    /// The second must be exactly two digits.
    ///
    /// This assumes the parser is positioned at the position where the
    /// first second digit should occur. Upon success, the parser will be
    /// positioned immediately after the second second digit.
    fn parse_second(&self) -> Result<i8, SecondError> {
        let number = self
            .parse_number_with_exactly_n_digits(2)
            .map_err(SecondError::Parse)?;
        let number = i8::try_from(number).map_err(|_| SecondError::Range)?;
        if !(0 <= number && number <= 59) {
            return Err(SecondError::Range);
        }
        Ok(number)
    }

    /// Parses a signed 64-bit integer expressed in exactly `n` digits.
    ///
    /// If `n` digits could not be found (or if the `TZ` string ends before
    /// `n` digits could be found), then this returns an error.
    ///
    /// This assumes that `n >= 1` and that the parser is positioned at the
    /// first digit. Upon success, the parser is positioned immediately
    /// after the `n`th digit.
    fn parse_number_with_exactly_n_digits(
        &self,
        n: usize,
    ) -> Result<i32, NumberError> {
        assert!(n >= 1, "numbers must have at least 1 digit");
        let mut number: i32 = 0;
        for _ in 0..n {
            if self.is_done() {
                return Err(NumberError::ExpectedLength);
            }
            let byte = self.byte();
            let digit = match byte.checked_sub(b'0') {
                None => {
                    return Err(NumberError::InvalidDigit);
                }
                Some(digit) if digit > 9 => {
                    return Err(NumberError::InvalidDigit);
                }
                Some(digit) => {
                    debug_assert!((0..=9).contains(&digit));
                    i32::from(digit)
                }
            };
            number = number
                .checked_mul(10)
                .and_then(|n| n.checked_add(digit))
                .ok_or(NumberError::TooBig)?;
            self.bump();
        }
        Ok(number)
    }

    /// Parses a signed 64-bit integer expressed with up to `n` digits and
    /// at least 1 digit.
    ///
    /// This assumes that `n >= 1` and that the parser is positioned at the
    /// first digit. Upon success, the parser is position immediately after
    /// the last digit (which can be at most `n`).
    fn parse_number_with_upto_n_digits(
        &self,
        n: usize,
    ) -> Result<i32, NumberError> {
        assert!(n >= 1, "numbers must have at least 1 digit");
        let mut number: i32 = 0;
        for i in 0..n {
            if self.is_done() || !self.byte().is_ascii_digit() {
                if i == 0 {
                    return Err(NumberError::Empty);
                }
                break;
            }
            let digit = i32::from(self.byte() - b'0');
            number = number
                .checked_mul(10)
                .and_then(|n| n.checked_add(digit))
                .ok_or(NumberError::TooBig)?;
            self.bump();
        }
        Ok(number)
    }

    /// Parses an optional sign.
    ///
    /// This assumes the parser is positioned at the position where a
    /// positive or negative sign is permitted. If one exists, then it
    /// is consumed and returned. Moreover, if one exists, then this
    /// guarantees that it is not the last byte in the input. That is, upon
    /// success, it is valid to call `self.byte()`.
    fn parse_optional_sign(&self) -> Result<Option<i8>, OptionalSignError> {
        if self.is_done() {
            return Ok(None);
        }
        Ok(match self.byte() {
            b'-' => {
                if !self.bump() {
                    return Err(OptionalSignError::ExpectedDigitAfterMinus);
                }
                Some(-1)
            }
            b'+' => {
                if !self.bump() {
                    return Err(OptionalSignError::ExpectedDigitAfterPlus);
                }
                Some(1)
            }
            _ => None,
        })
    }
}

/// Helper routines for parsing a POSIX `TZ` string.
impl<'s> Parser<'s> {
    /// Bump the parser to the next byte.
    ///
    /// If the end of the input has been reached, then `false` is returned.
    fn bump(&self) -> bool {
        if self.is_done() {
            return false;
        }
        self.pos.set(
            self.pos().checked_add(1).expect("pos cannot overflow usize"),
        );
        !self.is_done()
    }

    /// Returns true if the next call to `bump` would return false.
    fn is_done(&self) -> bool {
        self.pos() == self.tz.len()
    }

    /// Return the byte at the current position of the parser.
    ///
    /// This panics if the parser is positioned at the end of the TZ
    /// string.
    fn byte(&self) -> u8 {
        self.tz[self.pos()]
    }

    /// Return the byte at the current position of the parser. If the TZ
    /// string has been exhausted, then this returns `None`.
    fn maybe_byte(&self) -> Option<u8> {
        self.tz.get(self.pos()).copied()
    }

    /// Return the current byte offset of the parser.
    ///
    /// The offset starts at `0` from the beginning of the TZ string.
    fn pos(&self) -> usize {
        self.pos.get()
    }

    /// Returns the remaining bytes of the TZ string.
    ///
    /// This includes `self.byte()`. It may be empty.
    fn remaining(&self) -> &'s [u8] {
        &self.tz[self.pos()..]
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PosixTimeZoneError {
    kind: ErrorKind,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ErrorKind {
    AbbreviationDst(AbbreviationError),
    AbbreviationStd(AbbreviationError),
    Empty,
    ExpectedCommaAfterDst,
    FoundDstNoRule,
    FoundDstNoRuleWithOffset,
    FoundEndAfterComma,
    FoundRemaining,
    OffsetDst(PosixOffsetError),
    OffsetStd(PosixOffsetError),
    Rule(PosixRuleError),
}

impl core::fmt::Display for PosixTimeZoneError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::ErrorKind::*;
        match self.kind {
            AbbreviationDst(ref err) => {
                f.write_str("failed to parse DST time zone abbreviation: ")?;
                core::fmt::Display::fmt(err, f)
            }
            AbbreviationStd(ref err) => {
                f.write_str(
                    "failed to parse standard time zone abbreviation: ",
                )?;
                core::fmt::Display::fmt(err, f)
            }
            Empty => f.write_str(
                "an empty string is not a valid POSIX time zone \
                 transition rule",
            ),
            ExpectedCommaAfterDst => f.write_str(
                "expected `,` after parsing DST offset \
                 in POSIX time zone string",
            ),
            FoundDstNoRule => f.write_str(
                "found DST abbreviation in POSIX time zone string, \
                 but no transition rule \
                 (this is technically allowed by POSIX, but has \
                 unspecified behavior)",
            ),
            FoundDstNoRuleWithOffset => f.write_str(
                "found DST abbreviation and offset in POSIX time zone string, \
                 but no transition rule \
                 (this is technically allowed by POSIX, but has \
                 unspecified behavior)",
            ),
            FoundEndAfterComma => f.write_str(
                "after parsing DST offset in POSIX time zone string, \
                 found end of string after a trailing `,`",
            ),
            FoundRemaining => f.write_str(
                "expected entire POSIX TZ string to be a valid \
                 time zone transition rule, but found data after \
                 parsing a valid time zone transition rule",
            ),
            OffsetDst(ref err) => {
                f.write_str("failed to parse DST offset: ")?;
                core::fmt::Display::fmt(err, f)
            }
            OffsetStd(ref err) => {
                f.write_str("failed to parse standard offset: ")?;
                core::fmt::Display::fmt(err, f)
            }
            Rule(ref err) => core::fmt::Display::fmt(err, f),
        }
    }
}

impl From<ErrorKind> for PosixTimeZoneError {
    fn from(kind: ErrorKind) -> PosixTimeZoneError {
        PosixTimeZoneError { kind }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum PosixOffsetError {
    HourPosix(HourPosixError),
    IncompleteMinutes,
    IncompleteSeconds,
    Minute(MinuteError),
    OptionalSign(OptionalSignError),
    Second(SecondError),
}

impl core::fmt::Display for PosixOffsetError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::PosixOffsetError::*;
        match *self {
            HourPosix(ref err) => core::fmt::Display::fmt(err, f),
            IncompleteMinutes => f.write_str(
                "incomplete time in \
                 POSIX time zone string (missing minutes)",
            ),
            IncompleteSeconds => f.write_str(
                "incomplete time in \
                 POSIX time zone string (missing seconds)",
            ),
            Minute(ref err) => core::fmt::Display::fmt(err, f),
            Second(ref err) => core::fmt::Display::fmt(err, f),
            OptionalSign(ref err) => {
                f.write_str(
                    "failed to parse sign for time offset \
                     POSIX time zone string",
                )?;
                core::fmt::Display::fmt(err, f)
            }
        }
    }
}

impl From<HourPosixError> for PosixOffsetError {
    fn from(err: HourPosixError) -> PosixOffsetError {
        PosixOffsetError::HourPosix(err)
    }
}

impl From<MinuteError> for PosixOffsetError {
    fn from(err: MinuteError) -> PosixOffsetError {
        PosixOffsetError::Minute(err)
    }
}

impl From<OptionalSignError> for PosixOffsetError {
    fn from(err: OptionalSignError) -> PosixOffsetError {
        PosixOffsetError::OptionalSign(err)
    }
}

impl From<SecondError> for PosixOffsetError {
    fn from(err: SecondError) -> PosixOffsetError {
        PosixOffsetError::Second(err)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum PosixRuleError {
    DateTimeEnd(PosixDateTimeError),
    DateTimeStart(PosixDateTimeError),
    ExpectedEnd,
}

impl core::fmt::Display for PosixRuleError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::PosixRuleError::*;
        match *self {
            DateTimeEnd(ref err) => {
                f.write_str("failed to parse end of DST transition rule: ")?;
                core::fmt::Display::fmt(err, f)
            }
            DateTimeStart(ref err) => {
                f.write_str("failed to parse start of DST transition rule: ")?;
                core::fmt::Display::fmt(err, f)
            }
            ExpectedEnd => f.write_str(
                "expected end of DST rule after parsing the start \
                 of the DST rule",
            ),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum PosixDateTimeError {
    Date(PosixDateError),
    ExpectedTime,
    Time(PosixTimeError),
}

impl core::fmt::Display for PosixDateTimeError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::PosixDateTimeError::*;
        match *self {
            Date(ref err) => core::fmt::Display::fmt(err, f),
            ExpectedTime => f.write_str(
                "expected time specification after `/` following a date
                 specification in a POSIX time zone DST transition rule",
            ),
            Time(ref err) => core::fmt::Display::fmt(err, f),
        }
    }
}

impl From<PosixDateError> for PosixDateTimeError {
    fn from(err: PosixDateError) -> PosixDateTimeError {
        PosixDateTimeError::Date(err)
    }
}

impl From<PosixTimeError> for PosixDateTimeError {
    fn from(err: PosixTimeError) -> PosixDateTimeError {
        PosixDateTimeError::Time(err)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum PosixDateError {
    ExpectedJulianNoLeap,
    ExpectedMonthWeekWeekday,
    JulianLeap(PosixJulianLeapError),
    JulianNoLeap(PosixJulianNoLeapError),
    UnexpectedByte,
    WeekdayOfMonth(WeekdayOfMonthError),
}

impl core::fmt::Display for PosixDateError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::PosixDateError::*;
        match *self {
            ExpectedJulianNoLeap => f.write_str(
                "expected one-based Julian day after `J` in date \
                 specification of a POSIX time zone DST \
                 transition rule, but found the end of input",
            ),
            ExpectedMonthWeekWeekday => f.write_str(
                "expected month-week-weekday after `M` in date \
                 specification of a POSIX time zone DST \
                 transition rule, but found the end of input",
            ),
            JulianLeap(ref err) => core::fmt::Display::fmt(err, f),
            JulianNoLeap(ref err) => core::fmt::Display::fmt(err, f),
            UnexpectedByte => f.write_str(
                "expected `J`, a digit or `M` at the beginning of a date \
                 specification of a POSIX time zone DST transition rule",
            ),
            WeekdayOfMonth(ref err) => core::fmt::Display::fmt(err, f),
        }
    }
}

impl From<PosixJulianLeapError> for PosixDateError {
    fn from(err: PosixJulianLeapError) -> PosixDateError {
        PosixDateError::JulianLeap(err)
    }
}

impl From<PosixJulianNoLeapError> for PosixDateError {
    fn from(err: PosixJulianNoLeapError) -> PosixDateError {
        PosixDateError::JulianNoLeap(err)
    }
}

impl From<WeekdayOfMonthError> for PosixDateError {
    fn from(err: WeekdayOfMonthError) -> PosixDateError {
        PosixDateError::WeekdayOfMonth(err)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum PosixJulianNoLeapError {
    Parse(NumberError),
    Range,
}

impl core::fmt::Display for PosixJulianNoLeapError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::PosixJulianNoLeapError::*;
        match *self {
            Parse(ref err) => {
                f.write_str("invalid one-based Julian day digits: ")?;
                core::fmt::Display::fmt(err, f)
            }
            Range => f.write_str(
                "parsed one-based Julian day, but it's not in supported \
                 range of `1..=365`",
            ),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum PosixJulianLeapError {
    Parse(NumberError),
    Range,
}

impl core::fmt::Display for PosixJulianLeapError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::PosixJulianLeapError::*;
        match *self {
            Parse(ref err) => {
                f.write_str("invalid zero-based Julian day digits: ")?;
                core::fmt::Display::fmt(err, f)
            }
            Range => f.write_str(
                "parsed zero-based Julian day, but it's not in supported \
                 range of `0..=365`",
            ),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum AbbreviationError {
    Quoted(QuotedAbbreviationError),
    Unquoted(UnquotedAbbreviationError),
}

impl core::fmt::Display for AbbreviationError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::AbbreviationError::*;
        match *self {
            Quoted(ref err) => core::fmt::Display::fmt(err, f),
            Unquoted(ref err) => core::fmt::Display::fmt(err, f),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum UnquotedAbbreviationError {
    InvalidUtf8,
    TooLong,
    TooShort,
}

impl core::fmt::Display for UnquotedAbbreviationError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::UnquotedAbbreviationError::*;
        match *self {
            InvalidUtf8 => f.write_str(
                "unquoted time zone abbreviation must be valid UTF-8",
            ),
            TooLong => write!(
                f,
                "expected unquoted time zone abbreviation with at most \
                 {} bytes, but found an abbreviation that is longer",
                Abbreviation::capacity(),
            ),
            TooShort => f.write_str(
                "expected unquoted time zone abbreviation to have length of \
                 3 or more bytes, but an abbreviation that is shorter",
            ),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum QuotedAbbreviationError {
    InvalidUtf8,
    TooLong,
    TooShort,
    UnexpectedEnd,
    UnexpectedEndAfterOpening,
    UnexpectedLastByte,
}

impl core::fmt::Display for QuotedAbbreviationError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::QuotedAbbreviationError::*;
        match *self {
            InvalidUtf8 => f.write_str(
                "quoted time zone abbreviation must be valid UTF-8",
            ),
            TooLong => write!(
                f,
                "expected quoted time zone abbreviation with at most \
                 {} bytes, but found an abbreviation that is longer",
                Abbreviation::capacity(),
            ),
            TooShort => f.write_str(
                "expected quoted time zone abbreviation to have length of \
                 3 or more bytes, but an abbreviation that is shorter",
            ),
            UnexpectedEnd => f.write_str(
                "found non-empty quoted time zone abbreviation, but \
                 found end of input before an end-of-quoted abbreviation \
                 `>` character",
            ),
            UnexpectedEndAfterOpening => f.write_str(
                "found opening `<` quote for time zone abbreviation in \
                 POSIX time zone transition rule, and expected a name \
                 following it, but found the end of input instead",
            ),
            UnexpectedLastByte => f.write_str(
                "found non-empty quoted time zone abbreviation, but \
                 found did not find end-of-quoted abbreviation `>` \
                 character",
            ),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum WeekdayOfMonthError {
    ExpectedDayOfWeekAfterWeek,
    ExpectedDotAfterMonth,
    ExpectedDotAfterWeek,
    ExpectedWeekAfterMonth,
    Month(MonthError),
    WeekOfMonth(WeekOfMonthError),
    Weekday(WeekdayError),
}

impl core::fmt::Display for WeekdayOfMonthError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::WeekdayOfMonthError::*;
        match *self {
            ExpectedDayOfWeekAfterWeek => f.write_str(
                "expected day-of-week after week in POSIX time zone rule",
            ),
            ExpectedDotAfterMonth => {
                f.write_str("expected `.` after month in POSIX time zone rule")
            }
            ExpectedWeekAfterMonth => f.write_str(
                "expected week after month in POSIX time zone rule",
            ),
            ExpectedDotAfterWeek => {
                f.write_str("expected `.` after week in POSIX time zone rule")
            }
            Month(ref err) => core::fmt::Display::fmt(err, f),
            WeekOfMonth(ref err) => core::fmt::Display::fmt(err, f),
            Weekday(ref err) => core::fmt::Display::fmt(err, f),
        }
    }
}

impl From<MonthError> for WeekdayOfMonthError {
    fn from(err: MonthError) -> WeekdayOfMonthError {
        WeekdayOfMonthError::Month(err)
    }
}

impl From<WeekOfMonthError> for WeekdayOfMonthError {
    fn from(err: WeekOfMonthError) -> WeekdayOfMonthError {
        WeekdayOfMonthError::WeekOfMonth(err)
    }
}

impl From<WeekdayError> for WeekdayOfMonthError {
    fn from(err: WeekdayError) -> WeekdayOfMonthError {
        WeekdayOfMonthError::Weekday(err)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum PosixTimeError {
    HourIana(HourIanaError),
    HourPosix(HourPosixError),
    IncompleteMinutes,
    IncompleteSeconds,
    Minute(MinuteError),
    OptionalSign(OptionalSignError),
    Second(SecondError),
}

impl core::fmt::Display for PosixTimeError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::PosixTimeError::*;
        match *self {
            HourIana(ref err) => core::fmt::Display::fmt(err, f),
            HourPosix(ref err) => core::fmt::Display::fmt(err, f),
            IncompleteMinutes => f.write_str(
                "incomplete time zone transition time in \
                 POSIX time zone string (missing minutes)",
            ),
            IncompleteSeconds => f.write_str(
                "incomplete time zone transition time in \
                 POSIX time zone string (missing seconds)",
            ),
            Minute(ref err) => core::fmt::Display::fmt(err, f),
            Second(ref err) => core::fmt::Display::fmt(err, f),
            OptionalSign(ref err) => {
                f.write_str(
                    "failed to parse sign for time zone transition time",
                )?;
                core::fmt::Display::fmt(err, f)
            }
        }
    }
}

impl From<HourIanaError> for PosixTimeError {
    fn from(err: HourIanaError) -> PosixTimeError {
        PosixTimeError::HourIana(err)
    }
}

impl From<HourPosixError> for PosixTimeError {
    fn from(err: HourPosixError) -> PosixTimeError {
        PosixTimeError::HourPosix(err)
    }
}

impl From<MinuteError> for PosixTimeError {
    fn from(err: MinuteError) -> PosixTimeError {
        PosixTimeError::Minute(err)
    }
}

impl From<OptionalSignError> for PosixTimeError {
    fn from(err: OptionalSignError) -> PosixTimeError {
        PosixTimeError::OptionalSign(err)
    }
}

impl From<SecondError> for PosixTimeError {
    fn from(err: SecondError) -> PosixTimeError {
        PosixTimeError::Second(err)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum MonthError {
    Parse(NumberError),
    Range,
}

impl core::fmt::Display for MonthError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::MonthError::*;
        match *self {
            Parse(ref err) => {
                f.write_str("invalid month digits: ")?;
                core::fmt::Display::fmt(err, f)
            }
            Range => f.write_str(
                "parsed month, but it's not in supported \
                 range of `1..=12`",
            ),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum WeekOfMonthError {
    Parse(NumberError),
    Range,
}

impl core::fmt::Display for WeekOfMonthError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::WeekOfMonthError::*;
        match *self {
            Parse(ref err) => {
                f.write_str("invalid week-of-month digits: ")?;
                core::fmt::Display::fmt(err, f)
            }
            Range => f.write_str(
                "parsed week-of-month, but it's not in supported \
                 range of `1..=5`",
            ),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum WeekdayError {
    Parse(NumberError),
    Range,
}

impl core::fmt::Display for WeekdayError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::WeekdayError::*;
        match *self {
            Parse(ref err) => {
                f.write_str("invalid weekday digits: ")?;
                core::fmt::Display::fmt(err, f)
            }
            Range => f.write_str(
                "parsed weekday, but it's not in supported \
                 range of `0..=6` (with `0` corresponding to Sunday)",
            ),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum HourIanaError {
    Parse(NumberError),
    Range,
}

impl core::fmt::Display for HourIanaError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::HourIanaError::*;
        match *self {
            Parse(ref err) => {
                f.write_str("invalid hour digits: ")?;
                core::fmt::Display::fmt(err, f)
            }
            Range => f.write_str(
                "parsed hours, but it's not in supported \
                 range of `-167..=167`",
            ),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum HourPosixError {
    Parse(NumberError),
    Range,
}

impl core::fmt::Display for HourPosixError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::HourPosixError::*;
        match *self {
            Parse(ref err) => {
                f.write_str("invalid hour digits: ")?;
                core::fmt::Display::fmt(err, f)
            }
            Range => f.write_str(
                "parsed hours, but it's not in supported \
                 range of `0..=24`",
            ),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum MinuteError {
    Parse(NumberError),
    Range,
}

impl core::fmt::Display for MinuteError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::MinuteError::*;
        match *self {
            Parse(ref err) => {
                f.write_str("invalid minute digits: ")?;
                core::fmt::Display::fmt(err, f)
            }
            Range => f.write_str(
                "parsed minutes, but it's not in supported \
                 range of `0..=59`",
            ),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum SecondError {
    Parse(NumberError),
    Range,
}

impl core::fmt::Display for SecondError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::SecondError::*;
        match *self {
            Parse(ref err) => {
                f.write_str("invalid second digits: ")?;
                core::fmt::Display::fmt(err, f)
            }
            Range => f.write_str(
                "parsed seconds, but it's not in supported \
                 range of `0..=59`",
            ),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum NumberError {
    Empty,
    ExpectedLength,
    InvalidDigit,
    TooBig,
}

impl core::fmt::Display for NumberError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::NumberError::*;
        match *self {
            Empty => f.write_str("invalid number, no digits found"),
            ExpectedLength => f.write_str(
                "expected a fixed number of digits, \
                 but found incorrect number",
            ),
            InvalidDigit => f.write_str("expected digit in range `0..=9`"),
            TooBig => f.write_str(
                "parsed number too big to fit into a 32-bit signed integer",
            ),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum OptionalSignError {
    ExpectedDigitAfterMinus,
    ExpectedDigitAfterPlus,
}

impl core::fmt::Display for OptionalSignError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::OptionalSignError::*;
        match *self {
            ExpectedDigitAfterMinus => f.write_str(
                "expected digit after `-` sign, \
                 but got end of input",
            ),
            ExpectedDigitAfterPlus => f.write_str(
                "expected digit after `+` sign, \
                 but got end of input",
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn posix_time_zone(
        input: impl AsRef<[u8]>,
    ) -> PosixTimeZone<Abbreviation> {
        let input = input.as_ref();
        let tz = PosixTimeZone::parse(input).unwrap();
        #[cfg(feature = "alloc")]
        {
            use alloc::string::ToString;

            // While we're here, assert that converting the TZ back
            // to a string matches what we got. In the original version
            // of the POSIX TZ parser, we were very meticulous about
            // capturing the exact AST of the time zone. But I've
            // since simplified the data structure considerably such
            // that it is lossy in terms of what was actually parsed
            // (but of course, not lossy in terms of the semantic
            // meaning of the time zone).
            //
            // So to account for this, we serialize to a string and
            // then parse it back. We should get what we started with.
            let reparsed =
                PosixTimeZone::parse(tz.to_string().as_bytes()).unwrap();
            assert_eq!(tz, reparsed);
            assert_eq!(tz.to_string(), reparsed.to_string());
        }
        tz
    }

    fn parser(s: &str) -> Parser<'_> {
        Parser::new(s.as_bytes())
    }

    fn date(year: i16, month: i8, day: i8) -> IDate {
        IDate { year, month, day }
    }

    #[test]
    fn parse() {
        let p = parser("NZST-12NZDT,J60,J300");
        assert_eq!(
            p.parse().unwrap(),
            PosixTimeZone {
                std_abbrev: "NZST".into(),
                std_offset: PosixOffset { second: 12 * 60 * 60 },
                dst: Some(PosixDst {
                    abbrev: "NZDT".into(),
                    offset: PosixOffset { second: 13 * 60 * 60 },
                    rule: PosixRule {
                        start: PosixDayTime {
                            date: PosixDay::JulianOne(60),
                            time: PosixTime { second: 2 * 60 * 60 },
                        },
                        end: PosixDayTime {
                            date: PosixDay::JulianOne(300),
                            time: PosixTime { second: 2 * 60 * 60 },
                        },
                    },
                }),
            },
        );

        let p = Parser::new("NZST-12NZDT,J60,J300WAT");
        assert!(p.parse().is_err());
    }

    #[test]
    fn parse_posix_time_zone() {
        let p = Parser::new("NZST-12NZDT,M9.5.0,M4.1.0/3");
        assert_eq!(
            p.parse_posix_time_zone().unwrap(),
            PosixTimeZone {
                std_abbrev: "NZST".into(),
                std_offset: PosixOffset { second: 12 * 60 * 60 },
                dst: Some(PosixDst {
                    abbrev: "NZDT".into(),
                    offset: PosixOffset { second: 13 * 60 * 60 },
                    rule: PosixRule {
                        start: PosixDayTime {
                            date: PosixDay::WeekdayOfMonth {
                                month: 9,
                                week: 5,
                                weekday: 0,
                            },
                            time: PosixTime { second: 2 * 60 * 60 },
                        },
                        end: PosixDayTime {
                            date: PosixDay::WeekdayOfMonth {
                                month: 4,
                                week: 1,
                                weekday: 0,
                            },
                            time: PosixTime { second: 3 * 60 * 60 },
                        },
                    },
                }),
            },
        );

        let p = Parser::new("NZST-12NZDT,M9.5.0,M4.1.0/3WAT");
        assert_eq!(
            p.parse_posix_time_zone().unwrap(),
            PosixTimeZone {
                std_abbrev: "NZST".into(),
                std_offset: PosixOffset { second: 12 * 60 * 60 },
                dst: Some(PosixDst {
                    abbrev: "NZDT".into(),
                    offset: PosixOffset { second: 13 * 60 * 60 },
                    rule: PosixRule {
                        start: PosixDayTime {
                            date: PosixDay::WeekdayOfMonth {
                                month: 9,
                                week: 5,
                                weekday: 0,
                            },
                            time: PosixTime { second: 2 * 60 * 60 },
                        },
                        end: PosixDayTime {
                            date: PosixDay::WeekdayOfMonth {
                                month: 4,
                                week: 1,
                                weekday: 0,
                            },
                            time: PosixTime { second: 3 * 60 * 60 },
                        },
                    },
                }),
            },
        );

        let p = Parser::new("NZST-12NZDT,J60,J300");
        assert_eq!(
            p.parse_posix_time_zone().unwrap(),
            PosixTimeZone {
                std_abbrev: "NZST".into(),
                std_offset: PosixOffset { second: 12 * 60 * 60 },
                dst: Some(PosixDst {
                    abbrev: "NZDT".into(),
                    offset: PosixOffset { second: 13 * 60 * 60 },
                    rule: PosixRule {
                        start: PosixDayTime {
                            date: PosixDay::JulianOne(60),
                            time: PosixTime { second: 2 * 60 * 60 },
                        },
                        end: PosixDayTime {
                            date: PosixDay::JulianOne(300),
                            time: PosixTime { second: 2 * 60 * 60 },
                        },
                    },
                }),
            },
        );

        let p = Parser::new("NZST-12NZDT,J60,J300WAT");
        assert_eq!(
            p.parse_posix_time_zone().unwrap(),
            PosixTimeZone {
                std_abbrev: "NZST".into(),
                std_offset: PosixOffset { second: 12 * 60 * 60 },
                dst: Some(PosixDst {
                    abbrev: "NZDT".into(),
                    offset: PosixOffset { second: 13 * 60 * 60 },
                    rule: PosixRule {
                        start: PosixDayTime {
                            date: PosixDay::JulianOne(60),
                            time: PosixTime { second: 2 * 60 * 60 },
                        },
                        end: PosixDayTime {
                            date: PosixDay::JulianOne(300),
                            time: PosixTime { second: 2 * 60 * 60 },
                        },
                    },
                }),
            },
        );
    }

    #[test]
    fn parse_posix_dst() {
        let p = Parser::new("NZDT,M9.5.0,M4.1.0/3");
        assert_eq!(
            p.parse_posix_dst(&PosixOffset { second: 12 * 60 * 60 }).unwrap(),
            PosixDst {
                abbrev: "NZDT".into(),
                offset: PosixOffset { second: 13 * 60 * 60 },
                rule: PosixRule {
                    start: PosixDayTime {
                        date: PosixDay::WeekdayOfMonth {
                            month: 9,
                            week: 5,
                            weekday: 0,
                        },
                        time: PosixTime { second: 2 * 60 * 60 },
                    },
                    end: PosixDayTime {
                        date: PosixDay::WeekdayOfMonth {
                            month: 4,
                            week: 1,
                            weekday: 0,
                        },
                        time: PosixTime { second: 3 * 60 * 60 },
                    },
                },
            },
        );

        let p = Parser::new("NZDT,J60,J300");
        assert_eq!(
            p.parse_posix_dst(&PosixOffset { second: 12 * 60 * 60 }).unwrap(),
            PosixDst {
                abbrev: "NZDT".into(),
                offset: PosixOffset { second: 13 * 60 * 60 },
                rule: PosixRule {
                    start: PosixDayTime {
                        date: PosixDay::JulianOne(60),
                        time: PosixTime { second: 2 * 60 * 60 },
                    },
                    end: PosixDayTime {
                        date: PosixDay::JulianOne(300),
                        time: PosixTime { second: 2 * 60 * 60 },
                    },
                },
            },
        );

        let p = Parser::new("NZDT-7,J60,J300");
        assert_eq!(
            p.parse_posix_dst(&PosixOffset { second: 12 * 60 * 60 }).unwrap(),
            PosixDst {
                abbrev: "NZDT".into(),
                offset: PosixOffset { second: 7 * 60 * 60 },
                rule: PosixRule {
                    start: PosixDayTime {
                        date: PosixDay::JulianOne(60),
                        time: PosixTime { second: 2 * 60 * 60 },
                    },
                    end: PosixDayTime {
                        date: PosixDay::JulianOne(300),
                        time: PosixTime { second: 2 * 60 * 60 },
                    },
                },
            },
        );

        let p = Parser::new("NZDT+7,J60,J300");
        assert_eq!(
            p.parse_posix_dst(&PosixOffset { second: 12 * 60 * 60 }).unwrap(),
            PosixDst {
                abbrev: "NZDT".into(),
                offset: PosixOffset { second: -7 * 60 * 60 },
                rule: PosixRule {
                    start: PosixDayTime {
                        date: PosixDay::JulianOne(60),
                        time: PosixTime { second: 2 * 60 * 60 },
                    },
                    end: PosixDayTime {
                        date: PosixDay::JulianOne(300),
                        time: PosixTime { second: 2 * 60 * 60 },
                    },
                },
            },
        );

        let p = Parser::new("NZDT7,J60,J300");
        assert_eq!(
            p.parse_posix_dst(&PosixOffset { second: 12 * 60 * 60 }).unwrap(),
            PosixDst {
                abbrev: "NZDT".into(),
                offset: PosixOffset { second: -7 * 60 * 60 },
                rule: PosixRule {
                    start: PosixDayTime {
                        date: PosixDay::JulianOne(60),
                        time: PosixTime { second: 2 * 60 * 60 },
                    },
                    end: PosixDayTime {
                        date: PosixDay::JulianOne(300),
                        time: PosixTime { second: 2 * 60 * 60 },
                    },
                },
            },
        );

        let p = Parser::new("NZDT7,");
        assert!(p
            .parse_posix_dst(&PosixOffset { second: 12 * 60 * 60 })
            .is_err());

        let p = Parser::new("NZDT7!");
        assert!(p
            .parse_posix_dst(&PosixOffset { second: 12 * 60 * 60 })
            .is_err());
    }

    #[test]
    fn parse_abbreviation() {
        let p = Parser::new("ABC");
        assert_eq!(p.parse_abbreviation().unwrap(), "ABC");

        let p = Parser::new("<ABC>");
        assert_eq!(p.parse_abbreviation().unwrap(), "ABC");

        let p = Parser::new("<+09>");
        assert_eq!(p.parse_abbreviation().unwrap(), "+09");

        let p = Parser::new("+09");
        assert!(p.parse_abbreviation().is_err());
    }

    #[test]
    fn parse_unquoted_abbreviation() {
        let p = Parser::new("ABC");
        assert_eq!(p.parse_unquoted_abbreviation().unwrap(), "ABC");

        let p = Parser::new("ABCXYZ");
        assert_eq!(p.parse_unquoted_abbreviation().unwrap(), "ABCXYZ");

        let p = Parser::new("ABC123");
        assert_eq!(p.parse_unquoted_abbreviation().unwrap(), "ABC");

        let tz = "a".repeat(30);
        let p = Parser::new(&tz);
        assert_eq!(p.parse_unquoted_abbreviation().unwrap(), &*tz);

        let p = Parser::new("a");
        assert!(p.parse_unquoted_abbreviation().is_err());

        let p = Parser::new("ab");
        assert!(p.parse_unquoted_abbreviation().is_err());

        let p = Parser::new("ab1");
        assert!(p.parse_unquoted_abbreviation().is_err());

        let tz = "a".repeat(31);
        let p = Parser::new(&tz);
        assert!(p.parse_unquoted_abbreviation().is_err());

        let p = Parser::new(b"ab\xFFcd");
        assert!(p.parse_unquoted_abbreviation().is_err());
    }

    #[test]
    fn parse_quoted_abbreviation() {
        // The inputs look a little funny here, but that's because
        // 'parse_quoted_abbreviation' starts after the opening quote
        // has been parsed.

        let p = Parser::new("ABC>");
        assert_eq!(p.parse_quoted_abbreviation().unwrap(), "ABC");

        let p = Parser::new("ABCXYZ>");
        assert_eq!(p.parse_quoted_abbreviation().unwrap(), "ABCXYZ");

        let p = Parser::new("ABC>123");
        assert_eq!(p.parse_quoted_abbreviation().unwrap(), "ABC");

        let p = Parser::new("ABC123>");
        assert_eq!(p.parse_quoted_abbreviation().unwrap(), "ABC123");

        let p = Parser::new("ab1>");
        assert_eq!(p.parse_quoted_abbreviation().unwrap(), "ab1");

        let p = Parser::new("+09>");
        assert_eq!(p.parse_quoted_abbreviation().unwrap(), "+09");

        let p = Parser::new("-09>");
        assert_eq!(p.parse_quoted_abbreviation().unwrap(), "-09");

        let tz = alloc::format!("{}>", "a".repeat(30));
        let p = Parser::new(&tz);
        assert_eq!(
            p.parse_quoted_abbreviation().unwrap(),
            tz.trim_end_matches(">")
        );

        let p = Parser::new("a>");
        assert!(p.parse_quoted_abbreviation().is_err());

        let p = Parser::new("ab>");
        assert!(p.parse_quoted_abbreviation().is_err());

        let tz = alloc::format!("{}>", "a".repeat(31));
        let p = Parser::new(&tz);
        assert!(p.parse_quoted_abbreviation().is_err());

        let p = Parser::new(b"ab\xFFcd>");
        assert!(p.parse_quoted_abbreviation().is_err());

        let p = Parser::new("ABC");
        assert!(p.parse_quoted_abbreviation().is_err());

        let p = Parser::new("ABC!>");
        assert!(p.parse_quoted_abbreviation().is_err());
    }

    #[test]
    fn parse_posix_offset() {
        let p = Parser::new("5");
        assert_eq!(p.parse_posix_offset().unwrap().second, -5 * 60 * 60);

        let p = Parser::new("+5");
        assert_eq!(p.parse_posix_offset().unwrap().second, -5 * 60 * 60);

        let p = Parser::new("-5");
        assert_eq!(p.parse_posix_offset().unwrap().second, 5 * 60 * 60);

        let p = Parser::new("-12:34:56");
        assert_eq!(
            p.parse_posix_offset().unwrap().second,
            12 * 60 * 60 + 34 * 60 + 56,
        );

        let p = Parser::new("a");
        assert!(p.parse_posix_offset().is_err());

        let p = Parser::new("-");
        assert!(p.parse_posix_offset().is_err());

        let p = Parser::new("+");
        assert!(p.parse_posix_offset().is_err());

        let p = Parser::new("-a");
        assert!(p.parse_posix_offset().is_err());

        let p = Parser::new("+a");
        assert!(p.parse_posix_offset().is_err());

        let p = Parser::new("-25");
        assert!(p.parse_posix_offset().is_err());

        let p = Parser::new("+25");
        assert!(p.parse_posix_offset().is_err());

        // This checks that we don't accidentally permit IANA rules for
        // offset parsing. Namely, the IANA tzfile v3+ extension only applies
        // to transition times. But since POSIX says that the "time" for the
        // offset and transition is the same format, it would be an easy
        // implementation mistake to implement the more flexible rule for
        // IANA and have it accidentally also apply to the offset. So we check
        // that it doesn't here.
        let p = Parser { ianav3plus: true, ..Parser::new("25") };
        assert!(p.parse_posix_offset().is_err());
        let p = Parser { ianav3plus: true, ..Parser::new("+25") };
        assert!(p.parse_posix_offset().is_err());
        let p = Parser { ianav3plus: true, ..Parser::new("-25") };
        assert!(p.parse_posix_offset().is_err());
    }

    #[test]
    fn parse_rule() {
        let p = Parser::new("M9.5.0,M4.1.0/3");
        assert_eq!(
            p.parse_rule().unwrap(),
            PosixRule {
                start: PosixDayTime {
                    date: PosixDay::WeekdayOfMonth {
                        month: 9,
                        week: 5,
                        weekday: 0,
                    },
                    time: PosixTime { second: 2 * 60 * 60 },
                },
                end: PosixDayTime {
                    date: PosixDay::WeekdayOfMonth {
                        month: 4,
                        week: 1,
                        weekday: 0,
                    },
                    time: PosixTime { second: 3 * 60 * 60 },
                },
            },
        );

        let p = Parser::new("M9.5.0");
        assert!(p.parse_rule().is_err());

        let p = Parser::new(",M9.5.0,M4.1.0/3");
        assert!(p.parse_rule().is_err());

        let p = Parser::new("M9.5.0/");
        assert!(p.parse_rule().is_err());

        let p = Parser::new("M9.5.0,M4.1.0/");
        assert!(p.parse_rule().is_err());
    }

    #[test]
    fn parse_posix_datetime() {
        let p = Parser::new("J1");
        assert_eq!(
            p.parse_posix_datetime().unwrap(),
            PosixDayTime {
                date: PosixDay::JulianOne(1),
                time: PosixTime { second: 2 * 60 * 60 }
            },
        );

        let p = Parser::new("J1/3");
        assert_eq!(
            p.parse_posix_datetime().unwrap(),
            PosixDayTime {
                date: PosixDay::JulianOne(1),
                time: PosixTime { second: 3 * 60 * 60 }
            },
        );

        let p = Parser::new("M4.1.0/3");
        assert_eq!(
            p.parse_posix_datetime().unwrap(),
            PosixDayTime {
                date: PosixDay::WeekdayOfMonth {
                    month: 4,
                    week: 1,
                    weekday: 0,
                },
                time: PosixTime { second: 3 * 60 * 60 },
            },
        );

        let p = Parser::new("1/3:45:05");
        assert_eq!(
            p.parse_posix_datetime().unwrap(),
            PosixDayTime {
                date: PosixDay::JulianZero(1),
                time: PosixTime { second: 3 * 60 * 60 + 45 * 60 + 5 },
            },
        );

        let p = Parser::new("a");
        assert!(p.parse_posix_datetime().is_err());

        let p = Parser::new("J1/");
        assert!(p.parse_posix_datetime().is_err());

        let p = Parser::new("1/");
        assert!(p.parse_posix_datetime().is_err());

        let p = Parser::new("M4.1.0/");
        assert!(p.parse_posix_datetime().is_err());
    }

    #[test]
    fn parse_posix_date() {
        let p = Parser::new("J1");
        assert_eq!(p.parse_posix_date().unwrap(), PosixDay::JulianOne(1));
        let p = Parser::new("J365");
        assert_eq!(p.parse_posix_date().unwrap(), PosixDay::JulianOne(365));

        let p = Parser::new("0");
        assert_eq!(p.parse_posix_date().unwrap(), PosixDay::JulianZero(0));
        let p = Parser::new("1");
        assert_eq!(p.parse_posix_date().unwrap(), PosixDay::JulianZero(1));
        let p = Parser::new("365");
        assert_eq!(p.parse_posix_date().unwrap(), PosixDay::JulianZero(365));

        let p = Parser::new("M9.5.0");
        assert_eq!(
            p.parse_posix_date().unwrap(),
            PosixDay::WeekdayOfMonth { month: 9, week: 5, weekday: 0 },
        );
        let p = Parser::new("M9.5.6");
        assert_eq!(
            p.parse_posix_date().unwrap(),
            PosixDay::WeekdayOfMonth { month: 9, week: 5, weekday: 6 },
        );
        let p = Parser::new("M09.5.6");
        assert_eq!(
            p.parse_posix_date().unwrap(),
            PosixDay::WeekdayOfMonth { month: 9, week: 5, weekday: 6 },
        );
        let p = Parser::new("M12.1.1");
        assert_eq!(
            p.parse_posix_date().unwrap(),
            PosixDay::WeekdayOfMonth { month: 12, week: 1, weekday: 1 },
        );

        let p = Parser::new("a");
        assert!(p.parse_posix_date().is_err());

        let p = Parser::new("j");
        assert!(p.parse_posix_date().is_err());

        let p = Parser::new("m");
        assert!(p.parse_posix_date().is_err());

        let p = Parser::new("n");
        assert!(p.parse_posix_date().is_err());

        let p = Parser::new("J366");
        assert!(p.parse_posix_date().is_err());

        let p = Parser::new("366");
        assert!(p.parse_posix_date().is_err());
    }

    #[test]
    fn parse_posix_julian_day_no_leap() {
        let p = Parser::new("1");
        assert_eq!(p.parse_posix_julian_day_no_leap().unwrap(), 1);

        let p = Parser::new("001");
        assert_eq!(p.parse_posix_julian_day_no_leap().unwrap(), 1);

        let p = Parser::new("365");
        assert_eq!(p.parse_posix_julian_day_no_leap().unwrap(), 365);

        let p = Parser::new("3655");
        assert_eq!(p.parse_posix_julian_day_no_leap().unwrap(), 365);

        let p = Parser::new("0");
        assert!(p.parse_posix_julian_day_no_leap().is_err());

        let p = Parser::new("366");
        assert!(p.parse_posix_julian_day_no_leap().is_err());
    }

    #[test]
    fn parse_posix_julian_day_with_leap() {
        let p = Parser::new("0");
        assert_eq!(p.parse_posix_julian_day_with_leap().unwrap(), 0);

        let p = Parser::new("1");
        assert_eq!(p.parse_posix_julian_day_with_leap().unwrap(), 1);

        let p = Parser::new("001");
        assert_eq!(p.parse_posix_julian_day_with_leap().unwrap(), 1);

        let p = Parser::new("365");
        assert_eq!(p.parse_posix_julian_day_with_leap().unwrap(), 365);

        let p = Parser::new("3655");
        assert_eq!(p.parse_posix_julian_day_with_leap().unwrap(), 365);

        let p = Parser::new("366");
        assert!(p.parse_posix_julian_day_with_leap().is_err());
    }

    #[test]
    fn parse_weekday_of_month() {
        let p = Parser::new("9.5.0");
        assert_eq!(p.parse_weekday_of_month().unwrap(), (9, 5, 0));

        let p = Parser::new("9.1.6");
        assert_eq!(p.parse_weekday_of_month().unwrap(), (9, 1, 6));

        let p = Parser::new("09.1.6");
        assert_eq!(p.parse_weekday_of_month().unwrap(), (9, 1, 6));

        let p = Parser::new("9");
        assert!(p.parse_weekday_of_month().is_err());

        let p = Parser::new("9.");
        assert!(p.parse_weekday_of_month().is_err());

        let p = Parser::new("9.5");
        assert!(p.parse_weekday_of_month().is_err());

        let p = Parser::new("9.5.");
        assert!(p.parse_weekday_of_month().is_err());

        let p = Parser::new("0.5.0");
        assert!(p.parse_weekday_of_month().is_err());

        let p = Parser::new("13.5.0");
        assert!(p.parse_weekday_of_month().is_err());

        let p = Parser::new("9.0.0");
        assert!(p.parse_weekday_of_month().is_err());

        let p = Parser::new("9.6.0");
        assert!(p.parse_weekday_of_month().is_err());

        let p = Parser::new("9.5.7");
        assert!(p.parse_weekday_of_month().is_err());
    }

    #[test]
    fn parse_posix_time() {
        let p = Parser::new("5");
        assert_eq!(p.parse_posix_time().unwrap().second, 5 * 60 * 60);

        let p = Parser::new("22");
        assert_eq!(p.parse_posix_time().unwrap().second, 22 * 60 * 60);

        let p = Parser::new("02");
        assert_eq!(p.parse_posix_time().unwrap().second, 2 * 60 * 60);

        let p = Parser::new("5:45");
        assert_eq!(
            p.parse_posix_time().unwrap().second,
            5 * 60 * 60 + 45 * 60
        );

        let p = Parser::new("5:45:12");
        assert_eq!(
            p.parse_posix_time().unwrap().second,
            5 * 60 * 60 + 45 * 60 + 12
        );

        let p = Parser::new("5:45:129");
        assert_eq!(
            p.parse_posix_time().unwrap().second,
            5 * 60 * 60 + 45 * 60 + 12
        );

        let p = Parser::new("5:45:12:");
        assert_eq!(
            p.parse_posix_time().unwrap().second,
            5 * 60 * 60 + 45 * 60 + 12
        );

        let p = Parser { ianav3plus: true, ..Parser::new("+5:45:12") };
        assert_eq!(
            p.parse_posix_time().unwrap().second,
            5 * 60 * 60 + 45 * 60 + 12
        );

        let p = Parser { ianav3plus: true, ..Parser::new("-5:45:12") };
        assert_eq!(
            p.parse_posix_time().unwrap().second,
            -(5 * 60 * 60 + 45 * 60 + 12)
        );

        let p = Parser { ianav3plus: true, ..Parser::new("-167:45:12") };
        assert_eq!(
            p.parse_posix_time().unwrap().second,
            -(167 * 60 * 60 + 45 * 60 + 12),
        );

        let p = Parser::new("25");
        assert!(p.parse_posix_time().is_err());

        let p = Parser::new("12:2");
        assert!(p.parse_posix_time().is_err());

        let p = Parser::new("12:");
        assert!(p.parse_posix_time().is_err());

        let p = Parser::new("12:23:5");
        assert!(p.parse_posix_time().is_err());

        let p = Parser::new("12:23:");
        assert!(p.parse_posix_time().is_err());

        let p = Parser { ianav3plus: true, ..Parser::new("168") };
        assert!(p.parse_posix_time().is_err());

        let p = Parser { ianav3plus: true, ..Parser::new("-168") };
        assert!(p.parse_posix_time().is_err());

        let p = Parser { ianav3plus: true, ..Parser::new("+168") };
        assert!(p.parse_posix_time().is_err());
    }

    #[test]
    fn parse_month() {
        let p = Parser::new("1");
        assert_eq!(p.parse_month().unwrap(), 1);

        // Should this be allowed? POSIX spec is unclear.
        // We allow it because our parse does stop at 2
        // digits, so this seems harmless. Namely, '001'
        // results in an error.
        let p = Parser::new("01");
        assert_eq!(p.parse_month().unwrap(), 1);

        let p = Parser::new("12");
        assert_eq!(p.parse_month().unwrap(), 12);

        let p = Parser::new("0");
        assert!(p.parse_month().is_err());

        let p = Parser::new("00");
        assert!(p.parse_month().is_err());

        let p = Parser::new("001");
        assert!(p.parse_month().is_err());

        let p = Parser::new("13");
        assert!(p.parse_month().is_err());
    }

    #[test]
    fn parse_week() {
        let p = Parser::new("1");
        assert_eq!(p.parse_week().unwrap(), 1);

        let p = Parser::new("5");
        assert_eq!(p.parse_week().unwrap(), 5);

        let p = Parser::new("55");
        assert_eq!(p.parse_week().unwrap(), 5);

        let p = Parser::new("0");
        assert!(p.parse_week().is_err());

        let p = Parser::new("6");
        assert!(p.parse_week().is_err());

        let p = Parser::new("00");
        assert!(p.parse_week().is_err());

        let p = Parser::new("01");
        assert!(p.parse_week().is_err());

        let p = Parser::new("05");
        assert!(p.parse_week().is_err());
    }

    #[test]
    fn parse_weekday() {
        let p = Parser::new("0");
        assert_eq!(p.parse_weekday().unwrap(), 0);

        let p = Parser::new("1");
        assert_eq!(p.parse_weekday().unwrap(), 1);

        let p = Parser::new("6");
        assert_eq!(p.parse_weekday().unwrap(), 6);

        let p = Parser::new("00");
        assert_eq!(p.parse_weekday().unwrap(), 0);

        let p = Parser::new("06");
        assert_eq!(p.parse_weekday().unwrap(), 0);

        let p = Parser::new("60");
        assert_eq!(p.parse_weekday().unwrap(), 6);

        let p = Parser::new("7");
        assert!(p.parse_weekday().is_err());
    }

    #[test]
    fn parse_hour_posix() {
        let p = Parser::new("5");
        assert_eq!(p.parse_hour_posix().unwrap(), 5);

        let p = Parser::new("0");
        assert_eq!(p.parse_hour_posix().unwrap(), 0);

        let p = Parser::new("00");
        assert_eq!(p.parse_hour_posix().unwrap(), 0);

        let p = Parser::new("24");
        assert_eq!(p.parse_hour_posix().unwrap(), 24);

        let p = Parser::new("100");
        assert_eq!(p.parse_hour_posix().unwrap(), 10);

        let p = Parser::new("25");
        assert!(p.parse_hour_posix().is_err());

        let p = Parser::new("99");
        assert!(p.parse_hour_posix().is_err());
    }

    #[test]
    fn parse_hour_ianav3plus() {
        let new = |input| Parser { ianav3plus: true, ..Parser::new(input) };

        let p = new("5");
        assert_eq!(p.parse_hour_ianav3plus().unwrap(), 5);

        let p = new("0");
        assert_eq!(p.parse_hour_ianav3plus().unwrap(), 0);

        let p = new("00");
        assert_eq!(p.parse_hour_ianav3plus().unwrap(), 0);

        let p = new("000");
        assert_eq!(p.parse_hour_ianav3plus().unwrap(), 0);

        let p = new("24");
        assert_eq!(p.parse_hour_ianav3plus().unwrap(), 24);

        let p = new("100");
        assert_eq!(p.parse_hour_ianav3plus().unwrap(), 100);

        let p = new("1000");
        assert_eq!(p.parse_hour_ianav3plus().unwrap(), 100);

        let p = new("167");
        assert_eq!(p.parse_hour_ianav3plus().unwrap(), 167);

        let p = new("168");
        assert!(p.parse_hour_ianav3plus().is_err());

        let p = new("999");
        assert!(p.parse_hour_ianav3plus().is_err());
    }

    #[test]
    fn parse_minute() {
        let p = Parser::new("00");
        assert_eq!(p.parse_minute().unwrap(), 0);

        let p = Parser::new("24");
        assert_eq!(p.parse_minute().unwrap(), 24);

        let p = Parser::new("59");
        assert_eq!(p.parse_minute().unwrap(), 59);

        let p = Parser::new("599");
        assert_eq!(p.parse_minute().unwrap(), 59);

        let p = Parser::new("0");
        assert!(p.parse_minute().is_err());

        let p = Parser::new("1");
        assert!(p.parse_minute().is_err());

        let p = Parser::new("9");
        assert!(p.parse_minute().is_err());

        let p = Parser::new("60");
        assert!(p.parse_minute().is_err());
    }

    #[test]
    fn parse_second() {
        let p = Parser::new("00");
        assert_eq!(p.parse_second().unwrap(), 0);

        let p = Parser::new("24");
        assert_eq!(p.parse_second().unwrap(), 24);

        let p = Parser::new("59");
        assert_eq!(p.parse_second().unwrap(), 59);

        let p = Parser::new("599");
        assert_eq!(p.parse_second().unwrap(), 59);

        let p = Parser::new("0");
        assert!(p.parse_second().is_err());

        let p = Parser::new("1");
        assert!(p.parse_second().is_err());

        let p = Parser::new("9");
        assert!(p.parse_second().is_err());

        let p = Parser::new("60");
        assert!(p.parse_second().is_err());
    }

    #[test]
    fn parse_number_with_exactly_n_digits() {
        let p = Parser::new("1");
        assert_eq!(p.parse_number_with_exactly_n_digits(1).unwrap(), 1);

        let p = Parser::new("12");
        assert_eq!(p.parse_number_with_exactly_n_digits(2).unwrap(), 12);

        let p = Parser::new("123");
        assert_eq!(p.parse_number_with_exactly_n_digits(2).unwrap(), 12);

        let p = Parser::new("");
        assert!(p.parse_number_with_exactly_n_digits(1).is_err());

        let p = Parser::new("1");
        assert!(p.parse_number_with_exactly_n_digits(2).is_err());

        let p = Parser::new("12");
        assert!(p.parse_number_with_exactly_n_digits(3).is_err());
    }

    #[test]
    fn parse_number_with_upto_n_digits() {
        let p = Parser::new("1");
        assert_eq!(p.parse_number_with_upto_n_digits(1).unwrap(), 1);

        let p = Parser::new("1");
        assert_eq!(p.parse_number_with_upto_n_digits(2).unwrap(), 1);

        let p = Parser::new("12");
        assert_eq!(p.parse_number_with_upto_n_digits(2).unwrap(), 12);

        let p = Parser::new("12");
        assert_eq!(p.parse_number_with_upto_n_digits(3).unwrap(), 12);

        let p = Parser::new("123");
        assert_eq!(p.parse_number_with_upto_n_digits(2).unwrap(), 12);

        let p = Parser::new("");
        assert!(p.parse_number_with_upto_n_digits(1).is_err());

        let p = Parser::new("a");
        assert!(p.parse_number_with_upto_n_digits(1).is_err());
    }

    #[test]
    fn to_dst_civil_datetime_utc_range() {
        let tz = posix_time_zone("WART4WARST,J1/-3,J365/20");
        let dst_info = DstInfo {
            // We test this in other places. It's too annoying to write this
            // out here, and I didn't adopt snapshot testing until I had
            // written out these tests by hand. ¯\_(ツ)_/¯
            dst: tz.dst.as_ref().unwrap(),
            start: date(2024, 1, 1).at(1, 0, 0, 0),
            end: date(2024, 12, 31).at(23, 0, 0, 0),
        };
        assert_eq!(tz.dst_info_utc(2024), Some(dst_info));

        let tz = posix_time_zone("WART4WARST,J1/-4,J365/21");
        let dst_info = DstInfo {
            dst: tz.dst.as_ref().unwrap(),
            start: date(2024, 1, 1).at(0, 0, 0, 0),
            end: date(2024, 12, 31).at(23, 59, 59, 999_999_999),
        };
        assert_eq!(tz.dst_info_utc(2024), Some(dst_info));

        let tz = posix_time_zone("EST5EDT,M3.2.0,M11.1.0");
        let dst_info = DstInfo {
            dst: tz.dst.as_ref().unwrap(),
            start: date(2024, 3, 10).at(7, 0, 0, 0),
            end: date(2024, 11, 3).at(6, 0, 0, 0),
        };
        assert_eq!(tz.dst_info_utc(2024), Some(dst_info));
    }

    // See: https://github.com/BurntSushi/jiff/issues/386
    #[test]
    fn regression_permanent_dst() {
        let tz = posix_time_zone("XXX-2<+01>-1,0/0,J365/23");
        let dst_info = DstInfo {
            dst: tz.dst.as_ref().unwrap(),
            start: date(2087, 1, 1).at(0, 0, 0, 0),
            end: date(2087, 12, 31).at(23, 59, 59, 999_999_999),
        };
        assert_eq!(tz.dst_info_utc(2087), Some(dst_info));
    }

    #[test]
    fn reasonable() {
        assert!(PosixTimeZone::parse(b"EST5").is_ok());
        assert!(PosixTimeZone::parse(b"EST5EDT").is_err());
        assert!(PosixTimeZone::parse(b"EST5EDT,J1,J365").is_ok());

        let tz = posix_time_zone("EST24EDT,J1,J365");
        assert_eq!(
            tz,
            PosixTimeZone {
                std_abbrev: "EST".into(),
                std_offset: PosixOffset { second: -24 * 60 * 60 },
                dst: Some(PosixDst {
                    abbrev: "EDT".into(),
                    offset: PosixOffset { second: -23 * 60 * 60 },
                    rule: PosixRule {
                        start: PosixDayTime {
                            date: PosixDay::JulianOne(1),
                            time: PosixTime::DEFAULT,
                        },
                        end: PosixDayTime {
                            date: PosixDay::JulianOne(365),
                            time: PosixTime::DEFAULT,
                        },
                    },
                }),
            },
        );

        let tz = posix_time_zone("EST-24EDT,J1,J365");
        assert_eq!(
            tz,
            PosixTimeZone {
                std_abbrev: "EST".into(),
                std_offset: PosixOffset { second: 24 * 60 * 60 },
                dst: Some(PosixDst {
                    abbrev: "EDT".into(),
                    offset: PosixOffset { second: 25 * 60 * 60 },
                    rule: PosixRule {
                        start: PosixDayTime {
                            date: PosixDay::JulianOne(1),
                            time: PosixTime::DEFAULT,
                        },
                        end: PosixDayTime {
                            date: PosixDay::JulianOne(365),
                            time: PosixTime::DEFAULT,
                        },
                    },
                }),
            },
        );
    }

    #[test]
    fn posix_date_time_spec_to_datetime() {
        // For this test, we just keep the offset to zero to simplify things
        // a bit. We get coverage for non-zero offsets in higher level tests.
        let to_datetime = |daytime: &PosixDayTime, year: i16| {
            daytime.to_datetime(year, IOffset::UTC)
        };

        let tz = posix_time_zone("EST5EDT,J1,J365/5:12:34");
        assert_eq!(
            to_datetime(&tz.rule().start, 2023),
            date(2023, 1, 1).at(2, 0, 0, 0),
        );
        assert_eq!(
            to_datetime(&tz.rule().end, 2023),
            date(2023, 12, 31).at(5, 12, 34, 0),
        );

        let tz = posix_time_zone("EST+5EDT,M3.2.0/2,M11.1.0/2");
        assert_eq!(
            to_datetime(&tz.rule().start, 2024),
            date(2024, 3, 10).at(2, 0, 0, 0),
        );
        assert_eq!(
            to_datetime(&tz.rule().end, 2024),
            date(2024, 11, 3).at(2, 0, 0, 0),
        );

        let tz = posix_time_zone("EST+5EDT,M1.1.1,M12.5.2");
        assert_eq!(
            to_datetime(&tz.rule().start, 2024),
            date(2024, 1, 1).at(2, 0, 0, 0),
        );
        assert_eq!(
            to_datetime(&tz.rule().end, 2024),
            date(2024, 12, 31).at(2, 0, 0, 0),
        );

        let tz = posix_time_zone("EST5EDT,0/0,J365/25");
        assert_eq!(
            to_datetime(&tz.rule().start, 2024),
            date(2024, 1, 1).at(0, 0, 0, 0),
        );
        assert_eq!(
            to_datetime(&tz.rule().end, 2024),
            date(2024, 12, 31).at(23, 59, 59, 999_999_999),
        );

        let tz = posix_time_zone("XXX3EDT4,0/0,J365/23");
        assert_eq!(
            to_datetime(&tz.rule().start, 2024),
            date(2024, 1, 1).at(0, 0, 0, 0),
        );
        assert_eq!(
            to_datetime(&tz.rule().end, 2024),
            date(2024, 12, 31).at(23, 0, 0, 0),
        );

        let tz = posix_time_zone("XXX3EDT4,0/0,365");
        assert_eq!(
            to_datetime(&tz.rule().end, 2023),
            date(2023, 12, 31).at(23, 59, 59, 999_999_999),
        );
        assert_eq!(
            to_datetime(&tz.rule().end, 2024),
            date(2024, 12, 31).at(2, 0, 0, 0),
        );

        let tz = posix_time_zone("XXX3EDT4,J1/-167:59:59,J365/167:59:59");
        assert_eq!(
            to_datetime(&tz.rule().start, 2024),
            date(2024, 1, 1).at(0, 0, 0, 0),
        );
        assert_eq!(
            to_datetime(&tz.rule().end, 2024),
            date(2024, 12, 31).at(23, 59, 59, 999_999_999),
        );
    }

    #[test]
    fn posix_date_time_spec_time() {
        let tz = posix_time_zone("EST5EDT,J1,J365/5:12:34");
        assert_eq!(tz.rule().start.time, PosixTime::DEFAULT);
        assert_eq!(
            tz.rule().end.time,
            PosixTime { second: 5 * 60 * 60 + 12 * 60 + 34 },
        );
    }

    #[test]
    fn posix_date_spec_to_date() {
        let tz = posix_time_zone("EST+5EDT,M3.2.0/2,M11.1.0/2");
        let start = tz.rule().start.date.to_date(2023);
        assert_eq!(start, Some(date(2023, 3, 12)));
        let end = tz.rule().end.date.to_date(2023);
        assert_eq!(end, Some(date(2023, 11, 5)));
        let start = tz.rule().start.date.to_date(2024);
        assert_eq!(start, Some(date(2024, 3, 10)));
        let end = tz.rule().end.date.to_date(2024);
        assert_eq!(end, Some(date(2024, 11, 3)));

        let tz = posix_time_zone("EST+5EDT,J60,J365");
        let start = tz.rule().start.date.to_date(2023);
        assert_eq!(start, Some(date(2023, 3, 1)));
        let end = tz.rule().end.date.to_date(2023);
        assert_eq!(end, Some(date(2023, 12, 31)));
        let start = tz.rule().start.date.to_date(2024);
        assert_eq!(start, Some(date(2024, 3, 1)));
        let end = tz.rule().end.date.to_date(2024);
        assert_eq!(end, Some(date(2024, 12, 31)));

        let tz = posix_time_zone("EST+5EDT,59,365");
        let start = tz.rule().start.date.to_date(2023);
        assert_eq!(start, Some(date(2023, 3, 1)));
        let end = tz.rule().end.date.to_date(2023);
        assert_eq!(end, None);
        let start = tz.rule().start.date.to_date(2024);
        assert_eq!(start, Some(date(2024, 2, 29)));
        let end = tz.rule().end.date.to_date(2024);
        assert_eq!(end, Some(date(2024, 12, 31)));

        let tz = posix_time_zone("EST+5EDT,M1.1.1,M12.5.2");
        let start = tz.rule().start.date.to_date(2024);
        assert_eq!(start, Some(date(2024, 1, 1)));
        let end = tz.rule().end.date.to_date(2024);
        assert_eq!(end, Some(date(2024, 12, 31)));
    }

    #[test]
    fn posix_time_spec_to_civil_time() {
        let tz = posix_time_zone("EST5EDT,J1,J365/5:12:34");
        assert_eq!(
            tz.dst.as_ref().unwrap().rule.start.time.second,
            2 * 60 * 60,
        );
        assert_eq!(
            tz.dst.as_ref().unwrap().rule.end.time.second,
            5 * 60 * 60 + 12 * 60 + 34,
        );

        let tz = posix_time_zone("EST5EDT,J1/23:59:59,J365/24:00:00");
        assert_eq!(
            tz.dst.as_ref().unwrap().rule.start.time.second,
            23 * 60 * 60 + 59 * 60 + 59,
        );
        assert_eq!(
            tz.dst.as_ref().unwrap().rule.end.time.second,
            24 * 60 * 60,
        );

        let tz = posix_time_zone("EST5EDT,J1/-1,J365/167:00:00");
        assert_eq!(
            tz.dst.as_ref().unwrap().rule.start.time.second,
            -1 * 60 * 60,
        );
        assert_eq!(
            tz.dst.as_ref().unwrap().rule.end.time.second,
            167 * 60 * 60,
        );
    }

    #[test]
    fn parse_iana() {
        // Ref: https://github.com/chronotope/chrono/issues/1153
        let p = PosixTimeZone::parse(b"CRAZY5SHORT,M12.5.0/50,0/2").unwrap();
        assert_eq!(
            p,
            PosixTimeZone {
                std_abbrev: "CRAZY".into(),
                std_offset: PosixOffset { second: -5 * 60 * 60 },
                dst: Some(PosixDst {
                    abbrev: "SHORT".into(),
                    offset: PosixOffset { second: -4 * 60 * 60 },
                    rule: PosixRule {
                        start: PosixDayTime {
                            date: PosixDay::WeekdayOfMonth {
                                month: 12,
                                week: 5,
                                weekday: 0,
                            },
                            time: PosixTime { second: 50 * 60 * 60 },
                        },
                        end: PosixDayTime {
                            date: PosixDay::JulianZero(0),
                            time: PosixTime { second: 2 * 60 * 60 },
                        },
                    },
                }),
            },
        );

        assert!(PosixTimeZone::parse(b"America/New_York").is_err());
        assert!(PosixTimeZone::parse(b":America/New_York").is_err());
    }

    // See: https://github.com/BurntSushi/jiff/issues/407
    #[test]
    fn parse_empty_is_err() {
        assert!(PosixTimeZone::parse(b"").is_err());
    }

    // See: https://github.com/BurntSushi/jiff/issues/407
    #[test]
    fn parse_weird_is_err() {
        let s =
            b"AAAAAAAAAAAAAAACAAAAAAAAAAAAQA8AACAAAAAAAAAAAAAAAAACAAAAAAAAAAA";
        assert!(PosixTimeZone::parse(s).is_err());

        let s =
            b"<AAAAAAAAAAAAAAACAAAAAAAAAAAAQA>8<AACAAAAAAAAAAAAAAAAACAAAAAAAAAAA>";
        assert!(PosixTimeZone::parse(s).is_err());

        let s = b"PPPPPPPPPPPPPPPPPPPPnoofPPPAAA6DaPPPPPPPPPPPPPPPPPPPPPnoofPPPPP,n";
        assert!(PosixTimeZone::parse(s).is_err());

        let s = b"oooooooooovooooooooooooooooool9<ooooo2o-o-oooooookoorooooooooroo8";
        assert!(PosixTimeZone::parse(s).is_err());
    }
}
