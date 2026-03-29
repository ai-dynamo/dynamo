use crate::{
    error::{
        fmt::strtime::{Error as E, FormatError as FE},
        ErrorContext,
    },
    fmt::{
        buffer::BorrowedWriter,
        strtime::{
            month_name_abbrev, month_name_full, weekday_name_abbrev,
            weekday_name_full, BrokenDownTime, Config, Custom, Extension,
            Flag, Meridiem,
        },
    },
    tz::Offset,
    util::utf8,
    Error,
};

enum Item {
    AlreadyFormatted,
    Integer(ItemInteger),
    Fraction(ItemFraction),
    String(ItemString),
    Offset(ItemOffset),
}

struct ItemInteger {
    pad_byte: u8,
    pad_width: u8,
    number: i64,
}

impl ItemInteger {
    fn new(pad_byte: u8, pad_width: u8, n: impl Into<i64>) -> ItemInteger {
        ItemInteger { pad_byte, pad_width, number: n.into() }
    }
}

struct ItemFraction {
    width: Option<u8>,
    dot: bool,
    subsec: u32,
}

struct ItemString {
    case: Case,
    string: &'static str,
}

impl ItemString {
    fn new(case: Case, string: &'static str) -> ItemString {
        ItemString { case, string }
    }
}

struct ItemOffset {
    offset: Offset,
    colon: bool,
    minute: bool,
    second: bool,
}

impl ItemOffset {
    fn new(
        offset: Offset,
        colon: bool,
        minute: bool,
        second: bool,
    ) -> ItemOffset {
        ItemOffset { offset, colon, minute, second }
    }
}

pub(super) struct Formatter<
    'config,
    'fmt,
    'tm,
    'writer,
    'buffer,
    'data,
    'write,
    L,
> {
    pub(super) config: &'config Config<L>,
    pub(super) fmt: &'fmt [u8],
    pub(super) tm: &'tm BrokenDownTime,
    pub(super) wtr: &'writer mut BorrowedWriter<'buffer, 'data, 'write>,
}

impl<'config, 'fmt, 'tm, 'writer, 'buffer, 'data, 'write, L: Custom>
    Formatter<'config, 'fmt, 'tm, 'writer, 'buffer, 'data, 'write, L>
{
    #[inline(never)]
    pub(super) fn format(&mut self) -> Result<(), Error> {
        while !self.fmt.is_empty() {
            if self.f() != b'%' {
                if self.f().is_ascii() {
                    self.wtr.write_ascii_char(self.f())?;
                    self.bump_fmt();
                } else {
                    let ch = self.utf8_decode_and_bump()?;
                    self.wtr.write_char(ch)?;
                }
                continue;
            }
            if !self.bump_fmt() {
                if self.config.lenient {
                    self.wtr.write_ascii_char(b'%')?;
                    break;
                }
                return Err(E::UnexpectedEndAfterPercent.into());
            }
            let orig = self.fmt;
            if let Err(err) =
                self.parse_extension().and_then(|ext| self.format_one(&ext))
            {
                if !self.config.lenient {
                    return Err(err);
                }
                // `orig` is whatever failed to parse immediately after a `%`.
                // Since it failed, we write out the `%` and then proceed to
                // handle what failed to parse literally.
                self.wtr.write_ascii_char(b'%')?;
                // Reset back to right after parsing the `%`.
                self.fmt = orig;
            }
        }
        Ok(())
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn format_one(&mut self, ext: &Extension) -> Result<(), Error> {
        let i = |item: ItemInteger| Item::Integer(item);
        let s = |item: ItemString| Item::String(item);
        let o = |item: ItemOffset| Item::Offset(item);
        let failc =
            |directive, colons| E::DirectiveFailure { directive, colons };
        let fail = |directive| failc(directive, 0);

        // Parse extensions like padding/case options and padding width.
        let mut directive = self.f();
        let item = match directive {
            b'%' => self.fmt_literal("%").map(s).context(fail(b'%')),
            b'A' => self.fmt_weekday_full().map(s).context(fail(b'A')),
            b'a' => self.fmt_weekday_abbrev().map(s).context(fail(b'a')),
            b'B' => self.fmt_month_full().map(s).context(fail(b'B')),
            b'b' => self.fmt_month_abbrev().map(s).context(fail(b'b')),
            b'C' => self.fmt_century().map(i).context(fail(b'C')),
            b'c' => self.fmt_datetime(ext).context(fail(b'c')),
            b'D' => self.fmt_american_date().context(fail(b'D')),
            b'd' => self.fmt_day_zero().map(i).context(fail(b'd')),
            b'e' => self.fmt_day_space().map(i).context(fail(b'e')),
            b'F' => self.fmt_iso_date().context(fail(b'F')),
            b'f' => self.fmt_fractional(ext).context(fail(b'f')),
            b'G' => self.fmt_iso_week_year().map(i).context(fail(b'G')),
            b'g' => self.fmt_iso_week_year2().map(i).context(fail(b'g')),
            b'H' => self.fmt_hour24_zero().map(i).context(fail(b'H')),
            b'h' => self.fmt_month_abbrev().map(s).context(fail(b'b')),
            b'I' => self.fmt_hour12_zero().map(i).context(fail(b'H')),
            b'j' => self.fmt_day_of_year().map(i).context(fail(b'j')),
            b'k' => self.fmt_hour24_space().map(i).context(fail(b'k')),
            b'l' => self.fmt_hour12_space().map(i).context(fail(b'l')),
            b'M' => self.fmt_minute().map(i).context(fail(b'M')),
            b'm' => self.fmt_month().map(i).context(fail(b'm')),
            b'N' => self.fmt_nanoseconds(ext).context(fail(b'N')),
            b'n' => self.fmt_literal("\n").map(s).context(fail(b'n')),
            b'P' => self.fmt_ampm_lower().map(s).context(fail(b'P')),
            b'p' => self.fmt_ampm_upper(ext).map(s).context(fail(b'p')),
            b'Q' => match ext.colons {
                0 => self.fmt_iana_nocolon().context(fail(b'Q')),
                1 => self.fmt_iana_colon().context(failc(b'Q', 1)),
                _ => return Err(E::ColonCount { directive: b'Q' }.into()),
            },
            b'q' => self.fmt_quarter().map(i).context(fail(b'q')),
            b'R' => self.fmt_clock_nosecs().context(fail(b'R')),
            b'r' => self.fmt_12hour_time(ext).context(fail(b'r')),
            b'S' => self.fmt_second().map(i).context(fail(b'S')),
            b's' => self.fmt_timestamp().map(i).context(fail(b's')),
            b'T' => self.fmt_clock_secs().context(fail(b'T')),
            b't' => self.fmt_literal("\t").map(s).context(fail(b't')),
            b'U' => self.fmt_week_sun().map(i).context(fail(b'U')),
            b'u' => self.fmt_weekday_mon().map(i).context(fail(b'u')),
            b'V' => self.fmt_week_iso().map(i).context(fail(b'V')),
            b'W' => self.fmt_week_mon().map(i).context(fail(b'W')),
            b'w' => self.fmt_weekday_sun().map(i).context(fail(b'w')),
            b'X' => self.fmt_time(ext).context(fail(b'X')),
            b'x' => self.fmt_date(ext).context(fail(b'x')),
            b'Y' => self.fmt_year().map(i).context(fail(b'Y')),
            b'y' => self.fmt_year2().map(i).context(fail(b'y')),
            b'Z' => self.fmt_tzabbrev(ext).context(fail(b'Z')),
            b'z' => match ext.colons {
                0 => self.fmt_offset_nocolon().map(o).context(fail(b'z')),
                1 => self.fmt_offset_colon().map(o).context(failc(b'z', 1)),
                2 => self.fmt_offset_colon2().map(o).context(failc(b'z', 2)),
                3 => self.fmt_offset_colon3().map(o).context(failc(b'z', 3)),
                _ => return Err(E::ColonCount { directive: b'z' }.into()),
            },
            b'.' => {
                if !self.bump_fmt() {
                    return Err(E::UnexpectedEndAfterDot.into());
                }
                // Parse precision settings after the `.`, effectively
                // overriding any digits that came before it.
                let ext =
                    &Extension { width: self.parse_width()?, ..ext.clone() };
                directive = self.f();
                match directive {
                    b'f' => self
                        .fmt_dot_fractional(ext)
                        .context(E::DirectiveFailureDot { directive }),
                    _ => {
                        return Err(Error::from(
                            E::UnknownDirectiveAfterDot { directive },
                        ));
                    }
                }
            }
            _ => return Err(Error::from(E::UnknownDirective { directive })),
        }?;
        self.write_item(ext, &item).context(fail(directive))?;
        self.bump_fmt();
        Ok(())
    }

    /// Returns the byte at the current position of the format string.
    ///
    /// # Panics
    ///
    /// This panics when the entire format string has been consumed.
    fn f(&self) -> u8 {
        self.fmt[0]
    }

    /// Bumps the position of the format string.
    ///
    /// This returns true in precisely the cases where `self.f()` will not
    /// panic. i.e., When the end of the format string hasn't been reached yet.
    fn bump_fmt(&mut self) -> bool {
        self.fmt = &self.fmt[1..];
        !self.fmt.is_empty()
    }

    /// Decodes a Unicode scalar value from the beginning of `fmt` and advances
    /// the parser accordingly.
    ///
    /// If a Unicode scalar value could not be decoded, then an error is
    /// returned.
    ///
    /// It would be nice to just pass through bytes as-is instead of doing
    /// actual UTF-8 decoding, but since the `Write` trait only represents
    /// Unicode-accepting buffers, we need to actually do decoding here.
    ///
    /// # Errors
    ///
    /// Unless lenient parsing is enabled, this returns an error if UTF-8
    /// decoding failed. When lenient parsing is enabled, decoding errors
    /// are turned into the Unicode replacement codepoint via the
    /// "substitution of maximal subparts" strategy.
    ///
    /// # Panics
    ///
    /// When `self.fmt` is empty. i.e., Only call this when you know there is
    /// some remaining bytes to parse.
    #[cold]
    #[inline(never)]
    fn utf8_decode_and_bump(&mut self) -> Result<char, FE> {
        match utf8::decode(self.fmt).expect("non-empty fmt") {
            Ok(ch) => {
                self.fmt = &self.fmt[ch.len_utf8()..];
                return Ok(ch);
            }
            Err(err) if self.config.lenient => {
                self.fmt = &self.fmt[err.len()..];
                return Ok(char::REPLACEMENT_CHARACTER);
            }
            Err(_) => Err(FE::InvalidUtf8),
        }
    }

    /// Parses optional extensions before a specifier directive. That is, right
    /// after the `%`. If any extensions are parsed, the parser is bumped
    /// to the next byte. (If no next byte exists, then an error is returned.)
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_extension(&mut self) -> Result<Extension, Error> {
        if self.f().is_ascii_alphabetic() {
            return Ok(Extension { flag: None, width: None, colons: 0 });
        }
        let flag = self.parse_flag()?;
        let width = self.parse_width()?;
        let colons = self.parse_colons()?;
        Ok(Extension { flag, width, colons })
    }

    /// Parses an optional flag. And if one is parsed, the parser is bumped
    /// to the next byte. (If no next byte exists, then an error is returned.)
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_flag(&mut self) -> Result<Option<Flag>, Error> {
        let (flag, fmt) = Extension::parse_flag(self.fmt)?;
        self.fmt = fmt;
        Ok(flag)
    }

    /// Parses an optional width that comes after a (possibly absent) flag and
    /// before the specifier directive itself. And if a width is parsed, the
    /// parser is bumped to the next byte. (If no next byte exists, then an
    /// error is returned.)
    ///
    /// Note that this is also used to parse precision settings for `%f` and
    /// `%.f`. In the former case, the width is just re-interpreted as a
    /// precision setting. In the latter case, something like `%5.9f` is
    /// technically valid, but the `5` is ignored.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_width(&mut self) -> Result<Option<u8>, Error> {
        let (width, fmt) = Extension::parse_width(self.fmt)?;
        self.fmt = fmt;
        Ok(width)
    }

    /// Parses an optional number of colons (up to 3) immediately before a
    /// conversion specifier.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn parse_colons(&mut self) -> Result<u8, Error> {
        let (colons, fmt) = Extension::parse_colons(self.fmt)?;
        self.fmt = fmt;
        Ok(colons)
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn write_item(
        &mut self,
        ext: &Extension,
        item: &Item,
    ) -> Result<(), Error> {
        match *item {
            Item::AlreadyFormatted => Ok(()),
            Item::Integer(ref item) => ext.write_int(
                item.pad_byte,
                item.pad_width,
                item.number,
                self.wtr,
            ),
            Item::Fraction(ItemFraction { width, dot, subsec }) => {
                if dot {
                    self.wtr.write_ascii_char(b'.')?;
                }
                self.wtr.write_fraction(width, subsec)
            }
            Item::String(ref item) => {
                ext.write_str(item.case, item.string, self.wtr)
            }
            Item::Offset(ref item) => write_offset(
                item.offset,
                item.colon,
                item.minute,
                item.second,
                self.wtr,
            ),
        }
    }

    // These are the formatting functions. They are pretty much responsible
    // for getting what they need for the broken down time and reporting a
    // decent failure mode if what they need couldn't be found. And then,
    // of course, doing the actual formatting.

    /// %P
    fn fmt_ampm_lower(&self) -> Result<ItemString, Error> {
        let meridiem = match self.tm.meridiem() {
            Some(meridiem) => meridiem,
            None => {
                let hour = self.tm.hour().ok_or(FE::RequiresTime)?;
                if hour < 12 {
                    Meridiem::AM
                } else {
                    Meridiem::PM
                }
            }
        };
        let s = match meridiem {
            Meridiem::AM => "am",
            Meridiem::PM => "pm",
        };
        Ok(ItemString::new(Case::AsIs, s))
    }

    /// %p
    fn fmt_ampm_upper(&self, ext: &Extension) -> Result<ItemString, Error> {
        let meridiem = match self.tm.meridiem() {
            Some(meridiem) => meridiem,
            None => {
                let hour = self.tm.hour().ok_or(FE::RequiresTime)?;
                if hour < 12 {
                    Meridiem::AM
                } else {
                    Meridiem::PM
                }
            }
        };
        // Manually specialize this case to avoid hitting `write_str_cold`.
        let s = if matches!(ext.flag, Some(Flag::Swapcase)) {
            match meridiem {
                Meridiem::AM => "am",
                Meridiem::PM => "pm",
            }
        } else {
            match meridiem {
                Meridiem::AM => "AM",
                Meridiem::PM => "PM",
            }
        };
        Ok(ItemString::new(Case::AsIs, s))
    }

    /// %D
    fn fmt_american_date(&mut self) -> Result<Item, Error> {
        let ItemInteger { number, .. } = self.fmt_month()?;
        self.wtr.write_int_pad2(number.unsigned_abs())?;
        self.wtr.write_ascii_char(b'/')?;
        let ItemInteger { number, .. } = self.fmt_day_zero()?;
        self.wtr.write_int_pad2(number.unsigned_abs())?;
        self.wtr.write_ascii_char(b'/')?;
        let ItemInteger { number, .. } = self.fmt_year2()?;
        if number < 0 {
            self.wtr.write_ascii_char(b'-')?;
        }
        self.wtr.write_int_pad2(number.unsigned_abs())?;
        Ok(Item::AlreadyFormatted)
    }

    /// %R
    fn fmt_clock_nosecs(&mut self) -> Result<Item, Error> {
        let ItemInteger { number, .. } = self.fmt_hour24_zero()?;
        self.wtr.write_int_pad2(number.unsigned_abs())?;
        self.wtr.write_ascii_char(b':')?;
        let ItemInteger { number, .. } = self.fmt_minute()?;
        self.wtr.write_int_pad2(number.unsigned_abs())?;
        Ok(Item::AlreadyFormatted)
    }

    /// %T
    fn fmt_clock_secs(&mut self) -> Result<Item, Error> {
        let ItemInteger { number, .. } = self.fmt_hour24_zero()?;
        self.wtr.write_int_pad2(number.unsigned_abs())?;
        self.wtr.write_ascii_char(b':')?;
        let ItemInteger { number, .. } = self.fmt_minute()?;
        self.wtr.write_int_pad2(number.unsigned_abs())?;
        self.wtr.write_ascii_char(b':')?;
        let ItemInteger { number, .. } = self.fmt_second()?;
        self.wtr.write_int_pad2(number.unsigned_abs())?;
        Ok(Item::AlreadyFormatted)
    }

    /// %F
    fn fmt_iso_date(&mut self) -> Result<Item, Error> {
        let ItemInteger { number, .. } = self.fmt_year()?;
        if number < 0 {
            self.wtr.write_ascii_char(b'-')?;
        }
        self.wtr.write_int_pad4(number.unsigned_abs())?;
        self.wtr.write_ascii_char(b'-')?;
        let ItemInteger { number, .. } = self.fmt_month()?;
        self.wtr.write_int_pad2(number.unsigned_abs())?;
        self.wtr.write_ascii_char(b'-')?;
        let ItemInteger { number, .. } = self.fmt_day_zero()?;
        self.wtr.write_int_pad2(number.unsigned_abs())?;
        Ok(Item::AlreadyFormatted)
    }

    /// %d
    fn fmt_day_zero(&self) -> Result<ItemInteger, Error> {
        let day = self
            .tm
            .day()
            .or_else(|| self.tm.to_date().ok().map(|d| d.day()))
            .ok_or(FE::RequiresDate)?;
        Ok(ItemInteger::new(b'0', 2, day))
    }

    /// %e
    fn fmt_day_space(&self) -> Result<ItemInteger, Error> {
        let day = self
            .tm
            .day()
            .or_else(|| self.tm.to_date().ok().map(|d| d.day()))
            .ok_or(FE::RequiresDate)?;
        Ok(ItemInteger::new(b' ', 2, day))
    }

    /// %I
    fn fmt_hour12_zero(&self) -> Result<ItemInteger, Error> {
        let mut hour = self.tm.hour().ok_or(FE::RequiresTime)?;
        if hour == 0 {
            hour = 12;
        } else if hour > 12 {
            hour -= 12;
        }
        Ok(ItemInteger::new(b'0', 2, hour))
    }

    /// %H
    fn fmt_hour24_zero(&self) -> Result<ItemInteger, Error> {
        let hour = self.tm.hour().ok_or(FE::RequiresTime)?;
        Ok(ItemInteger::new(b'0', 2, hour))
    }

    /// %l
    fn fmt_hour12_space(&self) -> Result<ItemInteger, Error> {
        let mut hour = self.tm.hour().ok_or(FE::RequiresTime)?;
        if hour == 0 {
            hour = 12;
        } else if hour > 12 {
            hour -= 12;
        }
        Ok(ItemInteger::new(b' ', 2, hour))
    }

    /// %k
    fn fmt_hour24_space(&self) -> Result<ItemInteger, Error> {
        let hour = self.tm.hour().ok_or(FE::RequiresTime)?;
        Ok(ItemInteger::new(b' ', 2, hour))
    }

    /// %M
    fn fmt_minute(&self) -> Result<ItemInteger, Error> {
        let minute = self.tm.minute().ok_or(FE::RequiresTime)?;
        Ok(ItemInteger::new(b'0', 2, minute))
    }

    /// %m
    fn fmt_month(&self) -> Result<ItemInteger, Error> {
        let month = self
            .tm
            .month()
            .or_else(|| self.tm.to_date().ok().map(|d| d.month()))
            .ok_or(FE::RequiresDate)?;
        Ok(ItemInteger::new(b'0', 2, month))
    }

    /// %B
    fn fmt_month_full(&self) -> Result<ItemString, Error> {
        let month = self
            .tm
            .month()
            .or_else(|| self.tm.to_date().ok().map(|d| d.month()))
            .ok_or(FE::RequiresDate)?;
        Ok(ItemString::new(Case::AsIs, month_name_full(month)))
    }

    /// %b, %h
    fn fmt_month_abbrev(&self) -> Result<ItemString, Error> {
        let month = self
            .tm
            .month()
            .or_else(|| self.tm.to_date().ok().map(|d| d.month()))
            .ok_or(FE::RequiresDate)?;
        Ok(ItemString::new(Case::AsIs, month_name_abbrev(month)))
    }

    /// %Q
    fn fmt_iana_nocolon(&mut self) -> Result<Item, Error> {
        let Some(iana) = self.tm.iana_time_zone() else {
            let offset = self.tm.offset.ok_or(FE::RequiresTimeZoneOrOffset)?;
            return Ok(Item::Offset(ItemOffset::new(
                offset, false, true, false,
            )));
        };
        self.wtr.write_str(iana)?;
        Ok(Item::AlreadyFormatted)
    }

    /// %:Q
    fn fmt_iana_colon(&mut self) -> Result<Item, Error> {
        let Some(iana) = self.tm.iana_time_zone() else {
            let offset = self.tm.offset.ok_or(FE::RequiresTimeZoneOrOffset)?;
            return Ok(Item::Offset(ItemOffset::new(
                offset, true, true, false,
            )));
        };
        self.wtr.write_str(iana)?;
        Ok(Item::AlreadyFormatted)
    }

    /// %z
    fn fmt_offset_nocolon(&self) -> Result<ItemOffset, Error> {
        let offset = self.tm.offset.ok_or(FE::RequiresOffset)?;
        Ok(ItemOffset::new(offset, false, true, false))
    }

    /// %:z
    fn fmt_offset_colon(&self) -> Result<ItemOffset, Error> {
        let offset = self.tm.offset.ok_or(FE::RequiresOffset)?;
        Ok(ItemOffset::new(offset, true, true, false))
    }

    /// %::z
    fn fmt_offset_colon2(&self) -> Result<ItemOffset, Error> {
        let offset = self.tm.offset.ok_or(FE::RequiresOffset)?;
        Ok(ItemOffset::new(offset, true, true, true))
    }

    /// %:::z
    fn fmt_offset_colon3(&self) -> Result<ItemOffset, Error> {
        let offset = self.tm.offset.ok_or(FE::RequiresOffset)?;
        Ok(ItemOffset::new(offset, true, false, false))
    }

    /// %S
    fn fmt_second(&self) -> Result<ItemInteger, Error> {
        let second = self.tm.second().ok_or(FE::RequiresTime)?;
        Ok(ItemInteger::new(b'0', 2, second))
    }

    /// %s
    fn fmt_timestamp(&self) -> Result<ItemInteger, Error> {
        let timestamp =
            self.tm.to_timestamp().map_err(|_| FE::RequiresInstant)?;
        Ok(ItemInteger::new(b' ', 0, timestamp.as_second()))
    }

    /// %f
    fn fmt_fractional(&self, ext: &Extension) -> Result<Item, Error> {
        let subsec = self.tm.subsec.ok_or(FE::RequiresTime)?;
        let subsec = i32::from(subsec).unsigned_abs();
        // For %f, we always want to emit at least one digit. The only way we
        // wouldn't is if our fractional component is zero. One exception to
        // this is when the width is `0` (which looks like `%00f`), in which
        // case, we emit an error. We could allow it to emit an empty string,
        // but this seems very odd. And an empty string cannot be parsed by
        // `%f`.
        if ext.width == Some(0) {
            return Err(Error::from(FE::ZeroPrecisionFloat));
        }
        if subsec == 0 && ext.width.is_none() {
            return Ok(Item::String(ItemString::new(Case::AsIs, "0")));
        }
        Ok(Item::Fraction(ItemFraction {
            width: ext.width,
            dot: false,
            subsec,
        }))
    }

    /// %.f
    fn fmt_dot_fractional(&self, ext: &Extension) -> Result<Item, Error> {
        let Some(subsec) = self.tm.subsec else {
            return Ok(Item::AlreadyFormatted);
        };
        let subsec = i32::from(subsec).unsigned_abs();
        if subsec == 0 && ext.width.is_none() || ext.width == Some(0) {
            return Ok(Item::AlreadyFormatted);
        }
        Ok(Item::Fraction(ItemFraction {
            width: ext.width,
            dot: true,
            subsec,
        }))
    }

    /// %N
    fn fmt_nanoseconds(&self, ext: &Extension) -> Result<Item, Error> {
        let subsec = self.tm.subsec.ok_or(FE::RequiresTime)?;
        if ext.width == Some(0) {
            return Err(Error::from(FE::ZeroPrecisionNano));
        }
        let subsec = i32::from(subsec).unsigned_abs();
        // Since `%N` is actually an alias for `%9f`, when the precision
        // is missing, we default to 9.
        if ext.width.is_none() {
            return Ok(Item::Fraction(ItemFraction {
                width: Some(9),
                dot: false,
                subsec,
            }));
        }
        Ok(Item::Fraction(ItemFraction {
            width: ext.width,
            dot: false,
            subsec,
        }))
    }

    /// %Z
    fn fmt_tzabbrev(&mut self, ext: &Extension) -> Result<Item, Error> {
        let tz = self.tm.tz.as_ref().ok_or(FE::RequiresTimeZone)?;
        let ts = self.tm.to_timestamp().map_err(|_| FE::RequiresInstant)?;
        let oinfo = tz.to_offset_info(ts);
        ext.write_str(Case::Upper, oinfo.abbreviation(), self.wtr)?;
        Ok(Item::AlreadyFormatted)
    }

    /// %A
    fn fmt_weekday_full(&self) -> Result<ItemString, Error> {
        let weekday = self
            .tm
            .weekday
            .or_else(|| self.tm.to_date().ok().map(|d| d.weekday()))
            .ok_or(FE::RequiresDate)?;
        Ok(ItemString::new(Case::AsIs, weekday_name_full(weekday)))
    }

    /// %a
    fn fmt_weekday_abbrev(&self) -> Result<ItemString, Error> {
        let weekday = self
            .tm
            .weekday
            .or_else(|| self.tm.to_date().ok().map(|d| d.weekday()))
            .ok_or(FE::RequiresDate)?;
        Ok(ItemString::new(Case::AsIs, weekday_name_abbrev(weekday)))
    }

    /// %u
    fn fmt_weekday_mon(&self) -> Result<ItemInteger, Error> {
        let weekday = self
            .tm
            .weekday
            .or_else(|| self.tm.to_date().ok().map(|d| d.weekday()))
            .ok_or(FE::RequiresDate)?;
        Ok(ItemInteger::new(b' ', 0, weekday.to_monday_one_offset()))
    }

    /// %w
    fn fmt_weekday_sun(&self) -> Result<ItemInteger, Error> {
        let weekday = self
            .tm
            .weekday
            .or_else(|| self.tm.to_date().ok().map(|d| d.weekday()))
            .ok_or(FE::RequiresDate)?;
        Ok(ItemInteger::new(b' ', 0, weekday.to_sunday_zero_offset()))
    }

    /// %U
    fn fmt_week_sun(&self) -> Result<ItemInteger, Error> {
        // Short circuit if the week number was explicitly set.
        if let Some(weeknum) = self.tm.week_sun {
            return Ok(ItemInteger::new(b'0', 2, weeknum));
        }
        let day = self
            .tm
            .day_of_year()
            .or_else(|| self.tm.to_date().ok().map(|d| d.day_of_year()))
            .ok_or(FE::RequiresDate)?;
        let weekday = self
            .tm
            .weekday
            .or_else(|| self.tm.to_date().ok().map(|d| d.weekday()))
            .ok_or(FE::RequiresDate)?
            .to_sunday_zero_offset();
        // Example: 2025-01-05 is the first Sunday in 2025, and thus the start
        // of week 1. This means that 2025-01-04 (Saturday) is in week 0.
        //
        // So for 2025-01-05, day=5 and weekday=0. Thus we get 11/7 = 1.
        // For 2025-01-04, day=4 and weekday=6. Thus we get 4/7 = 0.
        let weeknum = (day + 6 - i16::from(weekday)) / 7;
        Ok(ItemInteger::new(b'0', 2, weeknum))
    }

    /// %V
    fn fmt_week_iso(&self) -> Result<ItemInteger, Error> {
        let weeknum = self
            .tm
            .iso_week()
            .or_else(|| {
                self.tm.to_date().ok().map(|d| d.iso_week_date().week())
            })
            .ok_or(FE::RequiresDate)?;
        Ok(ItemInteger::new(b'0', 2, weeknum))
    }

    /// %W
    fn fmt_week_mon(&self) -> Result<ItemInteger, Error> {
        // Short circuit if the week number was explicitly set.
        if let Some(weeknum) = self.tm.week_mon {
            return Ok(ItemInteger::new(b'0', 2, weeknum));
        }
        let day = self
            .tm
            .day_of_year()
            .or_else(|| self.tm.to_date().ok().map(|d| d.day_of_year()))
            .ok_or(FE::RequiresDate)?;
        let weekday = self
            .tm
            .weekday
            .or_else(|| self.tm.to_date().ok().map(|d| d.weekday()))
            .ok_or(FE::RequiresDate)?
            .to_sunday_zero_offset();
        // Example: 2025-01-06 is the first Monday in 2025, and thus the start
        // of week 1. This means that 2025-01-05 (Sunday) is in week 0.
        //
        // So for 2025-01-06, day=6 and weekday=1. Thus we get 12/7 = 1.
        // For 2025-01-05, day=5 and weekday=7. Thus we get 5/7 = 0.
        let weeknum = (day + 6 - ((i16::from(weekday) + 6) % 7)) / 7;
        Ok(ItemInteger::new(b'0', 2, weeknum))
    }

    /// %Y
    fn fmt_year(&self) -> Result<ItemInteger, Error> {
        let year = self
            .tm
            .year()
            .or_else(|| self.tm.to_date().ok().map(|d| d.year()))
            .ok_or(FE::RequiresDate)?;
        Ok(ItemInteger::new(b'0', 4, year))
    }

    /// %y
    fn fmt_year2(&self) -> Result<ItemInteger, Error> {
        let year = self
            .tm
            .year()
            .or_else(|| self.tm.to_date().ok().map(|d| d.year()))
            .ok_or(FE::RequiresDate)?;
        let year = year % 100;
        Ok(ItemInteger::new(b'0', 2, year))
    }

    /// %C
    fn fmt_century(&self) -> Result<ItemInteger, Error> {
        let year = self
            .tm
            .year()
            .or_else(|| self.tm.to_date().ok().map(|d| d.year()))
            .ok_or(FE::RequiresDate)?;
        let century = year / 100;
        Ok(ItemInteger::new(b' ', 0, century))
    }

    /// %G
    fn fmt_iso_week_year(&self) -> Result<ItemInteger, Error> {
        let year = self
            .tm
            .iso_week_year()
            .or_else(|| {
                self.tm.to_date().ok().map(|d| d.iso_week_date().year())
            })
            .ok_or(FE::RequiresDate)?;
        Ok(ItemInteger::new(b'0', 4, year))
    }

    /// %g
    fn fmt_iso_week_year2(&self) -> Result<ItemInteger, Error> {
        let year = self
            .tm
            .iso_week_year()
            .or_else(|| {
                self.tm.to_date().ok().map(|d| d.iso_week_date().year())
            })
            .ok_or(FE::RequiresDate)?;
        let year = year % 100;
        Ok(ItemInteger::new(b'0', 2, year))
    }

    /// %q
    fn fmt_quarter(&self) -> Result<ItemInteger, Error> {
        let month = self
            .tm
            .month()
            .or_else(|| self.tm.to_date().ok().map(|d| d.month()))
            .ok_or(FE::RequiresDate)?;
        let quarter = match month {
            1..=3 => 1,
            4..=6 => 2,
            7..=9 => 3,
            10..=12 => 4,
            _ => unreachable!(),
        };
        Ok(ItemInteger::new(b'0', 0, quarter))
    }

    /// %j
    fn fmt_day_of_year(&self) -> Result<ItemInteger, Error> {
        let day = self
            .tm
            .day_of_year()
            .or_else(|| self.tm.to_date().ok().map(|d| d.day_of_year()))
            .ok_or(FE::RequiresDate)?;
        Ok(ItemInteger::new(b'0', 3, day))
    }

    /// %n, %t
    fn fmt_literal(&self, literal: &'static str) -> Result<ItemString, Error> {
        Ok(ItemString::new(Case::AsIs, literal))
    }

    /// %c
    fn fmt_datetime(&mut self, ext: &Extension) -> Result<Item, Error> {
        self.config.custom.format_datetime(
            self.config,
            ext,
            self.tm,
            self.wtr,
        )?;
        Ok(Item::AlreadyFormatted)
    }

    /// %x
    fn fmt_date(&mut self, ext: &Extension) -> Result<Item, Error> {
        self.config.custom.format_date(self.config, ext, self.tm, self.wtr)?;
        Ok(Item::AlreadyFormatted)
    }

    /// %X
    fn fmt_time(&mut self, ext: &Extension) -> Result<Item, Error> {
        self.config.custom.format_time(self.config, ext, self.tm, self.wtr)?;
        Ok(Item::AlreadyFormatted)
    }

    /// %r
    fn fmt_12hour_time(&mut self, ext: &Extension) -> Result<Item, Error> {
        self.config.custom.format_12hour_time(
            self.config,
            ext,
            self.tm,
            self.wtr,
        )?;
        Ok(Item::AlreadyFormatted)
    }
}

/// Writes the given time zone offset to the writer.
///
/// When `colon` is true, the hour, minute and optional second components are
/// delimited by a colon. Otherwise, no delimiter is used.
///
/// When `minute` is true, the minute component is always printed. When
/// false, the minute component is only printed when it is non-zero (or if
/// the second component is non-zero).
///
/// When `second` is true, the second component is always printed. When false,
/// the second component is only printed when it is non-zero.
#[cfg_attr(feature = "perf-inline", inline(always))]
fn write_offset(
    offset: Offset,
    colon: bool,
    minute: bool,
    second: bool,
    wtr: &mut BorrowedWriter<'_, '_, '_>,
) -> Result<(), Error> {
    let total_seconds = offset.seconds().unsigned_abs();
    let hours = (total_seconds / (60 * 60)) as u8;
    let minutes = ((total_seconds / 60) % 60) as u8;
    let seconds = (total_seconds % 60) as u8;

    wtr.write_ascii_char(if offset.is_negative() { b'-' } else { b'+' })?;
    wtr.write_int_pad2(hours)?;
    if minute || minutes != 0 || seconds != 0 {
        if colon {
            wtr.write_ascii_char(b':')?;
        }
        wtr.write_int_pad2(minutes)?;
        if second || seconds != 0 {
            if colon {
                wtr.write_ascii_char(b':')?;
            }
            wtr.write_int_pad2(seconds)?;
        }
    }
    Ok(())
}

impl Extension {
    /// Writes the given string using the default case rule provided, unless
    /// an option in this extension config overrides the default case.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn write_str(
        &self,
        default: Case,
        string: &str,
        wtr: &mut BorrowedWriter<'_, '_, '_>,
    ) -> Result<(), Error> {
        if self.flag.is_none() && matches!(default, Case::AsIs) {
            return wtr.write_str(string);
        }
        self.write_str_cold(default, string, wtr)
    }

    #[cold]
    #[inline(never)]
    fn write_str_cold(
        &self,
        default: Case,
        string: &str,
        wtr: &mut BorrowedWriter<'_, '_, '_>,
    ) -> Result<(), Error> {
        let case = match self.flag {
            Some(Flag::Uppercase) => Case::Upper,
            Some(Flag::Swapcase) => default.swap(),
            _ => default,
        };
        match case {
            Case::AsIs => {
                wtr.write_str(string)?;
            }
            Case::Upper | Case::Lower => {
                for ch in string.chars() {
                    wtr.write_char(if matches!(case, Case::Upper) {
                        ch.to_ascii_uppercase()
                    } else {
                        ch.to_ascii_lowercase()
                    })?;
                }
            }
        }
        Ok(())
    }

    /// Writes the given integer using the given padding width and byte, unless
    /// an option in this extension config overrides a default setting.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn write_int(
        &self,
        pad_byte: u8,
        pad_width: u8,
        number: impl Into<i64>,
        wtr: &mut BorrowedWriter<'_, '_, '_>,
    ) -> Result<(), Error> {
        let number = number.into();
        let pad_byte = match self.flag {
            Some(Flag::PadZero) => b'0',
            Some(Flag::PadSpace) => b' ',
            _ => pad_byte,
        };
        let pad_width = if matches!(self.flag, Some(Flag::NoPad)) {
            0
        } else {
            self.width.unwrap_or(pad_width)
        };
        if number < 0 {
            // Special case formatting a negative integer with non-zero
            // padding. This case isn't handled by the `BorrowedWriter`
            // API, so just handle it specially instead of bloating up
            // the hot path.
            if pad_width > 0 {
                return Self::write_negative_int(
                    pad_byte,
                    pad_width,
                    number.unsigned_abs(),
                    wtr,
                );
            }
            wtr.write_ascii_char(b'-')?;
        }
        let number = number.unsigned_abs();
        match (pad_byte, pad_width) {
            (b'0', 2) if number <= 99 => wtr.write_int_pad2(number),
            (b' ', 2) if number <= 99 => wtr.write_int_pad2_space(number),
            (b'0', 4) if number <= 9999 => wtr.write_int_pad4(number),
            _ => wtr.write_int_pad(number, pad_byte, pad_width),
        }
    }

    /// Writes a negative integer with explicit width, where sign is part of
    /// total width.
    #[cold]
    #[inline(never)]
    fn write_negative_int(
        pad_byte: u8,
        pad_width: u8,
        number: u64,
        wtr: &mut BorrowedWriter<'_, '_, '_>,
    ) -> Result<(), Error> {
        // The sign is part of the total width, so offset the
        // padding to account for it.
        let mut pad_width = pad_width.saturating_sub(1);
        // When padding to spaces, we want the padding first
        // and then the `-` sign. So handle the padding here
        // explicitly. For padding to zeros, we handle it
        // normally (other than offsetting the padding width
        // above).
        if pad_byte == b' ' {
            let d = 1 + number.checked_ilog10().unwrap_or(0) as u8;
            for _ in 0..pad_width.saturating_sub(d) {
                wtr.write_ascii_char(b' ')?;
            }
            // Don't do any padding to the call below.
            // We could instead add a `BorrowedWriter::write_int`
            // routine, but I don't think it's worth doing for
            // this case.
            pad_width = 0;
        }
        wtr.write_ascii_char(b'-')?;
        wtr.write_int_pad(number, pad_byte, pad_width)
    }
}

/// The case to use when printing a string like weekday or TZ abbreviation.
#[derive(Clone, Copy, Debug)]
enum Case {
    AsIs,
    Upper,
    Lower,
}

impl Case {
    /// Swap upper to lowercase, and lower to uppercase.
    fn swap(self) -> Case {
        match self {
            Case::AsIs => Case::AsIs,
            Case::Upper => Case::Lower,
            Case::Lower => Case::Upper,
        }
    }
}

#[cfg(feature = "alloc")]
#[cfg(test)]
mod tests {
    use crate::{
        civil::{date, time, Date, DateTime, Time},
        fmt::strtime::{format, BrokenDownTime, Config, PosixCustom},
        tz::Offset,
        Timestamp, Zoned,
    };

    #[test]
    fn ok_format_american_date() {
        let f = |fmt: &str, date: Date| format(fmt, date).unwrap();

        insta::assert_snapshot!(f("%D", date(2024, 7, 9)), @"07/09/24");
        insta::assert_snapshot!(f("%-D", date(2024, 7, 9)), @"07/09/24");
        insta::assert_snapshot!(f("%3D", date(2024, 7, 9)), @"07/09/24");
        insta::assert_snapshot!(f("%03D", date(2024, 7, 9)), @"07/09/24");
    }

    #[test]
    fn ok_format_ampm() {
        let f = |fmt: &str, time: Time| format(fmt, time).unwrap();

        insta::assert_snapshot!(f("%H%P", time(9, 0, 0, 0)), @"09am");
        insta::assert_snapshot!(f("%H%P", time(11, 0, 0, 0)), @"11am");
        insta::assert_snapshot!(f("%H%P", time(23, 0, 0, 0)), @"23pm");
        insta::assert_snapshot!(f("%H%P", time(0, 0, 0, 0)), @"00am");

        insta::assert_snapshot!(f("%H%p", time(9, 0, 0, 0)), @"09AM");
        insta::assert_snapshot!(f("%H%p", time(11, 0, 0, 0)), @"11AM");
        insta::assert_snapshot!(f("%H%p", time(23, 0, 0, 0)), @"23PM");
        insta::assert_snapshot!(f("%H%p", time(0, 0, 0, 0)), @"00AM");

        insta::assert_snapshot!(f("%H%#p", time(9, 0, 0, 0)), @"09am");
    }

    #[test]
    fn ok_format_clock() {
        let f = |fmt: &str, time: Time| format(fmt, time).unwrap();

        insta::assert_snapshot!(f("%R", time(23, 59, 8, 0)), @"23:59");
        insta::assert_snapshot!(f("%T", time(23, 59, 8, 0)), @"23:59:08");
    }

    #[test]
    fn ok_format_day() {
        let f = |fmt: &str, date: Date| format(fmt, date).unwrap();

        insta::assert_snapshot!(f("%d", date(2024, 7, 9)), @"09");
        insta::assert_snapshot!(f("%0d", date(2024, 7, 9)), @"09");
        insta::assert_snapshot!(f("%-d", date(2024, 7, 9)), @"9");
        insta::assert_snapshot!(f("%_d", date(2024, 7, 9)), @" 9");

        insta::assert_snapshot!(f("%e", date(2024, 7, 9)), @" 9");
        insta::assert_snapshot!(f("%0e", date(2024, 7, 9)), @"09");
        insta::assert_snapshot!(f("%-e", date(2024, 7, 9)), @"9");
        insta::assert_snapshot!(f("%_e", date(2024, 7, 9)), @" 9");
    }

    #[test]
    fn ok_format_iso_date() {
        let f = |fmt: &str, date: Date| format(fmt, date).unwrap();

        insta::assert_snapshot!(f("%F", date(2024, 7, 9)), @"2024-07-09");
        insta::assert_snapshot!(f("%-F", date(2024, 7, 9)), @"2024-07-09");
        insta::assert_snapshot!(f("%3F", date(2024, 7, 9)), @"2024-07-09");
        insta::assert_snapshot!(f("%03F", date(2024, 7, 9)), @"2024-07-09");
    }

    #[test]
    fn ok_format_hour() {
        let f = |fmt: &str, time: Time| format(fmt, time).unwrap();

        insta::assert_snapshot!(f("%H", time(9, 0, 0, 0)), @"09");
        insta::assert_snapshot!(f("%H", time(11, 0, 0, 0)), @"11");
        insta::assert_snapshot!(f("%H", time(23, 0, 0, 0)), @"23");
        insta::assert_snapshot!(f("%H", time(0, 0, 0, 0)), @"00");

        insta::assert_snapshot!(f("%I", time(9, 0, 0, 0)), @"09");
        insta::assert_snapshot!(f("%I", time(11, 0, 0, 0)), @"11");
        insta::assert_snapshot!(f("%I", time(23, 0, 0, 0)), @"11");
        insta::assert_snapshot!(f("%I", time(0, 0, 0, 0)), @"12");

        insta::assert_snapshot!(f("%k", time(9, 0, 0, 0)), @" 9");
        insta::assert_snapshot!(f("%k", time(11, 0, 0, 0)), @"11");
        insta::assert_snapshot!(f("%k", time(23, 0, 0, 0)), @"23");
        insta::assert_snapshot!(f("%k", time(0, 0, 0, 0)), @" 0");

        insta::assert_snapshot!(f("%l", time(9, 0, 0, 0)), @" 9");
        insta::assert_snapshot!(f("%l", time(11, 0, 0, 0)), @"11");
        insta::assert_snapshot!(f("%l", time(23, 0, 0, 0)), @"11");
        insta::assert_snapshot!(f("%l", time(0, 0, 0, 0)), @"12");
    }

    #[test]
    fn ok_format_minute() {
        let f = |fmt: &str, time: Time| format(fmt, time).unwrap();

        insta::assert_snapshot!(f("%M", time(0, 9, 0, 0)), @"09");
        insta::assert_snapshot!(f("%M", time(0, 11, 0, 0)), @"11");
        insta::assert_snapshot!(f("%M", time(0, 23, 0, 0)), @"23");
        insta::assert_snapshot!(f("%M", time(0, 0, 0, 0)), @"00");
    }

    #[test]
    fn ok_format_month() {
        let f = |fmt: &str, date: Date| format(fmt, date).unwrap();

        insta::assert_snapshot!(f("%m", date(2024, 7, 14)), @"07");
        insta::assert_snapshot!(f("%m", date(2024, 12, 14)), @"12");
        insta::assert_snapshot!(f("%0m", date(2024, 7, 14)), @"07");
        insta::assert_snapshot!(f("%0m", date(2024, 12, 14)), @"12");
        insta::assert_snapshot!(f("%-m", date(2024, 7, 14)), @"7");
        insta::assert_snapshot!(f("%-m", date(2024, 12, 14)), @"12");
        insta::assert_snapshot!(f("%_m", date(2024, 7, 14)), @" 7");
        insta::assert_snapshot!(f("%_m", date(2024, 12, 14)), @"12");
    }

    #[test]
    fn ok_format_month_name() {
        let f = |fmt: &str, date: Date| format(fmt, date).unwrap();

        insta::assert_snapshot!(f("%B", date(2024, 7, 14)), @"July");
        insta::assert_snapshot!(f("%b", date(2024, 7, 14)), @"Jul");
        insta::assert_snapshot!(f("%h", date(2024, 7, 14)), @"Jul");

        insta::assert_snapshot!(f("%#B", date(2024, 7, 14)), @"July");
        insta::assert_snapshot!(f("%^B", date(2024, 7, 14)), @"JULY");
    }

    #[test]
    fn ok_format_offset_from_zoned() {
        if crate::tz::db().is_definitively_empty() {
            return;
        }

        let f = |fmt: &str, zdt: &Zoned| format(fmt, zdt).unwrap();

        let zdt = date(2024, 7, 14)
            .at(22, 24, 0, 0)
            .in_tz("America/New_York")
            .unwrap();
        insta::assert_snapshot!(f("%z", &zdt), @"-0400");
        insta::assert_snapshot!(f("%:z", &zdt), @"-04:00");

        let zdt = zdt.checked_add(crate::Span::new().months(5)).unwrap();
        insta::assert_snapshot!(f("%z", &zdt), @"-0500");
        insta::assert_snapshot!(f("%:z", &zdt), @"-05:00");
    }

    #[test]
    fn ok_format_offset_plain() {
        let o = |h: i8, m: i8, s: i8| -> Offset { Offset::hms(h, m, s) };
        let f = |fmt: &str, offset: Offset| {
            let mut tm = BrokenDownTime::default();
            tm.set_offset(Some(offset));
            tm.to_string(fmt).unwrap()
        };

        insta::assert_snapshot!(f("%z", o(0, 0, 0)), @"+0000");
        insta::assert_snapshot!(f("%:z", o(0, 0, 0)), @"+00:00");
        insta::assert_snapshot!(f("%::z", o(0, 0, 0)), @"+00:00:00");
        insta::assert_snapshot!(f("%:::z", o(0, 0, 0)), @"+00");

        insta::assert_snapshot!(f("%z", -o(4, 0, 0)), @"-0400");
        insta::assert_snapshot!(f("%:z", -o(4, 0, 0)), @"-04:00");
        insta::assert_snapshot!(f("%::z", -o(4, 0, 0)), @"-04:00:00");
        insta::assert_snapshot!(f("%:::z", -o(4, 0, 0)), @"-04");

        insta::assert_snapshot!(f("%z", o(5, 30, 0)), @"+0530");
        insta::assert_snapshot!(f("%:z", o(5, 30, 0)), @"+05:30");
        insta::assert_snapshot!(f("%::z", o(5, 30, 0)), @"+05:30:00");
        insta::assert_snapshot!(f("%:::z", o(5, 30, 0)), @"+05:30");

        insta::assert_snapshot!(f("%z", o(5, 30, 15)), @"+053015");
        insta::assert_snapshot!(f("%:z", o(5, 30, 15)), @"+05:30:15");
        insta::assert_snapshot!(f("%::z", o(5, 30, 15)), @"+05:30:15");
        insta::assert_snapshot!(f("%:::z", o(5, 30, 15)), @"+05:30:15");

        insta::assert_snapshot!(f("%z", o(5, 0, 15)), @"+050015");
        insta::assert_snapshot!(f("%:z", o(5, 0, 15)), @"+05:00:15");
        insta::assert_snapshot!(f("%::z", o(5, 0, 15)), @"+05:00:15");
        insta::assert_snapshot!(f("%:::z", o(5, 0, 15)), @"+05:00:15");
    }

    #[test]
    fn ok_format_second() {
        let f = |fmt: &str, time: Time| format(fmt, time).unwrap();

        insta::assert_snapshot!(f("%S", time(0, 0, 9, 0)), @"09");
        insta::assert_snapshot!(f("%S", time(0, 0, 11, 0)), @"11");
        insta::assert_snapshot!(f("%S", time(0, 0, 23, 0)), @"23");
        insta::assert_snapshot!(f("%S", time(0, 0, 0, 0)), @"00");
    }

    #[test]
    fn ok_format_subsec_nanosecond() {
        let f = |fmt: &str, time: Time| format(fmt, time).unwrap();
        let mk = |subsec| time(0, 0, 0, subsec);

        insta::assert_snapshot!(f("%f", mk(123_000_000)), @"123");
        insta::assert_snapshot!(f("%f", mk(0)), @"0");
        insta::assert_snapshot!(f("%3f", mk(0)), @"000");
        insta::assert_snapshot!(f("%3f", mk(123_000_000)), @"123");
        insta::assert_snapshot!(f("%6f", mk(123_000_000)), @"123000");
        insta::assert_snapshot!(f("%9f", mk(123_000_000)), @"123000000");
        insta::assert_snapshot!(f("%255f", mk(123_000_000)), @"123000000");

        insta::assert_snapshot!(f("%.f", mk(123_000_000)), @".123");
        insta::assert_snapshot!(f("%.f", mk(0)), @"");
        insta::assert_snapshot!(f("%3.f", mk(0)), @"");
        insta::assert_snapshot!(f("%.3f", mk(0)), @".000");
        insta::assert_snapshot!(f("%.3f", mk(123_000_000)), @".123");
        insta::assert_snapshot!(f("%.6f", mk(123_000_000)), @".123000");
        insta::assert_snapshot!(f("%.9f", mk(123_000_000)), @".123000000");
        insta::assert_snapshot!(f("%.255f", mk(123_000_000)), @".123000000");

        insta::assert_snapshot!(f("%3f", mk(123_456_789)), @"123");
        insta::assert_snapshot!(f("%6f", mk(123_456_789)), @"123456");
        insta::assert_snapshot!(f("%9f", mk(123_456_789)), @"123456789");

        insta::assert_snapshot!(f("%.0f", mk(123_456_789)), @"");
        insta::assert_snapshot!(f("%.3f", mk(123_456_789)), @".123");
        insta::assert_snapshot!(f("%.6f", mk(123_456_789)), @".123456");
        insta::assert_snapshot!(f("%.9f", mk(123_456_789)), @".123456789");

        insta::assert_snapshot!(f("%N", mk(123_000_000)), @"123000000");
        insta::assert_snapshot!(f("%N", mk(0)), @"000000000");
        insta::assert_snapshot!(f("%N", mk(000_123_000)), @"000123000");
        insta::assert_snapshot!(f("%3N", mk(0)), @"000");
        insta::assert_snapshot!(f("%3N", mk(123_000_000)), @"123");
        insta::assert_snapshot!(f("%6N", mk(123_000_000)), @"123000");
        insta::assert_snapshot!(f("%9N", mk(123_000_000)), @"123000000");
        insta::assert_snapshot!(f("%255N", mk(123_000_000)), @"123000000");
    }

    #[test]
    fn ok_format_tzabbrev() {
        if crate::tz::db().is_definitively_empty() {
            return;
        }

        let f = |fmt: &str, zdt: &Zoned| format(fmt, zdt).unwrap();

        let zdt = date(2024, 7, 14)
            .at(22, 24, 0, 0)
            .in_tz("America/New_York")
            .unwrap();
        insta::assert_snapshot!(f("%Z", &zdt), @"EDT");
        insta::assert_snapshot!(f("%^Z", &zdt), @"EDT");
        insta::assert_snapshot!(f("%#Z", &zdt), @"edt");

        let zdt = zdt.checked_add(crate::Span::new().months(5)).unwrap();
        insta::assert_snapshot!(f("%Z", &zdt), @"EST");
    }

    #[test]
    fn ok_format_iana() {
        if crate::tz::db().is_definitively_empty() {
            return;
        }

        let f = |fmt: &str, zdt: &Zoned| format(fmt, zdt).unwrap();

        let zdt = date(2024, 7, 14)
            .at(22, 24, 0, 0)
            .in_tz("America/New_York")
            .unwrap();
        insta::assert_snapshot!(f("%Q", &zdt), @"America/New_York");
        insta::assert_snapshot!(f("%:Q", &zdt), @"America/New_York");

        let zdt = date(2024, 7, 14)
            .at(22, 24, 0, 0)
            .to_zoned(crate::tz::offset(-4).to_time_zone())
            .unwrap();
        insta::assert_snapshot!(f("%Q", &zdt), @"-0400");
        insta::assert_snapshot!(f("%:Q", &zdt), @"-04:00");

        let zdt = date(2024, 7, 14)
            .at(22, 24, 0, 0)
            .to_zoned(crate::tz::TimeZone::UTC)
            .unwrap();
        insta::assert_snapshot!(f("%Q", &zdt), @"UTC");
        insta::assert_snapshot!(f("%:Q", &zdt), @"UTC");
    }

    #[test]
    fn ok_format_weekday_name() {
        let f = |fmt: &str, date: Date| format(fmt, date).unwrap();

        insta::assert_snapshot!(f("%A", date(2024, 7, 14)), @"Sunday");
        insta::assert_snapshot!(f("%a", date(2024, 7, 14)), @"Sun");

        insta::assert_snapshot!(f("%#A", date(2024, 7, 14)), @"Sunday");
        insta::assert_snapshot!(f("%^A", date(2024, 7, 14)), @"SUNDAY");

        insta::assert_snapshot!(f("%u", date(2024, 7, 14)), @"7");
        insta::assert_snapshot!(f("%w", date(2024, 7, 14)), @"0");
    }

    #[test]
    fn ok_format_year() {
        let f = |fmt: &str, date: Date| format(fmt, date).unwrap();

        insta::assert_snapshot!(f("%Y", date(2024, 7, 14)), @"2024");
        insta::assert_snapshot!(f("%Y", date(24, 7, 14)), @"0024");
        insta::assert_snapshot!(f("%Y", date(-24, 7, 14)), @"-024");

        insta::assert_snapshot!(f("%C", date(2024, 7, 14)), @"20");
        insta::assert_snapshot!(f("%C", date(1815, 7, 14)), @"18");
        insta::assert_snapshot!(f("%C", date(915, 7, 14)), @"9");
        insta::assert_snapshot!(f("%C", date(1, 7, 14)), @"0");
        insta::assert_snapshot!(f("%C", date(0, 7, 14)), @"0");
        insta::assert_snapshot!(f("%C", date(-1, 7, 14)), @"0");
        insta::assert_snapshot!(f("%C", date(-2024, 7, 14)), @"-20");
        insta::assert_snapshot!(f("%C", date(-1815, 7, 14)), @"-18");
        insta::assert_snapshot!(f("%C", date(-915, 7, 14)), @"-9");
    }

    #[test]
    fn ok_format_year_negative_padded() {
        let f = |fmt: &str, date: Date| format(fmt, date).unwrap();

        insta::assert_snapshot!(f("%06Y", date(-2025, 1, 13)), @"-02025");
        insta::assert_snapshot!(f("%06Y", date(-25, 1, 13)), @"-00025");
        insta::assert_snapshot!(f("%06Y", date(-1, 1, 13)), @"-00001");
        insta::assert_snapshot!(f("%08Y", date(-2025, 1, 13)), @"-0002025");
        insta::assert_snapshot!(f("%_6Y", date(-2025, 1, 13)), @" -2025");
        insta::assert_snapshot!(f("%_6Y", date(-25, 1, 13)), @"   -25");
        insta::assert_snapshot!(f("%_6Y", date(-1, 1, 13)), @"    -1");
        insta::assert_snapshot!(f("%_8Y", date(-2025, 1, 13)), @"   -2025");
        insta::assert_snapshot!(f("%06Y", date(2025, 1, 13)), @"002025");
        insta::assert_snapshot!(f("%_6Y", date(2025, 1, 13)), @"  2025");
        insta::assert_snapshot!(f("%Y", date(-2025, 1, 13)), @"-2025");
        insta::assert_snapshot!(f("%Y", date(-25, 1, 13)), @"-025");
        insta::assert_snapshot!(f("%_Y", date(-2025, 1, 13)), @"-2025");
        insta::assert_snapshot!(f("%_Y", date(-25, 1, 13)), @" -25");
        insta::assert_snapshot!(f("%03Y", date(-2025, 1, 13)), @"-2025");
        insta::assert_snapshot!(f("%_3Y", date(-2025, 1, 13)), @"-2025");
        insta::assert_snapshot!(f("%-6Y", date(-2025, 1, 13)), @"-2025");
        insta::assert_snapshot!(f("%04Y", date(-2025, 1, 13)), @"-2025");
        insta::assert_snapshot!(f("%05Y", date(-2025, 1, 13)), @"-2025");
    }

    #[test]
    fn ok_format_default_locale() {
        let f = |fmt: &str, date: DateTime| format(fmt, date).unwrap();

        insta::assert_snapshot!(
            f("%c", date(2024, 7, 14).at(0, 0, 0, 0)),
            @"2024 M07 14, Sun 00:00:00",
        );
        insta::assert_snapshot!(
            f("%c", date(24, 7, 14).at(0, 0, 0, 0)),
            @"0024 M07 14, Sun 00:00:00",
        );
        insta::assert_snapshot!(
            f("%c", date(-24, 7, 14).at(0, 0, 0, 0)),
            @"-024 M07 14, Wed 00:00:00",
        );
        insta::assert_snapshot!(
            f("%c", date(2024, 7, 14).at(17, 31, 59, 123_456_789)),
            @"2024 M07 14, Sun 17:31:59",
        );

        insta::assert_snapshot!(
            f("%r", date(2024, 7, 14).at(8, 30, 0, 0)),
            @"8:30:00 AM",
        );
        insta::assert_snapshot!(
            f("%r", date(2024, 7, 14).at(17, 31, 59, 123_456_789)),
            @"5:31:59 PM",
        );

        insta::assert_snapshot!(
            f("%x", date(2024, 7, 14).at(0, 0, 0, 0)),
            @"2024 M07 14",
        );

        insta::assert_snapshot!(
            f("%X", date(2024, 7, 14).at(8, 30, 0, 0)),
            @"08:30:00",
        );
        insta::assert_snapshot!(
            f("%X", date(2024, 7, 14).at(17, 31, 59, 123_456_789)),
            @"17:31:59",
        );
    }

    #[test]
    fn ok_format_compound_uppercase() {
        let f = |fmt: &str, date: DateTime| format(fmt, date).unwrap();

        insta::assert_snapshot!(
            f("%^c", date(2024, 7, 14).at(0, 0, 0, 0)),
            @"2024 M07 14, SUN 00:00:00",
        );
        insta::assert_snapshot!(
            f("%^x", date(2024, 7, 14).at(0, 0, 0, 0)),
            @"2024 M07 14",
        );
        insta::assert_snapshot!(
            f("%^X", date(2024, 7, 14).at(8, 30, 0, 0)),
            @"08:30:00",
        );
        insta::assert_snapshot!(
            f("%^r", date(2024, 7, 14).at(8, 30, 0, 0)),
            @"8:30:00 AM",
        );
    }

    #[test]
    fn ok_format_posix_locale() {
        let f = |fmt: &str, date: DateTime| {
            let config = Config::new().custom(PosixCustom::default());
            let tm = BrokenDownTime::from(date);
            tm.to_string_with_config(&config, fmt).unwrap()
        };

        insta::assert_snapshot!(
            f("%c", date(2024, 7, 14).at(0, 0, 0, 0)),
            @"Sun Jul 14 00:00:00 2024",
        );
        insta::assert_snapshot!(
            f("%c", date(24, 7, 14).at(0, 0, 0, 0)),
            @"Sun Jul 14 00:00:00 0024",
        );
        insta::assert_snapshot!(
            f("%c", date(-24, 7, 14).at(0, 0, 0, 0)),
            @"Wed Jul 14 00:00:00 -024",
        );
        insta::assert_snapshot!(
            f("%c", date(2024, 7, 14).at(17, 31, 59, 123_456_789)),
            @"Sun Jul 14 17:31:59 2024",
        );

        insta::assert_snapshot!(
            f("%r", date(2024, 7, 14).at(8, 30, 0, 0)),
            @"08:30:00 AM",
        );
        insta::assert_snapshot!(
            f("%r", date(2024, 7, 14).at(17, 31, 59, 123_456_789)),
            @"05:31:59 PM",
        );

        insta::assert_snapshot!(
            f("%x", date(2024, 7, 14).at(0, 0, 0, 0)),
            @"07/14/24",
        );

        insta::assert_snapshot!(
            f("%X", date(2024, 7, 14).at(8, 30, 0, 0)),
            @"08:30:00",
        );
        insta::assert_snapshot!(
            f("%X", date(2024, 7, 14).at(17, 31, 59, 123_456_789)),
            @"17:31:59",
        );
    }

    #[test]
    fn ok_format_year_2digit() {
        let f = |fmt: &str, date: Date| format(fmt, date).unwrap();

        insta::assert_snapshot!(f("%y", date(2024, 7, 14)), @"24");
        insta::assert_snapshot!(f("%y", date(2001, 7, 14)), @"01");
        insta::assert_snapshot!(f("%-y", date(2001, 7, 14)), @"1");
        insta::assert_snapshot!(f("%5y", date(2001, 7, 14)), @"00001");
        insta::assert_snapshot!(f("%-5y", date(2001, 7, 14)), @"1");
        insta::assert_snapshot!(f("%05y", date(2001, 7, 14)), @"00001");
        insta::assert_snapshot!(f("%_y", date(2001, 7, 14)), @" 1");
        insta::assert_snapshot!(f("%_5y", date(2001, 7, 14)), @"    1");

        insta::assert_snapshot!(f("%y", date(1824, 7, 14)), @"24");
        insta::assert_snapshot!(f("%g", date(1824, 7, 14)), @"24");
    }

    #[test]
    fn ok_format_iso_week_year() {
        let f = |fmt: &str, date: Date| format(fmt, date).unwrap();

        insta::assert_snapshot!(f("%G", date(2019, 11, 30)), @"2019");
        insta::assert_snapshot!(f("%G", date(19, 11, 30)), @"0019");
        insta::assert_snapshot!(f("%G", date(-19, 11, 30)), @"-019");

        // tricksy
        insta::assert_snapshot!(f("%G", date(2019, 12, 30)), @"2020");
    }

    #[test]
    fn ok_format_week_num() {
        let f = |fmt: &str, date: Date| format(fmt, date).unwrap();

        insta::assert_snapshot!(f("%U", date(2025, 1, 4)), @"00");
        insta::assert_snapshot!(f("%U", date(2025, 1, 5)), @"01");

        insta::assert_snapshot!(f("%W", date(2025, 1, 5)), @"00");
        insta::assert_snapshot!(f("%W", date(2025, 1, 6)), @"01");
    }

    #[test]
    fn ok_format_timestamp() {
        let f = |fmt: &str, ts: Timestamp| format(fmt, ts).unwrap();

        let ts = "1970-01-01T00:00Z".parse().unwrap();
        insta::assert_snapshot!(f("%s", ts), @"0");
        insta::assert_snapshot!(f("%3s", ts), @"  0");
        insta::assert_snapshot!(f("%03s", ts), @"000");
        insta::assert_snapshot!(f("%2s", ts), @" 0");

        let ts = "2025-01-20T13:09-05[US/Eastern]".parse().unwrap();
        insta::assert_snapshot!(f("%s", ts), @"1737396540");
    }

    #[test]
    fn ok_format_quarter() {
        let f = |fmt: &str, date: Date| format(fmt, date).unwrap();

        insta::assert_snapshot!(f("%q", date(2024, 3, 31)), @"1");
        insta::assert_snapshot!(f("%q", date(2024, 4, 1)), @"2");
        insta::assert_snapshot!(f("%q", date(2024, 7, 14)), @"3");
        insta::assert_snapshot!(f("%q", date(2024, 12, 31)), @"4");

        insta::assert_snapshot!(f("%2q", date(2024, 3, 31)), @"01");
        insta::assert_snapshot!(f("%02q", date(2024, 3, 31)), @"01");
        insta::assert_snapshot!(f("%_2q", date(2024, 3, 31)), @" 1");
    }

    #[test]
    fn err_format_subsec_nanosecond() {
        let f = |fmt: &str, time: Time| format(fmt, time).unwrap_err();
        let mk = |subsec| time(0, 0, 0, subsec);

        insta::assert_snapshot!(
            f("%00f", mk(123_456_789)),
            @"strftime formatting failed: %f failed: zero precision with %f is not allowed",
        );
    }

    #[test]
    fn err_format_timestamp() {
        let f = |fmt: &str, dt: DateTime| format(fmt, dt).unwrap_err();

        let dt = date(2025, 1, 20).at(13, 9, 0, 0);
        insta::assert_snapshot!(
            f("%s", dt),
            @"strftime formatting failed: %s failed: requires instant (a timestamp or a date, time and offset)",
        );
    }

    #[test]
    fn err_invalid_utf8() {
        let d = date(2025, 1, 20);
        insta::assert_snapshot!(
            format("abc %F xyz", d).unwrap(),
            @"abc 2025-01-20 xyz",
        );
        insta::assert_snapshot!(
            format(b"abc %F \xFFxyz", d).unwrap_err(),
            @"strftime formatting failed: invalid format string, it must be valid UTF-8",
        );
    }

    #[test]
    fn lenient() {
        fn f(
            fmt: impl AsRef<[u8]>,
            tm: impl Into<BrokenDownTime>,
        ) -> alloc::string::String {
            let config = Config::new().lenient(true);
            tm.into().to_string_with_config(&config, fmt).unwrap()
        }

        insta::assert_snapshot!(f("%z", date(2024, 7, 9)), @"%z");
        insta::assert_snapshot!(f("%:z", date(2024, 7, 9)), @"%:z");
        insta::assert_snapshot!(f("%Q", date(2024, 7, 9)), @"%Q");
        insta::assert_snapshot!(f("%+", date(2024, 7, 9)), @"%+");
        insta::assert_snapshot!(f("%F", date(2024, 7, 9)), @"2024-07-09");
        insta::assert_snapshot!(f("%T", date(2024, 7, 9)), @"%T");
        insta::assert_snapshot!(f("%F%", date(2024, 7, 9)), @"2024-07-09%");
        insta::assert_snapshot!(
            f(b"abc %F \xFFxyz", date(2024, 7, 9)),
            @"abc 2024-07-09 �xyz",
        );
        // Demonstrates substitution of maximal subparts.
        // Namely, `\xF0\x9F\x92` is a prefix of a valid
        // UTF-8 encoding of a codepoint, such as `💩`.
        // So the entire prefix should get substituted with
        // a single replacement character...
        insta::assert_snapshot!(
            f(b"%F\xF0\x9F\x92%Y", date(2024, 7, 9)),
            @"2024-07-09�2024",
        );
        // ... but \xFF is never part of a valid encoding.
        // So each instance gets its own replacement
        // character.
        insta::assert_snapshot!(
            f(b"%F\xFF\xFF\xFF%Y", date(2024, 7, 9)),
            @"2024-07-09���2024",
        );
    }

    // Regression test for a failed optimization where we would
    // inappropriately call `write_int_pad2` without checking that
    // the number to format was two digits.
    //
    // See: https://github.com/BurntSushi/jiff/issues/497
    #[test]
    fn regression_timestamp_2s() {
        use alloc::string::ToString;

        let ts: Timestamp = "2024-07-09T20:24:00Z".parse().unwrap();
        assert_eq!(ts.strftime("%2s").to_string(), "1720556640");

        let ts: Timestamp = "2024-07-09T20:24:00Z".parse().unwrap();
        assert_eq!(ts.strftime("%_2s").to_string(), "1720556640");

        let ts: Timestamp = "2024-07-09T20:24:00Z".parse().unwrap();
        assert_eq!(ts.strftime("%02s").to_string(), "1720556640");

        let ts: Timestamp = "2024-07-09T20:24:00Z".parse().unwrap();
        assert_eq!(ts.strftime("%04s").to_string(), "1720556640");
    }
}
