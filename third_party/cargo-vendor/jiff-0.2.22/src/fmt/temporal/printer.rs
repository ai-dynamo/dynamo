use core::time::Duration;

use crate::{
    civil::{Date, DateTime, ISOWeekDate, Time},
    error::{fmt::temporal::Error as E, Error},
    fmt::{
        buffer::{ArrayBuffer, BorrowedBuffer, BorrowedWriter},
        temporal::{Pieces, PiecesOffset, TimeZoneAnnotationKind},
        Write,
    },
    span::{Span, UnitSet},
    tz::{Offset, TimeZone},
    SignedDuration, Timestamp, Unit, Zoned,
};

/// Defines the "maximum" possible length (in bytes) of an RFC 9557 zoned
/// datetime.
///
/// In practice, the actual maximal string possible is I believe
/// `-009999-03-14T17:30:00.999999999-04:23[America/Argentina/ComodRivadavia]`,
/// which is 72 bytes. Obviously, the length of the IANA tzdb identifier
/// matters and can be highly variable.
///
/// So why is this called "reasonable" and not "maximum"? Well, it's the
/// IANA tzdb identifiers. They can be arbitrarily long. There also aren't
/// any rules about how long they can be. So in theory, IANA could allocate
/// a new identifier longer than `America/Argentina/ComodRivadavia`. With
/// that said, they do generally try to keep them succinct.
///
/// Separately from what IANA does, Jiff itself doesn't impose any restrictions
/// on the length of identifiers. Callers can pass in arbitrarily long
/// identifiers via `TimeZone::tzif` or by simply futzing with the names of
/// files in `/usr/share/zoneinfo`. It's also possible to use an arbitrarily
/// long identifier via the "pieces" `TimeZoneAnnotationName` API. Since we
/// don't impose any restrictions, we really do want to at least try to handle
/// arbitrarily long identifiers here.
///
/// Thus, we define a "reasonable" upper bound. When the RFC 9557 string we
/// want to serialize is known to be under this bound, then we'll use a "fast"
/// path with a fixed size buffer on the stack (or perhaps even write directly
/// into the spare capacity of a caller providd `String`). But when it's above
/// this bound, then we fall back to a slower path that uses a buffering
/// mechanism to permit arbitrarily long IANA tzdb identifiers.
///
/// For the most part, doing this dance doesn't come with a runtime cost,
/// primarily because we choose to sacrifice code size a bit by duplicating
/// some functions. We could have our cake and eat it too if we enforce a
/// maximum length on IANA tzdb identifiers. Then we could set a true
/// `MAX_ZONED_LEN` and avoid the case where buffering is needed.
const REASONABLE_ZONED_LEN: usize = 72;

/// Defines the "maximum" possible length (in bytes) of an RFC 9557 zoned
/// datetime via the "pieces" API.
///
/// This generally should be identical to `REASONABLE_ZONED_LEN`, except its
/// expected maximum is one byte longer. Namely, the pieces API currently
/// lets the caller roundtrip the "criticality" of the timestamp. i.e.,
/// the `!` in the time zone annotation. So it's one extra byte longer
/// than zoned datetimes.
///
/// Note that all the same considerations from variable length IANA tzdb
/// identifiers apply to the pieces printing just as it does to zoned datetime
/// printing.
const REASONABLE_PIECES_LEN: usize = 73;

/// Defines the maximum possible length (in bytes) of an RFC 3339 timestamp
/// that is always in Zulu time.
///
/// The longest possible string is `-009999-03-14T21:53:08.999999999Z`.
const MAX_TIMESTAMP_ZULU_LEN: usize = 33;

/// Defines the maximum possible length (in bytes) of an RFC 3339 timestamp
/// with an offset (up to minute precision).
///
/// The longest possible string is `-009999-03-15T23:53:07.999999999+25:59`.
const MAX_TIMESTAMP_OFFSET_LEN: usize = 38;

/// Defines the maximum possible length (in bytes) of an ISO 8601 datetime.
///
/// The longest possible string is `-009999-03-14T17:30:00.999999999`.
const MAX_DATETIME_LEN: usize = 32;

/// Defines the maximum possible length (in bytes) of an ISO 8601 date.
///
/// The longest possible string is `-009999-03-14`.
const MAX_DATE_LEN: usize = 13;

/// Defines the maximum possible length (in bytes) of an ISO 8601 time.
///
/// The longest possible string is `17:30:00.999999999`.
const MAX_TIME_LEN: usize = 18;

/// Defines the maximum possible length (in bytes) of an offset.
///
/// The longest possible string is `-25:59:59`.
const MAX_OFFSET_LEN: usize = 9;

/// Defines the maximum possible length (in bytes) of an ISO 8601 week date.
///
/// The longest possible string is `-009999-W11-3`.
const MAX_ISO_WEEK_DATE_LEN: usize = 13;

/// Defines the maximum possible length (in bytes) of a `Span` printed in the
/// Temporal ISO 8601 format.
///
/// The way I computed this length was by using the default printer (since
/// there's only one knob and it doesn't impact the length of the string)
/// and using a negative `Span` with each unit set to
/// its minimum value.
const MAX_SPAN_LEN: usize = 78;

/// Defines the maximum possible length (in bytes) of a duration printed in the
/// Temporal ISO 8601 format.
///
/// This applies to both signed and unsigned durations. Unsigned durations have
/// one more digit, but signed durations can have a negative sign.
const MAX_DURATION_LEN: usize = 35;

#[derive(Clone, Debug)]
pub(super) struct DateTimePrinter {
    lowercase: bool,
    separator: u8,
    precision: Option<u8>,
}

impl DateTimePrinter {
    pub(super) const fn new() -> DateTimePrinter {
        DateTimePrinter { lowercase: false, separator: b'T', precision: None }
    }

    pub(super) const fn lowercase(self, yes: bool) -> DateTimePrinter {
        DateTimePrinter { lowercase: yes, ..self }
    }

    pub(super) const fn separator(self, ascii_char: u8) -> DateTimePrinter {
        assert!(ascii_char.is_ascii(), "RFC3339 separator must be ASCII");
        DateTimePrinter { separator: ascii_char, ..self }
    }

    pub(super) const fn precision(
        self,
        precision: Option<u8>,
    ) -> DateTimePrinter {
        DateTimePrinter { precision, ..self }
    }

    pub(super) fn print_zoned(
        &self,
        zdt: &Zoned,
        wtr: &mut dyn Write,
    ) -> Result<(), Error> {
        /// The base size of an RFC 9557 zoned datetime string.
        ///
        /// 19 comes from the datetime component, e.g., `2025-01-01T00:00:00`
        /// 6 comes from the offset, e.g., `+05:30`
        /// 2 comes from the `[` and `]` for the time zone annotation
        ///
        /// Basically, we always need space for *at least* the above stuff.
        /// The actual time zone annotation name, negative year and fractional
        /// second component are all optional and can vary quite a bit in
        /// length.
        ///
        /// We do this calculation to get a tighter bound on the spare capacity
        /// needed when the caller provides a `&mut String` or a `&mut Vec<u8>`
        /// to write into. That is, we want to try hard not to over-allocate.
        ///
        /// Note that memory safety does not depend on us getting this
        /// calculation right. If we get it wrong, the printer will panic if
        /// it tries to print a string that exceeds the calculated amount.
        const BASE: usize = 19 + 6 + 2;

        // An IANA tzdb identifier is variable length data, so add its length
        // to the `BASE` for runtime allocation size. When there is no IANA
        // identifier, we could write a fixed offset (that's always 6 bytes) or
        // `Etc/Unknown` (11 bytes) for when the offset from UTC is not known.
        // In the non-IANA case, we just use an upper bound of 11, so we will
        // over-allocate a little in the fixed offset case.
        let mut runtime_allocation = BASE
            + zdt.time_zone().iana_name().map(|name| name.len()).unwrap_or(11);
        // A datetime before year 0 means we add a `-00` prefix. e.g.,
        // `-001234-01-01`.
        if zdt.year() < 0 {
            runtime_allocation += 3;
        }
        // If we're printing fractional seconds, then we need more room for
        // that. This potentially overallocates because we don't do the extra
        // work required to find a tighter bound.
        if zdt.subsec_nanosecond() != 0 || self.precision.is_some() {
            runtime_allocation += 10;
        }

        // The runtime allocation could be greater than what we assume is a
        // "reasonable" upper bound on the length of an RFC 9557 string. This
        // can only happen when an IANA tzdb identifier is very long. When
        // we're under the limit, we use a fast path.
        if runtime_allocation <= REASONABLE_ZONED_LEN {
            return BorrowedBuffer::with_writer::<REASONABLE_ZONED_LEN>(
                wtr,
                runtime_allocation,
                |bbuf| Ok(self.print_zoned_buf(zdt, bbuf)),
            );
        }

        // ... otherwise, we use a path with a buffered writer that is slower
        // but can deal with arbitrarily long IANA tzdb identifiers.
        let mut buf = ArrayBuffer::<REASONABLE_ZONED_LEN>::default();
        let mut bbuf = buf.as_borrowed();
        let mut wtr = BorrowedWriter::new(&mut bbuf, wtr);
        self.print_zoned_wtr(zdt, &mut wtr)?;
        wtr.finish()
    }

    fn print_zoned_buf(&self, zdt: &Zoned, bbuf: &mut BorrowedBuffer<'_>) {
        self.print_datetime_buf(&zdt.datetime(), bbuf);
        let tz = zdt.time_zone();
        if tz.is_unknown() {
            bbuf.write_str("Z[Etc/Unknown]");
        } else {
            self.print_offset_rounded_buf(&zdt.offset(), bbuf);
            self.print_time_zone_annotation_buf(&tz, &zdt.offset(), bbuf);
        }
    }

    fn print_zoned_wtr(
        &self,
        zdt: &Zoned,
        wtr: &mut BorrowedWriter<'_, '_, '_>,
    ) -> Result<(), Error> {
        self.print_datetime_wtr(&zdt.datetime(), wtr)?;
        let tz = zdt.time_zone();
        if tz.is_unknown() {
            wtr.write_str("Z[Etc/Unknown]")?;
        } else {
            self.print_offset_rounded_wtr(&zdt.offset(), wtr)?;
            self.print_time_zone_annotation_wtr(&tz, &zdt.offset(), wtr)?;
        }
        Ok(())
    }

    pub(super) fn print_timestamp(
        &self,
        timestamp: &Timestamp,
        wtr: &mut dyn Write,
    ) -> Result<(), Error> {
        let mut runtime_allocation = MAX_TIMESTAMP_ZULU_LEN;
        // Don't reserve room for fractional seconds if we don't use them.
        if timestamp.subsec_nanosecond() == 0 && self.precision.is_none() {
            runtime_allocation -= 10;
        }
        BorrowedBuffer::with_writer::<MAX_TIMESTAMP_ZULU_LEN>(
            wtr,
            runtime_allocation,
            |bbuf| Ok(self.print_timestamp_buf(timestamp, bbuf)),
        )
    }

    fn print_timestamp_buf(
        &self,
        timestamp: &Timestamp,
        bbuf: &mut BorrowedBuffer<'_>,
    ) {
        let dt = Offset::UTC.to_datetime(*timestamp);
        self.print_datetime_buf(&dt, bbuf);
        self.print_zulu_buf(bbuf);
    }

    pub(super) fn print_timestamp_with_offset(
        &self,
        timestamp: &Timestamp,
        offset: Offset,
        wtr: &mut dyn Write,
    ) -> Result<(), Error> {
        let mut runtime_allocation = MAX_TIMESTAMP_OFFSET_LEN;
        // Don't reserve room for fractional seconds if we don't use them.
        if timestamp.subsec_nanosecond() == 0 && self.precision.is_none() {
            runtime_allocation -= 10;
        }
        BorrowedBuffer::with_writer::<MAX_TIMESTAMP_OFFSET_LEN>(
            wtr,
            runtime_allocation,
            |bbuf| {
                Ok(self
                    .print_timestamp_with_offset_buf(timestamp, offset, bbuf))
            },
        )
    }

    fn print_timestamp_with_offset_buf(
        &self,
        timestamp: &Timestamp,
        offset: Offset,
        bbuf: &mut BorrowedBuffer<'_>,
    ) {
        let dt = offset.to_datetime(*timestamp);
        self.print_datetime_buf(&dt, bbuf);
        self.print_offset_rounded_buf(&offset, bbuf);
    }

    pub(super) fn print_datetime(
        &self,
        dt: &DateTime,
        wtr: &mut dyn Write,
    ) -> Result<(), Error> {
        let mut runtime_allocation = MAX_DATETIME_LEN;
        // Don't reserve room for fractional seconds if we don't use them.
        if dt.subsec_nanosecond() == 0 && self.precision.is_none() {
            runtime_allocation -= 10;
        }
        BorrowedBuffer::with_writer::<MAX_DATETIME_LEN>(
            wtr,
            runtime_allocation,
            |bbuf| Ok(self.print_datetime_buf(dt, bbuf)),
        )
    }

    fn print_datetime_buf(
        &self,
        dt: &DateTime,
        bbuf: &mut BorrowedBuffer<'_>,
    ) {
        self.print_date_buf(&dt.date(), bbuf);
        bbuf.write_ascii_char(if self.lowercase {
            self.separator.to_ascii_lowercase()
        } else {
            self.separator
        });
        self.print_time_buf(&dt.time(), bbuf);
    }

    fn print_datetime_wtr(
        &self,
        dt: &DateTime,
        wtr: &mut BorrowedWriter<'_, '_, '_>,
    ) -> Result<(), Error> {
        self.print_date_wtr(&dt.date(), wtr)?;
        wtr.write_ascii_char(if self.lowercase {
            self.separator.to_ascii_lowercase()
        } else {
            self.separator
        })?;
        self.print_time_wtr(&dt.time(), wtr)?;
        Ok(())
    }

    pub(super) fn print_date(
        &self,
        date: &Date,
        wtr: &mut dyn Write,
    ) -> Result<(), Error> {
        BorrowedBuffer::with_writer::<MAX_DATE_LEN>(
            wtr,
            MAX_DATE_LEN,
            |bbuf| Ok(self.print_date_buf(date, bbuf)),
        )
    }

    fn print_date_buf(&self, date: &Date, bbuf: &mut BorrowedBuffer<'_>) {
        let year = date.year();
        if year < 0 {
            bbuf.write_str("-00");
        }
        bbuf.write_int_pad4(year.unsigned_abs());
        bbuf.write_ascii_char(b'-');
        bbuf.write_int_pad2(date.month().unsigned_abs());
        bbuf.write_ascii_char(b'-');
        bbuf.write_int_pad2(date.day().unsigned_abs());
    }

    fn print_date_wtr(
        &self,
        date: &Date,
        wtr: &mut BorrowedWriter<'_, '_, '_>,
    ) -> Result<(), Error> {
        let year = date.year();
        if year < 0 {
            wtr.write_str("-00")?;
        }
        wtr.write_int_pad4(year.unsigned_abs())?;
        wtr.write_ascii_char(b'-')?;
        wtr.write_int_pad2(date.month().unsigned_abs())?;
        wtr.write_ascii_char(b'-')?;
        wtr.write_int_pad2(date.day().unsigned_abs())?;
        Ok(())
    }

    pub(super) fn print_time(
        &self,
        time: &Time,
        wtr: &mut dyn Write,
    ) -> Result<(), Error> {
        let mut runtime_allocation = MAX_TIME_LEN;
        // Don't reserve room for fractional seconds if we don't use them.
        if time.subsec_nanosecond() == 0 && self.precision.is_none() {
            runtime_allocation -= 10;
        }
        BorrowedBuffer::with_writer::<MAX_TIME_LEN>(
            wtr,
            runtime_allocation,
            |bbuf| Ok(self.print_time_buf(time, bbuf)),
        )
    }

    fn print_time_buf(&self, time: &Time, bbuf: &mut BorrowedBuffer<'_>) {
        bbuf.write_int_pad2(time.hour().unsigned_abs());
        bbuf.write_ascii_char(b':');
        bbuf.write_int_pad2(time.minute().unsigned_abs());
        bbuf.write_ascii_char(b':');
        bbuf.write_int_pad2(time.second().unsigned_abs());
        let fractional_nanosecond = time.subsec_nanosecond();
        if self.precision.map_or(fractional_nanosecond != 0, |p| p > 0) {
            bbuf.write_ascii_char(b'.');
            bbuf.write_fraction(
                self.precision,
                fractional_nanosecond.unsigned_abs(),
            );
        }
    }

    fn print_time_wtr(
        &self,
        time: &Time,
        wtr: &mut BorrowedWriter<'_, '_, '_>,
    ) -> Result<(), Error> {
        wtr.write_int_pad2(time.hour().unsigned_abs())?;
        wtr.write_ascii_char(b':')?;
        wtr.write_int_pad2(time.minute().unsigned_abs())?;
        wtr.write_ascii_char(b':')?;
        wtr.write_int_pad2(time.second().unsigned_abs())?;
        let fractional_nanosecond = time.subsec_nanosecond();
        if self.precision.map_or(fractional_nanosecond != 0, |p| p > 0) {
            wtr.write_ascii_char(b'.')?;
            wtr.write_fraction(
                self.precision,
                fractional_nanosecond.unsigned_abs(),
            )?;
        }
        Ok(())
    }

    pub(super) fn print_time_zone<W: Write>(
        &self,
        tz: &TimeZone,
        mut wtr: W,
    ) -> Result<(), Error> {
        // N.B. We use a `&mut dyn Write` here instead of an uninitialized
        // buffer (as in the other routines for this printer) because this
        // can emit a POSIX time zone string. We don't really have strong
        // guarantees about how long this string can be (although all sensible
        // values are pretty short). Since this API is not expected to be used
        // much, we don't spend the time to try and optimize this.
        //
        // If and when we get a borrowed buffer writer abstraction (for truly
        // variable length output), then we might consider using that here.
        self.print_time_zone_wtr(tz, &mut wtr)
    }

    fn print_time_zone_wtr(
        &self,
        tz: &TimeZone,
        wtr: &mut dyn Write,
    ) -> Result<(), Error> {
        if let Some(iana_name) = tz.iana_name() {
            return wtr.write_str(iana_name);
        }
        if tz.is_unknown() {
            return wtr.write_str("Etc/Unknown");
        }
        if let Ok(offset) = tz.to_fixed_offset() {
            // Kind of unfortunate, but it's probably better than
            // making `print_offset_full_precision` accept a `dyn Write`.
            let mut buf = ArrayBuffer::<MAX_OFFSET_LEN>::default();
            let mut bbuf = buf.as_borrowed();
            self.print_offset_full_precision(&offset, &mut bbuf);
            return wtr.write_str(bbuf.filled());
        }
        if let Some(posix_tz) = tz.posix_tz() {
            use core::fmt::Write as _;

            // This is rather circuitous, but I'm not sure how else to do it
            // without allocating an intermediate string. Or writing another
            // printing API for `PosixTimeZone`. (Which might actually not be
            // a bad idea. Perhaps using uninit buffers. But... who gives a
            // fuck about printing POSIX time zone strings?)
            return write!(crate::fmt::StdFmtWrite(wtr), "{posix_tz}")
                .map_err(|_| {
                    Error::from(crate::error::fmt::Error::StdFmtWriteAdapter)
                });
        }
        // Ideally this never actually happens, but it can, and there
        // are likely system configurations out there in which it does.
        // I can imagine "lightweight" installations that just have a
        // `/etc/localtime` as a TZif file that doesn't point to any IANA time
        // zone. In which case, serializing a time zone probably doesn't make
        // much sense.
        //
        // Anyway, if you're seeing this error and think there should be a
        // different behavior, please file an issue.
        Err(Error::from(E::PrintTimeZoneFailure))
    }

    pub(super) fn print_pieces<W: Write>(
        &self,
        pieces: &Pieces,
        mut wtr: W,
    ) -> Result<(), Error> {
        // N.B. We don't bother with writing into the spare capacity of a
        // `&mut String` here because with `Pieces` it's a little more
        // complicated to find a more precise upper bound on the length.
        // Plus, I don't think this API is commonly used, so it's not clear
        // that it's worth optimizing. But I'm open to a PR with benchmarks
        // if there's a good use case. ---AG
        let mut buf = ArrayBuffer::<REASONABLE_PIECES_LEN>::default();
        let mut bbuf = buf.as_borrowed();
        let mut wtr = BorrowedWriter::new(&mut bbuf, &mut wtr);
        self.print_pieces_wtr(pieces, &mut wtr)?;
        wtr.finish()
    }

    fn print_pieces_wtr(
        &self,
        pieces: &Pieces,
        wtr: &mut BorrowedWriter<'_, '_, '_>,
    ) -> Result<(), Error> {
        if let Some(time) = pieces.time() {
            let dt = DateTime::from_parts(pieces.date(), time);
            self.print_datetime_wtr(&dt, wtr)?;
            if let Some(poffset) = pieces.offset() {
                self.print_pieces_offset(&poffset, wtr)?;
            }
        } else if let Some(poffset) = pieces.offset() {
            // In this case, we have an offset but no time component. Since
            // `2025-01-02-05:00` isn't valid, we forcefully write out the
            // default time (which is what would be assumed anyway).
            let dt = DateTime::from_parts(pieces.date(), Time::midnight());
            self.print_datetime_wtr(&dt, wtr)?;
            self.print_pieces_offset(&poffset, wtr)?;
        } else {
            // We have no time and no offset, so we can just write the date.
            // It's okay to write this followed by an annotation, e.g.,
            // `2025-01-02[America/New_York]` or even `2025-01-02[-05:00]`.
            self.print_date_wtr(&pieces.date(), wtr)?;
        }
        // For the time zone annotation, a `Pieces` gives us the annotation
        // name or offset directly, where as with `Zoned`, we have a
        // `TimeZone`. So we hand-roll our own formatter directly from the
        // annotation.
        if let Some(ann) = pieces.time_zone_annotation() {
            wtr.write_ascii_char(b'[')?;
            if ann.is_critical() {
                wtr.write_ascii_char(b'!')?;
            }
            match *ann.kind() {
                TimeZoneAnnotationKind::Named(ref name) => {
                    wtr.write_str(name.as_str())?;
                }
                TimeZoneAnnotationKind::Offset(offset) => {
                    self.print_offset_rounded_wtr(&offset, wtr)?;
                }
            }
            wtr.write_ascii_char(b']')?;
        }
        Ok(())
    }

    pub(super) fn print_iso_week_date(
        &self,
        iso_week_date: &ISOWeekDate,
        wtr: &mut dyn Write,
    ) -> Result<(), Error> {
        BorrowedBuffer::with_writer::<MAX_ISO_WEEK_DATE_LEN>(
            wtr,
            MAX_ISO_WEEK_DATE_LEN,
            |bbuf| Ok(self.print_iso_week_date_buf(iso_week_date, bbuf)),
        )
    }

    fn print_iso_week_date_buf(
        &self,
        iso_week_date: &ISOWeekDate,
        bbuf: &mut BorrowedBuffer<'_>,
    ) {
        let year = iso_week_date.year();
        if year < 0 {
            bbuf.write_str("-00");
        }
        bbuf.write_int_pad4(year.unsigned_abs());
        bbuf.write_ascii_char(b'-');
        bbuf.write_ascii_char(if self.lowercase { b'w' } else { b'W' });
        bbuf.write_int_pad2(iso_week_date.week().unsigned_abs());
        bbuf.write_ascii_char(b'-');
        bbuf.write_int1(
            iso_week_date.weekday().to_monday_one_offset().unsigned_abs(),
        );
    }

    fn print_pieces_offset(
        &self,
        poffset: &PiecesOffset,
        wtr: &mut BorrowedWriter<'_, '_, '_>,
    ) -> Result<(), Error> {
        match *poffset {
            PiecesOffset::Zulu => self.print_zulu_wtr(wtr),
            PiecesOffset::Numeric(ref noffset) => {
                if noffset.offset().is_zero() && noffset.is_negative() {
                    wtr.write_str("-00:00")
                } else {
                    self.print_offset_rounded_wtr(&noffset.offset(), wtr)
                }
            }
        }
    }

    /// Formats the given offset into the writer given.
    ///
    /// If the given offset has non-zero seconds, then they are rounded to
    /// the nearest minute.
    fn print_offset_rounded_buf(
        &self,
        offset: &Offset,
        bbuf: &mut BorrowedBuffer<'_>,
    ) {
        bbuf.write_ascii_char(if offset.is_negative() { b'-' } else { b'+' });
        let (offset_hours, offset_minutes) = offset.round_to_nearest_minute();
        bbuf.write_int_pad2(offset_hours);
        bbuf.write_ascii_char(b':');
        bbuf.write_int_pad2(offset_minutes);
    }

    /// Formats the given offset into the writer given.
    ///
    /// If the given offset has non-zero seconds, then they are rounded to
    /// the nearest minute.
    fn print_offset_rounded_wtr(
        &self,
        offset: &Offset,
        wtr: &mut BorrowedWriter<'_, '_, '_>,
    ) -> Result<(), Error> {
        wtr.write_ascii_char(if offset.is_negative() { b'-' } else { b'+' })?;
        let (offset_hours, offset_minutes) = offset.round_to_nearest_minute();
        wtr.write_int_pad2(offset_hours)?;
        wtr.write_ascii_char(b':')?;
        wtr.write_int_pad2(offset_minutes)?;
        Ok(())
    }

    /// Formats the given offset into the writer given.
    ///
    /// If the given offset has non-zero seconds, then they are emitted as a
    /// third `:`-delimited component of the offset. If seconds are zero, then
    /// only the hours and minute components are emitted.
    fn print_offset_full_precision(
        &self,
        offset: &Offset,
        bbuf: &mut BorrowedBuffer<'_>,
    ) {
        bbuf.write_ascii_char(if offset.is_negative() { b'-' } else { b'+' });
        let hours = offset.part_hours().unsigned_abs();
        let minutes = offset.part_minutes().unsigned_abs();
        let seconds = offset.part_seconds().unsigned_abs();
        bbuf.write_int_pad2(hours);
        bbuf.write_ascii_char(b':');
        bbuf.write_int_pad2(minutes);
        if seconds > 0 {
            bbuf.write_ascii_char(b':');
            bbuf.write_int_pad2(seconds);
        }
    }

    /// Prints the "zulu" indicator.
    ///
    /// This should only be used when the offset is not known. For example,
    /// when printing a `Timestamp`.
    fn print_zulu_buf(&self, bbuf: &mut BorrowedBuffer<'_>) {
        bbuf.write_ascii_char(if self.lowercase { b'z' } else { b'Z' });
    }

    /// Prints the "zulu" indicator.
    ///
    /// This should only be used when the offset is not known. For example,
    /// when printing a `Timestamp`.
    fn print_zulu_wtr(
        &self,
        wtr: &mut BorrowedWriter<'_, '_, '_>,
    ) -> Result<(), Error> {
        wtr.write_ascii_char(if self.lowercase { b'z' } else { b'Z' })
    }

    /// Formats the given time zone name into the writer given as an RFC 9557
    /// time zone annotation.
    ///
    /// When the given time zone is not an IANA time zone name, then the offset
    /// is printed instead. (This means the offset will be printed twice, which
    /// is indeed an intended behavior of RFC 9557 for cases where a time zone
    /// name is not used or unavailable.)
    fn print_time_zone_annotation_buf(
        &self,
        time_zone: &TimeZone,
        offset: &Offset,
        bbuf: &mut BorrowedBuffer<'_>,
    ) {
        bbuf.write_ascii_char(b'[');
        if let Some(iana_name) = time_zone.iana_name() {
            bbuf.write_str(iana_name);
        } else {
            self.print_offset_rounded_buf(offset, bbuf);
        }
        bbuf.write_ascii_char(b']');
    }

    /// Formats the given time zone name into the writer given as an RFC 9557
    /// time zone annotation.
    ///
    /// When the given time zone is not an IANA time zone name, then the offset
    /// is printed instead. (This means the offset will be printed twice, which
    /// is indeed an intended behavior of RFC 9557 for cases where a time zone
    /// name is not used or unavailable.)
    fn print_time_zone_annotation_wtr(
        &self,
        time_zone: &TimeZone,
        offset: &Offset,
        wtr: &mut BorrowedWriter<'_, '_, '_>,
    ) -> Result<(), Error> {
        wtr.write_ascii_char(b'[')?;
        if let Some(iana_name) = time_zone.iana_name() {
            wtr.write_str(iana_name)?;
        } else {
            self.print_offset_rounded_wtr(offset, wtr)?;
        }
        wtr.write_ascii_char(b']')?;
        Ok(())
    }
}

impl Default for DateTimePrinter {
    fn default() -> DateTimePrinter {
        DateTimePrinter::new()
    }
}

/// A printer for Temporal spans.
///
/// Note that in Temporal, a "span" is called a "duration."
#[derive(Debug)]
pub(super) struct SpanPrinter {
    /// The designators to use.
    designators: &'static Designators,
}

impl SpanPrinter {
    /// Create a new Temporal span printer with the default configuration.
    pub(super) const fn new() -> SpanPrinter {
        SpanPrinter { designators: DESIGNATORS_UPPERCASE }
    }

    /// Use lowercase for unit designator labels.
    ///
    /// By default, unit designator labels are written in uppercase.
    pub(super) const fn lowercase(self, yes: bool) -> SpanPrinter {
        SpanPrinter {
            designators: if yes {
                DESIGNATORS_LOWERCASE
            } else {
                DESIGNATORS_UPPERCASE
            },
        }
    }

    /// Print the given span to the writer given.
    ///
    /// This only returns an error when the given writer returns an error.
    pub(super) fn print_span<W: Write>(
        &self,
        span: &Span,
        mut wtr: W,
    ) -> Result<(), Error> {
        let mut buf = ArrayBuffer::<MAX_SPAN_LEN>::default();
        let mut bbuf = buf.as_borrowed();
        self.print_span_impl(span, &mut bbuf);
        wtr.write_str(bbuf.filled())
    }

    fn print_span_impl(&self, span: &Span, bbuf: &mut BorrowedBuffer<'_>) {
        static SUBSECOND: UnitSet = UnitSet::from_slice(&[
            Unit::Millisecond,
            Unit::Microsecond,
            Unit::Nanosecond,
        ]);

        if span.is_negative() {
            bbuf.write_ascii_char(b'-');
        }
        bbuf.write_ascii_char(b'P');

        let units = span.units();
        if units.contains(Unit::Year) {
            bbuf.write_int(span.get_years_unsigned());
            bbuf.write_ascii_char(self.label(Unit::Year));
        }
        if units.contains(Unit::Month) {
            bbuf.write_int(span.get_months_unsigned());
            bbuf.write_ascii_char(self.label(Unit::Month));
        }
        if units.contains(Unit::Week) {
            bbuf.write_int(span.get_weeks_unsigned());
            bbuf.write_ascii_char(self.label(Unit::Week));
        }
        if units.contains(Unit::Day) {
            bbuf.write_int(span.get_days_unsigned());
            bbuf.write_ascii_char(self.label(Unit::Day));
        }

        if units.only_time().is_empty() {
            if units.only_calendar().is_empty() {
                bbuf.write_ascii_char(b'T');
                bbuf.write_ascii_char(b'0');
                bbuf.write_ascii_char(self.label(Unit::Second));
            }
            return;
        }

        bbuf.write_ascii_char(b'T');

        if units.contains(Unit::Hour) {
            bbuf.write_int(span.get_hours_unsigned());
            bbuf.write_ascii_char(self.label(Unit::Hour));
        }
        if units.contains(Unit::Minute) {
            bbuf.write_int(span.get_minutes_unsigned());
            bbuf.write_ascii_char(self.label(Unit::Minute));
        }

        // ISO 8601 (and Temporal) don't support writing out milliseconds,
        // microseconds or nanoseconds as separate components like for all
        // the other units. Instead, they must be incorporated as fractional
        // seconds. But we only want to do that work if we need to.
        let has_subsecond = !units.intersection(SUBSECOND).is_empty();
        if units.contains(Unit::Second) && !has_subsecond {
            bbuf.write_int(span.get_seconds_unsigned());
            bbuf.write_ascii_char(self.label(Unit::Second));
        } else if has_subsecond {
            // We want to combine our seconds, milliseconds, microseconds and
            // nanoseconds into one single value in terms of nanoseconds. Then
            // we can "balance" that out so that we have a number of seconds
            // and a number of nanoseconds not greater than 1 second. (Which is
            // our fraction.)
            let (seconds, millis, micros, nanos) = (
                Duration::from_secs(span.get_seconds_unsigned()),
                Duration::from_millis(span.get_milliseconds_unsigned()),
                Duration::from_micros(span.get_microseconds_unsigned()),
                Duration::from_nanos(span.get_nanoseconds_unsigned()),
            );
            // OK because the maximums for a span's seconds, millis, micros
            // and nanos combined all fit into a 96-bit integer. (This is
            // guaranteed by `Span::to_duration_invariant`.)
            let total = seconds + millis + micros + nanos;
            let (secs, subsecs) = (total.as_secs(), total.subsec_nanos());
            bbuf.write_int(secs);
            if subsecs != 0 {
                bbuf.write_ascii_char(b'.');
                bbuf.write_fraction(None, subsecs);
            }
            bbuf.write_ascii_char(self.label(Unit::Second));
        }
    }

    /// Print the given signed duration to the writer given.
    ///
    /// This only returns an error when the given writer returns an error.
    pub(super) fn print_signed_duration<W: Write>(
        &self,
        dur: &SignedDuration,
        mut wtr: W,
    ) -> Result<(), Error> {
        let mut buf = ArrayBuffer::<MAX_DURATION_LEN>::default();
        let mut bbuf = buf.as_borrowed();
        if dur.is_negative() {
            bbuf.write_ascii_char(b'-');
        }
        self.print_unsigned_duration_impl(&dur.unsigned_abs(), &mut bbuf);
        wtr.write_str(bbuf.filled())
    }

    /// Print the given unsigned duration to the writer given.
    ///
    /// This only returns an error when the given writer returns an error.
    pub(super) fn print_unsigned_duration<W: Write>(
        &self,
        dur: &Duration,
        mut wtr: W,
    ) -> Result<(), Error> {
        let mut buf = ArrayBuffer::<MAX_DURATION_LEN>::default();
        let mut bbuf = buf.as_borrowed();
        self.print_unsigned_duration_impl(dur, &mut bbuf);
        wtr.write_str(bbuf.filled())
    }

    fn print_unsigned_duration_impl(
        &self,
        dur: &Duration,
        bbuf: &mut BorrowedBuffer<'_>,
    ) {
        bbuf.write_ascii_char(b'P');
        bbuf.write_ascii_char(b'T');

        let (mut secs, nanos) = (dur.as_secs(), dur.subsec_nanos());
        let non_zero_greater_than_second = secs >= 60;
        if non_zero_greater_than_second {
            let hours = secs / (60 * 60);
            secs %= 60 * 60;
            let minutes = secs / 60;
            secs = secs % 60;
            if hours != 0 {
                bbuf.write_int(hours);
                bbuf.write_ascii_char(self.label(Unit::Hour));
            }
            if minutes != 0 {
                bbuf.write_int(minutes);
                bbuf.write_ascii_char(self.label(Unit::Minute));
            }
        }
        if !non_zero_greater_than_second || secs != 0 || nanos != 0 {
            bbuf.write_int(secs);
            if nanos != 0 {
                bbuf.write_ascii_char(b'.');
                bbuf.write_fraction(None, nanos);
            }
            bbuf.write_ascii_char(self.label(Unit::Second));
        }
    }

    /// Converts the ASCII uppercase unit designator label to lowercase if this
    /// printer is configured to use lowercase. Otherwise the label is returned
    /// unchanged.
    fn label(&self, unit: Unit) -> u8 {
        self.designators.designator(unit)
    }
}

#[derive(Clone, Debug)]
struct Designators {
    // Indexed by `Unit as usize`
    map: [u8; 10],
}

const DESIGNATORS_UPPERCASE: &'static Designators = &Designators {
    // N.B. ISO 8601 duration format doesn't have unit
    // designators for sub-second units.
    map: [0, 0, 0, b'S', b'M', b'H', b'D', b'W', b'M', b'Y'],
};

const DESIGNATORS_LOWERCASE: &'static Designators = &Designators {
    // N.B. ISO 8601 duration format doesn't have unit
    // designators for sub-second units.
    map: [0, 0, 0, b's', b'm', b'h', b'd', b'w', b'm', b'y'],
};

impl Designators {
    /// Returns the designator for the given unit.
    fn designator(&self, unit: Unit) -> u8 {
        self.map[unit as usize]
    }
}

#[cfg(feature = "alloc")]
#[cfg(test)]
mod tests {
    use alloc::string::{String, ToString};

    use crate::{
        civil::{date, time, Weekday},
        fmt::StdFmtWrite,
        span::ToSpan,
        util::b,
    };

    use super::*;

    #[test]
    fn print_zoned() {
        if crate::tz::db().is_definitively_empty() {
            return;
        }

        let p = || DateTimePrinter::new();
        let via_io = |dtp: DateTimePrinter, zdt| {
            let mut buf = String::new();
            dtp.print_zoned(&zdt, &mut StdFmtWrite(&mut buf)).unwrap();
            buf
        };
        let to_string = |dtp: DateTimePrinter, zdt| {
            let mut buf = String::new();
            dtp.print_zoned(&zdt, &mut buf).unwrap();
            let got_via_io = via_io(dtp, zdt);
            assert_eq!(
                buf, got_via_io,
                "expected writes to `&mut String` to match `&mut StdFmtWrite`"
            );
            buf
        };

        let dt = date(2024, 3, 10).at(5, 34, 45, 0);
        let zdt = dt.in_tz("America/New_York").unwrap();
        let got = to_string(p(), zdt);
        assert_eq!(got, "2024-03-10T05:34:45-04:00[America/New_York]");

        let dt = date(2024, 3, 10).at(5, 34, 45, 0);
        let zdt = dt.in_tz("America/New_York").unwrap();
        let zdt = zdt.with_time_zone(TimeZone::UTC);
        let got = to_string(p(), zdt);
        assert_eq!(got, "2024-03-10T09:34:45+00:00[UTC]");

        let dt = date(2024, 3, 10).at(5, 34, 45, 0);
        let zdt = dt.to_zoned(TimeZone::fixed(Offset::MIN)).unwrap();
        let got = to_string(p(), zdt);
        assert_eq!(got, "2024-03-10T05:34:45-25:59[-25:59]");

        let dt = date(2024, 3, 10).at(5, 34, 45, 0);
        let zdt = dt.to_zoned(TimeZone::fixed(Offset::MAX)).unwrap();
        let got = to_string(p(), zdt);
        assert_eq!(got, "2024-03-10T05:34:45+25:59[+25:59]");

        let dt = date(2024, 3, 10).at(5, 34, 45, 123_456_789);
        let zdt = dt.in_tz("America/New_York").unwrap();
        let got = to_string(p(), zdt);
        assert_eq!(
            got,
            "2024-03-10T05:34:45.123456789-04:00[America/New_York]"
        );

        let dt = date(2024, 3, 10).at(5, 34, 45, 0);
        let zdt = dt.in_tz("America/New_York").unwrap();
        let got = to_string(p().precision(Some(9)), zdt);
        assert_eq!(
            got,
            "2024-03-10T05:34:45.000000000-04:00[America/New_York]"
        );

        let dt = date(-9999, 3, 1).at(23, 59, 59, 999_999_999);
        let zdt = dt.in_tz("America/Argentina/ComodRivadavia").unwrap();
        let got = to_string(p().precision(Some(9)), zdt);
        assert_eq!(
            got,
            "-009999-03-01T23:59:59.999999999-04:23[America/Argentina/ComodRivadavia]",
        );

        // Inject a very long IANA tzdb identifier to ensure it's handled
        // properly.
        let tz = TimeZone::tzif(
            "Abc/Def/Ghi/Jkl/Mno/Pqr/Stu/Vwx/Yzz/Abc/Def/Ghi/Jkl/Mno/Pqr/Stu/Vwx/Yzz/Abc/Def/Ghi/Jkl/Mno/Pqr/Stu/Vwx/Yzz",
            crate::tz::testdata::TzifTestFile::get("America/New_York").data,
        ).unwrap();
        let dt = date(-9999, 3, 1).at(23, 59, 59, 999_999_999);
        let zdt = dt.to_zoned(tz).unwrap();
        let got = to_string(p().precision(Some(9)), zdt);
        assert_eq!(
            got,
            "-009999-03-01T23:59:59.999999999-04:56[Abc/Def/Ghi/Jkl/Mno/Pqr/Stu/Vwx/Yzz/Abc/Def/Ghi/Jkl/Mno/Pqr/Stu/Vwx/Yzz/Abc/Def/Ghi/Jkl/Mno/Pqr/Stu/Vwx/Yzz]",
        );
    }

    #[test]
    fn print_timestamp_zulu() {
        let p = || DateTimePrinter::new();
        let via_io = |dtp: DateTimePrinter, ts| {
            let mut buf = String::new();
            dtp.print_timestamp(&ts, &mut StdFmtWrite(&mut buf)).unwrap();
            buf
        };
        let to_string = |dtp: DateTimePrinter, ts| {
            let mut buf = String::new();
            dtp.print_timestamp(&ts, &mut buf).unwrap();
            let got_via_io = via_io(dtp, ts);
            assert_eq!(
                buf, got_via_io,
                "expected writes to `&mut String` to match `&mut StdFmtWrite`"
            );
            buf
        };

        let tz = TimeZone::fixed(-Offset::constant(4));

        let dt = date(2024, 3, 10).at(5, 34, 45, 0);
        let zoned: Zoned = dt.to_zoned(tz.clone()).unwrap();
        let got = to_string(p(), zoned.timestamp());
        assert_eq!(got, "2024-03-10T09:34:45Z");

        let dt = date(-2024, 3, 10).at(5, 34, 45, 0);
        let zoned: Zoned = dt.to_zoned(tz.clone()).unwrap();
        let got = to_string(p(), zoned.timestamp());
        assert_eq!(got, "-002024-03-10T09:34:45Z");

        let dt = date(2024, 3, 10).at(5, 34, 45, 123_456_789);
        let zoned: Zoned = dt.to_zoned(tz.clone()).unwrap();
        let got = to_string(p(), zoned.timestamp());
        assert_eq!(got, "2024-03-10T09:34:45.123456789Z");

        let dt = date(2024, 3, 10).at(5, 34, 45, 0);
        let zoned: Zoned = dt.to_zoned(tz.clone()).unwrap();
        let got = to_string(p().precision(Some(9)), zoned.timestamp());
        assert_eq!(got, "2024-03-10T09:34:45.000000000Z");
    }

    #[test]
    fn print_timestamp_with_offset() {
        let p = || DateTimePrinter::new();
        let via_io = |dtp: DateTimePrinter, ts, offset| {
            let mut buf = String::new();
            dtp.print_timestamp_with_offset(
                &ts,
                offset,
                &mut StdFmtWrite(&mut buf),
            )
            .unwrap();
            buf
        };
        let to_string = |dtp: DateTimePrinter, ts, offset| {
            let mut buf = String::new();
            dtp.print_timestamp_with_offset(&ts, offset, &mut buf).unwrap();
            let got_via_io = via_io(dtp, ts, offset);
            assert_eq!(
                buf, got_via_io,
                "expected writes to `&mut String` to match `&mut StdFmtWrite`"
            );
            buf
        };

        let tz = TimeZone::fixed(-Offset::constant(4));

        let dt = date(2024, 3, 10).at(5, 34, 45, 0);
        let zoned: Zoned = dt.to_zoned(tz.clone()).unwrap();
        let got = to_string(p(), zoned.timestamp(), zoned.offset());
        assert_eq!(got, "2024-03-10T05:34:45-04:00");

        let dt = date(-2024, 3, 10).at(5, 34, 45, 0);
        let zoned: Zoned = dt.to_zoned(tz.clone()).unwrap();
        let got = to_string(p(), zoned.timestamp(), zoned.offset());
        assert_eq!(got, "-002024-03-10T05:34:45-04:00");

        let dt = date(2024, 3, 10).at(5, 34, 45, 123_456_789);
        let zoned: Zoned = dt.to_zoned(tz.clone()).unwrap();
        let got = to_string(p(), zoned.timestamp(), zoned.offset());
        assert_eq!(got, "2024-03-10T05:34:45.123456789-04:00");

        let dt = date(2024, 3, 10).at(5, 34, 45, 0);
        let zoned: Zoned = dt.to_zoned(tz.clone()).unwrap();
        let got = to_string(
            p().precision(Some(9)),
            zoned.timestamp(),
            zoned.offset(),
        );
        assert_eq!(got, "2024-03-10T05:34:45.000000000-04:00");

        let dt = DateTime::MIN;
        let zoned: Zoned = dt.to_zoned(TimeZone::fixed(Offset::MIN)).unwrap();
        let got = to_string(
            p().precision(Some(9)),
            zoned.timestamp(),
            zoned.offset(),
        );
        assert_eq!(got, "-009999-01-01T00:00:00.000000000-25:59");
    }

    #[test]
    fn print_datetime() {
        let p = || DateTimePrinter::new();
        let via_io = |dtp: DateTimePrinter, dt| {
            let mut buf = String::new();
            dtp.print_datetime(&dt, &mut StdFmtWrite(&mut buf)).unwrap();
            buf
        };
        let to_string = |dtp: DateTimePrinter, dt| {
            let mut buf = String::new();
            dtp.print_datetime(&dt, &mut buf).unwrap();
            let got_via_io = via_io(dtp, dt);
            assert_eq!(
                buf, got_via_io,
                "expected writes to `&mut String` to match `&mut StdFmtWrite`"
            );
            buf
        };

        let dt = date(2024, 3, 10).at(5, 34, 45, 0);
        let got = to_string(p(), dt);
        assert_eq!(got, "2024-03-10T05:34:45");

        let dt = date(-2024, 3, 10).at(5, 34, 45, 0);
        let got = to_string(p(), dt);
        assert_eq!(got, "-002024-03-10T05:34:45");

        let dt = date(2024, 3, 10).at(5, 34, 45, 123_456_789);
        let got = to_string(p(), dt);
        assert_eq!(got, "2024-03-10T05:34:45.123456789");

        let dt = date(2024, 3, 10).at(5, 34, 45, 0);
        let got = to_string(p().precision(Some(9)), dt);
        assert_eq!(got, "2024-03-10T05:34:45.000000000");

        let dt = DateTime::MIN;
        let got = to_string(p().precision(Some(9)), dt);
        assert_eq!(got, "-009999-01-01T00:00:00.000000000");
    }

    #[test]
    fn print_date() {
        let p = || DateTimePrinter::new();
        let via_io = |dtp: DateTimePrinter, date| {
            let mut buf = String::new();
            dtp.print_date(&date, &mut StdFmtWrite(&mut buf)).unwrap();
            buf
        };
        let to_string = |dtp: DateTimePrinter, date| {
            let mut buf = String::new();
            dtp.print_date(&date, &mut buf).unwrap();
            let got_via_io = via_io(dtp, date);
            assert_eq!(
                buf, got_via_io,
                "expected writes to `&mut String` to match `&mut StdFmtWrite`"
            );
            buf
        };

        let d = date(2024, 3, 10);
        let got = to_string(p(), d);
        assert_eq!(got, "2024-03-10");

        let d = date(-2024, 3, 10);
        let got = to_string(p(), d);
        assert_eq!(got, "-002024-03-10");

        let d = date(2024, 3, 10);
        let got = to_string(p(), d);
        assert_eq!(got, "2024-03-10");

        let d = Date::MIN;
        let got = to_string(p().precision(Some(9)), d);
        assert_eq!(got, "-009999-01-01");
    }

    #[test]
    fn print_time() {
        let p = || DateTimePrinter::new();
        let via_io = |dtp: DateTimePrinter, time| {
            let mut buf = String::new();
            dtp.print_time(&time, &mut StdFmtWrite(&mut buf)).unwrap();
            buf
        };
        let to_string = |dtp: DateTimePrinter, time| {
            let mut buf = String::new();
            dtp.print_time(&time, &mut buf).unwrap();
            let got_via_io = via_io(dtp, time);
            assert_eq!(
                buf, got_via_io,
                "expected writes to `&mut String` to match `&mut StdFmtWrite`"
            );
            buf
        };

        let t = time(5, 34, 45, 0);
        let got = to_string(p(), t);
        assert_eq!(got, "05:34:45");

        let t = time(5, 34, 45, 0);
        let got = to_string(p(), t);
        assert_eq!(got, "05:34:45");

        let t = time(5, 34, 45, 123_456_789);
        let got = to_string(p(), t);
        assert_eq!(got, "05:34:45.123456789");

        let t = time(5, 34, 45, 0);
        let got = to_string(p().precision(Some(9)), t);
        assert_eq!(got, "05:34:45.000000000");

        let t = Time::MIN;
        let got = to_string(p().precision(Some(9)), t);
        assert_eq!(got, "00:00:00.000000000");

        let t = Time::MAX;
        let got = to_string(p().precision(Some(9)), t);
        assert_eq!(got, "23:59:59.999999999");
    }

    #[test]
    fn print_iso_week_date() {
        let p = || DateTimePrinter::new();
        let via_io = |dtp: DateTimePrinter, date| {
            let mut buf = String::new();
            dtp.print_iso_week_date(&date, &mut StdFmtWrite(&mut buf))
                .unwrap();
            buf
        };
        let to_string = |dtp: DateTimePrinter, date| {
            let mut buf = String::new();
            dtp.print_iso_week_date(&date, &mut buf).unwrap();
            let got_via_io = via_io(dtp, date);
            assert_eq!(
                buf, got_via_io,
                "expected writes to `&mut String` to match `&mut StdFmtWrite`"
            );
            buf
        };

        let d = ISOWeekDate::new(2024, 52, Weekday::Monday).unwrap();
        let got = to_string(p(), d);
        assert_eq!(got, "2024-W52-1");

        let d = ISOWeekDate::new(2004, 1, Weekday::Sunday).unwrap();
        let got = to_string(p(), d);
        assert_eq!(got, "2004-W01-7");

        let d = ISOWeekDate::MIN;
        let got = to_string(p(), d);
        assert_eq!(got, "-009999-W01-1");

        let d = ISOWeekDate::MAX;
        let got = to_string(p(), d);
        assert_eq!(got, "9999-W52-5");
    }

    #[test]
    fn print_pieces() {
        let p = || DateTimePrinter::new();
        let via_io = |dtp: DateTimePrinter, pieces| {
            let mut buf = String::new();
            dtp.print_pieces(&pieces, &mut StdFmtWrite(&mut buf)).unwrap();
            buf
        };
        let to_string = |dtp: DateTimePrinter, pieces| {
            let mut buf = String::new();
            dtp.print_pieces(&pieces, &mut buf).unwrap();
            let got_via_io = via_io(dtp, pieces);
            assert_eq!(
                buf, got_via_io,
                "expected writes to `&mut String` to match `&mut StdFmtWrite`"
            );
            buf
        };

        let pieces = Pieces::from(date(2024, 3, 10).at(5, 34, 45, 0))
            .with_offset(Offset::constant(-4))
            .with_time_zone_name("America/New_York");
        let got = to_string(p(), pieces);
        assert_eq!(got, "2024-03-10T05:34:45-04:00[America/New_York]");

        let pieces = Pieces::from(date(2024, 3, 10).at(5, 34, 45, 0))
            .with_offset(Offset::UTC)
            .with_time_zone_name("UTC");
        let got = to_string(p(), pieces);
        assert_eq!(got, "2024-03-10T05:34:45+00:00[UTC]");

        let pieces = Pieces::from(date(2024, 3, 10).at(5, 34, 45, 0))
            .with_offset(Offset::MIN)
            .with_time_zone_offset(Offset::MIN);
        let got = to_string(p(), pieces);
        assert_eq!(got, "2024-03-10T05:34:45-25:59[-25:59]");

        let pieces = Pieces::from(date(2024, 3, 10).at(5, 34, 45, 0))
            .with_offset(Offset::MAX)
            .with_time_zone_offset(Offset::MAX);
        let got = to_string(p(), pieces);
        assert_eq!(got, "2024-03-10T05:34:45+25:59[+25:59]");

        let pieces =
            Pieces::from(date(2024, 3, 10).at(5, 34, 45, 123_456_789))
                .with_offset(Offset::constant(-4))
                .with_time_zone_name("America/New_York");
        let got = to_string(p(), pieces);
        assert_eq!(
            got,
            "2024-03-10T05:34:45.123456789-04:00[America/New_York]"
        );

        let pieces = Pieces::from(date(2024, 3, 10).at(5, 34, 45, 0))
            .with_offset(Offset::constant(-4))
            .with_time_zone_name("America/New_York");
        let got = to_string(p().precision(Some(9)), pieces);
        assert_eq!(
            got,
            "2024-03-10T05:34:45.000000000-04:00[America/New_York]"
        );

        let pieces =
            Pieces::from(date(-9999, 3, 1).at(23, 59, 59, 999_999_999))
                .with_offset(Offset::constant(-4))
                .with_time_zone_name("America/Argentina/ComodRivadavia");
        let got = to_string(p().precision(Some(9)), pieces);
        assert_eq!(
            got,
            "-009999-03-01T23:59:59.999999999-04:00[America/Argentina/ComodRivadavia]",
        );

        let pieces =
            Pieces::parse(
                "-009999-03-01T23:59:59.999999999-04:00[!America/Argentina/ComodRivadavia]",
            ).unwrap();
        let got = to_string(p().precision(Some(9)), pieces);
        assert_eq!(
            got,
            "-009999-03-01T23:59:59.999999999-04:00[!America/Argentina/ComodRivadavia]",
        );

        // Inject a very long IANA tzdb identifier to ensure it's handled
        let name = "Abc/Def/Ghi/Jkl/Mno/Pqr/Stu/Vwx/Yzz/Abc/Def/Ghi/Jkl/Mno/Pqr/Stu/Vwx/Yzz/Abc/Def/Ghi/Jkl/Mno/Pqr/Stu/Vwx/Yzz";
        let pieces =
            Pieces::from(date(-9999, 3, 1).at(23, 59, 59, 999_999_999))
                .with_offset(Offset::constant(-4))
                .with_time_zone_name(name);
        let got = to_string(p().precision(Some(9)), pieces);
        assert_eq!(
            got,
            "-009999-03-01T23:59:59.999999999-04:00[Abc/Def/Ghi/Jkl/Mno/Pqr/Stu/Vwx/Yzz/Abc/Def/Ghi/Jkl/Mno/Pqr/Stu/Vwx/Yzz/Abc/Def/Ghi/Jkl/Mno/Pqr/Stu/Vwx/Yzz]",
        );
    }

    #[test]
    fn print_time_zone() {
        let p = || DateTimePrinter::new();
        let via_io = |dtp: DateTimePrinter, tz| {
            let mut buf = String::new();
            dtp.print_time_zone(&tz, &mut StdFmtWrite(&mut buf)).unwrap();
            buf
        };
        let to_string = |dtp: DateTimePrinter, tz| {
            let mut buf = String::new();
            dtp.print_time_zone(&tz, &mut buf).unwrap();
            let got_via_io = via_io(dtp, tz);
            assert_eq!(
                buf, got_via_io,
                "expected writes to `&mut String` to match `&mut StdFmtWrite`"
            );
            buf
        };

        let tztest =
            crate::tz::testdata::TzifTestFile::get("America/New_York");
        let tz = TimeZone::tzif(tztest.name, tztest.data).unwrap();
        let got = to_string(p(), tz);
        assert_eq!(got, "America/New_York");

        // Inject a very long IANA tzdb identifier to ensure it's handled
        // properly.
        let tz = TimeZone::tzif(
            "Abc/Def/Ghi/Jkl/Mno/Pqr/Stu/Vwx/Yzz/Abc/Def/Ghi/Jkl/Mno/Pqr/Stu/Vwx/Yzz/Abc/Def/Ghi/Jkl/Mno/Pqr/Stu/Vwx/Yzz",
            tztest.data,
        ).unwrap();
        let got = to_string(p(), tz);
        assert_eq!(
            got,
            "Abc/Def/Ghi/Jkl/Mno/Pqr/Stu/Vwx/Yzz/Abc/Def/Ghi/Jkl/Mno/Pqr/Stu/Vwx/Yzz/Abc/Def/Ghi/Jkl/Mno/Pqr/Stu/Vwx/Yzz",
        );

        let tz = TimeZone::UTC;
        let got = to_string(p(), tz);
        assert_eq!(got, "UTC");

        let tz = TimeZone::unknown();
        let got = to_string(p(), tz);
        assert_eq!(got, "Etc/Unknown");

        let tz = TimeZone::fixed(Offset::MIN);
        let got = to_string(p(), tz);
        assert_eq!(got, "-25:59:59");

        let tz = TimeZone::fixed(Offset::MAX);
        let got = to_string(p(), tz);
        assert_eq!(got, "+25:59:59");

        let tz = TimeZone::posix("EST5EDT,M3.2.0,M11.1.0").unwrap();
        let got = to_string(p(), tz);
        assert_eq!(got, "EST5EDT,M3.2.0,M11.1.0");

        let tz = TimeZone::posix(
            "ABCDEFGHIJKLMNOPQRSTUVWXY5ABCDEFGHIJKLMNOPQRSTUVWXYT,M3.2.0,M11.1.0",
        ).unwrap();
        let got = to_string(p(), tz);
        assert_eq!(
            got,
            "ABCDEFGHIJKLMNOPQRSTUVWXY5ABCDEFGHIJKLMNOPQRSTUVWXYT,M3.2.0,M11.1.0",
        );

        // This isn't a public API, but this lets us test the error
        // case: a valid time zone but without a succinct name.
        #[cfg(feature = "tz-system")]
        {
            let tz = TimeZone::tzif_system(tztest.data).unwrap();
            let mut buf = String::new();
            let err = p().print_time_zone(&tz, &mut buf).unwrap_err();
            assert_eq!(
                err.to_string(),
                "time zones without IANA identifiers that aren't \
                 either fixed offsets or a POSIX time zone can't be \
                 serialized (this typically occurs when this is a \
                 system time zone derived from `/etc/localtime` on \
                 Unix systems that isn't symlinked to an entry in \
                 `/usr/share/zoneinfo`)",
            );
        }
    }

    #[cfg(not(miri))]
    #[test]
    fn print_span_basic() {
        let p = |span: Span| -> String {
            let mut buf = String::new();
            SpanPrinter::new().print_span(&span, &mut buf).unwrap();
            buf
        };

        insta::assert_snapshot!(p(Span::new()), @"PT0S");
        insta::assert_snapshot!(p(1.second()), @"PT1S");
        insta::assert_snapshot!(p(-1.second()), @"-PT1S");
        insta::assert_snapshot!(p(
            1.second().milliseconds(1).microseconds(1).nanoseconds(1),
        ), @"PT1.001001001S");
        insta::assert_snapshot!(p(
            0.second().milliseconds(999).microseconds(999).nanoseconds(999),
        ), @"PT0.999999999S");
        insta::assert_snapshot!(p(
            1.year().months(1).weeks(1).days(1)
            .hours(1).minutes(1).seconds(1)
            .milliseconds(1).microseconds(1).nanoseconds(1),
        ), @"P1Y1M1W1DT1H1M1.001001001S");
        insta::assert_snapshot!(p(
            -1.year().months(1).weeks(1).days(1)
            .hours(1).minutes(1).seconds(1)
            .milliseconds(1).microseconds(1).nanoseconds(1),
        ), @"-P1Y1M1W1DT1H1M1.001001001S");
    }

    #[cfg(not(miri))]
    #[test]
    fn print_span_subsecond_positive() {
        let p = |span: Span| -> String {
            let mut buf = String::new();
            SpanPrinter::new().print_span(&span, &mut buf).unwrap();
            buf
        };

        // These are all sub-second trickery tests.
        insta::assert_snapshot!(p(
            0.second().milliseconds(1000).microseconds(1000).nanoseconds(1000),
        ), @"PT1.001001S");
        insta::assert_snapshot!(p(
            1.second().milliseconds(1000).microseconds(1000).nanoseconds(1000),
        ), @"PT2.001001S");
        insta::assert_snapshot!(p(
            0.second()
            .milliseconds(b::SpanMilliseconds::MAX),
        ), @"PT631107417600S");
        insta::assert_snapshot!(p(
            0.second()
            .microseconds(b::SpanMicroseconds::MAX),
        ), @"PT631107417600S");
        insta::assert_snapshot!(p(
            0.second()
            .nanoseconds(b::SpanNanoseconds::MAX),
        ), @"PT9223372036.854775807S");

        insta::assert_snapshot!(p(
            0.second()
            .milliseconds(b::SpanMilliseconds::MAX)
            .microseconds(999_999),
        ), @"PT631107417600.999999S");
        // This is 1 microsecond more than the maximum number of seconds
        // representable in a span.
        insta::assert_snapshot!(p(
            0.second()
            .milliseconds(b::SpanMilliseconds::MAX)
            .microseconds(1_000_000),
        ), @"PT631107417601S");
        insta::assert_snapshot!(p(
            0.second()
            .milliseconds(b::SpanMilliseconds::MAX)
            .microseconds(1_000_001),
        ), @"PT631107417601.000001S");
        // This is 1 nanosecond more than the maximum number of seconds
        // representable in a span.
        insta::assert_snapshot!(p(
            0.second()
            .milliseconds(b::SpanMilliseconds::MAX)
            .nanoseconds(1_000_000_000),
        ), @"PT631107417601S");
        insta::assert_snapshot!(p(
            0.second()
            .milliseconds(b::SpanMilliseconds::MAX)
            .nanoseconds(1_000_000_001),
        ), @"PT631107417601.000000001S");

        // The max millis, micros and nanos, combined.
        insta::assert_snapshot!(p(
            0.second()
            .milliseconds(b::SpanMilliseconds::MAX)
            .microseconds(b::SpanMicroseconds::MAX)
            .nanoseconds(b::SpanNanoseconds::MAX),
        ), @"PT1271438207236.854775807S");
        // The max seconds, millis, micros and nanos, combined.
        insta::assert_snapshot!(p(
            Span::new()
            .seconds(b::SpanSeconds::MAX)
            .milliseconds(b::SpanMilliseconds::MAX)
            .microseconds(b::SpanMicroseconds::MAX)
            .nanoseconds(b::SpanNanoseconds::MAX),
        ), @"PT1902545624836.854775807S");
    }

    #[cfg(not(miri))]
    #[test]
    fn print_span_subsecond_negative() {
        let p = |span: Span| -> String {
            let mut buf = String::new();
            SpanPrinter::new().print_span(&span, &mut buf).unwrap();
            buf
        };

        // These are all sub-second trickery tests.
        insta::assert_snapshot!(p(
            -0.second().milliseconds(1000).microseconds(1000).nanoseconds(1000),
        ), @"-PT1.001001S");
        insta::assert_snapshot!(p(
            -1.second().milliseconds(1000).microseconds(1000).nanoseconds(1000),
        ), @"-PT2.001001S");
        insta::assert_snapshot!(p(
            0.second()
            .milliseconds(b::SpanMilliseconds::MIN),
        ), @"-PT631107417600S");
        insta::assert_snapshot!(p(
            0.second()
            .microseconds(b::SpanMicroseconds::MIN),
        ), @"-PT631107417600S");
        insta::assert_snapshot!(p(
            0.second()
            .nanoseconds(b::SpanNanoseconds::MIN),
        ), @"-PT9223372036.854775807S");

        insta::assert_snapshot!(p(
            0.second()
            .milliseconds(b::SpanMilliseconds::MIN)
            .microseconds(999_999),
        ), @"-PT631107417600.999999S");
        // This is 1 microsecond more than the maximum number of seconds
        // representable in a span.
        insta::assert_snapshot!(p(
            0.second()
            .milliseconds(b::SpanMilliseconds::MIN)
            .microseconds(1_000_000),
        ), @"-PT631107417601S");
        insta::assert_snapshot!(p(
            0.second()
            .milliseconds(b::SpanMilliseconds::MIN)
            .microseconds(1_000_001),
        ), @"-PT631107417601.000001S");
        // This is 1 nanosecond more than the maximum number of seconds
        // representable in a span.
        insta::assert_snapshot!(p(
            0.second()
            .milliseconds(b::SpanMilliseconds::MIN)
            .nanoseconds(1_000_000_000),
        ), @"-PT631107417601S");
        insta::assert_snapshot!(p(
            0.second()
            .milliseconds(b::SpanMilliseconds::MIN)
            .nanoseconds(1_000_000_001),
        ), @"-PT631107417601.000000001S");

        // The max millis, micros and nanos, combined.
        insta::assert_snapshot!(p(
            0.second()
            .milliseconds(b::SpanMilliseconds::MIN)
            .microseconds(b::SpanMicroseconds::MIN)
            .nanoseconds(b::SpanNanoseconds::MIN),
        ), @"-PT1271438207236.854775807S");
        // The max seconds, millis, micros and nanos, combined.
        insta::assert_snapshot!(p(
            Span::new()
            .seconds(b::SpanSeconds::MIN)
            .milliseconds(b::SpanMilliseconds::MIN)
            .microseconds(b::SpanMicroseconds::MIN)
            .nanoseconds(b::SpanNanoseconds::MIN),
        ), @"-PT1902545624836.854775807S");
    }

    #[cfg(not(miri))]
    #[test]
    fn print_signed_duration() {
        let p = |secs, nanos| -> String {
            let dur = SignedDuration::new(secs, nanos);
            let mut buf = String::new();
            SpanPrinter::new().print_signed_duration(&dur, &mut buf).unwrap();
            buf
        };

        insta::assert_snapshot!(p(0, 0), @"PT0S");
        insta::assert_snapshot!(p(0, 1), @"PT0.000000001S");
        insta::assert_snapshot!(p(1, 0), @"PT1S");
        insta::assert_snapshot!(p(59, 0), @"PT59S");
        insta::assert_snapshot!(p(60, 0), @"PT1M");
        insta::assert_snapshot!(p(60, 1), @"PT1M0.000000001S");
        insta::assert_snapshot!(p(61, 1), @"PT1M1.000000001S");
        insta::assert_snapshot!(p(3_600, 0), @"PT1H");
        insta::assert_snapshot!(p(3_600, 1), @"PT1H0.000000001S");
        insta::assert_snapshot!(p(3_660, 0), @"PT1H1M");
        insta::assert_snapshot!(p(3_660, 1), @"PT1H1M0.000000001S");
        insta::assert_snapshot!(p(3_661, 0), @"PT1H1M1S");
        insta::assert_snapshot!(p(3_661, 1), @"PT1H1M1.000000001S");

        insta::assert_snapshot!(p(0, -1), @"-PT0.000000001S");
        insta::assert_snapshot!(p(-1, 0), @"-PT1S");
        insta::assert_snapshot!(p(-59, 0), @"-PT59S");
        insta::assert_snapshot!(p(-60, 0), @"-PT1M");
        insta::assert_snapshot!(p(-60, -1), @"-PT1M0.000000001S");
        insta::assert_snapshot!(p(-61, -1), @"-PT1M1.000000001S");
        insta::assert_snapshot!(p(-3_600, 0), @"-PT1H");
        insta::assert_snapshot!(p(-3_600, -1), @"-PT1H0.000000001S");
        insta::assert_snapshot!(p(-3_660, 0), @"-PT1H1M");
        insta::assert_snapshot!(p(-3_660, -1), @"-PT1H1M0.000000001S");
        insta::assert_snapshot!(p(-3_661, 0), @"-PT1H1M1S");
        insta::assert_snapshot!(p(-3_661, -1), @"-PT1H1M1.000000001S");

        insta::assert_snapshot!(
            p(i64::MIN, -999_999_999),
            @"-PT2562047788015215H30M8.999999999S",
        );
        insta::assert_snapshot!(
            p(i64::MAX, 999_999_999),
            @"PT2562047788015215H30M7.999999999S",
        );
    }

    #[cfg(not(miri))]
    #[test]
    fn print_unsigned_duration() {
        let p = |secs, nanos| -> String {
            let dur = Duration::new(secs, nanos);
            let mut buf = String::new();
            SpanPrinter::new()
                .print_unsigned_duration(&dur, &mut buf)
                .unwrap();
            buf
        };

        insta::assert_snapshot!(p(0, 0), @"PT0S");
        insta::assert_snapshot!(p(0, 1), @"PT0.000000001S");
        insta::assert_snapshot!(p(1, 0), @"PT1S");
        insta::assert_snapshot!(p(59, 0), @"PT59S");
        insta::assert_snapshot!(p(60, 0), @"PT1M");
        insta::assert_snapshot!(p(60, 1), @"PT1M0.000000001S");
        insta::assert_snapshot!(p(61, 1), @"PT1M1.000000001S");
        insta::assert_snapshot!(p(3_600, 0), @"PT1H");
        insta::assert_snapshot!(p(3_600, 1), @"PT1H0.000000001S");
        insta::assert_snapshot!(p(3_660, 0), @"PT1H1M");
        insta::assert_snapshot!(p(3_660, 1), @"PT1H1M0.000000001S");
        insta::assert_snapshot!(p(3_661, 0), @"PT1H1M1S");
        insta::assert_snapshot!(p(3_661, 1), @"PT1H1M1.000000001S");

        insta::assert_snapshot!(
            p(u64::MAX, 999_999_999),
            @"PT5124095576030431H15.999999999S",
        );
    }
}
