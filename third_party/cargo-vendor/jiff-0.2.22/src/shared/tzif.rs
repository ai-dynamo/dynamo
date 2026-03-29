use alloc::{string::String, vec};

use super::{
    util::{
        array_str::Abbreviation,
        itime::{IOffset, ITimestamp},
    },
    PosixTimeZone, TzifDateTime, TzifFixed, TzifIndicator, TzifLocalTimeType,
    TzifOwned, TzifTransitionInfo, TzifTransitionKind, TzifTransitions,
    TzifTransitionsOwned,
};

// These are Jiff min and max timestamp (in seconds) values.
//
// The TZif parser will clamp timestamps to this range. It's
// not ideal, but Jiff can't handle values outside of this range
// and completely refusing to use TZif data with pathological
// timestamps in typically irrelevant transitions is bad juju.
//
// Ref: https://github.com/BurntSushi/jiff/issues/163
// Ref: https://github.com/BurntSushi/jiff/pull/164
const TIMESTAMP_MIN: i64 = -377705023201;
const TIMESTAMP_MAX: i64 = 253402207200;

// Similarly for offsets, although in this case, if we find
// an offset outside of this range, we do actually error. This
// is because it could result in true incorrect datetimes for
// actual transitions.
//
// But our supported offset range is `-25:59:59..=+25:59:59`.
// There's no real time zone with offsets even close to those
// boundaries.
//
// If there is pathological data that we should ignore, then
// we should wait for a real bug report in order to determine
// the right way to ignore/clamp it.
const OFFSET_MIN: i32 = -93599;
const OFFSET_MAX: i32 = 93599;

// When fattening TZif data, this is the year to go up to.
//
// This year was chosen because it's what the "fat" TZif data generated
// by `zic` uses.
const FATTEN_UP_TO_YEAR: i16 = 2038;

// This is a "sanity" limit on the maximum number of transitions we'll
// add to TZif data when fattening them up.
//
// This is mostly just a defense-in-depth limit to avoid weird cases
// where a pathological POSIX time zone could be defined to create
// many transitions. It's not clear that this is actually possible,
// but I felt a little uneasy doing unbounded work that isn't linearly
// proportional to the input data. So, this limit is put into place for
// reasons of "good sense."
//
// For "normal" cases, there should be at most two transitions per
// year. So this limit permits 300/2=150 years of transition data.
// (Although we won't go above `FATTEN_UP_TO_YEAR`. See above.)
const FATTEN_MAX_TRANSITIONS: usize = 300;

impl TzifOwned {
    /// Parses the given data as a TZif formatted file.
    ///
    /// The name given is attached to the `Tzif` value returned, but is
    /// otherwise not significant.
    ///
    /// If the given data is not recognized to be valid TZif, then an error is
    /// returned.
    ///
    /// In general, callers may assume that it is safe to pass arbitrary or
    /// even untrusted data to this function and count on it not panicking
    /// or using resources that aren't limited to a small constant factor of
    /// the size of the data itself. That is, callers can reliably limit the
    /// resources used by limiting the size of the data given to this parse
    /// function.
    pub(crate) fn parse(
        name: Option<String>,
        bytes: &[u8],
    ) -> Result<TzifOwned, TzifError> {
        let original = bytes;
        let name = name.into();
        let (header32, rest) =
            Header::parse(4, bytes).map_err(TzifErrorKind::Header32)?;
        let (mut tzif, rest) = if header32.version == 0 {
            TzifOwned::parse32(name, header32, rest)?
        } else {
            TzifOwned::parse64(name, header32, rest)?
        };
        tzif.fatten();
        // This should come after fattening, because fattening may add new
        // transitions and we want to add civil datetimes to those.
        tzif.add_civil_datetimes_to_transitions();
        tzif.verify_posix_time_zone_consistency()?;
        // Compute the checksum using the entire contents of the TZif data.
        let tzif_raw_len = (rest.as_ptr() as usize)
            .checked_sub(original.as_ptr() as usize)
            .unwrap();
        let tzif_raw_bytes = &original[..tzif_raw_len];
        tzif.fixed.checksum = super::crc32::sum(tzif_raw_bytes);

        // Shrink all of our allocs so we don't keep excess capacity around.
        tzif.fixed.designations.shrink_to_fit();
        tzif.types.shrink_to_fit();
        tzif.transitions.timestamps.shrink_to_fit();
        tzif.transitions.civil_starts.shrink_to_fit();
        tzif.transitions.civil_ends.shrink_to_fit();
        tzif.transitions.infos.shrink_to_fit();

        Ok(tzif)
    }

    fn parse32<'b>(
        name: Option<String>,
        header32: Header,
        bytes: &'b [u8],
    ) -> Result<(TzifOwned, &'b [u8]), TzifError> {
        let mut tzif = TzifOwned {
            fixed: TzifFixed {
                name,
                version: header32.version,
                // filled in later
                checksum: 0,
                designations: String::new(),
                posix_tz: None,
            },
            types: vec![],
            transitions: TzifTransitions {
                timestamps: vec![],
                civil_starts: vec![],
                civil_ends: vec![],
                infos: vec![],
            },
        };
        let rest = tzif.parse_transitions(&header32, bytes)?;
        let rest = tzif.parse_transition_types(&header32, rest)?;
        let rest = tzif.parse_local_time_types(&header32, rest)?;
        let rest = tzif.parse_time_zone_designations(&header32, rest)?;
        let rest = tzif.parse_leap_seconds(&header32, rest)?;
        let rest = tzif.parse_indicators(&header32, rest)?;
        Ok((tzif, rest))
    }

    fn parse64<'b>(
        name: Option<String>,
        header32: Header,
        bytes: &'b [u8],
    ) -> Result<(TzifOwned, &'b [u8]), TzifError> {
        let (_, rest) =
            try_split_at(SplitAtError::V1, bytes, header32.data_block_len()?)?;
        let (header64, rest) =
            Header::parse(8, rest).map_err(TzifErrorKind::Header64)?;
        let mut tzif = TzifOwned {
            fixed: TzifFixed {
                name,
                version: header64.version,
                // filled in later
                checksum: 0,
                designations: String::new(),
                posix_tz: None,
            },
            types: vec![],
            transitions: TzifTransitions {
                timestamps: vec![],
                civil_starts: vec![],
                civil_ends: vec![],
                infos: vec![],
            },
        };
        let rest = tzif.parse_transitions(&header64, rest)?;
        let rest = tzif.parse_transition_types(&header64, rest)?;
        let rest = tzif.parse_local_time_types(&header64, rest)?;
        let rest = tzif.parse_time_zone_designations(&header64, rest)?;
        let rest = tzif.parse_leap_seconds(&header64, rest)?;
        let rest = tzif.parse_indicators(&header64, rest)?;
        let rest = tzif.parse_footer(&header64, rest)?;
        // Note that we don't check that the TZif data is fully valid. It is
        // possible for it to contain superfluous information. For example, a
        // non-zero local time type that is never referenced by a transition.
        Ok((tzif, rest))
    }

    fn parse_transitions<'b>(
        &mut self,
        header: &Header,
        bytes: &'b [u8],
    ) -> Result<&'b [u8], TzifError> {
        let (bytes, rest) = try_split_at(
            SplitAtError::TransitionTimes,
            bytes,
            header.transition_times_len()?,
        )?;
        let mut it = bytes.chunks_exact(header.time_size);
        // RFC 8536 says: "If there are no transitions, local time for all
        // timestamps is specified by the TZ string in the footer if present
        // and nonempty; otherwise, it is specified by time type 0."
        //
        // RFC 8536 also says: "Local time for timestamps before the first
        // transition is specified by the first time type (time type
        // 0)."
        //
        // So if there are no transitions, pushing this dummy one will result
        // in the desired behavior even when it's the only transition.
        // Similarly, since this is the minimum timestamp value, it will
        // trigger for any times before the first transition found in the TZif
        // data.
        self.transitions.add_with_type_index(TIMESTAMP_MIN, 0);
        while let Some(chunk) = it.next() {
            let mut timestamp = if header.is_32bit() {
                i64::from(from_be_bytes_i32(chunk))
            } else {
                from_be_bytes_i64(chunk)
            };
            if !(TIMESTAMP_MIN <= timestamp && timestamp <= TIMESTAMP_MAX) {
                // We really shouldn't error here just because the Unix
                // timestamp is outside what Jiff supports. Since what Jiff
                // supports is _somewhat_ arbitrary. But Jiff's supported
                // range is good enough for all realistic purposes, so we
                // just clamp an out-of-range Unix timestamp to the Jiff
                // min or max value.
                //
                // This can't result in the sorting order being wrong, but
                // it can result in a transition that is duplicative with
                // the dummy transition we inserted above. This should be
                // fine.
                let clamped = timestamp.clamp(TIMESTAMP_MIN, TIMESTAMP_MAX);
                // only-jiff-start
                warn!(
                    "found Unix timestamp `{timestamp}` that is outside \
                     Jiff's supported range, clamping to `{clamped}`",
                );
                // only-jiff-end
                timestamp = clamped;
            }
            self.transitions.add(timestamp);
        }
        assert!(it.remainder().is_empty());
        Ok(rest)
    }

    fn parse_transition_types<'b>(
        &mut self,
        header: &Header,
        bytes: &'b [u8],
    ) -> Result<&'b [u8], TransitionTypeError> {
        let (bytes, rest) = try_split_at(
            SplitAtError::TransitionTypes,
            bytes,
            header.transition_types_len(),
        )?;
        // We skip the first transition because it is our minimum dummy
        // transition.
        for (transition_index, &type_index) in (1..).zip(bytes) {
            if usize::from(type_index) >= header.tzh_typecnt {
                return Err(TransitionTypeError::ExceedsLocalTimeTypes);
            }
            self.transitions.infos[transition_index].type_index = type_index;
        }
        Ok(rest)
    }

    fn parse_local_time_types<'b>(
        &mut self,
        header: &Header,
        bytes: &'b [u8],
    ) -> Result<&'b [u8], TzifError> {
        let (bytes, rest) = try_split_at(
            SplitAtError::LocalTimeTypes,
            bytes,
            header.local_time_types_len()?,
        )?;
        let mut it = bytes.chunks_exact(6);
        while let Some(chunk) = it.next() {
            let offset = from_be_bytes_i32(&chunk[..4]);
            if !(OFFSET_MIN <= offset && offset <= OFFSET_MAX) {
                return Err(TzifError::from(
                    LocalTimeTypeError::InvalidOffset { offset },
                ));
            }
            let is_dst = chunk[4] == 1;
            let designation = (chunk[5], chunk[5]);
            self.types.push(TzifLocalTimeType {
                offset,
                is_dst,
                designation,
                indicator: TzifIndicator::LocalWall,
            });
        }
        assert!(it.remainder().is_empty());
        Ok(rest)
    }

    fn parse_time_zone_designations<'b>(
        &mut self,
        header: &Header,
        bytes: &'b [u8],
    ) -> Result<&'b [u8], TimeZoneDesignatorError> {
        let (bytes, rest) = try_split_at(
            SplitAtError::TimeZoneDesignations,
            bytes,
            header.time_zone_designations_len(),
        )?;
        self.fixed.designations = String::from_utf8(bytes.to_vec())
            .map_err(|_| TimeZoneDesignatorError::InvalidUtf8)?;
        // Holy hell, this is brutal. The boundary conditions are crazy.
        for typ in self.types.iter_mut() {
            let start = usize::from(typ.designation.0);
            let suffix = self
                .fixed
                .designations
                .get(start..)
                .ok_or(TimeZoneDesignatorError::InvalidStart)?;
            let len = suffix
                .find('\x00')
                .ok_or(TimeZoneDesignatorError::MissingNul)?;
            let end = start
                .checked_add(len)
                .ok_or(TimeZoneDesignatorError::InvalidLength)?;
            typ.designation.1 = u8::try_from(end)
                .map_err(|_| TimeZoneDesignatorError::InvalidEnd)?;
        }
        Ok(rest)
    }

    /// This parses the leap second corrections in the TZif data.
    ///
    /// Note that we only parse and verify them. We don't actually use them.
    /// Jiff effectively ignores leap seconds.
    fn parse_leap_seconds<'b>(
        &mut self,
        header: &Header,
        bytes: &'b [u8],
    ) -> Result<&'b [u8], TzifError> {
        let (bytes, rest) = try_split_at(
            SplitAtError::LeapSeconds,
            bytes,
            header.leap_second_len()?,
        )?;
        let chunk_len = header
            .time_size
            .checked_add(4)
            .expect("time_size plus 4 fits in usize");
        let mut it = bytes.chunks_exact(chunk_len);
        while let Some(chunk) = it.next() {
            let (occur_bytes, _corr_bytes) = chunk.split_at(header.time_size);
            let occur = if header.is_32bit() {
                i64::from(from_be_bytes_i32(occur_bytes))
            } else {
                from_be_bytes_i64(occur_bytes)
            };
            if !(TIMESTAMP_MIN <= occur && occur <= TIMESTAMP_MAX) {
                // only-jiff-start
                warn!(
                    "leap second occurrence `{occur}` is \
                     not in Jiff's supported range"
                )
                // only-jiff-end
            }
        }
        assert!(it.remainder().is_empty());
        Ok(rest)
    }

    fn parse_indicators<'b>(
        &mut self,
        header: &Header,
        bytes: &'b [u8],
    ) -> Result<&'b [u8], IndicatorError> {
        let (std_wall_bytes, rest) = try_split_at(
            SplitAtError::StandardWallIndicators,
            bytes,
            header.standard_wall_len(),
        )?;
        let (ut_local_bytes, rest) = try_split_at(
            SplitAtError::UTLocalIndicators,
            rest,
            header.ut_local_len(),
        )?;
        if std_wall_bytes.is_empty() && !ut_local_bytes.is_empty() {
            // This is a weird case, but technically possible only if all
            // UT/local indicators are 0. If any are 1, then it's an error,
            // because it would require the corresponding std/wall indicator
            // to be 1 too. Which it can't be, because there aren't any. So
            // we just check that they're all zeros.
            if ut_local_bytes.iter().any(|&byte| byte != 0) {
                return Err(IndicatorError::UtLocalNonZero);
            }
        } else if !std_wall_bytes.is_empty() && ut_local_bytes.is_empty() {
            for (i, &byte) in std_wall_bytes.iter().enumerate() {
                // Indexing is OK because Header guarantees that the number of
                // indicators is 0 or equal to the number of types.
                self.types[i].indicator = if byte == 0 {
                    TzifIndicator::LocalWall
                } else if byte == 1 {
                    TzifIndicator::LocalStandard
                } else {
                    return Err(IndicatorError::InvalidStdWallIndicator);
                };
            }
        } else if !std_wall_bytes.is_empty() && !ut_local_bytes.is_empty() {
            assert_eq!(std_wall_bytes.len(), ut_local_bytes.len());
            let it = std_wall_bytes.iter().zip(ut_local_bytes);
            for (i, (&stdwall, &utlocal)) in it.enumerate() {
                // Indexing is OK because Header guarantees that the number of
                // indicators is 0 or equal to the number of types.
                self.types[i].indicator = match (stdwall, utlocal) {
                    (0, 0) => TzifIndicator::LocalWall,
                    (1, 0) => TzifIndicator::LocalStandard,
                    (1, 1) => TzifIndicator::UTStandard,
                    (0, 1) => {
                        return Err(IndicatorError::InvalidUtWallCombination);
                    }
                    _ => return Err(IndicatorError::InvalidCombination),
                };
            }
        } else {
            // If they're both empty then we don't need to do anything. Every
            // local time type record already has the correct default for this
            // case set.
            debug_assert!(std_wall_bytes.is_empty());
            debug_assert!(ut_local_bytes.is_empty());
        }
        Ok(rest)
    }

    fn parse_footer<'b>(
        &mut self,
        _header: &Header,
        bytes: &'b [u8],
    ) -> Result<&'b [u8], FooterError> {
        if bytes.is_empty() {
            return Err(FooterError::UnexpectedEnd);
        }
        if bytes[0] != b'\n' {
            return Err(FooterError::MismatchEnd);
        }
        let bytes = &bytes[1..];
        // Only scan up to 1KB for a NUL terminator in case we somehow got
        // passed a huge block of bytes.
        let toscan = &bytes[..bytes.len().min(1024)];
        let nlat = toscan
            .iter()
            .position(|&b| b == b'\n')
            .ok_or(FooterError::TerminatorNotFound)?;
        let (bytes, rest) = bytes.split_at(nlat);
        if !bytes.is_empty() {
            // We could in theory limit TZ strings to their strict POSIX
            // definition here for TZif V2, but I don't think there is any
            // harm in allowing the extensions in V2 formatted TZif data. Note
            // that GNU tooling allows it via the `TZ` environment variable
            // even though POSIX doesn't specify it. This all seems okay to me
            // because the V3+ extension is a strict superset of functionality.
            let posix_tz = PosixTimeZone::parse(bytes)
                .map_err(FooterError::InvalidPosixTz)?;
            self.fixed.posix_tz = Some(posix_tz);
        }
        Ok(&rest[1..])
    }

    /// Validates that the POSIX TZ string we parsed (if one exists) is
    /// consistent with the last transition in this time zone. This is
    /// required by RFC 8536.
    ///
    /// RFC 8536 says, "If the string is nonempty and one or more
    /// transitions appear in the version 2+ data, the string MUST be
    /// consistent with the last version 2+ transition."
    fn verify_posix_time_zone_consistency(
        &self,
    ) -> Result<(), InconsistentPosixTimeZoneError> {
        // We need to be a little careful, since we always have at least one
        // transition (accounting for the dummy `Timestamp::MIN` transition).
        // So if we only have 1 transition and a POSIX TZ string, then we
        // should not validate it since it's equivalent to the case of 0
        // transitions and a POSIX TZ string.
        if self.transitions.timestamps.len() <= 1 {
            return Ok(());
        }
        let Some(ref tz) = self.fixed.posix_tz else {
            return Ok(());
        };
        let last = self
            .transitions
            .timestamps
            .last()
            .expect("last transition timestamp");
        let type_index = self
            .transitions
            .infos
            .last()
            .expect("last transition info")
            .type_index;
        let typ = &self.types[usize::from(type_index)];
        let (ioff, abbrev, is_dst) =
            tz.to_offset_info(ITimestamp::from_second(*last));
        if ioff.second != typ.offset {
            return Err(InconsistentPosixTimeZoneError::Offset);
        }
        if is_dst != typ.is_dst {
            return Err(InconsistentPosixTimeZoneError::Dst);
        }
        if abbrev != self.designation(&typ) {
            return Err(InconsistentPosixTimeZoneError::Designation);
        }
        Ok(())
    }

    /// Add civil datetimes to our transitions.
    ///
    /// This isn't strictly necessary, but it speeds up time zone lookups when
    /// the input is a civil datetime. It lets us do comparisons directly on
    /// the civil datetime as given, instead of needing to convert the civil
    /// datetime given to a timestamp first. (Even if we didn't do this, I
    /// believe we'd still need at least one additional timestamp that is
    /// offset, because TZ lookups for a civil datetime are done in local time,
    /// and the timestamps in TZif data are, of course, all in UTC.)
    fn add_civil_datetimes_to_transitions(&mut self) {
        fn to_datetime(timestamp: i64, offset: i32) -> TzifDateTime {
            let its = ITimestamp { second: timestamp, nanosecond: 0 };
            let ioff = IOffset { second: offset };
            let dt = its.to_datetime(ioff);
            TzifDateTime::new(
                dt.date.year,
                dt.date.month,
                dt.date.day,
                dt.time.hour,
                dt.time.minute,
                dt.time.second,
            )
        }

        let trans = &mut self.transitions;
        // Special case the first transition, which is always a dummy
        // transition establishing the lower bound. The loop below doesn't
        // handle this correctly because it isn't guaranteed to get a
        // `DateTime::MIN` value.
        trans.infos[0].kind = TzifTransitionKind::Unambiguous;
        trans.civil_starts[0] = TzifDateTime::MIN;
        for i in 1..trans.timestamps.len() {
            let timestamp = trans.timestamps[i];
            let offset = {
                let type_index = trans.infos[i].type_index;
                self.types[usize::from(type_index)].offset
            };
            let prev_offset = {
                let type_index = trans.infos[i.saturating_sub(1)].type_index;
                self.types[usize::from(type_index)].offset
            };

            if prev_offset == offset {
                // Equivalent offsets means there can never be any ambiguity.
                let start = to_datetime(timestamp, prev_offset);
                trans.infos[i].kind = TzifTransitionKind::Unambiguous;
                trans.civil_starts[i] = start;
            } else if prev_offset < offset {
                // When the offset of the previous transition is less, that
                // means there is some non-zero amount of time that is
                // "skipped" when moving to the next transition. Thus, we have
                // a gap. The start of the gap is the offset which gets us the
                // earliest time, i.e., the smaller of the two offsets.
                trans.infos[i].kind = TzifTransitionKind::Gap;
                trans.civil_starts[i] = to_datetime(timestamp, prev_offset);
                trans.civil_ends[i] = to_datetime(timestamp, offset);
            } else {
                // When the offset of the previous transition is greater, that
                // means there is some non-zero amount of time that will be
                // replayed on a wall clock in this time zone. Thus, we have
                // a fold. The start of the gold is the offset which gets us
                // the earliest time, i.e., the smaller of the two offsets.
                assert!(prev_offset > offset);
                trans.infos[i].kind = TzifTransitionKind::Fold;
                trans.civil_starts[i] = to_datetime(timestamp, offset);
                trans.civil_ends[i] = to_datetime(timestamp, prev_offset);
            }
        }
    }

    /// Fatten up this TZif data with additional transitions.
    ///
    /// These additional transitions often make time zone lookups faster, and
    /// they smooth out the performance difference between using "slim" and
    /// "fat" tzdbs.
    fn fatten(&mut self) {
        // Note that this is a crate feature for *both* `jiff` and
        // `jiff-static`.
        if !cfg!(feature = "tz-fat") {
            return;
        }
        let Some(posix_tz) = self.fixed.posix_tz.clone() else { return };
        let last =
            self.transitions.timestamps.last().expect("last transition");
        let mut i = 0;
        let mut prev = ITimestamp::from_second(*last);
        loop {
            if i > FATTEN_MAX_TRANSITIONS {
                // only-jiff-start
                warn!(
                    "fattening TZif data for `{name:?}` somehow generated \
                     more than {max} transitions, so giving up to avoid \
                     doing too much work",
                    name = self.fixed.name,
                    max = FATTEN_MAX_TRANSITIONS,
                );
                // only-jiff-end
                return;
            }
            i += 1;
            prev = match self.add_transition(&posix_tz, prev) {
                None => break,
                Some(next) => next,
            };
        }
    }

    /// If there's a transition strictly after the given timestamp for the
    /// given POSIX time zone, then add it to this TZif data.
    fn add_transition(
        &mut self,
        posix_tz: &PosixTimeZone<Abbreviation>,
        prev: ITimestamp,
    ) -> Option<ITimestamp> {
        let (its, ioff, abbrev, is_dst) = posix_tz.next_transition(prev)?;
        if its.to_datetime(IOffset::UTC).date.year >= FATTEN_UP_TO_YEAR {
            return None;
        }
        let type_index =
            self.find_or_create_local_time_type(ioff, abbrev, is_dst)?;
        self.transitions.add_with_type_index(its.second, type_index);
        Some(its)
    }

    /// Look for a local time type matching the data given.
    ///
    /// If one could not be found, then one is created and its index is
    /// returned.
    ///
    /// If one could not be found and one could not be created (e.g., the index
    /// would overflow `u8`), then `None` is returned.
    fn find_or_create_local_time_type(
        &mut self,
        offset: IOffset,
        abbrev: &str,
        is_dst: bool,
    ) -> Option<u8> {
        for (i, typ) in self.types.iter().enumerate() {
            if offset.second == typ.offset
                && abbrev == self.designation(typ)
                && is_dst == typ.is_dst
            {
                return u8::try_from(i).ok();
            }
        }
        let i = u8::try_from(self.types.len()).ok()?;
        let designation = self.find_or_create_designation(abbrev)?;
        self.types.push(TzifLocalTimeType {
            offset: offset.second,
            is_dst,
            designation,
            // Not really clear if this is correct, but Jiff
            // ignores this anyway, so ¯\_(ツ)_/¯.
            indicator: TzifIndicator::LocalWall,
        });
        Some(i)
    }

    /// Look for a designation (i.e., time zone abbreviation) matching the data
    /// given, and return its range into `self.fixed.designations`.
    ///
    /// If one could not be found, then one is created and its range is
    /// returned.
    ///
    /// If one could not be found and one could not be created (e.g., the range
    /// would overflow `u8`), then `None` is returned.
    fn find_or_create_designation(
        &mut self,
        needle: &str,
    ) -> Option<(u8, u8)> {
        let mut start = 0;
        while let Some(offset) = self.fixed.designations[start..].find('\0') {
            let end = start + offset;
            let abbrev = &self.fixed.designations[start..end];
            if needle == abbrev {
                return Some((start.try_into().ok()?, end.try_into().ok()?));
            }
            start = end + 1;
        }

        // Now we need to add a new abbreviation. This
        // should generally only happen for malformed TZif
        // data. i.e., TZif data with a POSIX time zone that
        // contains an TZ abbreviation that isn't found in
        // the TZif's designation list.
        //
        // And since we're guarding against malformed data,
        // the designation list might not end with NUL. If
        // not, add one.
        if !self.fixed.designations.ends_with('\0') {
            self.fixed.designations.push('\0');
        }
        let start = self.fixed.designations.len();
        self.fixed.designations.push_str(needle);
        self.fixed.designations.push('\0');
        let end = self.fixed.designations.len();
        Some((start.try_into().ok()?, end.try_into().ok()?))
    }

    fn designation(&self, typ: &TzifLocalTimeType) -> &str {
        let range =
            usize::from(typ.designation.0)..usize::from(typ.designation.1);
        // OK because we verify that the designation range on every local
        // time type is a valid range into `self.designations`.
        &self.fixed.designations[range]
    }
}

impl TzifTransitionsOwned {
    /// Add a single transition with the given timestamp.
    ///
    /// This also fills in the other columns (civil starts, civil ends and
    /// infos) with sensible default values. It is expected that callers will
    /// later fill them in.
    fn add(&mut self, timestamp: i64) {
        self.add_with_type_index(timestamp, 0);
    }

    /// Like `TzifTransitionsOwned::add`, but let's the caller provide a type
    /// index if it is known.
    fn add_with_type_index(&mut self, timestamp: i64, type_index: u8) {
        self.timestamps.push(timestamp);
        self.civil_starts.push(TzifDateTime::ZERO);
        self.civil_ends.push(TzifDateTime::ZERO);
        self.infos.push(TzifTransitionInfo {
            type_index,
            kind: TzifTransitionKind::Unambiguous,
        });
    }
}

/// The header for a TZif formatted file.
///
/// V2+ TZif format have two headers: one for V1 data, and then a second
/// following the V1 data block that describes another data block which uses
/// 64-bit timestamps. The two headers both have the same format and both
/// use 32-bit big-endian encoded integers.
#[derive(Debug)]
struct Header {
    /// The size of the timestamps encoded in the data block.
    ///
    /// This is guaranteed to be either 4 (for V1) or 8 (for the 64-bit header
    /// block in V2+).
    time_size: usize,
    /// The file format version.
    ///
    /// Note that this is either a NUL byte (for version 1), or an ASCII byte
    /// corresponding to the version number. That is, `0x32` for `2`, `0x33`
    /// for `3` or `0x34` for `4`. Note also that just because zoneinfo might
    /// have been recently generated does not mean it uses the latest format
    /// version. It seems like newer versions are only compiled by `zic` when
    /// they are needed. For example, `America/New_York` on my system (as of
    /// `2024-03-25`) has version `0x32`, but `Asia/Jerusalem` has version
    /// `0x33`.
    version: u8,
    /// Number of UT/local indicators stored in the file.
    ///
    /// This is checked to be either equal to `0` or equal to `tzh_typecnt`.
    tzh_ttisutcnt: usize,
    /// The number of standard/wall indicators stored in the file.
    ///
    /// This is checked to be either equal to `0` or equal to `tzh_typecnt`.
    tzh_ttisstdcnt: usize,
    /// The number of leap seconds for which data entries are stored in the
    /// file.
    tzh_leapcnt: usize,
    /// The number of transition times for which data entries are stored in
    /// the file.
    tzh_timecnt: usize,
    /// The number of local time types for which data entries are stored in the
    /// file.
    ///
    /// This is checked to be at least `1`.
    tzh_typecnt: usize,
    /// The number of bytes of time zone abbreviation strings stored in the
    /// file.
    ///
    /// This is checked to be at least `1`.
    tzh_charcnt: usize,
}

impl Header {
    /// Parse the header record from the given bytes.
    ///
    /// Upon success, return the header and all bytes after the header.
    ///
    /// The given `time_size` must be 4 or 8, corresponding to either the
    /// V1 header block or the V2+ header block, respectively.
    fn parse(
        time_size: usize,
        bytes: &[u8],
    ) -> Result<(Header, &[u8]), HeaderError> {
        assert!(time_size == 4 || time_size == 8, "time size must be 4 or 8");
        if bytes.len() < 44 {
            return Err(HeaderError::TooShort);
        }
        let (magic, rest) = bytes.split_at(4);
        if magic != b"TZif" {
            return Err(HeaderError::MismatchMagic);
        }
        let (version, rest) = rest.split_at(1);
        let (_reserved, rest) = rest.split_at(15);

        let (tzh_ttisutcnt_bytes, rest) = rest.split_at(4);
        let (tzh_ttisstdcnt_bytes, rest) = rest.split_at(4);
        let (tzh_leapcnt_bytes, rest) = rest.split_at(4);
        let (tzh_timecnt_bytes, rest) = rest.split_at(4);
        let (tzh_typecnt_bytes, rest) = rest.split_at(4);
        let (tzh_charcnt_bytes, rest) = rest.split_at(4);

        let tzh_ttisutcnt =
            from_be_bytes_u32_to_usize(tzh_ttisutcnt_bytes).map_err(|e| {
                HeaderError::ParseCount { kind: CountKind::Ut, convert: e }
            })?;
        let tzh_ttisstdcnt =
            from_be_bytes_u32_to_usize(tzh_ttisstdcnt_bytes).map_err(|e| {
                HeaderError::ParseCount { kind: CountKind::Std, convert: e }
            })?;
        let tzh_leapcnt =
            from_be_bytes_u32_to_usize(tzh_leapcnt_bytes).map_err(|e| {
                HeaderError::ParseCount { kind: CountKind::Leap, convert: e }
            })?;
        let tzh_timecnt =
            from_be_bytes_u32_to_usize(tzh_timecnt_bytes).map_err(|e| {
                HeaderError::ParseCount { kind: CountKind::Time, convert: e }
            })?;
        let tzh_typecnt =
            from_be_bytes_u32_to_usize(tzh_typecnt_bytes).map_err(|e| {
                HeaderError::ParseCount { kind: CountKind::Type, convert: e }
            })?;
        let tzh_charcnt =
            from_be_bytes_u32_to_usize(tzh_charcnt_bytes).map_err(|e| {
                HeaderError::ParseCount { kind: CountKind::Char, convert: e }
            })?;

        if tzh_ttisutcnt != 0 && tzh_ttisutcnt != tzh_typecnt {
            return Err(HeaderError::MismatchUtType);
        }
        if tzh_ttisstdcnt != 0 && tzh_ttisstdcnt != tzh_typecnt {
            return Err(HeaderError::MismatchStdType);
        }
        if tzh_typecnt < 1 {
            return Err(HeaderError::ZeroType);
        }
        if tzh_charcnt < 1 {
            return Err(HeaderError::ZeroChar);
        }

        let header = Header {
            time_size,
            version: version[0],
            tzh_ttisutcnt,
            tzh_ttisstdcnt,
            tzh_leapcnt,
            tzh_timecnt,
            tzh_typecnt,
            tzh_charcnt,
        };
        Ok((header, rest))
    }

    /// Returns true if this header is for a 32-bit data block.
    ///
    /// When false, it is guaranteed that this header is for a 64-bit data
    /// block.
    fn is_32bit(&self) -> bool {
        self.time_size == 4
    }

    /// Returns the size of the data block, in bytes, for this header.
    ///
    /// This returns an error if the arithmetic required to compute the
    /// length would overflow.
    ///
    /// This is useful for, e.g., skipping over the 32-bit V1 data block in
    /// V2+ TZif formatted files.
    fn data_block_len(&self) -> Result<usize, HeaderError> {
        let a = self.transition_times_len()?;
        let b = self.transition_types_len();
        let c = self.local_time_types_len()?;
        let d = self.time_zone_designations_len();
        let e = self.leap_second_len()?;
        let f = self.standard_wall_len();
        let g = self.ut_local_len();
        a.checked_add(b)
            .and_then(|z| z.checked_add(c))
            .and_then(|z| z.checked_add(d))
            .and_then(|z| z.checked_add(e))
            .and_then(|z| z.checked_add(f))
            .and_then(|z| z.checked_add(g))
            .ok_or(HeaderError::InvalidDataBlock { version: self.version })
    }

    fn transition_times_len(&self) -> Result<usize, HeaderError> {
        self.tzh_timecnt
            .checked_mul(self.time_size)
            .ok_or(HeaderError::InvalidTimeCount)
    }

    fn transition_types_len(&self) -> usize {
        self.tzh_timecnt
    }

    fn local_time_types_len(&self) -> Result<usize, HeaderError> {
        self.tzh_typecnt.checked_mul(6).ok_or(HeaderError::InvalidTypeCount)
    }

    fn time_zone_designations_len(&self) -> usize {
        self.tzh_charcnt
    }

    fn leap_second_len(&self) -> Result<usize, HeaderError> {
        let record_len = self
            .time_size
            .checked_add(4)
            .expect("4-or-8 plus 4 always fits in usize");
        self.tzh_leapcnt
            .checked_mul(record_len)
            .ok_or(HeaderError::InvalidLeapSecondCount)
    }

    fn standard_wall_len(&self) -> usize {
        self.tzh_ttisstdcnt
    }

    fn ut_local_len(&self) -> usize {
        self.tzh_ttisutcnt
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TzifError {
    kind: TzifErrorKind,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum TzifErrorKind {
    Footer(FooterError),
    Header(HeaderError),
    Header32(HeaderError),
    Header64(HeaderError),
    InconsistentPosixTimeZone(InconsistentPosixTimeZoneError),
    Indicator(IndicatorError),
    LocalTimeType(LocalTimeTypeError),
    SplitAt(SplitAtError),
    TimeZoneDesignator(TimeZoneDesignatorError),
    TransitionType(TransitionTypeError),
}

#[cfg(feature = "std")]
impl std::error::Error for TzifError {}

impl core::fmt::Display for TzifError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::TzifErrorKind::*;
        match self.kind {
            Footer(ref err) => {
                f.write_str("invalid TZif footer: ")?;
                err.fmt(f)
            }
            Header(ref err) => {
                f.write_str("invalid TZif header: ")?;
                err.fmt(f)
            }
            Header32(ref err) => {
                f.write_str("invalid 32-bit TZif header: ")?;
                err.fmt(f)
            }
            Header64(ref err) => {
                f.write_str("invalid 64-bit TZif header: ")?;
                err.fmt(f)
            }
            InconsistentPosixTimeZone(ref err) => {
                f.write_str(
                    "found inconsistency with \
                     POSIX time zone transition rule \
                     in TZif file footer: ",
                )?;
                err.fmt(f)
            }
            Indicator(ref err) => {
                f.write_str("failed to parse indicators: ")?;
                err.fmt(f)
            }
            LocalTimeType(ref err) => {
                f.write_str("failed to parse local time types: ")?;
                err.fmt(f)
            }
            SplitAt(ref err) => err.fmt(f),
            TimeZoneDesignator(ref err) => {
                f.write_str("failed to parse time zone designators: ")?;
                err.fmt(f)
            }
            TransitionType(ref err) => {
                f.write_str("failed to parse time zone transition types: ")?;
                err.fmt(f)
            }
        }
    }
}

impl From<TzifErrorKind> for TzifError {
    fn from(kind: TzifErrorKind) -> TzifError {
        TzifError { kind }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum TransitionTypeError {
    ExceedsLocalTimeTypes,
    Split(SplitAtError),
}

impl From<TransitionTypeError> for TzifError {
    fn from(err: TransitionTypeError) -> TzifError {
        TzifErrorKind::TransitionType(err).into()
    }
}

impl From<SplitAtError> for TransitionTypeError {
    fn from(err: SplitAtError) -> TransitionTypeError {
        TransitionTypeError::Split(err)
    }
}

impl core::fmt::Display for TransitionTypeError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::TransitionTypeError::*;

        match *self {
            ExceedsLocalTimeTypes => f.write_str(
                "found time zone transition type index \
                 that exceeds the number of local time types",
            ),
            Split(ref err) => err.fmt(f),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum LocalTimeTypeError {
    InvalidOffset { offset: i32 },
    Split(SplitAtError),
}

impl From<LocalTimeTypeError> for TzifError {
    fn from(err: LocalTimeTypeError) -> TzifError {
        TzifErrorKind::LocalTimeType(err).into()
    }
}

impl From<SplitAtError> for LocalTimeTypeError {
    fn from(err: SplitAtError) -> LocalTimeTypeError {
        LocalTimeTypeError::Split(err)
    }
}

impl core::fmt::Display for LocalTimeTypeError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::LocalTimeTypeError::*;

        match *self {
            InvalidOffset { offset } => write!(
                f,
                "found local time type with \
                 out-of-bounds time zone offset: {offset}, \
                 Jiff's allowed range is `{OFFSET_MIN}..={OFFSET_MAX}`"
            ),
            Split(ref err) => err.fmt(f),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum TimeZoneDesignatorError {
    InvalidEnd,
    InvalidLength,
    InvalidStart,
    InvalidUtf8,
    MissingNul,
    Split(SplitAtError),
}

impl From<TimeZoneDesignatorError> for TzifError {
    fn from(err: TimeZoneDesignatorError) -> TzifError {
        TzifErrorKind::TimeZoneDesignator(err).into()
    }
}

impl From<SplitAtError> for TimeZoneDesignatorError {
    fn from(err: SplitAtError) -> TimeZoneDesignatorError {
        TimeZoneDesignatorError::Split(err)
    }
}

impl core::fmt::Display for TimeZoneDesignatorError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::TimeZoneDesignatorError::*;

        match *self {
            InvalidEnd => f.write_str(
                "found invalid end of time zone designator \
                 for local time type",
            ),
            InvalidLength => f.write_str(
                "found invalid length of time zone designator \
                 for local time type",
            ),
            InvalidStart => f.write_str(
                "found invalid start of time zone designator \
                 for local time type",
            ),
            InvalidUtf8 => {
                f.write_str("found invalid UTF-8 in time zone designators")
            }
            MissingNul => f.write_str(
                "could not find NUL terminator for time zone designator",
            ),
            Split(ref err) => err.fmt(f),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum IndicatorError {
    InvalidCombination,
    InvalidStdWallIndicator,
    InvalidUtWallCombination,
    Split(SplitAtError),
    UtLocalNonZero,
}

impl From<IndicatorError> for TzifError {
    fn from(err: IndicatorError) -> TzifError {
        TzifErrorKind::Indicator(err).into()
    }
}

impl From<SplitAtError> for IndicatorError {
    fn from(err: SplitAtError) -> IndicatorError {
        IndicatorError::Split(err)
    }
}

impl core::fmt::Display for IndicatorError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::IndicatorError::*;

        match *self {
            InvalidCombination => f.write_str(
                "found invalid std/wall or UT/local value for \
                 local time type, each must be 0 or 1",
            ),
            InvalidStdWallIndicator => f.write_str(
                "found invalid std/wall indicator, \
                 expected it to be 0 or 1",
            ),
            InvalidUtWallCombination => f.write_str(
                "found invalid UT-wall combination for \
                 local time type, only local-wall, \
                 local-standard and UT-standard are allowed",
            ),
            Split(ref err) => err.fmt(f),
            UtLocalNonZero => f.write_str(
                "found non-zero UT/local indicator, \
                 but all such indicators should be zero",
            ),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum InconsistentPosixTimeZoneError {
    Designation,
    Dst,
    Offset,
}

impl From<InconsistentPosixTimeZoneError> for TzifError {
    fn from(err: InconsistentPosixTimeZoneError) -> TzifError {
        TzifErrorKind::InconsistentPosixTimeZone(err).into()
    }
}

impl core::fmt::Display for InconsistentPosixTimeZoneError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::InconsistentPosixTimeZoneError::*;

        match *self {
            Designation => f.write_str(
                "expected last transition in TZif file to have \
                 a time zone abbreviation matching the abbreviation \
                 derived from the POSIX time zone transition rule",
            ),
            Dst => f.write_str(
                "expected last transition in TZif file to have \
                 a DST status matching the status derived from the \
                 POSIX time zone transition rule",
            ),
            Offset => f.write_str(
                "expected last transition in TZif file to have \
                 DST offset matching the offset derived from the \
                 POSIX time zone transition rule",
            ),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum FooterError {
    InvalidPosixTz(crate::shared::posix::PosixTimeZoneError),
    MismatchEnd,
    TerminatorNotFound,
    UnexpectedEnd,
}

impl From<FooterError> for TzifError {
    fn from(err: FooterError) -> TzifError {
        TzifErrorKind::Footer(err).into()
    }
}

impl core::fmt::Display for FooterError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::FooterError::*;

        match *self {
            InvalidPosixTz(ref err) => {
                f.write_str("invalid POSIX time zone transition rule")?;
                core::fmt::Display::fmt(err, f)
            }
            MismatchEnd => f.write_str(
                "expected to find `\\n` at the beginning of \
                 the TZif file footer, \
                 but found something else instead",
            ),
            TerminatorNotFound => f.write_str(
                "expected to find `\\n` terminating \
                 the TZif file footer, \
                 but no line terminator could be found",
            ),
            UnexpectedEnd => f.write_str(
                "expected to find `\\n` at the beginning of \
                 the TZif file footer, \
                 but found unexpected end of data",
            ),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum HeaderError {
    InvalidDataBlock { version: u8 },
    InvalidLeapSecondCount,
    InvalidTimeCount,
    InvalidTypeCount,
    MismatchMagic,
    MismatchStdType,
    MismatchUtType,
    ParseCount { kind: CountKind, convert: U32UsizeError },
    TooShort,
    ZeroChar,
    ZeroType,
}

impl From<HeaderError> for TzifError {
    fn from(err: HeaderError) -> TzifError {
        TzifErrorKind::Header(err).into()
    }
}

impl core::fmt::Display for HeaderError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::HeaderError::*;

        match *self {
            InvalidDataBlock { version } => write!(
                f,
                "length of data block in V{version} TZif file is too big",
            ),
            InvalidLeapSecondCount => {
                f.write_str("number of leap seconds is too big")
            }
            InvalidTimeCount => {
                f.write_str("number of transition times is too big")
            }
            InvalidTypeCount => {
                f.write_str("number of local time types is too big")
            }
            MismatchMagic => f.write_str("magic bytes mismatch"),
            MismatchStdType => f.write_str(
                "expected number of standard/wall indicators to be zero \
                 or equal to the number of local time types",
            ),
            MismatchUtType => f.write_str(
                "expected number of UT/local indicators to be zero \
                 or equal to the number of local time types",
            ),
            ParseCount { ref kind, ref convert } => {
                write!(f, "failed to parse `{kind}`: {convert}")
            }
            TooShort => f.write_str("too short"),
            ZeroChar => f.write_str(
                "expected number of time zone abbreviations fo be at least 1",
            ),
            ZeroType => f.write_str(
                "expected number of local time types fo be at least 1",
            ),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum CountKind {
    Ut,
    Std,
    Leap,
    Time,
    Type,
    Char,
}

impl core::fmt::Display for CountKind {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::CountKind::*;
        match *self {
            Ut => f.write_str("tzh_ttisutcnt"),
            Std => f.write_str("tzh_ttisstdcnt"),
            Leap => f.write_str("tzh_leapcnt"),
            Time => f.write_str("tzh_timecnt"),
            Type => f.write_str("tzh_typecnt"),
            Char => f.write_str("tzh_charcnt"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum SplitAtError {
    V1,
    LeapSeconds,
    LocalTimeTypes,
    StandardWallIndicators,
    TimeZoneDesignations,
    TransitionTimes,
    TransitionTypes,
    UTLocalIndicators,
}

impl From<SplitAtError> for TzifError {
    fn from(err: SplitAtError) -> TzifError {
        TzifErrorKind::SplitAt(err).into()
    }
}

impl core::fmt::Display for SplitAtError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::SplitAtError::*;

        f.write_str("expected bytes for '")?;
        f.write_str(match *self {
            V1 => "v1 TZif",
            LeapSeconds => "leap seconds",
            LocalTimeTypes => "local time types",
            StandardWallIndicators => "standard/wall indicators",
            TimeZoneDesignations => "time zone designations",
            TransitionTimes => "transition times",
            TransitionTypes => "transition types",
            UTLocalIndicators => "UT/local indicators",
        })?;
        f.write_str("data block', but did not find enough bytes")?;
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct U32UsizeError;

impl core::fmt::Display for U32UsizeError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(
            f,
            "failed to parse integer because it is bigger than `{max}`",
            max = usize::MAX,
        )
    }
}

/// Splits the given slice of bytes at the index given.
///
/// If the index is out of range (greater than `bytes.len()`) then an error is
/// returned. The error message will include the `what` string given, which is
/// meant to describe the thing being split.
fn try_split_at<'b>(
    what: SplitAtError,
    bytes: &'b [u8],
    at: usize,
) -> Result<(&'b [u8], &'b [u8]), SplitAtError> {
    if at > bytes.len() {
        Err(what)
    } else {
        Ok(bytes.split_at(at))
    }
}

/// Interprets the given slice as an unsigned 32-bit big endian integer,
/// attempts to convert it to a `usize` and returns it.
///
/// # Panics
///
/// When `bytes.len() != 4`.
///
/// # Errors
///
/// This errors if the `u32` parsed from the given bytes cannot fit in a
/// `usize`.
fn from_be_bytes_u32_to_usize(bytes: &[u8]) -> Result<usize, U32UsizeError> {
    let n = from_be_bytes_u32(bytes);
    usize::try_from(n).map_err(|_| U32UsizeError)
}

/// Interprets the given slice as an unsigned 32-bit big endian integer and
/// returns it.
///
/// # Panics
///
/// When `bytes.len() != 4`.
fn from_be_bytes_u32(bytes: &[u8]) -> u32 {
    u32::from_be_bytes(bytes.try_into().unwrap())
}

/// Interprets the given slice as a signed 32-bit big endian integer and
/// returns it.
///
/// # Panics
///
/// When `bytes.len() != 4`.
fn from_be_bytes_i32(bytes: &[u8]) -> i32 {
    i32::from_be_bytes(bytes.try_into().unwrap())
}

/// Interprets the given slice as a signed 64-bit big endian integer and
/// returns it.
///
/// # Panics
///
/// When `bytes.len() != 8`.
fn from_be_bytes_i64(bytes: &[u8]) -> i64 {
    i64::from_be_bytes(bytes.try_into().unwrap())
}
