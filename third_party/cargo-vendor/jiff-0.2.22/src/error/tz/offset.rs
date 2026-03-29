use crate::{
    error,
    tz::{Offset, TimeZone},
};

#[derive(Clone, Debug)]
pub(crate) enum Error {
    ConvertDateTimeToTimestamp {
        offset: Offset,
    },
    OverflowAddSignedDuration,
    OverflowSignedDuration,
    ResolveRejectFold {
        given: Offset,
        before: Offset,
        after: Offset,
        tz: TimeZone,
    },
    ResolveRejectGap {
        given: Offset,
        before: Offset,
        after: Offset,
        tz: TimeZone,
    },
    ResolveRejectUnambiguous {
        given: Offset,
        offset: Offset,
        tz: TimeZone,
    },
    RoundOverflow,
}

impl From<Error> for error::Error {
    #[cold]
    #[inline(never)]
    fn from(err: Error) -> error::Error {
        error::ErrorKind::TzOffset(err).into()
    }
}

impl error::IntoError for Error {
    fn into_error(self) -> error::Error {
        self.into()
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::Error::*;

        match *self {
            ConvertDateTimeToTimestamp { offset } => write!(
                f,
                "converting datetime with time zone offset `{offset}` \
                 to timestamp overflowed",
            ),
            OverflowAddSignedDuration => f.write_str(
                "adding signed duration to time zone offset overflowed",
            ),
            OverflowSignedDuration => {
                f.write_str("signed duration overflows time zone offset")
            }
            ResolveRejectFold { given, before, after, ref tz } => write!(
                f,
                "datetime could not resolve to timestamp \
                 since `reject` conflict resolution was chosen, and \
                 because datetime has offset `{given}`, but the time \
                 zone `{tzname}` for the given datetime falls in a fold \
                 between offsets `{before}` and `{after}`, neither of which \
                 match the offset",
                tzname = tz.diagnostic_name(),
            ),
            ResolveRejectGap { given, before, after, ref tz } => write!(
                f,
                "datetime could not resolve to timestamp \
                 since `reject` conflict resolution was chosen, and \
                 because datetime has offset `{given}`, but the time \
                 zone `{tzname}` for the given datetime falls in a gap \
                 (between offsets `{before}` and `{after}`), and all \
                 offsets for a gap are regarded as invalid",
                tzname = tz.diagnostic_name(),
            ),
            ResolveRejectUnambiguous { given, offset, ref tz } => write!(
                f,
                "datetime could not resolve to a timestamp since \
                 `reject` conflict resolution was chosen, and because \
                 datetime has offset `{given}`, but the time \
                 zone `{tzname}` for the given datetime \
                 unambiguously has offset `{offset}`",
                tzname = tz.diagnostic_name(),
            ),
            RoundOverflow => f.write_str(
                "rounding time zone offset resulted in a duration \
                 that overflows",
            ),
        }
    }
}
