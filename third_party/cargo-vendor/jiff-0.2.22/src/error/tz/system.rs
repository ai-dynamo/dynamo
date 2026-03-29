#[cfg(not(feature = "tz-system"))]
pub(crate) use self::disabled::*;
#[cfg(feature = "tz-system")]
pub(crate) use self::enabled::*;

#[cfg(not(feature = "tz-system"))]
mod disabled {
    #[derive(Clone, Debug)]
    pub(crate) enum Error {}

    impl core::fmt::Display for Error {
        fn fmt(&self, _: &mut core::fmt::Formatter) -> core::fmt::Result {
            unreachable!()
        }
    }
}

#[cfg(feature = "tz-system")]
mod enabled {
    use crate::error;

    #[derive(Clone, Debug)]
    pub(crate) enum Error {
        FailedEnvTz,
        FailedEnvTzAsTzif,
        FailedPosixTzAndUtf8,
        FailedSystemTimeZone,
        FailedUnnamedTzifInvalid,
        FailedUnnamedTzifRead,
        #[cfg(windows)]
        WindowsMissingIanaMapping,
        #[cfg(windows)]
        WindowsTimeZoneKeyName,
        #[cfg(windows)]
        WindowsUtf16DecodeInvalid,
        #[cfg(windows)]
        WindowsUtf16DecodeNul,
    }

    impl From<Error> for error::Error {
        #[cold]
        #[inline(never)]
        fn from(err: Error) -> error::Error {
            error::ErrorKind::TzSystem(err).into()
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
                FailedEnvTz => f.write_str(
                    "`TZ` environment variable set, but failed to read value",
                ),
                FailedEnvTzAsTzif => f.write_str(
                    "failed to read `TZ` environment variable value \
                     as a TZif file after attempting (and failing) a tzdb \
                     lookup for that same value",
                ),
                FailedPosixTzAndUtf8 => f.write_str(
                    "failed to parse `TZ` environment variable as either \
                     a POSIX time zone transition string or as valid UTF-8",
                ),
                FailedSystemTimeZone => {
                    f.write_str("failed to find system time zone")
                }
                FailedUnnamedTzifInvalid => f.write_str(
                    "found invalid TZif data in unnamed time zone file",
                ),
                FailedUnnamedTzifRead => f.write_str(
                    "failed to read TZif data from unnamed time zone file",
                ),
                #[cfg(windows)]
                WindowsMissingIanaMapping => f.write_str(
                    "found Windows time zone name, \
                     but could not find a mapping for it to an \
                     IANA time zone name",
                ),
                #[cfg(windows)]
                WindowsTimeZoneKeyName => f.write_str(
                    "could not get `TimeZoneKeyName` from \
                     winapi `DYNAMIC_TIME_ZONE_INFORMATION`",
                ),
                #[cfg(windows)]
                WindowsUtf16DecodeInvalid => f.write_str(
                    "failed to convert `u16` slice to UTF-8 \
                     (invalid UTF-16)",
                ),
                #[cfg(windows)]
                WindowsUtf16DecodeNul => f.write_str(
                    "failed to convert `u16` slice to UTF-8 \
                     (no NUL terminator found)",
                ),
            }
        }
    }
}
