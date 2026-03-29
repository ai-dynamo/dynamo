use crate::error;

#[derive(Clone, Debug)]
pub(crate) enum Error {
    #[cfg(feature = "tzdb-concatenated")]
    ConcatenatedMissingIanaIdentifiers,
    #[cfg(all(feature = "std", not(feature = "tzdb-concatenated")))]
    DisabledConcatenated,
    #[cfg(all(feature = "std", not(feature = "tzdb-zoneinfo")))]
    DisabledZoneInfo,
    FailedTimeZone {
        #[cfg(feature = "alloc")]
        name: alloc::boxed::Box<str>,
    },
    FailedTimeZoneNoDatabaseConfigured {
        #[cfg(feature = "alloc")]
        name: alloc::boxed::Box<str>,
    },
    #[cfg(feature = "tzdb-zoneinfo")]
    ZoneInfoNoTzifFiles,
    #[cfg(feature = "tzdb-zoneinfo")]
    ZoneInfoStripPrefix,
}

impl Error {
    pub(crate) fn failed_time_zone(_time_zone_name: &str) -> Error {
        Error::FailedTimeZone {
            #[cfg(feature = "alloc")]
            name: _time_zone_name.into(),
        }
    }

    pub(crate) fn failed_time_zone_no_database_configured(
        _time_zone_name: &str,
    ) -> Error {
        Error::FailedTimeZoneNoDatabaseConfigured {
            #[cfg(feature = "alloc")]
            name: _time_zone_name.into(),
        }
    }
}

impl From<Error> for error::Error {
    #[cold]
    #[inline(never)]
    fn from(err: Error) -> error::Error {
        error::ErrorKind::TzDb(err).into()
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
            #[cfg(feature = "tzdb-concatenated")]
            ConcatenatedMissingIanaIdentifiers => f.write_str(
                "found no IANA time zone identifiers in \
                 concatenated tzdata file",
            ),
            #[cfg(all(feature = "std", not(feature = "tzdb-concatenated")))]
            DisabledConcatenated => {
                f.write_str("system concatenated tzdb unavailable")
            }
            #[cfg(all(feature = "std", not(feature = "tzdb-zoneinfo")))]
            DisabledZoneInfo => {
                f.write_str("system zoneinfo tzdb unavailable")
            }
            FailedTimeZone {
                #[cfg(feature = "alloc")]
                ref name,
            } => {
                #[cfg(feature = "alloc")]
                {
                    write!(
                        f,
                        "failed to find time zone `{name}` \
                         in time zone database",
                    )
                }
                #[cfg(not(feature = "alloc"))]
                {
                    f.write_str(
                        "failed to find time zone in time zone database",
                    )
                }
            }
            FailedTimeZoneNoDatabaseConfigured {
                #[cfg(feature = "alloc")]
                ref name,
            } => {
                #[cfg(feature = "std")]
                {
                    write!(
                        f,
                        "failed to find time zone `{name}` since there is no \
                         time zone database configured",
                    )
                }
                #[cfg(all(not(feature = "std"), feature = "alloc"))]
                {
                    write!(
                        f,
                        "failed to find time zone `{name}`, since there is no \
                         global time zone database configured (and is \
                         currently impossible to do so without Jiff's `std` \
                         feature enabled, if you need this functionality, \
                         please file an issue on Jiff's tracker with your \
                         use case)",
                    )
                }
                #[cfg(all(not(feature = "std"), not(feature = "alloc")))]
                {
                    f.write_str(
                        "failed to find time zone, since there is no \
                         global time zone database configured (and is \
                         currently impossible to do so without Jiff's `std` \
                         feature enabled, if you need this functionality, \
                         please file an issue on Jiff's tracker with your \
                         use case)",
                    )
                }
            }
            #[cfg(feature = "tzdb-zoneinfo")]
            ZoneInfoNoTzifFiles => f.write_str(
                "did not find any TZif files in zoneinfo time zone database",
            ),
            #[cfg(feature = "tzdb-zoneinfo")]
            ZoneInfoStripPrefix => f.write_str(
                "failed to strip zoneinfo time zone database directory \
                 path from path to TZif file",
            ),
        }
    }
}
