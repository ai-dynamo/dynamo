jiff-tzdb-platform
==================
This is an optional dependency of `jiff` that embeds the entire [IANA Time Zone
Database] into the compiled binary only on specific platforms. Specifically, it
embeds the binary [TZif] for each time zone.

The platforms for which this embeds the data are controlled via the Jiff
dependency on this crate. Namely, this crate merely serves as a target
dependent proxy. It exists because target specific Cargo features don't exist.

The idea is that this crate is an optional dependency that is enabled by
default, but due to its target specific nature, it is only enabled for specific
platforms that lack a system copy of the IANA Time Zone Database (i.e.,
Windows).

[IANA Time Zone Database]: https://www.iana.org/time-zones
[TZif]: https://datatracker.ietf.org/doc/html/rfc8536

### Documentation

https://docs.rs/jiff-tzdb-platform
