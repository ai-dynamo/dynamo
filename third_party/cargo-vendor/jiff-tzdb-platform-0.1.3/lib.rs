/*!
A crate that embeds data from the [IANA Time Zone Database] for platforms
without a system copy.

This crate re-exports [`jiff-tzdb`]. Its only purpose in existing is as a
target specific dependency of [Jiff]. Specifically, it allows a particular
dependency pattern that lets Jiff depend on `jiff-tzdb` on Windows by default
(without needing users to opt into it), while specifically *not* depending on
it by default for Unix platforms. This configuration is desirable because
Windows does not have a standard system copy of the Time Zone Database while
most Unix platforms do. And indeed, it is desirable to use the system copy
whenever possible.

[IANA Time Zone Database]: https://www.iana.org/time-zones
[`jiff-tzdb`]: https://docs.rs/jiff-tzdb
[Jiff]: https://docs.rs/jiff
*/

#![no_std]

pub extern crate jiff_tzdb;
