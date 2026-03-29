Jiff
====
Jiff is a datetime library for Rust that encourages you to jump into the
pit of success. The focus of this library is providing high level datetime
primitives that are difficult to misuse and have reasonable performance. Jiff
supports automatic and seamless integration with the Time Zone Database, DST
aware arithmetic and rounding, formatting and parsing zone aware datetimes
losslessly, opt-in Serde support and a whole lot more.

Jiff takes enormous inspiration from [Temporal], which is a [TC39] proposal to
improve datetime handling in JavaScript.

[![Build status](https://github.com/BurntSushi/jiff/workflows/ci/badge.svg)](https://github.com/BurntSushi/jiff/actions)
[![Crates.io](https://img.shields.io/crates/v/jiff.svg)](https://crates.io/crates/jiff)
[![Docs.rs](https://img.shields.io/docsrs/jiff)](https://docs.rs/jiff)

Dual-licensed under MIT or the [UNLICENSE](https://unlicense.org/).

[TC39]: https://tc39.es/
[Temporal]: https://tc39.es/proposal-temporal/docs/index.html

### Documentation

* [API documentation on docs.rs](https://docs.rs/jiff)
* [Comparison with `chrono`, `time`, `hifitime` and `icu`](COMPARE.md)
* [The API design rationale for Jiff](DESIGN.md)
* [Platform support](PLATFORM.md)
* [CHANGELOG](CHANGELOG.md)

### Example

Here is a quick example that shows how to parse a typical RFC 3339 instant,
convert it to a zone aware datetime, add a span of time and losslessly print
it:

```rust
use jiff::{Timestamp, ToSpan};

fn main() -> Result<(), jiff::Error> {
    let time: Timestamp = "2024-07-11T01:14:00Z".parse()?;
    let zoned = time.in_tz("America/New_York")?.checked_add(1.month().hours(2))?;
    assert_eq!(zoned.to_string(), "2024-08-10T23:14:00-04:00[America/New_York]");
    // Or, if you want an RFC3339 formatted string:
    assert_eq!(zoned.timestamp().to_string(), "2024-08-11T03:14:00Z");
    Ok(())
}
```

There are many more examples in the [documentation](https://docs.rs/jiff).

### Usage

Jiff is [on crates.io](https://crates.io/crates/jiff) and can be
used by adding `jiff` to your dependencies in your project's `Cargo.toml`.
Or more simply, just run `cargo add jiff`.

Here is a complete example that creates a new Rust project, adds a dependency
on `jiff`, creates the source code for a simple datetime program and then runs
it.

First, create the project in a new directory:

```text
$ cargo new jiff-example
$ cd jiff-example
```

Second, add a dependency on `jiff`:

```text
$ cargo add jiff
```

Third, edit `src/main.rs`. Delete what's there and replace it with this:

```rust
use jiff::{Unit, Zoned};

fn main() -> Result<(), jiff::Error> {
    let now = Zoned::now().round(Unit::Second)?;
    println!("{now}");
    Ok(())
}
```

Fourth, run it with `cargo run`:

```text
$ cargo run
   Compiling jiff v0.2.0 (/home/andrew/rust/jiff)
   Compiling jiff-play v0.2.0 (/home/andrew/tmp/scratch/rust/jiff-play)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.37s
     Running `target/debug/jiff-play`
2024-07-10T19:54:20-04:00[America/New_York]
```

The first time you run the program will show more output like above. But
subsequent runs shouldn't have to re-compile the dependencies.

### Crate features

Jiff has several crate features for customizing support for Rust's standard
library, `serde` support and whether to embed a copy of the Time Zone Database
into your binary.

The "[crate features](https://docs.rs/jiff/#crate-features)" section of the
documentation lists the full set of supported features.

### Future plans

My original plan was to release Jiff 1.0 in Summer 2025. It is now Winter 2026,
so that deadline has clearly slipped. I've continued to make progress toward
Jiff 1.0, but it has been more work than anticipated. My current hope is to
have Jiff 1.0 out by Spring or Summer 2026.

It's important to get Jiff 1.0 as "right" as possible. Namely, once it's
released, I plan to commit to its API indefinitely. At which point, users of
Jiff should feel comfortable using it as a stable base in public APIs.

### Performance

The most important design goal of Jiff is to be a high level datetime library
that makes it hard to do the wrong thing. Second to that is performance. Jiff
should have reasonable performance, but there are likely areas in which it
could improve. See the `bench` directory for benchmarks.

Note that performance is still an important goal. Some aspects of Jiff have
had optimization attention paid to them, but many still have not. It is a goal
to improve where we can, but performance will generally come second to API
comprehension and correctness.

### Platform support

The question of platform support in the context of datetime libraries comes up
primarily in relation to time zone support. Specifically:

* How should Jiff determine the time zone transitions for an IANA time zone
identifier like `Antarctica/Troll`?
* How should Jiff determine the default time zone for the current system?

Both of these require some level of platform interaction.

For discovering time zone transition data, Jiff relies on the
[IANA Time Zone Database]. On Unix systems, this is usually found at
`/usr/share/zoneinfo`, although it can be configured via the `TZDIR`
environment variable (which Jiff respects). On Windows, Jiff will automatically
embed a copy of the time zone database into the compiled library.

For discovering the system time zone, Jiff reads `/etc/localtime` on Unix. On
Windows, Jiff reads the Windows-specific time zone identifier via
[`GetDynamicTimeZoneInformation`] and then maps it to an IANA time zone
identifier via Unicode's [CLDR XML data].

I expect Jiff to grow more support for other platforms over time. Please file
issues, although I will likely be reliant on contributor pull requests for more
obscure platforms that aren't easy for me to test.

For more on platform support, see [`PLATFORM.md`](PLATFORM.md).

[IANA Time Zone Database]: https://en.wikipedia.org/wiki/Tz_database
[`GetDynamicTimeZoneInformation`]: https://learn.microsoft.com/en-us/windows/win32/api/timezoneapi/nf-timezoneapi-getdynamictimezoneinformation
[CLDR XML data]: https://github.com/unicode-org/cldr/raw/main/common/supplemental/windowsZones.xml

### Dependencies

At time of writing, it is no accident that Jiff has zero dependencies on Unix.
In general, my philosophy on adding new dependencies in an ecosystem crate like
Jiff is very conservative. I consider there to be two primary use cases for
adding new dependencies:

1. When a dependency is _practically_ required in order to interact with a
platform. For example, `windows-sys` for discovering the system time zone on
Windows.
2. When a dependency is necessary for inter-operability. For example, `serde`.
But even here, I expect to be conservative, where I'm generally only willing
to depend on things that have fewer breaking change releases than Jiff.

A secondary use case for new dependencies is if Jiff gets split into multiple
crates. I did a similar thing for the `regex` crate for very compelling
reasons. It is possible that will happen with Jiff as well, although there are
no plans for that. And in general, I expect the number of crates to stay small,
if only to make keep maintenance lightweight. (Managing lots of semver API
boundaries has a lot of overhead in my experience.)

### Minimum Rust version policy

This crate's minimum supported `rustc` version (MSRV) is `1.70.0`. Jiff
itself, and any dependencies of Jiff within this repository, will never use a
Rust version that requires a Rust compiler released in the previous 1 year.
(Dependents of Jiff in this repository, e.g., `jiff-icu`, may use a newer MSRV,
but never one that is newer than the crate it is trying to integrate with.)

Jiff does have other dependencies whose MSRV is outside the control of Jiff
(for example, `windows-sys` and `portable-atomic`), and it's possible there
may be a choice between "upgrade and increase the MSRV for those targets where
the dependency is used" and "don't upgrade, retain the MSRV but potentially
introduce additional copies of crates into downstream dependency trees." While
such instances should be quite rare because of Jiff's conservative dependency
philosophy, it's possible it could happen. In which case, I reserve the right
to make a judgment call that may, in practice, result in a newer-than-1-year
Rust compiler for targets where that dependency is used.

As is standard in the Rust ecosystem, Jiff does not consider an increase in
its MSRV to be a semver incompatible change. However, Jiff will limit MSRV
increases to minor version updates. For example, if Jiff `1.0` requires Rust
`1.20.0`, then Jiff `1.0.z` for all values of `z` will also require Rust
`1.20.0` or older. However, Jiff `1.y` for `y > 0` may require a newer minimum
version of Rust. (During Jiff's pre-1.0 releases, MSRV bumps will only occur
in semver incompatible releases.)
