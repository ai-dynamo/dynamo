use crate::util::{
    b::{BoundsError, SpecialBoundsError},
    sync::Arc,
};

pub(crate) mod civil;
pub(crate) mod duration;
pub(crate) mod fmt;
pub(crate) mod signed_duration;
pub(crate) mod span;
pub(crate) mod timestamp;
pub(crate) mod tz;
pub(crate) mod unit;
pub(crate) mod util;
pub(crate) mod zoned;

/// An error that can occur in this crate.
///
/// The most common type of error is a result of overflow. But other errors
/// exist as well:
///
/// * Time zone database lookup failure.
/// * Configuration problem. (For example, trying to round a span with calendar
/// units without providing a relative datetime.)
/// * An I/O error as a result of trying to open a time zone database from a
/// directory via
/// [`TimeZoneDatabase::from_dir`](crate::tz::TimeZoneDatabase::from_dir).
/// * Parse errors.
///
/// # Introspection is limited
///
/// Other than implementing the [`std::error::Error`] trait when the
/// `std` feature is enabled, the [`core::fmt::Debug`] trait and the
/// [`core::fmt::Display`] trait, this error type currently provides
/// very limited introspection capabilities. Simple predicates like
/// `Error::is_range` are provided, but the predicates are not
/// exhaustive. That is, there exist some errors that do not return
/// `true` for any of the `Error::is_*` predicates.
///
/// # Design
///
/// This crate follows the "One True God Error Type Pattern," where only one
/// error type exists for a variety of different operations. This design was
/// chosen after attempting to provide finer grained error types. But finer
/// grained error types proved difficult in the face of composition.
///
/// More about this design choice can be found in a GitHub issue
/// [about error types].
///
/// [about error types]: https://github.com/BurntSushi/jiff/issues/8
#[derive(Clone)]
pub struct Error {
    /// The internal representation of an error.
    ///
    /// This is in an `Arc` to make an `Error` cloneable. It could otherwise
    /// be automatically cloneable, but it embeds a `std::io::Error` when the
    /// `std` feature is enabled, which isn't cloneable.
    ///
    /// This also makes clones cheap. And it also make the size of error equal
    /// to one word (although a `Box` would achieve that last goal). This is
    /// why we put the `Arc` here instead of on `std::io::Error` directly.
    inner: Option<Arc<ErrorInner>>,
}

#[derive(Debug)]
#[cfg_attr(not(feature = "alloc"), derive(Clone))]
struct ErrorInner {
    kind: ErrorKind,
    #[cfg(feature = "alloc")]
    cause: Option<Error>,
}

impl Error {
    /// Creates a new error value from `core::fmt::Arguments`.
    ///
    /// It is expected to use [`format_args!`](format_args) from
    /// Rust's standard library (available in `core`) to create a
    /// `core::fmt::Arguments`.
    ///
    /// Callers should generally use their own error types. But in some
    /// circumstances, it can be convenient to manufacture a Jiff error value
    /// specifically.
    ///
    /// # Core-only environments
    ///
    /// In core-only environments without a dynamic memory allocator, error
    /// messages may be degraded in some cases. For example, if the given
    /// `core::fmt::Arguments` could not be converted to a simple borrowed
    /// `&str`, then this will ignore the input given and return an "unknown"
    /// Jiff error.
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::Error;
    ///
    /// let err = Error::from_args(format_args!("something failed"));
    /// assert_eq!(err.to_string(), "something failed");
    /// ```
    pub fn from_args<'a>(message: core::fmt::Arguments<'a>) -> Error {
        Error::from(ErrorKind::Adhoc(AdhocError::from_args(message)))
    }

    /// Returns true when this error originated as a result of a value being
    /// out of Jiff's supported range.
    ///
    /// # Example
    ///
    /// ```
    /// use jiff::civil::Date;
    ///
    /// assert!(Date::new(2025, 2, 29).unwrap_err().is_range());
    /// assert!("2025-02-29".parse::<Date>().unwrap_err().is_range());
    /// assert!(Date::strptime("%Y-%m-%d", "2025-02-29").unwrap_err().is_range());
    /// ```
    pub fn is_range(&self) -> bool {
        use self::ErrorKind::*;
        matches!(
            *self.root().kind(),
            Bounds(_) | SpecialBounds(_) | ITimeRange(_)
        )
    }

    /// Returns true when this error originated as a result of an invalid
    /// configuration of parameters to a function call.
    ///
    /// This particular error category is somewhat nebulous, but it's generally
    /// meant to cover errors that _could_ have been statically prevented by
    /// Jiff with more types in its API. Instead, a smaller API is preferred.
    ///
    /// # Example: invalid rounding options
    ///
    /// ```
    /// use jiff::{SpanRound, ToSpan, Unit};
    ///
    /// let span = 44.seconds();
    /// let err = span.round(
    ///     SpanRound::new().smallest(Unit::Second).increment(45),
    /// ).unwrap_err();
    /// // Rounding increments for seconds must divide evenly into `60`.
    /// // But `45` does not. Thus, this is a "configuration" error.
    /// assert!(err.is_invalid_parameter());
    /// ```
    ///
    /// # Example: invalid units
    ///
    /// One cannot round a span between dates to units less than days:
    ///
    /// ```
    /// use jiff::{civil::date, Unit};
    ///
    /// let date1 = date(2025, 3, 18);
    /// let date2 = date(2025, 12, 21);
    /// let err = date1.until((Unit::Hour, date2)).unwrap_err();
    /// assert!(err.is_invalid_parameter());
    /// ```
    ///
    /// Similarly, one cannot round a span between times to units greater than
    /// hours:
    ///
    /// ```
    /// use jiff::{civil::time, Unit};
    ///
    /// let time1 = time(9, 39, 0, 0);
    /// let time2 = time(17, 0, 0, 0);
    /// let err = time1.until((Unit::Day, time2)).unwrap_err();
    /// assert!(err.is_invalid_parameter());
    /// ```
    pub fn is_invalid_parameter(&self) -> bool {
        use self::civil::Error as CivilError;
        use self::ErrorKind::*;

        matches!(
            *self.root().kind(),
            UnitConfig(_)
                | Civil(
                    CivilError::IllegalTimeWithMicrosecond
                        | CivilError::IllegalTimeWithMillisecond
                        | CivilError::IllegalTimeWithNanosecond
                )
        )
    }

    /// Returns true when this error originated as a result of an operation
    /// failing because an appropriate Jiff crate feature was not enabled.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use jiff::tz::TimeZone;
    ///
    /// // This passes when the `tz-system` crate feature is NOT enabled.
    /// assert!(TimeZone::try_system().unwrap_err().is_crate_feature());
    /// ```
    pub fn is_crate_feature(&self) -> bool {
        matches!(*self.root().kind(), ErrorKind::CrateFeature(_))
    }
}

impl Error {
    #[inline(never)]
    #[cold]
    pub(crate) fn bounds(err: BoundsError) -> Error {
        Error::from(ErrorKind::Bounds(err))
    }

    #[inline(never)]
    #[cold]
    pub(crate) fn special_bounds(err: SpecialBoundsError) -> Error {
        Error::from(ErrorKind::SpecialBounds(err))
    }

    /// Creates a new error from the special "shared" error type.
    pub(crate) fn itime_range(
        err: crate::shared::util::itime::RangeError,
    ) -> Error {
        Error::from(ErrorKind::ITimeRange(err))
    }

    /// Creates a new error from the special TZif error type.
    #[cfg(feature = "alloc")]
    pub(crate) fn tzif(err: crate::shared::tzif::TzifError) -> Error {
        Error::from(ErrorKind::Tzif(err))
    }

    /// Creates a new error from the special `PosixTimeZoneError` type.
    pub(crate) fn posix_tz(
        err: crate::shared::posix::PosixTimeZoneError,
    ) -> Error {
        Error::from(ErrorKind::PosixTz(err))
    }

    /// A convenience constructor for building an I/O error.
    ///
    /// This returns an error that is just a simple wrapper around the
    /// `std::io::Error` type. In general, callers should always attach some
    /// kind of context to this error (like a file path).
    ///
    /// This is only available when the `std` feature is enabled.
    #[cfg(feature = "std")]
    #[inline(never)]
    #[cold]
    pub(crate) fn io(err: std::io::Error) -> Error {
        Error::from(ErrorKind::IO(IOError { err }))
    }

    /// Contextualizes this error by associating the given file path with it.
    ///
    /// This is a convenience routine for calling `Error::context` with a
    /// `FilePathError`.
    #[cfg(any(feature = "tzdb-zoneinfo", feature = "tzdb-concatenated"))]
    #[inline(never)]
    #[cold]
    pub(crate) fn path(self, path: impl Into<std::path::PathBuf>) -> Error {
        let err = Error::from(ErrorKind::FilePath(FilePathError {
            path: path.into(),
        }));
        self.context(err)
    }

    /*
    /// Creates a new "unknown" Jiff error.
    ///
    /// The benefit of this API is that it permits creating an `Error` in a
    /// `const` context. But the error message quality is currently pretty
    /// bad: it's just a generic "unknown Jiff error" message.
    ///
    /// This could be improved to take a `&'static str`, but I believe this
    /// will require pointer tagging in order to avoid increasing the size of
    /// `Error`. (Which is important, because of how many perf sensitive
    /// APIs return a `Result<T, Error>` in Jiff.
    pub(crate) const fn unknown() -> Error {
        Error { inner: None }
    }
    */

    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn context(self, consequent: impl IntoError) -> Error {
        self.context_impl(consequent.into_error())
    }

    #[inline(never)]
    #[cold]
    fn context_impl(self, _consequent: Error) -> Error {
        #[cfg(feature = "alloc")]
        {
            let mut err = _consequent;
            if err.inner.is_none() {
                err = Error::from(ErrorKind::Unknown);
            }
            let inner = err.inner.as_mut().unwrap();
            assert!(
                inner.cause.is_none(),
                "cause of consequence must be `None`"
            );
            // OK because we just created this error so the Arc
            // has one reference.
            Arc::get_mut(inner).unwrap().cause = Some(self);
            err
        }
        #[cfg(not(feature = "alloc"))]
        {
            // We just completely drop `self`. :-(
            //
            // 2025-12-21: ... actually, we used to drop self, but this
            // ends up dropping the root cause. And the root cause
            // is how the predicates on `Error` work. So we drop the
            // consequent instead.
            self
        }
    }

    /// Returns the root error in this chain.
    fn root(&self) -> &Error {
        // OK because `Error::chain` is guaranteed to return a non-empty
        // iterator.
        self.chain().last().unwrap()
    }

    /// Returns a chain of error values.
    ///
    /// This starts with the most recent error added to the chain. That is,
    /// the highest level context. The last error in the chain is always the
    /// "root" cause. That is, the error closest to the point where something
    /// has gone wrong.
    ///
    /// The iterator returned is guaranteed to yield at least one error.
    fn chain(&self) -> impl Iterator<Item = &Error> {
        #[cfg(feature = "alloc")]
        {
            let mut err = self;
            core::iter::once(err).chain(core::iter::from_fn(move || {
                err = err
                    .inner
                    .as_ref()
                    .and_then(|inner| inner.cause.as_ref())?;
                Some(err)
            }))
        }
        #[cfg(not(feature = "alloc"))]
        {
            core::iter::once(self)
        }
    }

    /// Returns the kind of this error.
    fn kind(&self) -> &ErrorKind {
        self.inner
            .as_ref()
            .map(|inner| &inner.kind)
            .unwrap_or(&ErrorKind::Unknown)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        let mut it = self.chain().peekable();
        while let Some(err) = it.next() {
            core::fmt::Display::fmt(err.kind(), f)?;
            if it.peek().is_some() {
                f.write_str(": ")?;
            }
        }
        Ok(())
    }
}

impl core::fmt::Debug for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        if !f.alternate() {
            core::fmt::Display::fmt(self, f)
        } else {
            let Some(ref inner) = self.inner else {
                return f
                    .debug_struct("Error")
                    .field("kind", &"None")
                    .finish();
            };
            #[cfg(feature = "alloc")]
            {
                f.debug_struct("Error")
                    .field("kind", &inner.kind)
                    .field("cause", &inner.cause)
                    .finish()
            }
            #[cfg(not(feature = "alloc"))]
            {
                f.debug_struct("Error").field("kind", &inner.kind).finish()
            }
        }
    }
}

/// The underlying kind of a [`Error`].
#[derive(Debug)]
#[cfg_attr(not(feature = "alloc"), derive(Clone))]
enum ErrorKind {
    Adhoc(AdhocError),
    Bounds(BoundsError),
    Civil(self::civil::Error),
    CrateFeature(CrateFeatureError),
    Duration(self::duration::Error),
    #[allow(dead_code)] // not used in some feature configs
    FilePath(FilePathError),
    Fmt(self::fmt::Error),
    FmtFriendly(self::fmt::friendly::Error),
    FmtOffset(self::fmt::offset::Error),
    FmtRfc2822(self::fmt::rfc2822::Error),
    FmtRfc9557(self::fmt::rfc9557::Error),
    FmtTemporal(self::fmt::temporal::Error),
    FmtUtil(self::fmt::util::Error),
    FmtStrtime(self::fmt::strtime::Error),
    FmtStrtimeFormat(self::fmt::strtime::FormatError),
    FmtStrtimeParse(self::fmt::strtime::ParseError),
    #[allow(dead_code)] // not used in some feature configs
    IO(IOError),
    ITimeRange(crate::shared::util::itime::RangeError),
    OsStrUtf8(self::util::OsStrUtf8Error),
    ParseInt(self::util::ParseIntError),
    ParseFraction(self::util::ParseFractionError),
    PosixTz(crate::shared::posix::PosixTimeZoneError),
    RoundingIncrement(self::util::RoundingIncrementError),
    SignedDuration(self::signed_duration::Error),
    Span(self::span::Error),
    UnitConfig(self::unit::UnitConfigError),
    SpecialBounds(SpecialBoundsError),
    Timestamp(self::timestamp::Error),
    TzAmbiguous(self::tz::ambiguous::Error),
    TzDb(self::tz::db::Error),
    TzConcatenated(self::tz::concatenated::Error),
    TzOffset(self::tz::offset::Error),
    TzPosix(self::tz::posix::Error),
    TzSystem(self::tz::system::Error),
    TzTimeZone(self::tz::timezone::Error),
    #[allow(dead_code)]
    TzZic(self::tz::zic::Error),
    #[cfg(feature = "alloc")]
    Tzif(crate::shared::tzif::TzifError),
    Unknown,
    Zoned(self::zoned::Error),
}

impl core::fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::ErrorKind::*;

        match *self {
            Adhoc(ref msg) => msg.fmt(f),
            Bounds(ref msg) => msg.fmt(f),
            Civil(ref err) => err.fmt(f),
            CrateFeature(ref err) => err.fmt(f),
            Duration(ref err) => err.fmt(f),
            FilePath(ref err) => err.fmt(f),
            Fmt(ref err) => err.fmt(f),
            FmtFriendly(ref err) => err.fmt(f),
            FmtOffset(ref err) => err.fmt(f),
            FmtRfc2822(ref err) => err.fmt(f),
            FmtRfc9557(ref err) => err.fmt(f),
            FmtUtil(ref err) => err.fmt(f),
            FmtStrtime(ref err) => err.fmt(f),
            FmtStrtimeFormat(ref err) => err.fmt(f),
            FmtStrtimeParse(ref err) => err.fmt(f),
            FmtTemporal(ref err) => err.fmt(f),
            IO(ref err) => err.fmt(f),
            ITimeRange(ref err) => err.fmt(f),
            OsStrUtf8(ref err) => err.fmt(f),
            ParseInt(ref err) => err.fmt(f),
            ParseFraction(ref err) => err.fmt(f),
            PosixTz(ref err) => err.fmt(f),
            RoundingIncrement(ref err) => err.fmt(f),
            SignedDuration(ref err) => err.fmt(f),
            Span(ref err) => err.fmt(f),
            UnitConfig(ref err) => err.fmt(f),
            SpecialBounds(ref msg) => msg.fmt(f),
            Timestamp(ref err) => err.fmt(f),
            TzAmbiguous(ref err) => err.fmt(f),
            TzDb(ref err) => err.fmt(f),
            TzConcatenated(ref err) => err.fmt(f),
            TzOffset(ref err) => err.fmt(f),
            TzPosix(ref err) => err.fmt(f),
            TzSystem(ref err) => err.fmt(f),
            TzTimeZone(ref err) => err.fmt(f),
            TzZic(ref err) => err.fmt(f),
            #[cfg(feature = "alloc")]
            Tzif(ref err) => err.fmt(f),
            Unknown => f.write_str("unknown Jiff error"),
            Zoned(ref err) => err.fmt(f),
        }
    }
}

impl From<ErrorKind> for Error {
    fn from(kind: ErrorKind) -> Error {
        #[cfg(feature = "alloc")]
        {
            Error { inner: Some(Arc::new(ErrorInner { kind, cause: None })) }
        }
        #[cfg(not(feature = "alloc"))]
        {
            Error { inner: Some(Arc::new(ErrorInner { kind })) }
        }
    }
}

/// A generic error message.
///
/// This used to be used to represent most errors in Jiff. But then I switched
/// to more structured error types (internally). We still keep this around to
/// support the `Error::from_args` public API, which permits users of Jiff to
/// manifest their own `Error` values from an arbitrary message.
#[cfg_attr(not(feature = "alloc"), derive(Clone))]
struct AdhocError {
    #[cfg(feature = "alloc")]
    message: alloc::boxed::Box<str>,
    #[cfg(not(feature = "alloc"))]
    message: &'static str,
}

impl AdhocError {
    fn from_args<'a>(message: core::fmt::Arguments<'a>) -> AdhocError {
        #[cfg(feature = "alloc")]
        {
            use alloc::string::ToString;

            let message = message.to_string().into_boxed_str();
            AdhocError { message }
        }
        #[cfg(not(feature = "alloc"))]
        {
            let message = message.as_str().unwrap_or(
                "unknown Jiff error (better error messages require \
                 enabling the `alloc` feature for the `jiff` crate)",
            );
            AdhocError { message }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for AdhocError {}

impl core::fmt::Display for AdhocError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        core::fmt::Display::fmt(&self.message, f)
    }
}

impl core::fmt::Debug for AdhocError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        core::fmt::Debug::fmt(&self.message, f)
    }
}

/// An error used whenever a failure is caused by a missing crate feature.
///
/// This enum doesn't necessarily contain every Jiff crate feature. It only
/// contains the features whose absence can result in an error.
#[derive(Clone, Debug)]
pub(crate) enum CrateFeatureError {
    #[cfg(not(feature = "tz-system"))]
    TzSystem,
    #[cfg(not(feature = "tzdb-concatenated"))]
    TzdbConcatenated,
    #[cfg(not(feature = "tzdb-zoneinfo"))]
    TzdbZoneInfo,
}

impl From<CrateFeatureError> for Error {
    #[cold]
    #[inline(never)]
    fn from(err: CrateFeatureError) -> Error {
        ErrorKind::CrateFeature(err).into()
    }
}

impl core::fmt::Display for CrateFeatureError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        #[allow(unused_imports)]
        use self::CrateFeatureError::*;

        f.write_str("operation failed because Jiff crate feature `")?;
        #[allow(unused_variables)]
        let name: &str = match *self {
            #[cfg(not(feature = "tz-system"))]
            TzSystem => "tz-system",
            #[cfg(not(feature = "tzdb-concatenated"))]
            TzdbConcatenated => "tzdb-concatenated",
            #[cfg(not(feature = "tzdb-zoneinfo"))]
            TzdbZoneInfo => "tzdb-zoneinfo",
        };
        #[allow(unreachable_code)]
        {
            core::fmt::Display::fmt(name, f)?;
            f.write_str("` is not enabled")
        }
    }
}

/// A `std::io::Error`.
///
/// This type is itself always available, even when the `std` feature is not
/// enabled. When `std` is not enabled, a value of this type can never be
/// constructed.
///
/// Otherwise, this type is a simple wrapper around `std::io::Error`. Its
/// purpose is to encapsulate the conditional compilation based on the `std`
/// feature.
#[cfg_attr(not(feature = "alloc"), derive(Clone))]
struct IOError {
    #[cfg(feature = "std")]
    err: std::io::Error,
}

#[cfg(feature = "std")]
impl std::error::Error for IOError {}

impl core::fmt::Display for IOError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        #[cfg(feature = "std")]
        {
            self.err.fmt(f)
        }
        #[cfg(not(feature = "std"))]
        {
            f.write_str("<BUG: SHOULD NOT EXIST>")
        }
    }
}

impl core::fmt::Debug for IOError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        #[cfg(feature = "std")]
        {
            f.debug_struct("IOError").field("err", &self.err).finish()
        }
        #[cfg(not(feature = "std"))]
        {
            f.write_str("<BUG: SHOULD NOT EXIST>")
        }
    }
}

#[cfg(feature = "std")]
impl From<std::io::Error> for IOError {
    fn from(err: std::io::Error) -> IOError {
        IOError { err }
    }
}

#[cfg_attr(not(feature = "alloc"), derive(Clone))]
struct FilePathError {
    #[cfg(feature = "std")]
    path: std::path::PathBuf,
}

#[cfg(feature = "std")]
impl std::error::Error for FilePathError {}

impl core::fmt::Display for FilePathError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        #[cfg(feature = "std")]
        {
            self.path.display().fmt(f)
        }
        #[cfg(not(feature = "std"))]
        {
            f.write_str("<BUG: SHOULD NOT EXIST>")
        }
    }
}

impl core::fmt::Debug for FilePathError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        #[cfg(feature = "std")]
        {
            f.debug_struct("FilePathError").field("path", &self.path).finish()
        }
        #[cfg(not(feature = "std"))]
        {
            f.write_str("<BUG: SHOULD NOT EXIST>")
        }
    }
}

/// A simple trait to encapsulate automatic conversion to `Error`.
///
/// This trait basically exists to make `Error::context` work without needing
/// to rely on public `From` impls. For example, without this trait, we might
/// otherwise write `impl From<String> for Error`. But this would make it part
/// of the public API. Which... maybe we should do, but at time of writing,
/// I'm starting very conservative so that we can evolve errors in semver
/// compatible ways.
pub(crate) trait IntoError {
    fn into_error(self) -> Error;
}

impl IntoError for Error {
    #[inline(always)]
    fn into_error(self) -> Error {
        self
    }
}

/// A trait for contextualizing error values.
///
/// This makes it easy to contextualize either `Error` or `Result<T, Error>`.
/// Specifically, in the latter case, it absolves one of the need to call
/// `map_err` everywhere one wants to add context to an error.
///
/// This trick was borrowed from `anyhow`.
pub(crate) trait ErrorContext<T, E> {
    /// Contextualize the given consequent error with this (`self`) error as
    /// the cause.
    ///
    /// This is equivalent to saying that "consequent is caused by self."
    ///
    /// Note that if an `Error` is given for `kind`, then this panics if it has
    /// a cause. (Because the cause would otherwise be dropped. An error causal
    /// chain is just a linked list, not a tree.)
    fn context(self, consequent: impl IntoError) -> Result<T, Error>;

    /// Like `context`, but hides error construction within a closure.
    ///
    /// This is useful if the creation of the consequent error is not otherwise
    /// guarded and when error construction is potentially "costly" (i.e., it
    /// allocates). The closure avoids paying the cost of contextual error
    /// creation in the happy path.
    ///
    /// Usually this only makes sense to use on a `Result<T, Error>`, otherwise
    /// the closure is just executed immediately anyway.
    fn with_context<C: IntoError>(
        self,
        consequent: impl FnOnce() -> C,
    ) -> Result<T, Error>;
}

impl<T, E> ErrorContext<T, E> for Result<T, E>
where
    E: IntoError,
{
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn context(self, consequent: impl IntoError) -> Result<T, Error> {
        self.map_err(|err| {
            err.into_error().context_impl(consequent.into_error())
        })
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn with_context<C: IntoError>(
        self,
        consequent: impl FnOnce() -> C,
    ) -> Result<T, Error> {
        self.map_err(|err| {
            err.into_error().context_impl(consequent().into_error())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // We test that our 'Error' type is the size we expect. This isn't an API
    // guarantee, but if the size increases, we really want to make sure we
    // decide to do that intentionally. So this should be a speed bump. And in
    // general, we should not increase the size without a very good reason.
    #[test]
    fn error_size() {
        let mut expected_size = core::mem::size_of::<usize>();
        if !cfg!(feature = "alloc") {
            // oooowwwwwwwwwwwch.
            //
            // Like, this is horrible, right? core-only environments are
            // precisely the place where one want to keep things slim. But
            // in core-only, I don't know of a way to introduce any sort of
            // indirection in the library level without using a completely
            // different API.
            //
            // This is what makes me doubt that core-only Jiff is actually
            // useful. In what context are people using a huge library like
            // Jiff but can't define a small little heap allocator?
            //
            // OK, this used to be `expected_size *= 10`, but I slimmed it down
            // to x3. Still kinda sucks right? If we tried harder, I think we
            // could probably slim this down more. And if we were willing to
            // sacrifice error message quality even more (like, all the way),
            // then we could make `Error` a zero sized type. Which might
            // actually be the right trade-off for core-only, but I'll hold off
            // until we have some real world use cases.
            //
            // OK... after switching to structured errors, this jumped
            // back up to `expected_size *= 6`. And that was with me being
            // conscientious about what data we store inside of error types.
            // Blech.
            //
            // 2026-01-14: A change to the `Offset` type made this move back
            // down to `expected_size *= 4`.
            expected_size *= 4;
        }
        assert_eq!(expected_size, core::mem::size_of::<Error>());
    }
}
