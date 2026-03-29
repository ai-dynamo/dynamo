use crate::error;

// At time of writing, the biggest TZif data file is a few KB. And the
// index block is tens of KB. So impose a limit that is a couple of orders
// of magnitude bigger, but still overall pretty small for... some systems.
// Anyway, I welcome improvements to this heuristic!
pub(crate) const ALLOC_LIMIT: usize = 10 * 1 << 20;

#[derive(Clone, Debug)]
pub(crate) enum Error {
    AllocRequestOverLimit,
    AllocFailed,
    AllocOverflow,
    ExpectedFirstSixBytes,
    ExpectedIanaName,
    ExpectedLastByte,
    #[cfg(test)]
    ExpectedMoreData,
    ExpectedVersion,
    FailedReadData,
    FailedReadHeader,
    FailedReadIndex,
    #[cfg(all(feature = "std", all(not(unix), not(windows))))]
    FailedSeek,
    InvalidIndexDataOffsets,
    InvalidLengthIndexBlock,
    #[cfg(all(feature = "std", windows))]
    InvalidOffsetOverflowFile,
    #[cfg(test)]
    InvalidOffsetOverflowSlice,
    #[cfg(test)]
    InvalidOffsetTooBig,
}

impl From<Error> for error::Error {
    #[cold]
    #[inline(never)]
    fn from(err: Error) -> error::Error {
        error::ErrorKind::TzConcatenated(err).into()
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
            AllocRequestOverLimit => write!(
                f,
                "attempted to allocate more than {ALLOC_LIMIT} bytes \
                 while reading concatenated TZif data, which \
                 exceeds a heuristic limit to prevent huge allocations \
                 (please file a bug if this error is inappropriate)",
            ),
            AllocFailed => f.write_str(
                "failed to allocate additional room \
                 for reading concatenated TZif data",
            ),
            AllocOverflow => {
                f.write_str("total allocation length overflowed `usize`")
            }
            ExpectedFirstSixBytes => f.write_str(
                "expected first 6 bytes of concatenated TZif header \
                 to be `tzdata`",
            ),
            ExpectedIanaName => f.write_str(
                "expected IANA time zone identifier to be valid UTF-8",
            ),
            ExpectedLastByte => f.write_str(
                "expected last byte of concatenated TZif header \
                 to be `NUL`",
            ),
            #[cfg(test)]
            ExpectedMoreData => f.write_str(
                "unexpected EOF, expected more bytes based on size \
                 of caller provided buffer",
            ),
            ExpectedVersion => f.write_str(
                "expected version in concatenated TZif header to \
                 be valid UTF-8",
            ),
            FailedReadData => f.write_str("failed to read TZif data block"),
            FailedReadHeader => {
                f.write_str("failed to read concatenated TZif header")
            }
            FailedReadIndex => f.write_str("failed to read index block"),
            #[cfg(all(feature = "std", all(not(unix), not(windows))))]
            FailedSeek => {
                f.write_str("failed to seek to offset in `std::fs::File`")
            }
            InvalidIndexDataOffsets => f.write_str(
                "invalid index and data offsets, \
                 expected index offset to be less than or equal \
                 to data offset",
            ),
            InvalidLengthIndexBlock => {
                f.write_str("length of index block is not a valid multiple")
            }
            #[cfg(all(feature = "std", windows))]
            InvalidOffsetOverflowFile => f.write_str(
                "offset overflow when reading from `std::fs::File`",
            ),
            #[cfg(test)]
            InvalidOffsetOverflowSlice => {
                f.write_str("offset overflowed `usize`")
            }
            #[cfg(test)]
            InvalidOffsetTooBig => {
                f.write_str("offset too big for given slice of data")
            }
        }
    }
}
