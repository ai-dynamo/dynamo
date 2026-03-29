use core::mem::MaybeUninit;

use crate::{fmt::Write, Error};

const MAX_CAPACITY: usize = u16::MAX as usize;
// From `u64::MAX.to_string().len()`.
const MAX_INTEGER_LEN: u8 = 20;
const MAX_PRECISION: usize = 9;

/// The minimum buffer length required for *any* of `BorrowedBuffer`'s
/// integer formatting APIs to work.
///
/// This relies on the fact that the maximum padding is clamped to `20`.
const BROAD_MINIMUM_BUFFER_LEN: usize = 20;

/// All integers in the range `0..=99`, zero padded.
const RADIX_100_ZERO: [u8; 200] = *b"00010203040506070809\
       10111213141516171819\
       20212223242526272829\
       30313233343536373839\
       40414243444546474849\
       50515253545556575859\
       60616263646566676869\
       70717273747576777879\
       80818283848586878889\
       90919293949596979899";

/// All integers in the range `0..=99`, space padded.
const RADIX_100_SPACE: [u8; 200] = *b" 0 1 2 3 4 5 6 7 8 9\
       10111213141516171819\
       20212223242526272829\
       30313233343536373839\
       40414243444546474849\
       50515253545556575859\
       60616263646566676869\
       70717273747576777879\
       80818283848586878889\
       90919293949596979899";

/// An uninitialized slice of bytes of fixed size.
///
/// This is typically used with `BorrowedBuffer`.
#[derive(Clone, Copy)]
pub(crate) struct ArrayBuffer<const N: usize> {
    data: [MaybeUninit<u8>; N],
}

impl<const N: usize> ArrayBuffer<N> {
    /// Return a writable buffer into this storage.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn as_borrowed<'data>(&mut self) -> BorrowedBuffer<'_> {
        BorrowedBuffer::from(&mut self.data)
    }

    /// Assume this entire buffer is initialized and return it as an array.
    ///
    /// # Safety
    ///
    /// Callers must ensure the entire buffer is initialized.
    unsafe fn assume_init(self) -> [u8; N] {
        // SAFETY: Callers are responsible for ensuring that `self.data`
        // is initialized. Otherwise, `MaybeUninit<u8>` and `u8` have the
        // same representation.
        unsafe {
            *(&self.data as *const [MaybeUninit<u8>; N] as *const [u8; N])
        }
    }
}

/// Construct an uninitialized buffer of data of size `N`.
impl<const N: usize> Default for ArrayBuffer<N> {
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn default() -> ArrayBuffer<N> {
        ArrayBuffer { data: [MaybeUninit::uninit(); N] }
    }
}

/// A borrowed buffer for writing into an uninitialized slice of bytes.
///
/// This can be used with, e.g., an `ArrayBuffer` as backing storage. This
/// type will managed actually writing to the backing storage, keeping track
/// of how much data has been written and exposing a safe API.
///
/// This type is principally used in Jiff's printer implementations. In
/// particular, this helps printers generate tighter and more efficient code.
/// Once printing is done, the data in the buffer is then copied to the caller
/// provided implementation of `jiff::fmt::Write`. This double write is
/// unfortunate, but it turned out that threading a `jiff::fmt::Write` down
/// through the printers and using it directly leads to slower code overall
/// *and* more code bloat. This is because a `BorrowedBuffer` is an incredibly
/// lightweight abstraction that never has to deal with I/O or growing an
/// allocation.
///
/// # Design
///
/// * Requires valid UTF-8 so that we can provide higher level safe APIs for
/// interfacing with `String`.
/// * Specifically panics when a write would exceed available capacity. This
/// introduces a branch, but effectively decouples "get the maximum size
/// correct" from "is memory safe."
#[derive(Debug)]
pub(crate) struct BorrowedBuffer<'data> {
    data: &'data mut [MaybeUninit<u8>],
    filled: u16,
}

impl<'data> BorrowedBuffer<'data> {
    /// A high level API for writing to a `jiff::fmt::Write` via a buffer of
    /// uninitialized bytes.
    ///
    /// When the `alloc` crate feature is enabled and `W` provides a
    /// `&mut Vec<u8>`, then the buffer is extracted directly from the
    /// spare capacity of the `Vec<u8>`.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn with_writer<const N: usize>(
        wtr: &mut dyn Write,
        _runtime_allocation: usize,
        mut with: impl FnMut(&mut BorrowedBuffer<'_>) -> Result<(), Error>,
    ) -> Result<(), Error> {
        // Specialize for the common case of `W = String` or `W = Vec<u8>`.
        // In effect, we write directly into the excess capacity of `W`
        // instead of first writing into a stack array and then copying that
        // into `W`.
        //
        // This can provide up to a 40% improvement in runtime in some
        // microbenchmarks.
        //
        // SAFETY: We only ever write valid UTF-8. Namely, `BorrowedBuffer`
        // enforces this invariant.
        #[cfg(feature = "alloc")]
        if let Some(buf) = unsafe { wtr.as_mut_vec() } {
            buf.reserve(_runtime_allocation);
            return BorrowedBuffer::with_vec_spare_capacity(buf, with);
        }
        let mut buf = ArrayBuffer::<N>::default();
        let mut bbuf = buf.as_borrowed();
        with(&mut bbuf)?;
        wtr.write_str(bbuf.filled())?;
        Ok(())
    }

    /// Provides a borrowed buffer into the first 255 bytes of the spare
    /// capacity in the given `Vec<u8>` and updates the length on `Vec<u8>`
    /// after the closure completes to account for any new data written.
    ///
    /// In effect, this safely encapsulates writing into the uninitialized
    /// portion of a `Vec<u8>`.
    ///
    /// If the provided closure panics, then there is no guarantee that the
    /// `buf`'s length will be updated to reflect what has been written.
    /// However, it is guaranteed that the length setting will not lead to
    /// undefined behavior.
    #[cfg(feature = "alloc")]
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn with_vec_spare_capacity<T>(
        buf: &'data mut alloc::vec::Vec<u8>,
        mut with: impl FnMut(&mut BorrowedBuffer<'_>) -> T,
    ) -> T {
        let mut bbuf = BorrowedBuffer::from_vec_spare_capacity(buf);
        let returned = with(&mut bbuf);
        let new_len = bbuf.len();
        // SAFETY: `BorrowedBuffer::len()` always reflects the number of
        // bytes that have been written to. Thus, the data up to the given new
        // length is guaranteed to be initialized.
        unsafe {
            buf.set_len(new_len);
        }
        returned
    }

    /// Build a borrowed buffer for writing into the spare capacity of a
    /// `Vec<u8>` allocation.
    ///
    /// This is limited only to the first `255` bytes of the spare capacity.
    #[cfg(feature = "alloc")]
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn from_vec_spare_capacity(
        vec: &'data mut alloc::vec::Vec<u8>,
    ) -> BorrowedBuffer<'data> {
        let data = vec.spare_capacity_mut();
        let len = data.len().min(MAX_CAPACITY);
        BorrowedBuffer::from(&mut data[..len])
    }

    /// Write the provided string to the available space in this buffer.
    ///
    /// # Panics
    ///
    /// When the available space is shorter than the length of the provided
    /// string. That is, when `capacity() - filled().len() < string.len()`.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn write_str(&mut self, string: &str) {
        // SAFETY: A `&str`, `&[u8]` and `&[MaybeUninit<u8>]` all have the
        // same representation in memory.
        let data: &[MaybeUninit<u8>] = unsafe {
            core::slice::from_raw_parts(
                string.as_ptr().cast::<MaybeUninit<u8>>(),
                string.len(),
            )
        };
        self.available()
            .get_mut(..string.len())
            .expect("string data exceeds available buffer space")
            .copy_from_slice(data);
        // Cast here will never truncate because `BorrowedBuffer` is limited
        // to `u16::MAX` in total capacity. Above will panic if `string.len()`
        // exceeds available capacity, which can never be above total capacity.
        // Thus, if we're here, `string.len() <= u16::MAX` is guaranteed to
        // hold.
        self.filled += string.len() as u16;
    }

    /// Writes the given Unicode scalar value (as UTF-8) to this writer.
    ///
    /// # Panics
    ///
    /// When the available space is shorter than the UTF-8 encoding of the
    /// given Unicode scalar value (up to and including 4 bytes).
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn write_char(&mut self, ch: char) {
        self.write_str(ch.encode_utf8(&mut [0; 4]));
    }

    /// Writes the given ASCII byte to this writer.
    ///
    /// # Panics
    ///
    /// When the available space is shorter than 1 or if `byte` is not ASCII.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn write_ascii_char(&mut self, byte: u8) {
        assert!(byte.is_ascii());
        self.available()
            .get_mut(0)
            .expect("insufficient buffer space to write one byte")
            .write(byte);
        self.filled += 1;
    }

    /// Writes the given `u8` integer to this buffer. No padding is performed.
    ///
    /// # Panics
    ///
    /// When the available space is insufficient to encode the number of
    /// digits in the decimal representation of `n`.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn write_int(&mut self, n: impl Into<u64>) {
        let mut n = n.into();
        let digits = digits(n);
        let mut remaining_digits = usize::from(digits);
        let available = self
            .available()
            .get_mut(..remaining_digits)
            .expect("u8 integer digits exceeds available buffer space");
        while remaining_digits > 0 {
            remaining_digits -= 1;
            // SAFETY: The assert above guarantees that `remaining_digits` is
            // always in bounds.
            unsafe {
                available
                    .get_unchecked_mut(remaining_digits)
                    .write(b'0' + ((n % 10) as u8));
            }
            n /= 10;
        }
        self.filled += u16::from(digits);
    }

    /// Writes the given `u8` integer to this buffer using the given padding.
    ///
    /// The maximum allowed padding is `20`. Any values bigger than that are
    /// silently clamped to `20`.
    ///
    /// This always pads with zeroes.
    ///
    /// # Panics
    ///
    /// When the available space is insufficient to encode the number of
    /// digits in the decimal representation of `n`.
    ///
    /// This also panics when `pad_byte` is not ASCII.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn write_int_pad0(&mut self, n: impl Into<u64>, pad_len: u8) {
        let mut n = n.into();
        let pad_len = pad_len.min(MAX_INTEGER_LEN);
        let digits = pad_len.max(digits(n));
        let mut remaining_digits = usize::from(digits);
        let available = self
            .available()
            .get_mut(..remaining_digits)
            .expect("u8 integer digits exceeds available buffer space");
        while remaining_digits > 0 {
            remaining_digits -= 1;
            // SAFETY: The assert above guarantees that `remaining_digits` is
            // always in bounds.
            unsafe {
                available
                    .get_unchecked_mut(remaining_digits)
                    .write(b'0' + ((n % 10) as u8));
            }
            n /= 10;
        }
        self.filled += u16::from(digits);
    }

    /// Writes the given `u8` integer to this buffer using the given padding.
    ///
    /// The maximum allowed padding is `20`. Any values bigger than that are
    /// silently clamped to `20`.
    ///
    /// # Panics
    ///
    /// When the available space is insufficient to encode the number of
    /// digits in the decimal representation of `n`.
    ///
    /// This also panics when `pad_byte` is not ASCII.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn write_int_pad(
        &mut self,
        n: impl Into<u64>,
        pad_byte: u8,
        pad_len: u8,
    ) {
        assert!(pad_byte.is_ascii(), "padding byte must be ASCII");

        let mut n = n.into();
        let pad_len = pad_len.min(MAX_INTEGER_LEN);
        let digits = pad_len.max(digits(n));
        let mut remaining_digits = usize::from(digits);
        let available = self
            .available()
            .get_mut(..remaining_digits)
            .expect("u8 integer digits exceeds available buffer space");
        while remaining_digits > 0 {
            remaining_digits -= 1;
            // SAFETY: The assert above guarantees that `remaining_digits` is
            // always in bounds.
            unsafe {
                available
                    .get_unchecked_mut(remaining_digits)
                    .write(b'0' + ((n % 10) as u8));
            }
            n /= 10;
            if n == 0 {
                break;
            }
        }
        while remaining_digits > 0 {
            remaining_digits -= 1;
            // SAFETY: The assert above guarantees that `remaining_digits` is
            // always in bounds.
            unsafe {
                available.get_unchecked_mut(remaining_digits).write(pad_byte);
            }
        }
        self.filled += u16::from(digits);
    }

    /// Writes the given integer as a 1-digit integer.
    ///
    /// # Panics
    ///
    /// When the available space is shorter than 1 or if `n > 9`.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn write_int1(&mut self, n: impl Into<u64>) {
        let n = n.into();
        // This is required for correctness. When `n > 9`, then the
        // `n as u8` below could result in writing an invalid UTF-8
        // byte. We don't mind incorrect results, but writing invalid
        // UTF-8 can lead to undefined behavior, and we want this API
        // to be sound.
        assert!(n <= 9);
        self.write_ascii_char(b'0' + (n as u8));
    }

    /// Writes the given integer as a 2-digit zero padded integer to this
    /// buffer.
    ///
    /// # Panics
    ///
    /// When the available space is shorter than 2 or if `n > 99`.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn write_int_pad2(&mut self, n: impl Into<u64>) {
        let n = n.into();
        // This is required for correctness. When `n > 99`, then the
        // last `n as u8` below could result in writing an invalid UTF-8
        // byte. We don't mind incorrect results, but writing invalid
        // UTF-8 can lead to undefined behavior, and we want this API
        // to be sound.
        //
        // We omit the final `% 10` because it makes micro-benchmark results
        // worse. This panicking check has a more modest impact.
        assert!(n <= 99);

        let dst = self
            .available()
            .get_mut(..2)
            .expect("padded 2 digit integer exceeds available buffer space");
        let radix_offset = ((n % 100) * 2) as usize;
        // SAFETY: Valid because of the assertion above. And also,
        // `RADIX_100_ZERO` always has exactly 200 elements and
        // `radix_offset` is never greater than 198. (In that case,
        // we do access element at index 199 below as well.)
        unsafe {
            dst.get_unchecked_mut(0)
                .write(*RADIX_100_ZERO.get_unchecked(radix_offset));
            dst.get_unchecked_mut(1)
                .write(*RADIX_100_ZERO.get_unchecked(radix_offset + 1));
        }
        self.filled += 2;
    }

    /// Writes the given integer as a 2-digit space padded integer to this
    /// buffer.
    ///
    /// # Panics
    ///
    /// When the available space is shorter than 2 or if `n > 99`.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn write_int_pad2_space(&mut self, n: impl Into<u64>) {
        let n = n.into();
        // This is required for correctness. When `n > 99`, then the
        // last `n as u8` below could result in writing an invalid UTF-8
        // byte. We don't mind incorrect results, but writing invalid
        // UTF-8 can lead to undefined behavior, and we want this API
        // to be sound.
        //
        // We omit the final `% 10` because it makes micro-benchmark results
        // worse. This panicking check has a more modest impact.
        assert!(n <= 99);

        let dst = self
            .available()
            .get_mut(..2)
            .expect("padded 2 digit integer exceeds available buffer space");
        let radix_offset = ((n % 100) * 2) as usize;
        // SAFETY: Valid because of the assertion above. And also,
        // `RADIX_100_ZERO` always has exactly 200 elements and
        // `radix_offset` is never greater than 198. (In that case,
        // we do access element at index 199 below as well.)
        unsafe {
            dst.get_unchecked_mut(0)
                .write(*RADIX_100_SPACE.get_unchecked(radix_offset));
            dst.get_unchecked_mut(1)
                .write(*RADIX_100_SPACE.get_unchecked(radix_offset + 1));
        }
        self.filled += 2;
    }

    /// Writes the given integer as a 4-digit zero padded integer to this
    /// buffer.
    ///
    /// # Panics
    ///
    /// When the available space is shorter than 4 or if `n > 9999`.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn write_int_pad4(&mut self, n: impl Into<u64>) {
        let mut n = n.into();
        // This is required for correctness. When `n > 9999`, then the
        // last `n as u8` below could result in writing an invalid UTF-8
        // byte. We don't mind incorrect results, but writing invalid
        // UTF-8 can lead to undefined behavior, and we want this API
        // to be sound.
        //
        // We omit the final `% 10` because it makes micro-benchmark results
        // worse. This panicking check has a more modest impact.
        assert!(n <= 9999);

        let dst = self
            .available()
            .get_mut(..4)
            .expect("padded 4 digit integer exceeds available buffer space");

        let radix_offset = ((n % 100) * 2) as usize;
        // SAFETY: Valid because of the assertion above. And also,
        // `RADIX_100_ZERO` always has exactly 200 elements and
        // `radix_offset` is never greater than 198. (In that case,
        // we do access element at index 199 below as well.)
        unsafe {
            dst.get_unchecked_mut(2)
                .write(*RADIX_100_ZERO.get_unchecked(radix_offset));
            dst.get_unchecked_mut(3)
                .write(*RADIX_100_ZERO.get_unchecked(radix_offset + 1));
        }

        n /= 100;
        let radix_offset = ((n % 100) * 2) as usize;
        // SAFETY: Valid because of the assertion above. And also,
        // `RADIX_100_ZERO` always has exactly 200 elements and
        // `radix_offset` is never greater than 198. (In that case,
        // we do access element at index 199 below as well.)
        unsafe {
            dst.get_unchecked_mut(0)
                .write(*RADIX_100_ZERO.get_unchecked(radix_offset));
            dst.get_unchecked_mut(1)
                .write(*RADIX_100_ZERO.get_unchecked(radix_offset + 1));
        }

        self.filled += 4;
    }

    /// Writes `n` as a fractional component to the given `precision`.
    ///
    /// When `precision` is absent, then it is automatically detected based
    /// on the value of `n`.
    ///
    /// When `precision` is bigger than `9`, then it is clamped to `9`.
    ///
    /// # Panics
    ///
    /// When the available space is shorter than the number of digits required
    /// to write `n` as a fractional value.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn write_fraction(
        &mut self,
        precision: Option<u8>,
        mut n: u32,
    ) {
        assert!(n <= 999_999_999);
        let mut buf = ArrayBuffer::<MAX_PRECISION>::default();
        for i in (0..MAX_PRECISION).rev() {
            unsafe {
                buf.data.get_unchecked_mut(i).write(b'0' + ((n % 10) as u8));
            }
            n /= 10;
        }

        let end = precision
            .map(|p| p.min(MAX_PRECISION as u8))
            .unwrap_or_else(|| {
                // SAFETY: The loop above is guaranteed to initialize `buf` in
                // its entirety.
                let buf = unsafe { buf.assume_init() };
                let mut end = MAX_PRECISION as u8;
                while end > 0 && buf[usize::from(end) - 1] == b'0' {
                    end -= 1;
                }
                end
            });

        let buf = &buf.data[..usize::from(end)];
        self.available()
            .get_mut(..buf.len())
            .expect("fraction exceeds available buffer space")
            .copy_from_slice(buf);
        self.filled += u16::from(end);
    }

    /// Clears this buffer so that there are no filled bytes.
    ///
    /// Note that no actual clearing of data is done, but this does lose
    /// track of data that has been initialized and data that hasn't been
    /// initialized.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn clear(&mut self) {
        self.filled = 0;
    }

    /// Returns the filled portion of this buffer.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn filled(&self) -> &str {
        // SAFETY: It is guaranteed that `..self.len()` is always a valid
        // range into `self.data` since `self.filled` is only increased upon
        // a valid write.
        let filled = unsafe { self.data.get_unchecked(..self.len()) };
        // SAFETY: Everything up to `self.filled` is guaranteed to be
        // initialized. Also, since `MaybeUninit<u8>` and `u8` have the same
        // representation, we can cast from `&[MaybeUninit<u8>]` to `&[u8]`.
        // Finally, the `BorrowedBuffer` API specifically guarantees that
        // all writes to `self.data` are valid UTF-8.
        unsafe {
            core::str::from_utf8_unchecked(core::slice::from_raw_parts(
                filled.as_ptr().cast::<u8>(),
                self.len(),
            ))
        }
    }

    /// Returns the available space in this buffer.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn available(&mut self) -> &mut [MaybeUninit<u8>] {
        // SAFETY: `self.len()` is guaranteed to be a valid offset for the
        // start of a slice into `self.data`.
        unsafe { self.data.get_unchecked_mut(self.len()..) }
    }

    /// Return the length of the "filled" in bytes.
    ///
    /// This is always equivalent to the length of the slice returned by
    /// `BorrowedBuffer::filled`.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn len(&self) -> usize {
        usize::from(self.filled)
    }

    /// Return the total unused capacity available to this buffer.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn available_capacity(&self) -> usize {
        self.capacity() - self.len()
    }

    /// Return the total capacity available to this buffer.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn capacity(&self) -> usize {
        self.data.len()
    }
}

/// Construct a borrowed buffer for writing into a `&mut [u8]`.
///
/// This typically isn't useful on its own since `&mut [u8]` is already
/// guaranteed to be initialized and doesn't require handling with
/// care. However, this is useful when using with APIs that expect a
/// `BorrowedBuffer`.
///
/// # Panics
///
/// When the slice exceeds the maximum capacity supported by `BorrowedBuffer`.
impl<'data> From<&'data mut [u8]> for BorrowedBuffer<'data> {
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn from(data: &'data mut [u8]) -> BorrowedBuffer<'data> {
        assert!(
            data.len() <= MAX_CAPACITY,
            "borrowed buffer only supports {MAX_CAPACITY} bytes"
        );
        let len = data.len();
        let data: *mut MaybeUninit<u8> = data.as_mut_ptr().cast();
        // SAFETY: The length hasn't changed and `MaybeUninit<u8>` and `u8`
        // are guaranteed to have the same representation in memory.
        let data = unsafe { core::slice::from_raw_parts_mut(data, len) };
        BorrowedBuffer { data, filled: 0 }
    }
}

/// Construct a borrowed buffer directly from a slice of uninitialized data.
///
/// # Panics
///
/// When the slice exceeds the maximum capacity supported by `BorrowedBuffer`.
impl<'data> From<&'data mut [MaybeUninit<u8>]> for BorrowedBuffer<'data> {
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn from(data: &'data mut [MaybeUninit<u8>]) -> BorrowedBuffer<'data> {
        assert!(
            data.len() <= MAX_CAPACITY,
            "borrowed buffer only supports {MAX_CAPACITY} bytes"
        );
        BorrowedBuffer { data, filled: 0 }
    }
}

/// Construct a borrowed buffer directly from an array of uninitialized data.
///
/// # Panics
///
/// When the array exceeds the maximum capacity supported by `BorrowedBuffer`.
impl<'data, const N: usize> From<&'data mut [MaybeUninit<u8>; N]>
    for BorrowedBuffer<'data>
{
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn from(data: &'data mut [MaybeUninit<u8>; N]) -> BorrowedBuffer<'data> {
        BorrowedBuffer::from(&mut data[..])
    }
}

/// A buffering abstraction on top of `BorrowedBuffer`.
///
/// This lets callers make use of a monomorphic uninitialized buffer while
/// writing variable length data. For example, in use with `strftime`, where
/// the length of the resulting string can be arbitrarily long.
///
/// Essentially, once the buffer is filled up, it is emptied by writing it
/// to an underlying `jiff::fmt::Write` implementation.
///
/// # Design
///
/// We specifically do not expose the underlying `BorrowedBuffer` in this API.
/// It is too error prone because it makes it ridiculously easy for the caller
/// to try to write too much data to the buffer, thus causing a panic.
///
/// Also, we require that the total capacity of the `BorrowedBuffer` given
/// is big enough such that any of the integer formatting routines will always
/// fit. This means we don't need to break up integer formatting to support
/// pathologically small buffer sizes, e.g., 0 or 1 bytes. This is fine because
/// this is a Jiff-internal abstraction.
///
/// Callers must call `BorrowedWriter::finish` when done to ensure the internal
/// buffer is properly flushed.
///
/// One somewhat unfortunate aspect of the design here is that the integer
/// formatting routines need to know how much data is going to be written. This
/// sometimes requires doing some work to figure out. And that work is usually
/// repeated by `BorrowedBuffer`. My hope at time of writing (2026-01-02) is
/// that compiler eliminates the duplication, but I haven't actually checked
/// this yet.
///
/// `BorrowedWriter::write_str` is the only method where there is some
/// awareness of the underlying `Write` implementation. This is because the
/// string can be of arbitrary length, and thus, may exceed the size of
/// the buffer. (In which case, we pass it through directly to the `Write`
/// implementation.)
pub(crate) struct BorrowedWriter<'buffer, 'data, 'write> {
    bbuf: &'buffer mut BorrowedBuffer<'data>,
    wtr: &'write mut dyn Write,
}

impl<'buffer, 'data, 'write> BorrowedWriter<'buffer, 'data, 'write> {
    /// Creates a new borrowed writer that buffers writes in `bbuf` and flushes
    /// them to `wtr`.
    ///
    /// # Panics
    ///
    /// When `BorrowedBuffer` is too small to handle formatting a single
    /// integer (including padding).
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn new(
        bbuf: &'buffer mut BorrowedBuffer<'data>,
        wtr: &'write mut dyn Write,
    ) -> BorrowedWriter<'buffer, 'data, 'write> {
        assert!(bbuf.capacity() >= BROAD_MINIMUM_BUFFER_LEN);
        BorrowedWriter { bbuf, wtr }
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn finish(self) -> Result<(), Error> {
        self.wtr.write_str(self.bbuf.filled())?;
        self.bbuf.clear();
        Ok(())
    }

    #[cold]
    #[inline(never)]
    pub(crate) fn flush(&mut self) -> Result<(), Error> {
        self.wtr.write_str(self.bbuf.filled())?;
        self.bbuf.clear();
        Ok(())
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn if_will_fill_then_flush(
        &mut self,
        additional: impl Into<usize>,
    ) -> Result<(), Error> {
        let n = additional.into();
        if self.bbuf.len().saturating_add(n) > self.bbuf.capacity() {
            self.flush()?;
        }
        Ok(())
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn write_str(&mut self, string: &str) -> Result<(), Error> {
        if string.len() > self.bbuf.available_capacity() {
            self.flush()?;
            if string.len() > self.bbuf.available_capacity() {
                return self.wtr.write_str(string);
            }
        }
        self.bbuf.write_str(string);
        Ok(())
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn write_char(&mut self, ch: char) -> Result<(), Error> {
        self.if_will_fill_then_flush(ch.len_utf8())?;
        self.bbuf.write_char(ch);
        Ok(())
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn write_ascii_char(&mut self, byte: u8) -> Result<(), Error> {
        if self.bbuf.available_capacity() == 0 {
            self.flush()?;
        }
        self.bbuf.write_ascii_char(byte);
        Ok(())
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn write_int_pad(
        &mut self,
        n: impl Into<u64>,
        pad_byte: u8,
        pad_len: u8,
    ) -> Result<(), Error> {
        let n = n.into();
        let pad_len = pad_len.min(MAX_INTEGER_LEN);
        let digits = pad_len.max(digits(n));
        self.if_will_fill_then_flush(digits)?;
        self.bbuf.write_int_pad(n, pad_byte, pad_len);
        Ok(())
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn write_int_pad2(
        &mut self,
        n: impl Into<u64>,
    ) -> Result<(), Error> {
        self.if_will_fill_then_flush(2usize)?;
        self.bbuf.write_int_pad2(n);
        Ok(())
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn write_int_pad2_space(
        &mut self,
        n: impl Into<u64>,
    ) -> Result<(), Error> {
        self.if_will_fill_then_flush(2usize)?;
        self.bbuf.write_int_pad2_space(n);
        Ok(())
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn write_int_pad4(
        &mut self,
        n: impl Into<u64>,
    ) -> Result<(), Error> {
        self.if_will_fill_then_flush(4usize)?;
        self.bbuf.write_int_pad4(n);
        Ok(())
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub(crate) fn write_fraction(
        &mut self,
        precision: Option<u8>,
        n: u32,
    ) -> Result<(), Error> {
        // It's hard to know up front how many digits we're going to print
        // without doing the work required to print the digits. So we just
        // assume this will always write 9 digits when called. We could do
        // a little better here when `precision` is not `None`, but I'm not
        // clear if it's worth it or not. I think in practice, for common
        // cases, our uninit buffer will be big enough anyway even when we're
        // pessimistic about the number of digits we'll print.
        self.if_will_fill_then_flush(9usize)?;
        self.bbuf.write_fraction(precision, n);
        Ok(())
    }
}

/// We come full circle and make a `BorrowedWriter` implement
/// `jiff::fmt::Write`.
///
/// This is concretely useful for `strftime` and passing a borrowed writer
/// to methods on the `Custom` trait.
impl<'buffer, 'data, 'write> Write for BorrowedWriter<'buffer, 'data, 'write> {
    fn write_str(&mut self, string: &str) -> Result<(), Error> {
        BorrowedWriter::write_str(self, string)
    }
}

/// Returns the number of digits in the decimal representation of `n`.
///
/// This calculation to figure out the number of digits to write in `n` is
/// the expense we incur by having our printers write forwards. If we instead
/// wrote backwards, then we could omit this calculation. I ended up choosing
/// this design because 1) most integer writes in datetime (not span) printing
/// are fixed 2 or 4 digits, and don't require this extra computation and 2)
/// writing backwards just overall seems like a pain.
#[cfg_attr(feature = "perf-inline", inline(always))]
fn digits(n: u64) -> u8 {
    // It's faster by about 1-5% (on microbenchmarks) to make this more
    // branch-y and specialize the smaller and more common integers in lieu
    // of calling `ilog10`.
    match n {
        0..=9 => 1,
        10..=99 => 2,
        100..=999 => 3,
        1000..=9999 => 4,
        _ => n.ilog10() as u8 + 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_str_array() {
        let mut buf = ArrayBuffer::<100>::default();
        let mut bbuf = buf.as_borrowed();
        bbuf.write_str("Hello, world!");
        assert_eq!(bbuf.filled(), "Hello, world!");
        let buf = bbuf.filled();
        assert_eq!(buf, "Hello, world!");
    }

    #[test]
    fn write_int_array() {
        let mut buf = ArrayBuffer::<100>::default();
        let mut bbuf = buf.as_borrowed();

        bbuf.write_int(0u64);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "0");
        }

        bbuf.clear();
        bbuf.write_int(1u64);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "1");
        }

        bbuf.clear();
        bbuf.write_int(10u64);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "10");
        }

        bbuf.clear();
        bbuf.write_int(100u64);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "100");
        }

        bbuf.clear();
        bbuf.write_int(u64::MAX);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "18446744073709551615");
        }
    }

    #[test]
    fn write_int_pad2() {
        let mut buf = ArrayBuffer::<100>::default();
        let mut bbuf = buf.as_borrowed();

        bbuf.write_int_pad2(0u64);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "00");
        }

        bbuf.clear();
        bbuf.write_int_pad2(1u64);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "01");
        }

        bbuf.clear();
        bbuf.write_int_pad2(10u64);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "10");
        }

        bbuf.clear();
        bbuf.write_int_pad2(99u64);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "99");
        }
    }

    #[test]
    #[should_panic]
    fn write_int_pad2_panic() {
        let mut buf = ArrayBuffer::<100>::default();
        let mut bbuf = buf.as_borrowed();
        // technically unspecified behavior,
        // but should not result in undefined behavior.
        bbuf.write_int_pad2(u64::MAX);
    }

    #[test]
    fn write_int_pad4() {
        let mut buf = ArrayBuffer::<100>::default();
        let mut bbuf = buf.as_borrowed();

        bbuf.write_int_pad4(0u64);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "0000");
        }

        bbuf.clear();
        bbuf.write_int_pad4(1u64);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "0001");
        }

        bbuf.clear();
        bbuf.write_int_pad4(10u64);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "0010");
        }

        bbuf.clear();
        bbuf.write_int_pad4(99u64);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "0099");
        }

        bbuf.clear();
        bbuf.write_int_pad4(999u64);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "0999");
        }

        bbuf.clear();
        bbuf.write_int_pad4(9999u64);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "9999");
        }
    }

    #[test]
    #[should_panic]
    fn write_int_pad4_panic() {
        let mut buf = ArrayBuffer::<100>::default();
        let mut bbuf = buf.as_borrowed();
        // technically unspecified behavior,
        // but should not result in undefined behavior.
        bbuf.write_int_pad4(u64::MAX);
    }

    #[test]
    fn write_int_pad_zero() {
        let mut buf = ArrayBuffer::<100>::default();
        let mut bbuf = buf.as_borrowed();

        bbuf.write_int_pad(0u64, b'0', 0);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "0");
        }

        bbuf.clear();
        bbuf.write_int_pad(0u64, b'0', 1);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "0");
        }

        bbuf.clear();
        bbuf.write_int_pad(0u64, b'0', 2);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "00");
        }

        bbuf.clear();
        bbuf.write_int_pad(0u64, b'0', 20);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "00000000000000000000");
        }

        bbuf.clear();
        // clamped to 20
        bbuf.write_int_pad(0u64, b'0', 21);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "00000000000000000000");
        }

        bbuf.clear();
        bbuf.write_int_pad(0u64, b' ', 2);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, " 0");
        }
    }

    #[test]
    fn write_int_pad_one() {
        let mut buf = ArrayBuffer::<100>::default();
        let mut bbuf = buf.as_borrowed();

        bbuf.write_int_pad(1u64, b'0', 0);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "1");
        }

        bbuf.clear();
        bbuf.write_int_pad(1u64, b'0', 1);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "1");
        }

        bbuf.clear();
        bbuf.write_int_pad(1u64, b'0', 2);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "01");
        }

        bbuf.clear();
        bbuf.write_int_pad(1u64, b'0', 20);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "00000000000000000001");
        }

        bbuf.clear();
        // clamped to 20
        bbuf.write_int_pad(1u64, b'0', 21);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "00000000000000000001");
        }

        bbuf.clear();
        bbuf.write_int_pad(1u64, b' ', 2);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, " 1");
        }
    }

    #[test]
    fn write_int_pad_max() {
        let mut buf = ArrayBuffer::<100>::default();
        let mut bbuf = buf.as_borrowed();

        bbuf.write_int_pad(u64::MAX, b'0', 0);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "18446744073709551615");
        }

        bbuf.clear();
        bbuf.write_int_pad(u64::MAX, b'0', 1);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "18446744073709551615");
        }

        bbuf.clear();
        bbuf.write_int_pad(u64::MAX, b'0', 2);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "18446744073709551615");
        }

        bbuf.clear();
        bbuf.write_int_pad(u64::MAX, b'0', 20);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "18446744073709551615");
        }

        bbuf.clear();
        // clamped to 20
        bbuf.write_int_pad(u64::MAX, b'0', 21);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "18446744073709551615");
        }

        bbuf.clear();
        bbuf.write_int_pad(u64::MAX, b' ', 2);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "18446744073709551615");
        }
    }

    #[test]
    #[should_panic]
    fn write_int_pad_non_ascii_panic() {
        let mut buf = ArrayBuffer::<100>::default();
        let mut bbuf = buf.as_borrowed();
        bbuf.write_int_pad(0u64, 0xFF, 0);
    }

    #[test]
    #[should_panic]
    fn write_int_pad_insufficient_capacity_panic() {
        let mut buf = ArrayBuffer::<19>::default();
        let mut bbuf = buf.as_borrowed();
        bbuf.write_int_pad(0u64, b'0', 20);
    }

    #[test]
    fn write_fraction_no_precision() {
        let mut buf = ArrayBuffer::<100>::default();
        let mut bbuf = buf.as_borrowed();

        bbuf.write_fraction(None, 0);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "");
        }

        bbuf.clear();
        bbuf.write_fraction(None, 1);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "000000001");
        }

        bbuf.clear();
        bbuf.write_fraction(None, 123_000_000);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "123");
        }

        bbuf.clear();
        bbuf.write_fraction(None, 999_999_999);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "999999999");
        }
    }

    #[test]
    fn write_fraction_precision() {
        let mut buf = ArrayBuffer::<100>::default();
        let mut bbuf = buf.as_borrowed();

        bbuf.write_fraction(Some(0), 0);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "");
        }

        bbuf.clear();
        bbuf.write_fraction(Some(1), 0);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "0");
        }

        bbuf.clear();
        bbuf.write_fraction(Some(9), 0);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "000000000");
        }

        bbuf.clear();
        bbuf.write_fraction(Some(0), 1);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "");
        }

        bbuf.clear();
        bbuf.write_fraction(Some(9), 1);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "000000001");
        }

        bbuf.clear();
        bbuf.write_fraction(Some(0), 123_000_000);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "");
        }

        bbuf.clear();
        bbuf.write_fraction(Some(1), 123_000_000);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "1");
        }

        bbuf.clear();
        bbuf.write_fraction(Some(2), 123_000_000);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "12");
        }

        bbuf.clear();
        bbuf.write_fraction(Some(3), 123_000_000);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "123");
        }

        bbuf.clear();
        bbuf.write_fraction(Some(6), 123_000_000);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "123000");
        }

        bbuf.clear();
        bbuf.write_fraction(Some(9), 123_000_000);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "123000000");
        }

        bbuf.clear();
        // clamps to 9
        bbuf.write_fraction(Some(10), 123_000_000);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "123000000");
        }

        bbuf.clear();
        bbuf.write_fraction(Some(0), 999_999_999);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "");
        }

        bbuf.clear();
        bbuf.write_fraction(Some(1), 999_999_999);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "9");
        }

        bbuf.clear();
        bbuf.write_fraction(Some(3), 999_999_999);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "999");
        }

        bbuf.clear();
        bbuf.write_fraction(Some(6), 999_999_999);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "999999");
        }

        bbuf.clear();
        bbuf.write_fraction(Some(9), 999_999_999);
        {
            let buf = bbuf.filled();
            assert_eq!(buf, "999999999");
        }
    }

    #[test]
    #[should_panic]
    fn write_fraction_too_big_panic() {
        let mut buf = ArrayBuffer::<100>::default();
        let mut bbuf = buf.as_borrowed();
        bbuf.write_fraction(None, 1_000_000_000);
    }
}
