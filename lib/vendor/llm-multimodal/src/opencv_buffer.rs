//! Safe wrapper for OpenCV's buffered video capture constructor.
#![allow(unsafe_code)]

use std::ffi::{c_char, c_void, CStr};

use bytes::Bytes;
use opencv::{
    core::{Mat, CV_8UC3},
    imgproc,
    prelude::*,
    traits::OpenCVFromExtern,
    videoio,
};

unsafe extern "C" {
    fn smg_opencv_capture_from_buffer(
        data: *const u8,
        size: usize,
        decoder_threads: i32,
        error: *mut c_char,
        error_capacity: usize,
    ) -> *mut c_void;
}

pub(crate) struct BufferedCapture {
    capture: videoio::VideoCapture,
    _bytes: Bytes,
}

pub(crate) struct RgbOutputBuffer {
    data: Vec<u8>,
}

impl RgbOutputBuffer {
    pub(crate) fn with_capacity(capacity: usize) -> Result<Self, String> {
        let mut data = Vec::new();
        data.try_reserve_exact(capacity).map_err(|error| {
            format!("failed to reserve {capacity} decoded video bytes: {error}")
        })?;
        Ok(Self { data })
    }

    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }

    pub(crate) fn push_bgr(
        &mut self,
        bgr_frame: &Mat,
        width: u32,
        height: u32,
    ) -> Result<(usize, usize), String> {
        let width_i32 = i32::try_from(width)
            .map_err(|_| format!("OpenCV RGB output width does not fit i32: {width}"))?;
        let height_i32 = i32::try_from(height)
            .map_err(|_| format!("OpenCV RGB output height does not fit i32: {height}"))?;
        let frame_size = usize::try_from(width)
            .ok()
            .and_then(|width| {
                usize::try_from(height)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .and_then(|pixels| pixels.checked_mul(3))
            .ok_or_else(|| "OpenCV RGB frame byte size overflow".to_string())?;
        self.data.try_reserve(frame_size).map_err(|error| {
            format!("failed to reserve {frame_size} decoded video bytes: {error}")
        })?;
        let offset = self.data.len();
        let output_data = self.data.spare_capacity_mut().as_mut_ptr().cast::<u8>();
        // SAFETY: `try_reserve` made at least `frame_size` writable bytes
        // available, and the Vec cannot move while `output` borrows that region.
        let mut output = unsafe {
            Mat::new_rows_cols_with_data_unsafe_def(
                height_i32,
                width_i32,
                CV_8UC3,
                output_data.cast(),
            )
        }
        .map_err(|error| error.to_string())?;
        imgproc::cvt_color_def(bgr_frame, &mut output, imgproc::COLOR_BGR2RGB)
            .map_err(|error| error.to_string())?;
        if output.data() != output_data
            || output.rows() != height_i32
            || output.cols() != width_i32
            || output.typ() != CV_8UC3
        {
            return Err("OpenCV replaced the caller-provided RGB output buffer".to_string());
        }
        drop(output);
        // SAFETY: cvtColor successfully initialized exactly `frame_size` bytes in
        // the caller-provided output region, whose identity was checked above.
        unsafe { self.data.set_len(offset + frame_size) };
        Ok((offset, frame_size))
    }

    pub(crate) fn into_bytes(self) -> Bytes {
        Bytes::from(self.data)
    }
}

impl BufferedCapture {
    pub(crate) fn capture_mut(&mut self) -> &mut videoio::VideoCapture {
        &mut self.capture
    }
}

pub(crate) fn open_capture(bytes: Bytes, decoder_threads: i32) -> Result<BufferedCapture, String> {
    let mut error = [0 as c_char; 512];
    // SAFETY: `BufferedCapture` owns `bytes` for at least as long as the capture.
    let capture = unsafe {
        smg_opencv_capture_from_buffer(
            bytes.as_ptr(),
            bytes.len(),
            decoder_threads,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if capture.is_null() {
        // SAFETY: the bridge always writes a NUL-terminated message on failure.
        return Err(unsafe { CStr::from_ptr(error.as_ptr()) }
            .to_string_lossy()
            .into_owned());
    }

    Ok(BufferedCapture {
        // SAFETY: the bridge returns a heap-allocated cv::VideoCapture compatible
        // with the opencv crate's generated ownership wrapper.
        capture: unsafe { videoio::VideoCapture::opencv_from_extern(capture) },
        _bytes: bytes,
    })
}

#[cfg(test)]
mod tests {
    use opencv::core::Vec3b;

    use super::*;

    #[test]
    fn rgb_output_buffer_writes_frames_directly_in_rgb_order() {
        let pixels = [Vec3b::from([1, 2, 3]), Vec3b::from([4, 5, 6])];
        let bgr = Mat::new_rows_cols_with_data(1, 2, &pixels)
            .unwrap()
            .try_clone()
            .unwrap();
        let mut output = RgbOutputBuffer::with_capacity(12).unwrap();

        assert_eq!(output.push_bgr(&bgr, 2, 1).unwrap(), (0, 6));
        assert_eq!(output.push_bgr(&bgr, 2, 1).unwrap(), (6, 6));

        let bytes = output.into_bytes();
        assert_eq!(&bytes[..], &[3, 2, 1, 6, 5, 4, 3, 2, 1, 6, 5, 4]);
    }

    #[test]
    fn rgb_output_buffer_supports_dimension_changes() {
        let wide_pixels = [Vec3b::from([1, 2, 3]), Vec3b::from([4, 5, 6])];
        let wide = Mat::new_rows_cols_with_data(1, 2, &wide_pixels)
            .unwrap()
            .try_clone()
            .unwrap();
        let square_pixels = [Vec3b::from([7, 8, 9])];
        let square = Mat::new_rows_cols_with_data(1, 1, &square_pixels)
            .unwrap()
            .try_clone()
            .unwrap();
        let mut output = RgbOutputBuffer::with_capacity(1).unwrap();

        assert_eq!(output.push_bgr(&wide, 2, 1).unwrap(), (0, 6));
        assert_eq!(output.push_bgr(&square, 1, 1).unwrap(), (6, 3));

        let bytes = output.into_bytes();
        assert_eq!(&bytes[..], &[3, 2, 1, 6, 5, 4, 9, 8, 7]);
    }
}
