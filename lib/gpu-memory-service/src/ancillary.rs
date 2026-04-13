// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SCM_RIGHTS file descriptor passing over Unix domain sockets.
//!
//! Provides helpers for sending and receiving file descriptors alongside
//! framed data over `UnixStream`. Used by the GPU Memory Service to pass
//! CUDA VMM allocation handles (exported as POSIX FDs) between processes.
//!
//! This operates at a lower level than the normal Velo framing — it uses
//! `sendmsg`/`recvmsg` with ancillary data (control messages) to transfer
//! file descriptors out-of-band.

use std::io::IoSlice;
use std::os::fd::RawFd;
use std::os::unix::io::AsRawFd;

use anyhow::{Context, Result};
use nix::sys::socket::{
    self, ControlMessage, ControlMessageOwned, MsgFlags, UnixAddr,
};

/// Send a frame with attached file descriptors via SCM_RIGHTS.
///
/// The data is sent as a single `sendmsg` call with the FDs attached as
/// ancillary data. The receiver must use [`recv_frame_with_fds`] to extract
/// both the data and the FDs.
///
/// # Arguments
/// * `stream` - The Unix stream (must be connected)
/// * `data` - Frame data to send
/// * `fds` - File descriptors to attach (may be empty)
pub async fn send_frame_with_fds(
    stream: &tokio::net::UnixStream,
    data: &[u8],
    fds: &[RawFd],
) -> Result<()> {
    let raw_fd = stream.as_raw_fd();

    // Wait for the stream to be writable
    stream
        .writable()
        .await
        .context("waiting for stream to be writable")?;

    // Perform the sendmsg in a blocking-safe way
    stream.try_io(tokio::io::Interest::WRITABLE, || {
        let iov = [IoSlice::new(data)];

        let cmsg: Vec<ControlMessage<'_>> = if fds.is_empty() {
            vec![]
        } else {
            vec![ControlMessage::ScmRights(fds)]
        };

        socket::sendmsg::<UnixAddr>(raw_fd, &iov, &cmsg, MsgFlags::empty(), None)
            .map_err(|e| std::io::Error::from_raw_os_error(e as i32))?;

        Ok(())
    })?;

    Ok(())
}

/// Receive a frame, extracting any attached file descriptors.
///
/// Returns the received data and any file descriptors that were attached
/// via SCM_RIGHTS.
///
/// **Note**: At most 4 file descriptors can be received per call. Any excess
/// FDs sent by the peer are silently dropped by the kernel.
///
/// # Arguments
/// * `stream` - The Unix stream (must be connected)
/// * `buf_size` - Maximum size of the data buffer
pub async fn recv_frame_with_fds(
    stream: &tokio::net::UnixStream,
    buf_size: usize,
) -> Result<(Vec<u8>, Vec<RawFd>)> {
    let raw_fd = stream.as_raw_fd();

    // Wait for the stream to be readable
    stream
        .readable()
        .await
        .context("waiting for stream to be readable")?;

    let result: std::io::Result<(Vec<u8>, Vec<RawFd>)> =
        stream.try_io(tokio::io::Interest::READABLE, || {
            let mut buf = vec![0u8; buf_size];
            let mut cmsg_buf = nix::cmsg_space!([RawFd; 4]);

            // recvmsg borrows buf through iov, so we scope the borrow
            let (bytes_read, fds) = {
                let mut iov = [std::io::IoSliceMut::new(&mut buf)];

                let msg = socket::recvmsg::<UnixAddr>(
                    raw_fd,
                    &mut iov,
                    Some(&mut cmsg_buf),
                    MsgFlags::empty(),
                )
                .map_err(|e| std::io::Error::from_raw_os_error(e as i32))?;

                let bytes_read = msg.bytes;

                // Extract file descriptors from control messages
                let mut fds = Vec::new();
                if let Ok(cmsgs) = msg.cmsgs() {
                    for cmsg in cmsgs {
                        if let ControlMessageOwned::ScmRights(received_fds) = cmsg {
                            fds.extend_from_slice(&received_fds);
                        }
                    }
                }

                (bytes_read, fds)
            };

            buf.truncate(bytes_read);
            Ok((buf, fds))
        });

    result.context("recvmsg with FDs")
}
