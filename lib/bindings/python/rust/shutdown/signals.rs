// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::{self, Read};
use std::net::Shutdown;
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicI32, AtomicU64, AtomicUsize, Ordering::SeqCst};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::time::Duration;

use pyo3::prelude::*;

// Only the main interpreter thread installs/restores process signal handlers.
// Dispatch itself may run on any thread, even while Python holds the GIL.
static PID: AtomicI32 = AtomicI32::new(0);
static WRITE_FD: AtomicI32 = AtomicI32::new(-1);
static IN_HANDLER: AtomicUsize = AtomicUsize::new(0);
static RECEIVED_NS: AtomicU64 = AtomicU64::new(0);

fn monotonic_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // CLOCK_MONOTONIC is also the clock used by Python time.monotonic on Unix.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

extern "C" fn on_signal(sig: libc::c_int) {
    // Only async-signal-safe operations here. Forked engine processes inherit
    // the handler but must never write to the parent's watchdog.
    let saved_errno = errno::errno();
    if PID.load(SeqCst) == unsafe { libc::getpid() } {
        IN_HANDLER.fetch_add(1, SeqCst);
        let fd = WRITE_FD.load(SeqCst);
        if fd >= 0 {
            let received = monotonic_ns();
            if RECEIVED_NS.swap(received, SeqCst) != 0 {
                unsafe { libc::_exit(70) }
            }
            // Only the first signal writes, into an otherwise empty socket.
            let timestamp = received.to_ne_bytes();
            let mut offset = 0;
            while offset < timestamp.len() {
                let bytes = &timestamp[offset..];
                let count = unsafe { libc::write(fd, bytes.as_ptr().cast(), bytes.len()) };
                if count > 0 {
                    offset += count as usize;
                } else if errno::errno().0 != libc::EINTR {
                    unsafe { libc::_exit(70) }
                }
            }
        }
        IN_HANDLER.fetch_sub(1, SeqCst);
    }
    // Preserve CPython's normal callback dispatch and event-loop wakeup fd.
    unsafe { pyo3::ffi::PyErr_SetInterruptEx(sig) };
    errno::set_errno(saved_errno);
}

/// Watch SIGTERM/SIGINT from delivery until Python can run its shutdown handler.
#[pyclass(name = "_ShutdownSignals")]
pub struct ShutdownSignals {
    writer: Option<UnixStream>,
    pid: i32,
    disarm: Option<Sender<()>>,
}

#[pymethods]
impl ShutdownSignals {
    #[new]
    #[pyo3(signature = (timeout_secs, parent_fd))]
    fn new(timeout_secs: f64, parent_fd: Option<i32>) -> PyResult<Self> {
        let timeout = Duration::try_from_secs_f64(timeout_secs)
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("invalid shutdown timeout"))?;
        let (reader, writer) = UnixStream::pair()?;
        let pid = unsafe { libc::getpid() };
        if PID.load(SeqCst) == pid && WRITE_FD.load(SeqCst) >= 0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "worker signal handlers already installed",
            ));
        }
        if PID.load(SeqCst) != pid {
            // After fork, handler frames on other parent threads no longer exist.
            WRITE_FD.store(-1, SeqCst);
            IN_HANDLER.store(0, SeqCst);
            PID.store(pid, SeqCst);
        }
        RECEIVED_NS.store(0, SeqCst);
        let (disarm, cancellation) = mpsc::channel();
        std::thread::Builder::new()
            .name("python-signal-watchdog".into())
            .spawn(move || {
                watch(reader, timeout, parent_fd, cancellation);
            })?;
        WRITE_FD.store(writer.as_raw_fd(), SeqCst);
        Ok(Self {
            writer: Some(writer),
            pid,
            disarm: Some(disarm),
        })
    }

    /// Interpose only after signal.signal installed its Python callback. The
    /// Python owner restores the original disposition before closing the socket.
    fn intercept(&self, sig: i32) -> PyResult<()> {
        if !matches!(sig, libc::SIGTERM | libc::SIGINT) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "expected SIGTERM or SIGINT",
            ));
        }
        let previous = unsafe { pyo3::ffi::PyOS_setsig(sig, on_signal) };
        if previous as usize == libc::SIG_ERR {
            return Err(io::Error::last_os_error().into());
        }
        Ok(())
    }

    fn received_at(&self) -> Option<f64> {
        let ns = RECEIVED_NS.load(SeqCst);
        (ns != 0).then_some(ns as f64 / 1e9)
    }

    fn close(&mut self) {
        if let Some(writer) = self.writer.take()
            && self.pid == unsafe { libc::getpid() }
        {
            self.disarm.take();
            WRITE_FD.store(-1, SeqCst);
            // An already-dispatched handler may have loaded the fd. Wait for
            // it before closing so it cannot write to a recycled descriptor.
            while IN_HANDLER.load(SeqCst) != 0 {
                std::hint::spin_loop();
            }
            // A forked engine may have inherited another copy of the writer.
            let _ = writer.shutdown(Shutdown::Write);
        }
    }
}

impl Drop for ShutdownSignals {
    fn drop(&mut self) {
        self.close();
    }
}

fn watch(
    mut reader: UnixStream,
    timeout: Duration,
    parent_fd: Option<i32>,
    cancellation: Receiver<()>,
) {
    let mut timestamp = [0u8; 8];
    if reader.read_exact(&mut timestamp).is_err() {
        return;
    }
    let mut deadline = u64::from_ne_bytes(timestamp) as f64 / 1e9 + timeout.as_secs_f64();
    if let Some(fd) = parent_fd {
        // pread shares neither the file offset nor ownership of the descriptor
        // with Python. An unreadable inherited deadline fails closed.
        let mut bytes = [0u8; 8];
        let read = unsafe { libc::pread(fd, bytes.as_mut_ptr().cast(), bytes.len(), 0) };
        deadline = deadline.min(if read == 8 {
            f64::from_ne_bytes(bytes)
        } else {
            0.0
        });
    }
    let remaining = Duration::from_secs_f64((deadline - monotonic_ns() as f64 / 1e9).max(0.0));
    if cancellation.recv_timeout(remaining) == Err(RecvTimeoutError::Timeout) {
        unsafe { libc::_exit(70) }
    }
}
