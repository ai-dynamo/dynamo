// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use aligned_vec::{AVec, ConstAlign};
use anyhow::Context;
use core::ffi::c_char;
use nix::fcntl::{FallocateFlags, fallocate};
use nix::unistd::{ftruncate, unlink};
use std::ffi::CStr;
use std::ffi::CString;
use std::fs::File;
use std::io::Write;
use std::os::unix::io::{FromRawFd, RawFd};
use std::path::Path;

const DISK_CACHE_KEY: &str = "DYN_KVBM_DISK_CACHE_DIR";
const DEFAULT_DISK_CACHE_DIR: &str = "/tmp/";
const DISK_ZEROFILL_FALLBACK_KEY: &str = "DYN_KVBM_DISK_ZEROFILL_FALLBACK";
const DISK_DISABLE_O_DIRECT_KEY: &str = "DYN_KVBM_DISK_DISABLE_O_DIRECT";
const DISK_ALLOCATOR_TYPE_KEY: &str = "DYN_KVBM_DISK_ALLOCATOR_TYPE";

/// Strategy for applying O_DIRECT to disk cache files.
///
/// Different storage backends require different approaches to setting O_DIRECT.
/// For example, IBM Storage Scale ignores `fcntl(F_SETFL, O_DIRECT)`,
/// requiring O_DIRECT to be passed at file open time instead.
///
/// Implementations are selected via the `DYN_KVBM_DISK_ALLOCATOR_TYPE` env var.
/// Customers can set this based on their storage backend, or contribute new
/// implementations for other providers.
pub trait DiskOpenStrategy: Send + Sync + std::fmt::Debug {
    /// Return a human-readable name for this strategy (for logging).
    fn name(&self) -> &str;

    /// Open a temporary file at the given path template and apply O_DIRECT
    /// as appropriate for the storage backend.
    ///
    /// `template_bytes` must contain a null-terminated path ending in "XXXXXX".
    /// `disable_o_direct` indicates the user has explicitly disabled O_DIRECT.
    ///
    /// Returns the raw file descriptor on success.
    fn open_temp_file(
        &self,
        template_bytes: &mut [u8],
        disable_o_direct: bool,
    ) -> Result<RawFd, StorageError>;
}

/// Default strategy: open with mkostemp(O_CLOEXEC), then apply O_DIRECT via fcntl.
///
/// This works on most POSIX filesystems (ext4, XFS, Lustre, etc.).
#[derive(Debug, Default)]
pub struct DefaultDirectIo;

impl DiskOpenStrategy for DefaultDirectIo {
    fn name(&self) -> &str {
        "default"
    }

    fn open_temp_file(
        &self,
        template_bytes: &mut [u8],
        disable_o_direct: bool,
    ) -> Result<RawFd, StorageError> {
        let raw_fd = unsafe {
            nix::libc::mkostemp(
                template_bytes.as_mut_ptr() as *mut c_char,
                nix::libc::O_CLOEXEC,
            )
        };

        if raw_fd < 0 {
            let file_name = CStr::from_bytes_with_nul(template_bytes)
                .unwrap()
                .to_str()
                .unwrap_or("<invalid utf8>");
            return Err(StorageError::AllocationFailed(format!(
                "Failed to create temp file {}: {}",
                file_name,
                std::io::Error::last_os_error()
            )));
        }

        if !disable_o_direct {
            use nix::fcntl::{FcntlArg, OFlag, fcntl};

            let current_flags = match fcntl(raw_fd, FcntlArg::F_GETFL) {
                Ok(flags) => OFlag::from_bits_truncate(flags),
                Err(e) => {
                    unsafe { nix::libc::close(raw_fd) };
                    let file_name = CStr::from_bytes_with_nul(template_bytes)
                        .unwrap()
                        .to_str()
                        .unwrap_or("<invalid utf8>");
                    let _ = unlink(file_name);
                    return Err(StorageError::AllocationFailed(format!(
                        "Failed to get file flags for {}: {}",
                        file_name, e
                    )));
                }
            };

            let new_flags = current_flags | OFlag::O_DIRECT;

            if let Err(e) = fcntl(raw_fd, FcntlArg::F_SETFL(new_flags)) {
                tracing::error!(
                    "Failed to set O_DIRECT on file descriptor {}: {}. \
                     This may indicate filesystem doesn't support O_DIRECT via fcntl. \
                     Consider setting {}=open-direct for filesystems like IBM Storage Scale, \
                     or {}=true to disable O_DIRECT entirely.",
                    raw_fd,
                    e,
                    DISK_ALLOCATOR_TYPE_KEY,
                    DISK_DISABLE_O_DIRECT_KEY
                );
                unsafe { nix::libc::close(raw_fd) };
                let file_name = CStr::from_bytes_with_nul(template_bytes)
                    .unwrap()
                    .to_str()
                    .unwrap_or("<invalid utf8>");
                let _ = unlink(file_name);
                return Err(StorageError::AllocationFailed(format!(
                    "Failed to set O_DIRECT: {}. Try {}=open-direct or {}=true",
                    e, DISK_ALLOCATOR_TYPE_KEY, DISK_DISABLE_O_DIRECT_KEY
                )));
            }

            tracing::debug!("O_DIRECT enabled via fcntl for disk cache (fd={})", raw_fd);
        } else {
            tracing::warn!(
                "O_DIRECT disabled via {}. GPU DirectStorage performance may be reduced.",
                DISK_DISABLE_O_DIRECT_KEY
            );
        }

        Ok(raw_fd)
    }
}

/// Open-direct strategy: pass O_DIRECT directly to mkostemp at file open time.
///
/// Some filesystems (e.g., IBM Storage Scale) ignore `fcntl(F_SETFL, O_DIRECT)`,
/// so O_DIRECT must be specified at file open time. This strategy passes O_DIRECT
/// as a flag to mkostemp instead of applying it post-creation via fcntl.
#[derive(Debug, Default)]
pub struct MkostempDirectIo;

impl DiskOpenStrategy for MkostempDirectIo {
    fn name(&self) -> &str {
        "open-direct"
    }

    fn open_temp_file(
        &self,
        template_bytes: &mut [u8],
        disable_o_direct: bool,
    ) -> Result<RawFd, StorageError> {
        let flags = if disable_o_direct {
            tracing::warn!(
                "O_DIRECT disabled via {}. GPU DirectStorage performance may be reduced.",
                DISK_DISABLE_O_DIRECT_KEY
            );
            nix::libc::O_CLOEXEC
        } else {
            nix::libc::O_CLOEXEC | nix::libc::O_DIRECT
        };

        let raw_fd =
            unsafe { nix::libc::mkostemp(template_bytes.as_mut_ptr() as *mut c_char, flags) };

        if raw_fd < 0 {
            let file_name = CStr::from_bytes_with_nul(template_bytes)
                .unwrap()
                .to_str()
                .unwrap_or("<invalid utf8>");
            return Err(StorageError::AllocationFailed(format!(
                "Failed to create temp file {}: {}",
                file_name,
                std::io::Error::last_os_error()
            )));
        }

        if !disable_o_direct {
            tracing::debug!("O_DIRECT enabled via mkostemp at open time (fd={})", raw_fd);
        }

        Ok(raw_fd)
    }
}

/// Create a `DiskOpenStrategy` from the `DYN_KVBM_DISK_ALLOCATOR_TYPE` env var.
///
/// Supported values:
/// - `"default"` (default): Apply O_DIRECT via fcntl after file creation.
/// - `"open-direct"`: Pass O_DIRECT to mkostemp at file open time (required for filesystems
///   like IBM Storage Scale where fcntl-based O_DIRECT is ignored).
fn disk_open_strategy_from_env() -> Result<Box<dyn DiskOpenStrategy>, StorageError> {
    match std::env::var(DISK_ALLOCATOR_TYPE_KEY).as_deref() {
        Ok("default") | Err(_) => {
            tracing::info!("Using default fcntl disk open strategy");
            Ok(Box::new(DefaultDirectIo))
        }
        Ok("open-direct") => {
            tracing::info!("Using open-direct disk open strategy (O_DIRECT via mkostemp)");
            Ok(Box::new(MkostempDirectIo))
        }
        Ok(unknown) => Err(StorageError::AllocationFailed(format!(
            "Unknown {}={:?}. Supported values: \"default\", \"open-direct\"",
            DISK_ALLOCATOR_TYPE_KEY, unknown
        ))),
    }
}

#[derive(Debug)]
pub struct DiskStorage {
    fd: u64,
    file_name: String,
    size: usize,
    handles: RegistrationHandles,
    unlinked: bool,
}

impl Local for DiskStorage {}
impl SystemAccessible for DiskStorage {}

const ZERO_BUF_SIZE: usize = 16 * 1024 * 1024; // 16MB
const PAGE_SIZE: usize = 4096; // Standard page size for O_DIRECT alignment

// Type alias for 4096-byte (page size) aligned vectors
type Align4096 = ConstAlign<4096>;

/// Create a page-aligned zero-filled buffer for O_DIRECT I/O operations.
/// On filesystems like Lustre, O_DIRECT requires both buffer address and I/O size
/// to be aligned to the filesystem block size (typically page size).
fn create_aligned_buffer(size: usize) -> anyhow::Result<AVec<u8, Align4096>> {
    // Round up to nearest page size to ensure alignment requirements
    let aligned_size = size.div_ceil(PAGE_SIZE) * PAGE_SIZE;

    // Create aligned vector with compile-time PAGE_SIZE alignment
    let mut buf = AVec::<u8, Align4096>::new(PAGE_SIZE);
    buf.resize(aligned_size, 0u8); // Zero-fill

    tracing::trace!(
        "Allocated aligned buffer: size={}, aligned_size={}, align={}",
        size,
        aligned_size,
        PAGE_SIZE
    );

    Ok(buf)
}

fn allocate_file(fd: RawFd, size: u64) -> anyhow::Result<()> {
    match fallocate(fd, FallocateFlags::empty(), 0, size as i64) {
        Ok(_) => {
            tracing::debug!("Successfully allocated {} bytes using fallocate()", size);
            Ok(())
        }
        Err(err) => match err {
            nix::errno::Errno::EOPNOTSUPP => {
                let do_zero_fill = std::env::var(DISK_ZEROFILL_FALLBACK_KEY).is_ok();
                if do_zero_fill {
                    tracing::warn!(
                        "fallocate() not supported on this filesystem, using zero-fill fallback. \
                         This may be slower but provides actual disk space allocation. \
                         Using page-aligned buffers (alignment={}) for O_DIRECT compatibility.",
                        PAGE_SIZE
                    );

                    // Use page-aligned buffer for O_DIRECT compatibility (required on Lustre)
                    let buf = create_aligned_buffer(ZERO_BUF_SIZE)
                        .context("Failed to allocate aligned zero buffer")?;

                    let mut file =
                        unsafe { File::from_raw_fd(nix::unistd::dup(fd).context("dup error")?) };

                    let mut written: u64 = 0;
                    while written < size {
                        // Calculate how much to write in this iteration.
                        // For O_DIRECT, we must write in multiples of page size, except possibly
                        // the last write.
                        let remaining = size - written;
                        let to_write = if remaining >= buf.len() as u64 {
                            // Full buffer write - always aligned
                            buf.len()
                        } else {
                            // Last partial write - round up to page size for O_DIRECT
                            let aligned = (remaining as usize).div_ceil(PAGE_SIZE) * PAGE_SIZE;
                            std::cmp::min(aligned, buf.len())
                        };

                        match file.write(&buf[..to_write]) {
                            Ok(n) => {
                                if n != to_write {
                                    tracing::error!(
                                        "Partial write detected: requested={}, written={}, \
                                         total_written={}/{}, fd={}, errno={:?}",
                                        to_write,
                                        n,
                                        written,
                                        size,
                                        fd,
                                        std::io::Error::last_os_error()
                                    );
                                    anyhow::bail!(
                                        "Partial write: expected {} bytes, wrote {} bytes (total {}/{})",
                                        to_write,
                                        n,
                                        written + n as u64,
                                        size
                                    );
                                }
                                written += n as u64;
                                tracing::trace!(
                                    "Zero-fill progress: {}/{} bytes ({:.1}%)",
                                    written,
                                    size,
                                    (written as f64 / size as f64) * 100.0
                                );
                            }
                            Err(e) => {
                                let errno = e.raw_os_error();
                                tracing::error!(
                                    "Zero-fill write failed: error={}, errno={:?}, \
                                     fd={}, to_write={}, written={}/{}, buf_addr={:p}, buf_align={}",
                                    e,
                                    errno,
                                    fd,
                                    to_write,
                                    written,
                                    size,
                                    buf.as_ptr(),
                                    PAGE_SIZE
                                );

                                // Provide specific guidance for common errors
                                if errno == Some(22) {
                                    // EINVAL - typically alignment issues
                                    anyhow::bail!(
                                        "Zero-fill write failed with EINVAL (errno 22). \
                                         This usually indicates O_DIRECT alignment issues. \
                                         Buffer is page-aligned ({}), but filesystem may require \
                                         different alignment. Try setting {}=true to disable O_DIRECT. \
                                         Original error: {}",
                                        PAGE_SIZE,
                                        DISK_DISABLE_O_DIRECT_KEY,
                                        e
                                    );
                                } else {
                                    anyhow::bail!("Zero-fill write failed: {}", e);
                                }
                            }
                        }
                    }

                    file.flush().context("Failed to flush zero-filled file")?;

                    // Truncate to exact size if we over-allocated due to alignment
                    if written > size {
                        tracing::debug!(
                            "Truncating file from {} to {} bytes (alignment padding)",
                            written,
                            size
                        );
                        ftruncate(fd, size as i64).context("Failed to truncate to exact size")?;
                    }

                    tracing::info!(
                        "Successfully zero-filled {} bytes using aligned buffers",
                        size
                    );
                    Ok(())
                } else {
                    tracing::warn!(
                        "fallocate() not supported on this filesystem, using truncate fallback. \
                         This may may not actually allocate disk space. \
                         Consider setting {}=true for slower zero-fill fallback.",
                        DISK_ZEROFILL_FALLBACK_KEY
                    );
                    // default fallback: set file length without zero-filling (does not really
                    // allocate)
                    ftruncate(fd, size as i64).context("truncate error")
                }
            }
            _ => Err(err.into()),
        },
    }
}

impl DiskStorage {
    pub fn new(size: usize, strategy: &dyn DiskOpenStrategy) -> Result<Self, StorageError> {
        // We need to open our file with some special flags that aren't supported by the tempfile crate.
        // Instead, we'll use the mkostemp function to create a temporary file with the correct flags.

        let specified_dir =
            std::env::var(DISK_CACHE_KEY).unwrap_or_else(|_| DEFAULT_DISK_CACHE_DIR.to_string());
        let file_path = Path::new(&specified_dir).join("dynamo-kvbm-disk-cache-XXXXXX");

        if !file_path.exists() {
            std::fs::create_dir_all(file_path.parent().unwrap()).unwrap();
        }

        tracing::debug!(
            "Allocating disk cache file at {} using {} strategy",
            file_path.display(),
            strategy.name()
        );

        let template = CString::new(file_path.to_str().unwrap()).unwrap();
        let mut template_bytes = template.into_bytes_with_nul();

        let disable_o_direct = std::env::var(DISK_DISABLE_O_DIRECT_KEY).is_ok();
        let raw_fd = strategy.open_temp_file(&mut template_bytes, disable_o_direct)?;

        let file_name = CStr::from_bytes_with_nul(template_bytes.as_slice())
            .unwrap()
            .to_str()
            .map_err(|e| {
                StorageError::AllocationFailed(format!("Failed to read temp file name: {}", e))
            })?
            .to_string();

        // We need to use fallocate to actually allocate the storage and create the blocks on disk.
        allocate_file(raw_fd, size as u64).map_err(|e| {
            StorageError::AllocationFailed(format!("Failed to allocate temp file: {}", e))
        })?;

        tracing::info!(
            "DiskStorage created: fd={}, file={}, size={} bytes",
            raw_fd,
            file_name,
            size
        );

        Ok(Self {
            fd: raw_fd as u64,
            file_name,
            size,
            handles: RegistrationHandles::new(),
            unlinked: false,
        })
    }

    pub fn fd(&self) -> u64 {
        self.fd
    }

    /// Unlink our temp file.
    /// This means that when this process terminates, the file will be automatically deleted by the OS.
    /// Unfortunately, GDS requires that files we try to register must be linked.
    /// To get around this, we unlink the file only after we've registered it with NIXL.
    pub fn unlink(&mut self) -> Result<(), StorageError> {
        if self.unlinked {
            return Ok(());
        }

        tracing::info!(
            "Unlinking temp file (fd={}, file={}). File will be deleted when fd closes.",
            self.fd,
            self.file_name
        );

        self.unlinked = true;

        unlink(self.file_name.as_str()).map_err(|e| {
            tracing::error!(
                "Failed to unlink temp file: fd={}, file={}, error={}",
                self.fd,
                self.file_name,
                e
            );
            StorageError::AllocationFailed(format!("Failed to unlink temp file: {}", e))
        })
    }

    pub fn unlinked(&self) -> bool {
        self.unlinked
    }
}

impl Drop for DiskStorage {
    fn drop(&mut self) {
        tracing::warn!(
            "DiskStorage being dropped: fd={}, file={}, size={} bytes, already_unlinked={}",
            self.fd,
            self.file_name,
            self.size,
            self.unlinked
        );

        self.handles.release();
        let _ = self.unlink();

        tracing::info!(
            "DiskStorage dropped and cleaned up: fd={}, file={}",
            self.fd,
            self.file_name
        );
    }
}

impl Storage for DiskStorage {
    fn storage_type(&self) -> StorageType {
        StorageType::Disk(self.fd())
    }

    fn addr(&self) -> u64 {
        0
    }

    fn size(&self) -> usize {
        self.size
    }

    unsafe fn as_ptr(&self) -> *const u8 {
        std::ptr::null()
    }

    unsafe fn as_mut_ptr(&mut self) -> *mut u8 {
        std::ptr::null_mut()
    }
}

impl RegisterableStorage for DiskStorage {
    fn register(
        &mut self,
        key: &str,
        handle: Box<dyn RegistationHandle>,
    ) -> Result<(), StorageError> {
        self.handles.register(key, handle)
    }

    fn is_registered(&self, key: &str) -> bool {
        self.handles.is_registered(key)
    }

    fn registration_handle(&self, key: &str) -> Option<&dyn RegistationHandle> {
        self.handles.registration_handle(key)
    }
}

pub struct DiskAllocator {
    strategy: Box<dyn DiskOpenStrategy>,
}

impl Default for DiskAllocator {
    fn default() -> Self {
        Self {
            strategy: Box::new(DefaultDirectIo),
        }
    }
}

impl DiskAllocator {
    /// Create a DiskAllocator by reading `DYN_KVBM_DISK_ALLOCATOR_TYPE` from the environment.
    pub fn from_env() -> Result<Self, StorageError> {
        Ok(Self {
            strategy: disk_open_strategy_from_env()?,
        })
    }

    /// Create a DiskAllocator with an explicit strategy.
    pub fn with_strategy(strategy: Box<dyn DiskOpenStrategy>) -> Self {
        Self { strategy }
    }
}

impl StorageAllocator<DiskStorage> for DiskAllocator {
    fn allocate(&self, size: usize) -> Result<DiskStorage, StorageError> {
        DiskStorage::new(size, self.strategy.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock writer that enforces strict O_DIRECT alignment rules like Lustre.
    /// This allows us to test the alignment logic without needing an actual Lustre filesystem.
    struct StrictODirectWriter {
        bytes_written: usize,
        writes: Vec<(usize, usize)>, // (address, size) of each write
    }

    impl StrictODirectWriter {
        fn new() -> Self {
            Self {
                bytes_written: 0,
                writes: Vec::new(),
            }
        }
    }

    impl std::io::Write for StrictODirectWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            let addr = buf.as_ptr() as usize;
            let size = buf.len();

            // Enforce Lustre-like O_DIRECT requirements
            if !addr.is_multiple_of(PAGE_SIZE) {
                eprintln!(
                    "EINVAL: Buffer address {:#x} not aligned to {} bytes",
                    addr, PAGE_SIZE
                );
                return Err(std::io::Error::from_raw_os_error(22)); // EINVAL
            }

            if !size.is_multiple_of(PAGE_SIZE) {
                eprintln!(
                    "EINVAL: Write size {} not aligned to {} bytes",
                    size, PAGE_SIZE
                );
                return Err(std::io::Error::from_raw_os_error(22)); // EINVAL
            }

            self.writes.push((addr, size));
            self.bytes_written += size;
            Ok(size)
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Test that aligned buffers satisfy strict O_DIRECT requirements.
    #[test]
    fn test_aligned_buffer_with_strict_writer() {
        let test_sizes = vec![
            1234,
            PAGE_SIZE,
            PAGE_SIZE + 1,
            16 * 1024 * 1024, // 16 MB
            1_000_000,
        ];

        for requested_size in test_sizes {
            let buf =
                create_aligned_buffer(requested_size).expect("Failed to create aligned buffer");

            let mut writer = StrictODirectWriter::new();

            // This should succeed - aligned buffer meets strict requirements
            let result = writer.write(&buf[..]);

            assert!(
                result.is_ok(),
                "Aligned buffer write failed for size {}: {:?}",
                requested_size,
                result.err()
            );

            assert_eq!(
                writer.bytes_written,
                buf.len(),
                "Bytes written mismatch for size {}",
                requested_size
            );
        }
    }

    /// Test that regular Vec<u8> FAILS with strict O_DIRECT writer.
    /// This demonstrates the bug that existed before the fix.
    #[test]
    fn test_unaligned_vec_fails_strict_writer() {
        let vec_buf = vec![0u8; 8192]; // 8KB, but not guaranteed aligned address

        let mut writer = StrictODirectWriter::new();
        let result = writer.write(&vec_buf);

        // This may fail with EINVAL if vec! didn't happen to allocate aligned memory
        // (which is common but not guaranteed)
        if let Err(err) = result {
            assert_eq!(
                err.raw_os_error(),
                Some(22),
                "Expected EINVAL (22), got {:?}",
                err
            );
            eprintln!("Confirmed: vec! buffer failed strict alignment check (as expected)");
        } else {
            eprintln!("Note: vec! happened to be aligned this time (lucky!), but not guaranteed");
        }
    }

    /// Test that the zero-fill write loop produces properly aligned write operations.
    #[test]
    fn test_zerofill_write_loop_alignment() {
        let test_sizes = vec![
            1_000_000,   // 1 MB non-aligned
            10_000_000,  // 10 MB non-aligned
            100_000_000, // 100 MB non-aligned
        ];

        for total_size in test_sizes {
            let buf = create_aligned_buffer(ZERO_BUF_SIZE).expect("Failed to create buffer");

            let mut writer = StrictODirectWriter::new();
            let mut written: u64 = 0;

            // Simulate the zero-fill loop from allocate_file()
            while written < total_size {
                let remaining = total_size - written;
                let to_write = if remaining >= buf.len() as u64 {
                    buf.len()
                } else {
                    let aligned = (remaining as usize).div_ceil(PAGE_SIZE) * PAGE_SIZE;
                    std::cmp::min(aligned, buf.len())
                };

                // This should always succeed with our aligned buffer
                writer.write_all(&buf[..to_write]).unwrap_or_else(|e| {
                    panic!(
                        "Write failed at offset {} for total size {}: {:?}",
                        written, total_size, e
                    )
                });

                written += to_write as u64;
            }

            assert!(
                written >= total_size,
                "Didn't write enough bytes for size {}",
                total_size
            );

            eprintln!(
                "Size {} passed: {} writes, {} total bytes",
                total_size,
                writer.writes.len(),
                writer.bytes_written
            );
        }
    }

    /// Integration test: Verify disk allocation with zero-fill fallback on filesystems
    /// that don't support fallocate (like Lustre). This exercises the O_DIRECT + aligned
    /// buffer code path that was failing before the fix.
    ///
    /// Run with: cargo test -- --ignored --nocapture test_zerofill_with_o_direct
    #[test]
    #[ignore]
    fn test_zerofill_with_o_direct() {
        unsafe {
            std::env::set_var(DISK_ZEROFILL_FALLBACK_KEY, "1");
        }

        // Test various sizes including non-page-aligned sizes that would fail with
        // unaligned buffers on Lustre
        let test_cases = vec![
            ("Small non-aligned", 1234),
            ("One page", PAGE_SIZE),
            ("Just over one page", PAGE_SIZE + 1),
            ("Multi-page non-aligned", 3 * PAGE_SIZE + 567),
            ("Large 10MB", 10 * 1024 * 1024),
        ];

        for (name, size) in test_cases {
            eprintln!("Testing: {} ({} bytes)", name, size);

            let strategy = DefaultDirectIo;
            let storage = DiskStorage::new(size, &strategy).unwrap_or_else(|e| {
                panic!("Failed to allocate {} bytes ({}): {:?}", size, name, e)
            });

            // Verify the file is actually the correct size
            assert_eq!(storage.size(), size, "Size mismatch for {}", name);

            // Verify we can read from the file (tests that data was actually written)
            let fd = storage.fd() as RawFd;
            let mut buf = vec![0u8; std::cmp::min(size, 4096)];

            let bytes_read = nix::sys::uio::pread(fd, &mut buf, 0)
                .unwrap_or_else(|e| panic!("Failed to read back data for {}: {:?}", name, e));

            assert!(bytes_read > 0, "No data read back for {}", name);
            assert!(
                buf.iter().all(|&b| b == 0),
                "File should be zero-filled for {}",
                name
            );

            eprintln!("{} passed", name);
        }

        unsafe {
            std::env::remove_var(DISK_ZEROFILL_FALLBACK_KEY);
        }
    }

    /// Test that O_DIRECT can be disabled and allocation still works.
    #[test]
    #[ignore]
    fn test_disable_o_direct() {
        unsafe {
            std::env::set_var(DISK_DISABLE_O_DIRECT_KEY, "1");
            std::env::set_var(DISK_ZEROFILL_FALLBACK_KEY, "1");
        }

        let size = 1024 * 1024;
        let strategy = DefaultDirectIo;
        let storage =
            DiskStorage::new(size, &strategy).expect("Failed to allocate with O_DIRECT disabled");

        assert_eq!(storage.size(), size);

        unsafe {
            std::env::remove_var(DISK_DISABLE_O_DIRECT_KEY);
            std::env::remove_var(DISK_ZEROFILL_FALLBACK_KEY);
        }
    }

    /// Test that disk_open_strategy_from_env returns DefaultDirectIo by default.
    #[test]
    fn test_strategy_from_env_default() {
        temp_env::with_var_unset(DISK_ALLOCATOR_TYPE_KEY, || {
            let strategy = disk_open_strategy_from_env().expect("default strategy should succeed");
            assert_eq!(strategy.name(), "default");
        });
    }

    /// Test that disk_open_strategy_from_env returns DefaultDirectIo for explicit "default".
    #[test]
    fn test_strategy_from_env_fcntl() {
        temp_env::with_var(DISK_ALLOCATOR_TYPE_KEY, Some("default"), || {
            let strategy = disk_open_strategy_from_env().expect("fcntl strategy should succeed");
            assert_eq!(strategy.name(), "default");
        });
    }

    /// Test that disk_open_strategy_from_env returns MkostempDirectIo for "open-direct".
    #[test]
    fn test_strategy_from_env_open_direct() {
        temp_env::with_var(DISK_ALLOCATOR_TYPE_KEY, Some("open-direct"), || {
            let strategy =
                disk_open_strategy_from_env().expect("open-direct strategy should succeed");
            assert_eq!(strategy.name(), "open-direct");
        });
    }

    /// Test that disk_open_strategy_from_env rejects unknown values.
    #[test]
    fn test_strategy_from_env_unknown() {
        temp_env::with_var(DISK_ALLOCATOR_TYPE_KEY, Some("not-a-real-backend"), || {
            let result = disk_open_strategy_from_env();
            assert!(result.is_err(), "unknown strategy should fail");
        });
    }
}
