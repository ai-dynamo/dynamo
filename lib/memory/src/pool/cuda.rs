// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CUDA memory pool for efficient device memory allocation in hot paths.
//!
//! This module provides a safe wrapper around CUDA's memory pool APIs, enabling
//! fast async allocations that avoid the overhead of cudaMalloc/cudaFree per call.
//! Memory is returned to the pool on free and reused for subsequent allocations.
//!
//! # Thread Safety
//!
//! [`CudaMemPool`] uses internal locking to serialize host-side calls to the CUDA
//! driver. This is required because `cuMemAllocFromPoolAsync` is not host-thread
//! reentrant. The GPU-side operations remain stream-ordered and asynchronous.

use anyhow::{Result, anyhow};
use cudarc::driver::sys::{
    self, CUmemAllocationType, CUmemLocationType, CUmemPool_attribute, CUmemPoolProps,
    CUmemoryPool, CUresult, CUstream,
};
use cudarc::driver::{CudaContext, CudaStream};
use std::ptr;
use std::sync::{Arc, Mutex};

/// Builder for creating a CUDA memory pool with configurable parameters.
///
/// # Example
/// ```ignore
/// let pool = CudaMemPoolBuilder::new(context, 64 * 1024 * 1024) // 64 MiB reserve
///     .release_threshold(32 * 1024 * 1024) // 32 MiB release threshold
///     .build()?;
/// ```
pub struct CudaMemPoolBuilder {
    /// CUDA context for the target device.
    context: Arc<CudaContext>,
    /// Bytes to pre-allocate to warm the pool.
    reserve_size: usize,
    /// Optional threshold above which memory is returned to the system on free.
    release_threshold: Option<u64>,
}

impl CudaMemPoolBuilder {
    /// Create a new builder with the required reserve size.
    ///
    /// # Arguments
    /// * `context` - CUDA context for the device
    /// * `reserve_size` - Number of bytes to pre-allocate to warm the pool
    pub fn new(context: Arc<CudaContext>, reserve_size: usize) -> Self {
        Self {
            context,
            reserve_size,
            release_threshold: None,
        }
    }

    /// Set the release threshold for the pool.
    ///
    /// Memory above this threshold is returned to the system when freed.
    /// If not set, no release threshold is configured (CUDA default behavior).
    pub fn release_threshold(mut self, threshold: u64) -> Self {
        self.release_threshold = Some(threshold);
        self
    }

    /// Build the CUDA memory pool.
    ///
    /// This will:
    /// 1. Create the pool
    /// 2. Set the release threshold if configured
    /// 3. Pre-allocate and free memory to warm the pool
    pub fn build(self) -> Result<CudaMemPool> {
        // Initialize pool properties
        let mut props: CUmemPoolProps = unsafe { std::mem::zeroed() };
        props.allocType = CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        props.location.type_ = CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        // CUDA 13.2 moved `id` into an anonymous union inside CUmemLocation;
        // the field path depends on the toolkit (see build.rs).
        #[cfg(not(cuda_mem_location_union))]
        {
            props.location.id = self.context.cu_device();
        }
        #[cfg(cuda_mem_location_union)]
        {
            // Writing to a union field is safe; only reads are `unsafe`.
            props.location.__bindgen_anon_1.id = self.context.cu_device();
        }

        let mut pool: CUmemoryPool = ptr::null_mut();

        // Create the pool
        let result = unsafe { sys::cuMemPoolCreate(&mut pool, &props) };
        if result != CUresult::CUDA_SUCCESS {
            return Err(anyhow!("cuMemPoolCreate failed with error: {:?}", result));
        }

        // Set release threshold if configured
        if let Some(threshold) = self.release_threshold {
            let result = unsafe {
                sys::cuMemPoolSetAttribute(
                    pool,
                    CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    &threshold as *const u64 as *mut std::ffi::c_void,
                )
            };
            if result != CUresult::CUDA_SUCCESS {
                // Clean up on failure
                unsafe { sys::cuMemPoolDestroy(pool) };
                return Err(anyhow!(
                    "cuMemPoolSetAttribute failed with error: {:?}",
                    result
                ));
            }
        }

        let cuda_pool = CudaMemPool {
            inner: Mutex::new(pool),
        };

        // Warm the pool by pre-allocating and freeing memory
        if self.reserve_size > 0 {
            // Create a temporary stream for warming
            let stream = self.context.new_stream()?;

            // Allocate to warm the pool (using safe variant)
            let ptr = cuda_pool.alloc_async(self.reserve_size, &stream)?;

            // Free back to pool (memory stays reserved)
            cuda_pool.free_async(ptr, &stream)?;

            // Synchronize to ensure operations complete
            // SAFETY: stream.cu_stream() is valid for the lifetime of `stream`
            let result = unsafe { sys::cuStreamSynchronize(stream.cu_stream()) };
            if result != CUresult::CUDA_SUCCESS {
                return Err(anyhow!(
                    "cuStreamSynchronize failed with error: {:?}",
                    result
                ));
            }
        }

        Ok(cuda_pool)
    }
}

/// Safe wrapper around a CUDA memory pool.
///
/// The pool amortizes allocation overhead by maintaining a reservoir of device memory.
/// Allocations are fast sub-allocations from this reservoir, and frees return memory
/// to the pool rather than the OS (until the release threshold is exceeded).
///
/// # Thread Safety
///
/// This type uses internal locking to serialize host-side calls to CUDA driver APIs.
/// `cuMemAllocFromPoolAsync` is not host-thread reentrant, so concurrent calls from
/// multiple threads must be serialized. The GPU-side operations remain asynchronous
/// and stream-ordered.
///
/// Use [`CudaMemPoolBuilder`] for configurable pool creation with pre-allocation.
pub struct CudaMemPool {
    /// Mutex protecting the pool handle for host-thread serialization.
    ///
    /// CUDA's `cuMemAllocFromPoolAsync` does not guarantee host-thread reentrancy,
    /// so all calls to the pool must be serialized on the host side.
    inner: Mutex<CUmemoryPool>,
}

// SAFETY: CudaMemPool is Send because the Mutex serializes all host-side access
// to the pool handle, and CUDA driver state is thread-safe when properly serialized.
unsafe impl Send for CudaMemPool {}

// SAFETY: CudaMemPool is Sync because all access to the pool handle goes through
// the Mutex, which serializes host-thread access. The CUDA driver requires this
// serialization because cuMemAllocFromPoolAsync is not host-thread reentrant.
unsafe impl Sync for CudaMemPool {}

impl CudaMemPool {
    /// Create a builder for a new CUDA memory pool.
    ///
    /// # Arguments
    /// * `context` - CUDA context for the device
    /// * `reserve_size` - Number of bytes to pre-allocate to warm the pool
    pub fn builder(context: Arc<CudaContext>, reserve_size: usize) -> CudaMemPoolBuilder {
        CudaMemPoolBuilder::new(context, reserve_size)
    }

    /// Allocate memory from the pool asynchronously.
    ///
    /// This is the safe variant that takes a `&CudaStream` reference, ensuring
    /// the stream is valid for the duration of the call.
    ///
    /// The allocation is stream-ordered; the memory is available for use
    /// after all preceding operations on the stream complete.
    ///
    /// # Host Serialization
    ///
    /// This method acquires an internal mutex because `cuMemAllocFromPoolAsync`
    /// is not host-thread reentrant. The allocation itself is stream-ordered on
    /// the GPU side.
    ///
    /// # Arguments
    /// * `size` - Size in bytes to allocate
    /// * `stream` - CUDA stream for async ordering
    ///
    /// # Returns
    /// Device pointer to the allocated memory
    pub fn alloc_async(&self, size: usize, stream: &CudaStream) -> Result<u64> {
        // SAFETY: stream.cu_stream() returns a valid handle owned by the CudaStream,
        // and the borrow ensures the stream lives for the duration of this call.
        unsafe { self.alloc_async_raw(size, stream.cu_stream()) }
    }

    /// Allocate memory from the pool asynchronously (raw stream handle variant).
    ///
    /// This is the unsafe variant for use when you have a raw `CUstream` handle
    /// from sources other than cudarc's `CudaStream`.
    ///
    /// # Host Serialization
    ///
    /// This method acquires an internal mutex because `cuMemAllocFromPoolAsync`
    /// is not host-thread reentrant.
    ///
    /// # Arguments
    /// * `size` - Size in bytes to allocate
    /// * `stream` - Raw CUDA stream handle for async ordering
    ///
    /// # Returns
    /// Device pointer to the allocated memory
    ///
    /// # Safety
    ///
    /// The caller must ensure that `stream` is a valid CUDA stream handle that
    /// will remain valid for the duration of this call.
    pub unsafe fn alloc_async_raw(&self, size: usize, stream: CUstream) -> Result<u64> {
        let pool = self
            .inner
            .lock()
            .map_err(|e| anyhow!("mutex poisoned: {}", e))?;

        let mut ptr: u64 = 0;

        let result = unsafe { sys::cuMemAllocFromPoolAsync(&mut ptr, size, *pool, stream) };

        if result != CUresult::CUDA_SUCCESS {
            return Err(anyhow!(
                "cuMemAllocFromPoolAsync failed with error: {:?}",
                result
            ));
        }

        Ok(ptr)
    }

    /// Free memory back to the pool asynchronously.
    ///
    /// This is the safe variant that takes a `&CudaStream` reference.
    ///
    /// The memory is returned to the pool's reservoir (not the OS) and can be
    /// reused by subsequent allocations. The free is stream-ordered.
    ///
    /// # Arguments
    /// * `ptr` - Device pointer previously allocated from this pool
    /// * `stream` - CUDA stream for async ordering
    pub fn free_async(&self, ptr: u64, stream: &CudaStream) -> Result<()> {
        // SAFETY: stream.cu_stream() returns a valid handle owned by the CudaStream,
        // and the borrow ensures the stream lives for the duration of this call.
        unsafe { self.free_async_raw(ptr, stream.cu_stream()) }
    }

    // NOTE: Unlike alloc_async_raw, this method does NOT acquire the pool mutex.
    // The mutex in alloc_async_raw ensures each allocation returns a unique pointer.
    // cuMemFreeAsync only enqueues a stream-ordered free operation for that unique
    // pointer - multiple threads can safely enqueue frees for different unique pointers
    // concurrently. The actual return-to-pool happens asynchronously on the GPU side.

    /// Free memory back to the pool asynchronously (raw stream handle variant).
    ///
    /// This is the unsafe variant for use when you have a raw `CUstream` handle.
    ///
    /// The memory is returned to the pool's reservoir (not the OS) and can be
    /// reused by subsequent allocations. The free is stream-ordered.
    ///
    /// # Arguments
    /// * `ptr` - Device pointer previously allocated from this pool
    /// * `stream` - Raw CUDA stream handle for async ordering
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` is a valid device pointer previously allocated from this pool
    /// - `stream` is a valid CUDA stream handle
    pub unsafe fn free_async_raw(&self, ptr: u64, stream: CUstream) -> Result<()> {
        let result = unsafe { sys::cuMemFreeAsync(ptr, stream) };

        if result != CUresult::CUDA_SUCCESS {
            return Err(anyhow!("cuMemFreeAsync failed with error: {:?}", result));
        }

        Ok(())
    }
}

impl Drop for CudaMemPool {
    fn drop(&mut self) {
        // No need to lock - we have &mut self so exclusive access is guaranteed
        let pool = self
            .inner
            .get_mut()
            .expect("mutex should not be poisoned during drop");

        // Destroy the pool, releasing all memory back to the system
        let result = unsafe { sys::cuMemPoolDestroy(*pool) };
        if result != CUresult::CUDA_SUCCESS {
            tracing::warn!("cuMemPoolDestroy failed with error: {:?}", result);
        }
    }
}

#[cfg(all(test, feature = "testing-cuda"))]
mod tests {
    use super::*;
    use cudarc::driver::{CudaEvent, CudaSlice, DevicePtr, PinnedHostSlice, result as cuda_result};
    use std::future::Future;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
    use std::time::{Duration, Instant};

    const WAIT: Duration = Duration::from_secs(5);
    const BYTES: usize = 1024 * 1024;
    const PROGRESS: Duration = Duration::from_secs(1);

    fn fail_stop(error: impl std::fmt::Debug) -> ! {
        eprintln!("GPU completion/ownership is unproven; stopping test process: {error:?}");
        std::process::abort();
    }

    fn native<T, E: std::fmt::Debug>(result: std::result::Result<T, E>) -> T {
        result.unwrap_or_else(|error| fail_stop(error))
    }

    fn join_finished<T>(finished: &Receiver<()>, worker: std::thread::JoinHandle<T>) -> T {
        if let Err(error) = finished.recv_timeout(WAIT) {
            fail_stop(error);
        }
        let deadline = Instant::now() + WAIT;
        while !worker.is_finished() {
            if Instant::now() >= deadline {
                fail_stop("GPU worker did not finish native teardown");
            }
            std::thread::yield_now();
        }
        match worker.join() {
            Ok(result) => result,
            Err(_) => fail_stop("GPU worker panicked"),
        }
    }

    struct StreamGate {
        release: Option<SyncSender<()>>,
        failed: Arc<AtomicBool>,
    }

    impl StreamGate {
        fn new(stream: &Arc<CudaStream>) -> Self {
            unsafe extern "C" fn wait(data: *mut std::ffi::c_void) {
                // The callback owns this box and must never call CUDA or unwind.
                let payload =
                    unsafe { Box::from_raw(data.cast::<(Receiver<()>, Arc<AtomicBool>)>()) };
                if payload.0.recv_timeout(WAIT).is_err() {
                    payload.1.store(true, Ordering::Release);
                }
            }
            native(stream.context().bind_to_thread());
            let (release, receiver) = sync_channel(1);
            let failed = Arc::new(AtomicBool::new(false));
            let payload = Box::into_raw(Box::new((receiver, failed.clone())));
            let result = unsafe {
                cuda_result::stream::launch_host_function(stream.cu_stream(), wait, payload.cast())
            };
            if let Err(error) = result {
                // An earlier asynchronous error does not prove the callback was not queued.
                // Do not free its argument while ownership is uncertain.
                fail_stop(error);
            }
            Self {
                release: Some(release),
                failed,
            }
        }

        fn release(&mut self) {
            if let Some(sender) = self.release.take() {
                let _ = sender.try_send(());
            }
        }

        fn assert_pending(&mut self, event: &CudaEvent) {
            let pending = match event.try_is_complete() {
                Ok(complete) => !complete,
                Err(error) => {
                    self.release();
                    fail_stop(error);
                }
            };
            if !pending {
                self.release();
            }
            assert!(pending, "copy must actually be pending behind the gate");
        }

        fn assert_released(&self) {
            assert!(
                !self.failed.load(Ordering::Acquire),
                "host-function gate timed out/disconnected"
            );
        }
    }

    impl Drop for StreamGate {
        fn drop(&mut self) {
            self.release();
        }
    }

    struct PoolBuffers<'a> {
        pool: &'a CudaMemPool,
        streams: [Arc<CudaStream>; 2],
        pointers: [Option<u64>; 2],
    }

    impl PoolBuffers<'_> {
        fn release(&mut self) -> Result<()> {
            for stream in &self.streams {
                stream.synchronize()?;
            }
            for slot in &mut self.pointers {
                if let Some(pointer) = *slot {
                    self.pool.free_async(pointer, &self.streams[0])?;
                    *slot = None;
                }
            }
            self.streams[0].synchronize()?;
            Ok(())
        }
    }

    impl Drop for PoolBuffers<'_> {
        fn drop(&mut self) {
            if let Err(error) = self.release() {
                fail_stop(error);
            }
        }
    }

    struct CopyOwners {
        stream: Arc<CudaStream>,
        fence: Option<CudaEvent>,
        _device: CudaSlice<u8>,
        _host: PinnedHostSlice<u8>,
        entered: Option<SyncSender<()>>,
        completed: Arc<AtomicBool>,
    }

    impl Drop for CopyOwners {
        fn drop(&mut self) {
            if let Some(entered) = self.entered.take() {
                let _ = entered.try_send(());
            }
            // Installed before submission; a pre-fence failure still drains the real stream.
            let result = match &self.fence {
                Some(fence) => fence.synchronize(),
                None => self.stream.synchronize(),
            };
            if let Err(error) = result {
                // Do not let automatic field destruction free an unproven live source.
                fail_stop(error);
            }
            self.completed.store(true, Ordering::Release);
        }
    }

    #[test]
    fn test_pool_creation_with_builder() {
        let context = CudaContext::new(0).unwrap();
        let pool = CudaMemPool::builder(context.clone(), BYTES)
            .release_threshold(64 * 1024 * 1024)
            .build()
            .unwrap();
        drop(pool);
        context.check_err().unwrap();
    }

    #[test]
    fn test_pool_creation_no_threshold() {
        let context = CudaContext::new(0).unwrap();
        let pool = CudaMemPool::builder(context.clone(), 0).build().unwrap();
        drop(pool);
        context.check_err().unwrap();
    }

    #[test]
    fn test_pool_async_roundtrip_orders_free_and_reuse() {
        let context = CudaContext::new(0).unwrap();
        {
            let upload = context.default_stream();
            let download = context.new_stream().unwrap();
            let pool = CudaMemPool::builder(context.clone(), 0).build().unwrap();
            let mut input = unsafe { context.alloc_pinned::<u8>(BYTES) }.unwrap();
            let mut output = unsafe { context.alloc_pinned::<u8>(BYTES) }.unwrap();
            let mut buffers = PoolBuffers {
                pool: &pool,
                streams: [upload.clone(), download.clone()],
                pointers: [None, None],
            };
            for seed in [19_u8, 173] {
                let expected: Vec<_> = (0..BYTES).map(|i| (i as u8).wrapping_add(seed)).collect();
                input.as_mut_slice().unwrap().copy_from_slice(&expected);
                for slot in &mut buffers.pointers {
                    *slot = Some(pool.alloc_async(BYTES, &upload).unwrap());
                }
                let [Some(source), Some(destination)] = buffers.pointers else {
                    unreachable!()
                };
                unsafe {
                    cuda_result::memcpy_htod_async(
                        source,
                        input.as_slice().unwrap(),
                        upload.cu_stream(),
                    )
                    .unwrap();
                }
                let ready = upload.record_event(None).unwrap();
                download.wait(&ready).unwrap();
                unsafe {
                    cuda_result::memcpy_dtod_async(
                        destination,
                        source,
                        BYTES,
                        download.cu_stream(),
                    )
                    .unwrap();
                    cuda_result::memcpy_dtoh_async(
                        output.as_mut_slice().unwrap(),
                        destination,
                        download.cu_stream(),
                    )
                    .unwrap();
                }
                let done = download.record_event(None).unwrap();
                done.synchronize().unwrap();
                assert_eq!(output.as_slice().unwrap(), expected.as_slice());
                buffers.release().unwrap();
            }
            let raw_default = upload.cu_stream();
            drop(buffers);
            drop(upload);
            unsafe { cuda_result::stream::synchronize(raw_default) }.unwrap();
        }
        context.check_err().unwrap();
    }

    #[test]
    fn test_cudarc_managed_source_drop_waits_for_last_reader() {
        let context = CudaContext::new(0).unwrap();
        let async_allocation = context.has_async_alloc();
        eprintln!("actual cudarc async allocation mode: {async_allocation}");
        assert!(
            async_allocation,
            "unsupported prerequisite for pending-reader/async-free pressure witness; not a synchronous-Drop regression"
        );
        {
            let upload = native(context.new_stream());
            let consumer = native(context.new_stream());
            let expected: Vec<_> = (0..BYTES).map(|i| (i as u8).wrapping_mul(37)).collect();
            let source = native(upload.clone_htod(&expected));
            let (source_address, record) = source.device_ptr(&upload);
            drop(record);
            let mut destination = native(consumer.alloc_zeros::<u8>(BYTES));
            native(upload.synchronize());
            native(consumer.synchronize());
            let mut gate = StreamGate::new(&consumer);
            native(consumer.memcpy_dtod(&source, &mut destination));
            let done = native(consumer.record_event(None));
            gate.assert_pending(&done);
            let (queued, queuing) = sync_channel(1);
            let (finished, finishing) = sync_channel(1);
            let pressure_stream = upload.clone();
            let owner_context = context.clone();
            let worker = std::thread::spawn(move || {
                drop(source);
                native(owner_context.check_err());
                let mut pressure = Vec::new();
                let mut addresses = Vec::new();
                for _ in 0..4 {
                    let allocation = native(pressure_stream.alloc_zeros::<u8>(BYTES));
                    let (address, record) = allocation.device_ptr(&pressure_stream);
                    drop(record);
                    addresses.push(address);
                    pressure.push(allocation);
                }
                // The acknowledgement precedes the host fence, not the queued allocation/copy.
                queued.send(addresses).unwrap();
                native(pressure_stream.synchronize());
                drop(pressure);
                drop(pressure_stream);
                native(owner_context.check_err());
                drop(owner_context);
                finished.send(()).unwrap();
            });
            let queued = queuing.recv_timeout(PROGRESS);
            let pending = done.try_is_complete();
            gate.release();
            let pending = native(pending);
            join_finished(&finishing, worker);
            let addresses = queued.expect("no async-drop/reuse witness before release; qualification incomplete, not proof of a Drop bug");
            assert!(
                addresses.contains(&source_address),
                "allocator did not reuse the source address; reuse sensitivity is unqualified"
            );
            assert!(!pending, "reader completed before the drop/reuse witness");
            native(done.synchronize());
            assert_eq!(native(consumer.clone_dtoh(&destination)), expected);
            gate.assert_released();
        }
        {
            let stream = native(context.new_stream());
            let expected = vec![91_u8; BYTES];
            let mut source = native(unsafe { context.alloc_pinned::<u8>(BYTES) });
            native(source.as_mut_slice()).copy_from_slice(&expected);
            let mut destination = native(stream.alloc_zeros::<u8>(BYTES));
            native(stream.synchronize());
            let mut gate = StreamGate::new(&stream);
            native(stream.memcpy_htod(&source, &mut destination));
            let done = native(stream.record_event(None));
            gate.assert_pending(&done);
            let (requested, requesting) = sync_channel(1);
            let (finished, finishing) = sync_channel(1);
            let owner_context = context.clone();
            let worker = std::thread::spawn(move || {
                requested.send(()).unwrap();
                drop(source);
                native(owner_context.check_err());
                drop(owner_context);
                finished.send(()).unwrap();
            });
            let requested = requesting.recv_timeout(PROGRESS);
            if requested.is_err() {
                gate.release();
                join_finished(&finishing, worker);
                panic!("pinned source drop was not requested in the bounded window");
            }
            // A returned Drop while this actual copy is blocked proves premature retirement.
            match finishing.recv_timeout(PROGRESS) {
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                other => fail_stop(("pinned source Drop returned before its copy fence", other)),
            }
            let pending = done.try_is_complete();
            gate.release();
            let pending = native(pending);
            join_finished(&finishing, worker);
            assert!(!pending);
            native(done.synchronize());
            assert_eq!(native(stream.clone_dtoh(&destination)), expected);
            gate.assert_released();
        }
        native(context.check_err());
    }

    #[test]
    fn test_cudarc_explicit_fences_survive_cancelled_waiter() {
        let context = CudaContext::new(0).unwrap();
        unsafe { context.disable_event_tracking() };
        for cancel_future in [false, true] {
            let stream = native(context.new_stream());
            let expected = vec![203_u8; BYTES];
            let mut source = native(unsafe { context.alloc_pinned::<u8>(BYTES) });
            native(source.as_mut_slice()).copy_from_slice(&expected);
            let device_source = native(stream.alloc_zeros::<u8>(BYTES));
            let mut destination = native(stream.alloc_zeros::<u8>(BYTES));
            native(stream.synchronize());
            let completed = Arc::new(AtomicBool::new(false));
            let (entered, entering) = sync_channel(1);
            let mut owners = CopyOwners {
                stream: stream.clone(),
                fence: None,
                _device: device_source,
                _host: source,
                entered: Some(entered),
                completed: completed.clone(),
            };
            // This guard is declared last, so unwind releases the gate before owners drain.
            let mut gate = StreamGate::new(&stream);
            native(stream.memcpy_htod(&owners._host, &mut owners._device));
            native(stream.memcpy_dtod(&owners._device, &mut destination));
            owners.fence = Some(native(stream.record_event(None)));
            let observer = native(stream.record_event(None));
            gate.assert_pending(&observer);
            let (finished, finishing) = sync_channel(1);
            let worker = std::thread::spawn(move || {
                if cancel_future {
                    let mut future = Box::pin(async move {
                        let _owners = owners;
                        std::future::pending::<()>().await;
                    });
                    let mut task = std::task::Context::from_waker(std::task::Waker::noop());
                    assert!(future.as_mut().poll(&mut task).is_pending());
                    drop(future);
                } else {
                    let fail_after_submission = || -> Result<()> {
                        let _owners = owners;
                        Err(anyhow!("intentional early return after submission"))
                    };
                    assert!(fail_after_submission().is_err());
                }
                finished.send(()).unwrap();
            });
            let entered = entering.recv_timeout(PROGRESS);
            let pending = observer.try_is_complete();
            // Sample before release; assert only after the worker has safely drained.
            let completed_before_release = completed.load(Ordering::Acquire);
            gate.release();
            let pending = native(pending);
            join_finished(&finishing, worker);
            assert!(
                entered.is_ok(),
                "caller-owned cancellation/error destructor was not entered"
            );
            assert!(!pending);
            assert!(
                !completed_before_release,
                "caller-owned destructor reported fence completion while copy was pending"
            );
            assert!(
                completed.load(Ordering::Acquire),
                "cancelled/error path did not complete its fence"
            );
            assert_eq!(native(stream.clone_dtoh(&destination)), expected);
            gate.assert_released();
        }
        native(context.check_err());
    }
}
