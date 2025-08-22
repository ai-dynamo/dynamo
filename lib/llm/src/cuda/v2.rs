// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// TODO: enable when https://github.com/rust-lang/rust/issues/83310 becomes stable
// #![deny(must_not_suspend)]

use core::future::Future;
use std::{marker::PhantomData, ptr::NonNull};

use super::{check_cuda, CudaContext, DriverError};

use ::cudarc::driver::sys::{
    cuCtxPopCurrent_v2, cuCtxPushCurrent_v2, cuDeviceGet, cuDevicePrimaryCtxRelease_v2,
    cuDevicePrimaryCtxRetain, CUctx_st, CUdevice, CUstream_st,
};

pub type CUcontext = NonNull<CUctx_st>;
pub type CUstream = NonNull<CUstream_st>;

pub struct PrimaryCtx {
    dev: CUdevice,
    ctx: NonNull<CUctx_st>,
}
impl PrimaryCtx {
    pub fn retain(device_ordinal: i32) -> Result<Self, DriverError> {
        unsafe {
            let mut dev = 0;
            check_cuda(cuDeviceGet(&mut dev, device_ordinal))?;
            let mut ctx = std::ptr::null_mut();
            check_cuda(cuDevicePrimaryCtxRetain(&mut ctx, dev))?;
            Ok(Self {
                dev,
                ctx: NonNull::new(ctx).expect("PrimaryCtx returned null context"),
            })
        }
    }
}

impl CudaContext for PrimaryCtx {
    #[inline]
    unsafe fn cu_context(&self) -> NonNull<CUctx_st> {
        self.ctx
    }
}

impl Drop for PrimaryCtx {
    fn drop(&mut self) {
        unsafe {
            let _ = cuDevicePrimaryCtxRelease_v2(self.dev);
        }
    }
}

struct PushGuard {
    _ns: std::rc::Rc<()>,
} // !Send/!Sync
impl PushGuard {
    fn push(ctx: &dyn CudaContext) -> Result<Self, DriverError> {
        unsafe {
            check_cuda(cuCtxPushCurrent_v2(ctx.cu_context().as_ptr()))?;
        }
        Ok(Self {
            _ns: std::rc::Rc::new(()),
        })
    }
}

impl Drop for PushGuard {
    fn drop(&mut self) {
        unsafe {
            let mut popped = std::ptr::null_mut();
            let _ = cuCtxPopCurrent_v2(&mut popped);
        }
    }
}

// TODO: enable when https://github.com/rust-lang/rust/issues/83310 becomes stable
// #[must_not_suspend = "Do not hold CUDA context token across .await; use enter_async_build()"]
#[derive(Clone, Copy)]
pub struct CudaCtxEntered<'a> {
    _l: PhantomData<&'a ()>,
    _ns: PhantomData<std::rc::Rc<()>>,
}
impl<'a> CudaCtxEntered<'a> {
    fn new(_g: &'a PushGuard) -> Self {
        Self {
            _l: PhantomData,
            _ns: PhantomData,
        }
    }
}

#[inline]
pub fn enter<R>(
    primary: &dyn CudaContext,
    f: impl FnOnce(CudaCtxEntered<'_>) -> R,
) -> Result<R, DriverError> {
    let guard = PushGuard::push(primary)?;
    let tok = CudaCtxEntered::new(&guard);
    let out = f(tok);
    drop(guard);
    Ok(out)
}

pub trait Resource {
    // What you get once you “unlock” the resource with `tok`.
    type Unlocked<'c>
    where
        Self: 'c;

    /// Produce a view/handle that is usable only while `tok` is in scope.
    fn unlock<'c>(&'c self, tok: CudaCtxEntered<'c>) -> Self::Unlocked<'c>;
}

/// A pack of resources: something that can be “unlocked” as a whole.
pub trait ResourcePack {
    type Unlocked<'c>
    where
        Self: 'c;

    /// Unlock *all* resources inside using the same token.
    fn unlock_all<'c>(&'c self, tok: CudaCtxEntered<'c>) -> Self::Unlocked<'c>;
}

pub struct OwnedStream {
    pub raw: CUstream,
}

impl Resource for OwnedStream {
    type Unlocked<'c> = Stream<'c>;

    fn unlock<'c>(&'c self, _tok: CudaCtxEntered<'c>) -> Self::Unlocked<'c> {
        Stream {
            raw: self.raw,
            _pd: PhantomData,
        }
    }
}

pub struct Stream<'c> {
    pub raw: CUstream,
    _pd: PhantomData<&'c ()>,
}

// The oneshot-backed future that *holds* your borrows:
#[must_use = "await to keep borrow_pack alive until GPU work completes"]
pub struct ScopedRx<'a, T, P> {
    rx: oneshot::Receiver<T>, // from the `oneshot` crate (not Tokio)
    _hold: P,                 // owns the borrow_pack => ties lifetimes
    _pd: core::marker::PhantomData<&'a ()>,
}
impl<'a, T, P> core::future::Future for ScopedRx<'a, T, P>
where
    oneshot::Receiver<T>: core::future::Future<Output = T> + Unpin,
{
    type Output = T;
    fn poll(
        self: core::pin::Pin<&mut Self>,
        cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<Self::Output> {
        // SAFETY: We're implementing Future, so this is safe to call
        let this = unsafe { self.get_unchecked_mut() };
        core::pin::Pin::new(&mut this.rx).poll(cx)
    }
}

impl<'a, T, P> ScopedRx<'a, T, P> {
    #[inline]
    fn new(rx: oneshot::Receiver<T>, pack: P) -> Self {
        Self {
            rx,
            _hold: pack,
            _pd: core::marker::PhantomData,
        }
    }
}
