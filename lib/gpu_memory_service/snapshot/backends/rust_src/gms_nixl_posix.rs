// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal in-process NIXL POSIX helper for GMS snapshot restore.
//!
//! The Python restore path owns pinned host buffers and CUDA H2D streams, so this
//! helper must be a shared library loaded into the Python process.  It avoids the
//! Python `nixl._api` import path and uses NIXL's C API via `dlopen`.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::ptr;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

const RTLD_NOW: c_int = 2;
const RTLD_GLOBAL: c_int = 0x100;

const NIXL_SUCCESS: c_int = 0;
const NIXL_IN_PROG: c_int = 1;
const NIXL_MEM_DRAM: c_int = 0;
const NIXL_MEM_FILE: c_int = 4;
const NIXL_XFER_OP_READ: c_int = 0;

#[link(name = "dl")]
extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

type Status = c_int;
type Agent = *mut c_void;
type Params = *mut c_void;
type Backend = *mut c_void;
type OptArgs = *mut c_void;
type RegDList = *mut c_void;
type XferDList = *mut c_void;
type XferReq = *mut c_void;

type CreateAgent = unsafe extern "C" fn(*const c_char, *mut Agent) -> Status;
type DestroyAgent = unsafe extern "C" fn(Agent) -> Status;
type CreateParams = unsafe extern "C" fn(*mut Params) -> Status;
type ParamsAdd = unsafe extern "C" fn(Params, *const c_char, *const c_char) -> Status;
type DestroyParams = unsafe extern "C" fn(Params) -> Status;
type CreateBackend = unsafe extern "C" fn(Agent, *const c_char, Params, *mut Backend) -> Status;
type DestroyBackend = unsafe extern "C" fn(Backend) -> Status;
type CreateOptArgs = unsafe extern "C" fn(*mut OptArgs) -> Status;
type DestroyOptArgs = unsafe extern "C" fn(OptArgs) -> Status;
type OptArgsAddBackend = unsafe extern "C" fn(OptArgs, Backend) -> Status;
type CreateRegDList = unsafe extern "C" fn(c_int, *mut RegDList) -> Status;
type DestroyRegDList = unsafe extern "C" fn(RegDList) -> Status;
type RegDListAddDesc =
    unsafe extern "C" fn(RegDList, usize, usize, u64, *const c_void, usize) -> Status;
type RegDListTrim = unsafe extern "C" fn(RegDList) -> Status;
type RegisterMem = unsafe extern "C" fn(Agent, RegDList, *mut c_void) -> Status;
type DeregisterMem = unsafe extern "C" fn(Agent, RegDList, *mut c_void) -> Status;
type CreateXferDList = unsafe extern "C" fn(c_int, *mut XferDList) -> Status;
type DestroyXferDList = unsafe extern "C" fn(XferDList) -> Status;
type XferDListAddDesc = unsafe extern "C" fn(XferDList, usize, usize, u64) -> Status;
type XferDListTrim = unsafe extern "C" fn(XferDList) -> Status;
type CreateXferReq = unsafe extern "C" fn(
    Agent,
    c_int,
    XferDList,
    XferDList,
    *const c_char,
    *mut XferReq,
    *mut c_void,
) -> Status;
type PostXferReq = unsafe extern "C" fn(Agent, XferReq, *mut c_void) -> Status;
type GetXferStatus = unsafe extern "C" fn(Agent, XferReq) -> Status;
type ReleaseXferReq = unsafe extern "C" fn(Agent, XferReq) -> Status;
type DestroyXferReq = unsafe extern "C" fn(XferReq) -> Status;

#[derive(Clone, Copy)]
struct NixlFns {
    create_agent: CreateAgent,
    destroy_agent: DestroyAgent,
    create_params: CreateParams,
    params_add: ParamsAdd,
    destroy_params: DestroyParams,
    create_backend: CreateBackend,
    destroy_backend: DestroyBackend,
    create_opt_args: CreateOptArgs,
    destroy_opt_args: DestroyOptArgs,
    opt_args_add_backend: OptArgsAddBackend,
    create_reg_dlist: CreateRegDList,
    destroy_reg_dlist: DestroyRegDList,
    reg_dlist_add_desc: RegDListAddDesc,
    reg_dlist_trim: RegDListTrim,
    register_mem: RegisterMem,
    deregister_mem: DeregisterMem,
    create_xfer_dlist: CreateXferDList,
    destroy_xfer_dlist: DestroyXferDList,
    xfer_dlist_add_desc: XferDListAddDesc,
    xfer_dlist_trim: XferDListTrim,
    create_xfer_req: CreateXferReq,
    post_xfer_req: PostXferReq,
    get_xfer_status: GetXferStatus,
    release_xfer_req: ReleaseXferReq,
    destroy_xfer_req: DestroyXferReq,
}

static NIXL: OnceLock<NixlFns> = OnceLock::new();

unsafe fn symbol<T: Copy>(handle: *mut c_void, name: &str) -> Result<T, String> {
    let cname = CString::new(name).map_err(|_| format!("invalid symbol {name}"))?;
    let ptr = dlsym(handle, cname.as_ptr());
    if ptr.is_null() {
        return Err(format!("dlsym({name}) failed: {}", dlerror_string()));
    }
    Ok(std::mem::transmute_copy(&ptr))
}

fn dlerror_string() -> String {
    unsafe {
        let err = dlerror();
        if err.is_null() {
            "unknown dynamic linker error".to_string()
        } else {
            CStr::from_ptr(err).to_string_lossy().into_owned()
        }
    }
}

fn load_nixl() -> Result<NixlFns, String> {
    let lib = CString::new("libnixl_capi.so").unwrap();
    unsafe {
        let handle = dlopen(lib.as_ptr(), RTLD_NOW | RTLD_GLOBAL);
        if handle.is_null() {
            return Err(format!(
                "dlopen(libnixl_capi.so) failed: {}",
                dlerror_string()
            ));
        }
        Ok(NixlFns {
            create_agent: symbol(handle, "nixl_capi_create_agent")?,
            destroy_agent: symbol(handle, "nixl_capi_destroy_agent")?,
            create_params: symbol(handle, "nixl_capi_create_params")?,
            params_add: symbol(handle, "nixl_capi_params_add")?,
            destroy_params: symbol(handle, "nixl_capi_destroy_params")?,
            create_backend: symbol(handle, "nixl_capi_create_backend")?,
            destroy_backend: symbol(handle, "nixl_capi_destroy_backend")?,
            create_opt_args: symbol(handle, "nixl_capi_create_opt_args")?,
            destroy_opt_args: symbol(handle, "nixl_capi_destroy_opt_args")?,
            opt_args_add_backend: symbol(handle, "nixl_capi_opt_args_add_backend")?,
            create_reg_dlist: symbol(handle, "nixl_capi_create_reg_dlist")?,
            destroy_reg_dlist: symbol(handle, "nixl_capi_destroy_reg_dlist")?,
            reg_dlist_add_desc: symbol(handle, "nixl_capi_reg_dlist_add_desc")?,
            reg_dlist_trim: symbol(handle, "nixl_capi_reg_dlist_trim")?,
            register_mem: symbol(handle, "nixl_capi_register_mem")?,
            deregister_mem: symbol(handle, "nixl_capi_deregister_mem")?,
            create_xfer_dlist: symbol(handle, "nixl_capi_create_xfer_dlist")?,
            destroy_xfer_dlist: symbol(handle, "nixl_capi_destroy_xfer_dlist")?,
            xfer_dlist_add_desc: symbol(handle, "nixl_capi_xfer_dlist_add_desc")?,
            xfer_dlist_trim: symbol(handle, "nixl_capi_xfer_dlist_trim")?,
            create_xfer_req: symbol(handle, "nixl_capi_create_xfer_req")?,
            post_xfer_req: symbol(handle, "nixl_capi_post_xfer_req")?,
            get_xfer_status: symbol(handle, "nixl_capi_get_xfer_status")?,
            release_xfer_req: symbol(handle, "nixl_capi_release_xfer_req")?,
            destroy_xfer_req: symbol(handle, "nixl_capi_destroy_xfer_req")?,
        })
    }
}

fn fns() -> Result<&'static NixlFns, String> {
    if let Some(fns) = NIXL.get() {
        return Ok(fns);
    }
    let loaded = load_nixl()?;
    let _ = NIXL.set(loaded);
    Ok(NIXL.get().expect("NIXL OnceLock just initialized"))
}

fn check(status: Status, what: &str) -> Result<(), String> {
    if status == NIXL_SUCCESS {
        Ok(())
    } else {
        Err(format!("{what} failed with status {status}"))
    }
}

fn copy_error(error: *mut c_char, error_len: usize, msg: &str) {
    if error.is_null() || error_len == 0 {
        return;
    }
    let bytes = msg.as_bytes();
    let n = bytes.len().min(error_len.saturating_sub(1));
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), error.cast::<u8>(), n);
        *error.add(n) = 0;
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GmsNixlPosixContextStats {
    pub dlopen_s: f64,
    pub create_agent_backend_s: f64,
    pub total_s: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GmsNixlPosixReadStats {
    pub register_file_s: f64,
    pub register_host_s: f64,
    pub create_req_s: f64,
    pub transfer_s: f64,
    pub cleanup_s: f64,
    pub total_s: f64,
}

pub struct GmsNixlPosixContext {
    f: &'static NixlFns,
    agent: Agent,
    backend: Backend,
    opt_args: OptArgs,
    params: Params,
}

impl GmsNixlPosixContext {
    fn new(
        agent_name: *const c_char,
        param_keys: *const *const c_char,
        param_values: *const *const c_char,
        param_count: usize,
        stats: &mut GmsNixlPosixContextStats,
    ) -> Result<Self, String> {
        let total_t0 = Instant::now();
        let dlopen_t0 = Instant::now();
        let f = fns()?;
        stats.dlopen_s = dlopen_t0.elapsed().as_secs_f64();

        let setup_t0 = Instant::now();
        let backend_name = CString::new("POSIX").unwrap();
        let mut ctx = Self {
            f,
            agent: ptr::null_mut(),
            backend: ptr::null_mut(),
            opt_args: ptr::null_mut(),
            params: ptr::null_mut(),
        };
        unsafe {
            check((f.create_agent)(agent_name, &mut ctx.agent), "create_agent")?;
            check((f.create_params)(&mut ctx.params), "create_params")?;
            if param_count > 0 {
                if param_keys.is_null() || param_values.is_null() {
                    return Err("backend param pointer is null".to_string());
                }
                for index in 0..param_count {
                    let key = *param_keys.add(index);
                    let value = *param_values.add(index);
                    if key.is_null() || value.is_null() {
                        return Err(format!("backend param {index} is null"));
                    }
                    check((f.params_add)(ctx.params, key, value), "params_add")?;
                }
            } else {
                let key_ios = CString::new("ios_pool_size").unwrap();
                let val_ios = CString::new("1024").unwrap();
                let key_queue = CString::new("kernel_queue_size").unwrap();
                let val_queue = CString::new("128").unwrap();
                check(
                    (f.params_add)(ctx.params, key_ios.as_ptr(), val_ios.as_ptr()),
                    "params_add(ios_pool_size)",
                )?;
                check(
                    (f.params_add)(ctx.params, key_queue.as_ptr(), val_queue.as_ptr()),
                    "params_add(kernel_queue_size)",
                )?;
            }
            check(
                (f.create_backend)(
                    ctx.agent,
                    backend_name.as_ptr(),
                    ctx.params,
                    &mut ctx.backend,
                ),
                "create_backend(POSIX)",
            )?;
            check((f.create_opt_args)(&mut ctx.opt_args), "create_opt_args")?;
            check(
                (f.opt_args_add_backend)(ctx.opt_args, ctx.backend),
                "opt_args_add_backend(POSIX)",
            )?;
        }
        stats.create_agent_backend_s = setup_t0.elapsed().as_secs_f64();
        stats.total_s = total_t0.elapsed().as_secs_f64();
        Ok(ctx)
    }
}

impl Drop for GmsNixlPosixContext {
    fn drop(&mut self) {
        unsafe {
            if !self.opt_args.is_null() {
                let _ = (self.f.destroy_opt_args)(self.opt_args);
                self.opt_args = ptr::null_mut();
            }
            if !self.backend.is_null() {
                let _ = (self.f.destroy_backend)(self.backend);
                self.backend = ptr::null_mut();
            }
            if !self.params.is_null() {
                let _ = (self.f.destroy_params)(self.params);
                self.params = ptr::null_mut();
            }
            if !self.agent.is_null() {
                let _ = (self.f.destroy_agent)(self.agent);
                self.agent = ptr::null_mut();
            }
        }
    }
}

struct ReadCleanup<'a> {
    f: &'a NixlFns,
    agent: Agent,
    opt_args: OptArgs,
    file_reg: RegDList,
    host_reg: RegDList,
    file_xfer: XferDList,
    host_xfer: XferDList,
    req: XferReq,
}

impl<'a> ReadCleanup<'a> {
    fn new(f: &'a NixlFns, agent: Agent, opt_args: OptArgs) -> Self {
        Self {
            f,
            agent,
            opt_args,
            file_reg: ptr::null_mut(),
            host_reg: ptr::null_mut(),
            file_xfer: ptr::null_mut(),
            host_xfer: ptr::null_mut(),
            req: ptr::null_mut(),
        }
    }

    unsafe fn cleanup(&mut self) {
        if !self.req.is_null() {
            let _ = (self.f.release_xfer_req)(self.agent, self.req);
            let _ = (self.f.destroy_xfer_req)(self.req);
            self.req = ptr::null_mut();
        }
        if !self.host_reg.is_null() {
            let _ = (self.f.deregister_mem)(self.agent, self.host_reg, self.opt_args);
        }
        if !self.file_reg.is_null() {
            let _ = (self.f.deregister_mem)(self.agent, self.file_reg, self.opt_args);
        }
        if !self.host_xfer.is_null() {
            let _ = (self.f.destroy_xfer_dlist)(self.host_xfer);
            self.host_xfer = ptr::null_mut();
        }
        if !self.file_xfer.is_null() {
            let _ = (self.f.destroy_xfer_dlist)(self.file_xfer);
            self.file_xfer = ptr::null_mut();
        }
        if !self.host_reg.is_null() {
            let _ = (self.f.destroy_reg_dlist)(self.host_reg);
            self.host_reg = ptr::null_mut();
        }
        if !self.file_reg.is_null() {
            let _ = (self.f.destroy_reg_dlist)(self.file_reg);
            self.file_reg = ptr::null_mut();
        }
    }
}

impl Drop for ReadCleanup<'_> {
    fn drop(&mut self) {
        unsafe { self.cleanup() };
    }
}

#[no_mangle]
pub extern "C" fn gms_nixl_posix_context_create(
    agent_name: *const c_char,
    param_keys: *const *const c_char,
    param_values: *const *const c_char,
    param_count: usize,
    context: *mut *mut GmsNixlPosixContext,
    stats: *mut GmsNixlPosixContextStats,
    error: *mut c_char,
    error_len: usize,
) -> c_int {
    if context.is_null() {
        copy_error(error, error_len, "context output pointer is null");
        return -1;
    }
    if agent_name.is_null() {
        copy_error(error, error_len, "agent_name is null");
        return -1;
    }
    let mut ctx_stats = GmsNixlPosixContextStats {
        dlopen_s: 0.0,
        create_agent_backend_s: 0.0,
        total_s: 0.0,
    };
    match GmsNixlPosixContext::new(
        agent_name,
        param_keys,
        param_values,
        param_count,
        &mut ctx_stats,
    ) {
        Ok(ctx) => {
            if !stats.is_null() {
                unsafe { *stats = ctx_stats };
            }
            unsafe { *context = Box::into_raw(Box::new(ctx)) };
            0
        }
        Err(msg) => {
            copy_error(error, error_len, &msg);
            -1
        }
    }
}

#[no_mangle]
pub extern "C" fn gms_nixl_posix_context_destroy(context: *mut GmsNixlPosixContext) {
    if context.is_null() {
        return;
    }
    unsafe { drop(Box::from_raw(context)) };
}

#[no_mangle]
pub extern "C" fn gms_nixl_posix_read(
    context: *mut GmsNixlPosixContext,
    agent_name: *const c_char,
    fd: c_int,
    file_offset: u64,
    host_ptr: u64,
    size: usize,
    stats: *mut GmsNixlPosixReadStats,
    error: *mut c_char,
    error_len: usize,
) -> c_int {
    let total_t0 = Instant::now();
    let mut out = GmsNixlPosixReadStats {
        register_file_s: 0.0,
        register_host_s: 0.0,
        create_req_s: 0.0,
        transfer_s: 0.0,
        cleanup_s: 0.0,
        total_s: 0.0,
    };

    let result = (|| -> Result<(), String> {
        if context.is_null() {
            return Err("context is null".to_string());
        }
        if agent_name.is_null() {
            return Err("agent_name is null".to_string());
        }
        if fd < 0 {
            return Err(format!("invalid fd {fd}"));
        }
        if host_ptr == 0 {
            return Err("host_ptr is null".to_string());
        }
        if size == 0 {
            return Ok(());
        }

        let ctx = unsafe { &mut *context };
        let f = ctx.f;
        let mut c = ReadCleanup::new(f, ctx.agent, ctx.opt_args);
        unsafe {
            let file_reg_t0 = Instant::now();
            check(
                (f.create_reg_dlist)(NIXL_MEM_FILE, &mut c.file_reg),
                "create_file_reg_dlist",
            )?;
            check(
                (f.reg_dlist_add_desc)(
                    c.file_reg,
                    file_offset as usize,
                    size,
                    fd as u64,
                    ptr::null(),
                    0,
                ),
                "file_reg_add_desc",
            )?;
            check((f.reg_dlist_trim)(c.file_reg), "file_reg_trim")?;
            check(
                (f.register_mem)(ctx.agent, c.file_reg, ctx.opt_args),
                "register_file",
            )?;
            out.register_file_s = file_reg_t0.elapsed().as_secs_f64();

            let host_reg_t0 = Instant::now();
            check(
                (f.create_reg_dlist)(NIXL_MEM_DRAM, &mut c.host_reg),
                "create_host_reg_dlist",
            )?;
            check(
                (f.reg_dlist_add_desc)(c.host_reg, host_ptr as usize, size, 0, ptr::null(), 0),
                "host_reg_add_desc",
            )?;
            check((f.reg_dlist_trim)(c.host_reg), "host_reg_trim")?;
            check(
                (f.register_mem)(ctx.agent, c.host_reg, ctx.opt_args),
                "register_host",
            )?;
            out.register_host_s = host_reg_t0.elapsed().as_secs_f64();

            let req_t0 = Instant::now();
            check(
                (f.create_xfer_dlist)(NIXL_MEM_FILE, &mut c.file_xfer),
                "create_file_xfer_dlist",
            )?;
            check(
                (f.xfer_dlist_add_desc)(c.file_xfer, file_offset as usize, size, fd as u64),
                "file_xfer_add_desc",
            )?;
            check((f.xfer_dlist_trim)(c.file_xfer), "file_xfer_trim")?;
            check(
                (f.create_xfer_dlist)(NIXL_MEM_DRAM, &mut c.host_xfer),
                "create_host_xfer_dlist",
            )?;
            check(
                (f.xfer_dlist_add_desc)(c.host_xfer, host_ptr as usize, size, 0),
                "host_xfer_add_desc",
            )?;
            check((f.xfer_dlist_trim)(c.host_xfer), "host_xfer_trim")?;
            check(
                (f.create_xfer_req)(
                    ctx.agent,
                    NIXL_XFER_OP_READ,
                    c.host_xfer,
                    c.file_xfer,
                    agent_name,
                    &mut c.req,
                    ctx.opt_args,
                ),
                "create_xfer_req",
            )?;
            out.create_req_s = req_t0.elapsed().as_secs_f64();

            let transfer_t0 = Instant::now();
            let post_status = (f.post_xfer_req)(ctx.agent, c.req, ptr::null_mut());
            if post_status != NIXL_SUCCESS && post_status != NIXL_IN_PROG {
                return Err(format!("post_xfer_req failed with status {post_status}"));
            }
            loop {
                let status = (f.get_xfer_status)(ctx.agent, c.req);
                if status == NIXL_SUCCESS {
                    break;
                }
                if status != NIXL_IN_PROG {
                    return Err(format!("get_xfer_status failed with status {status}"));
                }
                std::thread::sleep(Duration::from_micros(1000));
            }
            out.transfer_s = transfer_t0.elapsed().as_secs_f64();

            let cleanup_t0 = Instant::now();
            c.cleanup();
            out.cleanup_s = cleanup_t0.elapsed().as_secs_f64();
        }
        Ok(())
    })();

    out.total_s = total_t0.elapsed().as_secs_f64();
    if !stats.is_null() {
        unsafe { *stats = out };
    }
    match result {
        Ok(()) => 0,
        Err(msg) => {
            copy_error(error, error_len, &msg);
            -1
        }
    }
}
