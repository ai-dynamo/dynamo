/*
 * CUDA Intercept Library
 *
 * This library intercepts CUDA calls and returns appropriate error codes
 * to simulate various GPU failures (XIDs). Supports BOTH:
 *   - CUDA Runtime API (cuda* functions) - used by some applications
 *   - CUDA Driver API (cu* functions) - used by PyTorch, vLLM, etc.
 *
 * Supported XID types (set via CUDA_XID_TYPE environment variable):
 *   79  - GPU fell off bus (CUDA_ERROR_NO_DEVICE) - DEFAULT
 *   48  - Double-bit ECC error (CUDA_ERROR_ECC_UNCORRECTABLE)
 *   94  - Contained ECC error (CUDA_ERROR_ECC_UNCORRECTABLE)
 *   95  - Uncontained error (CUDA_ERROR_UNKNOWN)
 *   43  - GPU stopped responding (CUDA_ERROR_LAUNCH_TIMEOUT)
 *   74  - NVLink error (CUDA_ERROR_PEER_ACCESS_UNSUPPORTED)
 *
 * Intercepted Functions:
 *   Runtime API: cudaGetDeviceCount, cudaMalloc, cudaMemcpy, cudaLaunchKernel, etc.
 *   Driver API:  cuInit, cuDeviceGetCount, cuCtxCreate, cuMemAlloc, cuLaunchKernel, etc.
 *
 * Compile:
 *   gcc -shared -fPIC -ldl cuda_intercept.c -o cuda_intercept.so
 *
 * Use:
 *   export CUDA_FAULT_INJECTION_ENABLED=1
 *   export CUDA_XID_TYPE=79  # or 48, 94, 95, 43, 74
 *   LD_PRELOAD=/path/to/cuda_intercept.so python -m vllm.entrypoints.api_server
 *
 * Node Affinity Modes (when deploying via inject_into_pods.py):
 *   SOFT (default): preferredDuringScheduling - pods PREFER target node but can
 *                   spill to other nodes if target lacks GPU capacity. Useful for
 *                   testing partial failures (e.g., 4 pods on faulty node, 4 healthy).
 *   HARD (--hard-affinity): requiredDuringScheduling - ALL pods MUST fit on target
 *                   node or they stay Pending. Use only when node has enough GPUs.
 */

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int cudaError_t;
typedef void* cudaStream_t;

// dim3 structure for kernel launch dimensions
typedef struct {
  unsigned int x, y, z;
} dim3;

typedef struct cudaDeviceProp_st {
  char name[256];
  size_t totalGlobalMem;
  // ... other fields (we don't need them)
} cudaDeviceProp;

// CUDA error codes (from cuda_runtime_api.h)
#define cudaSuccess 0
#define cudaErrorNoDevice 100               // XID 79: GPU fell off bus
#define cudaErrorEccUncorrectable 214       // XID 48, 94: ECC errors
#define cudaErrorUnknown 999                // XID 95: Uncontained error
#define cudaErrorLaunchTimeout 6            // XID 43: GPU stopped responding
#define cudaErrorPeerAccessUnsupported 217  // XID 74: NVLink error

// XID error type mapping
typedef struct {
  int xid;
  cudaError_t cuda_error;
  const char* description;
} xid_mapping_t;

static const xid_mapping_t xid_mappings[] = {
    {79, cudaErrorNoDevice, "GPU fell off bus"},
    {48, cudaErrorEccUncorrectable, "Double-bit ECC error"},
    {94, cudaErrorEccUncorrectable, "Contained ECC error"},
    {95, cudaErrorUnknown, "Uncontained error"},
    {43, cudaErrorLaunchTimeout, "GPU stopped responding"},
    {74, cudaErrorPeerAccessUnsupported, "NVLink error"},
    {0, 0, NULL}  // Sentinel
};

// Get XID type and corresponding CUDA error
// Supports runtime toggling via /tmp/cuda_fault_enabled file
static void
get_fault_config(int* inject, int* xid_type, cudaError_t* error_code)
{
  static int initialized = 0;
  static int env_inject = 0;       // From environment variable (initial state)
  static int cached_xid = 79;      // Default to XID 79
  static cudaError_t cached_error = cudaErrorNoDevice;

  if (!initialized) {
    // Check if injection is enabled via environment (sets initial state)
    // Toggle file can override this at runtime!
    char* env = getenv("CUDA_FAULT_INJECTION_ENABLED");
    if (env) {
      env_inject = (strcmp(env, "1") == 0 || strcmp(env, "true") == 0);
    }

    // Get XID type
    char* xid_env = getenv("CUDA_XID_TYPE");
    if (xid_env) {
      cached_xid = atoi(xid_env);

      // Find corresponding CUDA error
      int found = 0;
      for (int i = 0; xid_mappings[i].description != NULL; i++) {
        if (xid_mappings[i].xid == cached_xid) {
          cached_error = xid_mappings[i].cuda_error;
          fprintf(
              stderr, "[CUDA FAULT INJECTION] Library loaded - XID %d (%s)\n", cached_xid, xid_mappings[i].description);
          found = 1;
          break;
        }
      }

      if (!found) {
        fprintf(stderr, "[CUDA FAULT INJECTION] WARNING: Unknown XID %d, defaulting to XID 79\n", cached_xid);
        cached_xid = 79;
        cached_error = cudaErrorNoDevice;
      }
    }

    initialized = 1;
  }

  // Runtime toggle: Check node-persistent fault marker on EVERY call
  // Use hostPath (/host-fault) so fault persists across pod restarts on same node
  // Pod reschedules to different node → no file there → automatic recovery!
  //
  // NOTE: Toggle file ALWAYS overrides env var! This allows:
  //   - Passthrough mode: env=0, then toggle file enables faults
  //   - Active mode: env=1, then toggle file can disable faults
  int runtime_inject = env_inject;  // Default to env var (can be 0 for passthrough)

  // Check hostPath first (persistent across restarts on same node)
  FILE* toggle_file = fopen("/host-fault/cuda_fault_enabled", "r");
  if (toggle_file) {
    char toggle_value[4] = {0};
    if (fgets(toggle_value, sizeof(toggle_value), toggle_file)) {
      runtime_inject = (toggle_value[0] == '1');
    }
    fclose(toggle_file);
  } else {
    // Fallback to ephemeral /tmp for backwards compatibility
    toggle_file = fopen("/tmp/cuda_fault_enabled", "r");
    if (toggle_file) {
      char toggle_value[4] = {0};
      if (fgets(toggle_value, sizeof(toggle_value), toggle_file)) {
        runtime_inject = (toggle_value[0] == '1');
      }
      fclose(toggle_file);
    }
  }

  *inject = runtime_inject;
  *xid_type = cached_xid;
  *error_code = cached_error;
}

// Check if fault should be injected
static int
should_inject_fault()
{
  int inject, xid;
  cudaError_t error;
  get_fault_config(&inject, &xid, &error);
  return inject;
}

// Get the error code to return
static cudaError_t
get_error_code()
{
  int inject, xid;
  cudaError_t error;
  get_fault_config(&inject, &xid, &error);
  return error;
}

// Log helper
static void
log_intercept(const char* func_name, cudaError_t error_code)
{
  if (should_inject_fault()) {
    int inject, xid;
    cudaError_t err;
    get_fault_config(&inject, &xid, &err);
    fprintf(stderr, "[XID %d SIM] %s() intercepted -> error %d\n", xid, func_name, error_code);
  }
}

// Intercept: Get device count
cudaError_t
cudaGetDeviceCount(int* count)
{
  if (should_inject_fault()) {
    cudaError_t error = get_error_code();
    log_intercept("cudaGetDeviceCount", error);
    if (count)
      *count = 0;
    return error;
  }

  // If disabled, call real function
  typedef cudaError_t (*real_func_t)(int*);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cudaGetDeviceCount");
  if (real_func) {
    return real_func(count);
  }
  return cudaErrorNoDevice;
}

// Intercept: Set device
cudaError_t
cudaSetDevice(int device)
{
  if (should_inject_fault()) {
    cudaError_t error = get_error_code();
    log_intercept("cudaSetDevice", error);
    return error;
  }

  typedef cudaError_t (*real_func_t)(int);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cudaSetDevice");
  if (real_func) {
    return real_func(device);
  }
  return cudaErrorNoDevice;
}

// Intercept: Get device
cudaError_t
cudaGetDevice(int* device)
{
  if (should_inject_fault()) {
    cudaError_t error = get_error_code();
    log_intercept("cudaGetDevice", error);
    return error;
  }

  typedef cudaError_t (*real_func_t)(int*);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cudaGetDevice");
  if (real_func) {
    return real_func(device);
  }
  return cudaErrorNoDevice;
}

// Intercept: Malloc
cudaError_t
cudaMalloc(void** devPtr, size_t size)
{
  if (should_inject_fault()) {
    cudaError_t error = get_error_code();
    log_intercept("cudaMalloc", error);
    return error;
  }

  typedef cudaError_t (*real_func_t)(void**, size_t);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cudaMalloc");
  if (real_func) {
    return real_func(devPtr, size);
  }
  return cudaErrorNoDevice;
}

// Intercept: Free
cudaError_t
cudaFree(void* devPtr)
{
  if (should_inject_fault()) {
    cudaError_t error = get_error_code();
    log_intercept("cudaFree", error);
    return error;
  }

  typedef cudaError_t (*real_func_t)(void*);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cudaFree");
  if (real_func) {
    return real_func(devPtr);
  }
  return cudaErrorNoDevice;
}

// Intercept: Memcpy
cudaError_t
cudaMemcpy(void* dst, const void* src, size_t count, int kind)
{
  if (should_inject_fault()) {
    cudaError_t error = get_error_code();
    log_intercept("cudaMemcpy", error);
    return error;
  }

  typedef cudaError_t (*real_func_t)(void*, const void*, size_t, int);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cudaMemcpy");
  if (real_func) {
    return real_func(dst, src, count, kind);
  }
  return cudaErrorNoDevice;
}

// Intercept: Device synchronize
cudaError_t
cudaDeviceSynchronize(void)
{
  if (should_inject_fault()) {
    cudaError_t error = get_error_code();
    log_intercept("cudaDeviceSynchronize", error);
    return error;
  }

  typedef cudaError_t (*real_func_t)(void);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cudaDeviceSynchronize");
  if (real_func) {
    return real_func();
  }
  return cudaErrorNoDevice;
}

// Intercept: Get device properties
cudaError_t
cudaGetDeviceProperties(cudaDeviceProp* prop, int device)
{
  if (should_inject_fault()) {
    cudaError_t error = get_error_code();
    log_intercept("cudaGetDeviceProperties", error);
    return error;
  }

  typedef cudaError_t (*real_func_t)(cudaDeviceProp*, int);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cudaGetDeviceProperties");
  if (real_func) {
    return real_func(prop, device);
  }
  return cudaErrorNoDevice;
}

// =============================================================================
// CRITICAL: Kernel launch interception - needed for inference fault injection
// =============================================================================

// Intercept: Kernel launch (CRITICAL for inference)
cudaError_t
cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim,
                 void** args, size_t sharedMem, cudaStream_t stream)
{
  if (should_inject_fault()) {
    cudaError_t error = get_error_code();
    log_intercept("cudaLaunchKernel", error);
    return error;
  }

  typedef cudaError_t (*real_func_t)(const void*, dim3, dim3, void**, size_t, cudaStream_t);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cudaLaunchKernel");
  if (real_func) {
    return real_func(func, gridDim, blockDim, args, sharedMem, stream);
  }
  return cudaErrorNoDevice;
}

// Intercept: Async memcpy (commonly used in inference)
cudaError_t
cudaMemcpyAsync(void* dst, const void* src, size_t count, int kind, cudaStream_t stream)
{
  if (should_inject_fault()) {
    cudaError_t error = get_error_code();
    log_intercept("cudaMemcpyAsync", error);
    return error;
  }

  typedef cudaError_t (*real_func_t)(void*, const void*, size_t, int, cudaStream_t);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cudaMemcpyAsync");
  if (real_func) {
    return real_func(dst, src, count, kind, stream);
  }
  return cudaErrorNoDevice;
}

// Intercept: Stream synchronize (wait for GPU work)
cudaError_t
cudaStreamSynchronize(cudaStream_t stream)
{
  if (should_inject_fault()) {
    cudaError_t error = get_error_code();
    log_intercept("cudaStreamSynchronize", error);
    return error;
  }

  typedef cudaError_t (*real_func_t)(cudaStream_t);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cudaStreamSynchronize");
  if (real_func) {
    return real_func(stream);
  }
  return cudaErrorNoDevice;
}

// =============================================================================
// CUDA DRIVER API INTERCEPTION
// PyTorch/vLLM often use Driver API (cu*) instead of Runtime API (cuda*)
// =============================================================================

// Driver API types
typedef int CUresult;
typedef void* CUdevice;
typedef void* CUcontext;
typedef void* CUdeviceptr;
typedef void* CUstream;
typedef void* CUfunction;

// Driver API error codes (from cuda.h)
#define CUDA_SUCCESS 0
#define CUDA_ERROR_NOT_INITIALIZED 3
#define CUDA_ERROR_NO_DEVICE 100
#define CUDA_ERROR_INVALID_DEVICE 101
#define CUDA_ERROR_ECC_UNCORRECTABLE 214
#define CUDA_ERROR_UNKNOWN 999
#define CUDA_ERROR_LAUNCH_TIMEOUT 702
#define CUDA_ERROR_PEER_ACCESS_UNSUPPORTED 217

// XID to Driver API error mapping
typedef struct {
  int xid;
  CUresult cu_error;
} xid_driver_mapping_t;

static const xid_driver_mapping_t xid_driver_mappings[] = {
    {79, CUDA_ERROR_NO_DEVICE},
    {48, CUDA_ERROR_ECC_UNCORRECTABLE},
    {94, CUDA_ERROR_ECC_UNCORRECTABLE},
    {95, CUDA_ERROR_UNKNOWN},
    {43, CUDA_ERROR_LAUNCH_TIMEOUT},
    {74, CUDA_ERROR_PEER_ACCESS_UNSUPPORTED},
    {0, 0}  // Sentinel
};

// Get Driver API error code for current XID
static CUresult
get_driver_error_code()
{
  int inject, xid;
  cudaError_t runtime_error;
  get_fault_config(&inject, &xid, &runtime_error);

  // Map XID to driver error
  for (int i = 0; xid_driver_mappings[i].xid != 0; i++) {
    if (xid_driver_mappings[i].xid == xid) {
      return xid_driver_mappings[i].cu_error;
    }
  }
  return CUDA_ERROR_NO_DEVICE;  // Default
}

// Log helper for Driver API
static void
log_driver_intercept(const char* func_name, CUresult error_code)
{
  if (should_inject_fault()) {
    int inject, xid;
    cudaError_t err;
    get_fault_config(&inject, &xid, &err);
    fprintf(stderr, "[XID %d SIM] %s() intercepted (Driver API) -> error %d\n", xid, func_name, error_code);
  }
}

// Intercept: cuInit (CRITICAL - first call PyTorch makes)
CUresult
cuInit(unsigned int flags)
{
  if (should_inject_fault()) {
    CUresult error = get_driver_error_code();
    log_driver_intercept("cuInit", error);
    return error;
  }

  typedef CUresult (*real_func_t)(unsigned int);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cuInit");
  if (real_func) {
    return real_func(flags);
  }
  return CUDA_ERROR_NO_DEVICE;
}

// Intercept: cuDeviceGetCount
CUresult
cuDeviceGetCount(int* count)
{
  if (should_inject_fault()) {
    CUresult error = get_driver_error_code();
    log_driver_intercept("cuDeviceGetCount", error);
    if (count)
      *count = 0;
    return error;
  }

  typedef CUresult (*real_func_t)(int*);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cuDeviceGetCount");
  if (real_func) {
    return real_func(count);
  }
  return CUDA_ERROR_NO_DEVICE;
}

// Intercept: cuDeviceGet
CUresult
cuDeviceGet(CUdevice* device, int ordinal)
{
  if (should_inject_fault()) {
    CUresult error = get_driver_error_code();
    log_driver_intercept("cuDeviceGet", error);
    return error;
  }

  typedef CUresult (*real_func_t)(CUdevice*, int);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cuDeviceGet");
  if (real_func) {
    return real_func(device, ordinal);
  }
  return CUDA_ERROR_NO_DEVICE;
}

// Intercept: cuDeviceGetAttribute
CUresult
cuDeviceGetAttribute(int* pi, int attrib, CUdevice dev)
{
  if (should_inject_fault()) {
    CUresult error = get_driver_error_code();
    log_driver_intercept("cuDeviceGetAttribute", error);
    return error;
  }

  typedef CUresult (*real_func_t)(int*, int, CUdevice);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cuDeviceGetAttribute");
  if (real_func) {
    return real_func(pi, attrib, dev);
  }
  return CUDA_ERROR_NO_DEVICE;
}

// Intercept: cuDeviceGetName
CUresult
cuDeviceGetName(char* name, int len, CUdevice dev)
{
  if (should_inject_fault()) {
    CUresult error = get_driver_error_code();
    log_driver_intercept("cuDeviceGetName", error);
    return error;
  }

  typedef CUresult (*real_func_t)(char*, int, CUdevice);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cuDeviceGetName");
  if (real_func) {
    return real_func(name, len, dev);
  }
  return CUDA_ERROR_NO_DEVICE;
}

// Intercept: cuCtxCreate (v1)
CUresult
cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev)
{
  if (should_inject_fault()) {
    CUresult error = get_driver_error_code();
    log_driver_intercept("cuCtxCreate", error);
    return error;
  }

  typedef CUresult (*real_func_t)(CUcontext*, unsigned int, CUdevice);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cuCtxCreate");
  if (real_func) {
    return real_func(pctx, flags, dev);
  }
  return CUDA_ERROR_NO_DEVICE;
}

// Intercept: cuCtxCreate_v2
CUresult
cuCtxCreate_v2(CUcontext* pctx, unsigned int flags, CUdevice dev)
{
  if (should_inject_fault()) {
    CUresult error = get_driver_error_code();
    log_driver_intercept("cuCtxCreate_v2", error);
    return error;
  }

  typedef CUresult (*real_func_t)(CUcontext*, unsigned int, CUdevice);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cuCtxCreate_v2");
  if (real_func) {
    return real_func(pctx, flags, dev);
  }
  return CUDA_ERROR_NO_DEVICE;
}

// Intercept: cuCtxGetCurrent
CUresult
cuCtxGetCurrent(CUcontext* pctx)
{
  if (should_inject_fault()) {
    CUresult error = get_driver_error_code();
    log_driver_intercept("cuCtxGetCurrent", error);
    return error;
  }

  typedef CUresult (*real_func_t)(CUcontext*);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cuCtxGetCurrent");
  if (real_func) {
    return real_func(pctx);
  }
  return CUDA_ERROR_NO_DEVICE;
}

// Intercept: cuCtxSetCurrent
CUresult
cuCtxSetCurrent(CUcontext ctx)
{
  if (should_inject_fault()) {
    CUresult error = get_driver_error_code();
    log_driver_intercept("cuCtxSetCurrent", error);
    return error;
  }

  typedef CUresult (*real_func_t)(CUcontext);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cuCtxSetCurrent");
  if (real_func) {
    return real_func(ctx);
  }
  return CUDA_ERROR_NO_DEVICE;
}

// Intercept: cuCtxSynchronize
CUresult
cuCtxSynchronize(void)
{
  if (should_inject_fault()) {
    CUresult error = get_driver_error_code();
    log_driver_intercept("cuCtxSynchronize", error);
    return error;
  }

  typedef CUresult (*real_func_t)(void);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cuCtxSynchronize");
  if (real_func) {
    return real_func();
  }
  return CUDA_ERROR_NO_DEVICE;
}

// Intercept: cuMemAlloc (v1)
CUresult
cuMemAlloc(CUdeviceptr* dptr, size_t bytesize)
{
  if (should_inject_fault()) {
    CUresult error = get_driver_error_code();
    log_driver_intercept("cuMemAlloc", error);
    return error;
  }

  typedef CUresult (*real_func_t)(CUdeviceptr*, size_t);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cuMemAlloc");
  if (real_func) {
    return real_func(dptr, bytesize);
  }
  return CUDA_ERROR_NO_DEVICE;
}

// Intercept: cuMemAlloc_v2
CUresult
cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize)
{
  if (should_inject_fault()) {
    CUresult error = get_driver_error_code();
    log_driver_intercept("cuMemAlloc_v2", error);
    return error;
  }

  typedef CUresult (*real_func_t)(CUdeviceptr*, size_t);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cuMemAlloc_v2");
  if (real_func) {
    return real_func(dptr, bytesize);
  }
  return CUDA_ERROR_NO_DEVICE;
}

// Intercept: cuMemFree (v1)
CUresult
cuMemFree(CUdeviceptr dptr)
{
  if (should_inject_fault()) {
    CUresult error = get_driver_error_code();
    log_driver_intercept("cuMemFree", error);
    return error;
  }

  typedef CUresult (*real_func_t)(CUdeviceptr);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cuMemFree");
  if (real_func) {
    return real_func(dptr);
  }
  return CUDA_ERROR_NO_DEVICE;
}

// Intercept: cuMemFree_v2
CUresult
cuMemFree_v2(CUdeviceptr dptr)
{
  if (should_inject_fault()) {
    CUresult error = get_driver_error_code();
    log_driver_intercept("cuMemFree_v2", error);
    return error;
  }

  typedef CUresult (*real_func_t)(CUdeviceptr);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cuMemFree_v2");
  if (real_func) {
    return real_func(dptr);
  }
  return CUDA_ERROR_NO_DEVICE;
}

// Intercept: cuMemcpyDtoH (Device to Host)
CUresult
cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount)
{
  if (should_inject_fault()) {
    CUresult error = get_driver_error_code();
    log_driver_intercept("cuMemcpyDtoH", error);
    return error;
  }

  typedef CUresult (*real_func_t)(void*, CUdeviceptr, size_t);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cuMemcpyDtoH");
  if (real_func) {
    return real_func(dstHost, srcDevice, ByteCount);
  }
  return CUDA_ERROR_NO_DEVICE;
}

// Intercept: cuMemcpyDtoH_v2
CUresult
cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount)
{
  if (should_inject_fault()) {
    CUresult error = get_driver_error_code();
    log_driver_intercept("cuMemcpyDtoH_v2", error);
    return error;
  }

  typedef CUresult (*real_func_t)(void*, CUdeviceptr, size_t);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cuMemcpyDtoH_v2");
  if (real_func) {
    return real_func(dstHost, srcDevice, ByteCount);
  }
  return CUDA_ERROR_NO_DEVICE;
}

// Intercept: cuMemcpyHtoD (Host to Device)
CUresult
cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount)
{
  if (should_inject_fault()) {
    CUresult error = get_driver_error_code();
    log_driver_intercept("cuMemcpyHtoD", error);
    return error;
  }

  typedef CUresult (*real_func_t)(CUdeviceptr, const void*, size_t);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cuMemcpyHtoD");
  if (real_func) {
    return real_func(dstDevice, srcHost, ByteCount);
  }
  return CUDA_ERROR_NO_DEVICE;
}

// Intercept: cuMemcpyHtoD_v2
CUresult
cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount)
{
  if (should_inject_fault()) {
    CUresult error = get_driver_error_code();
    log_driver_intercept("cuMemcpyHtoD_v2", error);
    return error;
  }

  typedef CUresult (*real_func_t)(CUdeviceptr, const void*, size_t);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cuMemcpyHtoD_v2");
  if (real_func) {
    return real_func(dstDevice, srcHost, ByteCount);
  }
  return CUDA_ERROR_NO_DEVICE;
}

// Intercept: cuMemcpyAsync
CUresult
cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream)
{
  if (should_inject_fault()) {
    CUresult error = get_driver_error_code();
    log_driver_intercept("cuMemcpyAsync", error);
    return error;
  }

  typedef CUresult (*real_func_t)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cuMemcpyAsync");
  if (real_func) {
    return real_func(dst, src, ByteCount, hStream);
  }
  return CUDA_ERROR_NO_DEVICE;
}

// Intercept: cuLaunchKernel (CRITICAL for inference)
CUresult
cuLaunchKernel(CUfunction f,
               unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
               unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
               unsigned int sharedMemBytes, CUstream hStream,
               void** kernelParams, void** extra)
{
  if (should_inject_fault()) {
    CUresult error = get_driver_error_code();
    log_driver_intercept("cuLaunchKernel", error);
    return error;
  }

  typedef CUresult (*real_func_t)(CUfunction,
                                  unsigned int, unsigned int, unsigned int,
                                  unsigned int, unsigned int, unsigned int,
                                  unsigned int, CUstream, void**, void**);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cuLaunchKernel");
  if (real_func) {
    return real_func(f, gridDimX, gridDimY, gridDimZ,
                     blockDimX, blockDimY, blockDimZ,
                     sharedMemBytes, hStream, kernelParams, extra);
  }
  return CUDA_ERROR_NO_DEVICE;
}

// Intercept: cuStreamSynchronize
CUresult
cuStreamSynchronize(CUstream hStream)
{
  if (should_inject_fault()) {
    CUresult error = get_driver_error_code();
    log_driver_intercept("cuStreamSynchronize", error);
    return error;
  }

  typedef CUresult (*real_func_t)(CUstream);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cuStreamSynchronize");
  if (real_func) {
    return real_func(hStream);
  }
  return CUDA_ERROR_NO_DEVICE;
}

// Intercept: cuStreamCreate
CUresult
cuStreamCreate(CUstream* phStream, unsigned int Flags)
{
  if (should_inject_fault()) {
    CUresult error = get_driver_error_code();
    log_driver_intercept("cuStreamCreate", error);
    return error;
  }

  typedef CUresult (*real_func_t)(CUstream*, unsigned int);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cuStreamCreate");
  if (real_func) {
    return real_func(phStream, Flags);
  }
  return CUDA_ERROR_NO_DEVICE;
}

// Intercept: cuModuleLoad (for loading kernels)
CUresult
cuModuleLoad(void* module, const char* fname)
{
  if (should_inject_fault()) {
    CUresult error = get_driver_error_code();
    log_driver_intercept("cuModuleLoad", error);
    return error;
  }

  typedef CUresult (*real_func_t)(void*, const char*);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cuModuleLoad");
  if (real_func) {
    return real_func(module, fname);
  }
  return CUDA_ERROR_NO_DEVICE;
}

// Intercept: cuModuleLoadData
CUresult
cuModuleLoadData(void* module, const void* image)
{
  if (should_inject_fault()) {
    CUresult error = get_driver_error_code();
    log_driver_intercept("cuModuleLoadData", error);
    return error;
  }

  typedef CUresult (*real_func_t)(void*, const void*);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cuModuleLoadData");
  if (real_func) {
    return real_func(module, image);
  }
  return CUDA_ERROR_NO_DEVICE;
}

// Intercept: cuModuleGetFunction
CUresult
cuModuleGetFunction(CUfunction* hfunc, void* hmod, const char* name)
{
  if (should_inject_fault()) {
    CUresult error = get_driver_error_code();
    log_driver_intercept("cuModuleGetFunction", error);
    return error;
  }

  typedef CUresult (*real_func_t)(CUfunction*, void*, const char*);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cuModuleGetFunction");
  if (real_func) {
    return real_func(hfunc, hmod, name);
  }
  return CUDA_ERROR_NO_DEVICE;
}
