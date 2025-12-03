/*
 * CUDA Intercept Library
 *
 * This library intercepts CUDA calls and returns appropriate error codes
 * to simulate various GPU failures (XIDs).
 *
 * Supported XID types (set via CUDA_XID_TYPE environment variable):
 *   79  - GPU fell off bus (CUDA_ERROR_NO_DEVICE) - DEFAULT
 *   48  - Double-bit ECC error (CUDA_ERROR_ECC_UNCORRECTABLE)
 *   94  - Contained ECC error (CUDA_ERROR_ECC_UNCORRECTABLE)
 *   95  - Uncontained error (CUDA_ERROR_UNKNOWN)
 *   43  - GPU stopped responding (CUDA_ERROR_LAUNCH_TIMEOUT)
 *   74  - NVLink error (CUDA_ERROR_PEER_ACCESS_UNSUPPORTED)
 *
 * Compile:
 *   gcc -shared -fPIC -ldl cuda_intercept.c -o cuda_intercept.so
 *
 * Use:
 *   export CUDA_FAULT_INJECTION_ENABLED=1
 *   export CUDA_XID_TYPE=79  # or 48, 94, 95, 43, 74
 *   LD_PRELOAD=/path/to/cuda_intercept.so python -m vllm.entrypoints.api_server
 */

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int cudaError_t;
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
static void
get_fault_config(int* inject, int* xid_type, cudaError_t* error_code)
{
  static int initialized = 0;
  static int cached_inject = 0;
  static int cached_xid = 79;  // Default to XID 79
  static cudaError_t cached_error = cudaErrorNoDevice;

  if (!initialized) {
    // Check if injection is enabled
    char* env = getenv("CUDA_FAULT_INJECTION_ENABLED");
    if (env) {
      cached_inject = (strcmp(env, "1") == 0 || strcmp(env, "true") == 0);
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
              stderr, "[CUDA FAULT INJECTION] ENABLED - Simulating XID %d (%s)\n", cached_xid,
              xid_mappings[i].description);
          found = 1;
          break;
        }
      }

      if (!found) {
        fprintf(stderr, "[CUDA FAULT INJECTION] WARNING: Unknown XID %d, defaulting to XID 79\n", cached_xid);
        cached_xid = 79;
        cached_error = cudaErrorNoDevice;
      }
    } else {
      fprintf(
          stderr, "[CUDA FAULT INJECTION] %s (default: XID 79 - GPU fell off bus)\n",
          cached_inject ? "ENABLED" : "DISABLED");
    }

    initialized = 1;
  }

  *inject = cached_inject;
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
