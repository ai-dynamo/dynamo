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
#include <time.h>
#include <unistd.h>

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

// Global call counter and GPU tracking
static volatile int cuda_call_count = 0;
static int after_step_threshold = 0;
static int after_step_initialized = 0;
static int after_seconds_threshold = 0;
static int after_seconds_initialized = 0;
static time_t start_time = 0;
static int target_gpu_id = -1;  // -1 = all GPUs, else specific GPU
static int target_gpu_initialized = 0;
static __thread int current_device = 0;  // Thread-local current GPU
static int debug_logging = 0;
static int debug_initialized = 0;

// Check if fault should be injected (with call count and GPU check)
static int
should_inject_fault()
{
  int inject, xid;
  cudaError_t error;
  get_fault_config(&inject, &xid, &error);

  // Initialize debug logging once
  if (!debug_initialized) {
    char* debug_env = getenv("CUDA_FAULT_DEBUG");
    if (debug_env && strcmp(debug_env, "1") == 0) {
      debug_logging = 1;
      fprintf(stderr, "[CUDA FAULT DEBUG] Verbose logging ENABLED\n");
    }
    debug_initialized = 1;
  }

  if (!inject) {
    if (debug_logging && cuda_call_count < 10) {
      fprintf(stderr, "[CUDA FAULT DEBUG] Injection disabled (count=%d)\n", cuda_call_count);
    }
    return 0;  // Injection disabled
  }

  // Initialize target GPU once
  if (!target_gpu_initialized) {
    char* gpu_env = getenv("CUDA_FAULT_GPU_ID");
    if (gpu_env) {
      target_gpu_id = atoi(gpu_env);
      fprintf(stderr, "[CUDA FAULT INJECTION] Targeting GPU %d only\n", target_gpu_id);
    } else {
      fprintf(stderr, "[CUDA FAULT INJECTION] Targeting ALL GPUs\n");
    }
    target_gpu_initialized = 1;
  }

  // Increment call count
  int current_count = __sync_fetch_and_add(&cuda_call_count, 1);

  // Debug logging for first few calls and around threshold
  if (debug_logging) {
    if (current_count < 10 || (after_step_threshold > 0 && current_count >= after_step_threshold - 5 &&
                               current_count <= after_step_threshold + 5)) {
      fprintf(
          stderr, "[CUDA FAULT DEBUG] Call #%d: current_device=%d, target_gpu=%d, threshold=%d\n", current_count,
          current_device, target_gpu_id, after_step_threshold);
    }
  }

  // Check if this call is for the target GPU
  if (target_gpu_id >= 0 && current_device != target_gpu_id) {
    if (debug_logging && current_count < 10) {
      fprintf(
          stderr, "[CUDA FAULT DEBUG] Call #%d: Wrong GPU (current=%d, target=%d) - skip\n", current_count,
          current_device, target_gpu_id);
    }
    return 0;  // Different GPU - don't inject
  }

  // Initialize time-based threshold once
  if (!after_seconds_initialized) {
    char* after_seconds_env = getenv("CUDA_FAULT_AFTER_SECONDS");
    if (after_seconds_env) {
      after_seconds_threshold = atoi(after_seconds_env);
      start_time = time(NULL);
      fprintf(stderr, "[CUDA FAULT INJECTION] Will inject after %d seconds from startup\n", after_seconds_threshold);
    }
    after_seconds_initialized = 1;
  }

  // Check time-based threshold first (more reliable than step count)
  if (after_seconds_threshold > 0) {
    time_t elapsed = time(NULL) - start_time;
    if (elapsed < after_seconds_threshold) {
      if (debug_logging && current_count < 10) {
        fprintf(
            stderr, "[CUDA FAULT DEBUG] Call #%d: %ld seconds elapsed, waiting for %d seconds\n", current_count,
            (long)elapsed, after_seconds_threshold);
      }
      return 0;  // Not enough time elapsed
    }
    // Time threshold reached - log the step count for reference
    if (current_count == after_seconds_threshold || (debug_logging && current_count % 100 == 0)) {
      fprintf(
          stderr, "[CUDA FAULT INFO] Time threshold reached at call #%d (after %ld seconds)\n", current_count,
          (long)elapsed);
    }
  }

  // Initialize after_step threshold once
  if (!after_step_initialized) {
    char* after_step_env = getenv("CUDA_FAULT_AFTER_STEP");
    if (after_step_env) {
      after_step_threshold = atoi(after_step_env);
      fprintf(stderr, "[CUDA FAULT INJECTION] Will inject after %d CUDA calls\n", after_step_threshold);
    }
    after_step_initialized = 1;
  }

  // Check step-based threshold (if no time-based threshold)
  if (after_step_threshold > 0 && after_seconds_threshold == 0 && current_count < after_step_threshold) {
    if (debug_logging && current_count < 10) {
      fprintf(
          stderr, "[CUDA FAULT DEBUG] Call #%d: Before threshold (%d) - skip\n", current_count, after_step_threshold);
    }
    return 0;  // Not yet - pass through
  }

  // All checks passed - inject fault!
  if (debug_logging || current_count == after_step_threshold) {
    fprintf(stderr, "[CUDA FAULT] Call #%d: INJECTING FAULT on GPU %d (XID %d)\n", current_count, current_device, xid);
  }

  return 1;  // Inject fault on target GPU!
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

// Intercept: Set device (also track current device for GPU-specific targeting)
cudaError_t
cudaSetDevice(int device)
{
  if (should_inject_fault()) {
    // SIMULATING GPU FAILURE:
    // Don't call real cudaSetDevice() because we're simulating that
    // the GPU doesn't exist (XID 79 = "GPU fell off bus").
    //
    // If we called real cudaSetDevice(2), we'd actually switch to GPU 2,
    // but we want to simulate GPU 2 is unavailable/broken.
    //
    // Returning error WITHOUT switching simulates:
    // "Tried to use GPU 2, but it's not there / fell off bus"

    cudaError_t error = get_error_code();
    log_intercept("cudaSetDevice", error);
    return error;  // Fail without switching device
  }

  // NO FAULT - Normal operation: actually switch to requested GPU
  typedef cudaError_t (*real_func_t)(int);
  real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cudaSetDevice");
  if (real_func) {
    cudaError_t result = real_func(device);

    if (result == cudaSuccess) {
      // GPU switch succeeded - track which device this thread is now using.
      // This enables GPU-specific fault injection: we can inject faults only
      // on calls targeting GPU 2, while letting GPU 0/1/3 work normally.
      //
      // Example: Thread calls cudaSetDevice(2), then later cudaMalloc().
      // cudaMalloc() has no device parameter, but we know it's for GPU 2
      // because we tracked the most recent cudaSetDevice(2) call.
      current_device = device;
    }
    // If failed: don't update current_device (we're still on previous GPU)

    return result;
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
