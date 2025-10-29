/*
 * Fake CUDA Library for XID 79 Simulation
 * 
 * This library intercepts CUDA calls and returns CUDA_ERROR_NO_DEVICE
 * to simulate a GPU falling off the bus (XID 79).
 * 
 * Compile:
 *   gcc -shared -fPIC fake_cuda_xid79.c -o fake_cuda_xid79.so
 * 
 * Use:
 *   LD_PRELOAD=/path/to/fake_cuda_xid79.so python -m vllm.entrypoints.api_server
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

typedef int cudaError_t;
typedef struct cudaDeviceProp_st {
    char name[256];
    size_t totalGlobalMem;
    // ... other fields (we don't need them)
} cudaDeviceProp;

// CUDA error codes
#define cudaSuccess 0
#define cudaErrorNoDevice 100

// Control flag - can be disabled via environment variable
static int should_inject_fault() {
    static int initialized = 0;
    static int inject = 1;
    
    if (!initialized) {
        char* env = getenv("CUDA_FAULT_INJECTION_ENABLED");
        if (env) {
            inject = (strcmp(env, "1") == 0 || strcmp(env, "true") == 0);
        }
        fprintf(stderr, "[CUDA FAULT INJECTION] %s (XID 79 simulation)\n", 
                inject ? "ENABLED" : "DISABLED");
        initialized = 1;
    }
    return inject;
}

// Log helper
static void log_intercept(const char* func_name) {
    if (should_inject_fault()) {
        fprintf(stderr, "[XID 79 SIM] %s() intercepted -> CUDA_ERROR_NO_DEVICE\n", func_name);
    }
}

// Intercept: Get device count
cudaError_t cudaGetDeviceCount(int* count) {
    if (should_inject_fault()) {
        log_intercept("cudaGetDeviceCount");
        if (count) *count = 0;
        return cudaErrorNoDevice;
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
cudaError_t cudaSetDevice(int device) {
    if (should_inject_fault()) {
        log_intercept("cudaSetDevice");
        return cudaErrorNoDevice;
    }
    
    typedef cudaError_t (*real_func_t)(int);
    real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cudaSetDevice");
    if (real_func) {
        return real_func(device);
    }
    return cudaErrorNoDevice;
}

// Intercept: Get device
cudaError_t cudaGetDevice(int* device) {
    if (should_inject_fault()) {
        log_intercept("cudaGetDevice");
        return cudaErrorNoDevice;
    }
    
    typedef cudaError_t (*real_func_t)(int*);
    real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cudaGetDevice");
    if (real_func) {
        return real_func(device);
    }
    return cudaErrorNoDevice;
}

// Intercept: Malloc
cudaError_t cudaMalloc(void** devPtr, size_t size) {
    if (should_inject_fault()) {
        log_intercept("cudaMalloc");
        return cudaErrorNoDevice;
    }
    
    typedef cudaError_t (*real_func_t)(void**, size_t);
    real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cudaMalloc");
    if (real_func) {
        return real_func(devPtr, size);
    }
    return cudaErrorNoDevice;
}

// Intercept: Free
cudaError_t cudaFree(void* devPtr) {
    if (should_inject_fault()) {
        log_intercept("cudaFree");
        return cudaErrorNoDevice;
    }
    
    typedef cudaError_t (*real_func_t)(void*);
    real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cudaFree");
    if (real_func) {
        return real_func(devPtr);
    }
    return cudaErrorNoDevice;
}

// Intercept: Memcpy
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, int kind) {
    if (should_inject_fault()) {
        log_intercept("cudaMemcpy");
        return cudaErrorNoDevice;
    }
    
    typedef cudaError_t (*real_func_t)(void*, const void*, size_t, int);
    real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cudaMemcpy");
    if (real_func) {
        return real_func(dst, src, count, kind);
    }
    return cudaErrorNoDevice;
}

// Intercept: Device synchronize
cudaError_t cudaDeviceSynchronize(void) {
    if (should_inject_fault()) {
        log_intercept("cudaDeviceSynchronize");
        return cudaErrorNoDevice;
    }
    
    typedef cudaError_t (*real_func_t)(void);
    real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cudaDeviceSynchronize");
    if (real_func) {
        return real_func();
    }
    return cudaErrorNoDevice;
}

// Intercept: Get device properties
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device) {
    if (should_inject_fault()) {
        log_intercept("cudaGetDeviceProperties");
        return cudaErrorNoDevice;
    }
    
    typedef cudaError_t (*real_func_t)(cudaDeviceProp*, int);
    real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cudaGetDeviceProperties");
    if (real_func) {
        return real_func(prop, device);
    }
    return cudaErrorNoDevice;
}

