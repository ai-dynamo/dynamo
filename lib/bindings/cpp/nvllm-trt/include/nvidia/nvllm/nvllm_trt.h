#ifndef __NVIDIA_NVLLM_TRT_C_API__
#define __NVIDIA_NVLLM_TRT_C_API__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

typedef enum
{
    NVLLM_TRT_ENGINE_SUCCESS              = 0,  // No error
    NVLLM_TRT_ENGINE_INVALID_REQUEST      = 1,  // Invalid request error
    NVLLM_TRT_ENGINE_SHUTDOWN_REQUIRED    = 2,  // Shutdown and join required before destroying
    NVLLM_TRT_ENGINE_SHUTDOWN_IN_PROGRESS = 3,  // Shutdown in progress
} nvllm_trt_engine_error_t;

// struct nvllm_trt_engine {};

// Forward declaration of the C++ class
typedef struct nvllm_trt_engine nvllm_trt_engine;
typedef nvllm_trt_engine* nvllm_trt_engine_t;
typedef uint64_t request_id_t;
typedef uint64_t client_id_t;

// Set the MPI Communicator for the TensorRT LLM Engine
// This function should be called before creating the engine
int nvllm_trt_mpi_session_set_communicator(void* world_comm_ptr);

// Functions to interact with nvllm_trt_engine_s
nvllm_trt_engine_t nvllm_trt_engine_create(const char* config_proto);

// Create a nvLLM TRT Engine from an instance of the engine
// This requires the raw engine pointer to be an instantiated object at the exact same
// commit version as the version of TRTLLM used to build the nvLLM C API.
// This is a workaround to enable the Dynamo TensorRT LLM backend to use nvLLM.
nvllm_trt_engine_t nvllm_trt_engine_unsafe_create_from_executor(void* engine);

// Source: Enqueue a streaming request via a json message to the request queue
request_id_t nvllm_trt_engine_enqueue_request(nvllm_trt_engine_t engine, client_id_t client_id, const char* req_proto);

// Sink: Pull off streaming responses from the response queue
char* nvllm_trt_engine_await_responses(nvllm_trt_engine_t engine);

// Sink: Pull off KvEvents from the event queue
char* nvllm_trt_engine_await_kv_events(nvllm_trt_engine_t engine);

// Get basic iteration stats
char* nvllm_trt_engine_await_iter_stats(nvllm_trt_engine_t engine);

// Free the memory allocated by nvllm_trt_engine_await_responses
void nvllm_trt_engine_free_responses(char* responses);

// Sink: Pull off streaming responses from the response queue
void nvllm_trt_engine_cancel_request(nvllm_trt_engine_t engine, uint64_t request_id);

// Initiate the shutdown sequence
void nvllm_trt_engine_shutdown(nvllm_trt_engine_t engine);

// // Await for the shutdown to complete; shutdown will be requested if not already requested
// void nvllm_trt_engine_join(nvllm_trt_engine_t engine);

// Destroy the engine
int nvllm_trt_engine_destroy(nvllm_trt_engine_t engine);

// Returns true (non-zero) once the engine has started pulling requests
// There is currently no stopping, so once an engine has started,
// it will always return true, even when complete.
// This call does not block; the user should use some backoff strategy
// to poll for detecting the start of the engine.
int nvllm_trt_engine_is_ready(nvllm_trt_engine_t engine);

// Returns true (non-zero) once the engine has stopped pulling requests
int nvllm_trt_engine_has_completed(nvllm_trt_engine_t engine);

// // Returns the major version number of the trtllm library
// int trtllm_version_major();

// // Returns the minor version number of the trtllm library
// int trtllm_version_minor();

// // Returns the patch version number of the trtllm library
// int trtllm_version_patch();

#ifdef __cplusplus
}
#endif

#endif  // __NVIDIA_NVLLM_TRT_C_API__
