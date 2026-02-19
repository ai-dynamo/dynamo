/*
Copyright 2025 NVIDIA Corporation.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package dynamo_kv_scorer provides the CGO/FFI bindings to the Dynamo Rust router.
//
// This package owns all CGO interactions with the libdynamo_llm_capi static library.
// The disagg plugin package imports the exported Go wrapper functions from here
// to call into the Rust router for prefill/decode worker selection and bookkeeping.
package dynamo_kv_scorer

/*
#cgo CPPFLAGS: -I${SRCDIR}/include
#cgo CXXFLAGS: -std=c++17
#cgo LDFLAGS: ${SRCDIR}/lib/libdynamo_llm_capi.a -lstdc++ -ldl -lpthread -lm

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>   // for free
#include <stdbool.h>

// Query router result codes (matches QueryRouterResult in Rust)
typedef uint32_t query_router_result_t;
enum {
    QUERY_ROUTER_OK = 0,
    QUERY_ROUTER_ERR_INVALID_HANDLE = 1,
    QUERY_ROUTER_ERR_INVALID_PARAM = 2,
    QUERY_ROUTER_ERR_INIT_FAILED = 3,
    QUERY_ROUTER_ERR_QUERY_FAILED = 4,
    QUERY_ROUTER_ERR_DISAGG_ENFORCED = 5,
    QUERY_ROUTER_ERR_TIMEOUT = 6,
};

// opaque handle forward-decl for Router bindings
struct RouterHandles;
typedef struct RouterHandles RouterHandles;

// Routing result from route functions
typedef struct {
    bool is_disaggregated;
    uint64_t prefill_worker_id;
    uint64_t decode_worker_id;
    uint32_t *token_ids;
    size_t token_count;
} CRoutingResult;

// Router bindings API
query_router_result_t create_routers(const char *namespace_c_str,
                                     const char *component_c_str,
                                     bool enforce_disagg,
                                     RouterHandles **out_handle);

query_router_result_t route_prefill_request(RouterHandles *handle,
                                            const char *request_json,
                                            const char *pods_json,
                                            CRoutingResult *out_result);

query_router_result_t route_decode_request(RouterHandles *handle,
                                           const char *request_json,
                                           const char *pods_json,
                                           bool is_disaggregated,
                                           CRoutingResult *out_result);

query_router_result_t add_request(RouterHandles *handle,
                                  const char *request_id,
                                  const uint32_t *token_ids,
                                  size_t token_count,
                                  uint64_t worker_id,
                                  uint32_t dp_rank);

query_router_result_t mark_prefill_complete(RouterHandles *handle,
                                            const char *request_id);

query_router_result_t free_request(RouterHandles *handle,
                                   const char *request_id);

void free_routing_result(CRoutingResult *result);

void destroy(RouterHandles *handle);
*/
import "C"

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"
	"unsafe"

	schedtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

// --------------------------- FFI integration ---------------------------

var (
	ffiOnce sync.Once
	ffiErr  error

	ffiNamespace     string
	ffiComponent     string
	ffiEnforceDisagg bool

	routerInitialized bool

	// Router handles (owned on the Rust side, opaque here)
	routerHandles      *C.struct_RouterHandles
	routerHandlesMutex sync.RWMutex
)

func loadDynamoConfig() {
	ffiNamespace = getEnvOrDefault("DYN_NAMESPACE", "vllm-agg")
	ffiComponent = "backend" // This is not the same as DYN_COMPONENT=epp (in this case)
	ffiEnforceDisagg = getEnvBoolOrDefault("DYN_ENFORCE_DISAGG", false)
	// Note: model name and kv_cache_block_size are now auto-discovered from the model card
	fmt.Printf("Dynamo KV Scorer: namespace=%s, component=%s, enforce_disagg=%v\n",
		ffiNamespace, ffiComponent, ffiEnforceDisagg)
}

func getEnvOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func getEnvBoolOrDefault(key string, def bool) bool {
	if v := os.Getenv(key); v != "" {
		switch strings.ToLower(v) {
		case "true", "1", "yes", "on":
			return true
		case "false", "0", "no", "off":
			return false
		}
	}
	return def
}

// initFFI initializes router handles using the Router bindings.
func initFFI() error {
	ffiOnce.Do(func() {
		loadDynamoConfig()

		ns := C.CString(ffiNamespace)
		cm := C.CString(ffiComponent)
		defer C.free(unsafe.Pointer(ns))
		defer C.free(unsafe.Pointer(cm))

		// Create router handles
		routerHandlesMutex.Lock()
		defer routerHandlesMutex.Unlock()

		rc := C.create_routers(
			ns,
			cm,
			C.bool(ffiEnforceDisagg),
			&routerHandles,
		)
		if rc != C.QUERY_ROUTER_OK {
			ffiErr = fmt.Errorf("create_routers failed with code %d", rc)
			return
		}
		routerInitialized = true
	})
	return ffiErr
}

// InitFFI exposes the FFI initialization for use by the disagg plugin package.
// It is idempotent — safe to call multiple times.
func InitFFI() error {
	return initFFI()
}

// podInfoJSON is the JSON-serializable representation of a backend.Pod (datalayer.PodInfo).
type podInfoJSON struct {
	Name        string            `json:"name"`
	Namespace   string            `json:"namespace"`
	PodName     string            `json:"podName"`
	Address     string            `json:"address"`
	Port        string            `json:"port"`
	MetricsHost string            `json:"metricsHost"`
	Labels      map[string]string `json:"labels"`
}

// metricsJSON is the JSON-serializable representation of backendmetrics.MetricsState (datalayer.Metrics).
type metricsJSON struct {
	ActiveModels            map[string]int `json:"activeModels"`
	WaitingModels           map[string]int `json:"waitingModels"`
	MaxActiveModels         int            `json:"maxActiveModels"`
	RunningQueueSize        int            `json:"runningQueueSize"`
	WaitingQueueSize        int            `json:"waitingQueueSize"`
	KVCacheUsagePercent     float64        `json:"kvCacheUsagePercent"`
	KvCacheMaxTokenCapacity int            `json:"kvCacheMaxTokenCapacity"`
	CacheBlockSize          int            `json:"cacheBlockSize"`
	CacheNumGPUBlocks       int            `json:"cacheNumGPUBlocks"`
	UpdateTime              time.Time      `json:"updateTime"`
}

// podJSON is the JSON-serializable representation of a schedtypes.Pod passed across the FFI boundary.
type podJSON struct {
	Pod     *podInfoJSON `json:"pod"`
	Metrics *metricsJSON `json:"metrics"`
}

// SerializePodsToJSON converts a slice of schedtypes.Pod into a JSON string
// suitable for passing across the C FFI boundary to the Rust router.
func SerializePodsToJSON(pods []schedtypes.Pod) (string, error) {
	out := make([]podJSON, 0, len(pods))
	for _, p := range pods {
		entry := podJSON{}

		if podInfo := p.GetPod(); podInfo != nil {
			entry.Pod = &podInfoJSON{
				Name:        podInfo.NamespacedName.Name,
				Namespace:   podInfo.NamespacedName.Namespace,
				PodName:     podInfo.PodName,
				Address:     podInfo.Address,
				Port:        podInfo.Port,
				MetricsHost: podInfo.MetricsHost,
				Labels:      podInfo.Labels,
			}
		}

		if m := p.GetMetrics(); m != nil {
			entry.Metrics = &metricsJSON{
				ActiveModels:            m.ActiveModels,
				WaitingModels:           m.WaitingModels,
				MaxActiveModels:         m.MaxActiveModels,
				RunningQueueSize:        m.RunningQueueSize,
				WaitingQueueSize:        m.WaitingQueueSize,
				KVCacheUsagePercent:     m.KVCacheUsagePercent,
				KvCacheMaxTokenCapacity: m.KvCacheMaxTokenCapacity,
				CacheBlockSize:          m.CacheBlockSize,
				CacheNumGPUBlocks:       m.CacheNumGPUBlocks,
				UpdateTime:              m.UpdateTime,
			}
		}

		out = append(out, entry)
	}

	data, err := json.Marshal(out)
	if err != nil {
		return "", fmt.Errorf("failed to serialize pods: %w", err)
	}
	return string(data), nil
}

func BuildOpenAIRequest(req *schedtypes.LLMRequest) (map[string]any, error) {
	requestBody := make(map[string]any)

	// Preserve the original message structure for correct chat template application
	if req == nil || req.Body == nil {
		return nil, fmt.Errorf("missing request body")
	}

	if req.Body.ChatCompletions != nil && len(req.Body.ChatCompletions.Messages) > 0 {
		messages := make([]map[string]any, 0, len(req.Body.ChatCompletions.Messages))
		anyNonEmpty := false
		for _, msg := range req.Body.ChatCompletions.Messages {
			content := msg.Content.PlainText()
			if strings.TrimSpace(content) != "" {
				anyNonEmpty = true
			}
			messages = append(messages, map[string]any{
				"role":    msg.Role,
				"content": content,
			})
		}
		if !anyNonEmpty {
			return nil, fmt.Errorf("empty chat messages")
		}
		requestBody["messages"] = messages
	} else if req.Body.Completions != nil && strings.TrimSpace(req.Body.Completions.Prompt) != "" {
		// Legacy completions format - wrap as single user message
		requestBody["messages"] = []map[string]any{
			{"role": "user", "content": req.Body.Completions.Prompt},
		}
	} else {
		return nil, fmt.Errorf("no messages or prompt provided")
	}

	// Model field is required by OpenAI spec but not used by the router's tokenizer
	// (tokenizer is determined by the discovered model card)
	if req != nil && strings.TrimSpace(req.TargetModel) != "" {
		requestBody["model"] = req.TargetModel
	} else {
		requestBody["model"] = "default"
	}
	return requestBody, nil
}

// CallAddRequest registers a request with the router's bookkeeping.
func CallAddRequest(requestID string, tokenData []int64, workerID uint64, dpRank uint32) error {
	if !routerInitialized {
		return fmt.Errorf("dynamo router not initialized")
	}

	routerHandlesMutex.RLock()
	router := routerHandles
	routerHandlesMutex.RUnlock()

	if router == nil {
		return fmt.Errorf("dynamo router handles not created")
	}

	// Convert token data from int64 to uint32
	tokens := make([]uint32, len(tokenData))
	for i, t := range tokenData {
		tokens[i] = uint32(t)
	}

	cRequestID := C.CString(requestID)
	defer C.free(unsafe.Pointer(cRequestID))

	var cTokens *C.uint32_t
	if len(tokens) > 0 {
		cTokens = (*C.uint32_t)(unsafe.Pointer(&tokens[0]))
	}

	rc := C.add_request(
		router,
		cRequestID,
		cTokens,
		C.size_t(len(tokens)),
		C.uint64_t(workerID),
		C.uint32_t(dpRank),
	)

	if rc != C.QUERY_ROUTER_OK {
		return fmt.Errorf("add_request failed with code %d", rc)
	}
	return nil
}

// CallMarkPrefillComplete marks prefill as completed for a request (bookkeeping).
func CallMarkPrefillComplete(requestID string) error {
	if !routerInitialized {
		return fmt.Errorf("dynamo router not initialized")
	}

	routerHandlesMutex.RLock()
	router := routerHandles
	routerHandlesMutex.RUnlock()

	if router == nil {
		return fmt.Errorf("dynamo router handles not created")
	}

	cRequestID := C.CString(requestID)
	defer C.free(unsafe.Pointer(cRequestID))

	rc := C.mark_prefill_complete(router, cRequestID)
	if rc != C.QUERY_ROUTER_OK {
		return fmt.Errorf("mark_prefill_complete failed with code %d", rc)
	}
	return nil
}

// CallFreeRequest cleans up router state for a completed/cancelled request (bookkeeping).
func CallFreeRequest(requestID string) error {
	if !routerInitialized {
		return fmt.Errorf("dynamo router not initialized")
	}

	routerHandlesMutex.RLock()
	router := routerHandles
	routerHandlesMutex.RUnlock()

	if router == nil {
		return fmt.Errorf("dynamo router handles not created")
	}

	cRequestID := C.CString(requestID)
	defer C.free(unsafe.Pointer(cRequestID))

	rc := C.free_request(router, cRequestID)
	if rc != C.QUERY_ROUTER_OK {
		return fmt.Errorf("free_request failed with code %d", rc)
	}
	return nil
}

// RoutingResult holds the result of a prefill or decode routing call.
type RoutingResult struct {
	WorkerID  uint64
	TokenData []int64
}

// CallRoutePrefillRequest routes a request to the best prefill worker.
// It tokenizes the request and queries only the prefill router.
func CallRoutePrefillRequest(requestJSON string, podsJSON string) (*RoutingResult, error) {
	if !routerInitialized {
		return nil, fmt.Errorf("dynamo router not initialized")
	}

	routerHandlesMutex.RLock()
	router := routerHandles
	routerHandlesMutex.RUnlock()
	if router == nil {
		return nil, fmt.Errorf("dynamo router handles not created")
	}

	cRequestJSON := C.CString(requestJSON)
	defer C.free(unsafe.Pointer(cRequestJSON))

	var cPodsJSON *C.char
	if podsJSON != "" {
		cPodsJSON = C.CString(podsJSON)
		defer C.free(unsafe.Pointer(cPodsJSON))
	}

	var result C.CRoutingResult
	rc := C.route_prefill_request(router, cRequestJSON, cPodsJSON, &result)
	if rc != C.QUERY_ROUTER_OK {
		return nil, fmt.Errorf("route_prefill_request failed with code %d", rc)
	}

	// Copy token IDs into Go memory
	count := int(result.token_count)
	var tokens64 []int64
	if count > 0 && result.token_ids != nil {
		src := unsafe.Slice((*uint32)(unsafe.Pointer(result.token_ids)), count)
		tokens64 = make([]int64, count)
		for i := 0; i < count; i++ {
			tokens64[i] = int64(src[i])
		}
	}

	workerID := uint64(result.prefill_worker_id)
	C.free_routing_result(&result)

	return &RoutingResult{WorkerID: workerID, TokenData: tokens64}, nil
}

// CallRouteDecodeRequest routes a request to the best decode worker.
// When isDisaggregated is true, overlap_score_weight=0 is used (KV cache transferred from prefill).
func CallRouteDecodeRequest(requestJSON string, podsJSON string, isDisaggregated bool) (*RoutingResult, error) {
	if !routerInitialized {
		return nil, fmt.Errorf("dynamo router not initialized")
	}

	routerHandlesMutex.RLock()
	router := routerHandles
	routerHandlesMutex.RUnlock()
	if router == nil {
		return nil, fmt.Errorf("dynamo router handles not created")
	}

	cRequestJSON := C.CString(requestJSON)
	defer C.free(unsafe.Pointer(cRequestJSON))

	var cPodsJSON *C.char
	if podsJSON != "" {
		cPodsJSON = C.CString(podsJSON)
		defer C.free(unsafe.Pointer(cPodsJSON))
	}

	var result C.CRoutingResult
	rc := C.route_decode_request(router, cRequestJSON, cPodsJSON, C.bool(isDisaggregated), &result)
	if rc != C.QUERY_ROUTER_OK {
		return nil, fmt.Errorf("route_decode_request failed with code %d", rc)
	}

	// Copy token IDs into Go memory
	count := int(result.token_count)
	var tokens64 []int64
	if count > 0 && result.token_ids != nil {
		src := unsafe.Slice((*uint32)(unsafe.Pointer(result.token_ids)), count)
		tokens64 = make([]int64, count)
		for i := 0; i < count; i++ {
			tokens64[i] = int64(src[i])
		}
	}

	workerID := uint64(result.decode_worker_id)
	C.free_routing_result(&result)

	return &RoutingResult{WorkerID: workerID, TokenData: tokens64}, nil
}

func cleanupDynamo() error {
	routerHandlesMutex.Lock()
	defer routerHandlesMutex.Unlock()

	if routerHandles != nil {
		C.destroy(routerHandles)
		routerHandles = nil
	}

	routerInitialized = false
	return nil
}
