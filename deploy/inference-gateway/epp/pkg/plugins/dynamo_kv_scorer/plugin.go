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
};

// opaque handle forward-decl for Router bindings
struct RouterHandles;
typedef struct RouterHandles RouterHandles;

// Routing result from route_chat_request
typedef struct {
    bool is_disaggregated;
    uint64_t prefill_worker_id;
    uint64_t decode_worker_id;
    uint32_t decode_dp_rank;
    uint32_t decode_overlap_blocks;
    uint32_t *token_ids;
    size_t token_count;
} CRoutingResult;

// Router bindings API (replaces Pipeline API)
query_router_result_t create_routers(const char *namespace_c_str,
                                     const char *component_c_str,
                                     const char *model_name_c_str,
                                     uint32_t block_size,
                                     bool enforce_disagg,
                                     bool wait_for_prefill,
                                     uint32_t prefill_timeout_secs,
                                     RouterHandles **out_handle);

query_router_result_t route_chat_request(RouterHandles *handle,
                                         const char *request_json,
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

bool is_disaggregated(RouterHandles *handle);
*/
import "C"

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"unsafe"

	log "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	rc "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	schedtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	PluginName            = "dynamo-kv-scorer"
	KVAwareScorerType     = "kv-aware-scorer"
	WorkerIDHeader        = "x-worker-instance-id"
	PrefillWorkerIDHeader = "x-prefill-instance-id"
	RoutingModeHeader     = "x-dynamo-routing-mode"
	// EnableLocalUpdatesHeader controls router bookkeeping in the Dynamo frontend.
	// Set to "false" for GAIE Stage 2 so the EPP handles bookkeeping via C FFI.
	EnableLocalUpdatesHeader = "x-enable-local-updates"

	// stateKey is the key used to store routing state in PluginState
	stateKey = "dynamo-routing-state"
)

// --------------------------- config / env ---------------------------

var warmupOnce sync.Once
var warmupErr error

type params struct{}

// DynamoRoutingState holds routing information passed from Score() to PreRequest().
// This is stored in PluginState keyed by request ID.
type DynamoRoutingState struct {
	WorkerID        string
	PrefillWorkerID string
	// TokenData holds the token IDs from the router.
	// Currently unused but stored for future implementation where tokens
	// may be passed to the worker via request body instead of headers.
	TokenData []int64
}

// Clone implements plugins.StateData interface.
func (s *DynamoRoutingState) Clone() plugins.StateData {
	if s == nil {
		return nil
	}
	clone := &DynamoRoutingState{
		WorkerID:        s.WorkerID,
		PrefillWorkerID: s.PrefillWorkerID,
	}
	if s.TokenData != nil {
		clone.TokenData = make([]int64, len(s.TokenData))
		copy(clone.TokenData, s.TokenData)
	}
	return clone
}

type KVAwareScorer struct {
	typedName      plugins.TypedName
	pluginState    *plugins.PluginState
	firstTokenSeen sync.Map // map[requestID]bool - tracks which requests have received first token
}

var _ plugins.Plugin = (*KVAwareScorer)(nil)
var _ framework.Scorer = (*KVAwareScorer)(nil)
var _ rc.PreRequest = (*KVAwareScorer)(nil)
var _ rc.ResponseStreaming = (*KVAwareScorer)(nil)
var _ rc.ResponseComplete = (*KVAwareScorer)(nil)

func NewKVAwareScorer(ctx context.Context) *KVAwareScorer {
	return &KVAwareScorer{
		typedName:   plugins.TypedName{Type: KVAwareScorerType, Name: PluginName},
		pluginState: plugins.NewPluginState(ctx),
	}
}

func (k *KVAwareScorer) WithName(name string) *KVAwareScorer { k.typedName.Name = name; return k }

func KVAwareScorerFactory(name string, raw json.RawMessage, handle plugins.Handle) (plugins.Plugin, error) {
	p := params{}
	_ = json.Unmarshal(raw, &p)

	s := NewKVAwareScorer(handle.Context()).WithName(name)

	// one-time FFI init (runtime + persistent pipeline)
	warmupOnce.Do(func() {
		defer func() {
			if r := recover(); r != nil {
				warmupErr = fmt.Errorf("Dynamo configuration error: %v", r)
			}
		}()
		warmupErr = initFFI()
	})
	if warmupErr != nil {
		return nil, fmt.Errorf("Dynamo FFI init for the Router failed: %w", warmupErr)
	}

	return s, nil
}

func (k *KVAwareScorer) TypedName() plugins.TypedName { return k.typedName }

// --------------------------- FFI integration ---------------------------

var (
	ffiOnce sync.Once
	ffiErr  error

	ffiNamespace     string
	ffiComponent     string
	ffiModel         string
	ffiKvBlockSize   uint32
	ffiEnforceDisagg bool

	routerInitialized bool

	// Router handles (owned on the Rust side, opaque here)
	routerHandles      *C.struct_RouterHandles
	routerHandlesMutex sync.RWMutex
)

func loadDynamoConfig() {
	ffiNamespace = getEnvOrDefault("DYN_NAMESPACE", "vllm-agg")
	ffiComponent = "backend" // This is not the same as DYN_COMPONENT=epp (in this case)
	ffiModel = getEnvOrDefault("DYN_MODEL", "Qwen/Qwen3-0.6B")
	ffiEnforceDisagg = getEnvBoolOrDefault("DYN_ENFORCE_DISAGG", false)

	kvBlockSizeStr := os.Getenv("DYN_KV_BLOCK_SIZE")
	if kvBlockSizeStr == "" {
		panic("DYN_KV_BLOCK_SIZE is required and must match the model card's kv_cache_block_size")
	}
	var tmp int64
	if n, err := fmt.Sscanf(kvBlockSizeStr, "%d", &tmp); err != nil || n != 1 {
		panic(fmt.Sprintf("DYN_KV_BLOCK_SIZE='%s' is not a valid integer", kvBlockSizeStr))
	}
	ffiKvBlockSize = uint32(tmp)
	if ffiKvBlockSize < 16 || ffiKvBlockSize > 8192 {
		panic(fmt.Sprintf("DYN_KV_BLOCK_SIZE=%d outside [16,8192]", ffiKvBlockSize))
	}
	if (ffiKvBlockSize & (ffiKvBlockSize - 1)) != 0 {
		panic(fmt.Sprintf("DYN_KV_BLOCK_SIZE=%d must be a power of 2", ffiKvBlockSize))
	}
	fmt.Printf("Dynamo KV Scorer: Loaded DYN_KV_BLOCK_SIZE=%d\n", ffiKvBlockSize)
}

func getEnvOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func getEnvInt64OrDefault(key string, def int64) int64 {
	if v := os.Getenv(key); v != "" {
		var p int64
		if n, err := fmt.Sscanf(v, "%d", &p); err == nil && n == 1 {
			return p
		}
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

// initFFI: initialize router handles using the new Router bindings.
func initFFI() error {
	ffiOnce.Do(func() {
		loadDynamoConfig()

		ns := C.CString(ffiNamespace)
		cm := C.CString(ffiComponent)
		model := C.CString(ffiModel)
		defer C.free(unsafe.Pointer(ns))
		defer C.free(unsafe.Pointer(cm))
		defer C.free(unsafe.Pointer(model))

		// Create router handles
		routerHandlesMutex.Lock()
		defer routerHandlesMutex.Unlock()

		waitForPrefill := getEnvBoolOrDefault("DYN_WAIT_FOR_PREFILL", false)
		prefillTimeoutSecs := uint32(getEnvInt64OrDefault("DYN_PREFILL_TIMEOUT_SEC", 300))

		rc := C.create_routers(
			ns,
			cm,
			model,
			C.uint32_t(ffiKvBlockSize),
			C.bool(ffiEnforceDisagg),
			C.bool(waitForPrefill),
			C.uint32_t(prefillTimeoutSecs),
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

// --------------------------- scoring ---------------------------

func (k *KVAwareScorer) Score(
	ctx context.Context,
	cycleState *schedtypes.CycleState,
	req *schedtypes.LLMRequest,
	pods []schedtypes.Pod,
) map[schedtypes.Pod]float64 {
	logger := log.FromContext(ctx)

	workerID, prefillWorkerID, tokenData, err := k.callDynamoRouter(ctx, req)
	if err != nil {
		logger.V(logutil.DEFAULT).Error(err, "Dynamo call failed; proceeding without worker id")
	} else if workerID != "" {
		logger.V(logutil.DEFAULT).Info(
			"Dynamo router selected worker",
			"workerID", workerID,
			"prefillWorkerID", prefillWorkerID,
			"tokenDataCount", len(tokenData),
		)

		// Store in request headers
		if req.Headers == nil {
			req.Headers = map[string]string{}
		}
		req.Headers[WorkerIDHeader] = workerID

		// Disable local updates in the Dynamo frontend router.
		// EPP handles bookkeeping via C FFI (add_request, mark_prefill_complete, free_request).
		req.Headers[EnableLocalUpdatesHeader] = "false"

		// Set routing mode and prefill worker ID based on disaggregated vs aggregated
		if prefillWorkerID != "" && prefillWorkerID != workerID {
			// Disaggregated mode: separate prefill and decode workers
			req.Headers[RoutingModeHeader] = "disaggregated"
			req.Headers[PrefillWorkerIDHeader] = prefillWorkerID
		} else {
			// Aggregated mode: single worker handles both prefill and decode
			req.Headers[RoutingModeHeader] = "aggregated"
		}

		// Store routing state for PreRequest to register with router bookkeeping.
		// This is the correct place to store state - PreRequest is called AFTER
		// scheduling is finalized, ensuring we only register committed requests.
		if req.RequestId != "" {
			routingState := &DynamoRoutingState{
				WorkerID:        workerID,
				PrefillWorkerID: prefillWorkerID,
				// TokenData is stored for future use. Currently not passed to workers
				// via headers (too large). May be passed via request body in future.
				TokenData: tokenData,
			}
			k.pluginState.Write(req.RequestId, plugins.StateKey(stateKey), routingState)
		}
	}

	out := make(map[schedtypes.Pod]float64, len(pods))
	for _, p := range pods {
		out[p] = 1.0
	}
	return out
}

// PreRequest is called after scheduling is finalized and before the request is sent to the worker.
// This is the place to register the request with the Dynamo router's bookkeeping,
// as we know the request WILL be dispatched (avoiding phantom bookkeeping entries).
func (k *KVAwareScorer) PreRequest(
	ctx context.Context,
	request *schedtypes.LLMRequest,
	schedulingResult *schedtypes.SchedulingResult,
) {
	logger := log.FromContext(ctx)

	if request == nil || request.RequestId == "" {
		logger.V(logutil.DEBUG).Info("PreRequest: no request ID, skipping router bookkeeping")
		return
	}

	// Read and delete the routing state stored by Score()
	state, err := plugins.ReadPluginStateKey[*DynamoRoutingState](
		k.pluginState, request.RequestId, plugins.StateKey(stateKey),
	)
	k.pluginState.Delete(request.RequestId) // Clean up state after reading

	if err != nil {
		// No state found means Score() didn't store routing info (e.g., router call failed)
		logger.V(logutil.DEBUG).Info("PreRequest: no routing state found, skipping router bookkeeping",
			"requestID", request.RequestId)
		return
	}

	// Register request with router bookkeeping now that scheduling is committed
	if addErr := k.callAddRequest(ctx, request.RequestId, state.TokenData, state.WorkerID, state.PrefillWorkerID); addErr != nil {
		logger.V(logutil.DEFAULT).Error(addErr, "PreRequest: failed to add request to router bookkeeping",
			"requestID", request.RequestId)
		return
	}

	logger.V(logutil.VERBOSE).Info("PreRequest: registered request with router bookkeeping",
		"requestID", request.RequestId,
		"workerID", state.WorkerID,
		"prefillWorkerID", state.PrefillWorkerID,
		"tokenCount", len(state.TokenData),
	)
}

// ResponseStreaming is called for each chunk of a streaming response.
// On the first token, it marks prefill as complete in the Dynamo router's bookkeeping.
func (k *KVAwareScorer) ResponseStreaming(
	ctx context.Context,
	request *schedtypes.LLMRequest,
	response *rc.Response,
	targetPod *backend.Pod,
) {
	if request == nil || request.RequestId == "" {
		return
	}

	// Check if we've already seen the first token for this request
	// LoadOrStore returns (value, loaded) - if loaded is false, this is the first time
	if _, alreadySeen := k.firstTokenSeen.LoadOrStore(request.RequestId, true); !alreadySeen {
		// This is the first token - mark prefill as complete
		logger := log.FromContext(ctx)
		if err := CallMarkPrefillComplete(request.RequestId); err != nil {
			logger.V(logutil.DEFAULT).Error(err, "ResponseStreaming: failed to mark prefill complete",
				"requestID", request.RequestId)
			return
		}
		logger.V(logutil.VERBOSE).Info("ResponseStreaming: marked prefill complete (first token received)",
			"requestID", request.RequestId)
	}
}

// ResponseComplete is called after the complete response is sent to the client.
// It cleans up the router bookkeeping state for the completed request by calling
// free_request to release resources associated with the request.
func (k *KVAwareScorer) ResponseComplete(
	ctx context.Context,
	request *schedtypes.LLMRequest,
	response *rc.Response,
	targetPod *backend.Pod,
) {
	logger := log.FromContext(ctx)

	if request == nil {
		logger.V(logutil.DEBUG).Info("ResponseComplete: request is nil, skipping cleanup")
		return
	}

	requestID := request.RequestId
	if requestID == "" {
		logger.V(logutil.DEBUG).Info("ResponseComplete: no request ID, skipping cleanup")
		return
	}

	// Clean up the first token tracking map
	k.firstTokenSeen.Delete(requestID)

	// Call the dynamo router to free the request bookkeeping
	if err := callFreeRequestInternal(requestID); err != nil {
		logger.V(logutil.DEFAULT).Error(err, "ResponseComplete: failed to free request",
			"requestID", requestID)
		return
	}

	logger.V(logutil.VERBOSE).Info("ResponseComplete: freed request from router",
		"requestID", requestID)
}

// --------------------------- router call ---------------------------

func (k *KVAwareScorer) callDynamoRouter(
	ctx context.Context,
	req *schedtypes.LLMRequest,
) (workerID string, prefillWorkerID string, tokenData []int64, err error) {
	logger := log.FromContext(ctx)

	if err := initFFI(); err != nil {
		logger.V(logutil.DEFAULT).Error(err, "FFI init failed")
		return "", "", nil, err
	}
	if !routerInitialized {
		return "", "", nil, fmt.Errorf("dynamo router not initialized")
	}

	routerHandlesMutex.RLock()
	currentRouter := routerHandles
	routerHandlesMutex.RUnlock()

	if currentRouter == nil {
		return "", "", nil, fmt.Errorf("dynamo router handles not created")
	}

	// Build OpenAI-compatible JSON request from the GAIE LLMRequest structure
	requestBody := buildOpenAIRequest(req)
	requestJSON, jsonErr := json.Marshal(requestBody)
	if jsonErr != nil {
		logger.V(logutil.DEFAULT).Error(jsonErr, "Failed to marshal OpenAI request")
		return "", "", nil, fmt.Errorf("marshal OpenAI request: %w", jsonErr)
	}
	cRequestJSON := C.CString(string(requestJSON))
	defer C.free(unsafe.Pointer(cRequestJSON))

	// Output result
	var result C.CRoutingResult

	// Call the router
	rc := C.route_chat_request(currentRouter, cRequestJSON, &result)
	if rc != C.QUERY_ROUTER_OK {
		return "", "", nil, fmt.Errorf("route_chat_request failed with code %d", rc)
	}

	// Copy tokens into Go memory
	count := int(result.token_count)
	var tokens64 []int64
	// TODO: Re-enable when tokens are enabled for request body passthrough
	// if count > 0 && result.token_ids != nil {
	// 	src := unsafe.Slice((*uint32)(unsafe.Pointer(result.token_ids)), count)
	// 	tokens64 = make([]int64, count)
	// 	for i := 0; i < count; i++ {
	// 		tokens64[i] = int64(src[i])
	// 	}
	// }

	// Free the routing result
	C.free_routing_result(&result)

	workerIDStr := fmt.Sprintf("%d", uint64(result.decode_worker_id))
	prefillWorkerIDStr := ""
	if result.is_disaggregated {
		prefillWorkerIDStr = fmt.Sprintf("%d", uint64(result.prefill_worker_id))
	}
	logger.V(logutil.DEFAULT).Info("Worker selection completed",
		"workerID", workerIDStr, "prefillWorkerID", prefillWorkerIDStr,
		"isDisaggregated", result.is_disaggregated, "tokenCount", count,
		"dpRank", result.decode_dp_rank, "overlapBlocks", result.decode_overlap_blocks)

	return workerIDStr, prefillWorkerIDStr, tokens64, nil
}

// buildOpenAIRequest constructs an OpenAI-compatible request from the GAIE LLMRequest structure.
// Preserves message roles for correct chat template application and tokenization.
func buildOpenAIRequest(req *schedtypes.LLMRequest) map[string]any {
	requestBody := make(map[string]any)

	// Preserve the original message structure for correct chat template application
	if req != nil && req.Body != nil {
		if req.Body.ChatCompletions != nil && len(req.Body.ChatCompletions.Messages) > 0 {
			messages := make([]map[string]any, 0, len(req.Body.ChatCompletions.Messages))
			for _, msg := range req.Body.ChatCompletions.Messages {
				messages = append(messages, map[string]any{
					"role":    msg.Role,
					"content": msg.Content.PlainText(),
				})
			}
			requestBody["messages"] = messages
		} else if req.Body.Completions != nil && req.Body.Completions.Prompt != "" {
			// Legacy completions format - wrap as single user message
			requestBody["messages"] = []map[string]any{
				{"role": "user", "content": req.Body.Completions.Prompt},
			}
		} else {
			requestBody["messages"] = []map[string]any{{"role": "user", "content": "default prompt"}}
		}
	} else {
		requestBody["messages"] = []map[string]any{{"role": "user", "content": "default prompt"}}
	}

	if req != nil && strings.TrimSpace(req.TargetModel) != "" {
		requestBody["model"] = req.TargetModel
	} else {
		requestBody["model"] = ffiModel
	}
	return requestBody
}

// --------------------------- router bookkeeping ---------------------------

// callAddRequest registers a request with the router's bookkeeping.
// This should be called after worker selection to track active requests.
func (k *KVAwareScorer) callAddRequest(
	ctx context.Context,
	requestID string,
	tokenData []int64,
	workerID string,
	prefillWorkerID string,
) error {
	logger := log.FromContext(ctx)

	if !routerInitialized {
		return fmt.Errorf("dynamo router not initialized")
	}

	routerHandlesMutex.RLock()
	currentRouter := routerHandles
	routerHandlesMutex.RUnlock()

	if currentRouter == nil {
		return fmt.Errorf("dynamo router handles not created")
	}

	// Parse worker ID (use decode worker for bookkeeping in disagg mode)
	var workerIDUint uint64
	if _, err := fmt.Sscanf(workerID, "%d", &workerIDUint); err != nil {
		return fmt.Errorf("invalid worker ID: %s", workerID)
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
		currentRouter,
		cRequestID,
		cTokens,
		C.size_t(len(tokens)),
		C.uint64_t(workerIDUint),
		C.uint32_t(0), // dp_rank = 0 for now
	)

	if rc != C.QUERY_ROUTER_OK {
		return fmt.Errorf("add_request failed with code %d", rc)
	}

	logger.V(logutil.VERBOSE).Info("Added request to router bookkeeping",
		"requestID", requestID, "workerID", workerID, "tokenCount", len(tokens))
	return nil
}

// CallMarkPrefillComplete marks prefill as completed for a request.
// Exported for use by response handlers.
func CallMarkPrefillComplete(requestID string) error {
	if !routerInitialized {
		return fmt.Errorf("dynamo router not initialized")
	}

	routerHandlesMutex.RLock()
	currentRouter := routerHandles
	routerHandlesMutex.RUnlock()

	if currentRouter == nil {
		return fmt.Errorf("dynamo router handles not created")
	}

	cRequestID := C.CString(requestID)
	defer C.free(unsafe.Pointer(cRequestID))

	rc := C.mark_prefill_complete(currentRouter, cRequestID)
	if rc != C.QUERY_ROUTER_OK {
		return fmt.Errorf("mark_prefill_complete failed with code %d", rc)
	}
	return nil
}

// callFreeRequestInternal cleans up router state for a completed/cancelled request.
func callFreeRequestInternal(requestID string) error {
	if !routerInitialized {
		return fmt.Errorf("dynamo router not initialized")
	}

	routerHandlesMutex.RLock()
	currentRouter := routerHandles
	routerHandlesMutex.RUnlock()

	if currentRouter == nil {
		return fmt.Errorf("dynamo router handles not created")
	}

	cRequestID := C.CString(requestID)
	defer C.free(unsafe.Pointer(cRequestID))

	rc := C.free_request(currentRouter, cRequestID)
	if rc != C.QUERY_ROUTER_OK {
		return fmt.Errorf("free_request failed with code %d", rc)
	}
	return nil
}

// --------------------------- shutdown ---------------------------

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
