/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package vllm implements the native control boundary for externally managed vLLM Elastic EP.
// It requires the correlated operation API proposed in vllm-project/vllm#62082.
// These controls do not establish traffic admission, physical release, or runtime conformance.
package vllm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
	"unicode/utf8"
)

const (
	statusPath                = "/is_scaling_elastic_ep"
	scalePath                 = "/scale_elastic_ep"
	defaultObservationTimeout = 5 * time.Second
	defaultScaleTimeout       = 12 * time.Minute
	maximumResponseBytes      = 1 << 20
)

// Phase is the native epoch-scoped resize phase, not the controller's operation phase.
type Phase string

const (
	// PhaseIdle means the runtime has not recorded an actual resize.
	PhaseIdle Phase = "idle"
	// PhasePreparing means old ranks and joiners are preparing the requested topology.
	PhasePreparing Phase = "preparing"
	// PhaseCommitting means membership is switching; the effective size may be only last-known evidence.
	PhaseCommitting Phase = "committing"
	// PhaseCompleted means the topology and all old-rank completion barriers finished.
	PhaseCompleted Phase = "completed"
	// PhaseFailed means the native operation failed, not necessarily without a membership mutation.
	PhaseFailed Phase = "failed"
)

// Status preserves one API process's identity and the world's latest native operation evidence.
// EffectiveDataParallelSize is only a last-known size when TopologyKnown is false.
// vLLM retains only its latest operation; callers must durably consume its result before another resize.
type Status struct {
	InstanceID                string  `json:"instance_id"`
	OperationID               *string `json:"operation_id"`
	Epoch                     *string `json:"epoch"`
	Phase                     Phase   `json:"phase"`
	RequestedDataParallelSize *int32  `json:"requested_data_parallel_size"`
	EffectiveDataParallelSize int32   `json:"effective_data_parallel_size"`
	TopologyKnown             bool    `json:"topology_known"`
	Error                     *string `json:"error"`
}

// ScaleRequest carries the same operation and target to every old rank, with that rank's own instance precondition.
// Neither identifier is optional in the Dynamo integration, even though vLLM accepts legacy uncorrelated requests.
type ScaleRequest struct {
	OperationID         string `json:"operation_id"`
	ExpectedInstanceID  string `json:"expected_instance_id"`
	NewDataParallelSize int32  `json:"new_data_parallel_size"`
	DrainTimeout        int32  `json:"drain_timeout"`
}

// HTTPError preserves invocation evidence without interpreting an HTTP response as a definitive membership rejection.
// In particular, a conflict or timeout on one rank does not prove that other ranks cannot commit.
type HTTPError struct {
	Path       string
	StatusCode int
	Body       string
}

// Error describes an unsuccessful native control request.
func (e *HTTPError) Error() string {
	return fmt.Sprintf("vLLM %s returned HTTP %d: %s", e.Path, e.StatusCode, e.Body)
}

// Client addresses one native vLLM engine, directly or through its Dynamo worker.
// It does not address a load-balanced service.
// The immutable base URL is private so an observed process cannot silently be retargeted.
type Client struct {
	baseURL            url.URL
	httpClient         *http.Client
	routePrefix        string
	ObservationTimeout time.Duration
	ScaleTimeout       time.Duration
}

// NewClient constructs one rank's control client without mutating httpClient.
// httpClient must not be nil. The URL identifies the native API process, not a Dynamo wrapper.
func NewClient(rawBaseURL string, httpClient *http.Client) (*Client, error) {
	// Reject ambiguous destinations rather than following a redirect to another engine process.
	parsed, err := url.Parse(rawBaseURL)
	if err != nil {
		return nil, fmt.Errorf("parse vLLM control URL: %w", err)
	}
	if (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Hostname() == "" ||
		parsed.User != nil || parsed.Opaque != "" || parsed.RawQuery != "" || parsed.ForceQuery ||
		parsed.Fragment != "" || parsed.RawPath != "" || (parsed.Path != "" && parsed.Path != "/") {
		return nil, fmt.Errorf("vLLM control URL must identify an http(s) process endpoint without credentials, path, query, or fragment")
	}

	// Keep identity-bound native requests on this process instead of following redirects.
	transport := *httpClient
	transport.CheckRedirect = func(_ *http.Request, _ []*http.Request) error {
		return http.ErrUseLastResponse
	}
	return &Client{
		baseURL: *parsed, httpClient: &transport,
		ObservationTimeout: defaultObservationTimeout, ScaleTimeout: defaultScaleTimeout,
	}, nil
}

// NewWorkerClient addresses one Dynamo Python worker's control plane.
// httpClient must not be nil. The worker forwards the same native AsyncLLM
// operation and status; it is not a second membership transaction authority.
func NewWorkerClient(rawBaseURL string, httpClient *http.Client) (*Client, error) {
	// Resolve and freeze the endpoint exactly as for a directly addressed native API.
	control, err := NewClient(rawBaseURL, httpClient)
	if err != nil {
		return nil, err
	}
	control.routePrefix = "/engine/control"
	return control, nil
}

// Observe reads native status once; both vLLM control routes use POST.
// Missing correlation or topology fields are unsupported, never implicit zero values or known topology.
func (c *Client) Observe(ctx context.Context) (Status, error) {
	// Bound status reads independently of a resize, preserving any shorter caller deadline.
	if c.ObservationTimeout <= 0 {
		return Status{}, fmt.Errorf("vLLM observation timeout must be positive")
	}
	ctx, cancel := context.WithTimeout(ctx, c.ObservationTimeout)
	defer cancel()
	return c.call(ctx, statusPath, nil)
}

// Scale invokes one rank's absolute operation and checks correlation on a successful response.
// A transport, HTTP, or response-validation error leaves the outcome inconclusive: observe before replaying.
// Replays must use the same request, including the originally observed instance ID.
func (c *Client) Scale(ctx context.Context, target ScaleRequest) (Status, error) {
	// Validate the complete target before issuing any native mutation.
	if err := target.validate(); err != nil {
		return Status{}, err
	}
	if c.ScaleTimeout <= 0 {
		return Status{}, fmt.Errorf("vLLM scale timeout must be positive")
	}
	ctx, cancel := context.WithTimeout(ctx, c.ScaleTimeout)
	defer cancel()
	status, err := c.call(ctx, scalePath, target)
	if err != nil {
		return Status{}, err
	}

	// A target already equal to the current size is a no-op and may return the previous operation.
	// It is not evidence that this operation committed, even when the effective count happens to match.
	if status.InstanceID != target.ExpectedInstanceID || status.OperationID == nil ||
		*status.OperationID != target.OperationID || status.RequestedDataParallelSize == nil ||
		*status.RequestedDataParallelSize != target.NewDataParallelSize {
		return Status{}, fmt.Errorf("vLLM scale response does not match the requested operation, target, and process instance")
	}
	return status, nil
}

func (r ScaleRequest) validate() error {
	// Match the upstream identifier constraints without inventing an uncorrelated fallback.
	if !validIdentifier(r.OperationID) || !validIdentifier(r.ExpectedInstanceID) {
		return fmt.Errorf("vLLM operation and expected instance IDs must each contain 1–256 characters and not be blank")
	}
	if r.NewDataParallelSize <= 0 || r.DrainTimeout <= 0 {
		return fmt.Errorf("vLLM DP target and drain timeout must be positive")
	}
	return nil
}

func (c *Client) call(ctx context.Context, path string, target any) (Status, error) {
	// Native and worker transports retain the exact same evidence decoder.
	payload, err := c.request(ctx, path, target)
	if err != nil {
		return Status{}, err
	}
	var wire struct {
		Status
		TopologyKnown *bool `json:"topology_known"`
	}
	if err := json.Unmarshal(payload, &wire); err != nil {
		return Status{}, fmt.Errorf("decode vLLM %s response: %w", path, err)
	}
	if wire.TopologyKnown == nil {
		return Status{}, fmt.Errorf("vLLM %s lacks correlated topology evidence; the operation-status API is required", path)
	}
	wire.Status.TopologyKnown = *wire.TopologyKnown
	if err := wire.Status.validate(); err != nil {
		return Status{}, fmt.Errorf("invalid vLLM %s status: %w", path, err)
	}
	return wire.Status, nil
}

func (c *Client) request(ctx context.Context, path string, target any) ([]byte, error) {
	// Only route placement changes; both transports use identical native evidence.
	path = c.routePrefix + path

	// Encode the exact request; observations intentionally have no JSON payload.
	var body io.Reader
	if target != nil {
		encoded, err := json.Marshal(target)
		if err != nil {
			return nil, fmt.Errorf("encode vLLM %s request: %w", path, err)
		}
		body = bytes.NewReader(encoded)
	}
	endpoint := c.baseURL
	endpoint.Path = path
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint.String(), body)
	if err != nil {
		return nil, fmt.Errorf("build vLLM %s request: %w", path, err)
	}
	if target != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	response, err := c.httpClient.Do(request)
	if err != nil {
		return nil, fmt.Errorf("call vLLM %s: %w", path, err)
	}
	defer func() { _ = response.Body.Close() }()

	// Bound and fully decode the response so trailing garbage cannot become membership evidence.
	payload, err := io.ReadAll(io.LimitReader(response.Body, maximumResponseBytes+1))
	if err != nil {
		return nil, fmt.Errorf("read vLLM %s response: %w", path, err)
	}
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return nil, &HTTPError{
			Path: path, StatusCode: response.StatusCode,
			Body: strings.TrimSpace(string(payload[:min(len(payload), 4096)])),
		}
	}
	if len(payload) > maximumResponseBytes {
		return nil, fmt.Errorf("vLLM %s response exceeds %d bytes", path, maximumResponseBytes)
	}

	return payload, nil
}

func (s Status) validate() error {
	// A process lifetime and an effective count are required even while topology is unknown.
	if !validIdentifier(s.InstanceID) || s.EffectiveDataParallelSize <= 0 {
		return fmt.Errorf("a process instance ID and positive effective DP size are required")
	}
	if s.Phase == PhaseIdle {
		if s.OperationID != nil || s.Epoch != nil || s.RequestedDataParallelSize != nil || s.Error != nil || !s.TopologyKnown {
			return fmt.Errorf("idle status must describe a known topology without an operation")
		}
		return nil
	}

	// Preserve native phase and certainty separately: Failed is not synonymous with rejected.
	if s.OperationID == nil || !validIdentifier(*s.OperationID) || s.Epoch == nil ||
		!validIdentifier(*s.Epoch) || s.RequestedDataParallelSize == nil || *s.RequestedDataParallelSize <= 0 {
		return fmt.Errorf("operation status requires operation ID, epoch, and a positive requested DP size")
	}
	switch s.Phase {
	case PhasePreparing, PhaseCommitting:
		if s.Error != nil {
			return fmt.Errorf("an active operation cannot carry a terminal error")
		}
	case PhaseCompleted:
		if s.Error != nil || !s.TopologyKnown || s.EffectiveDataParallelSize != *s.RequestedDataParallelSize {
			return fmt.Errorf("completed status requires a known topology equal to the target and no error")
		}
	case PhaseFailed:
		if s.Error == nil || strings.TrimSpace(*s.Error) == "" {
			return fmt.Errorf("failed status requires a native error")
		}
	default:
		return fmt.Errorf("unknown operation phase %q", s.Phase)
	}
	return nil
}

func validIdentifier(value string) bool {
	return strings.TrimSpace(value) != "" && utf8.ValidString(value) && utf8.RuneCountInString(value) <= 256
}
