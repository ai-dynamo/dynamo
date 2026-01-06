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

// Body Transformer - Envoy ext_proc service for nvext injection
//
// This service reads routing headers set by the Dynamo EPP and injects
// the nvext field into the request body for Dynamo inference backends.
//
// Headers read:
//   - x-worker-instance-id: Primary worker ID (required)
//   - x-prefiller-host-port: Prefill worker ID (disaggregated mode)
//   - x-dynamo-routing-mode: "aggregated" or "disaggregated"
//
// Body injection:
//   - Aggregated: {"nvext": {"backend_instance_id": <worker_id>}}
//   - Disaggregated: {"nvext": {"prefill_worker_id": <prefill>, "decode_worker_id": <decode>}}
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"strconv"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const (
	WorkerIDHeader        = "x-worker-instance-id"
	PrefillWorkerIDHeader = "x-prefiller-host-port"
	RoutingModeHeader     = "x-dynamo-routing-mode"
)

type server struct {
	extProcPb.UnimplementedExternalProcessorServer
}

func (s *server) Process(srv extProcPb.ExternalProcessor_ProcessServer) error {
	ctx := srv.Context()

	// Store headers across the request lifecycle
	var workerID string
	var prefillWorkerID string
	var routingMode string

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		req, err := srv.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return status.Errorf(codes.Unknown, "cannot receive stream request: %v", err)
		}

		var resp *extProcPb.ProcessingResponse

		switch v := req.Request.(type) {
		case *extProcPb.ProcessingRequest_RequestHeaders:
			// Extract routing headers
			for _, header := range v.RequestHeaders.Headers.Headers {
				switch header.Key {
				case WorkerIDHeader:
					workerID = string(header.RawValue)
				case PrefillWorkerIDHeader:
					prefillWorkerID = string(header.RawValue)
				case RoutingModeHeader:
					routingMode = string(header.RawValue)
				}
			}

			log.Printf("Headers received: worker_id=%s, prefill_id=%s, mode=%s",
				workerID, prefillWorkerID, routingMode)

			// Continue to body processing
			resp = &extProcPb.ProcessingResponse{
				Response: &extProcPb.ProcessingResponse_RequestHeaders{
					RequestHeaders: &extProcPb.HeadersResponse{},
				},
			}

		case *extProcPb.ProcessingRequest_RequestBody:
			body := v.RequestBody.Body

			if workerID != "" && v.RequestBody.EndOfStream {
				// Inject nvext into body
				modifiedBody, err := injectNvext(body, workerID, prefillWorkerID, routingMode)
				if err != nil {
					log.Printf("Error injecting nvext: %v", err)
					// Pass through original body on error
					resp = &extProcPb.ProcessingResponse{
						Response: &extProcPb.ProcessingResponse_RequestBody{
							RequestBody: &extProcPb.BodyResponse{},
						},
					}
				} else {
					log.Printf("Injected nvext, body size: %d -> %d", len(body), len(modifiedBody))
					resp = &extProcPb.ProcessingResponse{
						Response: &extProcPb.ProcessingResponse_RequestBody{
							RequestBody: &extProcPb.BodyResponse{
								Response: &extProcPb.CommonResponse{
									BodyMutation: &extProcPb.BodyMutation{
										Mutation: &extProcPb.BodyMutation_Body{
											Body: modifiedBody,
										},
									},
								},
							},
						},
					}
				}
			} else {
				// No worker ID or not end of stream - pass through
				resp = &extProcPb.ProcessingResponse{
					Response: &extProcPb.ProcessingResponse_RequestBody{
						RequestBody: &extProcPb.BodyResponse{},
					},
				}
			}

		case *extProcPb.ProcessingRequest_RequestTrailers:
			resp = &extProcPb.ProcessingResponse{
				Response: &extProcPb.ProcessingResponse_RequestTrailers{
					RequestTrailers: &extProcPb.TrailersResponse{},
				},
			}

		case *extProcPb.ProcessingRequest_ResponseHeaders:
			resp = &extProcPb.ProcessingResponse{
				Response: &extProcPb.ProcessingResponse_ResponseHeaders{
					ResponseHeaders: &extProcPb.HeadersResponse{},
				},
			}

		case *extProcPb.ProcessingRequest_ResponseBody:
			resp = &extProcPb.ProcessingResponse{
				Response: &extProcPb.ProcessingResponse_ResponseBody{
					ResponseBody: &extProcPb.BodyResponse{},
				},
			}

		default:
			log.Printf("Unknown request type: %T", v)
			continue
		}

		if resp != nil {
			if err := srv.Send(resp); err != nil {
				return status.Errorf(codes.Unknown, "failed to send response: %v", err)
			}
		}
	}
}

// injectNvext parses the JSON body and injects the nvext field
func injectNvext(body []byte, workerID, prefillWorkerID, routingMode string) ([]byte, error) {
	// Parse body as generic JSON
	var bodyMap map[string]interface{}
	if err := json.Unmarshal(body, &bodyMap); err != nil {
		return nil, fmt.Errorf("failed to parse body as JSON: %w", err)
	}

	// Parse worker ID
	workerIDInt, err := strconv.ParseInt(workerID, 10, 64)
	if err != nil {
		return nil, fmt.Errorf("invalid worker ID '%s': %w", workerID, err)
	}

	// Build nvext based on routing mode
	nvext := make(map[string]interface{})

	if routingMode == "disaggregated" && prefillWorkerID != "" {
		prefillIDInt, err := strconv.ParseInt(prefillWorkerID, 10, 64)
		if err != nil {
			// Fall back to aggregated mode if prefill ID is invalid
			nvext["backend_instance_id"] = workerIDInt
		} else {
			nvext["prefill_worker_id"] = prefillIDInt
			nvext["decode_worker_id"] = workerIDInt
		}
	} else {
		// Aggregated mode (default)
		nvext["backend_instance_id"] = workerIDInt
	}

	// Inject nvext into body
	bodyMap["nvext"] = nvext

	// Marshal back to JSON
	return json.Marshal(bodyMap)
}

func main() {
	port := flag.Int("port", 9003, "gRPC server port")
	flag.Parse()

	addr := fmt.Sprintf(":%d", *port)
	lis, err := net.Listen("tcp", addr)
	if err != nil {
		log.Fatalf("Failed to listen on %s: %v", addr, err)
	}

	grpcServer := grpc.NewServer()
	extProcPb.RegisterExternalProcessorServer(grpcServer, &server{})

	log.Printf("Body Transformer ext_proc server listening on %s", addr)
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("Failed to serve: %v", err)
	}
}

