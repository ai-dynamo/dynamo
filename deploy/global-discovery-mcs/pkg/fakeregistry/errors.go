// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package fakeregistry

import (
	"fmt"
	"strconv"

	"google.golang.org/genproto/googleapis/rpc/errdetails"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/protoadapt"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
)

// apiError returns a status error carrying an ErrorInfo with the registry
// domain and reason, followed by extra details.
func apiError(code codes.Code, reason, msg string, metadata map[string]string, extra ...protoadapt.MessageV1) error {
	st := status.New(code, msg)
	details := append([]protoadapt.MessageV1{&errdetails.ErrorInfo{
		Reason:   reason,
		Domain:   api.ErrorDomain,
		Metadata: metadata,
	}}, extra...)
	if withDetails, err := st.WithDetails(details...); err == nil {
		st = withDetails
	}
	return st.Err()
}

// errInvalid returns INVALID_ARGUMENT, reason INVALID_ARGUMENT, with a
// BadRequest naming the field, like the server.
func errInvalid(field, description string) error {
	return apiError(codes.InvalidArgument, api.ReasonInvalidArgument, fmt.Sprintf("invalid %s: %s", field, description), nil,
		&errdetails.BadRequest{
			FieldViolations: []*errdetails.BadRequest_FieldViolation{{Field: field, Description: description}},
		})
}

func errStaleGeneration(what string, got, current uint64) error {
	cur := strconv.FormatUint(current, 10)
	return apiError(codes.FailedPrecondition, api.ReasonStaleGeneration,
		fmt.Sprintf("%s generation %d does not exceed stored generation %d", what, got, current),
		map[string]string{api.MetadataCurrentGeneration: cur},
		&errdetails.PreconditionFailure{Violations: []*errdetails.PreconditionFailure_Violation{{
			Type:        api.ReasonStaleGeneration,
			Subject:     "generation",
			Description: "current generation " + cur,
		}}},
	)
}

func errEpochMismatch(what string, got, current uint64) error {
	return apiError(codes.Aborted, api.ReasonLeaseEpochMismatch,
		fmt.Sprintf("%s lease epoch %d is not the current epoch %d; register again", what, got, current), nil)
}

func errClusterNotRegistered(code codes.Code, clusterset, cluster string) error {
	return apiError(code, api.ReasonClusterNotRegistered,
		fmt.Sprintf("cluster %q is not registered in clusterset %q", cluster, clusterset), nil)
}

func errTooManyEndpoints(n int) error {
	return apiError(codes.InvalidArgument, api.ReasonTooManyEndpoints,
		fmt.Sprintf("%d endpoints exceed the limit of %d", n, api.MaxEndpointsPerExport), nil)
}

func errShuttingDown() error {
	return apiError(codes.Unavailable, api.ReasonShuttingDown, "fakeregistry: watch closed", nil)
}

// errNotFound returns NOT_FOUND, reason NOT_FOUND, for objects other than
// clusters, like the server.
func errNotFound(format string, args ...any) error {
	return apiError(codes.NotFound, api.ReasonNotFound, fmt.Sprintf(format, args...), nil)
}

// require returns INVALID_ARGUMENT for the first empty value, given as
// alternating field names and values.
func require(fieldsAndValues ...string) error {
	for i := 0; i+1 < len(fieldsAndValues); i += 2 {
		if fieldsAndValues[i+1] == "" {
			return errInvalid(fieldsAndValues[i], "required")
		}
	}
	return nil
}
