// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package api

import (
	"context"
	"errors"
	"fmt"
	"strconv"

	"google.golang.org/genproto/googleapis/rpc/errdetails"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/protoadapt"

	mcsapi "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/view"
)

// apiError returns a google.rpc.Status error with an ErrorInfo detail in the
// mcs.dynamo.nvidia.com domain followed by details.
func apiError(code codes.Code, reason, msg string, metadata map[string]string, details ...protoadapt.MessageV1) error {
	st := status.New(code, msg)
	all := append([]protoadapt.MessageV1{&errdetails.ErrorInfo{Reason: reason, Domain: mcsapi.ErrorDomain, Metadata: metadata}}, details...)
	withDetails, err := st.WithDetails(all...)
	if err != nil {
		return st.Err()
	}
	return withDetails.Err()
}

// invalidArgument returns INVALID_ARGUMENT with a BadRequest naming the
// fields.
func invalidArgument(violations []*errdetails.BadRequest_FieldViolation) error {
	msg := fmt.Sprintf("invalid request: %s: %s", violations[0].GetField(), violations[0].GetDescription())
	if n := len(violations) - 1; n > 0 {
		msg += fmt.Sprintf(" (and %d more)", n)
	}
	return apiError(codes.InvalidArgument, mcsapi.ReasonInvalidArgument, msg, nil, &errdetails.BadRequest{FieldViolations: violations})
}

func tooManyEndpoints(n int) error {
	msg := fmt.Sprintf("endpoint set has %d endpoints; at most %d are allowed", n, mcsapi.MaxEndpointsPerExport)
	return apiError(codes.InvalidArgument, mcsapi.ReasonTooManyEndpoints, msg, nil, &errdetails.BadRequest{
		FieldViolations: []*errdetails.BadRequest_FieldViolation{{Field: "endpoints", Description: msg}},
	})
}

// staleGeneration returns FAILED_PRECONDITION STALE_GENERATION with the
// stored generation in the ErrorInfo metadata and a PreconditionFailure.
func staleGeneration(what string, attempted, current uint64) error {
	msg := fmt.Sprintf("%s generation %d does not exceed stored generation %d", what, attempted, current)
	return apiError(codes.FailedPrecondition, mcsapi.ReasonStaleGeneration, msg,
		map[string]string{mcsapi.MetadataCurrentGeneration: strconv.FormatUint(current, 10)},
		&errdetails.PreconditionFailure{Violations: []*errdetails.PreconditionFailure_Violation{{
			Type:        mcsapi.ReasonStaleGeneration,
			Subject:     "generation",
			Description: fmt.Sprintf("current generation is %d", current),
		}}})
}

func clusterNotRegistered(code codes.Code, key store.ClusterKey) error {
	return apiError(code, mcsapi.ReasonClusterNotRegistered,
		fmt.Sprintf("cluster %q is not registered in clusterset %q", key.Cluster, key.ClusterSet), nil)
}

func notFound(format string, args ...any) error {
	return apiError(codes.NotFound, mcsapi.ReasonNotFound, fmt.Sprintf(format, args...), nil)
}

// storeError maps a store error that the caller did not handle itself.
func storeError(err error, op string) error {
	var stale store.ErrStaleGeneration
	switch {
	case errors.Is(err, context.Canceled):
		return status.Error(codes.Canceled, op+": "+err.Error())
	case errors.Is(err, context.DeadlineExceeded):
		return status.Error(codes.DeadlineExceeded, op+": "+err.Error())
	case errors.Is(err, store.ErrUnavailable):
		return apiError(codes.Unavailable, mcsapi.ReasonStoreUnavailable, op+": store unavailable", nil)
	case errors.Is(err, store.ErrLeaseEpochMismatch):
		return apiError(codes.Aborted, mcsapi.ReasonLeaseEpochMismatch, op+": lease epoch is not current; register again", nil)
	case errors.Is(err, store.ErrTooLarge):
		return apiError(codes.InvalidArgument, mcsapi.ReasonTooManyEndpoints, op+": record too large", nil)
	case errors.As(err, &stale):
		return staleGeneration(op, 0, stale.Current)
	}
	return apiError(codes.Internal, mcsapi.ReasonInternal, op+": "+err.Error(), nil)
}

// watchError maps the error that ended a watch.
func watchError(err error) error {
	switch {
	case err == nil:
		return nil
	case errors.Is(err, view.ErrTooManyWatchers):
		return apiError(codes.ResourceExhausted, mcsapi.ReasonTooManyWatchers, "the replica serves the maximum number of watches; retry later or elsewhere", nil)
	case errors.Is(err, view.ErrShuttingDown):
		return apiError(codes.Unavailable, mcsapi.ReasonShuttingDown, "the replica is shutting down; reconnect", nil)
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		return status.FromContextError(err).Err()
	}
	if _, ok := status.FromError(err); ok {
		return err
	}
	return apiError(codes.Internal, mcsapi.ReasonInternal, "watch: "+err.Error(), nil)
}

// ShuttingDownError is the UNAVAILABLE SHUTTING_DOWN error of a call that
// reaches a draining replica.
func ShuttingDownError() error {
	return apiError(codes.Unavailable, mcsapi.ReasonShuttingDown, "the replica is shutting down; retry on another replica", nil)
}

// InternalError is the INTERNAL error of a call that failed unexpectedly,
// for example a recovered panic.
func InternalError(msg string) error {
	return apiError(codes.Internal, mcsapi.ReasonInternal, msg, nil)
}
