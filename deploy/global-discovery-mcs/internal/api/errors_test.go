// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package api

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	mcsapi "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/view"
)

func TestStoreErrorMapping(t *testing.T) {
	cases := []struct {
		err    error
		code   codes.Code
		reason string
	}{
		{store.ErrUnavailable, codes.Unavailable, mcsapi.ReasonStoreUnavailable},
		{fmt.Errorf("wrapped: %w", store.ErrLeaseEpochMismatch), codes.Aborted, mcsapi.ReasonLeaseEpochMismatch},
		{store.ErrTooLarge, codes.InvalidArgument, mcsapi.ReasonTooManyEndpoints},
		{store.ErrStaleGeneration{Current: 3}, codes.FailedPrecondition, mcsapi.ReasonStaleGeneration},
		{errors.New("boom"), codes.Internal, mcsapi.ReasonInternal},
	}
	for _, tc := range cases {
		requireError(t, storeError(tc.err, "op"), tc.code, tc.reason)
	}
	assert.Equal(t, codes.Canceled, status.Code(storeError(context.Canceled, "op")))
	assert.Equal(t, codes.DeadlineExceeded, status.Code(storeError(context.DeadlineExceeded, "op")))
}

func TestWatchErrorMapping(t *testing.T) {
	assert.NoError(t, watchError(nil))
	requireError(t, watchError(view.ErrTooManyWatchers), codes.ResourceExhausted, mcsapi.ReasonTooManyWatchers)
	requireError(t, watchError(view.ErrShuttingDown), codes.Unavailable, mcsapi.ReasonShuttingDown)
	assert.Equal(t, codes.Canceled, status.Code(watchError(context.Canceled)))
	assert.Equal(t, codes.Unavailable, status.Code(watchError(status.Error(codes.Unavailable, "transport"))))
	requireError(t, watchError(errors.New("boom")), codes.Internal, mcsapi.ReasonInternal)
}

func TestInvalidArgumentMessage(t *testing.T) {
	var v validator
	v.add("a", "is bad")
	v.add("b", "is bad")
	st, _ := status.FromError(v.err())
	assert.Equal(t, "invalid request: a: is bad (and 1 more)", st.Message())
}

func TestAddressType(t *testing.T) {
	assert.Equal(t, mcsv1.AddressType_ADDRESS_TYPE_UNSPECIFIED, addressType(nil))
	assert.Equal(t, mcsv1.AddressType_ADDRESS_TYPE_IPV4, addressType([]string{"10.0.0.1"}))
	assert.Equal(t, mcsv1.AddressType_ADDRESS_TYPE_IPV6, addressType([]string{"fd00::1"}))
	assert.Equal(t, mcsv1.AddressType_ADDRESS_TYPE_FQDN, addressType([]string{"pod.example"}))
}

func TestGeneration(t *testing.T) {
	gen, assigned := generation(7, 3, t0)
	assert.Equal(t, uint64(7), gen)
	assert.False(t, assigned)
	gen, assigned = generation(0, 3, t0)
	assert.Equal(t, uint64(t0.UnixMilli()), gen)
	assert.True(t, assigned)
}
