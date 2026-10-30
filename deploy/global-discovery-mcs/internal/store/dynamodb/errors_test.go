// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamodb

import (
	"context"
	"errors"
	"fmt"
	"net"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	ddbtypes "github.com/aws/aws-sdk-go-v2/service/dynamodb/types"
	"github.com/aws/smithy-go"
	"github.com/stretchr/testify/assert"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
)

func TestMapError(t *testing.T) {
	opErr := func(err error) error {
		return &smithy.OperationError{ServiceID: "DynamoDB", OperationName: "PutItem", Err: err}
	}
	tests := []struct {
		name string
		err  error
		is   []error
		not  []error
	}{
		{"canceled", opErr(context.Canceled), []error{context.Canceled}, []error{store.ErrUnavailable}},
		{"deadline", opErr(context.DeadlineExceeded), []error{context.DeadlineExceeded}, []error{store.ErrUnavailable}},
		{"already unavailable", fmt.Errorf("x: %w", store.ErrUnavailable), []error{store.ErrUnavailable}, nil},
		{"already too large", store.ErrTooLarge, []error{store.ErrTooLarge}, nil},
		{"item collection", opErr(&ddbtypes.ItemCollectionSizeLimitExceededException{Message: aws.String("10 GB")}), []error{store.ErrTooLarge}, []error{store.ErrUnavailable}},
		{"item size", opErr(apiError("ValidationException", "Item size has exceeded the maximum allowed size")), []error{store.ErrTooLarge}, nil},
		{"update size", opErr(apiError("ValidationException", "Item size to update has exceeded the maximum allowed size")), []error{store.ErrTooLarge}, nil},
		{"validation", opErr(apiError("ValidationException", "One or more parameter values were invalid")), nil, []error{store.ErrUnavailable, store.ErrTooLarge}},
		{"serialization", opErr(apiError("SerializationException", "bad")), nil, []error{store.ErrUnavailable}},
		{"throttled", opErr(&ddbtypes.ProvisionedThroughputExceededException{Message: aws.String("slow down")}), []error{store.ErrUnavailable}, nil},
		{"request limit", opErr(&ddbtypes.RequestLimitExceeded{Message: aws.String("slow down")}), []error{store.ErrUnavailable}, nil},
		{"throttling", opErr(apiError("ThrottlingException", "Rate exceeded")), []error{store.ErrUnavailable}, nil},
		{"internal", opErr(&ddbtypes.InternalServerError{Message: aws.String("oops")}), []error{store.ErrUnavailable}, nil},
		{"missing table", opErr(&ddbtypes.ResourceNotFoundException{Message: aws.String("no table")}), []error{store.ErrUnavailable}, nil},
		{"access denied", opErr(apiError("AccessDeniedException", "no")), []error{store.ErrUnavailable}, nil},
		{"transport", opErr(&net.OpError{Op: "dial", Net: "tcp", Err: errors.New("connection refused")}), []error{store.ErrUnavailable}, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := mapError("PutItem", tt.err)
			assert.ErrorIs(t, got, tt.err, "the cause stays in the chain")
			for _, want := range tt.is {
				assert.ErrorIs(t, got, want)
			}
			for _, not := range tt.not {
				assert.NotErrorIs(t, got, not)
			}
		})
	}
	assert.NoError(t, mapError("PutItem", nil))
}

func TestConditionHelpers(t *testing.T) {
	old := map[string]ddbtypes.AttributeValue{attrGen: numU(3)}
	got, ok := conditionFailed(fmt.Errorf("wrapped: %w", &ddbtypes.ConditionalCheckFailedException{Item: old}))
	assert.True(t, ok)
	assert.Equal(t, old, got)
	_, ok = conditionFailed(errors.New("other"))
	assert.False(t, ok)

	assert.ErrorIs(t, errContention, store.ErrUnavailable)
}
