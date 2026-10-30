// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamodb

import (
	"context"
	"errors"
	"fmt"
	"strings"

	ddbtypes "github.com/aws/aws-sdk-go-v2/service/dynamodb/types"
	"github.com/aws/smithy-go"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
)

// mapError converts an SDK error into the store's error contract:
//
//   - context cancellation and deadlines are returned as they are;
//   - an item above the size limit and an LSI item collection above 10 GB
//     are store.ErrTooLarge;
//   - request validation and serialization errors are programming errors
//     and are returned wrapped, without a sentinel;
//   - everything else (throttling, capacity, internal errors, a missing
//     table, denied access, transport and DNS failures) is
//     store.ErrUnavailable, so the API reports STORE_UNAVAILABLE and
//     callers retry.
//
// Conditional check failures are handled by each operation before
// mapError is called.
func mapError(op string, err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return fmt.Errorf("dynamodb: %s: %w", op, err)
	}
	if errors.Is(err, store.ErrUnavailable) || errors.Is(err, store.ErrTooLarge) {
		return err
	}
	if _, ok := errors.AsType[*ddbtypes.ItemCollectionSizeLimitExceededException](err); ok {
		return fmt.Errorf("dynamodb: %s: %w: %w", op, store.ErrTooLarge, err)
	}
	if apiErr, ok := errors.AsType[smithy.APIError](err); ok {
		switch apiErr.ErrorCode() {
		case "ValidationException":
			if isItemSizeError(apiErr.ErrorMessage()) {
				return fmt.Errorf("dynamodb: %s: %w: %w", op, store.ErrTooLarge, err)
			}
			return fmt.Errorf("dynamodb: %s: %w", op, err)
		case "SerializationException":
			return fmt.Errorf("dynamodb: %s: %w", op, err)
		}
	}
	return fmt.Errorf("dynamodb: %s: %w: %w", op, store.ErrUnavailable, err)
}

// isItemSizeError reports whether a ValidationException message is about
// the 400 KB item size limit, for example "Item size has exceeded the
// maximum allowed size" or "Item size to update has exceeded the maximum
// allowed size".
func isItemSizeError(msg string) bool {
	msg = strings.ToLower(msg)
	return strings.Contains(msg, "item size") && strings.Contains(msg, "exceeded")
}

// conditionFailed reports whether err is a failed condition of a single
// item write and returns the item the condition was evaluated against,
// nil when the item did not exist. The write must have asked for
// ReturnValuesOnConditionCheckFailure ALL_OLD.
func conditionFailed(err error) (map[string]ddbtypes.AttributeValue, bool) {
	var ccf *ddbtypes.ConditionalCheckFailedException
	if !errors.As(err, &ccf) {
		return nil, false
	}
	return ccf.Item, true
}

// errContention is returned when an optimistic write keeps losing to
// concurrent writers of the same item.
var errContention = fmt.Errorf("%w: too many concurrent writes to one item", store.ErrUnavailable)
