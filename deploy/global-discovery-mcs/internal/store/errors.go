// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package store

import (
	"errors"
	"fmt"
)

var (
	// ErrNotFound reports that a record does not exist.
	ErrNotFound = errors.New("store: not found")
	// ErrLeaseEpochMismatch reports a heartbeat with an epoch other than the
	// current lease epoch. The writer must register again.
	ErrLeaseEpochMismatch = errors.New("store: lease epoch mismatch")
	// ErrTooLarge reports a record above a store limit.
	ErrTooLarge = errors.New("store: record too large")
	// ErrUnavailable reports that the backend cannot be reached or the store
	// is closed. Callers may retry.
	ErrUnavailable = errors.New("store: unavailable")
)

// ErrStaleGeneration reports a conditional put whose generation does not
// exceed the stored one. Current is the stored generation. Match it with
// errors.As, or with errors.Is(err, ErrStaleGeneration{}) when Current does
// not matter.
type ErrStaleGeneration struct {
	Current uint64
}

func (e ErrStaleGeneration) Error() string {
	return fmt.Sprintf("store: stale generation, current generation is %d", e.Current)
}

// Is matches any ErrStaleGeneration regardless of Current.
func (e ErrStaleGeneration) Is(target error) bool {
	_, ok := target.(ErrStaleGeneration)
	return ok
}

// StaleGeneration returns the stored generation of an ErrStaleGeneration in
// err's chain.
func StaleGeneration(err error) (current uint64, ok bool) {
	if stale, ok := errors.AsType[ErrStaleGeneration](err); ok {
		return stale.Current, true
	}
	return 0, false
}
