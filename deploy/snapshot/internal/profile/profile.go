// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package profile provides opt-in snapshot-agent timing records.
package profile

import (
	"context"
	"errors"
	"os"
	"time"

	"github.com/go-logr/logr"
	"github.com/google/uuid"
)

const Env = "GMS_SNAPSHOT_PROFILE"

var disabledSpan = &Span{}

// Operation carries fields shared by every span for one executor request.
type Operation struct {
	fields []any
}

// NewOperation creates a correlation scope with a unique operation ID.
func NewOperation(operation string, fields ...any) Operation {
	values := []any{
		"operation", operation,
		"operation_id", uuid.NewString(),
	}
	values = append(values, fields...)
	return Operation{fields: values}
}

// Start begins a span carrying this operation's correlation fields.
func (o Operation) Start(log logr.Logger, component, phase string, fields ...any) *Span {
	values := make([]any, 0, len(o.fields)+len(fields))
	values = append(values, o.fields...)
	values = append(values, fields...)
	return Start(log, component, phase, values...)
}

// Span records wall-clock endpoints and a monotonic duration.
type Span struct {
	enabled   bool
	log       logr.Logger
	component string
	phase     string
	fields    []any
	wallStart time.Time
	start     time.Time
}

// Start begins a span only when GMS_SNAPSHOT_PROFILE=1.
func Start(log logr.Logger, component, phase string, fields ...any) *Span {
	if os.Getenv(Env) != "1" {
		return disabledSpan
	}
	now := time.Now()
	return &Span{
		enabled:   true,
		log:       log,
		component: component,
		phase:     phase,
		fields:    append([]any(nil), fields...),
		wallStart: now,
		start:     now,
	}
}

// End emits one structured timing record.
func (s *Span) End(fields ...any) {
	if !s.enabled {
		return
	}
	wallEnd := time.Now()
	duration := wallEnd.Sub(s.start)
	values := []any{
		"event", "gms_snapshot_profile",
		"component", s.component,
		"phase", s.phase,
		"kind", "phase",
		"wall_start_ns", s.wallStart.UnixNano(),
		"wall_end_ns", wallEnd.UnixNano(),
		"duration_ns", duration.Nanoseconds(),
	}
	values = append(values, s.fields...)
	values = append(values, fields...)
	s.log.Info("GMS_SNAPSHOT_PROFILE", values...)
}

// EndStatus emits a success or categorized failure without recording the error text.
func (s *Span) EndStatus(err error, fields ...any) {
	if err == nil {
		s.End(append(fields, "status", "ok")...)
		return
	}
	s.End(append(fields, "status", "error", "error_category", errorCategory(err))...)
}

func errorCategory(err error) string {
	switch {
	case errors.Is(err, context.Canceled):
		return "context_canceled"
	case errors.Is(err, context.DeadlineExceeded):
		return "deadline_exceeded"
	case errors.Is(err, os.ErrNotExist):
		return "not_found"
	case errors.Is(err, os.ErrPermission):
		return "permission_denied"
	case errors.Is(err, os.ErrExist):
		return "already_exists"
	default:
		return "internal"
	}
}
