// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package profile

import (
	"errors"
	"strings"
	"testing"

	"github.com/go-logr/logr"
)

type captureSink struct {
	records [][]any
}

func (s *captureSink) Init(logr.RuntimeInfo) {}

func (s *captureSink) Enabled(int) bool { return true }

func (s *captureSink) Info(_ int, _ string, keysAndValues ...any) {
	s.records = append(s.records, append([]any(nil), keysAndValues...))
}

func (s *captureSink) Error(error, string, ...any) {}

func (s *captureSink) WithValues(...any) logr.LogSink { return s }

func (s *captureSink) WithName(string) logr.LogSink { return s }

func TestSpanGatingAndSchema(t *testing.T) {
	sink := &captureSink{}
	log := logr.New(sink)

	t.Setenv(Env, "0")
	Start(log, "snapshot-agent", "disabled").End()
	if len(sink.records) != 0 {
		t.Fatalf("disabled profile emitted fields: %v", sink.records)
	}

	t.Setenv(Env, "1")
	Start(log, "snapshot-agent", "ordinal_discovery", "pid", 123).EndStatus(nil)

	fields := recordFields(t, sink.records[0])
	for _, name := range []string{
		"event",
		"component",
		"phase",
		"kind",
		"wall_start_ns",
		"wall_end_ns",
		"duration_ns",
	} {
		if _, ok := fields[name]; !ok {
			t.Fatalf("profile record missing %q: %v", name, fields)
		}
	}
	if fields["event"] != "gms_snapshot_profile" ||
		fields["component"] != "snapshot-agent" ||
		fields["phase"] != "ordinal_discovery" {
		t.Fatalf("unexpected profile identity: %v", fields)
	}
	if fields["wall_end_ns"].(int64) < fields["wall_start_ns"].(int64) {
		t.Fatalf("wall endpoints are reversed: %v", fields)
	}
	if fields["duration_ns"].(int64) < 0 {
		t.Fatalf("negative duration: %v", fields)
	}
	if fields["pid"] != 123 || fields["status"] != "ok" {
		t.Fatalf("custom fields missing: %v", fields)
	}
}

func TestOperationCorrelationAndErrorStatus(t *testing.T) {
	t.Setenv(Env, "1")
	sink := &captureSink{}
	log := logr.New(sink)
	operation := NewOperation(
		"restore",
		"checkpoint_id", "checkpoint-123",
		"pod", "worker-0",
		"namespace", "inference",
		"container", "main",
	)

	operation.Start(log, "snapshot-agent", "restore_total").EndStatus(nil)
	operation.Start(log, "snapshot-agent", "manifest_read").
		EndStatus(errors.New("sensitive backend detail"))

	if len(sink.records) != 2 {
		t.Fatalf("got %d profile records, want 2", len(sink.records))
	}
	first := recordFields(t, sink.records[0])
	second := recordFields(t, sink.records[1])
	if first["operation_id"] == "" || first["operation_id"] != second["operation_id"] {
		t.Fatalf("operation IDs are not stable: first=%v second=%v", first, second)
	}
	for _, fields := range []map[string]any{first, second} {
		if fields["operation"] != "restore" ||
			fields["checkpoint_id"] != "checkpoint-123" ||
			fields["pod"] != "worker-0" ||
			fields["namespace"] != "inference" ||
			fields["container"] != "main" {
			t.Fatalf("operation identity missing: %v", fields)
		}
	}
	if first["status"] != "ok" {
		t.Fatalf("success status missing: %v", first)
	}
	if second["status"] != "error" || second["error_category"] != "internal" {
		t.Fatalf("categorized error status missing: %v", second)
	}
	for _, value := range sink.records[1] {
		if strings.Contains(strings.ToLower(toString(value)), "sensitive") {
			t.Fatalf("error text leaked into profile record: %v", sink.records[1])
		}
	}
}

func recordFields(t *testing.T, record []any) map[string]any {
	t.Helper()
	if len(record)%2 != 0 {
		t.Fatalf("profile record has odd field count: %v", record)
	}
	fields := make(map[string]any, len(record)/2)
	for i := 0; i < len(record); i += 2 {
		key, ok := record[i].(string)
		if !ok {
			t.Fatalf("profile key is not a string: %v", record[i])
		}
		fields[key] = record[i+1]
	}
	return fields
}

func toString(value any) string {
	if value == nil {
		return ""
	}
	if text, ok := value.(string); ok {
		return text
	}
	return ""
}
