// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package executor

import (
	"context"
	"errors"
	"path/filepath"
	"testing"

	"github.com/go-logr/logr"
	specs "github.com/opencontainers/runtime-spec/specs-go"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/profile"
	snapshotruntime "github.com/ai-dynamo/dynamo/deploy/snapshot/internal/runtime"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

type executorProfileSink struct {
	records []map[string]any
}

func (s *executorProfileSink) Init(logr.RuntimeInfo) {}

func (s *executorProfileSink) Enabled(int) bool { return true }

func (s *executorProfileSink) Info(_ int, message string, keysAndValues ...any) {
	if message != "GMS_SNAPSHOT_PROFILE" {
		return
	}
	record := make(map[string]any, len(keysAndValues)/2)
	for i := 0; i < len(keysAndValues); i += 2 {
		record[keysAndValues[i].(string)] = keysAndValues[i+1]
	}
	s.records = append(s.records, record)
}

func (s *executorProfileSink) Error(error, string, ...any) {}

func (s *executorProfileSink) WithValues(...any) logr.LogSink { return s }

func (s *executorProfileSink) WithName(string) logr.LogSink { return s }

type checkpointProfileRuntime struct{}

var _ snapshotruntime.Runtime = (*checkpointProfileRuntime)(nil)

func (*checkpointProfileRuntime) ResolveContainer(context.Context, string) (int, *specs.Spec, error) {
	return 0, nil, errors.New("sensitive runtime detail")
}

func (*checkpointProfileRuntime) ResolveContainerIDByPod(context.Context, string, string, string) (string, error) {
	return "", errors.New("not implemented")
}

func (*checkpointProfileRuntime) ResolveContainerByPod(context.Context, string, string, string) (int, *specs.Spec, error) {
	return 0, nil, errors.New("not implemented")
}

func (*checkpointProfileRuntime) Close() error { return nil }

func TestCheckpointProfileCorrelatesExecutorSpans(t *testing.T) {
	t.Setenv(profile.Env, "1")
	sink := &executorProfileSink{}
	req := CheckpointRequest{
		ContainerID:        "container-id",
		ContainerName:      "main",
		CheckpointID:       "checkpoint-123",
		CheckpointLocation: filepath.Join(t.TempDir(), "checkpoint-123"),
		PodName:            "worker-0",
		PodNamespace:       "inference",
	}

	err := Checkpoint(
		context.Background(),
		&checkpointProfileRuntime{},
		logr.New(sink),
		req,
		&types.AgentConfig{},
	)
	if err == nil {
		t.Fatal("expected container resolution to fail")
	}
	assertCorrelatedExecutorRecords(t, sink.records, "checkpoint", req.CheckpointID, req.PodName, req.PodNamespace, req.ContainerName)
	assertPhaseStatus(t, sink.records, "container_resolution", "error")
	assertPhaseStatus(t, sink.records, "host_inspect_total", "error")
	assertPhaseStatus(t, sink.records, "checkpoint_total", "error")
}

func TestRestoreProfileCorrelatesExecutorSpans(t *testing.T) {
	t.Setenv(profile.Env, "1")
	sink := &executorProfileSink{}
	req := RestoreRequest{
		CheckpointID:       "checkpoint-456",
		CheckpointLocation: filepath.Join(t.TempDir(), "missing-checkpoint"),
		PodName:            "worker-1",
		PodNamespace:       "inference",
		ContainerName:      "main",
	}

	_, err := Restore(context.Background(), nil, logr.New(sink), req)
	if err == nil {
		t.Fatal("expected manifest read to fail")
	}
	assertCorrelatedExecutorRecords(t, sink.records, "restore", req.CheckpointID, req.PodName, req.PodNamespace, req.ContainerName)
	assertPhaseStatus(t, sink.records, "manifest_read", "error")
	assertPhaseStatus(t, sink.records, "host_inspect_total", "error")
	assertPhaseStatus(t, sink.records, "restore_total", "error")
}

func assertCorrelatedExecutorRecords(
	t *testing.T,
	records []map[string]any,
	operation, checkpointID, pod, namespace, container string,
) {
	t.Helper()
	if len(records) < 2 {
		t.Fatalf("got %d profile records, want at least 2", len(records))
	}
	operationID, ok := records[0]["operation_id"].(string)
	if !ok || operationID == "" {
		t.Fatalf("first record has no operation ID: %v", records[0])
	}
	for _, record := range records {
		if record["operation_id"] != operationID ||
			record["operation"] != operation ||
			record["checkpoint_id"] != checkpointID ||
			record["pod"] != pod ||
			record["namespace"] != namespace ||
			record["container"] != container {
			t.Fatalf("profile record is not operation-correlated: %v", record)
		}
		if record["status"] == "error" {
			if record["error_category"] == nil {
				t.Fatalf("error profile record lacks category: %v", record)
			}
			for _, value := range record {
				if value == "sensitive runtime detail" {
					t.Fatalf("error text leaked into profile record: %v", record)
				}
			}
		}
	}
}

func assertPhaseStatus(t *testing.T, records []map[string]any, phase, status string) {
	t.Helper()
	for _, record := range records {
		if record["phase"] == phase {
			if record["status"] != status {
				t.Fatalf("phase %q status = %v, want %q: %v", phase, record["status"], status, record)
			}
			return
		}
	}
	t.Fatalf("profile phase %q not found: %v", phase, records)
}
