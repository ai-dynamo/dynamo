/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package powerbudget

import (
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestAgentReportSchema(t *testing.T) {
	t.Log("Serialize every required identity and per-GPU enforcement field")
	enforcedCapWatts := int64(350)
	report := AgentReport{
		Version:      AgentReportDocumentVersion,
		DGDUID:       "dgd-uid",
		Component:    "prefill",
		PodUID:       "pod-uid",
		Node:         "node-a",
		AllocationID: "allocation-id",
		GPUs: []AgentGPUReport{{
			UUID:               "GPU-1",
			RequestedWatts:     350,
			TargetWatts:        350,
			ConstraintMinWatts: 300,
			ConstraintMaxWatts: 700,
			PolicyOutcome:      AgentPolicyOutcomeAnnotated,
			WriteOutcome:       AgentWriteOutcomeSucceeded,
			ReadbackOutcome:    AgentReadbackOutcomeSucceeded,
			EnforcedCapWatts:   &enforcedCapWatts,
			Actuator:           AgentActuatorNVML,
			ObservedAt:         time.Date(2026, time.August, 15, 12, 34, 56, 0, time.UTC),
		}},
	}

	encoded, err := EncodeAgentReport(report)
	if err != nil {
		t.Fatalf("EncodeAgentReport() error = %v", err)
	}
	want := `{"version":1,"dgdUID":"dgd-uid","component":"prefill","podUID":"pod-uid","node":"node-a","allocationID":"allocation-id","gpus":[{"uuid":"GPU-1","requestedWatts":350,"targetWatts":350,"constraintMinWatts":300,"constraintMaxWatts":700,"policyOutcome":"annotated","writeOutcome":"succeeded","readbackOutcome":"succeeded","enforcedCapWatts":350,"actuator":"nvml","observedAt":"2026-08-15T12:34:56Z"}]}`
	if string(encoded) != want {
		t.Fatalf("encoded Agent report = %s, want %s", encoded, want)
	}

	t.Log("Decode the same document version without losing typed fields")
	decoded, err := DecodeAgentReport(encoded)
	if err != nil {
		t.Fatalf("DecodeAgentReport() error = %v", err)
	}
	if !reflect.DeepEqual(decoded, report) {
		t.Fatalf("decoded Agent report = %#v, want %#v", decoded, report)
	}

	t.Log("Reject unsupported document versions and encoded reports above 64 KiB")
	report.Version++
	if _, err := EncodeAgentReport(report); err == nil {
		t.Fatal("EncodeAgentReport() accepted an unsupported version")
	}
	oversized := []byte(strings.Repeat("x", MaxAgentReportBytes+1))
	if _, err := DecodeAgentReport(oversized); err == nil {
		t.Fatal("DecodeAgentReport() accepted an oversized annotation")
	}

	t.Log("Reject inconsistent enforcement evidence within a version-1 document")
	invalid := report
	invalid.Version = AgentReportDocumentVersion
	invalid.GPUs = append([]AgentGPUReport(nil), report.GPUs...)
	invalid.GPUs[0].WriteOutcome = AgentWriteOutcomeFailed
	if _, err := EncodeAgentReport(invalid); err == nil {
		t.Fatal("EncodeAgentReport() accepted enforcedCapWatts after a failed write")
	}
	invalid.GPUs[0].WriteOutcome = AgentWriteOutcomeSucceeded
	invalid.GPUs[0].ReadbackOutcome = AgentReadbackOutcomeFailed
	if _, err := EncodeAgentReport(invalid); err == nil {
		t.Fatal("EncodeAgentReport() accepted enforcedCapWatts after failed readback")
	}
}
