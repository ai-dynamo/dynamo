/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package powerbudget

import (
	"encoding/json"
	"fmt"
	"time"
)

const (
	// AgentReportAnnotation is the sole Pod annotation owned by the Power Agent.
	AgentReportAnnotation = "dynamo.nvidia.com/gpu-power-enforcement-report"
	// AgentReportDocumentVersion is the Phase 2 report decoding contract.
	AgentReportDocumentVersion = 1
	// MaxAgentReportBytes bounds one encoded report annotation to 64 KiB.
	MaxAgentReportBytes = 64 * 1024
)

// AgentPolicyOutcome records why the Agent selected the target cap.
type AgentPolicyOutcome string

const (
	AgentPolicyOutcomeAnnotated                   AgentPolicyOutcome = "annotated"
	AgentPolicyOutcomeSafeDefaultMissingOrInvalid AgentPolicyOutcome = "safe_default_missing_or_invalid"
	AgentPolicyOutcomeSafeDefaultConflict         AgentPolicyOutcome = "safe_default_conflict"
)

// AgentWriteOutcome records the hardware-write result.
type AgentWriteOutcome string

const (
	AgentWriteOutcomeSucceeded               AgentWriteOutcome = "succeeded"
	AgentWriteOutcomeFailed                  AgentWriteOutcome = "failed"
	AgentWriteOutcomeSkippedIdentityMismatch AgentWriteOutcome = "skipped_identity_mismatch"
)

// AgentReadbackOutcome records the immediate exact-identity cap readback result.
type AgentReadbackOutcome string

const (
	AgentReadbackOutcomeSucceeded    AgentReadbackOutcome = "succeeded"
	AgentReadbackOutcomeFailed       AgentReadbackOutcome = "failed"
	AgentReadbackOutcomeNotAttempted AgentReadbackOutcome = "not_attempted"
)

// AgentActuator identifies the hardware API used to apply the cap.
type AgentActuator string

const (
	AgentActuatorNVML AgentActuator = "nvml"
	AgentActuatorDCGM AgentActuator = "dcgm"
)

// AgentGPUReport is the exact apply/readback result for one assigned physical GPU.
type AgentGPUReport struct {
	UUID               string               `json:"uuid"`
	RequestedWatts     int64                `json:"requestedWatts"`
	TargetWatts        int64                `json:"targetWatts"`
	ConstraintMinWatts int64                `json:"constraintMinWatts"`
	ConstraintMaxWatts int64                `json:"constraintMaxWatts"`
	PolicyOutcome      AgentPolicyOutcome   `json:"policyOutcome"`
	WriteOutcome       AgentWriteOutcome    `json:"writeOutcome"`
	ReadbackOutcome    AgentReadbackOutcome `json:"readbackOutcome"`
	EnforcedCapWatts   *int64               `json:"enforcedCapWatts"`
	Actuator           AgentActuator        `json:"actuator"`
	ObservedAt         time.Time            `json:"observedAt"`
}

// AgentReport is one document-versioned report for all GPUs assigned to a
// power-managed workload container.
type AgentReport struct {
	Version      int              `json:"version"`
	DGDUID       string           `json:"dgdUID"`
	Component    string           `json:"component"`
	PodUID       string           `json:"podUID"`
	Node         string           `json:"node"`
	AllocationID string           `json:"allocationID"`
	GPUs         []AgentGPUReport `json:"gpus"`
}

// EncodeAgentReport serializes a supported report within the annotation limit.
func EncodeAgentReport(report AgentReport) ([]byte, error) {
	if err := ValidateAgentReport(report); err != nil {
		return nil, err
	}
	encoded, err := json.Marshal(report)
	if err != nil {
		return nil, fmt.Errorf("encode Agent report: %w", err)
	}
	if len(encoded) > MaxAgentReportBytes {
		return nil, fmt.Errorf("Agent report size %d exceeds %d bytes", len(encoded), MaxAgentReportBytes)
	}
	return encoded, nil
}

// DecodeAgentReport decodes a supported report within the annotation limit.
func DecodeAgentReport(encoded []byte) (AgentReport, error) {
	if len(encoded) == 0 {
		return AgentReport{}, fmt.Errorf("Agent report is empty")
	}
	if len(encoded) > MaxAgentReportBytes {
		return AgentReport{}, fmt.Errorf("Agent report size %d exceeds %d bytes", len(encoded), MaxAgentReportBytes)
	}
	var report AgentReport
	if err := json.Unmarshal(encoded, &report); err != nil {
		return AgentReport{}, fmt.Errorf("decode Agent report: %w", err)
	}
	if report.Version != AgentReportDocumentVersion {
		return AgentReport{}, fmt.Errorf("unsupported Agent report version %d", report.Version)
	}
	if err := ValidateAgentReport(report); err != nil {
		return AgentReport{}, err
	}
	return report, nil
}

// ValidateAgentReport validates the versioned structural and outcome contract.
func ValidateAgentReport(report AgentReport) error {
	if report.Version != AgentReportDocumentVersion {
		return fmt.Errorf("unsupported Agent report version %d", report.Version)
	}
	if report.DGDUID == "" || report.Component == "" || report.PodUID == "" || report.Node == "" || report.AllocationID == "" {
		return fmt.Errorf("Agent report identity fields must be nonempty")
	}
	if len(report.GPUs) == 0 {
		return fmt.Errorf("Agent report contains no GPUs")
	}
	seen := make(map[string]struct{}, len(report.GPUs))
	for index, gpu := range report.GPUs {
		if err := validateAgentGPUReport(gpu); err != nil {
			return fmt.Errorf("Agent report GPU %d: %w", index, err)
		}
		if _, found := seen[gpu.UUID]; found {
			return fmt.Errorf("Agent report contains duplicate GPU UUID %q", gpu.UUID)
		}
		seen[gpu.UUID] = struct{}{}
	}
	return nil
}

func validateAgentGPUReport(gpu AgentGPUReport) error {
	if gpu.UUID == "" {
		return fmt.Errorf("UUID is empty")
	}
	if gpu.RequestedWatts < 1 || gpu.TargetWatts < 1 || gpu.ConstraintMinWatts < 1 || gpu.ConstraintMaxWatts < gpu.ConstraintMinWatts {
		return fmt.Errorf("invalid requested/target/constraint watts requested=%d target=%d constraints=[%d,%d]", gpu.RequestedWatts, gpu.TargetWatts, gpu.ConstraintMinWatts, gpu.ConstraintMaxWatts)
	}
	if gpu.TargetWatts < gpu.ConstraintMinWatts || gpu.TargetWatts > gpu.ConstraintMaxWatts {
		return fmt.Errorf("target watts %d is outside live constraints", gpu.TargetWatts)
	}
	if !validPolicyOutcome(gpu.PolicyOutcome) || !validWriteOutcome(gpu.WriteOutcome) || !validReadbackOutcome(gpu.ReadbackOutcome) || !validActuator(gpu.Actuator) {
		return fmt.Errorf("unknown policy, write, readback, or actuator outcome")
	}
	if gpu.ObservedAt.IsZero() {
		return fmt.Errorf("observedAt is zero")
	}
	readbackProved := gpu.WriteOutcome == AgentWriteOutcomeSucceeded && gpu.ReadbackOutcome == AgentReadbackOutcomeSucceeded
	if gpu.ReadbackOutcome == AgentReadbackOutcomeSucceeded && gpu.WriteOutcome != AgentWriteOutcomeSucceeded {
		return fmt.Errorf("successful readback cannot follow an unsuccessful write")
	}
	if readbackProved && gpu.EnforcedCapWatts == nil {
		return fmt.Errorf("successful write and readback require enforcedCapWatts")
	}
	if !readbackProved && gpu.EnforcedCapWatts != nil {
		return fmt.Errorf("enforcedCapWatts requires successful write and readback")
	}
	if gpu.EnforcedCapWatts != nil && (*gpu.EnforcedCapWatts < gpu.ConstraintMinWatts || *gpu.EnforcedCapWatts > gpu.ConstraintMaxWatts) {
		return fmt.Errorf("enforcedCapWatts %d is outside live constraints", *gpu.EnforcedCapWatts)
	}
	return nil
}

func validPolicyOutcome(outcome AgentPolicyOutcome) bool {
	switch outcome {
	case AgentPolicyOutcomeAnnotated, AgentPolicyOutcomeSafeDefaultMissingOrInvalid, AgentPolicyOutcomeSafeDefaultConflict:
		return true
	default:
		return false
	}
}

func validWriteOutcome(outcome AgentWriteOutcome) bool {
	switch outcome {
	case AgentWriteOutcomeSucceeded, AgentWriteOutcomeFailed, AgentWriteOutcomeSkippedIdentityMismatch:
		return true
	default:
		return false
	}
}

func validReadbackOutcome(outcome AgentReadbackOutcome) bool {
	switch outcome {
	case AgentReadbackOutcomeSucceeded, AgentReadbackOutcomeFailed, AgentReadbackOutcomeNotAttempted:
		return true
	default:
		return false
	}
}

func validActuator(actuator AgentActuator) bool {
	switch actuator {
	case AgentActuatorNVML, AgentActuatorDCGM:
		return true
	default:
		return false
	}
}
