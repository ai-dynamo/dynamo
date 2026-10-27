/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	corev1 "k8s.io/api/core/v1"
)

// ErrUnsupportedVLLMProfileSource classifies declarations outside the external Elastic EP geometry contract.
var ErrUnsupportedVLLMProfileSource = errors.New("unsupported vLLM Engine Group profile source")

const (
	vllmMultiprocessingEnvVar = "VLLM_ENABLE_V1_MULTIPROCESSING"
	vllmElasticEPDrainEnvVar  = "VLLM_ELASTIC_EP_DRAIN_REQUESTS"
)

// UnsupportedVLLMProfileSourceReason is a stable classification for unsupported launch sources.
type UnsupportedVLLMProfileSourceReason string

const (
	UnsupportedVLLMProfileSourceReasonCommandForm       UnsupportedVLLMProfileSourceReason = "command-form"
	UnsupportedVLLMProfileSourceReasonShell             UnsupportedVLLMProfileSourceReason = "shell-or-wrapper"
	UnsupportedVLLMProfileSourceReasonEnvironment       UnsupportedVLLMProfileSourceReason = "environment"
	UnsupportedVLLMProfileSourceReasonExternalConfig    UnsupportedVLLMProfileSourceReason = "external-config"
	UnsupportedVLLMProfileSourceReasonAbbreviatedOption UnsupportedVLLMProfileSourceReason = "abbreviated-option"
	UnsupportedVLLMProfileSourceReasonMissingGeometry   UnsupportedVLLMProfileSourceReason = "missing-geometry"
	UnsupportedVLLMProfileSourceReasonElasticEP         UnsupportedVLLMProfileSourceReason = "external-elastic-ep"
	UnsupportedVLLMProfileSourceReasonCapacityLayout    UnsupportedVLLMProfileSourceReason = "capacity-layout"
	UnsupportedVLLMProfileSourceReasonProcessOwnership  UnsupportedVLLMProfileSourceReason = "process-ownership"
)

// UnsupportedVLLMProfileSourceError identifies the source boundary that prevents static resolution.
type UnsupportedVLLMProfileSourceError struct {
	Reason UnsupportedVLLMProfileSourceReason
	Detail string
}

func (e *UnsupportedVLLMProfileSourceError) Error() string {
	return fmt.Sprintf("%s: %s: %s", ErrUnsupportedVLLMProfileSource, e.Reason, e.Detail)
}

// Unwrap also classifies an unsupported engine declaration as an unsupported Engine Group profile.
func (e *UnsupportedVLLMProfileSourceError) Unwrap() []error {
	return []error{ErrUnsupportedVLLMProfileSource, enginegroup.ErrUnsupportedProfile}
}

// VLLMProfileGeometrySource describes one initial allocation's provider-resolved exec-form launch.
// Env must include effective relevant image/container environment, or the provider must neutralize it.
// MainContainerGPUs must describe the complete, dedicated GPU allocation visible to the engine.
// InitialReplicas may be zero to derive the creation seed from explicit global DP arguments.
// WorkloadRevisionDigest must exclude creation-time/live targets and per-allocation launch identities.
// Resolution does not mutate the source or establish runtime, bootstrap, or serving conformance.
type VLLMProfileGeometrySource struct {
	Command                    []string
	Args                       []string
	Env                        []corev1.EnvVar
	EnvFrom                    []corev1.EnvFromSource
	InitialReplicas            int32
	MainContainerGPUs          int64
	DedicatedMainGPUAllocation bool
	WorkloadRevisionDigest     string
}

// ResolvedVLLMElasticEPProfile contains immutable geometry and bounds in complete allocation units.
// Native targets multiply these counts by NativeMembersPerReplica; model-specific constraints still need validation.
type ResolvedVLLMElasticEPProfile struct {
	Geometry                enginegroup.ResolvedProfileGeometry
	InitialReplicas         int32
	MaximumReplicas         int32
	NativeMembersPerReplica int32
}

type vllmProfileDimensions struct {
	tp, pp, dcp, pcp int64
	dp, localDP      int64
	nodes, nodeRank  int64
	apiServers       int64
	maximumDP        int64
}

type vllmEngineGeometryProjection struct {
	Version        string          `json:"version"`
	TP             int64           `json:"tensorParallelSize"`
	Executor       string          `json:"executor"`
	All2AllBackend string          `json:"all2allBackend"`
	MaximumDP      int64           `json:"maximumDataParallelSize"`
	EPLB           json.RawMessage `json:"eplb"`
}

// ResolveVLLMElasticEPProfile resolves one Pod-local client owning a fixed native DP range.
// PP, context parallelism, multi-node TP, Ray and recovery are not enabled here.
func ResolveVLLMElasticEPProfile(source VLLMProfileGeometrySource) (ResolvedVLLMElasticEPProfile, error) {
	// Establish inspectable launch facts before interpreting their effective ownership.
	if source.InitialReplicas < 0 {
		return ResolvedVLLMElasticEPProfile{}, fmt.Errorf("initial replicas must not be negative, got %d", source.InitialReplicas)
	}
	values, flags, err := collectVLLMProfileOptions(source.Command, source.Args)
	if err != nil {
		return ResolvedVLLMElasticEPProfile{}, err
	}
	if err := validateVLLMProfileEnvironment(source); err != nil {
		return ResolvedVLLMElasticEPProfile{}, err
	}
	// Native auto mode fixes the creation seed without supplying a live global DP target.
	if flags["--elastic-ep-auto-bootstrap"] {
		if _, duplicate := values["--data-parallel-size"]; duplicate {
			return ResolvedVLLMElasticEPProfile{}, fmt.Errorf("automatic bootstrap cannot include --data-parallel-size")
		}
		initial, present := values["--elastic-ep-initial-dp-size"]
		if !present {
			return ResolvedVLLMElasticEPProfile{}, fmt.Errorf("automatic bootstrap requires --elastic-ep-initial-dp-size")
		}
		values["--data-parallel-size"] = initial
	}
	dimensions, err := resolveVLLMProfileDimensions(values)
	if err != nil {
		return ResolvedVLLMElasticEPProfile{}, err
	}
	if err := validateVLLMExternalGeometry(dimensions, values, flags); err != nil {
		return ResolvedVLLMElasticEPProfile{}, err
	}

	digest, err := digestVLLMEngineGeometry(dimensions, values)
	if err != nil {
		return ResolvedVLLMElasticEPProfile{}, err
	}

	// The local native engine range fills one indivisible Pod allocation.
	if dimensions.tp > math.MaxInt32 || dimensions.localDP > math.MaxInt32 || dimensions.tp*dimensions.localDP != source.MainContainerGPUs {
		return ResolvedVLLMElasticEPProfile{}, fmt.Errorf("Pod GPU allocation must equal TP times local DP size")
	}
	geometry, err := enginegroup.ResolveProfileGeometry(enginegroup.ProfileGeometryInput{
		Backend:        "vllm",
		GPUsPerReplica: source.MainContainerGPUs,
		CapacityRoles: []enginegroup.CapacityRoleGeometry{{
			Name:             enginegroup.MainCapacityRoleName,
			Class:            enginegroup.CapacityRoleClassRankOwningCapacity,
			EngineGPUsPerPod: source.MainContainerGPUs,
			DedicatedGPUs:    source.DedicatedMainGPUAllocation,
		}},
		EngineGeometryDigest:   digest,
		WorkloadRevisionDigest: source.WorkloadRevisionDigest,
	})
	if err != nil {
		return ResolvedVLLMElasticEPProfile{}, err
	}
	// Engine flags count native members; the Kubernetes target counts complete physical allocations.
	membersPerReplica := geometry.GPUsPerReplica / dimensions.tp
	if dimensions.dp%membersPerReplica != 0 || dimensions.maximumDP%membersPerReplica != 0 {
		return ResolvedVLLMElasticEPProfile{}, fmt.Errorf("initial/maximum DP sizes %d/%d must form complete %d-member allocations", dimensions.dp, dimensions.maximumDP, membersPerReplica)
	}
	initialReplicas := dimensions.dp / membersPerReplica
	if source.InitialReplicas != 0 && int64(source.InitialReplicas) != initialReplicas {
		return ResolvedVLLMElasticEPProfile{}, fmt.Errorf("initial replicas %d conflict with vLLM launch DP size %d", source.InitialReplicas, dimensions.dp)
	}
	return ResolvedVLLMElasticEPProfile{
		Geometry: geometry, InitialReplicas: int32(initialReplicas), MaximumReplicas: int32(dimensions.maximumDP / membersPerReplica),
		NativeMembersPerReplica: int32(membersPerReplica),
	}, nil
}

func resolveVLLMProfileDimensions(values map[string]string) (vllmProfileDimensions, error) {
	// Do not infer initial DP from the allocation target or native environment fallback.
	for _, option := range []string{"--data-parallel-size", "--elastic-ep-max-dp-size"} {
		if _, present := values[option]; !present {
			return vllmProfileDimensions{}, &UnsupportedVLLMProfileSourceError{
				Reason: UnsupportedVLLMProfileSourceReasonMissingGeometry, Detail: option + " must be explicit",
			}
		}
	}

	// Use only documented scalar defaults; local ownership is validated separately below.
	var dimensions vllmProfileDimensions
	options := []struct {
		name     string
		fallback int64
		result   *int64
	}{
		{"--tensor-parallel-size", 1, &dimensions.tp},
		{"--pipeline-parallel-size", 1, &dimensions.pp},
		{"--decode-context-parallel-size", 1, &dimensions.dcp},
		{"--prefill-context-parallel-size", 1, &dimensions.pcp},
		{"--data-parallel-size", 1, &dimensions.dp},
		{"--data-parallel-size-local", 1, &dimensions.localDP},
		{"--nnodes", 1, &dimensions.nodes},
		{"--api-server-count", 1, &dimensions.apiServers},
		{"--elastic-ep-max-dp-size", 1, &dimensions.maximumDP},
		{"--node-rank", 0, &dimensions.nodeRank},
	}
	for _, option := range options {
		value := option.fallback
		if literal, present := values[option.name]; present {
			parsed, err := strconv.ParseInt(literal, 10, 64)
			if err != nil || parsed < 0 || (parsed == 0 && option.name != "--node-rank") {
				return vllmProfileDimensions{}, fmt.Errorf("invalid vLLM option %s value %q", option.name, literal)
			}
			value = parsed
		}
		*option.result = value
	}
	return dimensions, nil
}

func validateVLLMExternalGeometry(d vllmProfileDimensions, values map[string]string, flags map[string]bool) error {
	// Require the current external Elastic EP path, not the older Ray-owned/internal-LB layout.
	for _, option := range []string{"--enable-expert-parallel", "--enable-eplb", "--enable-elastic-ep"} {
		if !flags[option] {
			return &UnsupportedVLLMProfileSourceError{Reason: UnsupportedVLLMProfileSourceReasonElasticEP, Detail: option + " is required"}
		}
	}
	if d.dp <= 1 {
		return &UnsupportedVLLMProfileSourceError{
			Reason: UnsupportedVLLMProfileSourceReasonElasticEP,
			Detail: "external Elastic EP requires initial DP > 1; native DP environment fallback is not an external launch declaration",
		}
	}
	if d.dp > math.MaxInt32 || d.maximumDP > math.MaxInt32 || d.maximumDP < d.dp {
		return fmt.Errorf("vLLM initial/maximum DP sizes %d/%d must fit the replica range and maximum must cover initial", d.dp, d.maximumDP)
	}
	if d.pp != 1 || d.dcp != 1 || d.pcp != 1 || d.nodes != 1 || d.nodeRank != 0 || d.apiServers != 1 {
		return &UnsupportedVLLMProfileSourceError{
			Reason: UnsupportedVLLMProfileSourceReasonCapacityLayout, Detail: "the profile requires PP=CP=1, one local node, node rank 0 and one API server per allocation",
		}
	}
	if d.localDP > d.dp || (d.localDP != 1 && !flags["--elastic-ep-auto-bootstrap"]) {
		return &UnsupportedVLLMProfileSourceError{Reason: UnsupportedVLLMProfileSourceReasonCapacityLayout, Detail: "a packed local DP range requires native automatic bootstrap and must fit the initial world"}
	}

	// Pin a local executor instead of allowing environment-dependent Ray auto-selection.
	executor := values["--distributed-executor-backend"]
	dpBackend := values["--data-parallel-backend"]
	if (executor != "mp" && executor != "uni") || (executor == "uni" && d.tp != 1) || (dpBackend != "" && dpBackend != "mp") {
		return &UnsupportedVLLMProfileSourceError{
			Reason: UnsupportedVLLMProfileSourceReasonProcessOwnership, Detail: "an explicit Pod-local mp executor (or uni at TP=1) and the mp DP backend are required",
		}
	}
	for _, option := range []string{"--headless", "--data-parallel-multi-port-external-lb", "--enable-fault-tolerance", "--device-ids"} {
		if _, present := values[option]; present || flags[option] {
			return &UnsupportedVLLMProfileSourceError{Reason: UnsupportedVLLMProfileSourceReasonProcessOwnership, Detail: option + " is outside this profile"}
		}
	}
	return validateVLLMExternalOwnership(d, values, flags)
}

func validateVLLMExternalOwnership(d vllmProfileDimensions, values map[string]string, flags map[string]bool) error {
	// Allocation indices are interpreted by native bootstrap, never by the operator or a shell wrapper.
	if index, present := values["--elastic-ep-allocation-index"]; present {
		if !flags["--elastic-ep-auto-bootstrap"] || flags["--data-parallel-hybrid-lb"] || flags["--data-parallel-external-lb"] ||
			values["--data-parallel-rank"] != "" || values["--data-parallel-start-rank"] != "" {
			return fmt.Errorf("automatic allocation index cannot be combined with rank or load-balancer ownership flags")
		}
		if index == engineGroupOrdinalRef {
			return nil
		}
		return validateVLLMInitialRank(index, d.dp/d.localDP)
	}

	// EngineArgs treats an explicit DP rank as external LB, even without the external-LB flag.
	rankLiteral, explicitRank := values["--data-parallel-rank"]
	startLiteral, explicitStart := values["--data-parallel-start-rank"]
	if (flags["--data-parallel-external-lb"] || explicitRank) && (flags["--data-parallel-hybrid-lb"] || explicitStart) {
		return fmt.Errorf("vLLM external and hybrid LB options cannot both be enabled")
	}
	if explicitRank {
		if flags["--elastic-ep-auto-bootstrap"] && rankLiteral == engineGroupOrdinalRef {
			return nil
		}
		return validateVLLMInitialRank(rankLiteral, d.dp)
	}
	if flags["--data-parallel-external-lb"] {
		return &UnsupportedVLLMProfileSourceError{
			Reason: UnsupportedVLLMProfileSourceReasonProcessOwnership, Detail: "single-node external LB requires an explicit data-parallel rank",
		}
	}

	// Local DP=1 becomes external only when hybrid/start-rank mode is actually selected.
	_, explicitLocal := values["--data-parallel-size-local"]
	if explicitLocal && (flags["--data-parallel-hybrid-lb"] || explicitStart) {
		if !explicitStart {
			startLiteral = "0"
		}
		return validateVLLMInitialRank(startLiteral, d.dp)
	}
	return &UnsupportedVLLMProfileSourceError{
		Reason: UnsupportedVLLMProfileSourceReasonProcessOwnership, Detail: "local DP=1 alone does not establish external ownership; internal LB manages the global DP set",
	}
}

func validateVLLMInitialRank(literal string, dp int64) error {
	rank, err := strconv.ParseInt(literal, 10, 64)
	if err != nil || rank < 0 || rank >= dp {
		return fmt.Errorf("vLLM initial data-parallel rank %q must be in [0, %d)", literal, dp)
	}
	return nil
}

func validateVLLMProfileEnvironment(source VLLMProfileGeometrySource) error {
	// Opaque imports can replace both process ownership and frontend selection.
	if len(source.EnvFrom) != 0 {
		return &UnsupportedVLLMProfileSourceError{Reason: UnsupportedVLLMProfileSourceReasonEnvironment, Detail: "envFrom must be resolved before profile extraction"}
	}
	for _, variable := range source.Env {
		relevant := strings.HasPrefix(variable.Name, "VLLM_DP_") || variable.Name == "VLLM_ELASTIC_EP_SCALE_UP_LAUNCH" ||
			variable.Name == vllmMultiprocessingEnvVar || variable.Name == "VLLM_USE_RUST_FRONTEND" || variable.Name == vllmElasticEPDrainEnvVar
		if !relevant {
			continue
		}
		if variable.ValueFrom != nil || hasKubernetesArgumentExpansion(variable.Value) {
			return &UnsupportedVLLMProfileSourceError{Reason: UnsupportedVLLMProfileSourceReasonEnvironment, Detail: variable.Name + " must be resolved to a literal value"}
		}
		if strings.HasPrefix(variable.Name, "VLLM_DP_") {
			continue // Explicit initial DP > 1 bypasses vLLM's entire native DP environment fallback.
		}
		value, err := strconv.ParseInt(variable.Value, 10, 64)
		mustBeEnabled := variable.Name == vllmMultiprocessingEnvVar || variable.Name == vllmElasticEPDrainEnvVar
		if err != nil || (mustBeEnabled && value == 0) || (!mustBeEnabled && value != 0) {
			return &UnsupportedVLLMProfileSourceError{Reason: UnsupportedVLLMProfileSourceReasonEnvironment, Detail: variable.Name + " conflicts with the initial Python/MP external profile"}
		}
	}
	return nil
}

func digestVLLMEngineGeometry(d vllmProfileDimensions, values map[string]string) (string, error) {
	// Canonicalize EPLB options without making their JSON key order part of profile identity.
	eplb := map[string]json.RawMessage{}
	if literal, present := values["--eplb-config"]; present {
		if err := json.Unmarshal([]byte(literal), &eplb); err != nil || eplb == nil {
			return "", fmt.Errorf("vLLM --eplb-config must be a JSON object")
		}
	}
	eplbJSON, err := json.Marshal(eplb)
	if err != nil {
		return "", fmt.Errorf("marshal vLLM EPLB geometry: %w", err)
	}
	backend := values["--all2all-backend"]
	if backend == "" {
		backend = "allgather_reducescatter"
	}

	// Initial DP and rank/address assignments are mutable dimensions; the reserved ceiling is not.
	encoded, err := json.Marshal(vllmEngineGeometryProjection{
		Version: "vllm-external-engine-group-geometry/v1", TP: d.tp,
		Executor: values["--distributed-executor-backend"], All2AllBackend: backend,
		MaximumDP: d.maximumDP, EPLB: eplbJSON,
	})
	if err != nil {
		return "", fmt.Errorf("marshal vLLM engine geometry: %w", err)
	}
	digest := sha256.Sum256(encoded)
	return "sha256:" + hex.EncodeToString(digest[:]), nil
}
