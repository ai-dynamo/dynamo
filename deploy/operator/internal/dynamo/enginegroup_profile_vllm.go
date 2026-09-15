/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package dynamo

import (
	"errors"
	"fmt"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
)

const (
	vllmEngineGroupBackend         = "vllm"
	vllmDynamoModule               = "dynamo.vllm"
	vllmServeSubcommand            = "serve"
	vllmTensorParallelSizeAlias    = "-tp"
	vllmPipelineParallelSizeAlias  = "-pp"
	vllmPrefillContextSizeFlag     = "--prefill-context-parallel-size"
	vllmPrefillContextSizeAlias    = "-pcp"
	vllmDataParallelSizeAlias      = "-dp"
	vllmDataParallelSizeLocalAlias = "-dpl"
	vllmDataParallelExternalLBFlag = "--data-parallel-external-lb"
	vllmDataParallelHybridLBFlag   = "--data-parallel-hybrid-lb"
	vllmNodesFlag                  = "--nnodes"
	vllmNodesAlias                 = "-n"
	vllmNodeRankFlag               = "--node-rank"
	vllmNodeRankAlias              = "-r"
	vllmMasterAddressFlag          = "--master-addr"
	vllmHeadlessFlag               = "--headless"
	vllmDeviceIDsFlag              = "--device-ids"
	vllmDPSizeEnvironment          = "VLLM_DP_SIZE"
)

// ErrUnsupportedVLLMProfileSource classifies vLLM declarations whose geometry source is not statically inspectable.
var ErrUnsupportedVLLMProfileSource = errors.New("unsupported vLLM Engine Group profile source")

// UnsupportedVLLMProfileSourceReason is a stable machine-readable explanation for an opaque vLLM source.
type UnsupportedVLLMProfileSourceReason string

const (
	// UnsupportedVLLMProfileSourceReasonCommandForm means the image entrypoint or executable is not explicit.
	UnsupportedVLLMProfileSourceReasonCommandForm UnsupportedVLLMProfileSourceReason = "command-form"
	// UnsupportedVLLMProfileSourceReasonShell means a shell or environment wrapper owns the effective argv.
	UnsupportedVLLMProfileSourceReasonShell UnsupportedVLLMProfileSourceReason = "shell-or-wrapper"
	// UnsupportedVLLMProfileSourceReasonEnvironment means runtime expansion may change the effective declaration.
	UnsupportedVLLMProfileSourceReasonEnvironment UnsupportedVLLMProfileSourceReason = "environment-expansion"
	// UnsupportedVLLMProfileSourceReasonExternalConfig means an external config may supply or override geometry.
	UnsupportedVLLMProfileSourceReasonExternalConfig UnsupportedVLLMProfileSourceReason = "external-config"
	// UnsupportedVLLMProfileSourceReasonAbbreviatedOption means vLLM may expand a long option the resolver cannot bind exactly.
	UnsupportedVLLMProfileSourceReasonAbbreviatedOption UnsupportedVLLMProfileSourceReason = "abbreviated-option"
	// UnsupportedVLLMProfileSourceReasonMissingGeometry means required geometry relies on engine defaults.
	UnsupportedVLLMProfileSourceReasonMissingGeometry UnsupportedVLLMProfileSourceReason = "missing-geometry"
	// UnsupportedVLLMProfileSourceReasonDistributedPlacement means process placement is not proven pod-local.
	UnsupportedVLLMProfileSourceReasonDistributedPlacement UnsupportedVLLMProfileSourceReason = "distributed-placement"
	// UnsupportedVLLMProfileSourceReasonOwnershipMode means vLLM does not expose one externally managed rank per pod.
	UnsupportedVLLMProfileSourceReasonOwnershipMode UnsupportedVLLMProfileSourceReason = "ownership-mode"
)

// UnsupportedVLLMProfileSourceError describes why a vLLM declaration cannot produce typed geometry.
type UnsupportedVLLMProfileSourceError struct {
	Reason UnsupportedVLLMProfileSourceReason
	Detail string
}

// Error returns the stable unsupported-source class and reason with human-readable detail.
func (e *UnsupportedVLLMProfileSourceError) Error() string {
	if e.Detail == "" {
		return fmt.Sprintf("%s: %s", ErrUnsupportedVLLMProfileSource, e.Reason)
	}

	return fmt.Sprintf("%s: %s: %s", ErrUnsupportedVLLMProfileSource, e.Reason, e.Detail)
}

// Unwrap classifies an opaque provider source as both source-specific and generically unsupported.
func (e *UnsupportedVLLMProfileSourceError) Unwrap() []error {
	return []error{ErrUnsupportedVLLMProfileSource, enginegroup.ErrUnsupportedProfile}
}

// VLLMProfileGeometrySource contains the concrete declarative inputs needed to resolve vLLM geometry.
// Command and Args must be provider-resolved Kubernetes exec-form argv after fixed launch
// rewriting; only the managed global-DP target may still be omitted. Environment contains
// resolved literal vLLM values, while HasUnresolvedDPEnvironment reports an envFrom source or
// an unresolved VLLM_DP_SIZE valueFrom. WorkloadRevisionDigest must exclude creation-time and
// live replica targets. Resolution does not mutate the slices or map.
type VLLMProfileGeometrySource struct {
	Command                    []string
	Args                       []string
	Environment                map[string]string
	HasUnresolvedDPEnvironment bool
	InitialReplicas            int32
	MainContainerGPUs          int64
	DedicatedMainGPUAllocation bool
	WorkloadRevisionDigest     string
}

type parsedVLLMProfileGeometry struct {
	tensorParallelSize       int64
	pipelineParallelSize     int64
	prefillContextSize       int64
	dataParallelSize         int64
	hasDataParallelSize      bool
	dataParallelSizeLocal    int64
	hasDataParallelSizeLocal bool
	enableElasticEP          bool
	dataParallelExternalLB   bool
	dataParallelHybridLB     bool
}

type vllmGeometryOption struct {
	canonical string
	aliases   []string
}

var vllmProfileGeometryOptions = []vllmGeometryOption{
	{canonical: tensorParallelSizeFlag, aliases: []string{tensorParallelSizeFlag, vllmTensorParallelSizeAlias}},
	{canonical: pipelineParallelSizeFlag, aliases: []string{pipelineParallelSizeFlag, vllmPipelineParallelSizeAlias}},
	{canonical: vllmPrefillContextSizeFlag, aliases: []string{vllmPrefillContextSizeFlag, vllmPrefillContextSizeAlias}},
	{canonical: dataParallelSizeFlag, aliases: []string{dataParallelSizeFlag, vllmDataParallelSizeAlias}},
	{canonical: dataParallelSizeLocalFlag, aliases: []string{dataParallelSizeLocalFlag, vllmDataParallelSizeLocalAlias}},
}

var vllmProfileSensitiveLongOptions = []string{
	"--config",
	tensorParallelSizeFlag,
	pipelineParallelSizeFlag,
	vllmPrefillContextSizeFlag,
	dataParallelSizeFlag,
	dataParallelSizeLocalFlag,
	enableElasticEPFlag,
	vllmDataParallelExternalLBFlag,
	vllmDataParallelHybridLBFlag,
	distributedExecutorFlag,
	dataParallelBackendFlag,
	vllmNodesFlag,
	vllmNodeRankFlag,
	vllmMasterAddressFlag,
	vllmMasterPortFlag,
	vllmHeadlessFlag,
	vllmDeviceIDsFlag,
}

// ResolveVLLMProfileGeometry parses strict vLLM argv and validates its creation-time DP assertion.
// Current vLLM Elastic EP owns the complete DP world through one internal client and Ray, which the
// narrow one-replica-per-pod Engine Group profile cannot represent. The resolver therefore reports
// valid current declarations as unsupported until vLLM exposes compatible external membership.
func ResolveVLLMProfileGeometry(source VLLMProfileGeometrySource) (enginegroup.ResolvedProfileGeometry, error) {
	// Require a valid creation-time target before comparing it with an optional vLLM assertion.
	if source.InitialReplicas <= 0 {
		return enginegroup.ResolvedProfileGeometry{}, fmt.Errorf("initial replicas must be positive, got %d", source.InitialReplicas)
	}

	// Parse only inspectable argv and reject declarations whose effective source is external.
	geometry, err := parseVLLMProfileGeometry(source.Command, source.Args)
	if err != nil {
		return enginegroup.ResolvedProfileGeometry{}, err
	}
	geometry, err = applyVLLMProfileEnvironment(geometry, source.Environment, source.HasUnresolvedDPEnvironment)
	if err != nil {
		return enginegroup.ResolvedProfileGeometry{}, err
	}

	if !geometry.enableElasticEP {
		return enginegroup.ResolvedProfileGeometry{}, &UnsupportedVLLMProfileSourceError{
			Reason: UnsupportedVLLMProfileSourceReasonOwnershipMode,
			Detail: "--enable-elastic-ep is required",
		}
	}
	if geometry.dataParallelExternalLB || geometry.dataParallelHybridLB {
		return enginegroup.ResolvedProfileGeometry{}, &UnsupportedVLLMProfileSourceError{
			Reason: UnsupportedVLLMProfileSourceReasonOwnershipMode,
			Detail: "vLLM Elastic EP is incompatible with external or hybrid DP load balancing",
		}
	}

	// Treat global DP as a creation-time assertion, never as the durable profile target.
	if geometry.hasDataParallelSize && geometry.dataParallelSize != int64(source.InitialReplicas) {
		return enginegroup.ResolvedProfileGeometry{}, fmt.Errorf(
			"initial replicas %d conflict with vLLM data parallel size %d",
			source.InitialReplicas,
			geometry.dataParallelSize,
		)
	}

	if geometry.hasDataParallelSizeLocal && geometry.dataParallelSizeLocal > int64(source.InitialReplicas) {
		return enginegroup.ResolvedProfileGeometry{}, fmt.Errorf(
			"vLLM data parallel size local %d exceeds global data parallel size %d",
			geometry.dataParallelSizeLocal,
			source.InitialReplicas,
		)
	}

	return enginegroup.ResolvedProfileGeometry{}, &UnsupportedVLLMProfileSourceError{
		Reason: UnsupportedVLLMProfileSourceReasonOwnershipMode,
		Detail: "current vLLM Elastic EP uses one internal client with Ray-managed DP capacity; " +
			"the narrow one-replica-per-pod Engine Group profile cannot represent that ownership",
	}
}

func parseVLLMProfileGeometry(command, args []string) (parsedVLLMProfileGeometry, error) {
	// Resolve the exact provider-owned invocation before interpreting any engine options.
	argv, err := resolveVLLMProfileArguments(command, args)
	if err != nil {
		return parsedVLLMProfileGeometry{}, err
	}

	// Respect the conventional end-of-options marker when inspecting vLLM arguments.
	optionEnd := len(argv)
	for index, token := range argv {
		if token == "--" {
			optionEnd = index
			break
		}
	}

	// Reject arguments that keep the effective geometry or placement outside static inspection.
	inspectableArgv := argv[:optionEnd]
	if err := validateInspectableVLLMProfileArguments(inspectableArgv); err != nil {
		return parsedVLLMProfileGeometry{}, err
	}

	// Consume every geometry option at most once across its canonical name and aliases.
	values := make(map[string]int64, len(vllmProfileGeometryOptions))
	for index := 0; index < len(inspectableArgv); index++ {
		option, literal, consumed, matched := splitVLLMGeometryOption(inspectableArgv, index)
		if !matched {
			continue
		}
		if _, exists := values[option.canonical]; exists {
			return parsedVLLMProfileGeometry{}, fmt.Errorf(
				"vLLM geometry argument %s is specified more than once",
				option.canonical,
			)
		}
		if strings.Contains(literal, "$") {
			return parsedVLLMProfileGeometry{}, &UnsupportedVLLMProfileSourceError{
				Reason: UnsupportedVLLMProfileSourceReasonEnvironment,
				Detail: fmt.Sprintf("geometry argument %s contains runtime expansion", option.canonical),
			}
		}

		value, err := parseVLLMGeometryLiteral(
			option.canonical,
			literal,
			option.canonical == dataParallelSizeLocalFlag,
		)
		if err != nil {
			return parsedVLLMProfileGeometry{}, err
		}
		values[option.canonical] = value
		index += consumed
	}

	// Require TP and PP explicitly because engine defaults are not an immutable profile contract.
	for _, flag := range []string{tensorParallelSizeFlag, pipelineParallelSizeFlag} {
		if _, exists := values[flag]; !exists {
			return parsedVLLMProfileGeometry{}, &UnsupportedVLLMProfileSourceError{
				Reason: UnsupportedVLLMProfileSourceReasonMissingGeometry,
				Detail: fmt.Sprintf("required geometry argument %s is not explicit", flag),
			}
		}
	}

	prefillContextSize, present := values[vllmPrefillContextSizeFlag]
	if !present {
		prefillContextSize = 1
	}
	_, hasDataParallelSize := values[dataParallelSizeFlag]
	_, hasDataParallelSizeLocal := values[dataParallelSizeLocalFlag]
	return parsedVLLMProfileGeometry{
		tensorParallelSize:       values[tensorParallelSizeFlag],
		pipelineParallelSize:     values[pipelineParallelSizeFlag],
		prefillContextSize:       prefillContextSize,
		dataParallelSize:         values[dataParallelSizeFlag],
		hasDataParallelSize:      hasDataParallelSize,
		dataParallelSizeLocal:    values[dataParallelSizeLocalFlag],
		hasDataParallelSizeLocal: hasDataParallelSizeLocal,
		enableElasticEP:          hasExactVLLMProfileFlag(inspectableArgv, enableElasticEPFlag),
		dataParallelExternalLB:   hasExactVLLMProfileFlag(inspectableArgv, vllmDataParallelExternalLBFlag),
		dataParallelHybridLB:     hasExactVLLMProfileFlag(inspectableArgv, vllmDataParallelHybridLBFlag),
	}, nil
}

func applyVLLMProfileEnvironment(
	geometry parsedVLLMProfileGeometry,
	environment map[string]string,
	hasUnresolvedEnvironment bool,
) (parsedVLLMProfileGeometry, error) {
	// Reject opaque Kubernetes sources because they may supply vLLM's native DP cardinality.
	if hasUnresolvedEnvironment {
		return parsedVLLMProfileGeometry{}, &UnsupportedVLLMProfileSourceError{
			Reason: UnsupportedVLLMProfileSourceReasonEnvironment,
			Detail: "envFrom or valueFrom may supply VLLM_DP_SIZE",
		}
	}

	// vLLM falls back to its native DP environment when effective CLI DP is at most one
	// and local DP is nonzero, even if --data-parallel-size=1 was explicit.
	effectiveDPSize := int64(1)
	if geometry.hasDataParallelSize {
		effectiveDPSize = geometry.dataParallelSize
	}
	effectiveLocalDPSize := int64(1)
	if geometry.hasDataParallelSizeLocal {
		effectiveLocalDPSize = geometry.dataParallelSizeLocal
	}
	if effectiveDPSize > 1 || effectiveLocalDPSize == 0 {
		return geometry, nil
	}

	literal, present := environment[vllmDPSizeEnvironment]
	if !present {
		return geometry, nil
	}
	value, err := parseVLLMGeometryLiteral(vllmDPSizeEnvironment, literal, false)
	if err != nil {
		return parsedVLLMProfileGeometry{}, err
	}
	geometry.dataParallelSize = value
	geometry.hasDataParallelSize = true

	return geometry, nil
}

func hasExactVLLMProfileFlag(argv []string, flag string) bool {
	// BooleanOptionalAction flags are present only as exact, value-free option tokens.
	for _, token := range argv {
		if normalizeVLLMOptionToken(token) == flag {
			return true
		}
	}

	return false
}

func resolveVLLMProfileArguments(command, args []string) ([]string, error) {
	// Require the executable explicitly because an image-owned entrypoint is not available to this resolver.
	if len(command) == 0 || command[0] == "" {
		return nil, &UnsupportedVLLMProfileSourceError{
			Reason: UnsupportedVLLMProfileSourceReasonCommandForm,
			Detail: "a non-empty direct executable command is required",
		}
	}

	// Reject shells and environment wrappers before recognizing a direct vLLM invocation.
	executable := filepath.Base(command[0])
	executableFields := strings.Fields(executable)
	if len(executableFields) > 0 && isProfileShellOrWrapper(executableFields[0]) {
		return nil, &UnsupportedVLLMProfileSourceError{
			Reason: UnsupportedVLLMProfileSourceReasonShell,
			Detail: fmt.Sprintf("command executable %q owns the effective argv", command[0]),
		}
	}

	// Build a private effective argv and accept only entrypoints whose argument ownership is known.
	effectiveArgv := make([]string, 0, len(command)+len(args))
	effectiveArgv = append(effectiveArgv, command...)
	effectiveArgv = append(effectiveArgv, args...)
	if isSupportedProfilePythonExecutable(executable) &&
		len(effectiveArgv) >= 3 &&
		effectiveArgv[1] == "-m" &&
		effectiveArgv[2] == vllmDynamoModule {
		return effectiveArgv[3:], nil
	}
	if executable == vllmEngineGroupBackend &&
		len(effectiveArgv) >= 2 &&
		effectiveArgv[1] == vllmServeSubcommand {
		return effectiveArgv[2:], nil
	}

	return nil, &UnsupportedVLLMProfileSourceError{
		Reason: UnsupportedVLLMProfileSourceReasonCommandForm,
		Detail: "expected a direct python -m dynamo.vllm or vllm serve invocation",
	}
}

func validateInspectableVLLMProfileArguments(argv []string) error {
	// Reject Kubernetes expansion only where it can change which argv option vLLM observes.
	for _, token := range argv {
		if !hasKubernetesArgumentExpansion(token) {
			continue
		}
		name, _, hasValue := strings.Cut(token, "=")
		if hasValue && !hasKubernetesArgumentExpansion(name) {
			continue
		}

		return &UnsupportedVLLMProfileSourceError{
			Reason: UnsupportedVLLMProfileSourceReasonEnvironment,
			Detail: fmt.Sprintf("argument %q contains option-bearing Kubernetes environment expansion", token),
		}
	}

	// Reject argparse abbreviations that could otherwise bypass an exact geometry or placement match.
	if err := validateNoAbbreviatedVLLMProfileOptions(argv); err != nil {
		return err
	}

	// External vLLM configuration can supply geometry that is absent from or conflicts with argv.
	for _, token := range argv {
		normalized := normalizeVLLMOptionToken(token)
		if normalized == "--config" || strings.HasPrefix(normalized, "--config=") {
			return &UnsupportedVLLMProfileSourceError{
				Reason: UnsupportedVLLMProfileSourceReasonExternalConfig,
				Detail: "--config may supply or override engine geometry",
			}
		}
	}

	// Reject launch controls that prevent proving one complete replica stays in this capacity pod.
	return validatePodLocalVLLMPlacement(argv)
}

func validateNoAbbreviatedVLLMProfileOptions(argv []string) error {
	// vLLM accepts unambiguous long-option prefixes, but profile resolution binds only exact declarations.
	for _, token := range argv {
		normalized := normalizeVLLMOptionToken(token)
		name, _, _ := strings.Cut(normalized, "=")
		if !strings.HasPrefix(name, "--") {
			continue
		}
		exact := false
		for _, option := range vllmProfileSensitiveLongOptions {
			if name == option {
				exact = true
				break
			}
		}
		if exact {
			continue
		}

		for _, option := range vllmProfileSensitiveLongOptions {
			if strings.HasPrefix(option, name) {
				return &UnsupportedVLLMProfileSourceError{
					Reason: UnsupportedVLLMProfileSourceReasonAbbreviatedOption,
					Detail: fmt.Sprintf(
						"argument %s may be interpreted as sensitive option %s",
						name,
						option,
					),
				}
			}
		}
	}

	return nil
}

func splitVLLMGeometryOption(argv []string, index int) (vllmGeometryOption, string, int, bool) {
	// Match exact and equals forms without confusing longer similarly prefixed options.
	token := normalizeVLLMOptionToken(argv[index])
	for _, option := range vllmProfileGeometryOptions {
		for _, alias := range option.aliases {
			if token == alias {
				if index+1 == len(argv) {
					return option, "", 0, true
				}

				return option, argv[index+1], 1, true
			}
			if strings.HasPrefix(token, alias+"=") {
				return option, strings.TrimPrefix(token, alias+"="), 0, true
			}
		}
	}

	return vllmGeometryOption{}, "", 0, false
}

func normalizeVLLMOptionToken(token string) string {
	// Match FlexibleArgumentParser by treating underscores and dashes alike in long option names.
	if !strings.HasPrefix(token, "--") {
		return token
	}
	name, value, hasValue := strings.Cut(token, "=")
	name = strings.ReplaceAll(name, "_", "-")
	if hasValue {
		return name + "=" + value
	}

	return name
}

func validatePodLocalVLLMPlacement(argv []string) error {
	// Any explicit distributed placement control falls outside the first pod-local profile.
	for _, token := range argv {
		normalized := normalizeVLLMOptionToken(token)
		name, _, _ := strings.Cut(normalized, "=")
		switch name {
		case distributedExecutorFlag,
			vllmNodesFlag,
			vllmNodesAlias,
			vllmNodeRankFlag,
			vllmNodeRankAlias,
			vllmMasterAddressFlag,
			vllmMasterPortFlag,
			vllmHeadlessFlag,
			vllmDeviceIDsFlag:
			return &UnsupportedVLLMProfileSourceError{
				Reason: UnsupportedVLLMProfileSourceReasonDistributedPlacement,
				Detail: fmt.Sprintf("placement option %s is outside the pod-local profile", name),
			}
		}
	}

	return nil
}

func parseVLLMGeometryLiteral(flag, literal string, allowZero bool) (int64, error) {
	// Parse only decimal cardinalities, allowing vLLM's control-process local-DP sentinel explicitly.
	value, err := strconv.ParseInt(literal, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("vLLM geometry argument %s must be an integer literal, got %q: %w", flag, literal, err)
	}
	if value < 0 || (!allowZero && value == 0) {
		return 0, fmt.Errorf("vLLM geometry argument %s must be positive, got %d", flag, value)
	}

	return value, nil
}
