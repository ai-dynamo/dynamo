/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"fmt"
	"path/filepath"
	"strings"
)

const vllmWorkerModule = "dynamo.vllm"

type vllmProfileOption struct {
	name    string
	aliases []string
	boolean bool
}

var vllmProfileOptions = []vllmProfileOption{
	{name: "--host"},
	{name: "--port"},
	{name: "--uds"},
	{name: "--ssl-keyfile"},
	{name: "--ssl-certfile"},
	{name: "--tensor-parallel-size", aliases: []string{"-tp"}},
	{name: "--pipeline-parallel-size", aliases: []string{"-pp"}},
	{name: "--decode-context-parallel-size", aliases: []string{"-dcp"}},
	{name: "--prefill-context-parallel-size", aliases: []string{"-pcp"}},
	{name: "--data-parallel-size", aliases: []string{"-dp"}},
	{name: "--data-parallel-size-local", aliases: []string{"-dpl"}},
	{name: "--data-parallel-rank", aliases: []string{"-dpn"}},
	{name: "--data-parallel-start-rank", aliases: []string{"-dpr"}},
	{name: "--data-parallel-backend", aliases: []string{"-dpb"}},
	{name: "--data-parallel-address", aliases: []string{"-dpa"}},
	{name: "--data-parallel-rpc-port", aliases: []string{"-dpp"}},
	{name: "--distributed-executor-backend"},
	{name: "--nnodes", aliases: []string{"-n"}},
	{name: "--node-rank", aliases: []string{"-r"}},
	{name: "--api-server-count"},
	{name: "--elastic-ep-max-dp-size"},
	{name: "--elastic-ep-initial-dp-size"},
	{name: "--elastic-ep-auto-bootstrap", boolean: true},
	{name: "--elastic-ep-allocation-index"},
	{name: "--all2all-backend"},
	{name: "--eplb-config"},
	{name: "--enable-expert-parallel", aliases: []string{"-ep"}, boolean: true},
	{name: "--enable-eplb", boolean: true},
	{name: "--enable-elastic-ep", boolean: true},
	{name: "--data-parallel-external-lb", aliases: []string{"-dpe"}, boolean: true},
	{name: "--data-parallel-hybrid-lb", aliases: []string{"-dph"}, boolean: true},
	{name: "--data-parallel-multi-port-external-lb", aliases: []string{"-dpm"}, boolean: true},
	{name: "--enable-fault-tolerance", boolean: true},
	{name: "--headless", boolean: true},
	{name: "--device-ids"},
	{name: "--disaggregation-mode"},
	{name: "--kv-transfer-config"},
	{name: "--enable-lora", boolean: true},
	{name: "--embedding-worker", boolean: true},
	{name: "--classify-worker", boolean: true},
	{name: "--realtime", boolean: true},
	{name: "--enable-rl", boolean: true},
	{name: "--gms-shadow-mode", boolean: true},
	{name: "--route-to-encoder", boolean: true},
}

func collectVLLMProfileOptions(command, args []string) (map[string]string, map[string]bool, error) {
	// Inspect only explicit exec-form entrypoints whose arguments retain native vLLM semantics.
	argv, err := resolveVLLMProfileArguments(command, args)
	if err != nil {
		return nil, nil, err
	}
	values := make(map[string]string)
	flags := make(map[string]bool)
	for index := 0; index < len(argv); index++ {
		token := argv[index]
		if token == "--" {
			return nil, nil, fmt.Errorf("vLLM profile requires explicit options without an argument terminator")
		}
		name, value, inline := strings.Cut(token, "=")
		if hasKubernetesArgumentExpansion(name) {
			return nil, nil, &UnsupportedVLLMProfileSourceError{Reason: UnsupportedVLLMProfileSourceReasonEnvironment, Detail: "an argument can expand into geometry options"}
		}
		if strings.HasPrefix(name, "--") {
			name = strings.ReplaceAll(name, "_", "-")
		}
		if name == "-c" || (strings.HasPrefix(name, "--") && strings.HasPrefix("--config", name)) || strings.HasPrefix(name, "--eplb-config.") {
			return nil, nil, &UnsupportedVLLMProfileSourceError{Reason: UnsupportedVLLMProfileSourceReasonExternalConfig, Detail: name + " must be resolved before profile extraction"}
		}

		// Normalize documented Boolean inverses before checking exact names or ambiguous prefixes.
		positive := name
		negated := strings.HasPrefix(name, "--no-") || strings.HasPrefix(name, "--disable-")
		if strings.HasPrefix(name, "--no-") {
			positive = "--" + strings.TrimPrefix(name, "--no-")
		} else if strings.HasPrefix(name, "--disable-") {
			positive = "--enable-" + strings.TrimPrefix(name, "--disable-")
		}
		option, matched := matchVLLMProfileOption(positive)
		if !matched {
			continue
		}
		if positive != option.name && !isVLLMProfileAlias(option, positive) {
			return nil, nil, &UnsupportedVLLMProfileSourceError{Reason: UnsupportedVLLMProfileSourceReasonAbbreviatedOption, Detail: name + " is not an exact supported option"}
		}
		if _, present := values[option.name]; present {
			return nil, nil, fmt.Errorf("vLLM option %s is specified more than once", option.name)
		}
		if _, present := flags[option.name]; present {
			return nil, nil, fmt.Errorf("vLLM option %s is specified more than once", option.name)
		}

		// Booleans are switches; value options accept split or equals syntax without interpolation.
		if option.boolean {
			if inline {
				return nil, nil, fmt.Errorf("vLLM flag %s does not accept a value", name)
			}
			flags[option.name] = !negated
			continue
		}
		if negated {
			return nil, nil, fmt.Errorf("vLLM option %s is not a Boolean switch", name)
		}
		if !inline {
			if index+1 >= len(argv) || strings.HasPrefix(argv[index+1], "-") {
				return nil, nil, fmt.Errorf("vLLM option %s requires a value", name)
			}
			index++
			value = argv[index]
		}
		if value == "" {
			return nil, nil, fmt.Errorf("vLLM option %s requires a nonempty value", name)
		}
		if hasKubernetesArgumentExpansion(value) && !isVLLMBootstrapArgument(option.name, value) {
			return nil, nil, &UnsupportedVLLMProfileSourceError{Reason: UnsupportedVLLMProfileSourceReasonEnvironment, Detail: name + " must be resolved to a literal value"}
		}
		values[option.name] = value
	}
	return values, flags, nil
}

func isVLLMBootstrapArgument(name, value string) bool {
	switch name {
	case "--data-parallel-rank", "--elastic-ep-allocation-index":
		return value == engineGroupOrdinalRef
	case "--data-parallel-address":
		return value == engineGroupRendezvousRef
	default:
		return false
	}
}

func resolveVLLMProfileArguments(command, args []string) ([]string, error) {
	// Caller-owned command and arguments may split the Python module or CLI prefix anywhere.
	argv := append(append([]string(nil), command...), args...)
	if len(command) == 0 {
		return nil, &UnsupportedVLLMProfileSourceError{Reason: UnsupportedVLLMProfileSourceReasonCommandForm, Detail: "an explicit vLLM executable is required"}
	}
	executable := filepath.Base(argv[0])
	if isProfileShellOrWrapper(executable) {
		return nil, &UnsupportedVLLMProfileSourceError{Reason: UnsupportedVLLMProfileSourceReasonShell, Detail: executable + " owns the effective vLLM argv"}
	}
	if isSupportedProfilePythonExecutable(executable) && len(argv) >= 3 && argv[1] == "-m" &&
		(argv[2] == vllmWorkerModule || argv[2] == "vllm.entrypoints.openai.api_server") {
		return argv[3:], nil
	}
	if executable == "vllm" && len(argv) >= 2 && argv[1] == "serve" {
		return argv[2:], nil
	}
	return nil, &UnsupportedVLLMProfileSourceError{Reason: UnsupportedVLLMProfileSourceReasonCommandForm, Detail: "expected vllm serve or a supported vLLM/Dynamo Python module"}
}

func matchVLLMProfileOption(name string) (vllmProfileOption, bool) {
	// Prefer exact aliases: -dp is not an abbreviation of -dpl or -dpn.
	for _, option := range vllmProfileOptions {
		if option.name == name || isVLLMProfileAlias(option, name) {
			return option, true
		}
	}
	if strings.HasPrefix(name, "--") {
		for _, option := range vllmProfileOptions {
			if strings.HasPrefix(option.name, name) {
				return option, true
			}
		}
	}
	return vllmProfileOption{}, false
}

func isVLLMProfileAlias(option vllmProfileOption, name string) bool {
	for _, alias := range option.aliases {
		if alias == name {
			return true
		}
	}
	return false
}
