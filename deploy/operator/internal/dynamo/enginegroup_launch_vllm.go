/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"fmt"
	"path/filepath"
	"strconv"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
)

// ResolveVLLMLaunchProfile validates native execution as well as declarative geometry before allocating capacity.
// Native servers and the existing Dynamo Python worker retain their own entrypoints.
func ResolveVLLMLaunchProfile(source VLLMProfileGeometrySource) (ResolvedVLLMElasticEPProfile, error) {
	// Native launch constraints are known before any Pod-specific assignment exists.
	profile, err := ResolveVLLMElasticEPProfile(source)
	if err != nil {
		return ResolvedVLLMElasticEPProfile{}, err
	}
	values, flags, err := collectVLLMProfileOptions(source.Command, source.Args)
	if err != nil {
		return ResolvedVLLMElasticEPProfile{}, err
	}
	if !flags["--elastic-ep-auto-bootstrap"] || !flags["--data-parallel-external-lb"] ||
		values["--data-parallel-rank"] != engineGroupOrdinalRef || values["--data-parallel-size-local"] != "1" ||
		values["--data-parallel-address"] != engineGroupRendezvousRef {
		return ResolvedVLLMElasticEPProfile{}, fmt.Errorf("vLLM Engine Group template must explicitly declare native automatic bootstrap, external LB, local DP=1, --data-parallel-rank=%s and --data-parallel-address=%s", engineGroupOrdinalRef, engineGroupRendezvousRef)
	}
	if _, _, err := ResolveVLLMControlPorts(source); err != nil {
		return ResolvedVLLMElasticEPProfile{}, err
	}
	if IsVLLMWorkerLaunch(source) {
		for _, option := range []string{"--enable-lora", "--embedding-worker", "--classify-worker", "--realtime", "--enable-rl", "--gms-shadow-mode", "--route-to-encoder"} {
			if flags[option] {
				return ResolvedVLLMElasticEPProfile{}, fmt.Errorf("managed vLLM worker admission does not support %s", option)
			}
		}
		if (values["--disaggregation-mode"] != "" && values["--disaggregation-mode"] != "agg") || values["--kv-transfer-config"] != "" {
			return ResolvedVLLMElasticEPProfile{}, fmt.Errorf("managed vLLM worker admission requires aggregated generation without a KV-transfer lifecycle")
		}
	}
	return profile, nil
}

// ResolveVLLMControlPorts derives the native HTTP and DP rendezvous ports from a literal declaration.
// This transport profile requires Pod-reachable HTTP, not a Unix socket or native TLS listener.
func ResolveVLLMControlPorts(source VLLMProfileGeometrySource) (int32, int32, error) {
	// Use the same option parser as launch validation, including duplicates and abbreviated-option rejection.
	values, _, err := collectVLLMProfileOptions(source.Command, source.Args)
	if err != nil {
		return 0, 0, err
	}
	if host := values["--host"]; (host != "" && host != "0.0.0.0" && host != "::") ||
		values["--uds"] != "" || values["--ssl-keyfile"] != "" || values["--ssl-certfile"] != "" {
		return 0, 0, fmt.Errorf("vLLM Engine Group control requires an HTTP listener on all Pod interfaces")
	}
	ports := []int32{8000, 13445}
	if IsVLLMWorkerLaunch(source) {
		if values["--port"] != "" {
			return 0, 0, fmt.Errorf("Dynamo worker control uses DYN_SYSTEM_PORT, not vLLM --port")
		}
		ports[0] = consts.DynamoSystemPort
		for _, variable := range source.Env {
			if variable.Name != "DYN_SYSTEM_PORT" {
				continue
			}
			port, err := strconv.ParseInt(variable.Value, 10, 32)
			if variable.ValueFrom != nil || err != nil || port < 1 || port > 32767 {
				return 0, 0, fmt.Errorf("Dynamo worker requires a literal positive DYN_SYSTEM_PORT")
			}
			ports[0] = int32(port)
		}
	}
	for i, name := range []string{"--port", "--data-parallel-rpc-port"} {
		if literal, present := values[name]; present {
			port, err := strconv.ParseInt(literal, 10, 32)
			if err != nil || port < 1 || port > 65535 {
				return 0, 0, fmt.Errorf("vLLM %s must be a valid TCP port", name)
			}
			ports[i] = int32(port)
		}
	}
	if ports[0] == ports[1] {
		return 0, 0, fmt.Errorf("vLLM HTTP and DP rendezvous ports must differ")
	}
	return ports[0], ports[1], nil
}

// IsVLLMWorkerLaunch identifies the existing Python worker, not a native API server or sidecar.
func IsVLLMWorkerLaunch(source VLLMProfileGeometrySource) bool {
	// Command/args may split the module prefix; do not mutate either caller-owned slice.
	argv := append(append([]string(nil), source.Command...), source.Args...)
	return len(argv) >= 3 && isSupportedProfilePythonExecutable(filepath.Base(argv[0])) &&
		argv[1] == "-m" && argv[2] == vllmWorkerModule
}
