/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// applyTRTLLMOverrides — TRT-LLM failover env injection and port staggering.
//
// TRT-LLM does not expose a --master-port CLI flag; torch.distributed picks
// up MASTER_PORT from the environment. When two engine containers share a
// pod network namespace and use TP>1, both torch.distributed groups would
// otherwise collide on the default 29500 TCP store.
//
// Env injected per engine:
//
//	DYN_TRTLLM_GMS_SHADOW_MODE=true       (contract for TRT-LLM shadow mode;
//	                                       not yet consumed by
//	                                       components/src/dynamo/trtllm or
//	                                       lib/gpu_memory_service/integrations/trtllm.
//	                                       Mirrors DYN_VLLM_GMS_SHADOW_MODE.)
//	MASTER_PORT=<base + engineID*stride>  (torch.distributed TCP store)
//	NNODES=<numberOfNodes>                (multinode only)
//
// TODOs (not blocking this change):
//   - DYN_KVBM_LEADER_ZMQ_PUB_PORT (default 56001) collides when KVBM
//     connector is enabled for both engines — stagger when that path ships
//     under failover.
//   - NIXL side-channel port for TRT-LLM cross-engine transfers is not yet
//     exposed via a knob; revisit if we run disagg/multi-engine NIXL under
//     failover.

package dynamo

import (
	"strconv"
	"strings"

	corev1 "k8s.io/api/core/v1"
)

const (
	trtllmMasterPortBase   = 29500
	trtllmMasterPortStride = 100
)

func applyTRTLLMOverrides(podSpec *corev1.PodSpec, numberOfNodes int32) {
	for i := range podSpec.Containers {
		c := &podSpec.Containers[i]
		if !strings.HasPrefix(c.Name, "engine-") {
			continue
		}

		engineID, _ := strconv.Atoi(strings.TrimPrefix(c.Name, "engine-"))

		c.Env = append(c.Env,
			corev1.EnvVar{Name: "DYN_TRTLLM_GMS_SHADOW_MODE", Value: "true"},
			corev1.EnvVar{
				Name:  "MASTER_PORT",
				Value: strconv.Itoa(trtllmMasterPortBase + engineID*trtllmMasterPortStride),
			},
		)

		if numberOfNodes > 1 {
			c.Env = append(c.Env,
				corev1.EnvVar{Name: "NNODES", Value: strconv.Itoa(int(numberOfNodes))},
			)
		}
	}
}
