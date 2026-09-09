/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	_ "embed"
	"fmt"
	"strconv"
	"strings"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

const (
	// LPUSSHPort is the SSH port used by Nova and LPU Agent workers.
	LPUSSHPort = commonconsts.MpiRunSshPort
	// LPUReadinessPort is the HTTP readiness port used by the LPU runtime.
	LPUReadinessPort = 19878
	lpuRDMAPort      = 19877

	lpuCommandTemplateTokenPrefix = "@@LPX_"
	lpuConfigMountPathToken       = "@@LPX_CONFIG_MOUNT_PATH@@"
	lpuSSHPortToken               = "@@LPX_SSH_PORT@@"
	lpuReadinessPortToken         = "@@LPX_READINESS_PORT@@"
	lpuRDMAPortToken              = "@@LPX_RDMA_PORT@@"

	embeddedShellSPDXHeader = `# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

`

	lpuPartitionSetEnvCommands = `SETENV_LINE="${SETENV_LINE} PARTITION_ID=\"$PARTITION_ID\""
SETENV_LINE="${SETENV_LINE} RANK_IN_PARTITION=\"$RANK_IN_PARTITION\""
`

	lpuV2ProbeCommandTemplate = `if (( NODE_COUNT > 1 )) && [[ "${PARTITION_RANK}" != "0" ]]; then
	cat < /dev/null > "/dev/tcp/127.0.0.1/${MPI_SSH_PORT:-@@LPX_SSH_PORT@@}"
else
	exec 3<>"/dev/tcp/127.0.0.1/${READINESS_PORT:-@@LPX_READINESS_PORT@@}"
	printf "GET /readyz HTTP/1.0\r\n\r\n" >&3
	read -r STATUS <&3
	[[ "${STATUS}" == *" 200 "* ]]
fi`
)

var (
	//go:embed commands/ssh_setup.sh
	sshSetupCommandSource string
	//go:embed commands/lpu_system_init.sh
	lpuSystemInitCommandSource string
	//go:embed commands/lpu_partition_metadata.sh
	lpuPartitionMetadataCommandSource string
	//go:embed commands/lpu_partition_env.sh
	lpuPartitionEnvironmentCommandSource string
	//go:embed commands/lpu_partition_agent.sh
	lpuPartitionRunCommandSource string
	//go:embed commands/lpu_worker.sh
	lpuWorkerRunCommandSource string

	lpuCommandTemplateReplacer = strings.NewReplacer(
		lpuConfigMountPathToken, lpuConfigMountPath,
		lpuSSHPortToken, strconv.Itoa(LPUSSHPort),
		lpuReadinessPortToken, strconv.Itoa(LPUReadinessPort),
		lpuRDMAPortToken, strconv.Itoa(lpuRDMAPort),
	)

	sshSetupCommands               = mustExpandLPUCommandTemplate(sshSetupCommandSource)
	lpuSystemInitCommands          = mustExpandLPUCommandTemplate(lpuSystemInitCommandSource)
	lpuPartitionMetadataCommand    = mustExpandLPUCommandTemplate(lpuPartitionMetadataCommandSource)
	lpuPartitionEnvironmentCommand = mustExpandLPUCommandTemplate(lpuPartitionEnvironmentCommandSource)
	lpuPartitionRunCommand         = lpuSystemInitCommands + "\n" + lpuPartitionMetadataCommand + "\n" + lpuPartitionEnvironmentCommand + "\n" + mustExpandLPUCommandTemplate(lpuPartitionRunCommandSource)
	lpuWorkerRunCommandTemplate    = mustExpandLPUCommandTemplate(lpuWorkerRunCommandSource)
	lpuV2ProbeCommand              = lpuPartitionMetadataCommand + "\n" + mustExpandLPUCommandTemplate(lpuV2ProbeCommandTemplate)

	lpuWorkerRunCommand          = lpuSystemInitCommands + "\n" + fmt.Sprintf(lpuWorkerRunCommandTemplate, sshSetupCommands, "")
	lpuPartitionWorkerRunCommand = lpuSystemInitCommands + "\n" + lpuPartitionMetadataCommand + "\n" + lpuPartitionEnvironmentCommand + "\n" + fmt.Sprintf(lpuWorkerRunCommandTemplate, sshSetupCommands, lpuPartitionSetEnvCommands)
)

func mustExpandLPUCommandTemplate(commandTemplate string) string {
	// Remove only the standard source header so adding SPDX does not roll rendered workloads.
	commandTemplate = strings.TrimPrefix(commandTemplate, embeddedShellSPDXHeader)

	// Resolve every Go-owned default through the package's single source of truth.
	command := lpuCommandTemplateReplacer.Replace(commandTemplate)
	if strings.Contains(command, lpuCommandTemplateTokenPrefix) {
		panic("unexpanded LPX command template token")
	}
	return command
}

const (
	lpuV2HTTPStartupFailureThreshold int32 = 60
	lpuV2ExecStartupFailureThreshold int32 = 240
)

func lpuV2StartupProbe(useHTTP bool) *corev1.Probe {
	probe := lpuV2ReadinessProbe(useHTTP)
	if useHTTP {
		probe.FailureThreshold = lpuV2HTTPStartupFailureThreshold
	} else {
		probe.FailureThreshold = lpuV2ExecStartupFailureThreshold
	}
	return probe
}

func lpuV2ReadinessProbe(useHTTP bool) *corev1.Probe {
	probe := &corev1.Probe{
		FailureThreshold: 6,
		PeriodSeconds:    5,
		SuccessThreshold: 1,
		TimeoutSeconds:   1,
	}
	if useHTTP {
		probe.HTTPGet = &corev1.HTTPGetAction{
			Path:   "/readyz",
			Port:   intstr.FromInt(LPUReadinessPort),
			Scheme: corev1.URISchemeHTTP,
		}
	} else {
		probe.Exec = &corev1.ExecAction{
			Command: []string{"/bin/bash", "-c", lpuV2ProbeCommand},
		}
	}
	return probe
}

func lpuSSHProbe() *corev1.Probe {
	return &corev1.Probe{
		FailureThreshold:    3,
		InitialDelaySeconds: 10,
		PeriodSeconds:       5,
		SuccessThreshold:    1,
		TimeoutSeconds:      1,
		ProbeHandler: corev1.ProbeHandler{
			TCPSocket: &corev1.TCPSocketAction{
				Port: intstr.FromInt(LPUSSHPort),
			},
		},
	}
}
