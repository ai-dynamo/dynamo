package dynamo

import (
	"fmt"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	ptr "k8s.io/utils/ptr"
)

type VLLMBackend struct{}

func (b *VLLMBackend) GenerateCommandAndArgs(componentType string, numberOfNodes int32, role Role, component *v1alpha1.DynamoComponentDeploymentOverridesSpec, multinodeDeploymentType consts.MultinodeDeploymentType) ([]string, []string) {
	cmd := []string{"/bin/sh", "-c"}
	args := buildVLLMArgs(component.DynamoConfig)
	var argStr string
	// Note: componentType parameter is used to generate different commands for different component types.
	// It's the caller's responsibility to ensure componentType matches component.ComponentType if needed.
	switch componentType {
	case commonconsts.ComponentTypeMain:
		argStr = "python3 -m dynamo.frontend --http-port 8000"
	case commonconsts.ComponentTypeWorker:
		if numberOfNodes == 1 {
			argStr = fmt.Sprintf("python3 -m dynamo.vllm %s", strings.Join(args, " "))
		} else if role == RoleWorker {
			if multinodeDeploymentType == consts.MultinodeDeploymentTypeGrove {
				argStr = "ray start --address=${GROVE_HEADLESS_SERVICE}:6379 --block"
			} else {
				argStr = "ray start --address=${LWS_LEADER_ADDRESS}:6379 --block"
			}
		} else {
			argStr = fmt.Sprintf("ray start --head --port=6379 && python3 -m dynamo.vllm %s", strings.Join(args, " "))
		}
	case commonconsts.ComponentTypePrefillWorker:
		if numberOfNodes == 1 {
			argStr = fmt.Sprintf("python3 -m dynamo.vllm --is-prefill-worker %s", strings.Join(args, " "))
		} else if role == RoleWorker {
			if multinodeDeploymentType == consts.MultinodeDeploymentTypeGrove {
				argStr = "ray start --address=${GROVE_HEADLESS_SERVICE}:6379 --block"
			} else {
				argStr = "ray start --address=${LWS_LEADER_ADDRESS}:6379 --block"
			}
		} else {
			argStr = fmt.Sprintf("ray start --head --port=6379 && python3 -m dynamo.vllm --is-prefill-worker %s", strings.Join(args, " "))
		}
	case commonconsts.ComponentTypeDecodeWorker:
		if numberOfNodes == 1 {
			argStr = fmt.Sprintf("python3 -m dynamo.vllm %s", strings.Join(args, " "))
		} else if role == RoleWorker {
			if multinodeDeploymentType == consts.MultinodeDeploymentTypeGrove {
				argStr = "ray start --address=${GROVE_HEADLESS_SERVICE}:6379 --block"
			} else {
				argStr = "ray start --address=${LWS_LEADER_ADDRESS}:6379 --block"
			}
		} else {
			argStr = fmt.Sprintf("ray start --head --port=6379 && python3 -m dynamo.vllm %s", strings.Join(args, " "))
		}
	default:
		argStr = fmt.Sprintf("python3 -m dynamo.vllm %s", strings.Join(args, " "))
	}
	return cmd, []string{argStr}
}

func (b *VLLMBackend) MergeArgs(defaultArgs, userArgs []string, multinode bool, role Role, componentType string, numberOfNodes int32, component *v1alpha1.DynamoComponentDeploymentOverridesSpec, multinodeDeploymentType consts.MultinodeDeploymentType) []string {
	if len(userArgs) == 0 {
		return defaultArgs
	}
	if multinode {
		multiArgs := buildVLLMArgs(component.DynamoConfig)
		switch role {
		case RoleLeader:
			return []string{fmt.Sprintf("ray start --head --port=6379 && %s", strings.Join(append(userArgs, multiArgs...), " "))}
		case RoleWorker:
			if multinodeDeploymentType == consts.MultinodeDeploymentTypeGrove {
				return []string{"ray start --address=${GROVE_HEADLESS_SERVICE}:6379 --block"}
			} else {
				return []string{"ray start --address=${LWS_LEADER_ADDRESS}:6379 --block"}
			}
		}
	}
	return userArgs
}

func buildVLLMArgs(
	dynamoConfig *v1alpha1.DynamoConfig,
) []string {
	baseFlags := map[string]*string{}
	// Set defaults from config
	if dynamoConfig != nil {
		if dynamoConfig.TensorParallelSize != nil {
			baseFlags["tensor-parallel-size"] = ptr.To(fmt.Sprintf("%d", *dynamoConfig.TensorParallelSize))
		}
		if dynamoConfig.DataParallelSize != nil {
			baseFlags["data-parallel-size"] = ptr.To(fmt.Sprintf("%d", *dynamoConfig.DataParallelSize))
		}
	}
	var flagOverrides map[string]*string
	var extraArgs []string
	if dynamoConfig != nil {
		flagOverrides = dynamoConfig.FlagOverrides
		extraArgs = dynamoConfig.ExtraArgs
	}
	return applyFlagOverridesAndExtraArgs(baseFlags, flagOverrides, extraArgs)
}
