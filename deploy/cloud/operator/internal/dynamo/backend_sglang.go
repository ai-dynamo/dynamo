package dynamo

import (
	"fmt"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	ptr "k8s.io/utils/ptr"
)

type SGLangBackend struct{}

func (b *SGLangBackend) GenerateCommandAndArgs(componentType string, numberOfNodes int32, role Role, component *v1alpha1.DynamoComponentDeploymentOverridesSpec, multinodeDeploymentType consts.MultinodeDeploymentType) ([]string, []string) {
	cmd := []string{"/bin/sh", "-c"}
	args := buildSGLangArgs(numberOfNodes, role, component.DynamoConfig, multinodeDeploymentType)
	var argStr string
	// Note: componentType parameter is used to generate different commands for different component types.
	// It's the caller's responsibility to ensure componentType matches component.ComponentType if needed.
	switch componentType {
	case commonconsts.ComponentTypeMain:
		argStr = "python3 -m dynamo.frontend --http-port=8000"
	case commonconsts.ComponentTypeWorker:
		argStr = fmt.Sprintf("python3 -m dynamo.sglang.worker %s", strings.Join(args, " "))
	case commonconsts.ComponentTypeDecodeWorker:
		argStr = fmt.Sprintf("python3 -m dynamo.sglang.decode_worker %s", strings.Join(args, " "))
	default:
		argStr = fmt.Sprintf("python3 -m dynamo.sglang.worker %s", strings.Join(args, " "))
	}
	return cmd, []string{argStr}
}

func (b *SGLangBackend) MergeArgs(defaultArgs, userArgs []string, multinode bool, role Role, componentType string, numberOfNodes int32, component *v1alpha1.DynamoComponentDeploymentOverridesSpec, multinodeDeploymentType consts.MultinodeDeploymentType) []string {
	if len(userArgs) == 0 {
		return defaultArgs
	}
	if !multinode {
		return userArgs
	}
	multiArgs := buildSGLangArgs(numberOfNodes, role, component.DynamoConfig, multinodeDeploymentType)
	return []string{strings.Join(append(userArgs, multiArgs...), " ")}
}

func buildSGLangArgs(
	numberOfNodes int32,
	role Role,
	dynamoConfig *v1alpha1.DynamoConfig,
	multinodeDeploymentType consts.MultinodeDeploymentType,
) []string {
	baseFlags := map[string]*string{}
	// Set distributed flags for multinode
	if numberOfNodes > 1 {
		if multinodeDeploymentType == consts.MultinodeDeploymentTypeGrove {
			baseFlags["dist-init-addr"] = ptr.To("${GROVE_HEADLESS_SERVICE}:29500")
		} else {
			baseFlags["dist-init-addr"] = ptr.To("${LWS_LEADER_ADDRESS}:29500")
		}
		baseFlags["nnodes"] = ptr.To(fmt.Sprintf("%d", numberOfNodes))
		if role == RoleLeader {
			baseFlags["node-rank"] = ptr.To("0")
		} else {
			if multinodeDeploymentType == consts.MultinodeDeploymentTypeGrove {
				baseFlags["node-rank"] = ptr.To("$((GROVE_PCLQ_POD_INDEX + 1))")
			} else {
				// todo : add node rank for LWS
			}
		}
	}
	// Set defaults from config
	if dynamoConfig != nil {
		if dynamoConfig.TensorParallelSize != nil {
			baseFlags["tp-size"] = ptr.To(fmt.Sprintf("%d", *dynamoConfig.TensorParallelSize))
		}
		if dynamoConfig.DataParallelSize != nil {
			baseFlags["dp-size"] = ptr.To(fmt.Sprintf("%d", *dynamoConfig.DataParallelSize))
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
