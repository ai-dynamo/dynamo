package conversion

import (
	"github.com/dynemo-ai/dynemo/deploy/compoundai/operator/api/compoundai/common"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/operator/api/compoundai/modelschemas"
)

func ConvertToDeploymentTargetResourceItem(src *common.ResourceItem) (dest *modelschemas.DeploymentTargetResourceItem) {
	if src == nil {
		return
	}
	dest = &modelschemas.DeploymentTargetResourceItem{
		CPU:    src.CPU,
		Memory: src.Memory,
		GPU:    src.GPU,
		Custom: src.Custom,
	}
	return
}

func ConvertToDeploymentTargetResources(src *common.Resources) (dest *modelschemas.DeploymentTargetResources) {
	if src == nil {
		return
	}
	dest = &modelschemas.DeploymentTargetResources{
		Requests: ConvertToDeploymentTargetResourceItem(src.Requests),
		Limits:   ConvertToDeploymentTargetResourceItem(src.Limits),
	}
	return
}

func ConvertFromDeploymentTargetResourceItem(src *modelschemas.DeploymentTargetResourceItem) (dest *common.ResourceItem) {
	if src == nil {
		return
	}
	dest = &common.ResourceItem{
		CPU:    src.CPU,
		Memory: src.Memory,
		GPU:    src.GPU,
		Custom: src.Custom,
	}
	return
}

func ConvertFromDeploymentTargetResources(src *modelschemas.DeploymentTargetResources) (dest *common.Resources) {
	if src == nil {
		return
	}
	dest = &common.Resources{
		Requests: ConvertFromDeploymentTargetResourceItem(src.Requests),
		Limits:   ConvertFromDeploymentTargetResourceItem(src.Limits),
	}
	return
}
