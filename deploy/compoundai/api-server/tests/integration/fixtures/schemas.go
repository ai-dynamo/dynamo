package fixtures

import "github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/schemas"

func DefaultCreateClusterSchema() schemas.CreateClusterSchema {
	return schemas.CreateClusterSchema{
		Description: "description",
		KubeConfig:  "",
		Name:        "default",
	}
}

func DefaultUpdateClusterSchema() schemas.UpdateClusterSchema {
	d := "description"
	kc := "kubeconfig"
	return schemas.UpdateClusterSchema{
		Description: &d,
		KubeConfig:  &kc,
	}
}

func DefaultListQuerySchema() schemas.ListQuerySchema {
	return schemas.ListQuerySchema{
		Start:  0,
		Count:  20,
		Search: nil,
	}
}

// DefaultCreateDeploymentSchema generates a default CreateDeploymentSchema
func DefaultCreateDeploymentSchema() schemas.CreateDeploymentSchema {
	return schemas.CreateDeploymentSchema{
		Name:                   "default-deployment",
		KubeNamespace:          "default-namespace",
		UpdateDeploymentSchema: DefaultUpdateDeploymentSchema(),
	}
}

// DefaultUpdateDeploymentSchema generates a default UpdateDeploymentSchema
func DefaultUpdateDeploymentSchema() schemas.UpdateDeploymentSchema {
	description := "default deployment"
	return schemas.UpdateDeploymentSchema{
		Targets: []*schemas.CreateDeploymentTargetSchema{
			DefaultCreateDeploymentTargetSchema(),
		},
		Description: &description,
		DoNotDeploy: false,
	}
}

// DefaultCreateDeploymentTargetSchema generates a default CreateDeploymentTargetSchema
func DefaultCreateDeploymentTargetSchema() *schemas.CreateDeploymentTargetSchema {
	return &schemas.CreateDeploymentTargetSchema{
		CompoundNim: "default-compound-nim",
		Version:     "default-version",
		Config:      DefaultDeploymentTargetConfig(),
	}
}

// DefaultDeploymentTargetConfig generates a default DeploymentTargetConfig
func DefaultDeploymentTargetConfig() *schemas.DeploymentTargetConfig {
	return &schemas.DeploymentTargetConfig{
		KubeResourceUid:     "default-uid",
		KubeResourceVersion: "v1",
		Resources:           DefaultResources(),
		HPAConf:             DefaultDeploymentTargetHPAConf(),
		DeploymentStrategy:  DefaultDeploymentStrategy(),
	}
}

// DefaultResources generates a default Resources struct
func DefaultResources() *schemas.Resources {
	return &schemas.Resources{
		Requests: &schemas.ResourceItem{
			CPU:    "500m",
			Memory: "1Gi",
		},
		Limits: &schemas.ResourceItem{
			CPU:    "1",
			Memory: "2Gi",
		},
	}
}

// DefaultDeploymentTargetHPAConf generates a default DeploymentTargetHPAConf
func DefaultDeploymentTargetHPAConf() *schemas.DeploymentTargetHPAConf {
	qps := int64(1000)
	return &schemas.DeploymentTargetHPAConf{
		CPU:         nil,
		GPU:         nil,
		Memory:      nil,
		QPS:         &qps,
		MinReplicas: int32Ptr(1),
		MaxReplicas: int32Ptr(5),
	}
}

// DefaultDeploymentStrategy generates a default DeploymentStrategy
func DefaultDeploymentStrategy() *schemas.DeploymentStrategy {
	strat := schemas.DeploymentStrategyRollingUpdate
	return &strat
}

// Helper function to return a pointer to an int32
func int32Ptr(i int32) *int32 {
	return &i
}
