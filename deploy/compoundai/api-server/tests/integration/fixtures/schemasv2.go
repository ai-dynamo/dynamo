package fixtures

import (
	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/schemas"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/schemasv2"
)

// DefaultScalingSpec generates a default ScalingSpec
func DefaultScalingSpec() schemasv2.ScalingSpec {
	return schemasv2.ScalingSpec{
		MinReplicas: 1,
		MaxReplicas: 10,
	}
}

// DefaultConfigOverridesSpec generates a default ConfigOverridesSpec
func DefaultConfigOverridesSpec() schemasv2.ConfigOverridesSpec {
	return schemasv2.ConfigOverridesSpec{
		Resources: *DefaultResources(),
	}
}

// DefaultServiceSpec generates a default ServiceSpec
func DefaultServiceSpec() schemasv2.ServiceSpec {
	return schemasv2.ServiceSpec{
		Scaling:          DefaultScalingSpec(),
		ConfigOverrides:  DefaultConfigOverridesSpec(),
		ExternalServices: map[string]schemas.ExternalService{},
		ColdStartTimeout: nil,
	}
}

// DefaultDeploymentConfigSchema generates a default DeploymentConfigSchema
func DefaultDeploymentConfigSchema() schemasv2.DeploymentConfigSchema {
	return schemasv2.DeploymentConfigSchema{
		AccessAuthorization: true,
		Envs: map[string]string{
			"ENV_VAR": "value",
		},
		Secrets: map[string]string{
			"SECRET_KEY": "secret-value",
		},
		Services: map[string]schemasv2.ServiceSpec{
			"default-service": DefaultServiceSpec(),
		},
	}
}

// DefaultUpdateDeploymentSchema generates a default UpdateDeploymentSchema
func DefaultUpdateDeploymentSchemaV2() schemasv2.UpdateDeploymentSchema {
	return schemasv2.UpdateDeploymentSchema{
		DeploymentConfigSchema: DefaultDeploymentConfigSchema(),
		CompoundNim:            "nvidia:123456",
	}
}

// DefaultCreateDeploymentSchema generates a default CreateDeploymentSchema
func DefaultCreateDeploymentSchemaV2() schemasv2.CreateDeploymentSchema {
	return schemasv2.CreateDeploymentSchema{
		Name:                   "default-deployment",
		UpdateDeploymentSchema: DefaultUpdateDeploymentSchemaV2(),
	}
}
