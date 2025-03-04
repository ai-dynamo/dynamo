package converters

import (
	"context"
	"fmt"
	"strings"

	"github.com/pkg/errors"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/mocks"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/models"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/schemas"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/services"
)

func ToDeploymentTargetSchema(ctx context.Context, deploymentTarget *models.DeploymentTarget) (*schemas.DeploymentTargetSchema, error) {
	if deploymentTarget == nil {
		return nil, nil
	}
	ss, err := ToDeploymentTargetSchemas(ctx, []*models.DeploymentTarget{deploymentTarget})
	if err != nil {
		return nil, err
	}
	return ss[0], nil
}

func ToDeploymentTargetSchemas(ctx context.Context, deploymentTargets []*models.DeploymentTarget) ([]*schemas.DeploymentTargetSchema, error) {
	resourceSchemasMap := make(map[string]*schemas.ResourceSchema, len(deploymentTargets))
	for _, target := range deploymentTargets {
		resourceSchemasMap[target.GetUid()] = ToResourceSchema(targetToResource(target), target.GetResourceType())
	}

	res := make([]*schemas.DeploymentTargetSchema, 0, len(deploymentTargets))
	for _, deploymentTarget := range deploymentTargets {
		creatorSchema := mocks.DefaultUser()

		compoundNimParts := strings.Split(deploymentTarget.CompoundNimVersionTag, ":")
		if len(compoundNimParts) != 2 {
			return nil, errors.Errorf("Invalid format for CompoundNIM version tag %s. Expected 2 parts got %d", deploymentTarget.CompoundNimVersionTag, len(compoundNimParts))
		}

		compoundNimVersionFullSchema, err := services.DatastoreService.GetCompoundNimVersion(ctx, compoundNimParts[0], compoundNimParts[1])
		if err != nil {
			compoundNimVersionFullSchema = nil // We shouldn't fail the request if this info is missing
		}

		resourceSchema, ok := resourceSchemasMap[deploymentTarget.GetUid()]
		if !ok {
			return nil, fmt.Errorf("resourceSchema not found for deploymentTarget %s", deploymentTarget.GetUid())
		}
		res = append(res, &schemas.DeploymentTargetSchema{
			ResourceSchema: *resourceSchema,
			DeploymentTargetTypeSchema: schemas.DeploymentTargetTypeSchema{
				Type: "stable",
			},
			Creator:            creatorSchema,
			CompoundNimVersion: compoundNimVersionFullSchema,
			Config:             deploymentTarget.Config,
		})
	}
	return res, nil
}

func targetToResource(deploymentTarget *models.DeploymentTarget) *models.Resource {
	return &models.Resource{
		BaseModel: deploymentTarget.BaseModel,
		Name:      deploymentTarget.GetUid(),
	}
}
