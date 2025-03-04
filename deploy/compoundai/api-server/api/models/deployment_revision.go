package models

import (
	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/schemas"
)

type DeploymentRevision struct {
	BaseModel
	CreatorAssociate
	DeploymentAssociate

	Status schemas.DeploymentRevisionStatus `json:"status"`
}

func (s *DeploymentRevision) GetName() string {
	return s.Uid.String()
}

func (s *DeploymentRevision) GetResourceType() schemas.ResourceType {
	return schemas.ResourceTypeDeploymentRevision
}
