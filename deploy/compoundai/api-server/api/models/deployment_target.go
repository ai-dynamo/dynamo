package models

import "github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/schemas"

type DeploymentTarget struct {
	BaseModel
	CreatorAssociate
	DeploymentAssociate
	DeploymentRevisionAssociate
	CompoundNimVersionAssociate
	DmsAssociate

	Config *schemas.DeploymentTargetConfig `json:"config"`
}

func (s *DeploymentTarget) GetName() string {
	return s.Uid.String()
}

func (s *DeploymentTarget) GetResourceType() schemas.ResourceType {
	return schemas.ResourceTypeDeploymentRevision
}
