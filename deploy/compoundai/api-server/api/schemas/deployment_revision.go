package schemas

type DeploymentRevisionSchema struct {
	ResourceSchema
	Creator *UserSchema               `json:"creator"`
	Status  DeploymentRevisionStatus  `json:"status" enum:"active,inactive"`
	Targets []*DeploymentTargetSchema `json:"targets"`
}

type DeploymentRevisionListSchema struct {
	BaseListSchema
	Items []*DeploymentRevisionSchema `json:"items"`
}

type GetDeploymentRevisionSchema struct {
	GetDeploymentSchema
	RevisionUid string `uri:"revisionUid" binding:"required"`
}
