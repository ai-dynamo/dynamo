package models

type ClusterAssociate struct {
	ClusterId              uint     `json:"cluster_id"`
	AssociatedClusterCache *Cluster `gorm:"foreignkey:ClusterId"`
}

type DeploymentAssociate struct {
	DeploymentId              uint        `json:"deployment_id"`
	AssociatedDeploymentCache *Deployment `gorm:"foreignkey:DeploymentId;constraint:OnDelete:CASCADE;"`
}

type DeploymentRevisionAssociate struct {
	DeploymentRevisionId              uint                `json:"deployment_revision_id"`
	AssociatedDeploymentRevisionCache *DeploymentRevision `gorm:"foreignkey:DeploymentRevisionId;constraint:OnDelete:CASCADE;"`
}

type CompoundNimVersionAssociate struct {
	CompoundNimVersionId  string `json:"compound_nim_version_id"`
	CompoundNimVersionTag string `json:"compound_nim_version_tag"`
}

type DmsAssociate struct {
	KubeRequestId    string
	KubeDeploymentId string
}

type OrganizationAssociate struct {
	OrganizationId string `json:"organization_id"` // Set via http headers
}

type CreatorAssociate struct {
	UserId string `json:"user_id"` // Set via http headers
}
