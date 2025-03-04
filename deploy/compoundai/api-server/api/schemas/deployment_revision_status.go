package schemas

type DeploymentRevisionStatus string

const (
	DeploymentRevisionStatusActive   DeploymentRevisionStatus = "active"
	DeploymentRevisionStatusInactive DeploymentRevisionStatus = "inactive"
)

func (d DeploymentRevisionStatus) Ptr() *DeploymentRevisionStatus {
	return &d
}

func DeploymentRevisionStatusPtr(status DeploymentRevisionStatus) *DeploymentRevisionStatus {
	return status.Ptr()
}
