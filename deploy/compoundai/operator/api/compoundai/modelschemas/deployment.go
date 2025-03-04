package modelschemas

type DeploymentStatus string

const (
	DeploymentStatusUnknown             DeploymentStatus = "unknown"
	DeploymentStatusNonDeployed         DeploymentStatus = "non-deployed"
	DeploymentStatusRunning             DeploymentStatus = "running"
	DeploymentStatusUnhealthy           DeploymentStatus = "unhealthy"
	DeploymentStatusFailed              DeploymentStatus = "failed"
	DeploymentStatusDeploying           DeploymentStatus = "deploying"
	DeploymentStatusTerminating         DeploymentStatus = "terminating"
	DeploymentStatusTerminated          DeploymentStatus = "terminated"
	DeploymentStatusImageBuilding       DeploymentStatus = "image-building"
	DeploymentStatusImageBuildFailed    DeploymentStatus = "image-build-failed"
	DeploymentStatusImageBuildSucceeded DeploymentStatus = "image-build-succeeded"
)

func (d DeploymentStatus) Ptr() *DeploymentStatus {
	return &d
}

type DeploymentMode string

const (
	DeploymentModeDeployment DeploymentMode = "deployment"
	DeploymentModeFunction   DeploymentMode = "function"
)

func (d DeploymentMode) Ptr() *DeploymentMode {
	return &d
}
