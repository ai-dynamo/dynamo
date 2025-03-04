package crds

var ApiVersion string = "nvidia.com/v1alpha1"

type CustomResourceType string

const (
	CompoundNimRequest    CustomResourceType = "CompoundAINimRequest"
	CompoundNimDeployment CustomResourceType = "CompoundAINimDeployment"
)
