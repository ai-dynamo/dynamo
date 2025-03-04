package v1alpha1

import (
	autoscalingv2 "k8s.io/api/autoscaling/v2"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

type PVC struct {
	// Create indicates to create a new PVC
	Create *bool `json:"create,omitempty"`
	// Name is the name of the PVC
	Name *string `json:"name,omitempty"`
	// StorageClass to be used for PVC creation. Leave it as empty if the PVC is already created.
	StorageClass string `json:"storageClass,omitempty"`
	// Size of the NIM cache in Gi, used during PVC creation
	Size resource.Quantity `json:"size,omitempty"`
	// VolumeAccessMode is the volume access mode of the PVC
	VolumeAccessMode corev1.PersistentVolumeAccessMode `json:"volumeAccessMode,omitempty"`
	MountPoint       *string                           `json:"mountPoint,omitempty"`
}

type Autoscaling struct {
	Enabled     bool                                           `json:"enabled,omitempty"`
	MinReplicas int                                            `json:"minReplicas,omitempty"`
	MaxReplicas int                                            `json:"maxReplicas,omitempty"`
	Behavior    *autoscalingv2.HorizontalPodAutoscalerBehavior `json:"behavior,omitempty"`
	Metrics     []autoscalingv2.MetricSpec                     `json:"metrics,omitempty"`
}
