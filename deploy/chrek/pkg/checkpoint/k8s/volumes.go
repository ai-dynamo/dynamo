// Package k8s provides Kubernetes-specific functionality for checkpoint operations.
// This includes volume type discovery via K8s API and containerd container discovery.
package k8s

import (
	"context"
	"fmt"
	"strings"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
)

// VolumeInfo contains Kubernetes volume information for a mount.
type VolumeInfo struct {
	VolumeName string // Name from pod.spec.volumes[].name
	VolumeType string // Type: emptyDir, configMap, secret, persistentVolumeClaim, etc.
	MountPath  string // Container path from volumeMounts[].mountPath
	SubPath    string // SubPath if specified
	ReadOnly   bool   // Whether mount is read-only

	// Type-specific details
	ConfigMapName string // For configMap volumes
	SecretName    string // For secret volumes
	PVCName       string // For persistentVolumeClaim volumes
}

// K8sClient wraps the Kubernetes clientset for volume discovery.
type K8sClient struct {
	clientset *kubernetes.Clientset
}

// NewK8sClient creates a new Kubernetes client.
// It attempts in-cluster config first, then falls back to kubeconfig.
func NewK8sClient() (*K8sClient, error) {
	config, err := rest.InClusterConfig()
	if err != nil {
		// Fall back to kubeconfig for local development
		config, err = clientcmd.BuildConfigFromFlags("", clientcmd.RecommendedHomeFile)
		if err != nil {
			return nil, fmt.Errorf("failed to create k8s config: %w", err)
		}
	}

	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create k8s clientset: %w", err)
	}

	return &K8sClient{clientset: clientset}, nil
}

// NewK8sClientWithConfig creates a client with explicit config.
func NewK8sClientWithConfig(config *rest.Config) (*K8sClient, error) {
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create k8s clientset: %w", err)
	}
	return &K8sClient{clientset: clientset}, nil
}

// GetPodVolumes returns volume information for all mounts in a container.
// Returns a map from mount path to VolumeInfo.
func (c *K8sClient) GetPodVolumes(ctx context.Context, namespace, podName, containerName string) (map[string]*VolumeInfo, error) {
	pod, err := c.clientset.CoreV1().Pods(namespace).Get(ctx, podName, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to get pod %s/%s: %w", namespace, podName, err)
	}

	return ExtractVolumeInfo(pod, containerName)
}

// ExtractVolumeInfo extracts volume information from a Pod spec.
// This is the core logic that maps volumeMounts to volumes and determines types.
func ExtractVolumeInfo(pod *corev1.Pod, containerName string) (map[string]*VolumeInfo, error) {
	// Build volume name -> type mapping from pod.spec.volumes
	volumeTypes := make(map[string]*volumeDetails)
	for _, vol := range pod.Spec.Volumes {
		volumeTypes[vol.Name] = getVolumeDetails(&vol)
	}

	// Find the target container
	var container *corev1.Container
	for i := range pod.Spec.Containers {
		if pod.Spec.Containers[i].Name == containerName {
			container = &pod.Spec.Containers[i]
			break
		}
	}
	if container == nil {
		// Try init containers
		for i := range pod.Spec.InitContainers {
			if pod.Spec.InitContainers[i].Name == containerName {
				container = &pod.Spec.InitContainers[i]
				break
			}
		}
	}
	if container == nil {
		return nil, fmt.Errorf("container %s not found in pod", containerName)
	}

	// Build mount path -> volume info mapping
	result := make(map[string]*VolumeInfo)
	for _, mount := range container.VolumeMounts {
		details, ok := volumeTypes[mount.Name]
		if !ok {
			continue // Mount references unknown volume
		}

		result[mount.MountPath] = &VolumeInfo{
			VolumeName:    mount.Name,
			VolumeType:    details.volumeType,
			MountPath:     mount.MountPath,
			SubPath:       mount.SubPath,
			ReadOnly:      mount.ReadOnly,
			ConfigMapName: details.configMapName,
			SecretName:    details.secretName,
			PVCName:       details.pvcName,
		}
	}

	return result, nil
}

// volumeDetails holds extracted volume type information.
type volumeDetails struct {
	volumeType    string
	configMapName string
	secretName    string
	pvcName       string
}

// getVolumeDetails extracts type and details from a Volume spec.
func getVolumeDetails(vol *corev1.Volume) *volumeDetails {
	d := &volumeDetails{volumeType: "unknown"}

	switch {
	case vol.EmptyDir != nil:
		d.volumeType = "emptyDir"
	case vol.ConfigMap != nil:
		d.volumeType = "configMap"
		d.configMapName = vol.ConfigMap.Name
	case vol.Secret != nil:
		d.volumeType = "secret"
		d.secretName = vol.Secret.SecretName
	case vol.PersistentVolumeClaim != nil:
		d.volumeType = "persistentVolumeClaim"
		d.pvcName = vol.PersistentVolumeClaim.ClaimName
	case vol.HostPath != nil:
		d.volumeType = "hostPath"
	case vol.Projected != nil:
		d.volumeType = "projected"
	case vol.DownwardAPI != nil:
		d.volumeType = "downwardAPI"
	case vol.CSI != nil:
		d.volumeType = "csi"
	case vol.NFS != nil:
		d.volumeType = "nfs"
	case vol.ISCSI != nil:
		d.volumeType = "iscsi"
	case vol.GCEPersistentDisk != nil:
		d.volumeType = "gcePersistentDisk"
	case vol.AWSElasticBlockStore != nil:
		d.volumeType = "awsElasticBlockStore"
	case vol.AzureDisk != nil:
		d.volumeType = "azureDisk"
	case vol.AzureFile != nil:
		d.volumeType = "azureFile"
	case vol.CephFS != nil:
		d.volumeType = "cephfs"
	case vol.Cinder != nil:
		d.volumeType = "cinder"
	case vol.FC != nil:
		d.volumeType = "fc"
	case vol.FlexVolume != nil:
		d.volumeType = "flexVolume"
	case vol.Flocker != nil:
		d.volumeType = "flocker"
	case vol.GitRepo != nil:
		d.volumeType = "gitRepo"
	case vol.Glusterfs != nil:
		d.volumeType = "glusterfs"
	case vol.PhotonPersistentDisk != nil:
		d.volumeType = "photonPersistentDisk"
	case vol.PortworxVolume != nil:
		d.volumeType = "portworxVolume"
	case vol.Quobyte != nil:
		d.volumeType = "quobyte"
	case vol.RBD != nil:
		d.volumeType = "rbd"
	case vol.ScaleIO != nil:
		d.volumeType = "scaleIO"
	case vol.StorageOS != nil:
		d.volumeType = "storageos"
	case vol.VsphereVolume != nil:
		d.volumeType = "vsphereVolume"
	case vol.Ephemeral != nil:
		d.volumeType = "ephemeral"
	}

	return d
}

// DetectVolumeTypeFromPath attempts to identify volume type from kubelet path patterns.
// This is a best-effort fallback; accurate volume types require K8s API access via GetPodVolumes.
func DetectVolumeTypeFromPath(hostPath string) (volumeType, volumeName string) {
	volumeType = "unknown"
	volumeName = ""

	// Map of path patterns to volume types
	patterns := map[string]string{
		"/kubernetes.io~empty-dir/":             "emptyDir",
		"/kubernetes.io~configmap/":             "configMap",
		"/kubernetes.io~secret/":                "secret",
		"/kubernetes.io~projected/":             "projected",
		"/kubernetes.io~downward-api/":          "downwardAPI",
		"/kubernetes.io~persistentvolumeclaim/": "persistentVolumeClaim",
		"/kubernetes.io~hostpath/":              "hostPath",
	}

	for pattern, vType := range patterns {
		if strings.Contains(hostPath, pattern) {
			volumeType = vType
			// Extract volume name from path
			parts := strings.Split(hostPath, pattern)
			if len(parts) > 1 {
				volumeName = strings.Split(parts[1], "/")[0]
			}
			break
		}
	}

	return volumeType, volumeName
}
