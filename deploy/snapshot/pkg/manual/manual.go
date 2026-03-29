package manual

import (
	"context"
	"errors"
	"fmt"
	"os"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"

	"gopkg.in/yaml.v3"
)

const (
	manualDiscoveryBackend               = "kubernetes"
	checkpointStatusAnnotation           = "nvidia.com/snapshot-checkpoint-status"
	restoreStatusAnnotation              = "nvidia.com/snapshot-restore-status"
	checkpointLocationAnnotation         = "nvidia.com/snapshot-checkpoint-location"
	checkpointStorageTypeAnnotation      = "nvidia.com/snapshot-checkpoint-storage-type"
	checkpointSourceLabel                = "nvidia.com/snapshot-is-checkpoint-source"
	restoreTargetLabel                   = "nvidia.com/snapshot-is-restore-target"
	checkpointHashLabel                  = "nvidia.com/snapshot-checkpoint-hash"
	defaultSeccompLocalhostProfile       = "profiles/block-iouring.json"
	defaultEventPlane                    = "zmq"
	defaultRequestPlane                  = "tcp"
	defaultImagePullPolicy               = "IfNotPresent"
	defaultSharedMemory                  = "8Gi"
	defaultGeneratedCheckpointHashPrefix = "manual-snapshot"
)

var dnsLabelPattern = regexp.MustCompile(`[^a-z0-9-]+`)

type CheckpointOptions struct {
	ManifestPath   string
	Namespace      string
	CheckpointHash string
	Timeout        time.Duration
}

type RestoreOptions struct {
	ManifestPath   string
	Namespace      string
	CheckpointHash string
	Timeout        time.Duration
}

type Result struct {
	Name               string
	Namespace          string
	Backend            string
	CheckpointHash     string
	CheckpointLocation string
	CheckpointJob      string
	RestorePod         string
	RestoreService     string
	Status             string
}

type selectedWorker struct {
	WorkloadName       string
	ManifestNamespace  string
	DGDName            string
	Backend            string
	Labels             map[string]string
	Annotations        map[string]string
	PodSpec            corev1.PodSpec
	MainContainerIndex int
	GPUCount           int
}

type runSpec struct {
	Namespace               string
	DGDName                 string
	WorkloadName            string
	ResourcePrefix          string
	CheckpointHash          string
	SeccompLocalhostProfile string
	CudaCheckpointLaunchJob bool
	Worker                  selectedWorker
	SnapshotStorage         snapshotStorage
	ServiceAccountName      string
	RoleName                string
	RoleBindingName         string
	CheckpointJobName       string
	CheckpointPodLabel      string
	RestorePodName          string
}

type snapshotStorage struct {
	PVCName  string
	BasePath string
}

func RunCheckpoint(ctx context.Context, opts CheckpointOptions) (*Result, error) {
	if err := validateCheckpointOptions(opts); err != nil {
		return nil, err
	}

	worker, err := loadWorker(opts.ManifestPath)
	if err != nil {
		return nil, err
	}

	clientset, defaultNamespace, err := loadClientset()
	if err != nil {
		return nil, err
	}

	namespace := resolveNamespace(opts.Namespace, worker.ManifestNamespace, defaultNamespace)
	storage, err := discoverSnapshotStorage(ctx, clientset, namespace)
	if err != nil {
		return nil, err
	}
	spec := buildRunSpec(namespace, opts.CheckpointHash, worker, storage)

	if err := ensureSupportResources(ctx, clientset, spec); err != nil {
		return nil, err
	}
	if err := createCheckpointJob(ctx, clientset, spec); err != nil {
		return nil, err
	}

	waitCtx, cancel := context.WithTimeout(ctx, opts.Timeout)
	defer cancel()
	status, err := waitForCheckpoint(waitCtx, clientset, spec)
	if err != nil {
		return nil, err
	}

	return &Result{
		Name:               spec.WorkloadName,
		Namespace:          spec.Namespace,
		Backend:            spec.Worker.Backend,
		CheckpointHash:     spec.CheckpointHash,
		CheckpointLocation: spec.checkpointLocation(),
		CheckpointJob:      spec.CheckpointJobName,
		Status:             status,
	}, nil
}

func RunRestore(ctx context.Context, opts RestoreOptions) (*Result, error) {
	if err := validateRestoreOptions(opts); err != nil {
		return nil, err
	}

	worker, err := loadWorker(opts.ManifestPath)
	if err != nil {
		return nil, err
	}

	clientset, defaultNamespace, err := loadClientset()
	if err != nil {
		return nil, err
	}

	namespace := resolveNamespace(opts.Namespace, worker.ManifestNamespace, defaultNamespace)
	storage, err := discoverSnapshotStorage(ctx, clientset, namespace)
	if err != nil {
		return nil, err
	}
	spec := buildRunSpec(namespace, opts.CheckpointHash, worker, storage)

	if err := ensureSupportResources(ctx, clientset, spec); err != nil {
		return nil, err
	}
	if err := createRestoreService(ctx, clientset, spec); err != nil {
		return nil, err
	}
	if err := createRestorePod(ctx, clientset, spec); err != nil {
		return nil, err
	}

	waitCtx, cancel := context.WithTimeout(ctx, opts.Timeout)
	defer cancel()
	status, err := waitForRestore(waitCtx, clientset, spec)
	if err != nil {
		return nil, err
	}

	return &Result{
		Name:               spec.WorkloadName,
		Namespace:          spec.Namespace,
		Backend:            spec.Worker.Backend,
		CheckpointHash:     spec.CheckpointHash,
		CheckpointLocation: spec.checkpointLocation(),
		RestorePod:         spec.RestorePodName,
		RestoreService:     spec.RestorePodName,
		Status:             status,
	}, nil
}

func validateCheckpointOptions(opts CheckpointOptions) error {
	var missing []string
	if strings.TrimSpace(opts.ManifestPath) == "" {
		missing = append(missing, "--manifest")
	}
	if len(missing) > 0 {
		return fmt.Errorf("missing required flags: %s", strings.Join(missing, ", "))
	}
	if opts.Timeout <= 0 {
		return fmt.Errorf("--timeout must be greater than zero")
	}
	return nil
}

func validateRestoreOptions(opts RestoreOptions) error {
	var missing []string
	if strings.TrimSpace(opts.ManifestPath) == "" {
		missing = append(missing, "--manifest")
	}
	if strings.TrimSpace(opts.CheckpointHash) == "" {
		missing = append(missing, "--checkpoint-hash")
	}
	if len(missing) > 0 {
		return fmt.Errorf("missing required flags: %s", strings.Join(missing, ", "))
	}
	if opts.Timeout <= 0 {
		return fmt.Errorf("--timeout must be greater than zero")
	}
	return nil
}

func loadClientset() (kubernetes.Interface, string, error) {
	loadingRules := clientcmd.NewDefaultClientConfigLoadingRules()
	clientConfig := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(loadingRules, &clientcmd.ConfigOverrides{})
	restConfig, err := clientConfig.ClientConfig()
	if err != nil {
		return nil, "", fmt.Errorf("load kubeconfig: %w", err)
	}

	namespace, _, err := clientConfig.Namespace()
	if err != nil {
		return nil, "", fmt.Errorf("resolve current namespace: %w", err)
	}
	if strings.TrimSpace(namespace) == "" {
		namespace = corev1.NamespaceDefault
	}

	clientset, err := kubernetes.NewForConfig(restConfig)
	if err != nil {
		return nil, "", fmt.Errorf("create kubernetes client: %w", err)
	}
	return clientset, namespace, nil
}

func discoverSnapshotStorage(ctx context.Context, clientset kubernetes.Interface, namespace string) (snapshotStorage, error) {
	daemonSets, err := clientset.AppsV1().DaemonSets(namespace).List(ctx, metav1.ListOptions{
		LabelSelector: "app.kubernetes.io/name=snapshot",
	})
	if err != nil {
		return snapshotStorage{}, fmt.Errorf("list snapshot-agent daemonsets in %s: %w", namespace, err)
	}
	if len(daemonSets.Items) == 0 {
		return snapshotStorage{}, fmt.Errorf("no snapshot-agent daemonset found in namespace %s", namespace)
	}

	slices.SortFunc(daemonSets.Items, func(a, b appsv1.DaemonSet) int {
		return strings.Compare(a.Name, b.Name)
	})

	for _, daemonSet := range daemonSets.Items {
		storage, ok := snapshotStorageFromDaemonSet(&daemonSet)
		if ok {
			return storage, nil
		}
	}

	names := make([]string, 0, len(daemonSets.Items))
	for _, daemonSet := range daemonSets.Items {
		names = append(names, daemonSet.Name)
	}
	return snapshotStorage{}, fmt.Errorf("snapshot-agent daemonset in %s does not mount a PVC-backed checkpoint volume (%s)", namespace, strings.Join(names, ", "))
}

func snapshotStorageFromDaemonSet(daemonSet *appsv1.DaemonSet) (snapshotStorage, bool) {
	if daemonSet == nil {
		return snapshotStorage{}, false
	}

	mountPaths := map[string]string{}
	for _, container := range daemonSet.Spec.Template.Spec.Containers {
		if container.Name != "agent" {
			continue
		}
		for _, mount := range container.VolumeMounts {
			if strings.TrimSpace(mount.MountPath) == "" {
				continue
			}
			mountPaths[mount.Name] = mount.MountPath
		}
	}

	for _, volume := range daemonSet.Spec.Template.Spec.Volumes {
		if volume.PersistentVolumeClaim == nil {
			continue
		}
		basePath, ok := mountPaths[volume.Name]
		if !ok || strings.TrimSpace(basePath) == "" {
			continue
		}
		claimName := strings.TrimSpace(volume.PersistentVolumeClaim.ClaimName)
		if claimName == "" {
			continue
		}
		return snapshotStorage{
			PVCName:  claimName,
			BasePath: strings.TrimRight(basePath, "/"),
		}, true
	}

	return snapshotStorage{}, false
}

func loadWorker(manifestPath string) (selectedWorker, error) {
	content, err := os.ReadFile(manifestPath)
	if err != nil {
		return selectedWorker{}, fmt.Errorf("read manifest %s: %w", manifestPath, err)
	}

	var pod corev1.Pod
	if err := yaml.Unmarshal(content, &pod); err != nil {
		return selectedWorker{}, fmt.Errorf("parse manifest %s: %w", manifestPath, err)
	}
	if kind := strings.TrimSpace(pod.Kind); kind != "" && kind != "Pod" {
		return selectedWorker{}, fmt.Errorf("manifest %s is kind %q, expected Pod", manifestPath, kind)
	}
	if len(pod.Spec.Containers) == 0 {
		return selectedWorker{}, fmt.Errorf("manifest %s has no containers", manifestPath)
	}
	if len(pod.Spec.Containers) > 1 {
		return selectedWorker{}, fmt.Errorf("manifest %s has %d containers; snapshotctl requires exactly one worker container", manifestPath, len(pod.Spec.Containers))
	}
	container := pod.Spec.Containers[0]
	if strings.TrimSpace(container.Image) == "" {
		return selectedWorker{}, fmt.Errorf("manifest %s: worker container image is required", manifestPath)
	}
	backend, err := inferBackend(container)
	if err != nil {
		return selectedWorker{}, fmt.Errorf("manifest %s: %w", manifestPath, err)
	}
	gpuCount, err := parseGPUCount(container.Resources)
	if err != nil {
		return selectedWorker{}, fmt.Errorf("manifest %s: %w", manifestPath, err)
	}

	return selectedWorker{
		WorkloadName:       sanitizeName(firstNonEmpty(pod.Name, graphNameForPod(pod), "snapshot-worker")),
		ManifestNamespace:  strings.TrimSpace(pod.Namespace),
		DGDName:            sanitizeName(firstNonEmpty(graphNameForPod(pod), pod.Name, "snapshot-worker")),
		Backend:            backend,
		Labels:             cloneMap(pod.Labels),
		Annotations:        cloneMap(pod.Annotations),
		PodSpec:            *pod.Spec.DeepCopy(),
		MainContainerIndex: 0,
		GPUCount:           gpuCount,
	}, nil
}

func graphNameForPod(pod corev1.Pod) string {
	return strings.TrimSpace(firstNonEmpty(
		pod.Annotations["nvidia.com/dyn-parent-dgd-name"],
		pod.Labels["nvidia.com/dynamo-graph-deployment-name"],
	))
}

func inferBackend(container corev1.Container) (string, error) {
	candidate := strings.ToLower(strings.Join(container.Command, " ") +
		" " + strings.Join(container.Args, " ") +
		" " + container.WorkingDir +
		" " + container.Image)

	switch {
	case strings.Contains(candidate, "trtllm"):
		return "", errors.New("trtllm manifests are not supported by snapshotctl")
	case strings.Contains(candidate, "dynamo.vllm"),
		strings.Contains(candidate, "/vllm"),
		strings.Contains(candidate, "vllm-runtime"):
		return "vllm", nil
	case strings.Contains(candidate, "dynamo.sglang"),
		strings.Contains(candidate, "/sglang"),
		strings.Contains(candidate, "sglang-runtime"):
		return "sglang", nil
	default:
		return "", errors.New("unable to infer backend from worker command/image")
	}
}

func parseGPUCount(resources corev1.ResourceRequirements) (int, error) {
	quantity, ok := resources.Limits[corev1.ResourceName("nvidia.com/gpu")]
	if !ok {
		return 0, errors.New("worker resources.limits[\"nvidia.com/gpu\"] is required")
	}
	value := quantity.Value()
	if value <= 0 {
		return 0, fmt.Errorf("worker resources.limits[\"nvidia.com/gpu\"] must be positive, got %q", quantity.String())
	}
	return int(value), nil
}

func resolveNamespace(explicitNamespace string, manifestNamespace string, currentNamespace string) string {
	if namespace := strings.TrimSpace(explicitNamespace); namespace != "" {
		return namespace
	}
	if namespace := strings.TrimSpace(manifestNamespace); namespace != "" {
		return namespace
	}
	if namespace := strings.TrimSpace(currentNamespace); namespace != "" {
		return namespace
	}
	return corev1.NamespaceDefault
}

func cloneMap(values map[string]string) map[string]string {
	if len(values) == 0 {
		return nil
	}
	cloned := make(map[string]string, len(values))
	for key, value := range values {
		cloned[key] = value
	}
	return cloned
}

func shouldUseCudaCheckpointLaunchJob(args []string) bool {
	for index := range args {
		var rawValue string
		switch {
		case args[index] == "--tensor-parallel-size", args[index] == "--tp":
			if index+1 < len(args) {
				rawValue = args[index+1]
			}
		case strings.HasPrefix(args[index], "--tensor-parallel-size="):
			rawValue = strings.TrimPrefix(args[index], "--tensor-parallel-size=")
		case strings.HasPrefix(args[index], "--tp="):
			rawValue = strings.TrimPrefix(args[index], "--tp=")
		}
		if rawValue == "" {
			continue
		}
		value, err := strconv.Atoi(strings.TrimSpace(rawValue))
		if err == nil && value > 1 {
			return true
		}
	}
	return false
}

func shortNameSuffix(value string) string {
	sanitized := sanitizeName(value)
	if len(sanitized) <= 12 {
		return sanitized
	}
	return strings.Trim(sanitized[len(sanitized)-12:], "-")
}

func buildRunSpec(namespace string, checkpointHash string, worker selectedWorker, storage snapshotStorage) runSpec {
	if strings.TrimSpace(checkpointHash) == "" {
		checkpointHash = fmt.Sprintf("%s-%d", defaultGeneratedCheckpointHashPrefix, time.Now().UTC().UnixNano())
	}
	resourcePrefix := sanitizeName(worker.WorkloadName + "-" + shortNameSuffix(checkpointHash))
	container := worker.mainContainer()

	return runSpec{
		Namespace:               namespace,
		DGDName:                 worker.DGDName,
		WorkloadName:            worker.WorkloadName,
		ResourcePrefix:          resourcePrefix,
		CheckpointHash:          checkpointHash,
		SeccompLocalhostProfile: defaultSeccompLocalhostProfile,
		CudaCheckpointLaunchJob: shouldUseCudaCheckpointLaunchJob(append(append([]string(nil), container.Command...), container.Args...)),
		Worker:                  worker,
		SnapshotStorage:         storage,
		ServiceAccountName:      sanitizeName(resourcePrefix + "-snapshot-sa"),
		RoleName:                sanitizeName(resourcePrefix + "-snapshot-role"),
		RoleBindingName:         sanitizeName(resourcePrefix + "-snapshot-binding"),
		CheckpointJobName:       sanitizeName(resourcePrefix + "-checkpoint"),
		CheckpointPodLabel:      sanitizeName(resourcePrefix + "-checkpoint"),
		RestorePodName:          sanitizeName(resourcePrefix + "-restore"),
	}
}

func (s runSpec) dynNamespace() string {
	return fmt.Sprintf("%s-%s", s.Namespace, s.DGDName)
}

func (s runSpec) checkpointLocation() string {
	return strings.TrimRight(s.SnapshotStorage.BasePath, "/") + "/" + s.CheckpointHash
}

func (s runSpec) tritonCacheDir() string {
	return "/home/dynamo/.cache/huggingface/dynamo/triton-cache/" + s.CheckpointHash
}

func (s runSpec) sharedAnnotations() map[string]string {
	return mergeStringMaps(s.Worker.Annotations, map[string]string{
		"nvidia.com/dyn-namespace":            s.dynNamespace(),
		"nvidia.com/dyn-component":            "worker",
		"nvidia.com/dyn-parent-dgd-name":      s.DGDName,
		"nvidia.com/dyn-parent-dgd-namespace": s.Namespace,
		"nvidia.com/dyn-discovery-backend":    manualDiscoveryBackend,
		checkpointLocationAnnotation:          s.checkpointLocation(),
		checkpointStorageTypeAnnotation:       "pvc",
	})
}

func (s runSpec) sharedEnvVars() []corev1.EnvVar {
	env := []corev1.EnvVar{
		{Name: "DYN_COMPONENT", Value: "worker"},
		{Name: "DYN_DISCOVERY_BACKEND", Value: manualDiscoveryBackend},
		{Name: "DYN_NAMESPACE", Value: s.dynNamespace()},
		{Name: "DYN_PARENT_DGD_K8S_NAME", Value: s.DGDName},
		{Name: "DYN_PARENT_DGD_K8S_NAMESPACE", Value: s.Namespace},
		{Name: "DYN_EVENT_PLANE", Value: defaultEventPlane},
		{Name: "DYN_REQUEST_PLANE", Value: defaultRequestPlane},
		{Name: "DYN_HEALTH_CHECK_ENABLED", Value: "false"},
		{Name: "DYN_SYSTEM_ENABLED", Value: "true"},
		{Name: "DYN_SYSTEM_PORT", Value: "9090"},
		{Name: "DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS", Value: `["generate"]`},
		{Name: "POD_NAME", ValueFrom: &corev1.EnvVarSource{
			FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.name"},
		}},
		{Name: "POD_NAMESPACE", ValueFrom: &corev1.EnvVarSource{
			FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.namespace"},
		}},
		{Name: "POD_UID", ValueFrom: &corev1.EnvVarSource{
			FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.uid"},
		}},
		{Name: "DYN_CHECKPOINT_HASH", Value: s.CheckpointHash},
		{Name: "DYN_CHECKPOINT_PATH", Value: s.SnapshotStorage.BasePath},
		{Name: "DYN_CHECKPOINT_LOCATION", Value: s.checkpointLocation()},
		{Name: "DYN_CHECKPOINT_STORAGE_TYPE", Value: "pvc"},
		{Name: "HF_HOME", Value: "/home/dynamo/.cache/huggingface"},
		{Name: "GLOO_SOCKET_IFNAME", Value: "lo"},
		{Name: "NCCL_SOCKET_IFNAME", Value: "lo"},
		{Name: "NCCL_DEBUG", Value: "ERROR"},
		{Name: "TORCH_CPP_LOG_LEVEL", Value: "ERROR"},
		{Name: "TORCH_DISTRIBUTED_DEBUG", Value: "OFF"},
		{Name: "CUDA_ERROR_LEVEL", Value: "10"},
	}

	if s.CudaCheckpointLaunchJob {
		env = append(env,
			corev1.EnvVar{Name: "NCCL_CUMEM_ENABLE", Value: "0"},
			corev1.EnvVar{Name: "NCCL_CUMEM_HOST_ENABLE", Value: "0"},
			corev1.EnvVar{Name: "NCCL_NVLS_ENABLE", Value: "0"},
			corev1.EnvVar{Name: "NCCL_P2P_DISABLE", Value: "0"},
			corev1.EnvVar{Name: "NCCL_SHM_DISABLE", Value: "1"},
			corev1.EnvVar{Name: "NCCL_IB_DISABLE", Value: "1"},
			corev1.EnvVar{Name: "TORCH_NCCL_ENABLE_MONITORING", Value: "0"},
		)
	}
	if s.Worker.Backend == "sglang" {
		env = append(env, corev1.EnvVar{Name: "TRITON_CACHE_DIR", Value: s.tritonCacheDir()})
	}

	return env
}

func (s runSpec) checkpointEnvVars() []corev1.EnvVar {
	shared := s.sharedEnvVars()
	withReadyFiles := make([]corev1.EnvVar, 0, len(shared)+2)
	for _, envVar := range shared {
		if envVar.Name == "DYN_CHECKPOINT_HASH" {
			withReadyFiles = append(withReadyFiles,
				corev1.EnvVar{Name: "DYN_CHECKPOINT_READY_FILE", Value: "/tmp/checkpoint-ready"},
				corev1.EnvVar{Name: "DYN_READY_FOR_CHECKPOINT_FILE", Value: "/tmp/ready-for-checkpoint"},
			)
		}
		withReadyFiles = append(withReadyFiles, envVar)
	}
	return withReadyFiles
}

func (s runSpec) sharedVolumes() []corev1.Volume {
	return []corev1.Volume{
		{
			Name: "checkpoint-storage",
			VolumeSource: corev1.VolumeSource{
				PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
					ClaimName: s.SnapshotStorage.PVCName,
				},
			},
		},
		{
			Name: "shared-memory",
			VolumeSource: corev1.VolumeSource{
				EmptyDir: &corev1.EmptyDirVolumeSource{
					Medium:    corev1.StorageMediumMemory,
					SizeLimit: quantityPtr(resource.MustParse(defaultSharedMemory)),
				},
			},
		},
		{
			Name: "podinfo",
			VolumeSource: corev1.VolumeSource{
				DownwardAPI: &corev1.DownwardAPIVolumeSource{
					Items: []corev1.DownwardAPIVolumeFile{
						{Path: "pod_name", FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.name"}},
						{Path: "pod_uid", FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.uid"}},
						{Path: "pod_namespace", FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.namespace"}},
						{Path: "dyn_namespace", FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.annotations['nvidia.com/dyn-namespace']"}},
						{Path: "dyn_component", FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.annotations['nvidia.com/dyn-component']"}},
						{Path: "dyn_parent_dgd_k8s_name", FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.annotations['nvidia.com/dyn-parent-dgd-name']"}},
						{Path: "dyn_parent_dgd_k8s_namespace", FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.annotations['nvidia.com/dyn-parent-dgd-namespace']"}},
						{Path: "dyn_discovery_backend", FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.annotations['nvidia.com/dyn-discovery-backend']"}},
					},
				},
			},
		},
	}
}

func (s runSpec) sharedVolumeMounts() []corev1.VolumeMount {
	return []corev1.VolumeMount{
		{Name: "checkpoint-storage", MountPath: s.SnapshotStorage.BasePath},
		{Name: "shared-memory", MountPath: "/dev/shm"},
		{Name: "podinfo", MountPath: "/etc/podinfo", ReadOnly: true},
	}
}

func (s runSpec) podSecurityContext() *corev1.PodSecurityContext {
	return &corev1.PodSecurityContext{
		SeccompProfile: &corev1.SeccompProfile{
			Type:             corev1.SeccompProfileTypeLocalhost,
			LocalhostProfile: stringPtr(s.SeccompLocalhostProfile),
		},
	}
}

func (s runSpec) podSpecBase() corev1.PodSpec {
	spec := *s.Worker.PodSpec.DeepCopy()
	spec.RestartPolicy = corev1.RestartPolicyNever
	spec.ServiceAccountName = s.ServiceAccountName
	spec.SecurityContext = s.podSecurityContext()
	return spec
}

func (s runSpec) checkpointCommandAndArgs() ([]string, []string) {
	container := s.Worker.mainContainer()
	command := append([]string(nil), container.Command...)
	args := append([]string(nil), container.Args...)
	if !containsFlag(args, "--event-plane") {
		args = append(args, "--event-plane", defaultEventPlane)
	}
	if !containsFlag(args, "--request-plane") {
		args = append(args, "--request-plane", defaultRequestPlane)
	}
	if !s.CudaCheckpointLaunchJob {
		return command, args
	}
	wrappedArgs := make([]string, 0, len(command)+len(args)+1)
	wrappedArgs = append(wrappedArgs, "--launch-job")
	wrappedArgs = append(wrappedArgs, command...)
	wrappedArgs = append(wrappedArgs, args...)
	return []string{"cuda-checkpoint"}, wrappedArgs
}

func buildServiceAccount(spec runSpec) *corev1.ServiceAccount {
	return &corev1.ServiceAccount{
		TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "ServiceAccount"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      spec.ServiceAccountName,
			Namespace: spec.Namespace,
		},
		ImagePullSecrets: append([]corev1.LocalObjectReference(nil), spec.Worker.PodSpec.ImagePullSecrets...),
	}
}

func buildRole(spec runSpec) *rbacv1.Role {
	return &rbacv1.Role{
		TypeMeta: metav1.TypeMeta{APIVersion: "rbac.authorization.k8s.io/v1", Kind: "Role"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      spec.RoleName,
			Namespace: spec.Namespace,
		},
		Rules: []rbacv1.PolicyRule{
			{
				APIGroups: []string{"discovery.k8s.io"},
				Resources: []string{"endpointslices"},
				Verbs:     []string{"get", "list", "watch"},
			},
			{
				APIGroups: []string{"nvidia.com"},
				Resources: []string{"dynamoworkermetadatas"},
				Verbs:     []string{"get", "list", "watch", "create", "update", "patch", "delete"},
			},
		},
	}
}

func buildRoleBinding(spec runSpec) *rbacv1.RoleBinding {
	return &rbacv1.RoleBinding{
		TypeMeta: metav1.TypeMeta{APIVersion: "rbac.authorization.k8s.io/v1", Kind: "RoleBinding"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      spec.RoleBindingName,
			Namespace: spec.Namespace,
		},
		Subjects: []rbacv1.Subject{
			{
				Kind:      "ServiceAccount",
				Name:      spec.ServiceAccountName,
				Namespace: spec.Namespace,
			},
		},
		RoleRef: rbacv1.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "Role",
			Name:     spec.RoleName,
		},
	}
}

func buildCheckpointJob(spec runSpec) *batchv1.Job {
	command, args := spec.checkpointCommandAndArgs()
	podSpec := spec.podSpecBase()
	container := spec.Worker.mainContainer()
	container.Command = command
	container.Args = args
	container.Env = mergeEnv(container.Env, spec.checkpointEnvVars())
	container.ReadinessProbe = &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			Exec: &corev1.ExecAction{
				Command: []string{"sh", "-c", "cat /tmp/checkpoint-ready 2>/dev/null || cat /tmp/ready-for-checkpoint"},
			},
		},
		InitialDelaySeconds: 15,
		PeriodSeconds:       2,
	}
	container.VolumeMounts = mergeVolumeMounts(container.VolumeMounts, spec.sharedVolumeMounts())
	podSpec.Containers = []corev1.Container{container}
	podSpec.Volumes = mergeVolumes(podSpec.Volumes, spec.sharedVolumes())

	return &batchv1.Job{
		TypeMeta: metav1.TypeMeta{APIVersion: "batch/v1", Kind: "Job"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      spec.CheckpointJobName,
			Namespace: spec.Namespace,
		},
		Spec: batchv1.JobSpec{
			BackoffLimit: int32Ptr(0),
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: spec.sharedAnnotations(),
					Labels: mergeStringMaps(spec.Worker.Labels, map[string]string{
						"app":                         spec.CheckpointPodLabel,
						"nvidia.com/dynamo-component": "worker",
						"nvidia.com/dynamo-graph-deployment-name": spec.DGDName,
						checkpointSourceLabel:                     "true",
						checkpointHashLabel:                       spec.CheckpointHash,
					}),
				},
				Spec: podSpec,
			},
		},
	}
}

func buildRestorePod(spec runSpec) *corev1.Pod {
	tunType := corev1.HostPathCharDev
	volumes := mergeVolumes(spec.sharedVolumes(), []corev1.Volume{{
		Name: "host-dev-net-tun",
		VolumeSource: corev1.VolumeSource{
			HostPath: &corev1.HostPathVolumeSource{
				Path: "/dev/net/tun",
				Type: &tunType,
			},
		},
	}})
	volumeMounts := mergeVolumeMounts(spec.sharedVolumeMounts(), []corev1.VolumeMount{{
		Name:      "host-dev-net-tun",
		MountPath: "/dev/net/tun",
	}})

	podSpec := spec.podSpecBase()
	container := spec.Worker.mainContainer()
	container.Command = []string{"sleep", "infinity"}
	container.Args = nil
	container.Env = mergeEnv(container.Env, spec.sharedEnvVars())
	container.Ports = []corev1.ContainerPort{{Name: "system", ContainerPort: 9090}}
	container.ReadinessProbe = &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			HTTPGet: &corev1.HTTPGetAction{
				Path: "/live",
				Port: intstr.FromString("system"),
			},
		},
		PeriodSeconds:    1,
		TimeoutSeconds:   4,
		FailureThreshold: 3,
	}
	container.LivenessProbe = &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			HTTPGet: &corev1.HTTPGetAction{
				Path: "/live",
				Port: intstr.FromString("system"),
			},
		},
		PeriodSeconds:    5,
		TimeoutSeconds:   4,
		FailureThreshold: 1,
	}
	container.StartupProbe = &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			HTTPGet: &corev1.HTTPGetAction{
				Path: "/live",
				Port: intstr.FromString("system"),
			},
		},
		PeriodSeconds:    1,
		TimeoutSeconds:   5,
		FailureThreshold: 720,
	}
	container.VolumeMounts = mergeVolumeMounts(container.VolumeMounts, volumeMounts)
	podSpec.Containers = []corev1.Container{container}
	podSpec.Volumes = mergeVolumes(podSpec.Volumes, volumes)

	return &corev1.Pod{
		TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "Pod"},
		ObjectMeta: metav1.ObjectMeta{
			Name:        spec.RestorePodName,
			Namespace:   spec.Namespace,
			Annotations: spec.sharedAnnotations(),
			Labels: mergeStringMaps(spec.Worker.Labels, map[string]string{
				"app":                         spec.RestorePodName,
				"nvidia.com/dynamo-component": "worker",
				"nvidia.com/dynamo-graph-deployment-name": spec.DGDName,
				"nvidia.com/checkpoint-restore":           "true",
				restoreTargetLabel:                        "true",
				checkpointHashLabel:                       spec.CheckpointHash,
			}),
		},
		Spec: podSpec,
	}
}

func buildRestoreService(spec runSpec) *corev1.Service {
	return &corev1.Service{
		TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "Service"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      spec.RestorePodName,
			Namespace: spec.Namespace,
			Labels: map[string]string{
				"nvidia.com/dynamo-component":             "worker",
				"nvidia.com/dynamo-discovery-backend":     manualDiscoveryBackend,
				"nvidia.com/dynamo-discovery-enabled":     "true",
				"nvidia.com/dynamo-graph-deployment-name": spec.DGDName,
				"nvidia.com/dynamo-namespace":             spec.dynNamespace(),
			},
		},
		Spec: corev1.ServiceSpec{
			Type: corev1.ServiceTypeClusterIP,
			Selector: map[string]string{
				"app": spec.RestorePodName,
			},
			Ports: []corev1.ServicePort{
				{
					Name:       "system",
					Port:       9090,
					Protocol:   corev1.ProtocolTCP,
					TargetPort: intstr.FromString("system"),
				},
			},
		},
	}
}

func ensureSupportResources(ctx context.Context, clientset kubernetes.Interface, spec runSpec) error {
	if err := ensureServiceAccount(ctx, clientset, buildServiceAccount(spec)); err != nil {
		return err
	}
	if err := ensureRole(ctx, clientset, buildRole(spec)); err != nil {
		return err
	}
	if err := ensureRoleBinding(ctx, clientset, buildRoleBinding(spec)); err != nil {
		return err
	}
	return nil
}

func ensureServiceAccount(ctx context.Context, clientset kubernetes.Interface, serviceAccount *corev1.ServiceAccount) error {
	if len(serviceAccount.ImagePullSecrets) == 0 {
		defaultServiceAccount, err := clientset.CoreV1().ServiceAccounts(serviceAccount.Namespace).Get(ctx, "default", metav1.GetOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			return fmt.Errorf("get default serviceaccount %s/default: %w", serviceAccount.Namespace, err)
		}
		if err == nil {
			serviceAccount = serviceAccount.DeepCopy()
			serviceAccount.ImagePullSecrets = append([]corev1.LocalObjectReference(nil), defaultServiceAccount.ImagePullSecrets...)
		}
	}
	_, err := clientset.CoreV1().ServiceAccounts(serviceAccount.Namespace).Create(ctx, serviceAccount, metav1.CreateOptions{})
	if err == nil || apierrors.IsAlreadyExists(err) {
		return nil
	}
	return fmt.Errorf("create serviceaccount %s/%s: %w", serviceAccount.Namespace, serviceAccount.Name, err)
}

func ensureRole(ctx context.Context, clientset kubernetes.Interface, role *rbacv1.Role) error {
	_, err := clientset.RbacV1().Roles(role.Namespace).Create(ctx, role, metav1.CreateOptions{})
	if err == nil || apierrors.IsAlreadyExists(err) {
		return nil
	}
	return fmt.Errorf("create role %s/%s: %w", role.Namespace, role.Name, err)
}

func ensureRoleBinding(ctx context.Context, clientset kubernetes.Interface, roleBinding *rbacv1.RoleBinding) error {
	_, err := clientset.RbacV1().RoleBindings(roleBinding.Namespace).Create(ctx, roleBinding, metav1.CreateOptions{})
	if err == nil || apierrors.IsAlreadyExists(err) {
		return nil
	}
	return fmt.Errorf("create rolebinding %s/%s: %w", roleBinding.Namespace, roleBinding.Name, err)
}

func createCheckpointJob(ctx context.Context, clientset kubernetes.Interface, spec runSpec) error {
	job := buildCheckpointJob(spec)
	_, err := clientset.BatchV1().Jobs(spec.Namespace).Create(ctx, job, metav1.CreateOptions{})
	if err == nil {
		return nil
	}
	if apierrors.IsAlreadyExists(err) {
		return fmt.Errorf("checkpoint job %s/%s already exists", spec.Namespace, spec.CheckpointJobName)
	}
	return fmt.Errorf("create checkpoint job %s/%s: %w", spec.Namespace, spec.CheckpointJobName, err)
}

func createRestoreService(ctx context.Context, clientset kubernetes.Interface, spec runSpec) error {
	service := buildRestoreService(spec)
	_, err := clientset.CoreV1().Services(spec.Namespace).Create(ctx, service, metav1.CreateOptions{})
	if err == nil {
		return nil
	}
	if apierrors.IsAlreadyExists(err) {
		return fmt.Errorf("restore service %s/%s already exists", spec.Namespace, spec.RestorePodName)
	}
	return fmt.Errorf("create restore service %s/%s: %w", spec.Namespace, spec.RestorePodName, err)
}

func createRestorePod(ctx context.Context, clientset kubernetes.Interface, spec runSpec) error {
	pod := buildRestorePod(spec)
	_, err := clientset.CoreV1().Pods(spec.Namespace).Create(ctx, pod, metav1.CreateOptions{})
	if err == nil {
		return nil
	}
	if apierrors.IsAlreadyExists(err) {
		return fmt.Errorf("restore pod %s/%s already exists", spec.Namespace, spec.RestorePodName)
	}
	return fmt.Errorf("create restore pod %s/%s: %w", spec.Namespace, spec.RestorePodName, err)
}

func waitForCheckpoint(ctx context.Context, clientset kubernetes.Interface, spec runSpec) (string, error) {
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for {
		job, err := clientset.BatchV1().Jobs(spec.Namespace).Get(ctx, spec.CheckpointJobName, metav1.GetOptions{})
		if err != nil {
			if ctx.Err() != nil {
				return "", fmt.Errorf("wait for checkpoint job %s/%s: %w", spec.Namespace, spec.CheckpointJobName, ctx.Err())
			}
			return "", fmt.Errorf("get checkpoint job %s/%s: %w", spec.Namespace, spec.CheckpointJobName, err)
		}

		status := strings.TrimSpace(job.Annotations[checkpointStatusAnnotation])
		if status == "completed" {
			return status, nil
		}
		if status == "failed" {
			return "", fmt.Errorf("checkpoint job %s/%s failed", spec.Namespace, spec.CheckpointJobName)
		}

		select {
		case <-ctx.Done():
			summaryCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			return "", fmt.Errorf("checkpoint job %s/%s timed out: %s", spec.Namespace, spec.CheckpointJobName, checkpointPodSummary(summaryCtx, clientset, spec))
		case <-ticker.C:
		}
	}
}

func waitForRestore(ctx context.Context, clientset kubernetes.Interface, spec runSpec) (string, error) {
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for {
		pod, err := clientset.CoreV1().Pods(spec.Namespace).Get(ctx, spec.RestorePodName, metav1.GetOptions{})
		if err != nil {
			if ctx.Err() != nil {
				return "", fmt.Errorf("wait for restore pod %s/%s: %w", spec.Namespace, spec.RestorePodName, ctx.Err())
			}
			return "", fmt.Errorf("get restore pod %s/%s: %w", spec.Namespace, spec.RestorePodName, err)
		}

		status := strings.TrimSpace(pod.Annotations[restoreStatusAnnotation])
		if status == "completed" {
			return status, nil
		}
		if status == "failed" {
			return "", fmt.Errorf("restore pod %s/%s failed", spec.Namespace, spec.RestorePodName)
		}
		if pod.Status.Phase == corev1.PodFailed {
			return "", fmt.Errorf("restore pod %s/%s entered phase Failed (%s)", spec.Namespace, spec.RestorePodName, pod.Status.Reason)
		}

		select {
		case <-ctx.Done():
			return "", fmt.Errorf("restore pod %s/%s timed out: phase=%s status=%q", spec.Namespace, spec.RestorePodName, pod.Status.Phase, status)
		case <-ticker.C:
		}
	}
}

func checkpointPodSummary(ctx context.Context, clientset kubernetes.Interface, spec runSpec) string {
	pods, err := clientset.CoreV1().Pods(spec.Namespace).List(ctx, metav1.ListOptions{
		LabelSelector: "batch.kubernetes.io/job-name=" + spec.CheckpointJobName,
	})
	if err != nil {
		return "unable to list checkpoint pod: " + err.Error()
	}
	if len(pods.Items) == 0 {
		return "no checkpoint pod created yet"
	}
	pod := pods.Items[0]
	parts := []string{fmt.Sprintf("pod=%s phase=%s", pod.Name, pod.Status.Phase)}
	for _, condition := range pod.Status.Conditions {
		if condition.Status == corev1.ConditionTrue || condition.Status == corev1.ConditionFalse {
			parts = append(parts, fmt.Sprintf("%s=%s", condition.Type, condition.Status))
		}
	}
	for _, status := range pod.Status.ContainerStatuses {
		if status.State.Waiting != nil {
			parts = append(parts, fmt.Sprintf("container=%s waiting=%s", status.Name, status.State.Waiting.Reason))
		}
		if status.State.Terminated != nil {
			parts = append(parts, fmt.Sprintf("container=%s terminated=%s", status.Name, status.State.Terminated.Reason))
		}
	}
	return strings.Join(parts, " ")
}

func (w selectedWorker) mainContainer() corev1.Container {
	return *w.PodSpec.Containers[w.MainContainerIndex].DeepCopy()
}

func containsFlag(args []string, name string) bool {
	for index := range args {
		if args[index] == name {
			return true
		}
		if strings.HasPrefix(args[index], name+"=") {
			return true
		}
	}
	return false
}

func compactStrings(values []string) []string {
	compacted := make([]string, 0, len(values))
	for _, value := range values {
		if trimmed := strings.TrimSpace(value); trimmed != "" {
			compacted = append(compacted, trimmed)
		}
	}
	return compacted
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if trimmed := strings.TrimSpace(value); trimmed != "" {
			return trimmed
		}
	}
	return ""
}

func mergeStringMaps(base map[string]string, overlays ...map[string]string) map[string]string {
	merged := cloneMap(base)
	for _, overlay := range overlays {
		if len(overlay) == 0 {
			continue
		}
		if merged == nil {
			merged = map[string]string{}
		}
		for key, value := range overlay {
			merged[key] = value
		}
	}
	return merged
}

func mergeEnv(base []corev1.EnvVar, overlays ...[]corev1.EnvVar) []corev1.EnvVar {
	merged := append([]corev1.EnvVar(nil), base...)
	indexByName := map[string]int{}
	for index, envVar := range merged {
		indexByName[envVar.Name] = index
	}
	for _, overlay := range overlays {
		for _, envVar := range overlay {
			if index, ok := indexByName[envVar.Name]; ok {
				merged[index] = envVar
				continue
			}
			indexByName[envVar.Name] = len(merged)
			merged = append(merged, envVar)
		}
	}
	return merged
}

func mergeVolumes(base []corev1.Volume, overlays []corev1.Volume) []corev1.Volume {
	merged := append([]corev1.Volume(nil), base...)
	indexByName := map[string]int{}
	for index, volume := range merged {
		indexByName[volume.Name] = index
	}
	for _, volume := range overlays {
		if index, ok := indexByName[volume.Name]; ok {
			merged[index] = volume
			continue
		}
		indexByName[volume.Name] = len(merged)
		merged = append(merged, volume)
	}
	return merged
}

func mergeVolumeMounts(base []corev1.VolumeMount, overlays []corev1.VolumeMount) []corev1.VolumeMount {
	merged := append([]corev1.VolumeMount(nil), base...)
	indexByName := map[string]int{}
	for index, mount := range merged {
		indexByName[mount.Name] = index
	}
	for _, mount := range overlays {
		if index, ok := indexByName[mount.Name]; ok {
			merged[index] = mount
			continue
		}
		indexByName[mount.Name] = len(merged)
		merged = append(merged, mount)
	}
	return merged
}

func sanitizeName(value string) string {
	value = strings.ToLower(value)
	value = dnsLabelPattern.ReplaceAllString(value, "-")
	value = strings.Trim(value, "-")
	if value == "" {
		value = "snapshot"
	}
	if len(value) > 63 {
		value = strings.Trim(value[:63], "-")
	}
	if value == "" {
		return "snapshot"
	}
	return value
}

func quantityPtr(q resource.Quantity) *resource.Quantity {
	return &q
}

func int32Ptr(value int32) *int32 {
	return &value
}

func stringPtr(value string) *string {
	return &value
}
