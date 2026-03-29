package manual

import (
	"context"
	"fmt"
	"os"
	"regexp"
	"slices"
	"strings"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"

	snapshotkube "github.com/ai-dynamo/dynamo/deploy/snapshot/pkg/kube"
	"gopkg.in/yaml.v3"
)

const (
	defaultGeneratedCheckpointHashPrefix = "manual-snapshot"
)

var dnsLabelPattern = regexp.MustCompile(`[^a-z0-9-]+`)

type CheckpointOptions struct {
	ManifestPath                 string
	Namespace                    string
	CheckpointHash               string
	DisableCudaCheckpointJobFile bool
	Timeout                      time.Duration
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
	CheckpointHash     string
	CheckpointLocation string
	CheckpointJob      string
	RestorePod         string
	Status             string
}

type selectedPod struct {
	Name              string
	ManifestNamespace string
	Labels            map[string]string
	Annotations       map[string]string
	PodSpec           corev1.PodSpec
	ContainerIndex    int
}

type runSpec struct {
	Namespace                    string
	WorkloadName                 string
	ResourcePrefix               string
	CheckpointHash               string
	DisableCudaCheckpointJobFile bool
	SeccompLocalhostProfile      string
	Pod                          selectedPod
	SnapshotStorage              snapshotStorage
	CheckpointJobName            string
	RestorePodName               string
}

type snapshotStorage struct {
	PVCName  string
	BasePath string
}

func RunCheckpoint(ctx context.Context, opts CheckpointOptions) (*Result, error) {
	if err := validateCheckpointOptions(opts); err != nil {
		return nil, err
	}

	pod, err := loadPod(opts.ManifestPath)
	if err != nil {
		return nil, err
	}

	clientset, currentNamespace, err := loadClientset()
	if err != nil {
		return nil, err
	}

	namespace := resolveNamespace(opts.Namespace, pod.ManifestNamespace, currentNamespace)
	storage, err := discoverSnapshotStorage(ctx, clientset, namespace)
	if err != nil {
		return nil, err
	}
	spec := buildRunSpec(namespace, opts.CheckpointHash, opts.DisableCudaCheckpointJobFile, pod, storage)

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

	pod, err := loadPod(opts.ManifestPath)
	if err != nil {
		return nil, err
	}

	clientset, currentNamespace, err := loadClientset()
	if err != nil {
		return nil, err
	}

	namespace := resolveNamespace(opts.Namespace, pod.ManifestNamespace, currentNamespace)
	storage, err := discoverSnapshotStorage(ctx, clientset, namespace)
	if err != nil {
		return nil, err
	}
	spec := buildRunSpec(namespace, opts.CheckpointHash, false, pod, storage)

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
		CheckpointHash:     spec.CheckpointHash,
		CheckpointLocation: spec.checkpointLocation(),
		RestorePod:         spec.RestorePodName,
		Status:             status,
	}, nil
}

func validateCheckpointOptions(opts CheckpointOptions) error {
	if strings.TrimSpace(opts.ManifestPath) == "" {
		return fmt.Errorf("missing required flags: --manifest")
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
	return snapshotStorage{}, fmt.Errorf(
		"snapshot-agent daemonset in %s does not mount a PVC-backed checkpoint volume (%s)",
		namespace,
		strings.Join(names, ", "),
	)
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

func loadPod(manifestPath string) (selectedPod, error) {
	content, err := os.ReadFile(manifestPath)
	if err != nil {
		return selectedPod{}, fmt.Errorf("read manifest %s: %w", manifestPath, err)
	}

	var pod corev1.Pod
	if err := yaml.Unmarshal(content, &pod); err != nil {
		return selectedPod{}, fmt.Errorf("parse manifest %s: %w", manifestPath, err)
	}
	if kind := strings.TrimSpace(pod.Kind); kind != "" && kind != "Pod" {
		return selectedPod{}, fmt.Errorf("manifest %s is kind %q, expected Pod", manifestPath, kind)
	}
	if len(pod.Spec.Containers) != 1 {
		return selectedPod{}, fmt.Errorf(
			"manifest %s has %d containers; snapshotctl requires exactly one worker container",
			manifestPath,
			len(pod.Spec.Containers),
		)
	}
	if strings.TrimSpace(pod.Spec.Containers[0].Image) == "" {
		return selectedPod{}, fmt.Errorf("manifest %s: worker container image is required", manifestPath)
	}

	return selectedPod{
		Name:              sanitizeName(firstNonEmpty(pod.Name, "snapshot-worker")),
		ManifestNamespace: strings.TrimSpace(pod.Namespace),
		Labels:            cloneMap(pod.Labels),
		Annotations:       cloneMap(pod.Annotations),
		PodSpec:           *pod.Spec.DeepCopy(),
		ContainerIndex:    0,
	}, nil
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

func buildRunSpec(namespace string, checkpointHash string, disableCudaCheckpointJobFile bool, pod selectedPod, storage snapshotStorage) runSpec {
	if strings.TrimSpace(checkpointHash) == "" {
		checkpointHash = fmt.Sprintf("%s-%d", defaultGeneratedCheckpointHashPrefix, time.Now().UTC().UnixNano())
	}

	resourcePrefix := sanitizeName(pod.Name + "-" + shortNameSuffix(checkpointHash))
	return runSpec{
		Namespace:                    namespace,
		WorkloadName:                 pod.Name,
		ResourcePrefix:               resourcePrefix,
		CheckpointHash:               checkpointHash,
		DisableCudaCheckpointJobFile: disableCudaCheckpointJobFile,
		SeccompLocalhostProfile:      snapshotkube.DefaultSeccompLocalhostProfile,
		Pod:                          pod,
		SnapshotStorage:              storage,
		CheckpointJobName:            sanitizeName(resourcePrefix + "-checkpoint"),
		RestorePodName:               pod.Name,
	}
}

func (s runSpec) checkpointLocation() string {
	return strings.TrimRight(s.SnapshotStorage.BasePath, "/") + "/" + s.CheckpointHash
}

func buildCheckpointJob(spec runSpec) (*batchv1.Job, error) {
	podSpec := *spec.Pod.PodSpec.DeepCopy()
	podSpec.RestartPolicy = corev1.RestartPolicyNever
	snapshotkube.InjectLocalhostSeccompProfile(&podSpec, spec.SeccompLocalhostProfile)
	snapshotkube.InjectCheckpointVolume(&podSpec, spec.SnapshotStorage.PVCName)

	container, err := buildCheckpointContainer(spec)
	if err != nil {
		return nil, err
	}
	podSpec.Containers = []corev1.Container{container}

	labels := cloneMap(spec.Pod.Labels)
	annotations := cloneMap(spec.Pod.Annotations)
	if labels == nil {
		labels = map[string]string{}
	}
	if annotations == nil {
		annotations = map[string]string{}
	}
	snapshotkube.ApplyCheckpointSourceMetadata(
		labels,
		annotations,
		spec.CheckpointHash,
		spec.checkpointLocation(),
		snapshotkube.StorageTypePVC,
	)

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
					Labels:      labels,
					Annotations: annotations,
				},
				Spec: podSpec,
			},
		},
	}, nil
}

func buildCheckpointContainer(spec runSpec) (corev1.Container, error) {
	container := spec.Pod.mainContainer()
	snapshotkube.InjectCheckpointVolumeMount(&container, spec.SnapshotStorage.BasePath)
	if spec.DisableCudaCheckpointJobFile {
		return container, nil
	}
	if len(container.Command) == 0 {
		return corev1.Container{}, fmt.Errorf(
			"manifest must set container.command when launch-job wrapping is enabled; use --disable-cuda-checkpoint-job-file to preserve the image entrypoint",
		)
	}
	container.Command, container.Args = snapshotkube.WrapWithCudaCheckpointLaunchJob(container.Command, container.Args)
	return container, nil
}

func buildRestorePod(spec runSpec) *corev1.Pod {
	podSpec := *spec.Pod.PodSpec.DeepCopy()
	podSpec.RestartPolicy = corev1.RestartPolicyNever
	snapshotkube.InjectLocalhostSeccompProfile(&podSpec, spec.SeccompLocalhostProfile)
	snapshotkube.InjectCheckpointVolume(&podSpec, spec.SnapshotStorage.PVCName)

	container := spec.Pod.mainContainer()
	snapshotkube.InjectCheckpointVolumeMount(&container, spec.SnapshotStorage.BasePath)
	snapshotkube.InjectRestoreTUN(&podSpec, &container)
	snapshotkube.SetRestorePlaceholderCommand(&container)
	podSpec.Containers = []corev1.Container{container}

	labels := cloneMap(spec.Pod.Labels)
	annotations := cloneMap(spec.Pod.Annotations)
	if labels == nil {
		labels = map[string]string{}
	}
	if annotations == nil {
		annotations = map[string]string{}
	}
	snapshotkube.ApplyRestoreTargetMetadata(
		labels,
		annotations,
		true,
		spec.CheckpointHash,
		spec.checkpointLocation(),
		snapshotkube.StorageTypePVC,
	)

	return &corev1.Pod{
		TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "Pod"},
		ObjectMeta: metav1.ObjectMeta{
			Name:        spec.RestorePodName,
			Namespace:   spec.Namespace,
			Labels:      labels,
			Annotations: annotations,
		},
		Spec: podSpec,
	}
}

func createCheckpointJob(ctx context.Context, clientset kubernetes.Interface, spec runSpec) error {
	job, err := buildCheckpointJob(spec)
	if err != nil {
		return err
	}
	_, err = clientset.BatchV1().Jobs(spec.Namespace).Create(ctx, job, metav1.CreateOptions{})
	if err == nil {
		return nil
	}
	if apierrors.IsAlreadyExists(err) {
		return fmt.Errorf("checkpoint job %s/%s already exists", spec.Namespace, spec.CheckpointJobName)
	}
	return fmt.Errorf("create checkpoint job %s/%s: %w", spec.Namespace, spec.CheckpointJobName, err)
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

		status := strings.TrimSpace(job.Annotations[snapshotkube.CheckpointStatusAnnotation])
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
			return "", fmt.Errorf(
				"checkpoint job %s/%s timed out: %s",
				spec.Namespace,
				spec.CheckpointJobName,
				checkpointPodSummary(summaryCtx, clientset, spec),
			)
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

		status := strings.TrimSpace(pod.Annotations[snapshotkube.RestoreStatusAnnotation])
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

func (p selectedPod) mainContainer() corev1.Container {
	return *p.PodSpec.Containers[p.ContainerIndex].DeepCopy()
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

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if trimmed := strings.TrimSpace(value); trimmed != "" {
			return trimmed
		}
	}
	return ""
}

func shortNameSuffix(value string) string {
	sanitized := sanitizeName(value)
	if len(sanitized) <= 12 {
		return sanitized
	}
	return strings.Trim(sanitized[len(sanitized)-12:], "-")
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

func int32Ptr(value int32) *int32 {
	return &value
}
