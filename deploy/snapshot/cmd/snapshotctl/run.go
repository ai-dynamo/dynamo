package main

import (
	"context"
	"fmt"
	"maps"
	"os"
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

	snapshotpodspec "github.com/ai-dynamo/dynamo/deploy/snapshot/podspec"
	"sigs.k8s.io/yaml"
)

const (
	defaultGeneratedCheckpointHashPrefix = "manual-snapshot"
)

type checkpointOptions struct {
	ManifestPath                 string
	Namespace                    string
	CheckpointHash               string
	DisableCudaCheckpointJobFile bool
	Timeout                      time.Duration
}

type restoreOptions struct {
	ManifestPath   string
	Namespace      string
	CheckpointHash string
	Timeout        time.Duration
}

type result struct {
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
}

type runSpec struct {
	Namespace                    string
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

func runCheckpointFlow(ctx context.Context, opts checkpointOptions) (*result, error) {
	if strings.TrimSpace(opts.ManifestPath) == "" {
		return nil, fmt.Errorf("missing required flags: --manifest")
	}
	if opts.Timeout <= 0 {
		return nil, fmt.Errorf("--timeout must be greater than zero")
	}

	pod, err := loadPod(opts.ManifestPath)
	if err != nil {
		return nil, err
	}

	clientset, currentNamespace, err := loadClientset()
	if err != nil {
		return nil, err
	}

	namespace := corev1.NamespaceDefault
	if currentNamespace != "" {
		namespace = currentNamespace
	}
	if pod.ManifestNamespace != "" {
		namespace = pod.ManifestNamespace
	}
	if opts.Namespace != "" {
		namespace = opts.Namespace
	}
	storage, err := discoverSnapshotStorage(ctx, clientset, namespace)
	if err != nil {
		return nil, err
	}
	spec := buildRunSpec(namespace, opts.CheckpointHash, opts.DisableCudaCheckpointJobFile, pod, storage)

	job, err := buildCheckpointJob(spec)
	if err != nil {
		return nil, err
	}
	_, err = clientset.BatchV1().Jobs(spec.Namespace).Create(ctx, job, metav1.CreateOptions{})
	if apierrors.IsAlreadyExists(err) {
		return nil, fmt.Errorf("checkpoint job %s/%s already exists", spec.Namespace, spec.CheckpointJobName)
	}
	if err != nil {
		return nil, err
	}

	waitCtx, cancel := context.WithTimeout(ctx, opts.Timeout)
	defer cancel()
	status, err := waitForCheckpoint(waitCtx, clientset, spec)
	if err != nil {
		return nil, err
	}

	return &result{
		Name:               spec.Pod.Name,
		Namespace:          spec.Namespace,
		CheckpointHash:     spec.CheckpointHash,
		CheckpointLocation: spec.checkpointLocation(),
		CheckpointJob:      spec.CheckpointJobName,
		Status:             status,
	}, nil
}

func runRestoreFlow(ctx context.Context, opts restoreOptions) (*result, error) {
	var missing []string
	if strings.TrimSpace(opts.ManifestPath) == "" {
		missing = append(missing, "--manifest")
	}
	if strings.TrimSpace(opts.CheckpointHash) == "" {
		missing = append(missing, "--checkpoint-hash")
	}
	if len(missing) != 0 {
		return nil, fmt.Errorf("missing required flags: %s", strings.Join(missing, ", "))
	}
	if opts.Timeout <= 0 {
		return nil, fmt.Errorf("--timeout must be greater than zero")
	}

	pod, err := loadPod(opts.ManifestPath)
	if err != nil {
		return nil, err
	}

	clientset, currentNamespace, err := loadClientset()
	if err != nil {
		return nil, err
	}

	namespace := corev1.NamespaceDefault
	if currentNamespace != "" {
		namespace = currentNamespace
	}
	if pod.ManifestNamespace != "" {
		namespace = pod.ManifestNamespace
	}
	if opts.Namespace != "" {
		namespace = opts.Namespace
	}
	storage, err := discoverSnapshotStorage(ctx, clientset, namespace)
	if err != nil {
		return nil, err
	}
	spec := buildRunSpec(namespace, opts.CheckpointHash, false, pod, storage)

	restorePod := buildRestorePod(spec)
	_, err = clientset.CoreV1().Pods(spec.Namespace).Create(ctx, restorePod, metav1.CreateOptions{})
	if apierrors.IsAlreadyExists(err) {
		return nil, fmt.Errorf("restore pod %s/%s already exists", spec.Namespace, spec.RestorePodName)
	}
	if err != nil {
		return nil, err
	}

	waitCtx, cancel := context.WithTimeout(ctx, opts.Timeout)
	defer cancel()
	status, err := waitForRestore(waitCtx, clientset, spec)
	if err != nil {
		return nil, err
	}

	return &result{
		Name:               spec.Pod.Name,
		Namespace:          spec.Namespace,
		CheckpointHash:     spec.CheckpointHash,
		CheckpointLocation: spec.checkpointLocation(),
		RestorePod:         spec.RestorePodName,
		Status:             status,
	}, nil
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
			}, nil
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

	name := strings.TrimSpace(pod.Name)
	if name == "" {
		name = "snapshot-worker"
	}

	return selectedPod{
		Name:              name,
		ManifestNamespace: strings.TrimSpace(pod.Namespace),
		Labels:            maps.Clone(pod.Labels),
		Annotations:       maps.Clone(pod.Annotations),
		PodSpec:           *pod.Spec.DeepCopy(),
	}, nil
}

func buildRunSpec(namespace string, checkpointHash string, disableCudaCheckpointJobFile bool, pod selectedPod, storage snapshotStorage) runSpec {
	if strings.TrimSpace(checkpointHash) == "" {
		checkpointHash = fmt.Sprintf("%s-%d", defaultGeneratedCheckpointHashPrefix, time.Now().UTC().UnixNano())
	}

	return runSpec{
		Namespace:                    namespace,
		CheckpointHash:               checkpointHash,
		DisableCudaCheckpointJobFile: disableCudaCheckpointJobFile,
		SeccompLocalhostProfile:      snapshotpodspec.DefaultSeccompLocalhostProfile,
		Pod:                          pod,
		SnapshotStorage:              storage,
		CheckpointJobName:            pod.Name + "-checkpoint",
		RestorePodName:               pod.Name,
	}
}

func (s runSpec) checkpointLocation() string {
	return strings.TrimRight(s.SnapshotStorage.BasePath, "/") + "/" + s.CheckpointHash
}

func buildCheckpointJob(spec runSpec) (*batchv1.Job, error) {
	podSpec := *spec.Pod.PodSpec.DeepCopy()
	podSpec.RestartPolicy = corev1.RestartPolicyNever
	snapshotpodspec.InjectLocalhostSeccompProfile(&podSpec, spec.SeccompLocalhostProfile)
	snapshotpodspec.InjectCheckpointVolume(&podSpec, spec.SnapshotStorage.PVCName)

	container := *spec.Pod.PodSpec.Containers[0].DeepCopy()
	snapshotpodspec.InjectCheckpointVolumeMount(&container, spec.SnapshotStorage.BasePath)
	if !spec.DisableCudaCheckpointJobFile {
		if len(container.Command) == 0 {
			return nil, fmt.Errorf(
				"manifest must set container.command when launch-job wrapping is enabled; use --disable-cuda-checkpoint-job-file to preserve the image entrypoint",
			)
		}
		container.Command, container.Args = snapshotpodspec.WrapWithCudaCheckpointLaunchJob(container.Command, container.Args)
	}
	podSpec.Containers = []corev1.Container{container}

	labels := maps.Clone(spec.Pod.Labels)
	annotations := maps.Clone(spec.Pod.Annotations)
	if labels == nil {
		labels = map[string]string{}
	}
	if annotations == nil {
		annotations = map[string]string{}
	}
	snapshotpodspec.ApplyCheckpointSourceMetadata(
		labels,
		annotations,
		spec.CheckpointHash,
		spec.checkpointLocation(),
		snapshotpodspec.StorageTypePVC,
	)
	zeroBackoffLimit := int32(0)

	return &batchv1.Job{
		TypeMeta: metav1.TypeMeta{APIVersion: "batch/v1", Kind: "Job"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      spec.CheckpointJobName,
			Namespace: spec.Namespace,
		},
		Spec: batchv1.JobSpec{
			BackoffLimit: &zeroBackoffLimit,
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

func buildRestorePod(spec runSpec) *corev1.Pod {
	podSpec := *spec.Pod.PodSpec.DeepCopy()
	podSpec.RestartPolicy = corev1.RestartPolicyNever
	snapshotpodspec.InjectLocalhostSeccompProfile(&podSpec, spec.SeccompLocalhostProfile)
	snapshotpodspec.InjectCheckpointVolume(&podSpec, spec.SnapshotStorage.PVCName)

	container := *spec.Pod.PodSpec.Containers[0].DeepCopy()
	snapshotpodspec.InjectCheckpointVolumeMount(&container, spec.SnapshotStorage.BasePath)
	snapshotpodspec.InjectRestoreTUN(&podSpec, &container)
	snapshotpodspec.SetRestorePlaceholderCommand(&container)
	podSpec.Containers = []corev1.Container{container}

	labels := maps.Clone(spec.Pod.Labels)
	annotations := maps.Clone(spec.Pod.Annotations)
	if labels == nil {
		labels = map[string]string{}
	}
	if annotations == nil {
		annotations = map[string]string{}
	}
	snapshotpodspec.ApplyRestoreTargetMetadata(
		labels,
		annotations,
		true,
		spec.CheckpointHash,
		spec.checkpointLocation(),
		snapshotpodspec.StorageTypePVC,
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

		status := strings.TrimSpace(job.Annotations[snapshotpodspec.CheckpointStatusAnnotation])
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

		status := strings.TrimSpace(pod.Annotations[snapshotpodspec.RestoreStatusAnnotation])
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
