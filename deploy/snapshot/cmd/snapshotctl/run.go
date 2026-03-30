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

	pod, clientset, namespace, storage, err := loadRunContext(ctx, opts.ManifestPath, opts.Namespace)
	if err != nil {
		return nil, err
	}

	checkpointHash := strings.TrimSpace(opts.CheckpointHash)
	if checkpointHash == "" {
		checkpointHash = fmt.Sprintf("%s-%d", defaultGeneratedCheckpointHashPrefix, time.Now().UTC().UnixNano())
	}
	checkpointLocation := strings.TrimRight(storage.BasePath, "/") + "/" + checkpointHash
	checkpointJobName := pod.Name + "-checkpoint"

	job, err := buildCheckpointJob(pod, namespace, checkpointJobName, checkpointHash, checkpointLocation, storage, opts.DisableCudaCheckpointJobFile)
	if err != nil {
		return nil, err
	}
	_, err = clientset.BatchV1().Jobs(namespace).Create(ctx, job, metav1.CreateOptions{})
	if apierrors.IsAlreadyExists(err) {
		return nil, fmt.Errorf("checkpoint job %s/%s already exists", namespace, checkpointJobName)
	}
	if err != nil {
		return nil, err
	}

	waitCtx, cancel := context.WithTimeout(ctx, opts.Timeout)
	defer cancel()
	status, err := waitForCheckpoint(waitCtx, clientset, namespace, checkpointJobName)
	if err != nil {
		return nil, err
	}

	return &result{
		Name:               pod.Name,
		Namespace:          namespace,
		CheckpointHash:     checkpointHash,
		CheckpointLocation: checkpointLocation,
		CheckpointJob:      checkpointJobName,
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

	pod, clientset, namespace, storage, err := loadRunContext(ctx, opts.ManifestPath, opts.Namespace)
	if err != nil {
		return nil, err
	}

	checkpointHash := strings.TrimSpace(opts.CheckpointHash)
	checkpointLocation := strings.TrimRight(storage.BasePath, "/") + "/" + checkpointHash

	restorePod := buildRestorePod(pod, namespace, checkpointHash, checkpointLocation, storage)
	_, err = clientset.CoreV1().Pods(namespace).Create(ctx, restorePod, metav1.CreateOptions{})
	if apierrors.IsAlreadyExists(err) {
		return nil, fmt.Errorf("restore pod %s/%s already exists", namespace, pod.Name)
	}
	if err != nil {
		return nil, err
	}

	waitCtx, cancel := context.WithTimeout(ctx, opts.Timeout)
	defer cancel()
	status, err := waitForRestore(waitCtx, clientset, namespace, pod.Name)
	if err != nil {
		return nil, err
	}

	return &result{
		Name:               pod.Name,
		Namespace:          namespace,
		CheckpointHash:     checkpointHash,
		CheckpointLocation: checkpointLocation,
		RestorePod:         pod.Name,
		Status:             status,
	}, nil
}

func loadRunContext(ctx context.Context, manifestPath string, namespaceOverride string) (*corev1.Pod, kubernetes.Interface, string, snapshotStorage, error) {
	pod, err := loadPod(manifestPath)
	if err != nil {
		return nil, nil, "", snapshotStorage{}, err
	}

	clientset, currentNamespace, err := loadClientset()
	if err != nil {
		return nil, nil, "", snapshotStorage{}, err
	}

	namespace := currentNamespace
	if namespace == "" {
		namespace = corev1.NamespaceDefault
	}
	if pod.Namespace != "" {
		namespace = pod.Namespace
	}
	if namespaceOverride != "" {
		namespace = namespaceOverride
	}

	storage, err := discoverSnapshotStorage(ctx, clientset, namespace)
	if err != nil {
		return nil, nil, "", snapshotStorage{}, err
	}
	return pod, clientset, namespace, storage, nil
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

func loadPod(manifestPath string) (*corev1.Pod, error) {
	content, err := os.ReadFile(manifestPath)
	if err != nil {
		return nil, fmt.Errorf("read manifest %s: %w", manifestPath, err)
	}

	var pod corev1.Pod
	if err := yaml.Unmarshal(content, &pod); err != nil {
		return nil, fmt.Errorf("parse manifest %s: %w", manifestPath, err)
	}
	if kind := strings.TrimSpace(pod.Kind); kind != "" && kind != "Pod" {
		return nil, fmt.Errorf("manifest %s is kind %q, expected Pod", manifestPath, kind)
	}
	if len(pod.Spec.Containers) != 1 {
		return nil, fmt.Errorf(
			"manifest %s has %d containers; snapshotctl requires exactly one worker container",
			manifestPath,
			len(pod.Spec.Containers),
		)
	}
	if strings.TrimSpace(pod.Spec.Containers[0].Image) == "" {
		return nil, fmt.Errorf("manifest %s: worker container image is required", manifestPath)
	}
	if strings.TrimSpace(pod.Name) == "" {
		return nil, fmt.Errorf("manifest %s: metadata.name is required", manifestPath)
	}

	pod.Namespace = strings.TrimSpace(pod.Namespace)
	return &pod, nil
}

func buildCheckpointJob(
	pod *corev1.Pod,
	namespace string,
	jobName string,
	checkpointHash string,
	checkpointLocation string,
	storage snapshotStorage,
	disableCudaCheckpointJobFile bool,
) (*batchv1.Job, error) {
	podSpec := *pod.Spec.DeepCopy()
	podSpec.RestartPolicy = corev1.RestartPolicyNever
	snapshotpodspec.InjectLocalhostSeccompProfile(&podSpec, snapshotpodspec.DefaultSeccompLocalhostProfile)
	snapshotpodspec.InjectCheckpointVolume(&podSpec, storage.PVCName)

	container := *pod.Spec.Containers[0].DeepCopy()
	snapshotpodspec.InjectCheckpointVolumeMount(&container, storage.BasePath)
	if !disableCudaCheckpointJobFile {
		if len(container.Command) == 0 {
			return nil, fmt.Errorf(
				"manifest must set container.command when launch-job wrapping is enabled; use --disable-cuda-checkpoint-job-file to preserve the image entrypoint",
			)
		}
		container.Command, container.Args = snapshotpodspec.WrapWithCudaCheckpointLaunchJob(container.Command, container.Args)
	}
	podSpec.Containers = []corev1.Container{container}

	labels := maps.Clone(pod.Labels)
	annotations := maps.Clone(pod.Annotations)
	if labels == nil {
		labels = map[string]string{}
	}
	if annotations == nil {
		annotations = map[string]string{}
	}
	snapshotpodspec.ApplyCheckpointSourceMetadata(
		labels,
		annotations,
		checkpointHash,
		checkpointLocation,
		snapshotpodspec.StorageTypePVC,
	)
	zeroBackoffLimit := int32(0)

	return &batchv1.Job{
		TypeMeta: metav1.TypeMeta{APIVersion: "batch/v1", Kind: "Job"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      jobName,
			Namespace: namespace,
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

func buildRestorePod(
	pod *corev1.Pod,
	namespace string,
	checkpointHash string,
	checkpointLocation string,
	storage snapshotStorage,
) *corev1.Pod {
	podSpec := *pod.Spec.DeepCopy()
	podSpec.RestartPolicy = corev1.RestartPolicyNever
	snapshotpodspec.InjectLocalhostSeccompProfile(&podSpec, snapshotpodspec.DefaultSeccompLocalhostProfile)
	snapshotpodspec.InjectCheckpointVolume(&podSpec, storage.PVCName)

	container := *pod.Spec.Containers[0].DeepCopy()
	snapshotpodspec.InjectCheckpointVolumeMount(&container, storage.BasePath)
	snapshotpodspec.InjectRestoreTUN(&podSpec, &container)
	snapshotpodspec.SetRestorePlaceholderCommand(&container)
	podSpec.Containers = []corev1.Container{container}

	labels := maps.Clone(pod.Labels)
	annotations := maps.Clone(pod.Annotations)
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
		checkpointHash,
		checkpointLocation,
		snapshotpodspec.StorageTypePVC,
	)

	return &corev1.Pod{
		TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "Pod"},
		ObjectMeta: metav1.ObjectMeta{
			Name:        pod.Name,
			Namespace:   namespace,
			Labels:      labels,
			Annotations: annotations,
		},
		Spec: podSpec,
	}
}

func waitForCheckpoint(ctx context.Context, clientset kubernetes.Interface, namespace string, jobName string) (string, error) {
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for {
		job, err := clientset.BatchV1().Jobs(namespace).Get(ctx, jobName, metav1.GetOptions{})
		if err != nil {
			if ctx.Err() != nil {
				return "", fmt.Errorf("wait for checkpoint job %s/%s: %w", namespace, jobName, ctx.Err())
			}
			return "", fmt.Errorf("get checkpoint job %s/%s: %w", namespace, jobName, err)
		}

		status := strings.TrimSpace(job.Annotations[snapshotpodspec.CheckpointStatusAnnotation])
		if status == "completed" {
			return status, nil
		}
		if status == "failed" {
			return "", fmt.Errorf("checkpoint job %s/%s failed", namespace, jobName)
		}

		select {
		case <-ctx.Done():
			summaryCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			pods, err := clientset.CoreV1().Pods(namespace).List(summaryCtx, metav1.ListOptions{
				LabelSelector: "batch.kubernetes.io/job-name=" + jobName,
			})
			summary := "no checkpoint pod created yet"
			if err != nil {
				summary = "unable to list checkpoint pod: " + err.Error()
			} else if len(pods.Items) != 0 {
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
				summary = strings.Join(parts, " ")
			}
			return "", fmt.Errorf(
				"checkpoint job %s/%s timed out: %s",
				namespace,
				jobName,
				summary,
			)
		case <-ticker.C:
		}
	}
}

func waitForRestore(ctx context.Context, clientset kubernetes.Interface, namespace string, podName string) (string, error) {
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for {
		pod, err := clientset.CoreV1().Pods(namespace).Get(ctx, podName, metav1.GetOptions{})
		if err != nil {
			if ctx.Err() != nil {
				return "", fmt.Errorf("wait for restore pod %s/%s: %w", namespace, podName, ctx.Err())
			}
			return "", fmt.Errorf("get restore pod %s/%s: %w", namespace, podName, err)
		}

		status := strings.TrimSpace(pod.Annotations[snapshotpodspec.RestoreStatusAnnotation])
		if status == "completed" {
			return status, nil
		}
		if status == "failed" {
			return "", fmt.Errorf("restore pod %s/%s failed", namespace, podName)
		}
		if pod.Status.Phase == corev1.PodFailed {
			return "", fmt.Errorf("restore pod %s/%s entered phase Failed (%s)", namespace, podName, pod.Status.Reason)
		}

		select {
		case <-ctx.Done():
			return "", fmt.Errorf("restore pod %s/%s timed out: phase=%s status=%q", namespace, podName, pod.Status.Phase, status)
		case <-ticker.C:
		}
	}
}
