package main

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/yaml"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
)

const (
	podScheduleTimeout  = 5 * time.Minute
	podScheduleInterval = 3 * time.Second
	podSnapshotPollInterval = 3 * time.Second
)

func loadRunContext(ctx context.Context, manifestPath string, namespaceOverride string, kubeContext string) (*corev1.Pod, kubernetes.Interface, client.Client, string, snapshotprotocol.Storage, error) {
	pod, err := loadPod(manifestPath)
	if err != nil {
		return nil, nil, nil, "", snapshotprotocol.Storage{}, err
	}

	clientset, restConfig, currentNamespace, err := loadClientset(kubeContext)
	if err != nil {
		return nil, nil, nil, "", snapshotprotocol.Storage{}, err
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

	scheme := runtime.NewScheme()
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(nvidiacomv1alpha1.AddToScheme(scheme))
	crClient, err := client.New(restConfig, client.Options{Scheme: scheme})
	if err != nil {
		return nil, nil, nil, "", snapshotprotocol.Storage{}, fmt.Errorf("create controller-runtime client: %w", err)
	}

	storage, err := discoverSnapshotStorage(ctx, clientset, namespace)
	if err != nil {
		return nil, nil, nil, "", snapshotprotocol.Storage{}, err
	}
	return pod, clientset, crClient, namespace, storage, nil
}

func loadClientset(kubeContext string) (kubernetes.Interface, *rest.Config, string, error) {
	loadingRules := clientcmd.NewDefaultClientConfigLoadingRules()
	clientConfig := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(loadingRules, &clientcmd.ConfigOverrides{
		CurrentContext: strings.TrimSpace(kubeContext),
	})
	restConfig, err := clientConfig.ClientConfig()
	if err != nil {
		return nil, nil, "", fmt.Errorf("load kubeconfig: %w", err)
	}
	restConfig.Timeout = 30 * time.Second

	namespace, _, err := clientConfig.Namespace()
	if err != nil {
		return nil, nil, "", fmt.Errorf("resolve current namespace: %w", err)
	}
	if strings.TrimSpace(namespace) == "" {
		namespace = corev1.NamespaceDefault
	}

	clientset, err := kubernetes.NewForConfig(restConfig)
	if err != nil {
		return nil, nil, "", fmt.Errorf("create kubernetes client: %w", err)
	}
	return clientset, restConfig, namespace, nil
}

// waitForSourcePod polls until the Job's source pod exists and has been scheduled to a node.
// Returns an actionable error if the pod never schedules within podScheduleTimeout so the caller
// does not create a PodSnapshot that would spin on a SourcePodNotFound terminal condition.
func waitForSourcePod(ctx context.Context, clientset kubernetes.Interface, namespace, jobName string, jobUID types.UID) (*corev1.Pod, error) {
	var sourcePod *corev1.Pod
	err := wait.PollUntilContextTimeout(ctx, podScheduleInterval, podScheduleTimeout, true, func(ctx context.Context) (bool, error) {
		pods, listErr := clientset.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{
			LabelSelector: "batch.kubernetes.io/job-name=" + jobName,
		})
		if listErr != nil {
			return false, listErr
		}
		for i := range pods.Items {
			pod := &pods.Items[i]
			if !isControlledByUID(pod, jobUID) {
				continue
			}
			if pod.Spec.NodeName == "" {
				return false, nil
			}
			sourcePod = pod
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		return nil, fmt.Errorf("source pod for job %s/%s not scheduled within %s: %w", namespace, jobName, podScheduleTimeout, err)
	}
	if sourcePod == nil {
		return nil, fmt.Errorf("source pod for job %s/%s not scheduled within %s; cannot create PodSnapshot", namespace, jobName, podScheduleTimeout)
	}
	return sourcePod, nil
}

func isControlledByUID(pod *corev1.Pod, uid types.UID) bool {
	for _, ref := range pod.OwnerReferences {
		if ref.Controller != nil && *ref.Controller && ref.UID == uid {
			return true
		}
	}
	return false
}

func podSnapshotName(jobName string) string {
	if len(jobName) <= 63 {
		return jobName
	}
	return strings.TrimRight(jobName[:63], "-.")
}

// createPodSnapshot creates a PodSnapshot in the given namespace, pinning the source pod's UID.
// On AlreadyExists it returns an actionable error naming the existing object.
// On Forbidden it surfaces a clear RBAC error.
func createPodSnapshot(ctx context.Context, crClient client.Client, namespace, snapName, podName string, podUID types.UID, checkpointID string) (*nvidiacomv1alpha1.PodSnapshot, error) {
	snap := &nvidiacomv1alpha1.PodSnapshot{
		TypeMeta: metav1.TypeMeta{
			APIVersion: nvidiacomv1alpha1.GroupVersion.String(),
			Kind:       "PodSnapshot",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      snapName,
			Namespace: namespace,
			Labels: map[string]string{
				snapshotprotocol.CheckpointIDLabel: checkpointID,
			},
		},
		Spec: nvidiacomv1alpha1.PodSnapshotSpec{
			Source: nvidiacomv1alpha1.PodSnapshotSource{
				PodRef: nvidiacomv1alpha1.PodReference{
					Name: podName,
					UID:  podUID,
				},
			},
		},
	}
	if err := crClient.Create(ctx, snap); err != nil {
		if apierrors.IsAlreadyExists(err) {
			return nil, fmt.Errorf("PodSnapshot %s/%s already exists; another capture may be in progress", namespace, snapName)
		}
		if apierrors.IsForbidden(err) {
			return nil, fmt.Errorf("RBAC error creating PodSnapshot %s/%s: missing verb 'create' on podsnapshots; grant it to the snapshotctl caller", namespace, snapName)
		}
		return nil, fmt.Errorf("create PodSnapshot %s/%s: %w", namespace, snapName, err)
	}
	return snap, nil
}

// waitForPodSnapshot polls the PodSnapshot until it reaches a terminal state (Ready or Failed).
// On success it returns the updated PodSnapshot. On failure it surfaces the Failed condition's
// Reason and Message for actionable diagnostics.
func waitForPodSnapshot(ctx context.Context, crClient client.Client, namespace, name string) (*nvidiacomv1alpha1.PodSnapshot, error) {
	var result *nvidiacomv1alpha1.PodSnapshot
	err := wait.PollUntilContextTimeout(ctx, podSnapshotPollInterval, ctxRemaining(ctx), true, func(ctx context.Context) (bool, error) {
		snap := &nvidiacomv1alpha1.PodSnapshot{}
		if getErr := crClient.Get(ctx, client.ObjectKey{Namespace: namespace, Name: name}, snap); getErr != nil {
			if apierrors.IsNotFound(getErr) {
				return false, nil
			}
			return false, getErr
		}
		if nvidiacomv1alpha1.IsPodSnapshotSucceeded(snap) {
			result = snap
			return true, nil
		}
		if nvidiacomv1alpha1.IsPodSnapshotFailed(snap) {
			return false, podSnapshotFailedError(snap)
		}
		return false, nil
	})
	if err != nil {
		return nil, err
	}
	return result, nil
}

// podSnapshotFailedError builds an error from the PodSnapshot's Failed condition Reason and Message.
func podSnapshotFailedError(snap *nvidiacomv1alpha1.PodSnapshot) error {
	for _, c := range snap.Status.Conditions {
		if c.Type == nvidiacomv1alpha1.PodSnapshotConditionFailed && c.Status == metav1.ConditionTrue {
			if c.Reason != "" || c.Message != "" {
				return fmt.Errorf("PodSnapshot %s/%s failed: %s: %s", snap.Namespace, snap.Name, c.Reason, c.Message)
			}
		}
	}
	return fmt.Errorf("PodSnapshot %s/%s failed", snap.Namespace, snap.Name)
}

// ctxRemaining returns the time remaining on ctx's deadline, or 1 hour if unset.
func ctxRemaining(ctx context.Context) time.Duration {
	if d, ok := ctx.Deadline(); ok {
		return time.Until(d)
	}
	return time.Hour
}

func discoverSnapshotStorage(ctx context.Context, clientset kubernetes.Interface, namespace string) (snapshotprotocol.Storage, error) {
	daemonSets, err := clientset.AppsV1().DaemonSets(namespace).List(ctx, metav1.ListOptions{
		LabelSelector: snapshotprotocol.SnapshotAgentLabelSelector,
	})
	if err != nil {
		return snapshotprotocol.Storage{}, fmt.Errorf("list snapshot-agent daemonsets in %s: %w", namespace, err)
	}

	return snapshotprotocol.DiscoverStorageFromDaemonSets(namespace, daemonSets.Items)
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
	if len(pod.Spec.Containers) == 0 {
		return nil, fmt.Errorf(
			"manifest %s has no worker containers; snapshotctl requires at least one worker container",
			manifestPath,
		)
	}
	// snapshotctl no longer guesses the workload container. Callers pass
	// --container / --containers (or pre-stamp the
	// nvidia.com/snapshot-target-containers annotation), which the protocol
	// layer then validates against the pod spec.
	if strings.TrimSpace(pod.Name) == "" {
		return nil, fmt.Errorf("manifest %s: metadata.name is required", manifestPath)
	}
	for i := range pod.Spec.Containers {
		if strings.TrimSpace(pod.Spec.Containers[i].Image) == "" {
			return nil, fmt.Errorf("manifest %s: container %q image is required", manifestPath, pod.Spec.Containers[i].Name)
		}
	}

	pod.Namespace = strings.TrimSpace(pod.Namespace)
	return &pod, nil
}
