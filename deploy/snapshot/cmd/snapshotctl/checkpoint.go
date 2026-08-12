package main

import (
	"context"
	"fmt"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
)

const defaultGeneratedCheckpointIDPrefix = "manual-snapshot"

type checkpointOptions struct {
	ManifestPath       string
	Namespace          string
	KubeContext        string
	CheckpointID       string
	Container          string
	CudaCheckpointWrap bool
	Timeout            time.Duration
}

type result struct {
	Name               string
	Namespace          string
	CheckpointID       string
	CheckpointLocation string
	CheckpointJob      string
	PodSnapshot        string
	BoundContent       string
	RestorePod         string
	Status             string
}

func runCheckpointFlow(ctx context.Context, opts checkpointOptions) (_ *result, retErr error) {
	if strings.TrimSpace(opts.ManifestPath) == "" {
		return nil, fmt.Errorf("missing required flags: --manifest")
	}
	if opts.Timeout <= 0 {
		return nil, fmt.Errorf("--timeout must be greater than zero")
	}

	pod, clientset, crClient, namespace, err := loadRunContext(opts.ManifestPath, opts.Namespace, opts.KubeContext)
	if err != nil {
		return nil, err
	}

	checkpointID := strings.TrimSpace(opts.CheckpointID)
	if checkpointID == "" {
		checkpointID = fmt.Sprintf("%s-%d", defaultGeneratedCheckpointIDPrefix, time.Now().UTC().UnixNano())
	}
	containers, err := reconcileTargetContainers(pod.Annotations, opts.Container, 1, 1)
	if err != nil {
		return nil, err
	}

	checkpointJobName := pod.Name + "-checkpoint"
	job, err := snapshotprotocol.NewCheckpointJob(&corev1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels:      pod.Labels,
			Annotations: pod.Annotations,
		},
		Spec: *pod.Spec.DeepCopy(),
	}, snapshotprotocol.CheckpointJobOptions{
		Namespace:       namespace,
		TargetContainer: containers[0],
		CheckpointID:    checkpointID,
		ArtifactVersion: snapshotprotocol.DefaultCheckpointArtifactVersion,
		SeccompProfile:  snapshotprotocol.DefaultSeccompLocalhostProfile,
		Name:            checkpointJobName,
		WrapLaunchJob:   opts.CudaCheckpointWrap,
	})
	if err != nil {
		return nil, err
	}
	createdJob, err := clientset.BatchV1().Jobs(namespace).Create(ctx, job, metav1.CreateOptions{})
	if apierrors.IsAlreadyExists(err) {
		return nil, fmt.Errorf("checkpoint job %s/%s already exists", namespace, checkpointJobName)
	}
	if err != nil {
		return nil, err
	}

	// Clean up the Job on any error after this point. The PodSnapshot is left in place
	// to aid debugging when the flow fails.
	defer func() {
		if retErr != nil {
			_ = clientset.BatchV1().Jobs(namespace).Delete(ctx, checkpointJobName, metav1.DeleteOptions{})
		}
	}()

	waitCtx, cancel := context.WithTimeout(ctx, opts.Timeout)
	defer cancel()

	sourcePod, err := waitForSourcePod(waitCtx, clientset, namespace, checkpointJobName, createdJob.UID)
	if err != nil {
		return nil, err
	}

	snapName := podSnapshotName(checkpointJobName)
	snap, err := createPodSnapshot(waitCtx, crClient, namespace, snapName, sourcePod.Name, sourcePod.UID, containers, checkpointID)
	if err != nil {
		return nil, err
	}

	snap, err = waitForPodSnapshot(waitCtx, crClient, namespace, snap.Name)
	if err != nil {
		return nil, err
	}

	boundContent, checkpointLocation, err := checkpointHandle(waitCtx, crClient, snap)
	if err != nil {
		return nil, err
	}

	res := &result{
		Name:               pod.Name,
		Namespace:          namespace,
		CheckpointID:       checkpointID,
		CheckpointLocation: checkpointLocation,
		CheckpointJob:      checkpointJobName,
		PodSnapshot:        snap.Name,
		BoundContent:       boundContent,
		Status:             "completed",
	}
	return res, nil
}

// checkpointHandle reads the agent-reported artifact location from the bound
// content object. An empty handle is valid: capture status can still be
// reported without predicting a storage path.
func checkpointHandle(ctx context.Context, crClient client.Client, snap *nvidiacomv1alpha1.PodSnapshot) (string, string, error) {
	if snap.Status.BoundPodSnapshotContentName == nil {
		return "", "", nil
	}
	boundContent := strings.TrimSpace(*snap.Status.BoundPodSnapshotContentName)
	if boundContent == "" {
		return "", "", nil
	}

	content := &nvidiacomv1alpha1.PodSnapshotContent{}
	if err := crClient.Get(ctx, client.ObjectKey{Name: boundContent}, content); err != nil {
		return "", "", fmt.Errorf("get bound PodSnapshotContent %s: %w", boundContent, err)
	}
	return boundContent, strings.TrimSpace(content.Status.SnapshotHandle), nil
}
