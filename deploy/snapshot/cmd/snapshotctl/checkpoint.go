package main

import (
	"context"
	"fmt"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
)

const defaultGeneratedCheckpointIDPrefix = "manual-snapshot"

type checkpointOptions struct {
	ManifestPath                 string
	Namespace                    string
	KubeContext                  string
	CheckpointID                 string
	Container                    string
	DisableCudaCheckpointJobFile bool
	Timeout                      time.Duration
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

	pod, clientset, crClient, namespace, storage, err := loadRunContext(ctx, opts.ManifestPath, opts.Namespace, opts.KubeContext)
	if err != nil {
		return nil, err
	}

	checkpointID := strings.TrimSpace(opts.CheckpointID)
	if checkpointID == "" {
		checkpointID = fmt.Sprintf("%s-%d", defaultGeneratedCheckpointIDPrefix, time.Now().UTC().UnixNano())
	}
	resolvedStorage, err := snapshotprotocol.ResolveCheckpointStorage(checkpointID, "", snapshotprotocol.Storage{
		Type:     snapshotprotocol.StorageTypePVC,
		PVCName:  storage.PVCName,
		BasePath: storage.BasePath,
	})
	if err != nil {
		return nil, err
	}

	targetValue, err := reconcileTargetContainers(pod.Annotations, opts.Container, 1, 1)
	if err != nil {
		return nil, err
	}
	annotations := map[string]string{}
	for k, v := range pod.Annotations {
		annotations[k] = v
	}
	annotations[snapshotprotocol.TargetContainersAnnotation] = targetValue

	checkpointJobName := pod.Name + "-checkpoint"
	job, err := snapshotprotocol.NewCheckpointJob(&corev1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels:      pod.Labels,
			Annotations: annotations,
		},
		Spec: *pod.Spec.DeepCopy(),
	}, snapshotprotocol.CheckpointJobOptions{
		Namespace:       namespace,
		CheckpointID:    checkpointID,
		ArtifactVersion: snapshotprotocol.DefaultCheckpointArtifactVersion,
		SeccompProfile:  snapshotprotocol.DefaultSeccompLocalhostProfile,
		Name:            checkpointJobName,
		WrapLaunchJob:   !opts.DisableCudaCheckpointJobFile,
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
	snap, err := createPodSnapshot(waitCtx, crClient, namespace, snapName, sourcePod.Name, sourcePod.UID, checkpointID)
	if err != nil {
		return nil, err
	}

	snap, err = waitForPodSnapshot(waitCtx, crClient, namespace, snap.Name)
	if err != nil {
		return nil, err
	}

	res := &result{
		Name:               pod.Name,
		Namespace:          namespace,
		CheckpointID:       checkpointID,
		CheckpointLocation: resolvedStorage.Location,
		CheckpointJob:      checkpointJobName,
		PodSnapshot:        snap.Name,
		Status:             "completed",
	}
	if snap.Status.BoundPodSnapshotContentName != nil && *snap.Status.BoundPodSnapshotContentName != "" {
		res.BoundContent = *snap.Status.BoundPodSnapshotContentName
	}
	return res, nil
}
