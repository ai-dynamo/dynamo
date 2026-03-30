package checkpoint

import (
	"strings"

	snapshotv1alpha1 "github.com/ai-dynamo/dynamo/deploy/snapshot/api/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func CheckpointRequestName(snapshotID string, artifactVersion string) string {
	return snapshotRequestName("checkpoint", snapshotID, artifactVersion)
}

func RestoreRequestName(snapshotID string, podName string) string {
	shortID := strings.TrimSpace(snapshotID)
	if len(shortID) > 16 {
		shortID = shortID[:16]
	}
	return snapshotRequestName("restore", podName, shortID)
}

func BuildRestoreRequest(namespace string, name string, snapshotID string, artifactVersion string, podName string) *snapshotv1alpha1.SnapshotRequest {
	return &snapshotv1alpha1.SnapshotRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: snapshotv1alpha1.SnapshotRequestSpec{
			Phase:           snapshotv1alpha1.SnapshotRequestPhaseRestore,
			SnapshotID:      snapshotID,
			ArtifactVersion: artifactVersion,
			TargetPodRef: &snapshotv1alpha1.SnapshotTargetPodRef{
				Name: podName,
			},
		},
	}
}

func snapshotRequestName(parts ...string) string {
	name := strings.ToLower(strings.Join(parts, "-"))
	name = strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z':
			return r
		case r >= '0' && r <= '9':
			return r
		case r == '-':
			return r
		default:
			return '-'
		}
	}, name)
	name = strings.Trim(name, "-")
	for strings.Contains(name, "--") {
		name = strings.ReplaceAll(name, "--", "-")
	}
	if len(name) > 63 {
		name = strings.Trim(name[:63], "-")
	}
	if name == "" {
		return "snapshot-request"
	}
	return name
}
