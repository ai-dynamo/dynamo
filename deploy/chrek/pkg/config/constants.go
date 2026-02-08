// constants.go defines shared constants used across checkpoint, restore, and operator packages.
//
// CROSS-REFERENCE: Some of these constants are duplicated in the operator because
// the operator is a separate Go module. If you change a value here, update the
// corresponding constant in deploy/operator/internal/consts/consts.go to match.
// Duplicated constants: CheckpointBasePath, RestoreMarkerFilePath, CheckpointReadyFilePath.
package config

const (
	// === Infrastructure constants (not user-configurable) ===

	// HostProcPath is the mount point for the host's /proc in DaemonSet pods.
	HostProcPath = "/host/proc"

	// CheckpointBasePath is the base directory for checkpoint storage (PVC mount point).
	// Duplicated in operator: consts.CheckpointBasePath
	// Also hard-coded in smart-entrypoint.sh
	CheckpointBasePath = "/checkpoints"

	// CRIULogDir is the directory where CRIU restore logs are copied for debugging.
	CRIULogDir = "/checkpoints/restore-logs"

	// === Path constants ===

	// DevShmDirName is the directory name for captured /dev/shm contents.
	DevShmDirName = "dev-shm"

	// RestoreMarkerFilePath is the path written after successful CRIU restore.
	// The operator injects this as DYN_RESTORE_MARKER_FILE for vLLM to read.
	// Duplicated in operator: consts.RestoreMarkerFilePath
	RestoreMarkerFilePath = "/tmp/dynamo-restored"

	// RestoreTriggerPath is the default path to the trigger file for trigger-based restore.
	RestoreTriggerPath = "/tmp/restore-trigger"

	// CheckpointReadyFilePath is the path the worker writes when the model
	// is loaded and ready for checkpointing (used by readiness probe).
	// Duplicated in operator: consts.CheckpointReadyFilePath
	CheckpointReadyFilePath = "/tmp/checkpoint-ready"

	// === Checkpoint artifact filenames ===

	// CheckpointDoneFilename is the marker file written to the checkpoint directory
	// after all checkpoint artifacts are complete. Used to detect checkpoint readiness.
	// Also hard-coded in vLLM for early-exit when checkpoint already exists.
	CheckpointDoneFilename = "checkpoint.done"

	// CheckpointDataFilename is the name of the metadata file in checkpoint directories.
	CheckpointDataFilename = "metadata.yaml"

	// DescriptorsFilename is the name of the file descriptors file.
	DescriptorsFilename = "descriptors.yaml"

	// RootfsDiffFilename is the name of the rootfs diff tar in checkpoint directories.
	RootfsDiffFilename = "rootfs-diff.tar"

	// DeletedFilesFilename is the name of the deleted files JSON in checkpoint directories.
	DeletedFilesFilename = "deleted-files.json"
)
