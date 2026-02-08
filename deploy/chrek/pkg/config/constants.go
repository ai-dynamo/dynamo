// constants.go defines shared constants used across checkpoint and restore packages.
package config

const (
	// === Infrastructure constants (not user-configurable) ===

	// HostProcPath is the mount point for the host's /proc in DaemonSet pods.
	HostProcPath = "/host/proc"

	// CRIULogDir is the directory where CRIU restore logs are copied for debugging.
	CRIULogDir = "/checkpoints/restore-logs"

	// === Path constants ===

	// DevShmDirName is the directory name for captured /dev/shm contents.
	DevShmDirName = "dev-shm"

	// RestoreTriggerPath is the default path to the trigger file for trigger-based restore.
	RestoreTriggerPath = "/tmp/restore-trigger"

	// === Kubernetes labels (must match operator consts) ===

	// KubeLabelCheckpointSource is the pod label that triggers automatic checkpointing.
	// Set by the operator on checkpoint-eligible pods.
	KubeLabelCheckpointSource = "nvidia.com/checkpoint-source"

	// KubeLabelCheckpointHash is the pod label specifying the checkpoint identity hash.
	// Set by the operator on checkpoint-eligible pods.
	KubeLabelCheckpointHash = "nvidia.com/checkpoint-hash"

	// === CRIU log and config filenames ===

	// DumpLogFilename is the CRIU dump (checkpoint) log filename.
	DumpLogFilename = "dump.log"

	// RestoreLogFilename is the CRIU restore log filename.
	RestoreLogFilename = "restore.log"

	// CheckpointCRIUConfFilename is the CRIU config file written at checkpoint time.
	CheckpointCRIUConfFilename = "criu.conf"

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
