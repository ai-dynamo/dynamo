// constants.go defines shared constants used across checkpoint and restore packages.
package checkpoint

const (
	// HostProcPath is the mount point for the host's /proc in DaemonSet pods.
	HostProcPath = "/host/proc"

	// DevShmDirName is the directory name for captured /dev/shm contents.
	DevShmDirName = "dev-shm"

	// KubeLabelCheckpointSource is the pod label that triggers automatic checkpointing.
	// Set by the operator on checkpoint-eligible pods.
	KubeLabelCheckpointSource = "nvidia.com/checkpoint-source"

	// KubeLabelCheckpointHash is the pod label specifying the checkpoint identity hash.
	// Set by the operator on checkpoint-eligible pods.
	KubeLabelCheckpointHash = "nvidia.com/checkpoint-hash"

	// DumpLogFilename is the CRIU dump (checkpoint) log filename.
	DumpLogFilename = "dump.log"

	// CheckpointCRIUConfFilename is the CRIU config file written at checkpoint time.
	CheckpointCRIUConfFilename = "criu.conf"

	// CheckpointDoneFilename is the marker file written to the checkpoint directory
	// after all checkpoint artifacts are complete. Used to detect checkpoint readiness.
	// Also hard-coded in vLLM for early-exit when checkpoint already exists.
	CheckpointDoneFilename = "checkpoint.done"

	// CheckpointManifestFilename is the name of the manifest file in checkpoint directories.
	CheckpointManifestFilename = "manifest.yaml"

	// DescriptorsFilename is the name of the file descriptors file.
	DescriptorsFilename = "descriptors.yaml"

	// RootfsDiffFilename is the name of the rootfs diff tar in checkpoint directories.
	RootfsDiffFilename = "rootfs-diff.tar"

	// DeletedFilesFilename is the name of the deleted files JSON in checkpoint directories.
	DeletedFilesFilename = "deleted-files.json"
)
