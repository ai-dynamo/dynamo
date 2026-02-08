// metadata.go provides backwards compatibility aliases for checkpoint data types.
// DEPRECATED: Use github.com/ai-dynamo/dynamo/deploy/chrek/pkg/config instead.
package common

import (
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/config"
)

// Constants - deprecated, use config package
const (
	// CheckpointDataFilename is the name of the checkpoint data file in checkpoint directories
	// Deprecated: Use config.CheckpointDataFilename
	CheckpointDataFilename = config.CheckpointDataFilename
	// DescriptorsFilename is the name of the file descriptors file
	// Deprecated: Use config.DescriptorsFilename
	DescriptorsFilename = config.DescriptorsFilename
)

// Type aliases for backwards compatibility
// Deprecated: Use types from github.com/ai-dynamo/dynamo/deploy/chrek/pkg/config

// CheckpointData stores information needed for cross-node restore
// Deprecated: Use config.CheckpointData
type CheckpointData = config.CheckpointData

// MountMetadata stores information about a mount for remapping during restore
// Deprecated: Use config.MountMetadata
type MountMetadata = config.MountMetadata

// NamespaceMetadata stores namespace information
// Deprecated: Use config.NamespaceMetadata
type NamespaceMetadata = config.NamespaceMetadata

// Function aliases for backwards compatibility
// Deprecated: Use functions from github.com/ai-dynamo/dynamo/deploy/chrek/pkg/config

// NewCheckpointData creates a new checkpoint data instance
// Deprecated: Use config.NewCheckpointData
var NewCheckpointData = config.NewCheckpointData

// SaveCheckpointData writes checkpoint data to a YAML file in the checkpoint directory
// Deprecated: Use config.SaveCheckpointData
var SaveCheckpointData = config.SaveCheckpointData

// LoadCheckpointData reads checkpoint data from a checkpoint directory
// Deprecated: Use config.LoadCheckpointData
var LoadCheckpointData = config.LoadCheckpointData

// SaveDescriptors writes file descriptor information to the checkpoint directory
// Deprecated: Use config.SaveDescriptors
var SaveDescriptors = config.SaveDescriptors

// LoadDescriptors reads file descriptor information from checkpoint directory
// Deprecated: Use config.LoadDescriptors
var LoadDescriptors = config.LoadDescriptors

// GetCheckpointDir returns the path to a checkpoint directory
// Deprecated: Use config.GetCheckpointDir
var GetCheckpointDir = config.GetCheckpointDir

// ListCheckpoints returns all checkpoint IDs in the base directory
// Deprecated: Use config.ListCheckpoints
var ListCheckpoints = config.ListCheckpoints

// GetCheckpointInfo returns checkpoint data for a specific checkpoint
// Deprecated: Use config.GetCheckpointInfo
var GetCheckpointInfo = config.GetCheckpointInfo

// DeleteCheckpoint removes a checkpoint directory
// Deprecated: Use config.DeleteCheckpoint
var DeleteCheckpoint = config.DeleteCheckpoint
