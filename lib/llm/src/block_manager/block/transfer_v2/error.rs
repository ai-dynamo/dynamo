use super::super::BlockError;
use thiserror::Error;

/// Error type for transfer operations in v2
#[derive(Debug, Error)]
pub enum TransferError {
    #[error("Builder configuration error: {0}")]
    BuilderError(String),
    #[error("Transfer execution failed: {0}")]
    ExecutionError(String),
    #[error("Incompatible block types provided: {0}")]
    IncompatibleTypes(String),
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
    #[error("Mismatched source/destination counts: {0} sources, {1} destinations")]
    CountMismatch(usize, usize),
    #[error("Block operation failed: {0}")]
    BlockError(String),
    // TODO: Add NIXL specific errors
    #[error("No blocks provided")]
    NoBlocksProvided,

    // #[error("Mismatched {0:?} block set index: {1} != {2}")]
    // MismatchedBlockSetIndex(BlockTarget, usize, usize),

    // #[error("Mismatched {0:?} worker ID: {1} != {2}")]
    // MismatchedWorkerID(BlockTarget, usize, usize),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
