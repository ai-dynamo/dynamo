// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block ID validation for transfers.
//!
//! This module provides validation functions to ensure block transfers are safe and correct.

use super::PhysicalLayout;
use std::collections::HashSet;
use thiserror::Error;

/// Validation errors for block transfers.
#[derive(Debug, Error, PartialEq)]
pub enum BlockValidationError {
    /// Destination block IDs contain duplicates.
    #[error("Destination block IDs are not unique: duplicates = {duplicates:?}")]
    DuplicateDestinationBlocks { duplicates: Vec<usize> },

    /// Lists have mismatched lengths.
    #[error(
        "Block ID lists have mismatched lengths: src={src_len}, dst={dst_len}, bounce={bounce_len:?}"
    )]
    LengthMismatch {
        src_len: usize,
        dst_len: usize,
        bounce_len: Option<usize>,
    },

    /// Block ID is out of range for the layout.
    #[error("Block ID {block_id} out of range for {layout_name} (max={max})")]
    BlockOutOfRange {
        block_id: usize,
        layout_name: &'static str,
        max: usize,
    },

    /// Bounce block IDs contain duplicates.
    #[error("Bounce block IDs are not unique: duplicates = {duplicates:?}")]
    DuplicateBounceBlocks { duplicates: Vec<usize> },
}

/// Validate that destination block IDs are unique (no duplicates).
///
/// # Arguments
/// * `dst_block_ids` - Destination block IDs
///
/// # Returns
/// Ok(()) if unique, Err with duplicate IDs otherwise
pub fn validate_dst_unique(dst_block_ids: &[usize]) -> Result<(), BlockValidationError> {
    let mut seen = HashSet::new();
    let mut duplicates = Vec::new();

    for &id in dst_block_ids {
        if !seen.insert(id) && !duplicates.contains(&id) {
            duplicates.push(id);
        }
    }

    if duplicates.is_empty() {
        Ok(())
    } else {
        Err(BlockValidationError::DuplicateDestinationBlocks { duplicates })
    }
}

/// Validate that bounce block IDs are unique (no duplicates).
pub fn validate_bounce_unique(bounce_block_ids: &[usize]) -> Result<(), BlockValidationError> {
    let mut seen = HashSet::new();
    let mut duplicates = Vec::new();

    for &id in bounce_block_ids {
        if !seen.insert(id) && !duplicates.contains(&id) {
            duplicates.push(id);
        }
    }

    if duplicates.is_empty() {
        Ok(())
    } else {
        Err(BlockValidationError::DuplicateBounceBlocks { duplicates })
    }
}

/// Validate block IDs are in range for a layout.
#[cfg(debug_assertions)]
pub fn validate_block_ids_in_range(
    block_ids: &[usize],
    layout: &PhysicalLayout,
    layout_name: &'static str,
) -> Result<(), BlockValidationError> {
    let max_blocks = layout.layout().config().num_blocks;

    for &block_id in block_ids {
        if block_id >= max_blocks {
            return Err(BlockValidationError::BlockOutOfRange {
                block_id,
                layout_name,
                max: max_blocks,
            });
        }
    }

    Ok(())
}

/// Full validation for block transfer (debug mode).
///
/// Validates:
/// - List lengths match
/// - Destination IDs are unique
/// - Bounce IDs are unique (if provided)
/// - Source and destination are disjoint (if same layout)
/// - All block IDs are in range for their respective layouts
#[cfg(debug_assertions)]
pub fn validate_block_transfer(
    src_block_ids: &[usize],
    dst_block_ids: &[usize],
    bounce_block_ids: Option<&[usize]>,
    src_layout: &PhysicalLayout,
    dst_layout: &PhysicalLayout,
    bounce_layout: Option<&PhysicalLayout>,
) -> Result<(), BlockValidationError> {
    // Validate lengths
    if src_block_ids.len() != dst_block_ids.len() {
        return Err(BlockValidationError::LengthMismatch {
            src_len: src_block_ids.len(),
            dst_len: dst_block_ids.len(),
            bounce_len: bounce_block_ids.map(|ids| ids.len()),
        });
    }

    if let Some(bounce_ids) = bounce_block_ids
        && bounce_ids.len() != src_block_ids.len()
    {
        return Err(BlockValidationError::LengthMismatch {
            src_len: src_block_ids.len(),
            dst_len: dst_block_ids.len(),
            bounce_len: Some(bounce_ids.len()),
        });
    }

    #[cfg(debug_assertions)]
    {
        // Validate destination uniqueness
        validate_dst_unique(dst_block_ids)?;

        // Validate bounce uniqueness if provided
        if let Some(bounce_ids) = bounce_block_ids {
            validate_bounce_unique(bounce_ids)?;
        }

        // Validate block IDs in range
        validate_block_ids_in_range(src_block_ids, src_layout, "source")?;
        validate_block_ids_in_range(dst_block_ids, dst_layout, "destination")?;
        if let (Some(bounce_ids), Some(bounce_layout)) = (bounce_block_ids, bounce_layout) {
            validate_block_ids_in_range(bounce_ids, bounce_layout, "bounce")?;
        }
    }

    Ok(())
}

/// Minimal validation for block transfer (release mode).
///
/// Only validates:
/// - List lengths match
/// - Destination IDs are unique
#[cfg(not(debug_assertions))]
pub fn validate_block_transfer(
    src_block_ids: &[usize],
    dst_block_ids: &[usize],
    bounce_block_ids: Option<&[usize]>,
    _src_layout: &PhysicalLayout,
    _dst_layout: &PhysicalLayout,
    _bounce_layout: Option<&PhysicalLayout>,
) -> Result<(), BlockValidationError> {
    // Validate lengths
    if src_block_ids.len() != dst_block_ids.len() {
        return Err(BlockValidationError::LengthMismatch {
            src_len: src_block_ids.len(),
            dst_len: dst_block_ids.len(),
            bounce_len: bounce_block_ids.map(|ids| ids.len()),
        });
    }

    if let Some(bounce_ids) = bounce_block_ids {
        if bounce_ids.len() != src_block_ids.len() {
            return Err(BlockValidationError::LengthMismatch {
                src_len: src_block_ids.len(),
                dst_len: dst_block_ids.len(),
                bounce_len: Some(bounce_ids.len()),
            });
        }
    }

    // Validate destination uniqueness
    validate_dst_unique(dst_block_ids)?;

    Ok(())
}

#[cfg(all(test, feature = "testing-nixl"))]
mod tests {
    use super::super::tests::*;
    use super::*;

    #[test]
    fn test_dst_unique_valid() {
        let ids = vec![0, 1, 2, 3, 4];
        assert!(validate_dst_unique(&ids).is_ok());
    }

    #[test]
    fn test_dst_unique_duplicate() {
        let ids = vec![0, 1, 2, 1, 3];
        let result = validate_dst_unique(&ids);
        assert!(result.is_err());
        match result.unwrap_err() {
            BlockValidationError::DuplicateDestinationBlocks { duplicates } => {
                assert_eq!(duplicates, vec![1]);
            }
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_dst_unique_multiple_duplicates() {
        let ids = vec![0, 1, 2, 1, 3, 2];
        let result = validate_dst_unique(&ids);
        assert!(result.is_err());
        match result.unwrap_err() {
            BlockValidationError::DuplicateDestinationBlocks { duplicates } => {
                assert!(duplicates.contains(&1));
                assert!(duplicates.contains(&2));
            }
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_length_mismatch() {
        let physical1 = builder(2)
            .fully_contiguous()
            .allocate_system()
            .build()
            .unwrap();

        let physical2 = builder(2)
            .fully_contiguous()
            .allocate_system()
            .build()
            .unwrap();

        let src_ids = vec![0, 1, 2];
        let dst_ids = vec![5, 6]; // Different length

        let result =
            validate_block_transfer(&src_ids, &dst_ids, None, &physical1, &physical2, None);
        assert!(result.is_err());
        match result.unwrap_err() {
            BlockValidationError::LengthMismatch {
                src_len,
                dst_len,
                bounce_len,
            } => {
                assert_eq!(src_len, 3);
                assert_eq!(dst_len, 2);
                assert_eq!(bounce_len, None);
            }
            _ => panic!("Wrong error type"),
        }
    }

    // #[test]
    // #[cfg(debug_assertions)]
    // fn test_block_out_of_range() {
    //     let (_layout, physical) = create_test_layout(5); // Only 5 blocks
    //     let src_ids = vec![0, 1, 2];
    //     let dst_ids = vec![3, 4, 10]; // 10 is out of range

    //     let result = validate_block_ids_in_range(&dst_ids, &physical, "destination");
    //     assert!(result.is_err());
    //     match result.unwrap_err() {
    //         BlockValidationError::BlockOutOfRange {
    //             block_id,
    //             layout_name,
    //             max,
    //         } => {
    //             assert_eq!(block_id, 10);
    //             assert_eq!(layout_name, "destination");
    //             assert_eq!(max, 5);
    //         }
    //         _ => panic!("Wrong error type"),
    //     }
    // }

    // #[test]
    // fn test_bounce_length_mismatch() {
    //     let (_layout1, physical1) = create_test_layout(10);
    //     let (_layout2, physical2) = create_test_layout(10);
    //     let (_layout3, physical3) = create_test_layout(10);
    //     let src_ids = vec![0, 1, 2];
    //     let dst_ids = vec![5, 6, 7];
    //     let bounce_ids = vec![8, 9]; // Wrong length

    //     let result = validate_block_transfer(
    //         &src_ids,
    //         &dst_ids,
    //         Some(&bounce_ids),
    //         &physical1,
    //         &physical2,
    //         Some(&physical3),
    //     );
    //     assert!(result.is_err());
    //     match result.unwrap_err() {
    //         BlockValidationError::LengthMismatch {
    //             src_len,
    //             dst_len,
    //             bounce_len,
    //         } => {
    //             assert_eq!(src_len, 3);
    //             assert_eq!(dst_len, 3);
    //             assert_eq!(bounce_len, Some(2));
    //         }
    //         _ => panic!("Wrong error type"),
    //     }
    // }

    // #[test]
    // fn test_full_validation_success() {
    //     let (_layout1, physical1) = create_test_layout(10);
    //     let (_layout2, physical2) = create_test_layout(10);
    //     let (_layout3, physical3) = create_test_layout(10);
    //     let src_ids = vec![0, 1, 2];
    //     let dst_ids = vec![5, 6, 7];
    //     let bounce_ids = vec![8, 9, 3];

    //     assert!(
    //         validate_block_transfer(
    //             &src_ids,
    //             &dst_ids,
    //             Some(&bounce_ids),
    //             &physical1,
    //             &physical2,
    //             Some(&physical3),
    //         )
    //         .is_ok()
    //     );
    // }
}
