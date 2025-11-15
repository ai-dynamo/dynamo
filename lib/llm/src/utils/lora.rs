// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use blake3;

/// Generate a deterministic integer ID from a LoRA name using blake3 hash.
/// Returns a signed int32 (range: 1 to 2,147,483,647).
///
/// # Arguments
///
/// * `lora_name` - The name of the LoRA adapter
///
/// # Returns
///
/// A signed 32-bit integer ID in the range 1 to 2,147,483,647
pub fn lora_name_to_hash_id(lora_name: &str) -> i32 {
    let hash = blake3::hash(lora_name.as_bytes());
    let hash_bytes = hash.as_bytes();

    // Take first 8 bytes and convert to u64
    let mut bytes_array = [0u8; 8];
    bytes_array.copy_from_slice(&hash_bytes[..8]);
    let hash_u64 = u64::from_be_bytes(bytes_array);

    // Convert to signed int32 range (0x7FFFFFFF = 2,147,483,647)
    // Use bitwise AND to ensure non-zero (LoRA IDs should be non-zero)
    let lora_id = (hash_u64 & 0x7FFFFFFF) as i32;

    // Ensure non-zero ID
    if lora_id == 0 { 1 } else { lora_id }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_name_to_hash_id() {
        // Test that function returns a valid ID
        let id = lora_name_to_hash_id("test_lora");
        assert!(id > 0);

        // Test determinism
        let id1 = lora_name_to_hash_id("test_lora");
        let id2 = lora_name_to_hash_id("test_lora");
        assert_eq!(id1, id2);

        // Test different names produce different IDs
        let id3 = lora_name_to_hash_id("different_lora");
        assert_ne!(id1, id3);

        // Test that zero is mapped to 1
        // We need to find a name that would hash to zero, but that's unlikely
        // Instead, let's verify the range constraint
        for i in 0..100 {
            let name = format!("lora_{}", i);
            let id = lora_name_to_hash_id(&name);
            assert!(id > 0);
        }
    }
}
