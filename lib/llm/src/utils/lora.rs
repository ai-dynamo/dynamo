// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use blake3;

/// Generate a deterministic signed int32 ID from a LoRA name using blake3 hash.
pub fn lora_name_to_id(lora_name: &str) -> i32 {
    let hash = blake3::hash(lora_name.as_bytes());
    let hash_bytes = hash.as_bytes();

    let mut bytes_array = [0u8; 8];
    bytes_array.copy_from_slice(&hash_bytes[..8]);
    let hash_u64 = u64::from_be_bytes(bytes_array);

    let lora_id: i32 = ((hash_u64 & 0x7FFFFFFF) as i32).abs();

    // Ensure non-zero ID
    if lora_id == 0 { 1 } else { lora_id }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_name_to_id() {
        let id = lora_name_to_id("test_lora");
        assert!(1 <= id);

        let id1 = lora_name_to_id("test_lora");
        let id2 = lora_name_to_id("test_lora");
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_lora_id_stability_across_version() {
        let id1 = lora_name_to_id("test_lora");
        assert_eq!(id1, 1983627077);
    }
}
