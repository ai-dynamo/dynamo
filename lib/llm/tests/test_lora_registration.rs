// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for LoRA registration functionality

#[cfg(test)]
mod tests {
    use dynamo_llm::utils::lora_name_to_hash_id;

    #[test]
    fn test_lora_hash_id_deterministic() {
        // Test that lora_name_to_hash_id produces consistent results
        let lora_name = "my-test-lora";
        let id1 = lora_name_to_hash_id(lora_name);
        let id2 = lora_name_to_hash_id(lora_name);

        assert_eq!(id1, id2, "Hash should be deterministic");
    }

    #[test]
    fn test_lora_hash_id_different_names() {
        // Test that different names produce different IDs
        let names = [
            "lora-adapter-1",
            "lora-adapter-2",
            "lora-adapter-3",
            "base-model",
            "another-lora",
        ];

        let ids: Vec<i32> = names.iter().map(|n| lora_name_to_hash_id(n)).collect();

        // All IDs should be unique
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                assert_ne!(
                    ids[i], ids[j],
                    "Different names should produce different IDs: {} vs {}",
                    names[i], names[j]
                );
            }
        }
    }

    #[test]
    fn test_lora_hash_id_range() {
        // Test that IDs are in the correct range (1 to 2,147,483,647)
        let test_names = (0..100).map(|i| format!("lora-{}", i)).collect::<Vec<_>>();

        for name in test_names {
            let id = lora_name_to_hash_id(&name);
            assert!(id > 0, "ID must be positive for {}", name);
        }
    }

    #[test]
    fn test_lora_hash_id_special_characters() {
        // Test that special characters in names work correctly
        let names = vec![
            "lora-with-dashes",
            "lora_with_underscores",
            "lora/with/slashes",
            "lora@special-chars",
            "UPPERCASE-LORA",
            "MixedCase-LoRA",
        ];

        for name in names {
            let id = lora_name_to_hash_id(name);
            assert!(id > 0, "ID must be positive for '{}'", name);
        }
    }

    #[test]
    fn test_lora_hash_id_empty_string() {
        // Test edge case: empty string
        let id = lora_name_to_hash_id("");
        assert!(id > 0, "Empty string should still produce valid ID");
    }

    #[test]
    fn test_lora_hash_id_long_name() {
        // Test with a very long name
        let long_name = "a".repeat(10000);
        let id = lora_name_to_hash_id(&long_name);
        assert!(id > 0, "Long name should produce valid ID");
    }

    #[test]
    fn test_lora_hash_collision_resistance() {
        // Test that similar names produce different IDs
        let similar_names = [
            "lora-adapter",
            "lora-adapter-",
            "lora-adapter-1",
            "lora-adapter1",
            "loraadapter",
            "lora_adapter",
        ];

        let ids: Vec<i32> = similar_names
            .iter()
            .map(|n| lora_name_to_hash_id(n))
            .collect();

        // Check all are unique
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                assert_ne!(
                    ids[i], ids[j],
                    "Similar names should still produce different IDs: '{}' vs '{}'",
                    similar_names[i], similar_names[j]
                );
            }
        }
    }

    #[test]
    fn test_lora_hash_batch_consistency() {
        // Test that generating many IDs in a batch maintains consistency
        let batch_size = 1000;
        let names: Vec<String> = (0..batch_size).map(|i| format!("lora-{}", i)).collect();

        // Generate IDs twice
        let ids_first: Vec<i32> = names.iter().map(|n| lora_name_to_hash_id(n)).collect();
        let ids_second: Vec<i32> = names.iter().map(|n| lora_name_to_hash_id(n)).collect();

        // Should be identical
        assert_eq!(
            ids_first, ids_second,
            "Batch generation should be consistent"
        );

        // All should be positive
        for (i, id) in ids_first.iter().enumerate() {
            assert!(*id > 0, "ID {} out of range for index {}", id, i);
        }
    }
}
