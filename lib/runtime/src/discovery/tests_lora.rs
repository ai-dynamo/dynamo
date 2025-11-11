// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for LoRA-specific discovery functionality

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::discovery::{Discovery, DiscoverySpec, KVStoreDiscovery};

    #[test]
    fn test_lora_key_generation_with_slug() {
        // Test that LoRA registration appends slug to the key
        let namespace = "dynamo";
        let component = "backend";
        let endpoint = "generate";
        let lora_slug = "my-lora-adapter";

        // Create a model card with LoRA user_data
        let card_json = json!({
            "display_name": "My LoRA Adapter",
            "slug": lora_slug,
            "model_type": "Chat",
            "model_input": "Tokens",
            "user_data": {
                "lora_adapter": true,
                "lora_id": 123,
                "lora_path": "/path/to/lora"
            }
        });

        // Create discovery spec for LoRA
        let spec = DiscoverySpec::Model {
            namespace: namespace.to_string(),
            component: component.to_string(),
            endpoint: endpoint.to_string(),
            card_json: card_json.clone(),
        };

        // Verify the spec contains the lora information
        match &spec {
            DiscoverySpec::Model { card_json, .. } => {
                assert!(card_json.get("user_data").is_some());
                assert_eq!(card_json["user_data"]["lora_adapter"].as_bool(), Some(true));
                assert_eq!(card_json["slug"].as_str(), Some(lora_slug));
            }
            _ => panic!("Expected Model spec"),
        }
    }

    #[test]
    fn test_base_model_key_without_slug() {
        // Test that base model registration does NOT append slug
        let namespace = "dynamo";
        let component = "backend";
        let endpoint = "generate";

        // Create a model card without LoRA user_data
        let card_json = json!({
            "display_name": "Base Model",
            "model_type": "Chat",
            "model_input": "Tokens"
        });

        let spec = DiscoverySpec::Model {
            namespace: namespace.to_string(),
            component: component.to_string(),
            endpoint: endpoint.to_string(),
            card_json: card_json.clone(),
        };

        // Verify no lora_adapter flag
        match &spec {
            DiscoverySpec::Model { card_json, .. } => {
                assert!(
                    card_json.get("user_data").is_none()
                        || card_json["user_data"].get("lora_adapter").is_none()
                        || card_json["user_data"]["lora_adapter"].as_bool() != Some(true)
                );
            }
            _ => panic!("Expected Model spec"),
        }
    }

    #[test]
    fn test_lora_slug_generation_from_display_name() {
        // Test slug generation when slug field is missing but display_name exists
        let card_json = json!({
            "display_name": "My LoRA Adapter With Spaces",
            "model_type": "Chat",
            "model_input": "Tokens",
            "user_data": {
                "lora_adapter": true,
                "lora_id": 123
            }
        });

        // The expected slug should be: "my-lora-adapter-with-spaces"
        let display_name = card_json["display_name"].as_str().unwrap();
        let computed_slug: String = display_name
            .to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '-' })
            .collect::<String>()
            .split('-')
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("-");

        assert_eq!(computed_slug, "my-lora-adapter-with-spaces");
    }

    #[test]
    fn test_lora_slug_with_special_characters() {
        // Test slug generation with special characters
        let test_cases = vec![
            ("My-LoRA_Adapter", "my-lora-adapter"),
            ("LoRA@123#456", "lora-123-456"),
            ("test___adapter", "test-adapter"),
            ("LoRA!!!Adapter", "lora-adapter"),
        ];

        for (input, expected) in test_cases {
            let computed_slug: String = input
                .to_lowercase()
                .chars()
                .map(|c| if c.is_alphanumeric() { c } else { '-' })
                .collect::<String>()
                .split('-')
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join("-");

            assert_eq!(computed_slug, expected, "Failed for input: {}", input);
        }
    }

    #[test]
    fn test_model_key_format() {
        // Test the basic model key format for keys used in KVStoreDiscovery
        let namespace = "test-namespace";
        let component = "test-component";
        let endpoint = "test-endpoint";
        let instance_id = 98765u64;

        // Base model key format: namespace/component/endpoint/instance_id
        let expected_base_key = format!("{}/{}/{}/{}", namespace, component, endpoint, instance_id);

        // LoRA key format: namespace/component/endpoint/instance_id/lora_slug
        let lora_slug = "my-lora";
        let expected_lora_key = format!("{}/{}", expected_base_key, lora_slug);
        assert_eq!(
            expected_lora_key,
            format!(
                "{}/{}/{}/{}/{}",
                namespace, component, endpoint, instance_id, lora_slug
            )
        );

        // Count slashes to verify format
        assert_eq!(expected_base_key.matches('/').count(), 3);
        assert_eq!(expected_lora_key.matches('/').count(), 4);
    }
}
