// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use dynamo_backend_common::{BackendError, ErrorType};
use serde_json::json;

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn lora_load_payload_validates_names_and_source_schemes_without_io() {
        for uri in [
            "file:///models/adapter",
            "hf://org/adapter",
            "s3://bucket/adapter",
        ] {
            let update =
                parse_load_lora(&json!({"lora_name": "adapter-a", "source": {"uri": uri}})).unwrap();
            assert_eq!(
                update,
                LoadLoraUpdate {
                    name: "adapter-a".into(),
                    uri: uri.into()
                }
            );
        }
        for body in [
            json!(null),
            json!({}),
            json!({"lora_name": " "}),
            json!({"lora_name": "adapter", "source": {"uri": ""}}),
            json!({"lora_name": "adapter", "source": {"uri": "https://example.com/adapter"}}),
            json!({"lora_name": "adapter", "source": {"uri": "file://host/adapter"}}),
            json!({"lora_name": "adapter", "source": {"uri": "file:///adapter?query=1"}}),
            json!({"lora_name": "adapter", "source": {"uri": "file:///adapter#fragment"}}),
        ] {
            let error = parse_load_lora(&body).unwrap_err();
            assert_eq!(
                error.error_type(),
                ErrorType::Backend(BackendError::InvalidArgument)
            );
        }
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn native_lora_inventory_requires_unique_usable_identity() {
        let adapter = |name: &str, id| pb::LoraAdapter {
            lora_name: name.into(),
            lora_id: id,
            source_path: "/models/adapter".into(),
        };
        let sorted = validate_inventory(vec![adapter("b", 2), adapter("a", 1)]).unwrap();
        assert_eq!(
            sorted
                .iter()
                .map(|item| item.lora_name.as_str())
                .collect::<Vec<_>>(),
            ["a", "b"]
        );
        for inventory in [
            vec![adapter(" ", 1)],
            vec![adapter("a", 0)],
            vec![adapter("a", -1)],
            vec![pb::LoraAdapter {
                source_path: String::new(),
                ..adapter("a", 1)
            }],
            vec![adapter("a", 1), adapter("a", 2)],
            vec![adapter("a", 1), adapter("b", 1)],
        ] {
            let error = validate_inventory(inventory).unwrap_err();
            assert_eq!(
                error.error_type(),
                ErrorType::Backend(BackendError::Unknown)
            );
        }
    }
}
