// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test support for the media protocol schemas.

use std::any::type_name;

use utoipa::PartialSchema;
use utoipa::openapi::schema::Object;
use utoipa::openapi::{RefOr, Schema};

/// The object schema of `T`. Panics when the schema of `T` is not an object.
pub(crate) fn object_schema<T: PartialSchema>() -> Object {
    match T::schema() {
        RefOr::T(Schema::Object(object)) => object,
        _ => panic!("the schema of {} is not an object", type_name::<T>()),
    }
}

/// The `default` of one property of `object`, as JSON. `Null` when the
/// property declares no default.
pub(crate) fn property_default(object: &Object, field: &str) -> serde_json::Value {
    serde_json::to_value(&object.properties[field]).unwrap()["default"].clone()
}
