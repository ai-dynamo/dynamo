// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use dynamo_backend_common::DynamoError;
use reqwest::{
    RequestBuilder,
    header::{AUTHORIZATION, HeaderValue},
};

use crate::client;

#[derive(Clone, Debug, Default)]
pub(crate) struct HttpAuth {
    api_key: Option<HeaderValue>,
    admin_api_key: Option<HeaderValue>,
}

impl HttpAuth {
    pub(crate) fn from_files(
        api_key: Option<&Path>,
        admin_api_key: Option<&Path>,
    ) -> Result<Self, DynamoError> {
        Ok(Self {
            api_key: api_key.map(read_key).transpose()?,
            admin_api_key: admin_api_key.map(read_key).transpose()?,
        })
    }

    pub(crate) fn generate(&self, request: RequestBuilder) -> RequestBuilder {
        authorize(request, self.api_key.as_ref())
    }

    pub(crate) fn abort(&self, request: RequestBuilder) -> RequestBuilder {
        authorize(
            request,
            self.admin_api_key.as_ref().or(self.api_key.as_ref()),
        )
    }
}

fn read_key(path: &Path) -> Result<HeaderValue, DynamoError> {
    let key = std::fs::read_to_string(path).map_err(|error| {
        client::invalid_arg(format!(
            "could not read SGLang HTTP key file {}: {error}",
            path.display()
        ))
    })?;
    let key = key.trim();
    if key.is_empty() {
        return Err(client::invalid_arg(
            "SGLang HTTP key file must not be empty",
        ));
    }
    let mut value = HeaderValue::from_str(&format!("Bearer {key}")).map_err(|_| {
        client::invalid_arg("SGLang HTTP key file contains an invalid bearer token")
    })?;
    value.set_sensitive(true);
    Ok(value)
}

fn authorize(request: RequestBuilder, value: Option<&HeaderValue>) -> RequestBuilder {
    match value {
        Some(value) => request.header(AUTHORIZATION, value.clone()),
        None => request,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_files_validate_headers_without_exposing_secrets() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("key");
        std::fs::write(&path, "test-secret\n").unwrap();
        let auth = HttpAuth::from_files(Some(&path), None).unwrap();
        assert!(!format!("{auth:?}").contains("test-secret"));
        assert!(auth.api_key.as_ref().unwrap().is_sensitive());
        for content in ["\n", "test-secret\ninjected: value"] {
            std::fs::write(&path, content).unwrap();
            let error = HttpAuth::from_files(Some(&path), None).unwrap_err();
            assert!(!error.to_string().contains("test-secret"));
        }
        assert!(HttpAuth::from_files(None, Some(&dir.path().join("missing"))).is_err());
    }
}
