// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Based on https://github.com/64bit/async-openai/ by Himanshu Neema
// Original Copyright (c) 2022 Himanshu Neema
// Licensed under MIT License (see ATTRIBUTIONS-Rust.md)
//
// Modifications Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
// Licensed under Apache 2.0

use super::{
    AssistantToolCodeInterpreterResources, AssistantToolFileSearchResources,
    AssistantToolResources, AssistantTools, AssistantToolsFileSearch, AssistantToolsFunction,
    CreateAssistantToolFileSearchResources, CreateAssistantToolResources, FunctionObject,
};

impl From<AssistantToolsFileSearch> for AssistantTools {
    fn from(value: AssistantToolsFileSearch) -> Self {
        Self::FileSearch(value)
    }
}

impl From<AssistantToolsFunction> for AssistantTools {
    fn from(value: AssistantToolsFunction) -> Self {
        Self::Function(value)
    }
}

impl From<FunctionObject> for AssistantToolsFunction {
    fn from(value: FunctionObject) -> Self {
        Self { function: value }
    }
}

impl From<FunctionObject> for AssistantTools {
    fn from(value: FunctionObject) -> Self {
        Self::Function(value.into())
    }
}

impl From<CreateAssistantToolFileSearchResources> for CreateAssistantToolResources {
    fn from(value: CreateAssistantToolFileSearchResources) -> Self {
        Self {
            code_interpreter: None,
            file_search: Some(value),
        }
    }
}

impl From<AssistantToolCodeInterpreterResources> for CreateAssistantToolResources {
    fn from(value: AssistantToolCodeInterpreterResources) -> Self {
        Self {
            code_interpreter: Some(value),
            file_search: None,
        }
    }
}

impl From<AssistantToolCodeInterpreterResources> for AssistantToolResources {
    fn from(value: AssistantToolCodeInterpreterResources) -> Self {
        Self {
            code_interpreter: Some(value),
            file_search: None,
        }
    }
}

impl From<AssistantToolFileSearchResources> for AssistantToolResources {
    fn from(value: AssistantToolFileSearchResources) -> Self {
        Self {
            code_interpreter: None,
            file_search: Some(value),
        }
    }
}
