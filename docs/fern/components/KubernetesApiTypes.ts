/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

export type KubernetesTypeKind = "resource" | "type" | "enum";

export interface KubernetesTypeRef {
    name: string;
    anchor: string;
}

export interface KubernetesField {
    name: string;
    type: string;
    default: string;
    required: boolean;
    description: string;
    validation: string;
}

export interface KubernetesEnumValue {
    name: string;
    description: string;
}

export interface KubernetesType {
    name: string;
    displayName: string;
    anchor: string;
    kind: KubernetesTypeKind;
    description: string;
    underlyingType: string;
    validation: string;
    appearsIn: KubernetesTypeRef[];
    fields: KubernetesField[];
    enumValues: KubernetesEnumValue[];
}

export interface KubernetesPackage {
    name: string;
    anchor: string;
    description: string;
    resourceTypes: KubernetesTypeRef[];
    types: KubernetesType[];
}

export interface KubernetesOperatorDefaultsSubsection {
    title: string;
    anchor: string;
    bodyMarkdown: string;
}

export interface KubernetesOperatorDefaults {
    introMarkdown: string;
    subsections: KubernetesOperatorDefaultsSubsection[];
}

export interface KubernetesReference {
    sourceHref: string;
    packages: KubernetesPackage[];
    operatorDefaults: KubernetesOperatorDefaults;
}
