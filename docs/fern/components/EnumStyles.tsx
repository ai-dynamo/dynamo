/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Reference-page enum badges. Render with the page because the NVIDIA global
 * theme replaces docs.yml CSS and footer configuration in hosted builds.
 */

const ENUM_CSS = `
.enum-values {
    display: inline-flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0.5rem;
}
.enum-values .enum-label {
    flex-shrink: 0;
    color: var(--grayscale-a11);
    font-size: var(--text-sm);
}
.enum-values .fern-docs-badge {
    border-radius: var(--radius-1) !important;
    height: 1.25rem;
    padding: 0 0.375rem;
    font-size: var(--text-xs);
    font-weight: 500;
    background-color: var(--grayscale-a3) !important;
    color: var(--grayscale-a11) !important;
    text-transform: none;
}
`;

export function EnumStyles() {
  return <style dangerouslySetInnerHTML={{ __html: ENUM_CSS }} />;
}
