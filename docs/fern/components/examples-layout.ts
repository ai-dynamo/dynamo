/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Keep Fern's content-width cap. Without a TOC, its mx-auto article otherwise
// centers in the space freed by the TOC and shifts the entire page to the right.
export const EXAMPLES_LAYOUT_CSS = `
main.fern-main:has(.dynamo-examples, .dynamo-example-viewer, .dynamo-diffusion):not(:has(> .fern-layout-content-wrapper ~ aside)) .fern-layout-guide > article {
  margin-left: 0 !important;
  margin-right: auto !important;
}
.dynamo-examples, .dynamo-example-viewer, .dynamo-diffusion {
  width: 100%;
  max-width: 100%;
  min-width: 0;
  margin-inline: 0;
}
`;
