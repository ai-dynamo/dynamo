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
 */

import * as fs from "node:fs";
import * as path from "node:path";

export interface GlobalRouterDeployment {
  readonly image: string;
  readonly config: Record<string, unknown>;
}

export function loadGlobalRouterDeployment(
  imageValue: unknown,
  configPathValue: unknown,
  cwd = process.cwd(),
): GlobalRouterDeployment | undefined {
  if (imageValue === undefined && configPathValue === undefined) {
    return undefined;
  }
  if (typeof imageValue !== "string" || imageValue.trim() === "") {
    throw new Error("-c routerImage must provide an explicit image URI and tag.");
  }
  if (typeof configPathValue !== "string" || configPathValue.trim() === "") {
    throw new Error("-c routerConfigPath must name a JSON config file.");
  }
  if (!hasExplicitImageVersion(imageValue)) {
    throw new Error("routerImage must include an explicit tag or sha256 digest.");
  }
  if (imageValue.endsWith(":latest")) {
    throw new Error("routerImage must not use the implicit latest tag.");
  }

  const configPath = path.resolve(cwd, configPathValue);
  let config: unknown;
  try {
    config = JSON.parse(fs.readFileSync(configPath, "utf8"));
  } catch (error) {
    throw new Error(`Could not read Global Router JSON config at ${configPath}.`, {
      cause: error,
    });
  }
  if (config === null || typeof config !== "object" || Array.isArray(config)) {
    throw new Error("Global Router config must be a JSON object.");
  }
  if ((config as Record<string, unknown>).listen !== "0.0.0.0:8080") {
    throw new Error("Global Router config listen must be 0.0.0.0:8080.");
  }
  return { image: imageValue, config: config as Record<string, unknown> };
}


function hasExplicitImageVersion(image: string): boolean {
  const digestSeparator = image.lastIndexOf("@");
  if (digestSeparator >= 0) {
    return /^sha256:[a-fA-F0-9]{64}$/.test(image.slice(digestSeparator + 1));
  }
  const finalPathSegment = image.slice(image.lastIndexOf("/") + 1);
  const tagSeparator = finalPathSegment.lastIndexOf(":");
  if (tagSeparator <= 0) {
    return false;
  }
  const tag = finalPathSegment.slice(tagSeparator + 1);
  return /^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$/.test(tag);
}
