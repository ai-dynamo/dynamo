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
 * EnterpriseArtifacts — the supported release line and its -enterprise
 * artifact table, for the newest RELEASES entry that carries an `enterprise`
 * date. Versioned docs snapshots import this shared component, so every
 * published version shows the current supported set without a snapshot patch.
 */

import { ENTERPRISE_ARTIFACTS, RELEASES } from "./releases.data";

export function EnterpriseArtifacts() {
  const release = RELEASES.find((r) => r.enterprise);
  if (!release) return null;
  const tag = release.version.replace(/^v/, "");
  const line = tag.split(".").slice(0, 2).join(".");

  return (
    <>
      <p>
        The Dynamo {line} release line is the current supported release. The following artifacts are published for
        it. Commercial support is limited to this exact set.
      </p>
      <table>
        <thead>
          <tr>
            <th>Component</th>
            <th>Artifact</th>
          </tr>
        </thead>
        <tbody>
          {ENTERPRISE_ARTIFACTS.map((artifact) => (
            <tr key={artifact.component}>
              <td>{artifact.component}</td>
              <td>
                <code>{artifact.ref.replace("{tag}", tag)}</code>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}
