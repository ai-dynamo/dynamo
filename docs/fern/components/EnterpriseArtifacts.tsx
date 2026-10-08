/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
