/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Recent pinned nightly wheel builds plus the rolling nightly runtime
 * container tags. Version data lives in releases.data.ts so the human page,
 * install selector, JSON, and llms-only tables stay aligned.
 */

import { NIGHTLY_BUILDS, type NightlyBuild } from "./releases.data";

const NIGHTLY_CSS = `
.dynref-nightly-row {
    display: grid;
    grid-template-columns: 170px minmax(0, 1fr) max-content;
    gap: 10px;
    align-items: center;
    padding: 11px 0;
    border-bottom: 1px solid var(--border, var(--grayscale-a5));
}

.dynref-nightly-row:last-child {
    border-bottom: 0;
}

.dynref-nightly-version {
    color: var(--pst-color-text-base);
    font-size: 13px;
    font-weight: 700;
    overflow-wrap: anywhere;
}

.dynref-nightly-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}

.dynref-nightly-actions {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: 6px;
}

.dynref-nightly-note {
    margin: 4px 0 0;
}

@media (max-width: 720px) {
    .dynref-nightly-row {
        grid-template-columns: minmax(0, 1fr);
        gap: 7px;
    }

    .dynref-nightly-actions {
        justify-content: flex-start;
    }
}
`;

const RUNTIME_CONTAINERS = [
  {
    label: "SGLang",
    clipboard: "docker pull nvcr.io/nvidia/ai-dynamo/sglang-runtime-nightly:latest",
  },
  {
    label: "TensorRT-LLM",
    clipboard: "docker pull nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime-nightly:latest",
  },
  {
    label: "vLLM",
    clipboard: "docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly:latest",
  },
];

function wheelInstall(version: string, extra: "sglang" | "vllm"): string {
  return `uv pip install --pre --extra-index-url https://pypi.nvidia.com/ "ai-dynamo[${extra}]==${version}"`;
}

function NightlyRow({ build }: { build: NightlyBuild }) {
  return (
    <div className="dynref-nightly-row">
      <div>
        <div className="dynref-mono dynref-nightly-version">{build.version}</div>
        <p className="dynref-muted dynref-nightly-note">{build.date}</p>
      </div>
      <div>
        <div className="dynref-nightly-tags">
          {build.packages.map((pkg) => (
            <span key={pkg} className="dynref-badge dynref-badge--blue">
              {pkg}
            </span>
          ))}
        </div>
        {build.note && <p className="dynref-muted dynref-nightly-note">{build.note}</p>}
      </div>
      <div className="dynref-nightly-actions">
        <button
          className="dynref-copy dynref-badge dynref-badge--blue"
          type="button"
          data-dynref-copy={wheelInstall(build.version, "sglang")}
          title={wheelInstall(build.version, "sglang")}
        >
          SGLang wheel
        </button>
        <button
          className="dynref-copy dynref-badge dynref-badge--blue"
          type="button"
          data-dynref-copy={wheelInstall(build.version, "vllm")}
          title={wheelInstall(build.version, "vllm")}
        >
          vLLM wheel
        </button>
      </div>
    </div>
  );
}

export function NightlyBuilds() {
  return (
    <>
      <style>{NIGHTLY_CSS}</style>
      <section className="dynref-panel">
        <div className="dynref-panel-header">
          <div>
            <p className="dynref-eyebrow">Nightly builds</p>
            <h3 className="dynref-h">Recent pinned wheel builds</h3>
          </div>
          <div className="dynref-nightly-actions">
            {RUNTIME_CONTAINERS.map((container) => (
              <button
                key={container.label}
                className="dynref-copy dynref-badge dynref-badge--amber"
                type="button"
                data-dynref-copy={container.clipboard}
                title={container.clipboard}
              >
                {container.label} latest
              </button>
            ))}
          </div>
        </div>
        <p className="dynref-muted">
          Nightly wheels are pinned by date. Runtime containers use rolling NGC tags, so the container buttons always pull the latest nightly image.
        </p>
        <div>
          {NIGHTLY_BUILDS.map((build) => (
            <NightlyRow key={build.version} build={build} />
          ))}
        </div>
      </section>
    </>
  );
}

export default NightlyBuilds;
