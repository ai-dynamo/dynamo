/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Kubernetes quickstart image selector. It mirrors the CLI install selector's
 * support-matrix rows while emitting Kubernetes image variables and XPU build
 * commands.
 */
"use client";

import { useState } from "react";

import {
  INSTALL_DATA,
  type InstallBackend,
  type InstallChannel,
  type InstallForm,
} from "./install-selector-data";
import { LOCAL_SELECTOR_CSS } from "./local-selector-styles";
import { CURRENT_TAG } from "./releases.data";

type Hardware = "nvidia" | "intel";

type Option<T extends string> = {
  id: T;
  label: string;
  sub?: string;
};

const HARDWARES: Option<Hardware>[] = [
  { id: "nvidia", label: "NVIDIA GPU", sub: "CUDA" },
  { id: "intel", label: "Intel XPU", sub: "source build" },
];

const BACKENDS: Option<InstallBackend>[] = [
  { id: "sglang", label: "SGLang" },
  { id: "trtllm", label: "TensorRT-LLM" },
  { id: "vllm", label: "vLLM" },
];

const CHANNELS: Option<InstallChannel>[] = [
  { id: "stable", label: "Stable release", sub: "QA-validated" },
  { id: "nightly", label: "Nightly", sub: "latest features" },
  { id: "source", label: "Source build", sub: "main branch" },
];

const FORMS: Option<InstallForm>[] = [
  { id: "container", label: "Container" },
  { id: "wheel", label: "Wheel" },
];

function backendEnabled(hardware: Hardware, backend: InstallBackend): boolean {
  return hardware === "nvidia" || backend === "vllm";
}

function channelEnabled(hardware: Hardware, channel: InstallChannel): boolean {
  return hardware === "nvidia" ? channel !== "source" : channel === "source";
}

function formEnabled(form: InstallForm): boolean {
  return form === "container";
}

function disabledReason(
  kind: "backend" | "channel" | "form",
  hardware: Hardware,
  label: string,
): string {
  if (kind === "backend") return `${label} is not currently part of the Intel XPU Kubernetes quickstart path.`;
  if (kind === "channel") {
    return hardware === "intel"
      ? "Intel XPU Kubernetes images are built from source rather than published as stable or nightly artifacts."
      : "Source build is only used for the Intel XPU Kubernetes path in this quickstart.";
  }
  return "Kubernetes deployments use container images rather than wheel installs.";
}

function normalizeRegistry(registry: string): string {
  return registry.trim().replace(/\/+$/, "");
}

const REGISTRY_PATTERN = /^(?=.{1,255}$)(?:[a-z0-9]+(?:[._-][a-z0-9]+)*(?::[0-9]+)?)(?:\/[a-z0-9]+(?:[._-][a-z0-9]+)*)*$/;

function registryIsValid(registry: string): boolean {
  const normalized = normalizeRegistry(registry);
  return REGISTRY_PATTERN.test(normalized);
}

function shellQuote(value: string): string {
  return `'${value.replace(/'/g, "'\\''")}'`;
}

function runtimeVersionFor(version: string | undefined): string {
  const semver = version?.match(/^(\d+\.\d+\.\d+)/)?.[1];
  return semver ?? CURRENT_TAG;
}

function commandFor(
  hardware: Hardware,
  backend: InstallBackend,
  channel: InstallChannel,
  dynamoVersion: string | undefined,
  registry: string,
): string {
  if (hardware === "intel") {
    const imageRegistry = normalizeRegistry(registry) || "<your-registry>/<namespace>";
    return [
      `export DYNAMO_VERSION=${CURRENT_TAG}`,
      `export IMAGE_REGISTRY=${shellQuote(imageRegistry)}`,
      'export XPU_IMAGE="${IMAGE_REGISTRY}/vllm-runtime-xpu:quickstart"',
      "",
      "python3 container/render.py \\",
      "  --framework=vllm \\",
      "  --device=xpu \\",
      "  --target=runtime \\",
      "  --output-short-filename",
      'docker build --tag "$XPU_IMAGE" --file container/rendered.Dockerfile .',
      'docker push "$XPU_IMAGE"',
    ].join("\n");
  }

  if (channel === "nightly") {
    return [
      `export DYNAMO_VERSION=${CURRENT_TAG}`,
      `export DYNAMO_RUNTIME_VERSION=${runtimeVersionFor(dynamoVersion)}`,
      'export DYNAMO_IMAGE="nvcr.io/nvidia/ai-dynamo/dynamo-planner:nightly"',
    ].join("\n");
  }

  const version = dynamoVersion ?? CURRENT_TAG;
  return [
    `export DYNAMO_VERSION=${version}`,
    'export DYNAMO_IMAGE="nvcr.io/nvidia/ai-dynamo/dynamo-planner:${DYNAMO_VERSION}"',
  ].join("\n");
}

function SelectorRow<T extends string>({
  label,
  options,
  selected,
  onSelect,
  isDisabled = () => false,
  disabledTitle,
}: {
  label: string;
  options: Option<T>[];
  selected: T;
  onSelect: (value: T) => void;
  isDisabled?: (value: T) => boolean;
  disabledTitle?: (option: Option<T>) => string;
}) {
  return (
    <div className="lqs-row">
      <span className="lqs-label">{label}</span>
      <div className="lqs-options" role="group" aria-label={label}>
        {options.map((option) => {
          const disabled = isDisabled(option.id);
          return (
            <button
              key={option.id}
              type="button"
              className="lqs-chip"
              aria-pressed={selected === option.id}
              disabled={disabled}
              title={disabled ? disabledTitle?.(option) : undefined}
              onClick={() => onSelect(option.id)}
            >
              {option.label}
              {option.sub && <span className="lqs-chip-sub">{option.sub}</span>}
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function KubernetesContainerSelector() {
  const [hardware, setHardware] = useState<Hardware>("nvidia");
  const [backend, setBackend] = useState<InstallBackend>("vllm");
  const [channel, setChannel] = useState<InstallChannel>("stable");
  const [versionIndex, setVersionIndex] = useState(0);
  const [form, setForm] = useState<InstallForm>("container");
  const [registry, setRegistry] = useState("");
  const [copyLabel, setCopyLabel] = useState("Copy");

  const entries = INSTALL_DATA[backend][channel].filter((candidate) => candidate.commands.container);
  const entry = entries[versionIndex] ?? entries[0];
  const command = commandFor(hardware, backend, channel, entry?.dynamo, registry);
  const hardwareLabel = hardware === "nvidia" ? "NVIDIA GPU" : "Intel XPU";
  const registryNeeded = hardware === "intel";
  const canCopy = !registryNeeded || registryIsValid(registry);
  const badge = channel === "stable" ? "Stable" : channel === "nightly" ? "Nightly" : "Source";
  const title = channel === "stable"
    ? `Dynamo ${entry?.dynamo}`
    : channel === "nightly"
      ? entry?.latest
        ? "Latest nightly"
        : `Nightly ${entry?.dynamo}`
      : "Build Dynamo XPU image";
  const role = channel === "stable"
    ? "Latest stable release that supports this version"
    : channel === "nightly"
      ? entry?.latest
        ? "Latest nightly container"
        : "Pinned nightly build"
      : "Intel XPU Kubernetes runtime";
  const versionRowLabel = `${INSTALL_DATA[backend].label} version`;

  function chooseHardware(next: Hardware) {
    setHardware(next);
    setVersionIndex(0);
    setForm("container");
    if (next === "intel") {
      setBackend("vllm");
      setChannel("source");
    } else {
      setChannel("stable");
    }
  }

  function chooseBackend(next: InstallBackend) {
    setBackend(next);
    setVersionIndex(0);
    setForm("container");
  }

  function chooseChannel(next: InstallChannel) {
    setChannel(next);
    setVersionIndex(0);
    setForm("container");
  }

  function resetCopyLabel(label = "Copy") {
    window.setTimeout(() => setCopyLabel(label), 1200);
  }

  async function copyCommand() {
    if (!canCopy) return;
    if (!navigator.clipboard?.writeText) {
      setCopyLabel("Copy failed");
      resetCopyLabel();
      return;
    }
    try {
      await navigator.clipboard.writeText(command);
      setCopyLabel("Copied!");
    } catch {
      setCopyLabel("Copy failed");
    }
    resetCopyLabel();
  }

  return (
    <>
      <style>{LOCAL_SELECTOR_CSS}</style>
      <section className="lqs-panel" aria-label="Kubernetes container image selector">
        <div className="lqs-head">
          <h3>Choose your Kubernetes build</h3>
          <p>Unavailable combinations remain visible to show the current support boundary.</p>
        </div>

        <SelectorRow label="Hardware" options={HARDWARES} selected={hardware} onSelect={chooseHardware} />
        <SelectorRow
          label="Backend"
          options={BACKENDS}
          selected={backend}
          onSelect={chooseBackend}
          isDisabled={(value) => !backendEnabled(hardware, value)}
          disabledTitle={(option) => disabledReason("backend", hardware, option.label)}
        />
        <SelectorRow
          label="Dynamo build"
          options={CHANNELS}
          selected={channel}
          onSelect={chooseChannel}
          isDisabled={(value) => !channelEnabled(hardware, value)}
          disabledTitle={(option) => disabledReason("channel", hardware, option.label)}
        />
        {entry && (
          <div className="lqs-row">
            <span className="lqs-label">{versionRowLabel}</span>
            <div className="lqs-options" role="group" aria-label={versionRowLabel}>
              {entries.map((version, index) => {
                const displayVersion = version.backend_version;
                const displayMeta = version.source
                  ? "from main"
                  : channel === "nightly"
                    ? version.latest
                      ? "latest nightly"
                      : version.date ?? "nightly"
                    : `Dynamo ${version.dynamo}`;
                return (
                  <button
                    key={`${version.backend_version}-${version.dynamo ?? index}`}
                    type="button"
                    className="lqs-chip"
                    aria-pressed={versionIndex === index}
                    onClick={() => {
                      setVersionIndex(index);
                      setForm("container");
                    }}
                  >
                    {displayVersion}
                    <span className="lqs-chip-sub">{displayMeta}</span>
                  </button>
                );
              })}
            </div>
          </div>
        )}
        <SelectorRow
          label="Install form"
          options={FORMS}
          selected={form}
          onSelect={setForm}
          isDisabled={(value) => !formEnabled(value)}
          disabledTitle={(option) => disabledReason("form", hardware, option.label)}
        />
        {registryNeeded && (
          <div className="lqs-row">
            <span className="lqs-label">Registry</span>
            <div className="lqs-field">
              <input
                className="lqs-input"
                value={registry}
                onChange={(event) => setRegistry(event.target.value)}
                placeholder="registry.example.com/my-team"
                aria-invalid={!canCopy}
              />
              <p className="lqs-hint">Use the registry/namespace where you can push the custom image.</p>
              {!canCopy && <p className="lqs-hint lqs-hint--error">Enter a lowercase Docker registry/namespace to enable copy.</p>}
            </div>
          </div>
        )}

        <div className="lqs-output">
          <div className={`lqs-rec lqs-rec--${channel}`}>
            <div className="lqs-eyebrow">{role}</div>
            <div className="lqs-title">
              <span className="lqs-badge">{badge}</span>
              {title}
            </div>
            <div className="lqs-support">
              {hardwareLabel} · {INSTALL_DATA[backend].label} {entry?.backend_version}
            </div>
          </div>
          <div className="lqs-command">
            {canCopy && (
              <button type="button" className="lqs-copy" onClick={copyCommand}>
                {copyLabel}
              </button>
            )}
            <pre>{command}</pre>
          </div>
        </div>
      </section>
    </>
  );
}

export default KubernetesContainerSelector;
