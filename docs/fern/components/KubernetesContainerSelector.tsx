/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Kubernetes quickstart image selector. It keeps the quickstart's release,
 * custom-build, and XPU paths in one copyable block.
 */
"use client";

import { useState } from "react";

import { CURRENT_TAG } from "./releases.data";
import { LOCAL_SELECTOR_CSS } from "./local-selector-styles";

type Hardware = "nvidia" | "intel";
type Build = "release" | "custom";

type Option<T extends string> = {
  id: T;
  label: string;
  sub?: string;
};

const HARDWARES: Option<Hardware>[] = [
  { id: "nvidia", label: "NVIDIA GPU", sub: "published images" },
  { id: "intel", label: "Intel XPU", sub: "source build" },
];

const BUILDS: Option<Build>[] = [
  { id: "release", label: "Dynamo release image", sub: CURRENT_TAG },
  { id: "custom", label: "Custom XPU image", sub: "build and push" },
];

function buildEnabled(hardware: Hardware, build: Build): boolean {
  return hardware === "nvidia" ? build === "release" : build === "custom";
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

function commandFor(hardware: Hardware, registry: string): string {
  if (hardware === "intel") {
    const imageRegistry = normalizeRegistry(registry) || "<your-registry>/<namespace>";
    return [
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

  return [
    `export DYNAMO_VERSION=${CURRENT_TAG}`,
    'export DYNAMO_IMAGE="nvcr.io/nvidia/ai-dynamo/dynamo-planner:${DYNAMO_VERSION}"',
  ].join("\n");
}

function ChoiceRow<T extends string>({
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
  const [build, setBuild] = useState<Build>("release");
  const [registry, setRegistry] = useState("");
  const [copyLabel, setCopyLabel] = useState("Copy");

  const command = commandFor(hardware, registry);
  const hardwareLabel = hardware === "nvidia" ? "NVIDIA GPU" : "Intel XPU";
  const buildLabel = hardware === "intel"
    ? "Custom XPU runtime"
    : "Published Dynamo image";
  const registryNeeded = hardware === "intel";
  const canCopy = !registryNeeded || registryIsValid(registry);

  function chooseHardware(next: Hardware) {
    setHardware(next);
    if (next === "intel") {
      setBuild("custom");
    } else {
      setBuild("release");
    }
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
          <h3>Choose your Kubernetes image path</h3>
          <p>Use the variables from this block in the deployment steps below.</p>
        </div>

        <ChoiceRow label="Hardware" options={HARDWARES} selected={hardware} onSelect={chooseHardware} />
        <ChoiceRow
          label="Container"
          options={BUILDS}
          selected={build}
          onSelect={setBuild}
          isDisabled={(value) => !buildEnabled(hardware, value)}
          disabledTitle={(option) => `${option.label} is not used by this quickstart path.`}
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
          <div className={`lqs-rec lqs-rec--${build === "release" ? "stable" : "source"}`}>
            <div className="lqs-eyebrow">Kubernetes quickstart</div>
            <div className="lqs-title">
              <span className="lqs-badge">{build === "release" ? "Use" : "Build"}</span>
              {buildLabel}
            </div>
            <div className="lqs-support">{hardwareLabel} / {hardware === "intel" ? "vLLM XPU runtime image" : "DGDR planner image"}</div>
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
