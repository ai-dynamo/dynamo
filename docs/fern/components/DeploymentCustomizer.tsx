/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tailors a Kubernetes example manifest in the browser: model, image registry
 * and tag, replicas and GPUs, environment variables, model caching, and the
 * Hugging Face token Secret. The output is a highlighted, copyable manifest
 * plus generated prerequisites and deploy steps.
 *
 * The source stays a native MDX Code embed. This component reads the embedded
 * file from the CodeBlock element Fern renders for it, so Fern still resolves
 * the file for every docs version and the page's Markdown twin keeps the
 * original source. When the manifest cannot be read or customized, the native
 * code block renders instead.
 */
"use client";

import {
  Children,
  Component,
  cloneElement,
  createContext,
  isValidElement,
  useContext,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactElement,
  type ReactNode,
} from "react";

import {
  analyzeManifest,
  customizeManifest,
  diffLines,
  hasPlaceholderImages,
  highlightBash,
  highlightYaml,
  initialEnvRows,
  isModelId,
  isTaggableImage,
  rewriteImage,
  splitLines,
  type ComponentOverride,
  type CustomizeResult,
  type CustomizeSettings,
  type EnvRow,
  type ImageSettings,
  type ManifestInfo,
  type OutputFile,
  type SecretMode,
  type StorageMode,
  type StorageSettings,
  type Token,
} from "./deployment-customizer";
import {
  DEFAULT_MOUNT_PATH,
  DEFAULT_NAMESPACE,
  DEFAULT_REGISTRY,
  IMAGE_TAGS,
  LATEST_TAG,
  MODEL_CATEGORIES,
  MODEL_OPTIONS,
  PUBLISHED_REPOS,
  modelCategory,
  type ModelCategory,
  type ModelOption,
} from "./deployment-customizer-data";
import { DEPLOYMENT_CUSTOMIZER_CSS } from "./DeploymentCustomizerStyles";
import { LOCAL_SELECTOR_CSS } from "./local-selector-styles";

// ---------------------------------------------------------------------------
// Settings shared by every manifest under one provider
// ---------------------------------------------------------------------------

interface SharedSettings {
  model: string;
  registry: string;
  tag: string | null;
  customTag: string;
  namespace: string;
  storage: StorageMode | null;
  pvcName: string;
  size: string;
  storageClass: string;
  accessMode: StorageSettings["accessMode"];
  mountPath: string;
  createPvc: boolean;
  downloadJob: boolean;
  secretMode: SecretMode;
  secretName: string;
  bundle: boolean;
}

const DEFAULT_SETTINGS: SharedSettings = {
  model: "",
  registry: "",
  tag: null,
  customTag: "",
  namespace: DEFAULT_NAMESPACE,
  storage: null,
  pvcName: "",
  size: "100Gi",
  storageClass: "",
  accessMode: "ReadWriteMany",
  mountPath: DEFAULT_MOUNT_PATH,
  createPvc: true,
  downloadJob: true,
  secretMode: "kubectl",
  secretName: "",
  bundle: false,
};

interface SettingsState {
  settings: SharedSettings;
  update: (patch: Partial<SharedSettings>) => void;
  reset: () => void;
  pageTag: string | null;
  /** Open state of each disclosure by key; a missing key uses the section's default. */
  disclosures: Record<string, boolean>;
  setOpen: (key: string, open: boolean) => void;
}

const SettingsContext = createContext<SettingsState | null>(null);

/** The release tag of a versioned docs URL such as /dynamo/v1.4.0/..., when it is a known tag. */
function usePageTag(): string | null {
  const [tag, setTag] = useState<string | null>(null);
  useEffect(() => {
    const match = window.location.pathname.match(/\/v(\d+\.\d+\.\d+)(?=\/|$)/);
    if (match && IMAGE_TAGS.some((option) => option.tag === match[1])) setTag(match[1]);
  }, []);
  return tag;
}

function useSettingsState(): SettingsState {
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [disclosures, setDisclosures] = useState<Record<string, boolean>>({});
  const pageTag = usePageTag();
  return {
    settings,
    update: (patch) => setSettings((current) => ({ ...current, ...patch })),
    reset: () => setSettings(DEFAULT_SETTINGS),
    pageTag,
    disclosures,
    setOpen: (key, open) => setDisclosures((current) => ({ ...current, [key]: open })),
  };
}

/**
 * Shares model, image, storage, and Secret choices, and which sections are
 * open, across the manifests it wraps.
 */
export function DeploymentCustomizerProvider({ children }: { children: ReactNode }) {
  const state = useSettingsState();
  return (
    <SettingsContext.Provider value={state}>
      <style dangerouslySetInnerHTML={{ __html: DEPLOYMENT_CUSTOMIZER_CSS }} />
      {children}
    </SettingsContext.Provider>
  );
}

// ---------------------------------------------------------------------------
// Reading the manifest from Fern's rendered code block
// ---------------------------------------------------------------------------

interface EmbeddedCode {
  code: string;
  title?: string;
}

function embeddedYaml(element: ReactElement): EmbeddedCode | null {
  const props = element.props as Record<string, unknown>;
  if (typeof props.code !== "string") return null;
  const language =
    typeof props.language === "string"
      ? props.language
      : typeof props.className === "string"
        ? props.className.replace(/^language-/, "")
        : "";
  const title = typeof props.title === "string" ? props.title : undefined;
  if (!/^ya?ml$/i.test(language) && !/\.ya?ml$/i.test(title ?? "")) return null;
  return { code: props.code, title };
}

function findEmbeddedYaml(node: ReactNode): EmbeddedCode | null {
  let found: EmbeddedCode | null = null;
  Children.forEach(node, (child) => {
    if (found || !isValidElement(child)) return;
    found = embeddedYaml(child) ?? findEmbeddedYaml((child.props as { children?: ReactNode }).children);
  });
  return found;
}

/** A page's "[Source: `file`](url)" link, shown in the customizer header. */
export interface SourceLink {
  href: string;
  label: string;
  /** The page's original paragraph, rendered when the customizer falls back. */
  element: ReactElement;
}

function isBlank(node: ReactNode): boolean {
  return node === null || node === undefined || typeof node === "boolean" || (typeof node === "string" && node.trim() === "");
}

function textOf(node: ReactNode): string {
  let text = "";
  Children.forEach(node, (child) => {
    if (typeof child === "string" || typeof child === "number") text += child;
    else if (isValidElement(child)) text += textOf((child.props as { children?: ReactNode }).children);
  });
  return text;
}

function codeTextOf(node: ReactNode): string | null {
  let found: string | null = null;
  Children.forEach(node, (child) => {
    if (found !== null || !isValidElement(child)) return;
    found = child.type === "code" ? textOf(child) : codeTextOf((child.props as { children?: ReactNode }).children);
  });
  return found;
}

/**
 * Reads a paragraph that holds only a link to the file embedded next. Matching
 * on the file name rather than the "Source:" text keeps other links in place.
 */
function sourceLinkFor(element: ReactElement, fileName: string | undefined): SourceLink | null {
  if (!fileName) return null;
  const content = Children.toArray((element.props as { children?: ReactNode }).children).filter((child) => !isBlank(child));
  const anchor = content[0];
  if (content.length !== 1 || !isValidElement(anchor)) return null;
  const href = (anchor.props as { href?: unknown }).href;
  if (typeof href !== "string" || !/^https?:\/\//.test(href)) return null;
  const path = href.split(/[?#]/)[0];
  if (!path.endsWith(`/${fileName}`)) return null;
  return { href, label: codeTextOf(anchor) ?? path.slice(path.lastIndexOf("/") + 1), element };
}

/**
 * Replaces each embedded YAML code block in `node` with a DeploymentCustomizer
 * that falls back to the original block. A source-link paragraph directly
 * before the block moves into the customizer's header.
 */
export function withDeploymentCustomizer(node: ReactNode, link: SourceLink | null = null): ReactNode {
  const items = Children.toArray(node);
  const out: ReactNode[] = [];
  let pending = link;
  items.forEach((child, index) => {
    if (!isValidElement(child)) {
      out.push(child);
      return;
    }
    const embedded = embeddedYaml(child);
    if (embedded) {
      out.push(
        <DeploymentCustomizer key={child.key} source={embedded.code} fileName={embedded.title} fallback={child} link={pending} />,
      );
      pending = null;
      return;
    }
    const next = items.slice(index + 1).find((item) => !isBlank(item));
    const nextYaml = isValidElement(next) ? findEmbeddedYaml(next) : null;
    const source = nextYaml ? sourceLinkFor(child, nextYaml.title) : null;
    if (source) {
      pending = source;
      return;
    }
    const children = (child.props as { children?: ReactNode }).children;
    if (children === undefined || typeof children === "string") {
      out.push(child);
      return;
    }
    const carried = pending && findEmbeddedYaml(child) ? pending : null;
    if (carried) pending = null;
    out.push(cloneElement(child, undefined, withDeploymentCustomizer(children, carried)));
  });
  return out;
}

class FallbackBoundary extends Component<{ fallback: ReactNode; children: ReactNode }, { failed: boolean }> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  render() {
    return this.state.failed ? this.props.fallback : this.props.children;
  }
}

function attempt<T>(run: () => T): T | null {
  try {
    return run();
  } catch {
    return null;
  }
}

export function DeploymentCustomizer({
  source,
  fileName,
  fallback,
  link,
  children,
}: {
  source?: string;
  fileName?: string;
  fallback?: ReactNode;
  link?: SourceLink | null;
  children?: ReactNode;
}) {
  // Wrapping native embeds: customize each YAML block and keep the rest as written.
  if (source === undefined) return <>{withDeploymentCustomizer(children)}</>;
  const code = fallback ?? children ?? null;
  // The page's original content: without the customizer header, the source link renders again.
  const native = link ? (
    <>
      {link.element}
      {code}
    </>
  ) : (
    code
  );
  return (
    <FallbackBoundary fallback={native}>
      <Customizer source={source} fileName={fileName ?? "deployment.yaml"} link={link ?? null} code={code} native={native} />
    </FallbackBoundary>
  );
}

export default DeploymentCustomizer;

// ---------------------------------------------------------------------------
// Customizer
// ---------------------------------------------------------------------------

const REGISTRY_PATTERN = /^(?=.{1,255}$)(?:[a-z0-9]+(?:[._-][a-z0-9]+)*(?::[0-9]+)?)(?:\/[a-z0-9]+(?:[._-][a-z0-9]+)*)*$/;
const TAG_PATTERN = /^[\w][\w.-]{0,127}$/;
const DNS_NAME = /^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$/;
const DNS_LABEL = /^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$/;
const QUANTITY = /^\d+(\.\d+)?(Ki|Mi|Gi|Ti|Pi|Ei|k|M|G|T|P|E)?$/;
const ENV_NAME = /^[-._a-zA-Z][-._a-zA-Z0-9]*$/;
/** Scale tables up to this many components start open; longer ones start as their summary. */
const OPEN_SCALE_ROWS = 4;

function Customizer({
  source,
  fileName,
  link,
  code,
  native,
}: {
  source: string;
  fileName: string;
  link: SourceLink | null;
  /** The native code block alone, shown under the panel when settings cannot be applied. */
  code: ReactNode;
  /** The page's original content, shown instead of the customizer. */
  native: ReactNode;
}) {
  const id = useId();
  const shared = useContext(SettingsContext);
  const own = useSettingsState();
  const { settings, update, reset, pageTag, disclosures, setOpen } = shared ?? own;
  const info = useMemo(() => attempt(() => analyzeManifest(source)), [source]);
  const initialEnv = useMemo(() => (info ? initialEnvRows(info) : []), [info]);

  const [names, setNames] = useState<Record<number, string>>({});
  const [components, setComponents] = useState<Record<string, ComponentOverride>>({});
  const [env, setEnv] = useState<EnvRow[]>(initialEnv);
  const [templateVars, setTemplateVars] = useState<Record<string, string>>({});

  const placeholders = info ? hasPlaceholderImages(info.images, DEFAULT_REGISTRY) : false;
  const tagChoice = settings.tag ?? (placeholders ? pageTag ?? LATEST_TAG : "keep");
  const customTag = settings.customTag.trim();
  const customTagValid = TAG_PATTERN.test(customTag);
  const registry = settings.registry.trim().replace(/\/+$/, "");
  const registryValid = registry === "" || REGISTRY_PATTERN.test(registry);
  const image: ImageSettings = {
    defaultRegistry: DEFAULT_REGISTRY,
    publishedRepos: PUBLISHED_REPOS,
    registry: registry && registryValid ? registry : null,
    tagMode: tagChoice === "keep" || (tagChoice === "custom" && !customTagValid) ? "keep" : tagChoice === "custom" ? "custom" : "release",
    tag: tagChoice === "custom" ? customTag : tagChoice,
  };

  const existingPvc = info?.modelPvc ?? null;
  const storageMode: StorageMode = settings.storage ?? (existingPvc ? "pvc" : "download");
  const pvcName = settings.pvcName.trim() || existingPvc?.claim || "model-cache";
  const pvcNameValid = DNS_NAME.test(pvcName);
  const size = settings.size.trim();
  const sizeValid = QUANTITY.test(size);
  const mountPath = existingPvc?.mountPath ?? (settings.mountPath.trim() || DEFAULT_MOUNT_PATH);
  const secretName = settings.secretName.trim() || info?.secret || "hf-token-secret";
  const secretNameValid = DNS_NAME.test(secretName);
  const model = settings.model.trim();
  const originalModel = info?.models[0] ?? null;
  const category = modelCategory(model) ?? modelCategory(originalModel);
  const chatModel = category === "text" || category === "vision";

  const customize: CustomizeSettings | null = info
    ? {
        fileName,
        model: model || null,
        image,
        names,
        components,
        env,
        storage: {
          mode: storageMode,
          pvcName: pvcNameValid ? pvcName : existingPvc?.claim || "model-cache",
          size: sizeValid ? size : "100Gi",
          storageClass: settings.storageClass.trim(),
          accessMode: settings.accessMode,
          mountPath,
          createPvc: settings.createPvc,
          downloadJob: settings.downloadJob,
        },
        secret: { mode: settings.secretMode, name: secretNameValid ? secretName : info.secret || "hf-token-secret" },
        templateVars,
        namespace: settings.namespace,
        bundle: settings.bundle,
        chatModel,
      }
    : null;
  const key = JSON.stringify(customize);
  const result: CustomizeResult | null = useMemo(
    () => (info && customize ? attempt(() => customizeManifest(source, info, customize)) : null),
    // `key` serializes every setting that feeds the transform.
    [source, info, key],
  );

  if (!info || !customize) return <>{native}</>;

  function resetAll() {
    reset();
    setNames({});
    setComponents({});
    setEnv(initialEnv);
    setTemplateVars({});
  }

  function setOverride(componentKey: string, patch: ComponentOverride) {
    setComponents((current) => ({ ...current, [componentKey]: { ...current[componentKey], ...patch } }));
  }

  const taggable = info.images.filter((ref) => isTaggableImage(ref, DEFAULT_REGISTRY));
  const resolvedImages = [...new Set(taggable.map((ref) => rewriteImage(ref, image)))];
  const dgds = info.graphs.filter((graph) => graph.kind === "DynamoGraphDeployment");
  const multiGraph = dgds.length > 1;
  const hasNodes = info.components.some((component) => component.nodes !== null);
  const totals = info.components.reduce(
    (sum, component) => {
      const override = components[component.key] ?? {};
      const replicas = override.replicas ?? component.replicas ?? 1;
      const nodes = override.nodes ?? component.nodes ?? 1;
      const gpus = override.gpus ?? component.gpus ?? 0;
      return { pods: sum.pods + replicas * nodes, gpus: sum.gpus + replicas * nodes * gpus };
    },
    { pods: 0, gpus: 0 },
  );
  const tpSynced = info.components.some(
    (component) => component.tp && component.gpus !== null && component.tp.value === component.gpus * (component.nodes ?? 1),
  );
  const selectedModel = MODEL_OPTIONS.find((option) => option.id === (model || originalModel));
  const claimInManifest = info.claims.includes(existingPvc?.claim ?? pvcName);
  const jobModel = originalModel
    ? model || originalModel
    : ["MODEL_NAME", "MODEL", "MODEL_ID"].map((name) => templateVars[name]?.trim()).find(Boolean) || null;
  const showSecret = info.secret !== null || (storageMode === "pvc" && settings.downloadJob && !!jobModel);
  const uiNotes: string[] = [];
  if (selectedModel?.gated && settings.secretMode === "none") {
    uiNotes.push(`${selectedModel.id} is gated on Hugging Face and needs a token Secret.`);
  }
  if (tagChoice === "custom" && !customTagValid) uiNotes.push("Enter a valid image tag to apply it.");

  // Open states are shared, so a section stays open or closed across examples.
  const isOpen = (key: string, fallback: boolean) => disclosures[key] ?? fallback;
  const disclosure = (key: string, fallback: boolean, summary: ReactNode): Disclosure => ({
    open: isOpen(key, fallback),
    onToggle: (open) => setOpen(key, open),
    summary,
  });
  const panelOpen = isOpen("panel", true);
  const moreOpen = isOpen("more", false);

  const scaleChanged = info.components.some((component) => {
    const override = components[component.key] ?? {};
    return (
      (override.replicas ?? component.replicas ?? 1) !== (component.replicas ?? 1) ||
      (override.gpus ?? component.gpus) !== component.gpus ||
      (override.nodes ?? component.nodes) !== component.nodes
    );
  });
  const scaleSummary = `${plural(totals.gpus, "GPU")} across ${plural(totals.pods, "pod")}${scaleChanged ? " · edited" : ""}`;
  const filledVars = info.templateVars.filter((name) => templateVars[name]?.trim()).length;
  const variablesSummary = `${plural(info.templateVars.length, "template variable")}, ${filledVars} filled`;

  const defaultStorage: StorageMode = existingPvc ? "pvc" : "download";
  const cacheChanged =
    storageMode !== defaultStorage ||
    (storageMode === "pvc" &&
      (settings.pvcName.trim() !== "" ||
        settings.size.trim() !== DEFAULT_SETTINGS.size ||
        settings.storageClass.trim() !== "" ||
        settings.accessMode !== DEFAULT_SETTINGS.accessMode ||
        (!existingPvc && settings.mountPath.trim() !== DEFAULT_SETTINGS.mountPath) ||
        settings.createPvc !== DEFAULT_SETTINGS.createPvc ||
        settings.downloadJob !== DEFAULT_SETTINGS.downloadJob));
  const cacheSummary =
    storageMode === "pvc" ? (
      <>
        Shared PVC <code>{pvcName}</code>
      </>
    ) : (
      "Download at startup"
    );
  const tokenChanged = showSecret && (settings.secretMode !== DEFAULT_SETTINGS.secretMode || settings.secretName.trim() !== "");
  const envNamed = env.filter((row) => row.name.trim() !== "");
  const envChanges = envChangeCount(initialEnv, env);
  const envScopeLabels = [...new Set(envNamed.map((row) => info.envScopes.find((scope) => scope.id === row.scope)?.label ?? ""))].filter(Boolean);
  const envSummary =
    envNamed.length === 0
      ? `None set${envChanges ? " · edited" : ""}`
      : `${plural(envNamed.length, "variable")} · ${envScopeLabels.join(", ")}${envChanges ? ` · ${envChanges} edited` : ""}`;
  // Bundling only has an effect when there is a Secret or model cache file to merge.
  const showOutput = settings.bundle || (result?.files.some((file) => file.id === "secret" || file.id === "model-cache") ?? false);
  const moreSections = ["Model cache", ...(showSecret ? ["HF token"] : []), ...(info.envScopes.length > 0 ? ["Env vars"] : [])];
  const moreChanges = [cacheChanged, tokenChanged, envChanges > 0, settings.bundle].filter(Boolean).length;

  return (
    <div className="dc">
      {!shared && <style dangerouslySetInnerHTML={{ __html: LOCAL_SELECTOR_CSS + DEPLOYMENT_CUSTOMIZER_CSS }} />}
      <section className="lqs-panel dc-panel" aria-labelledby={`${id}-title`}>
        <div className="dc-head">
          <div className="dc-head-text">
            <h3 id={`${id}-title`}>Customize this deployment</h3>
            <p>
              {link && (
                <>
                  <a className="dc-source" href={link.href} target="_blank" rel="noreferrer">
                    Source: <code>{link.label}</code>
                    <ExternalIcon />
                  </a>
                  <span className="dc-sep" aria-hidden="true">
                    ·
                  </span>
                </>
              )}
              Changes update the YAML below and stay in your browser.
            </p>
          </div>
          <div className="dc-head-actions">
            <button type="button" className="dc-button" onClick={resetAll}>
              <ResetIcon />
              Reset
            </button>
            <button
              type="button"
              className="dc-button"
              aria-expanded={panelOpen}
              aria-controls={`${id}-rows`}
              onClick={() => setOpen("panel", !panelOpen)}
            >
              {panelOpen ? "Hide options" : "Show options"}
            </button>
          </div>
        </div>
        <div id={`${id}-rows`} hidden={!panelOpen}>
          {(originalModel || model) && (
            <Row label="Model" htmlFor={`${id}-model`}>
              <ModelPicker id={`${id}-model`} value={model} fallback={originalModel} onChange={(next) => update({ model: next })} />
              <p className="dc-hint">
                Type any Hugging Face model ID or pick a suggestion.
                {selectedModel?.gated ? " This model is gated: accept its license on Hugging Face and use a token." : ""}
              </p>
              {model && !isModelId(model) && <p className="dc-hint dc-hint--warn">Hugging Face model IDs look like org/model-name.</p>}
            </Row>
          )}

          {taggable.length > 0 && (
            <Row label="Image">
              <div className="dc-field-row">
                <Field label="Registry" htmlFor={`${id}-registry`}>
                  <input
                    id={`${id}-registry`}
                    className="lqs-input dc-mono"
                    value={settings.registry}
                    placeholder={DEFAULT_REGISTRY}
                    spellCheck={false}
                    autoComplete="off"
                    aria-invalid={!registryValid}
                    onChange={(event) => update({ registry: event.target.value })}
                  />
                </Field>
                <Field label="Tag" htmlFor={`${id}-tag`} narrow>
                  <select
                    id={`${id}-tag`}
                    className="dc-select"
                    value={tagChoice}
                    onChange={(event) => update({ tag: event.target.value })}
                  >
                    <optgroup label="Releases">
                      {IMAGE_TAGS.map((option) => (
                        <option key={option.tag} value={option.tag}>
                          {option.tag}
                          {option.latest ? " (latest)" : ""}
                        </option>
                      ))}
                    </optgroup>
                    <optgroup label="Other">
                      <option value="keep">Keep manifest tags</option>
                      <option value="custom">Custom tag…</option>
                    </optgroup>
                  </select>
                </Field>
                {tagChoice === "custom" && (
                  <Field label="Custom tag" htmlFor={`${id}-custom-tag`} narrow>
                    <input
                      id={`${id}-custom-tag`}
                      className="lqs-input dc-mono"
                      value={settings.customTag}
                      placeholder="1.5.0"
                      spellCheck={false}
                      autoComplete="off"
                      aria-invalid={customTag !== "" && !customTagValid}
                      onChange={(event) => update({ customTag: event.target.value })}
                    />
                  </Field>
                )}
              </div>
              {!registryValid && <p className="dc-hint dc-hint--warn">Enter a lowercase registry path such as registry.example.com/dynamo.</p>}
              <p className="dc-hint">
                {registry ? "Images resolve to " : "Leave Registry empty for NGC. Images resolve to "}
                {resolvedImages.map((ref, index) => (
                  <span key={ref}>
                    {index > 0 ? ", " : ""}
                    <code>{ref}</code>
                  </span>
                ))}
                .
              </p>
            </Row>
          )}

          {info.components.length > 0 && (
            <Row label="Scale" disclosure={disclosure("scale", info.components.length <= OPEN_SCALE_ROWS, scaleSummary)}>
              <div className="dc-table-wrap">
                <table className="dc-table">
                  <thead>
                    <tr>
                      <th scope="col">Component</th>
                      <th scope="col">Replicas</th>
                      <th scope="col">GPUs per pod</th>
                      {hasNodes && <th scope="col">Nodes per replica</th>}
                    </tr>
                  </thead>
                  <tbody>
                    {info.components.map((component) => {
                      const override = components[component.key] ?? {};
                      const replicas = override.replicas ?? component.replicas ?? 1;
                      const gpus = override.gpus ?? component.gpus;
                      const nodes = override.nodes ?? component.nodes;
                      return (
                        <tr key={component.key}>
                          <td className="dc-component">
                            {component.name}
                            <small>
                              {multiGraph ? `${component.graph} · ` : ""}
                              {component.type || "component"}
                            </small>
                          </td>
                          <td>
                            <Stepper
                              label={`${component.name} replicas`}
                              value={replicas}
                              min={0}
                              max={64}
                              changed={replicas !== (component.replicas ?? 1)}
                              onChange={(value) => setOverride(component.key, { replicas: value })}
                            />
                          </td>
                          <td>
                            {component.gpus !== null && gpus !== null ? (
                              <Stepper
                                label={`${component.name} GPUs per pod`}
                                value={gpus}
                                min={1}
                                max={16}
                                changed={gpus !== component.gpus}
                                onChange={(value) => setOverride(component.key, { gpus: value })}
                              />
                            ) : (
                              <span className="dc-muted">{component.dra ? "DRA claim" : "None"}</span>
                            )}
                          </td>
                          {hasNodes && (
                            <td>
                              {component.nodes !== null && nodes !== null ? (
                                <Stepper
                                  label={`${component.name} nodes per replica`}
                                  value={nodes}
                                  min={1}
                                  max={64}
                                  changed={nodes !== component.nodes}
                                  onChange={(value) => setOverride(component.key, { nodes: value })}
                                />
                              ) : (
                                <span className="dc-muted">None</span>
                              )}
                            </td>
                          )}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <p className="dc-total">
                <strong>
                  {totals.gpus} GPU{totals.gpus === 1 ? "" : "s"}
                </strong>{" "}
                across {totals.pods} pod{totals.pods === 1 ? "" : "s"}.
                {tpSynced ? " Tensor-parallel flags that match the GPU count change with it." : ""}
              </p>
            </Row>
          )}

          {info.templateVars.length > 0 && (
            <Row label="Variables" disclosure={disclosure("variables", true, variablesSummary)}>
              <div className="dc-field-row">
                {info.templateVars.map((name) => (
                  <Field key={name} label={name} htmlFor={`${id}-var-${name}`}>
                    <input
                      id={`${id}-var-${name}`}
                      className="lqs-input dc-mono"
                      value={templateVars[name] ?? ""}
                      placeholder={/NAMESPACE$/.test(name) ? settings.namespace || DEFAULT_NAMESPACE : `\${${name}}`}
                      list={/MODEL/.test(name) ? `${id}-models` : undefined}
                      spellCheck={false}
                      autoComplete="off"
                      onChange={(event) => setTemplateVars((current) => ({ ...current, [name]: event.target.value }))}
                    />
                  </Field>
                ))}
              </div>
              <datalist id={`${id}-models`}>
                {MODEL_OPTIONS.map((option) => (
                  <option key={option.id} value={option.id} />
                ))}
              </datalist>
              <p className="dc-hint">Fill in values to substitute them here, or leave them for envsubst at apply time.</p>
            </Row>
          )}

          <Row label="Deploy">
            <div className="dc-field-row">
              {info.graphs.map((graph) => {
                const value = names[graph.doc] ?? "";
                const valid = value === "" || DNS_LABEL.test(value);
                const kind = graph.kind === "DynamoGraphDeploymentRequest" ? "DGDR" : "DGD";
                return (
                  <Field key={graph.doc} label={info.graphs.length > 1 ? `${kind} name` : "Deployment name"} htmlFor={`${id}-name-${graph.doc}`}>
                    <input
                      id={`${id}-name-${graph.doc}`}
                      className="lqs-input dc-mono"
                      value={value}
                      placeholder={graph.name}
                      spellCheck={false}
                      autoComplete="off"
                      aria-invalid={!valid}
                      onChange={(event) => setNames((current) => ({ ...current, [graph.doc]: event.target.value }))}
                    />
                  </Field>
                );
              })}
              <Field label="Namespace" htmlFor={`${id}-namespace`}>
                <input
                  id={`${id}-namespace`}
                  className="lqs-input dc-mono"
                  value={settings.namespace}
                  placeholder={DEFAULT_NAMESPACE}
                  spellCheck={false}
                  autoComplete="off"
                  onChange={(event) => update({ namespace: event.target.value })}
                />
              </Field>
            </div>
          </Row>

          <div className="dc-more" data-open={moreOpen}>
            <button
              type="button"
              className="dc-more-toggle"
              aria-expanded={moreOpen}
              aria-controls={`${id}-more`}
              aria-labelledby={`${id}-more-title`}
              aria-describedby={`${id}-more-list${moreChanges > 0 ? ` ${id}-more-badge` : ""}`}
              onClick={() => setOpen("more", !moreOpen)}
            >
              <span className="dc-more-title" id={`${id}-more-title`}>
                Additional options
              </span>
              <span className="dc-more-list" id={`${id}-more-list`}>
                {moreSections.join(" · ")}
              </span>
              {moreChanges > 0 && (
                <span className="dc-badge dc-badge--ok" id={`${id}-more-badge`}>
                  {moreChanges === 1 ? "1 change" : `${moreChanges} changes`}
                </span>
              )}
              <ChevronIcon />
            </button>
            <div id={`${id}-more`} hidden={!moreOpen}>
              <Row label="Model cache" disclosure={disclosure("cache", true, cacheSummary)}>
                {existingPvc ? (
                  <p className="dc-hint">
                    This manifest already mounts PVC <code>{existingPvc.claim}</code>
                    {existingPvc.mountPath ? (
                      <>
                        {" "}
                        at <code>{existingPvc.mountPath}</code>
                      </>
                    ) : null}
                    . Generate the claim and a download Job for it below.
                  </p>
                ) : (
                  <Chips
                    label="Model cache"
                    value={storageMode}
                    onChange={(value) => update({ storage: value as StorageMode })}
                    options={[
                      { id: "download", label: "Download at startup", sub: "each pod from Hugging Face" },
                      { id: "pvc", label: "Shared PVC", sub: "download once, mount everywhere" },
                    ]}
                  />
                )}
                {storageMode === "pvc" && (
                  <>
                    <div className="dc-field-row">
                      <Field label="PVC name" htmlFor={`${id}-pvc`}>
                        <input
                          id={`${id}-pvc`}
                          className="lqs-input dc-mono"
                          value={settings.pvcName}
                          placeholder={existingPvc?.claim ?? "model-cache"}
                          spellCheck={false}
                          autoComplete="off"
                          aria-invalid={!pvcNameValid}
                          onChange={(event) => update({ pvcName: event.target.value })}
                        />
                      </Field>
                      {!existingPvc && (
                        <Field label="Mount path" htmlFor={`${id}-mount`}>
                          <input
                            id={`${id}-mount`}
                            className="lqs-input dc-mono"
                            value={settings.mountPath}
                            placeholder={DEFAULT_MOUNT_PATH}
                            spellCheck={false}
                            autoComplete="off"
                            onChange={(event) => update({ mountPath: event.target.value })}
                          />
                        </Field>
                      )}
                    </div>
                    <div className="dc-checks">
                      <Check
                        label={claimInManifest ? "Create the PVC (defined in this manifest)" : "Create the PVC"}
                        checked={settings.createPvc && !claimInManifest}
                        disabled={claimInManifest}
                        onChange={(checked) => update({ createPvc: checked })}
                      />
                      <Check
                        label={jobModel ? "Download the model with a Job" : "Download the model with a Job (needs a model ID)"}
                        checked={settings.downloadJob && !!jobModel}
                        disabled={!jobModel}
                        onChange={(checked) => update({ downloadJob: checked })}
                      />
                    </div>
                    {settings.createPvc && !claimInManifest && (
                      <div className="dc-field-row">
                        <Field label="Size" htmlFor={`${id}-size`} narrow>
                          <input
                            id={`${id}-size`}
                            className="lqs-input dc-mono"
                            value={settings.size}
                            placeholder="100Gi"
                            spellCheck={false}
                            autoComplete="off"
                            aria-invalid={!sizeValid}
                            onChange={(event) => update({ size: event.target.value })}
                          />
                        </Field>
                        <Field label="Storage class" htmlFor={`${id}-class`}>
                          <input
                            id={`${id}-class`}
                            className="lqs-input dc-mono"
                            value={settings.storageClass}
                            placeholder="Cluster default"
                            spellCheck={false}
                            autoComplete="off"
                            onChange={(event) => update({ storageClass: event.target.value })}
                          />
                        </Field>
                        <Field label="Access mode" htmlFor={`${id}-access`} narrow>
                          <select
                            id={`${id}-access`}
                            className="dc-select"
                            value={settings.accessMode}
                            onChange={(event) => update({ accessMode: event.target.value as StorageSettings["accessMode"] })}
                          >
                            <option value="ReadWriteMany">ReadWriteMany</option>
                            <option value="ReadWriteOnce">ReadWriteOnce</option>
                          </select>
                        </Field>
                      </div>
                    )}
                    {(!pvcNameValid || !sizeValid) && (
                      <p className="dc-hint dc-hint--warn">Use a lowercase DNS name for the PVC and a quantity such as 100Gi for the size.</p>
                    )}
                    <p className="dc-hint">
                      {existingPvc
                        ? "The download Job writes to the cache path this manifest reads."
                        : "Workers set HF_HOME to the mount path and read the cached weights."}{" "}
                      Size the claim for the weights plus headroom, about 2 GB per billion parameters at BF16. ReadWriteMany needs a
                      shared storage class such as EFS, Filestore, or NFS.
                    </p>
                  </>
                )}
              </Row>

              {showSecret && (
                <Row label="HF token">
                  <Chips
                    label="Hugging Face token Secret"
                    value={settings.secretMode}
                    onChange={(value) => update({ secretMode: value as SecretMode })}
                    options={[
                      { id: "kubectl", label: "kubectl command", sub: "from $HF_TOKEN" },
                      { id: "yaml", label: "Secret YAML", sub: "paste the token" },
                      { id: "existing", label: "Already created" },
                      { id: "none", label: "No token", sub: "public models" },
                    ]}
                  />
                  {settings.secretMode !== "none" && (
                    <div className="dc-field-row">
                      <Field label="Secret name" htmlFor={`${id}-secret`}>
                        <input
                          id={`${id}-secret`}
                          className="lqs-input dc-mono"
                          value={settings.secretName}
                          placeholder={info.secret ?? "hf-token-secret"}
                          spellCheck={false}
                          autoComplete="off"
                          aria-invalid={!secretNameValid}
                          onChange={(event) => update({ secretName: event.target.value })}
                        />
                      </Field>
                    </div>
                  )}
                  <p className="dc-hint">
                    {settings.secretMode === "kubectl" && "Deploy steps create the Secret from the HF_TOKEN environment variable."}
                    {settings.secretMode === "yaml" && "Adds a Secret manifest; paste a token into it for gated or private models."}
                    {settings.secretMode === "existing" && "Uses a Secret that already exists in the namespace."}
                    {settings.secretMode === "none" && "Removes Secret references from the manifest. Use only with public models."}
                  </p>
                </Row>
              )}

              {info.envScopes.length > 0 && (
                <Row label="Env vars" disclosure={disclosure("env", false, envSummary)}>
                  <EnvEditor rows={env} scopes={info.envScopes} onChange={setEnv} />
                </Row>
              )}

              {showOutput && (
                <Row label="Output">
                  <div className="dc-checks">
                    <Check
                      label="Bundle the Secret and model cache into one file"
                      checked={settings.bundle}
                      onChange={(checked) => update({ bundle: checked })}
                    />
                  </div>
                </Row>
              )}
            </div>
          </div>
        </div>
      </section>

      {result ? (
        <Viewer
          id={id}
          source={source}
          result={result}
          notes={[...uiNotes, ...result.notes]}
          models={[model || originalModel || ""].filter(Boolean)}
        />
      ) : (
        <>
          <p className="dc-hint dc-hint--warn">These settings could not be applied; showing the original manifest.</p>
          {code}
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

interface Disclosure {
  open: boolean;
  onToggle: (open: boolean) => void;
  /** Stands in for the controls while the row is closed. */
  summary: ReactNode;
}

function plural(count: number, noun: string): string {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

/** Counts added, removed, and edited variables relative to the manifest's own. */
function envChangeCount(initial: EnvRow[], rows: EnvRow[]): number {
  const current = new Map(rows.map((row) => [row.id, row]));
  let changes = rows.filter((row) => row.orig === null && row.name.trim() !== "").length;
  for (const row of initial) {
    const now = current.get(row.id);
    if (!now || now.name !== row.name || now.value !== row.value) changes++;
  }
  return changes;
}

function Row({
  label,
  htmlFor,
  disclosure,
  children,
}: {
  label: string;
  htmlFor?: string;
  disclosure?: Disclosure;
  children: ReactNode;
}) {
  const id = useId();
  if (disclosure) {
    const { open, onToggle, summary } = disclosure;
    return (
      <div className="lqs-row dc-row" data-open={open}>
        <button
          type="button"
          className="lqs-label dc-row-toggle"
          id={id}
          aria-expanded={open}
          aria-controls={`${id}-body`}
          aria-describedby={open ? undefined : `${id}-summary`}
          onClick={() => onToggle(!open)}
        >
          {label}
          <ChevronIcon />
        </button>
        <div className="dc-row-content">
          {!open && (
            // Mouse shortcut only; keyboard and screen reader users get the summary through the toggle.
            <p className="dc-row-summary" id={`${id}-summary`} onClick={() => onToggle(true)}>
              {summary}
            </p>
          )}
          <div className="dc-stack" id={`${id}-body`} role="group" aria-labelledby={id} hidden={!open}>
            {children}
          </div>
        </div>
      </div>
    );
  }
  return (
    <div className="lqs-row">
      {htmlFor ? (
        <label className="lqs-label" htmlFor={htmlFor}>
          {label}
        </label>
      ) : (
        <span className="lqs-label" id={id}>
          {label}
        </span>
      )}
      <div className="dc-stack" role={htmlFor ? undefined : "group"} aria-labelledby={htmlFor ? undefined : id}>
        {children}
      </div>
    </div>
  );
}

function Field({ label, htmlFor, narrow, children }: { label: string; htmlFor: string; narrow?: boolean; children: ReactNode }) {
  return (
    <div className={`dc-field${narrow ? " dc-field--narrow" : ""}`}>
      <label className="dc-field-label" htmlFor={htmlFor}>
        {label}
      </label>
      {children}
    </div>
  );
}

function Chips({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: { id: string; label: string; sub?: string }[];
  onChange: (value: string) => void;
}) {
  return (
    <div className="lqs-options" role="group" aria-label={label}>
      {options.map((option) => (
        <button
          key={option.id}
          type="button"
          className="lqs-chip"
          aria-pressed={value === option.id}
          onClick={() => onChange(option.id)}
        >
          {option.label}
          {option.sub && <span className="lqs-chip-sub">{option.sub}</span>}
        </button>
      ))}
    </div>
  );
}

function Check({
  label,
  checked,
  disabled,
  onChange,
}: {
  label: string;
  checked: boolean;
  disabled?: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label className="dc-check" data-disabled={disabled || undefined}>
      <input type="checkbox" checked={checked} disabled={disabled} onChange={(event) => onChange(event.target.checked)} />
      {label}
    </label>
  );
}

function Stepper({
  label,
  value,
  min,
  max,
  changed,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  changed: boolean;
  onChange: (value: number) => void;
}) {
  const [draft, setDraft] = useState(String(value));
  useEffect(() => setDraft(String(value)), [value]);
  const accept = (raw: string) => {
    const next = Number(raw);
    return raw.trim() !== "" && Number.isInteger(next) && next >= min && next <= max ? next : null;
  };
  return (
    <span className="dc-stepper" data-changed={changed}>
      <button type="button" aria-label={`Decrease ${label}`} disabled={value <= min} onClick={() => onChange(value - 1)}>
        −
      </button>
      <input
        type="number"
        inputMode="numeric"
        aria-label={label}
        min={min}
        max={max}
        value={draft}
        onChange={(event) => {
          setDraft(event.target.value);
          const next = accept(event.target.value);
          if (next !== null) onChange(next);
        }}
        onBlur={(event) => {
          if (accept(event.target.value) === null) setDraft(String(value));
        }}
      />
      <button type="button" aria-label={`Increase ${label}`} disabled={value >= max} onClick={() => onChange(value + 1)}>
        +
      </button>
    </span>
  );
}

function ModelPicker({
  id,
  value,
  fallback,
  onChange,
}: {
  id: string;
  value: string;
  fallback: string | null;
  onChange: (value: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState("");
  const [active, setActive] = useState(0);
  const [draft, setDraft] = useState<string | null>(null);
  const wrapper = useRef<HTMLDivElement>(null);
  const listId = `${id}-list`;
  const shown = value || fallback || "";

  const preferred: ModelCategory = modelCategory(fallback) ?? "text";
  const query = filter.trim().toLowerCase();
  const matches = MODEL_OPTIONS.filter((option) => !query || option.id.toLowerCase().includes(query));
  const groups: { category: ModelCategory; options: ModelOption[] }[] = [];
  for (const option of [...matches].sort((a, b) => Number(b.category === preferred) - Number(a.category === preferred))) {
    let group = groups.find((candidate) => candidate.category === option.category);
    if (!group) groups.push((group = { category: option.category, options: [] }));
    group.options.push(option);
  }
  const flat = groups.flatMap((group) => group.options);

  useEffect(() => {
    if (!open) return;
    const close = (event: PointerEvent) => {
      if (!wrapper.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", close);
    return () => document.removeEventListener("pointerdown", close);
  }, [open]);

  useEffect(() => {
    if (open) document.getElementById(`${listId}-${active}`)?.scrollIntoView({ block: "nearest" });
  }, [open, active, listId]);

  function choose(next: string) {
    onChange(next === fallback ? "" : next);
    setDraft(null);
    setFilter("");
    setOpen(false);
  }

  /** Applies typed text: model IDs apply as you type; other text applies on Enter. */
  function commit(text: string, force: boolean) {
    const trimmed = text.trim();
    if (trimmed === "" || trimmed === fallback) onChange("");
    else if (force || isModelId(trimmed)) onChange(trimmed);
  }

  function onKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      if (!open) {
        setFilter("");
        setOpen(true);
        setActive(0);
      } else setActive((current) => Math.min(current + 1, flat.length - 1));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActive((current) => Math.max(current - 1, 0));
    } else if (event.key === "Enter") {
      if (open && flat[active]) {
        event.preventDefault();
        choose(flat[active].id);
      } else {
        if (draft !== null) commit(draft, true);
        setOpen(false);
      }
    } else if (event.key === "Escape" && open) {
      event.preventDefault();
      setOpen(false);
    }
  }

  let index = -1;
  return (
    <div className="dc-combobox" ref={wrapper}>
      <input
        id={id}
        className="lqs-input dc-mono"
        role="combobox"
        aria-expanded={open}
        aria-controls={listId}
        aria-autocomplete="list"
        aria-activedescendant={open && flat[active] ? `${listId}-${active}` : undefined}
        value={draft ?? shown}
        placeholder={fallback ?? "org/model-name"}
        spellCheck={false}
        autoComplete="off"
        onChange={(event) => {
          const next = event.target.value;
          commit(next, false);
          setDraft(next);
          setFilter(next);
          setOpen(true);
          setActive(0);
        }}
        onKeyDown={onKeyDown}
        onBlur={() => {
          if (draft !== null) commit(draft, false);
          setDraft(null);
          setFilter("");
        }}
      />
      <div className="dc-combobox-actions">
        <button
          type="button"
          className="dc-icon-button"
          aria-label="Use the example's model"
          title="Use the example's model"
          disabled={!value}
          onClick={() => choose(fallback ?? "")}
        >
          <ResetIcon />
        </button>
        <button
          type="button"
          className="dc-icon-button"
          aria-label="Show suggested models"
          aria-expanded={open}
          aria-controls={listId}
          tabIndex={-1}
          onClick={() => {
            setFilter("");
            setActive(0);
            setOpen(!open);
          }}
        >
          <ChevronIcon />
        </button>
      </div>
      {open && (
        <ul id={listId} className="dc-listbox" role="listbox" aria-label="Suggested models">
          {flat.length === 0 && (
            <li className="dc-listbox-empty" role="presentation">
              No suggestions. The typed model ID is used as is.
            </li>
          )}
          {groups.map((group) => (
            <GroupOptions key={group.category} label={MODEL_CATEGORIES[group.category]}>
              {group.options.map((option) => {
                index++;
                const position = index;
                return (
                  <li
                    key={option.id}
                    id={`${listId}-${position}`}
                    className="dc-option"
                    role="option"
                    aria-selected={option.id === shown}
                    data-active={position === active}
                    onMouseDown={(event) => event.preventDefault()}
                    onMouseEnter={() => setActive(position)}
                    onClick={() => choose(option.id)}
                  >
                    <span className="dc-option-id">{option.id}</span>
                    <span className="dc-option-meta">
                      {option.id === fallback && <span className="dc-badge dc-badge--ok">example</span>}
                      {option.gated && <span className="dc-badge dc-badge--warn">gated</span>}
                      <span>{option.size}</span>
                    </span>
                  </li>
                );
              })}
            </GroupOptions>
          ))}
        </ul>
      )}
    </div>
  );
}

function GroupOptions({ label, children }: { label: string; children: ReactNode }) {
  return (
    <>
      <li className="dc-listbox-group" role="presentation">
        {label}
      </li>
      {children}
    </>
  );
}

let nextEnvId = 0;

function EnvEditor({
  rows,
  scopes,
  onChange,
}: {
  rows: EnvRow[];
  scopes: ManifestInfo["envScopes"];
  onChange: (rows: EnvRow[]) => void;
}) {
  const patch = (rowId: string, change: Partial<EnvRow>) =>
    onChange(rows.map((row) => (row.id === rowId ? { ...row, ...change } : row)));
  const seen = new Set<string>();
  const duplicates = new Set<string>();
  for (const row of rows) {
    const key = `${row.scope}\u0000${row.name.trim()}`;
    if (row.name.trim() && seen.has(key)) duplicates.add(key);
    seen.add(key);
  }
  const invalid = rows.some((row) => row.name.trim() !== "" && !ENV_NAME.test(row.name.trim()));
  return (
    <div className="dc-env">
      {rows.length > 0 && (
        <div className="dc-env-row dc-env-head" aria-hidden="true">
          <span>Applies to</span>
          <span>Name</span>
          <span>Value</span>
          <span />
        </div>
      )}
      {rows.map((row) => {
        const name = row.name.trim();
        return (
          <div className="dc-env-row" key={row.id} data-new={row.orig === null}>
            <select
              className="dc-select"
              aria-label={`Components that receive ${name || "the new variable"}`}
              value={row.scope}
              disabled={row.orig !== null}
              onChange={(event) => patch(row.id, { scope: event.target.value })}
            >
              {scopes.map((scope) => (
                <option key={scope.id} value={scope.id}>
                  {scope.label}
                </option>
              ))}
            </select>
            <input
              className="lqs-input dc-mono"
              aria-label="Variable name"
              value={row.name}
              placeholder="NAME"
              spellCheck={false}
              autoComplete="off"
              aria-invalid={(name !== "" && !ENV_NAME.test(name)) || duplicates.has(`${row.scope}\u0000${name}`)}
              onChange={(event) => patch(row.id, { name: event.target.value })}
            />
            {row.from ? (
              <span className="dc-env-from" title={`valueFrom: ${row.from}`}>
                from {row.from}
              </span>
            ) : (
              <input
                className="lqs-input dc-mono"
                aria-label={`Value of ${name || "the new variable"}`}
                value={row.value}
                placeholder="value"
                spellCheck={false}
                autoComplete="off"
                onChange={(event) => patch(row.id, { value: event.target.value })}
              />
            )}
            <button
              type="button"
              className="dc-icon-button dc-icon-button--remove"
              aria-label={`Remove ${name || "variable"}`}
              title="Remove"
              onClick={() => onChange(rows.filter((candidate) => candidate.id !== row.id))}
            >
              <MinusIcon />
            </button>
          </div>
        );
      })}
      <button
        type="button"
        className="dc-button dc-env-add"
        onClick={() =>
          onChange([
            ...rows,
            { id: `new-${nextEnvId++}`, scope: scopes[0].id, name: "", value: "", from: null, orig: null },
          ])
        }
      >
        <PlusIcon />
        Add variable
      </button>
      {invalid && <p className="dc-hint dc-hint--warn">Variable names use letters, digits, and the characters - . _ and cannot start with a digit.</p>}
      {duplicates.size > 0 && <p className="dc-hint dc-hint--warn">A variable is set twice for the same components; the last one wins.</p>}
      <p className="dc-hint">All components maps to spec.env, which the operator prepends to every component.</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Viewer
// ---------------------------------------------------------------------------

interface ViewRow {
  key: string;
  tokens: Token[];
  ln: string;
  type: "same" | "add" | "del";
  line: number | null;
}

function Line({ row, flash }: { row: ViewRow; flash: boolean }) {
  const classes = ["dc-line"];
  if (row.type === "add") classes.push("dc-line--add");
  if (row.type === "del") classes.push("dc-line--del");
  if (flash) classes.push("dc-line--flash");
  return (
    <span
      className={classes.join(" ")}
      data-ln={row.ln}
      data-line={row.line ?? undefined}
      data-mark={row.type === "same" ? undefined : row.type === "add" ? "+" : "-"}
    >
      {row.tokens.map((token, index) =>
        token.c ? (
          <span key={index} className={`dc-t-${token.c}`}>
            {token.t}
          </span>
        ) : (
          token.t
        ),
      )}
    </span>
  );
}

function fileLabel(file: OutputFile): string {
  return file.id === "steps" ? "Deploy steps" : file.name;
}

function Viewer({
  id,
  source,
  result,
  notes,
  models,
}: {
  id: string;
  source: string;
  result: CustomizeResult;
  notes: string[];
  models: string[];
}) {
  const [activeId, setActiveId] = useState<OutputFile["id"]>("manifest");
  const [showDiff, setShowDiff] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [status, setStatus] = useState("");
  const [flash, setFlash] = useState<Set<number>>(() => new Set());
  const scroller = useRef<HTMLDivElement>(null);
  const previous = useRef<string | null>(null);
  const file = result.files.find((candidate) => candidate.id === activeId) ?? result.files[0];
  const isManifest = file.id === "manifest";
  const modelKey = models.join("\n");

  const original = useMemo(() => splitLines(source).lines, [source]);
  const diff = useMemo(() => diffLines(original, splitLines(result.manifest).lines), [original, result.manifest]);
  const added = useMemo(() => new Set(diff.filter((entry) => entry.type === "add").map((entry) => entry.line as number)), [diff]);
  const additions = diff.filter((entry) => entry.type === "add").length;
  const deletions = diff.filter((entry) => entry.type === "del").length;
  const changeCount = additions + deletions;
  const tokens = useMemo(
    () => (file.language === "yaml" ? highlightYaml(file.content, modelKey ? modelKey.split("\n") : []) : highlightBash(file.content)),
    [file.content, file.language, modelKey],
  );
  const originalTokens = useMemo(() => (showDiff && isManifest ? highlightYaml(source) : []), [showDiff, isManifest, source]);

  const rows: ViewRow[] =
    isManifest && showDiff
      ? diff.map((entry, index) => ({
          key: `d${index}`,
          tokens: entry.type === "del" ? originalTokens[entry.old as number] ?? [{ t: entry.text }] : tokens[entry.line as number] ?? [{ t: entry.text }],
          ln: entry.line === null ? "" : String(entry.line + 1),
          type: entry.type,
          line: entry.line,
        }))
      : tokens.map((lineTokens, index) => ({
          key: `l${index}`,
          tokens: lineTokens,
          ln: String(index + 1),
          type: isManifest && added.has(index) ? "add" : "same",
          line: index,
        }));

  useEffect(() => {
    const before = previous.current;
    previous.current = result.manifest;
    if (before === null || before === result.manifest) return;
    const changed = diffLines(splitLines(before).lines, splitLines(result.manifest).lines)
      .filter((entry) => entry.type === "add")
      .map((entry) => entry.line as number);
    setFlash(new Set(changed));
    const container = scroller.current;
    const target = container?.querySelector<HTMLElement>(`[data-line="${changed[0]}"]`);
    if (container && target && activeId === "manifest") {
      const top = target.offsetTop;
      if (top < container.scrollTop || top > container.scrollTop + container.clientHeight - 40) {
        const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
        container.scrollTo({ top: Math.max(0, top - container.clientHeight / 3), behavior: reduce ? "auto" : "smooth" });
      }
    }
    const timer = window.setTimeout(() => setFlash(new Set()), 1400);
    return () => window.clearTimeout(timer);
  }, [result.manifest, activeId]);

  function announce(message: string) {
    setStatus(message);
    window.setTimeout(() => setStatus(""), 1600);
  }

  async function copy() {
    try {
      await navigator.clipboard.writeText(file.content);
      announce("Copied");
    } catch {
      announce("Copy failed");
    }
  }

  function download() {
    const blob = new Blob([file.content], { type: file.language === "yaml" ? "application/yaml" : "text/x-shellscript" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = file.name;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    announce(`Downloaded ${file.name}`);
  }

  function onTabKey(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
    event.preventDefault();
    const next = result.files[(index + (event.key === "ArrowRight" ? 1 : -1) + result.files.length) % result.files.length];
    setActiveId(next.id);
    document.getElementById(`${id}-tab-${next.id}`)?.focus();
  }

  return (
    <div className="dc-view" data-expanded={expanded}>
      <div className="dc-toolbar">
        <div className="dc-tabs" role="tablist" aria-label="Generated files">
          {result.files.map((candidate, index) => (
            <button
              key={candidate.id}
              id={`${id}-tab-${candidate.id}`}
              type="button"
              className="dc-tab"
              role="tab"
              aria-selected={candidate.id === file.id}
              aria-controls={`${id}-panel`}
              tabIndex={candidate.id === file.id ? 0 : -1}
              onClick={() => setActiveId(candidate.id)}
              onKeyDown={(event) => onTabKey(event, index)}
            >
              {fileLabel(candidate)}
              {candidate.id === "manifest" && changeCount > 0 && (
                <span className="dc-tab-count" title={`${additions} added, ${deletions} removed lines`}>
                  {additions > 0 && <span className="dc-tab-count-add">+{additions}</span>}
                  {deletions > 0 && <span className="dc-tab-count-del">−{deletions}</span>}
                </span>
              )}
            </button>
          ))}
        </div>
        <div className="dc-actions">
          {isManifest && (
            <button
              type="button"
              className="dc-button"
              aria-pressed={showDiff}
              disabled={changeCount === 0}
              onClick={() => setShowDiff(!showDiff)}
            >
              <DiffIcon />
              Changes
            </button>
          )}
          <button type="button" className="dc-button" aria-pressed={expanded} onClick={() => setExpanded(!expanded)}>
            <ExpandIcon />
            {expanded ? "Collapse" : "Expand"}
          </button>
          <button type="button" className="dc-button" onClick={download}>
            <DownloadIcon />
            Download
          </button>
          <button type="button" className="dc-button" onClick={copy}>
            <CopyIcon />
            {status === "Copied" ? "Copied" : "Copy"}
          </button>
          <span className="dc-sr-only" role="status" aria-live="polite">
            {status}
          </span>
        </div>
      </div>
      {notes.length > 0 && (
        <ul className="dc-notes">
          {notes.map((note) => (
            <li key={note} className="dc-note">
              {note}
            </li>
          ))}
        </ul>
      )}
      <div
        className="dc-scroll"
        ref={scroller}
        id={`${id}-panel`}
        role="tabpanel"
        aria-labelledby={`${id}-tab-${file.id}`}
        tabIndex={0}
      >
        <pre className="dc-pre">
          <code>
            {rows.map((row) => (
              <Line key={row.key} row={row} flash={isManifest && row.line !== null && row.type !== "del" && flash.has(row.line)} />
            ))}
          </code>
        </pre>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Icons
// ---------------------------------------------------------------------------

function Icon({ children }: { children: ReactNode }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {children}
    </svg>
  );
}

function ResetIcon() {
  return (
    <Icon>
      <path d="M3 12a9 9 0 1 0 3-6.7" />
      <path d="M3 4v5h5" />
    </Icon>
  );
}

function ChevronIcon() {
  return (
    <Icon>
      <path d="m6 9 6 6 6-6" />
    </Icon>
  );
}

function ExternalIcon() {
  return (
    <Icon>
      <path d="M15 3h6v6" />
      <path d="M10 14 21 3" />
      <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
    </Icon>
  );
}

function CopyIcon() {
  return (
    <Icon>
      <rect x="9" y="9" width="12" height="12" rx="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </Icon>
  );
}

function DownloadIcon() {
  return (
    <Icon>
      <path d="M12 3v12" />
      <path d="m7 10 5 5 5-5" />
      <path d="M5 21h14" />
    </Icon>
  );
}

function DiffIcon() {
  return (
    <Icon>
      <path d="M12 3v8M8 7h8" />
      <path d="M8 17h8" />
      <rect x="3" y="3" width="18" height="18" rx="2" />
    </Icon>
  );
}

function ExpandIcon() {
  return (
    <Icon>
      <path d="m7 15 5 5 5-5" />
      <path d="m7 9 5-5 5 5" />
    </Icon>
  );
}

function MinusIcon() {
  return (
    <Icon>
      <circle cx="12" cy="12" r="9" />
      <path d="M8 12h8" />
    </Icon>
  );
}

function PlusIcon() {
  return (
    <Icon>
      <path d="M12 5v14M5 12h14" />
    </Icon>
  );
}
