"use client";

import { useEffect, useRef } from "react";

import { kubectlDocStyles } from "./kubectl-doc-styles";

export type KubeSchemaLine = {
  index: number;
  text: string;
  depth: number;
  field?: string;
  path?: string;
  code?: boolean;
  metadata?: boolean;
  required?: boolean;
  foldable?: boolean;
  collapsed?: boolean;
  detailId?: string;
  commentGroup?: string;
  tokens?: KubeSchemaToken[];
  comment?: KubeSchemaComment;
};

export type KubeSchemaToken = {
  k?: string;
  kind?: string;
  t?: string;
  text?: string;
};

export type KubeSchemaComment = {
  prefix: string;
  wrapPrefix: string;
  text: string;
};

export type KubeSchemaField = {
  id: string;
  path: string;
  type: string;
  required: boolean;
  description?: string;
  metadata?: string[];
};

export type KubeSchemaDocument = {
  apiVersion: string;
  group: string;
  version: string;
  kind: string;
  resource?: string;
  complete?: boolean;
  fullPayloadURL?: string;
  lines: KubeSchemaLine[];
  fields: KubeSchemaField[];
};

export type KubectlDocController = {
  destroy: () => void;
  focusPath?: (path: string, options?: { scroll?: boolean; updateHash?: boolean }) => boolean;
  expandPath?: (path: string) => boolean;
  collapsePath?: (path: string) => boolean;
  setFocused?: (value: boolean) => void;
  setFilter?: (filter: string) => void;
  clearFilter?: () => void;
  snapshot?: () => KubectlDocSnapshot;
};

export type KubectlDocSnapshot = {
  currentPath: string;
  filter: string;
  folds?: Array<{ path: string; expanded: boolean }>;
  hasFocus?: boolean;
};

type KubectlDocHost = HTMLDivElement & {
  __kubectlDocController?: KubectlDocController | null;
};

export type KubectlDocRuntime = {
  mount: (
    root: HTMLElement,
    options: {
      initialSchema: KubeSchemaDocument;
      filtering: boolean;
      detailsMode?: "inline-side" | "side-overlay";
      wrapControl?: boolean;
      wrapComments?: boolean;
      preloadFullSchema?: boolean;
      autoFocus?: boolean;
      loadFullSchema?: () => Promise<KubeSchemaDocument> | KubeSchemaDocument | false | void;
    },
  ) => KubectlDocController;
};

declare global {
  interface Window {
    KubectlDoc?: KubectlDocRuntime;
  }
}

export type KubeSchemaDocProps = {
  data: KubeSchemaDocument;
  filtering?: boolean;
  loadFullSchema?: () => Promise<KubeSchemaDocument> | KubeSchemaDocument | false | void;
  detailsMode?: "inline-side" | "side-overlay";
  wrapControl?: boolean;
  wrapComments?: boolean;
  preloadFullSchema?: boolean;
  autoFocus?: boolean;
  className?: string;
  injectStyles?: boolean;
  styleElementID?: string;
  runtimeLoader?: () => Promise<KubectlDocRuntime> | KubectlDocRuntime;
};

let runtimePromise: Promise<KubectlDocRuntime> | null = null;
const fullSchemaCache = new Map<string, Promise<KubeSchemaDocument> | KubeSchemaDocument>();
const defaultStyleElementID = "kubectl-doc-react-styles";

function ensureKubectlDocStyles(styleElementID: string) {
  if (typeof document === "undefined" || document.getElementById(styleElementID)) {
    return;
  }

  const style = document.createElement("style");
  style.id = styleElementID;
  style.textContent = kubectlDocStyles;
  document.head.appendChild(style);
}

function defaultRuntimeLoader() {
  if (!runtimePromise) {
    runtimePromise = import("./kubectl-doc-runtime.js").then(() => {
      if (!window.KubectlDoc) {
        throw new Error("kubectl-doc runtime did not register window.KubectlDoc");
      }
      return window.KubectlDoc;
    });
  }
  return runtimePromise;
}

function ensureKubectlDocRuntime(runtimeLoader?: () => Promise<KubectlDocRuntime> | KubectlDocRuntime) {
  if (typeof window === "undefined") {
    return Promise.reject(new Error("kubectl-doc runtime is only available in the browser"));
  }
  if (window.KubectlDoc) {
    return Promise.resolve(window.KubectlDoc);
  }
  return Promise.resolve(runtimeLoader ? runtimeLoader() : defaultRuntimeLoader()).then((runtime) => {
    if (!runtime) {
      throw new Error("kubectl-doc runtime loader did not return a runtime");
    }
    return runtime;
  });
}

function resolveSchemaSource(source: string) {
  if (source.startsWith("http://") || source.startsWith("https://") || source.startsWith("/")) {
    return source;
  }
  return new URL(source, window.location.href.replace(/\/$/, "")).toString();
}

function defaultLoadFullSchema(data: KubeSchemaDocument) {
  if (data.complete || !data.fullPayloadURL) {
    return undefined;
  }

  const source = resolveSchemaSource(data.fullPayloadURL);
  return () => {
    const cached = fullSchemaCache.get(source);
    if (cached) {
      return Promise.resolve(cached);
    }

    const request = fetch(source)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`${response.status} ${response.statusText}`);
        }
        return response.json() as Promise<KubeSchemaDocument>;
      })
      .then((schema) => {
        fullSchemaCache.set(source, schema);
        return schema;
      })
      .catch((error: unknown) => {
        fullSchemaCache.delete(source);
        throw error;
      });
    fullSchemaCache.set(source, request);
    return request;
  };
}

function activeController(root: KubectlDocHost | null, fallback?: KubectlDocController) {
  return root?.__kubectlDocController ?? fallback;
}

function restoreSnapshot(controller: KubectlDocController, snapshot: KubectlDocSnapshot | null) {
  if (!snapshot) {
    return;
  }

  snapshot.folds?.forEach((fold) => {
    if (fold.expanded) {
      controller.expandPath?.(fold.path);
    } else {
      controller.collapsePath?.(fold.path);
    }
  });
  if (snapshot.filter) {
    controller.setFilter?.(snapshot.filter);
  }
  if (snapshot.currentPath) {
    controller.focusPath?.(snapshot.currentPath, { scroll: false, updateHash: false });
  }
  if (snapshot.hasFocus) {
    controller.setFocused?.(true);
  }
}

function classNames(...values: Array<string | undefined>) {
  return values.filter(Boolean).join(" ");
}

export function KubeSchemaDoc({
  data,
  filtering = true,
  loadFullSchema,
  detailsMode = "side-overlay",
  wrapControl = false,
  wrapComments = true,
  preloadFullSchema = true,
  autoFocus = true,
  className,
  injectStyles = true,
  styleElementID = defaultStyleElementID,
  runtimeLoader,
}: KubeSchemaDocProps) {
  const rootRef = useRef<KubectlDocHost | null>(null);
  const snapshotRef = useRef<KubectlDocSnapshot | null>(null);

  useEffect(() => {
    let cancelled = false;
    let controller: KubectlDocController | undefined;

    if (injectStyles) {
      ensureKubectlDocStyles(styleElementID);
    }
    ensureKubectlDocRuntime(runtimeLoader)
      .then((runtime) => {
        if (cancelled || !rootRef.current) {
          return;
        }
        const previousSnapshot = snapshotRef.current;
        snapshotRef.current = null;
        rootRef.current.innerHTML = "";
        controller = runtime.mount(rootRef.current, {
          initialSchema: data,
          filtering,
          detailsMode,
          wrapControl,
          wrapComments,
          preloadFullSchema,
          autoFocus,
          loadFullSchema: loadFullSchema ?? defaultLoadFullSchema(data),
        });
        restoreSnapshot(controller, previousSnapshot);
      })
      .catch((error: unknown) => {
        console.error("kubectl-doc runtime failed to mount", error);
      });

    return () => {
      cancelled = true;
      const mountedController = activeController(rootRef.current, controller);
      snapshotRef.current = mountedController?.snapshot?.() ?? null;
      mountedController?.destroy();
    };
  }, [
    data,
    filtering,
    loadFullSchema,
    detailsMode,
    wrapControl,
    wrapComments,
    preloadFullSchema,
    autoFocus,
    injectStyles,
    styleElementID,
    runtimeLoader,
  ]);

  return <div ref={rootRef} className={classNames("kubectl-doc", "kdoc-embedded-host", "kdoc-react-host", className)} />;
}

export default KubeSchemaDoc;
