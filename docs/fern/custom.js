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

(() => {
  const NATIVE_SELECTOR = ".fern-variant-selector";
  const CUSTOM_SELECTOR_CLASS = "dynamo-variant-selector";

  // Inline icons avoid a flash of empty rows while the selector initializes.
  // Font Awesome Free icons (CC BY 4.0), Copyright Fonticons, Inc.
  const icons = {
    kubernetes: `
        <svg viewBox="0 0 512 512" aria-hidden="true" focusable="false">
          <path fill="currentColor" d="M256 316.1c3.5-.1 7.2 2 8.9 5.1L293 371.7c-3.6 1.2-7.3 2.3-11.2 3.1-21.4 4.9-42.7 3.4-62-3.2l27.9-50.4c1.7-3.1 4.9-5 8.2-5.1zm50.5-23.4l57.2 9.7c-8.4 23.5-24.4 43.9-45.8 57.5l-22.2-53.6c-2-4.7 .1-10.4 4.8-12.6 1.9-.9 4.1-1.3 6-.9zm-89.1 7.7c.5 2.1 .3 4.2-.5 6l-21.8 53.3c-20.5-13.2-36.9-32.9-45.7-57.1l56.7-9.6c5.1-.9 10.1 2.4 11.3 7.5zm58.8-41l-3.9 17.1-15.8 7.6-15.9-7.7-4-17.1 11-13.7 17.7 0 11 13.7zm-73.7-16.7c3.9 3.4 4.4 9.5 1.2 13.6-1.3 1.7-3.1 2.8-5 3.3l-55.3 16.2c-2.8-25.7 3.3-50.7 16.1-71.6l43.1 38.6zm150.7-38.5c6.4 10.4 11.2 22 14.1 34.6 2.9 12.4 3.6 24.8 2.4 36.8l-55.6-16c-5-1.4-8.1-6.7-6.9-11.7 .5-2.1 1.6-3.8 3.1-5.1l42.9-38.5zm-113.8 4.4c-.2 5.2-4.7 9.4-9.9 9.4-2.1 0-4.1-.7-5.7-1.8l-47.3-33.4c14.5-14.3 33.1-24.8 54.5-29.7 3.9-.9 7.8-1.6 11.7-2l-3.3 57.6zm30.8-57.6c25 3.1 48.1 14.4 65.8 31.7l-47.1 33.2c-4.2 3-10 2.3-13.3-1.8-1.3-1.7-2-3.6-2.1-5.6l-3.3-57.5zM254.5-1.3c5.9-.3 11.7 .9 17 3.4L455 89.7c4.7 2.3 8.9 5.6 12.2 9.7s5.6 8.8 6.8 13.9l45.3 196.9c1.2 5.1 1.2 10.4 0 15.5s-3.5 9.9-6.8 13.9L385.6 497.6c-3.3 4.1-7.5 7.4-12.2 9.6s-10 3.4-15.2 3.4l-203.6 0c-5.3 0-10.5-1.2-15.2-3.4s-8.9-5.5-12.2-9.6L.2 339.7c-.7-.9-1.4-1.8-2-2.8-2.6-3.9-4.3-8.3-5.1-12.9s-.7-9.3 .3-13.8L38.7 113.4c1.2-5.1 3.5-9.9 6.8-13.9s7.5-7.4 12.2-9.7L241.1 2.1c4.2-2 8.8-3.2 13.4-3.4zm1.8 67c-6.1 0-11 5.5-11 12.2 0 .1 0 .2 0 .3 0 .9-.1 2 0 2.8 .1 3.9 1 6.9 1.5 10.4 .9 7.7 1.7 14 1.2 19.9-.5 2.2-2.1 4.3-3.6 5.7l-.2 4.6c-6.6 .5-13.2 1.6-19.8 3.1-28.5 6.5-53.1 21.1-71.8 41-1.2-.8-3.3-2.3-4-2.8-2 .3-4 .9-6.5-.6-4.9-3.3-9.4-7.9-14.8-13.4-2.5-2.6-4.3-5.1-7.2-7.7-.7-.6-1.7-1.4-2.4-2-2.3-1.8-5-2.8-7.7-2.9-3.4-.1-6.6 1.2-8.8 3.9-3.8 4.7-2.5 12 2.7 16.2 .1 0 .1 .1 .2 .1 .7 .6 1.6 1.3 2.3 1.8 3.1 2.3 6 3.5 9.1 5.3 6.6 4.1 12 7.4 16.3 11.5 1.7 1.8 2 4.9 2.2 6.3l3.5 3.2c-18.8 28.4-27.6 63.4-22.4 99.1l-4.6 1.3c-1.2 1.6-2.9 4-4.7 4.7-5.6 1.8-12 2.4-19.7 3.3-3.6 .3-6.7 .1-10.5 .8-.8 .2-2 .4-2.9 .7l-.1 0-.2 .1c-6.5 1.6-10.7 7.5-9.3 13.4 1.3 5.9 7.7 9.4 14.2 8l.2 0c.1 0 .1-.1 .2-.1 .9-.2 2.1-.4 2.8-.6 3.8-1 6.5-2.5 9.8-3.8 7.3-2.6 13.3-4.8 19.2-5.6 2.4-.2 5 1.5 6.3 2.2l4.8-.8c11 34.1 34.1 61.7 63.3 79.1l-2 4.8c.7 1.9 1.5 4.4 1 6.2-2.1 5.5-5.8 11.4-9.9 17.9-2 3-4.1 5.3-5.9 8.8-.4 .8-1 2.1-1.4 3-2.8 6-.8 13 4.7 15.6 5.5 2.6 12.2-.1 15.2-6.2l0 0c.4-.9 1-2 1.4-2.8 1.6-3.6 2.1-6.6 3.2-10.1 2.9-7.3 4.5-14.9 8.5-19.7 1.1-1.3 2.9-1.8 4.8-2.3l2.5-4.5c25.5 9.8 54 12.4 82.5 5.9 6.5-1.5 12.8-3.4 18.8-5.7 .7 1.2 2 3.6 2.3 4.2 1.9 .6 3.9 .9 5.6 3.4 3 5.1 5 11.2 7.5 18.5 1.1 3.4 1.6 6.5 3.2 10.1 .4 .8 1 2 1.4 2.8 2.9 6.1 9.7 8.8 15.2 6.2 5.4-2.6 7.5-9.6 4.7-15.6-.4-.9-1-2.1-1.4-3-1.8-3.4-3.9-5.7-5.9-8.7-4.2-6.5-7.6-11.9-9.8-17.4-.9-2.8 .2-4.6 .8-6.5-.4-.5-1.3-3.2-1.8-4.4 30.4-17.9 52.8-46.6 63.3-79.6 1.4 .2 3.9 .7 4.7 .8 1.7-1.1 3.2-2.5 6.2-2.3 5.9 .8 11.9 3 19.2 5.6 3.4 1.3 6.1 2.8 9.8 3.8 .8 .2 1.9 .4 2.8 .6 .1 0 .1 0 .2 .1l.2 0c6.5 1.4 12.8-2.2 14.2-8s-2.8-11.8-9.3-13.4c-.9-.2-2.3-.6-3.2-.7-3.8-.7-6.9-.5-10.5-.8-7.7-.8-14-1.4-19.7-3.2-2.3-.9-4-3.7-4.8-4.8l-4.4-1.3c2.3-16.6 1.7-33.9-2.3-51.3-4-17.5-11.1-33.5-20.6-47.6 1.1-1 3.3-2.9 3.9-3.5 .2-2 0-4 2.1-6.2 4.3-4.1 9.8-7.4 16.3-11.5 3.1-1.8 6-3 9.1-5.3 .7-.5 1.7-1.3 2.4-1.9 5.3-4.2 6.5-11.4 2.7-16.2s-11.1-5.2-16.4-1c-.7 .6-1.8 1.4-2.4 2-2.9 2.5-4.8 5-7.2 7.7-5.4 5.5-9.9 10.1-14.8 13.4-2.1 1.2-5.3 .8-6.7 .7l-4.2 3c-23.8-25-56.2-41-91.2-44.1-.1-1.5-.2-4.1-.2-4.9-1.4-1.4-3.2-2.5-3.6-5.5-.5-5.9 .3-12.3 1.3-19.9 .5-3.6 1.4-6.6 1.5-10.4 0-.9 0-2.2 0-3.1 0-6.7-4.9-12.2-11-12.2z"></path>
        </svg>
      `,
    rectangleTerminal: `
        <svg viewBox="0 0 512 512" aria-hidden="true" focusable="false">
          <path fill="currentColor" d="M0 128C0 92.7 28.7 64 64 64l384 0c35.3 0 64 28.7 64 64l0 256c0 35.3-28.7 64-64 64L64 448c-35.3 0-64-28.7-64-64L0 128zm103 31c-9.4 9.4-9.4 24.6 0 33.9l63 63-63 63c-9.4 9.4-9.4 24.6 0 33.9s24.6 9.4 33.9 0l80-80c9.4-9.4 9.4-24.6 0-33.9l-80-80c-9.4-9.4-24.6-9.4-33.9 0zM248 336c-13.3 0-24 10.7-24 24s10.7 24 24 24l144 0c13.3 0 24-10.7 24-24s-10.7-24-24-24l-144 0z"></path>
        </svg>
      `,
    general: `
        <svg viewBox="0 0 512 512" aria-hidden="true" focusable="false">
          <path fill="currentColor" d="M256 512A256 256 0 1 0 256 0a256 256 0 1 0 0 512zM216 336h24V272H216c-13.3 0-24-10.7-24-24s10.7-24 24-24h48c13.3 0 24 10.7 24 24v88h8c13.3 0 24 10.7 24 24s-10.7 24-24 24H216c-13.3 0-24-10.7-24-24s10.7-24 24-24zm40-208a32 32 0 1 1 0 64 32 32 0 1 1 0-64z"></path>
        </svg>
      `,
    components: `
        <svg viewBox="0 0 512 512" aria-hidden="true" focusable="false">
          <path fill="currentColor" d="M0 416c0 17.7 14.3 32 32 32l54.7 0c12.3 28.3 40.5 48 73.3 48s61-19.7 73.3-48L480 448c17.7 0 32-14.3 32-32s-14.3-32-32-32l-246.7 0c-12.3-28.3-40.5-48-73.3-48s-61 19.7-73.3 48L32 384c-17.7 0-32 14.3-32 32zm128 0a32 32 0 1 1 64 0 32 32 0 1 1 -64 0zM320 256a32 32 0 1 1 64 0 32 32 0 1 1 -64 0zm32-80c-32.8 0-61 19.7-73.3 48L32 224c-17.7 0-32 14.3-32 32s14.3 32 32 32l246.7 0c12.3 28.3 40.5 48 73.3 48s61-19.7 73.3-48l54.7 0c17.7 0 32-14.3 32-32s-14.3-32-32-32l-54.7 0c-12.3-28.3-40.5-48-73.3-48zM192 128a32 32 0 1 1 0-64 32 32 0 1 1 0 64zm73.3-64C253 35.7 224.8 16 192 16s-61 19.7-73.3 48L32 64C14.3 64 0 78.3 0 96s14.3 32 32 32l86.7 0c12.3 28.3 40.5 48 73.3 48s61-19.7 73.3-48L480 128c17.7 0 32-14.3 32-32s-14.3-32-32-32L265.3 64z"></path>
        </svg>
      `,
    backends: `
        <svg viewBox="0 0 512 512" aria-hidden="true" focusable="false">
          <path fill="currentColor" d="M64 32C28.7 32 0 60.7 0 96v64c0 35.3 28.7 64 64 64H448c35.3 0 64-28.7 64-64V96c0-35.3-28.7-64-64-64H64zm280 72a24 24 0 1 1 0 48 24 24 0 1 1 0-48zm48 24a24 24 0 1 1 48 0 24 24 0 1 1 -48 0zM64 288c-35.3 0-64 28.7-64 64v64c0 35.3 28.7 64 64 64H448c35.3 0 64-28.7 64-64V352c0-35.3-28.7-64-64-64H64zm280 72a24 24 0 1 1 0 48 24 24 0 1 1 0-48zm56 24a24 24 0 1 1 48 0 24 24 0 1 1 -48 0z"></path>
        </svg>
      `,
    observability: `
        <svg viewBox="0 0 512 512" aria-hidden="true" focusable="false">
          <path fill="currentColor" d="M0 256a256 256 0 1 1 512 0A256 256 0 1 1 0 256zm320 96c0-26.9-16.5-49.9-40-59.3V88c0-13.3-10.7-24-24-24s-24 10.7-24 24V292.7c-23.5 9.5-40 32.5-40 59.3c0 35.3 28.7 64 64 64s64-28.7 64-64zM144 176a32 32 0 1 0 0-64 32 32 0 1 0 0 64zm-16 80a32 32 0 1 0 -64 0 32 32 0 1 0 64 0zm288 32a32 32 0 1 0 0-64 32 32 0 1 0 0 64zM400 144a32 32 0 1 0 -64 0 32 32 0 1 0 64 0z"></path>
        </svg>
      `,
    nixlConnect: `
        <svg viewBox="0 0 512 512" aria-hidden="true" focusable="false">
          <path fill="currentColor" d="M32 96l320 0V32c0-12.9 7.8-24.6 19.8-29.6s25.7-2.2 34.9 6.9l96 96c6 6 9.4 14.1 9.4 22.6s-3.4 16.6-9.4 22.6l-96 96c-9.2 9.2-22.9 11.9-34.9 6.9s-19.8-16.6-19.8-29.6V160L32 160c-17.7 0-32-14.3-32-32s14.3-32 32-32zM480 352c17.7 0 32 14.3 32 32s-14.3 32-32 32H160v64c0 12.9-7.8 24.6-19.8 29.6s-25.7 2.2-34.9-6.9l-96-96c-6-6-9.4-14.1-9.4-22.6s3.4-16.6 9.4-22.6l96-96c9.2-9.2 22.9-11.9 34.9-6.9s19.8 16.6 19.8 29.6l0 64H480z"></path>
        </svg>
      `
  };

  const selectorConfigurations = [
    {
      id: "user-guide",
      ariaLabel: "Deployment environment",
      variants: [
        {
          id: "kubernetes",
          label: "Kubernetes",
          landingPath: "/kubernetes/getting-started/introduction",
          icon: icons.kubernetes,
        },
        {
          id: "cli",
          label: "Local (CLI)",
          landingPath: "/cli/getting-started/introduction",
          // Keep this aligned with `icon: rectangle-terminal` in index.yml.
          icon: icons.rectangleTerminal,
        },
      ],
      match(pathname) {
        const match = pathname.match(/^(.*)\/(kubernetes|cli)(?:\/|$)/);

        if (match == null) {
          return null;
        }

        return { prefix: match[1], activeVariant: match[2] };
      },
    },
    {
      id: "reference",
      ariaLabel: "Reference category",
      variants: [
        {
          id: "general",
          label: "General",
          landingPath: "/reference/release-artifacts",
          icon: icons.general,
        },
        {
          id: "kubernetes-api",
          label: "Kubernetes API",
          landingPath: "/kubernetes-api/dynamo-graph-deployment",
          icon: icons.kubernetes,
        },
        {
          id: "components",
          label: "Components",
          landingPath: "/components/runtime-configuration",
          icon: icons.components,
        },
        {
          id: "backends",
          label: "Backends",
          landingPath: "/backends/v-llm-configuration",
          icon: icons.backends,
        },
        {
          id: "observability",
          label: "Observability",
          landingPath: "/observability/local-stack",
          icon: icons.observability,
        },
        {
          id: "nixl-connect",
          label: "NIXL Connect",
          landingPath: "/nixl-connect/overview",
          icon: icons.nixlConnect,
        },
      ],
      match(pathname, variants) {
        const match = pathname.match(
          /^(.*)\/(reference|kubernetes-api|components|backends|observability|nixl-connect)(?:\/|$)/,
        );

        if (match == null) {
          return null;
        }

        const activeVariant =
          match[2] === "reference" ? "general" : match[2];

        if (!variants.some(({ id }) => id === activeVariant)) {
          return null;
        }

        return { prefix: match[1], activeVariant };
      },
    },
  ];

  function findConfiguration(pathname) {
    for (const configuration of selectorConfigurations) {
      const route = configuration.match(pathname, configuration.variants);

      if (route != null) {
        return { configuration, route };
      }
    }

    return null;
  }

  function removeCustomSelectors() {
    document.querySelectorAll(`.${CUSTOM_SELECTOR_CLASS}`).forEach((selector) => {
      selector.remove();
    });
  }

  function setSelectedVariant(selector, variants, activeVariant) {
    selector.dataset.activeVariant = activeVariant;
    selector.style.setProperty(
      "--dynamo-variant-active-index",
      String(
        Math.max(0, variants.findIndex(({ id }) => id === activeVariant)),
      ),
    );

    variants.forEach((variant) => {
      const isActive = variant.id === activeVariant;
      const option = selector.querySelector(
        `[data-variant="${variant.id}"]`,
      );

      if (option == null) {
        return;
      }

      option.dataset.state = isActive ? "active" : "inactive";

      if (isActive) {
        option.setAttribute("aria-current", "page");
      } else {
        option.removeAttribute("aria-current");
      }
    });
  }

  function updateSelector(selector, configuration, prefix, activeVariant) {
    setSelectedVariant(selector, configuration.variants, activeVariant);

    configuration.variants.forEach((variant) => {
      const isActive = variant.id === activeVariant;
      const option = selector.querySelector(
        `[data-variant="${variant.id}"]`,
      );

      if (option == null) {
        return;
      }

      option.href = isActive
        ? `${window.location.pathname}${window.location.search}${window.location.hash}`
        : `${prefix}${variant.landingPath}`;
    });
  }

  function buildSelector(configuration, prefix, activeVariant) {
    const selector = document.createElement("nav");
    selector.className = CUSTOM_SELECTOR_CLASS;
    selector.dataset.selectorConfiguration = configuration.id;
    selector.setAttribute("aria-label", configuration.ariaLabel);

    configuration.variants.forEach((variant) => {
      const option = document.createElement("a");
      option.className = "dynamo-variant-selector-option";
      option.dataset.variant = variant.id;

      const icon = document.createElement("span");
      icon.className = "dynamo-variant-selector-icon";
      icon.innerHTML = variant.icon;

      const label = document.createElement("span");
      label.className = "dynamo-variant-selector-label";
      label.textContent = variant.label;

      option.append(icon, label);
      option.addEventListener("click", (event) => {
        if (
          option.dataset.state === "active" ||
          event.button !== 0 ||
          event.metaKey ||
          event.ctrlKey ||
          event.shiftKey ||
          event.altKey
        ) {
          return;
        }

        event.preventDefault();
        setSelectedVariant(selector, configuration.variants, variant.id);
        window.setTimeout(() => {
          window.location.assign(option.href);
        }, 140);
      });

      selector.append(option);
    });

    updateSelector(selector, configuration, prefix, activeVariant);
    return selector;
  }

  function enhanceVariantSelector() {
    const match = findConfiguration(window.location.pathname);

    if (match == null) {
      removeCustomSelectors();
      return;
    }

    const { configuration, route } = match;

    document.querySelectorAll(NATIVE_SELECTOR).forEach((nativeSelector) => {
      const container = nativeSelector.parentElement;

      if (container == null) {
        return;
      }

      let selector = container.querySelector(
        `:scope > .${CUSTOM_SELECTOR_CLASS}`,
      );

      if (
        selector != null &&
        selector.dataset.selectorConfiguration !== configuration.id
      ) {
        selector.remove();
        selector = null;
      }

      if (selector == null) {
        selector = buildSelector(
          configuration,
          route.prefix,
          route.activeVariant,
        );
        container.insertBefore(selector, nativeSelector);
      } else {
        updateSelector(
          selector,
          configuration,
          route.prefix,
          route.activeVariant,
        );
      }

      nativeSelector.setAttribute("aria-hidden", "true");
      nativeSelector.setAttribute("tabindex", "-1");
    });
  }

  let enhancementQueued = false;

  function queueEnhancement() {
    if (enhancementQueued) {
      return;
    }

    enhancementQueued = true;
    window.requestAnimationFrame(() => {
      enhancementQueued = false;
      enhanceVariantSelector();
    });
  }

  const observer = new MutationObserver(queueEnhancement);
  observer.observe(document.documentElement, { childList: true, subtree: true });
  window.addEventListener("popstate", queueEnhancement);
  queueEnhancement();
})();

// Reference-page click-to-copy: [data-dynref-copy] buttons (styled by
// ReferenceStyles.tsx) copy their payload, flash the .dynref-copied state,
// and swap their label to "Copied" for 1.2s. Buttons must carry plain-text
// labels — the swap replaces textContent.
(() => {
  if (typeof document === "undefined") return;
  document.addEventListener("click", (event) => {
    const el = event.target.closest("[data-dynref-copy]");
    if (!el || !navigator.clipboard) return;
    navigator.clipboard.writeText(el.getAttribute("data-dynref-copy"));
    if (el.dataset.dynrefRestore === undefined) {
      // Narrow chips (short tags) would GROW to fit "Copied", causing a width
      // jump — for those, the glyph flip + green state is the whole feedback.
      const swapText = el.offsetWidth >= 72;
      el.dataset.dynrefRestore = el.textContent;
      el.style.minWidth = `${el.offsetWidth}px`;
      if (swapText) el.textContent = "Copied";
      el.classList.add("dynref-copied");
      window.setTimeout(() => {
        if (swapText) el.textContent = el.dataset.dynrefRestore;
        delete el.dataset.dynrefRestore;
        el.style.minWidth = "";
        el.classList.remove("dynref-copied");
      }, 1200);
    }
  });
})();

// Hash deep-links into closed accordions: when the URL hash targets an anchor
// (e.g. #v120) that lives inside — or immediately before — a closed <details>
// accordion, open it and re-scroll the anchor into view. Generic: no
// page-specific ids; runs on load, on every hashchange, and (because the app
// hydrates after DOMContentLoaded) via a self-disconnecting MutationObserver
// that waits for the anchor to appear.
(() => {
  if (typeof document === "undefined") return;

  // Returns true once the anchor exists (whether or not an accordion needed
  // opening), so pending observers know to stand down.
  function openAccordionForHash() {
    const hash = window.location.hash;
    if (!hash || hash.length < 2) return true;
    const el = document.getElementById(hash.slice(1));
    if (el == null) return false;

    // Ancestor <details> first; otherwise the first <details> among the next
    // few forward siblings (anchor placed just before its accordion).
    let details = el.closest("details");
    if (details == null) {
      let sibling = el.nextElementSibling;
      for (let i = 0; i < 3 && sibling != null; i += 1) {
        const candidate =
          sibling.tagName === "DETAILS" ? sibling : sibling.querySelector("details");
        if (candidate != null) {
          details = candidate;
          break;
        }
        sibling = sibling.nextElementSibling;
      }
    }

    if (details != null && !details.open) {
      details.open = true;
      window.requestAnimationFrame(() => {
        el.scrollIntoView();
      });
      // Hydration (and the accordion's own hash rewrite) can reset the
      // scroll position after our first scroll — re-scroll once, later, if
      // the anchor fell back out of view.
      window.setTimeout(() => {
        const box = el.getBoundingClientRect();
        if (box.top < 0 || box.top > window.innerHeight) {
          el.scrollIntoView();
        }
      }, 400);
    }

    return true;
  }

  let observer = null;
  let observerDeadline = null;

  function stopObserving() {
    if (observer != null) {
      observer.disconnect();
      observer = null;
    }
    if (observerDeadline != null) {
      window.clearTimeout(observerDeadline);
      observerDeadline = null;
    }
  }

  // Try now; if the anchor is not rendered yet (client hydration), watch the
  // DOM until it appears, giving up after 10s so the observer never lingers.
  function openAccordionWhenReady() {
    stopObserving();
    if (openAccordionForHash()) return;

    observer = new MutationObserver(() => {
      if (openAccordionForHash()) stopObserving();
    });
    observer.observe(document.documentElement, { childList: true, subtree: true });
    observerDeadline = window.setTimeout(stopObserving, 10000);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", openAccordionWhenReady);
  } else {
    openAccordionWhenReady();
  }
  window.addEventListener("hashchange", openAccordionWhenReady);
})();
