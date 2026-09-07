/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

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
