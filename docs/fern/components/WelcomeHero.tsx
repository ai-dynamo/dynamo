/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * WelcomeHero — interactive continuation of the Welcome page heading.
 *
 * Fern renders the page title and subtitle from frontmatter. This component
 * starts immediately below them with a rotating Dynamo statement, the primary
 * quickstart action, a compact community rail, and the terminal demonstration.
 */
"use client";

import { useEffect, useState } from "react";
import { TerminalDemo } from "./TerminalDemo";

const DESCRIPTORS = [
  "open source.",
  "Kubernetes-native.",
  "modular by design.",
  "built for distributed inference.",
];

const CALENDAR_URL =
  "https://calendar.google.com/calendar/u/0/r?cid=Y19jMjQ0OGQyZWZiMDllYWMyZGRlZTFmMzQ1MjQxMjQxMzViZDNmNDU1NDg2ODc2OTA1OTEwNWUxOGUxYjk3ZThmQGdyb3VwLmNhbGVuZGFyLmdvb2dsZS5jb20";

function RotatingStatement() {
  const [descriptorIndex, setDescriptorIndex] = useState(0);
  const [displayed, setDisplayed] = useState(DESCRIPTORS[0]);
  const [deleting, setDeleting] = useState(false);
  const [reduceMotion, setReduceMotion] = useState(false);

  useEffect(() => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    const updatePreference = () => setReduceMotion(query.matches);
    updatePreference();
    query.addEventListener("change", updatePreference);
    return () => query.removeEventListener("change", updatePreference);
  }, []);

  useEffect(() => {
    if (reduceMotion) {
      setDescriptorIndex(0);
      setDisplayed(DESCRIPTORS[0]);
      setDeleting(false);
      return;
    }

    const descriptor = DESCRIPTORS[descriptorIndex];
    let delay = deleting ? 34 : 62;
    let next = displayed;
    let nextDeleting = deleting;
    let nextIndex = descriptorIndex;

    if (!deleting && displayed === descriptor) {
      delay = 1800;
      nextDeleting = true;
    } else if (deleting && displayed.length === 0) {
      delay = 260;
      nextDeleting = false;
      nextIndex = (descriptorIndex + 1) % DESCRIPTORS.length;
    } else if (deleting) {
      next = descriptor.slice(0, Math.max(0, displayed.length - 1));
    } else {
      next = descriptor.slice(0, displayed.length + 1);
    }

    const timer = window.setTimeout(() => {
      setDisplayed(next);
      setDeleting(nextDeleting);
      setDescriptorIndex(nextIndex);
    }, delay);

    return () => window.clearTimeout(timer);
  }, [deleting, descriptorIndex, displayed, reduceMotion]);

  return (
    <div className="dynamo-welcome__statement">
      <p aria-hidden="true">
        <span>Dynamo is </span>
        <span className="dynamo-welcome__typed">{displayed}</span>
        <span className="dynamo-welcome__cursor" />
      </p>
      <p className="dynamo-welcome__sr-only">
        Dynamo is open source, Kubernetes-native, modular by design, and built
        for distributed inference.
      </p>
    </div>
  );
}

function GitHubIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 2a10 10 0 0 0-3.16 19.49c.5.09.68-.22.68-.48v-1.87c-2.78.6-3.37-1.18-3.37-1.18-.45-1.16-1.11-1.47-1.11-1.47-.91-.62.07-.61.07-.61 1 .07 1.53 1.03 1.53 1.03.9 1.53 2.35 1.09 2.92.83.09-.65.35-1.09.64-1.34-2.22-.25-4.55-1.11-4.55-4.94 0-1.09.39-1.98 1.03-2.68-.1-.25-.45-1.27.1-2.64 0 0 .84-.27 2.75 1.02A9.58 9.58 0 0 1 12 6.82a9.6 9.6 0 0 1 2.5.34c1.9-1.29 2.74-1.02 2.74-1.02.55 1.37.2 2.39.1 2.64.64.7 1.03 1.59 1.03 2.68 0 3.84-2.34 4.68-4.57 4.93.36.31.68.92.68 1.86V21c0 .27.18.58.69.48A10 10 0 0 0 12 2Z" />
    </svg>
  );
}

function ChatIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M7 8h10M7 12h7m-9.5 7 1.2-3.2A7 7 0 1 1 19 12a7 7 0 0 1-9.8 6.4L4.5 19Z" />
    </svg>
  );
}

function CalendarIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M7 3v3m10-3v3M4 9h16M5 5h14a1 1 0 0 1 1 1v14H4V6a1 1 0 0 1 1-1Z" />
    </svg>
  );
}

function CommunityRail() {
  const links = [
    {
      label: "GitHub",
      href: "https://github.com/ai-dynamo/dynamo",
      icon: <GitHubIcon />,
    },
    {
      label: "Community Slack",
      href: "https://communityinviter.com/apps/cloud-native/cncf",
      icon: <ChatIcon />,
    },
    {
      label: "Community calendar",
      href: CALENDAR_URL,
      icon: <CalendarIcon />,
    },
  ];

  return (
    <nav
      className="dynamo-welcome__community"
      aria-label="Dynamo community links"
    >
      {links.map(({ label, href, icon }) => (
        <a
          key={label}
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          aria-label={label}
        >
          {icon}
          <span>{label}</span>
        </a>
      ))}
    </nav>
  );
}

export interface WelcomeHeroProps {
  /** Fern-rewritten path to the asciinema recording. */
  src: string;
}

export function WelcomeHero({ src }: WelcomeHeroProps) {
  return (
    <div className="dynamo-welcome">
      <CommunityRail />

      <section
        className="dynamo-welcome__intro"
        aria-label="Get started with Dynamo"
      >
        <RotatingStatement />
        <a className="dynamo-welcome__cta" href="/dynamo/dev/kubernetes">
          Get started
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="m9 18 6-6-6-6" />
          </svg>
        </a>
      </section>

      <div className="dynamo-welcome__terminal">
        <TerminalDemo
          src={src}
          title="Deploy Qwen3-235B with Dynamo"
          idleTimeLimit={1.5}
          speed={1.0}
          rows="25"
        />
      </div>
    </div>
  );
}

export default WelcomeHero;
