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

const STATEMENTS = [
  "works with vLLM, SGLang, and TensorRT-LLM.",
  "supports NVIDIA and AMD GPUs, and Intel XPUs.",
  "runs on Kubernetes, Slurm, or locally.",
  "scales one component or the full stack.",
];

const CALENDAR_URL =
  "https://calendar.google.com/calendar/u/0/r?cid=Y19jMjQ0OGQyZWZiMDllYWMyZGRlZTFmMzQ1MjQxMjQxMzViZDNmNDU1NDg2ODc2OTA1OTEwNWUxOGUxYjk3ZThmQGdyb3VwLmNhbGVuZGFyLmdvb2dsZS5jb20";

function RotatingStatement() {
  const [statementIndex, setStatementIndex] = useState(0);
  const [displayed, setDisplayed] = useState(STATEMENTS[0]);
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
      setStatementIndex(0);
      setDisplayed(STATEMENTS[0]);
      setDeleting(false);
      return;
    }

    const statement = STATEMENTS[statementIndex];
    let delay = deleting ? 34 : 62;
    let next = displayed;
    let nextDeleting = deleting;
    let nextIndex = statementIndex;

    if (!deleting && displayed === statement) {
      delay = 1800;
      nextDeleting = true;
    } else if (deleting && displayed.length === 0) {
      delay = 260;
      nextDeleting = false;
      nextIndex = (statementIndex + 1) % STATEMENTS.length;
    } else if (deleting) {
      next = statement.slice(0, Math.max(0, displayed.length - 1));
    } else {
      next = statement.slice(0, displayed.length + 1);
    }

    const timer = window.setTimeout(() => {
      setDisplayed(next);
      setDeleting(nextDeleting);
      setStatementIndex(nextIndex);
    }, delay);

    return () => window.clearTimeout(timer);
  }, [deleting, displayed, reduceMotion, statementIndex]);

  return (
    <div className="dynamo-welcome__statement">
      <p aria-hidden="true">
        <span>Dynamo </span>
        <span className="dynamo-welcome__typed">{displayed}</span>
        <span className="dynamo-welcome__cursor" />
      </p>
      <p className="dynamo-welcome__sr-only">
        Dynamo works with vLLM, SGLang, and TensorRT-LLM; supports NVIDIA and
        AMD GPUs and Intel XPUs; runs on Kubernetes, Slurm, or locally; and
        scales one component or the full stack.
      </p>
    </div>
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
      label: "Join community Slack",
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
        <a key={label} href={href} target="_blank" rel="noopener noreferrer">
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
