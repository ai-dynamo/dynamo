/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * WelcomeHero — interactive continuation of the Welcome page heading.
 *
 * Fern renders the page title and subtitle from frontmatter. This component
 * starts immediately below them with a rotating Dynamo statement, the primary
 * quickstart action, a community notification stack, and the terminal demonstration.
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

function SlackIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="slack-icon">
      <path d="M9.2 3.5a2 2 0 0 1 2 2v4h-2a2 2 0 0 1 0-4V3.5Zm0 7.5H5.5a2 2 0 1 0 0 4h3.7v-4Zm1.6 0h4v-5.5a2 2 0 1 1 4 0V11h-8Zm4 1.6h3.7a2 2 0 1 1 0 4h-3.7v-4Zm-1.6 0h-4v5.9a2 2 0 1 0 4 0v-5.9Z" />
    </svg>
  );
}

function WeChatIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M10.2 5C5.7 5 2.5 7.7 2.5 11c0 1.9 1.1 3.6 2.9 4.7l-.7 2.2 2.6-1.3c.9.2 1.8.4 2.9.4 4.5 0 7.7-2.7 7.7-6S14.7 5 10.2 5Z" />
      <path d="M13.7 9.2c4.4 0 7.8 2.5 7.8 5.8 0 1.8-1 3.4-2.8 4.5l.6 2-2.4-1.2c-1 .3-2 .5-3.2.5-3.8 0-6.8-1.9-7.6-4.5" />
      <path d="M7.4 9.3h.1m5.3 0h.1m-.5 4.2h.1m5 0h.1" />
    </svg>
  );
}

function CalendarIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M7 3v3m10-3v3M4 9h16M5 5h14a1 1 0 0 1 1 1v14H4V6a1 1 0 0 1 1-1Z" />
      <path d="M8 13h3v3H8z" />
    </svg>
  );
}

function CommunityRail() {
  const notifications = [
    {
      app: "CNCF Slack",
      message: "Join the #ai-dynamo channel",
      href: "https://communityinviter.com/apps/cloud-native/cncf",
      icon: <SlackIcon />,
      tone: "slack",
    },
    {
      app: "WeChat",
      message: "Ask to join the Dynamo community group",
      href: "/dynamo/dev/contribution-guide.zh-CN",
      icon: <WeChatIcon />,
      tone: "wechat",
    },
    {
      app: "Calendar",
      message: "See upcoming community events",
      href: CALENDAR_URL,
      icon: <CalendarIcon />,
      tone: "calendar",
    },
  ];

  return (
    <nav
      className="dynamo-welcome__community"
      aria-label="Dynamo community links"
    >
      {notifications.map(({ app, message, href, icon, tone }) => (
        <a
          key={app}
          className={`dynamo-welcome__notification dynamo-welcome__notification--${tone}`}
          href={href}
          target={href.startsWith("http") ? "_blank" : undefined}
          rel={href.startsWith("http") ? "noopener noreferrer" : undefined}
        >
          <span className="dynamo-welcome__notification-icon">{icon}</span>
          <span className="dynamo-welcome__notification-copy">
            <span className="dynamo-welcome__notification-app">
              {app}
              <small>now</small>
            </span>
            <span>{message}</span>
          </span>
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

      <CommunityRail />
    </div>
  );
}

export default WelcomeHero;
