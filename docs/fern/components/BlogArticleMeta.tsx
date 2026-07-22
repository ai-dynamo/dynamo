/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
"use client";

import { useEffect, useState } from "react";

type BlogAuthor = {
  name: string;
  href?: string;
  github?: string;
};

type BlogArticleMetaProps = {
  authors: BlogAuthor[];
  category: string;
  date: string;
  readTime: string;
};


const COVER_LABELS: Record<string, [string, string, string]> = {
  "Agentic AI": ["Harness", "KV Router", "Tools"],
  Architecture: ["Frontend", "KV Router", "Cache"],
  Ecosystem: ["Frontend", "Backend", "Stream"],
  Engineering: ["KV Events", "Indexer", "Route"],
  Kubernetes: ["Checkpoint", "Restore", "Ready"],
  Simulation: ["Workload", "DynoSim", "Frontier"],
};

function BlogArticleCover({ category }: { category: string }) {
  const labels = COVER_LABELS[category] ?? ["Request", "Dynamo", "Response"];
  const variant = category.toLowerCase().replace(/[^a-z0-9]+/g, "-");

  return (
    <div className={`dynamo-blog-cover dynamo-blog-cover--${variant}`} aria-hidden="true">
      <span className="dynamo-blog-cover__grid" />
      <span className="dynamo-blog-cover__glow dynamo-blog-cover__glow--one" />
      <span className="dynamo-blog-cover__glow dynamo-blog-cover__glow--two" />
      <span className="dynamo-blog-cover__beam" />
      <div className="dynamo-blog-cover__chrome">
        <span />
        <span />
        <span />
        <small>NVIDIA DYNAMO · {category.toUpperCase()}</small>
      </div>
      <div className="dynamo-blog-cover__flow">
        {labels.map((label, index) => (
          <div className="dynamo-blog-cover__node" key={label}>
            <small>0{index + 1}</small>
            <strong>{label}</strong>
            <span><i /></span>
          </div>
        ))}
      </div>
      <div className="dynamo-blog-cover__wordmark">D</div>
    </div>
  );
}

function ShareIcon() {
  return (
    <svg viewBox="0 0 20 20" aria-hidden="true">
      <circle cx="5" cy="10" r="2" />
      <circle cx="15" cy="5" r="2" />
      <circle cx="15" cy="15" r="2" />
      <path d="m6.8 9 6.4-3M6.8 11l6.4 3" fill="none" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  );
}

function initials(name: string) {
  if (name === "NVIDIA Dynamo team") return "D";

  return name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0])
    .join("")
    .toUpperCase();
}

function AuthorAvatar({ author }: { author: BlogAuthor }) {
  const label = author.github ? `${author.name} (@${author.github})` : author.name;
  const content = author.github ? (
    <img
      src={`https://github.com/${author.github}.png?size=64`}
      alt=""
      loading="lazy"
    />
  ) : (
    <span>{initials(author.name)}</span>
  );

  if (author.href) {
    return (
      <a
        className="dynamo-blog-author-avatar"
        href={author.href}
        target="_blank"
        rel="noopener noreferrer"
        title={label}
        aria-label={label}
      >
        {content}
      </a>
    );
  }

  return (
    <span className="dynamo-blog-author-avatar" title={label} aria-label={label}>
      {content}
    </span>
  );
}

export function BlogArticleMeta({ authors, category, date, readTime }: BlogArticleMetaProps) {
  const [shareLabel, setShareLabel] = useState("Share");

  useEffect(() => {
    const images = Array.from(
      document.querySelectorAll<HTMLImageElement>(
        "article:has(.dynamo-blog-article) img:not(.dynamo-blog-author-avatar img)",
      ),
    );

    if (images.length === 0) return;

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduceMotion || !("IntersectionObserver" in window)) {
      images.forEach((image) => image.dataset.revealed = "true");
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          (entry.target as HTMLImageElement).dataset.revealed = "true";
          observer.unobserve(entry.target);
        });
      },
      { rootMargin: "0px 0px -8%", threshold: 0.08 },
    );

    images.forEach((image, index) => {
      image.classList.add("dynamo-blog-image-reveal");
      image.style.setProperty("--dynamo-blog-image-delay", `${Math.min(index, 3) * 45}ms`);
      observer.observe(image);
    });

    return () => observer.disconnect();
  }, []);

  async function shareArticle() {
    const shareData = { title: document.title, url: window.location.href };

    try {
      if (navigator.share) {
        await navigator.share(shareData);
        return;
      }

      await navigator.clipboard.writeText(window.location.href);
      setShareLabel("Link copied");
      window.setTimeout(() => setShareLabel("Share"), 1800);
    } catch (error) {
      if ((error as Error).name !== "AbortError") {
        setShareLabel("Copy failed");
        window.setTimeout(() => setShareLabel("Share"), 1800);
      }
    }
  }

  return (
    <div className="dynamo-blog-article">
      <div className="dynamo-blog-article__byline">
        <span className="dynamo-blog-article__category">{category}</span>
        <div className="dynamo-blog-author-stack" aria-label="Authors">
          {authors.map((author) => (
            <AuthorAvatar author={author} key={author.name} />
          ))}
        </div>
        <span className="dynamo-blog-article__authors">
          By {authors.map((author, index) => {
            const separator = index === 0 ? "" : index === authors.length - 1 ? " and " : ", ";
            const name = author.href ? (
              <a href={author.href} target="_blank" rel="noopener noreferrer">{author.name}</a>
            ) : author.github ? (
              <a href={`https://github.com/${author.github}`} target="_blank" rel="noopener noreferrer">{author.name}</a>
            ) : (
              author.name
            );
            return <span key={author.name}>{separator}{name}</span>;
          })}
        </span>
      </div>
      <div className="dynamo-blog-article__details">
        <div className="dynamo-blog-meta-line">
          <time>{date}</time>
          <span aria-hidden="true">·</span>
          <span>{readTime}</span>
        </div>
        <div className="dynamo-blog-article__actions">
          <button type="button" onClick={shareArticle}>
            <ShareIcon /> {shareLabel}
          </button>
          <a
            href="https://github.com/ai-dynamo/dynamo/subscription"
            target="_blank"
            rel="noopener noreferrer"
          >
            Subscribe
          </a>
        </div>
      </div>
      <BlogArticleCover category={category} />
    </div>
  );
}
