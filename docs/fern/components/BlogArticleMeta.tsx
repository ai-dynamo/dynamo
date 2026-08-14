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

export function BlogArticleMeta({ authors, category, date, readTime }: BlogArticleMetaProps) {
  const [shareLabel, setShareLabel] = useState("Share");

  useEffect(() => {
    const images = Array.from(
      document.querySelectorAll<HTMLImageElement>(
        "article:has(.dynamo-blog-article) img",
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
    </div>
  );
}
