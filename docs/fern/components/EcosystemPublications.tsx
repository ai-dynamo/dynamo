/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * External publications — customer and partner articles about Dynamo.
 *
 * Compact link cards, one per article, in the same two-column rhythm and with
 * the same border, radius and hover treatment as the first-party cards in
 * BlogLanding. They carry less than those do (no summary, no reading time)
 * because we hold no editorial content for them: the card is a signpost to
 * someone else's article, so it shows who published it, when, and the title.
 *
 * Every link leaves the site. Nothing is mirrored or embedded, and that is not
 * only an editorial choice -- most of these publishers send X-Frame-Options or
 * a restrictive frame-ancestors, so an embed would render an empty box for
 * roughly two thirds of the list.
 *
 * Reuses the --dynamo-blog-* custom properties from BlogStyles.tsx, so a page
 * rendering this must render <BlogStyles /> too.
 */
import { PUBLICATIONS, type Publication } from "./publications.data";
import { RESEARCH_PAPERS, type ResearchPaper } from "./research-papers.data";
import { PUBLISHER_LOGOS } from "./publisher-logos.generated";

const PUBLICATIONS_CSS = `
.dynamo-pubs {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1rem;
}

.dynamo-pubs__card {
  display: flex;
  min-width: 0;
  flex-direction: column;
  gap: 0.6rem;
  padding: 1.1rem 1.2rem;
  border: 1px solid var(--dynamo-blog-rule);
  border-radius: 16px;
  background: var(--grayscale-a1);
  text-decoration: none;
  transition: border-color 180ms ease, box-shadow 180ms ease;
}

.dynamo-pubs__card:hover,
.dynamo-pubs__card:focus-visible {
  border-color: var(--dynamo-blog-green);
  box-shadow: 0 10px 26px rgba(0, 0, 0, 0.07);
}

.dark .dynamo-pubs__card:hover,
.dark .dynamo-pubs__card:focus-visible {
  box-shadow: 0 10px 26px rgba(0, 0, 0, 0.4);
}

.dynamo-pubs__top {
  display: flex;
  align-items: center;
  gap: 0.6rem;
}

.dynamo-pubs__mark {
  display: grid;
  flex: none;
  place-items: center;
  width: 1.9rem;
  height: 1.9rem;
  overflow: hidden;
  border-radius: 7px;
  background: rgba(118, 185, 0, 0.14);
  color: var(--dynamo-blog-green);
  font-size: 0.78rem;
  font-weight: 750;
}

/* Logo tile. Most of these marks are dark on transparent and would vanish
   against the dark theme, so the tile keeps a light face in both themes and
   the logo sits on it the way a favicon sits in a browser tab. */
.dynamo-pubs__mark--logo {
  background: #fff;
  box-shadow: inset 0 0 0 1px var(--dynamo-blog-rule);
}

.dynamo-pubs__mark--logo img {
  display: block;
  width: 1.25rem;
  height: 1.25rem;
  object-fit: contain;
  /* Fern's prose styles give every <img> a ~25px vertical margin. Inside a
     30px tile that pushes the logo almost entirely out of the clip box and
     leaves a sliver along the bottom edge. */
  margin: 0 !important;
}

.dynamo-pubs__partner {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  color: var(--dynamo-blog-ink);
  font-size: 0.82rem;
  font-weight: 650;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.dynamo-pubs__date {
  flex: none;
  color: var(--dynamo-blog-muted);
  font-size: 0.75rem;
}

.dynamo-pubs__title {
  color: var(--dynamo-blog-ink);
  font-size: 0.94rem;
  font-weight: 550;
  line-height: 1.45;
}

.dynamo-pubs__card:hover .dynamo-pubs__title,
.dynamo-pubs__card:focus-visible .dynamo-pubs__title {
  color: var(--dynamo-blog-green);
}

/* Fern's prose styles give svg a display of block, which drops the arrow onto a
   line of its own beneath the title instead of trailing the last word. Keep the
   braces out of this comment: a closing one ends the rule early and silently
   drops every declaration after it. */
.dynamo-pubs__title svg {
  display: inline-block !important;
  width: 11px;
  height: 11px;
  margin-left: 0.3rem;
  vertical-align: baseline;
  opacity: 0.5;
}

/* The arrow rides with the closing word. Inline alone is not enough: on a title
   whose last line is full, the arrow is its own break opportunity and wraps to
   a line by itself. */
.dynamo-pubs__title-end {
  white-space: nowrap;
}

/* Second section on the page, so it needs air above it. */
.dynamo-pubs__section + .dynamo-pubs__section {
  margin-top: 4rem;
}

/* Venue chip. Papers have no publisher logo to show, and the venue is the more
   useful signal anyway, so it takes the place the logo tile holds above. */
.dynamo-pubs__venue {
  display: inline-flex;
  flex: none;
  align-items: center;
  padding: 0.16rem 0.5rem;
  border: 1px solid var(--dynamo-blog-rule);
  border-radius: 999px;
  color: var(--dynamo-blog-muted);
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  white-space: nowrap;
}

@media (max-width: 768px) {
  .dynamo-pubs { grid-template-columns: minmax(0, 1fr); }
}
`;

function ExternalMark() {
  return (
    <svg viewBox="0 0 20 20" aria-hidden="true">
      <path
        d="M11 4h5v5M16 4l-7 7"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  );
}

/** Matches the icon BlogLanding puts inside its secondary button. */
function ExternalLinkIcon() {
  return (
    <svg viewBox="0 0 20 20" aria-hidden="true">
      <path d="M11 4h5v5M16 4l-7 7" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M15 11v4a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h4" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

/** Fallback mark: publisher initials, e.g. "Google Cloud" -> "GC". */
function initials(partner: string) {
  return partner
    .replace(/[/&]/g, " ")
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((word) => word[0]?.toUpperCase() ?? "")
    .join("");
}

function PublisherMark({ partner }: { partner: string }) {
  const logo = PUBLISHER_LOGOS[partner];
  if (!logo) {
    return (
      <span className="dynamo-pubs__mark" aria-hidden="true">
        {initials(partner)}
      </span>
    );
  }
  return (
    <span className="dynamo-pubs__mark dynamo-pubs__mark--logo" aria-hidden="true">
      <img src={logo} alt="" loading="lazy" decoding="async" />
    </span>
  );
}

function PublicationCard({ publication }: { publication: Publication }) {
  const { title, url, partner, date } = publication;
  // The arrow is tied to the closing word so the two wrap together and it never
  // ends up alone on a line. Split here rather than in a helper component: a
  // helper returning a fragment does not survive Fern's component transform.
  const words = title.trim().split(/\s+/);
  const lastWord = words.pop() ?? "";
  const leadingWords = words.join(" ");
  return (
    <a
      className="dynamo-pubs__card"
      href={url}
      target="_blank"
      rel="noopener noreferrer"
    >
      <span className="dynamo-pubs__top">
        <PublisherMark partner={partner} />
        <span className="dynamo-pubs__partner">{partner}</span>
        <span className="dynamo-pubs__date">{date}</span>
      </span>
      <span className="dynamo-pubs__title">
        {leadingWords ? `${leadingWords} ` : ""}
        <span className="dynamo-pubs__title-end">
          {lastWord}
          <ExternalMark />
        </span>
      </span>
    </a>
  );
}

function ResearchCard({ paper }: { paper: ResearchPaper }) {
  const { title, url, org, venue, date } = paper;
  const words = title.trim().split(/\s+/);
  const lastWord = words.pop() ?? "";
  const leadingWords = words.join(" ");
  return (
    <a
      className="dynamo-pubs__card"
      href={url}
      target="_blank"
      rel="noopener noreferrer"
    >
      <span className="dynamo-pubs__top">
        <span className="dynamo-pubs__venue">{venue}</span>
        <span className="dynamo-pubs__partner">{org}</span>
        <span className="dynamo-pubs__date">{date}</span>
      </span>
      <span className="dynamo-pubs__title">
        {leadingWords ? `${leadingWords} ` : ""}
        <span className="dynamo-pubs__title-end">
          {lastWord}
          <ExternalMark />
        </span>
      </span>
    </a>
  );
}

export function EcosystemPublications() {
  return (
    <div className="dynamo-blog-home">
      <style dangerouslySetInnerHTML={{ __html: PUBLICATIONS_CSS }} />
      <section
        className="dynamo-blog-latest dynamo-pubs__section"
        id="publications"
        aria-labelledby="publications-heading"
      >
        {/* Same heading structure as BlogLanding's "Latest articles", so the
            two pages in this tab open identically. */}
        <div className="dynamo-blog-section-heading">
          <div className="dynamo-blog-section-heading__copy">
            <span className="dynamo-blog-kicker">From the ecosystem</span>
            <h2 id="publications-heading">External publications</h2>
            <p>
              Deep dives, benchmarks, and deployment write-ups about Dynamo,
              published by the customers and partners running it.
            </p>
          </div>
          <div className="dynamo-blog-section-heading__actions">
            <a
              className="dynamo-blog-button dynamo-blog-button--secondary"
              href="https://github.com/ai-dynamo/dynamo/issues/new"
              target="_blank"
              rel="noopener noreferrer"
            >
              Suggest a publication
              <ExternalLinkIcon />
            </a>
          </div>
        </div>

        <div className="dynamo-pubs">
          {PUBLICATIONS.map((publication) => (
            <PublicationCard key={publication.url} publication={publication} />
          ))}
        </div>
      </section>

      <section
        className="dynamo-blog-latest dynamo-pubs__section"
        id="research"
        aria-labelledby="research-heading"
      >
        <div className="dynamo-blog-section-heading">
          <div className="dynamo-blog-section-heading__copy">
            <span className="dynamo-blog-kicker">From the literature</span>
            <h2 id="research-heading">Research publications</h2>
            <p>
              Papers that use, extend, or benchmark Dynamo, from the teams
              building on it and from the wider systems community.
            </p>
          </div>
        </div>

        <div className="dynamo-pubs">
          {RESEARCH_PAPERS.map((paper) => (
            <ResearchCard key={paper.url} paper={paper} />
          ))}
        </div>
      </section>
    </div>
  );
}

export default EcosystemPublications;
