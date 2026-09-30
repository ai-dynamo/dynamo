/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Shared card styles for the two publication lists: EcosystemPublications
 * (partner articles) and ResearchPublications (papers). They live on separate
 * pages now, so the CSS cannot sit inside either one — both render this.
 *
 * Named *Styles.tsx deliberately: check_style_components.py guards CSS
 * template literals by that filename, so keeping the CSS here means it is
 * covered without having to name the file in the hook's pattern.
 *
 * Builds on the --dynamo-blog-* custom properties from BlogStyles.tsx, so a
 * page rendering this must render <BlogStyles /> too.
 */
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

/**
 * Trailing external-link arrow on a card title. Shared, because both lists use
 * it and a second copy is a second thing to drift.
 */
export function ExternalMark() {
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

export function PublicationsStyles() {
  return <style dangerouslySetInnerHTML={{ __html: PUBLICATIONS_CSS }} />;
}

export default PublicationsStyles;
