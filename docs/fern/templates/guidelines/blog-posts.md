---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Writing Blog Posts
subtitle: Publish dated technical and project articles in the Dynamo Blog.
---

Use blog posts for dated editorial content, technical narratives, research summaries, and project
announcements. Keep canonical installation, task, architecture, and reference material in the
corresponding documentation tabs and link to it from the post.

## Step 1: Write the Post

Create an MDX file under `docs/fern/pages/blog/<year>/`; put supporting files under
`docs/fern/pages/blog/_assets/<topic>/`:

```text
docs/fern/pages/blog/2026/my-post.mdx
```

Use kebab-case for the filename. Add the standard SPDX header and metadata:

```yaml
---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Your Blog Post Title
description: A one-sentence summary for search results and social previews.
keywords: Dynamo, inference
last-updated: July 22, 2026
---
```

Import and render the editorial metadata component immediately after the frontmatter:

```mdx
import { BlogArticleMeta } from "@/components/BlogArticleMeta";
import { BlogStyles } from "@/components/BlogStyles";

<BlogStyles />

<BlogArticleMeta
  authors={[
    { name: "Author One", github: "verified-handle" },
    { name: "Author Two", href: "https://developer.nvidia.com/blog/author/verified-profile/" },
  ]}
  category="Engineering"
  date="July 22, 2026"
  readTime="8 min read"
/>
```

Estimate reading time from the body word count at approximately 200 words per minute.
Only add `github` after verifying that the account belongs to the credited author. Git blame and
commit authorship are not substitutes for editorial authorship. Authors without a verified profile
render with an initials avatar and name tooltip.

The `category` selects the article's generated editorial cover palette and flow labels. Supporting
images in the article body receive the standard focus-to-sharp scroll reveal automatically.

## Step 2: Add the Post to the Blog Navigation

Open `docs/fern/index.yml`, find the `blog` tab, and add the post at the correct position in
the reverse-chronological list:

```yaml
contents:
  - page: Your Blog Post Title
    path: pages/blog/2026/my-post.mdx
    slug: my-post
```

Keep the explicit `slug` stable after publication. The public URL is `/dynamo/dev/digest/my-post`.

## Step 3: Add the Post to the Landing Page

Open `docs/fern/components/BlogLanding.tsx` and add an entry to `ARTICLES` in reverse
chronological order:

```tsx
{
  title: "Your Blog Post Title",
  description: "A concise summary of the post.",
  href: "/dynamo/dev/digest/my-post",
  date: "July 22, 2026",
  readTime: "8 min read",
  category: "Engineering",
  art: "indexer",
},
```

Use an existing `art` treatment unless the post needs a deliberately new visual direction. The
landing page uses custom React and CSS rather than Fern card components.

## Step 4: Add the Sidebar Date Label

Add the post date to the Blog archive selectors in `docs/fern/components/BlogStyles.tsx`.
Scope each selector to the stable post slug so the label appears only beside that Blog entry. Follow
a neighboring post and update each sidebar date-label block that contains the existing article slugs.

## Quick Checklist

- [ ] MDX post includes SPDX frontmatter and metadata
- [ ] `BlogArticleMeta` includes author, publication date, category, and reading time
- [ ] Generated cover labels and category palette fit the article
- [ ] Post is added to the `blog` tab in `index.yml` in reverse chronological order
- [ ] Post is added to `BlogLanding.tsx`
- [ ] Sidebar date label is added to `BlogStyles.tsx`
- [ ] Internal links include the `.mdx` extension
- [ ] `fern check --local --warnings` passes without new errors
