# Adding a Blog Post

Follow these steps to publish a new blog post on the Dynamo docs site.

## Step 1: Write the Blog Post

Create a new Markdown file in this folder (`docs/blogs/`):

```text
docs/blogs/my-post-slug.md
```

Use kebab-case for the filename. The filename becomes the URL slug
(e.g., `my-post-slug.md` serves at `/dynamo/dev/blog/my-post-slug`).

Add frontmatter at the top of the file:

```yaml
---
title: Your Blog Post Title
description: A one-sentence summary shown in search results and social previews.
---
```

## Step 2: Add the Post to Navigation

Open `docs/versions/dev.yml` and add a page entry under the **Blog** section:

```yaml
  - section: Blog
    path: ../blogs/index.mdx
    slug: blog
    contents:
      - page: Your Blog Post Title
        path: ../blogs/my-post-slug.md
```

Each new post gets a `- page:` entry in the `contents` list. The `page`
value is the display name in the sidebar; the `path` points to your file.

## Step 3: Add a Card to the Landing Page

Open `docs/blogs/index.mdx` and add a `<Card>` inside the existing `<CardGroup>`:

```mdx
<Card
  title="Your Blog Post Title"
  icon="regular newspaper"
  href="/dynamo/dev/blog/my-post-slug"
>
  A brief summary of what this post covers (1-2 sentences).
</Card>
```

### Card Fields

| Field  | Required | Description                                              |
|--------|----------|----------------------------------------------------------|
| title  | Yes      | Post title displayed on the card                         |
| icon   | No       | Font Awesome icon class (e.g., `regular bolt`)           |
| href   | Yes      | URL path starting with `/dynamo/dev/blog/`               |
| (body) | Yes      | Short summary text inside the Card tags                  |

The `<CardGroup cols={2}>` wrapper is already in `index.mdx`. Just add your
`<Card>` inside it.

Browse icons at https://fontawesome.com/icons (Free tier).

## Step 4 (Optional): Add the Post to the Navbar Dropdown

Open `docs/docs.yml` and add a link under the Blog dropdown in `navbar-links`:

```yaml
navbar-links:
  - type: dropdown
    text: Blog
    links:
      - text: All Posts
        href: /dynamo/dev/blog
      - text: Your Blog Post Title
        href: /dynamo/dev/blog/my-post-slug
```

Only add featured/recent posts here. The dropdown should stay short (5 items max).
Posts that are no longer featured can be removed from the dropdown while remaining
accessible from the landing page and sidebar.

## Quick Checklist

- [ ] Blog post `.md` file created in `docs/blogs/`
- [ ] Page entry added to `docs/versions/dev.yml` under the Blog section
- [ ] Card added to `docs/blogs/index.mdx`
- [ ] (Optional) Link added to the Blog dropdown in `docs/docs.yml`
