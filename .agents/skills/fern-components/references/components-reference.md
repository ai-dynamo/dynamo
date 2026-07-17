# Fern Components Reference

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

Full syntax, props, variants, and examples for every Fern built-in component. Source of truth:
[`fern-api/docs` component library](https://github.com/fern-api/docs/tree/main/fern/products/docs/pages/component-library).
Read the relevant section before authoring; match prop names and casing exactly.

**Reminder:** components render only in `.mdx`. In `.md` pages in this repo, use GitHub-style callouts
(`> [!NOTE]`) — nothing else. MDX attributes are `camelCase` (`autoPlay`, `showLineNumbers`) and use
`className`, not `class`.

## Contents

- [Callout](#callout)
- [Badge](#badge)
- [Accordion / AccordionGroup](#accordion--accordiongroup)
- [Steps / Step](#steps--step)
- [Tabs / Tab](#tabs--tab)
- [Card / CardGroup](#card--cardgroup)
- [Button](#button)
- [Code blocks (fenced, Code, CodeBlocks, CodeGroup)](#code-blocks)
- [Frame](#frame)
- [Icon](#icon)
- [Files / Folder / File](#files--folder--file)
- [Tables (Sticky / Searchable)](#tables)
- [ParamField](#paramfield)
- [Indent](#indent)
- [Anchor](#anchor)
- [Aside](#aside)
- [Copy](#copy)
- [Download](#download)
- [Tooltip / Template](#tooltip--template)
- [Prompt](#prompt)
- [If](#if)
- [Versions / Version](#versions--version)
- [API Reference: EndpointRequestSnippet](#endpointrequestsnippet)
- [API Reference: EndpointResponseSnippet](#endpointresponsesnippet)
- [API Reference: EndpointSchemaSnippet](#endpointschemasnippet)
- [API Reference: Schema / SchemaSnippet](#schema--schemasnippet)
- [API Reference: RunnableEndpoint](#runnableendpoint)
- [API Reference: WebhookPayloadSnippet](#webhookpayloadsnippet)
- [Reusable snippets (Markdown src)](#reusable-snippets)
- [Custom React components](#custom-react-components)
- [Rich media (images, video, PDF, LaTeX, Mermaid)](#rich-media)

---

## Callout

Highlight important information. Each intent is its own tag; there is also a generic `<Callout intent>`.

```jsx
<Note>Additional context or supplementary information</Note>
<Info>Draws attention to important information</Info>
<Warning>A warning to watch out for</Warning>
<Success>A successful operation or positive outcome</Success>
<Error>A potential error</Error>
<Tip>A helpful tip</Tip>
<Launch>An announcement, styled with the docs' primary accent</Launch>
<Check>A checked status</Check>

<Warning title="Example callout" icon="skull-crossbones">
  Custom title and Font Awesome icon.
</Warning>
```

**Props:** `intent` (required on `<Callout>`: `info|warning|success|error|note|launch|tip|check`),
`title` (string), `icon` (Font Awesome name or React element; defaults per intent), `className`.

Use for a highlighted block; for a one-word status chip use a [Badge](#badge). **In `.md` files** write
`> [!NOTE]` / `[!TIP]` / `[!IMPORTANT]`→Info / `[!WARNING]` / `[!CAUTION]`→Error instead.

---

## Badge

Small inline label for status, version, or metadata.

```jsx
### Plant Care API <Badge intent="info">v2.1.0</Badge> <Badge intent="launch" minimal>New</Badge>

<Badge intent="success" outlined>Success, outlined</Badge>
<Badge intent="error" minimal outlined>Error, outlined + minimal</Badge>
```

**Props:** `intent` (required: `success|note|tip|warning|error|info|launch|check`), `minimal` (bool),
`outlined` (bool). Combine `minimal` + `outlined`. For longer notes use a [Callout](#callout).

---

## Accordion / AccordionGroup

Expandable sections; content stays searchable/SEO-indexed while collapsed.

```jsx
<AccordionGroup>
  <Accordion title="Section 1" defaultOpen={true}>
    Content for section 1, expanded by default.
  </Accordion>
  <Accordion title="Section 2">
    Content for section 2. Can nest components:
    <Note>A callout nested in an accordion.</Note>
  </Accordion>
</AccordionGroup>
```

**`<Accordion>` props:** `title` (string, required), `children` (required), `defaultOpen` (bool,
default false), `id` (string; auto-generated if omitted), `className`. Group multiple with
`<AccordionGroup>`. Accepts rich children (images, `<Frame>`, callouts, code). `title` generates **no**
anchor — add a heading or `<Anchor>` to deep-link.

---

## Steps / Step

Sequential instructions, auto-numbered with anchor links.

```jsx
<Steps>
  <Step title="Install the SDK">
    Install using your package manager.
  </Step>
  <Step title="Add your API key">
    Generate and configure your key.
    <Note>Steps accept nested components.</Note>
  </Step>
</Steps>
```

Alternative: markdown headings inside `<Steps>` become steps (with `toc={true}`, `##`→depth 2,
`###`→depth 3):

```jsx
<Steps toc={true}>
## Install the SDK
First, install it.
## Configure your key
Generate a key.
</Steps>
```

**`<Steps>` props:** `toc` (bool, default false — include steps in the table of contents), `tocDepth`
(`1|2|3`, default 3). **`<Step>` props:** `title` (string), `id` (auto if omitted), `className`.
`title` generates no anchor.

---

## Tabs / Tab

Switchable content panels. Tabs sharing a `language` value sync site-wide (with code blocks too).

```jsx
<Tabs>
  <Tab title="First tab">Content only in the first tab.</Tab>
  <Tab title="Second tab">Content only in the second tab.</Tab>
</Tabs>

<Tabs>
  <Tab title="TypeScript" language="typescript">
    ```typescript
    console.log("syncs across the site");
    ```
  </Tab>
  <Tab title="Python" language="python">
    ```python
    print("syncs across the site")
    ```
  </Tab>
</Tabs>
```

**`<Tab>` props:** `title` (required), `language` (optional — any string; enables global sync),
`children` (required), `id` (auto if omitted), `className`. Without `language`, tabs don't sync. `title`
generates no anchor.

---

## Card / CardGroup

Boxed content with optional icon, image, and link. Grid via `<CardGroup>`.

```jsx
<CardGroup cols={2}>
  <Card title="Watering" icon="regular droplet" href="/learn/docs/…">
    Learn how to water your plants.
  </Card>
  <Card title="Sunlight" icon="regular sun" href="/learn/docs/…">
    Sunlight requirements.
  </Card>
</CardGroup>

<Card title="Custom icon" icon={<img src="https://…/icon.png" alt="icon"/>} href="…">Image icon.</Card>
<Card title="With image" src="https://…/photo.jpg" imagePosition="left">Image beside content.</Card>
```

**`<CardGroup>` props:** `cols` (number, default 2). **`<Card>` props:** `title`, `icon` (Font Awesome
class e.g. `"brands python"`, `./path.svg`, or custom `<img>`), `href` (makes whole card clickable),
`iconPosition` (`top|left`, default top), `iconSize` (number; pixels = `iconSize * 4`, default 8),
`color` / `lightModeColor` / `darkModeColor` (hex), `src` (image URL), `imagePosition`
(`top|left|right|bottom`, default top), `imageWidth` / `imageHeight` (only to override auto-fit).
`title` generates no anchor.

---

## Button

Interactive button or link.

```jsx
<Button intent="primary">Primary</Button>
<Button intent="success" outlined large rounded>Large, outlined, rounded</Button>
<Button icon="download">Download</Button>
<Button rightIcon="arrow-right">Continue</Button>
<Button href="/learn/docs/getting-started/overview">Link button</Button>
<Button disabled>Disabled</Button>
```

**Props:** `intent` (`none|primary|success|warning|danger`, default none), `disabled`, `small`, `large`,
`icon` / `rightIcon` (Font Awesome name or node), `href`, `target` (`_self|_blank|_parent|_top`),
`minimal`, `outlined`, `full`, `rounded`, `active`, `mono`, `text` (node), `className`. All booleans
default false.

---

## Code blocks

Fenced ``` ``` ``` with an optional language and attributes after the identifier. Supported languages
(Shiki): `javascript`(`js`,`node`), `python`, `java`, `typescript`(`ts`), `csharp`, `cpp`, `c`, `php`,
`go`, `rust`, `ruby`, `swift`, `kotlin`, `sql`, `bash`(`shell`,`sh`), `curl`, `markdown`(`md`), `http`.
Unlisted languages render without highlighting.

    ```js Snippet title
    console.log("hello world")
    ```

    ```javascript {2-4,6}      // line highlight (inclusive; numbers/ranges/lists)
    ```

    ```javascript focus={2-4}  // dim non-focused lines (or comment [!code focus])
    ```

    ```python maxLines=10      // scroll after N lines; 0 disables the default 20-line cap
    ```

    ```txt wordWrap            // wrap long lines instead of horizontal scroll
    ```

    ```bash showLineNumbers={false}   // hide gutter + $/> prefixes
    ```

    ```javascript startLine={6}       // number the first line as 6
    ```

**Deep links** — make tokens clickable (JSON map; regex keys need `\\` escapes):

```jsx
<CodeBlock links={{"PlantClient": "/learn/docs/…", "/get\\w+/": "/learn/docs/…"}}>
```typescript
import { PlantClient } from "@plantstore/sdk";
```
</CodeBlock>
```

**Embed a file** with `<Code src>` (local path relative to docs, or a full raw URL). Same props as
`<CodeBlock>` plus `src` (required) and `lines` (e.g. `[1-3,5,7-10]`, 1-indexed):

```jsx
<Code src="snippets/example-code.js" />
<Code src="https://raw.githubusercontent.com/…/check.yml" title="Workflow" language="yaml" maxLines={15} />
```

**Tabbed / grouped code** — `<CodeBlocks>` renders a tabbed set; `<CodeGroup>` with `for=` creates a
custom sync group (e.g. npm/pnpm/yarn) independent of language:

```jsx
<CodeBlocks>
  ```ruby title="hello.rb"
  puts "Hello"
  ```
  ```php title="hello.php"
  <?php echo "Hello"; ?>
  ```
</CodeBlocks>

<CodeGroup>
  ```bash title="npm" for="npm"
  npm install plantstore
  ```
  ```bash title="pnpm" for="pnpm"
  pnpm add plantstore
  ```
</CodeGroup>
```

**Code block / `<CodeBlock>` props:** `language`, `title` (or inline after language, or `filename=`),
`highlight` (`{2-4,6}`), `focus`, `startLine`, `maxLines` (default 20; 0 = no limit), `wordWrap`
(default false), `showLineNumbers` (default true), `for` (sync group), `links` (object). **`<Code>`
adds:** `src` (required), `lines` (number[]). Same-language blocks auto-sync across the site.

---

## Frame

Container for an image/video with optional caption and background.

```jsx
<Frame caption="Beautiful mountains" background="subtle">
  <img src="./path/to/image.jpg" alt="Mountains"/>
</Frame>
```

**Props:** `caption` (string), `background` (`subtle|default`, default default), `className`.

---

## Icon

Font Awesome icon (all Pro styles) or a custom SVG.

```jsx
<Icon icon="seedling" /> Basic
<Icon icon="warning" size="7" /> Large
<Icon icon="check" color="#22C55E" /> Colored
<Icon icon="fa-duotone fa-heart" /> Duotone
<Icon icon="fa-brands fa-github" /> Brand
<Icon icon="./images/fern-leaf.svg" /> Custom SVG
```

**Props:** `icon` (required — Font Awesome name/class or `./path.svg`), `color` /
`lightModeColor` / `darkModeColor`, `size` (number; pixels = `size * 4`, default 4), `className`.

---

## Files / Folder / File

Visual file/directory tree.

```jsx
<Files>
  <Folder name="components" defaultOpen highlighted>
    <File name="accordion.mdx" comment="Collapsible sections"/>
    <File name="button.mdx" href="/docs/writing-content/components/button" />
    <Folder name="ui">
      <File name="card.mdx" />
    </Folder>
  </Folder>
  <File name="README.md" highlighted comment="Start here"/>
</Files>
```

**`<Files>`:** `children` (required), `className`. **`<Folder>`:** `name` (required), `defaultOpen`
(default false), `href`, `highlighted` (default false), `comment` (auto-prefixed `#`), `className`,
`children` (only `<File>`/`<Folder>`). **`<File>`:** `name` (required), `href`, `highlighted`,
`comment`, `className`. To indent non-file content, use [Indent](#indent).

---

## Tables

Plain Markdown tables work for short data. For longer/interactive data:

```jsx
<StickyTable>
| Plant | Light | Water |
|-------|-------|-------|
| Fern | Partial shade | Weekly |
</StickyTable>

<SearchableTable placeholder="Search plants…">
| Plant | Light | Water |
|-------|-------|-------|
| Fern | Partial shade | Weekly |
</SearchableTable>

<StickySearchableTable> … </StickySearchableTable>
```

Or HTML attributes on a `<table className="fern-table">`: `sticky`, `searchable`, `placeholder`
(combine `searchable sticky`). **Props:** `children` (a markdown table, required); `placeholder`
(searchable variants, default "Search…"). Style sticky tables via the `.fern-table.sticky` selector.

---

## ParamField

Document one API parameter / field / config key with consistent formatting. The standard field-doc row.

```jsx
<ParamField path="username" type="string" required={true}>
  The user's display name.
</ParamField>
<ParamField path="limit" type="number" default="50">
  Max items to return.
</ParamField>
<ParamField path="status" type="'active' | 'inactive' | 'pending'" default="active">
  Account status (union type).
</ParamField>
<ParamField path="api_key" type="string" deprecated={true}>
  Use OAuth instead.
</ParamField>
```

**Props:** `path` (name), `type` (required), `required` (bool → "Required" label), `default` (string),
`deprecated` (bool), `toc` (bool, default false — include in TOC). Pair with [Indent](#indent) for
nested object hierarchies.

---

## Indent

Left-indent any content (unlike `<Folder>`, which only takes files). Nest for multi-level hierarchies.

```jsx
<ParamField path="config" type="object" required={true}>Configuration object.</ParamField>
<Indent>
  <ParamField path="config.database" type="object" required={true}>DB settings.</ParamField>
  <Indent>
    <ParamField path="config.database.host" type="string" required={true}>Hostname.</ParamField>
  </Indent>
</Indent>
```

**Props:** `children` (required), `className` (includes `fern-indent` by default).

---

## Anchor

Linkable anchor for non-heading content (paragraphs, tables, code blocks). Headings already
auto-generate anchors — don't wrap them.

```jsx
<Anchor id="data">This sentence has a custom anchor</Anchor>. Link it via `#data`.

<Anchor id="api-endpoints">
| Endpoint | Method |
|----------|--------|
| `/plants` | GET |
</Anchor>
Link to the [table](#api-endpoints).
```

**Props:** `id` (required). Leave a blank line inside `<Anchor>` around tables/code fences.

---

## Aside

Sticky container that floats content to the right as the reader scrolls. Good for a code/endpoint
snippet beside prose.

```jsx
<Aside>
  <EndpointRequestSnippet endpoint='POST /chat/{domain}' />
</Aside>
```

No documented props beyond children.

---

## Copy

Make inline text click-to-copy. `clipboard=` copies a different value than what's shown.

```jsx
The current version is <Copy>v2.0</Copy>.
Install with <Copy clipboard="npm install -g bamboo-leaf-cli">npm install</Copy>.
```

**Props:** `children` (required — displayed text), `clipboard` (string — value actually copied).

---

## Download

Let users download an asset. Single file via `src` (must live in the `fern` folder), or bundle multiple
public URLs into a ZIP via `sources`.

```jsx
<Download src="./all-about-ferns.pdf">Download PDF</Download>

<Download src="./all-about-ferns.pdf"><Button intent="primary">Download PDF</Button></Download>

<Download sources={["https://…/logo-dark.svg","https://…/logo-light.svg"]} filename="brand.zip">
  <Button intent="primary">Download brand assets</Button>
</Download>
```

**Props:** `src` (single file, relative path, in `fern/`) **or** `sources` (string[] of public URLs —
mutually exclusive with `src`), `children` (required — the click target, usually a `<Button>`),
`filename` (ZIP name or override). If any `sources` file fails to fetch, the whole download fails.

---

## Tooltip / Template

Hover explanation for a term (`<Tooltip>`) or for variables inside a code block (`<Template>`).

```jsx
Add <Tooltip tip="A simple explanation">tooltips</Tooltip> to key terms.

You can include <Tooltip tip={<div><strong>Rich</strong> content with <code>code</code></div>} side="right">rich content</Tooltip>.

<Template
  data={{ API_KEY: "123456" }}
  tooltips={{ API_KEY: (<p>Your API key authenticates requests. Keep it secret.</p>) }}
>
```ts
const apiKey = "{{API_KEY}}";
```
</Template>
```

**Text tooltip props:** `tip` (string or node, required), `side` (`top|right|bottom|left`, default top),
`sideOffset` (number, default 4). **Code tooltip (`<Template>`) props:** `data` (object, required — the
values substituted into `{{KEY}}` in the code), `tooltips` (object — keys must match `data`). Style via
`.fern-mdx-tooltip-content` / `.fern-mdx-tooltip-trigger`.

---

## Prompt

Copyable AI prompt card with optional "open in" actions.

```jsx
<Prompt title="Generate a TypeScript SDK" actions={["cursor", "claude", "chatgpt"]}>
Generate a TypeScript SDK from my OpenAPI spec. Follow the quickstart.
</Prompt>

<Prompt
  title="Generate a TypeScript SDK"
  icon="code"
  actions={[{ label: "Open in Perplexity", url: "https://www.perplexity.ai/search?q={prompt}", icon: "magnifying-glass" }, "cursor"]}
>
…prompt text (markdown preserved)…
</Prompt>
```

**Props:** `children` (markdown, required — the prompt), `title` (string), `icon` (Font Awesome name or
image URL; default sparkle), `actions` (`(string|object)[]` — built-ins `cursor|claude|chatgpt`, or
`{label,url,icon?}`; `{prompt}` is the URL-encoded body placeholder; first action = primary button,
rest in a dropdown; copy button always present), `hidePrompt` (bool — title row only; requires
`title`), `singleLine` (bool — truncate to one line).

---

## If

Show/hide content by product, version, or authenticated role. Values match slugs in `docs.yml` / your
RBAC config.

```jsx
<If products={["orchids"]}>Only in the orchids product.</If>
<If versions={["v2"]}>Only in v2.</If>
<If roles={["admin"]}>Only for admins.</If>
<If products={["orchids"]} versions={["v2"]}>Orchids AND v2 (all conditions must match).</If>
<If products={["legacy"]} not>Every product except legacy.</If>
```

**Props:** `products` (string[]), `versions` (string[]), `roles` (string[]), `not` (bool, default
false — invert), `children` (required). Multiple props AND together.

---

## Versions / Version

Inline content switcher; selection persists in a URL query param. Distinct from site-wide versioning
(they can be used together).

```jsx
<Versions paramName="sdk-version">
  <Version version="v2" title="v2.0" default>
    ```bash
    npm install @fern/plant-sdk@2.0.0
    ```
    <Note>v2.0 has breaking changes.</Note>
  </Version>
  <Version version="v1" title="v1.0">
    ```bash
    npm install @fern/plant-sdk@1.0.0
    ```
  </Version>
</Versions>
```

**`<Versions>` props:** `paramName` (string, default `v` — set a unique one when using multiple on a
page), `className`. **`<Version>` props:** `version` (required, unique — used in the URL), `title`
(display; falls back to `version`), `default` (bool), `children` (required), `className`.

---

## API Reference components

These pull live data from an API definition already configured in your `docs.yml`. Endpoints are
addressed as `"METHOD /path"`; for namespaced APIs prefix with `namespace::` (e.g.
`payments::POST /chat/{domain}`). A bonus: when agents fetch the `.md` version of the page, Fern renders
these as structured Markdown / fenced code, so agents get the full spec without parsing HTML.

### EndpointRequestSnippet

Request code sample for an endpoint, with a language dropdown and a Try-It button.

```jsx
<EndpointRequestSnippet endpoint="POST /chat/{domain}" />
<EndpointRequestSnippet endpoint="POST /chat/{domain}" languages={["curl", "python", "typescript"]} />
<EndpointRequestSnippet endpoint="PUT /pet" example="ExampleWithMarkley" />
<EndpointRequestSnippet endpoint="POST /chat/{domain}" hideTryItButton={true} />
```

**Props:** `endpoint` (required), `example` (example name — use its `summary`/`docs` if set), `highlight`
(number | number[] | ranges), `languages` (string[] — filters + orders; includes `payload` for raw
body/query), `hideTryItButton` (bool, default false).

### EndpointResponseSnippet

Response sample for an endpoint.

```jsx
<EndpointResponseSnippet endpoint="POST /chat/{domain}" />
<EndpointResponseSnippet endpoint="GET /pet/{petId}" example="ExampleWithMarkley" />
```

**Props:** `endpoint` (required), `example`, `highlight`.

### EndpointSchemaSnippet

Endpoint schema (params/body/response) as a field breakdown. `selector` narrows it.

```jsx
<EndpointSchemaSnippet endpoint="POST /chat/{domain}" />
<EndpointSchemaSnippet endpoint="POST /chat/{domain}" selector="request.body" />
```

**Props:** `endpoint` (required), `selector` (`request | request.path | request.query | request.body |
response | response.body`). For any named type (not just endpoint schemas) use [Schema](#schema--schemasnippet).

### Schema / SchemaSnippet

`<Schema>` renders a named type's field breakdown; `<SchemaSnippet>` renders its JSON. Pair them to show
both. Only types **referenced by endpoints** are discoverable (not websocket/webhook-only types).

```jsx
<Schema type="AIChatConfig" />
<Schema type="AIChatConfig" include={["model", "provider"]} />
<SchemaSnippet type="AIChatConfig" title="The AIChatConfig Object" />
```

**`<Schema>` props:** `type` (required), `api` (string — which API; else first match), `description`
(bool), `include` (string[]), `exclude` (string[]), `excludeDeprecated` (bool), `className`.
**`<SchemaSnippet>` props:** `type` (required), `title`, `highlight`, `className`.

### RunnableEndpoint

Interactive request builder that makes real HTTP calls from the page.

```jsx
<RunnableEndpoint endpoint="POST /chat/{domain}" />
<RunnableEndpoint endpoint="POST /chat/{domain}" collapsed />
```

**Props:** `endpoint`, `example`, `defaultEnvironment` (matches an `x-fern-server-name`), `readonly`
(string[], e.g. `["environment"]`), `collapsed` (bool, default false), `className`.

### WebhookPayloadSnippet

Webhook payload schema, addressed by `operationId`.

```jsx
<WebhookPayloadSnippet webhook="on-conversation-completed" />
```

**Props:** `webhook` (required — the `operationId`; namespaced as `namespace::name`).

---

## Reusable snippets

Single-source a Markdown fragment: define once under any `snippets/` folder in `fern/`, reference with
`<Markdown src>`. `src` is an **absolute path with the `fern` folder as root** (same from any page).
Supports `{{parameters}}`.

```mdx
<!-- fern/snippets/watering-schedule.mdx -->
<Warning>Water your {{plant}} every {{interval}} days.</Warning>
```

```jsx
<Markdown src="/snippets/peace-lily.mdx" />
<Markdown src="/snippets/watering-schedule.mdx" plant="peace lily" interval="3" />
```

Good for constants (limits, prices, versions), repeated warnings, and standardized blocks. For a
constant, prefer a reusable snippet over a custom React component.

---

## Custom React components

Extend the library with your own SSR'd components when nothing built-in fits. Define `.tsx`/`.jsx`/`.mdx`
in a components dir, register it in `docs.yml`, import and use it in a page.

```tsx
// components/CustomCard.tsx
export const CustomCard = ({ title, text, link, sparkle = false }) => (
  <a href={link} className="block p-6 rounded-lg border">
    <h2>{title} {sparkle && "✨"}</h2>
    <p>{text}</p>
  </a>
);
```

```jsx
// in a page — @/ resolves to the fern folder root
import { CustomCard } from "@/components/CustomCard"
<CustomCard title="MyTitle" text="Hello" link="https://…" />
```

```yml
# docs.yml
experimental:
  mdx-components:
    - ./components
```

Advantages over bundling custom JS: no layout shift, faster load (shares Fern's React), SEO-indexable.
Don't use a component just to define a constant — use a [reusable snippet](#reusable-snippets).

---

## Rich media

Not components, but the standard ways to embed media in `.mdx`. MDX requires `camelCase` attributes
(`autoPlay`, not `autoplay`). Don't use `<embed>` (inconsistent browser support).

**Images** — Markdown or `<img>`; paths relative to the page (`./`, `../`) or the `fern` root (`/`):

```markdown
![Alt text](/assets/images/logo.png "Optional title")
```
```html
<img src="/assets/images/logo.png" width="500px" height="auto" />
```
Attributes: `src`, `alt`, `title`, `width`, `height`, `noZoom` (disable zoom). Wrap in a
[Frame](#frame) for a caption.

**Video** — HTML `<video>`:

```html
<video src="./demo.mp4" poster="./thumb.png" width="854" height="480" autoPlay loop playsInline muted controls></video>
```

**PDF / external site / YouTube / Loom** — `<iframe>`:

```jsx
<iframe src="./all-about-ferns.pdf" width="100%" height="500px" style={{ border: 'none', borderRadius: '0.5rem' }} />
<iframe src="https://www.loom.com/embed/…" width="100%" height="450px" frameBorder="0" allowFullScreen></iframe>
```

**LaTeX** — inline `$…$`, display `$$…$$`. Escape literal dollars as `\$10` to avoid accidental math.

**Mermaid** — a ```` ```mermaid ```` code fence:

    ```mermaid
    erDiagram
        CUSTOMER ||--o{ ORDER : places
    ```
