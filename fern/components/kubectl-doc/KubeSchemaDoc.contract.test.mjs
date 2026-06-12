import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import test from "node:test";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(here, "../../..");
const componentRoot = here;
const schemaAssetDir = join(repoRoot, "fern/kubectl-doc-schemas");
const schemaDoc = readFileSync(join(componentRoot, "KubeSchemaDoc.tsx"), "utf8");
const lazySchemaDoc = readFileSync(join(componentRoot, "LazyKubeSchemaDoc.tsx"), "utf8");
const generatedSourcesDoc = readFileSync(join(componentRoot, "schemaSources.generated.ts"), "utf8");

function readSchemaPayload(fileName) {
  const content = readFileSync(join(schemaAssetDir, fileName), "utf8");
  return {
    content,
    payload: JSON.parse(content),
  };
}

function lazySchemaSources() {
  const sources = new Map();
  const sourcePattern = /"([^"]+)": \{ initial: "([^"]+\.json)", full: "([^"]+\.json)" \}/g;
  let match;
  while ((match = sourcePattern.exec(generatedSourcesDoc)) !== null) {
    sources.set(match[1], { initial: match[2], full: match[3] });
  }
  return sources;
}

function slugName(value) {
  return value
    .replace(/([a-z0-9])([A-Z])/g, "$1-$2")
    .replace(/([A-Z]+)([A-Z][a-z])/g, "$1-$2")
    .replace(/([A-Za-z])([0-9])/g, "$1-$2")
    .toLowerCase();
}

test("KubeSchemaDoc consumes the shared kubectl-doc runtime instead of rendering schema lines in React", () => {
  assert.match(schemaDoc, /import \{ kubectlDocStyles \} from "\.\/kubectl-doc-styles";/);
  assert.match(schemaDoc, /const defaultStyleElementID = "kubectl-doc-react-styles";/);
  assert.match(schemaDoc, /function ensureKubectlDocStyles\(styleElementID: string\)/);
  assert.match(schemaDoc, /style\.textContent = kubectlDocStyles;/);
  assert.match(schemaDoc, /import\("\.\/kubectl-doc-runtime\.js"\)/);
  assert.match(schemaDoc, /runtime\.mount\(rootRef\.current, \{/);
  assert.match(schemaDoc, /initialSchema: data/);
  assert.match(schemaDoc, /detailsMode,/);
  assert.match(schemaDoc, /wrapControl,/);
  assert.match(schemaDoc, /wrapComments,/);
  assert.match(schemaDoc, /preloadFullSchema = true/);
  assert.match(schemaDoc, /preloadFullSchema,/);
  assert.match(schemaDoc, /autoFocus = true/);
  assert.match(schemaDoc, /autoFocus,/);
  assert.match(schemaDoc, /loadFullSchema: loadFullSchema \?\? defaultLoadFullSchema\(data\)/);
  assert.match(schemaDoc, /const fullSchemaCache = new Map<string, Promise<KubeSchemaDocument> \| KubeSchemaDocument>\(\);/);
  assert.match(schemaDoc, /fullSchemaCache\.get\(source\)/);
  assert.match(schemaDoc, /fullSchemaCache\.set\(source, request\)/);
  assert.match(schemaDoc, /function activeController\(root: KubectlDocHost \| null, fallback\?: KubectlDocController\)/);
  assert.match(schemaDoc, /restoreSnapshot\(controller, previousSnapshot\);/);
  assert.match(schemaDoc, /snapshotRef\.current = mountedController\?\.snapshot\?\.\(\) \?\? null;/);
  assert.match(schemaDoc, /mountedController\?\.destroy\(\);/);
  assert.match(schemaDoc, /return response\.json\(\) as Promise<KubeSchemaDocument>;/);
  assert.doesNotMatch(schemaDoc, /useState/);
  assert.doesNotMatch(schemaDoc, /visibleLines\.map/);
  assert.doesNotMatch(schemaDoc, /data\.lines\.map/);
  assert.doesNotMatch(schemaDoc, /<SchemaLine/);
  assert.doesNotMatch(schemaDoc, /function SchemaLine/);
  assert.doesNotMatch(schemaDoc, /parseSchemaPayload|atob|TextDecoder|response\.text/);
});

test("shared runtime keeps Fern overlay, scoped keyboard, and lazy full-payload state behavior", () => {
  const runtime = readFileSync(join(componentRoot, "kubectl-doc-runtime.js"), "utf8");
  const css = readFileSync(join(componentRoot, "kubectl-doc-styles.ts"), "utf8");

  assert.match(runtime, /function mount\(root, options\)/);
  assert.match(runtime, /renderSchema\(root, options\.initialSchema, options\);/);
  assert.match(runtime, /root\.classList\.toggle\("kdoc-details-side-overlay", scopedKeyboard\);/);
  assert.match(runtime, /function themeMatches\(value, mode\)/);
  assert.match(runtime, /themeMatches\(mdTheme, "dark"\)/);
  assert.match(runtime, /themeMatches\(mdTheme, "light"\)/);
  assert.match(runtime, /new MutationObserver\(syncHostTheme\)/);
  assert.match(runtime, /attributeFilter: \["class", "data-theme", "data-color-mode", "data-md-color-scheme", "data-kdoc-theme"\]/);
  assert.match(runtime, /function replaceHash\(path\)/);
  assert.match(runtime, /history\.pushState\(\{kubectlDocPath: path\}/);
  assert.match(runtime, /window\.addEventListener\("hashchange", handleHashNavigation\)/);
  assert.match(runtime, /focusPath\(path, \{scroll: scroll !== false, updateHash:false\}\)/);
  assert.match(runtime, /if\(scopedKeyboard && !hostHasFocus\(\)\)\{ return false; \}/);
  assert.match(runtime, /var keyTarget = document;/);
  assert.match(runtime, /keyTarget\.addEventListener\("keydown", handleCursorKey\);/);
  assert.match(runtime, /filter: filterQuery/);
  assert.match(runtime, /function renderPayloadToken\(token\)/);
  assert.match(runtime, /if\(line\.tokens && line\.tokens\.length\)\{/);
  assert.doesNotMatch(runtime, /function renderInlineYAML/);
  assert.doesNotMatch(runtime, /function renderYAMLCode/);
  assert.doesNotMatch(runtime, /function renderScalarToken/);
  assert.match(runtime, /function wantsFullProjectionForExpansion\(line\)/);
  assert.match(runtime, /function expandWithFullSchema\(line\)/);
  assert.match(runtime, /function toggleExpandedWithFullSchema\(line\)/);
  assert.match(runtime, /function scheduleFullSchemaPreload\(\)/);
  assert.match(runtime, /if\(mountedOptions\.preloadFullSchema === false \|\| fullSchema \|\| loadingFullSchema \|\| !mountedOptions\.loadFullSchema\)\{ return; \}/);
  assert.match(runtime, /requestFullSchema\(\);/);
  assert.match(runtime, /scheduleFullSchemaPreload\(\);/);
  assert.match(runtime, /function cancelFullSchemaPreload\(\)/);
  assert.match(runtime, /optionEnabled\(options, "autoFocus", root, "data-kdoc-auto-focus"\)/);
  assert.match(runtime, /if\(destroyed\)\{\s*fullSchemaCallbacks = \[\];\s*return;\s*\}/s);
  assert.match(runtime, /function scheduleInitialFocus\(\)/);
  assert.match(runtime, /if\(!autoFocus \|\| !scopedKeyboard\)\{ return; \}/);
  assert.match(runtime, /scheduleInitialFocus\(\);/);
  assert.match(runtime, /function cancelInitialFocus\(\)/);
  assert.match(runtime, /requestFullSchema\(function\(\)\{ renderFullFoldProjection\(path, true\); \}\);/);
  assert.match(runtime, /foldStates\.push\(\{path: state\.path, expanded: expanded\(state\.line\)\}\);/);
  assert.match(runtime, /function restoreFoldSnapshot\(targetController, foldStates\)/);
  assert.match(runtime, /function hostHasFocus\(\)/);
  assert.match(schemaDoc, /if \(snapshot\.hasFocus\) \{\s*controller\.setFocused\?\.\(true\);/s);
  assert.match(runtime, /function handleFocusIn\(\)\{\s*markHostFocused\(\);\s*requestFullSchema\(\);/s);
  assert.match(schemaDoc, /rootRef\.current\.innerHTML = "";/);
  assert.match(schemaDoc, /if \(snapshot\.filter\) \{\s*controller\.setFilter\?\.\(snapshot\.filter\);/s);
  assert.match(schemaDoc, /if \(snapshot\.currentPath\) \{\s*controller\.focusPath\?\.\(snapshot\.currentPath, \{ scroll: false, updateHash: false \}\);/s);
  assert.match(runtime, /folds: foldSnapshot\(\)/);
  assert.match(runtime, /hasFocus: hostHasFocus\(\)/);
  assert.match(runtime, /setFocused: function\(value\)/);

  assert.match(css, /\.kdoc-embedded-host\{/);
  assert.match(css, /\.kdoc-embedded-host \.kdoc-tree\{[^}]*overflow:hidden/);
  assert.match(css, /\.kubectl-doc\.kdoc-details-side-overlay:not\(\.kdoc-has-focus\) \.kdoc-details\{display:none\}/);
  assert.match(css, /\.kubectl-doc\.kdoc-details-side-overlay \.kdoc-details\{[^}]*position:fixed[^}]*z-index:2147483647/);
});

test("LazyKubeSchemaDoc delegates idle hydration to KubeSchemaDoc and loads JSON assets only", () => {
  assert.match(lazySchemaDoc, /import \{ schemaBaseURL, schemaSources \} from "\.\/schemaSources\.generated";/);
  assert.match(lazySchemaDoc, /const rootRef = useRef<HTMLDivElement>\(null\)/);
  assert.match(lazySchemaDoc, /const \[isVisible, setIsVisible\] = useState\(false\)/);
  assert.match(lazySchemaDoc, /const schemaGenerationRef = useRef\(0\)/);
  assert.match(lazySchemaDoc, /const loadingInitialRef = useRef\(false\)/);
  assert.match(lazySchemaDoc, /const fullLoadRef = useRef<Promise<KubeSchemaDocument> \| null>\(null\)/);
  assert.match(generatedSourcesDoc, /export const schemaBaseURL = "https:\/\/raw\.githubusercontent\.com\/ai-dynamo\/dynamo\/main\/fern\/kubectl-doc-schemas";/);
  assert.match(generatedSourcesDoc, /"DynamoGraphDeploymentSchema0": \{ initial: "dynamo-graph-deployment-schema-0\.json", full: "dynamo-graph-deployment-schema-0-full\.json" \}/);
  assert.match(lazySchemaDoc, /function schemaURL\(fileName: string\)/);
  assert.match(lazySchemaDoc, /schemaBaseURL\.replace\(\/\\\/\$\/, ""\)/);
  assert.match(lazySchemaDoc, /function fetchSchema\(fileName: string\)/);
  assert.match(lazySchemaDoc, /return response\.json\(\) as Promise<KubeSchemaDocument>;/);
  assert.match(lazySchemaDoc, /new IntersectionObserver/);
  assert.match(lazySchemaDoc, /rootMargin: "160px 0px"/);
  assert.match(lazySchemaDoc, /if \(!isVisible \|\| data \|\| loadingInitialRef\.current\) \{/);
  assert.match(lazySchemaDoc, /schemaGenerationRef\.current \+= 1;/);
  assert.match(lazySchemaDoc, /fetchSchema\(source\.initial\)/);
  assert.match(lazySchemaDoc, /if \(!cancelled && generation === schemaGenerationRef\.current\) \{\s*setData\(payload\);/s);
  assert.match(lazySchemaDoc, /if \(fullLoadRef\.current\) \{\s*return fullLoadRef\.current;/s);
  assert.match(lazySchemaDoc, /const source = schemaSources\[name\]\?\.full;/);
  assert.match(lazySchemaDoc, /const promise = fetchSchema\(source\)/);
  assert.match(lazySchemaDoc, /setData\(next\);\s*return next;/s);
  assert.match(lazySchemaDoc, /setError\(loadError instanceof Error \? loadError\.message : String\(loadError\)\);\s*fullLoadRef\.current = null;/s);
  assert.match(lazySchemaDoc, /fullLoadRef\.current = promise;\s*return promise;/s);
  assert.doesNotMatch(lazySchemaDoc, /finally\(\(\) => \{\s*if \(generation === schemaGenerationRef\.current\) \{\s*fullLoadRef\.current = null;/s);
  assert.match(lazySchemaDoc, /if \(!source \|\| data\?\.complete\) \{\s*return false;/s);
  assert.doesNotMatch(lazySchemaDoc, /return true;/);
  assert.match(lazySchemaDoc, /<div ref=\{rootRef\} className="kdoc-fern-lazy-frame">/);
  assert.doesNotMatch(lazySchemaDoc, /requestIdleCallback/);
  assert.doesNotMatch(lazySchemaDoc, /setTimeout\(callback, 1500\)/);
  assert.match(lazySchemaDoc, /<KubeSchemaDoc data=\{data\} filtering=\{filtering\} loadFullSchema={loadFull} \/>/);
  assert.match(lazySchemaDoc, /export default LazyKubeSchemaDoc;/);
  assert.doesNotMatch(lazySchemaDoc, /parseSchemaPayload|atob|TextDecoder|response\.text|schemaAssetPath|data-kdoc-schema-asset/);
  assert.doesNotMatch(generatedSourcesDoc, /\.md|docs\/assets|data-kdoc-schema-asset|import .*\.json|import\("/);

  const resetIndex = lazySchemaDoc.indexOf("useEffect(() => {\n    schemaGenerationRef.current += 1;\n    setData(null);");
  const resetEnd = lazySchemaDoc.indexOf("  }, [name]);", resetIndex);
  const visibilityFetchIndex = lazySchemaDoc.indexOf("if (!isVisible || data || loadingInitialRef.current)");
  assert.ok(resetIndex >= 0, "LazyKubeSchemaDoc should reset loaded data only in a schema-name effect");
  assert.ok(resetEnd > resetIndex, "LazyKubeSchemaDoc reset effect should depend only on name");
  assert.ok(
    resetEnd < visibilityFetchIndex,
    "LazyKubeSchemaDoc must not clear already-loaded data when visibility changes",
  );
});

test("KubeSchemaDoc keeps long YAML/comment lines inside the schema frame", () => {
  const css = readFileSync(join(componentRoot, "kubectl-doc-styles.ts"), "utf8");
  assert.match(css, /\.kdoc-embedded-host \.kdoc-tree\{[^}]*inline-size:100%[^}]*max-inline-size:100%[^}]*overflow:hidden/);
  assert.match(css, /\.kdoc-embedded-host \.kdoc-line\{[^}]*display:grid[^}]*grid-template-columns:24px minmax\(0,1fr\)[^}]*inline-size:100%[^}]*max-inline-size:100%[^}]*overflow:hidden[^}]*white-space:normal/);
  assert.match(css, /\.kdoc-embedded-host \.kdoc-line\[hidden\]\{display:none!important\}/);
  assert.match(css, /\.kdoc-embedded-host \.kdoc-yaml-text\{[^}]*display:block[^}]*min-inline-size:0[^}]*overflow-wrap:anywhere[^}]*white-space:pre-wrap/);
  assert.match(css, /\.kdoc-embedded-host \.kdoc-yaml-text \*\{[^}]*max-inline-size:100%[^}]*min-inline-size:0[^}]*overflow-wrap:anywhere/);
});

test("KubeSchemaDoc keeps field details as a focused high-z overlay", () => {
  const runtime = readFileSync(join(componentRoot, "kubectl-doc-runtime.js"), "utf8");
  const css = readFileSync(join(componentRoot, "kubectl-doc-styles.ts"), "utf8");
  assert.match(runtime, /root\.classList\.add\("kdoc-has-focus"\)/);
  assert.match(runtime, /root\.classList\.remove\("kdoc-has-focus"\)/);
  assert.match(css, /\.kdoc-embedded-host\{[^}]*position:relative/);
  assert.match(css, /\.kubectl-doc\.kdoc-details-side-overlay:not\(\.kdoc-has-focus\) \.kdoc-details\{display:none\}/);
  assert.match(css, /\.kubectl-doc\.kdoc-details-side-overlay \.kdoc-details\{[^}]*position:fixed[^}]*z-index:2147483647/);
});

test("multi-version API reference pages keep inactive versions behind Fern tabs", () => {
  const pages = [
    "dynamocomponentdeployment.mdx",
    "dynamographdeployment.mdx",
    "dynamographdeploymentrequest.mdx",
    "dynamographdeploymentscalingadapter.mdx",
  ];

  for (const page of pages) {
    const content = readFileSync(join(repoRoot, "docs/kubernetes/api-reference", page), "utf8");
    assert.match(content, /<Tabs>/, `${page} should render versions in tabs`);
    assert.match(content, /<Tab title="nvidia\.com\/v1beta1">/, `${page} should have a v1beta1 tab`);
    assert.match(content, /<Tab title="nvidia\.com\/v1alpha1">/, `${page} should have a v1alpha1 tab`);
    assert.doesNotMatch(content, /^## nvidia\.com\//m, `${page} should not mount versions as plain headings`);
  }
});

test("schema pages do not wrap the primary tree in a YAML disclosure", () => {
  assert.doesNotMatch(schemaDoc, /className="kdoc-fern-title"/);
  assert.doesNotMatch(schemaDoc, />YAML</);
  assert.doesNotMatch(schemaDoc, /<Accordion[^>]*YAML/);

  for (const page of readdirSync(join(repoRoot, "docs/kubernetes/api-reference")).filter((file) => file.endsWith(".mdx"))) {
    const content = readFileSync(join(repoRoot, "docs/kubernetes/api-reference", page), "utf8");
    assert.doesNotMatch(content, /<Accordion[^>]*YAML/, `${page} should not wrap the schema in a Fern accordion`);
    assert.doesNotMatch(content, /<details[\s\S]*?<summary>YAML/, `${page} should not wrap the schema in a details block`);
  }
});

test("generated schema JSON assets keep shallow and full data split", () => {
  const files = readdirSync(schemaAssetDir).filter((file) => file.endsWith(".json")).sort();
  const initialFiles = files.filter((file) => !file.endsWith("-full.json"));

  assert.ok(initialFiles.length > 0, "expected generated initial schema JSON assets");
  for (const file of initialFiles) {
    const fullFile = file.replace(/\.json$/, "-full.json");
    assert.ok(files.includes(fullFile), `${file} should have a matching full JSON asset`);

    const initial = readSchemaPayload(file);
    const full = readSchemaPayload(fullFile);
    assert.equal(initial.payload.complete, false, `${file} should be an initial shallow payload`);
    assert.equal(full.payload.complete, true, `${fullFile} should be a complete payload`);
    assert.ok(initial.payload.lines.length <= full.payload.lines.length, `${file} should not have more lines than ${fullFile}`);
    assert.ok(initial.payload.fields.length <= full.payload.fields.length, `${file} should not have more fields than ${fullFile}`);
    assert.ok(
      initial.payload.fullPayloadURL === `./${fullFile}`,
      `${file} should point to the matching full JSON asset`,
    );
    if (full.content.length > 1_000_000) {
      assert.ok(
        initial.content.length * 4 < full.content.length,
        `${file} should keep large initial payloads materially smaller than ${fullFile}: initial=${initial.content.length} full=${full.content.length}`,
      );
    }

    const metadata = initial.payload.lines.find((line) => line.path === "metadata");
    assert.ok(metadata?.collapsed, `${file} should keep metadata collapsed in the initial payload`);
    assert.ok(
      initial.payload.lines.some((line) => line.path === "metadata.name"),
      `${file} should include metadata.name before loading the full payload`,
    );
    assert.ok(
      initial.payload.lines.some((line) => line.path === "metadata.namespace"),
      `${file} should include metadata.namespace before loading the full payload`,
    );
    assert.equal(
      initial.payload.lines.some((line) => line.path === "status.phase"),
      false,
      `${file} should keep status descendants out of the initial payload`,
    );
    assert.ok(
      initial.payload.lines.some((line) => line.foldable && line.collapsed),
      `${file} should retain collapsed placeholders for hidden descendants`,
    );

    for (const line of [...initial.payload.lines, ...full.payload.lines]) {
      assert.equal(line.description, undefined, `${file}/${fullFile} should not duplicate descriptions in line records`);
      assert.equal(line.filterText, undefined, `${file}/${fullFile} should not duplicate filter text in line records`);
      assert.equal(line.text, undefined, `${file}/${fullFile} should not duplicate raw YAML text in line records`);
      assert.ok(line.comment || Array.isArray(line.tokens), `${file}/${fullFile} should carry structured line tokens`);
    }
    assert.doesNotMatch(initial.payload.fullPayloadURL ?? "", /^https?:/);
    assert.doesNotMatch(initial.payload.fullPayloadURL ?? "", /localhost|openapi/i);
    for (const content of [initial.content, full.content]) {
      const lower = content.toLowerCase();
      assert.equal(content.trimStart().startsWith("{"), true, `${file}/${fullFile} should be raw JSON`);
      assert.equal(lower.includes("```kubectl-doc-schema"), false, `${file}/${fullFile} should not contain Markdown fences`);
    }
  }
});

test("DynamoGraphDeployment keeps a small initial schema payload", () => {
  for (const index of [0, 1]) {
    const initial = readSchemaPayload(`dynamo-graph-deployment-schema-${index}.json`);
    const full = readSchemaPayload(`dynamo-graph-deployment-schema-${index}-full.json`);
    assert.equal(initial.payload.complete, false, `v${index} initial payload should be shallow`);
    assert.equal(full.payload.complete, true, `v${index} full payload should be complete`);
    assert.ok(
      initial.content.length < 150_000,
      `v${index} initial payload should remain below 150KB, got ${initial.content.length}`,
    );
    assert.ok(
      full.content.length > 2_000_000,
      `v${index} full payload should keep the complete generated schema, got ${full.content.length}`,
    );
    assert.ok(
      initial.content.length * 20 < full.content.length,
      `v${index} initial payload should be at least 20x smaller than full payload: initial=${initial.content.length} full=${full.content.length}`,
    );
  }
});

test("API reference pages, lazy source map, JSON assets, and Fern asset sync stay in sync", () => {
  const apiReferenceDir = join(repoRoot, "docs/kubernetes/api-reference");
  const docsIndex = readFileSync(join(repoRoot, "docs/index.yml"), "utf8");
  const workflow = readFileSync(join(repoRoot, ".github/workflows/fern-docs.yml"), "utf8");
  const files = new Set(readdirSync(schemaAssetDir).filter((file) => file.endsWith(".json")));
  const sources = lazySchemaSources();
  const usedNames = new Set();

  for (const page of readdirSync(apiReferenceDir).filter((file) => file.endsWith(".mdx")).sort()) {
    const content = readFileSync(join(apiReferenceDir, page), "utf8");
    for (const match of content.matchAll(/<LazyKubeSchemaDoc name=\{"([^"]+)"\} \/>/g)) {
      usedNames.add(match[1]);
    }
    assert.doesNotMatch(content, /data-kdoc-schema-asset|data-kdoc-schema-assets/, `${page} should not hide schema asset anchors`);
  }
  assert.ok(usedNames.size > 0, "expected API reference pages to use LazyKubeSchemaDoc");

  for (const name of usedNames) {
    const source = sources.get(name);
    assert.ok(source, `${name} should have a LazyKubeSchemaDoc source mapping`);

    const slug = slugName(name);
    assert.equal(source.initial, `${slug}.json`, `${name} initial source should match its JSON asset name`);
    assert.equal(source.full, `${slug}-full.json`, `${name} full source should match its JSON asset name`);
    assert.ok(files.has(source.initial), `${name} should have ${source.initial}`);
    assert.ok(files.has(source.full), `${name} should have ${source.full}`);
  }

  for (const name of sources.keys()) {
    assert.ok(usedNames.has(name), `${name} source mapping should be used by an API reference page`);
  }

  assert.doesNotMatch(docsIndex, /Dynamo[A-Za-z]+Schema\d/);
  assert.doesNotMatch(docsIndex, /hidden: true[\s\S]*schema/i);
  assert.match(workflow, /source-checkout\/fern\/components/);
  assert.match(workflow, /docs-checkout\/fern\/components/);
  assert.match(workflow, /RAW_SCHEMA_BASE_URL="https:\/\/raw\.githubusercontent\.com\/\$\{GITHUB_REPOSITORY\}\/\$\{GITHUB_REF_NAME\}\/fern\/kubectl-doc-schemas"/);
  assert.match(workflow, /schemaSources\.generated\.ts/);
  assert.doesNotMatch(workflow, /generated kubectl-doc JSON schema payloads\.[\s\S]*source-checkout\/docs\/assets/);
  assert.match(lazySchemaDoc, /fetchSchema\(source\.initial\)/);
  assert.match(lazySchemaDoc, /const promise = fetchSchema\(source\)/);
  assert.doesNotMatch(lazySchemaDoc, /schemaAssetPath|data-kdoc-schema-asset/);
});
