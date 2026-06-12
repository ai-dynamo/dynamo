(function(global){
  function ready(fn){ if(document.readyState !== "loading"){ fn(); } else { document.addEventListener("DOMContentLoaded", fn); } }
  function escapeHTML(value){
    return String(value || "").replace(/[&<>"']/g, function(ch){
      return {"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[ch];
    });
  }
  function attr(value){ return escapeHTML(value); }
  function optionEnabled(options, name, root, attributeName){
    if(Object.prototype.hasOwnProperty.call(options, name)){ return options[name] === true; }
    var value = root.getAttribute(attributeName);
    return value != null && (value === "" || value === "true" || value === "1");
  }
  function fieldMap(schema){
    var map = new Map();
    (schema.fields || []).forEach(function(field){ map.set(field.id, field); });
    return map;
  }
  function detailHTML(field){
    if(!field){ return ""; }
    var html = "<dl class=\"kdoc-detail-grid\">";
    html += "<div class=\"kdoc-detail-row\"><dt>Path</dt><dd><code class=\"kdoc-detail-code\">" + escapeHTML(field.path) + "</code></dd></div>";
    html += "<div class=\"kdoc-detail-row\"><dt>Type</dt><dd><code class=\"kdoc-detail-code\">" + escapeHTML(field.type) + "</code></dd></div>";
    html += "<div class=\"kdoc-detail-row\"><dt>Required</dt><dd><span class=\"kdoc-detail-badge " + (field.required ? "kdoc-detail-badge-required" : "kdoc-detail-badge-optional") + "\">" + (field.required ? "yes" : "no") + "</span></dd></div>";
    html += "</dl>";
    if(field.description){
      html += "<section class=\"kdoc-detail-section\"><h3>Description</h3><p class=\"kdoc-detail-description\">" + escapeHTML(field.description) + "</p></section>";
    }
    if(field.metadata && field.metadata.length){
      html += "<section class=\"kdoc-detail-section\"><h3>Validation and metadata</h3><ul class=\"kdoc-detail-list\">";
      field.metadata.forEach(function(item){ html += "<li><code>" + escapeHTML(item) + "</code></li>"; });
      html += "</ul></section>";
    }
    return html;
  }
  function isCommentLine(text){
    return /^\s*(?:-\s*)?#/.test(text || "");
  }
  function commentParts(text){
    var match = String(text || "").match(/^(\s*(?:-\s*)?#\s?)(.*)$/);
    if(!match){ return null; }
    var prefix = match[1];
    var nextPrefix = prefix;
    if(/-\s*#/.test(prefix)){
      nextPrefix = prefix.replace(/-\s*#\s?/, "  # ");
    }
    return {prefix: prefix, nextPrefix: nextPrefix, text: match[2]};
  }
  function renderCommentText(text){
    var parts = commentParts(text);
    if(!parts){ return escapeHTML(text); }
    return "<span class=\"kdoc-comment\" data-kdoc-comment data-kdoc-comment-prefix=\"" + attr(parts.prefix) + "\" data-kdoc-comment-wrap-prefix=\"" + attr(parts.nextPrefix) + "\" data-kdoc-comment-text=\"" + attr(parts.text) + "\"><span class=\"kdoc-yaml-comment kdoc-comment-prefix\">" + escapeHTML(parts.prefix) + "</span><span class=\"kdoc-yaml-comment kdoc-comment-body\">" + escapeHTML(parts.text) + "</span></span>";
  }
  function renderPayloadComment(comment){
    var prefix = comment.prefix || comment.p || "";
    var wrapPrefix = comment.wrapPrefix || comment.w || prefix;
    var text = comment.text || comment.t || "";
    return "<span class=\"kdoc-comment\" data-kdoc-comment data-kdoc-comment-prefix=\"" + attr(prefix) + "\" data-kdoc-comment-wrap-prefix=\"" + attr(wrapPrefix) + "\" data-kdoc-comment-text=\"" + attr(text) + "\"><span class=\"kdoc-yaml-comment kdoc-comment-prefix\">" + escapeHTML(prefix) + "</span><span class=\"kdoc-yaml-comment kdoc-comment-body\">" + escapeHTML(text) + "</span></span>";
  }
  function tokenClass(kind){
    switch(kind || ""){
    case "bool":
      return "kdoc-yaml-bool";
    case "comment":
      return "kdoc-yaml-comment";
    case "key":
      return "kdoc-yaml-key";
    case "null":
      return "kdoc-yaml-null";
    case "number":
      return "kdoc-yaml-number";
    case "placeholder":
      return "kdoc-yaml-placeholder";
    case "punct":
      return "kdoc-yaml-punct";
    case "required":
      return "kdoc-required-label";
    case "scalar":
      return "kdoc-yaml-scalar";
    case "string":
      return "kdoc-yaml-string";
    case "type-number":
      return "kdoc-yaml-type-number";
    default:
      return "";
    }
  }
  function tokenText(token){
    if(token.t != null){ return token.t; }
    if(token.text != null){ return token.text; }
    return "";
  }
  function renderPayloadToken(token){
    var text = tokenText(token);
    var className = tokenClass(token.k || token.kind);
    if(!className){ return escapeHTML(text); }
    return "<span class=\"" + className + "\">" + escapeHTML(text) + "</span>";
  }
  function lineText(line){
    if(line.text != null){ return String(line.text); }
    if(line.comment){
      return String(line.comment.prefix || line.comment.p || "") + String(line.comment.text || line.comment.t || "");
    }
    if(line.tokens && line.tokens.length){
      return line.tokens.map(tokenText).join("");
    }
    return "";
  }
  function renderLineYAML(line, text){
    if(line.comment){ return renderPayloadComment(line.comment); }
    if(line.tokens && line.tokens.length){
      return line.tokens.map(renderPayloadToken).join("");
    }
    if(isCommentLine(text)){ return renderCommentText(text); }
    return escapeHTML(text);
  }
  function lineDetailID(line, index){
    return line.detailId || ("line-" + (line.index != null ? line.index : index));
  }
  function renderLineHTML(line, index, fields){
    var field = line.detailId ? fields.get(line.detailId) : null;
    var text = lineText(line);
    var classes = "kdoc-line" + (text.trim() ? "" : " kdoc-blank");
    var fieldAttr = line.field ? " data-kdoc-field data-kdoc-field-name=\"" + attr(line.field) + "\" data-kdoc-filter-text=\"" + attr((line.field || "") + "\n" + (field && field.description ? field.description : "")) + "\"" : "";
    var commentGroupAttr = line.commentGroup ? " data-kdoc-comment-group=\"" + attr(line.commentGroup) + "\"" : "";
    var detailID = lineDetailID(line, index);
    var html = "<div class=\"" + classes + "\" role=\"treeitem\" data-kdoc-line" + fieldAttr + commentGroupAttr + " data-index=\"" + attr(line.index != null ? line.index : index) + "\" data-depth=\"" + attr(line.depth || 0) + "\" data-path=\"" + attr(line.path || "") + "\" data-detail-id=\"" + attr(detailID) + "\" data-detail=\"\" data-detail-html=\"" + attr(detailHTML(field)) + "\">";
    if(line.foldable){
      html += "<button class=\"kdoc-fold\" type=\"button\" aria-label=\"Toggle\" aria-expanded=\"" + (line.collapsed ? "false" : "true") + "\" data-kdoc-toggle></button>";
    } else {
      html += "<span class=\"kdoc-gutter\"></span>";
    }
    var commentText = line.comment || (!line.tokens && isCommentLine(text));
    html += "<span class=\"kdoc-yaml-text" + (commentText ? " kdoc-yaml-comment-text" : "") + "\">" + renderLineYAML(line, text) + "</span>";
    html += "</div>";
    return html;
  }
  function renderTreeHTML(schema, lines, fields){
    var html = "";
    (lines || (schema && schema.lines) || []).forEach(function(line, index){
      html += renderLineHTML(line, index, fields);
    });
    return html;
  }
  function renderSchema(root, schema, options){
    var fields = fieldMap(schema || {});
    var filtering = options.filtering !== false;
    var showWrapControl = options.wrapControl !== false;
    var wrapChecked = options.wrapComments !== false;
    var detailsMode = options.detailsMode || root.getAttribute("data-kdoc-details-mode") || "";
    root.classList.add("kubectl-doc");
    root.classList.toggle("kdoc-details-side-overlay", detailsMode === "side-overlay");
    root.classList.toggle("kdoc-embedded-host", detailsMode === "side-overlay" || root.classList.contains("kdoc-embedded-host"));
    root.classList.toggle("kdoc-filter-disabled", !filtering);
    root.setAttribute("data-kubectl-doc", "");
    if(!root.hasAttribute("tabindex")){ root.setAttribute("tabindex", "0"); }
    var html = "<div class=\"kdoc-layout\"><section class=\"kdoc-docs\"><div class=\"kdoc-filter-overlay\" data-kdoc-filter-overlay hidden></div><section class=\"kdoc-version\"><div class=\"kdoc-tree\" role=\"tree\" aria-label=\"" + attr((schema && schema.kind ? schema.kind : "Kubernetes") + " YAML schema") + "\">";
    html += renderTreeHTML(schema, schema.lines || [], fields);
    html += "</div></section></section><aside class=\"kdoc-details\" data-kdoc-details aria-live=\"polite\"><h2>Details</h2><div class=\"kdoc-detail-body\" data-kdoc-detail-body><p class=\"kdoc-detail-empty\">Select a field.</p></div></aside></div>";
    if(showWrapControl){
      html += "<div class=\"kdoc-view-controls\" aria-label=\"View options\"><label class=\"kdoc-wrap-toggle\"><input type=\"checkbox\" data-kdoc-wrap-comments" + (wrapChecked ? " checked" : "") + "><span class=\"kdoc-switch\" aria-hidden=\"true\"></span><span class=\"kdoc-wrap-label\">wrap</span></label></div>";
    }
    root.innerHTML = html;
  }
  function mount(root, options){
      options = options || {};
      if(!root){ return null; }
      if(root.__kubectlDocController){ return root.__kubectlDocController; }
      var mountStart = perfNow();
      if(options.initialSchema && !root.querySelector("[data-kdoc-line]")){
        renderSchema(root, options.initialSchema, options);
      }

      var tree = root.querySelector(".kdoc-tree");
      var lines = [];
      var comments = [];
      var details = root.querySelector("[data-kdoc-detail-body]");
      var wrapComments = root.querySelector("[data-kdoc-wrap-comments]");
      var filterOverlay = root.querySelector("[data-kdoc-filter-overlay]");
      var backURL = options.backURL || root.getAttribute("data-kdoc-back-url");
      var quitURL = options.quitURL || root.getAttribute("data-kdoc-quit-url");
      var resizeFrame = 0;
      var charWidthCache = 0;
      var commentColumnCache = 0;
      var currentLine = null;
      var filterQuery = "";
      var activeFilterState = null;
      var activeFilterProjection = null;
      var viewSchema = options.initialSchema || null;
      var fullSchema = viewSchema && viewSchema.complete !== false ? viewSchema : null;
      var fullSchemaIndex = null;
      var fullSchemaCallbacks = [];
      var renderedFullProjection = false;
      var foldStateByPath = Object.create(null);
      var lineStates = [];
      var stateByLine = new Map();
      var fieldStates = [];
      var detailFieldByID = new Map();
      var detailLineGroups = new Map();
      var detailLineStates = new Map();
      var allLineSet = new Set(lines);
      var commentStates = [];
      var commentGroups = [];
      var highlightedElements = [];
      var selectedLines = [];
      var filterVisibleLines = [];
      var filterScopeSection = null;
      var filterFoldSnapshot = null;
      var filterFoldSnapshotByPath = null;
      var filterFoldOverrides = Object.create(null);
      var filtering = options.filtering !== false;
      var loadingFullSchema = false;
      var pendingFullFilterRender = false;
      var mountedOptions = options;
      var controller = null;
      var staleBackdropTimers = [];
      var fullSchemaPreloadHandle = 0;
      var fullSchemaPreloadHandleType = "";
      var initialFocusHandle = 0;
      var initialFocusHandleType = "";
      var suppressHashNavigation = false;
      var destroyed = false;
      var explicitTheme = root.hasAttribute("data-kdoc-theme") && !root.hasAttribute("data-kdoc-managed-theme");
      var themeObserver = null;
      var detailsMode = options.detailsMode || root.getAttribute("data-kdoc-details-mode") || "";
      var scopedKeyboard = detailsMode === "side-overlay";
      var autoFocus = optionEnabled(options, "autoFocus", root, "data-kdoc-auto-focus");
      if(scopedKeyboard && !root.hasAttribute("tabindex")){ root.setAttribute("tabindex", "0"); }
      root.classList.toggle("kdoc-details-side-overlay", scopedKeyboard);
      root.classList.toggle("kdoc-embedded-host", scopedKeyboard || root.classList.contains("kdoc-embedded-host"));

      function perfNow(){
        return global.performance && performance.now ? performance.now() : Date.now();
      }
      function recordPerf(name, start, detail){
        var entry = {
          name: name,
          duration: Math.max(0, perfNow() - start),
          detail: detail || {}
        };
        global.__kubectlDocPerf = global.__kubectlDocPerf || [];
        global.__kubectlDocPerf.push(entry);
        try {
          root.dispatchEvent(new CustomEvent("kubectl-doc:perf", {detail: entry}));
        } catch(_err) {}
        return entry;
      }

      function themeValueFromElement(element, includeKdocTheme){
        if(!element){ return ""; }
        if(includeKdocTheme){
          var explicit = String(element.getAttribute("data-kdoc-theme") || "").toLowerCase();
          if(explicit === "dark" || explicit === "light"){ return explicit; }
        }
        var theme = String(element.getAttribute("data-theme") || "").toLowerCase();
        var colorMode = String(element.getAttribute("data-color-mode") || "").toLowerCase();
        var mdTheme = String(element.getAttribute("data-md-color-scheme") || "").toLowerCase();
        if(themeMatches(theme, "dark") || themeMatches(colorMode, "dark") || themeMatches(mdTheme, "dark") || mdTheme === "slate" || element.classList.contains("dark")){ return "dark"; }
        if(themeMatches(theme, "light") || themeMatches(colorMode, "light") || themeMatches(mdTheme, "light") || mdTheme === "default" || element.classList.contains("light")){ return "light"; }
        return "";
      }
      function themeMatches(value, mode){
        return value === mode || value.slice(-(mode.length + 1)) === "-" + mode;
      }
      function hostThemeValue(){
        var node = root;
        while(node){
          var value = themeValueFromElement(node, explicitTheme && node === root);
          if(value){ return value; }
          node = node.parentElement;
        }
        if(global.document){
          return themeValueFromElement(document.documentElement, false) || themeValueFromElement(document.body, false);
        }
        return "";
      }
      function syncHostTheme(){
        if(explicitTheme){ return; }
        var value = hostThemeValue();
        if(value){
          if(root.getAttribute("data-kdoc-theme") !== value){ root.setAttribute("data-kdoc-theme", value); }
          if(!root.hasAttribute("data-kdoc-managed-theme")){ root.setAttribute("data-kdoc-managed-theme", ""); }
        } else {
          if(root.hasAttribute("data-kdoc-theme")){ root.removeAttribute("data-kdoc-theme"); }
          if(root.hasAttribute("data-kdoc-managed-theme")){ root.removeAttribute("data-kdoc-managed-theme"); }
        }
      }
      function observeHostTheme(){
        if(explicitTheme || !global.MutationObserver || !global.document || !document.documentElement){ return; }
        syncHostTheme();
        themeObserver = new MutationObserver(syncHostTheme);
        themeObserver.observe(document.documentElement, {
          attributes: true,
          subtree: true,
          attributeFilter: ["class", "data-theme", "data-color-mode", "data-md-color-scheme", "data-kdoc-theme"]
        });
      }

      function normalizePath(path){
        return String(path || "").toLowerCase();
      }
      function decodedHashPath(){
        if(!global.location || !global.location.hash){ return ""; }
        var hash = String(global.location.hash || "").replace(/^#/, "");
        if(!hash){ return ""; }
        try {
          return decodeURIComponent(hash);
        } catch(_err) {
          return hash;
        }
      }
      function encodedPathHash(path){
        path = String(path || "");
        return path ? "#" + encodeURIComponent(path) : "";
      }
      function replaceHash(path){
        if(!global.location){ return; }
        var hash = encodedPathHash(path);
        var current = normalizePath(decodedHashPath());
        if(!hash || current === normalizePath(path)){ return; }
        if(global.history && history.pushState){
          history.pushState({kubectlDocPath: path}, "", location.pathname + location.search + hash);
          return;
        }
        suppressHashNavigation = true;
        location.hash = hash;
        if(global.setTimeout){
          setTimeout(function(){ suppressHashNavigation = false; }, 0);
        } else {
          suppressHashNavigation = false;
        }
      }

      function initializeFromDOM(){
        lines = Array.prototype.slice.call(root.querySelectorAll("[data-kdoc-line]"));
        comments = Array.prototype.slice.call(root.querySelectorAll("[data-kdoc-comment]"));
        activeFilterState = null;
        activeFilterProjection = null;
        lineStates = [];
        stateByLine = new Map();
        fieldStates = [];
        detailFieldByID = new Map();
        detailLineGroups = new Map();
        detailLineStates = new Map();
        allLineSet = new Set(lines);
        commentStates = [];
        commentGroups = [];
        highlightedElements = [];
        selectedLines = [];
        filterVisibleLines = [];

        lines.forEach(function(line, index){
          var detailID = line.getAttribute("data-detail-id") || "";
          var path = normalizePath(line.getAttribute("data-path") || "");
          var textElement = line.querySelector(".kdoc-yaml-text");
          var version = line.closest(".kdoc-version") || root;
          var state = {
            line: line,
            index: index,
            version: version,
            depth: Number(line.getAttribute("data-depth") || "0"),
            field: line.hasAttribute("data-kdoc-field"),
            filterText: (line.getAttribute("data-kdoc-filter-text") || "").toLowerCase(),
            path: path,
            pathParts: path ? path.split(".") : [],
            detailID: detailID,
            textTrim: line.textContent.trim(),
            textElement: textElement,
            textLower: textElement ? textElement.textContent.toLowerCase() : "",
            toggle: line.querySelector("[data-kdoc-toggle]"),
            fieldState: null,
            ancestors: [],
            descendants: [],
            pathHit: ""
          };
          lineStates.push(state);
          stateByLine.set(line, state);
          if(detailID){
            if(!detailLineGroups.has(detailID)){ detailLineGroups.set(detailID, []); }
            if(!detailLineStates.has(detailID)){ detailLineStates.set(detailID, []); }
            detailLineGroups.get(detailID).push(line);
            detailLineStates.get(detailID).push(state);
          }
          if(state.field){
            state.fieldState = state;
            fieldStates.push(state);
            if(detailID && !detailFieldByID.has(detailID)){ detailFieldByID.set(detailID, state); }
          }
          if(state.toggle && state.path && !Object.prototype.hasOwnProperty.call(foldStateByPath, state.path)){
            foldStateByPath[state.path] = state.toggle.getAttribute("aria-expanded") !== "false";
          }
        });
        lineStates.forEach(function(state){
          if(!state.field && state.detailID && detailFieldByID.has(state.detailID)){
            state.fieldState = detailFieldByID.get(state.detailID);
          }
        });
        var ancestorStack = [];
        fieldStates.forEach(function(state){
          while(ancestorStack.length && ancestorStack[ancestorStack.length - 1].depth >= state.depth){ ancestorStack.pop(); }
          state.ancestors = ancestorStack.slice();
          state.ancestors.forEach(function(ancestor){ ancestor.descendants.push(state); });
          ancestorStack.push(state);
        });
        comments.forEach(function(comment){
          var line = comment.closest("[data-kdoc-line]");
          commentStates.push({
            comment: comment,
            line: line,
            lineState: line ? lineState(line) : null,
            detailID: line ? line.getAttribute("data-detail-id") || "" : "",
            groupID: line ? line.getAttribute("data-kdoc-comment-group") || "" : "",
            firstPrefix: comment.getAttribute("data-kdoc-comment-prefix") || "",
            nextPrefix: comment.getAttribute("data-kdoc-comment-wrap-prefix") || comment.getAttribute("data-kdoc-comment-prefix") || "",
            text: comment.getAttribute("data-kdoc-comment-text") || "",
            commentGroup: null,
            wrapState: ""
          });
        });
        commentGroups = buildCommentGroups(commentStates);
      }

      function lineState(line){ return stateByLine.get(line) || null; }
      function button(line){ var state = lineState(line); return state ? state.toggle : line.querySelector("[data-kdoc-toggle]"); }
      function depth(line){ var state = lineState(line); return state ? state.depth : Number(line.getAttribute("data-depth") || "0"); }
      function buildSchemaIndex(schema){
        var fields = fieldMap(schema || {});
        var states = [];
        var stateFields = [];
        var stateDetailFields = new Map();
        var stateDetailLines = new Map();
        (schema.lines || []).forEach(function(line, index){
          var detailID = lineDetailID(line, index);
          var field = detailID ? fields.get(detailID) : null;
          var text = lineText(line);
          var path = normalizePath(line.path || "");
          var state = {
            model: line,
            index: index,
            depth: Number(line.depth || 0),
            field: !!line.field,
            filterText: (String(line.field || "") + "\n" + (field && field.description ? field.description : "")).toLowerCase(),
            path: path,
            pathParts: path ? path.split(".") : [],
            detailID: detailID,
            textTrim: text.trim(),
            textLower: text.toLowerCase(),
            ancestors: [],
            descendants: [],
            pathHit: ""
          };
          states.push(state);
          if(detailID){
            if(!stateDetailLines.has(detailID)){ stateDetailLines.set(detailID, []); }
            stateDetailLines.get(detailID).push(state);
          }
          if(state.field){
            state.fieldState = state;
            stateFields.push(state);
            if(detailID && !stateDetailFields.has(detailID)){ stateDetailFields.set(detailID, state); }
          }
        });
        states.forEach(function(state){
          if(!state.field && state.detailID && stateDetailFields.has(state.detailID)){
            state.fieldState = stateDetailFields.get(state.detailID);
          }
        });
        var stack = [];
        stateFields.forEach(function(state){
          while(stack.length && stack[stack.length - 1].depth >= state.depth){ stack.pop(); }
          state.ancestors = stack.slice();
          state.ancestors.forEach(function(ancestor){ ancestor.descendants.push(state); });
          stack.push(state);
        });
        return {schema: schema, fields: fields, lineStates: states, fieldStates: stateFields, detailLineStates: stateDetailLines};
      }
      function ensureFullSchemaIndex(){
        if(!fullSchema){ return null; }
        if(!fullSchemaIndex){ fullSchemaIndex = buildSchemaIndex(fullSchema); }
        return fullSchemaIndex;
      }
      function currentVersionSection(){
        if(currentLine){
          var state = lineState(currentLine);
          if(state && state.version){ return state.version; }
          var version = currentLine.closest(".kdoc-version");
          if(version){ return version; }
        }
        return root.querySelector(".kdoc-version") || root;
      }
      function compactCommentGroupText(states){
        return states.map(function(state){ return String(state.text || "").trim(); }).filter(Boolean).join(" ");
      }
      function newCommentGroup(state){
        var group = {
          key: state.groupID || "",
          detailID: state.detailID || "",
          firstPrefix: state.firstPrefix,
          nextPrefix: state.nextPrefix,
          states: [state],
          text: "",
          textLower: "",
          wrapState: ""
        };
        state.commentGroup = group;
        return group;
      }
      function canJoinCommentGroup(group, state){
        if(!group || !group.key || !state.groupID || !String(state.text || "").trim()){ return false; }
        if(group.key !== state.groupID || group.detailID !== state.detailID){ return false; }
        if(group.firstPrefix !== state.firstPrefix || group.nextPrefix !== state.nextPrefix){ return false; }
        var previous = group.states[group.states.length - 1];
        if(!previous || !previous.lineState || !state.lineState){ return false; }
        return previous.lineState.index + 1 === state.lineState.index;
      }
      function finishCommentGroup(group){
        if(!group){ return; }
        group.text = compactCommentGroupText(group.states);
        group.textLower = group.text.toLowerCase();
      }
      function buildCommentGroups(states){
        var groups = [];
        var current = null;
        states.forEach(function(state){
          if(!state.groupID || !String(state.text || "").trim()){
            finishCommentGroup(current);
            current = null;
            var single = newCommentGroup(state);
            finishCommentGroup(single);
            groups.push(single);
            return;
          }
          if(canJoinCommentGroup(current, state)){
            current.states.push(state);
            state.commentGroup = current;
            return;
          }
          finishCommentGroup(current);
          current = newCommentGroup(state);
          groups.push(current);
        });
        finishCommentGroup(current);
        return groups;
      }
      function expanded(line){ var b = button(line); return !b || b.getAttribute("aria-expanded") !== "false"; }
      function setExpanded(line, value, options){
        var b = button(line);
        if(!b){ return; }
        options = options || {};
        var state = lineState(line);
        if(state && state.path && !options.auto){ foldStateByPath[state.path] = !!value; }
        var next = value ? "true" : "false";
        if(b.getAttribute("aria-expanded") === next){ return; }
        b.setAttribute("aria-expanded", next);
        activeFilterProjection = null;
        if(filterQuery && !options.auto){
          if(state && state.path){ filterFoldOverrides[state.path] = !!value; }
        }
      }
      function modelExpanded(line, expandedResolver){
        if(!line || !line.foldable){ return true; }
        if(expandedResolver){
          var resolved = expandedResolver(line);
          if(resolved !== null && resolved !== undefined){ return !!resolved; }
        }
        var path = normalizePath(line.path || "");
        if(path && Object.prototype.hasOwnProperty.call(foldStateByPath, path)){ return !!foldStateByPath[path]; }
        return !line.collapsed;
      }
      function lineWithExpandedState(line, expandedValue){
        if(!line || !line.foldable){ return line; }
        var collapsed = !expandedValue;
        if(!!line.collapsed === collapsed){ return line; }
        var clone = {};
        Object.keys(line).forEach(function(key){ clone[key] = line[key]; });
        clone.collapsed = collapsed;
        return clone;
      }
      function projectedSchemaLines(index, expandedResolver, allowedStates){
        var visible = [];
        var hiddenDepth = -1;
        index.lineStates.forEach(function(state){
          if(allowedStates && !allowedStates.has(state)){ return; }
          if(hiddenDepth >= 0){
            if(state.textTrim === "" || state.depth > hiddenDepth){ return; }
            hiddenDepth = -1;
          }
          var expandedValue = modelExpanded(state.model, expandedResolver);
          visible.push(lineWithExpandedState(state.model, expandedValue));
          if(state.model.foldable && !expandedValue){ hiddenDepth = state.depth; }
        });
        return visible;
      }
      function renderSchemaProjection(schema, index, projection, focusPathValue, scroll, focusOptions){
        if(!tree){ return false; }
        var projectionStart = perfNow();
        clearFilterHighlights();
        tree.innerHTML = renderTreeHTML(schema, projection, index.fields);
        viewSchema = schema;
        renderedFullProjection = schema === fullSchema;
        commentColumnCache = 0;
        initializeFromDOM();
        applyCommentWrap();
        applyFolds();
        var nextFocusOptions = {};
        Object.keys(focusOptions || {}).forEach(function(key){ nextFocusOptions[key] = focusOptions[key]; });
        if(nextFocusOptions.scroll === undefined){ nextFocusOptions.scroll = scroll !== false; }
        nextFocusOptions.projectFull = false;
        if(!focusPathValue || !focusPath(focusPathValue, nextFocusOptions)){
          select(visibleFieldLines()[0] || lines[0], nextFocusOptions);
        }
        recordPerf("projection-render", projectionStart, {
          lines: projection.length,
          fields: fieldStates.length,
          complete: !!(schema && schema.complete)
        });
        return true;
      }
      function renderFullFoldProjection(focusPathValue, scroll, focusOptions){
        var index = ensureFullSchemaIndex();
        if(!index){ return false; }
        return renderSchemaProjection(fullSchema, index, projectedSchemaLines(index), focusPathValue, scroll, focusOptions);
      }
      function hasLoadedDescendants(line){
        var state = lineState(line);
        return !!(state && state.descendants && state.descendants.length);
      }
      function wantsFullProjectionForExpansion(line){
        return !!(line && !expanded(line) && !hasLoadedDescendants(line) && (fullSchema || (viewSchema && viewSchema.complete === false)));
      }
      function expandWithFullSchema(line){
        if(wantsFullProjectionForExpansion(line)){
          var path = line.getAttribute("data-path") || "";
          select(line, {scroll:false});
          setExpanded(line, true);
          if(fullSchema){
            renderFullFoldProjection(path, true);
          } else {
            requestFullSchema(function(){ renderFullFoldProjection(path, true); });
          }
          return true;
        }
        setExpanded(line, true);
        return false;
      }
      function toggleExpandedWithFullSchema(line){
        if(!expanded(line)){
          expandWithFullSchema(line);
          return;
        }
        setExpanded(line, false);
      }
      function setLineHidden(state, value){
        if(state.line.hidden !== value){ state.line.hidden = value; }
      }
      function nextContentDepth(index){
        for(var i = index; i < lineStates.length; i++){
          if(lineStates[i].textTrim !== ""){ return lineStates[i].depth; }
        }
        return null;
      }
      function cleanPathComponent(component){
        return String(component || "").replace(/\[\]$/, "");
      }
      function cleanPathComponents(parts){
        return parts.map(cleanPathComponent);
      }
      function pathComponentEqual(component, token){
        return component === token || cleanPathComponent(component) === token;
      }
      function pathComponentContains(component, token){
        return component.indexOf(token) >= 0 || cleanPathComponent(component).indexOf(token) >= 0;
      }
      function parsePathFilter(query){
        query = String(query || "").toLowerCase();
        var anchored = query.indexOf(".") === 0 && query.indexOf("...") !== 0;
        if(anchored){ query = query.slice(1); }
        if(!query || (!anchored && query.indexOf(".") < 0)){ return null; }

        var filter = {anchored: anchored, tokens: [], suffix: ""};
        for(var i = 0; i < query.length; ){
          if(query.slice(i, i + 3) === "..."){
            filter.tokens.push("...");
            i += 3;
            continue;
          }
          if(query[i] === "."){
            i++;
            continue;
          }

          var start = i;
          while(i < query.length && query[i] !== "."){ i++; }
          var token = query.slice(start, i);
          if(!token){ continue; }
          if(/\s/.test(token)){
            filter.suffix = query.slice(start);
            break;
          }
          filter.tokens.push(token);
        }
        if(!filter.tokens.length && !filter.suffix){ return null; }
        return filter;
      }
      function pathSuffixOverlapsFinalComponent(parts, suffix){
        var text = parts.join(".");
        var finalStart = text.length - parts[parts.length - 1].length;
        var offset = 0;
        while(offset <= text.length){
          var index = text.indexOf(suffix, offset);
          if(index < 0){ return false; }
          if(index + suffix.length > finalStart){ return true; }
          offset = index + 1;
        }
        return false;
      }
      function pathSuffixHighlight(parts, suffix){
        if(!parts.length){ return ""; }
        if(pathSuffixOverlapsFinalComponent(parts, suffix) || pathSuffixOverlapsFinalComponent(cleanPathComponents(parts), suffix)){
          var index = suffix.lastIndexOf(".");
          return index >= 0 ? suffix.slice(index + 1) : suffix;
        }
        return "";
      }
      function matchPathFilter(parts, partIndex, tokens, tokenIndex, suffix){
        if(tokenIndex === tokens.length){
          if(suffix){ return pathSuffixHighlight(parts.slice(partIndex), suffix); }
          return partIndex === parts.length ? "__match__" : "";
        }
        if(tokens[tokenIndex] === "..."){
          if(tokenIndex === tokens.length - 1 && !suffix){
            return cleanPathComponent(parts[parts.length - 1] || "");
          }
          for(var skip = partIndex; skip <= parts.length; skip++){
            var wildcardHit = matchPathFilter(parts, skip, tokens, tokenIndex + 1, suffix);
            if(wildcardHit){ return wildcardHit; }
          }
          return "";
        }
        if(partIndex >= parts.length){ return ""; }

        var token = tokens[tokenIndex];
        if(tokenIndex === tokens.length - 1 && !suffix){
          return partIndex === parts.length - 1 && pathComponentContains(parts[partIndex], token) ? token : "";
        }
        if(!pathComponentEqual(parts[partIndex], token)){ return ""; }
        return matchPathFilter(parts, partIndex + 1, tokens, tokenIndex + 1, suffix);
      }
      function pathFilterHighlightForState(state, filter){
        if(!filter || !state || !state.pathParts.length){ return ""; }
        var parts = state.pathParts;
        if(filter.anchored){
          var anchoredHit = matchPathFilter(parts, 0, filter.tokens, 0, filter.suffix);
          return anchoredHit === "__match__" ? "" : anchoredHit;
        }
        for(var start = 0; start < parts.length; start++){
          var hit = matchPathFilter(parts, start, filter.tokens, 0, filter.suffix);
          if(hit){ return hit === "__match__" ? "" : hit; }
        }
        return "";
      }
      function ancestorFieldLines(line){
        var state = lineState(line);
        if(!state || !state.fieldState){ return []; }
        return state.fieldState.ancestors.map(function(ancestor){ return ancestor.line; }).reverse();
      }
      function currentFilterState(){
        var query = filterQuery.toLowerCase();
        if(!query){ return null; }
        var scope = currentVersionSection();
        if(activeFilterState && activeFilterState.query === query && activeFilterState.scope === scope){ return activeFilterState; }

        var pathFilter = parsePathFilter(query);
        var directFields = new Set();
        fieldStates.forEach(function(state){
          if(state.version !== scope){ return; }
          state.pathHit = pathFilterHighlightForState(state, pathFilter);
          if(state.filterText.indexOf(query) >= 0 || state.pathHit){
            directFields.add(state);
          }
        });

        var includedFields = new Set();
        var allowedLines = new Set();
        var highlightLineStates = new Set();
        directFields.forEach(function(state){
          includedFields.add(state);
          state.ancestors.forEach(function(ancestor){ includedFields.add(ancestor); });
          state.descendants.forEach(function(descendant){ includedFields.add(descendant); });
          groupedLineStates(state).forEach(function(lineStateValue){
            allowedLines.add(lineStateValue.line);
            highlightLineStates.add(lineStateValue);
          });
        });
        includedFields.forEach(function(state){
          groupedLineStates(state).forEach(function(lineStateValue){ allowedLines.add(lineStateValue.line); });
        });

        var directLines = new Set();
        directFields.forEach(function(state){ directLines.add(state.line); });
        activeFilterState = {
          query: query,
          scope: scope,
          pathFilter: pathFilter,
          directFields: directFields,
          directLines: directLines,
          includedFields: includedFields,
          allowedLines: allowedLines,
          highlightLineStates: highlightLineStates
        };
        return activeFilterState;
      }
      function beginFilterSession(){
        if(filterFoldSnapshot){ return; }
        filterFoldSnapshot = foldSnapshot();
        filterFoldSnapshotByPath = Object.create(null);
        filterFoldSnapshot.forEach(function(item){ filterFoldSnapshotByPath[item.path] = item.expanded; });
        filterFoldOverrides = Object.create(null);
      }
      function endFilterSession(){
        filterFoldSnapshot = null;
        filterFoldSnapshotByPath = null;
        filterFoldOverrides = Object.create(null);
        activeFilterProjection = null;
      }
      function restoreFilterFoldSnapshot(){
        if(!filterFoldSnapshotByPath){ return; }
        fieldStates.forEach(function(state){
          if(Object.prototype.hasOwnProperty.call(filterFoldSnapshotByPath, state.path)){ setExpanded(state.line, filterFoldSnapshotByPath[state.path], {auto:true}); }
        });
      }
      function filterFoldOverride(path){
        return Object.prototype.hasOwnProperty.call(filterFoldOverrides, path) ? filterFoldOverrides[path] : null;
      }
      function filterSnapshotExpanded(path){
        return filterFoldSnapshotByPath && Object.prototype.hasOwnProperty.call(filterFoldSnapshotByPath, path) ? filterFoldSnapshotByPath[path] : null;
      }
      function groupedModelLineStates(index, state){
        if(!state.detailID){ return [state]; }
        return index.detailLineStates.get(state.detailID) || [state];
      }
      function fullFilterProjection(){
        var index = ensureFullSchemaIndex();
        var query = filterQuery.toLowerCase();
        if(!index || !query){ return null; }

        var pathFilter = parsePathFilter(query);
        var directFields = new Set();
        index.fieldStates.forEach(function(state){
          state.pathHit = pathFilterHighlightForState(state, pathFilter);
          if(state.filterText.indexOf(query) >= 0 || state.pathHit){
            directFields.add(state);
          }
        });

        var includedFields = new Set();
        var includedPaths = Object.create(null);
        var allowedStates = new Set();
        directFields.forEach(function(state){
          includedFields.add(state);
          state.ancestors.forEach(function(ancestor){ includedFields.add(ancestor); });
          state.descendants.forEach(function(descendant){ includedFields.add(descendant); });
          groupedModelLineStates(index, state).forEach(function(lineStateValue){ allowedStates.add(lineStateValue); });
        });
        includedFields.forEach(function(state){
          if(state.path){ includedPaths[state.path] = true; }
          groupedModelLineStates(index, state).forEach(function(lineStateValue){ allowedStates.add(lineStateValue); });
        });

        return {
          index: index,
          allowedStates: allowedStates,
          expandedResolver: function(line){
            var path = normalizePath(line.path || "");
            if(!path){ return null; }
            var override = filterFoldOverride(path);
            if(override !== null){ return override; }
            if(includedPaths[path]){ return true; }
            return filterSnapshotExpanded(path);
          }
        };
      }
      function renderFullFilterProjection(focusPathValue, scroll){
        var projection = fullFilterProjection();
        if(!projection){ return false; }
        return renderSchemaProjection(
          fullSchema,
          projection.index,
          projectedSchemaLines(projection.index, projection.expandedResolver, projection.allowedStates),
          focusPathValue,
          scroll
        );
      }
      function autoRevealFilterBranches(state){
        if(!state){ return; }
        fieldStates.forEach(function(fieldState){
          if(!fieldState.toggle){ return; }
          var override = filterFoldOverride(fieldState.path);
          if(override !== null){
            setExpanded(fieldState.line, override, {auto:true});
            return;
          }
          if(state.includedFields.has(fieldState)){
            setExpanded(fieldState.line, true, {auto:true});
            return;
          }
          var snapshot = filterSnapshotExpanded(fieldState.path);
          if(snapshot !== null){ setExpanded(fieldState.line, snapshot, {auto:true}); }
        });
      }
      function projectedFilterAllowedLines(state){
        autoRevealFilterBranches(state);
        if(activeFilterProjection && activeFilterProjection.state === state){ return activeFilterProjection.lines; }

        var visible = new Set();
        var hiddenDepth = -1;
        lineStates.forEach(function(lineStateValue){
          if(lineStateValue.version !== state.scope || !state.allowedLines.has(lineStateValue.line)){ return; }
          if(hiddenDepth >= 0){
            if(lineStateValue.textTrim === "" || lineStateValue.depth > hiddenDepth){ return; }
            hiddenDepth = -1;
          }
          visible.add(lineStateValue.line);
          if(lineStateValue.toggle && !expanded(lineStateValue.line)){ hiddenDepth = lineStateValue.depth; }
        });
        activeFilterProjection = {state: state, lines: visible};
        return visible;
      }
      function directFilterMatches(){
        var state = currentFilterState();
        return state ? state.directLines : new Set();
      }
      function directFilterMatchLines(){
        var direct = directFilterMatches();
        return visibleFieldLines().filter(function(line){ return direct.has(line); });
      }
      function filterAllowedLines(){
        var state = currentFilterState();
        return state ? projectedFilterAllowedLines(state) : allLineSet;
      }
      function lineVisible(line){
        if(filterQuery){
          var state = currentFilterState();
          var lineStateValue = lineState(line);
          return !!(state && lineStateValue && lineStateValue.version === state.scope && filterAllowedLines().has(line));
        }
        return !line.hidden;
      }
      function setFilterVisibleLines(allowed){
        root.classList.remove("kdoc-filtering");
        if(!filterQuery){
          if(filterScopeSection){ filterScopeSection.classList.remove("kdoc-filtering"); }
          filterVisibleLines.forEach(function(line){ line.classList.remove("kdoc-filter-visible"); });
          filterVisibleLines = [];
          filterScopeSection = null;
          return;
        }
        var scope = currentVersionSection();
        if(filterScopeSection && filterScopeSection !== scope){ filterScopeSection.classList.remove("kdoc-filtering"); }
        filterScopeSection = scope;
        if(filterScopeSection){ filterScopeSection.classList.add("kdoc-filtering"); }
        filterVisibleLines.forEach(function(line){
          if(!allowed.has(line)){ line.classList.remove("kdoc-filter-visible"); }
        });
        var next = [];
        allowed.forEach(function(line){
          if(line.hidden){ line.hidden = false; }
          line.classList.add("kdoc-filter-visible");
          next.push(line);
        });
        filterVisibleLines = next;
      }
      function applyFolds(){
        if(filterQuery){
          setFilterVisibleLines(filterAllowedLines());
          applyFilterHighlights();
          return;
        }
        setFilterVisibleLines(allLineSet);
        lineStates.forEach(function(state){ setLineHidden(state, false); });
        lineStates.forEach(function(state, index){
          var line = state.line;
          if(!lineVisible(line) || expanded(line)){ return; }
          var parentDepth = state.depth;
          for(var i = index + 1; i < lines.length; i++){
            var blank = lineStates[i].textTrim === "";
            var followingDepth = blank ? nextContentDepth(i + 1) : null;
            if(blank && followingDepth !== null && followingDepth <= parentDepth){ break; }
            if(!blank && lineStates[i].depth <= parentDepth){ break; }
            setLineHidden(lineStates[i], true);
          }
        });
        applyFilterHighlights();
      }
      function groupedLines(line){
        var id = line.getAttribute("data-detail-id");
        if(!id){ return [line]; }
        return detailLineGroups.get(id) || [line];
      }
      function groupedLineStates(state){
        if(!state.detailID){ return [state]; }
        return detailLineStates.get(state.detailID) || [state];
      }
      function fieldLineFor(line){
        var state = lineState(line);
        return state && state.fieldState ? state.fieldState.line : null;
      }
      function visibleFieldLines(){
        return fieldStates.filter(function(state){ return lineVisible(state.line); }).map(function(state){ return state.line; });
      }
      function firstVisibleLineInVersion(version){
        for(var i = 0; i < lineStates.length; i++){
          if(lineStates[i].version === version && lineVisible(lineStates[i].line)){ return lineStates[i].line; }
        }
        return null;
      }
      function firstVisibleFieldLineInVersion(version){
        for(var i = 0; i < fieldStates.length; i++){
          if(fieldStates[i].version === version && lineVisible(fieldStates[i].line)){ return fieldStates[i].line; }
        }
        return null;
      }
      function visibleFoldableLines(){
        return fieldStates.filter(function(state){ return lineVisible(state.line) && !!state.toggle; }).map(function(state){ return state.line; });
      }
      function currentFieldLine(){
        if(currentLine && lineVisible(currentLine)){ return currentLine; }
        return visibleFieldLines()[0] || null;
      }
      function lineIndex(collection, line){
        for(var i = 0; i < collection.length; i++){
          if(collection[i] === line){ return i; }
        }
        return -1;
      }
      function selectFieldByOffset(delta){
        var fields = visibleFieldLines();
        if(!fields.length){ return false; }
        var current = currentFieldLine();
        var index = lineIndex(fields, current);
        if(index < 0){ index = 0; }
        index = Math.max(0, Math.min(fields.length - 1, index + delta));
        select(fields[index], {scroll:true});
        return true;
      }
      function selectFirstField(){
        var fields = visibleFieldLines();
        if(!fields.length){ return false; }
        select(fields[0], {scroll:true});
        return true;
      }
      function selectLastField(){
        var fields = visibleFieldLines();
        if(!fields.length){ return false; }
        select(fields[fields.length - 1], {scroll:true});
        return true;
      }
      function pageFieldDistance(){
        var line = currentFieldLine();
        var height = 18;
        if(line){
          height = Math.max(line.getBoundingClientRect().height, height);
        }
        return Math.max(1, Math.floor(window.innerHeight / height / 2));
      }
      function parentField(line){
        if(!line){ return null; }
        var state = lineState(line);
        if(!state || !state.fieldState){ return null; }
        var ancestors = state.fieldState.ancestors;
        for(var i = ancestors.length - 1; i >= 0; i--){
          if(lineVisible(ancestors[i].line)){ return ancestors[i].line; }
        }
        return null;
      }
      function firstChildField(line){
        if(!line){ return null; }
        var currentDepth = depth(line);
        var fields = visibleFieldLines();
        var index = lineIndex(fields, line);
        for(var i = index + 1; i < fields.length; i++){
          if(depth(fields[i]) <= currentDepth){ return null; }
          return fields[i];
        }
        return null;
      }
      function toggleField(line){
        var toggle = button(line);
        if(!toggle){ return false; }
        toggleExpandedWithFullSchema(line);
        applyFolds();
        scheduleCommentWrap();
        select(line, {scroll:true});
        return true;
      }
      function collapseOrParent(){
        var line = currentFieldLine();
        if(!line){ return false; }
        if(button(line) && expanded(line)){
          setExpanded(line, false);
          applyFolds();
          scheduleCommentWrap();
          select(line, {scroll:true});
          return true;
        }
        var parent = parentField(line);
        if(!parent){ return false; }
        select(parent, {scroll:true});
        return true;
      }
      function expandOrChild(){
        var line = currentFieldLine();
        if(!line){ return false; }
        if(!button(line)){ return false; }
        if(!expanded(line)){
          expandWithFullSchema(line);
          applyFolds();
          scheduleCommentWrap();
          select(line, {scroll:true});
          return true;
        }
        var child = firstChildField(line);
        if(!child){ return false; }
        select(child, {scroll:true});
        return true;
      }
      function selectFoldable(delta){
        var foldable = visibleFoldableLines();
        if(!foldable.length){ return false; }
        var current = currentFieldLine();
        var index = lineIndex(foldable, current);
        if(index < 0){ index = delta > 0 ? -1 : 0; }
        index = (index + delta + foldable.length) % foldable.length;
        select(foldable[index], {scroll:true});
        return true;
      }
      function selectFilterMatch(delta){
        var matches = directFilterMatchLines();
        if(!matches.length){ return false; }
        var current = currentFieldLine();
        var index = lineIndex(matches, current);
        if(index < 0){
          index = delta > 0 ? 0 : matches.length - 1;
        } else {
          index = (index + delta + matches.length) % matches.length;
        }
        select(matches[index], {scroll:true});
        return true;
      }
      function cleanLineText(line){
        var comment = line.querySelector("[data-kdoc-comment]");
        if(comment){ return (comment.getAttribute("data-kdoc-comment-text") || "").trim(); }
        var text = line.querySelector(".kdoc-yaml-text").textContent.trim();
        if(text.indexOf("# ") === 0){ text = text.slice(2).trim(); }
        return text;
      }
      function escapeHTML(value){
        return String(value || "").replace(/[&<>"']/g, function(ch){
          return {"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[ch];
        });
      }
      function charWidth(){
        if(charWidthCache){ return charWidthCache; }
        var sample = document.createElement("span");
        sample.textContent = "0000000000";
        sample.style.position = "absolute";
        sample.style.visibility = "hidden";
        sample.style.whiteSpace = "pre";
        if(lines[0]){
          sample.style.font = window.getComputedStyle(lines[0]).font;
        }
        root.appendChild(sample);
        charWidthCache = Math.max(sample.getBoundingClientRect().width / 10, 1);
        sample.remove();
        return charWidthCache;
      }
      function contentWidth(element){
        if(!element){ return 0; }
        var rect = element.getBoundingClientRect ? element.getBoundingClientRect() : null;
        var width = element.clientWidth || (rect ? rect.width : 0);
        if(!width){ return 0; }
        var style = window.getComputedStyle ? window.getComputedStyle(element) : null;
        if(style){
          width -= parseFloat(style.paddingLeft || "0") + parseFloat(style.paddingRight || "0");
        }
        return Math.max(width, 0);
      }
      function visibleMeasureLine(){
        for(var i = 0; i < lines.length; i++){
          if(lineVisible(lines[i])){ return lines[i]; }
        }
        return lines[0] || null;
      }
      function commentLineChars(){
        if(commentColumnCache){ return commentColumnCache; }
        var line = visibleMeasureLine();
        var tree = root.querySelector(".kdoc-tree");
        var width = contentWidth(tree);
        if(!width && line){ width = contentWidth(line); }
        if(!width){ width = contentWidth(root); }
        if(!width && global.innerWidth){ width = Math.max(global.innerWidth - 48, 0); }
        var gutter = line ? line.querySelector(".kdoc-fold,.kdoc-gutter") : null;
        var text = line ? line.querySelector(".kdoc-yaml-text") : null;
        if(text && (!window.getComputedStyle || window.getComputedStyle(text).display !== "inline")){
          width = contentWidth(text) || width;
          gutter = null;
        }
        if(gutter){ width -= gutter.getBoundingClientRect().width; }
        commentColumnCache = Math.max(Math.floor(Math.max(width, 0) / charWidth()), 32);
        return commentColumnCache;
      }
      function splitLongWord(out, word, limit){
        while(word.length > limit){
          out.push(word.slice(0, limit));
          word = word.slice(limit);
        }
        return word;
      }
      function wrapCommentText(text, firstLimit, nextLimit){
        var words = String(text || "").trim().split(/\s+/).filter(Boolean);
        var out = [];
        var current = "";
        function limit(){ return out.length === 0 ? firstLimit : nextLimit; }
        words.forEach(function(word){
          var currentLimit = Math.max(limit(), 1);
          if(word.length > currentLimit){
            if(current){
              out.push(current);
              current = "";
              currentLimit = Math.max(limit(), 1);
            }
            word = splitLongWord(out, word, currentLimit);
            if(!word){ return; }
          }
          if(!current){
            current = word;
            return;
          }
          if(current.length + 1 + word.length <= Math.max(limit(), 1)){
            current += " " + word;
            return;
          }
          out.push(current);
          current = word;
        });
        if(current){ out.push(current); }
        return out.length ? out : [""];
      }
      function renderCommentLine(prefix, text){
        return "<span class=\"kdoc-comment-line\"><span class=\"kdoc-yaml-comment kdoc-comment-prefix\">" + escapeHTML(prefix) + "</span><span class=\"kdoc-yaml-comment kdoc-comment-body\">" + escapeHTML(text) + "</span></span>";
      }
      function renderOriginalComment(state, wrapState){
        if(state.line){ state.line.classList.remove("kdoc-comment-reflow-hidden"); }
        state.comment.innerHTML = "<span class=\"kdoc-yaml-comment kdoc-comment-prefix\">" + escapeHTML(state.firstPrefix) + "</span><span class=\"kdoc-yaml-comment kdoc-comment-body\">" + escapeHTML(state.text) + "</span>";
        state.wrapState = wrapState;
      }
      function renderWrappedComment(state, text, lineChars, wrapState){
        var firstLimit = Math.max(lineChars - state.firstPrefix.length, 8);
        var nextLimit = Math.max(lineChars - state.nextPrefix.length, 8);
        var chunks = wrapCommentText(text, firstLimit, nextLimit);
        state.comment.innerHTML = chunks.map(function(chunk, index){
          return renderCommentLine(index === 0 ? state.firstPrefix : state.nextPrefix, chunk);
        }).join("");
        state.wrapState = wrapState;
      }
      function groupVisible(group){
        return group.states.some(function(state){ return state.line && lineVisible(state.line); });
      }
      function renderCommentGroup(group, wrapped, lineChars){
        if(wrapped && !groupVisible(group)){ return false; }
        var wrapState = wrapped ? "wrap:" + lineChars + ":" + group.text : "nowrap";
        if(group.wrapState === wrapState){ return false; }
        if(!wrapped){
          group.states.forEach(function(state){ renderOriginalComment(state, wrapState); });
          group.wrapState = wrapState;
          return true;
        }
        if(group.states.length === 1 || !group.key){
          group.states.forEach(function(state){
            if(state.line){ state.line.classList.remove("kdoc-comment-reflow-hidden"); }
            renderWrappedComment(state, state.text, lineChars, wrapState);
          });
          group.wrapState = wrapState;
          return true;
        }
        group.states.forEach(function(state, index){
          if(state.line){ state.line.classList.toggle("kdoc-comment-reflow-hidden", index > 0); }
          if(index === 0){
            renderWrappedComment(state, group.text, lineChars, wrapState);
            return;
          }
          state.wrapState = wrapState;
        });
        group.wrapState = wrapState;
        return true;
      }
      function applyCommentWrap(){
        var wrapped = wrapComments ? wrapComments.checked : options.wrapComments !== false;
        var lineChars = wrapped ? commentLineChars() : 0;
        var changed = false;
        root.classList.toggle("kdoc-wrap-comments", wrapped);
        commentGroups.forEach(function(group){
          if(renderCommentGroup(group, wrapped, lineChars)){ changed = true; }
        });
        if(changed){ applyFilterHighlights(); }
      }
      function scheduleCommentWrap(){
        var wrapped = wrapComments ? wrapComments.checked : options.wrapComments !== false;
        if(!wrapped || resizeFrame){ return; }
        resizeFrame = window.requestAnimationFrame(function(){
          resizeFrame = 0;
          applyCommentWrap();
        });
      }
      function fallbackDetail(line){
        var path = line.getAttribute("data-path");
        var text = cleanLineText(line);
        var html = "";
        if(path){
          html += "<dl class=\"kdoc-detail-grid\"><div class=\"kdoc-detail-row\"><dt>Path</dt><dd><code class=\"kdoc-detail-code\">" + escapeHTML(path) + "</code></dd></div></dl>";
        }
        if(text){
          html += "<section class=\"kdoc-detail-section\"><p class=\"kdoc-detail-description\">" + escapeHTML(text) + "</p></section>";
        }
        return html || "<p class=\"kdoc-detail-empty\">No field details.</p>";
      }
      function showDetails(line){
        if(details){
          var detailHTML = line.getAttribute("data-detail-html");
          if(detailHTML){
            details.innerHTML = detailHTML;
          } else {
            details.innerHTML = fallbackDetail(line);
          }
        }
      }
      function updateFilterOverlay(){
        if(!filterOverlay){ return; }
        if(!filterQuery){
          filterOverlay.hidden = true;
          filterOverlay.textContent = "";
          return;
        }
        filterOverlay.hidden = false;
        filterOverlay.textContent = "filter: " + filterQuery;
      }
      function expandAncestors(line){
        ancestorFieldLines(line).forEach(function(ancestor){ setExpanded(ancestor, true); });
      }
      function clearFilter(){
        var line = currentLine;
        var path = line ? line.getAttribute("data-path") || "" : "";
        restoreFilterFoldSnapshot();
        filterQuery = "";
        activeFilterState = null;
        endFilterSession();
        updateFilterOverlay();
        if(line){ expandAncestors(line); }
        if(fullSchema && renderedFullProjection){
          renderFullFoldProjection(path, true);
          return;
        }
        applyFolds();
        scheduleCommentWrap();
        select(line || visibleFieldLines()[0] || lines[0], {scroll:true});
      }
      function acceptFilter(){
        var line = currentLine;
        var path = line ? line.getAttribute("data-path") || "" : "";
        visibleFieldLines().forEach(function(field){ expandAncestors(field); });
        filterQuery = "";
        activeFilterState = null;
        endFilterSession();
        updateFilterOverlay();
        if(fullSchema && renderedFullProjection){
          renderFullFoldProjection(path, true);
          return;
        }
        applyFolds();
        scheduleCommentWrap();
        select(line || visibleFieldLines()[0] || lines[0], {scroll:true});
      }
      function ensureFilteredFocus(){
        if(currentLine && lineVisible(currentLine)){
          select(currentLine, {scroll:true});
          return;
        }
        select(visibleFieldLines()[0] || lines[0], {scroll:true});
      }
      function recordFilterApply(start, value){
        recordPerf("filter-apply", start, {
          queryLength: String(value || "").length,
          lines: lines.length,
          fields: fieldStates.length,
          visibleLines: filterQuery ? filterVisibleLines.length : visibleFieldLines().length,
          full: !!fullSchema,
          renderedFull: !!renderedFullProjection
        });
      }
      function setFilter(value){
        var filterStart = perfNow();
        value = String(value || "");
        if(value && !filterQuery){ beginFilterSession(); }
        if(!value && filterQuery){
          clearFilter();
          return;
        }
        filterQuery = value;
        activeFilterState = null;
        activeFilterProjection = null;
        var path = currentLine ? currentLine.getAttribute("data-path") || "" : "";
        if(filtering && filterQuery && fullSchema){
          updateFilterOverlay();
          renderFullFilterProjection(path, true);
          ensureFilteredFocus();
          recordFilterApply(filterStart, value);
          return;
        }
        if(filtering && filterQuery && viewSchema && viewSchema.complete === false){
          updateFilterOverlay();
          if(pendingFullFilterRender){
            clearFilterHighlights();
            return;
          }
          pendingFullFilterRender = true;
          var requestedFullSchema = requestFullSchema(function(schema){
            pendingFullFilterRender = false;
            if(!schema || !filterQuery){ return; }
            var focusPathValue = currentLine ? currentLine.getAttribute("data-path") || path : path;
            var callbackFilterStart = perfNow();
            renderFullFilterProjection(focusPathValue, true);
            ensureFilteredFocus();
            recordFilterApply(callbackFilterStart, filterQuery);
          });
          if(requestedFullSchema){
            clearFilterHighlights();
            return;
          }
          pendingFullFilterRender = false;
        }
        updateFilterOverlay();
        applyFolds();
        ensureFilteredFocus();
        recordFilterApply(filterStart, value);
      }
      function filterKey(event){
        if(!filtering){ return ""; }
        if(event.key === "/" || event.key.length !== 1){ return ""; }
        if(event.key < " " || event.key === "\x7f"){ return ""; }
        return event.key;
      }
      function clearFilterHighlights(){
        highlightedElements.forEach(function(element){
          if(!element){ return; }
          Array.prototype.slice.call(element.querySelectorAll("mark.kdoc-filter-hit")).forEach(function(mark){
            mark.replaceWith(document.createTextNode(mark.textContent || ""));
          });
          element.normalize();
        });
        highlightedElements = [];
      }
      function highlightTextNode(node, query, needle){
        var value = node.nodeValue || "";
        var lower = value.toLowerCase();
        var index = lower.indexOf(needle);
        if(index < 0){ return; }
        var fragment = document.createDocumentFragment();
        var remaining = value;
        var remainingLower = lower;
        while(index >= 0){
          if(index > 0){ fragment.appendChild(document.createTextNode(remaining.slice(0, index))); }
          var hit = document.createElement("mark");
          hit.className = "kdoc-filter-hit";
          hit.textContent = remaining.slice(index, index + query.length);
          fragment.appendChild(hit);
          remaining = remaining.slice(index + query.length);
          remainingLower = remainingLower.slice(index + query.length);
          index = remainingLower.indexOf(needle);
        }
        if(remaining){ fragment.appendChild(document.createTextNode(remaining)); }
        node.replaceWith(fragment);
      }
      function highlightElement(element, query){
        var needle = query.toLowerCase();
        var walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT, {
          acceptNode: function(node){
            if(!node.nodeValue || node.parentElement.closest("mark.kdoc-filter-hit")){ return NodeFilter.FILTER_REJECT; }
            return node.nodeValue.toLowerCase().indexOf(needle) >= 0 ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
          }
        });
        var nodes = [];
        while(walker.nextNode()){ nodes.push(walker.currentNode); }
        nodes.forEach(function(node){ highlightTextNode(node, query, needle); });
        if(nodes.length && highlightedElements.indexOf(element) < 0){ highlightedElements.push(element); }
      }
      function highlightElementIfContains(element, textLower, query){
        if(!element || !query || textLower.indexOf(query) < 0){ return; }
        highlightElement(element, query);
      }
      function applyFilterHighlights(){
        clearFilterHighlights();
        if(!filterQuery){ return; }
        var query = filterQuery.toLowerCase();
        var filterState = currentFilterState();
        if(!filterState){ return; }
        var visibleLines = filterAllowedLines();
        var highlightedCommentGroups = new Set();
        filterState.highlightLineStates.forEach(function(state){
          if(state.commentGroup && state.commentGroup.states.length > 1){
            var group = state.commentGroup;
            if(highlightedCommentGroups.has(group)){ return; }
            highlightedCommentGroups.add(group);
            if(!group.states.some(function(groupState){ return visibleLines.has(groupState.line); })){ return; }
            var leader = group.states[0];
            if(!leader || !leader.textElement){ return; }
            highlightElementIfContains(leader.textElement, group.textLower, query);
            return;
          }
          if(!visibleLines.has(state.line)){ return; }
          var text = state.textElement;
          if(!text){ return; }
          highlightElementIfContains(text, state.textLower, query);
          var fieldState = state.fieldState || state;
          var pathHit = fieldState.pathHit || "";
          if(pathHit && state.textLower.indexOf(pathHit.toLowerCase()) >= 0){ highlightElement(text, pathHit); }
        });
      }
      function clearSelection(){
        root.querySelectorAll(".kdoc-selected").forEach(function(item){ item.classList.remove("kdoc-selected"); });
        selectedLines = [];
      }
      function documentTop(element){
        var rect = element && element.getBoundingClientRect ? element.getBoundingClientRect() : null;
        return rect ? rect.top + (window.pageYOffset || 0) : 0;
      }
      function fitsScrollContext(anchor, line){
        if(!anchor || !line || anchor === line){ return false; }
        return documentTop(line) - documentTop(anchor) < Math.max(160, window.innerHeight * 0.65);
      }
      function versionContextAnchor(version){
        if(!version || version === root){ return root; }
        return root.querySelector(".kdoc-version") === version ? root : version;
      }
      function scrollAnchorForSelection(line){
        var state = lineState(line);
        if(!state || !state.field || firstVisibleFieldLineInVersion(state.version) !== line){ return line; }
        var firstLine = firstVisibleLineInVersion(state.version);
        var context = versionContextAnchor(state.version);
        return fitsScrollContext(context, line) ? context : (firstLine || line);
      }
      function scrollSelectionIntoView(line){
        var anchor = scrollAnchorForSelection(line);
        if(!anchor || !anchor.scrollIntoView){ return; }
        anchor.scrollIntoView({block: anchor === line ? "nearest" : "start", inline:"nearest"});
      }
      function select(line, options){
        if(!line){ return; }
        options = options || {};
        var previousScope = currentVersionSection();
        var fieldLine = fieldLineFor(line);
        if(fieldLine){
          line = fieldLine;
        }
        currentLine = line;
        if(filterQuery && previousScope !== currentVersionSection()){
          activeFilterState = null;
          activeFilterProjection = null;
          applyFolds();
        }
        clearSelection();
        selectedLines = groupedLines(line);
        selectedLines.forEach(function(item){ item.classList.add("kdoc-selected"); });
        showDetails(line);
        if(options.scroll && line.scrollIntoView){
          scrollSelectionIntoView(line);
        }
        if(options.updateHash !== false){
          replaceHash(line.getAttribute("data-path") || "");
        }
      }
      function typingTarget(target){
        return !!(target && (target.closest("input,textarea,select") || target.isContentEditable));
      }
      function requestQuit(){
        if(!quitURL){ return false; }
        try {
          if(navigator.sendBeacon){
            navigator.sendBeacon(quitURL, "");
          } else {
            fetch(quitURL, {method:"POST", keepalive:true}).catch(function(){});
          }
        } catch(_err) {}
        return true;
      }
      function flushFullSchemaCallbacks(schema){
        var callbacks = fullSchemaCallbacks.slice();
        fullSchemaCallbacks = [];
        callbacks.forEach(function(item){ item(schema); });
      }
      function requestFullSchema(callback){
        if(fullSchema){
          ensureFullSchemaIndex();
          if(callback){ callback(fullSchema); }
          return true;
        }
        if(loadingFullSchema){
          if(callback){ fullSchemaCallbacks.push(callback); }
          return true;
        }
        if(!mountedOptions.loadFullSchema){ return false; }
        if(callback){ fullSchemaCallbacks.push(callback); }
        loadingFullSchema = true;
        var loadStart = perfNow();
        Promise.resolve(mountedOptions.loadFullSchema()).then(function(schema){
          loadingFullSchema = false;
          if(destroyed){
            fullSchemaCallbacks = [];
            return;
          }
          if(!schema){
            flushFullSchemaCallbacks(null);
            return;
          }
          var activateStart = perfNow();
          fullSchema = schema;
          fullSchemaIndex = buildSchemaIndex(schema);
          viewSchema = viewSchema || schema;
          mountedOptions.loadFullSchema = null;
          flushFullSchemaCallbacks(schema);
          recordPerf("full-schema-activate", activateStart, {
            lines: (schema.lines || []).length,
            fields: (schema.fields || []).length,
            renderedLines: lines.length
          });
          recordPerf("full-schema-load", loadStart, {
            lines: (schema.lines || []).length,
            fields: (schema.fields || []).length
          });
        }).catch(function(error){
          loadingFullSchema = false;
          flushFullSchemaCallbacks(null);
          if(global.console && console.error){ console.error("kubectl-doc schema failed to load", error); }
        });
        return true;
      }
      function cancelFullSchemaPreload(){
        if(!fullSchemaPreloadHandle){ return; }
        if(fullSchemaPreloadHandleType === "idle" && global.cancelIdleCallback){
          global.cancelIdleCallback(fullSchemaPreloadHandle);
        } else if(fullSchemaPreloadHandleType === "frame" && global.cancelAnimationFrame){
          global.cancelAnimationFrame(fullSchemaPreloadHandle);
        } else if(global.clearTimeout) {
          clearTimeout(fullSchemaPreloadHandle);
        }
        fullSchemaPreloadHandle = 0;
        fullSchemaPreloadHandleType = "";
      }
      function cancelInitialFocus(){
        if(!initialFocusHandle){ return; }
        if(initialFocusHandleType === "frame" && global.cancelAnimationFrame){
          global.cancelAnimationFrame(initialFocusHandle);
        } else if(global.clearTimeout) {
          clearTimeout(initialFocusHandle);
        }
        initialFocusHandle = 0;
        initialFocusHandleType = "";
      }
      function scheduleInitialFocus(){
        if(!autoFocus || !scopedKeyboard){ return; }
        var run = function(){
          initialFocusHandle = 0;
          initialFocusHandleType = "";
          if(destroyed){ return; }
          focusHost();
        };
        if(global.requestAnimationFrame){
          initialFocusHandle = global.requestAnimationFrame(run);
          initialFocusHandleType = "frame";
          return;
        }
        if(global.setTimeout){
          initialFocusHandle = setTimeout(run, 0);
          initialFocusHandleType = "timeout";
          return;
        }
        run();
      }
      function scheduleFullSchemaPreload(){
        if(mountedOptions.preloadFullSchema === false || fullSchema || loadingFullSchema || !mountedOptions.loadFullSchema){ return; }
        var run = function(){
          fullSchemaPreloadHandle = 0;
          fullSchemaPreloadHandleType = "";
          if(destroyed){ return; }
          requestFullSchema();
        };
        var runAfterFrame = function(){
          if(destroyed){ return; }
          if(global.requestIdleCallback){
            fullSchemaPreloadHandle = global.requestIdleCallback(run, {timeout: 1000});
            fullSchemaPreloadHandleType = "idle";
            return;
          }
          if(global.setTimeout){
            fullSchemaPreloadHandle = setTimeout(run, 0);
            fullSchemaPreloadHandleType = "timeout";
            return;
          }
          run();
        };
        if(global.requestAnimationFrame){
          fullSchemaPreloadHandle = global.requestAnimationFrame(runAfterFrame);
          fullSchemaPreloadHandleType = "frame";
          return;
        }
        runAfterFrame();
      }
      function handleCursorKey(event){
        if(event.defaultPrevented || typingTarget(event.target)){ return false; }
        if(scopedKeyboard && !hostHasFocus()){ return false; }
        if(event.altKey || event.ctrlKey || event.metaKey){ return false; }
        var handled = false;
        if(event.key === "Escape" && filterQuery){
          clearFilter();
          handled = true;
        } else if(event.key === "Enter" && filterQuery){
          acceptFilter();
          handled = true;
        } else if(filtering && event.key === "Backspace" && filterQuery){
          setFilter(filterQuery.slice(0, -1));
          handled = true;
        } else {
          var typed = filterKey(event);
          if(filtering && typed){
            setFilter(filterQuery + typed);
            handled = true;
          }
        }
        if(!handled){ switch(event.key){
        case "ArrowUp":
          handled = selectFieldByOffset(-1);
          break;
        case "ArrowDown":
          handled = selectFieldByOffset(1);
          break;
        case "ArrowLeft":
          handled = collapseOrParent();
          break;
        case "ArrowRight":
          handled = expandOrChild();
          break;
        case "Enter":
          handled = toggleField(currentFieldLine());
          break;
        case "Tab":
          handled = filterQuery ? selectFilterMatch(event.shiftKey ? -1 : 1) : selectFoldable(event.shiftKey ? -1 : 1);
          break;
        case "Home":
          handled = selectFirstField();
          break;
        case "End":
          handled = selectLastField();
          break;
        case "PageUp":
          handled = selectFieldByOffset(-pageFieldDistance());
          break;
        case "PageDown":
          handled = selectFieldByOffset(pageFieldDistance());
          break;
        case "Escape":
          if(backURL){
            window.location.href = backURL;
            handled = true;
          } else if(requestQuit()){
            handled = true;
          }
          break;
        } }
        if(handled){
          event.preventDefault();
          event.stopPropagation();
        }
        return handled;
      }

      function hostHasFocus(){
        return root.classList.contains("kdoc-has-focus") || !!(global.document && root.contains(document.activeElement));
      }
      function markHostFocused(){
        if(!scopedKeyboard){ return; }
        root.classList.add("kdoc-has-focus");
      }
      function focusHost(){
        if(!scopedKeyboard){ return; }
        if(root.focus){ root.focus({preventScroll:true}); }
        markHostFocused();
      }
      function handleRootClick(event){
        focusHost();
        var toggle = event.target.closest("[data-kdoc-toggle]");
        if(toggle){
          var line = toggle.closest("[data-kdoc-line]");
          toggleField(line);
          return;
        }
        var line = event.target.closest("[data-kdoc-line]");
        if(line){ select(line); }
      }
      function handleWrapChange(){
        applyCommentWrap();
      }
      function handleResize(){
        commentColumnCache = 0;
        scheduleCommentWrap();
      }
      function focusHashPath(scroll){
        var path = decodedHashPath();
        if(!path){ return false; }
        return focusPath(path, {scroll: scroll !== false, updateHash:false});
      }
      function handleHashNavigation(){
        if(suppressHashNavigation){ return; }
        focusHashPath(true);
      }
      function handleFocusIn(){
        markHostFocused();
        requestFullSchema();
      }
      function handleFocusOut(event){
        var next = event.relatedTarget;
        if(!next || !root.contains(next)){ root.classList.remove("kdoc-has-focus"); }
      }
      function elementVisible(element){
        if(!element){ return false; }
        var style = global.getComputedStyle ? global.getComputedStyle(element) : null;
        if(style && (style.display === "none" || style.visibility === "hidden" || style.opacity === "0")){ return false; }
        var rect = element.getBoundingClientRect ? element.getBoundingClientRect() : null;
        return !rect || (rect.width > 0 && rect.height > 0);
      }
      function releaseStaleConsentBackdrop(){
        if(!root.classList.contains("kdoc-embedded-host") || !global.document){ return; }
        var backdrop = document.querySelector(".onetrust-pc-dark-filter");
        if(!backdrop){ return; }
        var dialog = document.getElementById("onetrust-pc-sdk");
        var banner = document.getElementById("onetrust-banner-sdk");
        if(!elementVisible(dialog) && !elementVisible(banner)){
          backdrop.style.pointerEvents = "none";
        }
      }
      function scheduleConsentBackdropRelease(){
        releaseStaleConsentBackdrop();
        if(!root.classList.contains("kdoc-embedded-host") || !global.setTimeout){ return; }
        staleBackdropTimers.push(setTimeout(releaseStaleConsentBackdrop, 250));
        staleBackdropTimers.push(setTimeout(releaseStaleConsentBackdrop, 1000));
      }

      function decodePath(path){
        path = String(path || "");
        try {
          return decodeURIComponent(path);
        } catch(_err) {
          return path;
        }
      }
      function fieldStateByPath(path){
        path = normalizePath(path);
        for(var i = 0; i < fieldStates.length; i++){
          if(fieldStates[i].path === path){ return fieldStates[i]; }
        }
        return null;
      }
      function modelFieldStateByPath(index, path){
        path = normalizePath(path);
        if(!index){ return null; }
        for(var i = 0; i < index.fieldStates.length; i++){
          if(index.fieldStates[i].path === path){ return index.fieldStates[i]; }
        }
        return null;
      }
      function expandModelAncestors(state){
        if(!state || !state.ancestors){ return; }
        state.ancestors.forEach(function(ancestor){
          if(ancestor.path){ foldStateByPath[ancestor.path] = true; }
        });
      }
      function canonicalPathForState(state, fallback){
        return String((state && state.line && state.line.getAttribute("data-path")) || (state && state.model && state.model.path) || fallback || "");
      }
      function focusPath(path, options){
        var requestedPath = decodePath(path);
        var normalized = normalizePath(requestedPath);
        options = options || {scroll:true};
        if(!normalized){ return false; }

        var state = fieldStateByPath(normalized);
        if(state){
          expandAncestors(state.line);
          applyFolds();
          scheduleCommentWrap();
          select(state.line, options);
          return true;
        }

        if(options.projectFull === false){ return false; }

        var index = ensureFullSchemaIndex();
        var modelState = modelFieldStateByPath(index, normalized);
        if(modelState && fullSchema){
          expandModelAncestors(modelState);
          renderFullFoldProjection(canonicalPathForState(modelState, requestedPath), options.scroll !== false, options);
          return true;
        }

        if(viewSchema && viewSchema.complete === false){
          return requestFullSchema(function(schema){
            if(schema && !destroyed){ focusPath(requestedPath, options); }
          });
        }
        return false;
      }
      function setPathExpanded(path, value, options){
        path = normalizePath(path || "");
        if(!path){ return false; }
        for(var i = 0; i < fieldStates.length; i++){
          if(fieldStates[i].path === path){
            setExpanded(fieldStates[i].line, value, options);
            applyFolds();
            scheduleCommentWrap();
            return true;
          }
        }
        return false;
      }
      function foldSnapshot(){
        var foldStates = [];
        fieldStates.forEach(function(state){
          if(button(state.line)){
            foldStates.push({path: state.path, expanded: expanded(state.line)});
          }
        });
        return foldStates;
      }
      function restoreFoldSnapshot(targetController, foldStates){
        if(!targetController || !foldStates){ return; }
        foldStates.forEach(function(item){
          if(!item.path){ return; }
          if(item.expanded && targetController.expandPath){ targetController.expandPath(item.path); }
          if(!item.expanded && targetController.collapsePath){ targetController.collapsePath(item.path); }
        });
      }

      root.addEventListener("click", handleRootClick, true);
      root.addEventListener("focusin", handleFocusIn);
      root.addEventListener("focusout", handleFocusOut);
      var keyTarget = document;
      keyTarget.addEventListener("keydown", handleCursorKey);
      if(wrapComments){
        wrapComments.addEventListener("change", handleWrapChange);
      }
      observeHostTheme();
      window.addEventListener("resize", handleResize);
      window.addEventListener("hashchange", handleHashNavigation);
      window.addEventListener("popstate", handleHashNavigation);
      initializeFromDOM();
      applyCommentWrap();
      applyFolds();
      scheduleConsentBackdropRelease();
      if(!focusHashPath(true)){
        select(visibleFieldLines()[0] || lines[0], {updateHash:false});
      }
      recordPerf("mount", mountStart, {
        lines: lines.length,
        fields: fieldStates.length,
        complete: !!(viewSchema && viewSchema.complete)
      });
      scheduleFullSchemaPreload();
      scheduleInitialFocus();

      controller = {
        root: root,
        destroy: function(){
          destroyed = true;
          cancelFullSchemaPreload();
          cancelInitialFocus();
          root.removeEventListener("click", handleRootClick, true);
          root.removeEventListener("focusin", handleFocusIn);
          root.removeEventListener("focusout", handleFocusOut);
          keyTarget.removeEventListener("keydown", handleCursorKey);
          if(wrapComments){ wrapComments.removeEventListener("change", handleWrapChange); }
          window.removeEventListener("resize", handleResize);
          window.removeEventListener("hashchange", handleHashNavigation);
          window.removeEventListener("popstate", handleHashNavigation);
          if(themeObserver){ themeObserver.disconnect(); }
          if(!explicitTheme){
            root.removeAttribute("data-kdoc-theme");
            root.removeAttribute("data-kdoc-managed-theme");
          }
          staleBackdropTimers.forEach(function(timer){ clearTimeout(timer); });
          staleBackdropTimers = [];
          clearSelection();
          clearFilterHighlights();
          root.__kubectlDocController = null;
          root.classList.remove("kdoc-has-focus");
        },
        snapshot: function(){
          return {
            currentPath: currentLine ? currentLine.getAttribute("data-path") || "" : "",
            filter: filterQuery,
            folds: foldSnapshot(),
            hasFocus: hostHasFocus()
          };
        },
        focusPath: focusPath,
        expandPath: function(path){ return setPathExpanded(path, true); },
        collapsePath: function(path){ return setPathExpanded(path, false); },
        setFocused: function(value){
          if(value){ focusHost(); } else { root.classList.remove("kdoc-has-focus"); }
        },
        setFilter: setFilter,
        clearFilter: clearFilter
      };
      root.__kubectlDocController = controller;
      return controller;
  }

  global.KubectlDoc = global.KubectlDoc || {};
  global.KubectlDoc.mount = mount;
  ready(function(){
    document.querySelectorAll("[data-kubectl-doc]").forEach(function(root){
      mount(root);
    });
  });
})(window);
