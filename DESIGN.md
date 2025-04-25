# Dynamo Control (`dynamo-ctl`) - Design Document

## 1. Introduction

`dynamo-ctl` is a command-line interface (CLI) with a terminal user interface (TUI) built using Rust and `ratatui`. It provides users with the ability to monitor and manage Dynamo deployments. Its initial focus is on inspecting the state of a deployment namespace and manually scaling components (`VllmWorker`, `PrefillWorker`) by adding or removing them, mirroring the functionality exposed by `examples/planner/test_controller.py`.

**Target Users:** Developers and Operators managing Dynamo deployments.
**Technology Stack:** Rust, `ratatui`, `tokio`, `serde`.

## 2. Design Philosophy: "Neon Syndicate"

`dynamo-ctl` transcends a mere utility; it's the command center for next-generation AI inference. The design philosophy, "Neon Syndicate," embodies this with a sleek, modern, and subtly cyberpunk aesthetic, prioritizing clarity, information density, and a sense of sophisticated control. It draws inspiration from interfaces like the provided "Shadowwatch" mockup, focusing on dark themes punctuated by vibrant, data-driven highlights.

*   **Aesthetics & Theme:** Dark, sophisticated, data-centric. Imagine the control panel of a high-tech, clandestine organization managing powerful AI resources. Think deep blues, charcoal grays, and near-blacks for backgrounds, creating a high-contrast canvas for neon accents. The feel should be professional but with an edge – clean lines, sharp details, and efficient use of space.
*   **Color Palette:**
    *   **Base:** Dark Gray/Off-Black (`#1a1a1d` or similar) for primary backgrounds. Slightly lighter grays (`#2c2c2f`) for secondary panels or inactive elements.
    *   **Primary Accent:** Electric Cyan/Teal (`#00f0ff` or `#61f0de` inspired by mockup green) for primary interactive elements, borders, titles, and active indicators.
    *   **Secondary Accents:** Vibrant Magenta (`#ff00ff`), Bright Orange (`#ff8c00`), or Electric Green (`#39ff14`) for different data categories, alerts, status changes (e.g., scaling up/down), or highlighting specific metrics. Use sparingly for emphasis. Red (`#ff3333`) reserved for critical errors or destructive action confirmations.
    *   **Text:** Off-white or light gray (`#c5c6c7`) for standard text to reduce eye strain against dark backgrounds. Accent colors used for specific labels or data points.
*   **Typography:**
    *   **Primary:** A clean, modern sans-serif font (like Fira Sans, Inter, or similar if renderable).
    *   **Data/Code:** A crisp monospaced font (like Fira Code, JetBrains Mono) for displaying raw state data, logs, or code snippets, ensuring clarity and alignment. Use ligatures if possible.
*   **Layout & Structure:**
    *   **Modular Panels:** Employ `ratatui::widgets::Block` extensively to divide the UI into distinct functional areas (e.g., Namespace Info, Component Summary, State View, Actions, Status). Follow the mockup's multi-panel approach.
    *   **Information Density:** Balance showing relevant data without overwhelming the user. Use concise labels and clear visual hierarchy. Allow panels to be potentially collapsible or switchable in future versions.
    *   **Borders & Padding:** Sharp, single-line or double-line borders (`Borders::SINGLE`, `Borders::DOUBLE`), potentially using the primary accent color for focused panels. Consistent padding within blocks. Consider using box-drawing characters for subtle separators.
*   **Widget Styling:**
    *   **Blocks:** Titles should be distinct (e.g., bold, primary accent color). Borders indicate focus or category.
    *   **Tables/Lists:** Highlight selected rows with a solid block background (primary accent color, text inverted to dark). Minimal separators between rows/columns. Use accent colors within cells for status indicators (e.g., a green dot for 'Running', orange for 'Waiting').
    *   **Paragraphs (State View):** Use syntax highlighting if displaying JSON/YAML. Apply appropriate monospaced font.
    *   **Popups/Modals:** Appear centered or contextually. Dim the background UI slightly to draw focus. Use clear titles and action buttons (`[OK]`, `[Cancel]`).
    *   **Status Bar/Footer:** High contrast. Display keybindings clearly, perhaps using bracketed symbols `[N]` Namespace, `[A] Add`, `[R] Remove`, `[Q] Quit`. Status messages should be timestamped or use color to indicate severity (e.g., green for success, orange for warnings, red for errors).
*   **Feel & Interaction:**
    *   **Responsive:** Immediate feedback on keypresses. Loading states should be indicated clearly (e.g., a spinner or "Loading..." message in the status bar).
    *   **Intuitive Navigation:** Clear focus indication. Standard keybindings (`j`/`k` or arrows for lists, `Tab` potentially for panels).
    *   **Cyberpunk Elements (Subtle):** Consider occasional use of ASCII art or unicode symbols for icons (e.g., ⚡ for actions, ⚙️ for settings, 📈 for metrics - if fonts support), subtle "glitch" effects on text during errors (use very sparingly), or perhaps a slightly stylized border character set.

This "Neon Syndicate" theme aims to make `dynamo-ctl` not only functional but also visually engaging and reflective of the cutting-edge technology it helps manage.

## 3. Implementation Plan

This section outlines the development steps, focusing on delivering the v0.1 features first by replicating the local state file management approach, followed by potential future enhancements leveraging direct runtime integration.

**Phase 1: Core Functionality (v0.1 - State File Management)**

1.  **Project Setup (Done):**
    *   Create Rust binary project `dynamo-ctl` in `launch/`.
    *   Add dependencies: `ratatui`, `crossterm`, `tokio`, `serde`, `serde_json`, `glob`, `anyhow`, `figlet-rs`.
2.  **Initial TUI Skeleton (Done):**
    *   Implement terminal setup/teardown (`crossterm`, `ratatui`).
    *   Create main async loop (`tokio`).
    *   Implement initial screen:
        *   Display FIGlet banner (`DYNAMO_FIGLET`).
        *   Implement namespace text input widget.
        *   Basic event handling (typing, backspace, enter, quit).
        *   Apply basic "Neon Syndicate" styling (dark theme, cyan accents).
3.  **State Handling Module (`state.rs`):**
    *   Define Rust structs using `serde` to represent the structure of `~/.dynamo/state/{namespace}.json`. Key elements: `components` dictionary, potential resource info.
    *   Implement async function `load_state(namespace: &str) -> Result<State>`: Reads and deserializes the JSON file for the given namespace.
    *   Implement async function `save_state(namespace: &str, state: &State) -> Result<()>`: Serializes and writes the state back to the JSON file.
    *   Implement `add_component(state: &mut State, component_type: &str)`: Finds the next available index for the given component type (`VllmWorker` or `PrefillWorker`) and adds a corresponding entry to the `state.components` map. Mimics `LocalConnector.add_component` logic.
    *   Implement `remove_component(state: &mut State, component_type: &str)`: Finds the highest-indexed entry for the component type, removes it from `state.components`, and potentially clears associated resource info (e.g., `allocated_gpus`). Mimics `LocalConnector.remove_component` logic.
    *   Implement namespace discovery: Function to scan `~/.dynamo/state/` using `glob` and return a list of available namespace names (from `*.json` filenames).
    *   Use `anyhow` for robust error handling (file not found, JSON parsing errors, permission issues).
4.  **Namespace Selection View:**
    *   Modify the initial screen or create a new view/popup triggered after the banner.
    *   Use the namespace discovery function to list available namespaces.
    *   Use a `List` widget (`ratatui`) for selection.
    *   Handle navigation (up/down keys) and selection (Enter).
    *   On selection, load the state using `load_state` and transition to the main dashboard view.
5.  **Main Dashboard View (`ui.rs` or main module):**
    *   Refactor UI rendering logic into manageable functions/modules.
    *   Layout: Header (Namespace), Main Area (Component Summary, Raw State), Footer (Keybindings, Status).
    *   **Component Summary Widget:**
        *   Use a `Table` widget.
        *   Process the loaded `State` data.
        *   Display component types (`VllmWorker`, `PrefillWorker`) and the count of active instances for each.
        *   Apply "Neon Syndicate" table styling.
    *   **Raw State View Widget:**
        *   Use a `Paragraph` widget within a `Block`.
        *   Display the pretty-printed JSON string of the loaded `State`.
        *   Apply styling (monospace font, potential syntax highlighting in future).
    *   **Footer Widget:**
        *   Display static keybindings (`[N]amespace`, `[A]dd`, `[R]emove`, `[Q]uit`).
        *   Display dynamic status messages (e.g., "Loaded namespace 'xyz'", "State saved.", "Error: ...").
6.  **Action Handling:**
    *   Implement popups/modals for `Add Component` and `Remove Component` actions, triggered by `a` and `r` keys.
    *   Popups should prompt for component type (`VllmWorker`/`PrefillWorker`).
    *   Implement confirmation prompts before modifying state.
    *   On confirmation:
        *   Call the appropriate `add_component` or `remove_component` function on the in-memory `State`.
        *   Call `save_state` to persist changes to the JSON file.
        *   Update the UI (Component Summary, Raw State) to reflect the changes.
        *   Display success/error status messages in the footer.
7.  **Refinement & Styling:**
    *   Apply "Neon Syndicate" styling consistently across all widgets (colors, borders, fonts).
    *   Ensure responsive UI updates and clear feedback.
    *   Add comments and basic documentation.

**Phase 2: Direct Runtime Integration (Future Enhancements - Post v0.1)**

*   **Rust Runtime Bindings:** Investigate or develop Rust bindings for `libdynamo` or relevant parts of `DistributedRuntime`.
*   **ETCD State Reading:** Implement option to read component status directly from ETCD (`runtime.etcd_client().kv_get_prefix`) as an alternative or supplement to the local state file.
*   **NATS Metric Subscription:** Implement NATS client logic to subscribe to the `PrefillQueue` stream and display the live queue size.
*   **Component Metrics:** Explore ways to query metrics directly from `VllmWorker` components (e.g., KV cache usage) if exposed via the runtime or another mechanism.
*   **Direct Actions:** Potentially replace state file modification with direct calls to runtime APIs for scaling, if available and safe.

**Phase 3: Advanced Features (Future Enhancements)**

*   Log viewing.
*   Configuration management interface.
*   Improved state visualization (beyond raw JSON).
*   Filtering/searching components.

This phased approach ensures we deliver a functional tool based on the established `LocalConnector` pattern quickly, while paving the way for deeper integration with Dynamo's core systems later.

## 4. Core Concepts

*   **Namespace:** The fundamental identifier for a Dynamo deployment instance. All operations within `dynamo-ctl` are scoped to a selected namespace. State is persisted per namespace.
*   **State File:** The source of truth for `dynamo-ctl`'s view of the deployment. Located at `~/.dynamo/state/{namespace}.json`. Contains information about active components, watchers, and potentially resource allocations. `dynamo-ctl` reads this file and modifies it to enact changes (add/remove components).
*   **Components:** Scalable units within a namespace. Initially, `VllmWorker` and `PrefillWorker` are recognized.
*   **Component Watchers:** Entries within the state file (e.g., `{namespace}_VllmWorker_0`, `{namespace}_VllmWorker_1`) that represent instances of a component being managed. Adding/removing components manipulates these watcher entries.

## 5. Features (Initial Scope - v0.1)

*   **Namespace Management:**
    *   Scan `~/.dynamo/state/` directory to discover available namespaces (JSON files).
    *   Allow the user to select an active namespace upon startup or via a dedicated view/action.
*   **Dashboard View:**
    *   Display the currently selected namespace.
    *   List active component watchers grouped by type (`VllmWorker`, `PrefillWorker`).
    *   Show the count of active watchers for each component type.
    *   Display the raw content of the state file for the selected namespace.
*   **Component Management Actions:**
    *   Action to add a new `VllmWorker` instance (adds the next numbered watcher entry to the state file).
    *   Action to remove the latest `VllmWorker` instance (removes the highest numbered watcher entry from the state file and ensures related resource info is cleared).
    *   Action to add a new `PrefillWorker` instance.
    *   Action to remove the latest `PrefillWorker` instance.
    *   User confirmation prompts before modifying the state file.
*   **Status/Feedback:**
    *   Display status messages (e.g., "Loading state...", "Added VllmWorker_2", "Error reading state file").
    *   Clear visual indication of success or failure of actions.

## 6. TUI Design (`ratatui`)

*   **Layout:** A main vertical layout.
    *   **Header:** `Block` containing the `dynamo-ctl` title and the currently selected namespace.
    *   **Main Area:** A `Block` containing the primary view. Initially, this will show:
        *   A `Table` or `List` summarizing active component watchers (Type, Name/ID, Count).
        *   A `Paragraph` displaying the formatted JSON content of the state file.
    *   **Footer:** `Paragraph` displaying available keybindings and status messages.
*   **Widgets:**
    *   `Block`: Framing sections.
    *   `Paragraph`: For titles, status, keybindings, state file display.
    *   `Table` / `List`: To show component summary.
    *   Popups/Modals (potentially new windows or overlaid blocks): For namespace selection and action confirmations.
*   **Interaction:**
    *   `n`: Open namespace selection popup.
    *   `a`: Trigger "Add Component" flow (popup to choose type).
    *   `r`: Trigger "Remove Component" flow (popup to choose type).
    *   `q`: Quit the application.
    *   Arrow keys / `j`/`k`: Navigation within lists/tables (if needed later).
    *   `Enter`: Confirm selections in popups.
    *   `Esc`: Close popups/cancel actions.

## 7. Implementation Details

*   **Project Setup:** Standard Rust binary project (`cargo new dynamo-ctl`). Dependencies: `ratatui`, `crossterm` (backend for `ratatui`), `tokio` (for async file I/O and main loop), `serde`, `serde_json` (for state file parsing/serialization), `glob` (for namespace discovery), `anyhow` (for error handling).
*   **State Handling:**
    *   Define Rust structs mirroring the expected JSON structure of the state file (`~/.dynamo/state/{namespace}.json`). Use `serde` for deserialization/serialization.
    *   Implement functions corresponding to `LocalConnector`'s logic:
        *   `load_state(namespace: &str) -> Result<State, Error>`
        *   `save_state(namespace: &str, state: &State) -> Result<(), Error>`
        *   `add_component(state: &mut State, component_type: &str) -> bool` (Modifies state in memory)
        *   `remove_component(state: &mut State, component_type: &str) -> bool` (Modifies state in memory)
    *   File operations should be asynchronous (`tokio::fs`).
*   **TUI Loop:**
    *   Initialize `crossterm`.
    *   Enter the main loop (`loop`).
    *   Draw the UI based on the current application state (`terminal.draw(...)`).
    *   Poll for events (`crossterm::event::poll`, `crossterm::event::read`).
    *   Handle events (update app state, trigger actions like loading/saving state).
    *   Break loop on quit event.
    *   Restore terminal on exit.
*   **Error Handling:** Use `anyhow::Result` extensively. Display user-friendly errors in the status bar/footer. Prevent panics on file errors or JSON parsing errors.

## 8. Future Enhancements (Post v0.1)

*   **Live Metrics Display:** Integrate with Dynamo's metric system (if accessible) to show Prefill Queue Size, KV Cache Usage, etc. This might require understanding how the Python `planner` gets metrics (e.g., NATS, direct component calls).
*   **Log Viewing:** Stream or display logs from Dynamo components.
*   **Resource Visualization:** Show GPU allocation/usage more clearly than just raw state.
*   **Configuration Management:** View or even modify planner configurations (thresholds, budgets) if stored accessibly.
*   **More Sophisticated State View:** Instead of raw JSON, provide a structured, navigable view of the state.
*   **Direct Interaction:** Explore possibilities beyond state file manipulation if Dynamo exposes APIs or other RPC mechanisms.
*   **Filtering/Searching:** Useful if the number of components grows large.

This document serves as the initial blueprint. Details will be refined during implementation. 