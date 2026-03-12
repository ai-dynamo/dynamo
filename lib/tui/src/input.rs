// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Keyboard input handling.
//!
//! Maps crossterm key events to application actions, supporting
//! both vim-style (hjkl) and arrow-key navigation.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

/// Actions that can be triggered by keyboard input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Move selection up in the focused pane.
    MoveUp,
    /// Move selection down in the focused pane.
    MoveDown,
    /// Focus the previous (left) pane.
    FocusLeft,
    /// Focus the next (right) pane.
    FocusRight,
    /// Cycle focus to the next pane.
    CycleFocus,
    /// Force refresh all data sources.
    Refresh,
    /// Quit the application.
    Quit,
}

/// Map a key event to an application action.
pub fn map_key(event: KeyEvent) -> Option<Action> {
    // Ctrl+C always quits
    if event.modifiers.contains(KeyModifiers::CONTROL) && event.code == KeyCode::Char('c') {
        return Some(Action::Quit);
    }

    match event.code {
        // Navigation
        KeyCode::Up | KeyCode::Char('k') => Some(Action::MoveUp),
        KeyCode::Down | KeyCode::Char('j') => Some(Action::MoveDown),
        KeyCode::Left | KeyCode::Char('h') => Some(Action::FocusLeft),
        KeyCode::Right | KeyCode::Char('l') => Some(Action::FocusRight),
        KeyCode::Tab => Some(Action::CycleFocus),

        // Actions
        KeyCode::Char('r') => Some(Action::Refresh),
        KeyCode::Char('q') | KeyCode::Esc => Some(Action::Quit),

        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyEventKind;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn key_ctrl(code: KeyCode) -> KeyEvent {
        KeyEvent::new_with_kind(code, KeyModifiers::CONTROL, KeyEventKind::Press)
    }

    #[test]
    fn test_vim_navigation() {
        assert_eq!(map_key(key(KeyCode::Char('k'))), Some(Action::MoveUp));
        assert_eq!(map_key(key(KeyCode::Char('j'))), Some(Action::MoveDown));
        assert_eq!(map_key(key(KeyCode::Char('h'))), Some(Action::FocusLeft));
        assert_eq!(map_key(key(KeyCode::Char('l'))), Some(Action::FocusRight));
    }

    #[test]
    fn test_arrow_navigation() {
        assert_eq!(map_key(key(KeyCode::Up)), Some(Action::MoveUp));
        assert_eq!(map_key(key(KeyCode::Down)), Some(Action::MoveDown));
        assert_eq!(map_key(key(KeyCode::Left)), Some(Action::FocusLeft));
        assert_eq!(map_key(key(KeyCode::Right)), Some(Action::FocusRight));
    }

    #[test]
    fn test_quit_keys() {
        assert_eq!(map_key(key(KeyCode::Char('q'))), Some(Action::Quit));
        assert_eq!(map_key(key(KeyCode::Esc)), Some(Action::Quit));
        assert_eq!(map_key(key_ctrl(KeyCode::Char('c'))), Some(Action::Quit));
    }

    #[test]
    fn test_action_keys() {
        assert_eq!(map_key(key(KeyCode::Char('r'))), Some(Action::Refresh));
        assert_eq!(map_key(key(KeyCode::Tab)), Some(Action::CycleFocus));
    }

    #[test]
    fn test_unmapped_keys() {
        assert_eq!(map_key(key(KeyCode::Char('x'))), None);
        assert_eq!(map_key(key(KeyCode::F(1))), None);
        assert_eq!(map_key(key(KeyCode::Enter)), None);
    }

    #[test]
    fn test_shift_modifier_ignored_for_letters() {
        // Shift+j should not map (uppercase J is a different char)
        let shift_j = KeyEvent::new(KeyCode::Char('J'), KeyModifiers::SHIFT);
        assert_eq!(map_key(shift_j), None);
    }

    #[test]
    fn test_ctrl_c_overrides_other_ctrl() {
        // Ctrl+C should always quit regardless
        assert_eq!(map_key(key_ctrl(KeyCode::Char('c'))), Some(Action::Quit));
        // Ctrl+other should not map
        let ctrl_a = KeyEvent::new_with_kind(
            KeyCode::Char('a'),
            KeyModifiers::CONTROL,
            KeyEventKind::Press,
        );
        assert_eq!(map_key(ctrl_a), None);
    }

    #[test]
    fn test_all_nav_keys_exhaustive() {
        // Ensure every navigation key maps correctly
        let mappings = vec![
            (KeyCode::Up, Action::MoveUp),
            (KeyCode::Down, Action::MoveDown),
            (KeyCode::Left, Action::FocusLeft),
            (KeyCode::Right, Action::FocusRight),
            (KeyCode::Char('k'), Action::MoveUp),
            (KeyCode::Char('j'), Action::MoveDown),
            (KeyCode::Char('h'), Action::FocusLeft),
            (KeyCode::Char('l'), Action::FocusRight),
            (KeyCode::Tab, Action::CycleFocus),
            (KeyCode::Char('r'), Action::Refresh),
            (KeyCode::Char('q'), Action::Quit),
            (KeyCode::Esc, Action::Quit),
        ];
        for (code, expected_action) in mappings {
            assert_eq!(
                map_key(key(code)),
                Some(expected_action),
                "KeyCode::{:?} should map to {:?}",
                code,
                expected_action
            );
        }
    }

    #[test]
    fn test_function_keys_unmapped() {
        for i in 1..=12 {
            assert_eq!(map_key(key(KeyCode::F(i))), None);
        }
    }

    #[test]
    fn test_special_keys_unmapped() {
        let specials = vec![
            KeyCode::Backspace,
            KeyCode::Delete,
            KeyCode::Home,
            KeyCode::End,
            KeyCode::PageUp,
            KeyCode::PageDown,
            KeyCode::Insert,
        ];
        for code in specials {
            assert_eq!(
                map_key(key(code)),
                None,
                "KeyCode::{:?} should be unmapped",
                code
            );
        }
    }
}
