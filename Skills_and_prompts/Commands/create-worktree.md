---
allowed-tools: Bash(git worktree:*), Bash(git branch:*), Bash(ls:*), Bash(ln:*), Bash(code:*), Bash(claude:*), Bash(wt:*)
description: Create a new git worktree named '$ARGUMENTS' in .trees/$ARGUMENTS
---

## Runtime context
- Existing worktrees: !`git worktree list`

## Your task

Create a new worktree named '$ARGUMENTS' in the `.trees/$ARGUMENTS` folder.

Follow exactly these steps:
1. Check if `.trees/$ARGUMENTS` already exists. If it does, stop and tell the user the worktree already exists.
2. Create a new git worktree: `git worktree add .trees/$ARGUMENTS -b $ARGUMENTS`
3. Symlink the `.venv` folder into the worktree directory:
   - On Unix/Mac: `ln -s "$(pwd)/.venv" ".trees/$ARGUMENTS/.venv"`
   - On Windows: `New-Item -ItemType SymbolicLink -Path ".trees/$ARGUMENTS/.venv" -Target "$(pwd)/.venv"`
4. Ask the user: "Are you using VS Code? (yes/no)"
   - **Yes:** Run `code .trees/$ARGUMENTS` to launch a new VS Code instance in the worktree.
   - **No:** Open a new terminal tab in the worktree directory and start a Claude Code session automatically:
     - Detect the OS by checking `$OSTYPE` (bash) or `$env:OS` (PowerShell):
       - **Windows** — run: `wt new-tab -d "$PWD\.trees\$ARGUMENTS" -- claude`
       - **macOS** — run: `osascript -e 'tell application "Terminal" to do script "cd \"'"$PWD/.trees/$ARGUMENTS"'\" && claude"'`
       - **Linux (GNOME)** — run: `gnome-terminal --working-directory="$PWD/.trees/$ARGUMENTS" -- claude`
     - If the detected terminal emulator command fails (not installed), fall back to instructing the user to open a terminal in `.trees/$ARGUMENTS` and run `claude`.
