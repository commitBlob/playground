Your task is to create a new worktree named '$ARGUMENTS' in the .trees/$ARGUMENTS folder.

Follow exactly these steps:
1. Check if an existing folder in the .trees folder with the name $ARGUMENTS already exists. If it does, stop here and tell the user that worktree already exists
2. Create a new git worktree in .trees folder with the name $ARGUMENTS.
3. Symlink the .venv folder into the worktree directory.
4. Ask the user: "Are you using VS Code? (yes/no)"
   - If yes: Launch a new VS Code editor instance in that directory by running the `code` command.
   - If no: Open a new terminal tab and run `claude -w "$ARGUMENTS"` to start a Claude Code session in the new worktree.