Your task is to merge in the '$ARGUMENTS' worktree in the .trees/$ARGUMENTS folder.

Follow exactly these steps:
1. Change into .trees/$ARGUMENTS directory
2. Examin the changes that were made in the latest commit
3. Change back to the root directory
4. Merge in the worktree
5. There might be merge conflicts. Use "git status", "git diff --name-only --diff-filter=U", or "git ls-files -u" to list files that have merge conflicts
6. Manually resolve conflicts based on your knowledge of the changes
7. Once merged delete $ARGUMENTS worktree and corresponding branch