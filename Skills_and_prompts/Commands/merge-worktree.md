---
allowed-tools: Bash(git:*)
description: Merge the '$ARGUMENTS' worktree back into the main branch and clean up
---

## Runtime context
- Existing worktrees: !`git worktree list`
- Current branch: !`git branch --show-current`

## Your task

Merge the `$ARGUMENTS` worktree (located at `.trees/$ARGUMENTS`) into the current branch and clean up.

Follow exactly these steps:
1. Change into `.trees/$ARGUMENTS` and examine the latest commit: `git log -1 --stat`
2. Change back to the root project directory
3. Merge the branch: `git merge $ARGUMENTS`
4. If there are merge conflicts, identify them using `git status`, `git diff --name-only --diff-filter=U`, or `git ls-files -u`
5. Manually resolve all conflicts based on your knowledge of the changes made in the worktree
6. Stage resolved files and complete the merge commit
7. Delete the worktree and its branch:
   - `git worktree remove .trees/$ARGUMENTS`
   - `git branch -d $ARGUMENTS`
