<!-- Lines like this one are hidden when the description is rendered. Fill in each section below and delete any that does not apply. -->

## Summary
<!-- What changes and why, in one or two sentences. Write "Closes #<Issue Number>" if there is an issue. -->

## Changes
<!-- For a prompt or tool change, add the query you used to check it. -->
-

## AI use
<!-- Tick one. Whatever the level, you have read and verified every line. -->
- [ ] None
- [ ] Minor: autocomplete or suggestions for a few lines
- [ ] Some: AI drafted parts of the change, the rest was written by hand
- [ ] Most: AI wrote the bulk of the change, edited by hand
- [ ] All: AI wrote the whole change, reviewed and tested by hand

## Checklist
<!-- Tick each item before requesting review. If one does not apply, leave it unticked and say why in the summary. -->
- [ ] `DATABASE_URL="" uv run pytest` passes
- [ ] `uv run pytest evals` run, if prompts or tool selection changed
- [ ] Hooks pass (`devenv shell -- prek run --all-files`, or ruff and ty directly)
- [ ] README updated if a variable, route, or command changed
- [ ] No secrets, keys, or `.env` values in the diff
- [ ] Commits follow Conventional Commits, one change each, rebased onto `main`
- [ ] Title written like a commit subject
