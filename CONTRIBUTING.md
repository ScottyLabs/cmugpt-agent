# Contributing

Thank you for your interest in contributing to Bark. This repository contains
the agent, the Python service behind the Bark web app. The web app itself is
maintained in the
[Surface repository](https://git.cmu.dev/ScottyLabs/cmugpt-surface). The
project is hosted on
[git.cmu.dev](https://git.cmu.dev/ScottyLabs/cmugpt-agent). Every pull
request is checked by CI and deployed as a preview by
[Kennel](https://git.cmu.dev/ScottyLabs/kennel), the ScottyLabs deployment
platform. Setup and architecture are described in the [README](README.md).

## Overview

The contribution process, in outline:

1. For anything larger than a bug fix, start from an issue. Choose an
   existing one from the tracker or open a new one.
2. Set up a local environment as described in the README.
3. Make the change on a branch, with tests, and update the README where it
   is affected.
4. Write each commit as a single change with a Conventional Commits subject.
5. Open a pull request against `main`, complete the template, and wait for
   CI and the preview deployment.

## What to contribute

Bug fixes, unit tests for untested behavior, documentation fixes, and small
refactors that do not change behavior may be submitted directly as pull
requests. A bug fix should include a test that fails without it.

Larger changes should begin with an issue, so that the approach is agreed on
before the work is done. Check the
[issue tracker](https://git.cmu.dev/ScottyLabs/cmugpt-agent/issues) first,
since an issue you can take up may already exist, and open a new one
otherwise. Changes in this category include new tools or tool groups,
changes to how the agent selects tools, prompt changes that alter answers,
new dependencies, and any change to authentication, secrets, or deployment.

The following will not be accepted:

- generated code that the author has not read and verified (see AI
  assistance below),
- reformatting of files that the change does not otherwise touch, and
- a new dependency for functionality the standard library already provides.

## Questions, bugs, and security

Questions should be asked in Slack.

Bugs should be reported in the
[issue tracker](https://git.cmu.dev/ScottyLabs/cmugpt-agent/issues). A report
should include the query that was sent, the model in use, the answer
received, the answer expected, and whether the problem occurred on
bark.scottylabs.org or on a local build. For a local build, include the
agent's log lines for the request, which show which tools ran.

Security problems should be reported to a maintainer in Slack rather than in
a public issue, so that a fix can be released before the details become
public.

## Setting up

Follow the [Installation](README.md#installation) section of the README. It
installs the dependencies with uv, sets up the Postgres database used for
memory, and creates the `.env` file that holds the API keys. As the README
notes, the devenv shell is an alternative to installing Postgres by hand. It
also installs the git hooks that CI runs. Without it, the checks described
under Checks must be run manually before pushing.

The service listens on port 5055. To exercise a change through the web app,
run Surface locally with `AGENT_API_URL` set to `http://127.0.0.1:5055` and
the same `AGENT_SHARED_SECRET` in both `.env` files. The secret may be left
empty on both sides for local use, in which case authentication is skipped.
Production refuses an empty secret.

## Making your change

Create a branch from `main`. A branch should address a single concern.

### Tests

The unit tests run offline and must pass before every push:

```sh
DATABASE_URL="" uv run pytest
```

An empty `DATABASE_URL` selects the in-memory store in place of Postgres. The
tests do not require a database, and clearing the variable makes them
independent of whether Postgres is running. CI runs them the same way.

A change in behavior should be accompanied by a unit test that covers it. A
bug fix should include a test that fails without the fix.

The evals form a second, slower tier. They call OpenRouter and the MCP server
and are billed to whichever API keys are configured. Run them when a change
touches prompts or tool selection:

```sh
uv run pytest evals
```

They require `OPENROUTER_API_KEY` and `MCP_SERVER_URL` in the environment and
are skipped when either is missing.

### Checks

CI runs the git hooks that devenv installs: ruff, ty, treefmt, taplo,
gitleaks, the Nix linters statix and deadnix, and a check that rejects
non-ASCII punctuation in text files. It also runs semgrep and osv-scanner.
With devenv, all of these can be run locally before pushing:

```sh
devenv shell -- prek run --all-files
```

Without devenv, run the Python checks locally and leave the rest to CI:

```sh
uv run ruff check
uv run ruff format
uv run ty check
```

A failure in CI on formatting or punctuation is reported with the line in
question. Correct it by hand and push again.

### Documentation

A change that adds or renames an environment variable, route, or command
should update the README section that documents it, in the same pull
request.

### Dependencies

Dependencies are added with uv so that the lockfile stays in sync. Commit
`pyproject.toml` and `uv.lock` together. The production build resolves
packages from the lockfile, and a stale lockfile fails the build.

```sh
uv add <package>                # runtime
uv add --group dev <package>    # development only
```

## Code style

Most of the rules in this section are enforced by tools. The remainder are
applied in review.

### Tooling

ruff enforces pycodestyle (layout, including the 88-character line limit),
Pyflakes (unused and undefined names), pyupgrade (outdated syntax), bugbear
(likely bugs), simplify (needlessly complex constructs), isort (import
order), pep8-naming (see Naming), and the pydocstyle checks on docstring form
(see Docstrings). `ruff format` determines layout, and `ruff check` applies
fixes automatically.

Rules should not be disabled for the whole project. Where a single line
requires an exception, the directive should name the rule and be accompanied
by a reason, either after the directive or in a comment directly above it. A
directive that is no longer needed should be removed.

### Syntax and types

The project targets Python 3.12 and uses its syntax: `int | None` rather
than `Optional[int]`, and built-in generics such as `list[str]` rather than
the `typing` forms.

Every function signature should be annotated, including parameters and the
return type. ty checks the annotations in CI and treats warnings as errors.
`Any` should be used only where the shape of a value is unknown, such as a
request body before validation.

### Naming

Names follow [PEP 8](https://peps.python.org/pep-0008/#naming-conventions),
and ruff's pep8-naming rules enforce the conventions. Modules and packages
have short, lowercase names. Functions, methods, variables, and arguments are
`snake_case`. Classes are `CapWords`, and exception classes end in `Error`.
Module constants are `UPPER_CASE`. A helper that is an implementation detail
of its module should be prefixed with an underscore, so that the unprefixed
names of a module form its public interface.

### Docstrings

Docstrings follow [PEP 257](https://peps.python.org/pep-0257/) in form, and
ruff enforces the parts that can be checked: the summary fits on one line and
ends with a period, a blank line separates it from any further text, and the
closing quotes of a multi-line docstring are placed on their own line.

Every module begins with a docstring stating what the module is responsible
for and, where other modules depend on it, the invariant it maintains. One
short paragraph is sufficient.

### Comments

Comments follow [PEP 8](https://peps.python.org/pep-0008/#comments). They are
complete sentences, beginning with a capital letter and ending with a period.
A block comment is placed directly above the code it describes, at the same
indentation. An inline comment is separated from the statement by two spaces
and should be used sparingly, only where it says something the statement
does not. Comments must be kept up to date when the code changes. A comment
that contradicts the code is worse than no comment.

A comment should be added only where the code could be misread. It should
state the contract or invariant a maintainer must preserve, followed by the
reason or the consequence of violating it. Details should be concrete: the
unit, the bound, or the behavior on failure. A comment should not restate
what the code plainly does. Two or three sentences is the usual length.

### Configuration and logging

Configuration is read through `get_settings()` from `cmugpt.settings`, called
from within the function that needs the value rather than at import time. The
function reads the environment on each call, which allows tests to set a
variable for a single case and allows a rotated key to take effect without a
restart. `os.environ` should not be read directly.

Logging goes through a module logger, `logger = logging.getLogger(__name__)`,
never `print`. The warning level is used when the code falls back to a
degraded path, and the info level for events worth seeing in production. API
keys, the shared secret, and the text of user messages must never be logged.

## Commit messages

Commit subjects follow
[Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/):

```
<type>(<scope>): <description>
```

- `type` is one of `feat`, `fix`, `refactor`, `docs`, `test`, `build`, `ci`,
  `chore`, or `perf`.
- `scope` is optional and names the area: `api`, `graph`, `memory`, `maps`,
  `evals`, or `deploy`.
- `description` is written in the imperative mood and in lower case, without
  a trailing period. The complete subject should not exceed 72 characters.

### One change per commit

Each commit should contain a single logical change. A refactor that prepares
for a change in behavior belongs in a separate commit from that change. Most
commits need only a subject line. A body should be added when the diff alone
does not explain the reason for the change.

### AI tools

Commit messages should not mention AI tools. Co-author lines and generation
trailers are not used, and the git hook rejects co-author lines that name an
AI tool. AI use is recorded in the pull request instead, as described below.

## AI assistance

The use of AI tools to write code is permitted. The pull request template
asks how much of the change was produced with AI, on a scale from none to
all, and the answer should reflect what actually happened. Regardless of the
level, the author is responsible for every line of the pull request and
should have read all of it before committing.

Editor and agent state directories (`.claude`, `.cursor`, `.aider*`,
`.continue`, `.copilot`) are listed in `.gitignore` and should not be
force-added.

## Opening a pull request

### Before opening

- Rebase the branch onto `main` and squash any fixup commits, so that each
  commit stands on its own.
- Write the title in the form of a commit subject. The merge commit takes
  its message from the title.
- Complete the template. If the change resolves an issue, write
  `Closes #<Issue Number>` in the description.

### After opening

CI runs the hooks and the unit tests. Kennel builds the branch, deploys it as
a preview, and posts the URL in a comment on the pull request. Each push runs
the checks again and redeploys the preview, and closing the pull request
removes it. If a check fails, the CI log identifies the failing line, and
pushing a fix runs the check again.

### Review

Another member reviews the pull request, checking that it does what the
summary says, that the logic is sound and the code is clear, that tests cover
the change, that the README still matches, and that nothing sensitive appears
in the diff. If no review has arrived within a week, ask in Slack.

Review comments should be addressed in new commits while the review is open.
Once the pull request is approved, squash those commits into the ones they
amend and force-push. The reviewer then merges.
