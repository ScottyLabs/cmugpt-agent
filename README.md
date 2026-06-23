# Python Template

This project makes use of several excellent tools from [Astral](https://github.com/astral-sh), including [`uv`](https://github.com/astral-sh/uv), [`ruff`](https://github.com/astral-sh/ruff), and [`ty`](https://github.com/astral-sh/ty).

## Setup

1. Once you have [installed `uv`](https://docs.astral.sh/uv/getting-started/installation/), install dependencies with

```sh
uv sync
```

2. Install the pre-commit hooks using

```sh
uv run pre-commit autoupdate
uv run pre-commit install --install-hooks
```

3. VS Code will prompt you to install the recommended extensions, which you should accept. If you mistakenly closed it, you can find them in `.vscode/extensions.json`.

## Usage

- Format: `uv run ruff format`
- Typecheck: `uv run ty check`
- Lint: `uv run ruff check`

To run the FastAPI app locally with `uv` (the project uses `uv` for task execution), run:

```sh
uv run python src/main.py
```

You can set the `PORT` environment variable to change the listening port (defaults to `5000`):

```sh
PORT=8080 uv run python src/main.py
```

## Deployment (Kennel)

Production runs on Kennel via devenv and secretspec. Pushes to **Codeberg** `main` trigger deploys (GitHub mirror pushes do not).

URLs:

- https://api.cmugpt-agent.scottylabs.org (custom domain)
- https://cmugpt-agent-agent-main.scottylabs.net (default Kennel URL)

Validate locally before pushing:

```sh
SECRETSPEC_PROVIDER=dotenv://.env devenv build scottylabs.kennel.config
nix build .#packages.x86_64-linux.agent
```

Set production secrets (requires `cmugpt-agent-admins` group and `bao login -method=oidc`):

```sh
secretspec set -P prod OPENROUTER_API_KEY
secretspec set -P prod MCP_SERVER_URL
secretspec set -P prod AGENT_SHARED_SECRET
secretspec check -P prod
```

## Guidelines

You should not globally disable rules enforced by `ruff` or `ty`. If absolutely necessary, you can ignore them on a line-by-line basis:

For `ty`, use ignore directives in the following order of precedence, based on what is strictly necessary.

1. `# ty: ignore[<rule>]` for ignoring single rules
1. `# ty: ignore[rule1, rule2, ...]` for ignoring multiple rules
1. `# type: ignore` or `# type: ignore[<rule>]` for ignoring all violations on that line (even if a rule is specified!)
1. The decorator `@typing.no_type_check` to suppress all violations inside a function

For `ruff`, follow the same pattern.

1. `# noqa: <rule>` for ignoring single rules
1. `# noqa: rule1, rule2, ...` for ignoring multiple rules
1. `# noqa` for ignoring all violations on that line
1. `# ruff: noqa: <rule>` for ignoring a specific rule across an entire file
1. `# ruff: noqa` for ignoring all violations across an entire file
