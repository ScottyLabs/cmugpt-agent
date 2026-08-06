# Python Template

This project makes use of several excellent tools from [Astral](https://github.com/astral-sh), including [`uv`](https://github.com/astral-sh/uv), [`ruff`](https://github.com/astral-sh/ruff), and [`ty`](https://github.com/astral-sh/ty).

## Setup

1. Once you have [installed `uv`](https://docs.astral.sh/uv/getting-started/installation/), install dependencies with

```sh
uv sync
```

Create a `.env` file with `OPENROUTER_API_KEY`, `MCP_SERVER_URL`,
`OPENAI_API_KEY`, `AGENT_SHARED_SECRET`, and `DATABASE_URL`. For durable user
memory, create a PostgreSQL database with pgvector and point `DATABASE_URL` at
it (for example `postgresql:///cmugpt_agent?host=/tmp` for a local unix-socket
server):

```sh
createdb cmugpt_agent
psql -d cmugpt_agent -c 'CREATE EXTENSION IF NOT EXISTS vector;'
```

`OPENROUTER_API_KEY` powers chat and memory extraction. `OPENAI_API_KEY` is a
real OpenAI key used for `text-embedding-3-large` semantic search. The
`AGENT_SHARED_SECRET` is a random application-to-application bearer token shared
only with the Surface server; generate one with `openssl rand -hex 32` and never
put it in browser-visible configuration.

The embedding model uses a pgvector `halfvec(3072)` HNSW index. Startup
verifies that an existing `store_vectors` table matches this shape and refuses
to start against a database initialized for a different embedding model; in
that case rebuild the `store_vectors` and `vector_migrations` tables and
re-index any memory you need to retain. A fresh database needs no preparation
beyond `CREATE EXTENSION vector`.

Long-term memory stores only durable facts: facts distilled from chats and facts
the user explicitly asks CMUGPT to remember. Raw user/assistant turns are not
stored or recalled as memory. The clear-memory endpoint also purges the legacy
episode namespace so data written by older deployments can still be removed.

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

Verify that memory is actually durable:

```sh
curl -s http://localhost:5000/api/health
```

The response must report `memory.backend` as `postgres`, `memory.ready` as
`true`, and `memory.semantic_search` as `true`. An `in-memory` backend is only a
local-development fallback and resets on process restart.

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
secretspec set -P prod OPENAI_API_KEY
secretspec set -P prod MCP_SERVER_URL
secretspec set -P prod AGENT_SHARED_SECRET
secretspec check -P prod
```

`DATABASE_URL` is not an OpenBao secret: Kennel injects it into the process
environment from its platform-managed Postgres, so it is deliberately not
declared in `secretspec.toml`. The database in `devenv.nix` fills the same
role for local development.

Production must set `AGENT_ENV=production` (the `Procfile` already does). The
agent refuses to start in production if `DATABASE_URL` or
`AGENT_SHARED_SECRET` is missing from the environment.

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
