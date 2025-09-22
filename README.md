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

-   Format: `uv run ruff format`
-   Typecheck: `uv run ty check`
-   Lint: `uv run ruff check`

## Guidelines

You should not globally disable rules enforced by `ruff` or `ty`. If absolutely necessary, you can ignore them on a line-by-line basis:

For `ty`, use ignore directives in the following order of precedence, based on what is strictly necessary.

1. `# ty: ignore[<rule>]` for ignoring single rules
2. `# ty: ignore[rule1, rule2, ...]` for ignoring multiple rules
3. `# type: ignore` or `# type: ignore[<rule>]` for ignoring all violations on that line (even if a rule is specified!)
4. The decorator `@typing.no_type_check` to suppress all violations inside a function

For `ruff`, follow the same pattern.

1. `# noqa: <rule>` for ignoring single rules
2. `# noqa: rule1, rule2, ...` for ignoring multiple rules
3. `# noqa` for ignoring all violations on that line
4. `# ruff: noqa: <rule>` for ignoring a specific rule across an entire file
5. `# ruff: noqa` for ignoring all violations across an entire file
