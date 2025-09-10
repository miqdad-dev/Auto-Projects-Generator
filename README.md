# Auto Projects Generator

Automates creation of a new mini-hard project every 5 hours and daily using OpenAI (with optional Anthropic). Each project is created as its own brand-new GitHub repository (not inside this repo).

## Overview
- Reads `codex.md` to instruct an LLM to emit a complete, runnable project as fenced filename blocks.
- Chooses a random domain (backend, frontend, systems, etc.) and creates a dated folder `YYYY-MM-DD-<slug>`.
- Parses the model output and writes files to a temporary workspace.
- Creates a new GitHub repo named after the top-level project folder emitted by the model (or a date+slug if missing), pushes the project there.
- Schedules via GitHub Actions to run daily at 09:00 UTC and every 5 hours.

## Requirements
- Python 3.11+
- Dependencies from `requirements.txt`.
- Provider credentials via env (locally from `.env`, in CI from GitHub Secrets/Variables).
- GitHub Personal Access Token with `repo` scope to create/push new repos.

## Quick Start (Local)
- Copy `.env` and set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` as needed.
- Install deps: `python -m pip install -r requirements.txt`
- Set `GH_PAT` in your environment (PAT with `repo` scope). Optionally set `GITHUB_OWNER` and `GITHUB_VISIBILITY` (`public`|`private`).
- Run generator: `python scripts/generate_next.py`
- A new repository will be created under your GitHub account and pushed automatically.
  The push happens only if tests pass; otherwise the run aborts without creating the repo or pushing.

## Environment Variables
- `PROVIDER` in `{openai,anthropic}` (default: `openai`)
- `MODEL_NAME` (default: `gpt-4o` for OpenAI; `claude-3-5-sonnet-latest` for Anthropic)
- `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` (as applicable)
- Optional `TOPIC_PLAN` — CSV list to bias selection (e.g., `security,compilers,networking`).
- `GH_PAT` — GitHub Personal Access Token with `repo` scope (required for CI repo creation)
- `GITHUB_OWNER` — GitHub username/owner (optional; auto-detected if omitted)
- `GITHUB_VISIBILITY` — `public` or `private` (default: `public`)

## CI Setup
- Add repo secrets: `OPENAI_API_KEY` (and optionally `ANTHROPIC_API_KEY`), and `GH_PAT` (PAT with `repo` scope).
- Add repo variables: `PROVIDER` (e.g., `openai`), `MODEL_NAME` (e.g., `gpt-4o`), optionally `GITHUB_OWNER`, `GITHUB_VISIBILITY`.
- Workflow at `.github/workflows/auto.yml` handles scheduling and will create separate repos instead of committing here.

## Output
- No files are added to this repo by the workflow.
- Each run creates and pushes a new repository to your GitHub account.

## Notes & Tradeoffs
- Keeps dependencies minimal and parses strict fenced filename format.
- Includes a fallback minimal project when model output is malformed.
- Uses `.autogen/state.json` to avoid duplicate slugs.
