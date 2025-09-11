# Auto Projects Generator

Automates creation of a new mini-hard project every 5 hours and daily using OpenAI (with optional Anthropic). Each project is created as its own brand-new GitHub repository (not inside this repo).

## Overview
- Reads `codex.md` to instruct an LLM to emit a complete, runnable project as fenced filename blocks.
- Chooses a random domain among: frontend web/app, backend API, CLI/desktop app, data engineering, machine learning/AI, and game development.
- Biases the implementation language toward Java and Python (OOP), but may sometimes use JS/TS, Go, or Rust when appropriate.
- Parses the model output and writes files to a temporary workspace.
- Creates a new GitHub repo named after the top-level project folder emitted by the model (or a date+slug if missing), pushes the project there.
- Schedules via GitHub Actions to run daily at 09:00 UTC and every 5 hours.
- If a static site is detected (e.g., `index.html` at root or in `docs/`), the generator enables GitHub Pages so you can preview at `https://<owner>.github.io/<repo>/`.
- Node frontends: supports npm, Yarn, and pnpm (auto-detected via lockfile). It installs deps, runs the `build` script when present, and copies common outputs (`dist`, `build`, `out`, `public`) into `docs/` before commit. For Next.js projects, it attempts `next build` + `next export -o out` and publishes `out` to `docs/`.

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
 - For frontends or static games, it will try to prepare a `docs/` site and enable GitHub Pages automatically. If Pages cannot be enabled via API due to token permissions, you can enable it manually in the repo’s Settings → Pages (Deploy from a branch: `main` and `/docs`).

## Environment Variables
- `PROVIDER` in `{openai,anthropic}` (default: `openai`)
- `MODEL_NAME` (default: `gpt-4o` for OpenAI; `claude-3-5-sonnet-latest` for Anthropic)
- `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` (as applicable)
- Optional `TOPIC_PLAN` — CSV list to bias selection (e.g., `security,compilers,networking`).
- `GH_PAT` — GitHub Personal Access Token with `repo` scope (required for CI repo creation)
- `GITHUB_OWNER` — GitHub username/owner (optional; auto-detected if omitted)
- `GITHUB_VISIBILITY` — `public` or `private` (default: `public`)

## Output Quality Guarantees
- Strong, professional README in each generated project: overview, exact run commands, examples, architecture and tradeoffs, limitations, testing, and troubleshooting.
- Non-trivial logic (e.g., state machines, parsers, concurrency, or algorithms).
- Tests included and runnable: JUnit for Java projects (Maven/Gradle), pytest for Python, or an appropriate JS test runner.
- For data engineering, includes small sample data and validation; for ML/AI, includes dataset sample and train/eval scripts with metrics; for games, includes a playable loop.

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
