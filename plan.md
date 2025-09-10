# plan.md

**Objective:**
Automate creation of a new mini-hard project every 5 hours and once daily using OpenAI (with optional Anthropic fallback), and auto-commit to GitHub.

**Plan:**
1. **Repo & Skeleton**
   - Create repo `auto-projects`.
   - Add `.github/workflows/auto.yml`, `scripts/generate_next.py`, `requirements.txt`, `.gitignore`.

2. **Prompting Strategy**
   - Store the instruction in `codex.md`.
   - Generator script injects `codex.md` as the system/user prompt and enforces fenced filename parsing.

3. **Randomization & Rotation**
   - On each run, choose a random field from the predefined list.
   - Create a unique dated folder `YYYY-MM-DD-<slug>`.

4. **CI Scheduling**
   - GitHub Actions cron at 09:00 UTC and every 5 hours.
   - Manual trigger via `workflow_dispatch` for testing.

5. **Secrets & Config**
   - Secrets: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` (optional).
   - Variables: `PROVIDER`, `MODEL_NAME`, optional `TOPIC_PLAN` (CSV of themes).

6. **Quality Control**
   - Enforce tests in each project; run `pytest` or `make test` in workflow if provided.
   - Keep deps minimal; require explicit run commands in README.

7. **Commit Hygiene**
   - Conventional Commits: `feat(auto): add project <date-slug>`.
   - One folder per project to avoid merge conflicts.

8. **Extendability**
   - Add templates per domain over time.
   - Add cache to prevent duplicate themes; maintain an index in top-level `README.md`.

