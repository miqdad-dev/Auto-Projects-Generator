### Tasks

1. **Bootstrap Repo**
   - [ ] Create `auto-projects` repo, push default branch `main`.
   - [ ] Add `.gitignore` (Python, Node, general), `requirements.txt` (openai, anthropic, tiktoken).

2. **Save Prompts**
   - [ ] Add `codex.md` (this file) to repo root.

3. **Generator Script**
   - [ ] Implement `scripts/generate_next.py`:
       - Reads `codex.md` into a prompt.
       - Picks random field.
       - Calls OpenAI (or Anthropic) using env (`PROVIDER`, `MODEL_NAME`).
       - Parses fenced filename blocks and writes files to new dated folder.
       - If parsing fails, write a fallback minimal project.

4. **GitHub Actions**
   - [ ] Create `.github/workflows/auto.yml` with daily + 5-hour cron + manual dispatch.
   - [ ] Steps: checkout, setup Python 3.11, install deps, run generator, configure git, commit & push.

5. **Secrets & Variables**
   - [ ] Add repo secrets: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` (optional).
   - [ ] Add repo variables: `PROVIDER`=`openai`, `MODEL_NAME`=`gpt-4o` (or preferred).

6. **Validation**
   - [ ] Manually trigger workflow; verify a new dated folder is created with README, code, tests.
   - [ ] Run tests locally or via CI if present.

7. **Hardening**
   - [ ] Add duplicate-project detection (compare slugs/themes).
   - [ ] Add index updater that appends the new project to top-level `README.md`.
   - [ ] Optional: cache last N themes in `.autogen/state.json`.

8. **Maintenance**
   - [ ] Review output quality weekly; refine `codex.md` prompt.
   - [ ] Adjust cron windows or LOC targets as needed.

