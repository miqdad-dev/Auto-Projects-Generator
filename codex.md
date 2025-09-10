You are an elite code generator (OpenAI GPT-4o / GPT-4.1). Output a COMPLETE, RUNNABLE “mini-hard” project with non-trivial logic.

OBJECTIVE
- Build a new mini-hard project (beyond trivial examples) from a RANDOM programming field.
- Include GitHub automation to generate and commit NEW projects daily and every 5 hours.

RANDOM FIELD (choose 1 per run):
backend api; frontend web/app; systems programming; data engineering; machine learning/ai; distributed systems; devops/infrastructure; databases; networking; security; compilers/interpreters; robotics/iot; game dev; scripting/automation.

OUTPUT FORMAT (STRICT)
Return ONLY fenced file blocks, each like:

```path/filename
<file content>
````

REQUIRED FILES

1. A dated root folder: YYYY-MM-DD-<short-slug>  (use today’s UTC date).
2. Source code (≥1 meaningful module).
3. README.md that includes:

   * overview
   * run instructions (exact commands)
   * example usage
   * architecture notes & tradeoffs
4. Tests (unit/integration).
5. Dockerfile or docker-compose if relevant; otherwise Makefile or task runner.
6. Small sample data/fixtures if useful.
7. Lint/format config if appropriate.
8. A GitHub Actions workflow to schedule generation/commits.

AUTOMATION REQUIREMENTS

* Include: `.github/workflows/auto.yml`
* Triggers:

  * `cron: "0 9 * * *"`  # daily at 09:00 UTC
  * `cron: "0 */5 * * *"` # every 5 hours
  * `workflow_dispatch:`
* Workflow steps:

  * checkout
  * setup Python 3.11
  * install deps from `requirements.txt`
  * run `scripts/generate_next.py`
  * configure git user
  * commit & push if changes exist with a Conventional Commit message:
    `feat(auto): add project YYYY-MM-DD-<slug>`
* `scripts/generate_next.py` must:

  * pick a RANDOM field from the list
  * create a NEW dated folder (YYYY-MM-DD-<slug>) with a different project than before
  * call provider API (OpenAI or Anthropic) based on env:

    * `PROVIDER` in {"openai","anthropic"}
    * `OPENAI_API_KEY` for OpenAI (e.g., gpt-4o or gpt-4.1)
    * `ANTHROPIC_API_KEY` for Claude (e.g., claude-3-5-sonnet)
    * optional `MODEL_NAME`
  * embed a robust “file-emitter” prompt that asks the model to output files in the same fenced-filename format, then parse/write them to disk.

QUALITY BAR

* Non-trivial logic (algorithms, concurrency, state machines, streaming, parsers, protocol, indexing, etc.).
* Keep dependencies minimal. README must be accurate from a clean clone. LOC target: 200–600. Tests must run and be included.

DELIVERABLES

* Provide ALL files (source, README, tests, scripts, workflow, requirements, Makefile/Dockerfile, etc.) ONLY as fenced filename blocks exactly like:

```path/filename
<content>
```

