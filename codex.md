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

SPECIAL REQUIREMENTS FOR DATA ENGINEERING / MACHINE LEARNING PROJECTS:

When the selected field is "data engineering" or "machine learning/ai", create a HIGHLY ADVANCED, full-stack project with:

**Modern Stack Requirements:**
- Python 3.11+
- Apache Kafka (real-time streaming) 
- Apache Spark (ETL & processing)
- Airflow or Prefect (orchestration)
- FastAPI or Flask (API layer)
- PostgreSQL or MongoDB (storage)
- Docker + Docker Compose
- MLflow or Weights & Biases (model tracking)

**Professional Structure (root folder: <short-slug> - NO DATES):**
```
src/               # Core application code
├── ingestion/     # Data ingestion modules
├── processing/    # ETL and data processing
├── models/        # ML model definitions
├── api/           # FastAPI service code
└── monitoring/    # Monitoring and logging
data/              # Data storage directories
├── raw/           # Raw ingested data
├── processed/     # Cleaned and transformed data
└── models/        # Trained model artifacts
notebooks/         # Jupyter notebooks
├── eda/           # Exploratory data analysis
└── experiments/   # ML experiments
configs/           # Configuration files
tests/             # Unit and integration tests
├── unit/          # Unit tests
└── integration/   # Integration tests
pipeline/          # Pipeline definitions
└── airflow/       # Airflow DAGs
deployment/        # Deployment configurations
├── docker-compose.yml
└── kubernetes/    # K8s manifests
```

**End-to-End Flow Requirements:**
1. Real-time data ingestion from external APIs (financial/social media)
2. Clean, transform, validate data using Spark
3. Store in PostgreSQL with proper schema design
4. Train ML models (time series forecasting, classification, or NLP)
5. Serve predictions via REST API with Swagger docs
6. Monitor pipeline, log metrics, handle errors gracefully
7. Orchestrate with Airflow DAGs for end-to-end workflow management

**Core Files Must Include:**
- requirements.txt + pyproject.toml
- Dockerfile + docker-compose.yml with all services (Kafka, Spark, Airflow, PostgreSQL, MLflow)
- Makefile with commands for setup, test, run, deploy
- .env.example for configuration
- pytest configuration with 85%+ coverage target
- GitHub Actions CI/CD workflow for testing and deployment
- CLI interface using Typer for operations
- Comprehensive README with architecture diagrams, setup instructions, API docs

**Quality Standards:**
- Enterprise-grade architecture with proper separation of concerns
- Production-ready error handling, logging, and monitoring
- Comprehensive test coverage (unit + integration tests)
- Professional documentation with architecture diagrams
- Docker-based deployment with all services
- CLI interface for operations and management
- LOC target: 1500-3000+ (complex, production-grade codebase)

Apply these advanced requirements ONLY when the field is data engineering or machine learning/ai. For other fields, use the standard requirements above.

DELIVERABLES

* Provide ALL files (source, README, tests, scripts, workflow, requirements, Makefile/Dockerfile, etc.) ONLY as fenced filename blocks exactly like:

```path/filename
<content>
```

