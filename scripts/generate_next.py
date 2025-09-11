#!/usr/bin/env python3
import os
import re
import json
import time
import random
import pathlib
import subprocess
import tempfile
import shutil
from datetime import datetime, timezone


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

FIELDS = [
    "frontend web/app",
    "backend api",
    "app (cli/desktop)",
    "data engineering",
    "machine learning/ai",
    "game dev",
]

# Default field weights: data/ML/games are ~10x more likely
FIELD_WEIGHTS_DEFAULT = {
    "frontend web/app": 1,
    "backend api": 1,
    "app (cli/desktop)": 1,
    "data engineering": 10,
    "machine learning/ai": 10,
    "game dev": 10,
}

# Weighted language preferences: mostly Java and Python for OOP
LANGS = [
    "python", "python", "python",  # heavier weight
    "java", "java", "java",         # heavier weight
    "typescript",
    "javascript",
    "go",
    "rust",
]

NOUNS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey",
    "xray", "yankee", "zulu", "atlas", "orion", "vega", "nova", "ember",
]


def load_env_file(path: pathlib.Path) -> None:
    if not path.exists():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip("\"'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


def utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return text or "proj"


def choose_field() -> str:
    plan = os.getenv("TOPIC_PLAN", "").strip()
    # Start from either a planned subset or full set
    if plan:
        base = [x.strip() for x in plan.split(",") if x.strip()]
        # Keep only known fields to avoid typos
        candidates = [f for f in FIELDS if f in base]
        if not candidates:
            candidates = FIELDS[:]
    else:
        candidates = FIELDS[:]

    # Optional override via FIELD_WEIGHTS env: "field:weight,field:weight"
    weights_env = os.getenv("FIELD_WEIGHTS", "").strip()
    weights_map = FIELD_WEIGHTS_DEFAULT.copy()
    if weights_env:
        for pair in weights_env.split(","):
            if ":" in pair:
                k, v = pair.split(":", 1)
                k = k.strip()
                try:
                    w = int(v.strip())
                    if k in weights_map and w > 0:
                        weights_map[k] = w
                except ValueError:
                    pass

    weights = [weights_map.get(f, 1) for f in candidates]
    # Use random.choices for weighted selection
    return random.choices(candidates, weights=weights, k=1)[0]


def choose_language() -> str:
    return random.choice(LANGS)

def next_unique_name(base: str, exists_checker) -> str:
    candidate = base
    i = 2
    while exists_checker(candidate):
        candidate = f"{base}-{i}"
        i += 1
    return candidate


def github_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _is_masked_or_empty(value: str | None) -> bool:
    if value is None:
        return True
    v = value.strip().strip('"').strip("'")
    return v == "" or v == "***"


FENCE_RE = re.compile(r"(`{3,})([^\n\r`]+)\r?\n(.*?)\r?\n\1", re.DOTALL)


def parse_file_blocks(text: str) -> list[tuple[str, str]]:
    blocks = []
    for m in FENCE_RE.finditer(text):
        path = m.group(2).strip()
        content = m.group(3)
        if path:
            blocks.append((path, content))
    return blocks


def extract_top_folder(blocks: list[tuple[str, str]]) -> str | None:
    first_segments = set()
    for path, _ in blocks:
        seg = pathlib.PurePosixPath(path).parts[0] if "/" in path else pathlib.PurePath(path).parts[0]
        if seg:
            first_segments.add(seg)
    if len(first_segments) == 1:
        return next(iter(first_segments))
    return None


DATE_ORDERS = {"MDY", "DMY", "YMD"}


def format_repo_date(order: str) -> str:
    now = datetime.now(timezone.utc)
    y, m, d = now.year, now.month, now.day
    order = (order or "MDY").upper()
    if order not in DATE_ORDERS:
        order = "MDY"
    if order == "DMY":
        return f"{d:02d}-{m:02d}-{y}"
    if order == "YMD":
        return f"{y}-{m:02d}-{d:02d}"
    return f"{m:02d}-{d:02d}-{y}"


def strip_date_from_name(name: str) -> str:
    # Remove common leading or trailing date patterns
    n = name
    n = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", n)  # 2025-09-10-foo
    n = re.sub(r"-(\d{4}-\d{2}-\d{2})$", "", n)  # foo-2025-09-10
    n = re.sub(r"^\d{2}-\d{2}-\d{4}-", "", n)  # 10-09-2025-foo
    n = re.sub(r"-(\d{2}-\d{2}-\d{4})$", "", n)  # foo-10-09-2025
    n = re.sub(r"^\d{2}-\d{2}-\d{2}-", "", n)  # 10-09-25-foo
    n = re.sub(r"-(\d{2}-\d{2}-\d{2})$", "", n)  # foo-10-09-25
    return n


def ensure_readme_quality(project_root: pathlib.Path, project_title: str, field: str, language: str) -> None:
    rd = project_root / "README.md"
    if not rd.exists():
        rd.write_text(f"# {project_title}\n\n", encoding="utf-8")
    text = rd.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    # Force H1 to be just project title (no dates)
    if lines and lines[0].startswith("# "):
        lines[0] = f"# {project_title}"
    else:
        lines.insert(0, f"# {project_title}")

    content = "\n".join(lines)
    missing = []
    required_sections = [
        "Overview", "Features", "Architecture", "Setup", "Configuration",
        "Usage", "Examples", "Testing", "Deployment", "Hosting", "Troubleshooting"
    ]
    for sec in required_sections:
        if f"## {sec}" not in content:
            missing.append(sec)

    if missing or len(content) < 800:
        guide = (
            f"\n\n## Overview\n\n"
            f"{project_title} is a {field} project implemented in {language.capitalize()}. "
            f"It focuses on clear object-oriented design, maintainability, and practical examples.\n\n"
            f"## Features\n\n"
            f"- Clear project structure and modular components\n"
            f"- Non-trivial core logic (state machines, parsing, or algorithms)\n"
            f"- Comprehensive tests and examples\n\n"
            f"## Architecture\n\n"
            f"- Components: describe key modules/classes and responsibilities\n"
            f"- Data flow: outline request/response or pipeline stages\n"
            f"- Tradeoffs: briefly note choices (performance vs. simplicity, etc.)\n\n"
            f"## Setup\n\n"
            f"1. Ensure you have the required runtime (e.g., Java 17, Python 3.11, or Node 20).\n"
            f"2. Install dependencies (see below).\n\n"
            f"### Dependencies\n\n"
            f"- Java: Maven `mvn -q -DskipTests install` or Gradle `./gradlew build`\n"
            f"- Python: `python -m pip install -r requirements.txt`\n"
            f"- Node: `npm install` or `yarn install` or `pnpm install`\n\n"
            f"## Configuration\n\n"
            f"- Provide environment variables or config files as needed.\n"
            f"- Example: `.env` or `config.yaml`.\n\n"
            f"## Usage\n\n"
            f"Provide exact commands. For example:\n\n"
            f"- Java (Maven): `mvn -q exec:java`\n"
            f"- Java (Gradle): `./gradlew run`\n"
            f"- Python CLI: `python -m package.module --help`\n"
            f"- Web: `npm run dev` and open http://localhost:3000\n\n"
            f"## Examples\n\n"
            f"Show minimal inputs/outputs or curl examples for APIs.\n\n"
            f"## Testing\n\n"
            f"- Java: `mvn -q test` or `./gradlew test`\n"
            f"- Python: `pytest -q`\n"
            f"- Node: `npm test` / `yarn test` / `pnpm test`\n\n"
            f"## Deployment\n\n"
            f"- Docker: provide a `Dockerfile` or `docker-compose.yml` if applicable.\n"
            f"- Cloud: describe minimal steps for common providers (Heroku/Fly.io/Render).\n"
            f"- Data/ML: how to run pipelines or training jobs with parameters.\n\n"
            f"## Hosting\n\n"
            f"- Static web: publish `docs/` via GitHub Pages (Settings → Pages).\n"
            f"- Backend: deploy with Docker or a PaaS; expose port and configure env vars.\n\n"
            f"## Troubleshooting\n\n"
            f"- Common build errors and how to resolve them.\n"
            f"- Environment/version mismatches.\n"
        )
        content = content.rstrip() + guide

    rd.write_text(content, encoding="utf-8")


def is_safe_relpath(path: str) -> bool:
    p = pathlib.Path(path)
    if p.is_absolute():
        return False
    parts = p.parts
    return not any(part in ("..",) for part in parts)


def ensure_parent(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def call_openai(model: str, prompt: str) -> str:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model or "gpt-4o",
        messages=[
            {"role": "system", "content": "You are an elite code generator."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    return resp.choices[0].message.content or ""


def call_anthropic(model: str, prompt: str) -> str:
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)
    model = model or "claude-3-5-sonnet-latest"
    msg = client.messages.create(
        model=model,
        max_tokens=4000,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    parts = []
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


def build_prompt(field: str, today: str, language: str) -> str:
    codex_path = REPO_ROOT / "codex.md"
    codex = codex_path.read_text(encoding="utf-8") if codex_path.exists() else ""
    extra = (
        f"\n\nField for this run: {field}. Today's UTC date: {today}.\n"
        f"Primary implementation language: {language}. Favor OOP design, especially for Java/Python.\n"
        f"Output files for a new folder named <short-slug> (no dates in folder names).\n"
        f"Do not include any references to automation or generators in the files.\n"
        f"Do not include workflows that schedule generation of projects.\n"
        f"README must be professional and comprehensive and follow this structure exactly: \n"
        f"1) Title & Tagline (project name + one-line purpose)\n"
        f"2) Badges (optional)\n"
        f"3) Overview: what it does, the problem it solves, why it exists\n"
        f"4) Features: bullet list of capabilities\n"
        f"5) Getting Started: prerequisites, installation steps, and precise usage commands\n"
        f"6) Configuration: environment variables and an example .env\n"
        f"7) Workflow / Automation (only for this project itself, not how it was created): CI/CD, scheduled jobs if relevant\n"
        f"8) Examples / Screenshots: code snippets or structure examples\n"
        f"9) Testing: how to run tests and which framework\n"
        f"10) Deployment/Hosting: how to run in production or host (e.g., Pages/Docker/cloud)\n"
        f"11) Roadmap / Future Work (optional)\n"
        f"12) Contributing (optional)\n"
        f"13) License (explicit, e.g., MIT)\n"
        f"14) Acknowledgments (optional)\n"
        f"The README should highlight the problem, the approach to fix it, and how the solution works in clear, expert language.\n"
        f"Ensure non-trivial logic (state machines, concurrency, parsing, algorithms, or similar).\n"
        f"Tests are mandatory: JUnit for Java (Maven/Gradle) or pytest for Python; for JS use a standard test runner.\n"
        f"For frontend/web: produce a static site with an index.html (root or docs/) and clear build steps.\n"
        f"For data engineering: include sample data and a pipeline with validation.\n"
        f"For machine learning: prefer real datasets (e.g., Kaggle). Provide a script to download using the Kaggle API (do not commit large datasets). Include a small sample for tests and a train/eval script with metrics. Document KAGGLE_USERNAME/KAGGLE_KEY usage.\n"
        f"For games: deliver a playable loop and basic controls.\n"
        f"When external APIs or assets are needed (e.g., for games), integrate public endpoints/libraries and document keys/limits. Provide an offline fallback for tests.\n"
    )
    return f"{codex}\n{extra}"


def mm_dd_from_date_str(date_str: str) -> str | None:
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", date_str)
    if not m:
        return None
    return f"{m.group(2)}-{m.group(3)}-{m.group(1)}"


def extract_date_and_slug_from_folder(name: str) -> tuple[str | None, str | None]:
    # Expect YYYY-MM-DD-<slug>
    m = re.match(r"^(\d{4}-\d{2}-\d{2})-(.+)$", name)
    if not m:
        return None, None
    date_str = m.group(1)
    slug = slugify(m.group(2))
    return date_str, slug


def write_blocks(blocks: list[tuple[str, str]], project_root: pathlib.Path, root_prefix_to_strip: str | None = None) -> int:
    written = 0
    for rel_path, content in blocks:
        # Strip any leading project root in the model output to avoid duplication
        rp = rel_path
        prefix = (root_prefix_to_strip or project_root.name) + "/"
        if rp.startswith(prefix):
            rp = rp[len(prefix):]
        if not is_safe_relpath(rp):
            continue
        target = project_root / rp
        ensure_parent(target)
        target.write_text(content, encoding="utf-8")
        written += 1
    return written


def _w(path: pathlib.Path, content: str) -> None:
    ensure_parent(path)
    path.write_text(content, encoding="utf-8")


def fallback_frontend_static(project_root: pathlib.Path, title: str) -> None:
    (project_root / "docs").mkdir(parents=True, exist_ok=True)
    _w(project_root / "docs" / "index.html", f"""<!doctype html>
<html lang=\"en\"><head>
  <meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
  <title>{title}</title>
  <link rel=\"stylesheet\" href=\"style.css\">
</head><body>
  <main>
    <h1>{title}</h1>
    <p>Interactive demo: click to spawn particles. Use the README for full details.</p>
    <canvas id=\"c\" width=\"800\" height=\"400\"></canvas>
  </main>
  <script src=\"app.js\"></script>
</body></html>
""")
    _w(project_root / "docs" / "style.css", """
body{font-family:system-ui,Arial,sans-serif;margin:0;background:#0b1221;color:#e9eef7}
main{max-width:960px;margin:32px auto;padding:16px}
canvas{display:block;width:100%;background:#0f1730;border:1px solid #1f2a4a}
""")
    _w(project_root / "docs" / "app.js", """
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const P=[]; function rnd(n){return Math.random()*n}
canvas.addEventListener('click',e=>{for(let i=0;i<100;i++){P.push({x:e.offsetX,y:e.offsetY,vx:rnd(4)-2,vy:rnd(4)-2,l:100})}});
function step(){
  ctx.fillStyle='#0f1730';ctx.fillRect(0,0,canvas.width,canvas.height);
  for(const p of P){p.x+=p.vx;p.y+=p.vy;p.vy+=0.02;p.l--;}
  for(let i=P.length-1;i>=0;i--){if(P[i].l<=0)P.splice(i,1)}
  ctx.fillStyle='#72e5ff';
  for(const p of P){ctx.globalAlpha=Math.max(0,p.l/100);ctx.fillRect(p.x,p.y,2,2)}
  ctx.globalAlpha=1.0;requestAnimationFrame(step)
}step();
""")
    _w(project_root / "LICENSE", """
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
""")
    _w(project_root / "README.md", "# " + title + "\n\n" + """

An interactive HTML5 canvas demo showcasing a small particle system. This project demonstrates how to structure and host a static site problem: build a smooth, responsive animation with zero dependencies and deploy it reliably on GitHub Pages.

## Overview
This project addresses the problem of delivering a lightweight, portable interactive visual without any framework or backend. It uses a minimal, dependency-free JavaScript loop to animate particles spawned by user clicks. The solution focuses on clarity, performance, and ease of hosting.

## Features
- Dependency-free interactive animation (HTML/CSS/JS only)
- Particle physics loop with fade-out lifecycle
- Responsive canvas sizing and simple styling
- Ready for static hosting (GitHub Pages)

## Getting Started
### Prerequisites
- A modern web browser

### Installation
1. Clone the repo
   - `git clone <this repo>`
   - `cd <repo>`
2. No build step required

### Usage
- Open `docs/index.html` in a browser
- Click anywhere on the canvas to spawn particles

## Configuration
No runtime configuration required. You can tweak constants in `docs/app.js` (e.g., particle count, velocity, gravity).

## Examples / Screenshots
- See `docs/index.html` and `docs/app.js` for clean, commented code.

## Testing
This static demo is visual; no runtime tests are included. For CI validation, you could add a link checker or lighthouse CI later.

## Deployment / Hosting
- GitHub Pages: Settings → Pages → Deploy from a branch: `main`, folder: `/docs`
- After enabling, visit: `https://<owner>.github.io/<repo>/`

## Roadmap / Future Work
- Add UI toggles for particle physics
- Add FPS meter and performance profiling

## Contributing
Issues and PRs are welcome. Please describe the problem and proposed solution clearly.

## License
MIT

## Acknowledgments
Inspired by classic particle demos and the elegance of small, dependency-free visualizations.
""")


def fallback_backend_fastapi(project_root: pathlib.Path, title: str) -> None:
    _w(project_root / "requirements.txt", "fastapi==0.112.2\nuvicorn==0.30.6\npytest==8.3.3\nhttpx==0.27.2\n")
    _w(project_root / "app" / "main.py", """
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Service")

@app.get("/health")
def health():
    return {"status":"ok"}

@app.get("/fibonacci/{n}")
def fibonacci(n: int):
    if n < 0 or n > 1000:
        raise HTTPException(400, "n must be 0..1000")
    a,b=0,1
    for _ in range(n):
        a,b=b,a+b
    return {"n": n, "value": a}
""")
    _w(project_root / "tests" / "test_api.py", """
from fastapi.testclient import TestClient
from app.main import app

def test_health():
    c = TestClient(app)
    assert c.get('/health').json()['status']=='ok'

def test_fib():
    c = TestClient(app)
    assert c.get('/fibonacci/0').json()['value']==0
    assert c.get('/fibonacci/7').json()['value']==13
""")
    _w(project_root / "LICENSE", """
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
""")
    _w(project_root / "README.md", "# " + title + "\n\n" + """

Reliable FastAPI microservice for bounded computation with a clean API and test coverage.

## Overview
Problem: expose a safe, performant compute endpoint (e.g., Fibonacci) with input validation, health checks, and tests. Solution: a minimal FastAPI service with clear routing, validation, and an extensible structure for more endpoints.

## Features
- `GET /health` liveness endpoint
- `GET /fibonacci/{{n}}` computation with input guard (0..1000)
- Pytest-based tests with TestClient

## Getting Started
### Prerequisites
- Python 3.11+

### Installation
```bash
python -m pip install -r requirements.txt
```

### Run
```bash
uvicorn app.main:app --reload --port 8000
```

### Test
```bash
python -m pytest -q
```

## Configuration
No secrets required for local use. For production, consider environment variables for logging level, timeouts, or CORS.

## Examples
```bash
curl http://localhost:8000/health
curl http://localhost:8000/fibonacci/10
```

## Deployment / Hosting
- Container entrypoint: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
- Host on Fly.io, Render, Railway, or any container platform.

## Testing
- Framework: pytest with `fastapi.testclient`
- Run: `python -m pytest -q`

## Roadmap / Future Work
- Add metrics and tracing (OpenTelemetry)
- Add rate limiting and auth

## Contributing
PRs welcome; include tests and rationale.

## License
MIT
""")


def fallback_data_engineering(project_root: pathlib.Path, title: str) -> None:
    _w(project_root / "requirements.txt", "pandas==2.2.2\npytest==8.3.3\npyyaml==6.0.2\n")
    _w(project_root / "data" / "input.csv", """id,value\n1,10\n2,20\n3,30\n""")
    _w(project_root / "pipeline" / "transform.py", """
import pandas as pd
from pathlib import Path

def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['double'] = df['value'] * 2
    return df

def save(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)

def run(in_path: str, out_path: str) -> None:
    df = load(Path(in_path))
    out = transform(df)
    save(out, Path(out_path))

if __name__ == "__main__":
    run("data/input.csv", "data/output.csv")
""")
    _w(project_root / "tests" / "test_transform.py", """
from pipeline.transform import transform
import pandas as pd

def test_transform():
    df = pd.DataFrame({'id':[1,2], 'value':[5,6]})
    out = transform(df)
    assert list(out['double']) == [10,12]
""")
    _w(project_root / "LICENSE", """
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
""")
    _w(project_root / "README.md", "# " + title + "\n\n" + """

Deterministic CSV-to-CSV transformation pipeline with validation and tests.

## Overview
Problem: reliably transform tabular data with reproducible logic and test coverage. Solution: a small, explicit pipeline that loads, transforms, and saves CSV using pandas, with unit tests to prevent regressions.

## Features
- Load/transform/save pattern
- Deterministic computation and unit tests
- Simple configuration via CLI arguments

## Getting Started
### Prerequisites
- Python 3.11+

### Installation
```bash
python -m pip install -r requirements.txt
```

### Usage
```bash
python pipeline/transform.py  # reads data/input.csv and writes data/output.csv
```

## Configuration
Environment variables are not required. Adjust file paths by editing the script or adding CLI flags.

## Examples
Input (`data/input.csv`):
```
id,value
1,10
2,20
```
Output (`data/output.csv`):
```
id,value,double
1,10,20
2,20,40
```

## Testing
- Framework: pytest
- Run: `python -m pytest -q`

## Roadmap / Future Work
- Add schema validation (pydantic/cerberus)
- Add incremental processing and logging

## License
MIT
""")


def fallback_ml(project_root: pathlib.Path, title: str) -> None:
    _w(project_root / "requirements.txt", "scikit-learn==1.5.2\nnumpy==1.26.4\npytest==8.3.3\n")
    _w(project_root / "ml" / "train.py", """
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_and_eval(random_state: int = 42) -> float:
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    return accuracy_score(y_test, pred)

if __name__ == "__main__":
    acc = train_and_eval()
    print({"accuracy": acc})
""")
    _w(project_root / "tests" / "test_ml.py", """
from ml.train import train_and_eval

def test_accuracy_threshold():
    acc = train_and_eval()
    assert acc >= 0.85
""")
    _w(project_root / "LICENSE", """
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
""")
    _w(project_root / "README.md", f"""# {title}

Logistic regression classifier on the Iris dataset with reproducible train/eval and an accuracy threshold test.

## Overview
Problem: provide a minimal, reproducible ML baseline with explicit metrics and tests. Solution: a scikit-learn pipeline that trains a classifier and asserts a minimum accuracy, ensuring changes don’t silently degrade performance.

## Features
- Deterministic data split and training
- Accuracy metric printing and test gating
- Simple, dependency-light implementation

## Getting Started
### Prerequisites
- Python 3.11+

### Installation
```bash
python -m pip install -r requirements.txt
```

### Usage
```bash
python ml/train.py
```

## Configuration
Modify `random_state` or model hyperparameters in `ml/train.py`.

## Examples
Example output:
```
{"accuracy": 0.96}
```

## Testing
- Framework: pytest
- Run: `python -m pytest -q`
- Test criterion: accuracy >= 0.85

## Roadmap / Future Work
- Add model persistence and CLI args
- Add cross-validation and feature scaling

## License
MIT
""")


def fallback_cli_app(project_root: pathlib.Path, title: str) -> None:
    _w(project_root / "requirements.txt", "pytest==8.3.3\n")
    _w(project_root / "src" / "cli.py", """
import argparse, sys

def solve(s: str) -> str:
    # Reverse words while preserving whitespace
    parts = s.split(' ')
    return ' '.join(w[::-1] for w in parts)

def main(argv=None):
    p = argparse.ArgumentParser(description="String transformer")
    p.add_argument('text', help='input text')
    args = p.parse_args(argv)
    print(solve(args.text))

if __name__ == '__main__':
    main()
""")
    _w(project_root / "tests" / "test_cli.py", """
from src.cli import solve

def test_solve():
    assert solve('hello world') == 'olleh dlrow'
""")
    _w(project_root / "README.md", f"""# {title}

Command-line text transformer with tests.

## Run
- `python -m pip install -r requirements.txt`
- `python -m pytest -q`
- `python -m src.cli "some text"`
""")


def fallback_game_static(project_root: pathlib.Path, title: str) -> None:
    fallback_frontend_static(project_root, title)


def fallback_problem_project(field: str, language: str, project_root: pathlib.Path, title: str) -> None:
    f = field.lower()
    if "front" in f:
        return fallback_frontend_static(project_root, title)
    if "back" in f:
        return fallback_backend_fastapi(project_root, title)
    if "data" in f:
        return fallback_data_engineering(project_root, title)
    if "machine" in f or "ml" in f:
        return fallback_ml(project_root, title)
    if "game" in f:
        return fallback_game_static(project_root, title)
    return fallback_cli_app(project_root, title)


def run(cmd: list[str], cwd: pathlib.Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def run_ok(cmd: list[str], cwd: pathlib.Path | None = None) -> bool:
    try:
        subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception:
        return False


def detect_and_run_tests(project_root: pathlib.Path) -> bool:
    # 1) Makefile target
    mk = project_root / "Makefile"
    if mk.exists():
        try:
            content = mk.read_text(encoding="utf-8", errors="ignore")
            if re.search(r"^\s*test:\s*$", content, flags=re.MULTILINE):
                return run_ok(["make", "test"], cwd=project_root)
        except Exception:
            pass

    # 2) Node (package.json)
    pkg = project_root / "package.json"
    if pkg.exists():
        try:
            pkgj = json.loads(pkg.read_text(encoding="utf-8"))
            scripts = (pkgj.get("scripts") or {})
            if "test" in scripts and isinstance(scripts["test"], str):
                pm = select_pkg_manager(project_root)
                if not node_install(project_root, pm):
                    return False
                return run_ok(node_run_cmd(pm, "test"), cwd=project_root)
        except Exception:
            return False

    # 3) Python (pytest)
    has_py_tests = any(p.suffix == ".py" and ("tests" in p.parts or p.name.startswith("test_") or p.name.endswith("_test.py")) for p in project_root.rglob("*.py"))
    if has_py_tests:
        req = project_root / "requirements.txt"
        if req.exists():
            if not run_ok(["python", "-m", "pip", "install", "-r", str(req)]):
                return False
        else:
            # ensure pytest available
            if not run_ok(["python", "-m", "pip", "install", "pytest"]):
                return False
        return run_ok(["python", "-m", "pytest", "-q"], cwd=project_root)

    # 4) Rust
    if (project_root / "Cargo.toml").exists():
        return run_ok(["cargo", "test", "--quiet"], cwd=project_root)

    # 5) Go
    if (project_root / "go.mod").exists() or any(f.name.endswith("_test.go") for f in project_root.rglob("*_test.go")):
        return run_ok(["go", "test", "./..."], cwd=project_root)

    # 6) Maven
    mvnw = project_root / "mvnw"
    if mvnw.exists():
        mvnw.chmod(0o755)
        return run_ok([str(mvnw), "-q", "-DskipITs=false", "test"], cwd=project_root)
    if (project_root / "pom.xml").exists():
        return run_ok(["mvn", "-q", "test"], cwd=project_root)

    # 7) Gradle
    gradlew = project_root / "gradlew"
    if gradlew.exists():
        gradlew.chmod(0o755)
        return run_ok([str(gradlew), "test"], cwd=project_root)
    if (project_root / "build.gradle").exists() or (project_root / "build.gradle.kts").exists():
        return run_ok(["gradle", "test"], cwd=project_root)

    # If no tests detected, consider success
    return True


def prepare_data_and_assets(project_root: pathlib.Path) -> None:
    """Optionally fetch datasets or build assets before tests if enabled.

    Controlled by env ENABLE_DATA_FETCH ("1" to enable). Non-fatal on failure.
    Looks for common entry points: Makefile targets, Python/JS scripts.
    """
    flag = (os.getenv("ENABLE_DATA_FETCH", "0").strip().lower() in ("1", "true", "yes"))
    if not flag:
        return

    def try_run(cmd: list[str]) -> bool:
        try:
            return run_ok(cmd, cwd=project_root)
        except Exception:
            return False

    # Makefile targets commonly used
    mk = project_root / "Makefile"
    if mk.exists():
        content = mk.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"^\s*data:\s*$", content, flags=re.MULTILINE):
            try_run(["make", "data"])
        if re.search(r"^\s*assets:\s*$", content, flags=re.MULTILINE):
            try_run(["make", "assets"])
        if re.search(r"^\s*prepare:\s*$", content, flags=re.MULTILINE):
            try_run(["make", "prepare"])

    # Python data download scripts
    for p in [
        project_root / "scripts" / "download_data.py",
        project_root / "data" / "download.py",
        project_root / "tools" / "fetch_data.py",
    ]:
        if p.exists():
            try_run(["python", str(p)])

    # Node package.json scripts
    pkg = project_root / "package.json"
    if pkg.exists():
        try:
            pkgj = json.loads(pkg.read_text(encoding="utf-8"))
            scripts = (pkgj.get("scripts") or {})
            pm = select_pkg_manager(project_root)
            if "data" in scripts:
                try_run(node_run_cmd(pm, "data"))
            if "assets" in scripts:
                try_run(node_run_cmd(pm, "assets"))
            if "prepare" in scripts:
                try_run(node_run_cmd(pm, "prepare"))
        except Exception:
            pass


def select_pkg_manager(project_root: pathlib.Path) -> str:
    if (project_root / "pnpm-lock.yaml").exists():
        return "pnpm"
    if (project_root / "yarn.lock").exists():
        return "yarn"
    return "npm"


def node_install(project_root: pathlib.Path, pm: str) -> bool:
    if pm == "pnpm":
        # Rely on corepack providing pnpm; if not present, try to install
        if not run_ok(["pnpm", "-v"]):
            if not run_ok(["npm", "i", "-g", "pnpm"]):
                return False
        return run_ok(["pnpm", "install", "--frozen-lockfile"], cwd=project_root)
    if pm == "yarn":
        # Corepack should handle yarn
        return run_ok(["yarn", "install", "--frozen-lockfile"], cwd=project_root)
    # npm
    if (project_root / "package-lock.json").exists():
        return run_ok(["npm", "ci"], cwd=project_root)
    return run_ok(["npm", "install", "--no-audit", "--no-fund"], cwd=project_root)


def node_run_cmd(pm: str, script: str) -> list[str]:
    if pm == "pnpm":
        return ["pnpm", "run", script]
    if pm == "yarn":
        return ["yarn", script]
    return ["npm", "run", script]


def node_bin(project_root: pathlib.Path, name: str) -> pathlib.Path:
    # Return path to local node bin in node_modules
    p = project_root / "node_modules" / ".bin" / name
    if os.name == "nt":
        cmd = p.with_suffix(".cmd")
        return cmd if cmd.exists() else p
    return p


def prepare_static_site(project_root: pathlib.Path) -> str | None:
    """Detect or build static site content for GitHub Pages.

    Returns the pages path ("/" or "/docs") if a static site is ready, else None.
    """
    root_index = project_root / "index.html"
    docs_index = project_root / "docs" / "index.html"
    if root_index.exists():
        return "/"
    if docs_index.exists():
        return "/docs"

    # Try Node build into docs/
    pkg = project_root / "package.json"
    if pkg.exists():
        try:
            pkgj = json.loads(pkg.read_text(encoding="utf-8"))
            scripts = (pkgj.get("scripts") or {})
            pm = select_pkg_manager(project_root)
            if not node_install(project_root, pm):
                return None

            # If "build" exists, run it
            if "build" in scripts and isinstance(scripts["build"], str):
                if not run_ok(node_run_cmd(pm, "build"), cwd=project_root):
                    return None

            # Next.js static export if Next is present
            deps = {**(pkgj.get("dependencies") or {}), **(pkgj.get("devDependencies") or {})}
            if "next" in deps:
                next_path = node_bin(project_root, "next")
                if next_path.exists():
                    # Try build then export to out/
                    if not run_ok([str(next_path), "build"], cwd=project_root):
                        return None
                    # Next export; default out dir is out/
                    if not run_ok([str(next_path), "export", "-o", "out"], cwd=project_root):
                        return None

            # Common output dirs
            for out in ("docs", "dist", "build", "out", "public"):
                out_dir = project_root / out
                if out_dir.exists() and any(out_dir.iterdir()):
                    # Ensure docs/ contains site
                    docs_dir = project_root / "docs"
                    if out_dir.name != "docs":
                        if docs_dir.exists():
                            shutil.rmtree(docs_dir)
                        shutil.copytree(out_dir, docs_dir)
                    # Validate index
                    idx = docs_dir / "index.html"
                    if idx.exists():
                        return "/docs"
        except Exception:
            return None
    return None


def gh_enable_pages(token: str, api: str, owner: str, repo: str, branch: str, path: str) -> None:
    import requests
    payload = {"source": {"branch": branch, "path": path}}
    url_create = f"{api}/repos/{owner}/{repo}/pages"
    r = requests.post(url_create, headers=github_headers(token), json=payload, timeout=30)
    if r.status_code in (201, 204):
        return
    # If already exists or forbidden, try update endpoint
    url_update = f"{api}/repos/{owner}/{repo}/pages"
    r2 = requests.put(url_update, headers=github_headers(token), json=payload, timeout=30)
    # Do not raise; if it fails, user can enable manually
    try:
        r2.raise_for_status()
    except Exception:
        pass


def gh_get_owner(token: str, api: str) -> str:
    import requests
    r = requests.get(f"{api}/user", headers=github_headers(token), timeout=20)
    r.raise_for_status()
    return r.json()["login"]


def gh_repo_exists(token: str, api: str, owner: str, name: str) -> bool:
    import requests
    r = requests.get(f"{api}/repos/{owner}/{name}", headers=github_headers(token), timeout=20)
    return r.status_code == 200


def gh_create_repo(token: str, api: str, owner: str, name: str, description: str, visibility: str) -> dict:
    import requests
    payload = {
        "name": name,
        "description": description,
        "private": (visibility.lower() != "public"),
        "has_issues": True,
        "has_projects": False,
        "has_wiki": False,
        "auto_init": False,
    }
    # Try to create under the authenticated user
    url_user = f"{api}/user/repos"
    r = requests.post(url_user, headers=github_headers(token), json=payload, timeout=30)
    if r.status_code == 422 and "name already exists" in r.text.lower():
        raise FileExistsError(name)
    if r.status_code == 403 and owner:
        # Fallback: attempt org route if owner is an organization
        url_org = f"{api}/orgs/{owner}/repos"
        r2 = requests.post(url_org, headers=github_headers(token), json=payload, timeout=30)
        if r2.status_code == 422 and "name already exists" in r2.text.lower():
            raise FileExistsError(name)
        try:
            r2.raise_for_status()
        except requests.HTTPError as e:
            raise RuntimeError(
                "GitHub repo creation forbidden. Ensure GH_PAT has permissions: "
                "Classic token with 'repo' scope or Fine-grained token with "
                "Administration (Read/Write) and Contents (Read/Write), and access to the target owner."
            ) from e
        return r2.json()
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        if r.status_code == 403:
            raise RuntimeError(
                "GitHub repo creation forbidden. Ensure GH_PAT has permissions: "
                "Classic token with 'repo' scope or Fine-grained token with Administration (Read/Write) and Contents (Read/Write)."
            ) from e
        raise
    return r.json()


def git_init_and_push(workdir: pathlib.Path, remote_https_url: str, user: str, email: str) -> None:
    run(["git", "init"], cwd=workdir)
    run(["git", "config", "user.name", user], cwd=workdir)
    run(["git", "config", "user.email", email], cwd=workdir)
    run(["git", "add", "-A"], cwd=workdir)
    run(["git", "commit", "-m", "initial commit"], cwd=workdir)
    run(["git", "branch", "-M", "main"], cwd=workdir)
    run(["git", "remote", "add", "origin", remote_https_url], cwd=workdir)
    run(["git", "push", "-u", "origin", "main"], cwd=workdir)


def cleanse_automation_artifacts(project_root: pathlib.Path) -> None:
    # Remove known generator/automation markers to keep repo natural
    auto_wf = project_root / ".github" / "workflows" / "auto.yml"
    try:
        if auto_wf.exists():
            auto_wf.unlink()
            # remove empty parent dirs
            wf_dir = auto_wf.parent
            if wf_dir.exists() and not any(wf_dir.iterdir()):
                wf_dir.rmdir()
            gh_dir = wf_dir.parent
            if gh_dir.exists() and not any(gh_dir.iterdir()):
                gh_dir.rmdir()
    except Exception:
        pass


def main() -> int:
    load_env_file(REPO_ROOT / ".env")
    provider = (os.getenv("PROVIDER") or "openai").strip().lower()
    model = os.getenv("MODEL_NAME") or ("gpt-4o" if provider == "openai" else "claude-3-5-sonnet-latest")

    # Choose field and language
    field = choose_field()
    language = choose_language()
    today = utc_date()

    # Resolve GitHub settings
    gh_api = os.getenv("GITHUB_API", "https://api.github.com")
    gh_token = os.getenv("GH_PAT") or os.getenv("GITHUB_TOKEN")
    if _is_masked_or_empty(gh_token):
        raise RuntimeError("Missing or invalid GH_PAT: add a real Personal Access Token in Actions secrets as GH_PAT (with repo permissions)")
    # Prefer explicit owner to avoid calling /user when token scopes are limited
    gh_owner = os.getenv("GITHUB_OWNER")
    if not gh_owner:
        gh_owner = gh_get_owner(gh_token, gh_api)
    gh_visibility = os.getenv("GITHUB_VISIBILITY", "public")

    # Temp base dir for the new repo (not inside this repo)
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="autogen-"))

    # Build prompt
    prompt = build_prompt(field, today, language)

    # Call provider
    try:
        if provider == "openai":
            output = call_openai(model, prompt)
        elif provider == "anthropic":
            output = call_anthropic(model, prompt)
        else:
            raise RuntimeError(f"Unsupported PROVIDER: {provider}")
    except Exception as e:
        output = ""
        print(f"Provider call failed: {e}")

    # Parse and write blocks
    blocks = parse_file_blocks(output) if output else []
    # Derive project name from blocks' top folder if present, else synthesize
    top_folder = extract_top_folder(blocks) if blocks else None

    # Determine repo name as <slug>-MM-DD-YYYY
    # Prefer slug/date from the model's top folder if present
    repo_base: str
    if top_folder:
        d, s = extract_date_and_slug_from_folder(top_folder)
        if s:
            mdy = mm_dd_from_date_str(d) if d else None
            if not mdy:
                mdy = datetime.now(timezone.utc).strftime("%m-%d-%Y")
            repo_base = f"{s}-{mdy}"
        else:
            # No recognizable date-then-slug; treat the folder as slug
            s = slugify(top_folder)
            mdy = datetime.now(timezone.utc).strftime("%m-%d-%Y")
            repo_base = f"{s}-{mdy}"
    else:
        s = slugify(f"{field.split()[0]}-{random.choice(NOUNS)}")
        mdy = datetime.now(timezone.utc).strftime("%m-%d-%Y")
        repo_base = f"{s}-{mdy}"

    # Ensure unique name on GitHub
    def exists_checker(name: str) -> bool:
        try:
            return gh_repo_exists(gh_token, gh_api, gh_owner, name)
        except Exception:
            return False

    project_name = next_unique_name(repo_base, exists_checker)

    # Create working directory now that name is known
    project_root = tmpdir / project_name
    project_root.mkdir(parents=True, exist_ok=True)

    written = 0
    if blocks:
        written = write_blocks(blocks, project_root, root_prefix_to_strip=top_folder)

    # Fallback if nothing written
    if written == 0:
        title = f"{project_name}"
        fallback_problem_project(field, language, project_root, title)

    # Remove automation hints (if any)
    cleanse_automation_artifacts(project_root)

    # Strengthen README quality and ensure title is project-only (no dates)
    display_title = strip_date_from_name(project_name).replace("-", " ").title()
    try:
        ensure_readme_quality(project_root, display_title, field, language)
    except Exception:
        pass

    # Optionally fetch datasets/assets before tests
    try:
        prepare_data_and_assets(project_root)
    except Exception:
        pass

    # Gate push on tests passing
    print("Running tests to validate project before push...")
    if not detect_and_run_tests(project_root):
        print("Tests failed or could not be executed; skipping push.")
        return 3

    # Prepare GitHub Pages if this is a static site
    pages_path = prepare_static_site(project_root)

    # Create GitHub repo and push contents
    # Derive description from README title if available to avoid automation hints
    description = f"{project_name}"
    readme_path = project_root / "README.md"
    if readme_path.exists():
        try:
            for line in readme_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("# "):
                    description = line[2:].strip()
                    break
        except Exception:
            pass
    try:
        try:
            repo_info = gh_create_repo(gh_token, gh_api, gh_owner, project_name, description, gh_visibility)
        except FileExistsError:
            # Should be rare due to uniqueness, but retry with suffix
            slug2 = next_unique_slug(slug, exists_checker)
            project_name = f"{today}-{slug2}"
            repo_info = gh_create_repo(gh_token, gh_api, gh_owner, project_name, description, gh_visibility)
        https_url = repo_info.get("clone_url")
        # Prefer token-auth embed to push
        # Use x-access-token to avoid exposing username
        token_remote = f"https://x-access-token:{gh_token}@github.com/{gh_owner}/{project_name}.git"
        git_user = os.getenv("GIT_AUTHOR_NAME", "dev")
        git_email = os.getenv("GIT_AUTHOR_EMAIL", "dev@users.noreply.github.com")
        git_init_and_push(project_root, token_remote, git_user, git_email)
        print(f"Created and pushed: https://github.com/{gh_owner}/{project_name}")
        # Enable GitHub Pages if static site detected and repo is public
        if pages_path and (gh_visibility.lower() == "public"):
            try:
                gh_enable_pages(gh_token, gh_api, gh_owner, project_name, "main", pages_path)
                print(f"Pages enabled at: https://{gh_owner}.github.io/{project_name}/")
            except Exception:
                print("Could not enable GitHub Pages automatically. You can enable it in repo settings.")
    except Exception as e:
        print(f"GitHub push failed: {e}")
        return 2

    print(f"Generated and exported: {project_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
