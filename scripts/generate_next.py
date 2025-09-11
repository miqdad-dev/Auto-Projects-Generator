#!/usr/bin/env python3
import os
import re
import json
import time
import random
import pathlib
import subprocess
import tempfile
from datetime import datetime, timezone


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

FIELDS = [
    "backend api",
    "frontend web/app",
    "systems programming",
    "data engineering",
    "machine learning/ai",
    "distributed systems",
    "devops/infrastructure",
    "databases",
    "networking",
    "security",
    "compilers/interpreters",
    "robotics/iot",
    "game dev",
    "scripting/automation",
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
    if plan:
        choices = [x.strip() for x in plan.split(",") if x.strip()]
        if choices:
            return random.choice(choices)
    return random.choice(FIELDS)

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


def build_prompt(field: str, today: str) -> str:
    codex_path = REPO_ROOT / "codex.md"
    codex = codex_path.read_text(encoding="utf-8") if codex_path.exists() else ""
    extra = (
        f"\n\nField for this run: {field}. Today's UTC date: {today}.\n"
        f"Output files for a new folder named {today}-<short-slug>.\n"
        f"Do not include any references to automation or generators in the files.\n"
        f"Do not include workflows that schedule generation of projects.\n"
    )
    return f"{codex}\n{extra}"


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


def fallback_minimal_project(project_root: pathlib.Path, title: str) -> None:
    (project_root / "src").mkdir(parents=True, exist_ok=True)
    (project_root / "tests").mkdir(parents=True, exist_ok=True)
    (project_root / "README.md").write_text(
        f"# {title}\n\n"
        "A small event-driven state machine with tests.\n\n"
        "## Run\n\n"
        "python -m pytest -q\n\n"
        "## Overview\n\n"
        "Implements a minimal state machine with valid transitions.\n",
        encoding="utf-8",
    )
    (project_root / "src" / "app.py").write_text(
        "class Machine:\n"
        "    def __init__(self):\n        self.state='idle'\n"
        "    def send(self, event):\n"
        "        if self.state=='idle' and event=='start':\n            self.state='running'\n"
        "        elif self.state=='running' and event=='stop':\n            self.state='idle'\n"
        "        else:\n            raise ValueError('invalid transition')\n",
        encoding="utf-8",
    )
    (project_root / "tests" / "test_app.py").write_text(
        "from src.app import Machine\n\n"
        "def test_machine():\n"
        "    m = Machine()\n"
        "    m.send('start')\n"
        "    assert m.state=='running'\n"
        "    m.send('stop')\n"
        "    assert m.state=='idle'\n",
        encoding="utf-8",
    )


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
                install_cmd = ["npm", "ci"] if (project_root / "package-lock.json").exists() else ["npm", "install", "--no-audit", "--no-fund"]
                if not run_ok(install_cmd, cwd=project_root):
                    return False
                return run_ok(["npm", "test", "--silent"], cwd=project_root)
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

    # Choose field
    field = choose_field()
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
    prompt = build_prompt(field, today)

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
    if top_folder:
        project_name = top_folder
    else:
        base_slug = slugify(f"{field.split()[0]}-{random.choice(NOUNS)}")
        project_name = f"{today}-{base_slug}"

    # Ensure unique name on GitHub
    def exists_checker(name: str) -> bool:
        try:
            return gh_repo_exists(gh_token, gh_api, gh_owner, name)
        except Exception:
            return False

    project_name = next_unique_name(project_name, exists_checker)

    # Create working directory now that name is known
    project_root = tmpdir / project_name
    project_root.mkdir(parents=True, exist_ok=True)

    written = 0
    if blocks:
        written = write_blocks(blocks, project_root, root_prefix_to_strip=top_folder)

    # Fallback if nothing written
    if written == 0:
        title = f"{project_name}"
        fallback_minimal_project(project_root, title)

    # Remove automation hints (if any)
    cleanse_automation_artifacts(project_root)

    # Gate push on tests passing
    print("Running tests to validate project before push...")
    if not detect_and_run_tests(project_root):
        print("Tests failed or could not be executed; skipping push.")
        return 3

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
    except Exception as e:
        print(f"GitHub push failed: {e}")
        return 2

    print(f"Generated and exported: {project_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
